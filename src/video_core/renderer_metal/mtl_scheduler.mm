// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <unordered_map>
#include <vector>
#include "common/microprofile.h"
#include "common/thread.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

MICROPROFILE_DEFINE(Metal_WaitForWorker, "Metal", "Wait for worker", MP_RGB(255, 192, 192));
MICROPROFILE_DEFINE(Metal_Submit, "Metal", "Submit execution", MP_RGB(255, 192, 255));

namespace Metal {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr std::size_t CommandsPerChunk = 128;
constexpr NSUInteger CheckpointsPerHeap = 32;
constexpr auto WaitInterval = 10ms;

NSString* ToNSString(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}

NSError* MakeError(SubmissionStage stage, std::string_view message) {
    NSString* description = ToNSString(message);
    return [NSError errorWithDomain:@"org.azahar-emu.Metal.Scheduler"
                               code:static_cast<NSInteger>(stage)
                           userInfo:@{NSLocalizedDescriptionKey : description ?: @"Metal error"}];
}

NSError* DescribeException(SubmissionStage stage, std::exception_ptr error) {
    try {
        std::rethrow_exception(error);
    } catch (const MetalError& failure) {
        NSString* domain = ToNSString(failure.Domain());
        NSString* description = ToNSString(failure.what());
        return
            [NSError errorWithDomain:domain ?: @"org.azahar-emu.Metal"
                                code:failure.Code()
                            userInfo:@{NSLocalizedDescriptionKey : description ?: @"Metal error"}];
    } catch (const std::exception& failure) {
        return MakeError(stage, failure.what());
    } catch (...) {
        return MakeError(stage, "Non-standard exception during native Metal submission");
    }
}

struct TrackedUse {
    id<MTLResource> identity;
    id<MTLResource> resource;
    BufferRange range;
    BufferAccess access;
    MTLStages stages;
    u64 tick;
    u64 pass;
    QueueType queue;
};
using ResourceHistory = std::unordered_map<const void*, std::vector<TrackedUse>>;

id<MTLResource> GetBacking(const ResourceUse& use) {
    if (use.backing) {
        return use.backing;
    }
    if (![use.resource conformsToProtocol:@protocol(MTLTexture)]) {
        return use.resource;
    }
    id<MTLTexture> texture = static_cast<id<MTLTexture>>(use.resource);
    while (texture.parentTexture) {
        texture = texture.parentTexture;
    }
    if (id<MTLBuffer> buffer = texture.buffer) {
        return buffer;
    }
    return texture;
}

struct NativeCheckpoint {
    SubmissionCheckpoint metadata;
    id<MTL4CounterHeap> heap;
    NSUInteger index;
};

struct SubmissionResources {
    id<MTL4CommandBuffer> buffer = nil;
    id<MTLResidencySet> residency = nil;
    DeviceContext::WorkLease device;
    std::vector<id> objects;
    std::vector<MemoryAllocator::GPUBufferAccess> mapped_ranges;
    std::vector<ResourcePool::BindingLease> bindings;
    std::vector<std::shared_ptr<void>> owners;
    std::vector<NativeCheckpoint> checkpoints;

    std::optional<SubmissionCheckpoint> LastCheckpoint(bool may_have_run) const {
        if (checkpoints.empty()) {
            return {};
        }
        for (auto checkpoint = checkpoints.rbegin();
             may_have_run && checkpoint != checkpoints.rend(); ++checkpoint) {
            NSData* sample =
                [checkpoint->heap resolveCounterRange:NSMakeRange(checkpoint->index, 1)];
            if (sample.length == sizeof(u64)) {
                u64 value{};
                [sample getBytes:&value length:sizeof(value)];
                if (value != 0) {
                    return SubmissionCheckpoint{checkpoint->metadata.label, value};
                }
            }
        }
        return checkpoints.back().metadata;
    }
};

struct Diagnostics {
    void Record(SubmissionDiagnostic report) {
        std::scoped_lock lock(mutex);
        if (report.error && (!fault || report.tick < fault->tick)) {
            fault = report;
        }
        if (!last || report.tick > last->tick) {
            last = std::move(report);
        }
    }

    mutable std::mutex mutex;
    std::optional<SubmissionDiagnostic> fault;
    std::optional<SubmissionDiagnostic> last;
};

struct Queue {
    id<MTL4CommandQueue> native = nil;
    id<MTLSharedEvent> event = nil;
};

Queue MakeQueue(id<MTLDevice> device, NSString* name) {
    auto descriptor = [MTL4CommandQueueDescriptor new];
    descriptor.label = name;
    NSError* error = nil;
    auto queue = [device newMTL4CommandQueueWithDescriptor:descriptor error:&error];
    auto checked = CheckResource(queue, "Create native Metal 4 command queue", error);
    auto event = CheckResource([device newSharedEvent], "Create Metal queue synchronization event");
    event.label = name;
    return {checked, event};
}

constexpr std::size_t QueueIndex(QueueType queue) {
    return static_cast<std::size_t>(queue);
}

} // Anonymous namespace

struct EncodingContext::Impl {
    Impl(ResourcePool::CommandLease allocator, ResourcePool& pool, DeviceContext& context,
         const DeviceContext::WorkLease& device, ResourceHistory& history, u64 tick,
         QueueType queue, id<MTL4CommandBuffer> buffer)
        : allocator(std::move(allocator)), pool(pool), context(context), history(history),
          tick(tick), queue(queue), retained(std::make_shared<SubmissionResources>()) {
        retained->buffer = buffer;
        retained->device = device;
        [buffer beginCommandBufferWithAllocator:this->allocator.Get()];
    }

    ~Impl() {
        if (active) {
            [active endEncoding];
        }
        if (retained && !ended) {
            [retained->buffer endCommandBuffer];
        }
    }

    ResourcePool::CommandLease allocator;
    ResourcePool& pool;
    DeviceContext& context;
    ResourceHistory& history;
    const u64 tick;
    const QueueType queue;
    std::shared_ptr<SubmissionResources> retained;
    std::unordered_map<const void*, id<MTLResource>> residency;
    id<MTL4CommandEncoder> active = nil;
    u64 pass{};
    u64 cross_queue_tick{};
    bool ended{};
};

EncodingContext::EncodingContext(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

EncodingContext::~EncodingContext() = default;

id<MTL4CommandBuffer> EncodingContext::GetCommandBuffer() const noexcept {
    return impl->retained->buffer;
}

u64 EncodingContext::CurrentTick() const noexcept {
    return impl->tick;
}

QueueType EncodingContext::GetQueue() const noexcept {
    return impl->queue;
}

bool EncodingContext::OwnsWork(const DeviceContext::WorkLease& lease) const {
    return lease.Generation() == impl->retained->device.Generation() &&
           impl->context.GetLeaseStatus(lease) == WorkLeaseStatus::Valid;
}

ResourcePool& EncodingContext::GetResourcePool() const noexcept {
    return impl->pool;
}

void EncodingContext::MakeResident(id<MTLResource> resource) {
    if (!resource || resource.device != GetCommandBuffer().device) {
        throw std::invalid_argument("Metal residency requires a same-device resource");
    }
    impl->residency.try_emplace((__bridge const void*)resource, resource);
}

void EncodingContext::Synchronize(id<MTL4CommandEncoder> encoder,
                                  std::span<const ResourceUse> uses) {
    if (!encoder || encoder.commandBuffer != GetCommandBuffer() ||
        (impl->active && impl->active != encoder)) {
        throw std::logic_error("Metal synchronization requires the current submission encoder");
    }
    if (!impl->active) {
        impl->active = encoder;
        ++impl->pass;
    }

    struct Barrier {
        MTLStages after{};
        MTLStages before{};
        MTL4VisibilityOptions visibility = MTL4VisibilityOptionDevice;

        void Include(const TrackedUse& previous, const ResourceUse& use) {
            after |= previous.stages;
            before |= use.stages;
            if (previous.resource != use.resource) {
                visibility |= MTL4VisibilityOptionResourceAlias;
            }
        }
    };
    Barrier within_pass;
    Barrier across_passes;
    std::vector<TrackedUse> current;
    current.reserve(uses.size());
    for (const auto& use : uses) {
        if (!use.resource || use.stages == 0 || use.stages == MTLStageAll) {
            throw std::invalid_argument("Metal resource use requires a resource and stages");
        }
        if (use.access != BufferAccess::Read && use.access != BufferAccess::Write &&
            use.access != BufferAccess::ReadWrite) {
            throw std::invalid_argument("Invalid Metal resource access");
        }
        MakeResident(use.resource);
        id<MTLResource> backing = GetBacking(use);
        MakeResident(backing);
        const auto range = backing != use.resource && !use.backing ? BufferRange{} : use.range;
        if (range.length != 0 && [backing conformsToProtocol:@protocol(MTLBuffer)] &&
            !range.Fits(static_cast<id<MTLBuffer>>(backing).length)) {
            throw std::out_of_range("Metal resource access exceeds the backing allocation");
        }
        auto& previous = impl->history[(__bridge const void*)backing];
        for (const auto& prior : previous) {
            if (!ResourceUse::Conflicts(prior.access, prior.range, use.access, range)) {
                continue;
            }
            if (prior.queue != impl->queue) {
                impl->cross_queue_tick = std::max(impl->cross_queue_tick, prior.tick);
            } else if (prior.tick == impl->tick && prior.pass == impl->pass) {
                within_pass.Include(prior, use);
            } else {
                across_passes.Include(prior, use);
            }
        }
        current.push_back({backing, use.resource, range, use.access, use.stages, impl->tick,
                           impl->pass, impl->queue});
    }

    if (across_passes.after) {
        [encoder barrierAfterQueueStages:across_passes.after
                            beforeStages:across_passes.before
                       visibilityOptions:across_passes.visibility];
    }
    if (within_pass.after) {
        [encoder barrierAfterEncoderStages:within_pass.after
                       beforeEncoderStages:within_pass.before
                         visibilityOptions:within_pass.visibility];
    }

    for (auto& use : current) {
        auto& previous = impl->history.at((__bridge const void*)use.identity);
        auto match = std::ranges::find_if(previous, [&](const TrackedUse& item) {
            return item.resource == use.resource && item.tick == use.tick &&
                   item.pass == use.pass && item.range.offset == use.range.offset &&
                   item.range.length == use.range.length && item.access == use.access;
        });
        if (match == previous.end()) {
            previous.push_back(std::move(use));
        } else {
            match->stages |= use.stages;
        }
    }
}

void EncodingContext::EndPass(id<MTL4CommandEncoder> encoder) {
    if (!encoder || encoder.commandBuffer != GetCommandBuffer() ||
        (impl->active && impl->active != encoder)) {
        throw std::logic_error("Cannot end a foreign Metal pass");
    }
    [encoder endEncoding];
    impl->active = nil;
}

void EncodingContext::Retain(id object) {
    if (object) {
        impl->retained->objects.push_back(object);
    }
}

void EncodingContext::Retain(MemoryAllocator::GPUBufferAccess access) {
    MakeResident(access.GetBuffer());
    impl->retained->mapped_ranges.push_back(std::move(access));
}

void EncodingContext::Retain(ResourcePool::BindingLease table) {
    if (!table.Get()) {
        throw std::invalid_argument("Cannot retain an expired Metal argument table");
    }
    impl->retained->bindings.push_back(std::move(table));
}

void EncodingContext::Retain(std::shared_ptr<void> owner) {
    if (owner) {
        impl->retained->owners.push_back(std::move(owner));
    }
}

void EncodingContext::Checkpoint(std::string_view label) {
    if (impl->active || label.empty() || !ToNSString(label)) {
        throw std::invalid_argument("Metal checkpoint requires a valid label between passes");
    }
    auto& checkpoints = impl->retained->checkpoints;
    const auto index = checkpoints.size() % CheckpointsPerHeap;
    id<MTL4CounterHeap> heap;
    if (index == 0) {
        auto descriptor = [MTL4CounterHeapDescriptor new];
        descriptor.type = MTL4CounterHeapTypeTimestamp;
        descriptor.count = CheckpointsPerHeap;
        NSError* error = nil;
        auto allocated = [GetCommandBuffer().device newCounterHeapWithDescriptor:descriptor
                                                                           error:&error];
        heap = CheckResource(allocated, "Create Metal timestamp checkpoint storage", error);
        heap.label = @"Metaliris GPU checkpoints";
        [heap invalidateCounterRange:NSMakeRange(0, CheckpointsPerHeap)];
    } else {
        heap = checkpoints.back().heap;
    }
    checkpoints.push_back({{std::string(label), {}}, heap, index});
    [GetCommandBuffer() pushDebugGroup:ToNSString(label)];
    [GetCommandBuffer() writeTimestampIntoHeap:heap atIndex:index];
    [GetCommandBuffer() popDebugGroup];
}

struct Scheduler::State {
    struct Packet {
        u64 sequence{};
        u64 tick{};
        QueueType queue{};
        DeviceContext::WorkLease device;
        std::vector<std::unique_ptr<Command>> commands;
        std::optional<SubmissionOptions> submit;
    };

    struct Failure {
        SubmissionStage stage;
        NSError* error;
    };

    explicit State(DeviceContext& device, std::size_t command_capacity,
                   std::size_t binding_capacity)
        : device(device), completion(std::make_shared<MasterSemaphore>(device.GetDevice())),
          pool(device.GetDevice(), completion, command_capacity, binding_capacity),
          queues{MakeQueue(device.GetDevice(), @"Metaliris graphics"),
                 MakeQueue(device.GetDevice(), @"Metaliris transfers")} {
        worker = std::jthread([this](std::stop_token stop) { Run(stop); });
    }

    void StartBatch() {
        if (!producer_tick) {
            const u64 last = completion->Timeline().LastIssuedTick();
            if (last == std::numeric_limits<u64>::max()) {
                throw std::overflow_error("Metal scheduler exhausted its tickets");
            }
            producer_device = device.AcquireWork();
            producer_tick = last + 1;
        }
    }

    u64 Publish(std::optional<SubmissionOptions> submit) {
        if (on_dispatch) {
            on_dispatch();
        }
        std::unique_lock lock(mutex);
        if (closing || published == std::numeric_limits<u64>::max()) {
            throw std::logic_error("Metal worker cannot accept more packets");
        }
        auto& packet = packets.emplace_back();
        if (submit) {
            try {
                // The worker cannot consume the packet until the matching ticket is issued.
                static_cast<void>(completion->Issue());
            } catch (...) {
                packets.pop_back();
                throw;
            }
        }
        const auto tick = producer_tick;
        packet.sequence = ++published;
        packet.tick = tick;
        packet.queue = producer_queue;
        packet.device = producer_device;
        packet.submit = submit;
        packet.commands.swap(commands);
        if (submit) {
            producer_tick = 0;
            producer_device = {};
        }
        lock.unlock();
        changed.notify_all();
        return tick;
    }

    void Begin(const Packet& packet, std::stop_token stop) {
        const auto retired = completion->Timeline().RetiredTick();
        for (auto it = history.begin(); it != history.end();) {
            std::erase_if(it->second, [&](const TrackedUse& use) { return use.tick <= retired; });
            if (it->second.empty()) {
                it = history.erase(it);
            } else {
                ++it;
            }
        }

        while (!stop.stop_requested()) {
            if (auto lease = pool.AcquireCommand()) {
                id<MTL4CommandBuffer> buffer = CheckResource([device.GetDevice() newCommandBuffer],
                                                             "Create Metal 4 command buffer");
                buffer.label =
                    [NSString stringWithFormat:@"Metaliris submission %llu", packet.tick];
                auto impl = std::make_unique<EncodingContext::Impl>(
                    std::move(*lease), pool, device, packet.device, history, packet.tick,
                    packet.queue, buffer);
                active.reset(new EncodingContext(std::move(impl)));
                return;
            }
            const auto earliest = completion->Timeline().RetiredTick() + 1;
            static_cast<void>(
                completion->Timeline().WaitRetired(earliest, Clock::now() + WaitInterval));
        }
        throw std::runtime_error("Metal scheduler stopped while waiting for a command allocator");
    }

    CompletionStatus WaitForDependency(u64 tick, std::stop_token stop) {
        if (tick == 0) {
            return CompletionStatus::Complete;
        }
        while (!stop.stop_requested()) {
            const auto status = completion->Timeline().Wait(tick, Clock::now() + WaitInterval);
            if (status != CompletionStatus::TimedOut) {
                return status;
            }
        }
        return CompletionStatus::Closed;
    }

    void Commit(const Packet& packet, std::stop_token stop) {
        auto& encoding = *active->impl;
        const auto& options = *packet.submit;
        const auto& queue = queues[QueueIndex(packet.queue)];
        SubmissionStage stage = SubmissionStage::Encoding;
        auto failure = std::optional<Failure>{};
        MTL4CommitOptions* native_options = nil;
        id<MTL4CommandBuffer> buffer = encoding.retained->buffer;
        std::optional<DeviceContext::SubmissionGuard> guard;
        const auto other = QueueIndex(packet.queue) ^ 1;
        u64 cross_queue_tick = encoding.cross_queue_tick;
        const auto retired = completion->Timeline().RetiredTick();
        std::erase_if(routed_ticks,
                      [retired](const auto& route) { return route.first <= retired; });
        if (const auto route = routed_ticks.find(options.wait_for_tick);
            route != routed_ticks.end() && QueueIndex(route->second) == other) {
            cross_queue_tick = std::max(cross_queue_tick, options.wait_for_tick);
        }
        try {
            if (encoding.active) {
                throw std::logic_error("A Metal encoder was left open at submission");
            }
            if (!encoding.residency.empty()) {
                auto descriptor = [MTLResidencySetDescriptor new];
                descriptor.initialCapacity = encoding.residency.size();
                descriptor.label = buffer.label;
                NSError* error = nil;
                auto residency = [device.GetDevice() newResidencySetWithDescriptor:descriptor
                                                                             error:&error];
                encoding.retained->residency =
                    CheckResource(residency, "Create Metal submission residency", error);
                for (const auto& [identity, resource] : encoding.residency) {
                    [residency addAllocation:resource];
                }
                [residency commit];
                [buffer useResidencySet:residency];
            }
            active->Retain(options.drawable);
            [buffer endCommandBuffer];
            encoding.ended = true;

            stage = SubmissionStage::Dependency;
            const u64 wait = std::max(options.wait_for_tick, cross_queue_tick);
            const auto status = WaitForDependency(wait, stop);
            if (stop.stop_requested()) {
                stage = SubmissionStage::Canceled;
                throw std::runtime_error("Metal shutdown before queue submission");
            }
            if (status != CompletionStatus::Complete) {
                throw std::runtime_error("Required Metal producer did not complete successfully");
            }
            stage = SubmissionStage::DeviceGeneration;
            auto submission = device.AcquireSubmission(packet.device);
            if (!submission) {
                throw std::runtime_error("Stale or closed Metal device work lease");
            }
            guard.emplace(std::move(*submission));

            stage = SubmissionStage::Commit;
            auto& retained = encoding.retained;
            retained->owners.reserve(retained->owners.size() + retained->bindings.size() + 1);
            for (auto& binding : retained->bindings) {
                retained->owners.push_back(binding.RetireWith(packet.tick));
            }
            retained->owners.push_back(encoding.allocator.RetireWith(packet.tick));
            const auto generation = packet.device.Generation();
            const auto tick = packet.tick;
            const auto queue_type = packet.queue;
            const auto reports = diagnostics;
            const auto completed = completion->MakeCompletionHandler(tick);
            __block auto resources = std::move(retained);
            native_options = [MTL4CommitOptions new];
            [native_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              @autoreleasepool {
                  NSError* error = feedback.error;
                  if (!feedback) {
                      error = [NSError errorWithDomain:MTL4CommandQueueErrorDomain
                                                  code:MTL4CommandQueueErrorInternal
                                              userInfo:nil];
                  }
                  try {
                      auto checkpoint = resources->LastCheckpoint(feedback != nil);
                      reports->Record({tick, queue_type, generation, SubmissionStage::Execution,
                                       error, std::move(checkpoint)});
                  } catch (...) {
                      // Diagnostic collection must not prevent resource retirement.
                  }
                  resources.reset();
                  completed(feedback);
              }
            }];
            routed_ticks.emplace(packet.tick, packet.queue);
        } catch (...) {
            native_options = nil;
            auto error = std::current_exception();
            failure = Failure{stage, DescribeException(stage, error)};
            NoteWorkerError(error);
        }

        if (failure) {
            Fail(packet, *failure);
            return;
        }

        MICROPROFILE_SCOPE(Metal_Submit);
        if (cross_queue_tick) {
            [queue.native waitForEvent:queues[other].event value:cross_queue_tick];
        }
        if (options.drawable) {
            [queue.native waitForDrawable:options.drawable];
        }
        const id<MTL4CommandBuffer> buffers[] = {buffer};
        [queue.native commit:buffers count:1 options:native_options];
        [queue.native signalEvent:queue.event value:packet.tick];
        if (options.drawable) {
            [queue.native signalDrawable:options.drawable];
            if (options.minimum_present_duration) {
                [options.drawable presentAfterMinimumDuration:*options.minimum_present_duration];
            } else {
                [options.drawable present];
            }
        }
        active.reset();
    }

    void Fail(const Packet& packet, const Failure& failure) {
        std::optional<SubmissionCheckpoint> checkpoint;
        try {
            if (active && active->impl->retained) {
                checkpoint = active->impl->retained->LastCheckpoint(false);
            }
            diagnostics->Record({packet.tick, packet.queue, packet.device.Generation(),
                                 failure.stage, failure.error, std::move(checkpoint)});
        } catch (...) {
            // Retirement is independent of diagnostic storage.
        }
        active.reset();
        static_cast<void>(completion->ReportFailure(packet.tick, failure.error));
    }

    void NoteWorkerError(std::exception_ptr error) {
        std::scoped_lock lock(mutex);
        if (!worker_error) {
            worker_error = std::move(error);
        }
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("MetalWorker");
        for (;;) {
            Packet packet;
            {
                std::unique_lock lock(mutex);
                Common::CondvarWait(changed, lock, stop, [&] { return !packets.empty(); });
                if (stop.stop_requested()) {
                    break;
                }
                packet = std::move(packets.front());
                packets.pop_front();
            }

            @autoreleasepool {
                auto stage = SubmissionStage::Allocation;
                try {
                    if (!failure) {
                        if (completion->Timeline().FirstFault()) {
                            stage = SubmissionStage::Dependency;
                            throw std::runtime_error("Earlier Metal work faulted");
                        }
                        if (!active) {
                            Begin(packet, stop);
                        }
                        stage = SubmissionStage::Encoding;
                        for (auto& command : packet.commands) {
                            if (stop.stop_requested()) {
                                stage = SubmissionStage::Canceled;
                                throw std::runtime_error("Metal encoding canceled");
                            }
                            command->Execute(*active);
                            command.reset();
                        }
                    }
                } catch (...) {
                    if (stop.stop_requested()) {
                        stage = SubmissionStage::Canceled;
                    }
                    auto error = std::current_exception();
                    failure = Failure{stage, DescribeException(stage, error)};
                    NoteWorkerError(error);
                }
                // Destructors for unexecuted captures also precede the worker sequence fence.
                packet.commands.clear();
                if (packet.submit) {
                    if (failure) {
                        Fail(packet, *failure);
                    } else {
                        Commit(packet, stop);
                    }
                    failure.reset();
                }
            }

            {
                std::scoped_lock lock(mutex);
                executed = packet.sequence;
            }
            changed.notify_all();
        }

        @autoreleasepool {
            active.reset();
            history.clear();
            std::deque<Packet> abandoned;
            {
                std::scoped_lock lock(mutex);
                abandoned.swap(packets);
            }
            for (auto& packet : abandoned) {
                packet.commands.clear();
                if (packet.submit) {
                    Fail(packet, {SubmissionStage::Canceled,
                                  MakeError(SubmissionStage::Canceled, "Metal scheduler closed")});
                }
            }
        }
        changed.notify_all();
    }

    DeviceContext& device;
    std::shared_ptr<MasterSemaphore> completion;
    ResourcePool pool;
    std::array<Queue, 2> queues;
    std::shared_ptr<Diagnostics> diagnostics = std::make_shared<Diagnostics>();

    // Producer state is owned by the recording thread.
    QueueType producer_queue{};
    u64 producer_tick{};
    DeviceContext::WorkLease producer_device;
    std::vector<std::unique_ptr<Command>> commands;
    std::function<void()> on_submit;
    std::function<void()> on_dispatch;

    // A worker packet is published, then acknowledged only after executing its CPU commands.
    std::mutex mutex;
    std::condition_variable_any changed;
    std::deque<Packet> packets;
    u64 published{};
    u64 executed{};
    bool closing{};
    std::exception_ptr worker_error;

    // Only the worker accesses encoding state or submission hazard history.
    std::unique_ptr<EncodingContext> active;
    ResourceHistory history;
    std::unordered_map<u64, QueueType> routed_ticks;
    std::optional<Failure> failure;
    std::jthread worker;
};

Scheduler::Scheduler(DeviceContext& device, std::size_t command_capacity,
                     std::size_t binding_capacity)
    : state(std::make_unique<State>(device, command_capacity, binding_capacity)) {}

Scheduler::~Scheduler() {
    Stop();
}

void Scheduler::Stop() {
    {
        std::scoped_lock lock(state->mutex);
        if (state->closing) {
            return;
        }
        state->closing = true;
    }
    state->completion->Close();
    state->changed.notify_all();
    state->worker.request_stop();
    state->worker.join();
    state->commands.clear();
    state->producer_device = {};
}

bool Scheduler::Shutdown(CompletionTimeline::Deadline deadline) {
    Stop();
    const auto& timeline = state->completion->Timeline();
    return timeline.WaitRetired(timeline.LastIssuedTick(), deadline);
}

void Scheduler::SetQueue(QueueType queue) {
    if (QueueIndex(queue) >= state->queues.size()) {
        throw std::invalid_argument("Unknown Metal command queue");
    }
    if (state->producer_tick && queue != state->producer_queue) {
        throw std::logic_error("Cannot change Metal queue while a submission is open");
    }
    state->producer_queue = queue;
}

void Scheduler::Enqueue(std::unique_ptr<Command> command) {
    {
        std::scoped_lock lock(state->mutex);
        if (state->closing) {
            throw std::logic_error("Cannot record after Metal scheduler shutdown");
        }
        if (state->worker_error) {
            std::rethrow_exception(state->worker_error);
        }
    }
    if (state->commands.size() >= CommandsPerChunk) {
        DispatchWork();
    }
    state->StartBatch();
    state->commands.push_back(std::move(command));
}

void Scheduler::DispatchWork() {
    if (!state->commands.empty()) {
        static_cast<void>(state->Publish(std::nullopt));
    }
}

void Scheduler::WaitWorker() {
    MICROPROFILE_SCOPE(Metal_WaitForWorker);
    DispatchWork();
    std::unique_lock lock(state->mutex);
    const u64 target = state->published;
    state->changed.wait(lock, [&] { return state->executed >= target || state->closing; });
    if (state->worker_error) {
        std::rethrow_exception(state->worker_error);
    }
    if (state->executed < target) {
        throw std::runtime_error("Metal worker closed before the CPU fence");
    }
}

u64 Scheduler::Flush(const SubmissionOptions& options) {
    {
        std::scoped_lock lock(state->mutex);
        if (state->closing) {
            throw std::logic_error("Cannot flush a closed Metal scheduler");
        }
    }
    const auto tick = CurrentTick();
    if (options.wait_for_tick >= tick ||
        (options.minimum_present_duration &&
         (!options.drawable || !std::isfinite(*options.minimum_present_duration) ||
          *options.minimum_present_duration < 0))) {
        throw std::invalid_argument("Invalid Metal dependency or presentation duration");
    }
    if (state->on_submit) {
        state->on_submit();
    }
    state->StartBatch();
    const auto submitted = state->Publish(options);
    cached_state = StateFlags::AllDirty;
    return submitted;
}

CompletionStatus Scheduler::Finish(const SubmissionOptions& options,
                                   CompletionTimeline::Deadline deadline) {
    return state->completion->Timeline().Wait(Flush(options), deadline);
}

CompletionStatus Scheduler::Wait(u64 tick, CompletionTimeline::Deadline deadline) {
    if (tick > state->completion->Timeline().LastIssuedTick() && tick == CurrentTick()) {
        {
            std::scoped_lock lock(state->mutex);
            if (state->closing) {
                return CompletionStatus::Closed;
            }
        }
        static_cast<void>(Flush());
    }
    return state->completion->Timeline().Wait(tick, deadline);
}

u64 Scheduler::CurrentTick() const {
    const u64 last = state->completion->Timeline().LastIssuedTick();
    if (last == std::numeric_limits<u64>::max()) {
        throw std::overflow_error("Metal scheduler exhausted its tickets");
    }
    return last + 1;
}

bool Scheduler::IsFree(u64 tick) const {
    return tick <= state->completion->Timeline().RetiredTick();
}

const MasterSemaphore& Scheduler::GetMasterSemaphore() const noexcept {
    return *state->completion;
}

ResourcePool& Scheduler::GetResourcePool() noexcept {
    return state->pool;
}

std::optional<SubmissionDiagnostic> Scheduler::GetFault() const {
    std::scoped_lock lock(state->diagnostics->mutex);
    return state->diagnostics->fault;
}

std::optional<SubmissionDiagnostic> Scheduler::GetLastCompletion() const {
    std::scoped_lock lock(state->diagnostics->mutex);
    return state->diagnostics->last;
}

void Scheduler::RegisterOnSubmit(std::function<void()> callback) {
    state->on_submit = std::move(callback);
}

void Scheduler::RegisterOnDispatch(std::function<void()> callback) {
    state->on_dispatch = std::move(callback);
}

} // namespace Metal
