// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include "common/apple_utils.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core_timing.h"
#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_present_window.h"
#include "video_core/renderer_metal/mtl_shader_util.h"

namespace Metal {
namespace {

MICROPROFILE_DEFINE(Metal_WaitPresent, "Metal", "Wait For Present", MP_RGB(128, 128, 128));

constexpr u32 MaxFrameDimension = 16384;

constexpr std::string_view scale_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ScreenVertex {
    float4 position [[position]];
    float2 uv;
};

vertex ScreenVertex scale_vertex(uint index [[vertex_id]]) {
    const float2 uv(float((index & 1u) << 1u), float(index & 2u));
    return {float4(fma(uv, float2(2, -2), float2(-1, 1)), 0, 1), uv};
}

fragment float4 scale_fragment(ScreenVertex in [[stage_in]],
                               texture2d<float> image [[texture(0)]],
                               sampler filter [[sampler(0)]]) {
    return image.sample(filter, in.uv, level(0.0));
}
)MSL";

struct PresentationSettings {
    PresentPacing pacing;
    bool detect_refresh;
    bool linear;
};

PresentationSettings GetSettings() {
    PresentationSettings settings{
        .pacing = {.vsync = Settings::values.use_vsync.GetValue(),
                   .frame_limit = Settings::GetFrameLimit()},
        .detect_refresh = Settings::values.use_display_refresh_rate_detection.GetValue(),
        .linear = Settings::values.filter_mode.GetValue(),
    };
    auto& pacing = settings.pacing;
    if (pacing.vsync && pacing.frame_limit && pacing.frame_limit <= 100) {
        pacing.minimum_duration = 100. / (pacing.frame_limit * SCREEN_REFRESH_RATE);
    }
    return settings;
}

bool EqualExtent(VideoCore::Extent lhs, VideoCore::Extent rhs) {
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

NSError* PresentError(std::string_view message) {
    auto detail = [[NSString alloc] initWithBytes:message.data()
                                           length:message.size()
                                         encoding:NSUTF8StringEncoding];
    return [NSError errorWithDomain:@"Azahar.Metal.Presentation"
                               code:1
                           userInfo:detail ? @{NSLocalizedDescriptionKey : detail} : nil];
}

} // Anonymous namespace

struct PresentWindow::Allocation {
    id<MTLTexture> render_image;
    id<MTLTexture> resolved_image;
    MemoryBudget::Reservation render_charge;
    std::optional<MemoryBudget::Reservation> resolve_charge;
    DeviceContext::WorkLease device;
    VideoCore::Extent extent;
    u32 samples;
};

struct PresentWindow::State : std::enable_shared_from_this<State> {
    struct Slot {
        std::shared_ptr<Allocation> image;
        bool busy{};
    };

    struct Job {
        u64 rendered;
        u64 ticket;
        u32 slot;
        std::shared_ptr<Allocation> image;
        PresentationSettings settings;
    };

    State(Frontend::EmuWindow& window, Instance& instance, Scheduler& rendering,
          MemoryBudget& budget, bool asynchronous)
        : window(window), instance(instance), rendering(rendering), budget(budget),
          asynchronous(asynchronous), presentation(instance.GetContext()),
          swapchain(instance, presentation), descriptors(instance.GetContext()),
          compiler(instance.GetContext()), settings(GetSettings()) {}

    void Start() {
        if (asynchronous) {
            worker = std::thread([this] { Run(); });
        }
    }

    std::pair<id<MTLTexture>, MemoryBudget::Reservation> MakeTexture(VideoCore::Extent extent,
                                                                     u32 samples) {
        auto device = instance.GetDevice();
        auto descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
        descriptor.width = extent.width;
        descriptor.height = extent.height;
        descriptor.pixelFormat = FrameColorFormat;
        descriptor.sampleCount = samples;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        const auto size = [device heapTextureSizeAndAlignWithDescriptor:descriptor].size;
        auto charge = budget.TryReserve(MemoryPool::Texture, size, size);
        if (!charge) {
            throw std::runtime_error("Metal presentation texture exceeds the memory budget");
        }
        auto texture = CheckResource([device newTextureWithDescriptor:descriptor],
                                     "Allocate presentation texture");
        if (!charge->AdjustPhysical(texture.allocatedSize)) {
            throw std::runtime_error("Metal presentation texture exceeds the physical budget");
        }
        texture.label = @"Metaliris presentation frame";
        return {texture, std::move(*charge)};
    }

    std::shared_ptr<Allocation> Allocate(VideoCore::Extent extent, u32 samples,
                                         DeviceContext::WorkLease lease) {
        auto [image, charge] = MakeTexture(extent, samples);
        auto resolved = image;
        std::optional<MemoryBudget::Reservation> resolve_charge;
        if (samples > 1) {
            auto [target, target_charge] = MakeTexture(extent, 1);
            resolved = target;
            resolve_charge.emplace(std::move(target_charge));
        }
        return std::make_shared<Allocation>(Allocation{image, resolved, std::move(charge),
                                                       std::move(resolve_charge), std::move(lease),
                                                       extent, samples});
    }

    std::optional<Frame> Checkout(VideoCore::Extent extent, u32 samples,
                                  CompletionTimeline::Deadline deadline) {
        if (!extent.width || !extent.height || extent.width > MaxFrameDimension ||
            extent.height > MaxFrameDimension || !instance.SupportsSampleCount(samples)) {
            return {};
        }
        auto& context = instance.GetContext();
        const auto lease = context.AcquireWork();
        if (context.GetLeaseStatus(lease) != WorkLeaseStatus::Valid) {
            return {};
        }
        std::unique_lock lock(mutex);
        const auto ready = [this] {
            return closing ||
                   std::ranges::any_of(slots, [](const Slot& slot) { return !slot.busy; });
        };
        {
            MICROPROFILE_SCOPE(Metal_WaitPresent);
            if (deadline) {
                if (!changed.wait_until(lock, *deadline, ready)) {
                    return {};
                }
            } else {
                changed.wait(lock, ready);
            }
        }
        if (closing || context.GetLeaseStatus(lease) != WorkLeaseStatus::Valid) {
            return {};
        }
        const auto it = std::ranges::find(slots, false, &Slot::busy);
        const auto index = static_cast<u32>(it - slots.begin());
        it->busy = true;
        auto image = it->image;
        if (image && EqualExtent(image->extent, extent) && image->samples == samples &&
            context.GetLeaseStatus(image->device) == WorkLeaseStatus::Valid) {
            return Frame(weak_from_this(), std::move(image), index);
        }
        it->image.reset();
        image.reset();
        lock.unlock();

        try {
            image = Allocate(extent, samples, lease);
        } catch (...) {
            Return(index, false);
            throw;
        }
        lock.lock();
        if (closing || context.GetLeaseStatus(lease) != WorkLeaseStatus::Valid) {
            it->busy = false;
            lock.unlock();
            changed.notify_all();
            return {};
        }
        it->image = image;
        return Frame(weak_from_this(), std::move(image), index);
    }

    void Return(u32 slot, bool reuse) {
        {
            std::scoped_lock lock(mutex);
            if (!reuse || closing) {
                slots[slot].image.reset();
            }
            slots[slot].busy = false;
        }
        changed.notify_all();
    }

    bool Submit(Frame frame, u64 tick) {
        if (!frame || frame.owner.lock().get() != this || tick == 0 ||
            tick > rendering.GetMasterSemaphore().Timeline().LastIssuedTick()) {
            return false;
        }
        const auto captured_settings = GetSettings();
        {
            std::scoped_lock lock(mutex);
            if (closing || instance.GetContext().GetLeaseStatus(frame.allocation->device) !=
                               WorkLeaseStatus::Valid) {
                return false;
            }
            jobs.push_back({tick, 0, frame.slot, frame.allocation, captured_settings});
            try {
                jobs.back().ticket = completed.Issue();
            } catch (...) {
                jobs.pop_back();
                throw;
            }
            frame.allocation.reset();
        }
        if (asynchronous) {
            changed.notify_all();
        } else {
            Drain();
        }
        return true;
    }

    void Drain() {
        std::scoped_lock process(process_mutex);
        for (;;) {
            std::optional<Job> job;
            {
                std::scoped_lock lock(mutex);
                if (jobs.empty()) {
                    return;
                }
                job.emplace(std::move(jobs.front()));
                jobs.pop_front();
            }
            Process(std::move(*job));
        }
    }

    void Run() {
        Common::SetCurrentThreadName("MetalPresent");
        for (;;) {
            bool refresh;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [this] { return closing || surface_changed || !jobs.empty(); });
                if (closing && jobs.empty()) {
                    return;
                }
                refresh = std::exchange(surface_changed, false);
            }
            @autoreleasepool {
                if (refresh && !closing) {
                    try {
                        (void)UpdateSurface();
                    } catch (const std::exception& error) {
                        LOG_ERROR(Render, "Unable to update the Metal presentation surface: {}",
                                  error.what());
                    }
                }
                Drain();
            }
        }
    }

    bool UpdateSurface() {
        auto snapshot = window.GetNativeSurface();
        auto pacing = settings.pacing;
        const auto refresh_rate = snapshot.surface.refresh_rate;
        pacing.low_refresh_rate =
            settings.detect_refresh && (AppleUtils::IsLowPowerModeEnabled() ||
                                        (refresh_rate > 0. && refresh_rate < SCREEN_REFRESH_RATE));
        return swapchain.UpdateSurface(std::move(snapshot), pacing);
    }

    void Notify() {
        if (asynchronous) {
            {
                std::scoped_lock lock(mutex);
                surface_changed = !closing;
            }
            changed.notify_all();
        }
    }

    bool MoreFramesPending() {
        std::scoped_lock lock(mutex);
        return !jobs.empty();
    }

    id<MTLRenderPipelineState> ScalePipeline(MTLPixelFormat format) {
        auto& context = instance.GetContext();
        if (!scale_shader.GetLibrary(context)) {
            scale_shader = compiler.CompileLibrary(
                {std::string(scale_source), "Metaliris present scale", {}, {}});
            scale_pipeline = nil;
            samplers = {};
        }
        auto library = scale_shader.GetLibrary(context);
        if (!library) {
            throw std::runtime_error("Metal presentation MSL failure: " +
                                     scale_shader.GetDiagnostic().message);
        }
        if (scale_pipeline && scale_format == format) {
            return scale_pipeline;
        }
        auto vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = library;
        vertex.name = @"scale_vertex";
        auto fragment = [MTL4LibraryFunctionDescriptor new];
        fragment.library = library;
        fragment.name = @"scale_fragment";
        auto desc = [MTL4RenderPipelineDescriptor new];
        desc.vertexFunctionDescriptor = vertex;
        desc.fragmentFunctionDescriptor = fragment;
        desc.colorAttachments[0].pixelFormat = format;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        const auto lease = context.AcquireWork();
        const auto native =
            CheckResource(compiler.GetNativeCompiler(lease), "Create present compiler");
        auto guard = context.AcquireSubmission(lease);
        if (!guard) {
            throw std::runtime_error(
                "Metal presentation shader compiler belongs to a retired device");
        }
        NSError* error = nil;
        scale_pipeline = CheckResource([native newRenderPipelineStateWithDescriptor:desc
                                                                compilerTaskOptions:nil
                                                                              error:&error],
                                       "Compile native presentation scale", error);
        scale_format = format;
        return scale_pipeline;
    }

    id<MTLSamplerState> Sampler(bool linear) {
        auto& sampler = samplers[linear ? 1 : 0];
        if (!sampler) {
            auto descriptor = [MTLSamplerDescriptor new];
            descriptor.minFilter =
                linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            descriptor.magFilter = descriptor.minFilter;
            descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.supportArgumentBuffers = YES;
            sampler = CheckResource([instance.GetDevice() newSamplerStateWithDescriptor:descriptor],
                                    "Create native presentation sampler");
        }
        return sampler;
    }

    std::optional<u64> PresentJob(const Job& job) {
        auto acquired = swapchain.AcquireNextImage();
        if (acquired.status == DrawableStatus::SurfaceChanged && UpdateSurface()) {
            acquired = swapchain.AcquireNextImage();
        }
        if (acquired.status == DrawableStatus::DeviceUnavailable) {
            throw std::runtime_error("Metal presentation device became unavailable");
        }
        if (closing || !acquired.drawable) {
            return {};
        }
        auto& drawable = *acquired.drawable;
        const auto format = drawable.Texture().pixelFormat;
        const auto path = SelectPresentTransfer(job.image->extent, drawable.Extent(),
                                                FrameColorFormat, format, settings.linear);
        const auto source = job.image->resolved_image;
        if (path == PresentTransfer::Invalid) {
            throw std::invalid_argument("Invalid native presentation transfer");
        }
        if (path == PresentTransfer::Copy) {
            return swapchain.Present(
                std::move(drawable),
                [source, frame = job.image](EncodingContext& context, id<MTLTexture> target) {
                    const std::array uses{
                        ResourceUse{source, BufferAccess::Read, MTLStageBlit},
                        ResourceUse{target, BufferAccess::Write, MTLStageBlit},
                    };
                    auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                                 "Encode drawable copy in Metal 4");
                    context.Synchronize(encoder, uses);
                    [encoder copyFromTexture:source toTexture:target];
                    context.Retain(frame);
                    context.EndPass(encoder);
                });
        }

        const auto pipeline = ScalePipeline(format);
        const auto set = descriptors.CreateSet({.textures = 1, .samplers = 1});
        descriptors.AddImageSampler(set, 0, 0, source, Sampler(path == PresentTransfer::Linear));
        return swapchain.Present(std::move(drawable), [pipeline,
                                                       arguments = descriptors.Capture(set),
                                                       frame = job.image](EncodingContext& context,
                                                                          id<MTLTexture> target) {
            auto pass = [MTL4RenderPassDescriptor new];
            pass.colorAttachments[0].texture = target;
            pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            auto encoder =
                CheckResource([context.GetCommandBuffer() renderCommandEncoderWithDescriptor:pass],
                              "Encode drawable scaling in Metal 4");
            const ResourceUse use{target, BufferAccess::Write, MTLStageFragment};
            context.Synchronize(encoder, std::span{&use, 1});
            [encoder setRenderPipelineState:pipeline];
            arguments.BindRender(context, encoder, MTLRenderStageFragment);
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            context.Retain(frame);
            context.Retain(pipeline);
            context.EndPass(encoder);
        });
    }

    void Process(Job job) {
        @autoreleasepool {
            settings = job.settings;
            NSError* error = nil;
            bool reuse = false;
            try {
                const auto& render_timeline = rendering.GetMasterSemaphore().Timeline();
                const auto source = render_timeline.Wait(job.rendered);
                reuse = render_timeline.WaitRetired(job.rendered);
                if (source != CompletionStatus::Complete || !reuse) {
                    const auto fault = render_timeline.FirstFault();
                    error = fault && fault->error
                                ? fault->error
                                : PresentError("Metal source frame did not complete");
                } else if (!closing) {
                    if (!UpdateSurface()) {
                        throw std::runtime_error("Metal presentation surface configuration failed");
                    }
                    const bool superseded =
                        swapchain.GetPresentMode() == PresentMode::Mailbox && MoreFramesPending();
                    const auto ticket = superseded ? std::nullopt : PresentJob(job);
                    if (ticket) {
                        const auto& timeline = presentation.GetMasterSemaphore().Timeline();
                        const auto status = timeline.Wait(*ticket);
                        reuse = timeline.WaitRetired(*ticket);
                        if (status != CompletionStatus::Complete || !reuse) {
                            const auto fault = timeline.FirstFault();
                            error =
                                fault && fault->error
                                    ? fault->error
                                    : PresentError("Metal drawable submission did not complete");
                        }
                    }
                }
            } catch (const std::exception& exception) {
                error = PresentError(exception.what());
            }
            if (error) {
                LOG_ERROR(Render, "Metal presentation failed: {}",
                          error.localizedDescription.UTF8String);
            }
            Return(job.slot, reuse && !error);
            (void)completed.ReportCompletion(job.ticket, error);
        }
    }

    void Close() {
        std::scoped_lock close(close_mutex);
        {
            std::scoped_lock lock(mutex);
            if (closing.exchange(true)) {
                return;
            }
        }
        changed.notify_all();
        if (worker.joinable()) {
            worker.join();
        } else {
            Drain();
        }
        swapchain.Close();
        (void)presentation.Shutdown();
        {
            std::scoped_lock lock(mutex);
            for (auto& slot : slots) {
                slot.image.reset();
            }
        }
        completed.Close();
    }

    Frontend::EmuWindow& window;
    Instance& instance;
    Scheduler& rendering;
    MemoryBudget budget;
    const bool asynchronous;
    Scheduler presentation;
    Swapchain swapchain;
    DescriptorUpdateQueue descriptors;
    ShaderCompiler compiler;
    ShaderLibrary scale_shader;
    id<MTLRenderPipelineState> scale_pipeline = nil;
    MTLPixelFormat scale_format = MTLPixelFormatInvalid;
    std::array<id<MTLSamplerState>, 2> samplers{};
    PresentationSettings settings;
    CompletionTimeline completed;
    std::array<Slot, FrameCount> slots;
    std::deque<Job> jobs;
    std::condition_variable changed;
    std::mutex mutex;
    std::mutex process_mutex;
    std::mutex close_mutex;
    std::atomic_bool closing{};
    bool surface_changed{true};
    std::thread worker;
};

PresentWindow::Frame::Frame(std::weak_ptr<State> owner, std::shared_ptr<Allocation> allocation,
                            u32 slot)
    : owner(std::move(owner)), allocation(std::move(allocation)), slot(slot) {}

PresentWindow::Frame::Frame(Frame&& other) noexcept = default;

PresentWindow::Frame& PresentWindow::Frame::operator=(Frame&& other) noexcept {
    if (this != &other) {
        Release();
        owner = std::move(other.owner);
        allocation = std::move(other.allocation);
        slot = other.slot;
    }
    return *this;
}

PresentWindow::Frame::~Frame() {
    Release();
}

void PresentWindow::Frame::Release() {
    if (allocation) {
        if (const auto window = owner.lock()) {
            window->Return(slot, false);
        }
        allocation.reset();
    }
}

VideoCore::Extent PresentWindow::Frame::Extent() const noexcept {
    return allocation ? allocation->extent : VideoCore::Extent{0, 0};
}

u32 PresentWindow::Frame::Samples() const noexcept {
    return allocation ? allocation->samples : 0;
}

id<MTLTexture> PresentWindow::Frame::Texture() const noexcept {
    return allocation ? allocation->resolved_image : nil;
}

RenderPass PresentWindow::Frame::MakePass(MTLClearColor clear) const {
    if (!allocation) {
        throw std::logic_error("Cannot render into a released Metal presentation frame");
    }
    const auto extent = allocation->extent;
    RenderPass pass;
    pass.area = {0, 0, extent.width, extent.height};
    pass.width = extent.width;
    pass.height = extent.height;
    pass.samples = allocation->samples;
    pass.color.target.texture = allocation->render_image;
    pass.color.load_action = MTLLoadActionClear;
    pass.clear_color = clear;
    if (pass.samples > 1) {
        pass.color.resolve.texture = allocation->resolved_image;
        pass.color.store_action = MTLStoreActionMultisampleResolve;
    }
    return pass;
}

void PresentWindow::Frame::Retain(Scheduler& scheduler) const {
    if (!allocation) {
        throw std::invalid_argument("Cannot retain an empty Metal presentation frame");
    }
    scheduler.Record([image = allocation](EncodingContext& context) {
        if (!context.OwnsWork(image->device)) {
            throw std::invalid_argument(
                "Metal presentation frame belongs to a different work generation");
        }
        context.Retain(image);
    });
}

PresentWindow::PresentWindow(Frontend::EmuWindow& window, Instance& instance, Scheduler& rendering,
                             MemoryBudget& budget, bool async_presentation)
    : state(std::make_shared<State>(window, instance, rendering, budget, async_presentation)) {
    state->Start();
}

PresentWindow::~PresentWindow() {
    Close();
}

std::optional<PresentWindow::Frame> PresentWindow::GetRenderFrame(
    VideoCore::Extent extent, u32 samples, CompletionTimeline::Deadline deadline) {
    return state->Checkout(extent, samples, deadline);
}

bool PresentWindow::Present(Frame frame, u64 rendered_tick) {
    return state->Submit(std::move(frame), rendered_tick);
}

CompletionStatus PresentWindow::WaitPresent(CompletionTimeline::Deadline deadline) {
    const auto ticket = state->completed.LastIssuedTick();
    const auto result = state->completed.Wait(ticket, deadline);
    if (result == CompletionStatus::Faulted && !state->completed.WaitRetired(ticket, deadline)) {
        return CompletionStatus::TimedOut;
    }
    return result;
}

void PresentWindow::NotifySurfaceChanged() {
    state->Notify();
}

void PresentWindow::Close() {
    state->Close();
}

} // namespace Metal
