#import <Foundation/Foundation.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <latch>
#include <limits>
#include <thread>
#include <vector>
#include "video_core/renderer_metal/mtl_master_semaphore.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

auto Deadline() {
    return std::chrono::steady_clock::now() + 10s;
}

void CheckTimeline() {
    CompletionTimeline timeline;
    assert(timeline.Wait(0) == CompletionStatus::Complete);
    assert(timeline.Wait(1) == CompletionStatus::UnknownTicket);
    const auto first = timeline.Issue();
    const auto second = timeline.Issue();
    const auto third = timeline.Issue();
    assert(first == 1 && second == 2 && third == 3);
    assert(timeline.ReportCompletion(second));
    assert(!timeline.ReportCompletion(second));
    assert(!timeline.ReportCompletion(third + 1));
    assert(timeline.SuccessfulTick() == 0 && timeline.RetiredTick() == 0);
    assert(timeline.Wait(second, std::chrono::steady_clock::now()) == CompletionStatus::TimedOut);
    assert(!timeline.WaitRetired(second, std::chrono::steady_clock::now()));
    NSError* fault = [NSError errorWithDomain:@"MetalirisFixture" code:42 userInfo:nil];
    assert(timeline.ReportCompletion(third, fault));
    assert(timeline.Wait(third) == CompletionStatus::Faulted);
    assert(timeline.FirstFault()->ticket == third && timeline.FirstFault()->error == fault);
    assert(timeline.ReportCompletion(first));
    assert(timeline.RetiredTick() == third && timeline.SuccessfulTick() == second);
    assert(timeline.Wait(second) == CompletionStatus::Complete);
    assert(timeline.WaitRetired(third));

    CompletionTimeline earlier_fault;
    const auto a = earlier_fault.Issue();
    const auto b = earlier_fault.Issue();
    assert(earlier_fault.ReportCompletion(b, fault));
    assert(earlier_fault.ReportCompletion(a, fault));
    assert(earlier_fault.FirstFault()->ticket == a);
    assert(earlier_fault.RetiredTick() == b && earlier_fault.SuccessfulTick() == 0);

    CompletionTimeline parallel;
    const auto target = parallel.Issue();
    std::latch ready(24);
    std::atomic<int> released{};
    std::vector<std::jthread> waiters;
    for (int i = 0; i < 24; ++i) {
        waiters.emplace_back([&] {
            ready.count_down();
            if (parallel.Wait(target, Deadline()) == CompletionStatus::Closed) {
                ++released;
            }
        });
    }
    ready.wait();
    parallel.Close();
    waiters.clear();
    assert(released == 24 && parallel.RetiredTick() == 0);
    assert(parallel.ReportCompletion(target));
    assert(parallel.WaitRetired(target, Deadline()));
    try {
        (void)parallel.Issue();
        assert(false);
    } catch (const std::logic_error&) {
    }

    CompletionTimeline maximum(std::numeric_limits<u64>::max() - 1);
    const auto final = maximum.Issue();
    assert(final == std::numeric_limits<u64>::max());
    try {
        (void)maximum.Issue();
        assert(false);
    } catch (const std::overflow_error&) {
    }
    assert(maximum.ReportCompletion(final));
    assert(maximum.RetiredTick() == final);
}

void Copy(MasterSemaphore& timeline, id<MTLDevice> device, id<MTL4CommandQueue> queue,
          id<MTLBuffer> source, id<MTLBuffer> destination, u64 ticket,
          std::shared_ptr<void> keep_alive = {}) {
    NSError* error = nil;
    auto residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                                    error:&error];
    auto commands = [device newCommandBuffer];
    auto allocator = [device newCommandAllocator];
    assert(residency && commands && allocator);
    [residency addAllocation:source];
    [residency addAllocation:destination];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    auto encoder = [commands computeCommandEncoder];
    assert(encoder);
    [encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:4];
    [encoder endEncoding];
    [commands endCommandBuffer];
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:timeline.MakeCompletionHandler(ticket, std::move(keep_alive))];
    const id<MTL4CommandBuffer> buffers[] = {commands};
    [queue commit:buffers count:1 options:options];
    assert(timeline.Timeline().Wait(ticket, Deadline()) == CompletionStatus::Complete);
}

} // namespace

int main() {
    @autoreleasepool {
        CheckTimeline();
        std::puts("PASS CPU completion order, earliest fault, timeout, wrap prevention and concurrent shutdown");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING Metal 4 commit feedback and cross-queue copy ordering");
            return 2;
        }
        auto producer = [device newMTL4CommandQueue];
        auto consumer = [device newMTL4CommandQueue];
        auto src = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        auto shared = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        auto dst = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        assert(producer && consumer && src && shared && dst);
        *static_cast<u32*>(src.contents) = 0x7412abce;
        MasterSemaphore timeline(device);
        assert(timeline.GetDevice() == device);
        const auto checkpoint = timeline.Issue();
        auto keep_alive = std::make_shared<int>(3);
        std::weak_ptr<void> weak = keep_alive;
        Copy(timeline, device, producer, src, shared, checkpoint, std::move(keep_alive));
        assert(weak.expired());
        assert(timeline.WaitOnQueue(consumer, checkpoint, Deadline()) == CompletionStatus::Complete);
        Copy(timeline, device, consumer, shared, dst, timeline.Issue());
        assert(*static_cast<u32*>(dst.contents) == 0x7412abce);
        const auto failed = timeline.Issue();
        auto error = [NSError errorWithDomain:@"MetalirisFixture" code:9 userInfo:nil];
        assert(timeline.ReportFailure(failed, error));
        assert(timeline.WaitOnQueue(consumer, failed, Deadline()) == CompletionStatus::Faulted);
        std::puts("PASS live native feedback, keepalive release and cross-queue ordering");
    }
}
