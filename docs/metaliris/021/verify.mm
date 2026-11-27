#import <Foundation/Foundation.h>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include "video_core/renderer_metal/mtl_resource_pool.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

void CheckSlots() {
    ResourceSlots slots(3);
    const auto a = slots.Acquire(0);
    const auto b = slots.Acquire(0);
    const auto c = slots.Acquire(0);
    assert(a && b && c && !slots.Acquire(0));
    assert(slots.Submit(*a, 4));
    assert(slots.Submit(*b, 6));
    assert(!slots.Release(*a) && !slots.Submit(*a, 5));
    assert(slots.Release(*c));
    const auto d = slots.Acquire(3, c->index);
    assert(d && d->index == c->index && d->generation != c->generation);
    assert(!slots.Release(*c));
    assert(!slots.Acquire(3));
    const auto e = slots.Acquire(4, b->index);
    assert(e && e->index == a->index);
    assert(!slots.Submit(*e, 0));
    assert(!slots.Acquire(5));
    const auto f = slots.Acquire(6);
    assert(f && f->index == b->index);
    assert(!slots.Available(3, 9));
    assert(!slots.Available(f->index, 6));
    assert(slots.Release(*d) && slots.Release(*e) && slots.Release(*f));
    try {
        ResourceSlots invalid(0);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
}

void CheckDevice(id<MTLDevice> device) {
    auto completion = std::make_shared<MasterSemaphore>(device);
    auto queue = [device newMTL4CommandQueue];
    auto pool = std::make_unique<ResourcePool>(device, completion, 1, 2);
    auto command = pool->AcquireCommand();
    assert(queue && command && command->Get() && !pool->AcquireCommand());

    constexpr BindingLayout layout{1, 0, 0, false};
    constexpr std::array<u64, 1> x{41};
    constexpr std::array<u64, 1> y{42};
    auto owned = std::make_shared<int>(11);
    std::weak_ptr<void> weak = owned;
    unsigned initializations = 0;
    const auto init = [&](id<MTL4ArgumentTable> table) {
        assert(table.device == device);
        ++initializations;
    };
    id<MTL4ArgumentTable> original = nil;
    {
        auto first = pool->AcquireBinding(layout, x, owned, init);
        auto other = pool->AcquireBinding(layout, y, {}, init);
        assert(first && other && first->Get() != other->Get());
        assert(!pool->AcquireBinding(layout, x, {}, init));
        original = first->Get();
    }
    owned.reset();
    assert(!weak.expired() && initializations == 2);
    auto cached = pool->AcquireBinding(layout, x, {}, init);
    assert(cached && cached->Get() == original && initializations == 2);

    auto buffer = [device newCommandBuffer];
    assert(buffer);
    [buffer beginCommandBufferWithAllocator:command->Get()];
    [buffer endCommandBuffer];
    const auto ticket = completion->Issue();
    auto commands_pin = command->RetireWith(ticket);
    auto bindings_pin = cached->RetireWith(ticket);
    assert(!command->Get() && !cached->Get());
    pool->Trim();
    assert(!pool->AcquireCommand() && !weak.expired());
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:completion->MakeCompletionHandler(ticket, std::move(commands_pin))];
    bindings_pin.reset(); // Both pins hold the same pool state.
    const id<MTL4CommandBuffer> buffers[] = {buffer};
    [queue commit:buffers count:1 options:options];
    assert(completion->Timeline().Wait(ticket, std::chrono::steady_clock::now() + 10s) ==
           CompletionStatus::Complete);
    assert(pool->AcquireCommand());
    pool->Trim();
    assert(weak.expired());
}

} // namespace

int main() {
    @autoreleasepool {
        CheckSlots();
        std::puts("PASS bounded slot exhaustion, retirement, fragmentation, stale generations");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING real Metal allocator reset, immutable table reuse and pressure trim");
            return 2;
        }
        CheckDevice(device);
        std::puts("PASS real Metal allocator retirement, argument reuse and trim");
    }
}
