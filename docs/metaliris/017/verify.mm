#import <Foundation/Foundation.h>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <thread>
#include <sys/mman.h>
#include "common/memory_detect.h"
#include "video_core/renderer_metal/mtl_memory_util.h"

using namespace Metal;

namespace {

template <typename Exception, typename F>
void Reject(F f) {
    try {
        f();
        assert(false);
    } catch (const Exception&) {
    }
}

void CheckBudget() {
    MemoryBudget budget(100);
    MemoryBudget alias = budget;
    auto stream = budget.TryReserve(MemoryPool::Stream, 32, 48);
    auto texture = alias.TryReserve(MemoryPool::Texture, 40, 48);
    assert(stream && texture);
    assert(!alias.TryReserve(MemoryPool::Transient, 1, 5));
    auto usage = budget.GetUsage();
    assert(usage.physical_used == 96 && usage.physical_limit == 100);
    assert((usage.logical_used == std::array<u64, 4>{32, 40, 0, 0}));
    assert(!stream->AdjustPhysical(56) && budget.GetUsage().physical_used == 96);
    assert(stream->AdjustPhysical(52) && budget.GetUsage().physical_used == 100);
    assert(stream->AdjustPhysical(50) && budget.GetUsage().physical_used == 98);
    *stream = std::move(*texture);
    assert(budget.GetUsage().physical_used == 48);
    texture.reset();
    stream.reset();
    const auto initial = budget.GetUsage();
    assert(initial.physical_used == 0);
    auto imported = budget.TryReserve(MemoryPool::Readback, std::numeric_limits<u64>::max(), 0);
    assert(imported && budget.GetUsage().physical_used == 0);
    assert(!budget.TryReserve(MemoryPool::Readback, 1, 0));
    Reject<std::invalid_argument>([&] { (void)budget.TryReserve(MemoryPool::Count, 1, 0); });
    Reject<std::invalid_argument>([&] { (void)budget.TryReserve(MemoryPool::Stream, 0, 1); });
    imported.reset();

    MemoryBudget parallel(8);
    std::atomic<int> entered{0};
    std::atomic<int> successes{0};
    std::atomic<bool> release{false};
    std::array<std::jthread, 16> threads;
    for (auto& thread : threads) {
        thread = std::jthread([&] {
            auto reservation = parallel.TryReserve(MemoryPool::Stream, 1, 1);
            if (reservation) {
                successes.fetch_add(1);
            }
            entered.fetch_add(1);
            entered.notify_all();
            release.wait(false);
        });
    }
    for (int count = entered.load(); count != static_cast<int>(threads.size());
         count = entered.load()) {
        entered.wait(count);
    }
    assert(successes.load() == 8 && parallel.GetUsage().physical_used == 8);
    release.store(true);
    release.notify_all();
    for (auto& thread : threads) {
        thread.join();
    }
    assert(parallel.GetUsage().physical_used == 0);
}

void CheckHostRange() {
    const auto page = Common::GetPageSize();
    void* const addr = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    assert(addr != MAP_FAILED);
    auto* bytes = static_cast<std::byte*>(addr);
    assert(IsImportableHostRange({bytes, 2 * page}));
    assert(IsImportableHostRange({bytes + page, page}));
    assert(!IsImportableHostRange({bytes + 1, page}));
    assert(!IsImportableHostRange({bytes, page - 1}));
    assert(!IsImportableHostRange({}));
    assert(mprotect(bytes + page, page, PROT_READ) == 0);
    assert(IsImportableHostRange({bytes, page}));
    assert(!IsImportableHostRange({bytes, 2 * page}));
    assert(!IsImportableHostRange({bytes + page, page}));
    assert(munmap(addr, 2 * page) == 0);
}

void CheckDevice(id<MTLDevice> device) {
    const auto page = Common::GetPageSize();
    const auto cap = 32 * page;
    MemoryBudget budget(cap);
    auto pressure_charge = budget.TryReserve(MemoryPool::Transient, 1, cap);
    assert(pressure_charge);
    int recoveries = 0;
    MemoryAllocator allocator(device, budget, [&] {
        ++recoveries;
        pressure_charge.reset();
    });
    {
        auto buffer = allocator.AllocateBuffer(page, MemoryPool::Stream);
        assert(recoveries == 1 && buffer.Size() == page);
        auto mapped = buffer.MapWrite({0, 16});
        mapped.Bytes().front() = std::byte{0x68};
        Reject<std::logic_error>([&] { (void)buffer.AcquireGPUAccess({0, 1}, BufferAccess::Read); });
        Reject<std::logic_error>([&] { (void)buffer.MapRead({15, 2}); });
        auto moved = std::move(mapped);
        assert(mapped.Bytes().empty() && moved.Bytes().front() == std::byte{0x68});
        const auto independent = buffer.AcquireGPUAccess({16, 16}, BufferAccess::ReadWrite);
        assert(independent.GetBuffer().allocatedSize == budget.GetUsage().physical_used);
        Reject<std::out_of_range>([&] { (void)buffer.MapWrite({page, 1}); });
    }
    assert(budget.GetUsage().physical_used == 0);
    {
        const auto buffer = allocator.AllocateBuffer(page, MemoryPool::Transient,
                                                     MTLStorageModePrivate);
        Reject<std::logic_error>([&] { (void)buffer.MapRead({0, page}); });
        Reject<std::logic_error>([&] { (void)buffer.MapWrite({0, page}); });
    }
    std::weak_ptr<void> imported_memory;
    @autoreleasepool {
        void* data = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        assert(data != MAP_FAILED);
        auto owner = std::shared_ptr<void>(data, [page](void* ptr) { munmap(ptr, page); });
        imported_memory = owner;
        auto* bytes = static_cast<std::byte*>(data);
        Reject<std::invalid_argument>([&] {
            (void)allocator.ImportSharedBuffer({bytes + 1, page - 1}, owner, MemoryPool::Readback);
        });
        const auto gpu = [&] {
            auto buffer = allocator.ImportSharedBuffer({bytes, page}, std::move(owner),
                                                        MemoryPool::Readback);
            return buffer.AcquireGPUAccess({0, page}, BufferAccess::Write);
        }();
        assert(!imported_memory.expired());
        assert(gpu.GetBuffer().contents == data);
        assert(budget.GetUsage().physical_used == 0 && budget.GetUsage().logical_used[2] == page);
    }
    assert(imported_memory.expired() && budget.GetUsage().logical_used[2] == 0);
}

} // namespace

int main() {
    @autoreleasepool {
        CheckBudget();
        CheckHostRange();
        Reject<MetalError>([] { MemoryAllocator allocator(nil, MemoryBudget{8}); });
        std::puts("PASS unified budget concurrency, overflow, rollback and owned VM range validation");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::puts("PENDING real buffer allocation, scoped GPU/CPU overlap and zero-copy lifetime");
            return 2;
        }
        CheckDevice(device);
        std::puts("PASS real buffer allocation, scoped GPU/CPU overlap and zero-copy lifetime");
    }
}
