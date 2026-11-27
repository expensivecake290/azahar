// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <vector>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include "common/memory_detect.h"
#include "video_core/renderer_metal/mtl_memory_util.h"

namespace Metal {

struct MemoryBudget::State {
    explicit State(u64 limit) : usage{.physical_limit = limit} {}

    std::mutex mutex;
    MemoryUsage usage;
};

MemoryBudget::Reservation::Reservation(std::shared_ptr<State> state, MemoryPool pool, u64 logical,
                                       u64 physical)
    : state(std::move(state)), pool(pool), logical(logical), physical(physical) {}

MemoryBudget::Reservation::Reservation(Reservation&& other) noexcept
    : state(std::move(other.state)), pool(other.pool), logical(other.logical),
      physical(other.physical) {}

MemoryBudget::Reservation& MemoryBudget::Reservation::operator=(Reservation&& other) noexcept {
    if (this != &other) {
        Release();
        state = std::move(other.state);
        pool = other.pool;
        logical = other.logical;
        physical = other.physical;
    }
    return *this;
}

MemoryBudget::Reservation::~Reservation() {
    Release();
}

void MemoryBudget::Reservation::Release() noexcept {
    if (auto retained = std::move(state)) {
        std::scoped_lock lock(retained->mutex);
        retained->usage.logical_used[static_cast<std::size_t>(pool)] -= logical;
        retained->usage.physical_used -= physical;
    }
}

bool MemoryBudget::Reservation::AdjustPhysical(u64 size) {
    if (!state) {
        return false;
    }
    std::scoped_lock lock(state->mutex);
    auto& usage = state->usage;
    const u64 other_reservations = usage.physical_used - physical;
    if (size > usage.physical_limit - other_reservations) {
        return false;
    }
    usage.physical_used = other_reservations + size;
    physical = size;
    return true;
}

MemoryBudget::MemoryBudget(u64 physical_limit) : state(std::make_shared<State>(physical_limit)) {}

std::optional<MemoryBudget::Reservation> MemoryBudget::TryReserve(MemoryPool pool, u64 logical,
                                                                 u64 physical) const {
    const auto index = static_cast<std::size_t>(pool);
    if (index >= static_cast<std::size_t>(MemoryPool::Count) || logical == 0) {
        throw std::invalid_argument("Invalid Metal memory pool or reservation length");
    }
    std::scoped_lock lock(state->mutex);
    auto& usage = state->usage;
    if (logical > std::numeric_limits<u64>::max() - usage.logical_used[index] ||
        physical > usage.physical_limit - usage.physical_used) {
        return std::nullopt;
    }
    usage.logical_used[index] += logical;
    usage.physical_used += physical;
    return Reservation{state, pool, logical, physical};
}

MemoryUsage MemoryBudget::GetUsage() const {
    std::scoped_lock lock(state->mutex);
    return state->usage;
}

bool IsImportableHostRange(std::span<const std::byte> bytes) {
    const auto page_size = Common::GetPageSize();
    const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
    if (bytes.empty() || !address || page_size == 0 || address % page_size != 0 ||
        bytes.size() % page_size != 0 ||
        bytes.size() > std::numeric_limits<mach_vm_address_t>::max() - address) {
        return false;
    }

    const auto task = mach_task_self();
    mach_vm_address_t region_begin = address;
    mach_vm_size_t region_size{};
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto outcome = mach_vm_region(task, &region_begin, &region_size, VM_REGION_BASIC_INFO_64,
                                        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (MACH_PORT_VALID(object)) {
        mach_port_deallocate(task, object);
    }
    constexpr auto protections = VM_PROT_READ | VM_PROT_WRITE;
    if (outcome != KERN_SUCCESS || (info.protection & protections) != protections ||
        region_begin > address || address - region_begin >= region_size) {
        return false;
    }
    return bytes.size() <= region_size - (address - region_begin);
}

struct MemoryAllocator::BufferState {
    struct Use {
        BufferRange range;
        BufferAccess access;
        bool gpu;
        u64 id;
    };

    BufferState(MemoryBudget::Reservation reservation, id<MTLBuffer> buffer)
        : reservation(std::move(reservation)), buffer(buffer), length(buffer.length),
          storage(buffer.storageMode) {}

    MemoryBudget::Reservation reservation;
    const id<MTLBuffer> buffer;
    const std::size_t length;
    const MTLStorageMode storage;
    std::mutex mutex;
    std::vector<Use> uses;
    u64 next_id{1};
};

MemoryAllocator::BufferUse::BufferUse(std::shared_ptr<BufferState> state, u64 id)
    : state(std::move(state)), id(id) {}

MemoryAllocator::BufferUse::BufferUse(BufferUse&& other) noexcept
    : state(std::move(other.state)), id(other.id) {}

MemoryAllocator::BufferUse::~BufferUse() {
    if (state) {
        std::scoped_lock lock(state->mutex);
        std::erase_if(state->uses, [id = id](const auto& use) { return use.id == id; });
    }
}

id<MTLBuffer> MemoryAllocator::BufferUse::GetBuffer() const noexcept {
    return state ? state->buffer : nil;
}

MemoryAllocator::Buffer::Buffer(std::shared_ptr<BufferState> state) : state(std::move(state)) {}

std::size_t MemoryAllocator::Buffer::Size() const noexcept {
    return state ? state->length : 0;
}

MTLStorageMode MemoryAllocator::Buffer::StorageMode() const noexcept {
    return state ? state->storage : MTLStorageModePrivate;
}

MemoryAllocator::BufferUse MemoryAllocator::Buffer::Acquire(BufferRange range, BufferAccess access,
                                                            bool gpu) const {
    if (!range.Fits(Size())) {
        throw std::out_of_range("Metal buffer range is outside the allocation");
    }
    if (!gpu && state->storage != MTLStorageModeShared) {
        throw std::logic_error("CPU mapping requires Metal shared storage");
    }
    if (access != BufferAccess::Read && access != BufferAccess::Write &&
        access != BufferAccess::ReadWrite) {
        throw std::invalid_argument("Invalid Metal buffer access");
    }

    std::scoped_lock lock(state->mutex);
    for (const auto& existing : state->uses) {
        if (range.Overlaps(existing.range) && !(gpu && existing.gpu) &&
            (existing.access != BufferAccess::Read || access != BufferAccess::Read)) {
            throw std::logic_error("Overlapping CPU and GPU Metal buffer access");
        }
    }
    if (state->next_id == std::numeric_limits<u64>::max()) {
        throw std::overflow_error("Metal buffer use identifier exhausted");
    }
    const auto id = state->next_id;
    state->uses.push_back({range, access, gpu, id});
    ++state->next_id;
    return {state, id};
}

MemoryAllocator::ReadMap MemoryAllocator::Buffer::MapRead(BufferRange range) const {
    auto use = Acquire(range, BufferAccess::Read, false);
    const auto* bytes = CheckResource(static_cast<const std::byte*>(state->buffer.contents),
                                      "Map Metal buffer for reading");
    return {std::move(use), {bytes + range.offset, range.length}};
}

MemoryAllocator::WriteMap MemoryAllocator::Buffer::MapWrite(BufferRange range) const {
    auto use = Acquire(range, BufferAccess::Write, false);
    auto* bytes = CheckResource(static_cast<std::byte*>(state->buffer.contents),
                                "Map Metal buffer for writing");
    return {std::move(use), {bytes + range.offset, range.length}};
}

MemoryAllocator::GPUBufferAccess MemoryAllocator::Buffer::AcquireGPUAccess(BufferRange range,
                                                                           BufferAccess access) const {
    return {Acquire(range, access, true), range};
}

MemoryAllocator::MemoryAllocator(id<MTLDevice> device, MemoryBudget budget,
                                 std::function<void()> recover_pressure)
    : device(CheckResource(device, "Create Metal memory allocator")), budget(std::move(budget)),
      recover_pressure(std::move(recover_pressure)) {}

MemoryAllocator::Buffer MemoryAllocator::AllocateBuffer(std::size_t size, MemoryPool pool,
                                                         MTLStorageMode storage) const {
    if (storage != MTLStorageModeShared && storage != MTLStorageModePrivate) {
        throw std::invalid_argument("Unsupported Metal buffer storage mode");
    }
    return Allocate(size, pool, storage, {});
}

MemoryAllocator::Buffer MemoryAllocator::ImportSharedBuffer(std::span<std::byte> bytes,
                                                            std::shared_ptr<void> owner,
                                                            MemoryPool pool) const {
    if (!owner || owner.use_count() == 0 || owner.get() != bytes.data() ||
        !IsImportableHostRange(bytes)) {
        throw std::invalid_argument("Metal host import requires one owned writable VM page range");
    }
    return Allocate(bytes.size(), pool, MTLStorageModeShared, std::move(owner));
}

MemoryAllocator::Buffer MemoryAllocator::Allocate(std::size_t size, MemoryPool pool,
                                                   MTLStorageMode storage,
                                                   std::shared_ptr<void> host_allocation) const {
    if (size == 0 || size > device.maxBufferLength) {
        throw std::out_of_range("Metal buffer length exceeds the device limit");
    }
    const MTLResourceOptions options =
        (storage == MTLStorageModeShared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate) |
        MTLResourceHazardTrackingModeUntracked;
    const u64 estimated_size = host_allocation
                                   ? 0
                                   : std::max<u64>(size, [device heapBufferSizeAndAlignWithLength:size
                                                                                         options:options].size);
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (auto reservation = budget.TryReserve(pool, size, estimated_size)) {
            id<MTLBuffer> buffer = nil;
            if (host_allocation) {
                buffer = [device newBufferWithBytesNoCopy:host_allocation.get()
                                                   length:size
                                                  options:options
                                              deallocator:^(void*, NSUInteger) {
                                                  (void)host_allocation.get();
                                              }];
            } else {
                buffer = [device newBufferWithLength:size options:options];
            }
            if (buffer) {
                const u64 native_size = host_allocation ? 0 : buffer.allocatedSize;
                if (reservation->AdjustPhysical(native_size)) {
                    return Buffer{std::make_shared<BufferState>(std::move(*reservation), buffer)};
                }
            }
        }
        if (attempt != 0 || !recover_pressure) {
            break;
        }
        recover_pressure();
    }
    throw std::bad_alloc();
}

} // namespace Metal
