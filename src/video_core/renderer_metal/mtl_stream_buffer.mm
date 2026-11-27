// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include "common/literals.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_stream_buffer.h"

namespace Metal {
namespace {

using namespace Common::Literals;

std::string_view BufferTypeName(BufferType type) {
    switch (type) {
    case BufferType::Upload:
        return "Upload";
    case BufferType::Download:
        return "Download";
    case BufferType::Stream:
        return "Stream";
    }
    throw std::invalid_argument("Invalid Metal stream buffer type");
}

MemoryAllocator::Buffer AllocateStorage(const MemoryAllocator& allocator, id<MTLDevice> device,
                                        u32 preferred, BufferType type) {
    const auto type_name = BufferTypeName(type);
    const auto pool = type == BufferType::Download ? MemoryPool::Readback : MemoryPool::Stream;
    u64 attempt = std::min<u64>(preferred, device.maxBufferLength);
    if (attempt == 0) {
        throw std::invalid_argument("Metal stream buffer requires a positive capacity");
    }
    const u64 minimum = std::min<u64>(8_KiB, attempt);
    for (;;) {
        try {
            auto buffer = allocator.AllocateBuffer(attempt, pool, MTLStorageModeShared);
            auto access = buffer.AcquireGPUAccess({0, 1}, BufferAccess::Read);
            id<MTLBuffer> native = access.GetBuffer();
            if (native.device != device) {
                throw std::invalid_argument("Metal stream allocator and scheduler devices differ");
            }
            native.label = [NSString stringWithFormat:@"Metaliris %.*s stream (%llu bytes)",
                                                      static_cast<int>(type_name.size()),
                                                      type_name.data(), attempt];
            LOG_INFO(Render, "Created Metal {} stream with {} bytes", type_name, attempt);
            return buffer;
        } catch (const std::bad_alloc&) {
            if (attempt == minimum) {
                throw;
            }
        }
        attempt = std::max(minimum, attempt / 2);
    }
}

void WaitForSuccess(Scheduler& scheduler, u64 tick, CompletionTimeline::Deadline deadline) {
    switch (scheduler.Wait(tick, deadline)) {
    case CompletionStatus::Complete:
        return;
    case CompletionStatus::Faulted: {
        const auto error = scheduler.GetFault();
        throw MetalError("Wait for Metal stream bytes", error ? error->error : nil);
    }
    case CompletionStatus::TimedOut:
        throw std::runtime_error("Metal stream byte range wait expired");
    case CompletionStatus::Closed:
        throw std::runtime_error("Metal stream byte range was closed before completion");
    case CompletionStatus::UnknownTicket:
        throw std::logic_error("Metal stream byte range references an unknown submission");
    }
    throw std::logic_error("Invalid Metal stream completion status");
}

struct ReadbackState {
    bool has_writer{};
};

} // Anonymous namespace

StreamRing::StreamRing(u32 capacity) : capacity(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("Metal stream ring capacity must be positive");
    }
}

StreamRing::Placement StreamRing::Find(u32 size, u64 alignment) const {
    if (active) {
        throw std::logic_error("Metal stream ring already has a live reservation");
    }
    const auto placement = Locate(capacity, cursor, size, alignment);
    if (!placement) {
        throw std::out_of_range("Metal stream reservation exceeds the backing capacity");
    }
    return *placement;
}

StreamRing::Watches::const_iterator StreamRing::FirstOverlap(BufferRange range) const {
    if (range.length == 0) {
        return watches.end();
    }
    auto first = watches.lower_bound(range.offset);
    if (first != watches.begin()) {
        const auto previous = std::prev(first);
        if (range.Overlaps(previous->second.slice.range)) {
            return previous;
        }
    }
    return first;
}

u64 StreamRing::RequiredTick(Placement placement) const {
    u64 wait{};
    const auto range = placement.range;
    if (range.length != 0 && !range.Fits(capacity)) {
        throw std::out_of_range("Metal stream wait range exceeds the backing capacity");
    }
    for (auto watch = FirstOverlap(range);
         watch != watches.end() && range.Overlaps(watch->second.slice.range); ++watch) {
        if (!watch->second.readback_pin.expired()) {
            throw std::overflow_error("Metal stream wrap would overwrite a live readback slice");
        }
        wait = std::max(wait, watch->second.tick);
    }
    return wait;
}

StreamRing::Reservation StreamRing::Reserve(u32 size, u64 alignment, u64 completed_tick) {
    const auto placement = Find(size, alignment);
    if (RequiredTick(placement) > completed_tick) {
        throw std::logic_error("Cannot reserve Metal stream bytes before GPU retirement");
    }
    if (generation == std::numeric_limits<u64>::max()) {
        throw std::overflow_error("Metal stream has exhausted its range generations");
    }
    const auto range = placement.range;
    const auto first = FirstOverlap(range);
    auto last = first;
    while (last != watches.end() && range.Overlaps(last->second.slice.range)) {
        ++last;
    }
    watches.erase(first, last);
    active = Reservation{placement, ++generation};
    return *active;
}

StreamRing::Slice StreamRing::Commit(Reservation reservation, u32 size, u64 tick,
                                     std::weak_ptr<void> readback_pin) {
    if (!active || reservation.generation != active->generation) {
        throw std::logic_error("Cannot commit a stale or foreign Metal stream reservation");
    }
    const auto range = active->placement.range;
    if (size > range.length || (size != 0 && tick == 0)) {
        throw std::out_of_range("Invalid Metal stream commit size or ticket");
    }
    if (size == 0) {
        active.reset();
        return {{range.offset, 0}, 0};
    }
    const Slice slice{{range.offset, size}, active->generation};
    if (!watches.emplace(range.offset, Watch{slice, tick, std::move(readback_pin)}).second) {
        throw std::logic_error("Overlapping Metal stream commit");
    }
    cursor = static_cast<u32>(range.offset + size);
    active.reset();
    return slice;
}

void StreamRing::Cancel(Reservation reservation) noexcept {
    if (active && active->generation == reservation.generation) {
        active.reset();
    }
}

u64 StreamRing::UseTick(Slice slice) const {
    const auto watch = watches.find(slice.range.offset);
    if (slice.generation == 0 || watch == watches.end() ||
        watch->second.slice.generation != slice.generation ||
        watch->second.slice.range.length != slice.range.length) {
        throw std::logic_error("Metal stream slice is empty, stale or foreign");
    }
    return watch->second.tick;
}

void StreamRing::NoteUse(Slice slice, u64 tick) {
    const auto previous = UseTick(slice);
    if (tick < previous) {
        throw std::logic_error("Metal stream usage tickets must be monotonic");
    }
    watches.at(slice.range.offset).tick = tick;
}

struct StreamBuffer::State {
    State(const MemoryAllocator& allocator, Scheduler& scheduler, u32 preferred_size,
          BufferType type)
        : scheduler(scheduler), type(type),
          buffer(AllocateStorage(allocator, scheduler.GetMasterSemaphore().GetDevice(),
                                 preferred_size, type)),
          ring(static_cast<u32>(buffer.Size())) {}

    Scheduler& scheduler;
    const BufferType type;
    MemoryAllocator::Buffer buffer;
    StreamRing ring;
};

StreamBuffer::Mapping::Mapping(std::shared_ptr<State> owner, StreamRing::Reservation reservation,
                               std::optional<MemoryAllocator::WriteMap> cpu)
    : owner(std::move(owner)), reservation(reservation), cpu(std::move(cpu)) {}

StreamBuffer::Mapping::Mapping(Mapping&& other) noexcept
    : owner(std::move(other.owner)), reservation(other.reservation), cpu(std::move(other.cpu)) {}

StreamBuffer::Mapping::~Mapping() {
    if (owner) {
        cpu.reset();
        owner->ring.Cancel(reservation);
    }
}

StreamBuffer::Slice::Slice(std::weak_ptr<State> owner, StreamRing::Slice region,
                           std::shared_ptr<void> readback_pin)
    : owner(std::move(owner)), region(region), readback_pin(std::move(readback_pin)) {}

StreamBuffer::StreamBuffer(const MemoryAllocator& allocator, Scheduler& scheduler,
                           u32 preferred_size, BufferType type)
    : state(std::make_shared<State>(allocator, scheduler, preferred_size, type)) {}

StreamBuffer::~StreamBuffer() = default;

std::size_t StreamBuffer::Size() const noexcept {
    return state->buffer.Size();
}

StreamBuffer::Mapping StreamBuffer::Map(u32 size, u64 alignment,
                                        CompletionTimeline::Deadline deadline) {
    return Begin(size, alignment, false, deadline);
}

StreamBuffer::Mapping StreamBuffer::ReserveDownload(u32 size, u64 alignment,
                                                    CompletionTimeline::Deadline deadline) {
    return Begin(size, alignment, true, deadline);
}

StreamBuffer::Mapping StreamBuffer::Begin(u32 size, u64 alignment, bool download,
                                          CompletionTimeline::Deadline deadline) {
    if ((state->type == BufferType::Download) != download) {
        throw std::logic_error("Metal stream mapping mode does not match its buffer type");
    }
    const auto placement = state->ring.Find(size, alignment);
    const auto wait = state->ring.RequiredTick(placement);
    if (wait) {
        WaitForSuccess(state->scheduler, wait, deadline);
    }
    std::optional<MemoryAllocator::WriteMap> cpu;
    if (!download && size) {
        cpu.emplace(state->buffer.MapWrite(placement.range));
    }
    const auto completed = state->scheduler.GetMasterSemaphore().Timeline().SuccessfulTick();
    const auto reservation = state->ring.Reserve(size, alignment, completed);
    return Mapping{state, reservation, std::move(cpu)};
}

StreamBuffer::Slice StreamBuffer::Commit(Mapping&& mapping, u32 size) {
    if (mapping.owner != state) {
        throw std::invalid_argument("Metal stream mapping belongs to another buffer");
    }
    std::shared_ptr<ReadbackState> readback_pin;
    if (size && state->type == BufferType::Download) {
        readback_pin = std::make_shared<ReadbackState>();
    }
    const u64 tick = size ? state->scheduler.CurrentTick() : 0;
    const auto slice = state->ring.Commit(mapping.reservation, size, tick, readback_pin);
    mapping.cpu.reset();
    mapping.owner.reset();
    return Slice{state, slice, std::move(readback_pin)};
}

MemoryAllocator::GPUBufferAccess StreamBuffer::AcquireGPUAccess(const Slice& slice,
                                                                BufferAccess access) {
    if (slice.owner.lock() != state) {
        throw std::invalid_argument("Metal stream slice belongs to another or expired buffer");
    }
    static_cast<void>(state->ring.UseTick(slice.region));
    auto result = state->buffer.AcquireGPUAccess(slice.Range(), access);
    state->ring.NoteUse(slice.region, state->scheduler.CurrentTick());
    if (slice.readback_pin && access != BufferAccess::Read) {
        std::static_pointer_cast<ReadbackState>(slice.readback_pin)->has_writer = true;
    }
    return result;
}

MemoryAllocator::ReadMap StreamBuffer::ReadDownload(const Slice& slice,
                                                    CompletionTimeline::Deadline deadline) {
    if (slice.owner.lock() != state || !slice.readback_pin) {
        throw std::invalid_argument("Metal download requires a live readback slice");
    }
    if (!std::static_pointer_cast<ReadbackState>(slice.readback_pin)->has_writer) {
        throw std::logic_error("Metal download must be written by GPU work before reading");
    }
    WaitForSuccess(state->scheduler, state->ring.UseTick(slice.region), deadline);
    return state->buffer.MapRead(slice.Range());
}

} // namespace Metal
