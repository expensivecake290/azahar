// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <span>
#include "video_core/renderer_metal/mtl_master_semaphore.h"
#include "video_core/renderer_metal/mtl_memory_util.h"

namespace Metal {

enum class BufferType : u8 {
    Upload,
    Download,
    Stream,
};

class Scheduler;

/// Tracks logical byte ranges independently of Metal storage and queue submission.
class StreamRing {
public:
    struct Placement {
        BufferRange range;
        bool wrapped;
    };

    struct Reservation {
        Placement placement;
        u64 generation;
    };

    struct Slice {
        BufferRange range{};
        u64 generation{};
    };

    explicit StreamRing(u32 capacity);

    [[nodiscard]] static constexpr std::optional<Placement> Locate(u32 capacity, u32 cursor,
                                                                   u32 size, u64 alignment) {
        if (capacity == 0 || cursor > capacity || size > capacity) {
            return {};
        }
        const u64 remainder = alignment ? cursor % alignment : 0;
        const u64 padding = remainder ? alignment - remainder : 0;
        if (padding > capacity - cursor || size > capacity - cursor - padding) {
            return Placement{{0, size}, true};
        }
        return Placement{{static_cast<std::size_t>(cursor + padding), size}, false};
    }

    /// Look up a placement and the latest overlapping ticket without consuming the space.
    [[nodiscard]] Placement Find(u32 size, u64 alignment) const;
    [[nodiscard]] u64 RequiredTick(Placement placement) const;

    /// Refuses a reservation that overlaps work after completed_tick or a pinned download.
    [[nodiscard]] Reservation Reserve(u32 size, u64 alignment, u64 completed_tick);
    [[nodiscard]] Slice Commit(Reservation reservation, u32 size, u64 tick,
                               std::weak_ptr<void> readback_pin = {});
    void Cancel(Reservation reservation) noexcept;

    [[nodiscard]] u64 UseTick(Slice slice) const;
    void NoteUse(Slice slice, u64 tick);

private:
    struct Watch {
        Slice slice;
        u64 tick;
        std::weak_ptr<void> readback_pin;
    };

    using Watches = std::map<std::size_t, Watch>;
    [[nodiscard]] Watches::const_iterator FirstOverlap(BufferRange range) const;

    const u32 capacity;
    u32 cursor{};
    u64 generation{};
    std::optional<Reservation> active;
    Watches watches;
};

/// Shared-memory upload, readback and stream storage owned by one recording producer.
class StreamBuffer final {
    struct State;

public:
    class Mapping {
    public:
        Mapping(Mapping&&) noexcept;
        ~Mapping();

        /// The span is valid until this mapping is committed or destroyed.
        [[nodiscard]] std::span<std::byte> Bytes() const noexcept {
            return cpu ? cpu->Bytes() : std::span<std::byte>{};
        }

        [[nodiscard]] BufferRange Range() const noexcept {
            return reservation.placement.range;
        }

        [[nodiscard]] bool Wrapped() const noexcept {
            return reservation.placement.wrapped;
        }

    private:
        friend class StreamBuffer;
        Mapping(std::shared_ptr<State> owner, StreamRing::Reservation reservation,
                std::optional<MemoryAllocator::WriteMap> cpu);

        std::shared_ptr<State> owner;
        StreamRing::Reservation reservation;
        std::optional<MemoryAllocator::WriteMap> cpu;
    };

    class Slice {
    public:
        [[nodiscard]] BufferRange Range() const noexcept {
            return region.range;
        }

    private:
        friend class StreamBuffer;
        Slice(std::weak_ptr<State> owner, StreamRing::Slice region,
              std::shared_ptr<void> readback_pin);

        std::weak_ptr<State> owner;
        StreamRing::Slice region;
        std::shared_ptr<void> readback_pin;
    };

    StreamBuffer(const MemoryAllocator& allocator, Scheduler& scheduler, u32 preferred_size,
                 BufferType type = BufferType::Stream);
    ~StreamBuffer();
    StreamBuffer(const StreamBuffer&) = delete;
    StreamBuffer& operator=(const StreamBuffer&) = delete;

    [[nodiscard]] std::size_t Size() const noexcept;

    [[nodiscard]] Mapping Map(u32 size, u64 alignment = 0,
                              CompletionTimeline::Deadline deadline = {});
    [[nodiscard]] Mapping ReserveDownload(u32 size, u64 alignment = 0,
                                          CompletionTimeline::Deadline deadline = {});
    [[nodiscard]] Slice Commit(Mapping&& mapping, u32 size);

    /// Call before recording the command and retain the returned range in its encoding context.
    [[nodiscard]] MemoryAllocator::GPUBufferAccess AcquireGPUAccess(const Slice& slice,
                                                                    BufferAccess access);

    /// A download slice protects unread data across ring wraps. The map has its own CPU lease.
    [[nodiscard]] MemoryAllocator::ReadMap ReadDownload(const Slice& slice,
                                                        CompletionTimeline::Deadline deadline = {});

private:
    [[nodiscard]] Mapping Begin(u32 size, u64 alignment, bool download,
                                CompletionTimeline::Deadline deadline);

    std::shared_ptr<State> state;
};

} // namespace Metal
