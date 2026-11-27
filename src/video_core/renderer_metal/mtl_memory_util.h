// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include "video_core/renderer_metal/mtl_common.h"

namespace Metal {

enum class MemoryPool : u8 {
    Stream,
    Texture,
    Readback,
    Transient,
    Count,
};

struct MemoryUsage {
    u64 physical_limit{};
    u64 physical_used{};
    std::array<u64, static_cast<std::size_t>(MemoryPool::Count)> logical_used{};
};

/// A shared physical budget with separate logical accounting by resource use.
class MemoryBudget {
    struct State;

public:
    class Reservation {
    public:
        Reservation(Reservation&&) noexcept;
        Reservation& operator=(Reservation&&) noexcept;
        ~Reservation();

        /// Adjusts the physical charge to the native allocation's reported size. A failed
        /// adjustment leaves the existing charge intact.
        [[nodiscard]] bool AdjustPhysical(u64 size);

    private:
        friend class MemoryBudget;
        Reservation(std::shared_ptr<State> state, MemoryPool pool, u64 logical, u64 physical);
        void Release() noexcept;

        std::shared_ptr<State> state;
        MemoryPool pool{};
        u64 logical{};
        u64 physical{};
    };

    explicit MemoryBudget(u64 physical_limit);

    /// Imported memory can be charged logically with zero new physical bytes.
    [[nodiscard]] std::optional<Reservation> TryReserve(MemoryPool pool, u64 logical,
                                                         u64 physical) const;
    [[nodiscard]] MemoryUsage GetUsage() const;

private:
    std::shared_ptr<State> state;
};

struct BufferRange {
    std::size_t offset{};
    std::size_t length{};

    [[nodiscard]] constexpr bool Fits(std::size_t size) const noexcept {
        return length != 0 && offset <= size && length <= size - offset;
    }

    [[nodiscard]] constexpr bool Overlaps(BufferRange other) const noexcept {
        if (length == 0 || other.length == 0) {
            return false;
        }
        return offset <= other.offset ? other.offset - offset < length
                                      : offset - other.offset < other.length;
    }
};

enum class BufferAccess : u8 {
    Read,
    Write,
    ReadWrite,
};

/// Validates that the whole span is page-aligned, writable and inside one VM region.
[[nodiscard]] bool IsImportableHostRange(std::span<const std::byte> bytes);

class MemoryAllocator {
    struct BufferState;

    class BufferUse {
    public:
        BufferUse(std::shared_ptr<BufferState> state, u64 id);
        BufferUse(BufferUse&&) noexcept;
        ~BufferUse();

        [[nodiscard]] id<MTLBuffer> GetBuffer() const noexcept;

    private:
        std::shared_ptr<BufferState> state;
        u64 id{};
    };

public:
    class Buffer;

    /// Keep this lease through the completion callback for every submission using its range.
    /// The scheduler orders overlapping GPU operations. The lease prevents unsafe CPU mapping.
    class GPUBufferAccess {
    public:
        GPUBufferAccess(GPUBufferAccess&&) noexcept = default;

        [[nodiscard]] id<MTLBuffer> GetBuffer() const noexcept {
            return use.GetBuffer();
        }

        [[nodiscard]] BufferRange Range() const noexcept {
            return range;
        }

    private:
        friend class Buffer;
        GPUBufferAccess(BufferUse use, BufferRange range) : use(std::move(use)), range(range) {}

        BufferUse use;
        BufferRange range;
    };

    template <typename Byte>
    class MappedRange {
    public:
        MappedRange(MappedRange&& other) noexcept
            : use(std::move(other.use)), bytes(std::exchange(other.bytes, {})) {}

        [[nodiscard]] std::span<Byte> Bytes() const noexcept {
            return bytes;
        }

    private:
        friend class Buffer;
        MappedRange(BufferUse use, std::span<Byte> bytes) : use(std::move(use)), bytes(bytes) {}

        BufferUse use;
        std::span<Byte> bytes;
    };

    using ReadMap = MappedRange<const std::byte>;
    using WriteMap = MappedRange<std::byte>;

    class Buffer {
    public:
        [[nodiscard]] std::size_t Size() const noexcept;
        [[nodiscard]] MTLStorageMode StorageMode() const noexcept;

        [[nodiscard]] ReadMap MapRead(BufferRange range) const;
        [[nodiscard]] WriteMap MapWrite(BufferRange range) const;
        [[nodiscard]] GPUBufferAccess AcquireGPUAccess(BufferRange range, BufferAccess access) const;

    private:
        friend class MemoryAllocator;
        explicit Buffer(std::shared_ptr<BufferState> state);
        [[nodiscard]] BufferUse Acquire(BufferRange range, BufferAccess access, bool gpu) const;

        std::shared_ptr<BufferState> state;
    };

    /// The recovery callback may reclaim completed resources. It is called outside allocation
    /// locks, at most once per allocation, and must tolerate concurrent calls.
    MemoryAllocator(id<MTLDevice> device, MemoryBudget budget,
                    std::function<void()> recover_pressure = {});

    [[nodiscard]] Buffer AllocateBuffer(std::size_t size, MemoryPool pool,
                                         MTLStorageMode storage = MTLStorageModeShared) const;

    /// The owner must hold the complete supplied span and alias its first byte.
    [[nodiscard]] Buffer ImportSharedBuffer(std::span<std::byte> bytes, std::shared_ptr<void> owner,
                                             MemoryPool pool) const;

private:
    [[nodiscard]] Buffer Allocate(std::size_t size, MemoryPool pool, MTLStorageMode storage,
                                   std::shared_ptr<void> host_allocation) const;

    id<MTLDevice> device;
    MemoryBudget budget;
    std::function<void()> recover_pressure;
};

} // namespace Metal
