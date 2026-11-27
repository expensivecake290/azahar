// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include "video_core/renderer_metal/mtl_master_semaphore.h"

namespace Metal {

/// CPU ledger for a resource pool. The owner serializes changes and supplies the retirement tick.
class ResourceSlots {
public:
    struct Key {
        std::size_t index;
        u64 generation;
    };

    explicit ResourceSlots(std::size_t capacity);
    [[nodiscard]] std::optional<Key> Acquire(u64 retired,
                                             std::optional<std::size_t> preferred = {});
    [[nodiscard]] bool Release(Key key);
    [[nodiscard]] bool Submit(Key key, u64 ticket);
    [[nodiscard]] bool Available(std::size_t index, u64 retired) const;

private:
    struct Slot {
        u64 generation{};
        u64 ticket{};
        bool held{};
    };
    [[nodiscard]] bool Owns(Key key) const noexcept;

    std::vector<Slot> slots;
    std::size_t next{};
    u64 generation{};
};

struct BindingLayout {
    u32 buffers{};
    u32 textures{};
    u32 samplers{};
    bool dynamic_vertex_stride{};

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return buffers <= 31 && textures <= 128 && samplers <= 16;
    }

    bool operator==(const BindingLayout&) const = default;
};

class ResourcePool {
    struct State;
    enum class Kind { Command, Binding };

    class Lease {
    public:
        Lease(std::shared_ptr<State> state, ResourceSlots::Key key, Kind kind);
        Lease(Lease&&) noexcept;
        ~Lease();

        [[nodiscard]] std::shared_ptr<void> RetireWith(u64 ticket);
        [[nodiscard]] bool Active() const noexcept {
            return static_cast<bool>(state);
        }

    private:
        std::shared_ptr<State> state;
        ResourceSlots::Key key;
        Kind kind;
    };

public:
    class CommandLease {
    public:
        CommandLease(CommandLease&&) noexcept = default;
        [[nodiscard]] id<MTL4CommandAllocator> Get() const noexcept {
            return lease.Active() ? object : nil;
        }
        /// Transfer this checkout to Metal feedback before committing the corresponding ticket.
        [[nodiscard]] std::shared_ptr<void> RetireWith(u64 ticket) {
            return lease.RetireWith(ticket);
        }

    private:
        friend class ResourcePool;
        CommandLease(Lease lease, id<MTL4CommandAllocator> object);
        Lease lease;
        id<MTL4CommandAllocator> object;
    };

    class BindingLease {
    public:
        BindingLease(BindingLease&&) noexcept = default;
        /// The table is fully initialized and immutable while leased.
        [[nodiscard]] id<MTL4ArgumentTable> Get() const noexcept {
            return lease.Active() ? object : nil;
        }
        [[nodiscard]] std::shared_ptr<void> RetireWith(u64 ticket) {
            return lease.RetireWith(ticket);
        }

    private:
        friend class ResourcePool;
        BindingLease(Lease lease, id<MTL4ArgumentTable> object);
        Lease lease;
        id<MTL4ArgumentTable> object;
    };

    ResourcePool(id<MTLDevice> device, std::shared_ptr<MasterSemaphore> completion,
                 std::size_t command_capacity, std::size_t binding_capacity);
    ~ResourcePool();
    ResourcePool(const ResourcePool&) = delete;
    ResourcePool& operator=(const ResourcePool&) = delete;

    [[nodiscard]] std::optional<CommandLease> AcquireCommand();

    /// identity contains every binding's immutable value, including strides and owned resource
    /// identity. An exact match reuses an initialized table. The callback may only populate the
    /// new table during this call. Resource references remain alive for the cached table's lifetime.
    [[nodiscard]] std::optional<BindingLease> AcquireBinding(
        BindingLayout layout, std::span<const u64> identity, std::shared_ptr<void> resource_owner,
        const std::function<void(id<MTL4ArgumentTable>)>& initialize);

    /// Drop only cached native objects that are free according to the completion timeline.
    void Trim();

private:
    std::shared_ptr<State> state;
};

} // namespace Metal
