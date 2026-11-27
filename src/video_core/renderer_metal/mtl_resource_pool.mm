// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include "video_core/renderer_metal/mtl_resource_pool.h"

namespace Metal {

ResourceSlots::ResourceSlots(std::size_t capacity) : slots(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("Metal resource pool capacity must be positive");
    }
}

bool ResourceSlots::Owns(Key key) const noexcept {
    return key.index < slots.size() && slots[key.index].held &&
           slots[key.index].generation == key.generation;
}

bool ResourceSlots::Available(std::size_t index, u64 retired) const {
    return index < slots.size() && !slots[index].held && slots[index].ticket <= retired;
}

std::optional<ResourceSlots::Key> ResourceSlots::Acquire(u64 retired,
                                                         std::optional<std::size_t> preferred) {
    auto claim = [&](std::size_t index) -> Key {
        if (generation == std::numeric_limits<u64>::max()) {
            throw std::overflow_error("Metal resource pool exhausted its generations");
        }
        slots[index] = {++generation, 0, true};
        next = index + 1 == slots.size() ? 0 : index + 1;
        return {index, generation};
    };
    if (preferred && Available(*preferred, retired)) {
        return claim(*preferred);
    }
    for (std::size_t index = next; index < slots.size(); ++index) {
        if (Available(index, retired)) {
            return claim(index);
        }
    }
    for (std::size_t index = 0; index < next; ++index) {
        if (Available(index, retired)) {
            return claim(index);
        }
    }
    return std::nullopt;
}

bool ResourceSlots::Release(Key key) {
    if (!Owns(key)) {
        return false;
    }
    slots[key.index].held = false;
    return true;
}

bool ResourceSlots::Submit(Key key, u64 ticket) {
    if (!Owns(key) || ticket == 0) {
        return false;
    }
    slots[key.index].ticket = ticket;
    slots[key.index].held = false;
    return true;
}

struct ResourcePool::State {
    struct Binding {
        BindingLayout layout;
        std::vector<u64> identity;
        std::shared_ptr<void> resource_owner;
        id<MTL4ArgumentTable> table = nil;
    };

    State(id<MTLDevice> device, std::shared_ptr<MasterSemaphore> completion,
          std::size_t command_capacity, std::size_t binding_capacity)
        : device(device), completion(std::move(completion)), command_slots(command_capacity),
          binding_slots(binding_capacity), commands(command_capacity), bindings(binding_capacity) {}

    [[nodiscard]] ResourceSlots& Slots(Kind kind) {
        return kind == Kind::Command ? command_slots : binding_slots;
    }

    id<MTLDevice> device;
    std::shared_ptr<MasterSemaphore> completion;
    ResourceSlots command_slots;
    ResourceSlots binding_slots;
    std::vector<id<MTL4CommandAllocator>> commands;
    std::vector<Binding> bindings;
    std::mutex mutex;
};

ResourcePool::Lease::Lease(std::shared_ptr<State> state, ResourceSlots::Key key, Kind kind)
    : state(std::move(state)), key(key), kind(kind) {}

ResourcePool::Lease::Lease(Lease&& other) noexcept
    : state(std::move(other.state)), key(other.key), kind(other.kind) {}

ResourcePool::Lease::~Lease() {
    if (state) {
        std::scoped_lock lock(state->mutex);
        static_cast<void>(state->Slots(kind).Release(key));
    }
}

std::shared_ptr<void> ResourcePool::Lease::RetireWith(u64 ticket) {
    if (!state) {
        throw std::logic_error("Metal resource lease has already been transferred");
    }
    auto retained = state;
    std::scoped_lock lock(retained->mutex);
    const auto& timeline = retained->completion->Timeline();
    if (ticket <= timeline.RetiredTick() || ticket > timeline.LastIssuedTick()) {
        throw std::out_of_range("Metal resource lease requires an outstanding ticket");
    }
    if (!retained->Slots(kind).Submit(key, ticket)) {
        throw std::logic_error("Metal resource lease generation is stale");
    }
    state.reset();
    return retained;
}

ResourcePool::CommandLease::CommandLease(Lease lease, id<MTL4CommandAllocator> object)
    : lease(std::move(lease)), object(object) {}

ResourcePool::BindingLease::BindingLease(Lease lease, id<MTL4ArgumentTable> object)
    : lease(std::move(lease)), object(object) {}

ResourcePool::ResourcePool(id<MTLDevice> device, std::shared_ptr<MasterSemaphore> completion,
                           std::size_t command_capacity, std::size_t binding_capacity) {
    if (!device || !completion || completion->GetDevice() != device) {
        throw std::invalid_argument("Metal resource pool requires a same-device completion timeline");
    }
    state = std::make_shared<State>(device, std::move(completion), command_capacity,
                                    binding_capacity);
}

ResourcePool::~ResourcePool() = default;

std::optional<ResourcePool::CommandLease> ResourcePool::AcquireCommand() {
    std::unique_lock lock(state->mutex);
    const auto key = state->command_slots.Acquire(state->completion->Timeline().RetiredTick());
    if (!key) {
        return std::nullopt;
    }
    Lease lease{state, *key, Kind::Command};
    auto allocator = state->commands[key->index];
    lock.unlock();
    if (allocator) {
        [allocator reset];
    } else {
        auto descriptor = [MTL4CommandAllocatorDescriptor new];
        descriptor.label = @"Metaliris command storage";
        NSError* error = nil;
        allocator = [state->device newCommandAllocatorWithDescriptor:descriptor error:&error];
        if (!allocator) {
            Trim();
            error = nil;
            allocator = [state->device newCommandAllocatorWithDescriptor:descriptor error:&error];
        }
        allocator = CheckResource(allocator, "Allocate Metal 4 command storage", error);
        std::scoped_lock publish(state->mutex);
        state->commands[key->index] = allocator;
    }
    return CommandLease{std::move(lease), allocator};
}

std::optional<ResourcePool::BindingLease> ResourcePool::AcquireBinding(
    BindingLayout layout, std::span<const u64> identity, std::shared_ptr<void> resource_owner,
    const std::function<void(id<MTL4ArgumentTable>)>& initialize) {
    if (!layout.IsValid() || !initialize) {
        throw std::invalid_argument("Invalid Metal argument table layout or initialization");
    }
    std::unique_lock lock(state->mutex);
    const auto retired = state->completion->Timeline().RetiredTick();
    std::optional<std::size_t> preferred;
    for (std::size_t i = 0; i < state->bindings.size(); ++i) {
        const auto& binding = state->bindings[i];
        if (binding.table && binding.layout == layout && state->binding_slots.Available(i, retired) &&
            std::ranges::equal(binding.identity, identity)) {
            preferred = i;
            break;
        }
    }
    const auto key = state->binding_slots.Acquire(retired, preferred);
    if (!key) {
        return std::nullopt;
    }
    Lease lease{state, *key, Kind::Binding};
    if (preferred) {
        return BindingLease{std::move(lease), state->bindings[key->index].table};
    }
    auto previous = std::exchange(state->bindings[key->index], {});
    lock.unlock();

    auto descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = layout.buffers;
    descriptor.maxTextureBindCount = layout.textures;
    descriptor.maxSamplerStateBindCount = layout.samplers;
    descriptor.initializeBindings = YES;
    descriptor.supportAttributeStrides = layout.dynamic_vertex_stride;
    descriptor.label = @"Metaliris bindings";
    NSError* error = nil;
    auto table = [state->device newArgumentTableWithDescriptor:descriptor error:&error];
    if (!table) {
        previous = {};
        Trim();
        error = nil;
        table = [state->device newArgumentTableWithDescriptor:descriptor error:&error];
    }
    table = CheckResource(table, "Allocate Metal 4 argument table", error);
    initialize(table);
    State::Binding updated{layout, {identity.begin(), identity.end()}, std::move(resource_owner),
                           table};
    {
        std::scoped_lock publish(state->mutex);
        state->bindings[key->index] = std::move(updated);
    }
    return BindingLease{std::move(lease), table};
}

void ResourcePool::Trim() {
    std::vector<id<MTL4CommandAllocator>> commands;
    std::vector<State::Binding> bindings;
    commands.reserve(state->commands.size());
    bindings.reserve(state->bindings.size());
    std::scoped_lock lock(state->mutex);
    const auto retired = state->completion->Timeline().RetiredTick();
    for (std::size_t i = 0; i < state->commands.size(); ++i) {
        if (state->command_slots.Available(i, retired) && state->commands[i]) {
            commands.push_back(std::exchange(state->commands[i], nil));
        }
    }
    for (std::size_t i = 0; i < state->bindings.size(); ++i) {
        if (state->binding_slots.Available(i, retired) && state->bindings[i].table) {
            bindings.push_back(std::exchange(state->bindings[i], {}));
        }
    }
}

} // namespace Metal
