// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

namespace Metal {
namespace {

constexpr MTLRenderStages RenderStages = MTLRenderStageVertex | MTLRenderStageFragment |
                                         MTLRenderStageTile | MTLRenderStageObject |
                                         MTLRenderStageMesh;
constexpr MTLStages BindingStages = static_cast<MTLStages>(RenderStages) | MTLStageDispatch;

static_assert(static_cast<NSUInteger>(MTLRenderStageVertex) == MTLStageVertex);
static_assert(static_cast<NSUInteger>(MTLRenderStageFragment) == MTLStageFragment);
static_assert(static_cast<NSUInteger>(MTLRenderStageTile) == MTLStageTile);
static_assert(static_cast<NSUInteger>(MTLRenderStageObject) == MTLStageObject);
static_assert(static_cast<NSUInteger>(MTLRenderStageMesh) == MTLStageMesh);

void ValidateAccess(BufferAccess access, MTLStages stages) {
    if (access != BufferAccess::Read && access != BufferAccess::Write &&
        access != BufferAccess::ReadWrite) {
        throw std::invalid_argument("Invalid Metal argument access");
    }
    if ((stages & ~BindingStages) != 0) {
        throw std::invalid_argument("Invalid Metal argument shader stages");
    }
}

void ValidateResource(id<MTLDevice> actual_device, id<MTLDevice> expected_device) {
    if (actual_device && actual_device != expected_device) {
        throw std::invalid_argument("Metal argument resource belongs to a foreign device");
    }
}

u32 RequireElement(u32 capacity, u32 base, u32 index = 0) {
    const auto element = DescriptorUpdateQueue::Element(capacity, base, index);
    if (!element) {
        throw std::out_of_range("Metal argument binding index is outside its layout");
    }
    return *element;
}

u64 ResourceKey(id resource) {
    return reinterpret_cast<std::uintptr_t>((__bridge const void*)resource);
}

u64 NextGeneration(u64 generation) {
    if (generation == std::numeric_limits<u64>::max()) {
        throw std::overflow_error("Metal descriptor queue exhausted its generations");
    }
    return generation + 1;
}

struct BufferBinding {
    id<MTLBuffer> buffer = nil;
    BufferRange range{};
    BufferAccess access = BufferAccess::Read;
    std::optional<u32> stride;
    MTLStages stages{};
};

struct TextureBinding {
    id<MTLTexture> texture = nil;
    BufferAccess access = BufferAccess::Read;
    MTLStages stages{};
};

struct Bindings {
    explicit Bindings(BindingLayout layout)
        : layout(layout), buffers(layout.buffers), textures(layout.textures),
          samplers(layout.samplers) {}

    std::vector<u64> Identity(u64 device_generation) const {
        std::vector<u64> key;
        key.reserve(1 + buffers.size() * 8 + textures.size() * 4 + samplers.size() * 2);
        key.push_back(device_generation);
        for (const auto& binding : buffers) {
            const u64 entry[] = {ResourceKey(binding.buffer),
                                 binding.buffer.gpuAddress,
                                 binding.range.offset,
                                 binding.range.length,
                                 static_cast<u64>(binding.access),
                                 binding.stride.has_value(),
                                 binding.stride.value_or(0),
                                 static_cast<u64>(binding.stages)};
            key.insert(key.end(), std::begin(entry), std::end(entry));
        }
        for (const auto& binding : textures) {
            const u64 entry[] = {ResourceKey(binding.texture),
                                 binding.texture ? binding.texture.gpuResourceID._impl : 0,
                                 static_cast<u64>(binding.access),
                                 static_cast<u64>(binding.stages)};
            key.insert(key.end(), std::begin(entry), std::end(entry));
        }
        for (const auto& sampler : samplers) {
            key.push_back(ResourceKey(sampler));
            key.push_back(sampler ? sampler.gpuResourceID._impl : 0);
        }
        return key;
    }

    void Initialize(id<MTL4ArgumentTable> table) const {
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            const auto& entry = buffers[i];
            if (!entry.buffer) {
                continue;
            }
            const auto address = entry.buffer.gpuAddress + entry.range.offset;
            if (entry.stride) {
                [table setAddress:address attributeStride:*entry.stride atIndex:i];
            } else {
                [table setAddress:address atIndex:i];
            }
        }
        for (std::size_t i = 0; i < textures.size(); ++i) {
            if (const auto texture = textures[i].texture) {
                [table setTexture:texture.gpuResourceID atIndex:i];
            }
        }
        for (std::size_t i = 0; i < samplers.size(); ++i) {
            if (const auto sampler = samplers[i]) {
                [table setSamplerState:sampler.gpuResourceID atIndex:i];
            }
        }
    }

    void Synchronize(EncodingContext& context, id<MTL4CommandEncoder> encoder,
                     MTLStages stages) const {
        std::vector<ResourceUse> uses;
        uses.reserve(buffers.size() + textures.size());
        auto add = [&](id<MTLResource> resource, BufferAccess access, MTLStages binding_stages,
                       BufferRange range) {
            if (!resource) {
                return;
            }
            const auto active = binding_stages ? binding_stages & stages : stages;
            if (active) {
                uses.push_back({resource, access, active, range});
            } else {
                context.MakeResident(resource);
            }
        };
        for (const auto& binding : buffers) {
            add(binding.buffer, binding.access, binding.stages, binding.range);
        }
        for (const auto& binding : textures) {
            add(binding.texture, binding.access, binding.stages, {});
        }
        context.Synchronize(encoder, uses);
    }

    id<MTL4ArgumentTable> AllocateTransientTable(id<MTLDevice> device) const {
        auto descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = layout.buffers;
        descriptor.maxTextureBindCount = layout.textures;
        descriptor.maxSamplerStateBindCount = layout.samplers;
        descriptor.initializeBindings = YES;
        descriptor.supportAttributeStrides = layout.dynamic_vertex_stride;
        descriptor.label = @"Metaliris transient arguments";
        NSError* error = nil;
        auto table = [device newArgumentTableWithDescriptor:descriptor error:&error];
        table =
            CheckResource(table, "Allocate Metal arguments when the binding pool is full", error);
        Initialize(table);
        return table;
    }

    const BindingLayout layout;
    std::vector<BufferBinding> buffers;
    std::vector<TextureBinding> textures;
    std::vector<id<MTLSamplerState>> samplers;
};

} // Anonymous namespace

struct DescriptorUpdateQueue::SnapshotState {
    SnapshotState(std::shared_ptr<Bindings> bindings, const DeviceContext::WorkLease& work,
                  u64 generation)
        : bindings(std::move(bindings)), work(work), generation(generation) {}

    id<MTL4ArgumentTable> Prepare(EncodingContext& context, id<MTL4CommandEncoder> encoder,
                                  MTLStages stages) const {
        if (!context.OwnsWork(work)) {
            throw std::invalid_argument(
                "Metal argument generation belongs to another device context");
        }

        bindings->Synchronize(context, encoder, stages);
        id<MTL4ArgumentTable> table;
        {
            std::scoped_lock lock(table_mutex);
            if (!native) {
                const auto identity = bindings->Identity(work.Generation());
                auto lease = context.GetResourcePool().AcquireBinding(
                    bindings->layout, identity, bindings,
                    [&](id<MTL4ArgumentTable> table) { bindings->Initialize(table); });
                if (lease) {
                    auto pooled = lease->Get();
                    context.Retain(std::move(*lease));
                    native = pooled;
                } else {
                    // A full pool can contain leases owned by this still-open command buffer.
                    native = bindings->AllocateTransientTable(context.GetCommandBuffer().device);
                }
            }
            table = native;
        }
        // The table can be shared by exact immutable generations. Each submission keeps its own
        // references even if the cache later evicts the table's slot.
        context.Retain(table);
        context.Retain(bindings);
        return table;
    }

    const std::shared_ptr<Bindings> bindings;
    const DeviceContext::WorkLease work;
    const u64 generation;
    mutable std::mutex table_mutex;
    mutable id<MTL4ArgumentTable> native = nil;
};

struct DescriptorUpdateQueue::SetState {
    std::weak_ptr<void> queue_identity;
    std::shared_ptr<const SnapshotState> snapshot;
    std::shared_ptr<Bindings> pending;
};

struct DescriptorUpdateQueue::State {
    explicit State(DeviceContext& context, std::size_t max_writes)
        : context(context), max_writes(max_writes) {
        if (!max_writes) {
            throw std::invalid_argument("Metal descriptor queue capacity must be positive");
        }
    }

    void Validate(const SetState& target) const {
        if (target.queue_identity.lock() != identity ||
            context.GetLeaseStatus(target.snapshot->work) != WorkLeaseStatus::Valid) {
            throw std::invalid_argument(
                "Metal descriptor set is foreign or from an old generation");
        }
    }

    Bindings& BeginWrite(const std::shared_ptr<SetState>& target) {
        if (pending_writes == max_writes) {
            Flush();
        }
        if (!target->pending) {
            auto contents = std::make_shared<Bindings>(*target->snapshot->bindings);
            dirty.push_back(target);
            target->pending = std::move(contents);
        }
        ++pending_writes;
        return *target->pending;
    }

    void Flush() {
        if (pending_writes == 0) {
            return;
        }
        auto next = generation;
        std::vector<std::shared_ptr<const SnapshotState>> published;
        published.reserve(dirty.size());
        for (const auto& target : dirty) {
            if (context.GetLeaseStatus(target->snapshot->work) != WorkLeaseStatus::Valid) {
                published.push_back({});
                continue;
            }
            next = NextGeneration(next);
            published.push_back(
                std::make_shared<SnapshotState>(target->pending, target->snapshot->work, next));
        }
        for (std::size_t i = 0; i < dirty.size(); ++i) {
            auto& target = dirty[i];
            if (published[i]) {
                target->snapshot = std::move(published[i]);
            }
            target->pending.reset();
        }
        generation = next;
        dirty.clear();
        pending_writes = 0;
    }

    DeviceContext& context;
    const std::size_t max_writes;
    const std::shared_ptr<void> identity = std::make_shared<std::byte>();
    std::size_t pending_writes{};
    u64 generation{};
    std::vector<std::shared_ptr<SetState>> dirty;
};

DescriptorUpdateQueue::Set::Set(std::shared_ptr<SetState> state) : state(std::move(state)) {}

BindingLayout DescriptorUpdateQueue::Set::Layout() const noexcept {
    return state->snapshot->bindings->layout;
}

DescriptorUpdateQueue::Snapshot::Snapshot(std::shared_ptr<const SnapshotState> state)
    : state(std::move(state)) {}

u64 DescriptorUpdateQueue::Snapshot::Generation() const noexcept {
    return state->generation;
}

BindingLayout DescriptorUpdateQueue::Snapshot::Layout() const noexcept {
    return state->bindings->layout;
}

void DescriptorUpdateQueue::Snapshot::BindRender(EncodingContext& context,
                                                 id<MTL4RenderCommandEncoder> encoder,
                                                 MTLRenderStages stages) const {
    if (!stages || (stages & ~RenderStages) != 0) {
        throw std::invalid_argument("Metal render binding needs valid render stages");
    }
    const auto table = state->Prepare(context, encoder, static_cast<MTLStages>(stages));
    [encoder setArgumentTable:table atStages:stages];
}

void DescriptorUpdateQueue::Snapshot::BindCompute(EncodingContext& context,
                                                  id<MTL4ComputeCommandEncoder> encoder) const {
    const auto table = state->Prepare(context, encoder, MTLStageDispatch);
    [encoder setArgumentTable:table];
}

DescriptorUpdateQueue::DescriptorUpdateQueue(DeviceContext& context, std::size_t max_writes)
    : state(std::make_unique<State>(context, max_writes)) {}

DescriptorUpdateQueue::~DescriptorUpdateQueue() = default;

DescriptorUpdateQueue::Set DescriptorUpdateQueue::CreateSet(BindingLayout layout) {
    if (!layout.IsValid()) {
        throw std::invalid_argument("Metal argument table layout exceeds SDK bounds");
    }
    auto set = std::make_shared<SetState>();
    set->queue_identity = state->identity;
    const auto next = NextGeneration(state->generation);
    set->snapshot = std::make_shared<SnapshotState>(std::make_shared<Bindings>(layout),
                                                    state->context.AcquireWork(), next);
    state->generation = next;
    return Set(std::move(set));
}

void DescriptorUpdateQueue::Flush() {
    state->Flush();
}

DescriptorUpdateQueue::Snapshot DescriptorUpdateQueue::Capture(const Set& target) {
    state->Validate(*target.state);
    Flush();
    return Snapshot(target.state->snapshot);
}

void DescriptorUpdateQueue::AddStorageImage(const Set& target, u32 binding, id<MTLTexture> texture,
                                            BufferAccess access, MTLStages stages) {
    state->Validate(*target.state);
    ValidateAccess(access, stages);
    ValidateResource(texture.device, state->context.GetDevice());
    const auto index = RequireElement(target.Layout().textures, binding);
    auto& pending = state->BeginWrite(target.state);
    pending.textures[index] = {texture, access, stages};
}

void DescriptorUpdateQueue::AddImageSampler(const Set& target, u32 binding, u32 array_index,
                                            id<MTLTexture> texture, id<MTLSamplerState> sampler,
                                            std::optional<u32> sampler_binding, MTLStages stages) {
    state->Validate(*target.state);
    ValidateAccess(BufferAccess::Read, stages);
    ValidateResource(texture.device, state->context.GetDevice());
    ValidateResource(sampler.device, state->context.GetDevice());
    const auto layout = target.Layout();
    const auto texture_index = RequireElement(layout.textures, binding, array_index);
    const auto sampler_index =
        Element(layout.samplers, sampler_binding.value_or(binding), array_index);
    if ((sampler || sampler_binding) && !sampler_index) {
        throw std::out_of_range("Metal sampler binding index is outside its layout");
    }
    auto& pending = state->BeginWrite(target.state);
    pending.textures[texture_index] = {texture, BufferAccess::Read, stages};
    if (sampler_index) {
        pending.samplers[*sampler_index] = sampler;
    }
}

void DescriptorUpdateQueue::AddBuffer(const Set& target, u32 binding, id<MTLBuffer> buffer,
                                      std::size_t offset, std::optional<std::size_t> size,
                                      BufferAccess access, std::optional<u32> attribute_stride,
                                      MTLStages stages) {
    state->Validate(*target.state);
    ValidateAccess(access, stages);
    ValidateResource(buffer.device, state->context.GetDevice());
    const auto layout = target.Layout();
    const auto index = RequireElement(layout.buffers, binding);
    if (attribute_stride && (!layout.dynamic_vertex_stride || *attribute_stride == 0)) {
        throw std::invalid_argument("Invalid Metal argument buffer attribute stride");
    }
    BufferRange range;
    if (buffer) {
        const auto checked = BufferBindingRange(buffer.length, offset, size);
        if (!checked || buffer.gpuAddress == 0 ||
            checked->offset > std::numeric_limits<MTLGPUAddress>::max() - buffer.gpuAddress ||
            checked->length - 1 >
                std::numeric_limits<MTLGPUAddress>::max() - buffer.gpuAddress - checked->offset) {
            throw std::out_of_range("Metal argument buffer binding exceeds its range");
        }
        range = *checked;
    } else if (offset || size || attribute_stride) {
        throw std::invalid_argument("A nil Metal argument buffer cannot bind a range or stride");
    }
    auto& pending = state->BeginWrite(target.state);
    pending.buffers[index] = {buffer, range, access, attribute_stride, stages};
}

void DescriptorUpdateQueue::AddTexelBuffer(const Set& target, u32 binding,
                                           id<MTLTexture> buffer_view, MTLStages stages) {
    if (buffer_view && buffer_view.textureType != MTLTextureTypeTextureBuffer) {
        throw std::invalid_argument("Metal texel buffer requires a texture buffer view");
    }
    AddStorageImage(target, binding, buffer_view, BufferAccess::Read, stages);
}

} // namespace Metal
