// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <memory>
#include <optional>
#include "video_core/renderer_metal/mtl_memory_util.h"
#include "video_core/renderer_metal/mtl_resource_pool.h"

namespace Metal {

class EncodingContext;

/// Producer-owned descriptor updates. Captured snapshots may be encoded after later edits.
class DescriptorUpdateQueue {
    struct State;
    struct SetState;
    struct SnapshotState;

public:
    class Set {
    public:
        [[nodiscard]] BindingLayout Layout() const noexcept;

    private:
        friend class DescriptorUpdateQueue;
        explicit Set(std::shared_ptr<SetState> state);
        std::shared_ptr<SetState> state;
    };

    class Snapshot {
    public:
        [[nodiscard]] u64 Generation() const noexcept;
        [[nodiscard]] BindingLayout Layout() const noexcept;

        /// Binds immutable arguments and declares their resource uses on this encoder.
        void BindRender(EncodingContext& context, id<MTL4RenderCommandEncoder> encoder,
                        MTLRenderStages stages) const;
        void BindCompute(EncodingContext& context, id<MTL4ComputeCommandEncoder> encoder) const;

    private:
        friend class DescriptorUpdateQueue;
        explicit Snapshot(std::shared_ptr<const SnapshotState> state);
        std::shared_ptr<const SnapshotState> state;
    };

    explicit DescriptorUpdateQueue(DeviceContext& context, std::size_t max_writes = 2048);
    ~DescriptorUpdateQueue();
    DescriptorUpdateQueue(const DescriptorUpdateQueue&) = delete;
    DescriptorUpdateQueue& operator=(const DescriptorUpdateQueue&) = delete;

    [[nodiscard]] Set CreateSet(BindingLayout layout);
    void Flush();
    [[nodiscard]] Snapshot Capture(const Set& target);

    /// A nil resource clears the corresponding slot. A zero stage mask follows the encoder.
    void AddStorageImage(const Set& target, u32 binding, id<MTLTexture> texture,
                         BufferAccess access = BufferAccess::ReadWrite, MTLStages stages = {});

    /// For a paired sampler, sampler_binding defaults to the same base index as the texture.
    /// An explicit sampler index may be used with different shader binding namespaces.
    void AddImageSampler(const Set& target, u32 binding, u32 array_index, id<MTLTexture> texture,
                         id<MTLSamplerState> sampler, std::optional<u32> sampler_binding = {},
                         MTLStages stages = {});

    void AddBuffer(const Set& target, u32 binding, id<MTLBuffer> buffer, std::size_t offset,
                   std::optional<std::size_t> size = {}, BufferAccess access = BufferAccess::Read,
                   std::optional<u32> attribute_stride = {}, MTLStages stages = {});

    void AddTexelBuffer(const Set& target, u32 binding, id<MTLTexture> buffer_view,
                        MTLStages stages = {});

    [[nodiscard]] static constexpr std::optional<u32> Element(u32 capacity, u32 base,
                                                              u32 index) noexcept {
        if (base >= capacity || index >= capacity - base) {
            return {};
        }
        return base + index;
    }

    [[nodiscard]] static constexpr std::optional<BufferRange> BufferBindingRange(
        std::size_t length, std::size_t offset, std::optional<std::size_t> size) noexcept {
        if (offset >= length) {
            return {};
        }
        const BufferRange range{offset, size.value_or(length - offset)};
        return range.Fits(length) ? std::optional(range) : std::nullopt;
    }

private:
    std::unique_ptr<State> state;
};

} // namespace Metal
