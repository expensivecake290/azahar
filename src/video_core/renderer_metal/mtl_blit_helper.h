// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include "common/settings.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

namespace Metal {

enum class TextureFilterRoute : u8 {
    Fallback,
    MipChain,
    Anime4K,
    Bicubic,
    ScaleForce,
    xBRZ,
    MMPX,
};

[[nodiscard]] constexpr TextureFilterRoute SelectTextureFilter(Settings::TextureFilter filter,
                                                               u32 source_mip) noexcept {
    using enum Settings::TextureFilter;
    if (filter == NoFilter) {
        return TextureFilterRoute::Fallback;
    }
    if (source_mip != 0) {
        return TextureFilterRoute::MipChain;
    }
    switch (filter) {
    case Anime4K:
        return TextureFilterRoute::Anime4K;
    case Bicubic:
        return TextureFilterRoute::Bicubic;
    case ScaleForce:
        return TextureFilterRoute::ScaleForce;
    case xBRZ:
        return TextureFilterRoute::xBRZ;
    case MMPX:
        return TextureFilterRoute::MMPX;
    default:
        return TextureFilterRoute::Fallback;
    }
}

struct alignas(8) BlitSampling {
    std::array<float, 2> scale;
    std::array<float, 2> offset;
};

struct alignas(8) DepthStencilCopyRegion {
    std::array<u32, 2> source_offset;
    std::array<u32, 2> destination_offset;
    std::array<u32, 2> extent;
};

struct alignas(8) DepthReadbackRegion {
    std::array<u32, 2> source_offset;
    std::array<u32, 2> extent;
    u32 row_stride;
    u32 capacity;
};

static_assert(sizeof(BlitSampling) == 16);
static_assert(sizeof(DepthStencilCopyRegion) == 24);
static_assert(sizeof(DepthReadbackRegion) == 24);
static_assert(offsetof(DepthReadbackRegion, capacity) == 20);

[[nodiscard]] constexpr RenderArea BlitBounds(const Common::Rectangle<u32>& rect) noexcept {
    const u32 left = std::min(rect.left, rect.right);
    const u32 bottom = std::min(rect.bottom, rect.top);
    return {left, bottom, std::max(rect.left, rect.right) - left,
            std::max(rect.bottom, rect.top) - bottom};
}

/// Native texture coordinates increase from the origin of the native storage in both axes. A
/// relative reversal between the ordered source and destination corners mirrors just that axis.
[[nodiscard]] constexpr std::optional<BlitSampling> MakeBlitSampling(
    const VideoCore::TextureBlit& blit, u32 width, u32 height) noexcept {
    const auto source = BlitBounds(blit.src_rect);
    const auto destination = BlitBounds(blit.dst_rect);
    if (!source.Fits(width, height) || !destination.width || !destination.height) {
        return {};
    }
    const bool mirror_x =
        (blit.src_rect.left > blit.src_rect.right) != (blit.dst_rect.left > blit.dst_rect.right);
    const bool mirror_y =
        (blit.src_rect.bottom > blit.src_rect.top) != (blit.dst_rect.bottom > blit.dst_rect.top);
    const float sx = static_cast<float>(source.width) / static_cast<float>(width);
    const float sy = static_cast<float>(source.height) / static_cast<float>(height);
    return BlitSampling{
        {mirror_x ? -sx : sx, mirror_y ? -sy : sy},
        {static_cast<float>(source.x + (mirror_x ? source.width : 0)) / static_cast<float>(width),
         static_cast<float>(source.y + (mirror_y ? source.height : 0)) /
             static_cast<float>(height)}};
}

[[nodiscard]] constexpr std::optional<DepthReadbackRegion> MakeDepthReadbackRegion(
    const VideoCore::BufferTextureCopy& copy, u32 width, u32 height,
    u32 row_stride_words = 0) noexcept {
    constexpr u32 word_size = sizeof(u32);
    const auto area = BlitBounds(copy.texture_rect);
    const u32 stride = row_stride_words ? row_stride_words : area.width;
    const u32 capacity = copy.buffer_size / word_size;
    if (!area.Fits(width, height) || copy.buffer_offset % word_size != 0 ||
        copy.buffer_size % word_size != 0 || stride < area.width ||
        u64{area.height - 1} * stride + area.width > capacity) {
        return {};
    }
    return DepthReadbackRegion{{area.x, area.y}, {area.width, area.height}, stride, capacity};
}

enum class DepthCopyRoute : u8 {
    Invalid,
    Compute,
    PerSampleRender,
};

[[nodiscard]] constexpr DepthCopyRoute SelectDepthCopyRoute(u32 source, u32 destination) noexcept {
    if (source != destination || !SampleCountBit(source)) {
        return DepthCopyRoute::Invalid;
    }
    return source == 1 ? DepthCopyRoute::Compute : DepthCopyRoute::PerSampleRender;
}

class BlitHelper {
public:
    BlitHelper(Instance& instance, Scheduler& scheduler, RenderManager& passes,
               const MemoryAllocator& allocator);
    ~BlitHelper();
    BlitHelper(const BlitHelper&) = delete;
    BlitHelper& operator=(const BlitHelper&) = delete;

    [[nodiscard]] bool Filter(id<MTLTexture> source, id<MTLTexture> destination,
                              const VideoCore::TextureBlit& blit, Settings::TextureFilter filter,
                              float resolution_scale);

    [[nodiscard]] bool BlitDepthStencil(id<MTLTexture> source, id<MTLTexture> destination,
                                        const VideoCore::TextureBlit& blit);

    /// The multisample path preserves every sample. A supplied resolve updates the corresponding
    /// single-sample representation of the destination once the conversion has finished.
    [[nodiscard]] bool ConvertDS24S8ToRGBA8(id<MTLTexture> source, id<MTLTexture> destination,
                                            const VideoCore::TextureCopy& copy,
                                            RenderTexture resolve = {});

    [[nodiscard]] bool DepthToBuffer(id<MTLTexture> source, MemoryAllocator::GPUBufferAccess buffer,
                                     const VideoCore::BufferTextureCopy& copy,
                                     u32 row_stride_words = 0);

    [[nodiscard]] bool ResolveTexture(const RenderTexture& source, const RenderTexture& destination,
                                      TextureAspect aspect);

private:
    struct State;
    std::unique_ptr<State> state;
};

} // namespace Metal
