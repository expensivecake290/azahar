// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <bit>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <fmt/format.h>
#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/sampler_params.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_metal/mtl_blit_helper.h"

namespace Metal {

enum class TextureRepresentation : u8 {
    Current,
    Base,
    Scaled,
    Multisample,
    Normal,
    Feedback,
};

enum class TextureView : u8 {
    Sample,
    Attachment,
    Storage,
    Depth,
    Stencil,
};

struct TextureBufferLayout {
    std::size_t row_bytes;
    std::size_t row_pitch;
    std::size_t rows;
    std::size_t length;

    bool operator==(const TextureBufferLayout&) const = default;
};

/// Compute checked byte counts before acquiring any staging or texture regions. Edge blocks of
/// compressed formats occupy a complete block in memory even when the source extent is odd.
[[nodiscard]] constexpr std::optional<TextureBufferLayout> MakeTextureBufferLayout(
    VideoCore::Extent extent, u32 block_width, u32 block_height, u32 bytes_per_block,
    u32 row_alignment) noexcept {
    if (!extent.width || !extent.height || !block_width || !block_height || !bytes_per_block ||
        !std::has_single_bit(row_alignment)) {
        return {};
    }
    const u64 columns = (u64{extent.width} - 1) / block_width + 1;
    const u64 rows = (u64{extent.height} - 1) / block_height + 1;
    const u64 row_bytes = columns * bytes_per_block;
    if (row_bytes > std::numeric_limits<u64>::max() - (row_alignment - 1)) {
        return {};
    }
    const u64 pitch = (row_bytes + row_alignment - 1) & ~u64{row_alignment - 1};
    if (pitch > std::numeric_limits<std::size_t>::max() / rows) {
        return {};
    }
    return TextureBufferLayout{static_cast<std::size_t>(row_bytes), static_cast<std::size_t>(pitch),
                               static_cast<std::size_t>(rows),
                               static_cast<std::size_t>(rows * pitch)};
}

struct SamplerMetadata {
    using TextureConfig = VideoCore::SamplerParams::TextureConfig;
    TextureConfig::WrapMode wrap_s;
    TextureConfig::WrapMode wrap_t;
    std::array<float, 4> border_color;
    u32 lod_min;
    u32 lod_max;
    s32 lod_bias;

    [[nodiscard]] constexpr bool UsesGuestBorder() const noexcept {
        using enum TextureConfig::WrapMode;
        return wrap_s == ClampToBorder || wrap_t == ClampToBorder || wrap_s == ClampToBorder2 ||
               wrap_t == ClampToBorder2;
    }

    bool operator==(const SamplerMetadata&) const = default;
};

[[nodiscard]] constexpr SamplerMetadata GetSamplerMetadata(
    const VideoCore::SamplerParams& params) noexcept {
    const auto value = [&](unsigned shift) {
        return static_cast<float>((params.border_color >> shift) & 0xff) / 255.0f;
    };
    return {params.wrap_s,  params.wrap_t,  {value(0), value(8), value(16), value(24)},
            params.lod_min, params.lod_max, params.lod_bias};
}

struct TextureAllocation;
struct SurfaceStorage;
class TextureSurface;
class Framebuffer;

/// Shared ownership keeps both the Metal texture and its physical budget reservation alive until
/// the GPU retires the last submission using that allocation.
class TextureResource {
public:
    TextureResource() = default;

    [[nodiscard]] id<MTLTexture> Get() const noexcept;
    void Retain(EncodingContext& context) const;

    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(allocation);
    }

private:
    friend class TextureRuntime;
    explicit TextureResource(std::shared_ptr<TextureAllocation> allocation)
        : allocation(std::move(allocation)) {}

    std::shared_ptr<TextureAllocation> allocation;
};

class TextureRuntime {
public:
    TextureRuntime(Instance& instance, Scheduler& scheduler, RenderManager& passes,
                   const MemoryAllocator& allocator, MemoryBudget budget, u32 max_anisotropy = 1);
    ~TextureRuntime();
    TextureRuntime(const TextureRuntime&) = delete;
    TextureRuntime& operator=(const TextureRuntime&) = delete;

    [[nodiscard]] const Instance& GetInstance() const noexcept;
    [[nodiscard]] Scheduler& GetScheduler() const noexcept;
    [[nodiscard]] RenderManager& GetRenderManager() const noexcept;
    [[nodiscard]] u32 MaxAnisotropy() const noexcept;
    [[nodiscard]] bool SupportsCustomFormat(VideoCore::CustomPixelFormat format) const;
    [[nodiscard]] u64 GetResourceTick();
    [[nodiscard]] u64 GetResourceFreeTick();

    void Finish();
    [[nodiscard]] VideoCore::StagingData FindStaging(u32 size, bool upload);

    bool Reinterpret(TextureSurface& source, TextureSurface& destination,
                     const VideoCore::TextureCopy& copy);
    bool ClearTexture(TextureSurface& surface, const VideoCore::TextureClear& clear);
    bool CopyTextures(TextureSurface& source, TextureSurface& destination,
                      std::span<const VideoCore::TextureCopy> copies);
    bool CopyTextures(TextureSurface& source, TextureSurface& destination,
                      const VideoCore::TextureCopy& copy) {
        return CopyTextures(source, destination, std::span{&copy, 1});
    }
    bool BlitTextures(TextureSurface& source, TextureSurface& destination,
                      const VideoCore::TextureBlit& blit);
    void GenerateMipmaps(TextureSurface& surface);
    [[nodiscard]] bool NeedsConversion(const TextureSurface& surface) const;

private:
    friend class TextureSurface;
    friend class Framebuffer;
    struct State;
    std::unique_ptr<State> state;
};

struct ShadowTextureBinding {
    TextureResource image;
    std::optional<MemoryAllocator::Buffer> buffer;
    u32 row_pitch{};
};

class TextureSurface : public VideoCore::SurfaceBase {
public:
    TextureSurface(TextureRuntime& runtime, const VideoCore::SurfaceParams& params,
                   const VideoCore::SurfaceFlagBits& flags = {});
    TextureSurface(TextureRuntime& runtime, const VideoCore::SurfaceBase& original,
                   const VideoCore::Material* material);
    ~TextureSurface();
    TextureSurface(const TextureSurface&) = delete;
    TextureSurface& operator=(const TextureSurface&) = delete;
    TextureSurface(TextureSurface&&) noexcept;
    TextureSurface& operator=(TextureSurface&&) noexcept;

    [[nodiscard]] TextureAspect Aspect() const noexcept;
    [[nodiscard]] const FormatTraits& Format() const noexcept;
    [[nodiscard]] TextureResource Resource(
        TextureRepresentation representation = TextureRepresentation::Current) const noexcept;
    [[nodiscard]] id<MTLTexture> Image(
        TextureRepresentation representation = TextureRepresentation::Current) const noexcept;
    [[nodiscard]] id<MTLTexture> ImageView(
        TextureView view = TextureView::Sample,
        TextureRepresentation representation = TextureRepresentation::Current);
    [[nodiscard]] id<MTLTexture> CopyImageView(
        TextureRepresentation representation = TextureRepresentation::Current);
    [[nodiscard]] id<MTLTexture> DepthView() {
        return ImageView(TextureView::Depth);
    }
    [[nodiscard]] id<MTLTexture> StencilView() {
        return ImageView(TextureView::Stencil);
    }
    [[nodiscard]] id<MTLTexture> StorageView();
    /// Convert the current guest shadow words for sampling and retain the native allocation.
    [[nodiscard]] TextureResource ShadowImage();

    /// The returned allocation contains native shadow words and may be passed to a shader.
    [[nodiscard]] ShadowTextureBinding PrepareShadowStorage(bool texture_atomics);
    void CommitShadowStorage(const ShadowTextureBinding& binding);

    void Upload(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging);
    void UploadCustom(const VideoCore::Material* material, u32 level);
    void Download(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging);
    void ScaleUp(u32 new_scale, u8 new_sample_count);
    [[nodiscard]] u32 GetInternalBytesPerPixel() const;

private:
    friend class TextureRuntime;
    friend class Framebuffer;
    TextureRuntime* runtime;
    std::shared_ptr<SurfaceStorage> storage;
};

class Framebuffer : public VideoCore::FramebufferParams {
public:
    Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                TextureSurface* color, TextureSurface* depth);
    ~Framebuffer();
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&&) noexcept;
    Framebuffer& operator=(Framebuffer&&) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] VideoCore::PixelFormat Format(VideoCore::SurfaceType type) const noexcept;
    [[nodiscard]] u32 Scale() const noexcept;
    [[nodiscard]] u8 Samples() const noexcept;
    [[nodiscard]] VideoCore::Extent Extent() const noexcept;
    [[nodiscard]] RenderPass MakePass(RenderArea area);
    void Retain(EncodingContext& context) const;

private:
    struct State;
    std::unique_ptr<State> state;
};

class Sampler {
public:
    Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params);
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) noexcept = default;
    Sampler& operator=(Sampler&&) noexcept = default;

    [[nodiscard]] id<MTLSamplerState> Handle() const noexcept {
        return sampler;
    }

    [[nodiscard]] const SamplerMetadata& Metadata() const noexcept {
        return metadata;
    }

private:
    id<MTLSamplerState> sampler = nil;
    SamplerMetadata metadata;
};

class DebugScope {
public:
    template <typename... T>
    DebugScope(TextureRuntime& runtime, Common::Vec4f color, fmt::format_string<T...> format,
               T&&... args)
        : DebugScope(runtime, color, fmt::format(format, std::forward<T>(args)...)) {}
    DebugScope(TextureRuntime& runtime, Common::Vec4f color, std::string_view label);
    ~DebugScope();
    DebugScope(const DebugScope&) = delete;
    DebugScope& operator=(const DebugScope&) = delete;

private:
    Scheduler* scheduler{};
};

struct Traits {
    using Runtime = TextureRuntime;
    using Surface = Metal::TextureSurface;
    using Sampler = Metal::Sampler;
    using Framebuffer = Metal::Framebuffer;
    using DebugScope = Metal::DebugScope;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace Metal
