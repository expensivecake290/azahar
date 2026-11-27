// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <compare>
#include <span>
#include <string_view>
#include "video_core/custom_textures/custom_format.h"
#include "video_core/pica/regs_pipeline.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_metal/mtl_common.h"
#include "video_core/renderer_metal/mtl_platform.h"

namespace Metal {

enum class TextureAspect : u8 {
    None,
    Color,
    Depth,
    DepthStencil,
};

enum class TextureConversion : u8 {
    None,
    GuestColorToRGBA8,
    RepackGuestColor16,
    DecodeGuestTexture,
    Depth24ToFloat,
    SplitDepth24Stencil8,
};

struct FormatTraits {
    MTLPixelFormat native{MTLPixelFormatInvalid};
    TextureAspect aspect{TextureAspect::None};
    TextureConversion conversion{TextureConversion::None};
    MTLTextureUsage usage{MTLTextureUsageUnknown};
    VideoCore::PixelFormat guest_format{VideoCore::PixelFormat::Invalid};
    VideoCore::CustomPixelFormat custom_format{VideoCore::CustomPixelFormat::Invalid};
    u8 block_width{1};
    u8 block_height{1};
    u8 bytes_per_block{};
    bool filterable{};

    [[nodiscard]] bool IsSupported() const noexcept {
        return native != MTLPixelFormatInvalid;
    }

    auto operator<=>(const FormatTraits&) const = default;
};

struct VertexAttributeTraits {
    MTLVertexFormat native{MTLVertexFormatInvalid};
    u8 components{};
    u8 byte_width{};
    bool convert_integer_to_float{};

    auto operator<=>(const VertexAttributeTraits&) const = default;
};

/// Immutable format selection for one device's format capabilities.
class FormatTable {
public:
    explicit FormatTable(bool supports_bc_compression);

    [[nodiscard]] const FormatTraits& GetTraits(VideoCore::PixelFormat format) const noexcept;
    [[nodiscard]] const FormatTraits& GetTraits(VideoCore::CustomPixelFormat format) const noexcept;
    [[nodiscard]] const VertexAttributeTraits& GetTraits(
        Pica::PipelineRegs::VertexAttributeFormat format, u32 count) const noexcept;

    [[nodiscard]] const std::array<VertexAttributeTraits, 16>& GetAllVertexTraits() const noexcept {
        return vertex_formats;
    }

private:
    std::array<FormatTraits, VideoCore::PIXEL_FORMAT_COUNT> guest_formats;
    std::array<FormatTraits, static_cast<u32>(VideoCore::CustomPixelFormat::ASTC8) + 1>
        custom_formats;
    std::array<VertexAttributeTraits, 16> vertex_formats;
};

/// Encodes the Metal sample counts exposed to the renderer without unchecked shifts.
[[nodiscard]] constexpr u32 SampleCountBit(u32 count) noexcept {
    switch (count) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
        return count;
    default:
        return 0;
    }
}

struct DeviceCapabilities {
    bool raster_order_groups{};
    bool fragment_barycentrics{};
    bool bc_compression{};
    u32 sample_count_mask{};
    u64 max_buffer_length{};
    u64 recommended_working_set_size{};
};

/// Selects the configured device and owns the device context used by native work leases.
class Instance {
public:
    explicit Instance(u32 physical_device_index, bool require_spatial);
    ~Instance();

    [[nodiscard]] static const DeviceInfo& SelectDevice(std::span<const DeviceInfo> devices,
                                                        u32 index, bool require_spatial);

    [[nodiscard]] id<MTLDevice> GetDevice() const noexcept {
        return context.GetDevice();
    }

    [[nodiscard]] const DeviceInfo& GetDeviceInfo() const noexcept {
        return device;
    }

    [[nodiscard]] DeviceContext& GetContext() noexcept {
        return context;
    }

    [[nodiscard]] const FormatTable& GetFormats() const noexcept {
        return formats;
    }

    [[nodiscard]] const DeviceCapabilities& GetCapabilities() const noexcept {
        return capabilities;
    }

    [[nodiscard]] bool SupportsSampleCount(u32 count) const noexcept {
        return (capabilities.sample_count_mask & SampleCountBit(count)) != 0;
    }

    [[nodiscard]] std::string_view GetOSVersion() const noexcept {
        return os_version;
    }

    [[nodiscard]] bool CanCaptureTrace() const;
    [[nodiscard]] bool IsCaptureActive() const;

private:
    DeviceInfo device;
    DeviceContext context;
    DeviceCapabilities capabilities;
    FormatTable formats;
    std::string os_version;
};

} // namespace Metal
