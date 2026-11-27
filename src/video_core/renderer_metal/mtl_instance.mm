// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <stdexcept>
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_instance.h"

namespace Metal {

namespace {

using VideoCore::CustomPixelFormat;
using VideoCore::PixelFormat;
using VertexFormat = Pica::PipelineRegs::VertexAttributeFormat;

constexpr FormatTraits GuestColor(PixelFormat guest, MTLPixelFormat native,
                                  TextureConversion conversion, u8 bytes) {
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    if (guest == PixelFormat::RGBA8) {
        usage |= MTLTextureUsageShaderWrite;
    }
    return {.native = native,
            .aspect = TextureAspect::Color,
            .conversion = conversion,
            .usage = usage,
            .guest_format = guest,
            .bytes_per_block = bytes,
            .filterable = true};
}

constexpr FormatTraits GuestTexture(PixelFormat guest) {
    return {.native = MTLPixelFormatRGBA8Unorm,
            .aspect = TextureAspect::Color,
            .conversion = TextureConversion::DecodeGuestTexture,
            .usage = MTLTextureUsageShaderRead,
            .guest_format = guest,
            .bytes_per_block = 4,
            .filterable = true};
}

constexpr FormatTraits GuestDepth(PixelFormat guest, MTLPixelFormat native, TextureAspect aspect,
                                  TextureConversion conversion, u8 bytes) {
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    if (aspect == TextureAspect::DepthStencil) {
        usage |= MTLTextureUsagePixelFormatView;
    }
    return {.native = native,
            .aspect = aspect,
            .conversion = conversion,
            .usage = usage,
            .guest_format = guest,
            .bytes_per_block = bytes};
}

constexpr FormatTraits CustomTexture(CustomPixelFormat custom, MTLPixelFormat native,
                                    u8 width, u8 height, u8 bytes, bool supported = true) {
    return {.native = supported ? native : MTLPixelFormatInvalid,
            .aspect = TextureAspect::Color,
            .usage = supported ? MTLTextureUsageShaderRead : MTLTextureUsageUnknown,
            .custom_format = custom,
            .block_width = width,
            .block_height = height,
            .bytes_per_block = bytes,
            .filterable = supported};
}

constexpr std::array<VertexAttributeTraits, 16> MakeVertexTraits() {
    constexpr std::array<std::array<MTLVertexFormat, 4>, 4> formats{{
        {MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4},
        {MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4},
        {MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4},
        {MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4},
    }};
    static_assert(static_cast<u32>(VertexFormat::BYTE) == 0 &&
                  static_cast<u32>(VertexFormat::UBYTE) == 1 &&
                  static_cast<u32>(VertexFormat::SHORT) == 2 &&
                  static_cast<u32>(VertexFormat::FLOAT) == 3);
    constexpr std::array<u8, 4> scalar_size{1, 1, 2, 4};
    std::array<VertexAttributeTraits, 16> traits{};
    for (u8 type = 0; type < formats.size(); ++type) {
        for (u8 count = 1; count <= formats[type].size(); ++count) {
            traits[type * 4 + count - 1] = {formats[type][count - 1], count,
                                           static_cast<u8>(scalar_size[type] * count),
                                           type != static_cast<u8>(VertexFormat::FLOAT)};
        }
    }
    return traits;
}

DeviceCapabilities GetDeviceCapabilities(id<MTLDevice> device) {
    u32 sample_count_mask = 0;
    for (const u32 count : {1, 2, 4, 8, 16}) {
        if ([device supportsTextureSampleCount:count]) {
            sample_count_mask |= SampleCountBit(count);
        }
    }
    return {.raster_order_groups = static_cast<bool>(device.areRasterOrderGroupsSupported),
            .fragment_barycentrics = static_cast<bool>(device.supportsShaderBarycentricCoordinates),
            .bc_compression = static_cast<bool>(device.supportsBCTextureCompression),
            .sample_count_mask = sample_count_mask,
            .max_buffer_length = device.maxBufferLength,
            .recommended_working_set_size = device.recommendedMaxWorkingSetSize};
}

std::string ReadOSVersion() {
    NSString* version = NSProcessInfo.processInfo.operatingSystemVersionString;
    const char* utf8 = version.UTF8String;
    return utf8 ? utf8 : "";
}

} // namespace

FormatTable::FormatTable(bool supports_bc_compression)
    : guest_formats{{
          GuestColor(PixelFormat::RGBA8, MTLPixelFormatRGBA8Unorm,
                     TextureConversion::GuestColorToRGBA8, 4),
          GuestColor(PixelFormat::RGB8, MTLPixelFormatRGBA8Unorm,
                     TextureConversion::GuestColorToRGBA8, 4),
          GuestColor(PixelFormat::RGB5A1, MTLPixelFormatBGR5A1Unorm,
                     TextureConversion::RepackGuestColor16, 2),
          GuestColor(PixelFormat::RGB565, MTLPixelFormatB5G6R5Unorm,
                     TextureConversion::RepackGuestColor16, 2),
          GuestColor(PixelFormat::RGBA4, MTLPixelFormatABGR4Unorm,
                     TextureConversion::RepackGuestColor16, 2),
          GuestTexture(PixelFormat::IA8),
          GuestTexture(PixelFormat::RG8),
          GuestTexture(PixelFormat::I8),
          GuestTexture(PixelFormat::A8),
          GuestTexture(PixelFormat::IA4),
          GuestTexture(PixelFormat::I4),
          GuestTexture(PixelFormat::A4),
          GuestTexture(PixelFormat::ETC1),
          GuestTexture(PixelFormat::ETC1A4),
          GuestDepth(PixelFormat::D16, MTLPixelFormatDepth16Unorm, TextureAspect::Depth,
                     TextureConversion::None, 2),
          {}, // Reserved PICA depth encoding.
          GuestDepth(PixelFormat::D24, MTLPixelFormatDepth32Float, TextureAspect::Depth,
                     TextureConversion::Depth24ToFloat, 4),
          GuestDepth(PixelFormat::D24S8, MTLPixelFormatDepth32Float_Stencil8,
                     TextureAspect::DepthStencil, TextureConversion::SplitDepth24Stencil8, 8),
      }},
      custom_formats{{
          CustomTexture(CustomPixelFormat::RGBA8, MTLPixelFormatRGBA8Unorm, 1, 1, 4),
          CustomTexture(CustomPixelFormat::BC1, MTLPixelFormatBC1_RGBA, 4, 4, 8,
                        supports_bc_compression),
          CustomTexture(CustomPixelFormat::BC3, MTLPixelFormatBC3_RGBA, 4, 4, 16,
                        supports_bc_compression),
          CustomTexture(CustomPixelFormat::BC5, MTLPixelFormatBC5_RGUnorm, 4, 4, 16,
                        supports_bc_compression),
          CustomTexture(CustomPixelFormat::BC7, MTLPixelFormatBC7_RGBAUnorm, 4, 4, 16,
                        supports_bc_compression),
          CustomTexture(CustomPixelFormat::ASTC4, MTLPixelFormatASTC_4x4_LDR, 4, 4, 16),
          CustomTexture(CustomPixelFormat::ASTC6, MTLPixelFormatASTC_6x6_LDR, 6, 6, 16),
          CustomTexture(CustomPixelFormat::ASTC8, MTLPixelFormatASTC_8x6_LDR, 8, 6, 16),
      }},
      vertex_formats{MakeVertexTraits()} {}

const FormatTraits& FormatTable::GetTraits(PixelFormat format) const noexcept {
    static constexpr FormatTraits invalid{};
    const auto index = static_cast<u32>(format);
    return index < guest_formats.size() ? guest_formats[index] : invalid;
}

const FormatTraits& FormatTable::GetTraits(CustomPixelFormat format) const noexcept {
    static constexpr FormatTraits invalid{};
    const auto index = static_cast<u32>(format);
    return index < custom_formats.size() ? custom_formats[index] : invalid;
}

const VertexAttributeTraits& FormatTable::GetTraits(VertexFormat format, u32 count) const noexcept {
    static constexpr VertexAttributeTraits invalid{};
    const auto index = static_cast<u32>(format);
    if (index >= 4 || count < 1 || count > 4) {
        return invalid;
    }
    return vertex_formats[index * 4 + count - 1];
}

const DeviceInfo& Instance::SelectDevice(std::span<const DeviceInfo> devices, u32 index,
                                        bool require_spatial) {
    if (devices.empty()) {
        throw std::runtime_error("No Metal devices are available to this process");
    }
    if (index >= devices.size()) {
        throw std::out_of_range("Metal device index " + std::to_string(index) +
                                " is out of range (" + std::to_string(devices.size()) + " devices)");
    }
    const auto& selected = devices[index];
    switch (selected.support) {
    case DeviceSupport::Unavailable:
        throw std::runtime_error("The selected Metal device is unavailable");
    case DeviceSupport::MissingMetal4:
        throw std::runtime_error("The selected device does not support Metal 4");
    case DeviceSupport::MissingUnifiedMemory:
        throw std::runtime_error("The selected Metal device does not support unified memory");
    case DeviceSupport::Supported:
        break;
    }
    if (require_spatial && !selected.spatial_scaling) {
        throw std::runtime_error("The selected Metal device does not support MetalFX Spatial");
    }
    if (!selected.device) {
        throw std::runtime_error("The selected Metal device has no retained device handle");
    }
    return selected;
}

Instance::Instance(u32 physical_device_index, bool require_spatial)
    : device{SelectDevice(EnumerateDevices(), physical_device_index, require_spatial)},
      context{device.device}, capabilities{GetDeviceCapabilities(device.device)},
      formats{capabilities.bc_compression}, os_version{ReadOSVersion()} {
    LOG_INFO(Render, "Metal 4 device: {} (registry ID {}), {}", device.name, device.registry_id,
             os_version);
}

Instance::~Instance() {
    context.Close();
}

bool Instance::CanCaptureTrace() const {
    return CanCaptureGPUTrace();
}

bool Instance::IsCaptureActive() const {
    return MTLCaptureManager.sharedCaptureManager.isCapturing;
}

} // namespace Metal
