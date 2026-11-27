#import <Foundation/Foundation.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include "video_core/renderer_metal/mtl_instance.h"

using VideoCore::CustomPixelFormat;
using VideoCore::PixelFormat;
using Attribute = Pica::PipelineRegs::VertexAttributeFormat;

namespace {

void CheckTables() {
    const Metal::FormatTable table(true);
    const Metal::FormatTable without_bc(false);
    constexpr std::array pixel_formats{
        MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGR5A1Unorm,
        MTLPixelFormatB5G6R5Unorm, MTLPixelFormatABGR4Unorm, MTLPixelFormatRGBA8Unorm,
        MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm,
        MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm,
        MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth16Unorm,
        MTLPixelFormatInvalid, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8,
    };
    constexpr std::array block_bytes{4,4,2,2,2,4,4,4,4,4,4,4,4,4,2,0,4,8};
    for (u32 i = 0; i < pixel_formats.size(); ++i) {
        const auto guest = static_cast<PixelFormat>(i);
        const auto& traits = table.GetTraits(guest);
        assert(traits.native == pixel_formats[i]);
        assert(traits.bytes_per_block == block_bytes[i]);
        if (!traits.IsSupported()) {
            assert(traits == Metal::FormatTraits{});
            continue;
        }
        assert(traits.guest_format == guest);
        assert(traits.usage & MTLTextureUsageShaderRead);
        assert(bool(traits.usage & MTLTextureUsageRenderTarget) == (i < 5 || i >= 14));
        assert(bool(traits.usage & MTLTextureUsageShaderWrite) == (i == 0));
        if (i < 2) {
            assert(traits.conversion == Metal::TextureConversion::GuestColorToRGBA8);
        } else if (i < 5) {
            assert(traits.conversion == Metal::TextureConversion::RepackGuestColor16);
        } else if (i < 14) {
            assert(traits.conversion == Metal::TextureConversion::DecodeGuestTexture);
        }
    }
    assert(!table.GetTraits(PixelFormat::Invalid).IsSupported());
    assert(!table.GetTraits(static_cast<PixelFormat>(99)).IsSupported());
    assert(table.GetTraits(PixelFormat::D24).conversion == Metal::TextureConversion::Depth24ToFloat);
    assert(table.GetTraits(PixelFormat::D24S8).aspect == Metal::TextureAspect::DepthStencil);
    assert(table.GetTraits(PixelFormat::D24S8).conversion ==
           Metal::TextureConversion::SplitDepth24Stencil8);

    constexpr std::array custom_formats{
        MTLPixelFormatRGBA8Unorm, MTLPixelFormatBC1_RGBA, MTLPixelFormatBC3_RGBA,
        MTLPixelFormatBC5_RGUnorm, MTLPixelFormatBC7_RGBAUnorm, MTLPixelFormatASTC_4x4_LDR,
        MTLPixelFormatASTC_6x6_LDR, MTLPixelFormatASTC_8x6_LDR,
    };
    constexpr std::array width{1,4,4,4,4,4,6,8};
    constexpr std::array height{1,4,4,4,4,4,6,6};
    for (u32 i = 0; i < custom_formats.size(); ++i) {
        const auto format = static_cast<CustomPixelFormat>(i);
        const auto& traits = table.GetTraits(format);
        const auto& limited = without_bc.GetTraits(format);
        assert(traits.custom_format == format && limited.custom_format == format);
        assert(traits.guest_format == PixelFormat::Invalid);
        assert(traits.native == custom_formats[i] && traits.filterable);
        assert(traits.block_width == width[i] && traits.block_height == height[i]);
        assert(traits.bytes_per_block == (i == 0 ? 4 : i == 1 ? 8 : 16));
        assert(traits.conversion == Metal::TextureConversion::None);
        assert((limited.native == MTLPixelFormatInvalid) == (i >= 1 && i <= 4));
    }
    assert(!table.GetTraits(CustomPixelFormat::Invalid).IsSupported());

    constexpr std::array vertex_formats{
        MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4,
        MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4,
        MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4,
        MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4,
    };
    constexpr std::array scalar_size{1U,1U,2U,4U};
    for (u32 type = 0; type < scalar_size.size(); ++type) {
        const auto format = static_cast<Attribute>(type);
        for (u32 count = 1; count <= 4; ++count) {
            const auto& traits = table.GetTraits(format, count);
            assert(traits.native == vertex_formats[type * 4 + count - 1]);
            assert(traits.components == count && traits.byte_width == scalar_size[type] * count);
            assert(traits.convert_integer_to_float == (format != Attribute::FLOAT));
        }
        assert(table.GetTraits(format, 0) == Metal::VertexAttributeTraits{});
        assert(table.GetTraits(format, 5) == Metal::VertexAttributeTraits{});
    }
    assert(table.GetTraits(static_cast<Attribute>(99), 1) == Metal::VertexAttributeTraits{});
}

template <typename Action>
void ExpectError(Action action, std::string_view expected) {
    try {
        action();
        assert(false);
    } catch (const std::exception& e) {
        assert(std::string_view(e.what()).find(expected) != std::string_view::npos);
    }
}

void CheckSelectionFaults() {
    std::array<Metal::DeviceInfo, 1> devices{};
    ExpectError([] { (void)Metal::Instance::SelectDevice({}, 0, false); }, "No Metal devices");
    ExpectError([&] { (void)Metal::Instance::SelectDevice(devices, 1, false); }, "index");
    ExpectError([&] { (void)Metal::Instance::SelectDevice(devices, 0, false); }, "unavailable");
    devices[0].support = Metal::DeviceSupport::MissingMetal4;
    ExpectError([&] { (void)Metal::Instance::SelectDevice(devices, 0, false); }, "Metal 4");
    devices[0].support = Metal::DeviceSupport::MissingUnifiedMemory;
    ExpectError([&] { (void)Metal::Instance::SelectDevice(devices, 0, false); }, "unified memory");
    devices[0].support = Metal::DeviceSupport::Supported;
    ExpectError([&] { (void)Metal::Instance::SelectDevice(devices, 0, true); }, "MetalFX Spatial");
    ExpectError([&] { (void)Metal::Instance::SelectDevice(devices, 0, false); }, "device handle");
}

void CheckTexture(id<MTLDevice> device, const Metal::FormatTraits& traits) {
    if (!traits.IsSupported()) {
        return;
    }
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:traits.native
                                                                        width:24 height:24
                                                                     mipmapped:NO];
    descriptor.usage = traits.usage;
    descriptor.storageMode = MTLStorageModePrivate;
    const auto texture = [device newTextureWithDescriptor:descriptor];
    assert(texture && texture.pixelFormat == traits.native);
}

bool CheckLiveDevice() {
    const auto devices = Metal::EnumerateDevices();
    if (devices.empty()) {
        ExpectError([] { Metal::Instance selection(0, false); }, "No Metal devices");
        return false;
    }
    for (u32 index = 0; index < devices.size(); ++index) {
        const auto& info = devices[index];
        if (info.support != Metal::DeviceSupport::Supported) {
            continue;
        }
        Metal::Instance instance(index, info.spatial_scaling);
        id<MTLDevice> device = instance.GetDevice();
        assert(device.registryID == info.registry_id);
        assert(instance.GetContext().RegistryID() == info.registry_id);
        assert(!instance.GetOSVersion().empty());
        assert(instance.GetCapabilities().max_buffer_length == device.maxBufferLength);
        assert(instance.GetCapabilities().recommended_working_set_size ==
               device.recommendedMaxWorkingSetSize);
        for (u32 samples : {1, 2, 4, 8, 16}) {
            assert(instance.SupportsSampleCount(samples) ==
                   static_cast<bool>([device supportsTextureSampleCount:samples]));
        }
        for (u32 i = 0; i < VideoCore::PIXEL_FORMAT_COUNT; ++i) {
            CheckTexture(device, instance.GetFormats().GetTraits(static_cast<PixelFormat>(i)));
        }
        for (u32 i = 0; i <= static_cast<u32>(CustomPixelFormat::ASTC8); ++i) {
            CheckTexture(device, instance.GetFormats().GetTraits(static_cast<CustomPixelFormat>(i)));
        }
    }
    return std::ranges::any_of(devices, [](const auto& device) {
        return device.support == Metal::DeviceSupport::Supported;
    });
}

} // namespace

int main() {
    @autoreleasepool {
        CheckTables();
        CheckSelectionFaults();
        std::puts("PASS complete guest/custom/vertex traits and device-selection rejection matrix");
        if (!CheckLiveDevice()) {
            std::puts("PENDING live Metal 4 selection, limits and texture descriptors");
            return 2;
        }
        std::puts("PASS live Metal 4 selection, limits and texture descriptors");
    }
}
