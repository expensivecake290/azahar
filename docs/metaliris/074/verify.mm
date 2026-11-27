#include <array>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_metalfx_spatial.h"

using namespace Metal;

static_assert(std::is_copy_constructible_v<MetalFXSpatial::Image>);
static_assert(requires(MetalFXSpatial& spatial, std::span<const SpatialRequest> requests,
                       Scheduler& scheduler, const MetalFXSpatial::Image& output,
                       const DeviceContext::WorkLease& source,
                       id<MTLDevice> device, id<MTL4Compiler> compiler,
                       id<MTL4CommandBuffer> native, MTLFXSpatialScalerDescriptor* recipe,
                       id<MTL4FXSpatialScaler> effect) {
    { spatial.Apply(requests, true, 1, source) } -> std::same_as<MetalFXSpatial::Batch>;
    { spatial.IsSupported() } -> std::same_as<bool>;
    { output.Texture() } -> std::same_as<id<MTLTexture>>;
    { output.Retain(scheduler) } -> std::same_as<void>;
    {
        [recipe newSpatialScalerWithDevice:device compiler:compiler]
    } -> std::same_as<id<MTL4FXSpatialScaler>>;
    { [effect encodeToCommandBuffer:native] } -> std::same_as<void>;
});

int main() {
    using enum SpatialStatus;
    constexpr SpatialConfiguration defaults{{400, 240},
                                            {399, 239},
                                            {721, 433},
                                            MTLPixelFormatRGBA8Unorm,
                                            MTLPixelFormatBGRA8Unorm,
                                            MTLFXSpatialScalerColorProcessingModePerceptual};
    assert(CheckSpatialConfiguration(defaults, true, true) == Eligible);
    assert(CheckSpatialConfiguration(defaults, false, true) == Disabled);
    assert(CheckSpatialConfiguration(defaults, false, false) == Disabled);
    assert(CheckSpatialConfiguration(defaults, true, false) == UnsupportedDevice);
    auto config = defaults;
    for (const auto target :
         {VideoCore::Extent{399, 239}, VideoCore::Extent{700, 237}, VideoCore::Extent{397, 400}}) {
        config.output = target;
        assert(CheckSpatialConfiguration(config, true, true) == NotUpscaling);
    }
    for (const auto target :
         {VideoCore::Extent{0, 433}, VideoCore::Extent{721, 0}, VideoCore::Extent{16385, 433}}) {
        config.output = target;
        assert(CheckSpatialConfiguration(config, true, true) == InvalidInput);
    }
    config = defaults;
    config.content.width = 401;
    assert(CheckSpatialConfiguration(config, true, true) == InvalidInput);
    for (const auto format : {MTLPixelFormatInvalid, MTLPixelFormatDepth32Float,
                              MTLPixelFormatBGRA8Unorm_sRGB, MTLPixelFormatRGBA16Float}) {
        config = defaults;
        config.input_format = format;
        assert(CheckSpatialConfiguration(config, true, true) == UnsupportedFormat);
        config = defaults;
        config.output_format = format;
        assert(CheckSpatialConfiguration(config, true, true) == UnsupportedFormat);
    }
    config = defaults;
    config.color_mode = MTLFXSpatialScalerColorProcessingModeLinear;
    config.output_format = MTLPixelFormatRGBA16Float;
    assert(CheckSpatialConfiguration(config, true, true) == Eligible);
    config.color_mode = MTLFXSpatialScalerColorProcessingModeHDR;
    assert(CheckSpatialConfiguration(config, true, true) == UnsupportedFormat);
    config.input_format = MTLPixelFormatRGBA16Float;
    assert(CheckSpatialConfiguration(config, true, true) == Eligible);

    @autoreleasepool {
        auto descriptor = [MTLFXSpatialScalerDescriptor new];
        descriptor.colorProcessingMode = defaults.color_mode;
        descriptor.inputWidth = defaults.input.width;
        descriptor.outputWidth = defaults.output.width;
        assert(descriptor.colorProcessingMode == defaults.color_mode &&
               descriptor.inputWidth == 400 && descriptor.outputWidth == 721);
        std::puts("PASS native Metal 4 Spatial SDK selectors, extent, color and bypass contracts");
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![MTLFXSpatialScalerDescriptor supportsMetal4FX:device]) {
            std::puts(
                "PENDING physical Metal 4 Spatial output, synchronization, resize and stereo");
        }
    }
}
