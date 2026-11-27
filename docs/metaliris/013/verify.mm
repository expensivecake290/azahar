#include <concepts>
#include <type_traits>
#include "video_core/renderer_metal/mtl_instance.h"

using namespace Metal;

static_assert(!std::is_default_constructible_v<Instance>);
static_assert(!std::is_copy_constructible_v<Instance>);
static_assert(FormatTraits{}.native == MTLPixelFormatInvalid);
static_assert(FormatTraits{}.aspect == TextureAspect::None);
static_assert(VertexAttributeTraits{}.native == MTLVertexFormatInvalid);
static_assert(requires(const Instance& instance, const FormatTable& formats) {
    { instance.GetDevice() } -> std::same_as<id<MTLDevice>>;
    { instance.GetDeviceInfo() } -> std::same_as<const DeviceInfo&>;
    { formats.GetTraits(VideoCore::PixelFormat::D24S8) } -> std::same_as<const FormatTraits&>;
    { formats.GetTraits(VideoCore::CustomPixelFormat::ASTC8) } -> std::same_as<const FormatTraits&>;
    { formats.GetTraits(Pica::PipelineRegs::VertexAttributeFormat::FLOAT, 4) }
        -> std::same_as<const VertexAttributeTraits&>;
});
int main() {
    return SampleCountBit(0) != 0 || SampleCountBit(3) != 0 || SampleCountBit(16) != 16 ||
           SampleCountBit(32) != 0;
}
