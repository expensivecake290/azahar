#include <cassert>
#include <cstdio>
#include <limits>
#include "video_core/renderer_metal/shader/msl_fs_shader_gen.h"

using namespace Pica::Shader;
using namespace Pica::Shader::Generator::MSL;

#if defined(CHECK_NATIVE_CONSUMER)
FragmentShader BuildFragment(const FSConfig& config, const UserConfig& user) {
    const FragmentProfile profile{.color_format = VideoCore::PixelFormat::RGB565,
                                  .samples = 4,
                                  .raster_order_groups = true,
                                  .fragment_barycentrics = true,
                                  .texture_atomics = true};
    return GenerateFragmentShader(config, user, profile);
}
#else
int main() {
    static_assert(FragmentProfile{} != FragmentProfile{.samples = 4});
    using Logic = Pica::FramebufferRegs::LogicOp;
    using Blend = Pica::FramebufferRegs::BlendEquation;
    using Factor = Pica::FramebufferRegs::BlendFactor;
    static_assert(!LogicReadsDestination(Logic::NoOp));
    static_assert(LogicReadsDestination(Logic::Xor));
    static_assert(LogicReadsDestination(Logic::Or));
    static_assert(!LogicReadsDestination(Logic::Clear));
    static_assert(!LogicReadsDestination(Logic::Set));
    static_assert(!LogicReadsDestination(Logic::CopyInverted));
    static_assert(FragmentShadowBufferInfo{256, 128, 320}.WordCount() == 40960);
    static_assert(FragmentShadowBufferInfo{256, 128, 255}.WordCount() == 0);
    static_assert(FragmentShadowBufferInfo{0, 4, 4}.WordCount() == 0);
    constexpr auto max = std::numeric_limits<u32>::max();
    static_assert(FragmentShadowBufferInfo{max, max, max}.WordCount() == u64(max) * max);
    assert(RequiresMinMaxBlendShader({Blend::Min, Factor::One, Factor::SourceAlpha}));
    assert(!RequiresMinMaxBlendShader({Blend::Min, Factor::One, Factor::One}));
    assert(!RequiresMinMaxBlendShader({Blend::Add, Factor::Zero, Factor::One}));
    std::puts("PASS native fragment consumer, framebuffer feedback and shadow layout contracts");
}
#endif
