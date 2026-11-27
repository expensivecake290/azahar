#include <cassert>
#include <cstdio>
#include <limits>
#include "video_core/renderer_metal/pica_to_metal.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/renderer_software/sw_texturing.h"

int main() {
    using namespace PicaToMetal;
    using F = Pica::FramebufferRegs;
    using R = Pica::RasterizerRegs;
    using T = Pica::PipelineRegs::TriangleTopology;
    constexpr std::array equations{MTLBlendOperationAdd, MTLBlendOperationSubtract,
                                   MTLBlendOperationReverseSubtract, MTLBlendOperationMin,
                                   MTLBlendOperationMax, MTLBlendOperationAdd,
                                   MTLBlendOperationAdd, MTLBlendOperationAdd};
    for (u32 i = 0; i < equations.size(); ++i) {
        assert(BlendEquation(static_cast<F::BlendEquation>(i)) == equations[i]);
    }
    constexpr std::array factors{
        MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor,
        MTLBlendFactorOneMinusSourceColor, MTLBlendFactorDestinationColor,
        MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorSourceAlpha,
        MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorDestinationAlpha,
        MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorBlendColor,
        MTLBlendFactorOneMinusBlendColor, MTLBlendFactorBlendAlpha,
        MTLBlendFactorOneMinusBlendAlpha, MTLBlendFactorSourceAlphaSaturated, MTLBlendFactorOne,
    };
    constexpr std::array comparisons{
        MTLCompareFunctionNever, MTLCompareFunctionAlways, MTLCompareFunctionEqual,
        MTLCompareFunctionNotEqual, MTLCompareFunctionLess, MTLCompareFunctionLessEqual,
        MTLCompareFunctionGreater, MTLCompareFunctionGreaterEqual,
    };
    constexpr std::array stencil{
        MTLStencilOperationKeep, MTLStencilOperationZero, MTLStencilOperationReplace,
        MTLStencilOperationIncrementClamp, MTLStencilOperationDecrementClamp,
        MTLStencilOperationInvert, MTLStencilOperationIncrementWrap,
        MTLStencilOperationDecrementWrap,
    };
    for (u32 i = 0; i < factors.size(); ++i) {
        assert(BlendFunc(static_cast<F::BlendFactor>(i)) == factors[i]);
    }
    for (u32 i = 0; i < 8; ++i) {
        assert(CompareFunc(static_cast<F::CompareFunc>(i)) == comparisons[i]);
        assert(StencilOp(static_cast<F::StencilAction>(i)) == stencil[i]);
    }
    for (u32 i = 0; i < 16; ++i) {
        const auto op = static_cast<F::LogicOp>(i);
        for (u32 src = 0; src < 256; ++src) {
            for (u32 dst = 0; dst < 256; ++dst) {
                assert(static_cast<u8>(ApplyLogicOp(op, src, dst)) ==
                       SwRenderer::LogicOp(static_cast<u8>(src), static_cast<u8>(dst), op));
            }
        }
    }
    constexpr std::array samplers{
        MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToBorderColor,
        MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat,
        MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToBorderColor,
        MTLSamplerAddressModeRepeat, MTLSamplerAddressModeRepeat,
    };
    for (u32 i = 0; i < 8; ++i) {
        const auto mode = static_cast<TextureConfig::WrapMode>(i);
        const auto axis = WrapMode(mode);
        assert(axis.guest == mode && axis.native == samplers[i]);
        for (u32 size : {1, 3, 16, 31, 1024}) {
            for (s32 at : {std::numeric_limits<s32>::min(), -65, -31, -3, -1, 0, 1, 2,
                           15, 16, 31, 127, std::numeric_limits<s32>::max()}) {
                const bool use_border =
                    (mode == TextureConfig::ClampToBorder &&
                     (at < 0 || at >= static_cast<s32>(size))) ||
                    (mode == TextureConfig::ClampToBorder2 && at >= static_cast<s32>(size));
                const auto wrapped = WrapTexelCoordinate(mode, at, size);
                if (use_border) {
                    assert(!wrapped);
                } else {
                    assert(wrapped == static_cast<u32>(SwRenderer::GetWrappedTexCoord(mode, at, size)));
                }
            }
        }
    }
    assert(TextureFilterMode(TextureConfig::Nearest) == MTLSamplerMinMagFilterNearest);
    assert(TextureFilterMode(TextureConfig::Linear) == MTLSamplerMinMagFilterLinear);
    assert(TextureMipFilterMode(TextureConfig::Nearest) == MTLSamplerMipFilterNearest);
    assert(TextureMipFilterMode(TextureConfig::Linear) == MTLSamplerMipFilterLinear);
    assert(TextureMipFilterMode(TextureConfig::Nearest, false) == MTLSamplerMipFilterNotMipmapped);
    assert(!WrapTexelCoordinate(TextureConfig::Repeat, 0, 0));
    assert(BorderColor(0).native == MTLSamplerBorderColorTransparentBlack);
    assert(BorderColor(0xff000000).native == MTLSamplerBorderColorOpaqueBlack);
    assert(BorderColor(0xffffffff).native == MTLSamplerBorderColorOpaqueWhite);
    const auto custom = BorderColor(0x88442211);
    assert(!custom.native && custom.rgba8 == 0x88442211);
    const auto rgba = ColorRGBA8(custom.rgba8);
    assert(rgba.x == 17 / 255.0f && rgba.y == 34 / 255.0f &&
           rgba.z == 68 / 255.0f && rgba.w == 136 / 255.0f);
    assert(NeedsShaderAddressing(WrapMode(TextureConfig::ClampToBorder), custom, 16));
    assert(!NeedsShaderAddressing(WrapMode(TextureConfig::ClampToBorder), BorderColor(0), 16));
    assert(NeedsShaderAddressing(WrapMode(TextureConfig::ClampToEdge2), BorderColor(0), 16));
    assert(NeedsShaderAddressing(WrapMode(TextureConfig::Repeat), BorderColor(0), 31));
    assert(!NeedsShaderAddressing(WrapMode(TextureConfig::Repeat), BorderColor(0), 32));
    assert(PrimitiveTopology(T::List).native == MTLPrimitiveTypeTriangle);
    assert(PrimitiveTopology(T::Strip).native == MTLPrimitiveTypeTriangleStrip);
    assert(PrimitiveTopology(T::Fan).expand_fan && !PrimitiveTopology(T::Strip).expand_fan);
    assert(PrimitiveTopology(T::Shader).native == MTLPrimitiveTypeTriangle);
    assert(CullMode(R::CullMode::KeepAll, false) == MTLCullModeNone);
    assert(CullMode(R::CullMode::KeepAll2, true) == MTLCullModeNone);
    assert(CullMode(R::CullMode::KeepClockWise, false) == MTLCullModeBack);
    assert(CullMode(R::CullMode::KeepCounterClockWise, true) == MTLCullModeFront);
    assert(FrontFace(R::CullMode::KeepClockWise) == MTLWindingCounterClockwise);
    assert(FrontFace(R::CullMode::KeepCounterClockWise) == MTLWindingClockwise);
    for (u32 value = 0; value < 256; ++value) {
        assert(WidenIndex(static_cast<u8>(value)) == value);
    }
    std::puts("PASS Metal mappings, PICA CPU wrap oracle and 1,048,576 logic-operation cases");
}
