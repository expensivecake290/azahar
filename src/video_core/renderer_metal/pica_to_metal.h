// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <stdexcept>
#include "common/logging/log.h"
#include "video_core/pica/regs_internal.h"

namespace PicaToMetal {

using TextureConfig = Pica::TexturingRegs::TextureConfig;
using TextureFilter = TextureConfig::TextureFilter;

inline MTLSamplerMinMagFilter TextureFilterMode(TextureFilter mode) {
    switch (mode) {
    case TextureConfig::Nearest:
        return MTLSamplerMinMagFilterNearest;
    case TextureConfig::Linear:
        return MTLSamplerMinMagFilterLinear;
    }
    LOG_ERROR(Render, "Unknown Metal texture filter {}", static_cast<u32>(mode));
    return MTLSamplerMinMagFilterLinear;
}

inline MTLSamplerMipFilter TextureMipFilterMode(TextureFilter mode, bool mipmapped = true) {
    if (!mipmapped) {
        return MTLSamplerMipFilterNotMipmapped;
    }
    switch (mode) {
    case TextureConfig::Nearest:
        return MTLSamplerMipFilterNearest;
    case TextureConfig::Linear:
        return MTLSamplerMipFilterLinear;
    }
    LOG_ERROR(Render, "Unknown Metal mip filter {}", static_cast<u32>(mode));
    return MTLSamplerMipFilterLinear;
}

struct SamplerAxis {
    TextureConfig::WrapMode guest;
    MTLSamplerAddressMode native;
};

[[nodiscard]] constexpr SamplerAxis WrapMode(TextureConfig::WrapMode mode) {
    constexpr std::array modes{
        MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToBorderColor,
        MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat,
        MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToBorderColor,
        MTLSamplerAddressModeRepeat, MTLSamplerAddressModeRepeat,
    };
    return {mode, modes.at(static_cast<u32>(mode))};
}

struct SamplerBorder {
    u32 rgba8;
    std::optional<MTLSamplerBorderColor> native;
};

[[nodiscard]] constexpr SamplerBorder BorderColor(u32 rgba8) noexcept {
    switch (rgba8) {
    case 0:
        return {rgba8, MTLSamplerBorderColorTransparentBlack};
    case 0xff000000:
        return {rgba8, MTLSamplerBorderColorOpaqueBlack};
    case 0xffffffff:
        return {rgba8, MTLSamplerBorderColorOpaqueWhite};
    default:
        return {rgba8, std::nullopt};
    }
}

[[nodiscard]] constexpr bool NeedsShaderAddressing(SamplerAxis axis, const SamplerBorder& border,
                                                    u32 extent) noexcept {
    switch (axis.guest) {
    case TextureConfig::ClampToEdge2:
    case TextureConfig::ClampToBorder2:
        return true;
    case TextureConfig::ClampToBorder:
        return !border.native.has_value();
    case TextureConfig::Repeat:
    case TextureConfig::Repeat2:
    case TextureConfig::Repeat3:
    case TextureConfig::MirroredRepeat:
        return !std::has_single_bit(extent);
    default:
        return false;
    }
}

/// Resolve one integer filter tap in the asymmetric or custom-border shader path.
[[nodiscard]] constexpr std::optional<u32> WrapTexelCoordinate(TextureConfig::WrapMode mode,
                                                                s32 coordinate, u32 extent) {
    if (extent == 0) {
        return std::nullopt;
    }
    const auto raw = static_cast<u32>(coordinate);
    const bool negative = coordinate < 0;
    switch (mode) {
    case TextureConfig::ClampToEdge2:
        if (negative) {
            return raw % extent;
        }
        [[fallthrough]];
    case TextureConfig::ClampToEdge:
        return negative ? 0 : std::min(raw, extent - 1);
    case TextureConfig::ClampToBorder:
        if (negative || raw >= extent) {
            return std::nullopt;
        }
        return raw;
    case TextureConfig::ClampToBorder2:
        if (!negative && raw >= extent) {
            return std::nullopt;
        }
        [[fallthrough]];
    case TextureConfig::Repeat:
    case TextureConfig::Repeat2:
    case TextureConfig::Repeat3:
        return raw % extent;
    case TextureConfig::MirroredRepeat: {
        const u64 period = 2 * static_cast<u64>(extent);
        const u64 position = raw % period;
        return static_cast<u32>(position < extent ? position : period - 1 - position);
    }
    }
    throw std::invalid_argument("Invalid PICA texture wrap mode");
}

inline MTLBlendOperation BlendEquation(Pica::FramebufferRegs::BlendEquation equation) {
    constexpr std::array table{MTLBlendOperationAdd, MTLBlendOperationSubtract,
                               MTLBlendOperationReverseSubtract, MTLBlendOperationMin,
                               MTLBlendOperationMax};
    const auto index = static_cast<u32>(equation);
    if (index >= table.size()) {
        LOG_CRITICAL(Render, "Unknown Metal blend equation {}", index);
        return MTLBlendOperationAdd;
    }
    return table[index];
}

inline MTLBlendFactor BlendFunc(Pica::FramebufferRegs::BlendFactor factor) {
    constexpr std::array table{
        MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor,
        MTLBlendFactorOneMinusSourceColor, MTLBlendFactorDestinationColor,
        MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorSourceAlpha,
        MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorDestinationAlpha,
        MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorBlendColor,
        MTLBlendFactorOneMinusBlendColor, MTLBlendFactorBlendAlpha,
        MTLBlendFactorOneMinusBlendAlpha, MTLBlendFactorSourceAlphaSaturated,
    };
    const auto index = static_cast<u32>(factor);
    if (index >= table.size()) {
        LOG_CRITICAL(Render, "Unknown Metal blend factor {}", index);
        return MTLBlendFactorOne;
    }
    return table[index];
}

[[nodiscard]] constexpr u32 ApplyLogicOp(Pica::FramebufferRegs::LogicOp op, u32 src, u32 dst) {
    using Op = Pica::FramebufferRegs::LogicOp;
    switch (op) {
    case Op::Clear:
        return 0;
    case Op::And:
        return src & dst;
    case Op::AndReverse:
        return src & ~dst;
    case Op::Copy:
        return src;
    case Op::Set:
        return ~0U;
    case Op::CopyInverted:
        return ~src;
    case Op::NoOp:
        return dst;
    case Op::Invert:
        return ~dst;
    case Op::Nand:
        return ~(src & dst);
    case Op::Or:
        return src | dst;
    case Op::Nor:
        return ~(src | dst);
    case Op::Xor:
        return src ^ dst;
    case Op::Equiv:
        return ~(src ^ dst);
    case Op::AndInverted:
        return ~src & dst;
    case Op::OrReverse:
        return src | ~dst;
    case Op::OrInverted:
        return ~src | dst;
    }
    throw std::invalid_argument("Invalid PICA color logic operation");
}

[[nodiscard]] constexpr MTLCompareFunction CompareFunc(Pica::FramebufferRegs::CompareFunc func) {
    constexpr std::array table{
        MTLCompareFunctionNever, MTLCompareFunctionAlways, MTLCompareFunctionEqual,
        MTLCompareFunctionNotEqual, MTLCompareFunctionLess, MTLCompareFunctionLessEqual,
        MTLCompareFunctionGreater, MTLCompareFunctionGreaterEqual,
    };
    return table.at(static_cast<u32>(func));
}

[[nodiscard]] constexpr MTLStencilOperation StencilOp(Pica::FramebufferRegs::StencilAction action) {
    constexpr std::array table{
        MTLStencilOperationKeep, MTLStencilOperationZero, MTLStencilOperationReplace,
        MTLStencilOperationIncrementClamp, MTLStencilOperationDecrementClamp,
        MTLStencilOperationInvert, MTLStencilOperationIncrementWrap, MTLStencilOperationDecrementWrap,
    };
    return table.at(static_cast<u32>(action));
}

struct Primitive {
    MTLPrimitiveType native;
    bool expand_fan;
};

[[nodiscard]] constexpr Primitive PrimitiveTopology(Pica::PipelineRegs::TriangleTopology topology) {
    using Topology = Pica::PipelineRegs::TriangleTopology;
    switch (topology) {
    case Topology::List:
    case Topology::Shader:
        return {MTLPrimitiveTypeTriangle, false};
    case Topology::Strip:
        return {MTLPrimitiveTypeTriangleStrip, false};
    case Topology::Fan:
        return {MTLPrimitiveTypeTriangle, true};
    }
    throw std::invalid_argument("Invalid PICA triangle topology");
}

[[nodiscard]] constexpr MTLCullMode CullMode(Pica::RasterizerRegs::CullMode mode,
                                             bool flip_viewport) {
    using PicaCull = Pica::RasterizerRegs::CullMode;
    switch (mode) {
    case PicaCull::KeepAll:
    case PicaCull::KeepAll2:
        return MTLCullModeNone;
    case PicaCull::KeepClockWise:
    case PicaCull::KeepCounterClockWise:
        return flip_viewport ? MTLCullModeFront : MTLCullModeBack;
    }
    throw std::invalid_argument("Invalid PICA culling mode");
}

[[nodiscard]] constexpr MTLWinding FrontFace(Pica::RasterizerRegs::CullMode mode) {
    using PicaCull = Pica::RasterizerRegs::CullMode;
    switch (mode) {
    case PicaCull::KeepAll:
    case PicaCull::KeepAll2:
    case PicaCull::KeepClockWise:
        return MTLWindingCounterClockwise;
    case PicaCull::KeepCounterClockWise:
        return MTLWindingClockwise;
    }
    throw std::invalid_argument("Invalid PICA front-face mode");
}

[[nodiscard]] constexpr u16 WidenIndex(u8 index) noexcept {
    return index;
}

[[nodiscard]] inline Common::Vec4f ColorRGBA8(u32 color) {
    const auto rgba =
        Common::Vec4u{color & 0xff, (color >> 8) & 0xff, (color >> 16) & 0xff, color >> 24};
    return rgba / 255.0f;
}

} // namespace PicaToMetal
