// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <string>
#include <string_view>
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/shader/generator/pica_fs_config.h"
#include "video_core/shader/generator/shader_uniforms.h"

namespace Pica::Shader::Generator::MSL {

enum class FragmentFeedback : u8 {
    None,
    RasterOrderGroup,
    TextureCopy,
};

enum class ShadowStorage : u8 {
    None,
    AtomicTexture,
    AtomicBuffer,
};

struct FragmentProfile {
    VideoCore::PixelFormat color_format = VideoCore::PixelFormat::RGBA8;
    u32 samples{1};
    bool raster_order_groups{};
    bool fragment_barycentrics{};
    bool texture_atomics{};

    bool operator==(const FragmentProfile&) const = default;
};

struct FragmentRequirements {
    FragmentFeedback feedback{};
    ShadowStorage shadow_storage{};
    bool emulated_rgb_blend{};
    bool emulated_alpha_blend{};
    bool fragment_barycentrics{};
    bool writes_color{};
    bool writes_depth{};

    bool operator==(const FragmentRequirements&) const = default;
};

struct FragmentShader {
    std::string source;
    FragmentRequirements requirements;
};

struct FragmentBindings {
    static constexpr u32 Uniforms = 2;
    static constexpr u32 ShadowBuffer = 3;
    static constexpr u32 ShadowBufferInfo = 4;

    static constexpr u32 Texture0 = 0;
    static constexpr u32 Texture1 = 1;
    static constexpr u32 Texture2 = 2;
    static constexpr u32 LightingFogLut = 3;
    static constexpr u32 ProceduralLut = 4;
    static constexpr u32 ProceduralColorLut = 5;
    static constexpr u32 NormalMap = 6;
    static constexpr u32 FeedbackTexture = 7;
    static constexpr u32 ShadowTarget = 8;
    static constexpr u32 FirstShadowCubeFace = 9;
    static constexpr u32 ShadowCubeFaceCount = 6;

    static constexpr u32 FirstTextureSampler = 0;
    static constexpr u32 NormalMapSampler = 3;
};

struct alignas(16) FragmentShadowBufferInfo {
    u32 width;
    u32 height;
    /// Pitch in 32-bit depth/stencil elements.
    u32 row_pitch;
    u32 reserved{};

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return width != 0 && height != 0 && row_pitch >= width;
    }

    [[nodiscard]] constexpr u64 WordCount() const noexcept {
        return IsValid() ? static_cast<u64>(row_pitch) * height : 0;
    }
};
static_assert(sizeof(FragmentShadowBufferInfo) == 16);

/// NoOp preserves the destination with the pipeline color write mask.
[[nodiscard]] constexpr bool LogicReadsDestination(FramebufferRegs::LogicOp operation) noexcept {
    using enum FramebufferRegs::LogicOp;
    return operation != Clear && operation != Copy && operation != CopyInverted &&
           operation != Set && operation != NoOp;
}

[[nodiscard]] inline bool RequiresMinMaxBlendShader(const BlendConfig& blend) noexcept {
    return blend.RequiresMinMaxEmulation() && !blend.UsesNoOpFactors();
}

/// MSL declaration shared by the vertex/geometry outputs and ordinary fragment inputs.
[[nodiscard]] std::string_view FragmentVaryingsDeclaration() noexcept;

[[nodiscard]] FragmentRequirements DescribeFragment(const FSConfig& config,
                                                    const FragmentProfile& profile) noexcept;

/// Generate PICA fragment MSL solely from the captured shader, user and device configuration.
[[nodiscard]] FragmentShader GenerateFragmentShader(const FSConfig& config, const UserConfig& user,
                                                    const FragmentProfile& profile);

} // namespace Pica::Shader::Generator::MSL
