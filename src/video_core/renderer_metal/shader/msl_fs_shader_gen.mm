// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <bit>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <fmt/format.h>
#include "video_core/renderer_metal/shader/msl_fs_shader_gen.h"

namespace Pica::Shader::Generator::MSL {
namespace {

using BlendEquation = FramebufferRegs::BlendEquation;
using BlendFactor = FramebufferRegs::BlendFactor;
using CompareFunc = FramebufferRegs::CompareFunc;
using LogicOp = FramebufferRegs::LogicOp;
using TevStage = TexturingRegs::TevStageConfig;
using Source = TevStage::Source;
using Operation = TevStage::Operation;
using TextureType = TexturingRegs::TextureConfig::TextureType;
using WrapMode = TexturingRegs::TextureConfig::WrapMode;
using FogMode = TexturingRegs::FogMode;
using LightingSampler = LightingRegs::LightingSampler;

constexpr std::string_view Varyings = R"MSL(
#ifndef METALIRIS_FRAGMENT_VARYINGS
#define METALIRIS_FRAGMENT_VARYINGS
namespace MetalirisFragment {
struct Varyings {
    metal::float4 position [[position]];
    metal::float4 primary_color [[user(locn1)]];
    metal::float2 texcoord0 [[user(locn2)]];
    metal::float2 texcoord1 [[user(locn3)]];
    metal::float2 texcoord2 [[user(locn4)]];
    float texcoord0_w [[user(locn5)]];
    metal::float4 normquat [[user(locn6)]];
    metal::float3 view [[user(locn7)]];
};
}
#endif
)MSL";

constexpr std::string_view Uniforms = R"MSL(
namespace MetalirisFragment {
using namespace metal;

struct Light {
    alignas(16) packed_float3 specular_0;
    alignas(16) packed_float3 specular_1;
    alignas(16) packed_float3 diffuse;
    alignas(16) packed_float3 ambient;
    alignas(16) packed_float3 position;
    alignas(16) packed_float3 spot_direction;
    float dist_atten_bias;
    float dist_atten_scale;
};

struct Uniforms {
    int framebuffer_scale;
    int alphatest_ref;
    float depth_scale;
    float depth_offset;
    float shadow_bias_constant;
    float shadow_bias_linear;
    int scissor_x1;
    int scissor_y1;
    int scissor_x2;
    int scissor_y2;
    int fog_lut_offset;
    int proctex_noise_lut_offset;
    int proctex_color_map_offset;
    int proctex_alpha_map_offset;
    int proctex_lut_offset;
    int proctex_diff_lut_offset;
    float proctex_bias;
    int shadow_texture_bias;
    int4 lighting_lut_offset[6];
    alignas(16) packed_float3 fog_color;
    float2 proctex_noise_f;
    float2 proctex_noise_a;
    float2 proctex_noise_p;
    alignas(16) packed_float3 lighting_global_ambient;
    Light light_src[8];
    float4 const_color[6];
    float4 tev_combiner_buffer_color;
    alignas(16) packed_float3 tex_lod_bias;
    float4 tex_border_color[3];
    float4 blend_color;
};

static_assert(sizeof(Light) == 112, "PICA light layout");
static_assert(sizeof(Uniforms) == 0x530, "PICA fragment uniforms");
static_assert(__builtin_offsetof(Uniforms, light_src) == 240, "PICA light array");
static_assert(__builtin_offsetof(Uniforms, const_color) == 1136, "PICA TEV constants");
static_assert(__builtin_offsetof(Uniforms, blend_color) == 1312, "PICA blend uniform");

template <typename T>
inline T ByteRound(T value) {
    return round(value * 255.0f) * (1.0f / 255.0f);
}

inline float4 SamplePicaTexture(texture2d<float> tex, sampler state, float2 uv, float lod_bias) {
    const float2 texels = uv * float2(tex.get_width(), tex.get_height());
    const float2 delta = max(abs(dfdx(texels)), abs(dfdy(texels)));
    return tex.sample(state, uv, level(log2(max(delta.x, delta.y)) + lod_bias));
}
)MSL";

static_assert(sizeof(FSUniformData) == 0x530);
static_assert(offsetof(FSUniformData, light_src) == 240);
static_assert(offsetof(FSUniformData, const_color) == 1136);
static_assert(offsetof(FSUniformData, blend_color) == 1312);

constexpr std::string_view BarycentricInput = R"MSL(
struct BarycentricVaryings {
    float4 position [[position]];
    float4 primary_color [[user(locn1)]];
    float2 texcoord0 [[user(locn2)]];
    float2 texcoord1 [[user(locn3)]];
    float2 texcoord2 [[user(locn4)]];
    float texcoord0_w [[user(locn5)]];
    vertex_value<float4> normquat [[user(locn6)]];
    float3 view [[user(locn7)]];
    float3 barycentric [[barycentric_coord, center_perspective]];
};
)MSL";

constexpr std::string_view LightingHelpers = R"MSL(
inline float3 RotateQuaternion(float4 q, float3 v) {
    return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

inline float LightingLut(texture_buffer<float> lut, constant Uniforms& u, uint slot,
                         float value, bool unsigned_index) {
    const float scaled = value * (unsigned_index ? 256.0f : 128.0f);
    const float floored = clamp(floor(scaled), unsigned_index ? 0.0f : -128.0f,
                                unsigned_index ? 255.0f : 127.0f);
    const uint index = uint(int(floored) & 255);
    const uint offset = uint(u.lighting_lut_offset[slot >> 2u][slot & 3u]);
    const float2 entry = lut.read(offset + index).rg;
    return fma(entry.g, scaled - floored, entry.r);
}
)MSL";

constexpr std::string_view ShadowSampling = R"MSL(
inline float ShadowCompare(uint packed_pixel, uint depth) {
    return (packed_pixel >> 8u) <= depth ? 0.0f :
           float(packed_pixel & 255u) * (1.0f / 255.0f);
}

inline float ReadShadow(texture2d<uint, access::read> tex, int2 xy, int2 extent, uint depth,
                        bool clamp_edges) {
    if (clamp_edges) {
        xy = clamp(xy, int2(0), extent - 1);
    } else if (any(xy < 0) || any(xy >= extent)) {
        return 1.0f;
    }
    return ShadowCompare(tex.read(uint2(xy)).r, depth);
}

inline float4 SampleShadow(texture2d<uint, access::read> tex, float2 uv, float w, int bias,
                           bool clamp_edges) {
    if (!all(isfinite(uv))) {
        return float4(1.0f);
    }
    const uint depth = uint(max(0, int(min(abs(w), 1.0f) * 16777215.0f) - bias));
    const int2 extent = int2(tex.get_width(), tex.get_height());
    const float2 coord = float2(extent) * uv - 0.5f;
    if (!clamp_edges && (any(coord < -1.0f) || any(coord >= float2(extent)))) {
        return float4(1.0f);
    }
    const int2 i = int2(floor(coord));
    const float2 f = fract(coord);
    const float s00 = ReadShadow(tex, i, extent, depth, clamp_edges);
    const float s10 = ReadShadow(tex, i + int2(1, 0), extent, depth, clamp_edges);
    const float s01 = ReadShadow(tex, i + int2(0, 1), extent, depth, clamp_edges);
    const float s11 = ReadShadow(tex, i + int2(1, 1), extent, depth, clamp_edges);
    return float4(mix(mix(s00, s01, f.y), mix(s10, s11, f.y), f.x));
}
)MSL";

constexpr std::string_view ShadowUpdate = R"MSL(
inline uint UpdateShadow(uint previous, uint depth, uint opacity, constant Uniforms& u) {
    const uint previous_depth = previous >> 8u;
    if (depth >= previous_depth) {
        return previous;
    }
    if (opacity == 0u) {
        return (depth << 8u) | (previous & 255u);
    }
    const float bias = u.shadow_bias_constant +
                       u.shadow_bias_linear * float(depth) / float(previous_depth);
    const uint scaled = uint(clamp(float(opacity) / bias, 0.0f, 255.0f));
    return (previous & 0xFFFFFF00u) | min(scaled, previous & 255u);
}
)MSL";

constexpr std::string_view ProceduralHelpers = R"MSL(
inline float ProceduralLut(texture_buffer<float> lut, int offset, float value) {
    const float scaled = 128.0f * value;
    const float index = clamp(floor(scaled), 0.0f, 127.0f);
    const float2 entry = lut.read(uint(int(index) + offset)).rg;
    return saturate(fma(entry.g, scaled - index, entry.r));
}

inline int ProceduralRandom1D(int value) {
    const int table[16] = {0,4,10,8,4,9,7,12,5,15,13,14,11,15,2,11};
    return (((value % 9 + 2) * 3) & 15) ^ table[(value / 9) & 15];
}

inline float ProceduralRandom2D(int2 point) {
    const int table[16] = {10,2,15,8,0,7,4,5,5,13,2,6,13,9,3,14};
    const int x = ProceduralRandom1D(point.x);
    int y = ProceduralRandom1D(point.y);
    y += (x & 3) == 1 ? 4 : 0;
    y ^= (x & 1) * 6;
    y = ((y + 10 + x) & 15) ^ table[x];
    return fma(float(y), 2.0f / 15.0f, -1.0f);
}

inline float ProceduralNoise(texture_buffer<float> lut, constant Uniforms& u, float2 uv) {
    const float2 grid = 9.0f * u.proctex_noise_f * abs(uv + u.proctex_noise_p);
    const int2 point = int2(floor(grid));
    const float2 f = fract(grid);
    const float sum = f.x + f.y;
    const float g0 = ProceduralRandom2D(point) * sum;
    const float g1 = ProceduralRandom2D(point + int2(1,0)) * (sum - 1.0f);
    const float g2 = ProceduralRandom2D(point + int2(0,1)) * (sum - 1.0f);
    const float g3 = ProceduralRandom2D(point + int2(1,1)) * (sum - 2.0f);
    const float x = ProceduralLut(lut, u.proctex_noise_lut_offset, f.x);
    const float y = ProceduralLut(lut, u.proctex_noise_lut_offset, f.y);
    return mix(mix(g0, g1, x), mix(g2, g3, x), y);
}

inline float4 ProceduralColor(texture_buffer<float> lut, constant Uniforms& u, float value,
                              int lod, int width, const thread int* offsets, bool linear) {
    lod = clamp(lod, 0, 7);
    const float coord = value * float(max(1, width >> lod) - 1);
    const int base = offsets[lod];
    if (!linear) {
        return lut.read(uint(int(round(coord + float(base))) + u.proctex_lut_offset));
    }
    const int index = int(coord) + base;
    return lut.read(uint(index + u.proctex_lut_offset)) +
           fract(coord) * lut.read(uint(index + u.proctex_diff_lut_offset));
}
)MSL";

bool IsPassThrough(const TevStage& stage) {
    return stage.color_op == Operation::Replace && stage.alpha_op == Operation::Replace &&
           stage.color_source1 == Source::Previous && stage.alpha_source1 == Source::Previous &&
           stage.color_modifier1 == TevStage::ColorModifier::SourceColor &&
           stage.alpha_modifier1 == TevStage::AlphaModifier::SourceAlpha &&
           stage.GetColorMultiplier() == 1 && stage.GetAlphaMultiplier() == 1;
}

u32 OperandCount(Operation operation) {
    switch (operation) {
    case Operation::Replace:
        return 1;
    case Operation::Modulate:
    case Operation::Add:
    case Operation::AddSigned:
    case Operation::Subtract:
    case Operation::Dot3_RGB:
    case Operation::Dot3_RGBA:
        return 2;
    case Operation::Lerp:
    case Operation::MultiplyThenAdd:
    case Operation::AddThenMultiply:
        return 3;
    }
    return 0;
}

bool HasColor(const FragmentProfile& profile) {
    return profile.color_format != VideoCore::PixelFormat::Invalid;
}

std::array<u32, 4> ColorMasks(VideoCore::PixelFormat format) {
    using enum VideoCore::PixelFormat;
    switch (format) {
    case RGBA8:
        return {255, 255, 255, 255};
    case RGB8:
        return {255, 255, 255, 0};
    case RGB5A1:
        return {31, 31, 31, 1};
    case RGB565:
        return {31, 63, 31, 0};
    case RGBA4:
        return {15, 15, 15, 15};
    default:
        throw std::invalid_argument("Invalid PICA fragment color format");
    }
}

std::string_view LogicExpression(LogicOp op) {
    using enum LogicOp;
    switch (op) {
    case Clear:
        return "uint4(0u)";
    case And:
        return "src & dst";
    case AndReverse:
        return "src & ~dst";
    case Copy:
        return "src";
    case Set:
        return "mask";
    case CopyInverted:
        return "~src";
    case NoOp:
        return "dst";
    case Invert:
        return "~dst";
    case Nand:
        return "~(src & dst)";
    case Or:
        return "src | dst";
    case Nor:
        return "~(src | dst)";
    case Xor:
        return "src ^ dst";
    case Equiv:
        return "~(src ^ dst)";
    case AndInverted:
        return "~src & dst";
    case OrReverse:
        return "src | ~dst";
    case OrInverted:
        return "~src | dst";
    }
    return "src";
}

std::string_view FactorExpression(BlendFactor factor) {
    using enum BlendFactor;
    switch (factor) {
    case Zero:
        return "float4(0.0f)";
    case One:
        return "float4(1.0f)";
    case SourceColor:
        return "source_color";
    case OneMinusSourceColor:
        return "1.0f - source_color";
    case DestColor:
        return "dest_color";
    case OneMinusDestColor:
        return "1.0f - dest_color";
    case SourceAlpha:
        return "source_color.aaaa";
    case OneMinusSourceAlpha:
        return "1.0f - source_color.aaaa";
    case DestAlpha:
        return "dest_color.aaaa";
    case OneMinusDestAlpha:
        return "1.0f - dest_color.aaaa";
    case ConstantColor:
        return "u.blend_color";
    case OneMinusConstantColor:
        return "1.0f - u.blend_color";
    case ConstantAlpha:
        return "u.blend_color.aaaa";
    case OneMinusConstantAlpha:
        return "1.0f - u.blend_color.aaaa";
    case SourceAlphaSaturate:
        return "float4(float3(min(source_color.a, 1.0f - dest_color.a)), 1.0f)";
    }
    return "float4(1.0f)";
}

std::string_view AlphaReject(CompareFunc func) {
    using enum CompareFunc;
    switch (func) {
    case Equal:
        return "!=";
    case NotEqual:
        return "==";
    case LessThan:
        return ">=";
    case LessThanOrEqual:
        return ">";
    case GreaterThan:
        return "<=";
    case GreaterThanOrEqual:
        return "<";
    case Never:
    case Always:
        return {};
    }
    return {};
}

bool NeedsTexture(const TevStage& stage, Source desired, u32 index) {
    if (IsPassThrough(stage)) {
        return false;
    }
    const std::array rgb{stage.color_source1.Value(), stage.color_source2.Value(),
                         stage.color_source3.Value()};
    const std::array alpha{stage.alpha_source1.Value(), stage.alpha_source2.Value(),
                           stage.alpha_source3.Value()};
    const auto uses = [&](const std::array<Source, 3>& sources, Operation operation) {
        for (u32 i = 0; i < OperandCount(operation); ++i) {
            const auto source =
                index == 0 && sources[i] == Source::Previous ? sources[2] : sources[i];
            if (source == desired) {
                return true;
            }
        }
        return false;
    };
    return uses(rgb, stage.color_op) ||
           (stage.color_op != Operation::Dot3_RGBA && uses(alpha, stage.alpha_op));
}

std::array<bool, 4> TextureUse(const FSConfig& config, const UserConfig& user) {
    std::array<bool, 4> used{};
    for (u32 unit = 0; unit < used.size(); ++unit) {
        const auto source = static_cast<Source>(static_cast<u32>(Source::Texture0) + unit);
        for (u32 i = 0; i < config.texture.tev_stages.size(); ++i) {
            used[unit] |= NeedsTexture(config.texture.tev_stages[i], source, i);
        }
    }
    if (config.lighting.enable) {
        if (!user.use_custom_normal &&
            (config.lighting.bump_mode == LightingRegs::LightingBumpMode::NormalMap ||
             config.lighting.bump_mode == LightingRegs::LightingBumpMode::TangentMap)) {
            used[config.lighting.bump_selector] = true;
        }
        if (config.lighting.enable_shadow) {
            used[config.lighting.shadow_selector] = true;
        }
    }
    used[0] &= config.texture.texture0_type != TextureType::Disabled;
    used[3] &= config.proctex.enable != 0;
    return used;
}

class FragmentModule {
public:
    FragmentModule(const FSConfig& config, const UserConfig& user, const FragmentProfile& profile)
        : config(config), user(user), profile(profile),
          result{"", DescribeFragment(config, profile)}, used(TextureUse(config, user)) {}

    FragmentShader Generate();

private:
    template <typename... Args>
    void Emit(fmt::format_string<Args...> code, Args&&... args) {
        fmt::format_to(std::back_inserter(result.source), code, std::forward<Args>(args)...);
    }

    void DeclareEntry();
    void WriteSamples();
    void WriteProcedural(u32 coord);
    void WriteLighting();
    void WriteTev(u32 index);
    void WriteOutput();
    void WriteShadowOutput();
    void ReturnColor(std::string_view color);
    std::string TevSource(Source source, u32 index);
    std::string ColorOperand(Source source, TevStage::ColorModifier modifier, u32 index);
    std::string AlphaOperand(Source source, TevStage::AlphaModifier modifier, u32 index);

    const FSConfig& config;
    const UserConfig& user;
    const FragmentProfile& profile;
    FragmentShader result;
    const std::array<bool, 4> used;
};

} // Anonymous namespace

namespace {

void FragmentModule::DeclareEntry() {
    const auto& req = result.requirements;
    const bool shadow = req.shadow_storage != ShadowStorage::None;
    if (!shadow) {
        result.source += "struct FragmentOutput {\n";
        if (HasColor(profile)) {
            result.source += req.feedback == FragmentFeedback::RasterOrderGroup
                                 ? "    float4 color [[color(0), raster_order_group(0)]];\n"
                                 : "    float4 color [[color(0)]];\n";
        }
        result.source += "    float depth [[depth(any)]];\n};\n";
    }

    Emit("fragment {} pica_fragment({} in [[stage_in]],\n"
         "    constant Uniforms& u [[buffer({})]]",
         shadow ? "void" : "FragmentOutput",
         req.fragment_barycentrics ? "BarycentricVaryings" : "Varyings",
         FragmentBindings::Uniforms);
    for (u32 unit = 0; unit < 3; ++unit) {
        if (!used[unit]) {
            continue;
        }
        const auto type = unit == 0 ? config.texture.texture0_type.Value() : TextureType::Texture2D;
        if (type == TextureType::ShadowCube) {
            for (u32 face = 0; face < FragmentBindings::ShadowCubeFaceCount; ++face) {
                Emit(",\n    texture2d<uint, access::read> shadow{} [[texture({})]]", face,
                     FragmentBindings::FirstShadowCubeFace + face);
            }
        } else if (type == TextureType::Shadow2D) {
            Emit(",\n    texture2d<uint, access::read> tex0 [[texture({})]]", unit);
        } else {
            Emit(",\n    {} tex{} [[texture({})]], sampler s{} [[sampler({})]]",
                 type == TextureType::TextureCube ? "texturecube<float>" : "texture2d<float>", unit,
                 unit, unit, FragmentBindings::FirstTextureSampler + unit);
        }
    }
    if (used[3]) {
        Emit(",\n    texture_buffer<float> proc_rg [[texture({})]],"
             "\n    texture_buffer<float> proc_rgba [[texture({})]]",
             FragmentBindings::ProceduralLut, FragmentBindings::ProceduralColorLut);
    }
    if (config.lighting.enable || config.texture.fog_mode == FogMode::Fog) {
        Emit(",\n    texture_buffer<float> light_fog [[texture({})]]",
             FragmentBindings::LightingFogLut);
    }
    if (config.lighting.enable && user.use_custom_normal) {
        Emit(",\n    texture2d<float> normal_map [[texture({})]],"
             "\n    sampler normal_sampler [[sampler({})]]",
             FragmentBindings::NormalMap, FragmentBindings::NormalMapSampler);
    }
    if (req.feedback == FragmentFeedback::RasterOrderGroup) {
        Emit(",\n    float4 dest_color [[color(0), raster_order_group(0)]]");
    }
    if (req.feedback == FragmentFeedback::TextureCopy) {
        Emit(",\n    {} feedback_texture [[texture({})]]",
             profile.samples > 1 ? "texture2d_ms<float, access::read>"
                                 : "texture2d<float, access::read>",
             FragmentBindings::FeedbackTexture);
    }
    if (req.feedback != FragmentFeedback::None && profile.samples > 1) {
        Emit(",\n    uint sample_index [[sample_id]]");
    }
    if (req.shadow_storage == ShadowStorage::AtomicTexture) {
        Emit(",\n    texture2d<uint, access::read_write> shadow_target [[texture({})]]",
             FragmentBindings::ShadowTarget);
    }
    if (req.shadow_storage == ShadowStorage::AtomicBuffer) {
        Emit(",\n    device atomic_uint* shadow_target [[buffer({})]],"
             "\n    constant uint4& shadow_extent [[buffer({})]]",
             FragmentBindings::ShadowBuffer, FragmentBindings::ShadowBufferInfo);
    }
    Emit(") {{\n");
}

std::string_view ProcClamp(TexturingRegs::ProcTexClamp mode) {
    using enum TexturingRegs::ProcTexClamp;
    switch (mode) {
    case ToZero:
        return "v > 1.0f ? 0.0f : v";
    case ToEdge:
        return "min(v, 1.0f)";
    case SymmetricalRepeat:
        return "fract(v)";
    case MirroredRepeat:
        return "(int(v) & 1) ? 1.0f - fract(v) : fract(v)";
    case Pulse:
        return "v > 0.5f ? 1.0f : 0.0f";
    }
    return "min(v, 1.0f)";
}

std::string_view ProcCombine(TexturingRegs::ProcTexCombiner operation) {
    using enum TexturingRegs::ProcTexCombiner;
    switch (operation) {
    case U:
        return "pu";
    case U2:
        return "pu * pu";
    case V:
        return "pv";
    case V2:
        return "pv * pv";
    case Add:
        return "(pu + pv) * 0.5f";
    case Add2:
        return "(pu * pu + pv * pv) * 0.5f";
    case SqrtAdd2:
        return "min(sqrt(pu * pu + pv * pv), 1.0f)";
    case Min:
        return "min(pu, pv)";
    case Max:
        return "max(pu, pv)";
    case RMax:
        return "min(((pu + pv) * 0.5f + sqrt(pu * pu + pv * pv)) * 0.5f, 1.0f)";
    }
    return "0.0f";
}

std::string ProcShift(TexturingRegs::ProcTexShift mode, TexturingRegs::ProcTexClamp clamp,
                      char axis) {
    if (mode != TexturingRegs::ProcTexShift::Even && mode != TexturingRegs::ProcTexShift::Odd) {
        return "0.0f";
    }
    const bool even = mode == TexturingRegs::ProcTexShift::Even;
    const auto step = clamp == TexturingRegs::ProcTexClamp::MirroredRepeat ? "1.0f" : "0.5f";
    return fmt::format("float(((int(uv.{}) + {}) / 2) & 1) * {}", axis, even ? 1 : 0, step);
}

void FragmentModule::WriteProcedural(u32 coord) {
    const auto& proc = config.proctex;
    const auto filter = proc.lut_filter.Value();
    using enum TexturingRegs::ProcTexFilter;
    const bool linear =
        filter == Linear || filter == LinearMipmapNearest || filter == LinearMipmapLinear;
    const s32 width = std::clamp(proc.lut_width, 1, 255);
    const u16 max_lod = std::min<u16>(7, proc.lod_max);
    const u16 min_lod = std::min(max_lod, proc.lod_min);
    Emit("    {{\n"
         "        float2 uv = abs(in.texcoord{});\n"
         "        const float2 delta = max(abs(dfdx(uv)), abs(dfdy(uv)));\n"
         "        const float lod = clamp(u.proctex_bias == 0.0f ? 0.0f : "
         "log2(abs({}.0f * u.proctex_bias) * (delta.x + delta.y)), {}.0f, {}.0f);\n"
         "        const float u_shift = {};\n"
         "        const float v_shift = {};\n",
         coord, width, min_lod, max_lod, ProcShift(proc.u_shift, proc.u_clamp, 'y'),
         ProcShift(proc.v_shift, proc.v_clamp, 'x'));
    if (proc.noise_enable) {
        Emit("        uv = abs(uv + u.proctex_noise_a * ProceduralNoise(proc_rg, u, uv));\n");
    }
    Emit("        const auto clamp_u = [](float v) {{ return {}; }};\n"
         "        const auto clamp_v = [](float v) {{ return {}; }};\n"
         "        const float pu = clamp_u(uv.x + u_shift);\n"
         "        const float pv = clamp_v(uv.y + v_shift);\n"
         "        const float mapped = ProceduralLut(proc_rg, u.proctex_color_map_offset, {});\n"
         "        const int offsets[8] = {{{}, {}, {}, {}, 0xF0, 0xF8, 0xFC, 0xFE}};\n",
         ProcClamp(proc.u_clamp), ProcClamp(proc.v_clamp), ProcCombine(proc.color_combiner),
         proc.lut_offset0, proc.lut_offset1, proc.lut_offset2, proc.lut_offset3);

    const auto sample = [&](std::string_view level) {
        return fmt::format("ProceduralColor(proc_rgba, u, mapped, {}, {}, offsets, {})", level,
                           width, linear ? "true" : "false");
    };
    if (filter == NearestMipmapLinear || filter == LinearMipmapLinear) {
        Emit("        const int mip = int(lod);\n"
             "        samples[3] = mix({}, {}, fract(lod));\n",
             sample("mip"), sample("mip + 1"));
    } else {
        const bool mip = filter == NearestMipmapNearest || filter == LinearMipmapNearest;
        Emit("        samples[3] = {};\n", sample(mip ? "int(round(lod))" : "0"));
    }
    if (proc.separate_alpha) {
        Emit("        samples[3].a = ProceduralLut(proc_rg, u.proctex_alpha_map_offset, {});\n",
             ProcCombine(proc.alpha_combiner));
    }
    Emit("    }}\n");
}

void FragmentModule::WriteSamples() {
    Emit("    float4 samples[4] = {{}};\n");
    for (u32 unit = 0; unit < 3; ++unit) {
        if (!used[unit]) {
            continue;
        }
        const u32 coord = unit == 2 && config.texture.texture2_use_coord1 ? 1 : unit;
        Emit("    {{\n        const float2 uv = in.texcoord{};\n", coord);
        const bool border_s = config.texture.requested_wrap[unit].s == WrapMode::ClampToBorder ||
                              config.texture.texture_border_color[unit].enable_s;
        const bool border_t = config.texture.requested_wrap[unit].t == WrapMode::ClampToBorder ||
                              config.texture.texture_border_color[unit].enable_t;
        const auto type = unit == 0 ? config.texture.texture0_type.Value() : TextureType::Texture2D;
        switch (type) {
        case TextureType::Texture2D:
            Emit("        samples[{0}] = SamplePicaTexture(tex{0}, s{0}, uv, "
                 "u.tex_lod_bias[{0}]);\n",
                 unit);
            break;
        case TextureType::Projection2D:
            Emit("        samples[0] = tex0.sample(s0, uv / in.texcoord0_w);\n");
            break;
        case TextureType::TextureCube:
            Emit("        samples[0] = tex0.sample(s0, float3(uv, in.texcoord0_w));\n");
            break;
        case TextureType::Shadow2D:
            Emit("        samples[0] = SampleShadow(tex0, {}, in.texcoord0_w,"
                 " u.shadow_texture_bias, false);\n",
                 config.texture.shadow_texture_orthographic ? "uv" : "uv / in.texcoord0_w");
            break;
        case TextureType::ShadowCube:
            result.source += R"MSL(
        const float3 direction(uv, in.texcoord0_w);
        const float3 extent = abs(direction);
        if (extent.x > extent.y && extent.x > extent.z) {
            const float2 p = float2(direction.x < 0.0f ? direction.z : -direction.z, -direction.y);
            const float2 st = p / extent.x * 0.5f + 0.5f;
            samples[0] = direction.x > 0.0f
                ? SampleShadow(shadow0, st, extent.x, u.shadow_texture_bias, true)
                : SampleShadow(shadow1, st, extent.x, u.shadow_texture_bias, true);
        } else if (extent.y > extent.z) {
            const float2 p = float2(direction.x, direction.y < 0.0f ? -direction.z : direction.z);
            const float2 st = p / extent.y * 0.5f + 0.5f;
            samples[0] = direction.y > 0.0f
                ? SampleShadow(shadow2, st, extent.y, u.shadow_texture_bias, true)
                : SampleShadow(shadow3, st, extent.y, u.shadow_texture_bias, true);
        } else {
            const float2 p = float2(direction.z > 0.0f ? direction.x : -direction.x, -direction.y);
            const float2 st = p / extent.z * 0.5f + 0.5f;
            samples[0] = direction.z > 0.0f
                ? SampleShadow(shadow4, st, extent.z, u.shadow_texture_bias, true)
                : SampleShadow(shadow5, st, extent.z, u.shadow_texture_bias, true);
        }
)MSL";
            break;
        case TextureType::Disabled:
            break;
        default:
            Emit("        samples[0] = tex0.sample(s0, uv);\n");
            break;
        }
        if (border_s || border_t) {
            Emit("        if ({} || {}) {{ samples[{}] = u.tex_border_color[{}]; }}\n",
                 border_s ? "(uv.x < 0.0f || uv.x > 1.0f)" : "false",
                 border_t ? "(uv.y < 0.0f || uv.y > 1.0f)" : "false", unit, unit);
        }
        Emit("    }}\n");
    }
    if (used[3]) {
        const u32 coord = config.proctex.coord < 3 ? config.proctex.coord.Value() : 0;
        WriteProcedural(coord);
    }
}

void FragmentModule::WriteLighting() {
    Emit("    float4 primary_fragment = float4(0.0f);\n"
         "    float4 secondary_fragment = float4(0.0f);\n");
    const auto& lights = config.lighting;
    if (!lights.enable) {
        return;
    }
    Emit("    float3 surface_normal(0.0f, 0.0f, 1.0f);\n"
         "    float3 surface_tangent(1.0f, 0.0f, 0.0f);\n");
    if (user.use_custom_normal) {
        Emit("    surface_normal = 2.0f * normal_map.sample(normal_sampler, in.texcoord0).rgb - "
             "1.0f;\n");
    } else if (lights.bump_mode == LightingRegs::LightingBumpMode::NormalMap) {
        Emit("    surface_normal = 2.0f * samples[{}].rgb - 1.0f;\n", lights.bump_selector.Value());
        if (lights.bump_renorm) {
            Emit("    surface_normal.z = sqrt(max(1.0f - dot(surface_normal.xy, "
                 "surface_normal.xy), 0.0f));\n");
        }
    } else if (lights.bump_mode == LightingRegs::LightingBumpMode::TangentMap) {
        Emit("    surface_tangent = 2.0f * samples[{}].rgb - 1.0f;\n",
             lights.bump_selector.Value());
    }

    if (result.requirements.fragment_barycentrics) {
        result.source += R"MSL(
    const float4 q0 = in.normquat.get(vertex_index::first);
    const float4 q1 = in.normquat.get(vertex_index::second);
    const float4 q2 = in.normquat.get(vertex_index::third);
    const float4 normquat = q0 * in.barycentric.x +
        (dot(q0, q1) < 0.0f ? -q1 : q1) * in.barycentric.y +
        (dot(q0, q2) < 0.0f ? -q2 : q2) * in.barycentric.z;
)MSL";
    } else {
        Emit("    const float4 normquat = in.normquat;\n");
    }
    Emit("    const float4 q = normalize(normquat);\n"
         "    const float3 normal = RotateQuaternion(q, surface_normal);\n"
         "    const float3 tangent = RotateQuaternion(q, surface_tangent);\n"
         "    const float3 view = normalize(in.view);\n"
         "    float4 diffuse_sum(0.0f, 0.0f, 0.0f, 1.0f);\n"
         "    float4 specular_sum(0.0f, 0.0f, 0.0f, 1.0f);\n");
    if (lights.enable_shadow) {
        Emit("    const float4 shadow = {}samples[{}];\n", lights.shadow_invert ? "1.0f - " : "",
             lights.shadow_selector.Value());
    } else {
        Emit("    const float4 shadow(1.0f);\n");
    }

    const auto lut_value = [&](const LutConfig& lut, LightingSampler sampler, u32 light_num) {
        using enum LightingRegs::LightingLutInput;
        std::string_view position = "0.0f";
        switch (lut.type) {
        case NH:
            position = "dot(normal, normalize(half_vector))";
            break;
        case VH:
            position = "dot(view, normalize(half_vector))";
            break;
        case NV:
            position = "dot(normal, view)";
            break;
        case LN:
            position = "dot(light_vector, normal)";
            break;
        case SP:
            position = "dot(light_vector, float3(light.spot_direction))";
            break;
        case CP:
            if (lights.config == LightingRegs::LightingConfig::Config7) {
                position = "dot(normalize(half_vector) - normal * dot(normal, "
                           "normalize(half_vector)), tangent)";
            }
            break;
        }
        const auto coord = !lut.abs_input ? std::string(position)
                           : lights.lights[light_num].two_sided_diffuse
                               ? fmt::format("abs({})", position)
                               : fmt::format("max({}, 0.0f)", position);
        return fmt::format("(as_type<float>({}u) * LightingLut(light_fog, u, {}u, {}, {}))",
                           lut.scale_bits, static_cast<u32>(sampler), coord,
                           lut.abs_input ? "true" : "false");
    };
    const auto optional_lut = [&](const LutConfig& lut, LightingSampler sampler, u32 num,
                                  std::string_view otherwise) {
        if (lut.enable && LightingRegs::IsLightingSamplerSupported(lights.config, sampler)) {
            return lut_value(lut, sampler, num);
        }
        return std::string(otherwise);
    };

    const u32 count = std::min<u32>(lights.src_num, lights.lights.size());
    for (u32 i = 0; i < count; ++i) {
        const auto light = lights.lights[i];
        const u32 num = light.num;
        Emit("    {{\n"
             "        constant Light& light = u.light_src[{}];\n"
             "        const float3 displacement = float3(light.position){};\n"
             "        const float distance = length(displacement);\n"
             "        const float3 light_vector = normalize(displacement);\n"
             "        const float3 half_vector = view + light_vector;\n"
             "        const float cos_angle = {}(dot(light_vector, normal){});\n"
             "        const float highlight = {};\n",
             num, light.directional ? "" : " + in.view", light.two_sided_diffuse ? "abs" : "max",
             light.two_sided_diffuse ? "" : ", 0.0f",
             lights.clamp_highlights ? "sign(cos_angle)" : "1.0f");
        if (light.geometric_factor_0 || light.geometric_factor_1) {
            Emit("        const float norm_sq = dot(half_vector, half_vector);\n"
                 "        const float geometric = norm_sq == 0.0f ? 0.0f : min(cos_angle / "
                 "norm_sq, 1.0f);\n");
        }

        const auto spot =
            light.spot_atten_enable && LightingRegs::IsLightingSamplerSupported(
                                           lights.config, LightingSampler::SpotlightAttenuation)
                ? lut_value(lights.lut_sp, LightingRegs::SpotlightAttenuationSampler(num), num)
                : "1.0f";
        const auto distance =
            light.dist_atten_enable
                ? fmt::format("LightingLut(light_fog, u, {}u, "
                              "saturate(fma(light.dist_atten_scale, distance, "
                              "light.dist_atten_bias)), true)",
                              static_cast<u32>(LightingRegs::DistanceAttenuationSampler(num)))
                : "1.0f";
        const auto d0 = optional_lut(lights.lut_d0, LightingSampler::Distribution0, num, "1.0f");
        const auto d1 = optional_lut(lights.lut_d1, LightingSampler::Distribution1, num, "1.0f");
        Emit("        const float reflect_r = {};\n"
             "        const float reflect_g = {};\n"
             "        const float reflect_b = {};\n"
             "        const float attenuation = ({}) * ({});\n"
             "        const float3 specular_0 = ({}) * float3(light.specular_0){};\n"
             "        const float3 specular_1 = ({}) * float3(reflect_r, reflect_g, reflect_b) * "
             "float3(light.specular_1){};\n",
             optional_lut(lights.lut_rr, LightingSampler::ReflectRed, num, "1.0f"),
             optional_lut(lights.lut_rg, LightingSampler::ReflectGreen, num, "reflect_r"),
             optional_lut(lights.lut_rb, LightingSampler::ReflectBlue, num, "reflect_r"), distance,
             spot, d0, light.geometric_factor_0 ? " * geometric" : "", d1,
             light.geometric_factor_1 ? " * geometric" : "");

        if (i + 1 == count && lights.lut_fr.enable &&
            LightingRegs::IsLightingSamplerSupported(lights.config, LightingSampler::Fresnel)) {
            if (lights.enable_primary_alpha || lights.enable_secondary_alpha) {
                Emit("        const float fresnel = {};\n",
                     lut_value(lights.lut_fr, LightingSampler::Fresnel, num));
            }
            if (lights.enable_primary_alpha) {
                Emit("        diffuse_sum.a = fresnel;\n");
            }
            if (lights.enable_secondary_alpha) {
                Emit("        specular_sum.a = fresnel;\n");
            }
        }

        Emit("        diffuse_sum.rgb += (float3(light.diffuse) * cos_angle{} + "
             "float3(light.ambient)) * attenuation;\n"
             "        specular_sum.rgb += (specular_0 + specular_1) * highlight * attenuation{};\n"
             "    }}\n",
             lights.shadow_primary && light.shadow_enable ? " * shadow.rgb" : "",
             lights.shadow_secondary && light.shadow_enable ? " * shadow.rgb" : "");
    }
    if (lights.shadow_alpha && lights.enable_primary_alpha) {
        Emit("    diffuse_sum.a *= shadow.a;\n");
    }
    if (lights.shadow_alpha && lights.enable_secondary_alpha) {
        Emit("    specular_sum.a *= shadow.a;\n");
    }
    Emit("    diffuse_sum.rgb += float3(u.lighting_global_ambient);\n"
         "    primary_fragment = saturate(diffuse_sum);\n"
         "    secondary_fragment = saturate(specular_sum);\n");
}

std::string FragmentModule::TevSource(Source source, u32 index) {
    using enum Source;
    switch (source) {
    case PrimaryColor:
        return "primary_color";
    case PrimaryFragmentColor:
        return "primary_fragment";
    case SecondaryFragmentColor:
        return "secondary_fragment";
    case Texture0:
    case Texture1:
    case Texture2:
    case Texture3:
        return fmt::format("samples[{}]", static_cast<u32>(source) - static_cast<u32>(Texture0));
    case PreviousBuffer:
        return "combiner_buffer";
    case Constant:
        return fmt::format("u.const_color[{}]", index);
    case Previous:
        return "combiner_output";
    }
    return "float4(0.0f)";
}

std::string FragmentModule::ColorOperand(Source source, TevStage::ColorModifier modifier,
                                         u32 index) {
    using enum TevStage::ColorModifier;
    const TevStage stage = config.texture.tev_stages[index];
    if (index == 0 && source == Source::Previous) {
        source = stage.color_source3;
    }
    std::string_view lanes;
    switch (modifier) {
    case SourceColor:
    case OneMinusSourceColor:
        lanes = "rgb";
        break;
    case SourceAlpha:
    case OneMinusSourceAlpha:
        lanes = "aaa";
        break;
    case SourceRed:
    case OneMinusSourceRed:
        lanes = "rrr";
        break;
    case SourceGreen:
    case OneMinusSourceGreen:
        lanes = "ggg";
        break;
    case SourceBlue:
    case OneMinusSourceBlue:
        lanes = "bbb";
        break;
    default:
        return "float3(0.0f)";
    }
    return fmt::format("({}({}).{})", static_cast<u32>(modifier) & 1 ? "1.0f - " : "",
                       TevSource(source, index), lanes);
}

std::string FragmentModule::AlphaOperand(Source source, TevStage::AlphaModifier modifier,
                                         u32 index) {
    using enum TevStage::AlphaModifier;
    const TevStage stage = config.texture.tev_stages[index];
    if (index == 0 && source == Source::Previous) {
        source = stage.alpha_source3;
    }
    char lane;
    switch (modifier) {
    case SourceAlpha:
    case OneMinusSourceAlpha:
        lane = 'a';
        break;
    case SourceRed:
    case OneMinusSourceRed:
        lane = 'r';
        break;
    case SourceGreen:
    case OneMinusSourceGreen:
        lane = 'g';
        break;
    case SourceBlue:
    case OneMinusSourceBlue:
        lane = 'b';
        break;
    default:
        return "0.0f";
    }
    return fmt::format("({}({}).{})", static_cast<u32>(modifier) & 1 ? "1.0f - " : "",
                       TevSource(source, index), lane);
}

std::string Combine(Operation operation, bool rgb) {
    using enum Operation;
    switch (operation) {
    case Replace:
        return "a";
    case Modulate:
        return "a * b";
    case Add:
        return "a + b";
    case AddSigned:
        return "a + b - 0.5f";
    case Lerp:
        return "mix(b, a, c)";
    case Subtract:
        return "a - b";
    case MultiplyThenAdd:
        return "fma(a, b, c)";
    case AddThenMultiply:
        return rgb ? "min(a + b, float3(1.0f)) * c" : "min(a + b, 1.0f) * c";
    case Dot3_RGB:
    case Dot3_RGBA:
        return rgb ? "float3(4.0f * dot(a - 0.5f, b - 0.5f))" : "0.0f";
    }
    return rgb ? "float3(0.0f)" : "0.0f";
}

void FragmentModule::WriteTev(u32 index) {
    const TevStage stage = config.texture.tev_stages[index];
    if (!IsPassThrough(stage)) {
        Emit("    {{\n        float3 rgb;\n        float alpha;\n        {{\n");
        const std::array colors{stage.color_source1.Value(), stage.color_source2.Value(),
                                stage.color_source3.Value()};
        const std::array color_mods{stage.color_modifier1.Value(), stage.color_modifier2.Value(),
                                    stage.color_modifier3.Value()};
        for (u32 i = 0; i < OperandCount(stage.color_op); ++i) {
            Emit("            const float3 {} = {};\n", "abc"[i],
                 ColorOperand(colors[i], color_mods[i], index));
        }
        Emit("            rgb = ByteRound(saturate({}));\n        }}\n",
             Combine(stage.color_op, true));
        if (stage.color_op == Operation::Dot3_RGBA) {
            Emit("        alpha = rgb.r;\n");
        } else {
            Emit("        {{\n");
            const std::array alphas{stage.alpha_source1.Value(), stage.alpha_source2.Value(),
                                    stage.alpha_source3.Value()};
            const std::array alpha_mods{stage.alpha_modifier1.Value(),
                                        stage.alpha_modifier2.Value(),
                                        stage.alpha_modifier3.Value()};
            for (u32 i = 0; i < OperandCount(stage.alpha_op); ++i) {
                Emit("            const float {} = {};\n", "abc"[i],
                     AlphaOperand(alphas[i], alpha_mods[i], index));
            }
            Emit("            alpha = ByteRound(saturate({}));\n        }}\n",
                 Combine(stage.alpha_op, false));
        }
        Emit("        combiner_output = saturate(float4(rgb * {}.0f, alpha * {}.0f));\n"
             "    }}\n",
             stage.GetColorMultiplier(), stage.GetAlphaMultiplier());
    }
    Emit("    combiner_buffer = next_combiner_buffer;\n");
    if (config.TevStageUpdatesCombinerBufferColor(index)) {
        Emit("    next_combiner_buffer.rgb = combiner_output.rgb;\n");
    }
    if (config.TevStageUpdatesCombinerBufferAlpha(index)) {
        Emit("    next_combiner_buffer.a = combiner_output.a;\n");
    }
}

void FragmentModule::ReturnColor(std::string_view color) {
    if (result.requirements.shadow_storage != ShadowStorage::None) {
        Emit("    return;\n");
    } else if (HasColor(profile)) {
        Emit("    return FragmentOutput{{{}, saturate(depth)}};\n", color);
    } else {
        Emit("    return FragmentOutput{{saturate(depth)}};\n");
    }
}

void FragmentModule::WriteShadowOutput() {
    Emit("    const uint depth24 = uint(saturate(depth) * 16777215.0f);\n"
         "    const uint opacity = uint(saturate(combiner_output.g) * 255.0f);\n"
         "    const uint2 xy = uint2(in.position.xy);\n");
    if (result.requirements.shadow_storage == ShadowStorage::AtomicTexture) {
        result.source += R"MSL(
    if (xy.x >= shadow_target.get_width() || xy.y >= shadow_target.get_height()) {
        return;
    }
    uint4 expected = shadow_target.read(xy);
    while (true) {
        const uint updated = UpdateShadow(expected.r, depth24, opacity, u);
        if (updated == expected.r ||
            shadow_target.atomic_compare_exchange_weak(xy, &expected, uint4(updated, 0u, 0u, 0u))) {
            break;
        }
    }
)MSL";
    } else {
        result.source += R"MSL(
    if (xy.x >= shadow_extent.x || xy.y >= shadow_extent.y || shadow_extent.z < shadow_extent.x) {
        return;
    }
    device atomic_uint* pixel = shadow_target + ulong(xy.y) * ulong(shadow_extent.z) + xy.x;
    uint expected = atomic_load_explicit(pixel, memory_order_relaxed);
    while (true) {
        const uint updated = UpdateShadow(expected, depth24, opacity, u);
        if (updated == expected || atomic_compare_exchange_weak_explicit(pixel, &expected, updated,
                                      memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
)MSL";
    }
}

void FragmentModule::WriteOutput() {
    const auto req = result.requirements;
    if (req.shadow_storage != ShadowStorage::None) {
        WriteShadowOutput();
        return;
    }

    Emit("    combiner_output = ByteRound(combiner_output);\n");
    if (req.feedback == FragmentFeedback::TextureCopy) {
        Emit("    const float4 dest_color = feedback_texture.read(uint2(in.position.xy){});\n",
             profile.samples > 1 ? ", sample_index" : "");
    }
    if (req.emulated_rgb_blend || req.emulated_alpha_blend) {
        Emit("    const float4 source_color = combiner_output;\n");
        const auto blend = [&](const BlendConfig& axis, std::string_view lane) {
            Emit("    combiner_output.{0} = {1}(source_color.{0} * ({2}).{0}, "
                 "dest_color.{0} * ({3}).{0});\n",
                 lane, axis.eq == BlendEquation::Min ? "min" : "max",
                 FactorExpression(axis.src_factor), FactorExpression(axis.dst_factor));
        };
        if (req.emulated_rgb_blend) {
            blend(config.framebuffer.requested_rgb_blend, "rgb");
        }
        if (req.emulated_alpha_blend) {
            blend(config.framebuffer.requested_alpha_blend, "a");
        }
    }
    const auto logic = config.framebuffer.requested_logic_op;
    if (req.writes_color && !config.framebuffer.alphablend_enable && logic != LogicOp::Copy) {
        const auto [r, g, b, a] = ColorMasks(profile.color_format);
        Emit("    const uint4 mask({}u, {}u, {}u, {}u);\n"
             "    const uint4 src = uint4(round(saturate(combiner_output) * float4(mask)));\n",
             r, g, b, a);
        if (LogicReadsDestination(logic)) {
            Emit("    const uint4 dst = uint4(round(saturate(dest_color) * float4(mask)));\n");
        }
        Emit("    const uint4 packed = ({}) & mask;\n"
             "    combiner_output = select(float4(1.0f), float4(packed) / float4(max(mask, "
             "uint4(1u))), "
             "mask != uint4(0u));\n",
             LogicExpression(logic));
    }
    ReturnColor("combiner_output");
}

FragmentShader FragmentModule::Generate() {
    result.source = "#include <metal_stdlib>\n";
    if (result.requirements.fragment_barycentrics) {
        result.source += "#include <metal_vertex_value>\n";
    }
    result.source += Varyings;
    result.source += Uniforms;
    if (result.requirements.fragment_barycentrics) {
        result.source += BarycentricInput;
    }
    if (config.lighting.enable) {
        result.source += LightingHelpers;
    }
    if (used[0] && (config.texture.texture0_type == TextureType::Shadow2D ||
                    config.texture.texture0_type == TextureType::ShadowCube)) {
        result.source += ShadowSampling;
    }
    if (used[3]) {
        result.source += ProceduralHelpers;
    }
    if (result.requirements.shadow_storage != ShadowStorage::None) {
        result.source += ShadowUpdate;
    }

    DeclareEntry();
    if (config.framebuffer.alpha_test_func == CompareFunc::Never) {
        Emit("    discard_fragment();\n    const float depth = 0.0f;\n");
        ReturnColor("float4(0.0f)");
    } else {
        const auto scissor = config.framebuffer.scissor_test_mode.Value();
        if (scissor != RasterizerRegs::ScissorMode::Disabled) {
            Emit("    const bool inside = in.position.x >= float(u.scissor_x1) && "
                 "in.position.y >= float(u.scissor_y1) && in.position.x < float(u.scissor_x2) && "
                 "in.position.y < float(u.scissor_y2);\n"
                 "    if ({}inside) discard_fragment();\n",
                 scissor == RasterizerRegs::ScissorMode::Include ? "!" : "");
        }
        Emit("    float depth = fma(-in.position.z, u.depth_scale, u.depth_offset);\n");
        if (config.framebuffer.depthmap_enable == RasterizerRegs::DepthBuffering::WBuffering) {
            Emit("    depth /= in.position.w;\n");
        }
        WriteSamples();
        WriteLighting();
        Emit("    const float4 primary_color = ByteRound(in.primary_color);\n"
             "    float4 combiner_buffer(0.0f);\n"
             "    float4 next_combiner_buffer = u.tev_combiner_buffer_color;\n"
             "    float4 combiner_output(0.0f);\n");
        for (u32 i = 0; i < config.texture.tev_stages.size(); ++i) {
            WriteTev(i);
        }
        if (const auto comparison = AlphaReject(config.framebuffer.alpha_test_func);
            !comparison.empty()) {
            Emit(
                "    if (int(combiner_output.a * 255.0f) {} u.alphatest_ref) discard_fragment();\n",
                comparison);
        }

        switch (config.texture.fog_mode) {
        case FogMode::Fog:
            Emit("    const float fog_index = {} * 128.0f;\n"
                 "    const float fog_i = clamp(floor(fog_index), 0.0f, 127.0f);\n"
                 "    const float2 fog_entry = light_fog.read(uint(int(fog_i) + "
                 "u.fog_lut_offset)).rg;\n"
                 "    const float fog = saturate(fma(fog_entry.g, fog_index - fog_i, "
                 "fog_entry.r));\n"
                 "    combiner_output.rgb = mix(float3(u.fog_color), combiner_output.rgb, fog);\n",
                 config.texture.fog_flip ? "(1.0f - depth)" : "depth");
            WriteOutput();
            break;
        case FogMode::Gas:
            // The source renderer publishes a transparent fragment for gas mode.
            ReturnColor("float4(0.0f)");
            break;
        default:
            WriteOutput();
            break;
        }
    }
    Emit("}}\n}} // namespace MetalirisFragment\n");
    return std::move(result);
}

} // Anonymous namespace

std::string_view FragmentVaryingsDeclaration() noexcept {
    return Varyings;
}

FragmentRequirements DescribeFragment(const FSConfig& config,
                                      const FragmentProfile& profile) noexcept {
    FragmentRequirements result;
    result.fragment_barycentrics = config.lighting.enable && profile.fragment_barycentrics;
    const auto& framebuffer = config.framebuffer;
    if (framebuffer.shadow_rendering) {
        result.shadow_storage =
            profile.texture_atomics ? ShadowStorage::AtomicTexture : ShadowStorage::AtomicBuffer;
        return result;
    }

    result.writes_depth = true;
    if (!HasColor(profile)) {
        return result;
    }
    if (config.texture.fog_mode == FogMode::Gas) {
        result.writes_color = true;
        return result;
    }

    result.writes_color =
        framebuffer.alphablend_enable || framebuffer.requested_logic_op != LogicOp::NoOp;
    if (framebuffer.alphablend_enable) {
        result.emulated_rgb_blend = RequiresMinMaxBlendShader(framebuffer.requested_rgb_blend);
        result.emulated_alpha_blend = RequiresMinMaxBlendShader(framebuffer.requested_alpha_blend);
    }
    if (result.emulated_rgb_blend || result.emulated_alpha_blend ||
        (!framebuffer.alphablend_enable && LogicReadsDestination(framebuffer.requested_logic_op))) {
        result.feedback = profile.raster_order_groups ? FragmentFeedback::RasterOrderGroup
                                                      : FragmentFeedback::TextureCopy;
    }
    return result;
}

FragmentShader GenerateFragmentShader(const FSConfig& config, const UserConfig& user,
                                      const FragmentProfile& profile) {
    if (profile.samples > 16 || !std::has_single_bit(profile.samples)) {
        throw std::invalid_argument("Invalid PICA fragment sample count");
    }
    if (HasColor(profile) && !config.framebuffer.shadow_rendering) {
        static_cast<void>(ColorMasks(profile.color_format));
    }
    return FragmentModule(config, user, profile).Generate();
}

} // namespace Pica::Shader::Generator::MSL
