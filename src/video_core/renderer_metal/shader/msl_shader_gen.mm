// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <fmt/format.h>
#include "video_core/renderer_metal/shader/msl_shader_gen.h"

namespace Pica::Shader::Generator::MSL {
namespace {

using Semantic = RasterizerRegs::VSOutputAttributes::Semantic;
using Topology = PipelineRegs::TriangleTopology;
using GSMode = PipelineRegs::GSMode;

constexpr std::string_view RasterTypes = R"MSL(
#ifndef METALIRIS_RASTER_STAGE_TYPES
#define METALIRIS_RASTER_STAGE_TYPES
#include <metal_stdlib>

namespace MetalirisVertex {
using namespace metal;

struct RasterUniforms {
    uint enable_clip1;
    uint flip_viewport;
    float4 clip_coef;
)MSL";

#ifdef ENABLE_FREECAM
constexpr std::string_view FreeCamUniforms = R"MSL(
    float4x4 freecam_matrix;
    uint freecam_skip_2d;
};

static_assert(sizeof(RasterUniforms) == 112, "FreeCam raster uniform size");

inline float4 ApplyFreeCam(float4 value, constant RasterUniforms& uniforms) {
    if (uniforms.freecam_skip_2d && abs(value.w - 1.0f) < 0.0001f) {
        return value;
    }
    return uniforms.freecam_matrix * value;
}
)MSL";
#else
constexpr std::string_view FreeCamUniforms = R"MSL(
};

static_assert(sizeof(RasterUniforms) == 32, "Raster uniform size");

inline float4 ApplyFreeCam(float4 value, constant RasterUniforms&) {
    return value;
}
)MSL";
#endif

constexpr std::string_view RasterUtilities = R"MSL(
static_assert(__builtin_offsetof(RasterUniforms, clip_coef) == 16, "Raster clip coefficient");

struct ExpandedVertex {
    float4 position;
    float4 quaternion;
    float4 color;
    float4 texcoords01;
    float4 texcoord2;
    float4 view_and_texcoord0_w;
};

static_assert(sizeof(ExpandedVertex) == 96, "Expanded vertex stride");
static_assert(__builtin_offsetof(ExpandedVertex, view_and_texcoord0_w) == 80,
              "View and projective coordinate");

struct RasterVertex {
    float4 position [[position, invariant]];
    float4 primary_color [[user(locn1)]];
    float2 texcoord0 [[user(locn2)]];
    float2 texcoord1 [[user(locn3)]];
    float2 texcoord2 [[user(locn4)]];
    float texcoord0_w [[user(locn5)]];
    float4 normquat [[user(locn6)]];
    float3 view [[user(locn7)]];
    float clip_distances [[clip_distance]] [2];
};

struct RegisterBank {
    float4 attributes[16];
};

inline RasterVertex MakeRasterVertex(ExpandedVertex input, constant RasterUniforms& uniforms) {
    float4 pica_position = input.position;
    const float guest_depth = pica_position.z / pica_position.w;
    if (guest_depth > 0.0f && guest_depth < 0.000001f) {
        pica_position.z = 0.0f;
    }
    if (guest_depth < -1.0f && guest_depth > -1.00001f) {
        pica_position.z = -pica_position.w;
    }
    if (uniforms.flip_viewport) {
        pica_position.y = -pica_position.y;
    }

    RasterVertex result{};
    result.position = ApplyFreeCam(float4(pica_position.xy, -pica_position.z, pica_position.w),
                                   uniforms);
    result.primary_color = input.color;
    result.texcoord0 = input.texcoords01.xy;
    result.texcoord1 = input.texcoords01.zw;
    result.texcoord2 = input.texcoord2.xy;
    result.texcoord0_w = input.view_and_texcoord0_w.w;
    result.normquat = input.quaternion;
    result.view = input.view_and_texcoord0_w.xyz;
    result.clip_distances[0] = -pica_position.z;
    result.clip_distances[1] = uniforms.enable_clip1 ? dot(uniforms.clip_coef, pica_position) : 0.0f;
    return result;
}

inline void CorrectQuaternions(thread const ExpandedVertex& first,
                               thread ExpandedVertex& second, thread ExpandedVertex& third) {
    if (dot(first.quaternion, second.quaternion) < 0.0f) {
        second.quaternion = -second.quaternion;
    }
    if (dot(first.quaternion, third.quaternion) < 0.0f) {
        third.quaternion = -third.quaternion;
    }
}

} // namespace MetalirisVertex
#endif
)MSL";

constexpr std::string_view TransformOutputType = R"MSL(
#ifndef METALIRIS_TRANSFORM_OUTPUT_TYPES
#define METALIRIS_TRANSFORM_OUTPUT_TYPES

namespace MetalirisVertex {
using MetalirisPica::RunStatus;

struct TransformedVertex {
    RegisterBank registers;
    RunStatus status;
    uint reserved[3];
};

static_assert(sizeof(TransformedVertex) == 0x110, "Compact vertex program output");
static_assert(__builtin_offsetof(TransformedVertex, status) == 0x100, "VS execution status");

enum class GeometryStatus : uint {
    Complete,
    InvalidInput,
    ProgramFailure,
    OutputOverflow,
};

struct GeometryResult {
    GeometryStatus status;
    RunStatus program_status;
    uint failed_invocation;
    uint reserved;
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
};

static_assert(sizeof(GeometryResult) == 32, "Geometry result size");
static_assert(__builtin_offsetof(GeometryResult, vertex_count) == 16, "Indirect draw offset");

inline GeometryResult Failed(GeometryStatus status, uint invocation,
                             RunStatus program_status = RunStatus::Complete) {
    return {status, program_status, invocation, 0u, 0u, 0u, 0u, 0u};
}

inline GeometryResult Completed(uint vertices) {
    return {GeometryStatus::Complete, RunStatus::Complete, 0u, 0u, vertices, 1u, 0u, 0u};
}

} // namespace MetalirisVertex
#endif
)MSL";

struct InputAttribute {
    u32 location;
    std::string_view type;
    bool zero_w;
};

std::string RasterPreamble() {
    std::string result{RasterTypes};
    result += FreeCamUniforms;
    result += RasterUtilities;
    return result;
}

void AppendProgramTypes(std::string& source) {
    source += "#ifndef METALIRIS_PICA_PROGRAM_TYPES\n#define METALIRIS_PICA_PROGRAM_TYPES\n";
    source += DecompilerPreamble();
    source += "#endif\n";
    source += TransformOutputType;
}

void AppendSemanticConversion(std::string& source, const PicaGSConfigState& state,
                              u32 output_count) {
    if (state.vs_output_total > state.vs_output_attributes.size() || output_count > 16) {
        throw std::invalid_argument("Invalid PICA shader output semantic range");
    }
    const auto map = state.GetSemanticMaps();
    const auto semantic = [&](Semantic value) {
        const auto [attribute, component] = map[value];
        if (attribute >= output_count) {
            return std::string("1.0f");
        }
        return fmt::format("input.attributes[{}].{}", attribute, "xyzw"[component]);
    };
    using enum Semantic;
    source += fmt::format(
        R"MSL(
inline ExpandedVertex DecodeVertex(thread const RegisterBank& input) {{
    ExpandedVertex result{{}};
    result.position = float4({}, {}, {}, {});
    result.quaternion = float4({}, {}, {}, {});
    result.color = min(abs(float4({}, {}, {}, {})), float4(1.0f));
    result.texcoords01 = float4({}, {}, {}, {});
    result.texcoord2 = float4({}, {}, 0.0f, 0.0f);
    result.view_and_texcoord0_w = float4({}, {}, {}, {});
    return result;
}}
)MSL",
        semantic(POSITION_X), semantic(POSITION_Y), semantic(POSITION_Z), semantic(POSITION_W),
        semantic(QUATERNION_X), semantic(QUATERNION_Y), semantic(QUATERNION_Z),
        semantic(QUATERNION_W), semantic(COLOR_R), semantic(COLOR_G), semantic(COLOR_B),
        semantic(COLOR_A), semantic(TEXCOORD0_U), semantic(TEXCOORD0_V), semantic(TEXCOORD1_U),
        semantic(TEXCOORD1_V), semantic(TEXCOORD2_U), semantic(TEXCOORD2_V), semantic(VIEW_X),
        semantic(VIEW_Y), semantic(VIEW_Z), semantic(TEXCOORD0_W));
}

std::array<std::optional<InputAttribute>, 16> DescribeAttributes(const PicaVSConfigState& state,
                                                                 const VertexProfile& profile) {
    if (state.used_input_vertex_attributes > state.input_vertex_attributes.size() ||
        state.num_outputs > 16) {
        throw std::invalid_argument("Invalid native PICA vertex layout");
    }
    std::array<std::optional<InputAttribute>, 16> attributes{};
    for (u32 i = 0; i < state.used_input_vertex_attributes; ++i) {
        const u32 location = state.input_vertex_attributes[i].location;
        if (location >= attributes.size() || attributes[location]) {
            throw std::invalid_argument("Invalid or duplicate PICA vertex register location");
        }
        const auto flags = profile.load_flags[location];
        const bool sint = True(flags & AttribLoadFlags::Sint);
        const bool uint = True(flags & AttribLoadFlags::Uint);
        if ((sint && uint) || ((sint || uint) && True(flags & AttribLoadFlags::Float))) {
            throw std::invalid_argument("Multiple native types for a PICA vertex register");
        }
        const std::string_view type = sint ? "int4" : (uint ? "uint4" : "float4");
        attributes[location] = InputAttribute{location, type, True(flags & AttribLoadFlags::ZeroW)};
    }
    return attributes;
}

void AppendOutputMapping(std::string& source, const PicaVSConfigState& state) {
    std::array<bool, 16> mapped{};
    source += R"MSL(
inline RegisterBank PackOutput(thread const RegisterState& state) {
    RegisterBank result{};
    const uint bank = state.condition.z & 1u;
)MSL";
    for (u32 reg = 0; reg < state.output_map.size(); ++reg) {
        const u32 index = state.output_map[reg];
        if (index >= state.num_outputs) {
            continue;
        }
        if (mapped[index]) {
            throw std::invalid_argument("Duplicate compact PICA output register");
        }
        mapped[index] = true;
        source += fmt::format("    result.attributes[{}] = state.output[bank][{}];\n", index, reg);
    }
    if (static_cast<u32>(std::ranges::count(mapped, true)) != state.num_outputs) {
        throw std::invalid_argument("Incomplete compact PICA output register mapping");
    }
    source += "    return result;\n}\n";
}

void AppendVertexStages(std::string& source, const PicaVSConfigState& state,
                        const VertexProfile& profile) {
    const auto attributes = DescribeAttributes(state, profile);
    const bool vertex_inputs = state.used_input_vertex_attributes != 0;
    if (vertex_inputs) {
        source += "struct VertexAttributes {\n";
        for (const auto& attribute : attributes) {
            if (attribute) {
                source += fmt::format("    {} value{} [[attribute({})]];\n", attribute->type,
                                      attribute->location, attribute->location);
            }
        }
        source += "};\n";
    }

    source += fmt::format("vertex RasterVertex {}({}constant Uniforms& pica [[buffer({})]], "
                          "constant RasterUniforms& uniforms [[buffer({})]]) {{\n"
                          "    RegisterState registers{{}};\n",
                          VertexEntryPoints::Vertex,
                          vertex_inputs ? "VertexAttributes inputs [[stage_in]], " : "",
                          VertexBindings::PicaUniforms, VertexBindings::RasterUniforms);
    for (const auto& attribute : attributes) {
        if (!attribute) {
            continue;
        }
        source += fmt::format("    registers.input[{0}] = float4(inputs.value{0});\n",
                              attribute->location);
        if (attribute->zero_w) {
            source += fmt::format("    registers.input[{}].w = 0.0f;\n", attribute->location);
        }
    }
    source += R"MSL(
    for (uint i = 0u; i < 16u; ++i) {
        registers.output[0][i].w = 1.0f;
    }
    if (ExecuteVertex(registers, pica) != RunStatus::Complete) {
        RasterVertex discarded{};
        discarded.position = float4(0, 0, -1, 1);
        discarded.clip_distances[0] = -1.0f;
        return discarded;
    }
    const RegisterBank compact = PackOutput(registers);
    return MakeRasterVertex(DecodeVertex(compact), uniforms);
}

)MSL";
    source += fmt::format(R"MSL(
kernel void {}(device const RegisterBank* input [[buffer({})]],
               device TransformedVertex* output [[buffer({})]],
               constant uint4& parameters [[buffer({})]],
               constant Uniforms& pica [[buffer({})]],
               uint index [[thread_position_in_grid]]) {{
    if (index >= parameters.x) {{
        return;
    }}
    RegisterState registers{{}};
    for (uint i = 0u; i < 16u; ++i) {{
        registers.input[i] = input[index].attributes[i];
        registers.output[0][i].w = 1.0f;
    }}
    TransformedVertex result{{}};
    result.status = ExecuteVertex(registers, pica);
    if (result.status == RunStatus::Complete) {{
        result.registers = PackOutput(registers);
    }}
    output[index] = result;
}}
)MSL",
                          VertexEntryPoints::TransformVertices, VertexBindings::InputVertices,
                          VertexBindings::OutputVertices, VertexBindings::Parameters,
                          VertexBindings::PicaUniforms);
}

std::string_view TriangleIndices(Topology topology) {
    using enum Topology;
    switch (topology) {
    case List:
    case Shader:
        return "uint3(primitive * 3u) + uint3(0, 1, 2)";
    case Strip:
        return "uint3(primitive + (primitive & 1u), primitive + ((primitive & 1u) ^ 1u), primitive "
               "+ 2u)";
    case Fan:
        return "uint3(0u, primitive + 1u, primitive + 2u)";
    }
    throw std::invalid_argument("Invalid PICA triangle topology");
}

std::string_view PrimitiveCountExpression(Topology topology) {
    using enum Topology;
    switch (topology) {
    case List:
    case Shader:
        return "vertices / 3u";
    case Strip:
    case Fan:
        return "vertices < 3u ? 0u : vertices - 2u";
    }
    throw std::invalid_argument("Invalid PICA triangle topology");
}

} // Anonymous namespace

namespace {

void AppendFixedExpansion(std::string& source, const FixedGeometryConfig& config) {
    source += fmt::format(
        R"MSL(
inline uint TotalPrimitives(uint vertices) {{
    return {};
}}

inline uint3 PrimitiveIndices(uint primitive) {{
    return {};
}}

inline void LoadPrimitive(device const TransformedVertex* input, uint3 indices,
                          thread ExpandedVertex (&vertices)[3]) {{
    for (uint i = 0u; i < 3u; ++i) {{
        const RegisterBank register_values = input[indices[i]].registers;
        vertices[i] = DecodeVertex(register_values);
    }}
    {}
}}
)MSL",
        PrimitiveCountExpression(config.topology), TriangleIndices(config.topology),
        config.correct_quaternions ? "CorrectQuaternions(vertices[0], vertices[1], vertices[2]);"
                                   : "");
    switch (config.path) {
    case GeometryPath::Mesh:
        source += fmt::format(R"MSL(
struct PrimitivePayload {{
    ExpandedVertex vertices[3];
}};

[[object, max_total_threads_per_threadgroup(1)]]
void {}(object_data PrimitivePayload& payload [[payload]], mesh_grid_properties grid,
        device const TransformedVertex* input [[buffer({})]],
        constant uint4& parameters [[buffer({})]],
        uint3 group [[threadgroup_position_in_grid]]) {{
    const uint total = TotalPrimitives(parameters.x);
    if (parameters.y > total || group.x >= min(parameters.z, total - parameters.y)) {{
        return;
    }}
    const uint3 indices = PrimitiveIndices(parameters.y + group.x);
    if (input[indices.x].status != RunStatus::Complete ||
        input[indices.y].status != RunStatus::Complete ||
        input[indices.z].status != RunStatus::Complete) {{
        return;
    }}
    ExpandedVertex vertices[3];
    LoadPrimitive(input, indices, vertices);
    for (uint i = 0u; i < 3u; ++i) {{
        payload.vertices[i] = vertices[i];
    }}
    grid.set_threadgroups_per_grid(uint3(1, 1, 1));
}}

using PrimitiveMesh = mesh<RasterVertex, void, 3, 1, topology::triangle>;

[[mesh, max_total_threads_per_threadgroup(1)]]
void {}(PrimitiveMesh output, const object_data PrimitivePayload& payload [[payload]],
        constant RasterUniforms& uniforms [[buffer({})]]) {{
    for (uint i = 0u; i < 3u; ++i) {{
        output.set_vertex(i, MakeRasterVertex(payload.vertices[i], uniforms));
        output.set_index(i, uchar(i));
    }}
    output.set_primitive_count(1u);
}}
)MSL",
                              VertexEntryPoints::Object, VertexBindings::InputVertices,
                              VertexBindings::Parameters, VertexEntryPoints::Mesh,
                              VertexBindings::RasterUniforms);
        break;
    case GeometryPath::Compute:
        source += fmt::format(R"MSL(
kernel void {}(device const TransformedVertex* input [[buffer({})]],
               device ExpandedVertex* output [[buffer({})]],
               constant uint4& parameters [[buffer({})]],
               device GeometryResult& result [[buffer({})]],
               uint index [[thread_position_in_grid]]) {{
    if (index != 0u) {{
        return;
    }}
    const uint total = TotalPrimitives(parameters.x);
    if (parameters.y > total || parameters.z > total - parameters.y) {{
        result = Failed(GeometryStatus::InvalidInput, 0u);
        return;
    }}
    if (parameters.z > parameters.w / 3u) {{
        result = Failed(GeometryStatus::OutputOverflow, parameters.w / 3u);
        return;
    }}
    for (uint i = 0u; i < parameters.z; ++i) {{
        const uint3 indices = PrimitiveIndices(parameters.y + i);
        for (uint lane = 0u; lane < 3u; ++lane) {{
            const RunStatus status = input[indices[lane]].status;
            if (status != RunStatus::Complete) {{
                result = Failed(GeometryStatus::ProgramFailure, i, status);
                return;
            }}
        }}
        ExpandedVertex vertices[3];
        LoadPrimitive(input, indices, vertices);
        for (uint lane = 0u; lane < 3u; ++lane) {{
            output[i * 3u + lane] = vertices[lane];
        }}
    }}
    result = Completed(parameters.z * 3u);
}}
)MSL",
                              VertexEntryPoints::Expand, VertexBindings::InputVertices,
                              VertexBindings::OutputVertices, VertexBindings::Parameters,
                              VertexBindings::GeometryResult);
        break;
    case GeometryPath::Vertex:
    default:
        throw std::invalid_argument("Native fixed geometry requires a mesh or compute path");
    }
}

constexpr std::string_view GeometryContinuationType = R"MSL(
struct GeometryContinuation {
    RegisterState registers;
    Uniforms uniforms;
    RegisterBank emitter[3];
    ExpandedVertex primitive_vertices[2];
    uint primitive_count;
    uint primitive_cursor;
    uint primitive_ready;
    uint primitive_winding;
};

static_assert(sizeof(GeometryContinuation) == 0xE50, "Geometry continuation stride");
static_assert(__builtin_offsetof(GeometryContinuation, uniforms) == 0x430, "Geometry uniforms");
static_assert(__builtin_offsetof(GeometryContinuation, emitter) == 0xA80, "Geometry emit slots");
static_assert(__builtin_offsetof(GeometryContinuation, primitive_vertices) == 0xD80,
              "Geometry primitive assembler");
)MSL";

void AppendProgrammedEmitter(std::string& source, const ProgrammableGeometryConfig& config) {
    source += R"MSL(
struct GeometryEmitter {
    thread GeometryContinuation& state;
    device ExpandedVertex* output;
    uint capacity;
    uint count;

    bool WritePrimitive(ExpandedVertex first, ExpandedVertex second, ExpandedVertex third) {
        if (capacity - count < 3u) {
            return false;
        }
)MSL";
    if (config.correct_quaternions) {
        source += "        CorrectQuaternions(first, second, third);\n";
    }
    source += R"MSL(
        output[count++] = first;
        output[count++] = second;
        output[count++] = third;
        return true;
    }

    bool Submit(ExpandedVertex value) {
)MSL";
    switch (config.topology) {
    case Topology::List:
    case Topology::Shader:
        source += R"MSL(
        if (state.primitive_count < 2u) {
            state.primitive_vertices[state.primitive_count++] = value;
            return true;
        }
        state.primitive_count = 0u;
)MSL";
        if (config.topology == Topology::Shader) {
            source += R"MSL(
        if (state.primitive_winding != 0u) {
            state.primitive_winding = 0u;
            return WritePrimitive(state.primitive_vertices[1], state.primitive_vertices[0], value);
        }
)MSL";
        }
        source += R"MSL(
        return WritePrimitive(state.primitive_vertices[0], state.primitive_vertices[1], value);
)MSL";
        break;
    case Topology::Strip:
    case Topology::Fan:
        source += R"MSL(
        if (state.primitive_ready != 0u &&
            !WritePrimitive(state.primitive_vertices[0], state.primitive_vertices[1], value)) {
            return false;
        }
        state.primitive_vertices[state.primitive_cursor] = value;
        if (state.primitive_cursor == 1u) {
            state.primitive_ready = 1u;
        }
)MSL";
        if (config.topology == Topology::Strip) {
            source += "        state.primitive_cursor ^= 1u;\n";
        } else {
            source += "        state.primitive_cursor = 1u;\n";
        }
        source += "        return true;\n";
        break;
    default:
        throw std::invalid_argument("Invalid PICA geometry output topology");
    }
    source += R"MSL(
    }

    bool Emit(thread const float4* values, uint vertex_slot, bool primitive, bool reverse_winding) {
        thread RegisterBank& slot = state.emitter[vertex_slot];
)MSL";
    u32 packed_index{};
    for (u32 reg = 0; reg < 16; ++reg) {
        if (config.output_mask & (1u << reg)) {
            source +=
                fmt::format("        slot.attributes[{}] = values[{}];\n", packed_index++, reg);
        }
    }
    source += R"MSL(
        if (!primitive) {
            return true;
        }
        if (reverse_winding) {
            state.primitive_winding = 1u;
        }
        for (uint i = 0u; i < 3u; ++i) {
            if (!Submit(DecodeVertex(state.emitter[i]))) {
                return false;
            }
        }
        return true;
    }
};
)MSL";
}

void AppendGeometryInput(std::string& source, const ProgrammableGeometryConfig& config) {
    source += R"MSL(
inline bool ValidSize(uint vertices) {
)MSL";
    switch (config.mode) {
    case GSMode::Point:
        source += fmt::format("    return vertices == {}u;\n",
                              config.input_attribute_count / config.vertex_output_count);
        break;
    case GSMode::FixedPrimitive:
        source += fmt::format("    return vertices == {}u;\n", config.fixed_vertex_count);
        break;
    case GSMode::VariablePrimitive:
        source += fmt::format("    const uint complete = min(vertices, {}u);\n"
                              "    return vertices != 0u && vertices < 96u && "
                              "complete * {}u + (vertices - complete) < 96u;\n",
                              config.main_vertex_count, config.vertex_output_count);
        break;
    default:
        throw std::invalid_argument("Invalid PICA geometry input mode");
    }
    source += R"MSL(
}

inline void LoadInput(thread GeometryContinuation& state, device const TransformedVertex* input,
                      uint first, uint count) {
)MSL";
    if (config.mode == GSMode::Point) {
        for (u32 i = 0; i < config.input_attribute_count; ++i) {
            source += fmt::format(
                "    state.registers.input[{}] = input[first + {}u].registers.attributes[{}];\n",
                config.input_registers[i], i / config.vertex_output_count,
                i % config.vertex_output_count);
        }
    } else {
        if (config.mode == GSMode::VariablePrimitive) {
            source += fmt::format("    state.uniforms.f[0] = float4(float(count));\n"
                                  "    uint uniform_index = 1u;\n"
                                  "    const uint complete = min(count, {}u);\n",
                                  config.main_vertex_count);
        } else {
            source += fmt::format("    uint uniform_index = {}u;\n"
                                  "    const uint complete = count;\n",
                                  config.uniform_start);
        }
        source += fmt::format(R"MSL(
    for (uint i = 0u; i < complete; ++i) {{
        for (uint j = 0u; j < {}u; ++j) {{
            state.uniforms.f[uniform_index++] = input[first + i].registers.attributes[j];
        }}
    }}
)MSL",
                              config.vertex_output_count);
        if (config.mode == GSMode::VariablePrimitive) {
            source += R"MSL(
    for (uint i = complete; i < count; ++i) {
        state.uniforms.f[uniform_index++] = input[first + i].registers.attributes[0];
    }
)MSL";
        }
    }
    source += "}\n";
}

void ValidateGeometryConfig(const ProgrammableGeometryConfig& config) {
    bool valid = false;
    switch (config.mode) {
    case GSMode::Point:
        valid =
            config.vertex_output_count != 0 &&
            config.ValidInvocationSize(config.input_attribute_count / config.vertex_output_count);
        if (valid) {
            const auto registers =
                std::span(config.input_registers).first(config.input_attribute_count);
            valid = std::ranges::all_of(registers, [](u8 reg) { return reg < 16; });
        }
        break;
    case GSMode::FixedPrimitive:
        valid = config.ValidInvocationSize(config.fixed_vertex_count);
        break;
    case GSMode::VariablePrimitive:
        valid = config.ValidInvocationSize(1);
        break;
    }
    if (!valid) {
        throw std::invalid_argument("Invalid native PICA programmable geometry configuration");
    }
}

void AppendProgrammedKernel(std::string& source) {
    source += fmt::format(R"MSL(
kernel void {}(device const TransformedVertex* input [[buffer({})]],
               device ExpandedVertex* output [[buffer({})]],
               constant uint4& parameters [[buffer({})]],
               device const uint4* invocations [[buffer({})]],
               device GeometryContinuation& continuation [[buffer({})]],
               device GeometryResult& result [[buffer({})]],
               uint index [[thread_position_in_grid]]) {{
    if (index != 0u) {{
        return;
    }}

    GeometryContinuation state = continuation;
    if (parameters.w != 0u) {{
        state.primitive_count = 0u;
        state.primitive_cursor = 0u;
        state.primitive_ready = 0u;
        state.primitive_winding = 0u;
    }}
    if (state.primitive_count > 2u || state.primitive_cursor > 1u) {{
        result = Failed(GeometryStatus::InvalidInput, 0u);
        return;
    }}
    GeometryEmitter emitter{{state, output, parameters.z, 0u}};
    for (uint i = 0u; i < parameters.y; ++i) {{
        const uint2 invocation = invocations[i].xy;
        if (invocation.x > parameters.x || invocation.y > parameters.x - invocation.x ||
            !ValidSize(invocation.y)) {{
            result = Failed(GeometryStatus::InvalidInput, i);
            return;
        }}
        for (uint offset = 0u; offset < invocation.y; ++offset) {{
            const RunStatus status = input[invocation.x + offset].status;
            if (status != RunStatus::Complete) {{
                result = Failed(GeometryStatus::ProgramFailure, i, status);
                return;
            }}
        }}
        LoadInput(state, input, invocation.x, invocation.y);
        const RunStatus status = ExecuteGeometry(state.registers, state.uniforms, emitter);
        if (status != RunStatus::Complete) {{
            const auto failure = status == RunStatus::EmitOverflow ? GeometryStatus::OutputOverflow
                                                                  : GeometryStatus::ProgramFailure;
            result = Failed(failure, i, status);
            return;
        }}
        state.uniforms.b |= 0x8000u;
    }}
    continuation = state;
    result = Completed(emitter.count);
}}
)MSL",
                          VertexEntryPoints::Expand, VertexBindings::InputVertices,
                          VertexBindings::OutputVertices, VertexBindings::Parameters,
                          VertexBindings::InvocationList, VertexBindings::GeometryState,
                          VertexBindings::GeometryResult);
}

} // Anonymous namespace

std::string GenerateTrivialVertexShader() {
    std::string source = RasterPreamble();
    source +=
        fmt::format(R"MSL(
namespace MetalirisVertex {{
struct TrivialVertex {{
    float4 position [[attribute({})]];
    float4 color [[attribute({})]];
    float2 texcoord0 [[attribute({})]];
    float2 texcoord1 [[attribute({})]];
    float2 texcoord2 [[attribute({})]];
    float texcoord0_w [[attribute({})]];
    float4 quaternion [[attribute({})]];
    float3 view [[attribute({})]];
}};

vertex RasterVertex {}(TrivialVertex input [[stage_in]],
                       constant RasterUniforms& uniforms [[buffer({})]]) {{
    const ExpandedVertex expanded = {{
        input.position, input.quaternion, input.color, float4(input.texcoord0, input.texcoord1),
        float4(input.texcoord2, 0.0f, 0.0f), float4(input.view, input.texcoord0_w),
    }};
    return MakeRasterVertex(expanded, uniforms);
}}
}}
)MSL",
                    ATTRIBUTE_POSITION, ATTRIBUTE_COLOR, ATTRIBUTE_TEXCOORD0, ATTRIBUTE_TEXCOORD1,
                    ATTRIBUTE_TEXCOORD2, ATTRIBUTE_TEXCOORD0_W, ATTRIBUTE_NORMQUAT, ATTRIBUTE_VIEW,
                    VertexEntryPoints::Vertex, VertexBindings::RasterUniforms);
    return source;
}

std::string GenerateExpandedVertexShader() {
    std::string source = RasterPreamble();
    source += fmt::format(R"MSL(
namespace MetalirisVertex {{
vertex RasterVertex {}(device const ExpandedVertex* input [[buffer({})]],
                       constant RasterUniforms& uniforms [[buffer({})]],
                       uint index [[vertex_id]]) {{
    return MakeRasterVertex(input[index], uniforms);
}}
}}
)MSL",
                          VertexEntryPoints::ExpandedVertex, VertexBindings::InputVertices,
                          VertexBindings::RasterUniforms);
    return source;
}

DecompiledProgram GenerateVertexShader(const ShaderSetup& setup, const PicaVSConfig& config,
                                       const VertexProfile& profile) {
    auto program = DecompileProgram(setup.GetProgramCode(), setup.GetSwizzleData(),
                                    config.state.main_offset, {.function_name = "ExecuteVertex"});
    if (!program) {
        return program;
    }

    std::string source = RasterPreamble();
    AppendProgramTypes(source);
    source += program.source;
    source += R"MSL(
namespace MetalirisVertex {
using MetalirisPica::ExecuteVertex;
using MetalirisPica::RegisterState;
using MetalirisPica::Uniforms;
)MSL";
    AppendSemanticConversion(source, config.state.gs_state, config.state.num_outputs);
    AppendOutputMapping(source, config.state);
    AppendVertexStages(source, config.state, profile);
    source += "} // namespace MetalirisVertex\n";
    program.source = std::move(source);
    return program;
}

std::string GenerateFixedGeometryShader(const FixedGeometryConfig& config) {
    std::string source;
    if (config.path == GeometryPath::Mesh) {
        source += "#include <metal_mesh>\n";
    }
    source += RasterPreamble();
    AppendProgramTypes(source);
    source += R"MSL(
namespace MetalirisFixedGeometry {
using namespace MetalirisVertex;
)MSL";
    AppendSemanticConversion(source, config.semantics, config.semantics.vs_output_attributes_count);
    AppendFixedExpansion(source, config);
    source += "} // namespace MetalirisFixedGeometry\n";
    return source;
}

DecompiledProgram GenerateProgrammableGeometryShader(const ShaderSetup& setup,
                                                     const ProgrammableGeometryConfig& config) {
    ValidateGeometryConfig(config);
    auto program = DecompileProgram(setup.GetProgramCode(), setup.GetSwizzleData(), config.entry,
                                    {.kind = ProgramKind::Geometry,
                                     .function_name = "ExecuteGeometry",
                                     .uniform_storage = UniformStorage::Thread});
    if (!program) {
        return program;
    }

    std::string source = RasterPreamble();
    AppendProgramTypes(source);
    source += program.source;
    source += R"MSL(
namespace MetalirisProgrammableGeometry {
using namespace MetalirisVertex;
using MetalirisPica::ExecuteGeometry;
using MetalirisPica::RegisterState;
using MetalirisPica::Uniforms;
)MSL";
    AppendSemanticConversion(source, config.semantics, std::popcount(config.output_mask));
    source += GeometryContinuationType;
    AppendProgrammedEmitter(source, config);
    AppendGeometryInput(source, config);
    AppendProgrammedKernel(source);
    source += "} // namespace MetalirisProgrammableGeometry\n";
    program.source = std::move(source);
    return program;
}

} // namespace Pica::Shader::Generator::MSL
