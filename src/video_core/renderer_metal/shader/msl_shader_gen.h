// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include "video_core/pica/regs_pipeline.h"
#include "video_core/renderer_metal/shader/msl_shader_decompiler.h"
#include "video_core/shader/generator/shader_gen.h"

namespace Pica::Shader::Generator::MSL {

enum class GeometryPath : u8 {
    Vertex,
    Mesh,
    Compute,
};

struct GeometryProfile {
    bool mesh_shaders{};
    bool ordered_mesh_primitives{};
    bool fragment_barycentrics{};
};

/// Stateful PICA geometry and framebuffer feedback that needs ordered draws cannot be partitioned
/// into independent mesh threadgroups. Fans and per-primitive quaternion fixups require expansion.
[[nodiscard]] constexpr GeometryPath SelectGeometryPath(PipelineRegs::TriangleTopology topology,
                                                        const GeometryProfile& profile,
                                                        bool programmable_geometry, bool lighting,
                                                        bool serial_primitive_feedback) noexcept {
    using enum PipelineRegs::TriangleTopology;
    if (programmable_geometry || serial_primitive_feedback) {
        return GeometryPath::Compute;
    }
    if (topology != Fan && (!lighting || profile.fragment_barycentrics)) {
        return GeometryPath::Vertex;
    }
    return profile.mesh_shaders && profile.ordered_mesh_primitives ? GeometryPath::Mesh
                                                                   : GeometryPath::Compute;
}

[[nodiscard]] constexpr u32 PrimitiveCount(PipelineRegs::TriangleTopology topology,
                                           u32 vertex_count) noexcept {
    using enum PipelineRegs::TriangleTopology;
    switch (topology) {
    case List:
    case Shader:
        return vertex_count / 3;
    case Fan:
    case Strip:
        return vertex_count < 3 ? 0 : vertex_count - 2;
    }
    return 0;
}

[[nodiscard]] constexpr std::optional<std::array<u32, 3>> TriangleVertices(
    PipelineRegs::TriangleTopology topology, u32 vertex_count, u32 primitive) noexcept {
    using enum PipelineRegs::TriangleTopology;
    if (primitive >= PrimitiveCount(topology, vertex_count)) {
        return std::nullopt;
    }
    switch (topology) {
    case List:
    case Shader: {
        const u32 first = 3 * primitive;
        return std::array{first, first + 1, first + 2};
    }
    case Strip:
        return std::array{primitive + (primitive & 1), primitive + ((primitive & 1) ^ 1),
                          primitive + 2};
    case Fan:
        return std::array{0u, primitive + 1, primitive + 2};
    }
    return std::nullopt;
}

struct VertexBindings {
    static constexpr u32 PicaUniforms = 0;
    static constexpr u32 RasterUniforms = 1;
    static constexpr u32 InputVertices = 6;
    static constexpr u32 OutputVertices = 7;
    static constexpr u32 Parameters = 8;
    static constexpr u32 InvocationList = 9;
    static constexpr u32 GeometryState = 10;
    static constexpr u32 GeometryResult = 11;
};

struct VertexEntryPoints {
    static constexpr std::string_view Vertex = "pica_vertex";
    static constexpr std::string_view ExpandedVertex = "pica_expanded_vertex";
    static constexpr std::string_view TransformVertices = "pica_transform_vertices";
    static constexpr std::string_view Object = "pica_object";
    static constexpr std::string_view Mesh = "pica_mesh";
    static constexpr std::string_view Expand = "pica_expand";
};

struct VertexProfile {
    std::array<AttribLoadFlags, 16> load_flags{};
};

/// Compact outputs are in ascending enabled PICA output-register order. Expanded vertices hold
/// the guest clip position and varying values, before the native raster-position transform.
struct alignas(16) ExpandedVertex {
    RegisterState::Vector position;
    RegisterState::Vector quaternion;
    RegisterState::Vector color;
    RegisterState::Vector texcoords01;
    RegisterState::Vector texcoord2;
    /// XYZ contain the view vector, W contains the projective coordinate for texture unit 0.
    RegisterState::Vector view_and_texcoord0_w;
};

struct alignas(16) VertexTransformParams {
    u32 vertex_count;
    std::array<u32, 3> reserved{};
};

struct alignas(16) TransformedVertex {
    RegisterState::RegisterBank attributes;
    RunStatus status;
    std::array<u32, 3> reserved{};
};

struct alignas(16) FixedGeometryParams {
    u32 source_vertex_count;
    u32 first_primitive;
    u32 primitive_count;
    u32 output_vertex_capacity;

    /// For mesh draws the output capacity is supplied by each meshlet. Compute writes an
    /// ordered unindexed triangle list at the supplied buffer's base, including after a split.
    [[nodiscard]] constexpr bool IsValid(PipelineRegs::TriangleTopology topology,
                                         GeometryPath path) const noexcept {
        const u32 total = PrimitiveCount(topology, source_vertex_count);
        return first_primitive <= total && primitive_count <= total - first_primitive &&
               (path == GeometryPath::Mesh ||
                (path == GeometryPath::Compute && primitive_count <= output_vertex_capacity / 3));
    }
};

struct FixedGeometryConfig {
    PicaGSConfigState semantics{};
    PipelineRegs::TriangleTopology topology = PipelineRegs::TriangleTopology::List;
    GeometryPath path = GeometryPath::Compute;
    bool correct_quaternions{};
};

/// The native host parser consumes the variable-primitive count index and sends only vertices to
/// the vertex shader. Each complete geometry invocation refers to its consecutive compact outputs.
struct alignas(16) GeometryInvocation {
    u32 first_vertex;
    u32 vertex_count;
    std::array<u32, 2> reserved{};

    [[nodiscard]] constexpr bool InBounds(u32 source_vertex_count) const noexcept {
        return first_vertex <= source_vertex_count && vertex_count != 0 &&
               vertex_count <= source_vertex_count - first_vertex;
    }
};

struct ProgrammableGeometryConfig {
    PicaGSConfigState semantics{};
    PipelineRegs::GSMode mode = PipelineRegs::GSMode::Point;
    PipelineRegs::TriangleTopology topology = PipelineRegs::TriangleTopology::Shader;
    u32 entry{};
    u16 output_mask{};
    u8 vertex_output_count{};
    u8 input_attribute_count{};
    std::array<u8, 16> input_registers{};
    u32 uniform_start{};
    /// Number of complete vertex records at the start of a variable invocation; remaining
    /// records contribute only the first output vector.
    u32 main_vertex_count{};
    u32 fixed_vertex_count{};
    bool correct_quaternions{};

    [[nodiscard]] constexpr bool ValidInvocationSize(u32 vertices) const noexcept {
        if (vertices == 0 || vertex_output_count == 0 || vertex_output_count > 16) {
            return false;
        }
        using enum PipelineRegs::GSMode;
        switch (mode) {
        case Point:
            return input_attribute_count <= input_registers.size() &&
                   input_attribute_count / vertex_output_count == vertices &&
                   input_attribute_count % vertex_output_count == 0;
        case FixedPrimitive:
            return vertices == fixed_vertex_count && uniform_start <= 96 &&
                   vertices <= (96 - uniform_start) / vertex_output_count;
        case VariablePrimitive: {
            const u32 complete = std::min(vertices, main_vertex_count);
            return 1 + u64(complete) * vertex_output_count + (vertices - complete) <= 96;
        }
        }
        return false;
    }
};

struct alignas(16) ProgrammableGeometryParams {
    u32 source_vertex_count;
    u32 invocation_count;
    u32 output_vertex_capacity;
    /// PICA restart clears the primitive assembler, retaining shader and emitter state.
    u32 restart_primitive;
};

/// Stored on the device and retained between dependent geometry draws. A failed dispatch publishes
/// no continuation changes or vertices; the caller can grow output capacity and replay it.
struct alignas(16) GeometryContinuation {
    RegisterState registers{};
    VSPicaUniformData uniforms{};
    std::array<RegisterState::RegisterBank, 3> emitter{};
    std::array<ExpandedVertex, 2> primitive_vertices{};
    u32 primitive_count{};
    u32 primitive_cursor{};
    u32 primitive_ready{};
    u32 primitive_winding{};
};

enum class GeometryStatus : u32 {
    Complete,
    InvalidInput,
    ProgramFailure,
    OutputOverflow,
};

struct alignas(16) GeometryResult {
    GeometryStatus status;
    RunStatus program_status;
    u32 failed_invocation;
    u32 reserved{};
    /// MTLDrawPrimitivesIndirectArguments, with zero vertices on failure.
    u32 vertex_count;
    u32 instance_count;
    u32 vertex_start;
    u32 base_instance;
};

static_assert(sizeof(ExpandedVertex) == 96);
static_assert(offsetof(ExpandedVertex, view_and_texcoord0_w) == 80);
static_assert(sizeof(VertexTransformParams) == 16);
static_assert(offsetof(TransformedVertex, status) == 0x100);
static_assert(sizeof(TransformedVertex) == 0x110);
static_assert(sizeof(FixedGeometryParams) == 16);
static_assert(sizeof(GeometryInvocation) == 16);
static_assert(sizeof(ProgrammableGeometryParams) == 16);
static_assert(offsetof(GeometryContinuation, uniforms) == 0x430);
static_assert(offsetof(GeometryContinuation, emitter) == 0xA80);
static_assert(offsetof(GeometryContinuation, primitive_vertices) == 0xD80);
static_assert(sizeof(GeometryContinuation) == 0xE50);
static_assert(offsetof(GeometryResult, vertex_count) == 16);
static_assert(sizeof(GeometryResult) == 32);

/// Pass raster vertices using the original HardwareVertex stage-in attributes.
[[nodiscard]] std::string GenerateTrivialVertexShader();

/// Fetch native compute-expanded vertices through the InputVertices argument slot.
[[nodiscard]] std::string GenerateExpandedVertexShader();

/// Generate a native vertex entry and a compute transform entry that consumes input register banks
/// in guest submission order. Both apply the same program and compact output map.
[[nodiscard]] DecompiledProgram GenerateVertexShader(const ShaderSetup& setup,
                                                     const PicaVSConfig& config,
                                                     const VertexProfile& profile);

/// Generate an object/mesh pair or a compute shader over compact vertex outputs from the
/// vertex transform. Mesh dispatches one object per primitive with one mesh threadgroup.
[[nodiscard]] std::string GenerateFixedGeometryShader(const FixedGeometryConfig& config);

/// Generate the ordered native compute expansion for a programmed PICA geometry unit.
[[nodiscard]] DecompiledProgram GenerateProgrammableGeometryShader(
    const ShaderSetup& setup, const ProgrammableGeometryConfig& config);

} // namespace Pica::Shader::Generator::MSL
