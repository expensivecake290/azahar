#include <cassert>
#include <cstdio>
#include <limits>
#import <Metal/Metal.h>
#include "video_core/renderer_metal/shader/msl_shader_gen.h"

using namespace Pica;
using namespace Pica::Shader::Generator;
using namespace Pica::Shader::Generator::MSL;
using Topology = PipelineRegs::TriangleTopology;
using GeometryMode = PipelineRegs::GSMode;

static_assert(sizeof(MTLDrawPrimitivesIndirectArguments) == sizeof(GeometryResult) - 16);
static_assert(offsetof(GeometryResult, base_instance) - offsetof(GeometryResult, vertex_count) ==
              offsetof(MTLDrawPrimitivesIndirectArguments, baseInstance));

#if defined(CHECK_NATIVE_CONSUMER)
std::string BuildVertex(const ShaderSetup& setup, const PicaVSConfig& config, bool use_fixed,
                        bool use_mesh, bool serial_primitive_feedback) {
    const auto path = SelectGeometryPath(
        Topology::Strip, {.mesh_shaders = use_mesh, .ordered_mesh_primitives = use_mesh}, false,
        use_fixed, serial_primitive_feedback);
    if (path == GeometryPath::Vertex) {
        VertexProfile profile;
        profile.load_flags[0] = AttribLoadFlags::Float;
        profile.load_flags[1] = AttribLoadFlags::Sint | AttribLoadFlags::ZeroW;
        profile.load_flags[2] = AttribLoadFlags::Uint;
        return GenerateVertexShader(setup, config, profile).source;
    }

    FixedGeometryConfig fixed{
        .semantics = config.state.gs_state,
        .topology = Topology::Strip,
        .path = path,
        .correct_quaternions = use_fixed,
    };
    if (path == GeometryPath::Mesh) {
        auto descriptor = [MTL4MeshRenderPipelineDescriptor new];
        descriptor.objectFunctionDescriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.meshFunctionDescriptor = [MTL4LibraryFunctionDescriptor new];
    }
    return GenerateFixedGeometryShader(fixed) + GenerateExpandedVertexShader();
}

DecompiledProgram BuildProgrammedGeometry(const ShaderSetup& setup, const PicaVSConfig& config) {
    ProgrammableGeometryConfig geometry{
        .semantics = config.state.gs_state,
        .mode = GeometryMode::VariablePrimitive,
        .vertex_output_count = 3,
        .main_vertex_count = 2,
        .correct_quaternions = true,
    };
    return GenerateProgrammableGeometryShader(setup, geometry);
}
#else
int main() {
    constexpr GeometryProfile mesh{.mesh_shaders = true, .ordered_mesh_primitives = true};
    static_assert(SelectGeometryPath(Topology::List, {}, false, false, false) ==
                  GeometryPath::Vertex);
    static_assert(SelectGeometryPath(Topology::List, {.fragment_barycentrics = true}, false, true,
                                     false) == GeometryPath::Vertex);
    static_assert(SelectGeometryPath(Topology::Fan, mesh, false, false, false) ==
                  GeometryPath::Mesh);
    static_assert(SelectGeometryPath(Topology::Strip, mesh, false, true, false) ==
                  GeometryPath::Mesh);
    static_assert(SelectGeometryPath(Topology::Strip, {.mesh_shaders = true}, false, true, false) ==
                  GeometryPath::Compute);
    static_assert(SelectGeometryPath(Topology::Shader, mesh, true, true, false) ==
                  GeometryPath::Compute);
    static_assert(SelectGeometryPath(Topology::List, mesh, false, false, true) ==
                  GeometryPath::Compute);

    for (u32 size = 0; size < 37; ++size) {
        for (const auto topology :
             {Topology::List, Topology::Shader, Topology::Strip, Topology::Fan}) {
            std::array<u32, 2> slots{};
            u32 cursor = 0;
            bool ready = false;
            u32 primitive = 0;
            for (u32 vertex = 0; vertex < size; ++vertex) {
                if (topology == Topology::List || topology == Topology::Shader) {
                    if (cursor == 2) {
                        assert((TriangleVertices(topology, size, primitive++) ==
                                std::array{slots[0], slots[1], vertex}));
                        cursor = 0;
                    } else {
                        slots[cursor++] = vertex;
                    }
                } else {
                    if (ready) {
                        assert((TriangleVertices(topology, size, primitive++) ==
                                std::array{slots[0], slots[1], vertex}));
                    }
                    slots[cursor] = vertex;
                    ready |= cursor == 1;
                    cursor = topology == Topology::Fan ? 1 : !cursor;
                }
            }
            assert(primitive == PrimitiveCount(topology, size));
            assert(!TriangleVertices(topology, size, primitive));
        }
    }
    constexpr u32 max = std::numeric_limits<u32>::max();
    static_assert(TriangleVertices(Topology::List, max, max / 3 - 1) ==
                  std::array{max - 3, max - 2, max - 1});
    static_assert(TriangleVertices(Topology::Fan, max, max - 3) ==
                  std::array{0u, max - 2, max - 1});
    static_assert(
        !FixedGeometryParams{max, 0, max - 2, max}.IsValid(Topology::Fan, GeometryPath::Compute));
    static_assert(
        FixedGeometryParams{13, 4, 4, 12}.IsValid(Topology::Strip, GeometryPath::Compute));
    static_assert(FixedGeometryParams{13, 4, 4, 0}.IsValid(Topology::Strip, GeometryPath::Mesh));
    static_assert(
        !FixedGeometryParams{13, 4, 4, 11}.IsValid(Topology::Strip, GeometryPath::Compute));
    static_assert(!FixedGeometryParams{13, 12, 0, 0}.IsValid(Topology::Strip, GeometryPath::Mesh));
    static_assert(GeometryInvocation{max - 3, 3}.InBounds(max));
    static_assert(!GeometryInvocation{max - 3, 4}.InBounds(max));
    static_assert(!GeometryInvocation{1, 0}.InBounds(max));
    constexpr ProgrammableGeometryConfig point{.vertex_output_count = 4,
                                               .input_attribute_count = 12};
    static_assert(point.ValidInvocationSize(3));
    static_assert(!point.ValidInvocationSize(4));
    constexpr ProgrammableGeometryConfig fixed{.mode = GeometryMode::FixedPrimitive,
                                               .vertex_output_count = 4,
                                               .uniform_start = 84,
                                               .fixed_vertex_count = 3};
    static_assert(fixed.ValidInvocationSize(3));
    static_assert(!fixed.ValidInvocationSize(4));
    constexpr ProgrammableGeometryConfig variable{
        .mode = GeometryMode::VariablePrimitive, .vertex_output_count = 4, .main_vertex_count = 3};
    static_assert(variable.ValidInvocationSize(86));
    static_assert(!variable.ValidInvocationSize(87));
    static_assert(!variable.ValidInvocationSize(max));
    std::puts("PASS: native geometry ABI, path selection and independent primitive order oracle");
}
#endif
