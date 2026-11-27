#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"
#include "video_core/renderer_metal/shader/msl_fs_shader_gen.h"
#include "video_core/renderer_metal/shader/msl_shader_gen.h"

using namespace Metal;
using namespace Pica;
using namespace Pica::Shader::Generator;
using namespace Pica::Shader::Generator::MSL;
using namespace std::chrono_literals;
using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

namespace {

using Topology = PipelineRegs::TriangleTopology;
using Mode = PipelineRegs::GSMode;
using Semantic = RasterizerRegs::VSOutputAttributes::Semantic;
using Bank = RegisterState::RegisterBank;
constexpr u32 Side = 8;

Instruction Op(OpCode::Id opcode) {
    Instruction result{};
    result.opcode.Assign(opcode);
    return result;
}

Instruction Move(u32 destination, u32 source) {
    auto result = Op(OpCode::Id::MOV);
    result.common.dest = DestRegister(destination);
    result.common.src1 = SourceRegister(source);
    return result;
}

Instruction EmitVertex(u32 slot, bool emit, bool reverse) {
    auto result = Op(OpCode::Id::SETEMIT);
    result.setemit.vertex_id = slot;
    result.setemit.prim_emit = emit;
    result.setemit.winding = reverse;
    return result;
}

void SetProgram(ShaderSetup& setup, std::initializer_list<Instruction> instructions) {
    u32 index = 0;
    for (auto instruction : instructions) {
        setup.UpdateProgramCode(index++, instruction.hex);
    }
    SwizzlePattern identity{};
    identity.dest_mask = 15;
    for (u32 element = 0; element < 4; ++element) {
        identity.SetSelectorSrc1(element, static_cast<SwizzlePattern::Selector>(element));
    }
    setup.UpdateSwizzleData(0, identity.hex);
}

RasterizerRegs::VSOutputAttributes Semantics(Semantic x, Semantic y, Semantic z, Semantic w) {
    RasterizerRegs::VSOutputAttributes result{};
    result.map_x.Assign(x);
    result.map_y.Assign(y);
    result.map_z.Assign(z);
    result.map_w.Assign(w);
    return result;
}

PicaVSConfig VertexConfig() {
    using enum Semantic;
    PicaVSConfig config{};
    auto& state = config.state;
    state.num_outputs = 3;
    state.output_map.fill(16);
    state.output_map[0] = 0;
    state.output_map[3] = 1;
    state.output_map[5] = 2;
    state.used_input_vertex_attributes = 3;
    state.input_vertex_attributes[0].location = 0;
    state.input_vertex_attributes[1].location = 1;
    state.input_vertex_attributes[2].location = 2;
    auto& geometry = state.gs_state;
    geometry.vs_output_attributes_count = geometry.gs_output_attributes_count = 3;
    geometry.vs_output_total = 3;
    geometry.vs_output_attributes[0] = Semantics(POSITION_X, POSITION_Y, POSITION_Z, POSITION_W);
    geometry.vs_output_attributes[1] =
        Semantics(QUATERNION_X, QUATERNION_Y, QUATERNION_Z, QUATERNION_W);
    geometry.vs_output_attributes[2] = Semantics(COLOR_R, COLOR_G, COLOR_B, COLOR_A);
    return config;
}

ProgrammableGeometryConfig GeometryConfig(Mode mode) {
    ProgrammableGeometryConfig config{};
    config.semantics = VertexConfig().state.gs_state;
    config.semantics.vs_output_total = 1;
    config.mode = mode;
    config.output_mask = 1;
    config.vertex_output_count = mode == Mode::VariablePrimitive ? 2 : 1;
    config.input_attribute_count = 3;
    config.input_registers = {4, 6, 5};
    config.uniform_start = 8;
    config.main_vertex_count = 2;
    config.fixed_vertex_count = 3;
    config.correct_quaternions = true;
    return config;
}

std::array<u32, 3> PositionSources(Mode mode) {
    switch (mode) {
    case Mode::Point:
        return {4, 6, 5};
    case Mode::FixedPrimitive:
        return {32 + 8, 32 + 9, 32 + 10};
    case Mode::VariablePrimitive:
        return {32 + 1, 32 + 3, 32 + 5};
    }
    throw std::invalid_argument("Unknown geometry mode");
}

void SetGeometryProgram(ShaderSetup& setup, Mode mode) {
    const auto source = PositionSources(mode);
    SetProgram(setup, {Move(0, source[2]), EmitVertex(2, false, false), Op(OpCode::Id::EMIT),
                       Move(0, source[0]), EmitVertex(0, false, false), Op(OpCode::Id::EMIT),
                       Move(0, source[1]), EmitVertex(1, true, true), Op(OpCode::Id::EMIT),
                       Op(OpCode::Id::END)});
}

void Write(const std::filesystem::path& output, std::string_view name, std::string source,
           bool render = false) {
    if (render) {
        source += FragmentVaryingsDeclaration();
        source += R"MSL(
fragment metal::float4 fixture_fragment(MetalirisFragment::Varyings input [[stage_in]]) {
    return metal::float4(input.primary_color.rg, 0.5f + 0.25f * input.normquat.w, 1.0f);
}
)MSL";
    }
    std::ofstream file(output / (std::string(name) + ".metal"));
    file << source;
    assert(file.good());
}

void Generate(const std::filesystem::path& output) {
    Write(output, "trivial_vertex", GenerateTrivialVertexShader(), true);
    Write(output, "expanded_vertex", GenerateExpandedVertexShader(), true);
    ShaderSetup shader{};
    SetProgram(shader, {Move(0, 0), Move(3, 1), Move(5, 2), Op(OpCode::Id::END)});
    auto vertex_config = VertexConfig();
    VertexProfile profile{};
    profile.load_flags[0] = AttribLoadFlags::Float;
    profile.load_flags[1] = AttribLoadFlags::Sint | AttribLoadFlags::ZeroW;
    profile.load_flags[2] = AttribLoadFlags::Uint;
    const auto vertex = GenerateVertexShader(shader, vertex_config, profile);
    assert(vertex && vertex.diagnostics.empty());
    assert(vertex.source == GenerateVertexShader(shader, vertex_config, profile).source);
    assert(vertex.source.find("int4 value1 [[attribute(1)]]") != std::string::npos);
    assert(vertex.source.find("uint4 value2 [[attribute(2)]]") != std::string::npos);
    assert(vertex.source.find("registers.input[1].w = 0.0f") != std::string::npos);
    assert(vertex.source.find("result.attributes[2] = state.output[bank][5]") != std::string::npos);
    Write(output, "programmed_vertex", vertex.source, true);

    for (auto path : {GeometryPath::Compute, GeometryPath::Mesh}) {
        for (auto topology : {Topology::List, Topology::Strip, Topology::Fan}) {
            const bool mesh = path == GeometryPath::Mesh;
            auto source =
                GenerateFixedGeometryShader({vertex_config.state.gs_state, topology, path, true});
            Write(output,
                  (mesh ? "mesh_" : "compute_") + std::to_string(static_cast<u32>(topology)),
                  std::move(source), mesh);
        }
    }
    for (auto mode : {Mode::Point, Mode::FixedPrimitive, Mode::VariablePrimitive}) {
        ShaderSetup geometry{};
        SetGeometryProgram(geometry, mode);
        const auto generated = GenerateProgrammableGeometryShader(geometry, GeometryConfig(mode));
        assert(generated && generated.diagnostics.empty());
        assert(generated.source.find("thread const Uniforms& uniforms") != std::string::npos);
        assert(generated.source.find("state.uniforms.b |= 0x8000u") != std::string::npos);
        assert(generated.source.find("continuation = state") != std::string::npos);
        Write(output, "gs_" + std::to_string(static_cast<u32>(mode)), generated.source);
    }

    vertex_config.state.main_offset = MAX_PROGRAM_CODE_LENGTH;
    assert(!GenerateVertexShader(shader, vertex_config, profile));
    vertex_config.state.main_offset = 0;
    auto reject = [&](const PicaVSConfig& config) {
        try {
            static_cast<void>(GenerateVertexShader(shader, config, profile));
            assert(false);
        } catch (const std::invalid_argument&) {
        }
    };
    vertex_config.state.output_map[1] = 0;
    reject(vertex_config);
    vertex_config.state.output_map[1] = 16;
    vertex_config.state.used_input_vertex_attributes = 17;
    reject(vertex_config);
    std::puts("PASS: native source contracts, register maps, invalid layouts and 12 MSL variants");
}

class DeviceFixture {
public:
    DeviceFixture() : instance(0, false), scheduler(instance.GetContext()) {
        NSError* error = nil;
        compiler = [Device() newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        compiler = CheckResource(compiler, "Create native shader fixture compiler", error);
    }

    id<MTLDevice> Device() const {
        return instance.GetDevice();
    }

    id<MTLLibrary> Library(const std::filesystem::path& path) const {
        NSError* error = nil;
        auto lib = [Device() newLibraryWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                                         error:&error];
        return CheckResource(lib, "Load native shader fixture library", error);
    }

    static MTL4LibraryFunctionDescriptor* Function(id<MTLLibrary> library, std::string_view name) {
        auto descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library = library;
        descriptor.name = [[NSString alloc] initWithBytes:name.data()
                                                   length:name.size()
                                                 encoding:NSUTF8StringEncoding];
        return descriptor;
    }

    template <typename T>
    id<MTLBuffer> Shared(const T& value) const {
        return CheckResource([Device() newBufferWithBytes:&value
                                                   length:sizeof(value)
                                                  options:MTLResourceStorageModeShared |
                                                          MTLResourceHazardTrackingModeUntracked],
                             "Allocate native shader fixture buffer");
    }

    id<MTL4ArgumentTable> Table(u32 size) const {
        auto descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = size;
        NSError* error = nil;
        auto table = [Device() newArgumentTableWithDescriptor:descriptor error:&error];
        return CheckResource(table, "Allocate native shader fixture argument table", error);
    }

    struct Argument {
        u32 index;
        id<MTLBuffer> buffer;
        BufferAccess access;
    };

    void Dispatch(id<MTLLibrary> library, std::string_view function, u32 threads,
                  std::vector<Argument> arguments) {
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = Function(library, function);
        NSError* error = nil;
        auto state = [compiler newComputePipelineStateWithDescriptor:descriptor
                                                 compilerTaskOptions:nil
                                                               error:&error];
        state = CheckResource(state, "Compile native shader fixture compute state", error);
        auto table = Table(VertexBindings::GeometryResult + 1);
        for (const auto& argument : arguments) {
            [table setAddress:argument.buffer.gpuAddress atIndex:argument.index];
        }
        scheduler.Record(
            [state, table, threads, arguments = std::move(arguments)](EncodingContext& ctx) {
                auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
                std::vector<ResourceUse> uses;
                for (const auto& arg : arguments) {
                    uses.push_back({arg.buffer, arg.access, MTLStageDispatch});
                }
                ctx.Synchronize(encoder, uses);
                [encoder setComputePipelineState:state];
                [encoder setArgumentTable:table];
                [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                ctx.Retain(state);
                ctx.Retain(table);
                ctx.EndPass(encoder);
            });
        Wait();
    }

    void Wait() {
        assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
               CompletionStatus::Complete);
    }

    Instance instance;
    Scheduler scheduler;
    id<MTL4Compiler> compiler;
};

void CheckGeometryModes(DeviceFixture& fixture, const std::filesystem::path& directory) {
    const std::array<RegisterState::Vector, 6> positions{
        {{-1, -1, -0.5, 1},
         {3, -1, -0.5, 1},
         {-1, 3, -0.5, 1},
         {1, 1, -0.5, 1},
         {-3, 1, -0.5, 1},
         {1, -3, -0.5, 1}},
    };
    std::array<TransformedVertex, 6> input{};
    for (u32 i = 0; i < input.size(); ++i) {
        input[i].attributes[0] = positions[i];
        input[i].attributes[1] = {100, 200, 300, 400};
    }
    auto vertices = fixture.Shared(input);
    auto output = fixture.Shared(std::array<ExpandedVertex, 6>{});
    auto result = fixture.Shared(GeometryResult{});
    auto invocations = fixture.Shared(std::array<GeometryInvocation, 2>{{{0, 3}, {3, 3}}});
    auto continuation = fixture.Shared(GeometryContinuation{});
    auto parameters = fixture.Shared(ProgrammableGeometryParams{6, 2, 3, 0});
    const std::vector<DeviceFixture::Argument> args{
        {VertexBindings::InputVertices, vertices, BufferAccess::Read},
        {VertexBindings::OutputVertices, output, BufferAccess::Write},
        {VertexBindings::Parameters, parameters, BufferAccess::Read},
        {VertexBindings::InvocationList, invocations, BufferAccess::Read},
        {VertexBindings::GeometryState, continuation, BufferAccess::ReadWrite},
        {VertexBindings::GeometryResult, result, BufferAccess::Write},
    };

    for (const auto mode : {Mode::Point, Mode::FixedPrimitive, Mode::VariablePrimitive}) {
        const std::string name = "gs_" + std::to_string(static_cast<u32>(mode)) + ".metallib";
        const auto program = fixture.Library(directory / name);
        std::memset(continuation.contents, 0, sizeof(GeometryContinuation));
        auto& params = *static_cast<ProgrammableGeometryParams*>(parameters.contents);
        params.output_vertex_capacity = 3;
        fixture.Dispatch(program, VertexEntryPoints::Expand, 1, args);
        const auto& status = *static_cast<const GeometryResult*>(result.contents);
        const auto& state = *static_cast<const GeometryContinuation*>(continuation.contents);
        assert(status.status == GeometryStatus::OutputOverflow && status.vertex_count == 0);
        assert(status.program_status == RunStatus::EmitOverflow && status.failed_invocation == 1);
        const auto* state_bytes = static_cast<const u8*>(continuation.contents);
        for (u32 i = 0; i < sizeof(GeometryContinuation); ++i) {
            assert(state_bytes[i] == 0);
        }
        params.output_vertex_capacity = 6;
        fixture.Dispatch(program, VertexEntryPoints::Expand, 1, args);
        assert(status.status == GeometryStatus::Complete && status.vertex_count == 6);
        assert(status.instance_count == 1 && state.uniforms.b == 0x8000u);
        const auto* actual = static_cast<const ExpandedVertex*>(output.contents);
        const std::array<u32, 6> ordered{1, 0, 2, 4, 3, 5};
        for (u32 i = 0; i < positions.size(); ++i) {
            assert(actual[i].position == positions[ordered[i]]);
        }
    }
}

void CheckRaster(DeviceFixture& fixture, id<MTLRenderPipelineState> pipeline,
                 id<MTLBuffer> vertices, id<MTLBuffer> parameters, bool mesh, bool clip,
                 MemoryAllocator& allocator, id<MTLBuffer> pica = nil,
                 std::array<u8, 4> color = {128, 64, 191, 255}) {
    RenderManager manager(fixture.instance, fixture.scheduler, allocator);
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:Side
                                                                  height:Side
                                                               mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    auto target = CheckResource([fixture.Device() newTextureWithDescriptor:desc],
                                "Allocate native geometry fixture target");
    VSUniformData uniforms{};
    if (clip) {
        uniforms.enable_clip1 = 1;
        uniforms.clip_coef = {1, 0, 0, 0};
    }
    auto raster = fixture.Shared(uniforms);
    auto table = fixture.Table(VertexBindings::Parameters + 1);
    [table setAddress:raster.gpuAddress atIndex:VertexBindings::RasterUniforms];
    [table setAddress:vertices.gpuAddress atIndex:VertexBindings::InputVertices];
    if (parameters) {
        [table setAddress:parameters.gpuAddress atIndex:VertexBindings::Parameters];
    }
    if (pica) {
        [table setAddress:pica.gpuAddress atIndex:VertexBindings::PicaUniforms];
    }
    const RenderPass pass{
        .color = {.target = {target}, .load_action = MTLLoadActionClear},
        .area = {0, 0, Side, Side},
        .width = Side,
        .height = Side,
        .clear_color = MTLClearColorMake(0, 0, 0, 1),
    };
    manager.BeginRendering(pass);
    manager.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
        ctx.Retain(table);
        ctx.Retain(pipeline);
        ctx.MakeResident(raster);
        ctx.MakeResident(vertices);
        [encoder setViewport:MTLViewport{0, 0, Side, Side, 0, 1}];
        [encoder setRenderPipelineState:pipeline];
        if (mesh) {
            ctx.MakeResident(parameters);
            [encoder setArgumentTable:table atStages:MTLRenderStageObject | MTLRenderStageMesh];
            [encoder drawMeshThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
                  threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
        } else {
            if (pica) {
                ctx.MakeResident(pica);
            }
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
    });
    fixture.Wait();
    std::array<u8, Side * Side * 4> actual{};
    [target getBytes:actual.data()
         bytesPerRow:Side * 4
          fromRegion:MTLRegionMake2D(0, 0, Side, Side)
         mipmapLevel:0];
    const std::array<u8, 4> background{0, 0, 0, 255};
    for (u32 y = 0; y < Side; ++y) {
        for (u32 x = 0; x < Side; ++x) {
            const auto& expected = clip && x < Side / 2 ? background : color;
            assert(std::memcmp(actual.data() + (y * Side + x) * 4, expected.data(), 4) == 0);
        }
    }
}

void CheckFixedStages(DeviceFixture& fixture, const std::filesystem::path& directory) {
    const std::array<RegisterState::Vector, 3> positions{
        {{-1, -1, -0.5, 1}, {3, -1, -0.5, 1}, {-1, 3, -0.5, 1}},
    };
    std::array<Bank, 3> input{};
    for (u32 i = 0; i < input.size(); ++i) {
        input[i][0] = positions[i];
        input[i][1] = {0, 0, 0, i == 1 ? -1.0f : 1.0f};
        input[i][2] = {-0.5f, 0.25f, 0.75f, 1};
    }
    auto input_buffer = fixture.Shared(input);
    auto output = fixture.Shared(std::array<TransformedVertex, 3>{});
    auto uniforms = fixture.Shared(VSPicaUniformData{});
    auto parameters = fixture.Shared(VertexTransformParams{3});
    auto vertex_program = fixture.Library(directory / "programmed_vertex.metallib");
    fixture.Dispatch(vertex_program, VertexEntryPoints::TransformVertices, input.size(),
                     {{VertexBindings::PicaUniforms, uniforms, BufferAccess::Read},
                      {VertexBindings::InputVertices, input_buffer, BufferAccess::Read},
                      {VertexBindings::OutputVertices, output, BufferAccess::Write},
                      {VertexBindings::Parameters, parameters, BufferAccess::Read}});
    const auto* transformed = static_cast<const TransformedVertex*>(output.contents);
    for (u32 i = 0; i < input.size(); ++i) {
        assert(transformed[i].status == RunStatus::Complete);
        for (u32 channel = 0; channel < 3; ++channel) {
            assert(transformed[i].attributes[channel] == input[i][channel]);
        }
    }

    auto compute = fixture.Library(directory / "compute_0.metallib");
    auto expanded = fixture.Shared(std::array<ExpandedVertex, 3>{});
    auto result = fixture.Shared(GeometryResult{});
    auto fixed_parameters = fixture.Shared(FixedGeometryParams{3, 0, 1, 3});
    fixture.Dispatch(compute, VertexEntryPoints::Expand, 1,
                     {{VertexBindings::InputVertices, output, BufferAccess::Read},
                      {VertexBindings::OutputVertices, expanded, BufferAccess::Write},
                      {VertexBindings::Parameters, fixed_parameters, BufferAccess::Read},
                      {VertexBindings::GeometryResult, result, BufferAccess::Write}});
    assert(static_cast<const GeometryResult*>(result.contents)->vertex_count == 3);
    const auto* decoded = static_cast<const ExpandedVertex*>(expanded.contents);
    for (u32 i = 0; i < input.size(); ++i) {
        assert(decoded[i].position == positions[i]);
        assert(decoded[i].quaternion[3] == 1.0f);
        assert((decoded[i].color == RegisterState::Vector{0.5f, 0.25f, 0.75f, 1.0f}));
    }

    auto vertex = fixture.Library(directory / "expanded_vertex.metallib");
    auto conventional = [MTL4RenderPipelineDescriptor new];
    conventional.vertexFunctionDescriptor =
        DeviceFixture::Function(vertex, VertexEntryPoints::ExpandedVertex);
    conventional.fragmentFunctionDescriptor = DeviceFixture::Function(vertex, "fixture_fragment");
    conventional.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError* error = nil;
    auto vertex_pipeline = [fixture.compiler newRenderPipelineStateWithDescriptor:conventional
                                                              compilerTaskOptions:nil
                                                                            error:&error];
    vertex_pipeline =
        CheckResource(vertex_pipeline, "Compile compute-generated raster path", error);
    MemoryAllocator allocator(fixture.Device(), MemoryBudget(8 * 1024 * 1024));
    CheckRaster(fixture, vertex_pipeline, expanded, nil, false, false, allocator);
    CheckRaster(fixture, vertex_pipeline, expanded, nil, false, true, allocator);

    struct alignas(16) TypedVertex {
        RegisterState::Vector position;
        std::array<s32, 4> quaternion;
        std::array<u32, 4> color;
    };
    std::array<TypedVertex, 3> typed{};
    for (u32 i = 0; i < positions.size(); ++i) {
        typed[i] = {positions[i], {0, 0, -2, 2}, {9, 0, 2, 1}};
    }
    auto typed_vertices = fixture.Shared(typed);
    auto layout = [MTLVertexDescriptor new];
    const std::array formats{MTLVertexFormatFloat4, MTLVertexFormatInt4, MTLVertexFormatUInt4};
    const std::array offsets{offsetof(TypedVertex, position), offsetof(TypedVertex, quaternion),
                             offsetof(TypedVertex, color)};
    for (u32 i = 0; i < formats.size(); ++i) {
        layout.attributes[i].format = formats[i];
        layout.attributes[i].offset = offsets[i];
        layout.attributes[i].bufferIndex = VertexBindings::InputVertices;
    }
    layout.layouts[VertexBindings::InputVertices].stride = sizeof(TypedVertex);
    conventional.vertexFunctionDescriptor =
        DeviceFixture::Function(vertex_program, VertexEntryPoints::Vertex);
    conventional.fragmentFunctionDescriptor =
        DeviceFixture::Function(vertex_program, "fixture_fragment");
    conventional.vertexDescriptor = layout;
    auto typed_pipeline = [fixture.compiler newRenderPipelineStateWithDescriptor:conventional
                                                             compilerTaskOptions:nil
                                                                           error:&error];
    typed_pipeline = CheckResource(typed_pipeline, "Compile native typed vertex inputs", error);
    CheckRaster(fixture, typed_pipeline, typed_vertices, nil, false, false, allocator, uniforms,
                {255, 0, 128, 255});

    if (![fixture.Device() supportsFamily:MTLGPUFamilyApple7]) {
        std::puts("PENDING: mesh rendering requires a supporting Apple GPU.");
        return;
    }
    auto mesh = fixture.Library(directory / "mesh_0.metallib");
    auto native = [MTL4MeshRenderPipelineDescriptor new];
    native.objectFunctionDescriptor = DeviceFixture::Function(mesh, VertexEntryPoints::Object);
    native.meshFunctionDescriptor = DeviceFixture::Function(mesh, VertexEntryPoints::Mesh);
    native.fragmentFunctionDescriptor = DeviceFixture::Function(mesh, "fixture_fragment");
    native.maxTotalThreadsPerObjectThreadgroup = 1;
    native.maxTotalThreadsPerMeshThreadgroup = 1;
    native.maxTotalThreadgroupsPerMeshGrid = 1;
    native.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    auto mesh_pipeline = [fixture.compiler newRenderPipelineStateWithDescriptor:native
                                                            compilerTaskOptions:nil
                                                                          error:&error];
    mesh_pipeline = CheckResource(mesh_pipeline, "Compile native mesh raster fixture", error);
    CheckRaster(fixture, mesh_pipeline, output, fixed_parameters, true, false, allocator);
    CheckRaster(fixture, mesh_pipeline, output, fixed_parameters, true, true, allocator);
}

void Run(const std::filesystem::path& directory) {
    if (!MTLCreateSystemDefaultDevice()) {
        std::puts("PENDING: no process-visible Apple GPU. Native typed transform, "
                  "mesh versus compute clip and interpolation, and programmable geometry "
                  "winding, persistence and overflow replay require Metal 4 hardware.");
        return;
    }
    DeviceFixture fixture;
    CheckGeometryModes(fixture, directory);
    CheckFixedStages(fixture, directory);
    assert(fixture.scheduler.Shutdown(std::chrono::steady_clock::now() + 10s));
    std::puts("PASS: generated GPU vertex, fixed expansion and ordered geometry oracles");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        if (std::string_view(argv[1]) == "generate") {
            Generate(argv[2]);
        } else {
            assert(std::string_view(argv[1]) == "run");
            Run(argv[2]);
        }
    }
}
