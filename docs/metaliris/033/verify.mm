#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"
#include "video_core/renderer_metal/shader/msl_fs_shader_gen.h"

using namespace Metal;
using namespace Pica;
using namespace Pica::Shader;
using namespace Pica::Shader::Generator::MSL;
using namespace std::chrono_literals;

namespace {

using Tev = TexturingRegs::TevStageConfig;
using Logic = FramebufferRegs::LogicOp;
using TextureType = TexturingRegs::TextureConfig::TextureType;
constexpr u32 Side = 8;

constexpr std::string_view VertexSource = R"(
vertex MetalirisFragment::Varyings fixture_vertex(uint index [[vertex_id]]) {
    using namespace metal;
    MetalirisFragment::Varyings value{};
    const float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    value.position = float4(positions[index], 0.5, 1);
    value.primary_color = float4(1);
    value.texcoord0 = value.texcoord1 = value.texcoord2 = float2(0.5);
    value.texcoord0_w = 1;
    value.normquat = float4(0, 0, 0, 1);
    value.view = float3(0, 0, 1);
    return value;
}
)";

TevStageConfigRaw Stage(Tev::Source source) {
    Tev stage{};
    stage.color_source1.Assign(source);
    stage.alpha_source1.Assign(source);
    return {stage.sources_raw, stage.modifiers_raw, stage.ops_raw, stage.scales_raw};
}

FSConfig MakeConfig(Tev::Source first = Tev::Source::Constant) {
    FSConfig config{RegsInternal{}};
    config.framebuffer.alpha_test_func.Assign(FramebufferRegs::CompareFunc::Always);
    config.framebuffer.scissor_test_mode.Assign(RasterizerRegs::ScissorMode::Disabled);
    config.framebuffer.depthmap_enable.Assign(RasterizerRegs::DepthBuffering::ZBuffering);
    config.framebuffer.requested_logic_op = Logic::Copy;
    config.texture.texture0_type.Assign(TextureType::Disabled);
    config.lighting.enable.Assign(0);
    config.texture.tev_stages.fill(Stage(Tev::Source::Previous));
    config.texture.tev_stages.front() = Stage(first);
    return config;
}

class Corpus {
public:
    explicit Corpus(std::filesystem::path directory) : directory(std::move(directory)) {}

    FragmentShader Add(std::string_view name, const FSConfig& config, FragmentProfile profile = {},
                       UserConfig user = {}) {
        auto shader = GenerateFragmentShader(config, user, profile);
        assert(shader.requirements == DescribeFragment(config, profile));
        std::ofstream output(directory / (std::string(name) + ".metal"));
        output << shader.source << VertexSource;
        assert(output.good());
        ++count;
        return shader;
    }

    u32 count{};

private:
    std::filesystem::path directory;
};

void Generate(const std::filesystem::path& output) {
    Corpus corpus{output};
    const auto constant = MakeConfig();
    assert(corpus.Add("constant", constant).requirements.feedback == FragmentFeedback::None);
    auto config = constant;
    config.framebuffer.alpha_test_func.Assign(FramebufferRegs::CompareFunc::Never);
    corpus.Add("discard", config);
    config.framebuffer.alpha_test_func.Assign(FramebufferRegs::CompareFunc::GreaterThan);
    config.framebuffer.scissor_test_mode.Assign(RasterizerRegs::ScissorMode::Include);
    config.framebuffer.depthmap_enable.Assign(RasterizerRegs::DepthBuffering::WBuffering);
    config.texture.fog_mode.Assign(TexturingRegs::FogMode::Fog);
    config.texture.fog_flip.Assign(1);
    corpus.Add("depth_fog", config);
    config.texture.fog_mode.Assign(TexturingRegs::FogMode::Gas);
    assert(corpus.Add("gas", config).requirements.writes_color);

    for (u32 op = 0; op < 16; ++op) {
        config = constant;
        config.framebuffer.requested_logic_op = static_cast<Logic>(op);
        const auto requirements = DescribeFragment(config, {});
        assert(requirements.writes_color == (op != static_cast<u32>(Logic::NoOp)));
        assert((requirements.feedback == FragmentFeedback::TextureCopy) ==
               LogicReadsDestination(static_cast<Logic>(op)));
    }

    config = constant;
    config.framebuffer.requested_logic_op = Logic::Xor;
    assert(corpus.Add("xor_copy", config).requirements.feedback == FragmentFeedback::TextureCopy);
    assert(corpus.Add("xor_ordered", config, {.raster_order_groups = true}).requirements.feedback ==
           FragmentFeedback::RasterOrderGroup);
    auto msaa = corpus.Add("xor_multisample", config, {.samples = 4});
    assert(msaa.source.find("texture2d_ms<float") != std::string::npos);
    assert(msaa.source.find(", sample_index)") != std::string::npos);
    corpus.Add("xor_multisample_ordered", config, {.samples = 4, .raster_order_groups = true});
    auto rgba4 = corpus.Add("xor_rgba4", config, {.color_format = VideoCore::PixelFormat::RGBA4});
    assert(rgba4.source.find("15u, 15u, 15u, 15u") != std::string::npos);

    using Equation = FramebufferRegs::BlendEquation;
    using Factor = FramebufferRegs::BlendFactor;
    config.framebuffer.alphablend_enable.Assign(1);
    config.framebuffer.requested_rgb_blend = {Equation::Min, Factor::ConstantColor, Factor::One};
    config.framebuffer.requested_alpha_blend = {Equation::Max, Factor::One, Factor::One};
    const auto blend = corpus.Add("factor_min", config, {.raster_order_groups = true});
    assert(blend.requirements.emulated_rgb_blend);
    assert(!blend.requirements.emulated_alpha_blend);
    assert(blend.requirements.feedback == FragmentFeedback::RasterOrderGroup);
    config.framebuffer.requested_alpha_blend.dst_factor = Factor::SourceAlpha;
    const auto both = corpus.Add("factor_minmax", config);
    assert(both.requirements.emulated_rgb_blend && both.requirements.emulated_alpha_blend);

    config = constant;
    config.framebuffer.shadow_rendering.Assign(1);
    assert(corpus.Add("shadow_buffer", config).requirements.shadow_storage ==
           ShadowStorage::AtomicBuffer);
    assert(corpus.Add("shadow_texture", config, {.texture_atomics = true})
               .requirements.shadow_storage == ShadowStorage::AtomicTexture);

    config = MakeConfig(Tev::Source::Texture0);
    const std::array texture_types{TextureType::Texture2D, TextureType::TextureCube,
                                   TextureType::Projection2D, TextureType::Shadow2D,
                                   TextureType::ShadowCube};
    config.texture.texture_border_color[0].enable_s.Assign(1);
    for (u32 i = 0; i < texture_types.size(); ++i) {
        config.texture.texture0_type.Assign(texture_types[i]);
        corpus.Add("texture_" + std::to_string(i), config);
    }

    config = MakeConfig(Tev::Source::Texture3);
    config.proctex.enable.Assign(1);
    config.proctex.noise_enable.Assign(1);
    config.proctex.separate_alpha.Assign(1);
    config.proctex.lut_width = 128;
    config.proctex.lod_max = 7;
    for (u32 filter = 0; filter < 6; ++filter) {
        config.proctex.lut_filter.Assign(static_cast<TexturingRegs::ProcTexFilter>(filter));
        config.proctex.color_combiner.Assign(static_cast<TexturingRegs::ProcTexCombiner>(filter));
        config.proctex.alpha_combiner.Assign(
            static_cast<TexturingRegs::ProcTexCombiner>(filter + 3));
        config.proctex.u_clamp.Assign(static_cast<TexturingRegs::ProcTexClamp>(filter % 5));
        corpus.Add("procedural_" + std::to_string(filter), config);
    }

    config = MakeConfig(Tev::Source::PrimaryFragmentColor);
    config.lighting.enable.Assign(1);
    config.lighting.src_num.Assign(1);
    config.lighting.config.Assign(LightingRegs::LightingConfig::Config7);
    config.lighting.lights[0].spot_atten_enable.Assign(1);
    config.lighting.lights[0].dist_atten_enable.Assign(1);
    config.lighting.lights[0].geometric_factor_0.Assign(1);
    config.lighting.lut_d0.enable.Assign(1);
    config.lighting.lut_d0.type.Assign(LightingRegs::LightingLutInput::CP);
    config.lighting.lut_d0.scale_bits = std::bit_cast<u32>(1.0f);
    const auto lighting = corpus.Add("lighting", config);
    assert(!lighting.requirements.fragment_barycentrics);
    UserConfig user{};
    user.use_custom_normal.Assign(1);
    assert(corpus.Add("lighting_barycentric", config, {.fragment_barycentrics = true}, user)
               .requirements.fragment_barycentrics);

    auto multi_axis = MakeConfig();
    for (u32 operation = 0; operation <= 9; ++operation) {
        Tev stage{};
        stage.color_source1.Assign(Tev::Source::Constant);
        stage.color_source2.Assign(Tev::Source::Previous);
        stage.color_source3.Assign(Tev::Source::PrimaryColor);
        stage.alpha_source1.Assign(Tev::Source::Constant);
        stage.alpha_source2.Assign(Tev::Source::Previous);
        stage.alpha_source3.Assign(Tev::Source::PrimaryColor);
        stage.color_op.Assign(static_cast<Tev::Operation>(operation));
        stage.alpha_op.Assign(static_cast<Tev::Operation>(operation));
        multi_axis.texture.tev_stages[1 + operation % 5] = {stage.sources_raw, stage.modifiers_raw,
                                                            stage.ops_raw, stage.scales_raw};
        multi_axis.texture.combiner_buffer_input.Assign(0xA5);
        if (operation % 5 == 4) {
            corpus.Add("tev_" + std::to_string(operation), multi_axis);
        }
    }

    bool invalid_samples = false;
    try {
        static_cast<void>(GenerateFragmentShader(constant, {}, {.samples = 3}));
    } catch (const std::invalid_argument&) {
        invalid_samples = true;
    }
    assert(invalid_samples);
    std::printf(
        "PASS: fragment source/metadata oracle, all 16 logic ops and %u native MSL variants\n",
        corpus.count);
}

id<MTLRenderPipelineState> Pipeline(id<MTLDevice> device, const std::filesystem::path& path) {
    NSError* error = nil;
    auto url = [NSURL fileURLWithPath:@(path.c_str())];
    auto library = [device newLibraryWithURL:url error:&error];
    library = CheckResource(library, "Load generated fragment library", error);
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create fragment fixture compiler", error);
    auto vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = library;
    vertex.name = @"fixture_vertex";
    auto fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = library;
    fragment.name = @"pica_fragment";
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vertex;
    descriptor.fragmentFunctionDescriptor = fragment;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    auto pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                               compilerTaskOptions:nil
                                                             error:&error];
    return CheckResource(pipeline, "Compile generated fragment pipeline", error);
}

id<MTLTexture> Target(id<MTLDevice> device) {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:Side
                                                                  height:Side
                                                               mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    return CheckResource([device newTextureWithDescriptor:desc], "Allocate fragment target");
}

void CheckPixels(Instance& instance, const std::filesystem::path& directory,
                 std::string_view shader, bool copy, bool ordered,
                 const std::array<u8, 4>& expected) {
    auto device = instance.GetDevice();
    auto pipeline = Pipeline(device, directory / (std::string(shader) + ".metallib"));
    auto target = Target(device);
    id<MTLTexture> snapshot = copy ? Target(device) : nil;
    Pica::Shader::Generator::FSUniformData uniforms{};
    uniforms.depth_scale = -1.0f;
    uniforms.const_color[0] = {15.0f / 255, 170.0f / 255, 102.0f / 255, 1};
    auto buffer = [device newBufferWithBytes:&uniforms
                                      length:sizeof(uniforms)
                                     options:MTLResourceStorageModeShared];
    buffer = CheckResource(buffer, "Allocate fragment uniforms");
    auto descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = FragmentBindings::Uniforms + 1;
    descriptor.maxTextureBindCount = copy ? FragmentBindings::FeedbackTexture + 1 : 0;
    NSError* error = nil;
    auto table = [device newArgumentTableWithDescriptor:descriptor error:&error];
    table = CheckResource(table, "Allocate fixture argument table", error);
    [table setAddress:buffer.gpuAddress atIndex:FragmentBindings::Uniforms];
    if (copy) {
        [table setTexture:snapshot.gpuResourceID atIndex:FragmentBindings::FeedbackTexture];
    }
    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    RenderPass pass{
        .color = {.target = {target}, .load_action = MTLLoadActionClear},
        .area = {0, 0, Side, Side},
        .width = Side,
        .height = Side,
        .clear_color = MTLClearColorMake(85.0 / 255, 51.0 / 255, 204.0 / 255, 1),
        .tile_feedback = ordered,
    };
    manager.BeginRendering(pass);
    if (copy) {
        manager.CopyForFeedback({target}, {snapshot}, pass.area);
        pass.color.load_action = MTLLoadActionLoad;
        manager.BeginRendering(pass);
    }
    manager.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
        ctx.Retain(table);
        ctx.Retain(pipeline);
        ctx.MakeResident(buffer);
        if (copy) {
            ctx.MakeResident(snapshot);
        }
        [encoder setRenderPipelineState:pipeline];
        [encoder setViewport:MTLViewport{0, 0, Side, Side, 0, 1}];
        [encoder setArgumentTable:table atStages:MTLRenderStageFragment];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    });
    assert(scheduler.Finish({}, deadline) == CompletionStatus::Complete);
    std::array<u8, Side * Side * 4> actual{};
    [target getBytes:actual.data()
         bytesPerRow:Side * 4
          fromRegion:MTLRegionMake2D(0, 0, Side, Side)
         mipmapLevel:0];
    for (u32 pixel = 0; pixel < Side * Side; ++pixel) {
        assert(std::memcmp(actual.data() + pixel * 4, expected.data(), 4) == 0);
    }
    assert(scheduler.Shutdown(deadline));
}

void Run(const std::filesystem::path& directory) {
    if (!MTLCreateSystemDefaultDevice()) {
        std::puts("PENDING: Apple GPU unavailable; generated constant/discard and byte XOR on "
                  "ordered and copied attachments require native hardware.");
        return;
    }
    Instance instance(0, false);
    CheckPixels(instance, directory, "constant", false, false, {15, 170, 102, 255});
    CheckPixels(instance, directory, "discard", false, false, {85, 51, 204, 255});
    const std::array<u8, 4> xor_color{15 ^ 85, 170 ^ 51, 102 ^ 204, 255 ^ 255};
    CheckPixels(instance, directory, "xor_copy", true, false, xor_color);
    if (instance.GetCapabilities().raster_order_groups) {
        CheckPixels(instance, directory, "xor_ordered", false, true, xor_color);
    } else {
        std::puts("PENDING: native raster order groups require a supporting Metal 4 GPU.");
    }
    std::puts("PASS: available fragment pixel oracles with Metal API validation enabled.");
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
