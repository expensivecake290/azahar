#define main VerifyPhase058Standalone
#include "../058/verify.mm"
#undef main

#include <filesystem>
#include <nihstro/shader_bytecode.h>
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"

@interface CacheFixtureArchitecture : NSObject
@property(nonatomic, readonly) NSString* name;
@end

@implementation CacheFixtureArchitecture

- (NSString*)name {
    return @"phase-062-fixture";
}
@end

@interface CacheFixtureSerializer : NSObject
@end

@implementation CacheFixtureSerializer

- (BOOL)serializeAsArchiveAndFlushToURL:(NSURL*)url error:(NSError**)error {
    constexpr std::array<u8, 8> identity{'f', 'i', 'x', 't', 'u', 'r', 'e', 0};
    NSData* data = [NSData dataWithBytes:identity.data() length:identity.size()];
    return [data writeToURL:url options:0 error:error];
}
@end

@interface CacheFixtureDevice : PipelineFixtureDevice
@property(nonatomic) std::size_t captureCompilers;
@property(nonatomic) std::size_t liveOnlyCompilers;
@end

@implementation CacheFixtureDevice

- (MTLArchitecture*)architecture {
    return (MTLArchitecture*)[CacheFixtureArchitecture new];
}

- (id<MTL4PipelineDataSetSerializer>)newPipelineDataSetSerializerWithDescriptor:
    (MTL4PipelineDataSetSerializerDescriptor*)descriptor {
    assert(descriptor.configuration == MTL4PipelineDataSetSerializerConfigurationCaptureBinaries);
    return (id<MTL4PipelineDataSetSerializer>)[CacheFixtureSerializer new];
}

- (id<MTL4Compiler>)newCompilerWithDescriptor:(MTL4CompilerDescriptor*)descriptor
                                        error:(NSError**)error {
    if (descriptor.pipelineDataSetSerializer) {
        self.captureCompilers += 1;
    } else {
        self.liveOnlyCompilers += 1;
    }
    return [super newCompilerWithDescriptor:descriptor error:error];
}

- (id<MTL4Archive>)newArchiveWithURL:(NSURL*)url error:(NSError**)error {
    auto data = [NSData dataWithContentsOfURL:url];
    if (data.length != 8) {
        return nil;
    }
    auto archive = [PipelineFixtureArchive new];
    archive.device = (id<MTLDevice>)self;
    archive.hit = true;
    return (id<MTL4Archive>)archive;
}

@end

namespace {

namespace FS = std::filesystem;
namespace M = Pica::Shader::Generator::MSL;
using RStatus = ShaderCacheReadStatus;
using WStatus = ShaderCacheWriteStatus;

class CacheDirectory {
public:
    explicit CacheDirectory(const FS::path& output) {
        auto pattern = (output / "pipeline-cache-fixture-XXXXXXXX").string();
        assert(mkdtemp(pattern.data()));
        path = std::move(pattern);
    }
    ~CacheDirectory() {
        FS::remove_all(path);
    }

    FS::path path;
};

void SetBlock(PipelineFixtureCompiler* native) {
    std::lock_guard lock(native->mutex);
    native->blockRender = true;
}

void VerifyLifecycle(const FS::path& output) {
    CacheDirectory directory(output);
    auto holder = [CacheFixtureDevice new];
    auto native = holder.compiler;
    DeviceContext context((id<MTLDevice>)holder);
    PipelineCache cache(context, {true, true, true, true}, {directory.path, true, true, 1, 1, 1});
    const PipelineTitle first{200, ""};
    const PipelineTitle applet{0x00040030000000c9, ""};
    std::atomic_bool stop{};
    const ShaderSource source{"host-only SDK controlled shader input", "cache", "", {}};
    auto key = MakeKey(source, PipelineRoute::Vertex);

    std::vector<VideoCore::LoadCallbackStage> events;
    auto progress = [&](auto stage, auto current, auto total, const auto&) {
        assert(current <= total);
        events.push_back(stage);
    };
    auto load = cache.SwitchTitle(first, stop, progress);
    assert(load.disk_status == RStatus::Missing && load.completed == 0);
    assert((events == std::vector{VideoCore::LoadCallbackStage::Prepare,
                                  VideoCore::LoadCallbackStage::Complete}));
    assert(cache.CurrentTitle() == first && holder.captureCompilers == 1);

    SetBlock(native);
    auto original = std::async(std::launch::async, [&] { return cache.Require(key, true); });
    [native waitForAsyncRender];
    auto duplicate = std::async(std::launch::async, [&] { return cache.Require(key, true); });
    auto alternative = key;
    alternative.blend.write_mask = 1;
    const auto fallback = cache.Require(alternative, true);
    assert(fallback.GetStates(context) && native->renderSync == 1 && native->renderAsync == 1);
    assert(duplicate.wait_for(10ms) == std::future_status::timeout);
    [native unblockRender];
    const auto complete = original.get();
    const auto same = duplicate.get();
    assert(complete.GetStates(context)->render == same.GetStates(context)->render);
    assert(cache.Require(key, true).GetStates(context)->render ==
           complete.GetStates(context)->render);
    assert(native->renderAsync == 1);

    auto transient = key;
    transient.blend.write_mask = 7;
    auto ephemeral = cache.Require(transient, false);
    assert(ephemeral.GetStates(context) && holder.liveOnlyCompilers == 1);
    auto broken = key;
    broken.shaders.fragment->entry = "bad_native_entry";
    assert(cache.Require(broken, true).GetStatus(context) == PipelineBuildStatus::ShaderFailed);

    assert(cache.Save() == WStatus::Saved);
    assert(cache.Save() == WStatus::Missing);
    const auto scope =
        MakeShaderCacheIdentity(context, first.program_id, "", "metal4/truetruetruetrue");
    ShaderDiskCache files(directory.path, scope, true);
    const auto cached = files.Load(stop);
    assert(cached.status == RStatus::Ready && cached.records.size() == 2);
    assert(std::ranges::find(cached.records, ShaderCacheRecord{transient}) == cached.records.end());

    auto waiting = key;
    waiting.blend.write_mask = 10;
    SetBlock(native);
    auto old = std::async(std::launch::async, [&] { return cache.Require(waiting, true); });
    [native waitForAsyncRender];
    auto changing = std::async(std::launch::async, [&] { return cache.SwitchTitle(applet, stop); });
    assert(old.wait_for(10s) == std::future_status::ready);
    assert(old.get().GetStatus(context) == PipelineBuildStatus::Canceled);
    assert(changing.wait_for(20ms) == std::future_status::timeout);
    assert(complete.GetStatus(context) == PipelineBuildStatus::Canceled &&
           ephemeral.GetStatus(context) == PipelineBuildStatus::Canceled);
    [native unblockRender];
    assert(changing.get().disk_status == RStatus::Missing && cache.CurrentTitle().IsApplet());

    events.clear();
    const auto canceled =
        cache.SwitchTitle(first, stop, [&](auto stage, auto done, auto total, const auto& label) {
            progress(stage, done, total, label);
            if (stage == VideoCore::LoadCallbackStage::Preload && done == 1) {
                stop = true;
            }
        });
    assert(canceled.canceled && canceled.disk_status == RStatus::Canceled &&
           canceled.completed == 0);
    assert(events.back() == VideoCore::LoadCallbackStage::Preload);
    stop = false;
    std::vector<std::size_t> completed;
    load = cache.SwitchTitle(first, stop, [&](auto stage, auto done, auto total, const auto&) {
        if (stage == VideoCore::LoadCallbackStage::Build) {
            assert(total == 2);
            completed.push_back(done);
        }
    });
    assert(load.disk_status == RStatus::Ready && load.completed == 2 && load.failed == 0);
    assert((completed == std::vector<std::size_t>{0, 1, 2}));
    const auto prior = cache.Require(key, true);
    assert(prior.GetStates(context) && context.AdvanceGeneration());
    assert(prior.GetStatus(context) == PipelineBuildStatus::Canceled);
    assert(cache.Require(key, true).GetStates(context));

    static_cast<void>(cache.SwitchTitle(applet, stop));
    for (const auto& entry : FS::directory_iterator(directory.path / "warmup")) {
        if (entry.path().filename().string().starts_with("00000000000000c8-")) {
            std::ofstream marker(entry.path(), std::ios::binary);
            marker.put(1);
        }
    }
    assert(cache.PreviousLoadInterrupted(first));
    const auto recovery = cache.SwitchTitle(first, stop);
    assert(recovery.previous_load_interrupted && recovery.completed == 0 &&
           recovery.disk_status == RStatus::Recovered);
    assert(!cache.PreviousLoadInterrupted(first));
    assert(cache.Require(key, true).GetStates(context));
    cache.Close();
    assert(cache.Require(key, true).GetStatus(context) == PipelineBuildStatus::Canceled);
    cache.Close();

    PipelineCache sync(context, {}, {{}, false, false, 1, 1, 1});
    const auto sync_before = native->renderSync.load();
    assert(sync.Require(key, false).GetStates(context));
    assert(native->renderSync == sync_before + 1);
    assert(sync.Save() == WStatus::Disabled);
    std::puts(
        "PASS SDK callback fixture for duplicate required draws, queue-full independent sync, "
        "title retirement, stale/device generations, noncacheable shader isolation, warm "
        "counts, retryable cancellation and interrupted warmup cold recovery");
}

void Emit(const FS::path& output, const GraphicsPipelineKey& key, std::string_view label) {
    const std::array functions{&key.shaders.vertex, &key.shaders.fragment,  &key.shaders.object,
                               &key.shaders.mesh,   &key.shaders.transform, &key.shaders.expand};
    constexpr std::array names{"vertex", "fragment", "object", "mesh", "transform", "expand"};
    for (std::size_t i = 0; i < functions.size(); ++i) {
        if (!*functions[i]) {
            continue;
        }
        const auto& function = **functions[i];
        const auto& source = std::get<ShaderSource>(function.artifact).text;
        assert(source.find(function.entry) != std::string::npos);
        auto path = output / (std::string(label) + "-" + names[i] + ".metal");
        std::ofstream stream(path);
        stream << source;
        assert(stream.good());
    }
}

void VerifySelection(const FS::path& output) {
    auto holder = [CacheFixtureDevice new];
    DeviceContext context((id<MTLDevice>)holder);
    PipelineCache renderer(context, {false, true, true, false}, {});
    Pica::RegsInternal regs{};
    regs.lighting.disable.Assign(1);
    regs.framebuffer.output_merger.alpha_test.enable.Assign(0);
    regs.framebuffer.output_merger.logic_op.Assign(F::LogicOp::Copy);
    regs.vs.output_mask.Assign(1);
    regs.rasterizer.vs_output_total.Assign(1);
    using Semantic = Pica::RasterizerRegs::VSOutputAttributes::Semantic;
    auto& position = regs.rasterizer.vs_output_attributes[0];
    position.map_x.Assign(Semantic::POSITION_X);
    position.map_y.Assign(Semantic::POSITION_Y);
    position.map_z.Assign(Semantic::POSITION_Z);
    position.map_w.Assign(Semantic::POSITION_W);
    Pica::Shader::UserConfig user{};
    GraphicsPipelineKey fixed;
    fixed.attachments.color = MTLPixelFormatRGBA8Unorm;
    fixed.vertex_layout.bindings = {{4, 16, 1, MTLVertexStepFunctionPerVertex}};
    fixed.vertex_layout.attributes = {
        {0, 4, 0, MTLVertexFormatFloat4, P::VertexAttributeFormat::FLOAT, 4, false}};
    Pica::ShaderSetup vs;
    nihstro::Instruction end{};
    end.opcode.Assign(nihstro::OpCode::Id::END);
    vs.UpdateProgramCode(0, end.hex);
    PipelineShaderRequest request{
        regs, user, fixed, VideoCore::PixelFormat::RGBA8, &vs, nullptr, {}, 3, false, false};
    const auto direct = renderer.Select(request);
    assert(direct && direct.cacheable && direct.key.route == PipelineRoute::Vertex);
    Emit(output, direct.key, "programmable");
    request.fixed_state.raster.guest_topology = P::TriangleTopology::Fan;
    const auto mesh = renderer.Select(request);
    assert(mesh && mesh.key.route == PipelineRoute::Mesh && mesh.key.mesh.payload_bytes == 288);
    Emit(output, mesh.key, "object");
    request.vertex_count = 6;
    const auto ordered = renderer.Select(request);
    assert(ordered && ordered.key.route == PipelineRoute::ComputeExpansion);
    Emit(output, ordered.key, "ordered");
    request.fixed_state.raster.guest_topology = P::TriangleTopology::Strip;
    request.fixed_state.raster.index_type = PipelineIndexType::UInt16;
    const auto strip = renderer.Select(request);
    assert(strip && strip.key.route == PipelineRoute::ComputeExpansion &&
           strip.key.raster.index_type == PipelineIndexType::None);
    Emit(output, strip.key, "indexed-strip");
    regs.framebuffer.output_merger.logic_op.Assign(F::LogicOp::Xor);
    request.fixed_state.raster.guest_topology = P::TriangleTopology::List;
    request.fixed_state.raster.index_type = PipelineIndexType::None;
    request.fixed_state.attachments.samples = 4;
    const auto feedback = renderer.Select(request);
    assert(feedback && feedback.fragment.feedback == M::FragmentFeedback::TextureCopy &&
           feedback.key.route == PipelineRoute::ComputeExpansion);
    Emit(output, feedback.key, "feedback");
    regs.framebuffer.output_merger.logic_op.Assign(F::LogicOp::Copy);
    regs.framebuffer.output_merger.alphablend_enable.Assign(1);
    regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb.Assign(F::BlendEquation::Min);
    regs.framebuffer.output_merger.alpha_blending.factor_source_rgb.Assign(
        F::BlendFactor::ConstantColor);
    regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb.Assign(F::BlendFactor::One);
    user.use_custom_normal.Assign(1);
    const auto minimum = renderer.Select(request);
    assert(minimum && !minimum.cacheable && minimum.key.blend.rgb.shader_emulation &&
           !minimum.key.blend.alpha.shader_emulation);
    Emit(output, minimum.key, "minimum");
    regs.vs.main_offset.Assign(Pica::MAX_PROGRAM_CODE_LENGTH);
    const auto bad = renderer.Select(request);
    assert(!bad && bad.status == PipelineSelectionStatus::InvalidVertexProgram);
    request.vertex_setup = nullptr;
    request.geometry_setup = &vs;
    assert(renderer.Select(request).status == PipelineSelectionStatus::InvalidGeometryProgram);
    request.geometry_setup = nullptr;
    request.fixed_state.attachments.samples = 3;
    assert(renderer.Select(request).status == PipelineSelectionStatus::InvalidFragmentState);
    std::puts("PASS real PICA selection of vertex, mesh, ordered compute, indexed strip, feedback "
              "and independent blend paths with compiler corpus and typed failures");
}

void VerifyNativeCache(const FS::path& output, const char* shader_file) {
    CacheDirectory directory(output);
    std::ifstream input(shader_file);
    assert(input);
    const ShaderSource source{std::string{std::istreambuf_iterator<char>{input}, {}},
                              "graphics-cache native fixture",
                              "",
                              {}};
    Instance instance(0, false);
    auto& context = instance.GetContext();
    auto device = instance.GetDevice();
    const PipelineTitle title{0x00040000000007a1, ""};
    const auto profile = PipelineDeviceProfile::Detect(context);
    const auto key = MakeKey(source, PipelineRoute::Vertex);
    const std::atomic_bool stop{};
    for (bool asynchronous : {false, true}) {
        PipelineCache cache(context, profile, {directory.path, true, asynchronous, 2, 2, 8});
        const auto warmup = cache.SwitchTitle(title, stop);
        assert(warmup.completed == (asynchronous ? 1u : 0u) && warmup.failed == 0);
        const auto pipeline = cache.Require(key, true);
        assert(pipeline.GetStates(context));

        auto color = Texture(device, key.attachments.color, MTLStorageModeShared, 1);
        auto depth = Texture(device, key.attachments.depth, MTLStorageModePrivate, 1);
        Scheduler scheduler(context);
        MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
        RenderManager manager(instance, scheduler, allocator);
        RenderPass pass;
        pass.color = {.target = {color}, .load_action = MTLLoadActionClear};
        pass.depth = {.target = {depth}, .load_action = MTLLoadActionClear};
        pass.stencil = pass.depth;
        pass.area = {0, 0, Size, Size};
        pass.width = pass.height = Size;
        manager.BeginRendering(pass);
        manager.RecordDraw([&](EncodingContext& encoding, id<MTL4RenderCommandEncoder> encoder) {
            PipelineCache::Bind(encoding, context, encoder, pipeline,
                                {{0, 0, Size, Size, 0, 1}, {0, 0, Size, Size}, 0, 0}, {});
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        });
        manager.EndRendering();
        assert(scheduler.Finish({}, Clock::now() + 10s) == CompletionStatus::Complete);
        std::array<u8, Size * Size * 4> pixels;
        [color getBytes:pixels.data()
            bytesPerRow:Size * 4
             fromRegion:MTLRegionMake2D(0, 0, Size, Size)
            mipmapLevel:0];
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            assert(std::abs(int(pixels[i]) - 191) <= 1);
            assert(std::abs(int(pixels[i + 1]) - 64) <= 1);
            assert(std::abs(int(pixels[i + 2]) - 128) <= 1);
            assert(std::abs(int(pixels[i + 3]) - 128) <= 1);
        }
        const auto save = cache.Save();
        assert(save == WStatus::Saved || (asynchronous && save == WStatus::Missing));
    }
    std::puts(
        "PASS live native synchronous and asynchronous title pipeline plus cached draw readback");
}

} // Anonymous namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        assert(argc == 3);
        VerifyLifecycle(argv[1]);
        VerifySelection(argv[1]);
        const auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING live native Metal 4 title archive, cold and warm cache draws, "
                      "and retained native argument and pipeline lifetimes");
            return 2;
        }
        VerifyNativeCache(argv[1], argv[2]);
    }
}
