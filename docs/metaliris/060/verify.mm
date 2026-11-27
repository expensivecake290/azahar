#include <array>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <set>
#include <unistd.h>
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_shader_disk_cache.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

namespace FS = std::filesystem;
using R = ShaderCacheReadStatus;
using W = ShaderCacheWriteStatus;
using Clock = std::chrono::steady_clock;

class TestDirectory {
public:
    explicit TestDirectory(const FS::path& parent) {
        auto name = (parent / "cache-fixture-XXXXXXXX").string();
        assert(mkdtemp(name.data()));
        path = std::move(name);
    }

    ~TestDirectory() {
        FS::remove_all(path);
    }

    FS::path path;
};

std::vector<u8> Read(const FS::path& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input);
    return {std::istreambuf_iterator<char>{input}, {}};
}

void Write(const FS::path& path, std::span<const u8> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output.close();
    assert(output);
}

std::vector<FS::path> Files(const FS::path& directory, std::string_view type) {
    std::vector<FS::path> output;
    for (const auto& entry : FS::recursive_directory_iterator(directory)) {
        if (entry.path().extension() == type) {
            output.emplace_back(entry.path());
        }
    }
    return output;
}

ShaderCacheRecord Shader(CachedShaderStage stage, std::span<const u8> binary,
                         bool use_native_data) {
    if (use_native_data) {
        return CachedShaderFunction{
            stage,
            PipelineFunction{std::vector<u8>{binary.begin(), binary.end()}, "cache_fixture"}};
    }
    ShaderSource source{
        std::string(4800, ' ') + "kernel void cache_fixture() {}",
        "native fixture",
        "#define PHASE 60",
        {ShaderMath::Fast, ShaderOptimization::Size, false, {{"SAMPLES", "4"}, {"VALUE", "159"}}}};
    return CachedShaderFunction{stage, PipelineFunction{source, "cache_fixture"}};
}

GraphicsPipelineKey FullKey(const PipelineFunction& function) {
    using FB = Pica::FramebufferRegs;
    using P = Pica::PipelineRegs;
    using RS = Pica::RasterizerRegs;
    namespace MSL = Pica::Shader::Generator::MSL;

    GraphicsPipelineKey result;
    result.route = PipelineRoute::Vertex;
    result.shaders.vertex = function;
    result.shaders.fragment = function;
    result.vertex_layout.bindings = {{2, 4, 1, MTLVertexStepFunctionPerVertex},
                                     {4, 16, 0, MTLVertexStepFunctionConstant}};
    result.vertex_layout.attributes = {
        {0, 2, 0, MTLVertexFormatUChar4, P::VertexAttributeFormat::UBYTE, 4, true},
        {3, 4, 4, MTLVertexFormatFloat3, P::VertexAttributeFormat::FLOAT, 3, false}};
    result.attachments = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth32Float_Stencil8,
                          MTLPixelFormatDepth32Float_Stencil8, 4};
    result.raster = {P::TriangleTopology::List,
                     RS::CullMode::KeepCounterClockWise,
                     MTLPrimitiveTypeTriangle,
                     PipelineIndexType::UInt32,
                     true,
                     false};
    result.blend = {{FB::BlendFactor::ConstantColor, FB::BlendFactor::OneMinusSourceAlpha,
                     FB::BlendEquation::Min, true},
                    {FB::BlendFactor::SourceAlpha, FB::BlendFactor::DestAlpha,
                     FB::BlendEquation::ReverseSubtract, false},
                    FB::LogicOp::Xor,
                    7,
                    true};
    result.depth_stencil = {FB::CompareFunc::Equal,
                            FB::CompareFunc::GreaterThan,
                            FB::StencilAction::Invert,
                            FB::StencilAction::IncrementWrap,
                            FB::StencilAction::Replace,
                            0xc3,
                            0xf3,
                            true,
                            true,
                            true};
    result.fragment = {MSL::FragmentFeedback::TextureCopy, MSL::ShadowStorage::AtomicTexture, true,
                       true, true};
    result.mesh = {2, 3, 4, 197};
    return result;
}

void VerifyScope(const FS::path& root, std::span<const u8> binary) {
    const ShaderCacheIdentity identity{0x7f,          "fixture-content", "fixture-version",
                                       "fixture-gpu", "fixture-build",   "fixture-profile"};
    const auto first = Shader(CachedShaderStage::Vertex, binary, false);
    const auto other_stage = Shader(CachedShaderStage::Mesh, binary, true);
    const auto function = std::get<CachedShaderFunction>(first).function;
    const auto key = FullKey(function);
    std::atomic_bool stop{};

    ShaderDiskCache disabled(root / "disabled", identity, false);
    assert(!disabled.IsEnabled() && disabled.Load(stop).status == R::Disabled &&
           disabled.Store(first) == W::Disabled && disabled.Remove(first) == W::Disabled &&
           !FS::exists(root / "disabled"));
    ShaderDiskCache default_title(root / "unnamed", ShaderCacheIdentity{}, true);
    assert(!default_title.IsEnabled() && default_title.Store(first) == W::Disabled);
    ShaderDiskCache cache(root, identity, true);
    assert(cache.IsEnabled() && cache.Identity() == identity &&
           cache.Load(stop).status == R::Missing);
    assert(cache.Remove(first) == W::Missing);
    assert(cache.Store(first) == W::Saved && cache.Store(first) == W::Duplicate);
    assert(cache.Store(other_stage) == W::Saved && cache.Store(key) == W::Saved);

    auto expanded = key;
    expanded.route = PipelineRoute::ComputeExpansion;
    expanded.vertex_layout = {};
    expanded.shaders.transform = function;
    expanded.shaders.expand = function;
    assert(cache.Store(expanded) == W::Saved);
    auto mesh = key;
    mesh.route = PipelineRoute::Mesh;
    mesh.vertex_layout = {};
    mesh.shaders.vertex.reset();
    mesh.shaders.object = function;
    mesh.shaders.mesh = function;
    mesh.shaders.transform = function;
    assert(cache.Store(mesh) == W::Saved);

    auto invalid = key;
    invalid.shaders.vertex.reset();
    assert(cache.Store(invalid) == W::InvalidRecord);
    invalid = key;
    invalid.shaders.vertex->entry = "bad id";
    assert(cache.Store(invalid) == W::InvalidRecord);
    const auto bogus = Shader(CachedShaderStage::Object, std::array<u8, 3>{1, 2, 3}, true);
    assert(cache.Store(bogus) == W::InvalidRecord);

    const std::set<ShaderCacheRecord> expected{first, other_stage, key, expanded, mesh};
    const auto assert_snapshot = [&](const ShaderCacheContents& contents, R status,
                                     std::size_t rejects) {
        assert(contents.status == status && contents.rejected_records == rejects);
        assert((std::set<ShaderCacheRecord>{contents.records.begin(), contents.records.end()} ==
                expected));
    };
    std::size_t callbacks{};
    const auto contents =
        cache.Load(stop, [&](auto stage, auto current, auto total, const auto& label) {
            assert(stage == VideoCore::LoadCallbackStage::Preload && total == 5 &&
                   current == callbacks && !label.empty());
            ++callbacks;
        });
    assert(callbacks == 6);
    assert_snapshot(contents, R::Ready, 0);

    callbacks = 0;
    const auto canceled = cache.Load(stop, [&](auto, auto count, auto, const auto&) {
        ++callbacks;
        if (count == 2) {
            stop = true;
        }
    });
    assert(canceled.status == R::Canceled && canceled.records.empty() && callbacks == 3);
    assert(cache.Load(stop).status == R::Canceled);
    stop = false;

    for (const auto field : {&ShaderCacheIdentity::content_id, &ShaderCacheIdentity::source_version,
                             &ShaderCacheIdentity::device_architecture,
                             &ShaderCacheIdentity::system_build, &ShaderCacheIdentity::profile}) {
        auto mismatch = identity;
        (mismatch.*field) += "_different";
        ShaderDiskCache separate(root, mismatch, true);
        assert(separate.Load(stop).status == R::Missing);
    }
    auto mismatch = identity;
    ++mismatch.title_id;
    ShaderDiskCache title(root, mismatch, true);
    assert(title.Load(stop).status == R::Missing && title.Store(first) == W::Saved);
    assert(title.Load(stop).records == std::vector{first});
    assert_snapshot(cache.Load(stop), R::Ready, 0);

    const auto original_path = Files(root / "000000000000007f", ".mtlrecord").front();
    const auto bytes = Read(original_path);
    assert(bytes.size() > 9 &&
           std::ranges::equal(std::span(bytes).first(8), ShaderDiskCache::Magic));
    const auto shard = original_path.parent_path();
    const auto orphan = shard / ".staging-interrupted";
    FS::create_directory(orphan);
    Write(orphan / "record", std::span(bytes).first(7));
    assert_snapshot(cache.Load(stop), R::Ready, 0);

    auto corrupt = bytes;
    corrupt.back() ^= 0xa0;
    Write(shard / "bad_checksum.mtlrecord", corrupt);
    Write(shard / "truncated.mtlrecord", std::span(bytes).first(5));
    corrupt = bytes;
    corrupt[8] += 1;
    Write(shard / "future_version.mtlrecord", corrupt);
    const auto foreign = Files(root / "0000000000000080", ".mtlrecord").front();
    FS::copy_file(foreign, shard / "foreign_identity.mtlrecord");
    assert_snapshot(cache.Load(stop), R::Recovered, 4);

    Write(original_path, std::span(bytes).first(16));
    assert(cache.Load(stop).rejected_records == 5);
    std::size_t recovered{};
    for (const auto& record : expected) {
        const auto stored = cache.Store(record);
        assert(stored == W::Saved || stored == W::Duplicate);
        recovered += stored == W::Saved;
    }
    assert(recovered == 1);
    assert_snapshot(cache.Load(stop), R::Recovered, 4);
    assert(cache.Remove(other_stage) == W::Removed && cache.Remove(other_stage) == W::Missing);

    const auto exclusive = root / "foreign-only";
    ShaderDiskCache foreign_only(exclusive, identity, true);
    assert(foreign_only.Store(first) == W::Saved);
    const auto same_path = Files(exclusive, ".mtlrecord").front();
    Write(same_path, Read(foreign));
    assert(foreign_only.Load(stop).status == R::Incompatible);
    assert(foreign_only.Store(first) == W::Saved);
    assert(foreign_only.Load(stop).records == std::vector{first});

    const auto blocked = root / "is_file";
    Write(blocked, std::array<u8, 1>{'x'});
    ShaderDiskCache blocked_cache(blocked, identity, true);
    assert(blocked_cache.Store(first) == W::IOError);
    assert(blocked_cache.Load(stop).status == R::IOError);
    std::puts("PASS exact native key round trips, version/scope mismatch, recovery, checksum, "
              "truncated writes, disabled/unwritable paths and exact completion count");
}

void VerifyConcurrent(const FS::path& root, std::span<const u8> binary) {
    const ShaderCacheIdentity identity{91, "", "concurrent", "gpu", "build", ""};
    const auto duplicate = Shader(CachedShaderStage::ComputeTransform, binary, true);
    std::barrier start(8);
    std::vector<std::future<std::pair<W, W>>> tasks;
    for (u32 i = 0; i < 8; ++i) {
        tasks.push_back(std::async(std::launch::async, [&, i] {
            ShaderDiskCache cache(root, identity, true);
            auto unique = std::get<CachedShaderFunction>(
                Shader(CachedShaderStage::ComputeExpansion, binary, false));
            unique.function.entry += std::to_string(i);
            start.arrive_and_wait();
            const auto result = cache.Store(duplicate);
            return std::pair{result, cache.Store(unique)};
        }));
    }
    unsigned written{};
    for (auto& task : tasks) {
        const auto [shared, distinct] = task.get();
        assert((shared == W::Saved || shared == W::Duplicate) && distinct == W::Saved);
        written += shared == W::Saved;
    }
    assert(written == 1);

    ShaderDiskCache cache(root, identity, true);
    const std::atomic_bool stop{};
    const auto snapshot = cache.Load(stop);
    assert(snapshot.status == R::Ready && snapshot.records.size() == 9);
    assert(Files(root, ".mtlrecord").size() == 9);
    std::puts(
        "PASS eight independent writer handles with simultaneous duplicate and unique commits");
}

void Dispatch(DeviceContext& context, id<MTLComputePipelineState> pipeline) {
    assert(pipeline);
    auto device = context.GetDevice();
    const u32 zero{};
    auto buffer = CheckResource([device newBufferWithBytes:&zero
                                                    length:sizeof(zero)
                                                   options:MTLResourceStorageModeShared |
                                                           MTLResourceHazardTrackingModeUntracked],
                                "Allocate native archive check buffer");
    auto descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 1;
    NSError* error = nil;
    auto args = [device newArgumentTableWithDescriptor:descriptor error:&error];
    args = CheckResource(args, "Allocate native archive check arguments", error);
    [args setAddress:buffer.gpuAddress atIndex:0];
    Scheduler scheduler(context);
    scheduler.Record([=](EncodingContext& ctx) {
        const std::array uses{ResourceUse{buffer, BufferAccess::Write, MTLStageDispatch}};
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        ctx.Synchronize(encoder, uses);
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:args];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ctx.Retain(args);
        ctx.Retain(pipeline);
        ctx.EndPass(encoder);
    });
    assert(scheduler.Finish({}, Clock::now() + 10s) == CompletionStatus::Complete);
    assert(*static_cast<const u32*>(buffer.contents) == 0x4d495249u);
}

void VerifyGPU(id<MTLDevice> device, const FS::path& root, std::span<const u8> binary) {
    DeviceContext context(device);
    const auto identity = MakeShaderCacheIdentity(context, 661, "", "native");
    assert(identity.IsValid());
    ShaderDiskCache cache(root, identity, true);
    const std::atomic_bool stop{};
    assert(cache.LoadArchives(context, stop).status == R::Missing);

    auto serializer_desc = [MTL4PipelineDataSetSerializerDescriptor new];
    serializer_desc.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    auto serializer = [device newPipelineDataSetSerializerWithDescriptor:serializer_desc];
    assert(serializer);
    auto compiler_desc = [MTL4CompilerDescriptor new];
    compiler_desc.pipelineDataSetSerializer = serializer;
    NSError* error = nil;
    auto native = [device newCompilerWithDescriptor:compiler_desc error:&error];
    native = CheckResource(native, "Create native archive fixture compiler", error);
    ShaderCompiler libraries(context);
    const auto executable = libraries.LoadLibrary(binary, "phase 060");
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = executable.GetLibrary(context);
    function.name = @"cache_fixture";
    assert(function.library);
    auto desc = [MTL4ComputePipelineDescriptor new];
    desc.computeFunctionDescriptor = function;
    error = nil;
    auto first = [native newComputePipelineStateWithDescriptor:desc
                                           compilerTaskOptions:nil
                                                         error:&error];
    Dispatch(context, CheckResource(first, "Compile archived native pipeline", error));
    assert(cache.CommitArchive(context, serializer) == W::Saved);

    ShaderDiskCache reopened(root, identity, true);
    auto warm = reopened.LoadArchives(context, stop);
    assert(warm.status == R::Ready && warm.archives.count == 1);
    error = nil;
    auto restored = [warm.archives[0] newComputePipelineStateWithDescriptor:desc error:&error];
    Dispatch(context, CheckResource(restored, "Restore archived native pipeline", error));

    const auto archive_root = Files(root, ".mtlarchive");
    assert(archive_root.size() == 1);
    const auto archive = archive_root[0] / "pipelines.metallib";
    // Invalidate the name after releasing the loaded native archive's backing file.
    warm.archives = nil;
    FS::rename(archive, archive_root[0] / "previous.metallib");
    Write(archive, std::array<u8, 4>{'b', 'a', 'd', '\n'});
    const auto broken = reopened.LoadArchives(context, stop);
    assert(broken.status == R::Recovered && broken.archives.count == 0 &&
           broken.rejected_archives == 1);
    PipelineCompiler compiler(context, libraries);
    GraphicsPipelineKey recipe;
    recipe.route = PipelineRoute::Compute;
    recipe.shaders.transform =
        PipelineFunction{std::vector<u8>{binary.begin(), binary.end()}, "cache_fixture"};
    const auto rebuild = compiler.Build(recipe, broken.archives).GetStates(context);
    assert(rebuild);
    Dispatch(context, rebuild->transform);
    std::puts("PASS actual Metal 4 cold/warm pipeline bytes, native archive rejection and rebuild");
}

} // Anonymous namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        assert(argc == 3);
        TestDirectory corpus(argv[1]);
        const auto binary = Read(argv[2]);
        assert(HasMetalLibrarySignature(binary));
        VerifyScope(corpus.path / "scopes", binary);
        VerifyConcurrent(corpus.path / "concurrent", binary);
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts(
                "PENDING real Metal 4 pipeline archive commit, GPU cold/warm dispatch readback "
                "and invalid archive fallback");
            return 2;
        }
        VerifyGPU(device, corpus.path / "gpu", binary);
    }
}
