#include <cassert>
#include <cstdio>
#include <map>
#include <utility>
#include "video_core/renderer_metal/mtl_shader_disk_cache.h"

using namespace Metal;

static_assert(ShaderDiskCache::Magic[0] == 'M');
static_assert(ShaderDiskCache::FormatVersion == 1);
static_assert(ShaderDiskCache::MaxRecordSize >= 1024 * 1024);
static_assert(
    std::is_same_v<decltype(&ShaderDiskCache::Load),
                   ShaderCacheContents (ShaderDiskCache::*)(
                       const std::atomic_bool&, const VideoCore::DiskResourceLoadCallback&) const>);
static_assert(
    std::is_same_v<decltype(&ShaderDiskCache::Store),
                   ShaderCacheWriteStatus (ShaderDiskCache::*)(const ShaderCacheRecord&)>);
static_assert(std::is_same_v<decltype(&ShaderDiskCache::CommitArchive),
                             ShaderCacheWriteStatus (ShaderDiskCache::*)(
                                 const DeviceContext&, id<MTL4PipelineDataSetSerializer>)>);

void CompilerWitness(id<MTLDevice> device, NSURL* url) {
    NSError* error = nil;
    auto descriptor = [MTL4PipelineDataSetSerializerDescriptor new];
    descriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    auto serializer = [device newPipelineDataSetSerializerWithDescriptor:descriptor];
    auto compiler = [MTL4CompilerDescriptor new];
    compiler.pipelineDataSetSerializer = serializer;
    assert([serializer serializeAsArchiveAndFlushToURL:url error:&error]);
    id<MTL4Archive> archive = [device newArchiveWithURL:url error:&error];
    assert(archive && !error);
}

int main() {
    @autoreleasepool {
        ShaderCacheIdentity original{67, "", "fixture-version", "arch", "system-build", "profile"};
        assert(original.IsValid() && !ShaderCacheIdentity{}.IsValid());
        auto identity = original;
        identity.title_id = 0;
        assert(!identity.IsValid());
        identity.content_id = "fixture-homebrew-id";
        assert(identity.IsValid() && identity != original);
        for (const auto field :
             {&ShaderCacheIdentity::source_version, &ShaderCacheIdentity::device_architecture,
              &ShaderCacheIdentity::system_build}) {
            identity = original;
            (identity.*field).clear();
            assert(!identity.IsValid() && identity != original);
        }
        identity = original;
        identity.profile += "A";
        assert(identity.IsValid() && identity != original);

        std::map<ShaderCacheRecord, std::size_t> records;
        const PipelineFunction function{ShaderSource{"kernel native", "", "", {}}, "kernel_name"};
        for (const auto stage :
             {CachedShaderStage::Vertex, CachedShaderStage::Fragment, CachedShaderStage::Object,
              CachedShaderStage::Mesh, CachedShaderStage::ComputeTransform,
              CachedShaderStage::ComputeExpansion}) {
            assert(records.emplace(CachedShaderFunction{stage, function}, records.size()).second);
        }
        auto variant = function;
        std::get<ShaderSource>(variant.artifact).preamble = "#define PRECISE";
        assert(
            records
                .emplace(CachedShaderFunction{CachedShaderStage::Vertex, variant}, records.size())
                .second);
        variant.artifact = std::vector<u8>{'M', 'T', 'L', 'B', 1};
        assert(
            records
                .emplace(CachedShaderFunction{CachedShaderStage::Vertex, variant}, records.size())
                .second);
        GraphicsPipelineKey key;
        key.shaders.vertex = function;
        assert(records.emplace(key, records.size()).second);
        key.depth_stencil.depth_write = true;
        assert(records.emplace(key, records.size()).second);
        assert(records.size() == 10);

        VideoCore::LoadCallbackStage last = VideoCore::LoadCallbackStage::Prepare;
        VideoCore::DiskResourceLoadCallback callback = [&](VideoCore::LoadCallbackStage stage,
                                                           std::size_t complete, std::size_t total,
                                                           const std::string&) {
            assert(complete == total);
            last = stage;
        };
        callback(VideoCore::LoadCallbackStage::Preload, records.size(), records.size(), "fixture");
        assert(last == VideoCore::LoadCallbackStage::Preload);
        assert(ShaderCacheContents{}.status == ShaderCacheReadStatus::Missing &&
               ShaderCacheArchives{}.archives == nil);
        std::puts("PASS exact title/build/architecture/profile identity, cache record distinction, "
                  "progress ABI and native MTL4 archive declarations");
    }
}
