// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_pipeline_cache.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fmt/format.h>

#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mt_instance.h"

namespace Metal {

struct PipelineCache::Impl {
    id<MTLLibrary> present_library = nil;
    id<MTLRenderPipelineState> present_pipeline = nil;
    bool valid = false;
};

namespace {

constexpr std::string_view PRESENT_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

struct PresentVertexOut {
    float4 position [[position]];
};

vertex PresentVertexOut azahar_present_vs(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };

    PresentVertexOut out;
    out.position = float4(positions[vertex_id], 0.0, 1.0);
    return out;
}

fragment float4 azahar_present_fs() {
    return float4(0.0, 0.0, 0.0, 1.0);
}
)";

static std::string GetMetalShaderDir() {
    return FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + "metal" + DIR_SEP;
}

static std::string GetCacheMetadataPath() {
    return GetMetalShaderDir() + "present.cache";
}

static bool EnsureDirectories() {
    return FileUtil::CreateDir(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir)) &&
           FileUtil::CreateDir(GetMetalShaderDir());
}

static void SaveCacheMetadata(const Instance& instance) {
    if (!EnsureDirectories()) {
        return;
    }

    const u64 shader_hash =
        Common::ComputeHash64<Common::HashAlgo64::CityHash>(PRESENT_SHADER.data(),
                                                            PRESENT_SHADER.size());
    const std::string metadata =
        fmt::format("device={}\nshader_hash={:016x}\n", instance.GetDeviceName(), shader_hash);
    FileUtil::WriteStringToFile(true, GetCacheMetadataPath(), metadata);
}

} // Anonymous namespace

PipelineCache::PipelineCache(const Instance& instance) : impl{} {
    impl = std::make_unique<Impl>();

    if (!instance.IsValid()) {
        return;
    }

    id<MTL4Compiler> compiler = (__bridge id<MTL4Compiler>)instance.GetCompiler();
    if (compiler == nil) {
        return;
    }

    NSError* error = nil;

    MTL4LibraryDescriptor* library_descriptor = [[MTL4LibraryDescriptor alloc] init];
    library_descriptor.name = @"AzaharPresentLibrary";
    library_descriptor.source = [NSString stringWithUTF8String:PRESENT_SHADER.data()];
    library_descriptor.options = [[[MTLCompileOptions alloc] init] autorelease];
    impl->present_library = [[compiler newLibraryWithDescriptor:library_descriptor error:&error]
        retain];
    [library_descriptor release];
    if (impl->present_library == nil) {
        LOG_CRITICAL(Render_Metal, "Failed to compile present shader library: {}",
                     error != nil ? error.localizedDescription.UTF8String : "unknown error");
        return;
    }

    MTL4LibraryFunctionDescriptor* vertex_descriptor =
        [[MTL4LibraryFunctionDescriptor alloc] init];
    vertex_descriptor.library = impl->present_library;
    vertex_descriptor.name = @"azahar_present_vs";

    MTL4LibraryFunctionDescriptor* fragment_descriptor =
        [[MTL4LibraryFunctionDescriptor alloc] init];
    fragment_descriptor.library = impl->present_library;
    fragment_descriptor.name = @"azahar_present_fs";

    MTL4RenderPipelineDescriptor* pipeline_descriptor = [[MTL4RenderPipelineDescriptor alloc] init];
    pipeline_descriptor.vertexFunctionDescriptor = vertex_descriptor;
    pipeline_descriptor.fragmentFunctionDescriptor = fragment_descriptor;
    pipeline_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipeline_descriptor.rasterSampleCount = 1;

    impl->present_pipeline =
        [[compiler newRenderPipelineStateWithDescriptor:pipeline_descriptor
                                    compilerTaskOptions:nil
                                                  error:&error] retain];

    [pipeline_descriptor release];
    [fragment_descriptor release];
    [vertex_descriptor release];

    if (impl->present_pipeline == nil) {
        LOG_CRITICAL(Render_Metal, "Failed to build present pipeline: {}",
                     error != nil ? error.localizedDescription.UTF8String : "unknown error");
        return;
    }

    SaveCacheMetadata(instance);
    impl->valid = true;
}

PipelineCache::~PipelineCache() {
    if (!impl) {
        return;
    }
    [impl->present_pipeline release];
    [impl->present_library release];
}

bool PipelineCache::IsValid() const {
    return impl && impl->valid;
}

void* PipelineCache::GetPresentPipeline() const {
    return impl ? (__bridge void*)impl->present_pipeline : nullptr;
}

} // namespace Metal

#endif
