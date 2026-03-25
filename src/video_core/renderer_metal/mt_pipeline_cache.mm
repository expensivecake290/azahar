// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_pipeline_cache.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fmt/format.h>

#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mt_instance.h"

namespace Metal {

struct PipelineCache::Impl {
    id<MTLLibrary> present_library = nil;
    id<MTLRenderPipelineState> present_pipeline = nil;
    u64 present_shader_hash = 0;
    bool valid = false;
};

namespace {

constexpr std::string_view PRESENT_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

struct PresentVertex {
    float2 position;
    float2 tex_coord;
};

struct PresentVertexOut {
    float4 position [[position]];
    float2 tex_coord;
};

struct PresentScreenUniform {
    uint width;
    uint height;
    uint stride;
    uint format;
};

struct PresentFragmentUniforms {
    float opacity;
    uint mode;
    PresentScreenUniform left;
    PresentScreenUniform right;
};

vertex PresentVertexOut azahar_present_vs(const device PresentVertex* vertices [[buffer(0)]],
                                          uint vertex_id [[vertex_id]]) {
    PresentVertexOut out;
    out.position = float4(vertices[vertex_id].position, 0.0, 1.0);
    out.tex_coord = vertices[vertex_id].tex_coord;
    return out;
}

static float4 decode_pixel(const device uchar* data, PresentScreenUniform screen, float2 tex_coord) {
    if (data == nullptr || screen.width == 0u || screen.height == 0u || screen.stride == 0u) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    const uint x = min(static_cast<uint>(tex_coord.x * float(screen.width)), screen.width - 1u);
    const uint y = min(static_cast<uint>(tex_coord.y * float(screen.height)), screen.height - 1u);

    switch (screen.format) {
    case 0: {
        const uint index = ((y * screen.stride) + x) * 4u;
        return float4(float(data[index + 3u]) / 255.0, float(data[index + 2u]) / 255.0,
                      float(data[index + 1u]) / 255.0, float(data[index]) / 255.0);
    }
    case 1: {
        const uint index = ((y * screen.stride) + x) * 3u;
        return float4(float(data[index + 2u]) / 255.0, float(data[index + 1u]) / 255.0,
                      float(data[index]) / 255.0, 1.0);
    }
    case 2: {
        const uint index = ((y * screen.stride) + x) * 2u;
        const ushort pixel = ushort(data[index]) | (ushort(data[index + 1u]) << 8);
        return float4(float((pixel >> 11) & 0x1Fu) / 31.0, float((pixel >> 5) & 0x3Fu) / 63.0,
                      float(pixel & 0x1Fu) / 31.0, 1.0);
    }
    case 3: {
        const uint index = ((y * screen.stride) + x) * 2u;
        const ushort pixel = ushort(data[index]) | (ushort(data[index + 1u]) << 8);
        return float4(float((pixel >> 11) & 0x1Fu) / 31.0, float((pixel >> 6) & 0x1Fu) / 31.0,
                      float((pixel >> 1) & 0x1Fu) / 31.0, (pixel & 0x1u) ? 1.0 : 0.0);
    }
    case 4: {
        const uint index = ((y * screen.stride) + x) * 2u;
        const ushort pixel = ushort(data[index]) | (ushort(data[index + 1u]) << 8);
        return float4(float((pixel >> 12) & 0xFu) / 15.0, float((pixel >> 8) & 0xFu) / 15.0,
                      float((pixel >> 4) & 0xFu) / 15.0, float(pixel & 0xFu) / 15.0);
    }
    default:
        return float4(1.0, 0.0, 1.0, 1.0);
    }
}

fragment float4 azahar_present_fs(PresentVertexOut in [[stage_in]],
                                  float4 position [[position]],
                                  constant PresentFragmentUniforms& uniforms [[buffer(0)]],
                                  const device uchar* color_buffer_l [[buffer(1)]],
                                  const device uchar* color_buffer_r [[buffer(2)]]) {
    const float4 left = decode_pixel(color_buffer_l, uniforms.left, in.tex_coord);
    const float4 right =
        color_buffer_r != nullptr ? decode_pixel(color_buffer_r, uniforms.right, in.tex_coord) : left;
    float4 color = left;

    switch (uniforms.mode) {
    case 1:
        color = float4(left.r, right.g, right.b, max(left.a, right.a));
        break;
    case 2:
    case 3: {
        const bool odd_line = (static_cast<uint>(position.y) & 1u) != 0u;
        const bool use_right = uniforms.mode == 2 ? odd_line : !odd_line;
        color = use_right ? right : left;
        break;
    }
    default:
        break;
    }

    color.a *= uniforms.opacity;
    return color;
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

static u64 GetShaderHash() {
    return Common::ComputeHash64<Common::HashAlgo64::CityHash>(PRESENT_SHADER.data(),
                                                               PRESENT_SHADER.size());
}

static void SaveCacheMetadata(const Instance& instance, u64 shader_hash) {
    if (!EnsureDirectories()) {
        return;
    }

    const std::string metadata =
        fmt::format("version=2\ndevice={}\nshader_hash={:016x}\n", instance.GetDeviceName(),
                    shader_hash);
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

    impl->present_shader_hash = GetShaderHash();

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

    SaveCacheMetadata(instance, impl->present_shader_hash);
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

u64 PipelineCache::GetPresentShaderHash() const {
    return impl ? impl->present_shader_hash : 0;
}

} // namespace Metal

#endif
