#pragma once

#include <array>
#include <chrono>
#include <span>
#include <stdexcept>
#include <vector>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

namespace VerifyFilter {

using namespace Metal;
using namespace std::chrono_literals;

using Pixel = std::array<u8, 4>;

struct alignas(8) Uniforms {
    std::array<f32, 2> scale{1, 1};
    std::array<f32, 2> offset{0, 0};
    f32 resolution_scale{1.0f};
};
static_assert(offsetof(Uniforms, resolution_scale) == 16);

struct VertexDraw {
    NSString* name = @"full_screen_triangle_vert";
    MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
    u32 count = 3;
};

class Fixture {
public:
    Fixture(const char* vertex_path, const char* fragment_path, NSString* fragment_name,
            MTLPixelFormat format = MTLPixelFormatRGBA8Unorm, VertexDraw vertices = {})
        : instance(0, false), scheduler(instance.GetContext()),
          allocator(instance.GetDevice(), MemoryBudget(8 * 1024 * 1024)),
          manager(instance, scheduler, allocator), vertices(vertices) {
        auto device = instance.GetDevice();
        NSError* error = nil;
        auto vertex = [device newLibraryWithURL:[NSURL fileURLWithPath:@(vertex_path)]
                                          error:&error];
        vertex = CheckResource(vertex, "Load native full-screen vertex library", error);
        compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        compiler = CheckResource(compiler, "Create native filter compiler", error);
        vertex_function = [MTL4LibraryFunctionDescriptor new];
        vertex_function.library = vertex;
        vertex_function.name = vertices.name;
        SetFragment(fragment_path, fragment_name, format);
        auto sampler_descriptor = [MTLSamplerDescriptor new];
        sampler_descriptor.minFilter = MTLSamplerMinMagFilterLinear;
        sampler_descriptor.magFilter = MTLSamplerMinMagFilterLinear;
        sampler_descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler_descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler_descriptor.lodMinClamp = 0;
        sampler_descriptor.lodMaxClamp = 0;
        sampler_descriptor.supportArgumentBuffers = YES;
        sampler = CheckResource([device newSamplerStateWithDescriptor:sampler_descriptor],
                                "Create native clamp-to-edge filter sampler");
    }

    void SetFragment(const char* path, NSString* name, MTLPixelFormat target_format) {
        NSError* error = nil;
        auto fragment = [Device() newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
        fragment = CheckResource(fragment, "Load native filter library", error);
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = fragment;
        function.name = name;
        auto descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = vertex_function;
        descriptor.fragmentFunctionDescriptor = function;
        descriptor.colorAttachments[0].pixelFormat = target_format;
        auto state = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                                compilerTaskOptions:nil
                                                              error:&error];
        pipeline = CheckResource(state, "Compile native filter pipeline", error);
        format = target_format;
    }

    void Close() {
        if (!scheduler.Shutdown(Deadline())) {
            throw std::runtime_error("Filter fixture did not retire");
        }
    }

    template <typename Data>
    id<MTLTexture> Source(u32 width, u32 height, MTLPixelFormat source_format,
                          std::span<const Data> data) {
        if (data.size() != u64(width) * height) {
            throw std::invalid_argument("Filter fixture input does not match texture dimensions");
        }
        auto source = Texture(Device(), width, height, source_format, MTLTextureUsageShaderRead);
        [source replaceRegion:MTLRegionMake2D(0, 0, width, height)
                  mipmapLevel:0
                    withBytes:data.data()
                  bytesPerRow:width * sizeof(Data)];
        return source;
    }

    template <typename Result>
    std::vector<Result> Draw(u32 width, u32 height, const Uniforms& uniforms,
                             std::vector<id<MTLTexture>> sources) {
        return Read<Result>(Render(width, height, uniforms, std::move(sources)));
    }

    id<MTLTexture> Render(u32 width, u32 height, const Uniforms& uniforms,
                          std::vector<id<MTLTexture>> sources) {
        auto device = Device();
        auto target = Texture(device, width, height, format,
                              MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
        auto buffer =
            CheckResource([device newBufferWithBytes:&uniforms
                                              length:sizeof(uniforms)
                                             options:MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeUntracked],
                          "Allocate native filter constants");
        auto arguments = [MTL4ArgumentTableDescriptor new];
        arguments.maxBufferBindCount = 1;
        arguments.maxTextureBindCount = sources.size();
        arguments.maxSamplerStateBindCount = 1;
        NSError* error = nil;
        auto table = [device newArgumentTableWithDescriptor:arguments error:&error];
        table = CheckResource(table, "Build native filter fixture argument table", error);
        [table setSamplerState:sampler.gpuResourceID atIndex:0];
        [table setAddress:buffer.gpuAddress atIndex:0];
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (sources[i]) {
                [table setTexture:sources[i].gpuResourceID atIndex:i];
            }
        }
        manager.BeginRendering({
            .color = {.target = {target}, .load_action = MTLLoadActionClear},
            .area = {0, 0, width, height},
            .width = width,
            .height = height,
        });
        manager.RecordDraw([table, buffer, width, height, sources = std::move(sources),
                            pipeline = pipeline, sampler = sampler, vertices = vertices](
                               EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            std::vector<ResourceUse> uses{
                {buffer, BufferAccess::Read, MTLStageVertex | MTLStageFragment}};
            for (const auto& texture : sources) {
                if (texture) {
                    uses.push_back({texture, BufferAccess::Read, MTLStageFragment});
                }
            }
            ctx.Synchronize(encoder, uses);
            ctx.Retain(pipeline);
            ctx.Retain(table);
            ctx.Retain(sampler);
            [encoder setRenderPipelineState:pipeline];
            [encoder setViewport:MTLViewport{0, 0, static_cast<double>(width),
                                             static_cast<double>(height), 0, 1}];
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
            [encoder drawPrimitives:vertices.primitive vertexStart:0 vertexCount:vertices.count];
        });
        manager.EndRendering();
        return target;
    }

    template <typename Result>
    std::vector<Result> Read(id<MTLTexture> target) {
        const auto width = target.width;
        const auto height = target.height;
        if (scheduler.Finish({}, Deadline()) != CompletionStatus::Complete) {
            throw std::runtime_error("Filter fixture draw failed");
        }

        std::vector<Result> pixels(u64(width) * height);
        [target getBytes:pixels.data()
             bytesPerRow:width * sizeof(Result)
              fromRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0];
        return pixels;
    }

    id<MTLDevice> Device() const {
        return instance.GetDevice();
    }

private:
    static std::chrono::steady_clock::time_point Deadline() {
        return std::chrono::steady_clock::now() + 10s;
    }

    static id<MTLTexture> Texture(id<MTLDevice> device, u32 width, u32 height,
                                  MTLPixelFormat format, MTLTextureUsage usage) {
        auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:width
                                                                            height:height
                                                                         mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = usage;
        return CheckResource([device newTextureWithDescriptor:descriptor],
                             "Allocate native filter fixture texture");
    }

    Instance instance;
    Scheduler scheduler;
    MemoryAllocator allocator;
    RenderManager manager;
    MTLPixelFormat format;
    VertexDraw vertices;
    id<MTL4Compiler> compiler = nil;
    MTL4LibraryFunctionDescriptor* vertex_function = nil;
    id<MTLSamplerState> sampler = nil;
    id<MTLRenderPipelineState> pipeline = nil;
};

} // namespace VerifyFilter
