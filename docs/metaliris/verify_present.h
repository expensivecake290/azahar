#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <span>
#include <vector>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

namespace VerifyPresent {

using namespace Metal;
using namespace std::chrono_literals;

using Pixel = std::array<u8, 4>;
using Vec2 = std::array<float, 2>;
struct Vertex {
    Vec2 position;
    Vec2 texcoord;
};
static_assert(sizeof(Vertex) == 16 && offsetof(Vertex, texcoord) == 8);

struct alignas(16) Uniforms {
    std::array<float, 16> matrix{};
    std::array<float, 4> input_resolution{};
    std::array<float, 4> output_resolution{};
    s32 left_screen{};
    s32 right_screen{};
    s32 layer{};
    s32 reverse_interlaced{};
};
static_assert(sizeof(Uniforms) == 112);
static_assert(offsetof(Uniforms, input_resolution) == 64);
static_assert(offsetof(Uniforms, output_resolution) == 80);
static_assert(offsetof(Uniforms, left_screen) == 96);
static_assert(offsetof(Uniforms, reverse_interlaced) == 108);

constexpr Pixel Background{19, 28, 60, 33};

constexpr std::array<float, 16> Orthographic(u32 width, u32 height) {
    return {2.0f / width, 0, 0, -1, 0, 2.0f / height, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1};
}

constexpr std::array<Vertex, 4> Vertices(float left, float top, float right, float bottom,
                                         Vec2 first = {0, 0}, Vec2 last = {1, 1}) {
    return {{{{left, top}, first},
             {{right, top}, {last[0], first[1]}},
             {{left, bottom}, {first[0], last[1]}},
             {{right, bottom}, last}}};
}

class Fixture {
public:
    Fixture(const char* vertex_path, const char* fragment_path, NSString* fragment_name,
            bool use_stream, MTLPixelFormat target_format = MTLPixelFormatRGBA8Unorm)
        : instance(0, false), scheduler(instance.GetContext()),
          allocator(instance.GetDevice(), MemoryBudget(8 * 1024 * 1024)),
          manager(instance, scheduler, allocator), use_stream(use_stream),
          target_format(target_format) {
        auto device = instance.GetDevice();
        auto vertex = Entry(vertex_path, use_stream ? @"present_vert" : @"fixture_present_vertex");
        auto fragment = Entry(fragment_path, fragment_name);
        NSError* error = nil;
        auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                    error:&error];
        compiler = CheckResource(compiler, "Create presentation fixture compiler", error);
        auto state = [MTL4RenderPipelineDescriptor new];
        state.vertexFunctionDescriptor = vertex;
        state.fragmentFunctionDescriptor = fragment;
        state.colorAttachments[0].pixelFormat = target_format;
        if (use_stream) {
            auto layout = [MTLVertexDescriptor new];
            layout.attributes[0].format = MTLVertexFormatFloat2;
            layout.attributes[0].offset = offsetof(Vertex, position);
            layout.attributes[0].bufferIndex = 1;
            layout.attributes[1].format = MTLVertexFormatFloat2;
            layout.attributes[1].offset = offsetof(Vertex, texcoord);
            layout.attributes[1].bufferIndex = 1;
            layout.layouts[1].stride = sizeof(Vertex);
            state.vertexDescriptor = layout;
        }
        pipeline = [compiler newRenderPipelineStateWithDescriptor:state
                                              compilerTaskOptions:nil
                                                            error:&error];
        pipeline = CheckResource(pipeline, "Compile native presentation fixture", error);
    }

    id<MTLTexture> Source(u32 width, u32 height, std::span<const Pixel> pixels) const {
        assert(pixels.size() == u64(width) * height);
        auto source = Texture(width, height, MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead);
        [source replaceRegion:MTLRegionMake2D(0, 0, width, height)
                  mipmapLevel:0
                    withBytes:pixels.data()
                  bytesPerRow:width * sizeof(Pixel)];
        return source;
    }

    template <typename Output = Pixel>
    std::vector<Output> Draw(u32 width, u32 height, const Uniforms& uniforms,
                             const std::array<id<MTLTexture>, 3>& screens,
                             std::span<const Vertex> vertices = {}, u32 first_vertex = 0,
                             MTLSamplerMinMagFilter filter = MTLSamplerMinMagFilterNearest) {
        auto device = instance.GetDevice();
        auto target = Texture(width, height, target_format, MTLTextureUsageRenderTarget);
        const auto options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        auto data = CheckResource([device newBufferWithBytes:&uniforms
                                                      length:sizeof(uniforms)
                                                     options:options],
                                  "Allocate native presentation constants");
        id<MTLBuffer> stream = nil;
        if (use_stream) {
            assert(vertices.size() >= first_vertex + 4);
            stream = CheckResource([device newBufferWithBytes:vertices.data()
                                                       length:vertices.size_bytes()
                                                      options:options],
                                   "Allocate native presentation vertex stream");
        }
        auto sampler_desc = [MTLSamplerDescriptor new];
        sampler_desc.minFilter = filter;
        sampler_desc.magFilter = filter;
        sampler_desc.lodMinClamp = 0;
        sampler_desc.lodMaxClamp = 0;
        sampler_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler_desc.supportArgumentBuffers = YES;
        auto sampler = CheckResource([device newSamplerStateWithDescriptor:sampler_desc],
                                     "Allocate native presentation sampler");

        auto args = [MTL4ArgumentTableDescriptor new];
        args.maxBufferBindCount = stream ? 2 : 1;
        args.maxTextureBindCount = 3;
        args.maxSamplerStateBindCount = 1;
        NSError* error = nil;
        auto table = [device newArgumentTableWithDescriptor:args error:&error];
        table = CheckResource(table, "Allocate native presentation argument table", error);
        [table setAddress:data.gpuAddress atIndex:0];
        if (stream) {
            [table setAddress:stream.gpuAddress atIndex:1];
        }
        for (u32 i = 0; i < screens.size(); ++i) {
            [table setTexture:screens[i].gpuResourceID atIndex:i];
        }
        [table setSamplerState:sampler.gpuResourceID atIndex:0];
        manager.BeginRendering({
            .color = {.target = {target}, .load_action = MTLLoadActionClear},
            .area = {0, 0, width, height},
            .width = width,
            .height = height,
            .clear_color = MTLClearColorMake(Background[0] / 255.0, Background[1] / 255.0,
                                             Background[2] / 255.0, Background[3] / 255.0),
        });
        manager.RecordDraw([=, pipeline = pipeline](EncodingContext& ctx,
                                                    id<MTL4RenderCommandEncoder> encoder) {
            std::vector<ResourceUse> uses{
                {data, BufferAccess::Read, MTLStageVertex | MTLStageFragment}};
            if (stream) {
                uses.push_back({stream, BufferAccess::Read, MTLStageVertex});
            }
            for (auto screen : screens) {
                uses.push_back({screen, BufferAccess::Read, MTLStageFragment});
            }
            ctx.Synchronize(encoder, uses);
            ctx.Retain(pipeline);
            ctx.Retain(table);
            ctx.Retain(sampler);
            [encoder setRenderPipelineState:pipeline];
            [encoder setViewport:MTLViewport{0, 0, double(width), double(height), 0, 1}];
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                        vertexStart:first_vertex
                        vertexCount:4];
        });
        manager.EndRendering();
        assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
               CompletionStatus::Complete);
        std::vector<Output> pixels(u64(width) * height);
        [target getBytes:pixels.data()
             bytesPerRow:width * sizeof(Output)
              fromRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0];
        return pixels;
    }

    void Close() {
        assert(scheduler.Shutdown(std::chrono::steady_clock::now() + 10s));
    }

private:
    MTL4LibraryFunctionDescriptor* Entry(const char* path, NSString* name) const {
        NSError* issue = nil;
        auto library = [instance.GetDevice() newLibraryWithURL:[NSURL fileURLWithPath:@(path)]
                                                         error:&issue];
        auto descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library =
            CheckResource(library, "Load native screen presentation shader", issue);
        descriptor.name = name;
        return descriptor;
    }

    id<MTLTexture> Texture(u32 width, u32 height, MTLPixelFormat format,
                           MTLTextureUsage usage) const {
        auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:width
                                                                            height:height
                                                                         mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = usage;
        return CheckResource([instance.GetDevice() newTextureWithDescriptor:descriptor],
                             "Allocate native presentation fixture texture");
    }

    Instance instance;
    Scheduler scheduler;
    MemoryAllocator allocator;
    RenderManager manager;
    bool use_stream;
    MTLPixelFormat target_format;
    id<MTLRenderPipelineState> pipeline = nil;
};

} // namespace VerifyPresent
