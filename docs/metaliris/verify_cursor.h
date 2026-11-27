#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

namespace VerifyCursor {

using namespace Metal;
using namespace std::chrono_literals;

using Pixel = std::array<u8, 4>;
using Position = std::array<float, 2>;
constexpr u32 Width = 11;
constexpr u32 Height = 7;
constexpr Pixel Background{17, 200, 51, 39};
constexpr MTLScissorRect Vertical{5, 1, 1, 5};
constexpr MTLScissorRect Horizontal{3, 3, 5, 1};

constexpr bool Contains(MTLScissorRect rect, u32 x, u32 y) {
    return x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height;
}

constexpr Pixel Expected(u32 x, u32 y) {
    const bool vertical = Contains(Vertical, x, y);
    const bool horizontal = Contains(Horizontal, x, y);
    if (horizontal != vertical) {
        return {255 - Background[0], 255 - Background[1], 255 - Background[2], 255};
    }
    return {Background[0], Background[1], Background[2], vertical ? u8{255} : Background[3]};
}

constexpr Position GuestPosition(NSUInteger x, NSUInteger y) {
    return {float(2 * x) / Width - 1, float(2 * y) / Height - 1};
}

constexpr std::array<Position, 15> Vertices() {
    std::array<Position, 15> result{};
    u32 offset = 3;
    for (const auto rect : {Vertical, Horizontal}) {
        const auto tl = GuestPosition(rect.x, rect.y);
        const auto tr = GuestPosition(rect.x + rect.width, rect.y);
        const auto bl = GuestPosition(rect.x, rect.y + rect.height);
        const auto br = GuestPosition(rect.x + rect.width, rect.y + rect.height);
        for (const auto vertex : {tl, tr, br, tl, br, bl}) {
            result[offset++] = vertex;
        }
    }
    return result;
}

inline void Run(const char* vertex_path, const char* fragment_path, bool use_stream) {
    Instance instance(0, false);
    auto device = instance.GetDevice();
    NSError* error = nil;
    const auto load = [&](const char* path) {
        NSError* issue = nil;
        auto lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&issue];
        return CheckResource(lib, "Load native cursor verification shader", issue);
    };
    auto vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = load(vertex_path);
    vertex.name = use_stream ? @"cursor_vert" : @"fixture_cursor_vertex";
    auto fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = load(fragment_path);
    fragment.name = @"cursor_frag";
    auto desc = [MTL4RenderPipelineDescriptor new];
    desc.vertexFunctionDescriptor = vertex;
    desc.fragmentFunctionDescriptor = fragment;
    if (use_stream) {
        auto input = [MTLVertexDescriptor new];
        input.attributes[0].format = MTLVertexFormatFloat2;
        input.attributes[0].bufferIndex = 0;
        input.layouts[0].stride = sizeof(Position);
        desc.vertexDescriptor = input;
    }

    auto color = desc.colorAttachments[0];
    color.pixelFormat = MTLPixelFormatRGBA8Unorm;
    color.blendingState = MTL4BlendStateEnabled;
    color.sourceRGBBlendFactor = MTLBlendFactorOneMinusDestinationColor;
    color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceColor;
    color.sourceAlphaBlendFactor = MTLBlendFactorOne;
    color.destinationAlphaBlendFactor = MTLBlendFactorZero;
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create native cursor compiler", error);
    auto pipeline = [compiler newRenderPipelineStateWithDescriptor:desc
                                               compilerTaskOptions:nil
                                                             error:&error];
    pipeline = CheckResource(pipeline, "Compile native cursor pipeline", error);

    auto texture = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                      width:Width
                                                                     height:Height
                                                                  mipmapped:NO];
    texture.storageMode = MTLStorageModeShared;
    texture.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    texture.usage = MTLTextureUsageRenderTarget;
    auto target =
        CheckResource([device newTextureWithDescriptor:texture], "Allocate cursor target");

    id<MTL4ArgumentTable> table = nil;
    id<MTLBuffer> positions = nil;
    if (use_stream) {
        constexpr auto data = Vertices();
        positions =
            CheckResource([device newBufferWithBytes:data.data()
                                              length:sizeof(data)
                                             options:MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeUntracked],
                          "Allocate verification cursor vertices");
        auto args = [MTL4ArgumentTableDescriptor new];
        args.maxBufferBindCount = 1;
        table = [device newArgumentTableWithDescriptor:args error:&error];
        table = CheckResource(table, "Create native cursor argument table", error);
        [table setAddress:positions.gpuAddress atIndex:0];
    }

    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    manager.BeginRendering({
        .color = {.target = {target}, .load_action = MTLLoadActionClear},
        .area = {0, 0, Width, Height},
        .width = Width,
        .height = Height,
        .clear_color = MTLClearColorMake(Background[0] / 255.0, Background[1] / 255.0,
                                         Background[2] / 255.0, Background[3] / 255.0),
    });
    manager.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
        [encoder setRenderPipelineState:pipeline];
        [encoder setViewport:MTLViewport{0, 0, Width, Height, 0, 1}];
        ctx.Retain(pipeline);
        if (use_stream) {
            ctx.MakeResident(positions);
            ctx.Retain(table);
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:3 vertexCount:12];
        } else {
            for (const auto rect : {Vertical, Horizontal}) {
                [encoder setScissorRect:rect];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }
        }
    });
    manager.EndRendering();
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    assert(scheduler.Finish({}, deadline) == CompletionStatus::Complete);
    std::array<Pixel, Width * Height> pixels{};
    [target getBytes:pixels.data()
         bytesPerRow:Width * sizeof(Pixel)
          fromRegion:MTLRegionMake2D(0, 0, Width, Height)
         mipmapLevel:0];
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            assert(pixels[y * Width + x] == Expected(x, y));
        }
    }
    assert(scheduler.Shutdown(deadline));
}

} // namespace VerifyCursor
