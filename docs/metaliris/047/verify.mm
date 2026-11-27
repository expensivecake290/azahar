#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

constexpr u32 SourceWidth = 5;
constexpr u32 SourceHeight = 3;
constexpr u32 Width = 9;
constexpr u32 Height = 7;
constexpr MTLScissorRect Scissor{1, 0, 8, 6};
constexpr float SentinelDepth = 0.875f;
constexpr u32 SentinelStencil = 0xD7;

struct alignas(8) Transform {
    std::array<float, 2> scale;
    std::array<float, 2> offset;
};
static_assert(sizeof(Transform) == 16 && offsetof(Transform, offset) == 8);
constexpr Transform Identity{{1, 1}, {0, 0}};
constexpr Transform Reversed{{1.4f, -1.2f}, {-0.2f, 1.1f}};
using Pixel = std::array<u32, 2>;

u32 Nearest(double normalized, u32 size) {
    return std::min(static_cast<u32>(std::clamp(normalized, 0.0, 1.0) * size), size - 1);
}

Pixel Expected(u32 x, u32 y) {
    if (x < Scissor.x || y < Scissor.y || x >= Scissor.x + Scissor.width ||
        y >= Scissor.y + Scissor.height) {
        return {std::bit_cast<u32>(SentinelDepth), SentinelStencil};
    }
    const u32 column =
        Nearest((x + 0.5) * Reversed.scale[0] / Width + Reversed.offset[0], SourceWidth);
    const u32 row =
        Nearest((y + 0.5) * Reversed.scale[1] / Height + Reversed.offset[1], SourceHeight);
    const u32 index = row * SourceWidth + column;
    return {std::bit_cast<u32>(float(index) / 16.0f), index == 14 ? 255 : 17 * index};
}

void CPU() {
    assert(Nearest(-100, 5) == 0);
    assert(Nearest(100, 5) == 4);
    assert(Nearest(1, 3) == 2);
    assert(Nearest(0.499, 2) == 0);
    assert(Nearest(0.501, 2) == 1);
    assert((Expected(8, 0) == Pixel{std::bit_cast<u32>(0.875f), 255}));
    assert((Expected(1, 5) == Pixel{0, 0}));
    assert((Expected(0, 0) == Pixel{std::bit_cast<u32>(SentinelDepth), SentinelStencil}));
    std::puts(
        "PASS: independent nearest depth/stencil offsets, clamped endpoints and scissor oracle");
}

MTL4LibraryFunctionDescriptor* Entry(id<MTLLibrary> library, NSString* name) {
    auto desc = [MTL4LibraryFunctionDescriptor new];
    desc.library = library;
    desc.name = name;
    return desc;
}

id<MTLRenderPipelineState> Pipeline(id<MTL4Compiler> compiler, id<MTLLibrary> vertex,
                                    id<MTLLibrary> fragment, NSString* entry) {
    auto desc = [MTL4RenderPipelineDescriptor new];
    desc.vertexFunctionDescriptor = Entry(vertex, @"full_screen_triangle_vert");
    desc.fragmentFunctionDescriptor = Entry(fragment, entry);
    NSError* error = nil;
    auto state = [compiler newRenderPipelineStateWithDescriptor:desc
                                            compilerTaskOptions:nil
                                                          error:&error];
    return CheckResource(state, "Compile depth/stencil blit fixture graphics state", error);
}

id<MTLTexture> Texture(id<MTLDevice> device, u32 width, u32 height) {
    auto desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.usage =
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    return CheckResource([device newTextureWithDescriptor:desc], "Allocate native depth/stencil");
}

void GPU(const char* shader_path, const char* fixture_path, const char* vertex_path) {
    Instance instance(0, false);
    auto device = instance.GetDevice();
    NSError* error = nil;
    const auto load = [&](const char* path) {
        NSError* issue = nil;
        auto lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&issue];
        return CheckResource(lib, "Load native aspect fixture library", issue);
    };
    auto library = load(shader_path);
    auto fixture = load(fixture_path);
    auto vertex = load(vertex_path);
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create native depth/stencil compiler", error);
    auto initialize = Pipeline(compiler, vertex, fixture, @"fixture_seed");
    auto convert = Pipeline(compiler, vertex, library, @"blit_depth_stencil_frag");
    auto descriptor = [MTLDepthStencilDescriptor new];
    descriptor.depthCompareFunction = MTLCompareFunctionAlways;
    descriptor.depthWriteEnabled = YES;
    auto stencil_descriptor = [MTLStencilDescriptor new];
    stencil_descriptor.stencilCompareFunction = MTLCompareFunctionAlways;
    stencil_descriptor.depthStencilPassOperation = MTLStencilOperationReplace;
    descriptor.frontFaceStencil = stencil_descriptor;
    descriptor.backFaceStencil = stencil_descriptor;
    auto state = CheckResource([device newDepthStencilStateWithDescriptor:descriptor],
                               "Create always-replace depth/stencil state");

    auto source = Texture(device, SourceWidth, SourceHeight);
    auto destination = Texture(device, Width, Height);
    auto src_stencil =
        CheckResource([source newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8],
                      "Create source stencil view");
    auto dst_stencil =
        CheckResource([destination newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8],
                      "Create destination stencil view");
    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    const auto draw = [&](id<MTLTexture> target, u32 width, u32 height, const Transform& transform,
                          id<MTLRenderPipelineState> pipeline, bool blit) {
        auto constants =
            CheckResource([device newBufferWithBytes:&transform
                                              length:sizeof(transform)
                                             options:MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeUntracked],
                          "Allocate native depth/stencil transform");
        auto args = [MTL4ArgumentTableDescriptor new];
        args.maxBufferBindCount = 1;
        args.maxTextureBindCount = blit ? 2 : 0;
        auto table = [device newArgumentTableWithDescriptor:args error:&error];
        table = CheckResource(table, "Allocate native depth/stencil arguments", error);
        [table setAddress:constants.gpuAddress atIndex:0];
        if (blit) {
            [table setTexture:source.gpuResourceID atIndex:0];
            [table setTexture:src_stencil.gpuResourceID atIndex:1];
        }
        manager.BeginRendering({
            .depth = {.target = {target}, .load_action = MTLLoadActionClear},
            .stencil = {.target = {target}, .load_action = MTLLoadActionClear},
            .area = {0, 0, width, height},
            .width = width,
            .height = height,
            .clear_depth = SentinelDepth,
            .clear_stencil = SentinelStencil,
        });
        manager.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            ctx.MakeResident(constants);
            if (blit) {
                const std::array uses{
                    ResourceUse{source, BufferAccess::Read, MTLStageFragment},
                    ResourceUse{src_stencil, BufferAccess::Read, MTLStageFragment, {}, source},
                };
                ctx.Synchronize(encoder, uses);
                [encoder setScissorRect:Scissor];
            }
            ctx.Retain(state);
            ctx.Retain(table);
            ctx.Retain(pipeline);
            [encoder setRenderPipelineState:pipeline];
            [encoder setDepthStencilState:state];
            [encoder setViewport:MTLViewport{0, 0, double(width), double(height), 0, 1}];
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        });
        manager.EndRendering();
    };
    draw(source, SourceWidth, SourceHeight, Identity, initialize, false);
    draw(destination, Width, Height, Reversed, convert, true);

    auto compute = [MTL4ComputePipelineDescriptor new];
    compute.computeFunctionDescriptor = Entry(fixture, @"fixture_read");
    auto read = [compiler newComputePipelineStateWithDescriptor:compute
                                            compilerTaskOptions:nil
                                                          error:&error];
    read = CheckResource(read, "Compile native aspect readback", error);
    auto buffer = CheckResource([device newBufferWithLength:Width * Height * sizeof(Pixel)
                                                    options:MTLResourceStorageModeShared |
                                                            MTLResourceHazardTrackingModeUntracked],
                                "Allocate native aspect readback");
    auto args = [MTL4ArgumentTableDescriptor new];
    args.maxBufferBindCount = 1;
    args.maxTextureBindCount = 2;
    auto table = [device newArgumentTableWithDescriptor:args error:&error];
    table = CheckResource(table, "Allocate aspect readback arguments", error);
    [table setAddress:buffer.gpuAddress atIndex:0];
    [table setTexture:destination.gpuResourceID atIndex:0];
    [table setTexture:dst_stencil.gpuResourceID atIndex:1];
    scheduler.Record([=](EncodingContext& ctx) {
        const std::array uses{
            ResourceUse{destination, BufferAccess::Read, MTLStageDispatch},
            ResourceUse{dst_stencil, BufferAccess::Read, MTLStageDispatch, {}, destination},
            ResourceUse{buffer, BufferAccess::Write, MTLStageDispatch},
        };
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        ctx.Synchronize(encoder, uses);
        [encoder setComputePipelineState:read];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(Width, Height, 1)
            threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
        ctx.Retain(read);
        ctx.Retain(table);
        ctx.EndPass(encoder);
    });

    const auto deadline = std::chrono::steady_clock::now() + 10s;
    assert(scheduler.Finish({}, deadline) == CompletionStatus::Complete);
    const auto* output = static_cast<const Pixel*>(buffer.contents);
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            assert(output[y * Width + x] == Expected(x, y));
        }
    }
    assert(scheduler.Shutdown(deadline));
    std::puts("PASS: native exact depth/stencil words and untouched scissor neighbors");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 4);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts(
                "PENDING: native stencil export and depth/stencil readback require a Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2], argv[3]);
        }
    }
}
