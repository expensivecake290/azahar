#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

constexpr u32 SourceWidth = 12;
constexpr u32 SourceHeight = 8;
constexpr u32 Width = 11;
constexpr u32 Height = 7;
constexpr u32 SampleCount = 4;

using Pixel = std::array<u8, 4>;
using Coordinate = std::array<u32, 2>;

struct alignas(8) CopyRegion {
    Coordinate source;
    Coordinate destination;
    Coordinate extent;
};
static_assert(sizeof(CopyRegion) == 24);
static_assert(offsetof(CopyRegion, destination) == 8 && offsetof(CopyRegion, extent) == 16);

constexpr std::array<Pixel, SampleCount> Expected{
    {{0, 0, 0, 0}, {33, 17, 34, 51}, {112, 100, 135, 174}, {255, 255, 255, 255}},
};
constexpr Pixel Resolved{100, 93, 106, 120};
constexpr Pixel Sentinel{120, 18, 53, 219};
constexpr CopyRegion Region{{2, 1}, {3, 1}, {7, 5}};
constexpr CopyRegion Edge{{11, 7}, {10, 6}, {1, 1}};

constexpr u32 Pack(Pixel bytes) {
    return static_cast<u32>(bytes[0]) | (static_cast<u32>(bytes[1]) << 8) |
           (static_cast<u32>(bytes[2]) << 16) | (static_cast<u32>(bytes[3]) << 24);
}

bool Copied(const CopyRegion& copy, u32 x, u32 y) {
    if (x < copy.destination[0] || y < copy.destination[1]) {
        return false;
    }
    const u32 horizontal = x - copy.destination[0];
    const u32 vertical = y - copy.destination[1];
    return horizontal < copy.extent[0] && vertical < copy.extent[1] &&
           u64(copy.source[0]) + horizontal < SourceWidth &&
           u64(copy.source[1]) + vertical < SourceHeight;
}

void CheckCPU() {
    for (u32 lane = 0; lane < 4; ++lane) {
        u32 sum = 0;
        for (const auto& sample : Expected) {
            sum += sample[lane];
        }
        assert(sum == Resolved[lane] * SampleCount);
    }
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            assert(Copied(Region, x, y) == (x >= 3 && x <= 9 && y >= 1 && y <= 5));
            assert(Copied(Edge, x, y) == (x == 10 && y == 6));
            assert(!Copied({{0, 0}, {0, 0}, {8, 0}}, x, y));
            assert(!Copied({{0, 0}, {std::numeric_limits<u32>::max(), 0}, {8, 8}}, x, y));
            assert(!Copied({{0, std::numeric_limits<u32>::max()}, {0, 0}, {8, 8}}, x, y));
        }
    }
    std::puts("PASS: independent MSAA sample, resolve byte and rectangle oracle");
}

MTL4LibraryFunctionDescriptor* Function(id<MTLLibrary> library, NSString* name) {
    auto descriptor = [MTL4LibraryFunctionDescriptor new];
    descriptor.library = library;
    descriptor.name = name;
    return descriptor;
}

id<MTLRenderPipelineState> RenderPipeline(id<MTL4Compiler> compiler, id<MTLLibrary> library,
                                          NSString* vertex, NSString* fragment,
                                          MTLPixelFormat format) {
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = Function(library, vertex);
    descriptor.fragmentFunctionDescriptor = Function(library, fragment);
    descriptor.rasterSampleCount = SampleCount;
    if (format != MTLPixelFormatInvalid) {
        descriptor.colorAttachments[0].pixelFormat = format;
    }
    NSError* error = nil;
    auto pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                               compilerTaskOptions:nil
                                                             error:&error];
    return CheckResource(pipeline, "Build multisample fixture pipeline", error);
}

void DrawTriangle(EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder,
                  id<MTLRenderPipelineState> pipeline, u32 width, u32 height) {
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{0, 0, static_cast<double>(width), static_cast<double>(height),
                                     0, 1}];
    ctx.Retain(pipeline);
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format, u32 width, u32 height,
                       u32 samples, MTLTextureUsage usage) {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                   width:width
                                                                  height:height
                                                               mipmapped:NO];
    desc.storageMode = samples > 1 ? MTLStorageModePrivate : MTLStorageModeShared;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    if (samples > 1) {
        desc.textureType = MTLTextureType2DMultisample;
        desc.sampleCount = samples;
    }
    desc.usage = usage;
    return CheckResource([device newTextureWithDescriptor:desc],
                         "Allocate multisample fixture image");
}

void Run(const char* shader_path, const char* fixture_path) {
    Instance instance(0, false);
    if (!instance.SupportsSampleCount(SampleCount)) {
        std::puts("PENDING: exact 4x MSAA checks require a Metal 4 GPU supporting 4 samples.");
        return;
    }
    auto device = instance.GetDevice();
    NSError* error = nil;
    auto shader = [device newLibraryWithURL:[NSURL fileURLWithPath:@(shader_path)] error:&error];
    shader = CheckResource(shader, "Load native multisample conversion library", error);
    auto fixture = [device newLibraryWithURL:[NSURL fileURLWithPath:@(fixture_path)] error:&error];
    fixture = CheckResource(fixture, "Load native multisample validation library", error);
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create native multisample verification compiler", error);

    auto depth_stencil = Texture(
        device, MTLPixelFormatDepth32Float_Stencil8, SourceWidth, SourceHeight, SampleCount,
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
    auto stencil =
        CheckResource([depth_stencil newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8],
                      "Create multisample stencil view");
    auto color = Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, SampleCount,
                         MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    auto resolved =
        Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, 1, MTLTextureUsageRenderTarget);

    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    auto initialize = RenderPipeline(compiler, fixture, @"fixture_vertex", @"fixture_samples",
                                     MTLPixelFormatInvalid);
    auto depth_state = [MTLDepthStencilDescriptor new];
    depth_state.depthCompareFunction = MTLCompareFunctionAlways;
    depth_state.depthWriteEnabled = YES;
    auto stencil_state = [MTLStencilDescriptor new];
    stencil_state.stencilCompareFunction = MTLCompareFunctionAlways;
    stencil_state.depthStencilPassOperation = MTLStencilOperationReplace;
    depth_state.frontFaceStencil = stencil_state;
    depth_state.backFaceStencil = stencil_state;
    auto state = CheckResource([device newDepthStencilStateWithDescriptor:depth_state],
                               "Build per-sample depth/stencil fixture state");
    manager.BeginRendering({
        .depth = {.target = {depth_stencil}, .load_action = MTLLoadActionClear},
        .stencil = {.target = {depth_stencil}, .load_action = MTLLoadActionClear},
        .area = {0, 0, SourceWidth, SourceHeight},
        .width = SourceWidth,
        .height = SourceHeight,
        .samples = SampleCount,
    });
    manager.RecordDraw(
        [initialize, state](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            [encoder setDepthStencilState:state];
            ctx.Retain(state);
            DrawTriangle(ctx, encoder, initialize, SourceWidth, SourceHeight);
        });
    manager.EndRendering();

    auto convert = RenderPipeline(compiler, shader, @"d24s8_to_rgba8_ms_vertex",
                                  @"d24s8_to_rgba8_ms_fragment", MTLPixelFormatRGBA8Unorm);
    manager.BeginRendering({
        .color = {.target = {color},
                  .resolve = {resolved},
                  .load_action = MTLLoadActionClear,
                  .store_action = MTLStoreActionStoreAndMultisampleResolve},
        .area = {0, 0, Width, Height},
        .width = Width,
        .height = Height,
        .samples = SampleCount,
        .clear_color = MTLClearColorMake(Sentinel[0] / 255.0, Sentinel[1] / 255.0,
                                         Sentinel[2] / 255.0, Sentinel[3] / 255.0),
    });
    const auto draw = [&](const CopyRegion& copy) {
        auto region =
            CheckResource([device newBufferWithBytes:&copy
                                              length:sizeof(copy)
                                             options:MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeUntracked],
                          "Allocate sample conversion copy parameters");
        auto desc = [MTL4ArgumentTableDescriptor new];
        desc.maxBufferBindCount = 1;
        desc.maxTextureBindCount = 2;
        auto table = [device newArgumentTableWithDescriptor:desc error:&error];
        table = CheckResource(table, "Build sample conversion argument table", error);
        [table setAddress:region.gpuAddress atIndex:0];
        [table setTexture:depth_stencil.gpuResourceID atIndex:0];
        [table setTexture:stencil.gpuResourceID atIndex:1];
        manager.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            const std::array uses{
                ResourceUse{depth_stencil, BufferAccess::Read, MTLStageFragment},
                ResourceUse{stencil, BufferAccess::Read, MTLStageFragment, {}, depth_stencil},
                ResourceUse{region, BufferAccess::Read, MTLStageFragment},
            };
            ctx.Synchronize(encoder, uses);
            ctx.Retain(table);
            [encoder setArgumentTable:table atStages:MTLRenderStageFragment];
            DrawTriangle(ctx, encoder, convert, Width, Height);
        });
    };
    draw(Region);
    draw(Edge);
    draw({{0, 0}, {0, 0}, {8, 0}});
    draw({{0, 0}, {std::numeric_limits<u32>::max(), 0}, {8, 8}});
    draw({{0, std::numeric_limits<u32>::max()}, {0, 0}, {8, 8}});
    manager.EndRendering();

    auto read_desc = [MTL4ComputePipelineDescriptor new];
    read_desc.computeFunctionDescriptor = Function(fixture, @"fixture_read_samples");
    auto read_samples = [compiler newComputePipelineStateWithDescriptor:read_desc
                                                    compilerTaskOptions:nil
                                                                  error:&error];
    read_samples = CheckResource(read_samples, "Build multisample fixture readback", error);
    auto output =
        CheckResource([device newBufferWithLength:Width * Height * SampleCount * sizeof(u32)
                                          options:MTLResourceStorageModeShared |
                                                  MTLResourceHazardTrackingModeUntracked],
                      "Allocate multisample fixture readback");
    auto read_args = [MTL4ArgumentTableDescriptor new];
    read_args.maxBufferBindCount = 1;
    read_args.maxTextureBindCount = 1;
    auto table = [device newArgumentTableWithDescriptor:read_args error:&error];
    table = CheckResource(table, "Create sample readback argument table", error);
    [table setAddress:output.gpuAddress atIndex:0];
    [table setTexture:color.gpuResourceID atIndex:0];
    scheduler.Record([=](EncodingContext& ctx) {
        const std::array uses{
            ResourceUse{color, BufferAccess::Read, MTLStageDispatch},
            ResourceUse{output, BufferAccess::Write, MTLStageDispatch},
        };
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        ctx.Synchronize(encoder, uses);
        [encoder setComputePipelineState:read_samples];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(Width, Height, SampleCount)
            threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
        ctx.Retain(read_samples);
        ctx.Retain(table);
        ctx.EndPass(encoder);
    });

    const auto deadline = std::chrono::steady_clock::now() + 10s;
    assert(scheduler.Finish({}, deadline) == CompletionStatus::Complete);
    std::array<Pixel, Width * Height> pixels{};
    [resolved getBytes:pixels.data()
           bytesPerRow:Width * sizeof(Pixel)
            fromRegion:MTLRegionMake2D(0, 0, Width, Height)
           mipmapLevel:0];
    const auto* native_samples = static_cast<const u32*>(output.contents);
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            const auto index = y * Width + x;
            const bool active = Copied(Region, x, y) || Copied(Edge, x, y);
            for (u32 sample = 0; sample < SampleCount; ++sample) {
                assert(native_samples[index * SampleCount + sample] ==
                       Pack(active ? Expected[sample] : Sentinel));
            }
            assert(pixels[index] == (active ? Resolved : Sentinel));
        }
    }
    assert(scheduler.Shutdown(deadline));
    std::puts("PASS: exact retained multisample byte identities, resolved values and guard pixels");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CheckCPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: multisample depth/stencil render, sample readback and resolve "
                      "require a process-visible Metal 4 GPU.");
        } else {
            Run(argv[1], argv[2]);
        }
    }
}
