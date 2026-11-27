#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

constexpr u32 SrcWidth = 12;
constexpr u32 SrcHeight = 8;
constexpr u32 DstWidth = 11;
constexpr u32 DstHeight = 7;

using Pixel = std::array<u8, 4>;
using Coordinate = std::array<u32, 2>;

struct alignas(8) CopyRegion {
    Coordinate source;
    Coordinate destination;
    Coordinate extent;
};

static_assert(offsetof(CopyRegion, source) == 0 && offsetof(CopyRegion, destination) == 8 &&
              offsetof(CopyRegion, extent) == 16 && sizeof(CopyRegion) == 24);

constexpr Pixel Sentinel{0xCD, 0xB3, 0xD2, 0xA9};
constexpr CopyRegion OddCopy{{2, 1}, {3, 1}, {7, 5}};
constexpr CopyRegion EdgeCopy{{11, 7}, {10, 6}, {1, 1}};

struct DepthStencilCase {
    float depth;
    u8 stencil;
    Pixel expected;
};

constexpr std::array Cases{
    DepthStencilCase{0.0f, 0x00, {0x00, 0x00, 0x00, 0x00}},
    DepthStencilCase{0.5f, 0xFF, {0xFF, 0x80, 0x00, 0x00}},
    DepthStencilCase{1.0f, 0x7E, {0x7E, 0xFF, 0xFF, 0xFF}},
    DepthStencilCase{0x1p-24f, 0xAB, {0xAB, 0x00, 0x00, 0x01}},
    DepthStencilCase{0x1.fffffep-1f, 0x34, {0x34, 0xFF, 0xFF, 0xFF}},
    DepthStencilCase{float(0x123456) * 0x1p-24f, 0x69, {0x69, 0x12, 0x34, 0x56}},
    DepthStencilCase{float(0xFEDCBA) * 0x1p-24f, 0xDF, {0xDF, 0xFE, 0xDC, 0xBA}},
};

Pixel Reference(const DepthStencilCase& sample) {
    const double depth = sample.depth;
    const u64 fixed = std::min<u64>(0xFFFFFFFF, static_cast<u64>(std::ldexp(depth, 32)));
    return {sample.stencil, u8(fixed / (1ULL << 24)), u8(fixed / (1ULL << 16)),
            u8(fixed / (1ULL << 8))};
}

bool Writable(const CopyRegion& copy, Coordinate id) {
    return id[0] < copy.extent[0] && id[1] < copy.extent[1] &&
           u64(copy.source[0]) + id[0] < SrcWidth && u64(copy.source[1]) + id[1] < SrcHeight &&
           u64(copy.destination[0]) + id[0] < DstWidth &&
           u64(copy.destination[1]) + id[1] < DstHeight;
}

void CheckCPU() {
    for (const auto& sample : Cases) {
        assert(Reference(sample) == sample.expected);
    }
    for (u32 y = 0; y < 16; ++y) {
        for (u32 x = 0; x < 16; ++x) {
            assert(Writable(OddCopy, {x, y}) == (x < 7 && y < 5));
            assert(Writable(EdgeCopy, {x, y}) == (x == 0 && y == 0));
            assert(!Writable({{0, 0}, {0, 0}, {0, 8}}, {x, y}));
            assert(!Writable({{std::numeric_limits<u32>::max(), 0}, {0, 0}, {8, 8}}, {x, y}));
            assert(!Writable({{0, 0}, {1, std::numeric_limits<u32>::max()}, {8, 8}}, {x, y}));
        }
    }
    std::puts("PASS: independent depth/stencil byte, endpoint and copy bounds oracles");
}

void PrepareClear(RenderManager& manager, RenderPass pass) {
    manager.BeginRendering(pass);
    manager.EndRendering();
}

void Run(const char* path) {
    Instance instance(0, false);
    auto device = instance.GetDevice();
    auto depth_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:SrcWidth
                                                          height:SrcHeight
                                                       mipmapped:NO];
    depth_desc.storageMode = MTLStorageModePrivate;
    depth_desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    depth_desc.usage =
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    auto depth = CheckResource([device newTextureWithDescriptor:depth_desc],
                               "Allocate native D24S8 fixture depth");
    auto stencil = CheckResource([depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8],
                                 "Create native stencil view");

    auto color_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:DstWidth
                                                          height:DstHeight
                                                       mipmapped:NO];
    color_desc.storageMode = MTLStorageModeShared;
    color_desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    color_desc.usage = MTLTextureUsageShaderWrite;
    auto color = CheckResource([device newTextureWithDescriptor:color_desc],
                               "Allocate native RGBA8 fixture color");
    std::array<Pixel, DstWidth * DstHeight> initial{};
    initial.fill(Sentinel);
    [color replaceRegion:MTLRegionMake2D(0, 0, DstWidth, DstHeight)
             mipmapLevel:0
               withBytes:initial.data()
             bytesPerRow:DstWidth * sizeof(Pixel)];

    NSError* error = nil;
    auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
    library = CheckResource(library, "Load native D24S8 shader", error);
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create native D24S8 fixture compiler", error);
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"d24s8_to_rgba8_comp";
    auto compute_descriptor = [MTL4ComputePipelineDescriptor new];
    compute_descriptor.computeFunctionDescriptor = function;
    auto pipeline = [compiler newComputePipelineStateWithDescriptor:compute_descriptor
                                                compilerTaskOptions:nil
                                                              error:&error];
    pipeline = CheckResource(pipeline, "Build native D24S8 compute pipeline", error);

    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    RenderPass clear{
        .depth = {.target = {depth}, .load_action = MTLLoadActionClear},
        .stencil = {.target = {depth}, .load_action = MTLLoadActionClear},
        .area = {0, 0, SrcWidth, SrcHeight},
        .width = SrcWidth,
        .height = SrcHeight,
        .clear_depth = Cases[1].depth,
        .clear_stencil = Cases[1].stencil,
    };
    PrepareClear(manager, clear);
    for (u32 i = 0; i < Cases.size(); ++i) {
        clear.area = {2 + i, 1, 1, 1};
        clear.clear_depth = Cases[i].depth;
        clear.clear_stencil = Cases[i].stencil;
        PrepareClear(manager, clear);
    }
    clear.area = {11, 7, 1, 1};
    clear.clear_depth = Cases[5].depth;
    clear.clear_stencil = Cases[5].stencil;
    PrepareClear(manager, clear);

    const auto dispatch = [&](const CopyRegion& region) {
        auto buffer =
            CheckResource([device newBufferWithBytes:&region
                                              length:sizeof(region)
                                             options:MTLResourceStorageModeShared |
                                                     MTLResourceHazardTrackingModeUntracked],
                          "Allocate native copy region constants");
        auto descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 1;
        descriptor.maxTextureBindCount = 3;
        auto arguments = [device newArgumentTableWithDescriptor:descriptor error:&error];
        arguments = CheckResource(arguments, "Create native D24S8 argument table", error);
        [arguments setAddress:buffer.gpuAddress atIndex:0];
        [arguments setTexture:depth.gpuResourceID atIndex:0];
        [arguments setTexture:stencil.gpuResourceID atIndex:1];
        [arguments setTexture:color.gpuResourceID atIndex:2];
        scheduler.Record([=](EncodingContext& ctx) {
            const std::array uses{
                ResourceUse{depth, BufferAccess::Read, MTLStageDispatch},
                ResourceUse{stencil, BufferAccess::Read, MTLStageDispatch, {}, depth},
                ResourceUse{color, BufferAccess::Write, MTLStageDispatch},
                ResourceUse{buffer, BufferAccess::Read, MTLStageDispatch},
            };
            auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
            ctx.Synchronize(encoder, uses);
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:arguments];
            [encoder dispatchThreadgroups:MTLSizeMake(2, 2, 1)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            ctx.Retain(pipeline);
            ctx.Retain(arguments);
            ctx.EndPass(encoder);
        });
    };

    dispatch(OddCopy);
    dispatch(EdgeCopy);
    dispatch({{0, 0}, {0, 0}, {0, 8}});
    dispatch({{std::numeric_limits<u32>::max(), 0}, {0, 0}, {8, 8}});
    dispatch({{0, 0}, {0, std::numeric_limits<u32>::max()}, {8, 8}});
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    assert(scheduler.Finish({}, deadline) == CompletionStatus::Complete);
    std::array<Pixel, DstWidth * DstHeight> actual{};
    [color getBytes:actual.data()
        bytesPerRow:DstWidth * sizeof(Pixel)
         fromRegion:MTLRegionMake2D(0, 0, DstWidth, DstHeight)
        mipmapLevel:0];
    for (u32 y = 0; y < DstHeight; ++y) {
        for (u32 x = 0; x < DstWidth; ++x) {
            Pixel expected = Sentinel;
            if (x == DstWidth - 1 && y == DstHeight - 1) {
                expected = Cases[5].expected;
            } else if (y >= 1 && y <= 5 && x >= 3 && x <= 9) {
                expected = Cases[y == 1 ? x - 3 : 1].expected;
            }
            assert(actual[y * DstWidth + x] == expected);
        }
    }
    assert(scheduler.Shutdown(deadline));
    std::puts("PASS: exact native D24S8 byte conversion, odd/edge copy and untouched destination");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2);
        CheckCPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native Depth32Float_Stencil8->RGBA8 test needs a Metal 4 GPU.");
        } else {
            Run(argv[1]);
        }
    }
}
