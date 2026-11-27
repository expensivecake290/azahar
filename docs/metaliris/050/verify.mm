#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

constexpr u32 SourceWidth = 11;
constexpr u32 SourceHeight = 7;
constexpr u32 PrefixWords = 4;
constexpr u32 AllocationWords = 64;
constexpr u32 GuardWord = 0xD3A7B1C5;
using Coordinate = std::array<u32, 2>;

struct alignas(8) Region {
    Coordinate origin;
    Coordinate extent;
    u32 stride;
    u32 capacity;
};
static_assert(sizeof(Region) == 24 && offsetof(Region, stride) == 16 &&
              offsetof(Region, capacity) == 20);

struct Sample {
    float depth;
    u32 stencil;
    u32 word;
};
constexpr std::array Samples{
    Sample{0.0f, 0, 0x00000000},
    Sample{0.5f, 0xA5, 0x7FFFFFA5},
    Sample{1.0f, 0xFF, 0xFFFFFFFF},
    Sample{0.25f, 3, 0x3FFFFF03},
    Sample{0.75f, 0x87, 0xBFFFFF87},
    Sample{0x1p-24f, 0x1C, 0x0000001C},
    Sample{0x1.fffffep-1f, 90, 0xFFFFFE5A},
};

constexpr Region Odd{{2, 1}, {7, 5}, 11, 55};
constexpr Region Partial{{2, 1}, {7, 5}, 11, 27};
constexpr Region Edge{{10, 6}, {1, 1}, 1, 1};
constexpr std::array Regions{
    Odd,
    Partial,
    Edge,
    Region{{0, 0}, {7, 0}, 11, 55},
    Region{{2, 1}, {7, 5}, 6, 55},
    Region{{std::numeric_limits<u32>::max(), 0}, {7, 6}, 7, 42},
};

bool Written(const Region& r, u32 row, u32 column) {
    return r.stride >= r.extent[0] && row < r.extent[1] && column < r.extent[0] &&
           u64(r.origin[0]) + column < SourceWidth && u64(r.origin[1]) + row < SourceHeight &&
           u64(r.stride) * row + column < r.capacity;
}

u32 Reference(const Sample& sample) {
    const auto depth = static_cast<u32>(static_cast<float>(double(sample.depth) * 16777215.0));
    return 256u * depth + sample.stencil;
}

u32 At(u32 x, u32 y) {
    return y == 1 && x >= 2 && x < 2 + Samples.size() ? Samples[x - 2].word : Samples[1].word;
}

void CPU() {
    for (const auto& sample : Samples) {
        assert(Reference(sample) == sample.word);
    }
    assert(Written(Odd, 4, 6));
    assert(!Written(Odd, 0, 7));
    assert(Written(Partial, 2, 4));
    assert(!Written(Partial, 2, 5));
    assert(Written(Edge, 0, 0));
    assert(!Written(Edge, 0, 1));
    for (const auto& invalid : std::span(Regions).subspan(3)) {
        assert(!Written(invalid, 0, 0));
    }
    std::puts("PASS: independent D24 truncation, stencil packing, row stride and capacity oracles");
}

void GPU(const char* path) {
    Instance instance(0, false);
    auto device = instance.GetDevice();
    auto desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:SourceWidth
                                                          height:SourceHeight
                                                       mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.usage =
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    auto depth =
        CheckResource([device newTextureWithDescriptor:desc], "Allocate D24 readback source");
    auto stencil = CheckResource([depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8],
                                 "Create S8 readback view");
    NSError* error = nil;
    auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
    library = CheckResource(library, "Load native D24 staging shader", error);
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create D24 staging compiler", error);
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"depth_to_buffer_comp";
    auto compute = [MTL4ComputePipelineDescriptor new];
    compute.computeFunctionDescriptor = function;
    auto pipeline = [compiler newComputePipelineStateWithDescriptor:compute
                                                compilerTaskOptions:nil
                                                              error:&error];
    pipeline = CheckResource(pipeline, "Compile D24 staging shader", error);

    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    RenderPass clear{
        .depth = {.target = {depth}, .load_action = MTLLoadActionClear},
        .stencil = {.target = {depth}, .load_action = MTLLoadActionClear},
        .area = {0, 0, SourceWidth, SourceHeight},
        .width = SourceWidth,
        .height = SourceHeight,
        .clear_depth = Samples[1].depth,
        .clear_stencil = Samples[1].stencil,
    };
    manager.BeginRendering(clear);
    manager.EndRendering();
    for (u32 i = 0; i < Samples.size(); ++i) {
        clear.area = {2 + i, 1, 1, 1};
        clear.clear_depth = Samples[i].depth;
        clear.clear_stencil = Samples[i].stencil;
        manager.BeginRendering(clear);
        manager.EndRendering();
    }

    for (const auto& region : Regions) {
        std::array<u32, AllocationWords> initial{};
        initial.fill(GuardWord);
        const auto options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        auto output = CheckResource([device newBufferWithBytes:initial.data()
                                                        length:sizeof(initial)
                                                       options:options],
                                    "Allocate padded depth readback output");
        auto parameters = CheckResource([device newBufferWithBytes:&region
                                                            length:sizeof(region)
                                                           options:options],
                                        "Allocate depth readback region");
        auto args = [MTL4ArgumentTableDescriptor new];
        args.maxBufferBindCount = 2;
        args.maxTextureBindCount = 2;
        auto table = [device newArgumentTableWithDescriptor:args error:&error];
        table = CheckResource(table, "Build D24 readback argument table", error);
        [table setAddress:parameters.gpuAddress atIndex:0];
        [table setAddress:output.gpuAddress + PrefixWords * sizeof(u32) atIndex:1];
        [table setTexture:depth.gpuResourceID atIndex:0];
        [table setTexture:stencil.gpuResourceID atIndex:1];

        scheduler.Record([=](EncodingContext& ctx) {
            const std::array uses{
                ResourceUse{depth, BufferAccess::Read, MTLStageDispatch},
                ResourceUse{stencil, BufferAccess::Read, MTLStageDispatch, {}, depth},
                ResourceUse{parameters, BufferAccess::Read, MTLStageDispatch},
                ResourceUse{output,
                            BufferAccess::Write,
                            MTLStageDispatch,
                            {PrefixWords * sizeof(u32), region.capacity * sizeof(u32)}},
            };
            auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
            ctx.Synchronize(encoder, uses);
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreadgroups:MTLSizeMake(2, 2, 1)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            ctx.Retain(table);
            ctx.Retain(pipeline);
            ctx.EndPass(encoder);
        });
        assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
               CompletionStatus::Complete);
        const auto* data = static_cast<const u32*>(output.contents);
        for (u32 word = 0; word < AllocationWords; ++word) {
            u32 value = GuardWord;
            if (word >= PrefixWords) {
                const u32 index = word - PrefixWords;
                const u32 row = index / region.stride;
                const u32 column = index % region.stride;
                if (Written(region, row, column)) {
                    value = At(region.origin[0] + column, region.origin[1] + row);
                }
            }
            assert(data[word] == value);
        }
    }
    assert(scheduler.Shutdown(std::chrono::steady_clock::now() + 10s));
    std::puts("PASS: exact native D24S8 packed words and guard regions");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts(
                "PENDING: exact native D24S8 readback requires a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1]);
        }
    }
}
