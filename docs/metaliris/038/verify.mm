#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

constexpr u32 Width = 10;
constexpr u32 Height = 9;
constexpr MTLViewport Viewport{1, 2, 7, 5, 0, 1};
constexpr MTLScissorRect Scissor{2, 3, 4, 3};

struct alignas(8) TextureTransform {
    std::array<f32, 2> scale;
    std::array<f32, 2> offset;
};
static_assert(sizeof(TextureTransform) == 16 && offsetof(TextureTransform, offset) == 8);

constexpr std::array Transforms{
    TextureTransform{{1, 1}, {0, 0}},
    TextureTransform{{-0.5f, 0.75f}, {0.75f, 0.125f}},
    TextureTransform{{0, 0}, {1, 0}},
};

std::array<f32, 2> Expected(const TextureTransform& transform, u32 x, u32 y) {
    const double horizontal = (x + 0.5 - Viewport.originX) / Viewport.width;
    const double vertical = (y + 0.5 - Viewport.originY) / Viewport.height;
    return {static_cast<f32>(horizontal * transform.scale[0] + transform.offset[0]),
            static_cast<f32>(vertical * transform.scale[1] + transform.offset[1])};
}

void CPU() {
    const auto top = Expected(Transforms[0], 1, 2);
    const auto bottom = Expected(Transforms[0], 7, 6);
    assert(std::abs(top[0] - 1.0 / 14.0) < 1e-7);
    assert(std::abs(top[1] - 0.1) < 1e-7);
    assert(std::abs(bottom[0] - 13.0 / 14.0) < 1e-7);
    assert(std::abs(bottom[1] - 0.9) < 1e-7);
    const auto reversed = Expected(Transforms[1], 1, 2);
    assert(std::abs(reversed[0] - (0.75 - 1.0 / 28.0)) < 1e-7);
    assert(std::abs(reversed[1] - 0.2) < 1e-7);
    assert((Expected(Transforms[2], 7, 6) == std::array<f32, 2>{1, 0}));
    std::puts("PASS: independent full-screen triangle viewport and affine texture oracle");
}

void GPU(const char* vertex_path, const char* fragment_path) {
    Instance instance(0, false);
    auto device = instance.GetDevice();
    NSError* error = nil;
    auto vertex = [device newLibraryWithURL:[NSURL fileURLWithPath:@(vertex_path)] error:&error];
    vertex = CheckResource(vertex, "Load native full-screen vertex library", error);
    auto fragment = [device newLibraryWithURL:[NSURL fileURLWithPath:@(fragment_path)]
                                        error:&error];
    fragment = CheckResource(fragment, "Load native viewport fixture library", error);
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create native viewport fixture compiler", error);
    auto vs = [MTL4LibraryFunctionDescriptor new];
    vs.library = vertex;
    vs.name = @"full_screen_triangle_vert";
    auto fs = [MTL4LibraryFunctionDescriptor new];
    fs.library = fragment;
    fs.name = @"fixture_uv";
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vs;
    descriptor.fragmentFunctionDescriptor = fs;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    auto pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                               compilerTaskOptions:nil
                                                             error:&error];
    pipeline = CheckResource(pipeline, "Build native viewport fixture pipeline", error);
    auto texture =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:Width
                                                          height:Height
                                                       mipmapped:NO];
    texture.storageMode = MTLStorageModeShared;
    texture.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    texture.usage = MTLTextureUsageRenderTarget;
    auto image =
        CheckResource([device newTextureWithDescriptor:texture], "Allocate viewport fixture");

    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    const RenderPass clear{
        .color = {.target = {image}, .load_action = MTLLoadActionClear},
        .area = {0, 0, Width, Height},
        .width = Width,
        .height = Height,
        .clear_color = MTLClearColorMake(-1, -1, 0, 1),
    };

    for (const auto& transform : Transforms) {
        auto parameters = CheckResource([device newBufferWithBytes:&transform
                                                            length:sizeof(transform)
                                                           options:MTLResourceStorageModeShared],
                                        "Allocate viewport transform");
        auto args = [MTL4ArgumentTableDescriptor new];
        args.maxBufferBindCount = 1;
        auto table = [device newArgumentTableWithDescriptor:args error:&error];
        table = CheckResource(table, "Create viewport argument table", error);
        [table setAddress:parameters.gpuAddress atIndex:0];
        for (const bool clipped : {false, true}) {
            manager.BeginRendering(clear);
            manager.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
                ctx.Retain(table);
                ctx.Retain(pipeline);
                ctx.MakeResident(parameters);
                [encoder setViewport:Viewport];
                if (clipped) {
                    [encoder setScissorRect:Scissor];
                }
                [encoder setRenderPipelineState:pipeline];
                [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            });
            manager.EndRendering();
            assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
                   CompletionStatus::Complete);
            using Pixel = std::array<_Float16, 4>;
            std::array<Pixel, Width * Height> pixels{};
            [image getBytes:pixels.data()
                bytesPerRow:Width * sizeof(Pixel)
                 fromRegion:MTLRegionMake2D(0, 0, Width, Height)
                mipmapLevel:0];
            for (u32 y = 0; y < Height; ++y) {
                for (u32 x = 0; x < Width; ++x) {
                    const bool covered = clipped ? x >= 2 && x < 6 && y >= 3 && y < 6
                                                 : x >= 1 && x < 8 && y >= 2 && y < 7;
                    const auto want =
                        covered ? Expected(transform, x, y) : std::array<f32, 2>{-1, -1};
                    const auto& pixel = pixels[y * Width + x];
                    assert(std::abs(float(pixel[0]) - want[0]) < 0.00049f);
                    assert(std::abs(float(pixel[1]) - want[1]) < 0.00049f);
                    assert(pixel[2] == 0 && pixel[3] == 1);
                }
            }
        }
    }
    assert(scheduler.Shutdown(std::chrono::steady_clock::now() + 10s));
    std::puts("PASS: native full viewport coverage, restricted scissor and top-left UV");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: full-screen triangle viewport pixels need a native Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
