#include <array>
#include <cassert>
#include <cstdio>
#include "video_core/renderer_metal/mtl_metalfx_spatial.h"

using namespace Metal;

namespace {

std::array<u8, 4> Pixel(Scheduler& scheduler, const MemoryAllocator& allocator,
                        id<MTLTexture> texture, u64 dependency, u32 x, u32 y) {
    constexpr BufferRange range{0, 256};
    auto buffer = allocator.AllocateBuffer(range.length, MemoryPool::Readback);
    scheduler.Record(
        [texture, x, y, range, access = buffer.AcquireGPUAccess(range, BufferAccess::Write)](
            EncodingContext& context) mutable {
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Read MetalFX Spatial output");
            const std::array uses{
                ResourceUse{texture, BufferAccess::Read, MTLStageBlit},
                ResourceUse{access.GetBuffer(), BufferAccess::Write, MTLStageBlit, range}};
            context.Synchronize(encoder, uses);
            [encoder copyFromTexture:texture
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(x, y, 0)
                              sourceSize:MTLSizeMake(1, 1, 1)
                                toBuffer:access.GetBuffer()
                       destinationOffset:0
                  destinationBytesPerRow:256
                destinationBytesPerImage:256];
            context.Retain(std::move(access));
            context.EndPass(encoder);
        });
    assert(scheduler.Finish({.wait_for_tick = dependency}) == CompletionStatus::Complete);
    const auto map = buffer.MapRead(range);
    const auto bytes = map.Bytes();
    return {std::to_integer<u8>(bytes[0]), std::to_integer<u8>(bytes[1]),
            std::to_integer<u8>(bytes[2]), std::to_integer<u8>(bytes[3])};
}

bool VerifyDevice() {
    Instance instance(0, false);
    auto& context = instance.GetContext();
    Scheduler scheduler(context);
    MemoryBudget budget(96 * 1024 * 1024);
    MemoryAllocator allocator(instance.GetDevice(), budget);
    RenderManager manager(instance, scheduler, allocator);
    TextureRuntime runtime(instance, scheduler, manager, allocator, budget);
    MetalFXSpatial spatial(context, scheduler, budget);

    const std::array blank{SpatialRequest{}};
    const auto disabled = spatial.Apply(blank, false, 0, {});
    assert(disabled.completion == 0 && disabled.results[0].status == SpatialStatus::Disabled);
    if (!spatial.IsSupported()) {
        const auto unsupported = spatial.Apply(blank, true, 0, {});
        assert(unsupported.completion == 0 &&
               unsupported.results[0].status == SpatialStatus::UnsupportedDevice);
        std::puts("PASS typed unsupported MetalFX bypass; PENDING a Spatial-capable Metal 4 GPU");
        return false;
    }
    assert(spatial.Apply(blank, true, 0, {}).results[0].status == SpatialStatus::InvalidDependency);

    VideoCore::SurfaceParams params;
    params.addr = 0x100000;
    params.width = 400;
    params.height = 240;
    params.pixel_format = VideoCore::PixelFormat::RGBA8;
    params.UpdateParams();

    constexpr std::array input_colors{Common::Vec4f{1, 0, 0, 1}, Common::Vec4f{0, 1, 0, 1},
                                      Common::Vec4f{0, 0, 1, 1}};
    std::array<TextureSurface, 3> inputs{TextureSurface(runtime, params),
                                         TextureSurface(runtime, params),
                                         TextureSurface(runtime, params)};
    std::array<SpatialRequest, 3> requests;
    const auto source_scope = context.AcquireWork();
    for (u32 i = 0; i < requests.size(); ++i) {
        const VideoCore::TextureClear clear{0, 0, params.GetRect(), {.color = input_colors[i]}};
        assert(runtime.ClearTexture(inputs[i], clear));
        requests[i] = {inputs[i].Resource(), {400, 240}, {721, 433}};
    }
    const auto produced = scheduler.Flush();

    DeviceContext foreign(instance.GetDevice());
    const auto foreign_result = spatial.Apply(requests, true, produced, foreign.AcquireWork());
    assert(!foreign_result.completion &&
           foreign_result.results[0].status == SpatialStatus::InvalidDependency);
    auto bad_format = requests[0];
    bad_format.output_format = MTLPixelFormatDepth32Float;
    const auto rejected = spatial.Apply(std::span{&bad_format, 1}, true, produced, source_scope);
    assert(!rejected.completion && rejected.results[0].status == SpatialStatus::UnsupportedFormat);

    const auto output = spatial.Apply(requests, true, produced, source_scope);
    assert(output.completion > produced && output.results.size() == 3);
    constexpr std::array<std::array<u8, 4>, 3> expected{std::array<u8, 4>{255, 0, 0, 255},
                                                        std::array<u8, 4>{0, 255, 0, 255},
                                                        std::array<u8, 4>{0, 0, 255, 255}};
    for (u32 i = 0; i < requests.size(); ++i) {
        const auto& result = output.results[i];
        assert(result.status == SpatialStatus::Queued && result.image);
        auto image = result.image.Texture();
        assert(image && image != inputs[i].Image() && image.width == 721 && image.height == 433);
        assert(image.storageMode == MTLStorageModePrivate);
        result.image.Retain(scheduler);
        assert(Pixel(scheduler, allocator, image, output.completion, 360, 216) == expected[i]);
        assert(Pixel(scheduler, allocator, inputs[i].Image(), output.completion, 200, 120) ==
               expected[i]);
    }

    auto resized = requests[0];
    resized.output = {733, 439};
    const auto next = spatial.Apply(std::span{&resized, 1}, true, produced, source_scope);
    assert(next.completion > output.completion && next.results[0].status == SpatialStatus::Queued);
    auto larger = next.results[0].image.Texture();
    assert(larger.width == 733 && larger.height == 439);
    assert(output.results[0].image.Texture().width == 721);
    next.results[0].image.Retain(scheduler);
    assert(Pixel(scheduler, allocator, larger, next.completion, 400, 240) == expected[0]);
    assert(!scheduler.GetFault());
    std::puts("PASS physical Metal 4 Spatial odd resize, per-screen output and unchanged source");
    return true;
}

} // Anonymous namespace

int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING real Metal 4 Spatial odd-size and per-screen texture output");
            return 2;
        }
        return VerifyDevice() ? 0 : 2;
    }
}
