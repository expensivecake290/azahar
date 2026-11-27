#import <AppKit/AppKit.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include "core/frontend/emu_window.h"
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_present_window.h"

using namespace Metal;

namespace {

class Window final : public Frontend::EmuWindow {
public:
    using EmuWindow::PublishNativeSurface;

    void PollEvents() override {}
};

void CheckPresent(bool asynchronous) {
    Window window;
    Instance instance(0, false);
    Scheduler rendering(instance.GetContext());
    MemoryBudget budget(64 * 1024 * 1024);
    MemoryAllocator allocator(instance.GetDevice(), budget);
    RenderManager passes(instance, rendering, allocator);
    PresentWindow present(window, instance, rendering, budget, asynchronous);
    constexpr VideoCore::Extent extent{9, 7};

    std::array<std::optional<PresentWindow::Frame>, PresentWindow::FrameCount> held;
    for (auto& frame : held) {
        frame = present.GetRenderFrame(extent);
        assert(frame && frame->Extent().width == 9 && frame->Texture());
    }
    assert(held[0]->Texture() != held[1]->Texture() && held[0]->Texture() != held[2]->Texture());
    assert(!present.GetRenderFrame(extent, 1, std::chrono::steady_clock::now()));
    held[0].reset();
    assert(present.GetRenderFrame(extent, 1, std::chrono::steady_clock::now()));
    held = {};
    assert(!present.GetRenderFrame({0, 7}));
    assert(!present.GetRenderFrame(extent, 3));
    assert(!present.Present(std::move(*present.GetRenderFrame(extent)), 0));

    const u32 samples = instance.SupportsSampleCount(4) ? 4 : 1;
    auto frame = present.GetRenderFrame(extent, samples);
    assert(frame && frame->Samples() == samples);
    const auto pass = frame->MakePass(MTLClearColorMake(0.25, 0.50, 0.75, 1));
    assert(pass.color.target.texture && frame->Texture().sampleCount == 1);
    if (samples > 1) {
        assert(pass.color.resolve.texture == frame->Texture() &&
               pass.color.store_action == MTLStoreActionMultisampleResolve);
    }
    frame->Retain(rendering);
    passes.BeginRendering(pass);
    passes.EndRendering();

    constexpr std::size_t pitch = 256;
    const BufferRange range{0, pitch * extent.height};
    auto host = allocator.AllocateBuffer(range.length, MemoryPool::Readback);
    auto source = frame->Texture();
    rendering.Record([host = host.AcquireGPUAccess(range, BufferAccess::Write), source,
                      range](EncodingContext& context) mutable {
        auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                     "Read the real Metal frame resolve");
        const std::array uses{
            ResourceUse{source, BufferAccess::Read, MTLStageBlit},
            ResourceUse{host.GetBuffer(), BufferAccess::Write, MTLStageBlit, range}};
        context.Synchronize(encoder, uses);
        [encoder copyFromTexture:source
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(9, 7, 1)
                            toBuffer:host.GetBuffer()
                   destinationOffset:range.offset
              destinationBytesPerRow:pitch
            destinationBytesPerImage:range.length];
        context.Retain(std::move(host));
        context.EndPass(encoder);
    });
    assert(present.Present(std::move(*frame), rendering.Flush()));
    assert(present.WaitPresent(std::chrono::steady_clock::now() + std::chrono::seconds(20)) ==
           CompletionStatus::Complete);
    {
        const auto read = host.MapRead(range);
        for (u32 y = 0; y < extent.height; ++y) {
            for (u32 x = 0; x < extent.width; ++x) {
                const auto color = read.Bytes().subspan(y * pitch + x * 4, 4);
                assert(std::to_integer<u8>(color[0]) == 191);
                assert(std::to_integer<u8>(color[1]) == 128);
                assert(std::to_integer<u8>(color[2]) == 64);
                assert(std::to_integer<u8>(color[3]) == 255);
            }
        }
    }

    [NSApplication sharedApplication];
    NSWindow* display = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 80, 64)
                                                    styleMask:NSWindowStyleMaskBorderless
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
    display.contentView.wantsLayer = YES;
    CAMetalLayer* layer = [CAMetalLayer layer];
    display.contentView.layer = layer;
    [display orderFront:nil];
    Frontend::EmuWindow::NativeSurface surface{
        .handle = RetainLayer(layer), .width = 17, .height = 11, .scale = 1, .refresh_rate = 60};
    assert(window.PublishNativeSurface(surface, Layout::DefaultFrameLayout(17, 11, false, false)));
    present.NotifySurfaceChanged();
    for (const auto size : {extent, VideoCore::Extent{17, 11}}) {
        auto onscreen = present.GetRenderFrame(size);
        assert(onscreen);
        onscreen->Retain(rendering);
        passes.BeginRendering(onscreen->MakePass(MTLClearColorMake(0, 1, 0, 1)));
        passes.EndRendering();
        assert(present.Present(std::move(*onscreen), rendering.Flush()));
    }
    assert(present.WaitPresent(std::chrono::steady_clock::now() + std::chrono::seconds(20)) ==
           CompletionStatus::Complete);
    assert(layer.drawableSize.width == 17 && layer.drawableSize.height == 11);

    surface.suspended = true;
    assert(window.PublishNativeSurface(surface, Layout::DefaultFrameLayout(17, 11, false, false)));
    present.NotifySurfaceChanged();
    auto final = present.GetRenderFrame(extent);
    assert(final);
    final->Retain(rendering);
    passes.BeginRendering(final->MakePass(MTLClearColorMake(1, 0, 1, 1)));
    passes.EndRendering();
    assert(present.Present(std::move(*final), rendering.Flush()));
    present.Close();
    assert(!present.GetRenderFrame(extent));
    assert(!rendering.GetFault());
    [display orderOut:nil];
    std::printf("PASS physical frame pool, resolve, headless, scaled and matched surface with %s "
                "worker mode\n",
                asynchronous ? "asynchronous" : "synchronous");
}

} // Anonymous namespace

int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING physical Metal 4 frame pool, native color resolve, copy/scale and "
                      "worker lifecycle");
            return 2;
        }
        CheckPresent(false);
        CheckPresent(true);
    }
}
