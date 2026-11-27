#import <AppKit/AppKit.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_swapchain.h"

using namespace Metal;

namespace {

void ClearDrawable(EncodingContext& context, id<MTLTexture> texture) {
    auto pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.50, 0.75, 1);
    auto encoder =
        CheckResource([context.GetCommandBuffer() renderCommandEncoderWithDescriptor:pass],
                      "Verify native drawable rendering");
    const ResourceUse use{texture, BufferAccess::Write, MTLStageFragment};
    context.Synchronize(encoder, std::span{&use, 1});
    context.EndPass(encoder);
}

void VerifyDevice() {
    [NSApplication sharedApplication];
    Instance instance(0, false);
    Scheduler scheduler(instance.GetContext());
    Swapchain chain(instance, scheduler);
    assert(chain.AcquireNextImage().status == DrawableStatus::NoSurface);

    Frontend::EmuWindow::NativeSurfaceSnapshot snapshot;
    snapshot.layout = Layout::DefaultFrameLayout(10, 7, false, false);
    assert(!chain.UpdateSurface(snapshot, {.frame_limit = -1}));
    assert(
        !chain.UpdateSurface(snapshot, {.frame_limit = std::numeric_limits<double>::quiet_NaN()}));
    assert(chain.UpdateSurface(snapshot, {}));
    assert(chain.GetPixelFormat() == MTLPixelFormatInvalid);
    assert(chain.GetExtent().width == 0);
    assert(chain.GetSurface()->IsHeadless());

    NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 64, 64)
                                                   styleMask:NSWindowStyleMaskBorderless
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    NSView* view = window.contentView;
    view.wantsLayer = YES;
    CAMetalLayer* layer = [CAMetalLayer layer];
    view.layer = layer;
    [window orderFront:nil];

    snapshot.surface = {
        .handle = RetainLayer(layer), .width = 64, .height = 64, .scale = 1, .refresh_rate = 60};
    snapshot.generation = 1;
    const PresentPacing fixed{.minimum_duration = 1.0 / 60};
    assert(chain.UpdateSurface(snapshot, fixed));
    assert(chain.GetImageCount() == 3 && chain.GetPresentMode() == PresentMode::Fifo);
    assert(chain.GetPixelFormat() == MTLPixelFormatBGRA8Unorm && layer.opaque && layer.colorspace);
    assert(layer.allowsNextDrawableTimeout && layer.displaySyncEnabled);
    snapshot.layout = Layout::DefaultFrameLayout(7, 10, false, true);
    assert(chain.UpdateSurface(snapshot, fixed));
    assert(chain.GetSurface()->GetSnapshot().layout.height == 10);

    std::vector<Swapchain::Drawable> held;
    for (u32 i = 0; i < 3; ++i) {
        auto value = chain.AcquireNextImage();
        assert(value.status == DrawableStatus::Ready && value.drawable);
        assert(value.drawable->Texture().width == 64);
        held.push_back(std::move(*value.drawable));
    }
    assert(chain.AcquireNextImage().status == DrawableStatus::Backpressure);
    auto stale = snapshot;
    snapshot.surface.width = 96;
    snapshot.surface.height = 80;
    snapshot.surface.scale = 2;
    snapshot.surface.refresh_rate = 120;
    snapshot.surface.fullscreen = true;
    ++snapshot.generation;
    assert(chain.UpdateSurface(snapshot, fixed));
    assert(!chain.UpdateSurface(stale, fixed));
    assert(!chain.Present(std::move(held.back()), ClearDrawable));
    held.clear();
    assert(layer.drawableSize.width == 96 && layer.drawableSize.height == 80);

    auto next = chain.AcquireNextImage();
    assert(next.status == DrawableStatus::Ready && next.drawable);
    assert(next.drawable->Extent().height == 80);
    const auto ticket = chain.Present(std::move(*next.drawable), ClearDrawable);
    assert(ticket && *ticket != 0);
    snapshot.surface.suspended = true;
    ++snapshot.generation;
    assert(chain.UpdateSurface(snapshot, fixed));
    assert(chain.AcquireNextImage().status == DrawableStatus::Suspended);
    chain.Close();
    assert(chain.AcquireNextImage().status == DrawableStatus::Closed);
    assert(!chain.UpdateSurface(snapshot, fixed));
    assert(scheduler.Wait(*ticket) == CompletionStatus::Complete);
    assert(!scheduler.GetFault());
    [window orderOut:nil];
    std::puts(
        "PASS physical native drawable pressure, resize, stale lease, encode, present and close");
}

} // Anonymous namespace

int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING live Metal 4 window with drawable exhaustion, resize, stale lease "
                      "and present");
            return 2;
        }
        VerifyDevice();
    }
}
