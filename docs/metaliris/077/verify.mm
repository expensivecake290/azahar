#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cassert>
#include <cstdio>
#include <thread>
#include "video_core/renderer_metal/mtl_platform.h"
#include "video_core/renderer_metal/mtl_surface_adapter.h"

using namespace Metal;

namespace {

void VerifyLayers() {
    SurfaceAdapter adapter;
    const auto headless = adapter.Capture();
    assert(headless && headless->IsHeadless());
    assert(adapter.Detach() && !adapter.Attach(nullptr));
    const auto wrong = [NSObject new];
    assert(!adapter.Attach((__bridge void*)wrong));

    auto* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 91, 57)];
    view.wantsLayer = YES;
    auto* former = [CALayer layer];
    view.layer = former;
    assert(adapter.Attach((__bridge void*)view));
    auto* native_layer = view.layer;
    assert(native_layer != former && [native_layer isKindOfClass:CAMetalLayer.class]);

    const auto first = adapter.Capture({.portrait = true});
    assert(first && first->width == 91 && first->height == 57 && first->portrait &&
           first->suspended && !first->IsHeadless());
    assert(adapter.Attach((__bridge void*)view));
    const auto repeated = adapter.Capture();
    assert(repeated && repeated->handle == first->handle &&
           !repeated->handle.owner_before(first->handle) &&
           !first->handle.owner_before(repeated->handle));
    Surface retained({.surface = *first, .generation = 19});
    assert(retained.GetLayer() == native_layer);
    assert(!retained.IsDrawableAvailable());

    std::thread forbidden([&] {
        @autoreleasepool {
            assert(!adapter.Capture());
            assert(!adapter.Attach((__bridge void*)view));
            assert(!adapter.Detach());
        }
    });
    forbidden.join();

    [view setFrameSize:NSMakeSize(123, 51)];
    const auto resized = adapter.Capture({.fullscreen = true});
    assert(resized && resized->width == 123 && resized->height == 51 && resized->fullscreen);
    assert(resized->handle == first->handle);

    auto* externally_replaced = [CALayer layer];
    view.layer = externally_replaced;
    assert(!adapter.Capture());
    assert(adapter.Detach() && view.layer == externally_replaced);
    assert(retained.GetLayer() == native_layer);
    assert(adapter.Capture()->IsHeadless());

    auto* external_metal = [CAMetalLayer layer];
    view.layer = external_metal;
    assert(adapter.Attach((__bridge void*)view));
    assert(view.layer == external_metal && adapter.Detach() && view.layer == external_metal);

    view.layer = former;
    auto release = std::make_unique<SurfaceAdapter>();
    assert(release->Attach((__bridge void*)view));
    auto* owned = view.layer;
    std::thread destroy([lease = std::move(release)]() mutable { lease.reset(); });
    destroy.join();
    assert(view.layer == owned);
    for (u32 step = 0; step < 5 && view.layer == owned; ++step) {
        [NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.04]];
    }
    assert(view.layer == former);
    std::puts("PASS AppKit main-thread attach, adoption, detach, off-main rejection, queued "
              "destruction and retained surface identity");
}

void VerifyDisplay() {
    [NSApplication sharedApplication];
    auto* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 101, 57)
                                               styleMask:NSWindowStyleMaskBorderless
                                                 backing:NSBackingStoreBuffered
                                                   defer:NO];
    auto* view = window.contentView;
    SurfaceAdapter adapter;
    assert(adapter.Attach((__bridge void*)view));
    [window orderFront:nil];
    const auto onscreen = adapter.Capture();
    assert(onscreen && onscreen->IsDrawableAvailable());
    const auto expected = [view convertRectToBacking:view.bounds];
    assert(onscreen->width == static_cast<u32>(std::round(expected.size.width)) &&
           onscreen->height == static_cast<u32>(std::round(expected.size.height)));
    assert(onscreen->scale == window.backingScaleFactor);
    assert(onscreen->refresh_rate > 0);
    const auto suspended = adapter.Capture({.suspended = true});
    assert(suspended && !suspended->IsDrawableAvailable());
    [window orderOut:nil];
    assert(!adapter.Capture()->IsDrawableAvailable());
    assert(adapter.Detach());
    std::puts("PASS real window backing size, display refresh and suspension");
}

} // Anonymous namespace

int main() {
    @autoreleasepool {
        VerifyLayers();
        const auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING live Apple display Retina/refresh/fullscreen and native "
                      "presentation retirement");
            return 2;
        }
        VerifyDisplay();
    }
}
