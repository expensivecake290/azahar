// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CATransaction.h>
#include <dispatch/dispatch.h>

#include <cmath>
#include <utility>
#include "video_core/renderer_metal/mtl_platform.h"
#include "video_core/renderer_metal/mtl_surface_adapter.h"

namespace Metal {
namespace {

double RefreshRate(NSScreen* screen) {
    if (!screen) {
        return 0.;
    }
    NSNumber* display = screen.deviceDescription[@"NSScreenNumber"];
    if (display) {
        if (auto mode = CGDisplayCopyDisplayMode(display.unsignedIntValue)) {
            const double rate = CGDisplayModeGetRefreshRate(mode);
            CGDisplayModeRelease(mode);
            if (std::isfinite(rate) && rate > 0.) {
                return rate;
            }
        }
    }
    return static_cast<double>(std::max<NSInteger>(0, screen.maximumFramesPerSecond));
}

} // Anonymous namespace

struct SurfaceAdapter::State {
    void Disconnect() {
        auto* attached_view = view;
        if (installed && attached_view && attached_view.layer == layer) {
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            attached_view.layer = former_layer;
            attached_view.wantsLayer = former_wants_layer;
            [CATransaction commit];
        }

        view = nil;
        layer = nil;
        former_layer = nil;
        owner.reset();
        installed = false;
        former_wants_layer = false;
    }

    __weak NSView* view = nil;
    CAMetalLayer* layer = nil;
    CALayer* former_layer = nil;
    std::shared_ptr<void> owner;
    bool installed{};
    bool former_wants_layer{};
};

SurfaceAdapter::SurfaceAdapter() : state(std::make_shared<State>()) {}

SurfaceAdapter::~SurfaceAdapter() {
    if ([NSThread isMainThread]) {
        state->Disconnect();
        return;
    }
    const auto cleanup = std::move(state);
    dispatch_async(dispatch_get_main_queue(), ^{
      cleanup->Disconnect();
    });
}

bool SurfaceAdapter::Attach(void* native_view) {
    if (![NSThread isMainThread] || !native_view) {
        return false;
    }
    id candidate = (__bridge id)native_view;
    if (![candidate isKindOfClass:NSView.class]) {
        return false;
    }
    NSView* view = candidate;
    CALayer* current = view.layer;
    if (state->view == view && current == state->layer) {
        return true;
    }

    const bool adopt = [current isKindOfClass:CAMetalLayer.class];
    auto* metal_layer = adopt ? (CAMetalLayer*)current : [CAMetalLayer layer];
    if (!metal_layer) {
        return false;
    }
    auto owner = RetainLayer(metal_layer);

    state->Disconnect();
    state->view = view;
    state->layer = metal_layer;
    state->owner = std::move(owner);
    state->installed = !adopt;
    if (!adopt) {
        state->former_wants_layer = view.wantsLayer;
        state->former_layer = view.layer;
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        view.wantsLayer = YES;
        view.layer = metal_layer;
        [CATransaction commit];
    }
    return true;
}

std::optional<Frontend::EmuWindow::NativeSurface> SurfaceAdapter::Capture(
    SurfaceOptions options) const {
    if (![NSThread isMainThread]) {
        return std::nullopt;
    }
    Frontend::EmuWindow::NativeSurface result{
        .fullscreen = options.fullscreen,
        .portrait = options.portrait,
        .suspended = options.suspended,
    };
    auto* view = state->view;
    if (!view) {
        return result;
    }
    if (view.layer != state->layer) {
        return std::nullopt;
    }

    auto* window = view.window;
    const auto bounds = [view convertRectToBacking:view.bounds];
    const auto extent = DrawableExtentForBackingSize(bounds.size.width, bounds.size.height);
    const auto scale = window ? window.backingScaleFactor : state->layer.contentsScale;
    if (!extent || !std::isfinite(scale) || scale <= 0. ||
        scale > std::numeric_limits<f32>::max()) {
        return std::nullopt;
    }
    result.handle = state->owner;
    result.width = extent->width;
    result.height = extent->height;
    result.scale = static_cast<f32>(scale);
    result.refresh_rate = RefreshRate(window.screen);
    result.fullscreen |= (window.styleMask & NSWindowStyleMaskFullScreen) != 0;
    result.suspended |= !window || !window.visible || window.miniaturized ||
                        [view isHiddenOrHasHiddenAncestor] || !extent->width;
    return result;
}

bool SurfaceAdapter::Detach() {
    if (![NSThread isMainThread]) {
        return false;
    }
    state->Disconnect();
    return true;
}

} // namespace Metal
