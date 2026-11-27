// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <CoreGraphics/CoreGraphics.h>
#import <QuartzCore/CATransaction.h>
#include <TargetConditionals.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include "common/microprofile.h"
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_swapchain.h"

namespace Metal {
namespace {

MICROPROFILE_DEFINE(Metal_AcquireDrawable, "Metal", "Acquire Drawable", MP_RGB(185, 66, 245));
MICROPROFILE_DEFINE(Metal_PresentDrawable, "Metal", "Submit Drawable", MP_RGB(66, 185, 245));

constexpr u32 MaxDrawableDimension = 16384;
constexpr MTLPixelFormat DrawableFormat = MTLPixelFormatBGRA8Unorm;
constexpr bool SupportsImmediate = TARGET_OS_OSX != 0;

bool IsValidSurface(const Frontend::EmuWindow::NativeSurface& surface) {
    return std::isfinite(surface.scale) && surface.scale > 0.f &&
           std::isfinite(surface.refresh_rate) && surface.refresh_rate >= 0. &&
           (surface.width == 0) == (surface.height == 0);
}

void ConfigureLayer(CAMetalLayer* layer, id<MTLDevice> device, VideoCore::Extent extent, f32 scale,
                    PresentMode mode, u32 drawable_count) {
    auto colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!colorspace) {
        throw std::runtime_error("Unable to create the Metal drawable color space");
    }

    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    layer.device = device;
    layer.pixelFormat = DrawableFormat;
    layer.colorspace = colorspace;
    layer.opaque = YES;
    layer.framebufferOnly = NO;
    layer.presentsWithTransaction = NO;
    layer.allowsNextDrawableTimeout = YES;
    layer.maximumDrawableCount = drawable_count;
#if TARGET_OS_OSX
    layer.displaySyncEnabled = mode != PresentMode::Immediate;
#endif
    layer.contentsScale = scale;
    layer.drawableSize = CGSizeMake(extent.width, extent.height);
    [CATransaction commit];
    CFRelease(colorspace);
}

} // Anonymous namespace

struct Swapchain::Generation {
    Surface surface;
    DeviceContext::WorkLease device;
    PresentPacing pacing;
    VideoCore::Extent extent;
    PresentMode mode;
    u32 drawable_count;
};

struct Swapchain::State {
    State(Instance& instance, Scheduler& scheduler)
        : context(instance.GetContext()), scheduler(scheduler) {}

    DeviceContext& context;
    Scheduler& scheduler;
    std::shared_ptr<const Generation> current;
    Layout::FramebufferLayout layout{};
    bool closed{};
};

Swapchain::Drawable::Drawable(id<CAMetalDrawable> native, VideoCore::Extent extent,
                              std::shared_ptr<const Generation> generation)
    : native(native), extent(extent), generation(std::move(generation)) {}

Swapchain::Swapchain(Instance& instance, Scheduler& scheduler)
    : state(std::make_unique<State>(instance, scheduler)) {}

Swapchain::~Swapchain() {
    Close();
}

bool Swapchain::UpdateSurface(Frontend::EmuWindow::NativeSurfaceSnapshot snapshot,
                              PresentPacing pacing) {
    if (state->closed || !IsValidSurface(snapshot.surface) || !std::isfinite(pacing.frame_limit) ||
        pacing.frame_limit < 0. ||
        (pacing.minimum_duration &&
         (!std::isfinite(*pacing.minimum_duration) || *pacing.minimum_duration < 0.))) {
        return false;
    }
    const auto previous = state->current;
    if (previous) {
        const auto& old = previous->surface.GetSnapshot();
        if (snapshot.generation < old.generation) {
            return false;
        }
        if (snapshot.generation == old.generation) {
            const auto& handle = snapshot.surface.handle;
            const auto& old_handle = old.surface.handle;
            if (snapshot.surface != old.surface || handle.owner_before(old_handle) ||
                old_handle.owner_before(handle)) {
                return false;
            }
            if (pacing == previous->pacing &&
                state->context.GetLeaseStatus(previous->device) == WorkLeaseStatus::Valid) {
                state->layout = snapshot.layout;
                return true;
            }
        }
    }

    const auto layout = snapshot.layout;
    std::optional<Surface> surface;
    try {
        surface.emplace(std::move(snapshot));
    } catch (const std::invalid_argument&) {
        return false;
    }

    const auto lease = state->context.AcquireWork();
    if (state->context.GetLeaseStatus(lease) != WorkLeaseStatus::Valid) {
        return false;
    }
    const auto mode = pacing.SelectMode(true, SupportsImmediate);
    const u32 count = surface->IsDrawableAvailable() ? (mode == PresentMode::Immediate ? 2 : 3) : 0;
    const auto& request = surface->GetSnapshot().surface;
    const VideoCore::Extent extent{count ? std::min(request.width, MaxDrawableDimension) : 0,
                                   count ? std::min(request.height, MaxDrawableDimension) : 0};
    const auto next = std::make_shared<const Generation>(
        Generation{std::move(*surface), lease, pacing, extent, mode, count});
    if (count) {
        ConfigureLayer(next->surface.GetLayer(), state->context.GetDevice(), extent, request.scale,
                       mode, count);
    }
    state->layout = layout;
    state->current = next;
    return true;
}

Swapchain::Acquisition Swapchain::AcquireNextImage() {
    if (state->closed) {
        return {DrawableStatus::Closed, {}};
    }
    const auto active = state->current;
    if (!active || active->surface.IsHeadless()) {
        return {DrawableStatus::NoSurface, {}};
    }
    if (!active->drawable_count) {
        return {DrawableStatus::Suspended, {}};
    }
    if (state->context.GetLeaseStatus(active->device) != WorkLeaseStatus::Valid) {
        return {DrawableStatus::DeviceUnavailable, {}};
    }

    MICROPROFILE_SCOPE(Metal_AcquireDrawable);
    @autoreleasepool {
        auto layer = active->surface.GetLayer();
        auto drawable = [layer nextDrawable];
        if (!drawable) {
            return {DrawableStatus::Backpressure, {}};
        }
        auto image = drawable.texture;
        if (!image || drawable.layer != layer || image.width != active->extent.width ||
            image.height != active->extent.height || image.pixelFormat != DrawableFormat) {
            return {DrawableStatus::SurfaceChanged, {}};
        }
        if (state->context.GetLeaseStatus(active->device) != WorkLeaseStatus::Valid ||
            image.device != state->context.GetDevice()) {
            return {DrawableStatus::DeviceUnavailable, {}};
        }
        return {DrawableStatus::Ready, Drawable(drawable, active->extent, active)};
    }
}

std::optional<u64> Swapchain::Present(Drawable drawable, DrawableEncoder encode,
                                      u64 wait_for_tick) {
    const auto active = state->current;
    if (state->closed || !active || active != drawable.generation || !drawable.native || !encode ||
        state->context.GetLeaseStatus(active->device) != WorkLeaseStatus::Valid ||
        wait_for_tick >= state->scheduler.CurrentTick()) {
        return {};
    }

    MICROPROFILE_SCOPE(Metal_PresentDrawable);
    const auto native = drawable.native;
    const auto duration =
        active->mode == PresentMode::Fifo ? active->pacing.minimum_duration : std::nullopt;
    state->scheduler.SetQueue(QueueType::Graphics);
    state->scheduler.Record([native, active, encode = std::move(encode)](EncodingContext& context) {
        if (!context.OwnsWork(active->device)) {
            throw std::runtime_error("Metal drawable belongs to a retired device generation");
        }
        context.Retain(active->surface.GetSnapshot().surface.handle);
        context.Retain(native);
        const auto texture = native.texture;
        context.MakeResident(texture);
        encode(context, texture);
    });
    return state->scheduler.Flush({wait_for_tick, native, duration});
}

std::optional<Surface> Swapchain::GetSurface() const {
    if (!state->current) {
        return {};
    }
    auto snapshot = state->current->surface.GetSnapshot();
    snapshot.layout = state->layout;
    return Surface(std::move(snapshot));
}

VideoCore::Extent Swapchain::GetExtent() const noexcept {
    return state->current ? state->current->extent : VideoCore::Extent{0, 0};
}

MTLPixelFormat Swapchain::GetPixelFormat() const noexcept {
    return GetImageCount() ? DrawableFormat : MTLPixelFormatInvalid;
}

u32 Swapchain::GetImageCount() const noexcept {
    return state->current ? state->current->drawable_count : 0;
}

PresentMode Swapchain::GetPresentMode() const noexcept {
    return state->current ? state->current->mode : PresentMode::Fifo;
}

void Swapchain::Close() {
    state->closed = true;
    state->current.reset();
}

} // namespace Metal
