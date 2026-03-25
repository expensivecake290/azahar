// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_present_window.h"

#if defined(__APPLE__)

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Metal/Metal.h>

#include "common/logging/log.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_metal/mt_instance.h"
#include "video_core/renderer_metal/mt_pipeline_cache.h"

namespace Metal {

struct PresentWindow::Impl {
    Frontend::EmuWindow& emu_window;
    const Instance& instance;
    const PipelineCache& pipeline_cache;
    NSView* host_view = nil;
    CAMetalLayer* metal_layer = nil;
    bool valid = false;

    Impl(Frontend::EmuWindow& emu_window_, const Instance& instance_,
         const PipelineCache& pipeline_cache_)
        : emu_window{emu_window_}, instance{instance_}, pipeline_cache{pipeline_cache_} {}
};

namespace {
static void UpdateDrawableSize(CAMetalLayer* metal_layer, NSView* host_view, float scale) {
    metal_layer.contentsScale = scale;
    metal_layer.frame = host_view.bounds;
    metal_layer.drawableSize =
        CGSizeMake(host_view.bounds.size.width * scale, host_view.bounds.size.height * scale);
}

} // Anonymous namespace

PresentWindow::PresentWindow(Frontend::EmuWindow& emu_window, const Instance& instance,
                             const PipelineCache& pipeline_cache)
    : impl{std::make_unique<Impl>(emu_window, instance, pipeline_cache)} {
    const auto& window_info = emu_window.GetWindowInfo();
    impl->host_view = (__bridge NSView*)window_info.render_view;

    if (!instance.IsValid() || impl->host_view == nil) {
        LOG_CRITICAL(Render_Metal, "Metal present window missing required Apple host view");
        return;
    }

    impl->metal_layer = [[CAMetalLayer alloc] init];
    impl->metal_layer.device = (__bridge id<MTLDevice>)instance.GetDevice();
    impl->metal_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    impl->metal_layer.framebufferOnly = NO;
    impl->metal_layer.opaque = YES;
    impl->metal_layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
    impl->metal_layer.contentsGravity = kCAGravityTopLeft;

    [impl->host_view setWantsLayer:YES];
    [impl->host_view.layer addSublayer:impl->metal_layer];
    UpdateDrawableSize(impl->metal_layer, impl->host_view, window_info.render_surface_scale);

    impl->valid = true;
}

PresentWindow::~PresentWindow() {
    if (!impl) {
        return;
    }
    [impl->metal_layer removeFromSuperlayer];
    [impl->metal_layer release];
}

void PresentWindow::NotifySurfaceChanged() {
    if (!impl || !impl->valid) {
        return;
    }
    const auto& window_info = impl->emu_window.GetWindowInfo();
    UpdateDrawableSize(impl->metal_layer, impl->host_view, window_info.render_surface_scale);
}

void PresentWindow::Present(const Layout::FramebufferLayout& layout, const Common::Vec4f& clear_color) {
    (void)layout;

    if (!impl || !impl->valid) {
        return;
    }

    const auto& window_info = impl->emu_window.GetWindowInfo();
    UpdateDrawableSize(impl->metal_layer, impl->host_view, window_info.render_surface_scale);

    id<CAMetalDrawable> drawable = [impl->metal_layer nextDrawable];
    if (drawable == nil) {
        LOG_ERROR(Render_Metal, "Failed to acquire CAMetalDrawable");
        return;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)impl->instance.GetDevice();
    id<MTL4CommandAllocator> allocator =
        (__bridge id<MTL4CommandAllocator>)impl->instance.GetCommandAllocator();
    id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)impl->instance.GetCommandQueue();
    id<MTLRenderPipelineState> pipeline =
        (__bridge id<MTLRenderPipelineState>)impl->pipeline_cache.GetPresentPipeline();

    id<MTL4CommandBuffer> command_buffer = [device newCommandBuffer];
    [command_buffer beginCommandBufferWithAllocator:allocator];

    MTL4RenderPassDescriptor* render_pass = [[MTL4RenderPassDescriptor alloc] init];
    render_pass.colorAttachments[0].texture = drawable.texture;
    render_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    render_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    render_pass.colorAttachments[0].clearColor =
        MTLClearColorMake(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    render_pass.renderTargetWidth = drawable.texture.width;
    render_pass.renderTargetHeight = drawable.texture.height;

    id<MTL4RenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:render_pass];
    if (encoder != nil) {
        if (pipeline != nil) {
            [encoder setRenderPipelineState:pipeline];
        }
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
    }

    [render_pass release];

    [command_buffer endCommandBuffer];
    id<MTL4CommandBuffer> buffers[] = {command_buffer};
    [queue commit:buffers count:1];
    [queue signalDrawable:drawable];
    [drawable present];
    [command_buffer release];
}

bool PresentWindow::IsValid() const {
    return impl && impl->valid;
}

} // namespace Metal

#endif
