// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_present_window.h"

#if defined(__APPLE__)

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <simd/simd.h>

#include <array>

#include "common/logging/log.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/renderer_metal/mt_instance.h"
#include "video_core/renderer_metal/mt_pipeline_cache.h"

namespace Metal {

namespace {

enum class PresentMode : u32 {
    Mono = 0,
    Anaglyph = 1,
    Interlaced = 2,
    ReverseInterlaced = 3,
};

struct PresentVertex {
    simd_float2 position;
    simd_float2 tex_coord;
};

struct PresentFragmentUniforms {
    float opacity;
    u32 mode;
};

struct ScreenTexture {
    id<MTLTexture> texture = nil;
    u32 width = 0;
    u32 height = 0;
};

static void UpdateDrawableSize(CAMetalLayer* metal_layer, NSView* host_view, float scale) {
    metal_layer.contentsScale = scale;
    metal_layer.frame = host_view.bounds;
    metal_layer.drawableSize =
        CGSizeMake(host_view.bounds.size.width * scale, host_view.bounds.size.height * scale);
}

static std::array<simd_float2, 4> MakeTexCoords(Layout::DisplayOrientation orientation) {
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        return {{
            {1.0f, 0.0f},
            {1.0f, 1.0f},
            {0.0f, 0.0f},
            {0.0f, 1.0f},
        }};
    case Layout::DisplayOrientation::Portrait:
        return {{
            {1.0f, 1.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {0.0f, 0.0f},
        }};
    case Layout::DisplayOrientation::LandscapeFlipped:
        return {{
            {0.0f, 1.0f},
            {0.0f, 0.0f},
            {1.0f, 1.0f},
            {1.0f, 0.0f},
        }};
    case Layout::DisplayOrientation::PortraitFlipped:
        return {{
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 1.0f},
        }};
    }

    return {{
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        {0.0f, 1.0f},
    }};
}

static std::array<PresentVertex, 4> MakeVertices(const Layout::FramebufferLayout& layout, float x,
                                                 float y, float w, float h,
                                                 Layout::DisplayOrientation orientation) {
    const float left = (x / static_cast<float>(layout.width)) * 2.0f - 1.0f;
    const float right = ((x + w) / static_cast<float>(layout.width)) * 2.0f - 1.0f;
    const float top = 1.0f - (y / static_cast<float>(layout.height)) * 2.0f;
    const float bottom = 1.0f - ((y + h) / static_cast<float>(layout.height)) * 2.0f;
    const auto tex_coords = MakeTexCoords(orientation);

    return {{
        {{left, top}, tex_coords[0]},
        {{right, top}, tex_coords[1]},
        {{left, bottom}, tex_coords[2]},
        {{right, bottom}, tex_coords[3]},
    }};
}

static Layout::DisplayOrientation GetOrientation(const Layout::FramebufferLayout& layout) {
    return layout.is_rotated ? Layout::DisplayOrientation::Landscape
                             : Layout::DisplayOrientation::Portrait;
}

static id<MTLSamplerState> CreateSampler(id<MTLDevice> device, MTLSamplerMinMagFilter filter) {
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = filter;
    descriptor.magFilter = filter;
    descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:descriptor];
    [descriptor release];
    return sampler;
}

static void ReleaseTexture(ScreenTexture& texture) {
    [texture.texture release];
    texture.texture = nil;
    texture.width = 0;
    texture.height = 0;
}

} // Anonymous namespace

struct PresentWindow::Impl {
    Frontend::EmuWindow& emu_window;
    const Instance& instance;
    const PipelineCache& pipeline_cache;
    NSView* host_view = nil;
    CAMetalLayer* metal_layer = nil;
    std::array<ScreenTexture, 3> screen_textures{};
    std::array<id<MTLSamplerState>, 2> samplers{};
    bool valid = false;

    Impl(Frontend::EmuWindow& emu_window_, const Instance& instance_,
         const PipelineCache& pipeline_cache_)
        : emu_window{emu_window_}, instance{instance_}, pipeline_cache{pipeline_cache_} {}
};

PresentWindow::PresentWindow(Frontend::EmuWindow& emu_window, const Instance& instance,
                             const PipelineCache& pipeline_cache)
    : impl{std::make_unique<Impl>(emu_window, instance, pipeline_cache)} {
    const auto& window_info = emu_window.GetWindowInfo();
    impl->host_view = (__bridge NSView*)window_info.render_view;

    if (!instance.IsValid() || impl->host_view == nil) {
        LOG_CRITICAL(Render_Metal, "Metal present window missing required Apple host view");
        return;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)instance.GetDevice();

    impl->metal_layer = [[CAMetalLayer alloc] init];
    impl->metal_layer.device = device;
    impl->metal_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    impl->metal_layer.framebufferOnly = NO;
    impl->metal_layer.opaque = YES;
    impl->metal_layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
    impl->metal_layer.contentsGravity = kCAGravityTopLeft;

    impl->samplers[0] = CreateSampler(device, MTLSamplerMinMagFilterNearest);
    impl->samplers[1] = CreateSampler(device, MTLSamplerMinMagFilterLinear);
    if (impl->samplers[0] == nil || impl->samplers[1] == nil) {
        LOG_CRITICAL(Render_Metal, "Failed to create Metal present samplers");
        return;
    }

    [impl->host_view setWantsLayer:YES];
    [impl->host_view.layer addSublayer:impl->metal_layer];
    UpdateDrawableSize(impl->metal_layer, impl->host_view, window_info.render_surface_scale);

    impl->valid = true;
}

PresentWindow::~PresentWindow() {
    if (!impl) {
        return;
    }

    for (auto& texture : impl->screen_textures) {
        ReleaseTexture(texture);
    }
    for (auto& sampler : impl->samplers) {
        [sampler release];
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

static bool EnsureTexture(ScreenTexture& screen_texture, id<MTLDevice> device, u32 width,
                          u32 height) {
    if (screen_texture.texture != nil && screen_texture.width == width &&
        screen_texture.height == height) {
        return true;
    }

    ReleaseTexture(screen_texture);

    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModeShared;
    screen_texture.texture = [device newTextureWithDescriptor:descriptor];
    screen_texture.width = width;
    screen_texture.height = height;
    return screen_texture.texture != nil;
}

static void DrawScreen(id<MTL4RenderCommandEncoder> encoder, const Layout::FramebufferLayout& layout,
                       const Common::Rectangle<u32>& rect, const ScreenTexture& left_screen,
                       const ScreenTexture& right_screen, float opacity, PresentMode mode,
                       id<MTLSamplerState> sampler) {
    if (left_screen.texture == nil) {
        return;
    }

    const auto vertices =
        MakeVertices(layout, static_cast<float>(rect.left), static_cast<float>(rect.top),
                     static_cast<float>(rect.GetWidth()), static_cast<float>(rect.GetHeight()),
                     GetOrientation(layout));
    const PresentFragmentUniforms uniforms{opacity, static_cast<u32>(mode)};
    id<MTLTexture> right_texture = right_screen.texture != nil ? right_screen.texture
                                                               : left_screen.texture;

    [encoder setVertexBytes:vertices.data() length:sizeof(vertices) atIndex:0];
    [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    [encoder setFragmentTexture:left_screen.texture atIndex:0];
    [encoder setFragmentTexture:right_texture atIndex:1];
    [encoder setFragmentSamplerState:sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

static void DrawTopScreen(id<MTL4RenderCommandEncoder> encoder, const Layout::FramebufferLayout& layout,
                          const std::array<ScreenTexture, 3>& screen_textures,
                          id<MTLSamplerState> sampler) {
    if (!layout.top_screen_enabled) {
        return;
    }

    const int leftside = Settings::values.swap_eyes_3d.GetValue() ? 1 : 0;
    const int rightside = Settings::values.swap_eyes_3d.GetValue() ? 0 : 1;
    const auto& rect = layout.top_screen;

    switch (layout.render_3d_mode) {
    case Settings::StereoRenderOption::Off: {
        const int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
        DrawScreen(encoder, layout, rect, screen_textures[eye], screen_textures[eye], 1.0f,
                   PresentMode::Mono, sampler);
        break;
    }
    case Settings::StereoRenderOption::SideBySide: {
        DrawScreen(encoder, layout,
                   Common::Rectangle<u32>{rect.left / 2, rect.top, rect.right / 2, rect.bottom},
                   screen_textures[leftside], screen_textures[leftside], 1.0f, PresentMode::Mono,
                   sampler);
        DrawScreen(
            encoder, layout,
            Common::Rectangle<u32>{(rect.left / 2) + (layout.width / 2), rect.top,
                                   (rect.right / 2) + (layout.width / 2), rect.bottom},
            screen_textures[rightside], screen_textures[rightside], 1.0f, PresentMode::Mono,
            sampler);
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawScreen(encoder, layout, rect, screen_textures[leftside], screen_textures[leftside],
                   1.0f, PresentMode::Mono, sampler);
        DrawScreen(encoder, layout,
                   rect.TranslateX(static_cast<int>(layout.width / 2)), screen_textures[rightside],
                   screen_textures[rightside], 1.0f, PresentMode::Mono, sampler);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawScreen(encoder, layout, rect, screen_textures[leftside], screen_textures[leftside],
                   1.0f, PresentMode::Mono, sampler);
        DrawScreen(
            encoder, layout,
            Common::Rectangle<u32>{layout.cardboard.top_screen_right_eye + (layout.width / 2),
                                   rect.top,
                                   layout.cardboard.top_screen_right_eye + (layout.width / 2) +
                                       rect.GetWidth(),
                                   rect.bottom},
            screen_textures[rightside], screen_textures[rightside], 1.0f, PresentMode::Mono,
            sampler);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
        DrawScreen(encoder, layout, rect, screen_textures[leftside], screen_textures[rightside],
                   1.0f, PresentMode::Anaglyph, sampler);
        break;
    case Settings::StereoRenderOption::Interlaced:
        DrawScreen(encoder, layout, rect, screen_textures[leftside], screen_textures[rightside],
                   1.0f, PresentMode::Interlaced, sampler);
        break;
    case Settings::StereoRenderOption::ReverseInterlaced:
        DrawScreen(encoder, layout, rect, screen_textures[leftside], screen_textures[rightside],
                   1.0f, PresentMode::ReverseInterlaced, sampler);
        break;
    }
}

static void DrawBottomScreen(id<MTL4RenderCommandEncoder> encoder,
                             const Layout::FramebufferLayout& layout,
                             const std::array<ScreenTexture, 3>& screen_textures, float opacity,
                             id<MTLSamplerState> sampler) {
    if (!layout.bottom_screen_enabled) {
        return;
    }

    const auto& rect = layout.bottom_screen;

    switch (layout.render_3d_mode) {
    case Settings::StereoRenderOption::Off:
        DrawScreen(encoder, layout, rect, screen_textures[2], screen_textures[2], opacity,
                   PresentMode::Mono, sampler);
        break;
    case Settings::StereoRenderOption::SideBySide:
        DrawScreen(encoder, layout,
                   Common::Rectangle<u32>{rect.left / 2, rect.top, rect.right / 2, rect.bottom},
                   screen_textures[2], screen_textures[2], opacity, PresentMode::Mono, sampler);
        DrawScreen(
            encoder, layout,
            Common::Rectangle<u32>{(rect.left / 2) + (layout.width / 2), rect.top,
                                   (rect.right / 2) + (layout.width / 2), rect.bottom},
            screen_textures[2], screen_textures[2], opacity, PresentMode::Mono, sampler);
        break;
    case Settings::StereoRenderOption::SideBySideFull:
        DrawScreen(encoder, layout, rect, screen_textures[2], screen_textures[2], opacity,
                   PresentMode::Mono, sampler);
        DrawScreen(encoder, layout, rect.TranslateX(static_cast<int>(layout.width / 2)),
                   screen_textures[2], screen_textures[2], opacity, PresentMode::Mono, sampler);
        break;
    case Settings::StereoRenderOption::CardboardVR:
        DrawScreen(encoder, layout, rect, screen_textures[2], screen_textures[2], opacity,
                   PresentMode::Mono, sampler);
        DrawScreen(
            encoder, layout,
            Common::Rectangle<u32>{layout.cardboard.bottom_screen_right_eye + (layout.width / 2),
                                   rect.top,
                                   layout.cardboard.bottom_screen_right_eye + (layout.width / 2) +
                                       rect.GetWidth(),
                                   rect.bottom},
            screen_textures[2], screen_textures[2], opacity, PresentMode::Mono, sampler);
        break;
    case Settings::StereoRenderOption::Anaglyph:
        DrawScreen(encoder, layout, rect, screen_textures[2], screen_textures[2], opacity,
                   PresentMode::Anaglyph, sampler);
        break;
    case Settings::StereoRenderOption::Interlaced:
        DrawScreen(encoder, layout, rect, screen_textures[2], screen_textures[2], opacity,
                   PresentMode::Interlaced, sampler);
        break;
    case Settings::StereoRenderOption::ReverseInterlaced:
        DrawScreen(encoder, layout, rect, screen_textures[2], screen_textures[2], opacity,
                   PresentMode::ReverseInterlaced, sampler);
        break;
    }
}

void PresentWindow::Present(const Layout::FramebufferLayout& layout, const Common::Vec4f& clear_color,
                            const std::array<ScreenInfo, 3>& screen_infos) {
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

    for (std::size_t i = 0; i < screen_infos.size(); i++) {
        const auto& screen = screen_infos[i];
        if (!screen.valid || screen.pixels == nullptr || screen.width == 0 || screen.height == 0) {
            continue;
        }
        if (!EnsureTexture(impl->screen_textures[i], device, screen.width, screen.height)) {
            LOG_ERROR(Render_Metal, "Failed to allocate Metal screen texture {} ({}x{})", i,
                      screen.width, screen.height);
            continue;
        }

        MTLRegion region = MTLRegionMake2D(0, 0, screen.width, screen.height);
        [impl->screen_textures[i].texture replaceRegion:region
                                            mipmapLevel:0
                                              withBytes:screen.pixels
                                            bytesPerRow:screen.width * 4];
    }

    id<MTL4CommandAllocator> allocator =
        (__bridge id<MTL4CommandAllocator>)impl->instance.GetCommandAllocator();
    id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)impl->instance.GetCommandQueue();
    id<MTLRenderPipelineState> pipeline =
        (__bridge id<MTLRenderPipelineState>)impl->pipeline_cache.GetPresentPipeline();
    id<MTLSamplerState> sampler = impl->samplers[Settings::values.filter_mode.GetValue() ? 1 : 0];

    if (pipeline == nil || sampler == nil) {
        LOG_ERROR(Render_Metal, "Metal present resources are not ready");
        return;
    }

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
        [encoder setRenderPipelineState:pipeline];
        [encoder setCullMode:MTLCullModeNone];

        if (!Settings::values.swap_screen.GetValue()) {
            DrawTopScreen(encoder, layout, impl->screen_textures, sampler);
            DrawBottomScreen(encoder, layout, impl->screen_textures, layout.bottom_opacity,
                             sampler);
        } else {
            DrawBottomScreen(encoder, layout, impl->screen_textures, 1.0f, sampler);
            DrawTopScreen(encoder, layout, impl->screen_textures, sampler);
        }

        if (layout.additional_screen_enabled) {
            const auto& additional_screen = layout.additional_screen;
            if (!Settings::values.swap_screen.GetValue()) {
                DrawScreen(encoder, layout, additional_screen, impl->screen_textures[0],
                           impl->screen_textures[1], 1.0f, PresentMode::Mono, sampler);
            } else {
                DrawScreen(encoder, layout, additional_screen, impl->screen_textures[2],
                           impl->screen_textures[2], 1.0f, PresentMode::Mono, sampler);
            }
        }

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
