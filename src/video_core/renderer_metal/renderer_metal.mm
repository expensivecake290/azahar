// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_metal/renderer_metal.h"

#include <stdexcept>

namespace Metal {

RendererMetal::RendererMetal(Core::System& system, Pica::PicaCore& pica_,
                             Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window)
    : RendererBase{system, window, secondary_window}, pica{pica_}, instance{
          window, Settings::values.physical_device.GetValue()},
      pipeline_cache{instance}, runtime{instance},
      main_present_window{window, instance, pipeline_cache},
      rasterizer{system.Memory(), pica, system.CustomTexManager(), *this, render_window, instance,
                 runtime, pipeline_cache} {
    if (!instance.IsValid()) {
        throw std::runtime_error("Metal backend initialization failed");
    }
    if (!pipeline_cache.IsValid()) {
        throw std::runtime_error("Metal present pipeline initialization failed");
    }
    if (!main_present_window.IsValid()) {
        throw std::runtime_error("Metal present window initialization failed");
    }
    if (secondary_window) {
        secondary_present_window_ptr =
            std::make_unique<PresentWindow>(*secondary_window, instance, pipeline_cache);
    }
    LOG_INFO(Render_Metal, "Using Metal device: {}", instance.GetDeviceName());
}

RendererMetal::~RendererMetal() {
    runtime.Finish();
}

void RendererMetal::NotifySurfaceChanged(bool second) {
    if (second) {
        if (secondary_present_window_ptr) {
            secondary_present_window_ptr->NotifySurfaceChanged();
        }
        return;
    }
    main_present_window.NotifySurfaceChanged();
}

void RendererMetal::PrepareRendertarget() {
    const auto& framebuffer_config = pica.regs.framebuffer_config;
    const auto& regs_lcd = pica.regs_lcd;

    for (u32 i = 0; i < 3; i++) {
        const u32 fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = framebuffer_config[fb_id];
        const auto color_fill = fb_id == 0 ? regs_lcd.color_fill_top : regs_lcd.color_fill_bottom;
        LoadFBToScreenInfo(i, framebuffer, i == 1, color_fill);
    }
}

void RendererMetal::LoadFBToScreenInfo(u32 screen_index, const Pica::FramebufferConfig& framebuffer,
                                       bool right_eye, const Pica::ColorFill& color_fill) {
    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0) {
        right_eye = false;
    }

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
            ? (right_eye ? framebuffer.address_right1 : framebuffer.address_left1)
            : (right_eye ? framebuffer.address_right2 : framebuffer.address_left2);

    present_screens[screen_index] = {};

    const u32 width = color_fill.is_enabled ? 1 : framebuffer.width;
    const u32 height = color_fill.is_enabled ? 1 : framebuffer.height;
    if (width == 0 || height == 0) {
        return;
    }

    const u32 bpp = Pica::BytesPerPixel(framebuffer.color_format);
    const u32 pixel_stride = framebuffer.stride / bpp;
    if (pixel_stride * bpp != framebuffer.stride ||
        (!color_fill.is_enabled && framebuffer.width > pixel_stride)) {
        LOG_CRITICAL(Render_Metal, "Invalid Metal display stride {} for {}x{} format {}",
                     framebuffer.stride, framebuffer.width.Value(), framebuffer.height.Value(),
                     framebuffer.format);
        throw std::runtime_error("Metal backend rejected invalid display framebuffer state");
    }

    if (!rasterizer.AccelerateDisplay(screen_index, framebuffer, framebuffer_addr, pixel_stride,
                                      color_fill, present_screens[screen_index])) {
        LOG_CRITICAL(Render_Metal,
                     "Metal backend requires native accelerated display surfaces; CPU fallback is disabled");
        throw std::runtime_error(
            "Metal backend requires native accelerated display surfaces; CPU fallback removed");
    }
}

void RendererMetal::SwapBuffers() {
    system.perf_stats->StartSwap();
    PrepareRendertarget();

    const Common::Vec4f clear_color{
        Settings::values.bg_red.GetValue(),
        Settings::values.bg_green.GetValue(),
        Settings::values.bg_blue.GetValue(),
        1.0f,
    };

    main_present_window.Present(render_window.GetFramebufferLayout(), clear_color, present_screens);
    if (secondary_present_window_ptr) {
        secondary_present_window_ptr->Present(secondary_window->GetFramebufferLayout(), clear_color,
                                              present_screens);
    }

    system.perf_stats->EndSwap();
    EndFrame();
}

} // namespace Metal
