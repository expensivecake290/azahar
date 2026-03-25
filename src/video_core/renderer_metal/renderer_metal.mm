// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/color.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_metal/renderer_metal.h"

#include <stdexcept>

namespace Metal {

namespace {

static Common::Vec4<u8> DecodeFramebufferPixel(Pica::PixelFormat format, const u8* pixel) {
    switch (format) {
    case Pica::PixelFormat::RGBA8:
        return Common::Color::DecodeRGBA8(pixel);
    case Pica::PixelFormat::RGB8:
        return Common::Color::DecodeRGB8(pixel);
    case Pica::PixelFormat::RGB565:
        return Common::Color::DecodeRGB565(pixel);
    case Pica::PixelFormat::RGB5A1:
        return Common::Color::DecodeRGB5A1(pixel);
    case Pica::PixelFormat::RGBA4:
        return Common::Color::DecodeRGBA4(pixel);
    }

    UNREACHABLE();
}

} // Anonymous namespace

RendererMetal::RendererMetal(Core::System& system, Pica::PicaCore& pica_,
                             Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window)
    : RendererBase{system, window, secondary_window}, memory{system.Memory()}, pica{pica_},
      instance{window, Settings::values.physical_device.GetValue()}, pipeline_cache{instance},
      runtime{instance}, main_present_window{window, instance, pipeline_cache},
      rasterizer{memory, pica, system.CustomTexManager(), *this, render_window, instance, runtime,
                 pipeline_cache} {
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
        LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1, color_fill);
    }
}

void RendererMetal::LoadFBToScreenInfo(const Pica::FramebufferConfig& framebuffer,
                                       ScreenBuffer& screen_info, bool right_eye,
                                       const Pica::ColorFill& color_fill) {
    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0) {
        right_eye = false;
    }

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
            ? (right_eye ? framebuffer.address_right1 : framebuffer.address_left1)
            : (right_eye ? framebuffer.address_right2 : framebuffer.address_left2);
    const u32 width = color_fill.is_enabled ? 1 : framebuffer.width;
    const u32 height = color_fill.is_enabled ? 1 : framebuffer.height;

    screen_info.valid = false;
    screen_info.width = width;
    screen_info.height = height;

    if (width == 0 || height == 0) {
        return;
    }

    screen_info.pixels.resize(static_cast<std::size_t>(width) * height * 4);

    if (color_fill.is_enabled) {
        const Common::Vec4<u8> fill_pixel{color_fill.color_r, color_fill.color_g, color_fill.color_b,
                                          255};
        std::memcpy(screen_info.pixels.data(), fill_pixel.AsArray(), 4);
        screen_info.valid = true;
        return;
    }

    if (framebuffer_addr == 0) {
        LOG_ERROR(Render_Metal, "Framebuffer address is zero");
        return;
    }

    const u32 bpp = Pica::BytesPerPixel(framebuffer.color_format);
    const u32 pixel_stride = framebuffer.stride / bpp;
    if (pixel_stride * bpp != framebuffer.stride || framebuffer.width > pixel_stride) {
        LOG_ERROR(Render_Metal, "Invalid framebuffer stride {} for {}x{} format {}",
                  framebuffer.stride, framebuffer.width.Value(), framebuffer.height.Value(),
                  framebuffer.format);
        return;
    }

    rasterizer.FlushRegion(framebuffer_addr, framebuffer.stride * framebuffer.height);

    const u8* framebuffer_data = memory.GetPhysicalPointer(framebuffer_addr);
    if (framebuffer_data == nullptr) {
        LOG_ERROR(Render_Metal, "Failed to map framebuffer memory at 0x{:08x}", framebuffer_addr);
        return;
    }

    for (u32 y = 0; y < framebuffer.height; y++) {
        for (u32 x = 0; x < framebuffer.width; x++) {
            const u8* pixel = framebuffer_data + ((y * pixel_stride) + x) * bpp;
            const Common::Vec4<u8> color = DecodeFramebufferPixel(framebuffer.color_format, pixel);
            std::memcpy(screen_info.pixels.data() + ((y * framebuffer.width) + x) * 4,
                        color.AsArray(), 4);
        }
    }

    screen_info.valid = true;
}

void RendererMetal::SwapBuffers() {
    system.perf_stats->StartSwap();
    PrepareRendertarget();

    std::array<Metal::ScreenInfo, 3> present_screens{};
    for (std::size_t i = 0; i < screen_infos.size(); i++) {
        present_screens[i] = {
            .pixels = screen_infos[i].pixels.empty() ? nullptr : screen_infos[i].pixels.data(),
            .width = screen_infos[i].width,
            .height = screen_infos[i].height,
            .opacity = 1.0f,
            .valid = screen_infos[i].valid,
        };
    }

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
