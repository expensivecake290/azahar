// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_metal/renderer_metal.h"

#include <stdexcept>

namespace Metal {

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

void RendererMetal::SwapBuffers() {
    system.perf_stats->StartSwap();

    const Common::Vec4f clear_color{
        Settings::values.bg_red.GetValue(),
        Settings::values.bg_green.GetValue(),
        Settings::values.bg_blue.GetValue(),
        1.0f,
    };

    main_present_window.Present(render_window.GetFramebufferLayout(), clear_color);
    if (secondary_present_window_ptr) {
        secondary_present_window_ptr->Present(secondary_window->GetFramebufferLayout(), clear_color);
    }

    system.perf_stats->EndSwap();
    EndFrame();
}

} // namespace Metal
