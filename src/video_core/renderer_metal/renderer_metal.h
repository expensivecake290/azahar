// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"
#include "video_core/pica/regs_external.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_metal/mt_instance.h"
#include "video_core/renderer_metal/mt_pipeline_cache.h"
#include "video_core/renderer_metal/mt_present_window.h"
#include "video_core/renderer_metal/mt_rasterizer.h"
#include "video_core/renderer_metal/mt_texture_runtime.h"

namespace Core {
class System;
}

namespace Pica {
class PicaCore;
}

namespace Metal {

class RendererMetal : public VideoCore::RendererBase {
public:
    explicit RendererMetal(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window,
                           Frontend::EmuWindow* secondary_window);
    ~RendererMetal() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    void NotifySurfaceChanged(bool second) override;

    void SwapBuffers() override;
    void TryPresent(int timeout_ms, bool is_secondary) override {}

private:
    void PrepareRendertarget();
    void LoadFBToScreenInfo(u32 screen_index, const Pica::FramebufferConfig& framebuffer,
                            bool right_eye, const Pica::ColorFill& color_fill);

    Pica::PicaCore& pica;
    Instance instance;
    PipelineCache pipeline_cache;
    TextureRuntime runtime;
    PresentWindow main_present_window;
    RasterizerMetal rasterizer;
    std::unique_ptr<PresentWindow> secondary_present_window_ptr;
    std::array<Metal::ScreenInfo, 3> present_screens;
};

} // namespace Metal
