// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <vector>

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

namespace Memory {
class MemorySystem;
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
    struct ScreenBuffer {
        std::vector<u8> pixels;
        u32 width = 0;
        u32 height = 0;
        bool valid = false;
    };

    void PrepareRendertarget();
    void LoadFBToScreenInfo(const Pica::FramebufferConfig& framebuffer, ScreenBuffer& screen_info,
                            bool right_eye, const Pica::ColorFill& color_fill);

    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    Instance instance;
    PipelineCache pipeline_cache;
    TextureRuntime runtime;
    PresentWindow main_present_window;
    RasterizerMetal rasterizer;
    std::unique_ptr<PresentWindow> secondary_present_window_ptr;
    std::array<ScreenBuffer, 3> screen_infos;
};

} // namespace Metal
