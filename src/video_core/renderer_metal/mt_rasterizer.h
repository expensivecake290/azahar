// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/renderer_metal/mt_present_window.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
class PicaCore;
}

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class CustomTexManager;
class RendererBase;
}

namespace Metal {

class Instance;
class TextureRuntime;
class PipelineCache;

class RasterizerMetal : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerMetal(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                             VideoCore::CustomTexManager& custom_tex_manager,
                             VideoCore::RendererBase& renderer, Frontend::EmuWindow& emu_window,
                             const Instance& instance, TextureRuntime& runtime,
                             PipelineCache& pipeline_cache);
    ~RasterizerMetal() override;

    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;

    void DrawTriangles() override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplay(u32 screen_index, const Pica::FramebufferConfig& config,
                           PAddr framebuffer_addr, u32 pixel_stride,
                           const Pica::ColorFill& color_fill,
                           ScreenInfo& screen_info);

private:
    void LogUnsupported(const char* function_name);

private:
    TextureRuntime& runtime;
    bool warned = false;
};

} // namespace Metal
