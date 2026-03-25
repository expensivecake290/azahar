// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_accelerated.h"

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

private:
    void LogUnsupported(const char* function_name);

private:
    bool warned = false;
};

} // namespace Metal
