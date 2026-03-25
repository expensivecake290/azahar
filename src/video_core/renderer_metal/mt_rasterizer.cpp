// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/renderer_metal/mt_pipeline_cache.h"
#include "video_core/renderer_metal/mt_rasterizer.h"
#include "video_core/renderer_metal/mt_texture_runtime.h"

namespace Metal {

RasterizerMetal::RasterizerMetal(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                 VideoCore::CustomTexManager& custom_tex_manager,
                                 VideoCore::RendererBase& renderer,
                                 Frontend::EmuWindow& emu_window, const Instance& instance,
                                 TextureRuntime& runtime, PipelineCache& pipeline_cache)
    : RasterizerAccelerated{memory, pica} {
    (void)custom_tex_manager;
    (void)renderer;
    (void)emu_window;
    (void)instance;
    (void)runtime;
    (void)pipeline_cache;
}

RasterizerMetal::~RasterizerMetal() = default;

void RasterizerMetal::LoadDefaultDiskResources(
    const std::atomic_bool& stop_loading, const VideoCore::DiskResourceLoadCallback& callback) {
    (void)stop_loading;
    if (callback) {
        callback(VideoCore::LoadCallbackStage::Prepare, 0, 0, "");
        callback(VideoCore::LoadCallbackStage::Complete, 0, 0, "");
    }
}

void RasterizerMetal::DrawTriangles() {
    vertex_batch.clear();
    LogUnsupported("DrawTriangles");
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::FlushRegion(PAddr addr, u32 size) {
    (void)addr;
    (void)size;
}

void RasterizerMetal::InvalidateRegion(PAddr addr, u32 size) {
    (void)addr;
    (void)size;
}

void RasterizerMetal::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    (void)addr;
    (void)size;
}

void RasterizerMetal::ClearAll(bool flush) {
    (void)flush;
    vertex_batch.clear();
}

void RasterizerMetal::LogUnsupported(const char* function_name) {
    if (warned) {
        return;
    }
    warned = true;
    LOG_WARNING(Render_Metal,
                "Metal backend skeleton invoked {} before accelerated draw parity is implemented",
                function_name);
}

} // namespace Metal
