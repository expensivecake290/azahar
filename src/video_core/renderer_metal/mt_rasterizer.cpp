// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_rasterizer.h"

#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_metal/mt_pipeline_cache.h"
#include "video_core/renderer_metal/mt_texture_runtime.h"

namespace Metal {

RasterizerMetal::RasterizerMetal(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                 VideoCore::CustomTexManager& custom_tex_manager,
                                 VideoCore::RendererBase& renderer,
                                 Frontend::EmuWindow& emu_window, const Instance& instance,
                                 TextureRuntime& runtime_, PipelineCache& pipeline_cache)
    : RasterizerAccelerated{memory, pica}, runtime{runtime_} {
    (void)custom_tex_manager;
    (void)renderer;
    (void)emu_window;
    (void)instance;
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
    runtime.ClearDisplayTextures();
}

bool RasterizerMetal::AccelerateDisplay(u32 screen_index, const Pica::FramebufferConfig& config,
                                        PAddr framebuffer_addr, u32 pixel_stride,
                                        const Pica::ColorFill& color_fill,
                                        ScreenInfo& screen_info) {
    if (color_fill.is_enabled) {
        const Common::Vec4<u8> fill_pixel{color_fill.color_r, color_fill.color_g, color_fill.color_b,
                                          255};
        const auto uploaded =
            runtime.SynchronizeDisplaySurface(screen_index, 0, fill_pixel.AsArray(), 4, 1, 1, 1,
                                              Pica::PixelFormat::RGBA8);
        screen_info = {
            .buffer = uploaded.buffer,
            .width = uploaded.width,
            .height = uploaded.height,
            .pixel_stride = uploaded.pixel_stride,
            .pixel_format = uploaded.pixel_format,
            .texcoords = uploaded.texcoords,
            .valid = uploaded.valid,
        };
        return uploaded.valid;
    }

    if (framebuffer_addr == 0 || config.width == 0 || config.height == 0) {
        return false;
    }
    if (pixel_stride < config.width) {
        LOG_ERROR(Render_Metal, "Invalid display pixel stride {} for width {}", pixel_stride,
                  config.width.Value());
        return false;
    }

    FlushRegion(framebuffer_addr, config.stride * config.height);
    const u8* framebuffer_data = memory.GetPhysicalPointer(framebuffer_addr);
    if (framebuffer_data == nullptr) {
        LOG_ERROR(Render_Metal, "Failed to map framebuffer memory at 0x{:08x}", framebuffer_addr);
        return false;
    }

    const auto uploaded =
        runtime.SynchronizeDisplaySurface(screen_index, framebuffer_addr, framebuffer_data,
                                          config.stride * config.height, config.width,
                                          config.height, pixel_stride, config.color_format);
    screen_info = {
        .buffer = uploaded.buffer,
        .width = uploaded.width,
        .height = uploaded.height,
        .pixel_stride = uploaded.pixel_stride,
        .pixel_format = uploaded.pixel_format,
        .texcoords = uploaded.texcoords,
        .valid = uploaded.valid,
    };
    return uploaded.valid;
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
