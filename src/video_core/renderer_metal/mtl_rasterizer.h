// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <memory>
#include <optional>
#include <span>
#include "video_core/rasterizer_accelerated.h"
#include "video_core/renderer_metal/mtl_texture_runtime.h"

namespace VideoCore {
class CustomTexManager;
class RendererBase;
} // namespace VideoCore

namespace Pica {
struct FramebufferConfig;
}

namespace Metal {

class DescriptorUpdateQueue;
struct PipelineCacheOptions;

struct ScreenInfo {
    [[nodiscard]] static constexpr std::optional<Common::Rectangle<f32>> Coordinates(
        const Common::Rectangle<u32>& rect, u32 width, u32 height) noexcept {
        if (rect.left >= rect.right || rect.bottom >= rect.top || rect.right > width ||
            rect.top > height) {
            return {};
        }
        return Common::Rectangle<f32>{
            static_cast<f32>(rect.bottom) / height, static_cast<f32>(rect.left) / width,
            static_cast<f32>(rect.top) / height, static_cast<f32>(rect.right) / width};
    }

    TextureResource image;
    Common::Rectangle<f32> texcoords{};
};

/// Reads an unsigned PICA index. The largest value in either format is a vertex index.
[[nodiscard]] constexpr std::optional<u32> ReadVertexIndex(std::span<const u8> data, u32 index,
                                                           bool index_u16) noexcept {
    const std::size_t width = index_u16 ? 2 : 1;
    if (index >= data.size() / width) {
        return {};
    }
    const auto offset = index * width;
    return index_u16 ? u32{data[offset]} | (u32{data[offset + 1]} << 8) : u32{data[offset]};
}

class RasterizerMetal final : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerMetal(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                             VideoCore::CustomTexManager& custom_tex_manager,
                             VideoCore::RendererBase& renderer, Instance& instance,
                             Scheduler& scheduler, RenderManager& render_manager,
                             DescriptorUpdateQueue& update_queue, MemoryAllocator& allocator,
                             MemoryBudget& budget, PipelineCacheOptions options);
    ~RasterizerMetal() override;

    void TickFrame();
    void DrawTriangles() override;
    bool AccelerateDrawBatch(bool is_indexed) override;

    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;

    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const Pica::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);

    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;
    void SwitchDiskResources(u64 title_id) override;
    bool DiskResourcesLoadCrashed(u64 title_id) override;

private:
    struct State;
    std::unique_ptr<State> state;
};

} // namespace Metal
