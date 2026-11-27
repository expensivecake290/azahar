// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <compare>
#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include "common/math_util.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

namespace VideoCore {
enum class PixelFormat : u32;
}

namespace Metal {

class Instance;

struct RenderArea {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};

    [[nodiscard]] static constexpr std::optional<RenderArea> FromRectangle(
        const Common::Rectangle<u32>& rect) noexcept {
        if (rect.right <= rect.left || rect.top <= rect.bottom) {
            return {};
        }
        return RenderArea{rect.left, rect.bottom, rect.right - rect.left, rect.top - rect.bottom};
    }

    [[nodiscard]] constexpr bool Fits(u32 target_width, u32 target_height) const noexcept {
        return width && height && x <= target_width && y <= target_height &&
               width <= target_width - x && height <= target_height - y;
    }

    [[nodiscard]] constexpr bool Covers(u32 target_width, u32 target_height) const noexcept {
        return x == 0 && y == 0 && width == target_width && height == target_height;
    }

    bool operator==(const RenderArea&) const = default;
};

struct RenderTexture {
    id<MTLTexture> texture = nil;
    u32 level{};
    u32 slice{};
    u32 depth_plane{};

    bool operator==(const RenderTexture&) const = default;
};

struct RenderAttachment {
    RenderTexture target;
    RenderTexture resolve;
    MTLLoadAction load_action = MTLLoadActionLoad;
    MTLStoreAction store_action = MTLStoreActionStore;

    [[nodiscard]] bool CanContinueWith(const RenderAttachment& next) const noexcept {
        return target == next.target && resolve == next.resolve &&
               store_action == next.store_action &&
               (next.load_action == MTLLoadActionLoad ||
                (load_action == MTLLoadActionDontCare &&
                 next.load_action == MTLLoadActionDontCare));
    }
};

/// Native pipeline compatibility. Load, store and clear values belong to the pass instance.
struct RenderFormats {
    MTLPixelFormat color{MTLPixelFormatInvalid};
    MTLPixelFormat depth{MTLPixelFormatInvalid};
    MTLPixelFormat stencil{MTLPixelFormatInvalid};
    u32 samples{1};

    auto operator<=>(const RenderFormats&) const = default;
};

struct RenderPass {
    RenderAttachment color;
    RenderAttachment depth;
    RenderAttachment stencil;
    RenderArea area;
    u32 width{};
    u32 height{};
    u32 samples{1};
    MTLClearColor clear_color{};
    double clear_depth{1.0};
    u32 clear_stencil{};
    /// Shadow draws may write a fragment storage texture without a fixed-function attachment.
    id<MTLTexture> shadow_storage = nil;
    /// The pipeline uses native framebuffer feedback with render-order groups.
    bool tile_feedback{};

    [[nodiscard]] bool CanContinueWith(const RenderPass& next) const noexcept {
        return area == next.area && width == next.width && height == next.height &&
               samples == next.samples && shadow_storage == next.shadow_storage &&
               tile_feedback == next.tile_feedback && color.CanContinueWith(next.color) &&
               depth.CanContinueWith(next.depth) && stencil.CanContinueWith(next.stencil);
    }

    [[nodiscard]] RenderFormats Formats() const noexcept;
};

/// Producer-side render pass state with one worker-owned Metal 4 render encoder.
class RenderManager {
    struct State;

public:
    RenderManager(const Instance& instance, Scheduler& scheduler, const MemoryAllocator& allocator);
    ~RenderManager();
    RenderManager(const RenderManager&) = delete;
    RenderManager& operator=(const RenderManager&) = delete;

    void BeginRendering(const RenderPass& pass);
    void BeginRendering(RenderPass pass, const Common::Rectangle<u32>& draw_rect);
    void EndRendering();

    template <typename Draw>
        requires std::invocable<Draw&, EncodingContext&, id<MTL4RenderCommandEncoder>>
    void RecordDraw(Draw&& draw) {
        RequireActivePass();
        scheduler.Record(
            [pass = state, draw = std::forward<Draw>(draw)](EncodingContext& ctx) mutable {
                std::invoke(draw, ctx, RequireEncoder(*pass, ctx));
            });
    }

    [[nodiscard]] RenderFormats GetRenderpass(VideoCore::PixelFormat color,
                                              VideoCore::PixelFormat depth,
                                              u32 sample_count = 1) const;

    /// Split the current pass and copy preserved color into a distinct subresource.
    /// Equal-sample multisample attachments preserve each destination sample independently.
    void CopyForFeedback(const RenderTexture& source, const RenderTexture& destination,
                         RenderArea area);

private:
    void RequireActivePass() const;
    [[nodiscard]] static id<MTL4RenderCommandEncoder> RequireEncoder(const State& state,
                                                                     EncodingContext& context);

    const Instance& instance;
    Scheduler& scheduler;
    std::shared_ptr<State> state;
};

} // namespace Metal
