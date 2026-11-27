// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#import <QuartzCore/CAMetalLayer.h>

#include <functional>
#include <memory>
#include <optional>
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_metal/mtl_platform.h"

namespace Metal {

class EncodingContext;
class Instance;
class Scheduler;

enum class PresentMode : u8 {
    Fifo,
    Mailbox,
    Immediate,
};

struct PresentPacing {
    bool vsync{true};
    double frame_limit{100.};
    bool low_refresh_rate{};
    std::optional<double> minimum_duration{};

    [[nodiscard]] constexpr PresentMode SelectMode(bool has_mailbox,
                                                   bool has_immediate) const noexcept {
        if (!vsync) {
            return has_immediate ? PresentMode::Immediate
                   : has_mailbox ? PresentMode::Mailbox
                                 : PresentMode::Fifo;
        }
        if (!low_refresh_rate && frame_limit != 0 && frame_limit <= 100) {
            return PresentMode::Fifo;
        }
        return has_mailbox     ? PresentMode::Mailbox
               : has_immediate ? PresentMode::Immediate
                               : PresentMode::Fifo;
    }

    bool operator==(const PresentPacing&) const = default;
};

enum class DrawableStatus : u8 {
    Ready,
    NoSurface,
    Suspended,
    Backpressure,
    SurfaceChanged,
    DeviceUnavailable,
    Closed,
};

/// One presentation producer owns a swapchain. Acquire may wait for Core Animation and runs
/// on that producer, outside the emulator and frontend event loops.
class Swapchain final {
    struct Generation;
    struct State;

public:
    class Drawable {
    public:
        Drawable(const Drawable&) = delete;
        Drawable& operator=(const Drawable&) = delete;
        Drawable(Drawable&&) noexcept = default;
        Drawable& operator=(Drawable&&) noexcept = default;

        [[nodiscard]] id<MTLTexture> Texture() const noexcept {
            return native.texture;
        }

        [[nodiscard]] VideoCore::Extent Extent() const noexcept {
            return native ? extent : VideoCore::Extent{0, 0};
        }

    private:
        friend class Swapchain;
        explicit Drawable(id<CAMetalDrawable> native, VideoCore::Extent extent,
                          std::shared_ptr<const Generation> generation);

        id<CAMetalDrawable> native;
        VideoCore::Extent extent;
        std::shared_ptr<const Generation> generation;
    };

    struct Acquisition {
        DrawableStatus status;
        std::optional<Drawable> drawable;
    };

    Swapchain(Instance& instance, Scheduler& scheduler);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    /// Adopt a newly published surface or apply changed pacing to the current generation.
    bool UpdateSurface(Frontend::EmuWindow::NativeSurfaceSnapshot snapshot, PresentPacing pacing);

    [[nodiscard]] Acquisition AcquireNextImage();

    using DrawableEncoder = std::function<void(EncodingContext&, id<MTLTexture>)>;

    /// Consume a drawable from the live surface. A retired lease returns no ticket before the
    /// encoder runs. The dependency, if supplied, belongs to the bound native scheduler.
    [[nodiscard]] std::optional<u64> Present(Drawable drawable, DrawableEncoder encode,
                                             u64 wait_for_tick = 0);

    [[nodiscard]] std::optional<Surface> GetSurface() const;
    [[nodiscard]] VideoCore::Extent GetExtent() const noexcept;
    [[nodiscard]] MTLPixelFormat GetPixelFormat() const noexcept;
    [[nodiscard]] u32 GetImageCount() const noexcept;
    [[nodiscard]] PresentMode GetPresentMode() const noexcept;

    /// Stop new acquisitions and release the current surface. Committed drawables remain owned
    /// until their native work retires.
    void Close();

private:
    std::unique_ptr<State> state;
};

} // namespace Metal
