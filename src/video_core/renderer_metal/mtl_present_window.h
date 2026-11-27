// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <memory>
#include <optional>
#include "video_core/renderer_metal/mtl_render_manager.h"
#include "video_core/renderer_metal/mtl_swapchain.h"

namespace Metal {

enum class PresentTransfer : u8 {
    Invalid,
    Copy,
    Nearest,
    Linear,
};

[[nodiscard]] constexpr PresentTransfer SelectPresentTransfer(VideoCore::Extent source,
                                                              VideoCore::Extent destination,
                                                              MTLPixelFormat source_format,
                                                              MTLPixelFormat destination_format,
                                                              bool linear) noexcept {
    const auto compatible = [](MTLPixelFormat format) {
        return format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatRGBA8Unorm ||
               format == MTLPixelFormatBGRA8Unorm_sRGB || format == MTLPixelFormatRGBA8Unorm_sRGB;
    };
    if (!source.width || !source.height || !destination.width || !destination.height ||
        !compatible(source_format) || !compatible(destination_format)) {
        return PresentTransfer::Invalid;
    }
    const bool same_extent =
        source.width == destination.width && source.height == destination.height;
    if (same_extent && source_format == destination_format) {
        return PresentTransfer::Copy;
    }
    return !same_extent && linear ? PresentTransfer::Linear : PresentTransfer::Nearest;
}

/// One independent presentation stream. The window, device and rendering scheduler outlive it.
class PresentWindow final {
    struct Allocation;
    struct State;

public:
    static constexpr u32 FrameCount = 3;
    static constexpr MTLPixelFormat FrameColorFormat = MTLPixelFormatBGRA8Unorm;

    class Frame {
    public:
        Frame(Frame&& other) noexcept;
        Frame& operator=(Frame&& other) noexcept;
        Frame(const Frame&) = delete;
        Frame& operator=(const Frame&) = delete;
        ~Frame();

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(allocation);
        }

        [[nodiscard]] VideoCore::Extent Extent() const noexcept;
        [[nodiscard]] u32 Samples() const noexcept;
        [[nodiscard]] id<MTLTexture> Texture() const noexcept;
        [[nodiscard]] RenderPass MakePass(MTLClearColor clear) const;
        /// Keep the allocation alive in the producer's pending submission before drawing.
        void Retain(Scheduler& scheduler) const;

    private:
        friend struct State;
        Frame(std::weak_ptr<State> owner, std::shared_ptr<Allocation> allocation, u32 slot);
        void Release();

        std::weak_ptr<State> owner;
        std::shared_ptr<Allocation> allocation;
        u32 slot{};
    };

    PresentWindow(Frontend::EmuWindow& window, Instance& instance, Scheduler& rendering,
                  MemoryBudget& budget, bool async_presentation);
    ~PresentWindow();
    PresentWindow(const PresentWindow&) = delete;
    PresentWindow& operator=(const PresentWindow&) = delete;

    /// Checkout a budgeted native color frame at the requested pixel size and sample count.
    [[nodiscard]] std::optional<Frame> GetRenderFrame(VideoCore::Extent extent, u32 samples = 1,
                                                      CompletionTimeline::Deadline deadline = {});

    /// The renderer must retain the frame when encoding and supply its already issued ticket.
    /// Ownership is transferred to this window until the last GPU user retires.
    [[nodiscard]] bool Present(Frame frame, u64 rendered_tick);

    /// Wait for the current presentation work, including headless or superseded frame retirement.
    [[nodiscard]] CompletionStatus WaitPresent(CompletionTimeline::Deadline deadline = {});

    void NotifySurfaceChanged();
    void Close();

private:
    std::shared_ptr<State> state;
};

} // namespace Metal
