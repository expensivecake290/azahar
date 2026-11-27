// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include "core/frontend/emu_window.h"
#include "video_core/rasterizer_cache/utils.h"

namespace Metal {

struct SurfaceOptions {
    bool fullscreen{};
    bool portrait{};
    bool suspended{};
};

[[nodiscard]] constexpr std::optional<VideoCore::Extent> DrawableExtentForBackingSize(
    double width, double height) noexcept {
    constexpr double largest = std::numeric_limits<u32>::max();
    constexpr double max_dimension = 16384;
    if (!(width >= 0 && width <= largest && height >= 0 && height <= largest)) {
        return std::nullopt;
    }
    const auto pixels_wide = static_cast<u32>(std::min(width + 0.5, max_dimension));
    const auto pixels_high = static_cast<u32>(std::min(height + 0.5, max_dimension));
    return pixels_wide && pixels_high ? VideoCore::Extent{pixels_wide, pixels_high}
                                      : VideoCore::Extent{0, 0};
}

/// Owns the native view attachment on the main thread. Frontends publish the captured surface
/// together with their layout via EmuWindow. Snapshots retain their layers independently.
class SurfaceAdapter final {
public:
    SurfaceAdapter();
    ~SurfaceAdapter();
    SurfaceAdapter(const SurfaceAdapter&) = delete;
    SurfaceAdapter& operator=(const SurfaceAdapter&) = delete;

    /// The opaque handle is a live NSView supplied by the frontend.
    [[nodiscard]] bool Attach(void* native_view);

    /// Returns headless state when detached. Invalid geometry and off-main callers have no result.
    [[nodiscard]] std::optional<Frontend::EmuWindow::NativeSurface> Capture(
        SurfaceOptions options = {}) const;

    /// Detach the AppKit view on the main thread. Previously published layers retain ownership.
    [[nodiscard]] bool Detach();

private:
    struct State;
    std::shared_ptr<State> state;
};

} // namespace Metal
