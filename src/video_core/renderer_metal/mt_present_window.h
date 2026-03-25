// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <memory>

#include "common/common_types.h"
#include "common/math_util.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_external.h"

namespace Frontend {
class EmuWindow;
}

namespace Layout {
struct FramebufferLayout;
}

namespace Metal {

class Instance;
class PipelineCache;

struct ScreenInfo {
    void* buffer = nullptr;
    u32 width = 0;
    u32 height = 0;
    u32 pixel_stride = 0;
    Pica::PixelFormat pixel_format = Pica::PixelFormat::RGBA8;
    Common::Rectangle<float> texcoords{0.0f, 0.0f, 1.0f, 1.0f};
    bool valid = false;
};

class PresentWindow {
public:
    explicit PresentWindow(Frontend::EmuWindow& emu_window, const Instance& instance,
                           const PipelineCache& pipeline_cache);
    ~PresentWindow();

    void NotifySurfaceChanged();
    void Present(const Layout::FramebufferLayout& layout, const Common::Vec4f& clear_color,
                 const std::array<ScreenInfo, 3>& screen_infos);

    [[nodiscard]] bool IsValid() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
