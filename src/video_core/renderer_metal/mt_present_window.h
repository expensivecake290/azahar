// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>

#include "common/vector_math.h"

namespace Frontend {
class EmuWindow;
}

namespace Layout {
struct FramebufferLayout;
}

namespace Metal {

class Instance;
class PipelineCache;

class PresentWindow {
public:
    explicit PresentWindow(Frontend::EmuWindow& emu_window, const Instance& instance,
                           const PipelineCache& pipeline_cache);
    ~PresentWindow();

    void NotifySurfaceChanged();
    void Present(const Layout::FramebufferLayout& layout, const Common::Vec4f& clear_color);

    [[nodiscard]] bool IsValid() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
