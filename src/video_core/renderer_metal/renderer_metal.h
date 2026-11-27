// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <memory>
#include "video_core/renderer_base.h"

namespace Pica {
class PicaCore;
}

namespace Metal {

class RendererMetal final : public VideoCore::RendererBase {
public:
    explicit RendererMetal(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window,
                           Frontend::EmuWindow* secondary_window);
    ~RendererMetal() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override;

    void NotifySurfaceChanged(bool is_secondary) override;
    void SwapBuffers() override;

    using VideoCore::RendererBase::TryPresent;
    void TryPresent(int timeout_ms, bool is_secondary) override;

private:
    void LimitFramesAheadOfGpu() override;

    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
