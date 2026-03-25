// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string>

#include "common/common_types.h"
#include "common/math_util.h"
#include "video_core/pica/regs_external.h"

namespace Metal {

class Instance;

struct DisplayTextureInfo {
    void* buffer = nullptr;
    u32 width = 0;
    u32 height = 0;
    u32 pixel_stride = 0;
    Pica::PixelFormat pixel_format = Pica::PixelFormat::RGBA8;
    Common::Rectangle<float> texcoords{0.0f, 0.0f, 1.0f, 1.0f};
    bool valid = false;
};

class TextureRuntime {
public:
    explicit TextureRuntime(const Instance& instance);
    ~TextureRuntime();

    void Finish();
    void SwitchDiskResources(u64 title_id);
    DisplayTextureInfo SynchronizeDisplaySurface(u32 index, PAddr addr, const u8* pixels, u32 size,
                                                 u32 width, u32 height, u32 pixel_stride,
                                                 Pica::PixelFormat pixel_format);
    void ClearDisplayTextures();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
