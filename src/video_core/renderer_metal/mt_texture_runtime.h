// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string>

#include "common/common_types.h"

namespace Metal {

class Instance;

class TextureRuntime {
public:
    explicit TextureRuntime(const Instance& instance);
    ~TextureRuntime();

    void Finish();
    void SwitchDiskResources(u64 title_id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
