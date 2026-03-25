// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string>

#include "common/common_types.h"

namespace Frontend {
class EmuWindow;
}

namespace Metal {

class Instance {
public:
    explicit Instance(Frontend::EmuWindow& window, u32 physical_device_index);
    ~Instance();

    [[nodiscard]] bool IsValid() const;
    [[nodiscard]] bool SupportsMetal4() const;

    [[nodiscard]] void* GetDevice() const;
    [[nodiscard]] void* GetCommandQueue() const;
    [[nodiscard]] void* GetCommandAllocator() const;
    [[nodiscard]] void* GetCompiler() const;

    [[nodiscard]] std::string GetDeviceName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
