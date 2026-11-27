// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#import <Metal/Metal.h>

#include <memory>
#include <string>
#include <vector>
#include "common/common_types.h"
#include "core/frontend/emu_window.h"

@class CAMetalLayer;

namespace Metal {

enum class DeviceSupport : u8 {
    Unavailable,
    MissingMetal4,
    MissingUnifiedMemory,
    Supported,
};

struct DeviceInfo {
    id<MTLDevice> device = nil;
    std::string name;
    u64 registry_id{};
    DeviceSupport support{DeviceSupport::Unavailable};
    bool mesh_shaders{};
    bool spatial_scaling{};
};

[[nodiscard]] DeviceInfo ProbeDevice(id<MTLDevice> device);

/// Returns all detected devices, with the system default first and remaining devices by registry ID.
[[nodiscard]] std::vector<DeviceInfo> EnumerateDevices();

[[nodiscard]] bool CanCaptureGPUTrace();

/// Publish this retained object as the macOS native surface handle.
[[nodiscard]] std::shared_ptr<void> RetainLayer(CAMetalLayer* layer);

/// A retained native presentation surface paired with one published layout and generation.
class Surface {
public:
    /// The snapshot must be headless or have a handle created by RetainLayer.
    explicit Surface(Frontend::EmuWindow::NativeSurfaceSnapshot snapshot);

    [[nodiscard]] CAMetalLayer* GetLayer() const noexcept;

    [[nodiscard]] bool IsHeadless() const noexcept {
        return snapshot.surface.IsHeadless();
    }

    [[nodiscard]] bool IsDrawableAvailable() const noexcept {
        return snapshot.surface.IsDrawableAvailable();
    }

    [[nodiscard]] const Frontend::EmuWindow::NativeSurfaceSnapshot& GetSnapshot() const noexcept {
        return snapshot;
    }

private:
    Frontend::EmuWindow::NativeSurfaceSnapshot snapshot;
};

} // namespace Metal
