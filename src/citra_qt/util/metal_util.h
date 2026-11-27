// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include "qwindowdefs.h"

#ifdef ENABLE_METAL
#include <string>
#include <vector>
#include <QString>

namespace Metal {
class SurfaceAdapter;
}
#endif

namespace MetalUtil {

void* CreateMetalLayer(WId window);

#ifdef ENABLE_METAL
bool AttachMetalSurface(Metal::SurfaceAdapter& adapter, WId window);

struct DeviceInfo {
    std::string name;
    bool supported{};
    bool spatial_scaling{};
};

/// The order matches the native renderer's physical device indices.
std::vector<DeviceInfo> EnumerateDevices();
bool CanCaptureGPUTrace();
#endif

} // namespace MetalUtil

#ifdef ENABLE_METAL
std::vector<QString> GetMetalPhysicalDevices();
#endif
