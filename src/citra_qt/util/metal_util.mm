// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include "citra_qt/util/metal_util.h"

#ifdef ENABLE_METAL
#include <utility>
#include "video_core/renderer_metal/mtl_platform.h"
#include "video_core/renderer_metal/mtl_surface_adapter.h"
#endif

namespace MetalUtil {

void* CreateMetalLayer(WId winId) {
    // Qt's MetalSurface QWindow on macOS does not always back the NSView
    // with a CAMetalLayer immediately; MoltenVK 1.3+ aborts in
    // MVKSurface::getNaturalExtent() if the layer is not CAMetalLayer.
    // Force-install a fresh CAMetalLayer on the view.
    CAMetalLayer* metalLayer = [CAMetalLayer layer];
    NSView* view = (__bridge NSView*)reinterpret_cast<void*>(winId);
    view.wantsLayer = YES;
    view.layer = metalLayer;
    return (__bridge void*)metalLayer;
}

#ifdef ENABLE_METAL
bool AttachMetalSurface(Metal::SurfaceAdapter& adapter, WId window) {
    return adapter.Attach(reinterpret_cast<void*>(window));
}

std::vector<DeviceInfo> EnumerateDevices() {
    auto native = Metal::EnumerateDevices();
    std::vector<DeviceInfo> devices;
    devices.reserve(native.size());
    for (auto& device : native) {
        devices.push_back({std::move(device.name),
                           device.support == Metal::DeviceSupport::Supported,
                           device.spatial_scaling});
    }
    return devices;
}

bool CanCaptureGPUTrace() {
    return Metal::CanCaptureGPUTrace();
}
#endif

} // namespace MetalUtil
