// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>
#include "video_core/renderer_metal/mtl_platform.h"

namespace Metal {

namespace {

struct ReleaseLayer {
    void* retained_handle;

    void operator()(void* handle) const noexcept {
        (void)CFBridgingRelease(handle);
    }
};

} // namespace

DeviceInfo ProbeDevice(id<MTLDevice> device) {
    if (!device) {
        return {};
    }
    DeviceInfo result;
    result.device = device;
    result.registry_id = device.registryID;
    if (const char* name = device.name.UTF8String) {
        result.name = name;
    }
    if (![device supportsFamily:MTLGPUFamilyMetal4]) {
        result.support = DeviceSupport::MissingMetal4;
        return result;
    }
    if (!device.hasUnifiedMemory) {
        result.support = DeviceSupport::MissingUnifiedMemory;
        return result;
    }
    result.support = DeviceSupport::Supported;
    result.mesh_shaders = [device supportsFamily:MTLGPUFamilyApple7];
    result.spatial_scaling = [MTLFXSpatialScalerDescriptor supportsMetal4FX:device];
    return result;
}

std::vector<DeviceInfo> EnumerateDevices() {
    NSArray<id<MTLDevice>>* native_devices = MTLCopyAllDevices();
    std::vector<DeviceInfo> devices;
    devices.reserve(native_devices.count);
    for (id<MTLDevice> device in native_devices) {
        devices.push_back(ProbeDevice(device));
    }
    std::ranges::sort(devices, {}, &DeviceInfo::registry_id);
    const auto duplicates = std::ranges::unique(devices, {}, &DeviceInfo::registry_id);
    devices.erase(duplicates.begin(), duplicates.end());

    if (id<MTLDevice> system_default = MTLCreateSystemDefaultDevice()) {
        const auto match = std::ranges::find(devices, system_default.registryID,
                                             &DeviceInfo::registry_id);
        if (match == devices.end()) {
            devices.insert(devices.begin(), ProbeDevice(system_default));
        } else {
            std::rotate(devices.begin(), match, std::next(match));
        }
    }
    return devices;
}

bool CanCaptureGPUTrace() {
    return [MTLCaptureManager.sharedCaptureManager
        supportsDestination:MTLCaptureDestinationGPUTraceDocument];
}

std::shared_ptr<void> RetainLayer(CAMetalLayer* layer) {
    if (!layer) {
        return {};
    }
    void* handle = (__bridge_retained void*)layer;
    return std::shared_ptr<void>(handle, ReleaseLayer{handle});
}

Surface::Surface(Frontend::EmuWindow::NativeSurfaceSnapshot snapshot_)
    : snapshot(std::move(snapshot_)) {
    const auto& handle = snapshot.surface.handle;
    if (handle) {
        const auto* owner = std::get_deleter<ReleaseLayer>(handle);
        if (!owner || owner->retained_handle != handle.get()) {
            throw std::invalid_argument("Metal surface handle was not created by RetainLayer");
        }
    }
}

CAMetalLayer* Surface::GetLayer() const noexcept {
    return (__bridge CAMetalLayer*)snapshot.surface.handle.get();
}

} // namespace Metal
