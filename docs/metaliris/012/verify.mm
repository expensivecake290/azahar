#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CATransaction.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>
#include "video_core/renderer_metal/mtl_platform.h"

namespace {

void CheckSurfaces() {
    using Window = Frontend::EmuWindow;
    Metal::Surface headless({});
    assert(headless.IsHeadless() && !headless.IsDrawableAvailable() && !headless.GetLayer());
    assert(!Metal::RetainLayer(nil));

    __weak CAMetalLayer* weak_layer;
    @autoreleasepool {
        [CATransaction begin];
        CAMetalLayer* layer = [CAMetalLayer new];
        assert(layer);
        weak_layer = layer;
        Window::NativeSurfaceSnapshot snapshot{
            .surface = {.handle = Metal::RetainLayer(layer), .width = 640, .height = 480,
                        .scale = 2.0f, .refresh_rate = 120.0},
            .layout = {}, .generation = 7,
        };
        snapshot.layout.width = 640;
        snapshot.layout.height = 480;
        Metal::Surface surface(snapshot);
        layer = nil;
        assert(weak_layer && surface.GetLayer() == weak_layer);
        assert(!surface.IsHeadless() && surface.IsDrawableAvailable());
        assert(surface.GetSnapshot().generation == 7 && surface.GetSnapshot().layout.width == 640);
        snapshot.surface.suspended = true;
        assert(!Metal::Surface(snapshot).IsDrawableAvailable());

        int unrelated = 1;
        auto wrong = snapshot;
        wrong.surface.handle = std::shared_ptr<void>(wrong.surface.handle, &unrelated);
        try {
            (void)Metal::Surface(wrong);
            assert(false);
        } catch (const std::invalid_argument&) {
        }
        wrong.surface.handle = std::make_shared<int>(42);
        try {
            (void)Metal::Surface(wrong);
            assert(false);
        } catch (const std::invalid_argument&) {
        }
        [CATransaction commit];
    }
    [CATransaction flush];
    assert(!weak_layer);
}

bool CheckDevices() {
    const auto absent = Metal::ProbeDevice(nil);
    assert(absent.support == Metal::DeviceSupport::Unavailable && !absent.device);
    assert(absent.name.empty() && absent.registry_id == 0 && !absent.mesh_shaders);
    assert(!absent.spatial_scaling);
    assert(Metal::CanCaptureGPUTrace() == static_cast<bool>(
              [MTLCaptureManager.sharedCaptureManager supportsDestination:MTLCaptureDestinationGPUTraceDocument]));

    const auto devices = Metal::EnumerateDevices();
    id<MTLDevice> system_default = MTLCreateSystemDefaultDevice();
    if (!system_default && MTLCopyAllDevices().count == 0) {
        assert(devices.empty());
        return false;
    }
    if (system_default) {
        assert(!devices.empty() && devices.front().registry_id == system_default.registryID);
    }
    std::unordered_set<u64> unique;
    const std::size_t first_nonsystem = system_default ? 1 : 0;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& info = devices[i];
        assert(info.device && info.registry_id == info.device.registryID);
        assert(unique.insert(info.registry_id).second);
        if (i > first_nonsystem) {
            assert(devices[i - 1].registry_id < info.registry_id);
        }
        if (![info.device supportsFamily:MTLGPUFamilyMetal4]) {
            assert(info.support == Metal::DeviceSupport::MissingMetal4);
        } else if (!info.device.hasUnifiedMemory) {
            assert(info.support == Metal::DeviceSupport::MissingUnifiedMemory);
        } else {
            assert(info.support == Metal::DeviceSupport::Supported);
            assert(info.mesh_shaders ==
                   static_cast<bool>([info.device supportsFamily:MTLGPUFamilyApple7]));
            assert(info.spatial_scaling ==
                   static_cast<bool>([MTLFXSpatialScalerDescriptor supportsMetal4FX:info.device]));
        }
    }
    return true;
}

} // namespace

int main() {
    @autoreleasepool {
        CheckSurfaces();
        const bool has_device = CheckDevices();
        std::puts("PASS headless and native layer leases, suspension, invalid ownership and nil device");
        if (!has_device) {
            std::puts("PENDING live Metal device enumeration and capability matrix");
            return 2;
        }
        std::puts("PASS live device enumeration, Metal 4, unified memory, mesh and Spatial capability");
    }
}
