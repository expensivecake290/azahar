#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cassert>
#include <cstdio>
#include <thread>
#include "citra_qt/util/metal_util.h"
#include "video_core/renderer_metal/mtl_platform.h"
#include "video_core/renderer_metal/mtl_surface_adapter.h"

namespace {

WId WindowId(NSView* view) {
    return reinterpret_cast<WId>((__bridge void*)view);
}

void ProbeBridge() {
    auto* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 71, 39)];
    Metal::SurfaceAdapter adapter;
    assert(!MetalUtil::AttachMetalSurface(adapter, WId{}));
    assert(MetalUtil::AttachMetalSurface(adapter, WindowId(view)));
    const auto surface = adapter.Capture();
    assert(surface && surface->width == 71 && surface->height == 39);
    Metal::Surface native({.surface = *surface, .generation = 1});
    assert(native.GetLayer() == view.layer && native.GetLayer());
    std::thread other([&] { assert(!MetalUtil::AttachMetalSurface(adapter, WindowId(view))); });
    other.join();
    assert(adapter.Detach());
    assert(native.GetLayer());

    auto* existing_consumer = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 32, 16)];
    const auto raw = MetalUtil::CreateMetalLayer(WindowId(existing_consumer));
    assert(raw == (__bridge void*)existing_consumer.layer);
    assert([existing_consumer.layer isKindOfClass:CAMetalLayer.class]);

    const auto expected = Metal::EnumerateDevices();
    const auto reported = MetalUtil::EnumerateDevices();
    assert(expected.size() == reported.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& device = expected[i];
        assert(reported[i].name == device.name &&
               reported[i].supported == (device.support == Metal::DeviceSupport::Supported) &&
               reported[i].spatial_scaling == device.spatial_scaling);
    }
    assert(MetalUtil::CanCaptureGPUTrace() == Metal::CanCaptureGPUTrace());
    std::printf("PASS real Qt WId and AppKit ownership bridge, ordered device enumeration (%zu "
                "devices), capture capability\n",
                reported.size());
    if (reported.empty()) {
        std::puts(
            "PENDING physical native Metal device properties, Qt live drawable and trace capture");
    }
}

} // Anonymous namespace

int main() {
    @autoreleasepool {
        ProbeBridge();
    }
}
