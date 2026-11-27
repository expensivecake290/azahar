#include <concepts>
#include <cstdio>
#include "citra_qt/util/metal_util.h"

static_assert(std::same_as<decltype(MetalUtil::CreateMetalLayer(WId{})), void*>);
#ifdef ENABLE_METAL
using NativeBridge = bool (*)(Metal::SurfaceAdapter&, WId);
static_assert(std::same_as<decltype(&MetalUtil::AttachMetalSurface), NativeBridge>);
static_assert(std::same_as<decltype(MetalUtil::EnumerateDevices()),
                           std::vector<MetalUtil::DeviceInfo>>);
static_assert(std::same_as<decltype(MetalUtil::CanCaptureGPUTrace()), bool>);
static_assert(std::same_as<decltype(GetMetalPhysicalDevices()), std::vector<QString>>);
#endif

int main() {
#ifdef ENABLE_METAL
    std::puts("PASS selected native Qt WId bridge ABI");
#else
    std::puts("PASS existing Qt layer bridge ABI with native mode disabled");
#endif
}
