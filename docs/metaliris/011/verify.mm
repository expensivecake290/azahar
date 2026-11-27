#include <concepts>
#include <type_traits>
#include "video_core/renderer_metal/mtl_platform.h"

static_assert(std::same_as<decltype(Metal::ProbeDevice(nil)), Metal::DeviceInfo>);
static_assert(std::same_as<decltype(Metal::EnumerateDevices()), std::vector<Metal::DeviceInfo>>);
static_assert(std::same_as<decltype(Metal::RetainLayer(nil)), std::shared_ptr<void>>);
static_assert(std::is_constructible_v<Metal::Surface, Frontend::EmuWindow::NativeSurfaceSnapshot>);
static_assert(requires(const Metal::Surface& surface) {
    { surface.GetLayer() } -> std::same_as<CAMetalLayer*>;
    { surface.IsHeadless() } -> std::same_as<bool>;
    { surface.IsDrawableAvailable() } -> std::same_as<bool>;
    { surface.GetSnapshot() } -> std::same_as<const Frontend::EmuWindow::NativeSurfaceSnapshot&>;
});
