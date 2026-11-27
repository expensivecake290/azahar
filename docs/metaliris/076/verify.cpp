#include <cassert>
#include <concepts>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_surface_adapter.h"

using namespace Metal;

static_assert(!std::is_copy_constructible_v<SurfaceAdapter>);
static_assert(requires(SurfaceAdapter& adapter, void* view) {
    { adapter.Attach(view) } -> std::same_as<bool>;
    { adapter.Capture({}) } -> std::same_as<std::optional<Frontend::EmuWindow::NativeSurface>>;
    { adapter.Detach() } -> std::same_as<bool>;
});

int main() {
    const auto odd = DrawableExtentForBackingSize(901.499, 571.5);
    assert(odd && odd->width == 901 && odd->height == 572);
    const auto retina = DrawableExtentForBackingSize(800 * 2, 377.5 * 2);
    assert(retina && retina->width == 1600 && retina->height == 755);
    const auto clamped = DrawableExtentForBackingSize(20000, 16383.7);
    assert(clamped && clamped->width == 16384 && clamped->height == 16384);
    for (double width : {0., 0.25, 0.49}) {
        const auto empty = DrawableExtentForBackingSize(width, 12);
        assert(empty && !empty->width && !empty->height);
    }
    for (double invalid : {-1., std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(), 4294967296.}) {
        assert(!DrawableExtentForBackingSize(invalid, 72));
        assert(!DrawableExtentForBackingSize(72, invalid));
    }
    Frontend::EmuWindow::NativeSurface headless;
    assert(headless.IsHeadless() && !headless.IsDrawableAvailable());
    std::puts(
        "PASS frontend-neutral C++ surface ABI, exact Retina rounding, bounds and headless state");
}
