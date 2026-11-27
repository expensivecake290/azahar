#include <array>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_swapchain.h"

using namespace Metal;
static_assert(std::is_nothrow_move_constructible_v<Swapchain::Drawable>);
static_assert(!std::is_copy_constructible_v<Swapchain::Drawable>);
static_assert(
    std::is_same_v<decltype(std::declval<Swapchain::Drawable>().Texture()), id<MTLTexture>>);
static_assert(
    std::is_same_v<decltype(std::declval<Swapchain&>().GetSurface()), std::optional<Surface>>);
static_assert(requires(Swapchain& swap, Swapchain::Drawable drawable,
                       Frontend::EmuWindow::NativeSurfaceSnapshot snapshot,
                       Swapchain::DrawableEncoder encode) {
    { swap.UpdateSurface(snapshot, PresentPacing{}) } -> std::same_as<bool>;
    { swap.AcquireNextImage() } -> std::same_as<Swapchain::Acquisition>;
    { swap.Present(std::move(drawable), encode, 0) } -> std::same_as<std::optional<u64>>;
});

int main() {
    @autoreleasepool {
        using enum PresentMode;
        constexpr std::array capabilities{std::array{false, false}, std::array{false, true},
                                          std::array{true, false}, std::array{true, true}};
        const std::array preferred_when_unsynced{Fifo, Immediate, Mailbox, Immediate};
        const std::array preferred_when_limited{Fifo, Immediate, Mailbox, Mailbox};
        for (u32 i = 0; i < capabilities.size(); ++i) {
            const auto [mailbox, immediate] = capabilities[i];
            for (double frame_limit : {0., 40., 99.999, 100., 100.001, 200.}) {
                for (bool low_refresh : {false, true}) {
                    const PresentPacing fast{false, frame_limit, low_refresh};
                    const PresentPacing sync{true, frame_limit, low_refresh};
                    assert(fast.SelectMode(mailbox, immediate) == preferred_when_unsynced[i]);
                    assert(sync.SelectMode(mailbox, immediate) ==
                           (low_refresh || frame_limit == 0 || frame_limit > 100
                                ? preferred_when_limited[i]
                                : Fifo));
                }
            }
        }
        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.allowsNextDrawableTimeout = YES;
        layer.maximumDrawableCount = 3;
        layer.drawableSize = CGSizeMake(901, 571);
        assert(layer.allowsNextDrawableTimeout && layer.maximumDrawableCount == 3);
        assert(layer.drawableSize.width == 901 && layer.drawableSize.height == 571);
        std::puts(
            "PASS native drawable ABI, 96 pacing cases and CAMetalLayer size/pool configuration");
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING physical Metal 4 drawable lifecycle and pacing tests");
        }
    }
}
