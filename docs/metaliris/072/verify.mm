#include <array>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_present_window.h"

using namespace Metal;

static_assert(std::is_nothrow_move_constructible_v<PresentWindow::Frame>);
static_assert(!std::is_copy_constructible_v<PresentWindow::Frame>);
static_assert(PresentWindow::FrameColorFormat == MTLPixelFormatBGRA8Unorm);
static_assert(requires(PresentWindow& present, PresentWindow::Frame frame,
                       Scheduler& scheduler) {
    { present.GetRenderFrame({9, 7}, 4) } -> std::same_as<std::optional<PresentWindow::Frame>>;
    { frame.MakePass(MTLClearColorMake(0, 0, 0, 1)) } -> std::same_as<RenderPass>;
    { frame.Retain(scheduler) } -> std::same_as<void>;
    { present.Present(std::move(frame), 1) } -> std::same_as<bool>;
    { present.WaitPresent() } -> std::same_as<CompletionStatus>;
});

int main() {
    using enum PresentTransfer;
    using Extent = VideoCore::Extent;
    constexpr std::array formats{MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA8Unorm,
                                 MTLPixelFormatBGRA8Unorm_sRGB, MTLPixelFormatRGBA8Unorm_sRGB};
    constexpr Extent original{9, 7};
    constexpr Extent odd_scaled{17, 11};
    for (const auto source : formats) {
        for (const auto target : formats) {
            for (bool filter : {false, true}) {
                assert(SelectPresentTransfer(original, original, source, target, filter) ==
                       (source == target ? Copy : Nearest));
                assert(SelectPresentTransfer(original, odd_scaled, source, target, filter) ==
                       (filter ? Linear : Nearest));
            }
        }
    }
    assert(SelectPresentTransfer({0, 7}, original, formats[0], formats[0], true) == Invalid);
    assert(SelectPresentTransfer(original, {9, 0}, formats[0], formats[0], true) == Invalid);
    assert(SelectPresentTransfer(original, original, MTLPixelFormatDepth32Float, formats[0],
                                 false) == Invalid);
    assert(SelectPresentTransfer(original, original, formats[0], MTLPixelFormatInvalid, false) ==
           Invalid);
    @autoreleasepool {
        auto pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        assert(pass.colorAttachments[0].storeAction == MTLStoreActionMultisampleResolve);
    }
    std::puts("PASS native movable frame, resolve ABI and 68 exact presentation transfer cases");
}
