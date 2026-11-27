#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include "../verify_present.h"

namespace {

using namespace VerifyPresent;

constexpr u32 SrcWidth = 5;
constexpr u32 SrcHeight = 3;
constexpr u32 DstWidth = 9;
constexpr u32 DstHeight = 7;
constexpr Pixel Invalid{0, 0, 0, 255};

constexpr Pixel Pattern(u32 screen, u32 x, u32 y) {
    return {u8(60 * screen + 10 * x), u8(10 * screen + 50 * y), u8(170 - 30 * screen - 11 * x),
            u8(100 + 40 * screen + 8 * y)};
}

constexpr Pixel Expected(s32 screen, u32 x, u32 y) {
    if (screen < 0 || screen >= 3) {
        return Invalid;
    }
    const u32 source_x = (2 * x + 1) * SrcWidth / (2 * DstWidth);
    const u32 source_y = (2 * y + 1) * SrcHeight / (2 * DstHeight);
    return Pattern(screen, source_x, source_y);
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"present_frag", false);
    std::array<id<MTLTexture>, 3> screens{};
    for (u32 screen = 0; screen < screens.size(); ++screen) {
        std::array<Pixel, SrcWidth * SrcHeight> pattern{};
        for (u32 y = 0; y < SrcHeight; ++y) {
            for (u32 x = 0; x < SrcWidth; ++x) {
                pattern[y * SrcWidth + x] = Pattern(screen, x, y);
            }
        }
        screens[screen] = fixture.Source(SrcWidth, SrcHeight, pattern);
    }
    for (const s32 id : {0, 1, 2, -1, 3, std::numeric_limits<s32>::max()}) {
        Uniforms data{.left_screen = id, .right_screen = 2, .layer = 1, .reverse_interlaced = 1};
        const auto actual = fixture.Draw(DstWidth, DstHeight, data, screens);
        for (u32 y = 0; y < DstHeight; ++y) {
            for (u32 x = 0; x < DstWidth; ++x) {
                assert(actual[y * DstWidth + x] == Expected(id, x, y));
            }
        }
    }
    fixture.Close();
    std::puts("PASS: exact native RGB/alpha for all three screen views and invalid selectors");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        static_assert(Expected(-1, 4, 3) == Invalid);
        static_assert(Expected(3, 4, 3) == Invalid);
        static_assert(Expected(1, 0, 0) == Pixel{60, 10, 140, 140});
        static_assert(Expected(2, 8, 6) == Pixel{160, 120, 66, 196});
        static_assert(Expected(0, 4, 3) == Pixel{20, 50, 148, 108});
        std::puts("PASS: independent three-screen RGBA selection, bounds and odd-pixel oracle");
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native presentation screen selection requires a Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
