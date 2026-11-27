#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include "../verify_present.h"

namespace {

using namespace VerifyPresent;

using Color = std::array<double, 4>;
constexpr std::array<Pixel, 3> Sources{{
    {255, 0, 0, 63},
    {0, 255, 0, 197},
    {0, 0, 255, 91},
}};
constexpr std::array EyePairs{
    std::array{0, 1},  std::array{1, 2}, std::array{2, 0},  std::array{2, 2},
    std::array{-1, 2}, std::array{1, 3}, std::array{-1, 3},
};

Color Expected(s32 left, s32 right) {
    constexpr std::array<std::array<double, 3>, 3> LeftContributions{{
        {0.437, -0.062, -0.048},
        {0.449, -0.062, -0.050},
        {0.164, -0.024, -0.017},
    }};
    constexpr std::array<std::array<double, 3>, 3> RightContributions{{
        {-0.011, 0.377, -0.026},
        {-0.032, 0.761, -0.093},
        {-0.007, 0.009, 1.234},
    }};
    constexpr Pixel no_source{0, 0, 0, 255};
    const auto a = left >= 0 && left < 3 ? Sources[left] : no_source;
    const auto b = right >= 0 && right < 3 ? Sources[right] : no_source;
    Color color{0, 0, 0, a[3] / 255.0};
    for (u32 output = 0; output < 3; ++output) {
        for (u32 input = 0; input < 3; ++input) {
            color[output] += (a[input] * LeftContributions[input][output] +
                              b[input] * RightContributions[input][output]) /
                             255.0;
        }
    }
    return color;
}

void CPU() {
    const auto red_green = Expected(0, 1);
    assert(std::abs(red_green[0] - 0.405) < 1e-14);
    assert(std::abs(red_green[1] - 0.699) < 1e-14);
    assert(std::abs(red_green[2] + 0.141) < 1e-14);
    assert(red_green[3] == 63.0 / 255.0);
    const auto blue = Expected(-1, 2);
    assert(std::abs(blue[2] - 1.234) < 1e-14 && blue[3] == 1);
    assert((Expected(-1, 3) == Color{0, 0, 0, 1}));
    std::puts("PASS: independent Dubois channel projection, signed range and left-alpha oracle");
}

template <typename Output>
void GPU(const char* shader, const char* vertex, MTLPixelFormat format) {
    constexpr u32 width = 7;
    constexpr u32 height = 5;
    Fixture fixture(vertex, shader, @"present_anaglyph_frag", true, format);
    std::array<id<MTLTexture>, 3> screens{};
    for (u32 screen = 0; screen < Sources.size(); ++screen) {
        std::array<Pixel, 15> surface{};
        surface.fill(Sources[screen]);
        screens[screen] = fixture.Source(5, 3, surface);
    }
    constexpr auto geometry = Vertices(0, 0, width, height);
    for (const auto [left, right] : EyePairs) {
        Uniforms uniforms{.matrix = Orthographic(width, height),
                          .left_screen = left,
                          .right_screen = right,
                          .layer = 1};
        const auto pixels = fixture.Draw<Output>(width, height, uniforms, screens, geometry);
        const auto expected = Expected(left, right);
        for (const auto& pixel : pixels) {
            for (u32 lane = 0; lane < 4; ++lane) {
                if constexpr (std::is_same_v<Output, Pixel>) {
                    const auto byte = std::lround(std::clamp(expected[lane], 0.0, 1.0) * 255);
                    assert(std::abs(pixel[lane] - byte) <= 1);
                } else {
                    assert(std::abs(double(pixel[lane]) - expected[lane]) < 0.0006);
                }
            }
        }
    }
    fixture.Close();
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native anaglyph projection pixels require a Metal 4 GPU.");
        } else {
            GPU<Pixel>(argv[1], argv[2], MTLPixelFormatRGBA8Unorm);
            GPU<std::array<_Float16, 4>>(argv[1], argv[2], MTLPixelFormatRGBA16Float);
            std::puts("PASS: native normalized and signed half-float anaglyph pixel outputs");
        }
    }
}
