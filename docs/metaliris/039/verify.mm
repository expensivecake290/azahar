#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr u32 SourceWidth = 5;
constexpr u32 SourceHeight = 3;
constexpr u32 DestWidth = 9;
constexpr u32 DestHeight = 7;

constexpr std::array<Pixel, SourceWidth * SourceHeight> Gradient() {
    std::array<Pixel, SourceWidth * SourceHeight> values{};
    for (u32 y = 0; y < SourceHeight; ++y) {
        for (u32 x = 0; x < SourceWidth; ++x) {
            values[y * SourceWidth + x] = {u8(30 * x), u8(50 * y), u8(7 * x + 30 * y),
                                           u8(40 + 25 * x + 35 * y)};
        }
    }
    return values;
}

std::array<double, 4> Weights(double t) {
    const double s = 1 - t;
    return {s * s * s / 6, (3 * t * t * t - 6 * t * t + 4) / 6, (3 * s * s * s - 6 * s * s + 4) / 6,
            t * t * t / 6};
}

Pixel DirectBSpline(std::span<const Pixel> data, u32 width, u32 height, double u, double v) {
    const double x = u * width - 0.5;
    const double y = v * height - 0.5;
    const int left = static_cast<int>(std::floor(x));
    const int top = static_cast<int>(std::floor(y));
    const auto cx = Weights(x - left);
    const auto cy = Weights(y - top);
    std::array<double, 4> value{};
    for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
            const auto row = static_cast<u32>(std::clamp(top + dy - 1, 0, int(height) - 1));
            const auto column = static_cast<u32>(std::clamp(left + dx - 1, 0, int(width) - 1));
            const auto& sample = data[row * width + column];
            for (u32 lane = 0; lane < 4; ++lane) {
                value[lane] += cx[dx] * cy[dy] * sample[lane];
            }
        }
    }
    return {u8(std::lround(value[0])), u8(std::lround(value[1])), u8(std::lround(value[2])),
            u8(std::lround(value[3]))};
}

void CPU() {
    for (int i = 0; i <= 32; ++i) {
        const auto w = Weights(i / 32.0);
        assert(std::abs(w[0] + w[1] + w[2] + w[3] - 1) < 1e-14);
        for (const auto weight : w) {
            assert(weight >= 0);
        }
    }
    const auto gradient = Gradient();
    assert(DirectBSpline(gradient, SourceWidth, SourceHeight, -1, -1) == gradient.front());
    assert(DirectBSpline(gradient, SourceWidth, SourceHeight, 2, 2) == gradient.back());
    assert(
        (DirectBSpline(gradient, SourceWidth, SourceHeight, 0.5, 0.5) == Pixel{60, 50, 44, 125}));
    constexpr std::array<Pixel, 1> solid{{{9, 21, 164, 239}}};
    assert(DirectBSpline(solid, 1, 1, -0.25, 1.125) == solid.front());
    std::puts("PASS: independent direct 16-tap bicubic oracle and all four channel checks");
}

void GPU(const char* fragment, const char* vertex) {
    Fixture fixture(vertex, fragment, @"bicubic_frag");
    const auto gradient = Gradient();
    auto source =
        fixture.Source<Pixel>(SourceWidth, SourceHeight, MTLPixelFormatRGBA8Unorm, gradient);
    const std::array transforms{Uniforms{}, Uniforms{{1.4f, -1.5f}, {-0.2f, 1.25f}}};
    for (const auto& transform : transforms) {
        const auto actual = fixture.Draw<Pixel>(DestWidth, DestHeight, transform, {source});
        for (u32 y = 0; y < DestHeight; ++y) {
            for (u32 x = 0; x < DestWidth; ++x) {
                const double u = (x + 0.5) * transform.scale[0] / DestWidth + transform.offset[0];
                const double v = (y + 0.5) * transform.scale[1] / DestHeight + transform.offset[1];
                const auto expected = DirectBSpline(gradient, SourceWidth, SourceHeight, u, v);
                for (u32 lane = 0; lane < expected.size(); ++lane) {
                    assert(std::abs(actual[y * DestWidth + x][lane] - expected[lane]) <= 1);
                }
            }
        }
    }
    constexpr std::array<Pixel, 1> solid{{{9, 21, 164, 239}}};
    auto one = fixture.Source<Pixel>(1, 1, MTLPixelFormatRGBA8Unorm, solid);
    for (const auto& pixel : fixture.Draw<Pixel>(DestWidth, DestHeight, transforms[1], {one})) {
        assert(pixel == solid.front());
    }
    fixture.Close();
    std::puts("PASS: native bicubic pixels match the direct separable 16-tap oracle");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native bicubic pixels require a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
