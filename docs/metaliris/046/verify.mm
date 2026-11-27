#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr u32 Width = 5;
constexpr u32 Height = 3;
using RG = std::array<_Float16, 2>;
using Gradient = std::array<RG, Width * Height>;

constexpr Gradient Synthetic() {
    Gradient data{};
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            data[y * Width + x] = {_Float16((int(x) - 2) * 0.125 + y * 0.25),
                                   _Float16(x * 0.25 + y * 0.5)};
        }
    }
    return data;
}

constexpr std::array<Pixel, Width * Height> Colors() {
    std::array<Pixel, Width * Height> data{};
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            data[y * Width + x] = {u8(45 * x), u8(70 * y), u8(30 * (x + y)), u8((x & 1) * 255)};
        }
    }
    return data;
}

constexpr double Luma(Pixel rgba) {
    return (2627.0 * rgba[0] + 6780.0 * rgba[1] + 593.0 * rgba[2]) / 2550000.0;
}

constexpr Gradient HorizontalReference() {
    constexpr auto colors = Colors();
    Gradient result{};
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            const u32 row = y * Width;
            const double left = Luma(colors[row + (x == 0 ? 0 : x - 1)]);
            const double center = Luma(colors[row + x]);
            const double right = Luma(colors[row + std::min(x + 1, Width - 1)]);
            result[row + x] = {_Float16(right - left), _Float16(left + 2 * center + right)};
        }
    }
    return result;
}

double Expected(const Gradient& input, u32 x, u32 y) {
    const auto above = input[(y == 0 ? 0 : y - 1) * Width + x];
    const auto center = input[y * Width + x];
    const auto below = input[std::min(Height - 1, y + 1) * Width + x];
    const double dx = double(above[0]) + 2 * double(center[0]) + double(below[0]);
    const double dy = double(above[1]) - double(below[1]);
    return 1.0 - std::sqrt(dx * dx + dy * dy);
}

void CPU() {
    constexpr auto gradient = Synthetic();
    assert(std::abs(Expected(gradient, 2, 1) - (1.0 - std::sqrt(2.0))) < 1e-14);
    assert(std::abs(Expected(gradient, 2, 0) - (1.0 - std::sqrt(0.3125))) < 1e-14);
    assert(Expected(gradient, 4, 2) < 0.0);
    constexpr auto computed = HorizontalReference();
    assert(Expected(computed, 1, 1) != Expected(computed, 0, 0));
    std::puts("PASS: independent vertical gradient norm, negative range and full X/Y oracle");
}

void GPU(const char* shader, const char* vertex, const char* horizontal) {
    Fixture fixture(vertex, shader, @"y_gradient_frag", MTLPixelFormatR16Float,
                    {@"tex_coord_vert", MTLPrimitiveTypeTriangleStrip, 4});
    const auto check = [](const std::vector<_Float16>& actual, const Gradient& input) {
        for (u32 y = 0; y < Height; ++y) {
            for (u32 x = 0; x < Width; ++x) {
                assert(std::abs(double(actual[y * Width + x]) - Expected(input, x, y)) < 0.003);
            }
        }
    };
    constexpr auto input = Synthetic();
    auto gradient = fixture.Source<RG>(Width, Height, MTLPixelFormatRG16Float, input);
    check(fixture.Draw<_Float16>(Width, Height, {}, {nil, nil, gradient}), input);

    constexpr std::array<RG, 1> flat{{{0, 1}}};
    auto uniform = fixture.Source<RG>(1, 1, MTLPixelFormatRG16Float, flat);
    for (const auto value : fixture.Draw<_Float16>(9, 7, {}, {nil, nil, uniform})) {
        assert(value == 1);
    }

    constexpr auto colors = Colors();
    auto source = fixture.Source<Pixel>(Width, Height, MTLPixelFormatRGBA8Unorm, colors);
    fixture.SetFragment(horizontal, @"x_gradient_frag", MTLPixelFormatRG16Float);
    auto intermediate = fixture.Render(Width, Height, {}, {source});
    fixture.SetFragment(shader, @"y_gradient_frag", MTLPixelFormatR16Float);
    auto vertical = fixture.Render(Width, Height, {}, {nil, nil, intermediate});
    check(fixture.Read<_Float16>(vertical), HorizontalReference());
    fixture.Close();
    std::puts("PASS: exact native X-to-Y intermediate and signed R16Float output checks");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 4);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native vertical gradient and X-to-Y reuse require a Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2], argv[3]);
        }
    }
}
