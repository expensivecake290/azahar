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

constexpr std::array<Pixel, Width * Height> Image() {
    std::array<Pixel, Width * Height> pixels{};
    constexpr std::array reds{0u, 255u, 40u, 200u, 40u};
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            pixels[y * Width + x] = {u8(reds[x]), u8((220 - 35 * x) / (y + 1)), u8(41 * y),
                                     u8(x % 2 ? 255 : 0)};
        }
    }
    return pixels;
}

constexpr double Luma(Pixel value) {
    return (2627.0 * value[0] + 6780.0 * value[1] + 593.0 * value[2]) / 2550000.0;
}

constexpr std::array<double, 2> Expected(u32 x, u32 y) {
    constexpr auto input = Image();
    const auto row = y * Width;
    const double a = Luma(input[row + (x == 0 ? 0 : x - 1)]);
    const double b = Luma(input[row + x]);
    const double c = Luma(input[row + std::min(Width - 1, x + 1)]);
    return {c - a, a + 2 * b + c};
}

void CPU() {
    static_assert(Luma({255, 255, 255, 0}) == 1.0);
    static_assert(Luma({255, 255, 255, 255}) == 1.0);
    static_assert(Luma({0, 0, 0, 255}) == 0.0);
    static_assert(Expected(0, 0)[0] > 0);
    static_assert(Expected(2, 0)[0] < 0);
    static_assert(Expected(1, 0)[1] > 1);
    static_assert(Expected(1, 2)[1] != Expected(1, 0)[1]);
    std::puts("PASS: independent X gradient signed, smooth and alpha-independent luma oracle");
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"x_gradient_frag", MTLPixelFormatRG16Float,
                    {@"tex_coord_vert", MTLPrimitiveTypeTriangleStrip, 4});
    constexpr auto input = Image();
    auto texture = fixture.Source<Pixel>(Width, Height, MTLPixelFormatRGBA8Unorm, input);
    using RG = std::array<_Float16, 2>;
    const auto actual = fixture.Draw<RG>(Width, Height, {}, {texture});
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            const auto expected = Expected(x, y);
            assert(std::abs(actual[y * Width + x][0] - expected[0]) <= 0.002);
            assert(std::abs(actual[y * Width + x][1] - expected[1]) <= 0.002);
        }
    }

    constexpr std::array<Pixel, 1> solid{{{255, 255, 255, 0}}};
    auto edge = fixture.Source<Pixel>(1, 1, MTLPixelFormatRGBA8Unorm, solid);
    for (const auto& pixel : fixture.Draw<RG>(7, 3, {}, {edge})) {
        assert(pixel[0] == 0 && pixel[1] == 4);
    }
    fixture.Close();
    std::puts("PASS: native X gradient signed and unsaturated RG16Float pixels");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native X gradient RG16Float pixels require a Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
