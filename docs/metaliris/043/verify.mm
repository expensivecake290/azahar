#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr double Coordinate(u32 index, u32 extent) {
    return (index + 0.5) / extent;
}

void CPU() {
    static_assert(Coordinate(0, 1) == 0.5);
    static_assert(Coordinate(0, 9) + Coordinate(8, 9) == 1.0);
    static_assert(Coordinate(0, 7) < 0.1);
    static_assert(Coordinate(6, 7) > 0.9);
    constexpr std::array<std::array<double, 2>, 4> screen_corners{
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}},
    };
    for (u32 corner = 0; corner < screen_corners.size(); ++corner) {
        const auto [u, v] = screen_corners[corner];
        assert((2 * u - 1 == (corner % 2 == 0 ? -1 : 1)));
        assert((1 - 2 * v == (corner < 2 ? 1 : -1)));
    }
    std::puts("PASS: independent triangle-strip corner, orientation and pixel-center oracle");
}

void GPU(const char* vertex, const char* fragment) {
    Fixture fixture(vertex, fragment, @"fixture_uv", MTLPixelFormatRGBA16Float,
                    {@"tex_coord_vert", MTLPrimitiveTypeTriangleStrip, 4});
    using Result = std::array<_Float16, 4>;
    for (const auto dimensions : {std::array{1u, 1u}, std::array{9u, 7u}}) {
        const auto [width, height] = dimensions;
        const auto pixels = fixture.Draw<Result>(width, height, {}, {});
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const auto& pixel = pixels[y * width + x];
                assert(std::abs(double(pixel[0]) - Coordinate(x, width)) < 0.00049);
                assert(std::abs(double(pixel[1]) - Coordinate(y, height)) < 0.00049);
                assert(pixel[2] == 0 && pixel[3] == 1);
            }
        }
    }
    fixture.Close();
    std::puts("PASS: native normalized strip UV, full coverage and continuous interpolation");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native strip pixels require a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
