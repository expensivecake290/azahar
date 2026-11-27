#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr u32 Side = 7;
constexpr u32 Middle = Side / 2;
using Image = std::array<Pixel, Side * Side>;
constexpr Pixel Low{20, 60, 100, 0};
constexpr Pixel High{20, 60, 100, 255};
constexpr double RoundedOrigin = 0.7071067811865475244;

constexpr u32 Cell(int x, int y) {
    return (Middle + y) * Side + Middle + x;
}

constexpr Image Corner(int x, int y, bool rounded) {
    Image image{};
    image.fill(Low);
    image[Cell(x, 0)] = High;
    image[Cell(0, y)] = High;
    image[Cell(x, y)] = High;
    if (rounded) {
        image[Cell(-x, y)] = High;
        image[Cell(x, -y)] = High;
    }
    return image;
}

u8 AlphaAt(int sx, int sy, u32 x, u32 y, u32 scale, bool rounded) {
    // The input colors differ only in alpha. Their distance is 1, so the center has one
    // active diagonal blend and the signed perpendicular forms a simple scalar curve.
    const double px = sx * ((x + 0.5) / scale - 0.5);
    const double py = sy * ((y + 0.5) / scale - 0.5);
    const double origin = rounded ? RoundedOrigin : 0.5;
    const double t = std::clamp(0.5 + 0.5 * scale * (px + py - origin), 0.0, 1.0);
    return u8(std::lround(255 * t * t * (3.0 - 2.0 * t)));
}

void CPU() {
    static_assert(offsetof(Uniforms, resolution_scale) == 16);
    assert(AlphaAt(1, 1, 0, 0, 2, false) == 0);
    assert(AlphaAt(1, 1, 1, 1, 2, false) == 128);
    assert(AlphaAt(1, 1, 1, 1, 2, true) == 53);
    assert(AlphaAt(1, 1, 3, 3, 4, false) == 255);
    assert(AlphaAt(-1, -1, 0, 0, 4, false) == 255);
    assert(AlphaAt(1, -1, 3, 0, 4, false) == 255);
    assert(AlphaAt(-1, 1, 0, 3, 4, false) == 255);
    assert(AlphaAt(1, 1, 0, 0, 1, false) == 40);
    std::puts("PASS: independent xBRZ diagonal, rounded mask and free-scale alpha oracles");
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"xbrz_freescale_frag");
    for (const int sx : {-1, 1}) {
        for (const int sy : {-1, 1}) {
            for (const bool rounded : {false, true}) {
                const auto corner = Corner(sx, sy, rounded);
                auto texture = fixture.Source<Pixel>(Side, Side, MTLPixelFormatRGBA8Unorm, corner);
                for (const u32 scale : {1u, 2u, 4u}) {
                    const auto pixels = fixture.Draw<Pixel>(
                        Side * scale, Side * scale, {.resolution_scale = float(scale)}, {texture});
                    for (u32 y = 0; y < scale; ++y) {
                        for (u32 x = 0; x < scale; ++x) {
                            const auto& pixel =
                                pixels[(Middle * scale + y) * Side * scale + Middle * scale + x];
                            assert(pixel[0] == Low[0] && pixel[1] == Low[1] && pixel[2] == Low[2]);
                            assert(std::abs(pixel[3] - AlphaAt(sx, sy, x, y, scale, rounded)) <= 1);
                        }
                    }
                }
            }
        }
    }
    constexpr std::array<Pixel, 1> solid{{{16, 77, 240, 195}}};
    auto texture = fixture.Source<Pixel>(1, 1, MTLPixelFormatRGBA8Unorm, solid);
    const auto output =
        fixture.Draw<Pixel>(9, 7, {{-1.2f, 1.5f}, {1.05f, -0.15f}, 2.5f}, {texture});
    for (const auto& pixel : output) {
        assert(pixel == solid.front());
    }
    fixture.Close();
    std::puts("PASS: native xBRZ directional, rounded, scale and clamped-edge pixels");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native xBRZ output requires a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
