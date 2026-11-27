#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr u32 Side = 7;
constexpr u32 Middle = Side / 2;
using SourceImage = std::array<Pixel, Side * Side>;
using Quadrants = std::array<Pixel, 4>;

constexpr Pixel Center{200, 100, 40, 32};
constexpr Pixel Neighbor{40, 20, 8, 160};
constexpr Pixel Half{120, 60, 24, 96};
constexpr Pixel Quarter{160, 80, 32, 64};
constexpr Pixel DarkTransparent{24, 28, 32, 0};
constexpr Pixel LightOpaque{216, 220, 224, 255};
constexpr Pixel Accent{40, 200, 104, 80};
constexpr Pixel AlphaBlend{120, 124, 128, 128};

struct Pattern {
    SourceImage source;
    Quadrants expected;
};

constexpr double Luma(Pixel value) {
    return (0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2]) * (255.0 - value[3]);
}

constexpr Pixel Average(Pixel a, Pixel b) {
    return {u8((u32(a[0]) + b[0] + 1) / 2), u8((u32(a[1]) + b[1] + 1) / 2),
            u8((u32(a[2]) + b[2] + 1) / 2), u8((u32(a[3]) + b[3] + 1) / 2)};
}

constexpr u32 Cell(int x, int y) {
    return (Middle + y) * Side + Middle + x;
}

constexpr Pattern Corner(int x, int y, u32 selected) {
    Pattern pattern{};
    pattern.source.fill(Center);
    pattern.source[Cell(x, 0)] = Neighbor;
    pattern.source[Cell(0, y)] = Neighbor;
    pattern.expected.fill(Center);
    pattern.expected[selected] = Half;
    return pattern;
}

constexpr Pattern HorizontalContinuation() {
    Pattern pattern{};
    pattern.source.fill(Center);
    pattern.source[Cell(1, 0)] = Neighbor;
    pattern.source[Cell(0, -1)] = Neighbor;
    pattern.source[Cell(0, 1)] = Neighbor;
    pattern.expected = {Center, Quarter, Center, Half};
    return pattern;
}

constexpr Pattern AlphaCorner() {
    Pattern pattern{};
    pattern.source.fill(DarkTransparent);
    pattern.source[Cell(-1, 0)] = LightOpaque;
    pattern.source[Cell(0, -1)] = LightOpaque;
    pattern.source[Cell(-1, -1)] = Accent;
    pattern.expected = {AlphaBlend, DarkTransparent, DarkTransparent, DarkTransparent};
    return pattern;
}

constexpr std::array Patterns{
    Corner(-1, -1, 0), Corner(1, -1, 1),         Corner(-1, 1, 2),
    Corner(1, 1, 3),   HorizontalContinuation(), AlphaCorner(),
};

void CPU() {
    static_assert(Average(Center, Neighbor) == Half);
    static_assert(Average(Center, Half) == Quarter);
    static_assert(Average(DarkTransparent, LightOpaque) == AlphaBlend);
    static_assert(Luma(DarkTransparent) > Luma(LightOpaque));
    static_assert(Luma(Center) > Luma(Neighbor));
    for (u32 i = 0; i < 4; ++i) {
        assert(Patterns[i].source[Cell(0, 0)] == Center);
        for (u32 q = 0; q < 4; ++q) {
            assert(Patterns[i].expected[q] == (i == q ? Half : Center));
        }
    }
    assert(Patterns[4].expected[1] == Average(Center, Patterns[4].expected[3]));
    assert(Patterns[5].expected[0] == Average(DarkTransparent, LightOpaque));
    std::puts("PASS: MMPX directional palette, ordered continuation and alpha-luma byte oracles");
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"mmpx_frag");
    for (const auto& pattern : Patterns) {
        auto source = fixture.Source<Pixel>(Side, Side, MTLPixelFormatRGBA8Unorm, pattern.source);
        const auto pixels = fixture.Draw<Pixel>(Side * 2, Side * 2, {}, {source});
        for (u32 y = 0; y < 2; ++y) {
            for (u32 x = 0; x < 2; ++x) {
                const auto& pixel = pixels[(Middle * 2 + y) * Side * 2 + Middle * 2 + x];
                const auto& expected = pattern.expected[2 * y + x];
                for (u32 channel = 0; channel < 4; ++channel) {
                    // Quantization for an odd half-byte can choose an adjacent RGBA8 value.
                    const int tolerance = pattern.source[Cell(0, 0)] == DarkTransparent &&
                                                  channel == 3 && y == 0 && x == 0
                                              ? 1
                                              : 0;
                    assert(std::abs(pixel[channel] - expected[channel]) <= tolerance);
                }
            }
        }
    }
    constexpr std::array<Pixel, 1> solid{{{41, 23, 172, 251}}};
    auto source = fixture.Source<Pixel>(1, 1, MTLPixelFormatRGBA8Unorm, solid);
    for (auto pixel : fixture.Draw<Pixel>(9, 7, {{-1.5f, 2.0f}, {1.25f, -0.5f}}, {source})) {
        assert(pixel == solid.front());
    }
    fixture.Close();
    std::puts("PASS: native MMPX quadrants, exact palette blends and clamped edges");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts(
                "PENDING: native MMPX quadrants and edges require a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
