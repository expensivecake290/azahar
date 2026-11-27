#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr u32 ColorWidth = 5;
constexpr u32 ColorHeight = 3;
constexpr u32 LumaWidth = 7;
constexpr u32 LumaHeight = 5;
constexpr u32 Center = 4;

using Direction = std::array<u32, 3>;
constexpr std::array Directions{
    Direction{0, 1, 2}, Direction{8, 7, 6}, Direction{5, 1, 2}, Direction{6, 3, 7},
    Direction{5, 8, 2}, Direction{3, 0, 6}, Direction{5, 8, 7}, Direction{1, 3, 0},
};

template <u32 Width, u32 Height>
constexpr u32 AroundCenter(u32 slot) {
    return ((Height / 2 - 1) + slot / 3) * Width + (Width / 2 - 1) + slot % 3;
}

constexpr std::array<Pixel, ColorWidth * ColorHeight> Colors() {
    std::array<Pixel, ColorWidth * ColorHeight> texels{};
    for (u32 i = 0; i < texels.size(); ++i) {
        texels[i] = {u8(10 * i), u8(220 - 11 * i), u8(15 + 5 * i), u8(35 + 10 * i)};
    }
    return texels;
}

constexpr Pixel At(u32 index) {
    return Colors()[AroundCenter<ColorWidth, ColorHeight>(index)];
}

constexpr Pixel Expected(const Direction& direction) {
    Pixel result{};
    for (u32 lane = 0; lane < 4; ++lane) {
        const u32 weighted = 2 * At(Center)[lane] + At(direction[0])[lane] +
                             At(direction[1])[lane] + At(direction[2])[lane];
        result[lane] = u8((weighted + 2) / 5);
    }
    return result;
}

constexpr auto LumaPlane(const Direction& direction, _Float16 center_luma = 0.25f) {
    std::array<_Float16, LumaWidth * LumaHeight> plane{};
    plane.fill(0.125f);
    plane[AroundCenter<LumaWidth, LumaHeight>(Center)] = center_luma;
    for (const auto slot : direction) {
        plane[AroundCenter<LumaWidth, LumaHeight>(slot)] = 0.875f;
    }
    return plane;
}

void CPU() {
    static_assert(Expected(Directions[0]) == Pixel{40, 176, 35, 75});
    static_assert(Expected(Directions[1]) == Pixel{100, 110, 65, 135});
    static_assert(At(Center) == Pixel{70, 143, 50, 105});
    constexpr auto first = LumaPlane(Directions[0], _Float16(0.39990234375f));
    constexpr auto second = LumaPlane(Directions[0], _Float16(0.400146484375f));
    static_assert(first[AroundCenter<LumaWidth, LumaHeight>(Center)] < 0.4f);
    static_assert(second[AroundCenter<LumaWidth, LumaHeight>(Center)] > 0.4f);
    for (const auto& direction : Directions) {
        const auto luma = LumaPlane(direction);
        for (const auto slot : direction) {
            assert((luma[AroundCenter<LumaWidth, LumaHeight>(slot)] == 0.875f));
        }
    }
    std::puts("PASS: independent eight-direction refinement, RGBA weights and threshold oracle");
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"refine_frag");
    constexpr auto colors = Colors();
    auto image = fixture.Source<Pixel>(ColorWidth, ColorHeight, MTLPixelFormatRGBA8Unorm, colors);
    const auto check = [&](const auto& luma, const Pixel& expected) {
        auto luma_tex =
            fixture.Source<_Float16>(LumaWidth, LumaHeight, MTLPixelFormatR16Float, luma);
        const auto pixels = fixture.Draw<Pixel>(ColorWidth, ColorHeight, {}, {image, luma_tex});
        const auto& actual = pixels[AroundCenter<ColorWidth, ColorHeight>(Center)];
        for (u32 lane = 0; lane < 4; ++lane) {
            assert(std::abs(actual[lane] - expected[lane]) <= 1);
        }
    };
    for (const auto& direction : Directions) {
        check(LumaPlane(direction), Expected(direction));
    }
    check(LumaPlane(Directions[0], _Float16(0.39990234375f)), Expected(Directions[0]));
    check(LumaPlane(Directions[0], _Float16(0.400146484375f)), At(Center));
    auto tied = LumaPlane(Directions[0], 0.25f);
    tied.fill(0.25f);
    check(tied, At(Center));
    auto overlapping = LumaPlane(Directions[0]);
    overlapping[AroundCenter<LumaWidth, LumaHeight>(5)] = 0.625f;
    overlapping[AroundCenter<LumaWidth, LumaHeight>(8)] = 0.625f;
    check(overlapping, Expected(Directions[0]));
    fixture.Close();
    std::puts("PASS: native refinement selects each direction and preserves threshold fallbacks");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native refinement pixels require a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
