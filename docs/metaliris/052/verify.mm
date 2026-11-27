#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include "../verify_present.h"

namespace {

using namespace VerifyPresent;

constexpr u32 Width = 9;
constexpr u32 Height = 7;
constexpr u32 SourceWidth = 5;
constexpr u32 SourceHeight = 3;
constexpr u32 RectWidth = 6;
constexpr u32 RectHeight = 5;

constexpr Pixel Pattern(u32 x, u32 y) {
    return {u8(10 + 25 * x), u8(60 + 60 * y), u8(80 + 10 * x + 5 * y), u8(100 + 11 * x + 9 * y)};
}

constexpr Pixel Expected(u32 x, u32 y, u32 left, u32 top) {
    if (x < left || x >= left + RectWidth || y < top || y >= top + RectHeight) {
        return Background;
    }
    const u32 sx = (2 * RectWidth - (2 * (x - left) + 1)) * SourceWidth / (2 * RectWidth);
    const u32 sy = (2 * (y - top) + 1) * SourceHeight / (2 * RectHeight);
    return Pattern(sx, sy);
}

void CPU() {
    constexpr auto matrix = Orthographic(Width, Height);
    static_assert(matrix[3] == -1.0f && matrix[7] == -1.0f && matrix[15] == 1.0f);
    static_assert(matrix[0] * Width == 2.0f && matrix[5] * Height == 2.0f);
    static_assert(Expected(0, 0, 1, 1) == Background);
    static_assert(Expected(1, 1, 1, 1) == Pattern(4, 0));
    static_assert(Expected(6, 5, 1, 1) == Pattern(0, 2));
    std::puts("PASS: independent presentation transform, odd rectangle, UV and layout oracle");
}

void GPU(const char* vertex, const char* shader) {
    Fixture fixture(vertex, shader, @"present_frag", true);
    std::array<Pixel, SourceWidth * SourceHeight> color{};
    for (u32 y = 0; y < SourceHeight; ++y) {
        for (u32 x = 0; x < SourceWidth; ++x) {
            color[y * SourceWidth + x] = Pattern(x, y);
        }
    }
    auto input = fixture.Source(SourceWidth, SourceHeight, color);
    constexpr std::array<Pixel, 1> empty{{{0, 0, 0, 0}}};
    auto unused = fixture.Source(1, 1, empty);
    constexpr auto rectangle = Vertices(1, 1, 7, 6, {1, 0}, {0, 1});
    std::array<Vertex, 7> vertices{};
    std::copy(rectangle.begin(), rectangle.end(), vertices.begin() + 3);
    for (const bool shifted : {false, true}) {
        Uniforms uniforms{.matrix = Orthographic(Width, Height), .left_screen = 1};
        uniforms.matrix[11] = 2.0f;
        uniforms.matrix[15] = 3.0f;
        if (shifted) {
            uniforms.matrix[3] += 2.0f / Width;
            uniforms.matrix[7] -= 2.0f / Height;
        }
        const auto result =
            fixture.Draw(Width, Height, uniforms, {unused, input, unused}, vertices, 3);
        for (u32 y = 0; y < Height; ++y) {
            for (u32 x = 0; x < Width; ++x) {
                assert(result[y * Width + x] == Expected(x, y, shifted ? 2 : 1, shifted ? 0 : 1));
            }
        }
    }
    fixture.Close();
    std::puts("PASS: native strided present vertex, orientation and transformed output pixels");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts(
                "PENDING: native presentation vertex/fragment output requires a Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
