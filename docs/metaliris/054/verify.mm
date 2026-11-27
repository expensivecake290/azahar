#include <array>
#include <cstdio>
#include "../verify_present.h"

namespace {

using namespace VerifyPresent;

constexpr u32 Width = 9;
constexpr u32 Height = 7;
constexpr std::array<Pixel, 3> Colors{{
    {17, 48, 99, 113},
    {181, 26, 78, 67},
    {27, 197, 219, 251},
}};
constexpr Pixel Invalid{0, 0, 0, 255};

constexpr s32 Row(u32 index, u32 axis_length, bool negative) {
    return negative ? s32(index) + 1 - s32(axis_length) : s32(index);
}

constexpr Pixel Expected(s32 row, s32 reverse, s32 left, s32 right) {
    const s32 eye = row % 2 == reverse ? left : right;
    return eye >= 0 && eye < s32(Colors.size()) ? Colors[eye] : Invalid;
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"present_interlaced_frag", true);
    std::array<id<MTLTexture>, 3> screens{};
    for (u32 i = 0; i < screens.size(); ++i) {
        std::array<Pixel, 15> samples{};
        samples.fill(Colors[i]);
        screens[i] = fixture.Source(5, 3, samples);
    }
    for (const bool rotated : {false, true}) {
        for (const bool negative : {false, true}) {
            auto vertices = Vertices(0, 0, Width, Height);
            if (rotated) {
                for (auto& vertex_data : vertices) {
                    std::swap(vertex_data.texcoord[0], vertex_data.texcoord[1]);
                }
            }
            if (negative) {
                for (auto& vertex_data : vertices) {
                    vertex_data.texcoord[0] -= 1.0f;
                }
            }
            const u32 axis_length = rotated ? Height : Width;
            for (const s32 reverse : {0, 1}) {
                for (const s32 left : {0, 2, -1}) {
                    const Uniforms uniforms{
                        .matrix = Orthographic(Width, Height),
                        .output_resolution = {float(axis_length), 13, 0, 0},
                        .left_screen = left,
                        .right_screen = 1,
                        .reverse_interlaced = reverse,
                    };
                    const auto actual = fixture.Draw(Width, Height, uniforms, screens, vertices);
                    for (u32 y = 0; y < Height; ++y) {
                        for (u32 x = 0; x < Width; ++x) {
                            const auto row = Row(rotated ? y : x, axis_length, negative);
                            assert(actual[y * Width + x] == Expected(row, reverse, left, 1));
                        }
                    }
                }
            }
        }
    }
    fixture.Close();
    std::puts("PASS: native normal/reverse parity with rotated and signed UV and exact RGBA");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        static_assert(Row(0, 9, true) == -8 && Row(8, 9, true) == 0);
        static_assert(Expected(-1, 1, 0, 1) == Colors[1]);
        static_assert(Expected(1, 1, 0, 1) == Colors[0]);
        static_assert(Expected(2, 0, -1, 1) == Invalid);
        static_assert(Expected(2, 1, -1, 1) == Colors[1]);
        std::puts("PASS: independent signed interlace parity, rotation and eye selection oracle");
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts(
                "PENDING: native interlaced output parity requires a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
