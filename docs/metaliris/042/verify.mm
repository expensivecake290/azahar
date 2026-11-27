#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "../verify_filter.h"

namespace {

using namespace VerifyFilter;

constexpr int Width = 5;
constexpr int Height = 3;
constexpr int TargetWidth = 9;
constexpr int TargetHeight = 7;
using Image = std::array<Pixel, Width * Height>;
using Color = std::array<double, 4>;

Image Edge(bool alpha_only) {
    Image image{};
    constexpr std::array shape{0u,  16u, 236u, 24u, 100u, 20u,  220u, 72u,
                               42u, 32u, 160u, 28u, 240u, 112u, 176u};
    for (u32 index = 0; index < image.size(); ++index) {
        image[index] = {u8(alpha_only ? 115 : shape[index]), 40, 120,
                        u8(alpha_only ? shape[index] : 255)};
    }
    return image;
}

Color Linear(const Image& image, double x, double y) {
    const double base_x = std::floor(x);
    const double base_y = std::floor(y);
    const double relative_x = x - base_x;
    const double relative_y = y - base_y;
    Color result{};
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            const int row = std::clamp(static_cast<int>(base_y) + j, 0, Height - 1);
            const int column = std::clamp(static_cast<int>(base_x) + i, 0, Width - 1);
            const double area =
                (j ? relative_y : 1 - relative_y) * (i ? relative_x : 1 - relative_x);
            for (u32 lane = 0; lane < result.size(); ++lane) {
                result[lane] += area * image[row * Width + column][lane] / 255.0;
            }
        }
    }
    return result;
}

struct Tap {
    double point;
    double weight;
};

std::array<Tap, 2> Taps(double x) {
    const auto base = std::trunc(x);
    const auto frac = x - base;
    const double complement = 1.0 - frac;
    const double w0 = complement * complement * complement / 6;
    const double w3 = frac * frac * frac / 6;
    const double w1 = (4 + frac * frac * (3 * frac - 6)) / 6;
    const double w2 = (4 + complement * complement * (3 * complement - 6)) / 6;
    const double left = w0 + w1;
    const double right = w2 + w3;
    return {{{base - 1 + w1 / left, left}, {base + 1 + w3 / right, right}}};
}

Color Cubic(const Image& image, double x, double y) {
    Color result{};
    for (const auto& row : Taps(y)) {
        for (const auto& column : Taps(x)) {
            const auto sample = Linear(image, column.point, row.point);
            for (u32 lane = 0; lane < result.size(); ++lane) {
                result[lane] += row.weight * column.weight * sample[lane];
            }
        }
    }
    return result;
}

Color Reference(const Image& image, double u, double v) {
    const double x = u * Width - 0.5;
    const double y = v * Height - 0.5;
    const auto center = Linear(image, x, y);
    // In the fixture, green and blue are constant. The first matrix column sums to 0.6.
    const auto distance = [&](int dx, int dy) {
        const auto color = Linear(image, x + dx, y + dy);
        const double diff = 0.6 * (color[0] - center[0]);
        return std::sqrt((diff * diff + std::abs(color[3] - center[3])) * color[3] * center[3]);
    };
    const std::array differences{
        distance(-1, 1) - distance(1, -1),
        distance(0, 1) - distance(0, -1),
        distance(1, 1) - distance(-1, -1),
        distance(1, 0) - distance(-1, 0),
    };
    const double total = distance(-1, 1) + distance(0, 1) + distance(1, 1) + distance(1, 0) +
                         distance(1, -1) + distance(0, -1) + distance(-1, -1) + distance(-1, 0);
    if (total == 0) {
        return center;
    }
    const double horizontal = differences[3] + differences[2] - differences[0];
    const double vertical = differences[1] + differences[2] + differences[0];
    const double limit = std::hypot(horizontal, vertical) / total;
    return Cubic(image, x - std::clamp(horizontal, -limit, limit),
                 y - std::clamp(vertical, -limit, limit));
}

void CPU() {
    const auto edge = Edge(false);
    const auto alpha = Edge(true);
    const auto signed_fraction = Taps(-0.25);
    assert(signed_fraction[0].point > -1 && signed_fraction[0].point < 0);
    assert(signed_fraction[1].point < 1);
    assert(std::abs(signed_fraction[0].weight + signed_fraction[1].weight - 1.0) < 1e-14);
    const auto result = Reference(edge, 0.52, 0.47);
    assert(std::abs(result[0] - Linear(edge, 0.52 * Width - 0.5, 0.47 * Height - 0.5)[0]) > 0.02);
    assert(std::abs(result[1] - 40.0 / 255.0) < 1e-14);
    assert(std::abs(Reference(alpha, 0.6, 0.5)[0] - 115.0 / 255.0) < 1e-14);
    Image transparent = edge;
    for (auto& pixel : transparent) {
        pixel[3] = 0;
    }
    assert(Reference(transparent, 0.52, 0.47) ==
           Linear(transparent, 0.52 * Width - 0.5, 0.47 * Height - 0.5));
    std::puts("PASS: independent ScaleForce edge, alpha and signed cubic coordinate oracles");
}

void GPU(const char* shader, const char* vertex) {
    Fixture fixture(vertex, shader, @"scale_force_frag");
    auto transparent = Edge(false);
    for (auto& pixel : transparent) {
        pixel[3] = 0;
    }
    Image flat{};
    flat.fill({41, 40, 120, 122});
    const std::array transforms{Uniforms{}, Uniforms{{1.2f, -1.4f}, {-0.1f, 1.2f}}};
    for (const auto& image : {Edge(false), Edge(true), transparent, flat}) {
        auto input = fixture.Source<Pixel>(Width, Height, MTLPixelFormatRGBA8Unorm, image);
        for (const auto& transform : transforms) {
            const auto actual = fixture.Draw<Pixel>(TargetWidth, TargetHeight, transform, {input});
            for (int y = 0; y < TargetHeight; ++y) {
                for (int x = 0; x < TargetWidth; ++x) {
                    const double u =
                        (x + 0.5) * transform.scale[0] / TargetWidth + transform.offset[0];
                    const double v =
                        (y + 0.5) * transform.scale[1] / TargetHeight + transform.offset[1];
                    const auto expected = Reference(image, u, v);
                    for (u32 lane = 0; lane < 4; ++lane) {
                        const auto byte = std::lround(255 * std::clamp(expected[lane], 0.0, 1.0));
                        assert(std::abs(actual[y * TargetWidth + x][lane] - byte) <= 2);
                    }
                }
            }
        }
    }
    fixture.Close();
    std::puts("PASS: native ScaleForce contrast and skip branches match RGBA output oracles");
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        CPU();
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native ScaleForce pixels require a process-visible Metal 4 GPU.");
        } else {
            GPU(argv[1], argv[2]);
        }
    }
}
