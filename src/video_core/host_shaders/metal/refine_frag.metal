// MIT License
//
// Copyright (c) 2019 bloc97
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>

using namespace metal;

struct RefineInput {
    float2 texcoord [[user(locn0)]];
};

struct ColorLuma {
    float4 color;
    float luminance;
};

static float Minimum(ColorLuma a, ColorLuma b, ColorLuma c) {
    return min(min(a.luminance, b.luminance), c.luminance);
}

static float Maximum(ColorLuma a, ColorLuma b, ColorLuma c) {
    return max(max(a.luminance, b.luminance), c.luminance);
}

static float4 Refine(ColorLuma center, ColorLuma a, ColorLuma b, ColorLuma c) {
    constexpr float Strength = 0.6f;
    return center.color * (1.0f - Strength) + ((a.color + b.color + c.color) / 3.0f) * Strength;
}

fragment float4 refine_frag(RefineInput input [[stage_in]], texture2d<float> image [[texture(0)]],
                            texture2d<float> luminance [[texture(1)]],
                            sampler linear_edge [[sampler(0)]]) {
    const auto fetch = [&](int2 offset) -> ColorLuma {
        return {image.sample(linear_edge, input.texcoord, level(0.0f), offset),
                luminance.sample(linear_edge, input.texcoord, level(0.0f), offset).r};
    };

    const ColorLuma center = fetch(int2(0));
    if (center.luminance > 0.4f) {
        return center.color;
    }

    const ColorLuma tl = fetch(int2(-1, -1));
    const ColorLuma t = fetch(int2(0, -1));
    const ColorLuma tr = fetch(int2(1, -1));
    const ColorLuma l = fetch(int2(-1, 0));
    const ColorLuma r = fetch(int2(1, 0));
    const ColorLuma bl = fetch(int2(-1, 1));
    const ColorLuma b = fetch(int2(0, 1));
    const ColorLuma br = fetch(int2(1, 1));

    float bright = Minimum(tl, t, tr);
    if (bright > center.luminance && bright > Maximum(br, b, bl)) {
        return Refine(center, tl, t, tr);
    }
    bright = Minimum(br, b, bl);
    if (bright > center.luminance && bright > Maximum(tl, t, tr)) {
        return Refine(center, br, b, bl);
    }

    if (Minimum(r, t, tr) > Maximum(center, l, b)) {
        return Refine(center, r, t, tr);
    }
    if (Minimum(bl, l, b) > Maximum(center, r, t)) {
        return Refine(center, bl, l, b);
    }

    bright = Minimum(r, br, tr);
    if (bright > center.luminance && bright > Maximum(l, tl, bl)) {
        return Refine(center, r, br, tr);
    }
    bright = Minimum(l, tl, bl);
    if (bright > center.luminance && bright > Maximum(r, br, tr)) {
        return Refine(center, l, tl, bl);
    }

    if (Minimum(r, br, b) > Maximum(center, l, t)) {
        return Refine(center, r, br, b);
    }
    if (Minimum(t, l, tl) > Maximum(center, r, b)) {
        return Refine(center, t, l, tl);
    }
    return center.color;
}
