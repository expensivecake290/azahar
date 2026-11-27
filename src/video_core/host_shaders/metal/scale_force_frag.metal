// from https://github.com/BreadFish64/ScaleFish/tree/master/scale_force

// MIT License
//
// Copyright (c) 2020 BreadFish64
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

struct FilterInput {
    float2 texcoord [[user(locn0)]];
};

static float4 Cubic(float fraction) {
    const float3 n = float3(1.0f, 2.0f, 3.0f) - fraction;
    const float3 s = n * n * n;
    const float a = s.x;
    const float b = s.y - 4.0f * s.x;
    const float c = s.z - 4.0f * s.y + 6.0f * s.x;
    return float4(a, b, c, 6.0f - a - b - c) / 6.0f;
}

static float4 SampleBicubic(texture2d<float> image, sampler linear_edge, float2 uv, float2 size) {
    float2 integral;
    const float2 fraction = modf(uv * size - 0.5f, integral);
    const float4 x = Cubic(fraction.x);
    const float4 y = Cubic(fraction.y);
    const float4 pairs(x.xz + x.yw, y.xz + y.yw);
    const float4 bases = integral.xxyy + float4(-0.5f, 1.5f, -0.5f, 1.5f);
    const float4 offsets = (bases + float4(x.yw, y.yw) / pairs) / size.xxyy;

    const float4 low_low = image.sample(linear_edge, offsets.xz, level(0.0f));
    const float4 high_low = image.sample(linear_edge, offsets.yz, level(0.0f));
    const float4 low_high = image.sample(linear_edge, offsets.xw, level(0.0f));
    const float4 high_high = image.sample(linear_edge, offsets.yw, level(0.0f));
    const float sx = pairs.x / (pairs.x + pairs.y);
    const float sy = pairs.z / (pairs.z + pairs.w);
    return mix(mix(high_high, low_high, sx), mix(high_low, low_low, sx), sy);
}

static float Distance(float4 color, float4 center) {
    const float3 k(0.2627f, 0.6780f, 0.0593f);
    const float3x3 ycbcr(k * 0.6f,
                         float3(-0.5f * k.r / (1.0f - k.b), -0.5f * k.g / (1.0f - k.b), 0.5f),
                         float3(0.5f, -0.5f * k.g / (1.0f - k.r), -0.5f * k.b / (1.0f - k.r)));
    const float difference = dot(float3(1.0f), ycbcr * (color.rgb - center.rgb));
    return sqrt((difference * difference + abs(center.a - color.a)) * color.a * center.a);
}

fragment float4 scale_force_frag(FilterInput input [[stage_in]],
                                 texture2d<float> image [[texture(0)]],
                                 sampler linear_edge [[sampler(0)]]) {
    const auto fetch = [&](int2 delta) {
        return image.sample(linear_edge, input.texcoord, level(0.0f), delta);
    };
    const float4 center = fetch(int2(0));
    const float4 positive(Distance(fetch(int2(-1, 1)), center), Distance(fetch(int2(0, 1)), center),
                          Distance(fetch(int2(1, 1)), center), Distance(fetch(int2(1, 0)), center));
    const float4 negative(
        Distance(fetch(int2(1, -1)), center), Distance(fetch(int2(0, -1)), center),
        Distance(fetch(int2(-1, -1)), center), Distance(fetch(int2(-1, 0)), center));
    const float total_distance = dot(positive + negative, float4(1.0f));
    if (total_distance == 0.0f) {
        return center;
    }

    const float4 difference = positive - negative;
    const float2 displacement = difference.wy + difference.zz + float2(-difference.x, difference.x);
    const float limit = length(displacement) / total_distance;
    const float2 size(image.get_width(), image.get_height());
    const float2 offset = clamp(displacement, -limit, limit) / size;
    return SampleBicubic(image, linear_edge, input.texcoord - offset, size);
}
