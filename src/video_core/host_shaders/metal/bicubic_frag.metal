// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct FilterInput {
    float2 texcoord [[user(locn0)]];
};

static float4 CubicWeights(float coordinate) {
    const float4 distance = float4(1, 2, 3, 4) - coordinate;
    const float4 cubes = distance * distance * distance;
    const float a = cubes.x;
    const float b = cubes.y - 4.0f * a;
    const float c = cubes.z - 4.0f * cubes.y + 6.0f * a;
    return float4(a, b, c, 6.0f - a - b - c) * (1.0f / 6.0f);
}

fragment float4 bicubic_frag(FilterInput input [[stage_in]], texture2d<float> image [[texture(0)]],
                             sampler linear_edge [[sampler(0)]]) {
    const float2 size(image.get_width(), image.get_height());
    const float2 texel = input.texcoord * size - 0.5f;
    const float2 base = floor(texel);
    const float2 fraction = texel - base;
    const float4 cx = CubicWeights(fraction.x);
    const float4 cy = CubicWeights(fraction.y);
    const float4 base_offsets = base.xxyy + float4(-0.5f, 1.5f, -0.5f, 1.5f);
    const float4 pair_weights(cx.xz + cx.yw, cy.xz + cy.yw);
    const float4 coord =
        (base_offsets + float4(cx.yw, cy.yw) / pair_weights) / size.xxyy;

    const float4 low_low = image.sample(linear_edge, coord.xz, level(0.0f));
    const float4 high_low = image.sample(linear_edge, coord.yz, level(0.0f));
    const float4 low_high = image.sample(linear_edge, coord.xw, level(0.0f));
    const float4 high_high = image.sample(linear_edge, coord.yw, level(0.0f));
    const float sx = pair_weights.x / (pair_weights.x + pair_weights.y);
    const float sy = pair_weights.z / (pair_weights.z + pair_weights.w);
    return mix(mix(high_high, low_high, sx), mix(high_low, low_low, sx), sy);
}
