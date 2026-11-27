// Copyright 2022-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct PresentVertex {
    float2 position [[attribute(0)]];
    float2 texcoord [[attribute(1)]];
};

struct PresentOutput {
    float4 position [[position]];
    float2 texcoord [[user(locn0)]];
};

struct PresentUniforms {
    float4x4 modelview;
    float4 input_resolution;
    float4 output_resolution;
    int left_screen;
    int right_screen;
    int layer;
    int reverse_interlaced;
};

static_assert(sizeof(PresentUniforms) == 112, "Presentation uniform size");
static_assert(__builtin_offsetof(PresentUniforms, output_resolution) == 80,
              "Presentation output resolution offset");

vertex PresentOutput present_vert(PresentVertex input [[stage_in]],
                                   constant PresentUniforms& uniforms [[buffer(0)]]) {
    const float4 position = float4(input.position, 0.0f, 1.0f) * uniforms.modelview;
    return {float4(position.x, -position.y, 0.0f, 1.0f), input.texcoord};
}
