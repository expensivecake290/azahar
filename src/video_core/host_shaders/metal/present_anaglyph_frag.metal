// Copyright 2022-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct PresentInput {
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
static_assert(__builtin_offsetof(PresentUniforms, right_screen) == 100, "Right screen offset");

fragment float4 present_anaglyph_frag(PresentInput input [[stage_in]],
                                       array<texture2d<float>, 3> screens [[texture(0)]],
                                       sampler screen_sampler [[sampler(0)]],
                                       constant PresentUniforms& uniforms [[buffer(0)]]) {
    const auto sample = [&](int id) {
        return uint(id) < screens.size()
                   ? screens[uint(id)].sample(screen_sampler, input.texcoord, level(0.0f))
                   : float4(0.0f, 0.0f, 0.0f, 1.0f);
    };

    // Red-cyan projection coefficients from Eric Dubois's least-squares conversion.
    const float3x3 left(float3(0.437f, 0.449f, 0.164f), float3(-0.062f, -0.062f, -0.024f),
                        float3(-0.048f, -0.050f, -0.017f));
    const float3x3 right(float3(-0.011f, -0.032f, -0.007f), float3(0.377f, 0.761f, 0.009f),
                         float3(-0.026f, -0.093f, 1.234f));
    const float4 a = sample(uniforms.left_screen);
    const float4 b = sample(uniforms.right_screen);
    return float4(a.rgb * left + b.rgb * right, a.a);
}
