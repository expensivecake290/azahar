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
static_assert(__builtin_offsetof(PresentUniforms, left_screen) == 96, "Left screen offset");

fragment float4 present_frag(PresentInput input [[stage_in]],
                              array<texture2d<float>, 3> screens [[texture(0)]],
                              sampler screen_sampler [[sampler(0)]],
                              constant PresentUniforms& uniforms [[buffer(0)]]) {
    const uint screen = uint(uniforms.left_screen);
    if (screen >= screens.size()) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return screens[screen].sample(screen_sampler, input.texcoord, level(0.0f));
}
