// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Azahar Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <metal_stdlib>

using namespace metal;

struct TextureTransform {
    float2 scale;
    float2 offset;
};

static_assert(sizeof(TextureTransform) == 16, "Blit vertex texture transform");

struct FullScreenVertex {
    float4 position [[position]];
    float2 texcoord [[user(locn0)]];
};

vertex FullScreenVertex full_screen_triangle_vert(
    uint index [[vertex_id]], constant TextureTransform& transform [[buffer(0)]]) {
    const float2 uv(float((index & 1u) << 1u), float(index & 2u));
    // Metal maps positive clip-space Y to the top of the viewport.
    const float2 position = fma(uv, float2(2.0f, -2.0f), float2(-1.0f, 1.0f));
    return {float4(position, 0.0f, 1.0f), fma(uv, transform.scale, transform.offset)};
}
