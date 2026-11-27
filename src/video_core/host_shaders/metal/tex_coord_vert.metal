// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct TextureVertex {
    float4 position [[position]];
    float2 texcoord [[user(locn0)]];
};

vertex TextureVertex tex_coord_vert(uint index [[vertex_id]]) {
    const float2 coordinate(float(index & 1u), float((index >> 1u) & 1u));
    const float2 position = fma(coordinate, float2(2.0f, -2.0f), float2(-1.0f, 1.0f));
    return {float4(position, 0.0f, 1.0f), coordinate};
}
