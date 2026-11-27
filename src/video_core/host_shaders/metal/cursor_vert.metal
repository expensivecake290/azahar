// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct CursorInput {
    float2 position [[attribute(0)]];
};

struct CursorOutput {
    float4 position [[position]];
};

vertex CursorOutput cursor_vert(CursorInput input [[stage_in]]) {
    return {float4(input.position.x, -input.position.y, 0.0f, 1.0f)};
}
