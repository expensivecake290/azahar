// Copyright 2022-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct BlitInput {
    float2 texcoord [[user(locn0)]];
};

struct DepthStencil {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

static uint2 NearestCoordinate(float2 uv, uint2 size) {
    return uint2(clamp(floor(uv * float2(size)), float2(0.0f), float2(size - 1u)));
}

fragment DepthStencil blit_depth_stencil_frag(
    BlitInput input [[stage_in]], depth2d<float, access::read> depth [[texture(0)]],
    texture2d<uint, access::read> stencil [[texture(1)]]) {
    const uint2 depth_size(depth.get_width(), depth.get_height());
    const uint2 stencil_size(stencil.get_width(), stencil.get_height());
    return {depth.read(NearestCoordinate(input.texcoord, depth_size)),
            stencil.read(NearestCoordinate(input.texcoord, stencil_size)).r & 0xFFu};
}
