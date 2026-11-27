// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <metal_stdlib>

using namespace metal;

struct CopyRegion {
    uint2 src_offset;
    uint2 dst_offset;
    uint2 extent;
};

static_assert(sizeof(CopyRegion) == 24, "Multisample D24S8 copy parameters");

struct CopyVertex {
    float4 position [[position]];
};

vertex CopyVertex d24s8_to_rgba8_ms_vertex(uint vertex_id [[vertex_id]]) {
    const float2 position(vertex_id == 1u ? 3.0f : -1.0f, vertex_id == 2u ? 3.0f : -1.0f);
    return {float4(position, 0.0f, 1.0f)};
}

// Multisample textures are read-only in MSL. Writing a native color attachment at sample
// frequency preserves each sample and permits a tile-local resolve when one is requested.
fragment float4 d24s8_to_rgba8_ms_fragment(CopyVertex input [[stage_in]], uint sample [[sample_id]],
                                           depth2d_ms<float, access::read> depth [[texture(0)]],
                                           texture2d_ms<uint, access::read> stencil [[texture(1)]],
                                           constant CopyRegion& region [[buffer(0)]]) {
    const uint2 dst = uint2(input.position.xy);
    if (any(dst < region.dst_offset)) {
        discard_fragment();
        return float4(0.0f);
    }

    const uint2 index = dst - region.dst_offset;
    const uint2 src_size(min(depth.get_width(), stencil.get_width()),
                         min(depth.get_height(), stencil.get_height()));
    if (any(index >= region.extent) || any(region.src_offset >= src_size) ||
        any(index >= src_size - region.src_offset) || sample >= depth.get_num_samples() ||
        sample >= stencil.get_num_samples()) {
        discard_fragment();
        return float4(0.0f);
    }

    const uint2 src = region.src_offset + index;
    const float depth_value = depth.read(src, sample);
    const uint bits = min(uint(clamp(depth_value, 0.0f, 1.0f) * 16777216.0f), 0xFFFFFFu);
    const uint4 components(stencil.read(src, sample).r & 0xFFu, (bits >> 16u) & 0xFFu,
                           (bits >> 8u) & 0xFFu, bits & 0xFFu);
    return float4(components) * (1.0f / 255.0f);
}
