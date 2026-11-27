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

static_assert(sizeof(CopyRegion) == 24, "D24S8 copy parameters");

kernel void d24s8_to_rgba8_comp(depth2d<float, access::read> depth [[texture(0)]],
                                texture2d<uint, access::read> stencil [[texture(1)]],
                                texture2d<float, access::write> color [[texture(2)]],
                                constant CopyRegion& region [[buffer(0)]],
                                uint2 index [[thread_position_in_grid]]) {
    const uint2 src_size(min(depth.get_width(), stencil.get_width()),
                         min(depth.get_height(), stencil.get_height()));
    const uint2 dst_size(color.get_width(), color.get_height());
    if (any(index >= region.extent) || any(region.src_offset >= src_size) ||
        any(region.dst_offset >= dst_size) || any(index >= src_size - region.src_offset) ||
        any(index >= dst_size - region.dst_offset)) {
        return;
    }

    const uint2 src = region.src_offset + index;
    const uint2 dst = region.dst_offset + index;
    const float depth_value = depth.read(src);
    // The three most significant bytes of the normalized 32-bit fixed-point value.
    const uint bits = min(uint(clamp(depth_value, 0.0f, 1.0f) * 16777216.0f), 0xFFFFFFu);
    const uint4 components(stencil.read(src).r & 0xFFu, (bits >> 16u) & 0xFFu, (bits >> 8u) & 0xFFu,
                           bits & 0xFFu);
    color.write(float4(components) * (1.0f / 255.0f), dst);
}
