// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct ReadbackRegion {
    uint2 src_offset;
    uint2 extent;
    uint row_stride;
    uint capacity;
};

static_assert(sizeof(ReadbackRegion) == 24, "Depth readback region layout");

kernel void depth_to_buffer_comp(depth2d<float, access::read> depth [[texture(0)]],
                                 texture2d<uint, access::read> stencil [[texture(1)]],
                                 constant ReadbackRegion& region [[buffer(0)]],
                                 device uint* output [[buffer(1)]],
                                 uint2 index [[thread_position_in_grid]]) {
    const uint2 size(min(depth.get_width(), stencil.get_width()),
                     min(depth.get_height(), stencil.get_height()));
    if (any(index >= region.extent) || region.row_stride < region.extent.x ||
        any(region.src_offset >= size) || any(index >= size - region.src_offset)) {
        return;
    }

    const ulong word = ulong(index.y) * region.row_stride + index.x;
    if (word >= region.capacity) {
        return;
    }

    const uint2 source = region.src_offset + index;
    const uint depth24 = uint(clamp(depth.read(source), 0.0f, 1.0f) * 16777215.0f);
    const uint stencil8 = stencil.read(source).r & 0xFFu;
    output[word] = (depth24 << 8u) | stencil8;
}
