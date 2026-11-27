#include <metal_stdlib>
using namespace metal;

struct DepthStencil {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

vertex float4 fill_depth_vertex(uint index [[vertex_id]]) {
    return float4(index == 1 ? 3.f : -1.f, index == 2 ? 3.f : -1.f, 0.f, 1.f);
}

fragment DepthStencil fill_depth_samples(uint index [[sample_id]]) {
    return {float(index + 1) / 8.f, index + 0x2du};
}

kernel void inspect_samples(texture2d_ms<float, access::read> packed [[texture(0)]],
                            device uint* values [[buffer(0)]],
                            uint index [[thread_position_in_grid]]) {
    const uint4 bytes = uint4(round(packed.read(uint2(2, 2), index) * 255.f));
    values[index] = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
}
