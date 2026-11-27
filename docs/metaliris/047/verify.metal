#include <metal_stdlib>

using namespace metal;

struct DepthStencil {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

fragment DepthStencil fixture_seed(float4 position [[position]]) {
    const uint2 coordinate = uint2(position.xy);
    const uint index = coordinate.x + 5u * coordinate.y;
    return {float(index) / 16.0f, index == 14u ? 255u : 17u * index};
}

kernel void fixture_read(depth2d<float, access::read> depth [[texture(0)]],
                         texture2d<uint, access::read> stencil [[texture(1)]],
                         device uint2* output [[buffer(0)]],
                         uint2 index [[thread_position_in_grid]]) {
    if (index.x < depth.get_width() && index.y < depth.get_height()) {
        output[index.x + depth.get_width() * index.y] =
            uint2(as_type<uint>(depth.read(index)), stencil.read(index).r);
    }
}
