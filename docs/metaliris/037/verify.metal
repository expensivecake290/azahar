#include <metal_stdlib>

using namespace metal;

struct Position {
    float4 value [[position]];
};

vertex Position fixture_vertex(uint index [[vertex_id]]) {
    return {float4(index == 1u ? 3.0f : -1.0f, index == 2u ? 3.0f : -1.0f, 0.0f, 1.0f)};
}

struct DepthAndStencil {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

fragment DepthAndStencil fixture_samples(uint sample [[sample_id]]) {
    const float depths[4] = {0.0f, float(0x112233u) * 0x1p-24f,
                            float(0x6487AEu) * 0x1p-24f, 1.0f};
    const uint stencils[4] = {0x00u, 0x21u, 0x70u, 0xFFu};
    return {depths[sample], stencils[sample]};
}

kernel void fixture_read_samples(texture2d_ms<float, access::read> color [[texture(0)]],
                                 device uint* output [[buffer(0)]],
                                 uint3 index [[thread_position_in_grid]]) {
    if (index.x < color.get_width() && index.y < color.get_height() &&
        index.z < color.get_num_samples()) {
        const uint4 bytes = uint4(round(color.read(index.xy, index.z) * 255.0f));
        const uint slot = (index.y * color.get_width() + index.x) * color.get_num_samples() +
                          index.z;
        output[slot] = bytes.r + (bytes.g << 8u) + (bytes.b << 16u) + (bytes.a << 24u);
    }
}
