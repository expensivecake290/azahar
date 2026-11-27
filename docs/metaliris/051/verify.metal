#include <metal_stdlib>

using namespace metal;

struct PresentVaryings {
    float4 position [[position]];
    float2 texcoord [[user(locn0)]];
};

vertex PresentVaryings fixture_present_vertex(uint vertex_id [[vertex_id]]) {
    const float2 uv(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u));
    return {float4(2.0f * uv.x - 1.0f, 1.0f - 2.0f * uv.y, 0.0f, 1.0f), uv};
}
