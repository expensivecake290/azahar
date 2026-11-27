#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position full_triangle(uint vid [[vertex_id]]) {
    return {float4(vid == 1 ? 3.0f : -1.0f, vid == 2 ? 3.0f : -1.0f, 0.0f, 1.0f)};
}

struct OrderedColor {
    float4 value [[color(0), raster_order_group(0)]];
};

fragment OrderedColor ordered_increment(OrderedColor previous) {
    return {previous.value + float4(1.0f / 255.0f, 0.0f, 0.0f, 0.0f)};
}

struct SampleValues {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

fragment SampleValues distinct_samples(uint sample [[sample_id]]) {
    return {0.25f + float(sample) / 32.0f, 0x55u + sample};
}

fragment float4 distinct_color(uint sample [[sample_id]]) {
    return float4(float(sample + 1u) / 255.0f, 0.0f, 0.0f, 1.0f);
}

kernel void read_samples(texture2d_ms<float, access::read> texture [[texture(0)]],
                         device uint* values [[buffer(0)]], uint sample [[thread_position_in_grid]]) {
    values[sample] = uint(round(texture.read(uint2(4), sample).r * 255.0f));
}
