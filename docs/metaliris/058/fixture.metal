#include <metal_stdlib>
using namespace metal;

struct Varying {
    float4 position [[position]];
};

constant float2 positions[3] = {{-1.f, -1.f}, {3.f, -1.f}, {-1.f, 3.f}};

vertex Varying triangle(uint index [[vertex_id]]) {
    return {float4(positions[index], .25f, 1.f)};
}

struct Fragment {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment Fragment paint() {
    return {float4(.75f, .25f, .50f, .50f), .25f};
}

struct Primitive {
    uint index;
};

[[object, max_total_threads_per_threadgroup(1)]]
void object_stage(object_data Primitive& payload [[payload]], mesh_grid_properties grid) {
    payload.index = 0;
    grid.set_threadgroups_per_grid(uint3(1));
}

using TestMesh = mesh<Varying, void, 3, 1, topology::triangle>;

[[mesh, max_total_threads_per_threadgroup(1)]]
void mesh_stage(TestMesh output, const object_data Primitive& payload [[payload]]) {
    for (uint i = 0; i < 3; ++i) {
        output.set_vertex(i, {float4(positions[(i + payload.index) % 3], .25f, 1.f)});
        output.set_index(i, uchar(i));
    }
    output.set_primitive_count(1);
}

kernel void transform_stage(device uint* output [[buffer(0)]]) {
    output[0] = 41;
}

kernel void expand_stage(device uint* output [[buffer(0)]]) {
    output[0] += 1;
}
