#include <metal_stdlib>
using namespace metal;

kernel void gate0_write(device uint* result [[buffer(0)]],
                        uint index [[thread_position_in_grid]]) {
    result[index] = 0x4d544c34u;
}

struct Vertex {
    float4 position [[position]];
};

struct TrianglePayload {
    float scale;
};

[[object]] void gate0_object(object_data TrianglePayload& payload [[payload]],
                             mesh_grid_properties grid) {
    payload.scale = 1.0f;
    grid.set_threadgroups_per_grid(uint3(1));
}

[[mesh]] void gate0_mesh(const object_data TrianglePayload& payload [[payload]],
                         mesh<Vertex, void, 3, 1, topology::triangle> triangle) {
    triangle.set_vertex(0, Vertex{float4(-payload.scale, -1, 0, 1)});
    triangle.set_vertex(1, Vertex{float4(3 * payload.scale, -1, 0, 1)});
    triangle.set_vertex(2, Vertex{float4(-payload.scale, 3, 0, 1)});
    triangle.set_index(0, 0);
    triangle.set_index(1, 1);
    triangle.set_index(2, 2);
    triangle.set_primitive_count(1);
}

fragment float4 gate0_fragment() {
    return float4(1, 0, 0, 1);
}
