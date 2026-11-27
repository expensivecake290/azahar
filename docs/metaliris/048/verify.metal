#include <metal_stdlib>

using namespace metal;

struct Position {
    float4 value [[position]];
};

vertex Position fixture_cursor_vertex(uint index [[vertex_id]]) {
    return {float4(index == 1u ? 3.0f : -1.0f, index == 2u ? 3.0f : -1.0f, 0.0f, 1.0f)};
}
