#include <metal_stdlib>

using namespace metal;

struct TextureCoordinate {
    float2 uv [[user(locn0)]];
};

fragment float4 fixture_uv(TextureCoordinate input [[stage_in]]) {
    return float4(input.uv, 0.0f, 1.0f);
}
