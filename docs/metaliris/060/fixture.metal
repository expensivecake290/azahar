#include <metal_stdlib>
using namespace metal;

kernel void cache_fixture(device uint* output [[buffer(0)]]) {
    output[0] = 0x4d495249u;
}
