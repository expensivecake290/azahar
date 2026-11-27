#include <metal_stdlib>
using namespace metal;

kernel void verify_generation(device const uint* source [[buffer(0)]],
                              device uint* destination [[buffer(1)]],
                              texture2d<float> image [[texture(0)]],
                              texture_buffer<uint, access::read> lookup [[texture(1)]],
                              texture2d<uint, access::write> storage [[texture(2)]],
                              sampler point [[sampler(1)]]) {
    const uint sample = uint(image.sample(point, float2(0.5f), level(0.0f)).r * 16.0f);
    const uint result = source[0] + sample + lookup.read(0u).r;
    destination[0] = result;
    storage.write(uint4(result & 0xFFu), uint2(0));
}
