// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct FilterInput {
    float2 texcoord [[user(locn0)]];
};

static float Luma(float4 color) {
    return dot(color.rgb, float3(0.2126f, 0.7152f, 0.0722f)) * (1.0f - color.a);
}

static bool Same(float4 a, float4 b) {
    return all(a == b);
}

static bool Same(float4 base, float4 a, float4 b) {
    return Same(base, a) && Same(base, b);
}

static bool Same(float4 base, float4 a, float4 b, float4 c) {
    return Same(base, a, b) && Same(base, c);
}

static bool Same(float4 base, float4 a, float4 b, float4 c, float4 d) {
    return Same(base, a, b, c) && Same(base, d);
}

static bool AnySame(float4 base, float4 a, float4 b, float4 c) {
    return Same(base, a) || Same(base, b) || Same(base, c);
}

static bool NoneSame(float4 base, float4 a, float4 b) {
    return !Same(base, a) && !Same(base, b);
}

static bool NoneSame(float4 base, float4 a, float4 b, float4 c, float4 d) {
    return NoneSame(base, a, b) && NoneSame(base, c, d);
}

fragment float4 mmpx_frag(FilterInput input [[stage_in]],
                          texture2d<float, access::read> image [[texture(0)]]) {
    const int2 size(image.get_width(), image.get_height());
    const float2 texel = input.texcoord * float2(size);
    const float2 integral = floor(texel);
    const int2 center = int2(clamp(integral, float2(-3.0f), float2(size + 2)));
    const auto pixel = [&](int x, int y) {
        return image.read(uint2(clamp(center + int2(x, y), int2(0), size - 1)));
    };

    // The nine nearest texels are named A through I, in row order.
    const float4 a = pixel(-1, -1);
    const float4 b = pixel(0, -1);
    const float4 c = pixel(1, -1);
    const float4 d = pixel(-1, 0);
    const float4 e = pixel(0, 0);
    const float4 f = pixel(1, 0);
    const float4 g = pixel(-1, 1);
    const float4 h = pixel(0, 1);
    const float4 i = pixel(1, 1);
    if (Same(e, a, b, c, d) && Same(e, f, g, h, i)) {
        return e;
    }

    // Both vertical continuation rules in the source use the Y+2 sample.
    const float4 south2 = pixel(0, 2);
    const float4 west2 = pixel(-2, 0);
    const float4 east2 = pixel(2, 0);
    const float bl = Luma(b);
    const float dl = Luma(d);
    const float el = Luma(e);
    const float fl = Luma(f);
    const float hl = Luma(h);

    float4 top_left = e;
    float4 top_right = e;
    float4 bottom_left = e;
    float4 bottom_right = e;
    if (Same(d, b) && NoneSame(d, h, f) && (el >= dl || Same(e, a)) && AnySame(e, a, c, g) &&
        (el < dl || !Same(a, d) || !Same(e, south2) || !Same(e, west2))) {
        top_left = mix(d, top_left, 0.5f);
    }
    if (Same(b, f) && NoneSame(b, d, h) && (el >= bl || Same(e, c)) && AnySame(e, a, c, i) &&
        (el < bl || !Same(c, b) || !Same(e, south2) || !Same(e, east2))) {
        top_right = mix(b, top_right, 0.5f);
    }
    if (Same(h, d) && NoneSame(h, f, b) && (el >= hl || Same(e, g)) && AnySame(e, a, g, i) &&
        (el < hl || !Same(g, h) || !Same(e, south2) || !Same(e, west2))) {
        bottom_left = mix(h, bottom_left, 0.5f);
    }
    if (Same(f, h) && NoneSame(f, b, d) && (el >= fl || Same(e, i)) && AnySame(e, c, g, i) &&
        (el < fl || !Same(i, h) || !Same(e, east2) || !Same(e, south2))) {
        bottom_right = mix(f, bottom_right, 0.5f);
    }

    if (!Same(e, f) && Same(e, c, i, d, west2) && Same(f, b, h) && !Same(f, pixel(3, 0))) {
        bottom_right = mix(bottom_right, f, 0.5f);
        top_right = mix(top_right, bottom_right, 0.5f);
    }
    if (!Same(e, d) && Same(e, a, g, f, east2) && Same(d, b, h) && !Same(d, pixel(-3, 0))) {
        bottom_left = mix(bottom_left, d, 0.5f);
        top_left = mix(top_left, bottom_left, 0.5f);
    }
    if (!Same(e, h) && Same(e, g, i, b, south2) && Same(h, d, f) && !Same(h, pixel(0, 3))) {
        bottom_right = mix(bottom_right, h, 0.5f);
        bottom_left = mix(bottom_left, bottom_right, 0.5f);
    }
    if (!Same(e, b) && Same(e, a, c, h, south2) && Same(b, d, f) && !Same(b, pixel(0, -3))) {
        top_right = mix(top_right, b, 0.5f);
        top_left = mix(top_left, top_right, 0.5f);
    }

    if (bl < el && Same(e, g, h, i, south2) && NoneSame(e, a, d, c, f)) {
        top_right = mix(top_right, b, 0.5f);
        top_left = mix(top_left, top_right, 0.5f);
    }
    if (hl < el && Same(e, a, b, c, south2) && NoneSame(e, d, g, i, f)) {
        bottom_right = mix(bottom_right, h, 0.5f);
        bottom_left = mix(bottom_left, bottom_right, 0.5f);
    }
    if (fl < el && Same(e, a, d, g, west2) && NoneSame(e, b, c, i, h)) {
        bottom_right = mix(bottom_right, f, 0.5f);
        top_right = mix(top_right, bottom_right, 0.5f);
    }
    if (dl < el && Same(e, c, f, i, east2) && NoneSame(e, b, a, g, h)) {
        bottom_left = mix(bottom_left, d, 0.5f);
        top_left = mix(top_left, bottom_left, 0.5f);
    }

    if (!Same(h, b)) {
        if (NoneSame(h, a, e) && !Same(h, c)) {
            if (Same(h, g, f, east2) && NoneSame(h, d, pixel(2, -1))) {
                bottom_left = mix(bottom_right, bottom_left, 0.5f);
            }
            if (Same(h, i, d, west2) && NoneSame(h, f, pixel(-2, -1))) {
                bottom_right = mix(bottom_left, bottom_right, 0.5f);
            }
        }
        if (NoneSame(b, i, g) && !Same(b, e)) {
            if (Same(b, a, f, east2) && NoneSame(b, d, pixel(2, 1))) {
                top_left = mix(top_right, bottom_left, 0.5f);
            }
            if (Same(b, c, d, west2) && NoneSame(b, f, pixel(-2, 1))) {
                top_right = mix(top_left, top_right, 0.5f);
            }
        }
    }

    if (!Same(f, d)) {
        if (NoneSame(d, i, e) && !Same(d, c)) {
            if (Same(d, a, h, south2) && NoneSame(d, b, pixel(1, 2))) {
                top_left = mix(bottom_left, top_left, 0.5f);
            }
            if (Same(d, g, b, south2) && NoneSame(d, h, pixel(1, 2))) {
                bottom_left = mix(top_left, bottom_left, 0.5f);
            }
        }
        if (NoneSame(f, e, a) && !Same(f, g)) {
            if (Same(f, c, h, south2) && NoneSame(f, b, pixel(-1, 2))) {
                top_right = mix(bottom_right, top_right, 0.5f);
            }
            if (Same(f, i, b, south2) && NoneSame(f, h, pixel(-1, -2))) {
                bottom_right = mix(top_right, bottom_right, 0.5f);
            }
        }
    }

    const bool left = texel.x - integral.x < 0.5f;
    return texel.y - integral.y < 0.5f ? (left ? top_left : top_right)
                                       : (left ? bottom_left : bottom_right);
}
