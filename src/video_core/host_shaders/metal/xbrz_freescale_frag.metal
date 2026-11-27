// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <metal_stdlib>

using namespace metal;

struct FilterInput {
    float2 texcoord [[user(locn0)]];
};

struct FilterUniforms {
    float2 tex_scale;
    float2 tex_offset;
    float resolution_scale;
};

static_assert(__builtin_offsetof(FilterUniforms, resolution_scale) == 16,
              "xBRZ filter scale offset");

enum class Blend {
    None,
    Normal,
    Dominant,
};

constant constexpr float SteepDirectionThreshold = 2.2f;
constant constexpr float DominantDirectionThreshold = 3.6f;
constant constexpr float HalfDiagonal = 0.7071067811865475244f;

static bool Equal(float4 a, float4 b) {
    return all(a == b);
}

static float Distance(float4 a, float4 b) {
    const float3 k(0.2627f, 0.6780f, 0.0593f);
    const float3x3 matrix(k, float3(-0.5f * k.r / (1.0f - k.b), -0.5f * k.g / (1.0f - k.b), 0.5f),
                          float3(0.5f, -0.5f * k.g / (1.0f - k.r), -0.5f * k.b / (1.0f - k.r)));
    const float4 diff = a - b;
    const float d = length(diff.rgb * matrix);
    return sqrt(a.a * b.a * d * d + diff.a * diff.a);
}

static bool Similar(float4 a, float4 b) {
    return Distance(a, b) < (30.0f / 255.0f);
}

static Blend Classify(float along, float opposed, float4 center, float4 first, float4 second) {
    if (!(along < opposed) || Equal(center, first) || Equal(center, second)) {
        return Blend::None;
    }
    return DominantDirectionThreshold * along < opposed ? Blend::Dominant : Blend::Normal;
}

static float Ratio(float2 coordinate, float2 origin, float2 direction, float scale) {
    const float2 relative = coordinate - origin;
    const float2 projected = direction * (dot(relative, direction) / dot(direction, direction));
    const float side = sign(dot(relative, float2(-direction.y, direction.x)));
    return smoothstep(-HalfDiagonal, HalfDiagonal, side * length((relative - projected) * scale));
}

static float4 BlendPixel(float4 center, float4 first, float4 second) {
    return Distance(center, first) < Distance(center, second) ? first : second;
}

fragment float4 xbrz_freescale_frag(FilterInput input [[stage_in]],
                                    texture2d<float, access::read> image [[texture(0)]],
                                    constant FilterUniforms& uniforms [[buffer(0)]]) {
    const int2 size(image.get_width(), image.get_height());
    const float2 texel = input.texcoord * float2(size);
    const float2 integral = floor(texel);
    const float2 position = texel - integral - 0.5f;
    const int2 center = int2(clamp(integral, float2(-2.0f), float2(size + 1)));
    const auto pixel = [&](int x, int y) {
        return image.read(uint2(clamp(center + int2(x, y), int2(0), size - 1)));
    };

    // The nine central source texels, in row order.
    const float4 a = pixel(-1, -1);
    const float4 b = pixel(0, -1);
    const float4 c = pixel(1, -1);
    const float4 d = pixel(-1, 0);
    const float4 e = pixel(0, 0);
    const float4 f = pixel(1, 0);
    const float4 g = pixel(-1, 1);
    const float4 h = pixel(0, 1);
    const float4 i = pixel(1, 1);
    Blend top_left = Blend::None;
    Blend top_right = Blend::None;
    Blend bottom_left = Blend::None;
    Blend bottom_right = Blend::None;

    if (!((Equal(e, f) && Equal(h, i)) || (Equal(e, h) && Equal(f, i)))) {
        const float hf = Distance(g, e) + Distance(e, c) + Distance(pixel(0, 2), i) +
                         Distance(i, pixel(2, 0)) + 4.0f * Distance(h, f);
        const float ei = Distance(d, h) + Distance(h, pixel(1, 2)) + Distance(b, f) +
                         Distance(f, pixel(2, 1)) + 4.0f * Distance(e, i);
        bottom_right = Classify(hf, ei, e, f, h);
    }
    if (!((Equal(d, e) && Equal(g, h)) || (Equal(d, g) && Equal(e, h)))) {
        const float ge = Distance(pixel(-2, 1), d) + Distance(d, b) + Distance(pixel(-1, 2), h) +
                         Distance(h, f) + 4.0f * Distance(g, e);
        const float dh = Distance(pixel(-2, 0), g) + Distance(g, pixel(0, 2)) + Distance(a, e) +
                         Distance(e, i) + 4.0f * Distance(d, h);
        bottom_left = Classify(dh, ge, e, d, h);
    }
    if (!((Equal(b, c) && Equal(e, f)) || (Equal(b, e) && Equal(c, f)))) {
        const float ec = Distance(d, b) + Distance(b, pixel(1, -2)) + Distance(h, f) +
                         Distance(f, pixel(2, -1)) + 4.0f * Distance(e, c);
        const float bf = Distance(a, e) + Distance(e, i) + Distance(pixel(0, -2), c) +
                         Distance(c, pixel(2, 0)) + 4.0f * Distance(b, f);
        top_right = Classify(bf, ec, e, b, f);
    }
    if (!((Equal(a, b) && Equal(d, e)) || (Equal(a, d) && Equal(b, e)))) {
        const float db = Distance(pixel(-2, 0), a) + Distance(a, pixel(0, -2)) + Distance(g, e) +
                         Distance(e, c) + 4.0f * Distance(d, b);
        const float ae = Distance(pixel(-2, -1), d) + Distance(d, h) + Distance(pixel(-1, -2), b) +
                         Distance(b, f) + 4.0f * Distance(a, e);
        top_left = Classify(db, ae, e, d, b);
    }

    const float scale = uniforms.resolution_scale;
    float4 result = e;
    if (bottom_right != Blend::None) {
        const float fg = Distance(f, g);
        const float hc = Distance(h, c);
        const bool line =
            bottom_right == Blend::Dominant ||
            !((top_right != Blend::None && !Similar(e, g)) ||
              (bottom_left != Blend::None && !Similar(e, c)) ||
              (Similar(g, h) && Similar(h, i) && Similar(i, f) && Similar(f, c) && !Similar(e, i)));
        float2 origin(0.0f, HalfDiagonal);
        float2 direction(1.0f, -1.0f);
        if (line) {
            const bool shallow = SteepDirectionThreshold * fg <= hc && !Equal(e, g) && !Equal(d, g);
            const bool steep = SteepDirectionThreshold * hc <= fg && !Equal(e, c) && !Equal(b, c);
            origin.y = shallow ? 0.25f : 0.5f;
            direction.x += shallow ? 1.0f : 0.0f;
            direction.y -= steep ? 1.0f : 0.0f;
        }
        result = mix(result, BlendPixel(e, h, f), Ratio(position, origin, direction, scale));
    }

    if (bottom_left != Blend::None) {
        const float ha = Distance(h, a);
        const float di = Distance(d, i);
        const bool line =
            bottom_left == Blend::Dominant ||
            !((bottom_right != Blend::None && !Similar(e, a)) ||
              (top_left != Blend::None && !Similar(e, i)) ||
              (Similar(a, d) && Similar(d, g) && Similar(g, h) && Similar(h, i) && !Similar(e, g)));
        float2 origin(-HalfDiagonal, 0.0f);
        float2 direction(1.0f, 1.0f);
        if (line) {
            const bool shallow = SteepDirectionThreshold * ha <= di && !Equal(e, a) && !Equal(b, a);
            const bool steep = SteepDirectionThreshold * di <= ha && !Equal(e, i) && !Equal(f, i);
            origin.x = shallow ? -0.25f : -0.5f;
            direction.y += shallow ? 1.0f : 0.0f;
            direction.x += steep ? 1.0f : 0.0f;
        }
        result = mix(result, BlendPixel(e, h, d), Ratio(position, origin, direction, scale));
    }

    if (top_right != Blend::None) {
        const float bi = Distance(b, i);
        const float fa = Distance(f, a);
        const bool line =
            top_right == Blend::Dominant ||
            !((top_left != Blend::None && !Similar(e, i)) ||
              (bottom_right != Blend::None && !Similar(e, a)) ||
              (Similar(i, f) && Similar(f, c) && Similar(c, b) && Similar(b, a) && !Similar(e, c)));
        float2 origin(HalfDiagonal, 0.0f);
        float2 direction(-1.0f, -1.0f);
        if (line) {
            const bool shallow = SteepDirectionThreshold * bi <= fa && !Equal(e, i) && !Equal(h, i);
            const bool steep = SteepDirectionThreshold * fa <= bi && !Equal(e, a) && !Equal(d, a);
            origin.x = shallow ? 0.25f : 0.5f;
            direction.y -= shallow ? 1.0f : 0.0f;
            direction.x -= steep ? 1.0f : 0.0f;
        }
        result = mix(result, BlendPixel(e, f, b), Ratio(position, origin, direction, scale));
    }

    if (top_left != Blend::None) {
        const float dc = Distance(d, c);
        const float bg = Distance(b, g);
        const bool line =
            top_left == Blend::Dominant ||
            !((bottom_left != Blend::None && !Similar(e, c)) ||
              (top_right != Blend::None && !Similar(e, g)) ||
              (Similar(c, b) && Similar(b, a) && Similar(a, d) && Similar(d, g) && !Similar(e, a)));
        float2 origin(0.0f, -HalfDiagonal);
        float2 direction(-1.0f, 1.0f);
        if (line) {
            const bool shallow = SteepDirectionThreshold * dc <= bg && !Equal(e, c) && !Equal(f, c);
            const bool steep = SteepDirectionThreshold * bg <= dc && !Equal(e, g) && !Equal(h, g);
            origin.y = shallow ? -0.25f : -0.5f;
            direction.x -= shallow ? 1.0f : 0.0f;
            direction.y += steep ? 1.0f : 0.0f;
        }
        result = mix(result, BlendPixel(e, d, b), Ratio(position, origin, direction, scale));
    }
    return result;
}
