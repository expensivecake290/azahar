// MIT License
//
// Copyright (c) 2019 bloc97
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>

using namespace metal;

struct GradientInput {
    float2 texcoord [[user(locn0)]];
};

fragment float y_gradient_frag(GradientInput input [[stage_in]],
                               texture2d<float> gradient [[texture(2)]],
                               sampler linear_edge [[sampler(0)]]) {
    const float2 next = gradient.sample(linear_edge, input.texcoord, level(0.0f), int2(0, 1)).rg;
    const float2 center = gradient.sample(linear_edge, input.texcoord, level(0.0f)).rg;
    const float2 previous =
        gradient.sample(linear_edge, input.texcoord, level(0.0f), int2(0, -1)).rg;

    const float2 response(next.x + 2.0f * center.x + previous.x, previous.y - next.y);
    return 1.0f - length(response);
}
