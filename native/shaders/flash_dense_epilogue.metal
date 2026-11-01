// Adapted from MLX 0.32.2 reduction/reduce_col.h and hadamard.h.
// Copyright © 2023-2024 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

short i = short(thread_position_in_threadgroup.x);
uint block = threadgroup_position_in_grid.y;
uint row = block / 20u;
uint column_base = (block % 20u) * 128u;
uint partial_base = row * (2u * WIDTH) + column_base;
threadgroup half buf[128];

#pragma clang loop unroll(full)
for (short j = 0; j < 4; ++j) {
    short index = j * 32 + i * 4;
    #pragma clang loop unroll(full)
    for (short r = 0; r < 4; ++r) {
        // The stock S2 reduction initializes each lane with positive zero.
        volatile float p0 = partials[partial_base + uint(index + r)] + 0.0f;
        volatile float p1 = partials[partial_base + WIDTH + uint(index + r)] + 0.0f;
        buf[index + r] = half(p1 + p0);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float x[16];
#pragma clang loop unroll(full)
for (short r = 0; r < 16; ++r) x[r] = float(buf[i * 16 + r]);
short h = 1;
#pragma clang loop unroll(full)
for (short stage = 0; stage < 4; ++stage) {
    #pragma clang loop unroll(full)
    for (short pair = 0; pair < 8; ++pair) {
        short k = pair & (h - 1);
        short j = ((pair - k) << 1) + k;
        float a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
    }
    h <<= 1;
}
#pragma clang loop unroll(full)
for (short r = 0; r < 16; ++r) buf[i * 16 + r] = half(x[r]);
threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
for (short t = 0; t < 2; ++t) {
    short index = i + t * 8;
    #pragma clang loop unroll(full)
    for (short r = 0; r < 8; ++r) x[r] = float(buf[index + 16 * r]);
    h = 1;
    #pragma clang loop unroll(full)
    for (short stage = 0; stage < 3; ++stage) {
        #pragma clang loop unroll(full)
        for (short pair = 0; pair < 4; ++pair) {
            short k = pair & (h - 1);
            short j = ((pair - k) << 1) + k;
            float a = x[j], b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        h <<= 1;
    }
    #pragma clang loop unroll(full)
    for (short r = 0; r < 8; ++r) buf[index + 16 * r] = half(x[r]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float scale = as_type<float>(uint(SCALE_BITS));
#pragma clang loop unroll(full)
for (short j = 0; j < 4; ++j) {
    short index = j * 32 + i * 4;
    #pragma clang loop unroll(full)
    for (short r = 0; r < 4; ++r) {
        uint column = column_base + uint(index + r);
        half rotated = half(float(buf[index + r]) * scale);
        output[row * WIDTH + column] = half(rotated * svh[column]);
    }
}
