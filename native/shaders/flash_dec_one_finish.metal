// Exact finish adapted from MLX 0.32.2 / flash_dense_epilogue.metal.
// Arrival publication adapted from turboexl3 087c757, dec_gemv_arrive.metal.
// Copyright © 2023-2024 Apple Inc. Copyright (c) 2026 David Tai.
// MIT licensed; see THIRD_PARTY_NOTICES.md and the adjacent source credits.

static_assert(MLXL3_QMV_NT == 1u && MLXL3_QMV_SG == 4u && N_SPLITS == 2,
              "DEC-ONE preserves the canonical S2/NT1/SG4 partition");
static_assert(MLXL3_MATRIX_ROWS == MLXL3_QMV_MB && MLXL3_QMV_MB <= 2u,
              "DEC-ONE supports one complete B1/B2 row group");
uint finish_block = tile_n / 8u;
device atomic_uint* counters = reinterpret_cast<device atomic_uint*>(state + DEC_PARTIAL_WORDS);
threadgroup uint arrived_last;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
threadgroup_barrier(mem_flags::mem_device);
if (tid == 0u) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint old = atomic_fetch_add_explicit(counters + finish_block, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    arrived_last = uint(old == 15u);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (arrived_last == 0u) return;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

device half* final_output = reinterpret_cast<device half*>(state + DEC_OUTPUT_WORD);
threadgroup half finish_buf[128];
for (uint finish_row = 0u; finish_row < MLXL3_MATRIX_ROWS; ++finish_row) {
    uint column_base = finish_block * 128u;
    uint partial_base = finish_row * 5120u + column_base;
    short i = short(tid);
    if (tid < 8u) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) {
            short index = j * 32 + i * 4;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; ++r) {
                volatile float p0 = yhat[partial_base + uint(index + r)] + 0.0f;
                volatile float p1 = yhat[partial_base + 2560u + uint(index + r)] + 0.0f;
                finish_buf[index + r] = half(p1 + p0);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        float x[16];
        #pragma clang loop unroll(full)
        for (short r = 0; r < 16; ++r) x[r] = float(finish_buf[i * 16 + r]);
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
        for (short r = 0; r < 16; ++r) finish_buf[i * 16 + r] = half(x[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        float x[8];
        #pragma clang loop unroll(full)
        for (short t = 0; t < 2; ++t) {
            short index = i + t * 8;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 8; ++r) x[r] = float(finish_buf[index + 16 * r]);
            short h = 1;
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
            for (short r = 0; r < 8; ++r) finish_buf[index + 16 * r] = half(x[r]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        float scale = as_type<float>(uint(DEC_SCALE_BITS));
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) {
            short index = j * 32 + i * 4;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; ++r) {
                uint column = column_base + uint(index + r);
                half rotated = half(float(finish_buf[index + r]) * scale);
                final_output[finish_row * 2560u + column] = half(rotated * svh[column]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
