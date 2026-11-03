// Dense and resident-MoE row projections retain their established accumulation.
#define K int(p.k)
#define CB p.cb
#define PACKED_U32 (8u * p.k)
#define TILES_K (p.rows / 16u)
#define TILES_N (p.cols / 16u)
#define INPUT_DIMS p.rows
#define OUTPUT_DIMS p.cols
#define N_SPLITS grid_size.z
#define MLXL3_K_BITS p.k
#define MLXL3_MATRIX_ROWS p.count
#define MLXL3_QMV_MB 1u
#define MLXL3_FUSE_OUTPUT 0
#define mlxl3_decode_codeword(x, mode) codeword(x, mode)
#define DENSE_QMV_ARGUMENTS const device half* xhat [[buffer(0)]], \
    const device uint* trellis [[buffer(1)]], \
    const device half* svh [[buffer(2)]], \
    device float* yhat [[buffer(3)]], \
    constant Params& p [[buffer(4)]], \
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
    uint3 grid_size [[threadgroups_per_grid]]
#undef MLXL3_REFERENCE_ACCUMULATION
#define MLXL3_REFERENCE_ACCUMULATION 0

#undef MLXL3_QMV_SG
#define MLXL3_QMV_SG 4u

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
kernel void dense_qmv_s4_n1(DENSE_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void dense_qmv_s4_n2(DENSE_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 4u
kernel void dense_qmv_s4_n4(DENSE_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_SG
#define MLXL3_QMV_SG 8u

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
kernel void dense_qmv_s8_n1(DENSE_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void dense_qmv_s8_n2(DENSE_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 4u
kernel void dense_qmv_s8_n4(DENSE_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

kernel void dense_finish128(const device float* partials [[buffer(0)]],
    const device half* svh [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* output [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    uint base = block * 128, col = base % p.cols;
    uint first = (base / p.cols * p.count) * p.cols + col;
    for (uint i = lane; i < 128; i += 64) {
        const uint stripes = min(p.count, 8u);
        float sum = 0.0f;
        for (uint stripe = 0; stripe < stripes; ++stripe) {
            float part = 0.0f;
            for (uint split = stripe; split < p.count; split += stripes) {
                part = partials[first + split * p.cols + i] + part;
            }
            sum = stripe == 0 ? part : part + sum;
        }
        values[i] = float(half(sum));
    }
    hadamard128(values, lane);
    for (uint i = lane; i < 128; i += 64) {
        half y = half(values[i] * as_type<float>(0x3db504f3u)) * svh[col+i];
        output[base+i] = p.cb ? half(y + bias[col+i]) : y;
    }
}

#undef DENSE_QMV_ARGUMENTS
