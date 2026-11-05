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

// Tier 2 argument buffers bind an independent projection per grid Y.
struct MappedQmvJob {
    const device half* input;
    const device uint* weights;
    const device half* scale;
    device float* output;
};

#define MAPPED_QMV_ARGUMENTS const device MappedQmvJob* jobs [[buffer(0)]], \
    constant Params& p [[buffer(1)]], \
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
    uint3 grid_size [[threadgroups_per_grid]]
#define MAPPED_QMV_BIND \
    const device MappedQmvJob& job = jobs[threadgroup_position_in_grid.y]; \
    const device half* xhat = job.input; \
    const device uint* trellis = job.weights; \
    const device half* svh = job.scale; \
    device float* yhat = job.output;
#define MLXL3_INDEPENDENT_QMV_JOBS

#undef MLXL3_QMV_SG
#define MLXL3_QMV_SG 4u

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void mapped_qmv_s4_n2(MAPPED_QMV_ARGUMENTS) {
    MAPPED_QMV_BIND
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 4u
kernel void mapped_qmv_s4_n4(MAPPED_QMV_ARGUMENTS) {
    MAPPED_QMV_BIND
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_SG
#define MLXL3_QMV_SG 8u

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void mapped_qmv_s8_n2(MAPPED_QMV_ARGUMENTS) {
    MAPPED_QMV_BIND
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 4u
kernel void mapped_qmv_s8_n4(MAPPED_QMV_ARGUMENTS) {
    MAPPED_QMV_BIND
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_INDEPENDENT_QMV_JOBS
#undef MAPPED_QMV_BIND
#undef MAPPED_QMV_ARGUMENTS

struct QwenStageScales {
    const device half* su[3];
    const device half* sv[3];
};

inline void dense_finish_block(const device float* partials, const device half* svh,
    const device half* bias, device half* output, uint width, uint splits, bool has_bias,
    uint lane, uint block, threadgroup float* values) {
    uint base = block * 128, col = base % width;
    uint first = (base / width * splits) * width + col;
    for (uint i = lane; i < 128; i += 64) {
        const uint stripes = min(splits, 8u);
        float sum = 0.0f;
        for (uint stripe = 0; stripe < stripes; ++stripe) {
            float part = 0.0f;
            for (uint split = stripe; split < splits; split += stripes) {
                part = partials[first + split * width + i] + part;
            }
            sum = stripe == 0 ? part : part + sum;
        }
        values[i] = float(half(sum));
    }
    hadamard128(values, lane);
    for (uint i = lane; i < 128; i += 64) {
        half y = half(values[i] * as_type<float>(0x3db504f3u)) * svh[col+i];
        output[base+i] = has_bias ? half(y + bias[col+i]) : y;
    }
}

kernel void dense_finish128(const device float* partials [[buffer(0)]],
    const device half* svh [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* output [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    dense_finish_block(partials, svh, bias, output, p.cols, p.count, bool(p.cb), lane, block, values);
}

kernel void qwen_stage_finish(
    const device float* partials [[buffer(0)]],
    const device QwenStageScales* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& splits [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    uint route = group.y;
    const device half* sv = scales[route].sv[2];
    dense_finish_block(partials + route * splits * 2048u, sv, sv,
                       output + route * 2048u, 2048u, splits, false,
                       lane, group.x, values);
}


#undef DENSE_QMV_ARGUMENTS
