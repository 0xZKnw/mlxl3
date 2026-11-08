#ifndef MLXL3_DEVICE_HOT_SLOTS
#define MLXL3_DEVICE_HOT_SLOTS 53u
#endif
#if MLXL3_DEVICE_RESIDENT_ROUTE
// The argument encoder supplies pointers; no CPU/GPU address representation is assumed.
struct DeviceHotBank { array<const device uchar*, MLXL3_DEVICE_HOT_SLOTS> records [[id(0)]]; };
inline uint device_hot_slot(const device uint* ids, const device uint* map, uint route) {
    uint id = ids[route];
    return id < 512u && map[id] < MLXL3_DEVICE_HOT_SLOTS ? map[id] : 0xffffffffu;
}

kernel void device_hot_gu_prepare(const device float* input [[buffer(0)]],
    constant DeviceHotBank& bank [[buffer(1)]], const device uint* ids [[buffer(2)]],
    const device uint* map [[buffer(3)]], device half* output [[buffer(4)]],
    constant ulong* offsets [[buffer(5)]], uint lane [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint slot = device_hot_slot(ids, map, group.y);
    if (slot == 0xffffffffu) return; // Uniform over this entire CTA, before pointer dereference.
    const device uchar* record = bank.records[slot];
    const device half* gate_suh = reinterpret_cast<const device half*>(record + offsets[0]);
    const device half* up_suh = reinterpret_cast<const device half*>(record + offsets[1]);
    threadgroup float gate[128], up[128];
    uint start = group.x * 128u, base = group.y * 5120u;
    for (uint i = lane; i < 128u; i += 64u) {
        half x = half(input[start+i]);
        gate[i] = float(x * gate_suh[start+i]);
        up[i] = float(x * up_suh[start+i]);
    }
    hadamard128(gate, lane);
    hadamard128(up, lane);
    for (uint i = lane; i < 128u; i += 64u) {
        output[base+start+i] = half(gate[i] * as_type<float>(0x3db504f3u));
        output[base+2560u+start+i] = half(up[i] * as_type<float>(0x3db504f3u));
    }
}

#define DEVICE_HOT_QMV_ARGS const device half* prepared [[buffer(0)]], \
    constant DeviceHotBank& bank [[buffer(1)]], const device uint* ids [[buffer(2)]], \
    const device uint* map [[buffer(3)]], device float* partial [[buffer(4)]], \
    constant ExpertAxisParams& ep [[buffer(5)]], \
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
    uint3 axis_group [[threadgroup_position_in_grid]], uint3 axis_grid [[threadgroups_per_grid]]
#define DEVICE_HOT_QMV_LOCALS \
    uint expert = axis_group.z; \
    uint slot = device_hot_slot(ids, map, expert); \
    if (slot == 0xffffffffu) return; \
    const device uchar* record = bank.records[slot]; \
    const device half* xhat = prepared + expert * ep.stride; \
    const device uint* trellis = reinterpret_cast<const device uint*>(record + ep.trellis); \
    const device half* svh = reinterpret_cast<const device half*>(record + ep.svh); \
    device float* yhat = partial + expert * uint(OUTPUT_DIMS); \
    constant Params& p = ep.p; \
    uint3 threadgroup_position_in_grid = uint3(axis_group.x,0,0); \
    uint3 grid_size = uint3(axis_grid.x,1,1);

#undef K
#define K 2
#undef MLXL3_REFERENCE_ACCUMULATION
#ifndef MLXL3_FLASH_REFERENCE_ACCUMULATION
#define MLXL3_FLASH_REFERENCE_ACCUMULATION 1
#endif
#define MLXL3_REFERENCE_ACCUMULATION MLXL3_FLASH_REFERENCE_ACCUMULATION
#undef MLXL3_QMV_SG
#define MLXL3_QMV_SG 4u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u
#undef MLXL3_FUSE_OUTPUT
#define MLXL3_FUSE_OUTPUT 0
#undef MLXL3_MATRIX_ROWS
#define MLXL3_MATRIX_ROWS p.count
#undef CB
#define CB 2u
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 2u
#undef PACKED_U32
#define PACKED_U32 16u
#undef N_SPLITS
#define N_SPLITS 1u
#undef INPUT_DIMS
#define INPUT_DIMS 2560u
#undef OUTPUT_DIMS
#define OUTPUT_DIMS 640u
#undef TILES_K
#define TILES_K 160u
#undef TILES_N
#define TILES_N 40u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void device_hot_qmv_gu(DEVICE_HOT_QMV_ARGS) {
    DEVICE_HOT_QMV_LOCALS
#include "_qmv_tile_kernel.metal"
}
#undef INPUT_DIMS
#define INPUT_DIMS 640u
#undef OUTPUT_DIMS
#define OUTPUT_DIMS 2560u
#undef TILES_K
#define TILES_K 40u
#undef TILES_N
#define TILES_N 160u
kernel void device_hot_qmv_down(DEVICE_HOT_QMV_ARGS) {
    DEVICE_HOT_QMV_LOCALS
#include "_qmv_tile_kernel.metal"
}
#undef DEVICE_HOT_QMV_ARGS
#undef DEVICE_HOT_QMV_LOCALS
#undef INPUT_DIMS
#define INPUT_DIMS p.rows
#undef OUTPUT_DIMS
#define OUTPUT_DIMS p.cols
#undef TILES_K
#define TILES_K (p.rows / 16u)
#undef TILES_N
#define TILES_N (p.cols / 16u)
#undef N_SPLITS
#define N_SPLITS grid_size.z
#undef K
#define K int(p.k)
#undef CB
#define CB p.cb
#undef MLXL3_K_BITS
#define MLXL3_K_BITS p.k
#undef PACKED_U32
#define PACKED_U32 (8u * p.k)
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
#undef MLXL3_REFERENCE_ACCUMULATION
#define MLXL3_REFERENCE_ACCUMULATION 0
#undef MLXL3_QMV_SG
#define MLXL3_QMV_SG 8u

// One 64-thread CTA owns one route's H128; its arithmetic is flash_finish128's.
kernel void device_hot_finish(const device float* partial [[buffer(0)]],
    constant DeviceHotBank& bank [[buffer(1)]], const device uint* ids [[buffer(2)]],
    const device uint* map [[buffer(3)]], device half* output [[buffer(4)]],
    const device half* gate [[buffer(5)]], constant uint4& p [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    uint route = group.y, base = group.x * 128u, width = p.x;
    uint slot = device_hot_slot(ids, map, route);
    if (slot == 0xffffffffu) {
        for (uint i = lane; i < 128u; i += 64u) output[route*width+base+i] = half(0);
        return;
    }
    const device half* svh = reinterpret_cast<const device half*>(bank.records[slot] + p.y);
    threadgroup float values[128];
    for (uint i = lane; i < 128u; i += 64u) values[i] = float(half(partial[route*width+base+i]));
    hadamard128(values, lane);
    for (uint i = lane; i < 128u; i += 64u) {
        half y = half(values[i] * as_type<float>(0x3db504f3u)) * svh[base+i];
        if (p.z != 0u) {
            half g = gate[route*width+base+i];
            half tail = half(1) / (half(1) + metal::exp(metal::abs(g)));
            half sigmoid = g < half(0) ? tail : half(1) - tail;
            y = (g * sigmoid) * y;
        }
        output[route*width+base+i] = y;
    }
}
kernel void device_hot_down_prepare(const device half* input [[buffer(0)]],
    constant DeviceHotBank& bank [[buffer(1)]], const device uint* ids [[buffer(2)]],
    const device uint* map [[buffer(3)]], device half* output [[buffer(4)]],
    constant ulong& offset [[buffer(5)]], uint lane [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint slot = device_hot_slot(ids, map, group.y);
    if (slot == 0xffffffffu) return;
    const device half* suh = reinterpret_cast<const device half*>(bank.records[slot] + offset);
    threadgroup float values[128];
    uint start = group.x * 128u;
    for (uint i = lane; i < 128u; i += 64u) values[i] = float(half(input[group.y*640u+start+i] * suh[start+i]));
    hadamard128(values, lane);
    for (uint i = lane; i < 128u; i += 64u) output[group.y*5120u+start+i] = half(values[i] * as_type<float>(0x3db504f3u));
}
#if MLXL3_AR20_TESTS
kernel void device_hot_compiled(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) out[0] = MLXL3_DEVICE_RESIDENT_ROUTE;
}
#endif
#endif
