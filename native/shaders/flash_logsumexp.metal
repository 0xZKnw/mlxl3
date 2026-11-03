// Flash FP16 vocabulary normalizer. Adapted from MLX 0.32.2, Copyright © 2025 Apple Inc. (MIT).
kernel void flash_logsumexp(device const half* in [[buffer(0)]],device half* out [[buffer(1)]],
    constant int& axis_size [[buffer(2)]],uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS=4;
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_max[SIMD_SIZE];
  threadgroup float local_normalizer[SIMD_SIZE];

  // Get the max and the normalizer in one go
  float prevmax;
  float maxval = (-FLT_MAX);
  float normalizer = 0;
  for (int r = 0; r < int((axis_size + N_READS * lsize - 1)/(N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    float vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = float(in[offset + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        vals[i] =
            (offset + i < axis_size) ? float(in[offset + i]) : (-INFINITY);
      }
    }
    prevmax = maxval;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }
    normalizer *= fast::exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer += fast::exp(vals[i] - maxval);
    }
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);

  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[simd_lane_id]);
  normalizer *= fast::exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_normalizer[simd_lane_id]);

  if (lid == 0) {
    out[0] = isinf(maxval) ? half(maxval) : half(log(normalizer) + maxval);
  }
}
