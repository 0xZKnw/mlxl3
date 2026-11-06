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

// A raw-logits maximum is not equivalent: Half subtraction can introduce ties.
kernel void flash_normalized_greedy(const device half* logits [[buffer(0)]],
    const device half* lse [[buffer(1)]],device uint* out [[buffer(2)]],
    constant uint2& params [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  uint count=params.x;
  if(count!=248320u) {if(tid==0) out[0]=0xffffffffu;return;}
  threadgroup float values[32];
  threadgroup uint indices[32], errors[32];
  float best=-INFINITY;uint index=0xffffffffu;
  uint invalid=uint(!isfinite(float(lse[0])));
  for(uint i=tid;i<count;i+=1024u) {
    float raw=float(logits[i]);
    float value=float(half(raw-float(lse[0])));
    invalid|=uint((params.y!=0u && !isfinite(raw)) || isnan(value) || value==INFINITY);
    if(isfinite(value)&&(value>best||(value==best&&i<index))) {best=value;index=i;}
  }
  float maximum=simd_max(best);
  uint winner=simd_min(best==maximum?index:0xffffffffu);
  uint bad=simd_or(invalid);
  if(lane==0) {values[sg]=maximum;indices[sg]=winner;errors[sg]=bad;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) {
    maximum=simd_max(values[lane]);
    winner=simd_min(values[lane]==maximum?indices[lane]:0xffffffffu);
    bad=simd_or(errors[lane]);
    if(lane==0) out[0]=bad?0xffffffffu:winner;
  }
}
