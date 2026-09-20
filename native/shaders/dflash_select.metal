#if DFLASH_STAGE == 1
constexpr uint K = 16, Shards = 8, VectorTokens = 8, ChunkVectors = 16;
uint group = threadgroup_position_in_grid.x;
uint position = group / Shards;
uint shard = group % Shards;
ulong row_start = ulong(position) * DFLASH_VOCABULARY;
uint shard_tokens = (DFLASH_VOCABULARY + Shards - 1) / Shards;
uint begin = min(shard * shard_tokens, DFLASH_VOCABULARY);
uint end = min(begin + shard_tokens, DFLASH_VOCABULARY);
uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) %
                 VectorTokens);
head = min(head, end - begin);
uint vectors = (end - begin - head) / VectorTokens;
uint vector_begin = begin + head;
uint tail_begin = vector_begin + vectors * VectorTokens;
device const bfloat *row = logits + row_start;
device const uint4 *vector_row =
    reinterpret_cast<device const uint4 *>(row + vector_begin);

float values[K];
uint ids[K];
for (uint i = 0; i < K; ++i) {
  values[i] = -INFINITY;
  ids[i] = 0xffffffffu;
}
if (thread_position_in_threadgroup.x < head) {
  uint token = begin + thread_position_in_threadgroup.x;
  float value = float(row[token]);
  if (dflash_top_beats(value, token, values[K - 1], ids[K - 1]))
    dflash_top16_insert(values, ids, value, token);
}
if (thread_position_in_threadgroup.x < end - tail_begin) {
  uint token = tail_begin + thread_position_in_threadgroup.x;
  float value = float(row[token]);
  if (dflash_top_beats(value, token, values[K - 1], ids[K - 1]))
    dflash_top16_insert(values, ids, value, token);
}

threadgroup float maxima[256];
threadgroup float thresholds[8];
for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
  uint4 loaded[ChunkVectors];
  float best = -INFINITY;
  for (uint i = 0; i < ChunkVectors; ++i) {
    uint index = chunk + thread_position_in_threadgroup.x + i * 256;
    loaded[i] = index < vectors ? vector_row[index] : uint4(0u);
    if (index < vectors) {
      for (uint word = 0; word < 4; ++word) {
        float low = as_type<float>(loaded[i][word] << 16);
        float high = as_type<float>(loaded[i][word] & 0xffff0000u);
        best = max(best, max(low, high));
      }
    }
  }
  maxima[thread_position_in_threadgroup.x] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint above = 0;
  for (uint other = 0; other < 256; ++other)
    above += maxima[other] > best ? 1u : 0u;
  float candidate = simd_min(above < K ? best : INFINITY);
  if (thread_index_in_simdgroup == 0)
    thresholds[simdgroup_index_in_threadgroup] = candidate;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float threshold = thresholds[0];
  for (uint other = 1; other < 8; ++other)
    threshold = min(threshold, thresholds[other]);
  for (uint i = 0; i < ChunkVectors; ++i) {
    uint index = chunk + thread_position_in_threadgroup.x + i * 256;
    if (index >= vectors)
      continue;
    uint token = vector_begin + index * VectorTokens;
    for (uint word = 0; word < 4; ++word) {
      float low = as_type<float>(loaded[i][word] << 16);
      float high = as_type<float>(loaded[i][word] & 0xffff0000u);
      if (low >= threshold &&
          dflash_top_beats(low, token + 2 * word, values[K - 1], ids[K - 1]))
        dflash_top16_insert(values, ids, low, token + 2 * word);
      if (high >= threshold &&
          dflash_top_beats(high, token + 2 * word + 1, values[K - 1],
                           ids[K - 1]))
        dflash_top16_insert(values, ids, high, token + 2 * word + 1);
    }
  }
}

threadgroup float round_values[2][8];
threadgroup uint round_ids[2][8];
for (uint rank = 0; rank < K; ++rank) {
  float head_value = values[0];
  uint head_id = ids[0];
  float best;
  uint best_id;
  dflash_simd_best(head_value, head_id, best, best_id);
  uint slot = rank & 1;
  if (thread_index_in_simdgroup == 0) {
    round_values[slot][simdgroup_index_in_threadgroup] = best;
    round_ids[slot][simdgroup_index_in_threadgroup] = best_id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  best = round_values[slot][0];
  best_id = round_ids[slot][0];
  for (uint other = 1; other < 8; ++other) {
    float value = round_values[slot][other];
    uint token = round_ids[slot][other];
    if (dflash_top_beats(value, token, best, best_id)) {
      best = value;
      best_id = token;
    }
  }
  if (thread_position_in_threadgroup.x == rank) {
    partial_ids[group * K + rank] = best_id;
    partial_values[group * K + rank] = best;
  }
  if (head_value == best && head_id == best_id)
    dflash_top_pop<K>(values, ids);
}
#elif DFLASH_STAGE == 2
constexpr uint Candidates = 16, Rank = 256;
uint position = threadgroup_position_in_grid.x;
threadgroup uint successors[Candidates];
threadgroup uint predecessors[Candidates];
if (simdgroup_index_in_threadgroup == 0) {
  float value;
  uint token;
  dflash_merge_shards(partial_ids, partial_values, position,
                      thread_index_in_simdgroup, value, token);
  if (thread_index_in_simdgroup < Candidates) {
    candidates[position * Candidates + thread_index_in_simdgroup] = token;
    unary[position * Candidates + thread_index_in_simdgroup] = bfloat(value);
    successors[thread_index_in_simdgroup] = token;
  }
} else if (simdgroup_index_in_threadgroup == 1) {
  uint token = anchor[0];
  if (position > 0) {
    float value;
    dflash_merge_shards(partial_ids, partial_values, position - 1,
                        thread_index_in_simdgroup, value, token);
  }
  if (thread_index_in_simdgroup < Candidates)
    predecessors[thread_index_in_simdgroup] = token;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr uint TaskCandidates = 8, Dims = Rank / 32;
uint tasks = (position > 0 ? Candidates : 1) *
             (Candidates / TaskCandidates);
device const bfloat *row_hidden = selector + ulong(position) * Rank;
for (uint task = simdgroup_index_in_threadgroup; task < tasks; task += 8) {
  uint predecessor_index = task / (Candidates / TaskCandidates);
  uint first_candidate = task % (Candidates / TaskCandidates) * TaskCandidates;
  uint safe_predecessor =
      min(predecessors[predecessor_index], DFLASH_VOCABULARY - 1u);
  float context[Dims];
  float successor_values[TaskCandidates][Dims];
  for (uint i = 0; i < Dims; ++i) {
    uint dim = thread_index_in_simdgroup + i * 32;
    context[i] = float(predecessor[safe_predecessor * Rank + dim]) *
                 float(row_hidden[dim]);
  }
  for (uint j = 0; j < TaskCandidates; ++j) {
    uint candidate =
        min(successors[first_candidate + j], DFLASH_VOCABULARY - 1u);
    for (uint i = 0; i < Dims; ++i)
      successor_values[j][i] =
          float(successor[candidate * Rank + thread_index_in_simdgroup +
                          i * 32]);
  }
  for (uint j = 0; j < TaskCandidates; ++j) {
    float score = 0.0f;
    for (uint i = 0; i < Dims; ++i)
      score += context[i] * successor_values[j][i];
    score = simd_sum(score);
    if (thread_index_in_simdgroup == 0)
      edges[(position * Candidates + predecessor_index) * Candidates +
            first_candidate + j] = score;
  }
}
#elif DFLASH_STAGE == 3
uint predecessor_index = 0;
for (uint position = 0; position < DFLASH_POSITIONS; ++position) {
  uint selected = 0;
  for (uint candidate = 1; candidate < 16; ++candidate) {
    float score = float(unary[position * 16 + candidate]) +
                  DFLASH_EDGE_SCALE *
                      edges[(position * 16 + predecessor_index) * 16 + candidate];
    float previous = float(unary[position * 16 + selected]) +
                     DFLASH_EDGE_SCALE *
                         edges[(position * 16 + predecessor_index) * 16 + selected];
    if (score > previous)
      selected = candidate;
  }
  predecessor_index = selected;
  tokens[position] = candidates[position * 16 + selected];
}
#elif DFLASH_STAGE == 4
uint row = threadgroup_position_in_grid.x;
float best = -INFINITY;
uint best_id = 0xffffffffu;
for (uint token = thread_position_in_threadgroup.x; token < DFLASH_VOCABULARY;
     token += 256) {
  float value = float(logits[row * DFLASH_VOCABULARY + token]);
  if (!isnan(value) && dflash_top_beats(value, token, best, best_id)) {
    best = value;
    best_id = token;
  }
}
float simd_value;
uint simd_id;
dflash_simd_best(best, best_id, simd_value, simd_id);
threadgroup float group_values[8];
threadgroup uint group_ids[8];
if (thread_index_in_simdgroup == 0) {
  group_values[simdgroup_index_in_threadgroup] = simd_value;
  group_ids[simdgroup_index_in_threadgroup] = simd_id;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (thread_position_in_threadgroup.x == 0) {
  best = group_values[0];
  best_id = group_ids[0];
  for (uint group = 1; group < 8; ++group) {
    if (dflash_top_beats(group_values[group], group_ids[group], best, best_id)) {
      best = group_values[group];
      best_id = group_ids[group];
    }
  }
  tokens[row] = best_id;
}
#endif
