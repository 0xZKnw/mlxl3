uint row = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
float value;
uint token;
dflash_merge_shards(partial_ids, partial_values, row, lane, value, token);
bool valid = simd_all(lane >= 16u || (token < 248320u && isfinite(value)));
if (lane < 16u) {
  candidates[row * 16u + lane] = min(token, 248319u);
  unary[row * 16u + lane] = value;
}
if (lane == 0u)
  valid_rows[row] = uint(valid);
