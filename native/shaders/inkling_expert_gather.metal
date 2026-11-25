// DFLASH2-2026-11-24-INK2M-P4-2: the members' transformed input rows of every live tile in the lane order of the
// transposed TensorOps kernel (`exl3_lane_source_index`), LANE_TOKENS rows per (tile, projection), zero padded.
uint i = thread_position_in_grid.x;
if (i >= uint(MAX_TILES) * uint(PROJECTIONS) * uint(LANE_TOKENS) * uint(INPUT_DIMS)) return;
uint k = i % uint(INPUT_DIMS);
uint row = i / uint(INPUT_DIMS);
uint tile = row / (uint(PROJECTIONS) * uint(LANE_TOKENS));
uint rem = row - tile * uint(PROJECTIONS) * uint(LANE_TOKENS);
uint projection = rem / uint(LANE_TOKENS);
uint token = rem - projection * uint(LANE_TOKENS);
uint count = tile_info[tile * 2u + 1u];
if (count == 0u) return;
uint within = k & 15u;
uint source = (k & ~15u) | ((within & 12u) >> 1u) | (within & 1u) | ((within & 2u) << 2u);
half value = half(0.0h);
if (token < count) {
    uint route = members[tile * uint(LANE_TOKENS) + token];
    value = x[(ulong(route) * uint(PROJECTIONS) + projection) * uint(INPUT_DIMS) + source];
}
xg[i] = value;
