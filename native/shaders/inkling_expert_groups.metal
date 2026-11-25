// DFLASH2-2026-11-24-INK2M-P4-2: experts chosen by MIN_MEMBERS..LANE_TOKENS routes of one MoE call (ROUTES routes,
// threads 0..ROUTES-1): `tile_info` = {expert, members} per tile in the order of each expert's first route
// (unused tiles {0, 0}), `members` = route indices (0xffffffff padding), `shared` = 1 for the routes of a tile.
threadgroup uint chosen[ROUTES];
threadgroup uint counts[ROUTES];
threadgroup uint firsts[ROUTES];
uint r = thread_position_in_threadgroup.x;
if (r < ROUTES) chosen[r] = selected[r];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (r < ROUTES) {
    uint expert = chosen[r];
    uint count = 0u;
    uint first = 1u;
    for (uint q = 0u; q < ROUTES; ++q) {
        if (chosen[q] == expert) {
            count += 1u;
            if (q < r) first = 0u;
        }
    }
    counts[r] = count;
    firsts[r] = first;
    shared[r] = (count >= MIN_MEMBERS && count <= LANE_TOKENS) ? 1u : 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (r != 0u) return;
uint tiles = 0u;
for (uint q = 0u; q < ROUTES; ++q) {
    if (firsts[q] == 0u || counts[q] < MIN_MEMBERS || counts[q] > LANE_TOKENS || tiles >= MAX_TILES) continue;
    uint expert = chosen[q];
    uint written = 0u;
    for (uint u = q; u < ROUTES; ++u) {
        if (chosen[u] == expert) {
#if GROUP_MUTATION
            // oracle mutation: the last member of every tile is dropped
            if (written + 1u == counts[q]) break;
#endif
            members[tiles * LANE_TOKENS + written] = u;
            written += 1u;
        }
    }
    for (uint i = written; i < LANE_TOKENS; ++i) members[tiles * LANE_TOKENS + i] = 0xffffffffu;
    tile_info[tiles * 2u] = expert;
    tile_info[tiles * 2u + 1u] = written;
    tiles += 1u;
}
for (uint t = tiles; t < MAX_TILES; ++t) {
    tile_info[t * 2u] = 0u;
    tile_info[t * 2u + 1u] = 0u;
    for (uint i = 0u; i < LANE_TOKENS; ++i) members[t * LANE_TOKENS + i] = 0xffffffffu;
}
