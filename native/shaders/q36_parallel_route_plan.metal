// One thread per expert. The sorted order is shared with the canonical route plan.
threadgroup uint starts[257];
threadgroup uint sums[256];
uint lo = 0, hi = SLOTS;
while (lo < hi) {
    uint mid = (lo + hi) / 2;
    if (selected[order[mid]] < tid) lo = mid + 1;
    else hi = mid;
}
starts[tid] = lo;
if (tid == 255) starts[256] = SLOTS;
threadgroup_barrier(mem_flags::mem_threadgroup);
uint count = starts[tid + 1] - lo;
uint blocks = (count + 63u) / 64u;
sums[tid] = blocks;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint step = 1; step < 256; step *= 2) {
    uint previous = tid >= step ? sums[tid - step] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sums[tid] += previous;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint start_block = sums[tid] - blocks;
for (uint b = 0; b < blocks; ++b) {
    uint idx = start_block + b;
    block_table[idx] = tid;
    block_table[MAX_BLOCKS + idx] = lo + 64u * b;
    block_table[2u * MAX_BLOCKS + idx] = min(64u, count - 64u * b);
}
if (tid == 255) block_count[0] = sums[tid];
