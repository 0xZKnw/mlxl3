threadgroup float input_sums[64];
uint group = threadgroup_position_in_grid.x;
for (uint tile = group; tile < DFLASH_OUTPUT / DFLASH_TILE;
     tile += DFLASH_GROUPS) {
  dflash_q4_tile<DFLASH_TILE, DFLASH_PIPELINED>(
      const_cast<device bfloat *>(input),
      const_cast<device uchar *>(weights),
      const_cast<device bfloat *>(scales),
      const_cast<device bfloat *>(biases), output, input_sums,
      DFLASH_OUTPUT_BEGIN + tile * DFLASH_TILE, thread_index_in_simdgroup,
      simdgroup_index_in_threadgroup);
}
