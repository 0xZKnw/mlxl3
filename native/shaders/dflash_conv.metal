constexpr uint Rows = 8;
constexpr uint Hidden = 2048;
constexpr uint ConvGroups = Hidden / 16;
constexpr uint Dynamic = 4 * ConvGroups;

uint group = threadgroup_position_in_grid.x;
for (uint element = group * 256 + thread_index_in_threadgroup;
     element < Rows * Hidden; element += DFLASH_GROUPS * 256) {
  uint row = element / Hidden;
  uint channel = element % Hidden;
  uint conv_group = channel / 16;
  uint kind = DFLASH_FINISH ? 1 : 0;
  uint coefficient = kind * 2;
  float value =
      float(input[element]) *
      (float(base[coefficient * Hidden + channel]) +
       float(dynamic[row * Dynamic + coefficient * ConvGroups + conv_group]));
  if (row > 0) {
    value +=
        float(input[(row - 1) * Hidden + channel]) *
        (float(base[(coefficient + 1) * Hidden + channel]) +
         float(dynamic[row * Dynamic +
                       (coefficient + 1) * ConvGroups + conv_group]));
  }
  if (DFLASH_FINISH)
    value += float(residual[element]);
  output[element] = bfloat(value);
}
