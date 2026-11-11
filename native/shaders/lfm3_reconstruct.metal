// Rotated-basis EXL3 reconstruction; addressing matches the QMM kernel.
uint index = thread_position_in_grid.x;
uint row = index / OUTPUT_DIMS;
uint col = index % OUTPUT_DIMS;
uint local = (row & 15u) * 16u + (col & 15u);
uint source = uint(mlxl3_perm_inv[local]);
uint position = source >> 1u;
int begin = int(position * 16u + 8u) - 16 + 2048;
int end = begin + 24;
uint word0 = uint(begin / 32) % 64u;
uint word1 = uint((end - 1) / 32) % 64u;
uint shift = uint(((end - 1) / 32 + 1) * 32 - end) + ((source & 1u) ? 0u : 8u);
ulong tile = ulong(row / 16u) * (OUTPUT_DIMS / 16u) + col / 16u;
const device uint* words = trellis + tile * 64u;
ulong merged = (ulong(words[word0]) << 32) | ulong(words[word1]);
uint codeword = uint(merged >> shift) & 0xffffu;
weights[index] = half(mlxl3_decode_codeword(codeword, 0));
