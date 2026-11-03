uint i = thread_position_in_grid.x;
if (i >= 10240u) return;
out[i] = fma(float(branch[i % 2560u]), gate[i / 2560u], streams[i]);
