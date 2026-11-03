uint i = thread_position_in_grid.x;
if (i >= 10240u) return;
// Preserve the graph's rounded FP32 product before addition, without FMA.
volatile float product = float(branch[i % 2560u]) * gate[i / 2560u];
out[i] = streams[i] + product;
