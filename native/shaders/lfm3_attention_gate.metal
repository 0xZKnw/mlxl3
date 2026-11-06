// Per-head sigmoid gate and [1,H,T,D] -> [1,T,H*D], as in ExLlamaV3 LFM3.
uint index = thread_position_in_grid.x;
uint gate_index = index / HEAD_DIM;
if (gate_index < GATE_VALUES) {
    uint token = gate_index / HEADS;
    uint head = gate_index % HEADS;
    uint channel = index % HEAD_DIM;
    uint source = (head * TIME + token) * HEAD_DIM + channel;
    half scale = half(1.0f / (1.0f + exp(-float(gate[gate_index]))));
    y[index] = attention[source] * scale;
}
