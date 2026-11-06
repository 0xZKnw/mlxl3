// LFM3 B*X -> depthwise causal conv -> C gate; ExLlamaV3 ShortConv semantics.
uint channel = thread_position_in_grid.x;
if (channel < DIM) {
    half current = bcx[channel] * bcx[2 * DIM + channel];
    float sum = 0.0f;
    for (uint tap = 0; tap < STATE_VALUES / DIM; ++tap) {
        sum += float(state[tap * DIM + channel]) * float(weight[7 * channel + tap]);
        next_state[tap * DIM + channel] = tap < 5
            ? state[(tap + 1) * DIM + channel] : current;
    }
    sum += float(current) * float(weight[7 * channel + 6]);
    y[channel] = bcx[DIM + channel] * half(sum);
}
