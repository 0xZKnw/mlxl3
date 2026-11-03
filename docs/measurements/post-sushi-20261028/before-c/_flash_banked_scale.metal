uint i = thread_position_in_grid.x;
if (i >= uint(x_shape[0])) return;
uint row = i / WIDTH;
uint slot = slots[row / PROJECTIONS];
half scale = half(0);
if (slot < CAPACITY) {
    scale = bank[(slot * PROJECTIONS + row % PROJECTIONS) * WIDTH + i % WIDTH];
}
out[i] = x[i] * scale;
