uint index = thread_position_in_grid.x;
if (index >= ROWS * WIDTH) return;
uint row = index / WIDTH;
uint column = index % WIDTH;
float normalized;
switch (row) {
    case 0: normalized = normed0[column]; break;
#if ROWS > 1
    case 1: normalized = normed1[column]; break;
#endif
#if ROWS > 2
    case 2: normalized = normed2[column]; break;
#endif
#if ROWS > 3
    case 3: normalized = normed3[column]; break;
#endif
#if ROWS > 4
    case 4: normalized = normed4[column]; break;
#endif
#if ROWS > 5
    case 5: normalized = normed5[column]; break;
#endif
#if ROWS > 6
    case 6: normalized = normed6[column]; break;
#endif
#if ROWS > 7
    case 7: normalized = normed7[column]; break;
#endif
    default: return;
}
float gate = float(z[index]);
float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gate)));
// The stock sigmoid and product each materialize FP32 before the next operation.
volatile float probability = gate < 0.0f ? y : 1.0f - y;
volatile float product = normalized * probability;
output[index] = half(product);
