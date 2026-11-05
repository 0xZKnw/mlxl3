uint i = thread_position_in_grid.x;
if (i >= uint(order_shape[0]) * WIDTH) return;
uint row = i / WIDTH;
uint column = i % WIDTH;
uint source = order[row];
uint hot_rows = uint(hot_shape[0]);
half value;
if (source < hot_rows) {
    value = hot[source * WIDTH + column];
} else {
    value = cold[(source - hot_rows) * WIDTH + column];
}
ACT_T converted = ACT_T(value);
PRODUCT_T product = PRODUCT_T(converted);
output[i] = product;
