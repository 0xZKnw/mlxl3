// Reapply recorded FP32 updates without recomputing gates or dot products.
constexpr uint values_per_lane = Dk / 4;
auto hv = thread_position_in_grid.z;
auto lane = thread_index_in_simdgroup;
auto dv = thread_position_in_grid.y * 8 + lane / 4;
auto first_key = (lane & 3) * values_per_lane;
auto hk = hv / (Hv / Hk);
auto base = (hv * Dv + dv) * Dk + first_key;
float state[values_per_lane];
for (uint i = 0; i < values_per_lane; ++i) {
  state[i] = state_in[base + i];
}
for (uint t = 0; t < RETAINED; ++t) {
  float gt = decay_tape[t * Hv + hv];
  float delta = delta_tape[(t * Hv + hv) * Dv + dv];
  auto key_base = (t * Hk + hk) * Dk + first_key;
  for (uint i = 0; i < values_per_lane; ++i) {
    state[i] = state[i] * gt;
    state[i] = state[i] + float(k[key_base + i]) * delta;
  }
}
for (uint i = 0; i < values_per_lane; ++i) {
  state_out[base + i] = state[i];
}
