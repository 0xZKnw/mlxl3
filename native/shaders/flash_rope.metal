// Text-only half-split MRoPE from mlx-vlm/models/rope_utils.py.
// MIT licensed; see THIRD_PARTY_NOTICES.md.
uint elem = thread_position_in_grid.x;
const int half_dim = ROTARY / 2;
const int slots = half_dim + WIDTH - ROTARY;
if (elem >= uint(HEADS * slots)) return;
int slot = int(elem) % slots;
int base = (int(elem) / slots) * WIDTH;
if (slot >= half_dim) {
    int pass = base + ROTARY + slot - half_dim;
    out[pass] = x[pass];
    return;
}
float angle = float(position[0]) * float(inv_freq[slot]);
float c = metal::cos(angle);
float s = metal::sin(angle);
int idx = base + slot;
float xv = float(x[idx]);
float xp = float(x[idx + half_dim]);
out[idx] = half(xv * c - xp * s);
out[idx + half_dim] = half(xp * c + xv * s);
