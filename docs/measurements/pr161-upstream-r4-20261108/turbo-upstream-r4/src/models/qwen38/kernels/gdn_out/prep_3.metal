    for (uint j = 0; j < 4u; ++j) xp[row * VD + head * 128u + L + 32u * j] = half(v[j] * HAD_SCALE);
