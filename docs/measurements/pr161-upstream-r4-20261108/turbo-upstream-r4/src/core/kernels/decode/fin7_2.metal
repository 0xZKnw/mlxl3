    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    float g[4]; float u[4];
    for (uint q = 0; q < 4u; ++q) {
        const uint o = b * 128u + L + 32u * q;
        float gg = 0.0f, uu = 0.0f;
