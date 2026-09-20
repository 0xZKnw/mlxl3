// Fused Qwen Gated DeltaNet convolution, SiLU and Q/K normalization.

            uint lane = thread_position_in_threadgroup.x;
            uint head = threadgroup_position_in_grid.y;
            uint time = threadgroup_position_in_grid.z;

            if (head < uint(KEY_HEADS)) {
                half q_values[4];
                half k_values[4];
                float q_sum = 0.0f;
                float k_sum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    uint dim = lane * 4u + i;
                    uint q_channel = head * uint(KEY_DIM) + dim;
                    uint k_channel = uint(KEYS) + q_channel;
                    q_values[i] = mlxl3_gdn_conv_silu(
                        qkv, conv_state, conv_weight, time, q_channel
                    );
                    k_values[i] = mlxl3_gdn_conv_silu(
                        qkv, conv_state, conv_weight, time, k_channel
                    );
                    float q_value = float(q_values[i]);
                    float k_value = float(k_values[i]);
                    q_sum += q_value * q_value;
                    k_sum += k_value * k_value;
                }
                q_sum = simd_sum(q_sum);
                k_sum = simd_sum(k_sum);
                float q_inv = metal::precise::rsqrt(
                    q_sum / float(KEY_DIM) + 1.0e-6f
                );
                float k_inv = metal::precise::rsqrt(
                    k_sum / float(KEY_DIM) + 1.0e-6f
                );
                ulong output_base =
                    (ulong(time) * uint(KEY_HEADS) + head) * uint(KEY_DIM)
                    + lane * 4u;
                for (uint i = 0u; i < 4u; ++i) {
                    half q_normalized = half(float(q_values[i]) * q_inv);
                    half k_normalized = half(float(k_values[i]) * k_inv);
                    q[output_base + i] = q_normalized * q_scale;
                    k[output_base + i] = k_normalized * k_scale;
                }
            }

            if (head < uint(VALUE_HEADS)) {
                for (uint dim = lane; dim < uint(VALUE_DIM); dim += 32u) {
                    uint channel = uint(2 * KEYS) + head * uint(VALUE_DIM) + dim;
                    ulong output_index =
                        (ulong(time) * uint(VALUE_HEADS) + head)
                        * uint(VALUE_DIM) + dim;
                    v[output_index] = mlxl3_gdn_conv_silu(
                        qkv, conv_state, conv_weight, time, channel
                    );
                }
            }

            if (time == 0u) {
                if (head < uint(KEY_HEADS)) {
                    for (uint i = 0u; i < 4u; ++i) {
                        uint dim = lane * 4u + i;
                        uint q_channel = head * uint(KEY_DIM) + dim;
                        uint k_channel = uint(KEYS) + q_channel;
                        for (uint position = 0u; position < uint(CONV_HISTORY); ++position) {
                            uint source = uint(TIME) + position;
                            state_out[position * uint(CONV_DIMS) + q_channel] =
                                mlxl3_gdn_input(qkv, conv_state, source, q_channel);
                            state_out[position * uint(CONV_DIMS) + k_channel] =
                                mlxl3_gdn_input(qkv, conv_state, source, k_channel);
                        }
                    }
                }
                if (head < uint(VALUE_HEADS)) {
                    for (uint dim = lane; dim < uint(VALUE_DIM); dim += 32u) {
                        uint channel = uint(2 * KEYS) + head * uint(VALUE_DIM) + dim;
                        for (uint position = 0u; position < uint(CONV_HISTORY); ++position) {
                            uint source = uint(TIME) + position;
                            state_out[position * uint(CONV_DIMS) + channel] =
                                mlxl3_gdn_input(qkv, conv_state, source, channel);
                        }
                    }
                }
            }
