# MLXL3 engine 1.4.1

Includes [PR #25](https://github.com/0xZKnw/mlxl3/pull/25): use four output tiles for three measured dense single-token EXL3 projections on Apple M5, preserving weights, codebooks and reduction order. Other shapes retain their existing dispatch. Set `MLXL3_DENSE_DECODE_NT4=0` to restore the previous geometry.

The source campaign found about 5% improvement in eager native forward on its tested M5 and protocol. Bridge throughput remains inconclusive; no universal speedup is claimed. [Measurements and scope](https://github.com/0xZKnw/mlxl3/blob/main/docs/qwen27-m5-round2.md).

Retains engine 1.4.0 automatic Qwen dense/MoE MTP head selection and cleanup. Experimental norm fusion remains off by default.

Independent engine update for Apple Silicon/macOS 26.2+, MLXL3 Desktop 1.4.0+, bridge protocol 1 and MLX 0.32.2. No model weights included. Restart Desktop after installing the engine update.
