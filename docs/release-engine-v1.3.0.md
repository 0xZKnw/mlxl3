# MLXL3 engine 1.3.0

- Native Qwen MTP depths 1, 2 and 3, with target-verified greedy tokens and exact accepted-cache repair.
- Device-side recursive draft chain: a single host transfer per proposal block; K/V-only history maintenance retains the earlier MTP optimization.
- Small EXL3 verify kernels share decoded weights at two/four rows; the large vocabulary projection has a three-row tile for MTP2. A SIMD top-8 router adapted from MTPLX preserves the existing FP16 scores and tie order. Internal expert routes remain on the device.
- Resident-model `tune_mtp` bridge request measures baseline/MTP1/MTP2/MTP3 after warmup, checks output parity and acceptance, and returns the best eligible depth with progress and per-mode rates. Does not invoke MCP tools or write conversation history.
- Optional `mtp_depth` generation field defaults to 1 for existing clients. Bridge protocol remains 1. New ready capabilities advertise `mtp_max_depth`, `mtp_tune_supported` and a configuration fingerprint. Desktop 1.3.0 provides the controls and persistent selection.

Independent engine update for Apple Silicon/macOS 26.2+. No model weights included. [Validation](desktop-v1.3.0-validation.md).
