# MLXL3 Engine 2.0.16

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release adds **Liquid d1 decision models**: [d1-2B](https://huggingface.co/LiquidAI/d1-2B) (text, pictures) and [d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) (text, pictures, audio). A decision model answers named, typed questions about a state in **one forward pass, with no generated tokens**, and returns calibrated probabilities.

- New commands: `decide`, `decide-bridge` (JSON lines, used by Desktop 2.0.7) and `decide-serve` (local HTTP API, `POST /v1/decide`).
- Native audio input for d1-Omni: 16-bit PCM WAV clips, at any sample rate, and spoken questions.
- Chat, MTP and DFlash2 are unchanged: greedy LFM3 and LFM3-VL outputs are identical to 2.0.15 (details below).

## New in this release

### Decision models (d1)

- **What it does.** A request gives a state (text, any JSON value, or none when media are the whole state), optional media, and named questions. Each answer is read at the question's answer slot in the same pass:
  - `noul` (yes/no), `choice`, `choice` with `multiple: true`, `score` (2 to 10 ordered levels): a softmax over the option tokens, with the model's calibrated temperature;
  - `number` with `range` (an estimate with a p10–p90 interval), `box` and `boxes` (edges in thousandths of the picture);
  - `text`, `number` and `date` read from the state through d1's pointer, with their character span; numbers and dates are parsed with the reference's strict readers (decimal marks, currencies, scale words, day/month order).
- **How it works.** The engine reproduces d1's prompt layout, option codes and `set` attention: full-attention layers let state tokens see the whole sequence and question tokens see the state and their own question; convolution and sliding-window layers stay causal. Up to 16 questions are packed after one state and read in one pass; more questions take further passes.
- **Example.**
  ```sh
  mlxl3 decide d1-2B-EXL3-8bpw --request request.json
  ```
  with `request.json`:
  ```json
  {"state": "I was charged twice and my parcel never arrived. Please refund one charge.",
   "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                 "team": {"type": "choice", "instructions": "Which team should handle this?",
                          "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels"}}}}
  ```
- **Checkpoints.** d1 EXL3 exports are not published on the Hub yet. The engine recognizes a decision model by `d1_attention` in its `config.json`. We converted both models to 8 bpw with ExLlamaV3 1.6.6: the language model and the vision tower are quantized; d1-Omni's audio encoder and projector are kept in BF16, unmodified.

### Audio input for d1-Omni

- 16-bit PCM WAV, mono or multichannel, at any sample rate from 1 to 384 kHz, up to five minutes per clip. Channels are averaged, then resampled to 16 kHz with the same Hann-windowed sinc as the reference.
- The engine computes the 128-bin log-mel spectrogram on the CPU, then runs the convolutional downsampler, the 17-layer LFM2 audio encoder and the projector on MLX. A new MLX operation provides the stride-2 convolutions.
- **Spoken questions.** A question's `instructions` can be an audio clip (`{"type": "audio", "path": ...}`) instead of text.
- Video is not supported.

### Bridge and HTTP API

- **`mlxl3 decide-bridge <model>`**: one `{"id", "request"}` JSON line in, one `result` or `error` line out. It announces `{"type": "ready", "images", "audio", ...}` on load, and answers `{"type": "ping"}`. Local paths and `data:` URLs are both accepted.
- **`mlxl3 decide-serve <model> --port 8100 --api-key-stdin`**:
  - `GET /health`, `GET /v1/models`, `GET /v1/capabilities` and `POST /v1/decide`;
  - bearer key required; listening outside loopback requires a key;
  - media only as `data:` URLs (local paths are refused);
  - at most 8 queued requests (HTTP 429 beyond), 96 MiB per request.
  It prints the same `{"type": "ready", "port": ...}` line as `mlxl3 serve`.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

Desktop 2.0.7 is the first Desktop with the Decision mode that uses these commands. Older Desktop versions run this engine unchanged; they simply do not show decision models.

New Rust dependencies, enabled with the `chat` feature: `fancy-regex` 0.17, `unicode-normalization` 0.1.26, and `regex` as a direct dependency.

## Verification and limits

- **Decision quality** against the official d1 code run in FP32 (Transformers 5.19, A10G, pinned revisions `d1-2B@a6bce774` and `d1-Omni-2B@51918b53`), with the binary extracted from this archive and the 8 bpw exports:
  - **d1-2B**, 12 fixtures (text, nested JSON, special tokens in the state, 30 options, 20 questions in two passes, a 1,376-token state, two pictures):
    - token IDs, slot positions and character offsets identical to the reference;
    - on the 70 option slots, mean KL 8.8e-5 and maximum 8.3e-4 nats, over the 128,000-token vocabulary;
    - option probabilities within 0.0088;
    - identical answers, including parsed numbers and dates.
  - **d1-Omni-2B**, 18 fixtures (the same 12, plus synthesized speech, audio with a text state, a spoken question, French at 22,050 Hz, a pure tone, and a picture with audio): identical layout and identical final answers on all 18.
    - Audio stages on speech: log-mel within 3.9e-4 and projected features within 0.0013 relative L2.
    - **Three gates failed and are kept as is:**
      - the pure tone: the log-mel gap is 0.02, only in cells at the reference's float32 FFT noise floor (energy below 1e-9);
      - two box fixtures: KL up to 0.066 on box-edge slots against a 0.05 gate. With an unquantized vision tower, both pass (maximum 0.0061 and 0.0023), but they stay above d1-2B's level on the same code.
  - Number and date readers: 120 of 120 cases identical to the reference Python functions.
- **Chat unchanged.** We compared this archive with the published 2.0.15 archive on the same machine:
  - greedy text identical for LFM3-1.7B (French and Rust prompts) and for LFM3-VL-2B with a picture;
  - an ABBA of 8 runs on LFM3-1.7B: same text hash on every run, decode 52.5–53.4 tok/s for 2.0.16 against 52.3–53.1 for 2.0.15, within drift.
  Qwen and Gemma models were not rerun. Their code paths only gained an unused parameter.
- **Bridge and server.** Run with the released binary and d1-Omni:
  - text, audio, a spoken question, and a picture with audio sent as `data:` URLs (340 ms);
  - question order preserved in the answers;
  - error replies for invalid JSON, unknown question types, video, local paths over HTTP, a missing key (401) and unknown routes (404).
- **Released archive, extracted into a new directory:**
  - its four files match the manifest: release tag `engine-v2.0.16`, version 2.0.16, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files;
  - the three Mach-O files pass strict signature and arm64 checks;
  - `mlx.metallib` is identical to MLX 0.32.2;
  - `runtime-info` reports version 2.0.16, MLX, chat and source `40440b0bc2a5`.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy (`-D warnings`) passed for `chat`, `mlx,chat` and `direct-metal,chat`.
  - CPU tests: 601 passed, 10 ignored.
  - All MLX test targets: 664 passed, 312 ignored.
  - Direct-Metal library and binary: 560 passed, 141 ignored.
  - The ignored tests are physical-GPU or fixture tests.
- **Not verified:**
  - Kani was not run on the new code.
  - Speed and memory of decisions were not measured, except the single latencies above.
  - Only an Apple M5 MacBook with 24 GB was used.
  - Sequences longer than 8,192 tokens are refused.
  - Hosted CI on this commit had not completed when this was written.

### Known issues

- The 8 bpw d1-Omni vision tower is less accurate on box edges than d1-2B's (see above). Answers matched the reference on our fixtures.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for d1-2B, d1-Omni-2B, the LFM3 architecture and the d1 reference code, prompt format and calibration. Thanks also to [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and converter, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, [Hugging Face Transformers](https://github.com/huggingface/transformers) for the LFM3-Omni feature extractor this release follows, and [Modal](https://modal.com) for the conversion and reference GPUs.

## Source and assets

Engine source commit `40440b0bc2a51adfb7c7e5ca327a34e1767c833e`: `main` at `e86d0034f` (engine 2.0.15) plus [PR 180](https://github.com/0xZKnw/mlxl3/pull/180) (d1 decision models, audio input, Decision mode) and the version bump.

- `MLXL3-Engine-v2.0.16-arm64.tar.gz`: 68,865,479 bytes, SHA-256 `146d35340795b9e97af367bdbfa54e84d6272c82c2c3245fc23340788cfd38cd`.
- `MLXL3-Engine-v2.0.16-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `8160cb04b437f09799ebe3a7c36da497d9064eb2c78e68bd055bb49eafb4f60b`.
- `mlxl3`: 14,555,904 bytes, SHA-256 `e14a2317765debbc9ef3a67269fbf426e700acfc9affa9ada00b770fd90cffb2`.
- Detailed records: `opti.md` (entries `INTEGRATION-2026-11-18-D1`, `INTEGRATION-2026-11-18-D1-OMNI-AUDIO` and `RELEASE-ENGINE-2.0.16-DESKTOP-2.0.7`) and `docs/d1-decision-2026-11-18.md`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
