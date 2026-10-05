# MLXL3 Desktop 1.3.0 · build 22

- MTP1, MTP2 and MTP3 for compatible Qwen3.5/3.6 models. Select the depth in the MTP generation settings.
- **Tune MTP** compares ordinary greedy decoding with all three depths on the loaded model, warms every candidate, verifies identical output and automatically saves the fastest eligible mode for this model, head, context, engine and Mac.
- The tuner shows progress, four measured token rates and the chosen setting. Stop the test at any time; cancellation and errors preserve the previous setting and chat history. Baseline wins when the MTP gain is within 3%; candidates with no accepted proposals cannot win.
- Recursive draft tokens stay on the GPU until the entire chain is ready. Accepted history uses exact K/V-only repair, avoiding unused attention, MoE and vocabulary outputs. Target verification remains exact greedy decoding.

Requires engine 1.3.0 for the new depths and tuning. The app includes it; the engine is also available through its independent update channel. Older engines retain MTP1 and show an update explanation. Existing MTP heads are reused.

Apple Silicon, macOS 26.2 or later. Short local tuning is indicative: the winning depth can change with hardware, model, prompt and context. [Validation and measured example](desktop-v1.3.0-validation.md).
