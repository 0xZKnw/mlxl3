# Routed expert local scratch experiment

Status: CPU-qualified local prototype; physical correctness, composed-model validation and throughput remain unverified. Default is OFF (`MLXL3_NATIVE_ROUTED_SCRATCH=0`); explicit `1` selects the experiment. Baseline main `21cfc9ad0b6679f9e6975bf6f02bc7a5b7b96d47`, whose native sources match PR118 `34adade01`.

Only immutable routed K2/MUL1 records from `FlashExpert::from_backing`, one row, splits1, 2560→640→2560 can reuse two local temporaries (5120 and10240bytes) across gate, up and down in the same retained command buffer. The four destinations remain independent. Shared/dense/other dimensions or batch sizes use the original path. The strict flag is captured once per Metal context; malformed values reject eligible routed encodes only.

Expected requested allocations per expert:10→6,35840→24320bytes. These are arithmetic predictions pending actual physical counters, not RAM savings or measured throughput. Ten dispatches and all shaders/rounding remain unchanged. Tracked write-after-read hazards may reduce GPU overlap. There is no persistent pool, inter-expert/wave sharing, new offset or new kernel.

## Physical component protocol (not yet run)

Use the frozen stock `runtime_gpu` executable with MLX, then the newly composed native `libtest`, exclusively after the root GPU grant. The existing stock oracle performs actual Exl3Linear gate/up/SwiGLU/down and writes `stock.json`. The native test requires bit-exact finite, nonzero 2560-word outputs, explicit actualnewBuffer10/6 and three bindings to the same two nonzero scratch identities. It also checks four nonalias destinations, F16/F32 reverse padded views, varied inputs, two experts plus a duplicate in one command, owners dropped before commit, poisoned scratch, unchanged prepared tail after down, insufficient lengths/alias rejection and error-before-commit recovery, malformed-setting fallback and batch/shared fallback. The report is written only after every check succeeds.

Generate disposable fixture packets without importing MLX (MODEL is the real Flash Next checkpoint path):

```python
import json, os
from pathlib import Path
import numpy as np
root = Path(os.environ["FIXTURES"])
root.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(751204)
cases = []
for replay in range(2):
    for expert in range(2):
        x = rng.normal(0, 0.0625, 2560).astype(np.float16)
        cases.append(dict(op="metal-checkpoint-expert", path=os.environ["MODEL"],
            prefix=f"model.language_model.layers.0.mlp.experts.{expert}",
            x=x.view(np.uint16).tolist()))
x = rng.normal(0, 0.0625, 2560).astype(np.float16)
cases.append(dict(op="metal-checkpoint-expert", path=os.environ["MODEL"],
    prefix="model.language_model.layers.0.mlp.shared_expert", x=x.view(np.uint16).tolist()))
(root / "cases.json").write_text(json.dumps(cases))
```

Run each process with a bounded timeout and join/kill its process group on error; record exact artifact SHA256s and environment options. Required commands (variables are explicit paths):

```sh
MLXL3_PROJECTION_FIXTURES="$FIXTURES" "$STOCK" flash_projection_stock_oracle --exact --ignored --test-threads=1
MLXL3_NATIVE_ROUTED_SCRATCH=1 MLXL3_ROUTED_SCRATCH_FIXTURES="$FIXTURES" "$NATIVE" gpu::projection::routed_scratch_tests::physical_routed_expert_scratch --exact --ignored --test-threads=1
```

This gate mutates only the context's test-only accessible setting to exercise OFF/ON/invalid on one device. The production setting stays immutable. Individual component parity does not validate the composed model. Root must still check states/routes/logits and actual AR CLI budgets/cancellation before a speed pilot. No MTP/DFlash campaign is requested.

## Verification limits

Kani checks the actual production selector, prefix capacity arithmetic and strict flag parser, with symbolic u32 extents/u64 capacities, all routing booleans and all three setting states; UTF8 strings have length0…4. No assumptions or stubs. The selector's36-byte array comparison needs unwind40, parser unwind10; arithmetic is loop-free. Initial wrapper unwind4 failed in builtin memcmp and is retained separately. The corrected actual package attempt is authoritative. These CPU bounded checks do not prove Metal, allocator ownership, asynchronous hazards or full generation. Existing long grouped-route/SpecFill baseline proof limits are not repaired or rerun by this experiment.

After CPU checks, independent review strengthened only the physical fixture guard: count distinct prefixes on the routed slice excluding shared, and require the first two routed prefixes to differ. Production code is unchanged. That last assertion-only edit was not recompiled under the root coordination instruction; the composed rebuild must compile it before physical execution. `verification.json` preserves both source fingerprints.
