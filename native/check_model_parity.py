"""Strict forced-token comparison of native models and Python production.

Run only on a physical Apple GPU, with the same libmlx and checkpoint. Compares
every logit, plus live recurrent/KV values when the native model exports them.
Token-wise prefill on both sides isolates the architecture port; no speed claim
is made.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx
import numpy as np
from mlxl3.checkpoint import load_exl3_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/mlxl3-rs"))
    parser.add_argument("--tokens", default="1,2,3,19,225,4096,17,7")
    parser.add_argument("--ungrouped-reference", action="store_true",
        help="diagnostic only: disables Python QKV grouping, not production parity")
    args = parser.parse_args()
    if args.ungrouped_reference:
        import mlxl3.checkpoint as loader
        loader.fuse_compatible_linear_groups = lambda model: 0
    tokens = [int(token) for token in args.tokens.split(",")]
    config = json.loads((args.model / "config.json").read_text())
    model_type = config["model_type"]
    exact = model_type != "gemma4"
    compared = 0
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode="w+") as errors:
        directory = Path(directory)
        model, _, _ = load_exl3_model(args.model, lazy=False)
        cache = model.make_cache()
        state_names: list[list[str]] = []
        for step, token in enumerate(tokens):
            expected = model(mx.array([[token]], dtype=mx.int32), cache=cache)
            mx.eval(expected)
            np.save(directory / f"logits-{step}.npy", np.asarray(expected))
            names = []
            if model_type in ("lfm2", "lfm2_moe"):
                for index, layer in enumerate(cache):
                    if hasattr(layer, "keys"):
                        values = zip(("keys", "values"), layer.state)
                    else:
                        values = (("conv_state", layer[0]),)
                    for suffix, value in values:
                        name = f"model.layers.{index}.{suffix}"
                        mx.eval(value)
                        np.save(directory / f"state-{step}-{len(names)}.npy", np.asarray(value))
                        names.append(name)
            state_names.append(names)
        del expected, cache, model
        gc.collect()
        mx.clear_cache()

        command = [str(args.binary.resolve()), "forward", str(args.model), "--tokens", args.tokens]
        if model_type in ("lfm2", "lfm2_moe"):
            command.append("--states")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for step, token in enumerate(tokens):
                line = process.stdout.readline()
                if not line:
                    errors.seek(0)
                    raise AssertionError(f"native engine exited early: {errors.read()}")
                actual = json.loads(line)
                assert actual["token"] == token
                if actual["states"]:
                    assert [state["name"] for state in actual["states"]] == state_names[step]
                    # Check states first so a divergent layer is identified before
                    # the inevitable final-logit mismatch.
                    for index, state in enumerate(actual["states"]):
                        reference = np.load(directory / f"state-{step}-{index}.npy")
                        assert state["shape"] == list(reference.shape), state["name"]
                        np.testing.assert_array_equal(state["data"], reference.view(np.uint8).ravel(),
                            err_msg=f"step {step}, cache {state['name']}")
                logits = np.load(directory / f"logits-{step}.npy")
                assert logits.dtype == np.float16, f"unexpected logits dtype {logits.dtype}"
                native = np.asarray(actual["logits"], dtype=np.uint16).view(np.float16)
                reference = logits.ravel()
                if exact:
                    np.testing.assert_array_equal(native.view(np.uint16), reference.view(np.uint16),
                        err_msg=f"step {step}, complete logits")
                else:
                    native32 = native.astype(np.float32)
                    reference32 = reference.astype(np.float32)
                    assert int(native32.argmax()) == int(reference32.argmax()), f"step {step}, top-1"
                    assert float(np.max(np.abs(native32 - reference32))) <= 0.5, f"step {step}, max error"
                    p = np.exp(native32 - native32.max(), dtype=np.float64); p /= p.sum()
                    q = np.exp(reference32 - reference32.max(), dtype=np.float64); q /= q.sum()
                    assert float(np.sum(q * np.log(q / p))) <= 0.005, f"step {step}, KL"
                compared += 1
            assert not process.stdout.read().strip(), "unexpected extra model output"
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
    print(json.dumps({"status":"passed", "steps":compared, "model":str(args.model),
        "production_reference": not args.ungrouped_reference,
        "comparison": "all logits bit-for-bit; caches when exported" if exact else
            "same top-1, max abs <= 0.5 and KL <= 0.005 against cached production Gemma",
        "performance":"not measured"}))


if __name__ == "__main__":
    main()
