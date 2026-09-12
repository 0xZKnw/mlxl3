"""Strict forced-token comparison of the native LFM2 port and Python production.

Run only on a physical Apple GPU, with the same libmlx and checkpoint. Compares
every logit and every live recurrent/KV value, not just greedy text. Token-wise
prefill on both sides isolates the architecture port; no speed claim is made.
"""
from __future__ import annotations
import argparse
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
    model, _, _ = load_exl3_model(args.model, lazy=False)
    cache = model.make_cache()
    compared = 0
    with tempfile.TemporaryFile(mode="w+") as errors:
        process = subprocess.Popen([str(args.binary.resolve()), "forward", str(args.model),
            "--tokens", args.tokens, "--states"], stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for step, token in enumerate(tokens):
                expected = model(mx.array([[token]], dtype=mx.int32), cache=cache)
                mx.eval(expected)
                line = process.stdout.readline()
                if not line:
                    errors.seek(0)
                    raise AssertionError(f"native engine exited early: {errors.read()}")
                actual = json.loads(line)
                assert actual["token"] == token
                states = {}
                for index, layer in enumerate(cache):
                    if hasattr(layer, "keys"):
                        keys, values = layer.state
                        states[f"model.layers.{index}.keys"] = keys
                        states[f"model.layers.{index}.values"] = values
                    else:
                        states[f"model.layers.{index}.conv_state"] = layer[0]
                assert {state["name"] for state in actual["states"]} == set(states)
                # Check states first so a divergent layer is identified before
                # the inevitable final-logit mismatch.
                for state in sorted(actual["states"], key=lambda s: int(s["name"].split(".")[2])):
                    reference = np.asarray(states[state["name"]])
                    assert state["shape"] == list(reference.shape), state["name"]
                    np.testing.assert_array_equal(state["data"], reference.view(np.uint8).ravel(),
                        err_msg=f"step {step}, cache {state['name']}")
                logits = np.asarray(expected)
                assert logits.dtype == np.float16, f"unexpected logits dtype {logits.dtype}"
                np.testing.assert_array_equal(actual["logits"], logits.view(np.uint16).ravel(),
                    err_msg=f"step {step}, complete logits")
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
        "comparison":"all logits and caches bit-for-bit", "performance":"not measured"}))


if __name__ == "__main__":
    main()
