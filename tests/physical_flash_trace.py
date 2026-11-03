"""Check the user-visible Flash forward state wire format against a separate stock CLI."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from physical_flash_run import digest, execute


def state_digest(data: bytes, count: int) -> str:
    """Hash exactly count finite, little-endian FP32 words.

    raises: ValueError
    post: len(_) == 64
    """
    if count <= 0 or len(data) != count * 4 or not np.isfinite(np.frombuffer(data, "<f4")).all():
        raise ValueError("invalid FP32 state length or value")
    return hashlib.sha256(data).hexdigest()


def observe(cli, direct, model, layers, batch):
    args = ["forward", model, "--tokens", "248044", "--states"]
    if batch:
        args.append("--batch")
    output, _, images = execute(cli, args, direct)
    row = json.loads(output)
    del output
    assert row["token"] == 248044
    logits = row["logits"]
    assert len(logits) == 248320 and all(type(x) is int and 0 <= x <= 65535 for x in logits)
    bits = np.asarray(logits, dtype="<u2")
    assert np.isfinite(bits.view("<f2")).all()
    expected = []
    if not batch:
        for i, attention in enumerate(layers):
            expected.append((f"layer_{i}", [1, 10240]))
            if attention == "linear_attention":
                expected.append((f"gdn_state_{i}", [1, 48, 128, 128]))
        assert len(expected) == 84
    tensors = []
    for state, (name, shape) in zip(row["states"], expected, strict=True):
        assert (state["name"], state["shape"]) == (name, shape)
        digest_ = state_digest(bytes(state.pop("data")), math.prod(shape))
        tensors.append({"name": name, "shape": shape, "sha256": digest_})
    return {"logits_sha256": hashlib.sha256(bits.tobytes()).hexdigest(), "tensors": tensors}, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["cli", "stock", "model", "report"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    layers = json.loads((args.model / "config.json").read_text())["text_config"]["layer_types"]
    assert len(layers) == 48 and set(layers) == {"linear_attention", "full_attention"}
    report = {
        "status": "incomplete",
        "cli_sha256": digest(args.cli),
        "stock_sha256": digest(args.stock),
    }
    with args.report.open("x") as sink:
        try:
            for batch, key in [(False, "step"), (True, "batch")]:
                stock, _ = observe(args.stock, False, args.model, layers, batch)
                native, images = observe(args.cli, True, args.model, layers, batch)
                report[key] = {"oracle": stock, "direct": native, "direct_images": images}
                if stock != native:
                    raise ValueError(f"{key} state or logits differ")
            assert (
                report["step"]["direct"]["logits_sha256"]
                == report["batch"]["direct"]["logits_sha256"]
            )
            report["status"] = "verified"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            json.dump(report, sink, indent=2)
            sink.write("\n")


if __name__ == "__main__":
    main()
