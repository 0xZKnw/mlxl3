"""Capture official BF16 LFM2.5 logits independently of the historical MLX teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from contextlib import nullcontext
from pathlib import Path

from generate_lfm25_trace import SOURCE_HASHES

REVISION = "1769516ac588c4c558089c94f27d96b0dc67b183"
WINDOW_HASHES = {
    "window1": "bdafc8c2a7c8ef0322c186013d1b99f7bff96c45fc91f47e9125e9c98fd62061",
    "window5": "23b146a0a8b43a1b0495f9fc9bd6882e7ed628be7cd7c7aa4751082dade7f7a4",
}
SOURCE_SHARDS = {
    "model-00001-of-00010.safetensors": "f7b4d9106ff00f1e4529c01bfa2b2e62578cf237cc8c39f792768c510e386b6c",
    "model-00002-of-00010.safetensors": "e33aab029127a29dc3f3ec60e995c3ea37608c517d0e33f0c12bd84ea1031700",
    "model-00003-of-00010.safetensors": "68bd39f975ed9baf0d489772ac302b926f426baf7140d88007c34b64561a6809",
    "model-00004-of-00010.safetensors": "1355d0f20369f5c7697d57eb1d6aa8f954fa12c479e58a688db7817657fc89e4",
    "model-00005-of-00010.safetensors": "a4fec7fbe3c4f3f08afe357abddba837314fe6f220936c515042ea1ed7dfddcd",
    "model-00006-of-00010.safetensors": "d7f59eb54c2c6dab7a54258d1bc3665d35a304301e49cba4a492a430de183ad1",
    "model-00007-of-00010.safetensors": "13f6524d71c8c3190b9d1d083b266a7dff569566a0ab2b14e732e9df08fdb5f7",
    "model-00008-of-00010.safetensors": "51ebc311324fd321baf5d9c12ef0d6f0dc06b88b2e7b95899c63cf33179c2b35",
    "model-00009-of-00010.safetensors": "491ab1cff839626e02c5f996fe90ef044592ce14df6471249567027d4adb4dad",
    "model-00010-of-00010.safetensors": "1d3f7178c0c7d1b146bffe7c5ac41874b06b95ef04538693204502258fe2c1b4",
}


def valid_token(token: int) -> int:
    """Reject invalid imposed IDs, including bools.

    raises: ValueError
    post: 0 <= __return__ < 125017
    post: __return__ == token
    """
    if type(token) is not int or not 0 <= token < 125017:
        raise ValueError("Invalid source token ID")
    return token


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_windows(plan: dict) -> dict[str, list[int]]:
    windows = plan.get("windows") if isinstance(plan, dict) else None
    if not isinstance(windows, dict) or set(windows) != set(WINDOW_HASHES):
        raise ValueError("Expected exactly the historical windows 1 and 5")
    for name, tokens in windows.items():
        if not isinstance(tokens, list) or len(tokens) != 1024:
            raise ValueError("Expected 1024 imposed tokens per window")
        for token in tokens:
            valid_token(token)
        if hashlib.sha256(struct.pack("<1024I", *tokens)).hexdigest() != WINDOW_HASHES[name]:
            raise ValueError("Historical diagnostic tokens changed")
    return windows


def restore_biases(model, source: Path, index: dict) -> dict:
    import torch
    from safetensors import safe_open

    hashes = {}
    for name, shard in index.items():
        if not name.endswith("expert_bias"):
            continue
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            original = handle.get_tensor(name)
        if (
            original.dtype != torch.float32
            or original.shape != (64,)
            or not torch.isfinite(original).all().item()
        ):
            raise ValueError("Invalid original routing bias")
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, original.to(model.device))
        actual = model.get_buffer(name).cpu()
        if actual.dtype != torch.float32 or not torch.equal(
            actual.view(torch.uint8), original.view(torch.uint8)
        ):
            raise ValueError("Source routing bias was rounded")
        hashes[name] = hashlib.sha256(original.numpy().tobytes()).hexdigest()
    if len(hashes) != 38:
        raise ValueError("Expected exactly 38 FP32 routing biases")
    return hashes


def load_source(source: Path, report: dict):
    import importlib.metadata

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    package = Path(transformers.__file__).parent
    actual = {name: file_sha(package / name) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES or importlib.metadata.version("transformers") != "5.16.1":
        raise ValueError("Unexpected official Transformers sources/version")
    report["transformers_sources"] = actual
    report["versions"] = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")
    }
    if (
        file_sha(source / "config.json")
        != "b0c46662afe3881e49cf542e683bf5ea8a5e0b780a429b6141a48defc33531bf"
    ):
        raise ValueError("Source configuration changed")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    if set(index.values()) != set(SOURCE_SHARDS):
        raise ValueError("Unexpected original checkpoint shards")
    report["source_files"] = {
        "config.json": file_sha(source / "config.json"),
        index_path.name: file_sha(index_path),
    }
    for name, expected in SOURCE_SHARDS.items():
        actual = file_sha(source / name)
        if actual != expected:
            raise ValueError("Source weight SHA mismatch: " + name)
        report["source_files"][name] = actual
    torch.backends.cuda.matmul.allow_tf32 = False
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(source),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
        experts_implementation="grouped_mm",
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if (
        loading.get("unexpected_keys")
        or loading.get("mismatched_keys")
        or set(loading.get("missing_keys", [])) - {"lm_head.weight"}
    ):
        raise ValueError(f"Incomplete official source load: {loading}")
    if (
        not model.config.tie_word_embeddings
        or model.get_input_embeddings().weight.data_ptr()
        != model.get_output_embeddings().weight.data_ptr()
    ):
        raise ValueError("Source output head is not the tied embedding")
    model.eval()
    report["routing_bias_sha256"] = restore_biases(model, source, index)
    if model.config.vocab_size != 128000 or any(
        p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in model.parameters()
    ):
        raise ValueError("Source precision/device/vocabulary contract failed")
    report["parameter_dtype"] = "torch.bfloat16"
    report["attention_implementation"] = model.config._attn_implementation
    report["gpu"] = torch.cuda.get_device_name()
    return model


def capture_window(model, tokens: list[int], phase: str, path: Path, progress) -> dict:
    import numpy as np
    import torch

    if phase not in ("prefill", "decode") or not tokens or path.exists():
        raise ValueError("Invalid capture phase, empty sequence or existing output")
    vocabulary = model.config.vocab_size
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(tokens), vocabulary)
    )
    dtypes, cache = set(), None
    completed = 0
    context = model._optimize_model_for_decode() if phase == "decode" else nullcontext()
    with context, torch.inference_mode():
        implementation = model.get_experts_implementation()
        steps = [tokens] if phase == "prefill" else [[token] for token in tokens]
        for part in steps:
            output = model(
                input_ids=torch.tensor([part], device=model.device),
                past_key_values=cache,
                use_cache=phase == "decode",
                logits_to_keep=0,
            )
            logits = output.logits
            if (
                logits.shape != (1, len(part), vocabulary)
                or not torch.isfinite(logits).all().item()
            ):
                raise ValueError("Invalid official reference logits shape/finitude")
            dtypes.add(str(logits.dtype))
            values[completed : completed + len(part)] = logits.float().cpu().numpy()[0]
            completed += len(part)
            if phase == "decode":
                cache = output.past_key_values
                if cache is None or cache.get_seq_length() != completed:
                    raise ValueError("Missing, stale or misaligned decode cache")
            if completed % 128 == 0 or completed == len(tokens):
                values.flush()
                progress(completed)
            del logits, output
    if completed != len(tokens) or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite reference file")
    values.flush()
    del values, cache
    return {
        "shape": [len(tokens), vocabulary],
        "storage_dtype": "float32",
        "logits_dtypes": sorted(dtypes),
        "experts_implementation": implementation,
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
    }


def run(source: Path, plan: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "publication": False,
        "stage": "source",
        "windows": {},
        "source": str(source),
        "source_revision": REVISION,
        "script_sha256": file_sha(Path(__file__)),
    }
    model, original_mm = None, None

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = folder / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(folder / "report.json")

    save()
    try:
        from transformers.integrations import moe

        windows = validate_windows(plan)
        (folder / "input.json").write_text(json.dumps(plan, allow_nan=False) + "\n")
        report["input_sha256"] = file_sha(folder / "input.json")
        model = load_source(source, report)
        save()
        native_calls = 0
        original_mm = moe._grouped_mm

        def guarded_mm(x, weight, offs):
            nonlocal native_calls
            if not moe._can_use_grouped_mm(x, weight, offs):
                raise RuntimeError("Refusing grouped_mm fallback")
            native_calls += 1
            return original_mm(x, weight, offs)

        moe._grouped_mm = guarded_mm
        for phase in ("prefill", "decode"):
            for name, tokens in windows.items():
                report["stage"] = f"{phase}/{name}"
                before = native_calls

                def progress(rows, phase=phase, name=name, tokens=tokens):
                    if time.monotonic() - started > 740:
                        raise TimeoutError("Bounded official reference capture expired")
                    report["progress"] = {
                        "phase": phase,
                        "window": name,
                        "tokens": rows,
                        "total": len(tokens),
                    }
                    save()
                    print(json.dumps(report["progress"]), flush=True)

                result = capture_window(
                    model, tokens, phase, folder / f"{phase}-{name}.npy", progress
                )
                required = "grouped_mm" if phase == "prefill" else "batched_mm"
                if set(result["experts_implementation"].values()) != {required} or (
                    phase == "prefill" and native_calls <= before
                ):
                    raise ValueError("Official experts implementation was not executed")
                result["native_grouped_calls"] = native_calls - before
                result["token_sha256"] = WINDOW_HASHES[name]
                report["windows"][f"{phase}-{name}"] = result
                save()
        report.update(status="complete", stage="done")
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if original_mm is not None:
            moe._grouped_mm = original_mm
        del model
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    run(args.source, json.loads(args.plan.read_text()), args.output)


if __name__ == "__main__":
    main()
