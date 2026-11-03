"""Recorded lossless head operation; native logits must be checked separately afterward."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def require_alias(head, embedding):
    import torch

    if (
        head.shape != embedding.shape
        or not head.numel()
        or not head.is_floating_point()
        or not embedding.is_floating_point()
    ):
        raise ValueError("Invalid tied weight shapes or dtypes")
    a, b = head.half(), embedding.half()
    if (
        not torch.isfinite(head).all().item()
        or not torch.isfinite(embedding).all().item()
        or not torch.isfinite(a).all().item()
        or not torch.isfinite(b).all().item()
        or not torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    ):
        raise ValueError("Head is not a finite bitexact F16 embedding alias")


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def descriptor_entries(
    storage: dict[str, dict[str, dict[str, dict]]], mapping: dict[str, str]
) -> dict[str, dict]:
    """Validate described tensors while retaining the format's raw conv/router tensors.

    raises: ValueError
    post: all(key in mapping for key in __return__)
    post: len(__return__) <= len(mapping)
    """
    entries = {}
    for item in storage.values():
        if "stored_tensors" not in item:
            raise ValueError("Missing stored tensor metadata")
        for key, info in item["stored_tensors"].items():
            if key in entries or key not in mapping:
                raise ValueError("Duplicate or orphan descriptor tensor")
            entries[key] = info
    if any(
        key not in entries and not key.endswith((".conv.conv.weight", ".feed_forward.expert_bias"))
        for key in mapping
    ):
        raise ValueError("Unexpected undescribed checkpoint tensor")
    return entries


def compact(source, output, report_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if output.exists() or report_path.exists():
        raise FileExistsError("Preserve existing model and report attempts")
    report = {"status": "running", "parity": False, "source": str(source), "output": str(output)}
    started = time.monotonic()
    try:
        config = json.loads((source / "config.json").read_text())
        index = json.loads((source / "model.safetensors.index.json").read_text())
        descriptor = json.loads((source / "quantization_config.json").read_text())
        if (
            config.get("model_type") != "lfm2_moe"
            or not config.get("tie_word_embeddings")
            or (config["vocab_size"], config["hidden_size"]) != (128000, 2048)
            or descriptor.get("quant_method") != "exl3"
        ):
            raise ValueError("Expected the registered tied LFM2.5-24B checkpoint")
        mapping, storage = index["weight_map"], descriptor["tensor_storage"]
        original_entries = descriptor_entries(storage, mapping)
        passthrough = set(mapping) - set(original_entries)
        head_key, embedding_key = "lm_head.weight", "model.embed_tokens.weight"
        if set(storage["lm_head"]["stored_tensors"]) != {head_key}:
            raise ValueError("Unexpected extra head tensors")
        if any(Path(name).name != name for name in mapping.values()):
            raise ValueError("Non-flat checkpoint shard mapping")
        head_shard = mapping[head_key]
        with (
            safe_open(source / head_shard, framework="pt", device="cpu") as heads,
            safe_open(source / mapping[embedding_key], framework="pt", device="cpu") as embeds,
        ):
            head, embedding = heads.get_tensor(head_key), embeds.get_tensor(embedding_key)
            if head.shape != (128000, 2048) or head.dtype != torch.float16:
                raise ValueError("Unexpected complete head shape or dtype")
            for start in range(0, len(head), 1024):
                require_alias(head[start : start + 1024], embedding[start : start + 1024])
            mutant = head[:1].clone()
            mutant.view(torch.int16)[0, 0].bitwise_xor_(1)
            try:
                require_alias(mutant, embedding[:1])
            except ValueError:
                report["negative_mutation_detected"] = True
            else:
                raise ValueError("The alias guard missed a head-bit mutation")
            removed = head.numel() * head.element_size()
            report.update(head_values=head.numel(), head_weights_f16_bitexact=True)
        shutil.copytree(source, output)
        tensors = load_file(str(source / head_shard), device="cpu")
        del tensors[head_key]
        temporary = output / (head_shard + ".tmp")
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        temporary.replace(output / head_shard)
        rewritten = load_file(str(output / head_shard), device="cpu")
        if tensors.keys() != rewritten.keys() or any(
            a.shape != rewritten[name].shape
            or a.dtype != rewritten[name].dtype
            or not torch.equal(
                a.reshape(-1).view(torch.uint8), rewritten[name].reshape(-1).view(torch.uint8)
            )
            for name, a in tensors.items()
        ):
            raise ValueError("Rewriting the head shard changed other tensors")
        report["verified_unchanged_tensors_in_rewritten_shard"] = len(tensors)
        del tensors, rewritten
        del mapping[head_key]
        index["metadata"]["total_size"] -= removed
        del storage["lm_head"]
        (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        (output / "quantization_config.json").write_text(json.dumps(descriptor, indent=2) + "\n")
        expected = descriptor_entries(storage, mapping)
        if set(mapping) - set(expected) != passthrough:
            raise ValueError("Head removal changed the raw passthrough set")
        seen, total_size, finite_values, float_tensors = set(), 0, 0, 0
        report["shards"] = {}
        for name in sorted(set(mapping.values())):
            path = output / name
            digest = file_sha(path)
            if name != head_shard and digest != file_sha(source / name):
                raise ValueError("An untouched shard changed")
            with safe_open(path, framework="pt", device="cpu") as handle:
                if name == head_shard and handle.metadata() != {"format": "pt"}:
                    raise ValueError("Missing native-compatible Safetensors metadata")
                shard_keys = handle.keys()
                for key in shard_keys:
                    value = handle.get_tensor(key)
                    size = value.numel() * value.element_size()
                    if key in seen or mapping[key] != name or not value.numel():
                        raise ValueError("Invalid or duplicate tensor metadata")
                    if key in expected:
                        info = expected[key]
                        if (
                            list(value.shape) != info["shape"]
                            or size != info["n_bytes"]
                            or str(value.dtype) != info["dtype"]
                        ):
                            raise ValueError("Descriptor tensor metadata changed")
                    else:
                        with safe_open(source / name, framework="pt", device="cpu") as original:
                            old = original.get_tensor(key)
                            if (
                                value.shape != old.shape
                                or value.dtype != old.dtype
                                or not torch.equal(
                                    value.reshape(-1).view(torch.uint8),
                                    old.reshape(-1).view(torch.uint8),
                                )
                            ):
                                raise ValueError("A raw passthrough tensor changed")
                    seen.add(key)
                    total_size += size
                    if value.is_floating_point():
                        flat = value.reshape(-1)
                        for start in range(0, flat.numel(), 1048576):
                            if not torch.isfinite(flat[start : start + 1048576]).all().item():
                                raise ValueError("Nonfinite compact checkpoint tensor")
                        float_tensors += 1
                        finite_values += value.numel()
            report["shards"][name] = {"bytes": path.stat().st_size, "sha256": digest}
        if seen != set(mapping) or total_size != index["metadata"]["total_size"]:
            raise ValueError("Incomplete compact checkpoint")
        report.update(
            status="cpu-verified-native-pending",
            removed_payload_bytes=removed,
            finite_float_values=finite_values,
            float_tensors=float_tensors,
            tensor_count=len(seen),
            raw_passthrough_tensors=len(passthrough),
            weight_bytes=sum(v["bytes"] for v in report["shards"].values()),
        )
        report["effective_bpw"] = report["weight_bytes"] * 8 / 23971587712
    except BaseException as error:
        report.update(status="failed", parity=False, error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("source_directory output_directory report.json")
    compact(*(Path(arg) for arg in sys.argv[1:]))
