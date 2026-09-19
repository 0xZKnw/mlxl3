"""Header-only validation for checkpoints emitted by the optional converter."""

from __future__ import annotations

import json
import re
from pathlib import Path

from safetensors import safe_open


def quantization_config(model_path: str | Path) -> dict:
    config = json.loads((Path(model_path) / "quantization_config.json").read_text())
    if not isinstance(config, dict) or config.get("quant_method") != "exl3":
        raise ValueError("expected an EXL3 quantization config")
    return config


def list_exl3_modules(model_path: str | Path) -> list[str]:
    storage = quantization_config(model_path).get("tensor_storage", {})
    if not isinstance(storage, dict) or not all(isinstance(v, dict) for v in storage.values()):
        raise ValueError("invalid EXL3 tensor_storage")
    modules = sorted(key for key, value in storage.items() if value.get("quant_format") == "exl3")
    if not modules:
        raise ValueError("checkpoint contains no EXL3 modules")
    return modules


def validate_checkpoint_files(root: str | Path) -> None:
    """Read headers only and reject incomplete or mixed checkpoint shards."""
    root = Path(root)
    descriptor = quantization_config(root)
    modules = list_exl3_modules(root)
    config_path = root / "config.json"
    model_type = json.loads(config_path.read_text()).get("model_type") if config_path.is_file() else None
    tensors: dict[str, tuple[int, ...]] = {}
    owners: dict[str, str] = {}
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise ValueError("checkpoint has no safetensors weights")
    for file in files:
        with safe_open(file, framework="numpy") as handle:
            for key in handle:
                if key in tensors:
                    raise ValueError(f"duplicate tensor {key}")
                tensors[key] = tuple(handle.get_slice(key).get_shape())
                owners[key] = file.name
    for metadata in descriptor["tensor_storage"].values():
        for key, spec in metadata.get("stored_tensors", {}).items():
            shape = tensors.get(key)
            if (
                shape is None
                and model_type == "lfm2_moe"
                and re.fullmatch(r"model\.layers\.\d+\.feed_forward\.gate\.expert_bias", key)
            ):
                shape = tensors.get(key.replace(".gate.expert_bias", ".expert_bias"))
            if tuple(spec["shape"]) != shape:
                raise ValueError(f"missing or invalid checkpoint tensor: {key}")
    for prefix in modules:
        shape = tensors.get(prefix + ".trellis", ())
        if len(shape) != 3 or shape[-1] not in range(16, 129, 16) or min(shape) <= 0:
            raise ValueError(f"invalid or missing EXL3 trellis: {prefix}")
        for primary, legacy, length in (
            ("suh", "su", shape[0] * 16),
            ("svh", "sv", shape[1] * 16),
        ):
            if tensors.get(prefix + "." + primary, tensors.get(prefix + "." + legacy)) != (length,):
                raise ValueError(f"invalid or missing EXL3 scale: {prefix}.{primary}")
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text()).get("weight_map", {})
        for key, filename in mapping.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or owners.get(key) != filename
            ):
                raise ValueError(f"incomplete checkpoint shard: {filename}")
