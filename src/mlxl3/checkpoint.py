"""Load standard EXL3 safetensors into MLX-LM architectures."""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import _get_classes

from mlxl3.codec.codebook import CodebookMode
from mlxl3.linear import EXL3Linear, fuse_compatible_linear_groups
from mlxl3.moe import EXL3SwitchGLU

_SERIALIZED_SUFFIXES = (".trellis", ".suh", ".svh", ".su", ".sv", ".mul1", ".mcg")
_LING_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_LFM_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.feed_forward\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>w1|w2|w3)$"
)


@dataclass(frozen=True)
class LoadReport:
    replaced: tuple[str, ...]
    skipped: tuple[str, ...]


def quantization_config(model_path: str | Path) -> dict:
    path = Path(model_path) / "quantization_config.json"
    with path.open() as handle:
        config = json.load(handle)
    if config.get("quant_method") != "exl3":
        raise ValueError(f"expected quant_method='exl3', got {config.get('quant_method')!r}")
    return config


def list_exl3_modules(model_path: str | Path) -> list[str]:
    storage = quantization_config(model_path).get("tensor_storage", {})
    return sorted(key for key, value in storage.items() if value.get("quant_format") == "exl3")


def _load_all_safetensors(model_path: Path) -> dict[str, mx.array]:
    files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors under {model_path}")
    weights: dict[str, mx.array] = {}
    for file in files:
        loaded = mx.load(file)
        if not isinstance(loaded, dict):
            raise TypeError(f"expected named safetensors in {file}")
        weights.update(loaded)
    return weights


def _sanitized_prefix(model: nn.Module, prefix: str) -> str:
    sentinel = mx.zeros((1,), dtype=mx.uint16)
    # Architecture sanitizers normally rewrite public ``.weight`` names, not
    # EXL3's serialized suffixes.  Probe with a weight key to obtain the actual
    # MLX module path, then read trellis/scales from the untouched disk prefix.
    probe = {f"{prefix}.weight": sentinel}
    if hasattr(model, "sanitize"):
        probe = model.sanitize(probe)
    if len(probe) != 1:
        raise ValueError(f"model sanitize dropped or expanded EXL3 path {prefix!r}")
    key = next(iter(probe))
    if not key.endswith(".weight"):
        raise ValueError(f"unexpected sanitized EXL3 key {key!r}")
    return key.removesuffix(".weight")


def _scale(weights: dict[str, mx.array], prefix: str, primary: str, legacy: str) -> mx.array:
    value = weights.get(f"{prefix}.{primary}")
    if value is None:
        value = weights.get(f"{prefix}.{legacy}")
    if value is None:
        raise KeyError(f"missing {prefix}.{primary}/{legacy}")
    return value


def _mode(weights: dict[str, mx.array], prefix: str) -> CodebookMode:
    if f"{prefix}.mul1" in weights:
        return CodebookMode.MUL1
    if f"{prefix}.mcg" in weights:
        return CodebookMode.MCG
    return CodebookMode.DEFAULT


def _expert_spec(prefix: str) -> tuple[str, int, int, str, str] | None:
    match = _LING_EXPERT_RE.match(prefix)
    if match is not None:
        return (
            "ling",
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("projection"),
            f"model.layers.{match.group('layer')}.mlp.switch_mlp",
        )
    match = _LFM_EXPERT_RE.match(prefix)
    if match is None:
        return None
    projection = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[match.group("projection")]
    return (
        "lfm2_moe",
        int(match.group("layer")),
        int(match.group("expert")),
        projection,
        f"model.layers.{match.group('layer')}.feed_forward.switch_mlp",
    )


def _stack_moe_experts(
    disk_modules: list[str],
    weights: dict[str, mx.array],
) -> tuple[list[tuple[str, nn.Module]], set[str]]:
    """Build one mapped SwitchGLU per supported MoE layer."""

    grouped: dict[tuple[str, int, str], dict[int, dict[str, str]]] = {}
    for prefix in disk_modules:
        spec = _expert_spec(prefix)
        if spec is None:
            continue
        architecture, layer, expert, projection, module_path = spec
        grouped.setdefault((architecture, layer, module_path), {}).setdefault(expert, {})[
            projection
        ] = prefix

    replacements: list[tuple[str, nn.Module]] = []
    consumed: set[str] = set()
    required = {"gate_proj", "up_proj", "down_proj"}
    for (architecture, layer, module_path), experts in sorted(grouped.items()):
        expert_ids = sorted(experts)
        if expert_ids != list(range(len(expert_ids))):
            raise ValueError(f"layer {layer} has non-contiguous EXL3 expert ids")
        if any(set(experts[expert]) != required for expert in expert_ids):
            raise ValueError(f"layer {layer} is missing one or more EXL3 expert projections")

        all_prefixes = [
            experts[expert][projection]
            for expert in expert_ids
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]
        bits = {int(weights[f"{prefix}.trellis"].shape[-1] // 16) for prefix in all_prefixes}
        modes = {_mode(weights, prefix) for prefix in all_prefixes}
        if len(bits) != 1 or len(modes) != 1:
            raise ValueError(
                f"layer {layer} experts require uniform K/codebook for grouped Metal QMV"
            )

        gates = [experts[expert]["gate_proj"] for expert in expert_ids]
        ups = [experts[expert]["up_proj"] for expert in expert_ids]
        downs = [experts[expert]["down_proj"] for expert in expert_ids]
        module = EXL3SwitchGLU(
            gu_trellis=mx.concatenate(
                [weights[f"{prefix}.trellis"] for prefix in gates + ups],
                axis=1,
            ),
            gu_suh=mx.stack(
                [
                    mx.stack(
                        [
                            _scale(weights, gates[index], "suh", "su"),
                            _scale(weights, ups[index], "suh", "su"),
                        ]
                    )
                    for index in range(len(expert_ids))
                ]
            ),
            gu_svh=mx.stack(
                [
                    mx.stack(
                        [
                            _scale(weights, gates[index], "svh", "sv"),
                            _scale(weights, ups[index], "svh", "sv"),
                        ]
                    )
                    for index in range(len(expert_ids))
                ]
            ),
            down_trellis=mx.concatenate(
                [weights[f"{prefix}.trellis"] for prefix in downs],
                axis=1,
            ),
            down_suh=mx.stack([_scale(weights, prefix, "suh", "su") for prefix in downs]),
            down_svh=mx.stack([_scale(weights, prefix, "svh", "sv") for prefix in downs]),
            bits=bits.pop(),
            mode=modes.pop(),
        )
        replacements.append((module_path, module))
        consumed.update(all_prefixes)
    return replacements, consumed


def _build_model_with_output_head(
    model_class: type[nn.Module],
    args: object,
    config: dict,
    disk_modules: list[str],
) -> nn.Module:
    """Expose an untied EXL3 head when an MLX architecture hardcodes tying."""

    if "lm_head" not in disk_modules or config.get("model_type") != "lfm2_moe":
        return model_class(args)

    class LFM2MoeWithOutputHead(model_class):
        def __init__(self, model_args: object):
            super().__init__(model_args)
            self.lm_head = nn.Linear(
                model_args.hidden_size,
                model_args.vocab_size,
                bias=False,
            )

        def __call__(self, inputs, cache=None, input_embeddings=None):
            hidden = self.model(inputs, cache, input_embeddings)
            return self.lm_head(hidden)

    return LFM2MoeWithOutputHead(args)


def load_exl3_model(
    model_path: str | Path,
    *,
    lazy: bool = True,
    strict_modules: bool = True,
) -> tuple[nn.Module, dict, LoadReport]:
    """Build an MLX-LM model and replace its serialized EXL3 projections."""

    root = Path(model_path)
    with (root / "config.json").open() as handle:
        config = json.load(handle)
    disk_modules = list_exl3_modules(root)
    if "lm_head" in disk_modules:
        # EXL3 checkpoints may carry a separately quantized high-bit head even
        # when the source model tied it to the embeddings. Expose the slot so
        # the serialized checkpoint, not the stale source flag, wins.
        if "text_config" in config:
            config["text_config"] = dict(config["text_config"])
            config["text_config"]["tie_word_embeddings"] = False
        else:
            config["tie_word_embeddings"] = False
    model_class, args_class = _get_classes(config)
    args = args_class.from_dict(config)
    model = _build_model_with_output_head(model_class, args, config, disk_modules)

    raw_weights = _load_all_safetensors(root)
    weights = dict(raw_weights)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    leaves = dict(
        tree_flatten(
            model.leaf_modules(),
            is_leaf=lambda value: isinstance(value, nn.Module),
        )
    )
    replacements, grouped_experts = _stack_moe_experts(disk_modules, raw_weights)
    replaced: list[str] = []
    skipped: list[str] = []
    for disk_prefix in disk_modules:
        if disk_prefix in grouped_experts:
            replaced.append(disk_prefix)
            continue
        prefix = _sanitized_prefix(model, disk_prefix)
        if prefix not in leaves:
            skipped.append(prefix)
            continue
        trellis = raw_weights[f"{disk_prefix}.trellis"]
        bits = trellis.shape[-1] // 16
        module = EXL3Linear(
            trellis,
            _scale(raw_weights, disk_prefix, "suh", "su"),
            _scale(raw_weights, disk_prefix, "svh", "sv"),
            bits=bits,
            mode=_mode(raw_weights, disk_prefix),
            bias=raw_weights.get(f"{disk_prefix}.bias"),
        )
        replacements.append((prefix, module))
        replaced.append(prefix)

    if skipped and strict_modules:
        preview = ", ".join(skipped[:5])
        raise ValueError(f"{len(skipped)} EXL3 module paths are absent from the model: {preview}")
    if replacements:
        model.update_modules(tree_unflatten(replacements))

    ordinary_weights = []
    for key, value in weights.items():
        if key.endswith(_SERIALIZED_SUFFIXES):
            continue
        if mx.issubdtype(value.dtype, mx.floating):
            value = value.astype(mx.float32 if key.endswith("A_log") else mx.float16)
        ordinary_weights.append((key, value))
    model.load_weights(ordinary_weights, strict=False)
    fuse_compatible_linear_groups(model)
    model.eval()
    if not lazy:
        mx.eval(model.parameters())
    return model, config, LoadReport(tuple(replaced), tuple(skipped))
