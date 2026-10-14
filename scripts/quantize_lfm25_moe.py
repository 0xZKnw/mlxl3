"""Layerwise BF16 -> EXL3 conversion for native LFM2 MoE on small Apple GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references/PonyExl3"))


def expected_projections(config: dict) -> set[str]:
    if config.get("model_type") != "lfm2_moe" or not config.get("tie_word_embeddings"):
        raise ValueError("Expected native lfm2_moe with tied embeddings")
    kinds = config["layer_types"]
    layers, dense, experts = (
        config[key] for key in ("num_hidden_layers", "num_dense_layers", "num_experts")
    )
    if len(kinds) != layers or layers < 1 or not 0 <= dense <= layers or experts < 1:
        raise ValueError("Invalid LFM layer/expert configuration")
    keys = set()
    for number, kind in enumerate(kinds):
        prefix = f"model.layers.{number}."
        if kind == "full_attention":
            keys.update(
                prefix + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj", "out_proj")
            )
        elif kind == "conv":
            keys.update(prefix + "conv." + name for name in ("in_proj", "out_proj"))
        else:
            raise ValueError(f"Unsupported LFM layer type: {kind}")
        mlps = (
            [prefix + "feed_forward"]
            if number < dense
            else [prefix + f"feed_forward.experts.{expert}" for expert in range(experts)]
        )
        keys.update(mlp + "." + name for mlp in mlps for name in ("w1", "w2", "w3"))
    return keys


def capture_lfm_moe(
    source_dir,
    module_keys,
    output,
    *,
    text_path,
    rows=2048,
    seq_len=1024,
    max_seqs=None,
    dtype="float16",
    progress=None,
    hidden_output=None,
):
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.lfm2_moe import Model, ModelArgs
    from ponyexl3.convert import cancel
    from ponyexl3.convert.capture import CalibrationCaptureSummary, _ActivationCollector
    from ponyexl3.convert.fixtures import SafetensorIndex
    from quantize_ling import capture_experts, save_rows
    from transformers import PreTrainedTokenizerFast

    source_dir, folder = Path(source_dir), Path(output)
    config = json.loads((source_dir / "config.json").read_text())
    selected = set(module_keys)
    if not selected or not selected <= expected_projections(config):
        raise ValueError("Empty or unsupported LFM calibration selection")
    if rows < 1 or seq_len < 1 or dtype != "float16" or (max_seqs is not None and max_seqs < 1):
        raise ValueError("LFM calibration needs positive rows/sequences and float16 capture")
    count = max_seqs or (rows + seq_len - 1) // seq_len
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source_dir)
    tokens = tokenizer.encode(Path(text_path).read_text())
    if count * seq_len < rows or len(tokens) < count * seq_len:
        raise ValueError("Calibration corpus/sequences must provide the requested real rows")
    index = SafetensorIndex(source_dir)
    signature = {
        "modules": sorted(selected),
        "rows": rows,
        "seq_len": seq_len,
        "max_seqs": max_seqs,
        "source": str(source_dir.resolve()),
        "config": hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest(),
        "index": hashlib.sha256(
            (source_dir / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "shards": {
            name: [(source_dir / name).stat().st_size, (source_dir / name).stat().st_mtime_ns]
            for name in sorted(set(index.weight_map.values()))
        },
        "text": hashlib.sha256(Path(text_path).read_bytes()).hexdigest(),
    }
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.json"
    if (
        hidden_output is None
        and complete.is_file()
        and json.loads(complete.read_text()) == signature
        and all((folder / (key + ".npy")).is_file() for key in selected)
    ):
        return CalibrationCaptureSummary(
            str(folder),
            str(source_dir),
            len(selected),
            len(selected),
            rows,
            seq_len,
            count,
            dtype,
            [],
            [],
        )
    complete.unlink(missing_ok=True)
    model = Model(ModelArgs.from_dict(config))
    model.eval()

    def load_keys(keys):
        return model.sanitize(
            {
                key: mx.array(index.read_tensor(key)).astype(
                    mx.float32 if index.tensor_info(key).dtype == "F32" else mx.bfloat16
                )
                for key in keys
            }
        )

    embedding = load_keys(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    hidden = embedding[mx.array(tokens[: count * seq_len]).reshape(count, seq_len)]
    mx.eval(hidden)
    del embedding
    model.model.embed_tokens = None
    original_call = nn.Linear.__call__
    try:
        for number, layer in enumerate(model.layers):
            cancel.raise_if_requested()
            prefix = f"model.layers.{number}."
            weights = load_keys([key for key in index.weight_map if key.startswith(prefix)])
            layer.load_weights(
                [(key.removeprefix(prefix), value) for key, value in weights.items()]
            )
            mx.eval(layer.parameters())
            del weights
            keys = [key for key in selected if key.startswith(prefix) and ".experts." not in key]
            collector = _ActivationCollector(keys, rows=rows, dtype=dtype)
            paths = dict(layer.named_modules())
            by_id = {
                id(
                    paths[
                        re.sub(
                            r"\.w([123])$",
                            lambda m: (
                                "." + {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[m[1]]
                            ),
                            key.removeprefix(prefix),
                        )
                    ]
                ): key
                for key in keys
            }

            def record_linear(mod, x, by_id=by_id, collector=collector):
                if id(mod) in by_id:
                    collector.add(by_id[id(mod)], x)
                return original_call(mod, x)

            nn.Linear.__call__ = record_linear
            moe_inputs, next_hidden = [], []
            experts_selected = any(
                key.startswith(prefix + "feed_forward.experts.") for key in selected
            )
            for sequence in hidden:
                cache = KVCache() if layer.is_attention_layer else ArraysCache(size=1)
                chunks = []
                for start in range(0, seq_len, 128):
                    cancel.raise_if_requested()
                    x = sequence[None, start : start + 128]
                    mask = (
                        create_attention_mask(x, cache, return_array=True)
                        if layer.is_attention_layer
                        else None
                    )
                    branch = layer.self_attn if layer.is_attention_layer else layer.conv
                    h = x + branch(layer.operator_norm(x), mask=mask, cache=cache)
                    inputs = layer.ffn_norm(h)
                    y = h + layer.feed_forward(inputs)
                    collector.flush(y)
                    if experts_selected:
                        moe_inputs.append(np.array(inputs[0].astype(mx.float16)))
                    chunks.append(y[0])
                next_hidden.append(mx.concatenate(chunks))
            nn.Linear.__call__ = original_call
            hidden = mx.stack(next_hidden)
            mx.eval(hidden)
            captured, missing, _ = collector.arrays()
            if missing:
                raise ValueError(f"Missing LFM calibration: {missing}")
            for key, value in captured.items():
                save_rows(folder, key, value)
            if moe_inputs:
                capture_experts(
                    folder,
                    prefix + "feed_forward",
                    layer.feed_forward.switch_mlp,
                    mx.array(np.concatenate(moe_inputs)[:rows]).astype(mx.bfloat16),
                    selected,
                    names=("w1", "w3", "w2"),
                )
            model.layers[number] = None
            del (
                layer,
                collector,
                paths,
                captured,
                next_hidden,
                chunks,
                cache,
                moe_inputs,
                inputs,
                y,
                h,
                branch,
                by_id,
                record_linear,
            )
            mx.clear_cache()
            print(
                f"[lfm:calib] layer {number + 1}/{config['num_hidden_layers']} complete", flush=True
            )
        if hidden_output is not None:
            norm = load_keys(["model.embedding_norm.weight"])["model.embedding_norm.weight"]
            normalized = mx.fast.rms_norm(hidden, norm, config["norm_eps"])
            np.save(hidden_output, np.array(normalized.astype(mx.float32)))
    finally:
        nn.Linear.__call__ = original_call
    missing = sorted(key for key in selected if not (folder / (key + ".npy")).is_file())
    if missing:
        raise ValueError(f"Incomplete LFM calibration: {missing[:8]}")
    complete.write_text(json.dumps(signature))
    return CalibrationCaptureSummary(
        str(folder),
        str(source_dir),
        len(selected),
        len(selected),
        rows,
        seq_len,
        count,
        dtype,
        [],
        [],
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--module-limit", type=int)
    args, _ = parser.parse_known_args()
    config = json.loads((args.in_dir / "config.json").read_text())
    expected = expected_projections(config)
    from ponyexl3.cli import convert
    from ponyexl3.convert import capture, direct, driver, metal_search, mlx_trellis
    from ponyexl3.convert.discovery import discover_exl3_module_keys
    from ponyexl3.convert.fixtures import SafetensorIndex
    from quantize_lfm26 import patch_converter
    from quantize_ling import ActivationFiles, measure_in_groups

    from mlxl3_quantizer import (
        list_exl3_modules,
        pack_trellis_mlx,
        quantize_tiles_mlx,
        validate_checkpoint_files,
    )

    patch_converter()
    keys = set(discover_exl3_module_keys(args.in_dir, include_routed_experts=True))
    if keys != expected:
        raise ValueError(
            f"Incomplete LFM plan: missing={sorted(expected - keys)[:8]}, extra={sorted(keys - expected)[:8]}"
        )
    # The 0.908 scale belongs to Ling's measured checkpoint/recipe only.
    os.environ.pop("PONYEXL3_FIXED_EXPERT_SCALE", None)
    if "--include-routed-experts" not in sys.argv:
        sys.argv.append("--include-routed-experts")
    capture.capture_calibration_activations = convert.capture_calibration_activations = (
        capture_lfm_moe
    )
    metal_search.quantize_tiles_mlx = quantize_tiles_mlx
    mlx_trellis.pack_trellis_mlx = pack_trellis_mlx
    original_plain = direct.read_source_plain_tensors

    def read_plain(source, keys):
        tensors = original_plain(source, keys)
        index = SafetensorIndex(source)
        for key in keys:
            if index.tensor_info(key).dtype == "F32":
                tensors[key] = index.read_tensor(key)
        return tensors

    direct.read_source_plain_tensors = driver.read_source_plain_tensors = read_plain
    original_load, original_measure = (
        convert.load_calibration_activations_map,
        convert.measure_ldlq_candidates,
    )
    convert.load_calibration_activations_map = lambda path: (
        ActivationFiles(path) if Path(path).is_dir() else original_load(path)
    )
    convert.measure_ldlq_candidates = lambda *a, **kw: measure_in_groups(original_measure, *a, **kw)
    result = convert.main()
    if result == 0:
        validate_checkpoint_files(args.out_dir)
        converted = set(list_exl3_modules(args.out_dir))
        metadata = json.loads((args.out_dir / "quantization_config.json").read_text())
        output_keys = json.loads((args.out_dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for key in converted:
            for mode in ("mcg", "mul1"):
                if (
                    metadata["tensor_storage"][key].get(mode + "_multiplier")
                    and key + "." + mode not in output_keys
                ):
                    raise ValueError(f"Missing runtime codebook marker: {key}.{mode}")
        if not args.module_limit and converted != expected:
            raise ValueError("Incomplete LFM EXL3 projection coverage")
        if not args.module_limit:
            missing = [
                key
                for key in SafetensorIndex(args.in_dir).weight_map
                if key not in output_keys and key.removesuffix(".weight") not in converted
            ]
            if missing:
                raise ValueError(f"Missing unquantized LFM source tensors: {missing[:8]}")
        if (args.in_dir / "LICENSE").is_file():
            shutil.copy2(args.in_dir / "LICENSE", args.out_dir / "LICENSE")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
