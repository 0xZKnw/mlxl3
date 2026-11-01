"""Layerwise BF16 -> EXL3 conversion for native LFM2 MoE on small Apple GPUs.

--routed-calibration records each expert's actual SwitchLinear inputs. Increase
--calibration-max-seqs to cover cold experts; --calibration-rows caps each matrix.
The default preserves the all-expert v1 recipe pending full-model v2 evaluation.
"""

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


def input_row_stride(input_rows: int, route_slots: int) -> int:
    if input_rows < 1 or route_slots < 1 or route_slots % input_rows:
        raise ValueError("Cannot align SwitchLinear inputs with expert indices")
    return route_slots // input_rows


def calibration_source_identity(source_dir, index, text_path):
    return {
        "source": str(source_dir.resolve()),
        "config": hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest(),
        "index": hashlib.sha256(
            (source_dir / "model.safetensors.index.json").read_bytes()
            if (source_dir / "model.safetensors.index.json").is_file()
            else json.dumps(index.weight_map, sort_keys=True).encode()
        ).hexdigest(),
        "shards": {
            name: [(source_dir / name).stat().st_size, (source_dir / name).stat().st_mtime_ns]
            for name in sorted(set(index.weight_map.values()))
        },
        "text": hashlib.sha256(Path(text_path).read_bytes()).hexdigest(),
    }


def check_calibration_recipe(work_dir: Path, out_dir: Path, recipe: dict) -> Path:
    marker = work_dir / "lfm_calibration_recipe.json"
    if marker.exists():
        if json.loads(marker.read_text()) != recipe:
            raise ValueError("LFM calibration recipe changed; choose new --work-dir/--out-dir")
    elif recipe["expert_calibration"] == "routed-switch-v1" and any(
        path.exists() and any(path.iterdir()) for path in (work_dir, out_dir)
    ):
        raise ValueError("Routed calibration requires new --work-dir/--out-dir")
    return marker


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
    routed=False,
):
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.lfm2_moe import Model, ModelArgs
    from mlx_lm.models.switch_layers import SwitchLinear
    from ponyexl3.cli.convert import _write_json_atomic
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
        **calibration_source_identity(source_dir, index, text_path),
    }
    # Below 192 rows Pony changes the activation Hadamard backend.
    expert_min_rows = min(rows, 256)
    if routed:
        signature.update(expert_calibration="routed-switch-v1", expert_min_rows=expert_min_rows)
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.json"
    coverage_path = folder / "coverage.json"
    if (
        hidden_output is None
        and complete.is_file()
        and json.loads(complete.read_text()) == signature
        and all((folder / (key + ".npy")).is_file() for key in selected)
    ):
        if routed:
            try:
                saved = json.loads(coverage_path.read_text())
                if (
                    not isinstance(saved, dict)
                    or saved.get("status") != "complete"
                    or saved.get("signature") != signature
                ):
                    raise ValueError("Invalid resumed LFM coverage")
                records = saved.get("modules")
                if not isinstance(records, dict) or set(records) != selected:
                    raise ValueError("Incomplete resumed LFM coverage")
                for key in selected:
                    values = np.load(folder / (key + ".npy"), allow_pickle=False, mmap_mode="r")
                    if not isinstance(values, np.ndarray):
                        values.close()
                        raise TypeError(f"Expected an array in LFM calibration: {key}")
                    minimum = expert_min_rows if ".experts." in key else rows
                    width = index.tensor_info(key + ".weight").shape[1]
                    if (
                        values.ndim != 2
                        or values.shape[1] != width
                        or not minimum <= values.shape[0] <= rows
                        or values.dtype != np.float16
                        or not np.isfinite(values).all()
                        or not isinstance(records[key], dict)
                        or records[key].get("shape") != list(values.shape)
                        or records[key].get("sha256")
                        != hashlib.sha256(values.tobytes()).hexdigest()
                    ):
                        raise ValueError(f"Invalid resumed LFM calibration: {key}")
            except (OSError, ValueError, TypeError, EOFError) as error:
                complete.unlink(missing_ok=True)
                _write_json_atomic(
                    coverage_path,
                    {
                        "status": "failed",
                        "signature": signature,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                raise
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
    coverage = {
        "status": "running",
        "signature": signature,
        "tokens": count * seq_len,
        "expert_min_rows": expert_min_rows,
        "row_cap": rows,
        "modules": {},
    }
    if routed:
        _write_json_atomic(coverage_path, coverage)
    original_call = nn.Linear.__call__
    original_switch_call = SwitchLinear.__call__

    def load_keys(keys):
        return model.sanitize(
            {
                key: mx.array(index.read_tensor(key)).astype(
                    mx.float32 if index.tensor_info(key).dtype == "F32" else mx.bfloat16
                )
                for key in keys
            }
        )

    try:
        model = Model(ModelArgs.from_dict(config))
        model.eval()
        embedding = load_keys(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
        hidden = embedding[mx.array(tokens[: count * seq_len]).reshape(count, seq_len)]
        mx.eval(hidden)
        del embedding
        model.model.embed_tokens = None
        for number, layer in enumerate(model.layers):
            cancel.raise_if_requested()
            prefix = f"model.layers.{number}."
            weights = load_keys([key for key in index.weight_map if key.startswith(prefix)])
            layer.load_weights(
                [(key.removeprefix(prefix), value) for key, value in weights.items()]
            )
            mx.eval(layer.parameters())
            del weights
            keys = sorted(
                key
                for key in selected
                if key.startswith(prefix) and (routed or ".experts." not in key)
            )
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
                if ".experts." not in key
            }
            observed = dict.fromkeys(keys, 0)

            def record_linear(mod, x, by_id=by_id, collector=collector, observed=observed):
                if id(mod) in by_id:
                    observed[by_id[id(mod)]] += x.size // x.shape[-1]
                    collector.add(by_id[id(mod)], x)
                return original_call(mod, x)

            nn.Linear.__call__ = record_linear
            switch_keys = {}
            if routed:
                for key in keys:
                    if ".experts." in key:
                        expert, projection = key.rsplit(".", 2)[-2:]
                        name = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[projection]
                        mod = paths["feed_forward.switch_mlp." + name]
                        switch_keys.setdefault(id(mod), []).append((key, int(expert)))

                def record_switch(
                    mod,
                    x,
                    indices,
                    sorted_indices=False,
                    switch_keys=switch_keys,
                    collector=collector,
                    observed=observed,
                ):
                    logical = switch_keys.get(id(mod), [])
                    if logical:
                        flat = x.reshape(-1, x.shape[-1])
                        ids = np.asarray(indices).reshape(-1)
                        stride = input_row_stride(flat.shape[0], ids.size)
                        if (
                            ids.dtype.kind not in "iu"
                            or np.any(ids < 0)
                            or np.any(ids >= mod.num_experts)
                        ):
                            raise ValueError("Invalid routed expert indices")
                        for key, expert in logical:
                            positions = np.flatnonzero(ids == expert)
                            observed[key] += positions.size
                            need = rows - collector.counts[key]
                            if need > 0 and positions.size:
                                collector.add(key, flat[mx.array(positions[:need] // stride)])
                    return original_switch_call(mod, x, indices, sorted_indices=sorted_indices)

                SwitchLinear.__call__ = record_switch
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
                    if experts_selected and not routed:
                        moe_inputs.append(np.array(inputs[0].astype(mx.float16)))
                    chunks.append(y[0])
                next_hidden.append(mx.concatenate(chunks))
            nn.Linear.__call__ = original_call
            SwitchLinear.__call__ = original_switch_call
            hidden = mx.stack(next_hidden)
            mx.eval(hidden)
            if routed:
                captured, missing = {}, []
                for key in keys:
                    minimum = expert_min_rows if ".experts." in key else rows
                    stored = collector.counts[key]
                    coverage["modules"][key] = {
                        "observed_rows": int(observed[key]),
                        "stored_rows": stored,
                        "required_rows": minimum,
                    }
                    if stored < minimum:
                        missing.append(key)
                    else:
                        captured[key] = np.concatenate(collector.chunks[key])
                _write_json_atomic(coverage_path, coverage)
            else:
                captured, missing, _ = collector.arrays()
            if missing:
                raise ValueError(
                    f"Insufficient LFM calibration: {missing[:8]}; "
                    "increase --calibration-max-seqs or supply a richer --calibration-text"
                )
            for key, value in captured.items():
                if routed:
                    coverage["modules"][key].update(
                        shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest()
                    )
                gate_key = key.removesuffix("w3") + "w1"
                if routed and ".experts." in key and key.endswith("w3") and gate_key in captured:
                    if not np.array_equal(value, captured[gate_key]):
                        raise ValueError(f"Gate/up calibration mismatch: {key}")
                    target = folder / (key + ".npy")
                    target.unlink(missing_ok=True)
                    os.link(folder / (gate_key + ".npy"), target)
                else:
                    if routed:
                        # A legacy folder hard-links inputs across different experts.
                        (folder / (key + ".npy")).unlink(missing_ok=True)
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
                observed,
                switch_keys,
            )
            mx.clear_cache()
            print(
                f"[lfm:calib] layer {number + 1}/{config['num_hidden_layers']} complete", flush=True
            )
        if hidden_output is not None:
            norm = load_keys(["model.embedding_norm.weight"])["model.embedding_norm.weight"]
            normalized = mx.fast.rms_norm(hidden, norm, config["norm_eps"])
            np.save(hidden_output, np.array(normalized.astype(mx.float32)))
        missing = sorted(key for key in selected if not (folder / (key + ".npy")).is_file())
        if missing:
            raise ValueError(f"Incomplete LFM calibration: {missing[:8]}")
        _write_json_atomic(complete, signature)
        if routed:
            coverage["status"] = "complete"
            _write_json_atomic(coverage_path, coverage)
    except BaseException as error:
        complete.unlink(missing_ok=True)
        if routed:
            coverage["status"] = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, cancel.ConversionCancelled))
                else "failed"
            )
            coverage["error"] = f"{type(error).__name__}: {error}"
            _write_json_atomic(coverage_path, coverage)
        raise
    finally:
        nn.Linear.__call__ = original_call
        SwitchLinear.__call__ = original_switch_call
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
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--calibration-text", type=Path)
    parser.add_argument("--routed-calibration", action="store_true")
    args, _ = parser.parse_known_args()
    config = json.loads((args.in_dir / "config.json").read_text())
    from mlxl3_quantizer.options import excluded_projection

    expected = {key for key in expected_projections(config) if not excluded_projection(key)}
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
    work_dir = args.work_dir or convert._default_work_dir(args.out_dir)
    recipe = {
        "expert_calibration": "routed-switch-v1" if args.routed_calibration else "all-experts-v1",
        **calibration_source_identity(
            args.in_dir,
            SafetensorIndex(args.in_dir),
            args.calibration_text or capture.default_calibration_text(),
        ),
    }
    marker = check_calibration_recipe(work_dir, args.out_dir, recipe)
    if args.routed_calibration:
        sys.argv.remove("--routed-calibration")
    # The 0.908 scale belongs to Ling's measured checkpoint/recipe only.
    os.environ.pop("PONYEXL3_FIXED_EXPERT_SCALE", None)
    if "--include-routed-experts" not in sys.argv:
        sys.argv.append("--include-routed-experts")

    def capture_with_recipe(*a, **kw):
        convert._write_json_atomic(marker, recipe)
        return capture_lfm_moe(*a, **kw, routed=args.routed_calibration)

    capture.capture_calibration_activations = convert.capture_calibration_activations = (
        capture_with_recipe
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
