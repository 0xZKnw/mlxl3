"""Ling/Bailing EXL3 conversion with layerwise BF16 calibration on small Macs.

Uses common observed MoE inputs for every expert; down inputs are calculated
with that expert's own BF16 gate/up weights, never borrowed from another expert.
Calibration is disk-backed so 128 experts do not multiply resident RAM.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references/PonyExl3"))


class ActivationFiles(Mapping):
    """Bounded read cache: do not hold one mmap/file descriptor per expert."""
    def __init__(self, folder):
        self.paths = {p.stem: p for p in Path(folder).glob("*.npy")}
        self.read = lru_cache(maxsize=8)(lambda path: np.load(path, allow_pickle=False))

    def __getitem__(self, key):
        return self.read(self.paths[key])

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)

    def __contains__(self, key):
        return key in self.paths


def measure_in_groups(original, source, plan, modules, **kwargs):
    """Pony builds its pending list eagerly; bound it without changing candidates."""
    from ponyexl3.convert.measure import _measurement_summary

    records, started = [], time.perf_counter()
    checkpoint = kwargs.get("checkpoint_path")
    progress = kwargs.get("progress")
    for start in range(0, len(modules), 32):
        options = dict(kwargs)
        if checkpoint:
            path = Path(checkpoint)
            options["checkpoint_path"] = path.with_name(f"{path.stem}-{start:05d}.json")
        if progress:
            def report(event, data, offset=start):
                progress(event, {**data, "index": data.get("index", 0) + offset,
                                 "total": len(modules)})
            options["progress"] = report
        result = original(source, plan, modules[start:start + 32], **options)
        records.extend(result["records"])
    return _measurement_summary(
        modules=modules, candidate_bits=kwargs.get("candidate_bits"),
        shrinkages=kwargs.get("hessian_shrinkages", [0.0]),
        score_metric=kwargs["score_metric"], records=records,
        elapsed_s=time.perf_counter() - started,
        candidate_bits_by_module=kwargs.get("candidate_bits_by_module"))


def save_rows(folder, key, values):
    if isinstance(values, mx.array):
        values = values.astype(mx.float16)
    values = np.asarray(values, dtype=np.float16)
    if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
        raise ValueError(f"Invalid calibration rows for {key}")
    np.save(folder / (key + ".npy"), values)


def capture_experts(folder, prefix, switch, inputs, selected):
    """The gate/up share inputs; each down projection gets its own SwiGLU."""
    from mlx_lm.models.activations import swiglu

    shared_inputs = [
        f"{prefix}.experts.{expert}.{proj}"
        for expert in range(switch.gate_proj.num_experts)
        for proj in ("gate_proj", "up_proj")
        if f"{prefix}.experts.{expert}.{proj}" in selected
    ]
    if shared_inputs:
        save_rows(folder, shared_inputs[0], inputs)
        source = folder / (shared_inputs[0] + ".npy")
        for key in shared_inputs[1:]:
            target = folder / (key + ".npy")
            target.unlink(missing_ok=True)
            os.link(source, target)
    down_experts = [
        expert for expert in range(switch.gate_proj.num_experts)
        if f"{prefix}.experts.{expert}.down_proj" in selected
    ]
    for start in range(0, len(down_experts), 8):
        experts = down_experts[start:start + 8]
        gate = mx.matmul(
            inputs[None], switch.gate_proj.weight[experts].transpose(0, 2, 1)
        )
        up = mx.matmul(
            inputs[None], switch.up_proj.weight[experts].transpose(0, 2, 1)
        )
        down_inputs = swiglu(gate, up)
        mx.eval(down_inputs)
        for index, expert in enumerate(experts):
            save_rows(folder, f"{prefix}.experts.{expert}.down_proj", down_inputs[index])
        del gate, up, down_inputs
    mx.clear_cache()


def capture_ling(source_dir, module_keys, output, *, text_path, rows=2048,
                 seq_len=1024, max_seqs=None, dtype="float16", progress=None,
                 hidden_output=None):
    from mlx_lm.models.bailing_moe_v3 import Model, ModelArgs
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import ArraysCache, KVCache
    from transformers import PreTrainedTokenizerFast
    from ponyexl3.convert.capture import _ActivationCollector, CalibrationCaptureSummary
    from ponyexl3.convert.fixtures import SafetensorIndex

    if rows < 1 or seq_len < 1 or dtype != "float16":
        raise ValueError("Ling calibration needs positive rows/sequence and float16 capture")
    source_dir, folder = Path(source_dir), Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    selected = set(module_keys)
    signature = {"modules": sorted(selected), "rows": rows, "seq_len": seq_len,
                 "max_seqs": max_seqs, "source": str(source_dir.resolve()),
                 "text_sha256": hashlib.sha256(Path(text_path).read_bytes()).hexdigest()}
    complete = folder / "complete.json"
    if hidden_output is None and complete.is_file() and json.loads(complete.read_text()) == signature:
        if all((folder / (key + ".npy")).is_file() for key in selected):
            count = max_seqs or max(1, (rows + seq_len - 1) // seq_len)
            return CalibrationCaptureSummary(str(folder), str(source_dir), len(selected), len(selected),
                                             rows, seq_len, count, dtype, [], [])
    config = json.loads((source_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    model.eval()
    index = SafetensorIndex(source_dir)

    def load_keys(keys):
        weights = {}
        for key in keys:
            info = index.tensor_info(key)
            weights[key] = mx.array(index.read_tensor(key)).astype(
                mx.float32 if info.dtype == "F32" else mx.bfloat16)
        return model.sanitize(weights)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(source_dir)
    tokens = tokenizer.encode(Path(text_path).read_text())
    count = max_seqs or max(1, (rows + seq_len - 1) // seq_len)
    if count * seq_len < rows or len(tokens) < count * seq_len:
        raise ValueError("Calibration corpus/sequences must provide the requested real rows")
    embedding = load_keys(["model.word_embeddings.weight"])["model.word_embeddings.weight"]
    hidden = embedding[mx.array(tokens[:count * seq_len]).reshape(count, seq_len)]
    mx.eval(hidden)
    del embedding
    model.model.word_embeddings = None

    original_call = nn.Linear.__call__
    try:
        for number in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{number}."
            layer = model.layers[number]
            weights = load_keys([key for key in index.weight_map if key.startswith(prefix)])
            layer.load_weights([(key.removeprefix(prefix), value) for key, value in weights.items()])
            mx.eval(layer.parameters())
            del weights
            keys = [key for key in module_keys if key.startswith(prefix) and ".experts." not in key]
            collector = _ActivationCollector(keys, rows=rows, dtype=dtype)
            paths = dict(layer.named_modules())
            by_id = {id(paths[key.removeprefix(prefix)]): key for key in keys}

            def record_linear(mod, x):
                if id(mod) in by_id:
                    collector.add(by_id[id(mod)], x)
                return original_call(mod, x)

            nn.Linear.__call__ = record_linear
            # Record the actual normalized inputs at the MoE boundary.
            moe_inputs = []
            next_hidden = []
            for sequence in hidden:
                cache = ArraysCache(size=4) if layer.is_linear else KVCache()
                chunks = []
                for start in range(0, seq_len, 128):
                    x = sequence[None, start:start + 128]
                    mask = None if layer.is_linear else create_attention_mask(x, cache, return_array=True)
                    h = x + layer.attention(layer.input_layernorm(x), mask, cache)
                    mlp_input = layer.post_attention_layernorm(h)
                    y = h + layer.mlp(mlp_input)
                    collector.flush(y)
                    if any(key.startswith(prefix + "mlp.experts.") for key in selected):
                        moe_inputs.append(np.array(mlp_input[0].astype(mx.float16)))
                    chunks.append(y[0])
                next_hidden.append(mx.concatenate(chunks))
            nn.Linear.__call__ = original_call
            hidden = mx.stack(next_hidden)
            mx.eval(hidden)
            captured, missing, _ = collector.arrays()
            if missing:
                raise ValueError(f"Missing Ling calibration: {missing}")
            for key, value in captured.items():
                save_rows(folder, key, value)
            if moe_inputs:
                inputs = mx.array(np.concatenate(moe_inputs)[:rows]).astype(mx.bfloat16)
                capture_experts(folder, prefix + "mlp", layer.mlp.switch_mlp, inputs, selected)
                del inputs
            model.layers[number] = None
            del layer, collector, paths, captured, next_hidden, chunks, cache, moe_inputs
            mx.clear_cache()
            print(f"[ling:calib] layer {number + 1}/{config['num_hidden_layers']} complete", flush=True)
        if "lm_head" in selected or hidden_output is not None:
            weight = load_keys(["model.norm.weight"])["model.norm.weight"]
            normalized = mx.fast.rms_norm(hidden, weight, config["rms_norm_eps"])
            if "lm_head" in selected:
                save_rows(folder, "lm_head", normalized.reshape(-1, config["hidden_size"])[:rows])
            if hidden_output is not None:
                np.save(hidden_output, np.array(normalized.astype(mx.float32)))
    finally:
        nn.Linear.__call__ = original_call
    missing = sorted(key for key in selected if not (folder / (key + ".npy")).is_file())
    if missing:
        raise ValueError(f"Incomplete Ling calibration: {missing[:8]}")
    complete.write_text(json.dumps(signature))
    return CalibrationCaptureSummary(str(folder), str(source_dir), len(selected), len(selected),
                                     rows, seq_len, count, dtype, [], [])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--module-limit", type=int)
    parser.add_argument("--bits", type=float, required=True)
    parser.add_argument("--codebook", default="mcg")
    args, _ = parser.parse_known_args()
    config = json.loads((args.in_dir / "config.json").read_text())
    if config.get("model_type") != "bailing_hybrid":
        raise ValueError("Expected Ling/Bailing hybrid source")
    if "--include-routed-experts" not in sys.argv:
        sys.argv.append("--include-routed-experts")
    if args.bits == 4.0 and args.codebook == "mcg":
        os.environ["PONYEXL3_FIXED_EXPERT_SCALE"] = "0.908"
    else:
        os.environ.pop("PONYEXL3_FIXED_EXPERT_SCALE", None)
    # Reuse the existing marker export and exact Metal-search adapters.
    from quantize_lfm26 import patch_converter
    patch_converter()
    from ponyexl3.convert import capture, direct, driver, metal_search, reuse
    from ponyexl3.convert.fixtures import SafetensorIndex
    from mlxl3_quantizer import quantize_tiles_mlx
    capture.capture_calibration_activations = capture_ling
    original_plain = direct.read_source_plain_tensors
    def read_plain(source, keys):
        tensors = original_plain(source, keys)
        index = SafetensorIndex(source)
        for key in keys:
            if index.tensor_info(key).dtype == "F32":
                tensors[key] = index.read_tensor(key)
        return tensors
    direct.read_source_plain_tensors = driver.read_source_plain_tensors = read_plain
    metal_search.quantize_tiles_mlx = quantize_tiles_mlx
    original_enable = reuse.enable
    reuse.enable = lambda max_bytes=None: original_enable(6 << 30 if max_bytes is None else max_bytes)
    from ponyexl3.cli import convert
    original_measure = convert.measure_ldlq_candidates
    convert.measure_ldlq_candidates = lambda *a, **kw: measure_in_groups(original_measure, *a, **kw)
    original_load = convert.load_calibration_activations_map
    convert.load_calibration_activations_map = lambda path: (
        ActivationFiles(path)
        if Path(path).is_dir() else original_load(path))
    result = convert.main()
    if result == 0:
        from mlxl3_quantizer import list_exl3_modules, validate_checkpoint_files
        from ponyexl3.convert.discovery import discover_exl3_module_keys
        metadata_path = args.out_dir / "quantization_config.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["bits"] = args.bits
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        # Native MLX architecture, no remote Python required by this export.
        config.pop("auto_map", None)
        (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        validate_checkpoint_files(args.out_dir)
        if not args.module_limit:
            expected = discover_exl3_module_keys(args.in_dir, include_routed_experts=True)
            if set(list_exl3_modules(args.out_dir)) != set(expected):
                raise ValueError("Incomplete Ling EXL3 projection coverage")
            output_keys = json.loads((args.out_dir / "model.safetensors.index.json").read_text())["weight_map"]
            missing = [key for key in SafetensorIndex(args.in_dir).weight_map
                       if key not in output_keys and key.removesuffix(".weight") not in expected]
            if missing:
                raise ValueError(f"Missing Ling source tensors: {missing[:8]}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
