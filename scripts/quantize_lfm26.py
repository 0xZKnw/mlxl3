"""LFM2.5-2.6B adapter for the checked-out PonyExl3 Metal LDLQ converter.

Run with the same arguments as ponyexl3.cli.convert; --module-limit is for pilots.
Leaves embeddings tied and unquantized (FP16 export). Target bpw refers to quantized linear weights;
the total file size also includes embeddings, norms and convolution kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references" / "PonyExl3"))


def is_lfm_projection(key: str) -> bool:
    return (
        re.fullmatch(r"model\.layers\.\d+\.(feed_forward\.w[123]|conv\.(in_proj|out_proj))", key)
        is not None
    )


def patch_converter():
    from ponyexl3.convert import capture, direct, discovery, driver, measure, reuse

    original_filter = discovery._is_exl3_linear_key
    original_group = driver._sibling_group_signature
    original_tensors = direct._layer_tensors
    original_capture = capture.capture_calibration_activations
    original_enable = reuse.enable
    # Both K=4/5 candidates total ~2.74 GB: keep them until the emit stage.
    reuse.enable = lambda max_bytes=None: original_enable(
        4 << 30 if max_bytes is None else max_bytes
    )
    discovery._is_exl3_linear_key = lambda key: is_lfm_projection(key) or original_filter(key)

    def sibling(key):
        if re.fullmatch(r"model\.layers\.\d+\.feed_forward\.w[13]", key):
            return key.rsplit(".", 1)[0] + "._w1w3"
        return original_group(key)

    driver._sibling_group_signature = sibling
    measure._sibling_group_signature = sibling

    def layer_tensors(layer):
        tensors = original_tensors(layer)
        # EXL3 runtimes inspect these tensors, not only quantization_config.json.
        for name, multiplier in (("mcg", 0xCBAC1FED), ("mul1", 0x83DCD12D)):
            if getattr(layer, name):
                tensors[f"{layer.key}.{name}"] = np.array(multiplier, dtype=np.uint32).view(
                    np.int32
                )
        return tensors

    direct._layer_tensors = layer_tensors

    def capture_all(*args, **kwargs):
        summary = original_capture(*args, **kwargs)
        if summary.missing or summary.captured_count != summary.module_count:
            raise ValueError(f"Incomplete LFM calibration: {summary.missing}")
        return summary

    capture.capture_calibration_activations = capture_all


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--module-limit", type=int)
    parser.add_argument(
        "--reference-metal-search",
        action="store_true",
        help="Use the original PonyExl3 search for A/B comparisons",
    )
    args, _ = parser.parse_known_args()
    config = json.loads((args.in_dir / "config.json").read_text())
    if config.get("model_type") != "lfm2" or not config.get("tie_word_embeddings"):
        raise ValueError("This adapter expects a native LFM2 checkpoint with tied embeddings")
    patch_converter()
    if args.reference_metal_search:
        sys.argv.remove("--reference-metal-search")
    else:
        from ponyexl3.convert import metal_search

        from mlxl3.quantize import quantize_tiles_mlx

        metal_search.quantize_tiles_mlx = quantize_tiles_mlx
    from ponyexl3.convert.discovery import discover_exl3_module_keys

    keys = discover_exl3_module_keys(args.in_dir)
    expected = sum(7 if kind == "full_attention" else 5 for kind in config["layer_types"])
    if len(keys) != expected:
        raise ValueError(
            f"Incomplete LFM conversion plan: {len(keys)} linears, expected {expected}"
        )
    print(
        f"[mlxl3] LFM coverage: {len(keys)} linears; tied embeddings exported in FP16", flush=True
    )
    from ponyexl3.cli.convert import main as convert

    result = convert()
    if result == 0:
        shutil.copy2(args.in_dir / "LICENSE", args.out_dir / "LICENSE")
        metadata = json.loads((args.out_dir / "quantization_config.json").read_text())
        converted = [
            k for k, v in metadata["tensor_storage"].items() if v.get("quant_format") == "exl3"
        ]
        if not args.module_limit and set(converted) != set(keys):
            raise ValueError("The finished checkpoint does not cover every LFM projection")
        output_index = json.loads((args.out_dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for key in converted:
            info = metadata["tensor_storage"][key]
            for mode in ("mcg", "mul1"):
                if info.get(mode + "_multiplier") and f"{key}.{mode}" not in output_index:
                    raise ValueError(f"Missing runtime codebook marker: {key}.{mode}")
        if not args.module_limit:
            source_index = json.loads((args.in_dir / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            missing = [
                k
                for k in source_index
                if k not in output_index and k.removesuffix(".weight") not in converted
            ]
            if missing:
                raise ValueError(f"Missing unquantized source tensors: {missing}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
