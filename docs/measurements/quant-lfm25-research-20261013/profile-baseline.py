"""Profile the production LFM expert groups, with disposable synthetic BF16 weights."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

STAGES = ("batched_prep_s", "batched_g_scale_s", "batched_state_s", "batched_ldlq_s")


def validate_report(report, repeats, compared=False):
    variants = ("reference", "candidate") if compared else ("candidate",)
    if (
        repeats < 1
        or not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("parity") is not True
        or not isinstance(report.get("runs"), list)
        or len(report["runs"]) != 2 * (repeats + 1) * len(variants)
    ):
        raise ValueError("Incomplete child profile")
    expected = {
        (pair, group, variant)
        for pair in range(-1, repeats)
        for group in ("gate_up", "down")
        for variant in variants
    }
    seen, signatures = set(), {}
    for row in report["runs"]:
        if (
            not isinstance(row, dict)
            or type(row.get("pair")) is not int
            or not isinstance(row.get("group"), str)
            or not isinstance(row.get("variant"), str)
        ):
            raise ValueError("Invalid child profile pass")
        identity = (row["pair"], row["group"], row["variant"])
        if identity not in expected or identity in seen:
            raise ValueError("Incomplete child profile coverage")
        seen.add(identity)
        keys = {
            f"model.layers.2.feed_forward.experts.{expert}.w{projection}"
            for expert in range(4)
            for projection in ((1, 3) if row["group"] == "gate_up" else (2,))
        }
        hashes = row.get("hashes")
        if (
            not isinstance(hashes, dict)
            or set(hashes) != keys
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in hashes.values()
            )
        ):
            raise ValueError("Invalid child profile hashes")
        if row["group"] in signatures and signatures[row["group"]] != hashes:
            raise ValueError("Child profile diverged")
        signatures[row["group"]] = hashes
        stages = row.get("stages")
        if not isinstance(stages, dict) or set(stages) != set(STAGES):
            raise ValueError("Incomplete child profile stages")
        seconds = row.get("seconds")
        values = (seconds, *stages.values())
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
            or seconds <= 0
            or any(value < 0 for value in stages.values())
        ):
            raise ValueError("Invalid child profile timing")
        for field in (
            "peak_mlx_bytes",
            "cache_mlx_bytes",
            "active_mlx_bytes",
            "cache_purges",
            "process_rss_bytes",
        ):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError("Invalid child profile memory")


def read_report(output):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite child profile JSON")
        return number

    return json.loads(output.read_text(), parse_float=finite_float, parse_constant=finite_float)


def write_report(output, report):
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def fingerprints(results, expected):
    if (
        not results
        or {r.module_key for r in results} != set(expected)
        or len(results) != len(expected)
    ):
        raise ValueError("Incomplete profile result")
    hashes = {}
    for result in results:
        layer = result.layer
        rows, cols = expected[result.module_key]
        layer.validate()
        if layer.trellis.shape != (rows // 16, cols // 16, 64):
            raise ValueError("Unexpected profile trellis shape")
        if not np.isfinite(result.stats["inner_mse"]) or result.stats["inner_mse"] < 0:
            raise ValueError("Invalid inner error")
        digest = hashlib.sha256()
        for value, shape in (
            (layer.trellis, layer.trellis.shape),
            (layer.suh, (rows,)),
            (layer.svh, (cols,)),
        ):
            if (
                value is None
                or value.shape != shape
                or not value.size
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid profile tensor")
            digest.update(value.tobytes())
        hashes[result.module_key] = digest.hexdigest()
    return hashes


def run(
    output,
    repeats=3,
    reference_source=None,
    reverse_order=False,
    scratch_dir=None,
    reference_hessian_source=None,
):
    if repeats < 1:
        raise ValueError("Positive repetitions required")
    import mlx.core as mx
    from ponyexl3.convert import hessian, metal_search, mlx_trellis
    from ponyexl3.convert.discovery import write_quantization_plan
    from quantize_lfm26 import patch_converter

    from mlxl3_quantizer import pack_trellis_mlx, quantize_tiles_mlx

    variants = {"candidate": quantize_tiles_mlx}
    pipelines = {"candidate": hessian.ldlq_quantize_group}
    reference_hash = None
    patch_converter()
    metal_search.quantize_tiles_mlx = quantize_tiles_mlx
    mlx_trellis.pack_trellis_mlx = pack_trellis_mlx
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "in_progress",
        "parity": None,
        "scope": "synthetic full expert group; not the 24B",
        "seed": 139,
        "rows": 2048,
        "bits": 4,
        "codebook": "MCG",
        "repeats": repeats,
        "device": mx.device_info(),
        "parity_scope": "repeated/cross-variant trellis and scale hashes; no whole-model quality claim",
        "reference_sha256": reference_hash,
        "reference_hessian_sha256": None,
        "reverse_order": reverse_order,
        "runs": [],
    }

    def save():
        write_report(output, report)

    save()
    clear_cache = mx.clear_cache
    purges = 0

    def counted_clear():
        nonlocal purges
        purges += 1
        clear_cache()

    try:
        if reference_source is not None:
            reference_source = Path(reference_source)
            spec = importlib.util.spec_from_file_location("lfm25_reference_metal", reference_source)
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            variants = {"reference": reference.quantize_tiles_mlx, **variants}
            reference_hash = hashlib.sha256(reference_source.read_bytes()).hexdigest()
        if reference_hessian_source is not None:
            reference_hessian_source = Path(reference_hessian_source)
            name = "lfm25_reference_hessian"
            spec = importlib.util.spec_from_file_location(name, reference_hessian_source)
            reference_hessian = importlib.util.module_from_spec(spec)
            sys.modules[name] = reference_hessian
            spec.loader.exec_module(reference_hessian)
            variants = {"reference": variants.get("reference", quantize_tiles_mlx), **variants}
            pipelines["reference"] = reference_hessian.ldlq_quantize_group
            report["reference_hessian_sha256"] = hashlib.sha256(
                reference_hessian_source.read_bytes()
            ).hexdigest()
        report["reference_sha256"] = reference_hash
        save()
        with tempfile.TemporaryDirectory(dir=scratch_dir) as directory:
            source, plan = Path(directory) / "source", Path(directory) / "plan"
            source.mkdir()
            rng = np.random.default_rng(139)
            groups = {"gate_up": {}, "down": {}}
            weights, acts = {}, {}
            common = rng.standard_normal((2048, 2048), dtype=np.float32).astype(np.float16)
            for expert in range(4):
                for projection in (1, 3, 2):
                    key = f"model.layers.2.feed_forward.experts.{expert}.w{projection}"
                    shape = (1536, 2048) if projection == 2 else (2048, 1536)
                    groups["down" if projection == 2 else "gate_up"][key] = shape
                    public = rng.standard_normal(shape, dtype=np.float32) / np.sqrt(shape[0])
                    weights[key + ".weight"] = mx.array(public.T).astype(mx.bfloat16)
                    acts[key] = (
                        common
                        if projection != 2
                        else rng.standard_normal((2048, 1536), dtype=np.float32).astype(np.float16)
                    )
            mx.save_safetensors(source / "model.safetensors", weights)
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {key: "model.safetensors" for key in weights}, "metadata": {}}
                )
            )
            del weights, public
            mx.clear_cache()
            write_quantization_plan(
                source, plan, bits=4, codebook="mcg", include_routed_experts=True, copy_assets=False
            )
            signatures = {}
            mx.clear_cache = counted_clear
            for pair in range(-1, repeats):
                order = ("gate_up", "down") if pair % 2 == 0 else ("down", "gate_up")
                for name in order:
                    variant_order = list(variants)
                    if (pair % 2 == 1) != reverse_order:
                        variant_order.reverse()
                    for variant in variant_order:
                        metal_search.quantize_tiles_mlx = variants[variant]
                        expected = groups[name]
                        mx.reset_peak_memory()
                        purges = 0
                        started = time.perf_counter()
                        results = pipelines.get(variant, hessian.ldlq_quantize_group)(
                            source,
                            plan,
                            list(expected),
                            scale_mode="computed",
                            search_backend="metal",
                            calibration_activations_by_module=acts,
                            quant_bits_by_module={key: 4 for key in expected},
                        )
                        seconds = time.perf_counter() - started
                        signature = fingerprints(results, expected)
                        if name in signatures and signature != signatures[name]:
                            raise ValueError("Profile repeat diverged")
                        signatures[name] = signature
                        stats = results[0].stats
                        row = {
                            "pair": pair,
                            "group": name,
                            "variant": variant,
                            "seconds": seconds,
                            "peak_mlx_bytes": mx.get_peak_memory(),
                            "cache_mlx_bytes": mx.get_cache_memory(),
                            "active_mlx_bytes": mx.get_active_memory(),
                            "cache_purges": purges,
                            "process_rss_bytes": int(
                                subprocess.check_output(
                                    ["ps", "-o", "rss=", "-p", str(os.getpid())],
                                    timeout=5,
                                    text=True,
                                ).strip()
                            )
                            * 1024,
                            "hashes": signature,
                            "stages": {field: stats[field] for field in STAGES},
                            "scales": {
                                r.module_key: r.stats["regularize_g_scale"] for r in results
                            },
                        }
                        if (
                            seconds <= 0
                            or not np.isfinite(seconds)
                            or any(not np.isfinite(v) or v < 0 for v in row["stages"].values())
                        ):
                            raise ValueError("Invalid profile timing")
                        report["runs"].append(row)
                        save()
                        print(
                            f"{name} {variant} pair={pair}: {seconds:.3f}s, stages={row['stages']}",
                            flush=True,
                        )
                        del results
            report.update(status="complete", parity=True)
    except BaseException as exc:
        report.update(status="failed", parity=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        mx.clear_cache = clear_cache
        save()
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--reference-hessian-source", type=Path)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("Positive repetitions required")
    if args.child:
        return run(
            args.output,
            args.repeats,
            args.reference_source,
            args.reverse_order,
            args.scratch_dir,
            args.reference_hessian_source,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "in_progress", "parity": None, "runs": []}
    write_report(args.output, report)
    try:
        # subprocess.run kills and joins the child on timeout or interruption.
        with tempfile.TemporaryDirectory(prefix="mlxl3-lfm-profile-") as scratch:
            subprocess.run(
                [sys.executable, __file__, *argv, "--child", "--scratch-dir", scratch],
                timeout=240,
                check=True,
            )
        report = read_report(args.output)
        validate_report(
            report, args.repeats, bool(args.reference_source or args.reference_hessian_source)
        )
        return report
    except BaseException as exc:
        try:
            partial = read_report(args.output)
            if isinstance(partial, dict):
                report = partial
        except (OSError, ValueError):
            pass
        if not isinstance(report, dict):
            report = {"runs": []}
        report.update(
            status="failed", parity=False, supervisor_error=f"{type(exc).__name__}: {exc}"
        )
        write_report(args.output, report)
        raise


if __name__ == "__main__":
    main()
