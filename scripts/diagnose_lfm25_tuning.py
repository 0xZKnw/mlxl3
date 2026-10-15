"""Run pinned diagnostic23/36 unchanged with an explicit cooperative-GEMM tune cache."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import struct
import subprocess
import sys
import time
import types
from pathlib import Path

from diagnose_lfm25_routes import WINDOW_HASHES, validate_plan

STAGES = ("cold-a", "warm-a", "cold-b", "routes-a")
LEGACY_SHA256 = "99c650616348c59a2d5a2c28f5b4519051a3a6295c305329f2aa0b2e5a617910"
LEGACY_OUTPUT = Path("/data/diagnostic23")
MODEL = Path("/data/v2/output")


def cold_stage(stage: str) -> bool:
    """Select the required initial state of the persistent tuning file.

    raises: ValueError
    post: __return__ == stage.startswith('cold-')
    post: __return__ or stage in ('warm-a', 'routes-a')
    """
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("Unknown registered tuning stage")
    return stage in ("cold-a", "cold-b")


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cache_snapshot(path: Path) -> dict:
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError("Tuning cache must be an explicit regular file")
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": file_sha(path) if exists else None,
    }


def cache_records(path: Path) -> dict:
    """Read the pinned little-endian DiskCacheHeader/RecordV1, without changing it."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    if (
        len(data) < 48
        or data[:8] != b"EX3ATUNE"
        or struct.unpack_from("<II", data, 8) != (1, 32)
        or (len(data) - 16) % 32
    ):
        raise ValueError("Invalid or incomplete pinned tuning cache format")
    records = {}
    for key, tag, block, sms, concurrency, _, _ in struct.iter_unpack("<QiiiiII", data[16:]):
        value = {"tag": tag, "block_dim": block, "num_sms": sms, "concurrency": concurrency}
        if (
            block <= 0
            or sms <= 0
            or concurrency <= 0
            or f"{key:016x}" in records
            and records[f"{key:016x}"] != value
        ):
            raise ValueError("Invalid or conflicting tuning-cache geometry")
        records[f"{key:016x}"] = value
    return records


def load_legacy(path: Path):
    if file_sha(path) != LEGACY_SHA256:
        raise ValueError("The archived diagnostic23 source changed")
    namespace = runpy.run_path(str(path), run_name="mlxl3_diagnostic23_original")
    original = inspect.unwrap(namespace["diagnose"].get_raw_f())
    if original.__code__.co_filename != str(path):
        raise RuntimeError("The SDK callable does not refer to the pinned original source")
    return original


def validate_legacy_report(report: dict) -> dict:
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or set(report.get("windows", {})) != set(WINDOW_HASHES)
    ):
        raise ValueError("Incomplete original diagnostic23 capture")
    for name, row in report["windows"].items():
        digest = row.get("sha256", "")
        if (
            row.get("shape") != [1024, 128000]
            or row.get("dtype") != "float16"
            or row.get("bytes") != 262144128
            or row.get("token_sha256") != WINDOW_HASHES[name]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Incomplete original diagnostic23 output evidence")
    return report


def flags() -> dict:
    import torch

    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("EXLLAMAV3_", "EXL3_"))
        },
    }


def runtime_metadata() -> dict:
    import torch

    gpu = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    if not driver:
        raise ValueError("Missing NVIDIA driver version")
    files = {}
    for name in ("config.json", "quantization_config.json"):
        path = MODEL / name
        metadata = json.loads(path.read_text())
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Incomplete model configuration")
        fields = (
            "architectures",
            "model_type",
            "dtype",
            "hidden_size",
            "num_hidden_layers",
            "num_experts",
            "num_experts_per_tok",
            "rope_parameters",
            "tie_word_embeddings",
            "quant_method",
        )
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha(path),
            "metadata": {key: metadata[key] for key in fields if key in metadata},
        }
        if "tensor_storage" in metadata:
            files[name]["tensor_storage_entries"] = len(metadata["tensor_storage"])
    return {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "exllamav3", "numpy", "safetensors", "modal")
        },
        "python": sys.version,
        "gpu": {
            "name": gpu.name,
            "total_memory": gpu.total_memory,
            "multi_processor_count": gpu.multi_processor_count,
            "compute_capability": [gpu.major, gpu.minor],
        },
        "cuda": torch.version.cuda,
        "driver": driver,
        "flags_before": flags(),
        "config_files": files,
    }


def run_stage(stage, plan, output: Path, cache: Path, legacy: Path) -> dict:
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    capture = output / "capture"
    report = {
        "status": "running",
        "stage": stage,
        "publication": False,
        "causal_interpretation": False,
        "cache_path": str(cache),
        "capture_report": str(capture / "report.json"),
        "legacy_sha256": LEGACY_SHA256,
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        cold = cold_stage(stage)
        validate_plan(plan)
        if file_sha(legacy) != LEGACY_SHA256:
            raise ValueError("The archived diagnostic23 source changed")
        if any(name == "exllamav3" or name.startswith("exllamav3.") for name in sys.modules):
            raise RuntimeError("Each tuning stage requires a fresh process before ExLlama import")
        if not cache.is_absolute() or not cache.parent.is_dir():
            raise ValueError("Use the registered absolute cache file in an existing directory")
        report["cache_before"] = cache_snapshot(cache)
        if (
            cold
            and report["cache_before"]["exists"]
            or not cold
            and (not report["cache_before"]["exists"] or report["cache_before"]["bytes"] == 0)
        ):
            raise ValueError(
                "Cold stages require an absent cache; warm stages require a nonempty cache"
            )
        report["cache_records_before"] = cache_records(cache)
        if stage != "routes-a" and (LEGACY_OUTPUT.exists() or LEGACY_OUTPUT.is_symlink()):
            raise FileExistsError("Preserve the existing original diagnostic23 output")
        os.environ["EXLLAMAV3_TUNE_CACHE"] = str(cache)
        report["runtime"] = runtime_metadata()
        report["script_sha256"] = file_sha(Path(__file__))
        save()
        faulthandler.dump_traceback_later(60, repeat=True)
        try:
            if stage == "routes-a":
                from diagnose_lfm25_routes import run
                from diagnose_lfm25_routes_modal import validate_result

                result = validate_result(run(plan, capture))
            else:
                original = load_legacy(legacy)
                previous = original.__globals__["volume"]
                original.__globals__["volume"] = types.SimpleNamespace(commit=lambda: None)
                try:
                    result = original(plan["windows"], plan["shards"])
                finally:
                    original.__globals__["volume"] = previous
                    if LEGACY_OUTPUT.exists():
                        LEGACY_OUTPUT.rename(capture)
                validate_legacy_report(result)
        finally:
            faulthandler.cancel_dump_traceback_later()
        saved = json.loads((capture / "report.json").read_text())
        if saved != result:
            raise ValueError("Returned and persisted numerical capture reports differ")
        report["runtime"]["flags_after"] = flags()
        report["cache_after"] = cache_snapshot(cache)
        report["cache_records_after"] = cache_records(cache)
        if not report["cache_after"]["exists"] or not report["cache_records_after"]:
            raise ValueError("The actual kernel did not persist a nonempty tuning cache")
        before, after = report["cache_records_before"], report["cache_records_after"]
        if any(after.get(key) != value for key, value in before.items()):
            raise ValueError("Warm-cache existing geometries were removed or changed")
        report["cache_reuse"] = {
            "existing_records_preserved": True,
            "new_records": sorted(set(after) - set(before)),
            "bytes_unchanged": report["cache_before"]["sha256"] == report["cache_after"]["sha256"],
        }
        report["capture_report_sha256"] = file_sha(capture / "report.json")
        report["status"] = "complete"
        save()
        return report
    except BaseException as error:
        report.update(
            status="failed", error=f"{type(error).__name__}: {error}", causal_interpretation=False
        )
        try:
            report["cache_after"] = cache_snapshot(cache)
        except (ValueError, OSError) as metadata_error:
            report["cache_after_error"] = str(metadata_error)
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("legacy", type=Path)
    args = parser.parse_args(argv)
    return run_stage(
        args.stage, json.loads(args.plan.read_text()), args.output, args.cache, args.legacy
    )


if __name__ == "__main__":
    main()
