from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import diagnose_lfm25_routes as routes
import diagnose_lfm25_tuning as target
import pytest

LEGACY = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/quant-lfm25-diagnostics35-36-20261015/diagnostic23-original.py.txt"
)


def cache_bytes(*keys):
    return b"EX3ATUNE\x01\0\0\0\x20\0\0\0" + b"".join(
        struct.pack("<QiiiiII", key, 2, 128, 132, 1, 0, 0) for key in keys
    )


def plan():
    return {
        "windows": {name: list(range(1024)) for name in ("window1", "window5")},
        "shards": routes.V2_SHARDS.copy(),
    }


def legacy_report():
    return {
        "status": "complete",
        "publication": False,
        "windows": {
            name: {
                "shape": [1024, 128000],
                "dtype": "float16",
                "bytes": 262144128,
                "sha256": "0" * 64,
                "token_sha256": digest,
            }
            for name, digest in target.WINDOW_HASHES.items()
        },
    }


@pytest.fixture(autouse=True)
def synthetic_plan(monkeypatch):
    digest = hashlib.sha256(struct.pack("<1024I", *range(1024))).hexdigest()
    monkeypatch.setattr(routes, "WINDOW_HASHES", {"window1": digest, "window5": digest})
    monkeypatch.setattr(target, "WINDOW_HASHES", {"window1": digest, "window5": digest})


@pytest.mark.parametrize("stage", target.STAGES)
def test_registered_stage_cache_contract(stage):
    assert target.cold_stage(stage) == stage.startswith("cold-")


@pytest.mark.parametrize("stage", ["cold", "warm-b", "", False, None, 1])
def test_invalid_stage_fails_before_any_capture(stage):
    with pytest.raises(ValueError):
        target.cold_stage(stage)


def test_real_archived_source_imports_through_sdk_without_importing_cuda(monkeypatch):
    before = sys.path.copy()
    try:
        assert (
            target.file_sha(LEGACY)
            == target.LEGACY_SHA256
            == "99c650616348c59a2d5a2c28f5b4519051a3a6295c305329f2aa0b2e5a617910"
        )
        original = target.load_legacy(LEGACY)
        assert original.__name__ == "diagnose" and original.__code__.co_filename == str(LEGACY)
        assert "Cache" in original.__code__.co_names and "Model" in original.__code__.co_names
        assert "exllamav3" not in sys.modules
        assert (
            original.__globals__["SOURCE_HASHES"]
            == __import__("quantize_lfm25_modal").SOURCE_HASHES
        )
    finally:
        sys.path[:] = before
        monkeypatch.delitem(sys.modules, "mlxl3_diagnostic23_original", raising=False)


def test_changed_original_bytes_are_refused_before_sdk_import(tmp_path):
    bad = tmp_path / "original.py"
    bad.write_bytes(LEGACY.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        target.load_legacy(bad)


def test_cache_snapshot_and_pinned_binary_records(tmp_path):
    cache = tmp_path / "cache-a.bin"
    assert target.cache_snapshot(cache) == {
        "path": str(cache),
        "exists": False,
        "bytes": 0,
        "sha256": None,
    }
    assert target.cache_records(cache) == {}
    cache.write_bytes(cache_bytes(1, 8))
    values = target.cache_records(cache)
    assert set(values) == {"0000000000000001", "0000000000000008"}
    assert values["0000000000000008"] == {
        "tag": 2,
        "block_dim": 128,
        "num_sms": 132,
        "concurrency": 1,
    }
    assert target.cache_snapshot(cache)["bytes"] == 80


@pytest.mark.parametrize(
    "kind", ["empty", "magic", "format", "size", "partial", "invalid_geometry", "conflicting"]
)
def test_corrupt_cache_is_not_silently_treated_as_warm(tmp_path, kind):
    data = cache_bytes(1)
    if kind == "empty":
        data = b""
    elif kind == "magic":
        data = b"INVALID!" + data[8:]
    elif kind == "format":
        data = data[:8] + struct.pack("<I", 2) + data[12:]
    elif kind == "size":
        data = data[:12] + struct.pack("<I", 16) + data[16:]
    elif kind == "partial":
        data = data[:-1]
    elif kind == "invalid_geometry":
        data = data[:28] + struct.pack("<i", 0) + data[32:]
    else:
        data += struct.pack("<QiiiiII", 1, 3, 128, 132, 1, 0, 0)
    cache = tmp_path / "cache.bin"
    cache.write_bytes(data)
    with pytest.raises(ValueError):
        target.cache_records(cache)


def test_directory_or_symlink_is_not_an_explicit_cache_file(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError):
        target.cache_snapshot(cache)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        target.cache_snapshot(link)


@pytest.mark.parametrize("failure", ["status", "window", "shape", "bytes", "dtype", "token", "sha"])
def test_legacy_report_requires_two_complete_actual_windows(failure):
    data = legacy_report()
    row = data["windows"]["window1"]
    if failure == "status":
        data["status"] = "failed"
    elif failure == "window":
        data["windows"].pop("window5")
    elif failure == "shape":
        row["shape"] = [0, 128000]
    elif failure == "bytes":
        row["bytes"] = 0
    elif failure == "dtype":
        row["dtype"] = "float32"
    elif failure == "token":
        row["token_sha256"] = "0" * 64
    else:
        row["sha256"] = "z" * 64
    with pytest.raises(ValueError):
        target.validate_legacy_report(data)


def setup_run(tmp_path, monkeypatch, *, failure=None):
    cache = tmp_path / "cache-a.bin"
    output = tmp_path / "out"
    legacy_output = tmp_path / "diagnostic23"
    monkeypatch.setattr(target, "LEGACY_OUTPUT", legacy_output)
    observed = []

    def metadata():
        assert os.environ["EXLLAMAV3_TUNE_CACHE"] == str(cache)
        observed.append("metadata")
        return {"test_only": True}

    monkeypatch.setattr(target, "runtime_metadata", metadata)
    monkeypatch.setattr(target, "flags", lambda: {"test_only": True})
    old_volume = object()

    def raw(windows, shards):
        assert os.environ["EXLLAMAV3_TUNE_CACHE"] == str(cache)
        assert windows == plan()["windows"] and shards == plan()["shards"]
        assert raw.__globals__["volume"] is not old_volume
        raw.__globals__["volume"].commit()
        observed.append("original")
        legacy_output.mkdir()
        data = legacy_report()
        if failure == "incomplete":
            data["windows"].pop("window1")
        (legacy_output / "report.json").write_text(json.dumps(data))
        cache.write_bytes(cache_bytes(1, 2) if failure != "retune" else cache_bytes(3))
        if failure == "exception":
            raise RuntimeError("original child failure")
        if failure == "different_report":
            return {**data, "extra": True}
        return data

    monkeypatch.setitem(raw.__globals__, "volume", old_volume)

    def load(path):
        assert path == LEGACY and os.environ["EXLLAMAV3_TUNE_CACHE"] == str(cache)
        observed.append("load")
        return raw

    monkeypatch.setattr(target, "load_legacy", load)
    return cache, output, legacy_output, observed, raw, old_volume


@pytest.mark.parametrize("stage", ["cold-a", "cold-b", "warm-a"])
def test_original_entrypoint_keeps_math_and_sets_cache_before_import(tmp_path, monkeypatch, stage):
    cache, output, original_path, observed, raw, old_volume = setup_run(tmp_path, monkeypatch)
    if stage == "warm-a":
        cache.write_bytes(cache_bytes(1))
    result = target.run_stage(stage, plan(), output, cache, LEGACY)
    assert (
        result["status"] == "complete"
        and result["stage"] == stage
        and not result["causal_interpretation"]
    )
    assert observed == ["metadata", "load", "original"]
    assert not original_path.exists() and (output / "capture/report.json").exists()
    assert raw.__globals__["volume"] is old_volume
    assert result["cache_before"]["exists"] == (stage == "warm-a")
    assert result["cache_after"] == target.cache_snapshot(cache)
    assert result["cache_reuse"]["existing_records_preserved"]
    before = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        target.run_stage(stage, plan(), output, cache, LEGACY)
    assert (output / "run.json").read_bytes() == before


@pytest.mark.parametrize("failure", ["exception", "incomplete", "different_report", "retune"])
def test_original_failure_preserves_capture_and_restores_only_volume(
    tmp_path, monkeypatch, failure
):
    cache, output, original_path, _observed, raw, old_volume = setup_run(
        tmp_path, monkeypatch, failure=failure
    )
    cache.write_bytes(cache_bytes(1))
    with pytest.raises((RuntimeError, ValueError)):
        target.run_stage("warm-a", plan(), output, cache, LEGACY)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert (output / "capture/report.json").exists() and not original_path.exists()
    assert raw.__globals__["volume"] is old_volume


@pytest.mark.parametrize(
    "failure",
    [
        "cold_existing",
        "warm_absent",
        "warm_empty",
        "cache_directory",
        "legacy_existing",
        "loaded_exllama",
        "changed_plan",
    ],
)
def test_invalid_preconditions_never_start_runtime_or_modify_existing_outputs(
    tmp_path, monkeypatch, failure
):
    cache, output, original_path, observed, _raw, _old_volume = setup_run(tmp_path, monkeypatch)
    stage = "cold-a"
    data = plan()
    if failure == "cold_existing":
        cache.write_bytes(cache_bytes(1))
    elif failure in ("warm_absent", "warm_empty"):
        stage = "warm-a"
        if failure == "warm_empty":
            cache.touch()
    elif failure == "cache_directory":
        cache.mkdir()
    elif failure == "legacy_existing":
        original_path.mkdir()
        (original_path / "keep").write_text("preserve")
    elif failure == "loaded_exllama":
        monkeypatch.setitem(sys.modules, "exllamav3", SimpleNamespace())
    else:
        data["windows"]["window1"][0] = 9
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        target.run_stage(stage, data, output, cache, LEGACY)
    assert json.loads((output / "run.json").read_text())["status"] == "failed" and not observed
    if failure == "legacy_existing":
        assert (original_path / "keep").read_text() == "preserve"


def test_routes_stage_calls_unmodified_diagnostic36_entrypoint(tmp_path, monkeypatch):
    cache, output, original_path, observed, _raw, _old_volume = setup_run(tmp_path, monkeypatch)
    cache.write_bytes(cache_bytes(1))
    import diagnose_lfm25_routes_modal as validator

    def capture(data, path):
        assert data == plan() and path == output / "capture"
        path.mkdir()
        result = {"status": "complete", "test_only": True}
        (path / "report.json").write_text(json.dumps(result))
        cache.write_bytes(cache_bytes(1, 2))
        observed.append("routes36")
        return result

    monkeypatch.setattr(routes, "run", capture)
    monkeypatch.setattr(validator, "validate_result", lambda value: value)
    result = target.run_stage("routes-a", plan(), output, cache, LEGACY)
    assert result["status"] == "complete" and observed == ["metadata", "routes36"]
    assert not original_path.exists()


def test_cli_routes_all_five_positionals_to_registered_api(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan()))
    calls = []
    monkeypatch.setattr(
        target, "run_stage", lambda *args: calls.append(args) or {"status": "complete"}
    )
    output, cache = tmp_path / "out", tmp_path / "cache.bin"
    assert target.main(["cold-a", str(path), str(output), str(cache), str(LEGACY)]) == {
        "status": "complete"
    }
    assert calls == [("cold-a", plan(), output, cache, LEGACY)]


@pytest.mark.parametrize("failure", [None, "driver", "config"])
def test_runtime_metadata_records_configs_and_versions_without_changing_flags(
    tmp_path, monkeypatch, failure
):
    properties = SimpleNamespace(
        name="H100 fake", total_memory=80 * 1024**3, multi_processor_count=132, major=9, minor=0
    )
    fake = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda _: properties),
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        get_float32_matmul_precision=lambda: "highest",
        are_deterministic_algorithms_enabled=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(target, "MODEL", tmp_path)
    monkeypatch.setattr(target.importlib.metadata, "version", lambda _: "test-only")
    for name in ("config.json", "quantization_config.json"):
        (tmp_path / name).write_text(
            json.dumps(
                {"model_type": "lfm2_moe", "quant_method": "exl3", "tensor_storage": {"x": "f16"}}
                if failure != "config"
                else []
            )
        )

    def driver(command, **kwargs):
        assert command == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        assert kwargs["timeout"] == 10 and kwargs["check"] is True
        return SimpleNamespace(stdout="550.0\n" if failure != "driver" else "")

    monkeypatch.setattr(target.subprocess, "run", driver)
    if failure:
        with pytest.raises(ValueError):
            target.runtime_metadata()
    else:
        result = target.runtime_metadata()
        assert result["driver"] == "550.0" and result["cuda"] == "12.8"
        assert set(result["config_files"]) == {"config.json", "quantization_config.json"}
        assert result["flags_before"]["matmul_allow_tf32"] is False
        assert result["flags_before"]["cudnn_allow_tf32"] is True
        assert result["config_files"]["config.json"]["sha256"] == target.file_sha(
            tmp_path / "config.json"
        )


@pytest.mark.parametrize("stage", ["cold-a", "routes-a"])
@pytest.mark.parametrize("failure", [False, True])
def test_traceback_timer_covers_capture_and_is_cancelled_on_success_or_error(
    tmp_path, monkeypatch, stage, failure
):
    cache, output, _original_path, _observed, raw, _old_volume = setup_run(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(
        target.faulthandler,
        "dump_traceback_later",
        lambda seconds, *, repeat: events.append(("armed", seconds, repeat)),
    )
    monkeypatch.setattr(
        target.faulthandler, "cancel_dump_traceback_later", lambda: events.append("cancelled")
    )

    def invoked():
        assert events == [("armed", 60, True)]
        events.append("capture")

    if stage == "cold-a":

        def legacy(windows, shards):
            invoked()
            result = raw(windows, shards)
            if failure:
                raise RuntimeError("interrupted legacy capture")
            return result

        monkeypatch.setattr(target, "load_legacy", lambda _: legacy)
    else:
        import diagnose_lfm25_routes_modal as validator

        cache.write_bytes(cache_bytes(1))

        def capture(plan, path):
            invoked()
            if failure:
                raise RuntimeError("interrupted routes capture")
            path.mkdir()
            result = {"status": "complete", "test_only": True}
            (path / "report.json").write_text(json.dumps(result))
            cache.write_bytes(cache_bytes(1, 2))
            return result

        monkeypatch.setattr(routes, "run", capture)
        monkeypatch.setattr(validator, "validate_result", lambda result: result)
    if failure:
        with pytest.raises(RuntimeError, match="interrupted"):
            target.run_stage(stage, plan(), output, cache, LEGACY)
        assert json.loads((output / "run.json").read_text())["status"] == "failed"
    else:
        assert target.run_stage(stage, plan(), output, cache, LEGACY)["status"] == "complete"
    assert events == [("armed", 60, True), "capture", "cancelled"]
