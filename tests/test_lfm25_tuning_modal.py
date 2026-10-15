from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def legacy_capture():
    row = {"shape": [1024, 128000], "dtype": "float16", "bytes": 262144128, "sha256": "0" * 64}
    return {
        "status": "complete",
        "windows": {name: copy.deepcopy(row) for name in ("window1", "window5")},
    }


def routes_capture():
    row = legacy_capture()["windows"]["window1"]
    row.update(
        routes={"bytes": 100, "layers": 38, "tokens_per_layer": 1024, "top_k": 4},
        replay_bitexact=True,
        normal_comparison={"kl": 0.0, "tokens": 1024, "greedy_matches": 1024},
    )
    return {
        "status": "complete",
        "ids_only": "approved",
        "arms": {
            arm: {name: copy.deepcopy(row) for name in ("window1", "window5")}
            for arm in ("source", "normal", "own_routes", "source_routes", "own_ids", "source_ids")
        },
    }


@pytest.fixture
def runner(tmp_path, monkeypatch):
    import diagnose_lfm25_tuning_modal as target

    legacy = tmp_path / "legacy.py"
    legacy.write_bytes(b"pinned legacy fixture")
    monkeypatch.setattr(target, "LEGACY", legacy)
    monkeypatch.setattr(target, "LEGACY_SHA256", hashlib.sha256(legacy.read_bytes()).hexdigest())
    monkeypatch.setattr(target, "DATA", tmp_path / "tuning37")
    monkeypatch.setattr(target, "LEGACY_CAPTURE", tmp_path / "diagnostic23")
    monkeypatch.setattr(target, "validate_plan", lambda plan: plan)
    commits = []
    monkeypatch.setattr(target, "volume", SimpleNamespace(commit=lambda: commits.append(True)))
    return target, commits


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("cold-a", "cache-a.bin"),
        ("warm-a", "cache-a.bin"),
        ("cold-b", "cache-b.bin"),
        ("routes-a", "cache-a.bin"),
    ],
)
def test_registered_cache_selection(stage, expected):
    from diagnose_lfm25_tuning_modal import cache_name

    assert cache_name(stage) == expected
    with pytest.raises(ValueError):
        cache_name("unexpected")


def install_child(target, monkeypatch, failure=None):
    commands = []

    def child(command, log, timeout):
        assert len(command) == 8 and command[2] == "/opt/mlxl3/diagnose_lfm25_tuning.py"
        stage, plan, folder, cache, legacy = command[3:]
        folder, cache = Path(folder), Path(cache)
        assert stage == target.STAGES[len(commands)]
        assert timeout == (540 if stage == "routes-a" else 180)
        assert json.loads(Path(plan).read_text()) == {"frozen": True}
        assert Path(legacy) == target.LEGACY
        assert Path(log) == target.DATA / f"{stage}.log"
        assert not folder.exists() and not target.LEGACY_CAPTURE.exists()
        before = target.cache_snapshot(cache)
        assert before["exists"] == (stage in ("warm-a", "routes-a"))
        commands.append(command)
        failing = stage == "routes-a" if failure == "route_arm" else stage == "warm-a"
        if failing and failure in ("timeout", "error"):
            raise (
                TimeoutError("silent stage")
                if failure == "timeout"
                else RuntimeError("stage failed")
            )
        folder.mkdir()
        capture_dir = folder / "capture"
        capture_dir.mkdir()
        cache.write_bytes((cache.read_bytes() if cache.exists() else b"") + stage.encode())
        if failure == "tamper_a" and stage == "cold-b":
            (target.DATA / "cache-a.bin").write_bytes(b"unexpected independent-child mutation")
        capture = routes_capture() if stage == "routes-a" else legacy_capture()
        if failing and failure == "incomplete_capture":
            capture["status"] = "running"
        if failing and failure == "window":
            capture["windows"].pop("window5")
        if failing and failure == "route_arm":
            capture["arms"].pop("source_ids")
        path = capture_dir / "report.json"
        path.write_text(json.dumps(capture))
        envelope = {
            "status": "complete",
            "stage": stage,
            "cache_path": str(cache),
            "cache_before": before,
            "cache_after": target.cache_snapshot(cache),
            "capture_report": str(path),
            "legacy_sha256": target.LEGACY_SHA256,
            "publication": False,
            "causal_interpretation": False,
        }
        if failing and failure == "incomplete_run":
            envelope["status"] = "running"
        elif failing and failure == "wrong_cache":
            envelope["cache_path"] = str(target.DATA / "cache-b.bin")
        elif failing and failure == "wrong_sha":
            envelope["legacy_sha256"] = "1" * 64
        elif failing and failure == "before":
            envelope["cache_before"]["sha256"] = "2" * 64
        elif failing and failure == "after":
            envelope["cache_after"]["sha256"] = "2" * 64
        elif failing and failure == "left_legacy":
            target.LEGACY_CAPTURE.mkdir()
        elif failing and failure == "lost_cache":
            cache.unlink()
        run = folder / "run.json"
        run.write_text(
            "{truncated"
            if failing and failure == "invalid_json"
            else "[]"
            if failing and failure == "list_json"
            else json.dumps(envelope)
        )

    monkeypatch.setattr(target.base, "run_logged", child)
    return commands


def test_four_fresh_children_run_in_registered_order_with_shared_and_independent_cache(
    runner, monkeypatch
):
    target, commits = runner
    assert target.STAGES == ("cold-a", "warm-a", "cold-b", "routes-a")
    assert target.VOLUME == "mlxl3-lfm25-v3-20261015"
    commands = install_child(target, monkeypatch)
    result = target.execute({"frozen": True})
    assert len(commands) == 4 and commits == [True]
    assert [command[3] for command in commands] == list(target.STAGES)
    assert [command[6] for command in commands] == [
        str(target.DATA / name)
        for name in ("cache-a.bin", "cache-a.bin", "cache-b.bin", "cache-a.bin")
    ]
    assert result["status"] == "complete" and result["causal_interpretation"] is False
    assert result["stage"] == "awaiting_cpu_comparison" and len(result["stages"]) == 4
    assert (
        result["stages"]["warm-a"]["run"]["cache_before"]
        == result["stages"]["cold-a"]["run"]["cache_after"]
    )
    assert (
        result["stages"]["routes-a"]["run"]["cache_before"]
        == result["stages"]["warm-a"]["run"]["cache_after"]
    )
    assert (target.DATA / "cache-a.bin").read_bytes() == b"cold-awarm-aroutes-a"
    assert (target.DATA / "cache-b.bin").read_bytes() == b"cold-b"
    before = (target.DATA / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        target.execute({"frozen": True})
    assert (target.DATA / "report.json").read_bytes() == before and len(commands) == 4


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "error",
        "invalid_json",
        "list_json",
        "incomplete_run",
        "incomplete_capture",
        "window",
        "wrong_cache",
        "wrong_sha",
        "before",
        "after",
        "left_legacy",
        "lost_cache",
        "route_arm",
    ],
)
def test_child_failure_or_incomplete_evidence_stops_without_causal_claim(
    runner, monkeypatch, failure
):
    target, commits = runner
    commands = install_child(target, monkeypatch, failure)
    with pytest.raises((ValueError, TypeError, RuntimeError, FileExistsError, TimeoutError)):
        target.execute({"frozen": True})
    data = json.loads((target.DATA / "report.json").read_text())
    assert (
        data["status"] == "failed"
        and data["causal_interpretation"] is False
        and data["publication"] is False
    )
    assert data["launcher_error"] and commits == [True]
    assert len(commands) == (4 if failure == "route_arm" else 2)
    assert data["stages"]["cold-a"]["run"]["status"] == "complete"


def test_independent_child_cannot_change_the_shared_cache(runner, monkeypatch):
    target, commits = runner
    commands = install_child(target, monkeypatch, "tamper_a")
    with pytest.raises(ValueError, match="cache changed"):
        target.execute({"frozen": True})
    assert len(commands) == 3 and commits == [True]
    assert json.loads((target.DATA / "report.json").read_text())["status"] == "failed"


def test_legacy_preexisting_source_and_invalid_plan_are_rejected_before_subprocess(
    runner, monkeypatch
):
    target, commits = runner
    commands = install_child(target, monkeypatch)
    target.LEGACY_CAPTURE.mkdir()
    with pytest.raises(FileExistsError):
        target.execute({"frozen": True})
    target.LEGACY_CAPTURE.rmdir()
    target.LEGACY.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="legacy"):
        target.execute({"frozen": True})
    assert not target.DATA.exists() and not commits and not commands


@pytest.mark.parametrize("kind", ["legacy", "routes"])
def test_capture_validator_requires_complete_positive_shapes_and_six_route_arms(kind):
    from diagnose_lfm25_tuning_modal import validate_capture

    data = legacy_capture() if kind == "legacy" else routes_capture()
    stage = "cold-a" if kind == "legacy" else "routes-a"
    assert validate_capture(data, stage) is data
    if kind == "legacy":
        data["windows"]["window1"]["shape"] = [0, 128000]
    else:
        data["ids_only"] = "not_executed_binding_absent"
        data["arms"].pop("own_ids")
        data["arms"].pop("source_ids")
    with pytest.raises(ValueError):
        validate_capture(data, stage)


def test_main_uses_stdlib_plan_and_single_remote_call_without_rpc(tmp_path, monkeypatch):
    import diagnose_lfm25_tuning_modal as target

    monkeypatch.setattr(target, "ROOT", tmp_path)
    (tmp_path / "build").mkdir()
    plan = {"frozen": True}
    (tmp_path / "build/diagnostic23-input.json").write_text(json.dumps(plan))
    calls = []
    monkeypatch.setattr(target, "validate_plan", lambda p: p)
    monkeypatch.setattr(
        target.diagnose,
        "remote",
        lambda p: calls.append(p) or {"status": "complete", "causal_interpretation": False},
    )
    target.main()
    assert calls == [plan]
    assert (
        json.loads((tmp_path / "build/lfm25-tuning37-cloud.json").read_text())["status"]
        == "complete"
    )


def test_real_cli_help_without_credentials_or_gpu(tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ, MODAL_CONFIG_PATH=str(tmp_path / "modal.toml"), NO_COLOR="1")
    for key in ("MODAL_PROFILE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-m", "modal", "run", "scripts/diagnose_lfm25_tuning_modal.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--profile TEXT" in result.stdout


def test_real_sdk_enforces_copy_before_lazy_mount_without_rpc():
    import asyncio

    import diagnose_lfm25_tuning_modal as target
    from modal._image import _Image
    from modal._utils.async_utils import synchronizer

    image = synchronizer._translate_in(target.image)
    snapshots = {}

    class OfflineResolver:
        async def load(self, obj, context):
            assert not isinstance(obj, _Image)

    async def visit(node):
        if node in snapshots:
            return
        snapshots[node] = dict(node.__dict__)
        for dependency in node._deps_():
            if isinstance(dependency, _Image):
                await visit(dependency)
        if node._rep == "Image(local files)":
            await node._load(node, OfflineResolver(), SimpleNamespace(app_id=None), None)
        else:
            with pytest.raises(AssertionError):
                await node._load(node, OfflineResolver(), SimpleNamespace(app_id=None), None)
            node._hydrate("im-offline-test", SimpleNamespace(), None)

    try:
        asyncio.run(visit(image))
    finally:
        for node, state in snapshots.items():
            node.__dict__.clear()
            node.__dict__.update(state)


def test_remote_module_import_uses_the_actual_image_pythonpath(tmp_path):
    import inspect
    import os
    import shlex
    import shutil
    import subprocess
    import sys

    import diagnose_lfm25_tuning_modal as target
    from modal._utils.async_utils import synchronizer

    environment, visited = {}, set()

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        for dependency in node._deps_():
            if hasattr(dependency, "_rep"):
                visit(dependency)
        if "env." in node._rep:
            builder = inspect.getclosurevars(node._load).nonlocals["dockerfile_function"]
            for command in builder("2025.06").commands:
                if command.startswith("ENV "):
                    key, value = shlex.split(command[4:])[0].split("=", 1)
                    environment[key] = value

    visit(synchronizer._translate_in(target.image))
    root = Path(__file__).resolve().parents[1]
    module_root, payload = tmp_path / "module-root", tmp_path / "image-payload"
    module_root.mkdir()
    payload.mkdir()
    for name in ("diagnose_lfm25_tuning_modal", "quantize_lfm25_modal"):
        shutil.copyfile(root / "scripts" / f"{name}.py", module_root / f"{name}.py")
    for name in ("diagnose_lfm25_routes", "diagnose_lfm25_routes_modal", "diagnose_lfm25_tuning"):
        shutil.copyfile(root / "scripts" / f"{name}.py", payload / f"{name}.py")
    bootstrap = module_root / "remote_import.py"
    bootstrap.write_text(
        "import sys\nfrom pathlib import Path\nimport modal\n"
        "modal.is_local = lambda: False\n"
        "import diagnose_lfm25_tuning_modal\n"
        "import diagnose_lfm25_routes, diagnose_lfm25_tuning\n"
        "assert diagnose_lfm25_tuning_modal.STAGES == ('cold-a','warm-a','cold-b','routes-a')\n"
        "assert Path(diagnose_lfm25_routes.__file__).parent == Path(sys.argv[1])\n"
        "assert Path(diagnose_lfm25_tuning.__file__).parent == Path(sys.argv[1])\n"
        "assert not any(Path(p).resolve() == Path(sys.argv[2]) for p in sys.path if p)\n"
        "print('remote module and actual copied helpers imported without RPC/GPU')\n"
    )
    runtime_path = os.pathsep.join(
        str(payload) if item == "/opt/mlxl3" else item
        for item in environment.get("PYTHONPATH", "").split(os.pathsep)
        if item
    )
    env = dict(os.environ, PYTHONPATH=runtime_path, MODAL_CONFIG_PATH=str(tmp_path / "modal.toml"))
    for key in ("MODAL_PROFILE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, str(bootstrap), str(payload), str(root / "scripts")],
        cwd=module_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert environment["PYTHONPATH"] == "/opt/mlxl3"
    assert "actual copied helpers imported" in result.stdout
