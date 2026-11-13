"""Exercise the actual bundle metadata entry point with disposable artifacts."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_manifest import runtime_backend
from package_engine import manifest


@pytest.fixture
def project(tmp_path):
    for name in [
        "scripts/build_manifest.py",
        "scripts/package_engine.py",
        "Cargo.toml",
        "LICENSES/MTPLX-NOTICE.txt",
        "LICENSES/TensorFold-NOTICE.txt",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/MLX-VLM-MIT.txt",
        "apps/MLXL3Studio/Vendor/PonyExl3/NOTICE",
        "apps/MLXL3Studio/Resources/Info.plist",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def setup_runtime(root, backend):
    runtime = root / "App/Contents/Resources/runtime"
    runtime.mkdir(parents=True)
    (runtime / "mlxl3").write_bytes(b"fixture executable")
    if backend == "mlx":
        for name in ["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"]:
            (runtime / name).write_bytes(name.encode())
    data = manifest(runtime, "2.0.2", backend)
    (runtime / "engine.json").write_text(json.dumps(data))
    return runtime, data


def run(root, **environment):
    destination = root / "App/Contents/Resources/build-info.json"
    env = {**os.environ, "MLXL3_MACOS_SDK": "/fixture/SDK"}
    env.pop("MLXL3_ENGINE_BACKEND", None)
    env.update(environment)
    result = subprocess.run(
        [sys.executable, str(root / "scripts/build_manifest.py"), str(destination)],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    return result, destination


@pytest.mark.parametrize("backend", ["metal", "mlx"])
@pytest.mark.parametrize("flag", [None, "metal", "mlx"])
def test_metadata_uses_verified_artifact_with_unset_or_stale_build_flag(project, backend, flag):
    runtime, _ = setup_runtime(project, backend)
    result, destination = run(project, **({} if flag is None else {"MLXL3_ENGINE_BACKEND": flag}))
    assert result.returncode == 0, result.stderr
    data = json.loads(destination.read_text())
    assert data["runtime"] == ("Rust + Metal" if backend == "metal" else "Rust + MLX 0.32.2")
    assert data["engine"] == {
        "backend": backend,
        "version": "2.0.2",
        "sha256": hashlib.sha256((runtime / "mlxl3").read_bytes()).hexdigest(),
    }
    assert data["quantizer_runtime"] == "PonyExl3 + MLX"
    assert data["swift_sdk"] == "/fixture/SDK" and data["commit"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "schema",
        "boolean",
        "backend",
        "hash",
        "size",
        "empty",
        "extra",
        "symlink",
        "json",
    ],
)
def test_invalid_bundled_runtime_never_gets_successful_build_info(project, fault):
    runtime, data = setup_runtime(project, "metal")
    if fault == "missing":
        (runtime / "mlxl3").unlink()
    elif fault == "schema":
        data["schema"] = 3
    elif fault == "boolean":
        data["bridge_protocol"] = True
    elif fault == "backend":
        data["backend"] = "cpu"
    elif fault == "hash":
        data["files"]["mlxl3"]["sha256"] = "a" * 64
    elif fault == "size":
        data["files"]["mlxl3"]["size"] = True
    elif fault == "empty":
        (runtime / "mlxl3").write_bytes(b"")
    elif fault == "extra":
        (runtime / "hidden.dylib").write_bytes(b"foreign")
    elif fault == "symlink":
        original = runtime / "engine.json"
        original.rename(runtime / "foreign.json")
        original.symlink_to(runtime / "foreign.json")
    if fault != "symlink":
        (runtime / "engine.json").write_text("[" if fault == "json" else json.dumps(data))
    result, destination = run(project)
    assert result.returncode != 0
    assert not destination.exists()


@pytest.mark.parametrize(
    "schema,backend",
    [(True, None), (0, None), (1, "metal"), (2, None), (2, "mlx"), (2, "cpu"), (3, "metal")],
)
def test_only_the_two_explicit_archive_contracts_are_accepted(schema, backend):
    with pytest.raises(ValueError):
        runtime_backend(schema, backend)


@pytest.mark.parametrize("backend", ["metal", "mlx"])
def test_bundle_records_exact_engine_release_tag(project, backend):
    runtime, _ = setup_runtime(project, backend)
    payload = manifest(runtime, "2.0.2", backend, release_tag="engine-v2.0.2-beta.2")
    (runtime / "engine.json").write_text(json.dumps(payload))
    result, destination = run(project)
    assert result.returncode == 0, result.stderr
    identity = json.loads(destination.read_text())["engine"]
    assert identity["version"] == "2.0.2"
    assert identity["release_tag"] == "engine-v2.0.2-beta.2"
