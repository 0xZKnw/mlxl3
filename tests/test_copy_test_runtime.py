"""The local test app must copy only a compatible, intact installed runtime."""

import hashlib
import importlib.util
import json
import plistlib
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "copy_test_runtime", Path(__file__).resolve().parents[1] / "scripts/copy-test-runtime.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "Library/Application Support/io.mlxl3.desktop/Engines"
    source = root / "v2.0.2"
    source.mkdir(parents=True)
    (root / "current.json").write_text('{"version":"2.0.2"}')
    files = {}
    for name in ["mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"]:
        data = ("fixture-" + name).encode()
        path = source / name
        path.write_bytes(data)
        path.chmod(0o755)
        files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "version": "2.0.2",
        "schema": 1,
        "bridge_protocol": 1,
        "architecture": "arm64",
        "minimum_app_version": "1.4.0",
        "files": files,
    }
    (source / "engine.json").write_text(json.dumps(manifest))
    (source / "NOTICE.txt").write_text("Preserve notices")
    bundle = tmp_path / "Preview.app"
    (bundle / "Contents").mkdir(parents=True)
    (bundle / "Contents/Info.plist").write_bytes(
        plistlib.dumps({"CFBundleShortVersionString": "1.4.2"})
    )
    return source, bundle, manifest


def test_copies_verified_runtime_without_modifying_installation(installed):
    source, bundle, _ = installed
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    runtime.copy_runtime(bundle)
    destination = bundle / "Contents/Resources/runtime"
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert (destination / "mlxl3").stat().st_mode & 0o111
    with pytest.raises(FileExistsError):
        runtime.copy_runtime(bundle)


@pytest.mark.parametrize("value", [None, True, 3, "", "1.2", "../2.0.2", "1.2.3/..", "2.0.2-beta"])
def test_rejects_invalid_versions(value):
    with pytest.raises(ValueError):
        runtime.version(value)


@pytest.mark.parametrize(
    "problem",
    [
        "disabled",
        "hash",
        "size",
        "symlink",
        "executable",
        "schema",
        "protocol",
        "architecture",
        "minimum",
        "version",
        "extra",
    ],
)
def test_refuses_bad_runtime_before_copying(installed, problem):
    source, bundle, manifest = installed
    if problem == "disabled":
        (source / "disabled").touch()
    elif problem == "hash":
        (source / "libmlx.dylib").write_bytes(b"X" * manifest["files"]["libmlx.dylib"]["size"])
    elif problem == "size":
        manifest["files"]["mlxl3"]["size"] = -1
    elif problem == "symlink":
        (source / "mlxl3").unlink()
        (source / "mlxl3").symlink_to(source / "libmlx.dylib")
    elif problem == "executable":
        (source / "mlxl3").chmod(0o600)
    elif problem == "schema":
        manifest["schema"] = 2
    elif problem == "protocol":
        manifest["bridge_protocol"] = 2
    elif problem == "architecture":
        manifest["architecture"] = "x86_64"
    elif problem == "minimum":
        manifest["minimum_app_version"] = "99.0.0"
    elif problem == "version":
        manifest["version"] = "2.0.1"
    else:
        manifest["files"]["unexpected"] = {"size": 1, "sha256": "0" * 64}
    (source / "engine.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runtime.copy_runtime(bundle)
    assert not (bundle / "Contents/Resources/runtime").exists()
