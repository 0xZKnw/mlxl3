"""Release archives must be reproducible, complete and independently verifiable."""

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_engine", Path(__file__).resolve().parents[1] / "scripts/package_engine.py"
)
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


@pytest.mark.parametrize(
    "version,minimum",
    [
        ("1.2.0", "1.2.0"),
        ("1.3.9", "1.2.0"),
        ("1.4.0", "1.4.0"),
        ("1.4.10", "1.4.0"),
        ("2.0.0", "1.4.0"),
    ],
)
def test_mtp_target_selection_requires_compatible_desktop(runtime, version, minimum):
    assert engine.manifest(runtime, version)["minimum_app_version"] == minimum


@pytest.fixture
def runtime(tmp_path):
    folder = tmp_path / "runtime"
    folder.mkdir()
    for index, name in enumerate(engine.FILES):
        (folder / name).write_bytes(bytes([index + 1]) * (index + 1) * 13)
    return folder


def test_archive_reproducible_and_hashes_match(runtime, tmp_path):
    first = engine.package(runtime, tmp_path / "one", "1.2.0")
    second = engine.package(runtime, tmp_path / "two", "1.2.0")
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        assert set(archive.getnames()) == {"engine.json", *engine.FILES}
        manifest = json.load(archive.extractfile("engine.json"))
        assert archive.getmember("engine.json").size <= 64 * 1024
        assert manifest["bridge_protocol"] == 1
        assert manifest["architecture"] == "arm64"
        assert "Powered by MTPLX" in manifest["notices"]["MTPLX"]
        assert "Apache License" in manifest["notices"]["Apache-2.0"]
        for name in engine.FILES:
            member = archive.getmember(name)
            assert member.isfile() and member.mtime == 0 and member.uid == member.gid == 0
            data = archive.extractfile(name).read()
            assert manifest["files"][name] == {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }


@pytest.mark.parametrize(
    "version", ["../1.2.0", "v1.2.0", "1.2", "01.2.0", "1.2.0-beta", "1.2.0\n", "1.2.0/../2.0.0"]
)
def test_invalid_version_never_publishes(runtime, tmp_path, version):
    with pytest.raises(ValueError):
        engine.package(runtime, tmp_path / "dist", version)
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("failure", ["missing", "empty", "symlink", "directory", "oversize"])
def test_invalid_file_never_publishes(runtime, tmp_path, failure):
    source = runtime / "mlxl3"
    source.unlink()
    if failure == "empty":
        source.touch()
    elif failure == "symlink":
        source.symlink_to(runtime / "libmlx.dylib")
    elif failure == "directory":
        source.mkdir()
    elif failure == "oversize":
        with source.open("wb") as stream:
            stream.truncate(engine.MAX_FILE + 1)
    with pytest.raises(ValueError):
        engine.package(runtime, tmp_path / "dist", "1.2.0")
    assert not (tmp_path / "dist").exists()


def test_changed_runtime_preserves_previous_archive(runtime, tmp_path, monkeypatch):
    folder = tmp_path / "dist"
    original = engine.package(runtime, folder, "1.2.0")
    before = original.read_bytes()
    real_manifest = engine.manifest
    calls = 0

    def changed(path, version):
        nonlocal calls
        calls += 1
        if calls == 2:
            (path / "mlxl3").write_bytes(b"modified during package")
        return real_manifest(path, version)

    monkeypatch.setattr(engine, "manifest", changed)
    with pytest.raises(ValueError, match="changed"):
        engine.package(runtime, folder, "1.2.0")
    assert original.read_bytes() == before
    assert not list(folder.glob("*.partial"))


def test_manifest_symlink_is_rejected(runtime, tmp_path):
    protected = tmp_path / "protected"
    protected.write_bytes(b"keep")
    (runtime / "engine.json").symlink_to(protected)
    with pytest.raises(ValueError, match="linked"):
        engine.package(runtime, tmp_path / "dist", "1.2.0")
    assert protected.read_bytes() == b"keep"
