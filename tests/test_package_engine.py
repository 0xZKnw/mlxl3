"""Release archives must be reproducible, complete and independently verifiable."""

import hashlib
import importlib.util
import json
import plistlib
import shutil
import subprocess
import sys
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


@pytest.fixture
def cli_project(tmp_path):
    root = tmp_path / "copied-project"
    script = root / "scripts/package_engine.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(engine.__file__), script)
    (root / "Cargo.toml").write_text('[package]\nname = "engine-fixture"\nversion = "1.4.5"\n')
    info = root / "apps/MLXL3Studio/Resources/Info.plist"
    info.parent.mkdir(parents=True)
    info.write_bytes(plistlib.dumps({"CFBundleShortVersionString": "1.4.4"}))
    notices = root / "LICENSES"
    notices.mkdir()
    for name in ("MTPLX-NOTICE.txt", "Apache-2.0.txt"):
        shutil.copyfile(
            Path(engine.__file__).resolve().parents[1] / "LICENSES" / name, notices / name
        )
    return root


def run_package_cli(root, *arguments):
    return subprocess.run(
        [sys.executable, str(root / "scripts/package_engine.py"), *map(str, arguments)],
        cwd=root.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def test_cli_default_uses_engine_cargo_version_and_verifiable_archive(cli_project, runtime):
    output = cli_project / "dist"
    result = run_package_cli(cli_project, runtime, output)
    assert result.returncode == 0, result.stderr
    archive_path = output / "MLXL3-Engine-v1.4.5-arm64.tar.gz"
    assert result.stdout.strip() == str(archive_path)
    assert list(output.iterdir()) == [archive_path]
    with tarfile.open(archive_path) as archive:
        payload = json.load(archive.extractfile("engine.json"))
        assert payload["version"] == "1.4.5"
        assert payload == json.loads((runtime / "engine.json").read_text())
        assert set(archive.getnames()) == {"engine.json", *engine.FILES}
        for name in engine.FILES:
            data = archive.extractfile(name).read()
            assert data == (runtime / name).read_bytes()
            assert payload["files"][name] == {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }


@pytest.mark.parametrize("cargo", [None, b"[broken", b'[package]\nversion = "../bad"\n'])
def test_cli_explicit_version_does_not_read_cargo_or_desktop(cli_project, runtime, cargo):
    path = cli_project / "Cargo.toml"
    if cargo is None:
        path.unlink()
    else:
        path.write_bytes(cargo)
    (cli_project / "apps/MLXL3Studio/Resources/Info.plist").unlink()
    output = cli_project / "dist"
    result = run_package_cli(cli_project, runtime, output, "--version", "2.3.4")
    assert result.returncode == 0, result.stderr
    archive_path = output / "MLXL3-Engine-v2.3.4-arm64.tar.gz"
    assert result.stdout.strip() == str(archive_path)
    with tarfile.open(archive_path) as archive:
        assert json.load(archive.extractfile("engine.json"))["version"] == "2.3.4"


@pytest.mark.parametrize(
    "cargo",
    [
        None,
        b"[broken",
        b"\xff",
        b"",
        b"[package]\nname='fixture'\n",
        b"package=[]\n",
        b"[package]\nversion=145\n",
        b"[package]\nversion=[]\n",
        b"[package]\nversion='../1.4.5'\n",
        b"[package]\nversion='1.4'\n",
        b"[package]\nversion='01.4.5'\n",
        b"[package]\nversion=''\n",
    ],
)
def test_cli_bad_cargo_never_publishes(cli_project, runtime, cargo):
    path = cli_project / "Cargo.toml"
    if cargo is None:
        path.unlink()
    else:
        path.write_bytes(cargo)
    output = cli_project / "dist"
    result = run_package_cli(cli_project, runtime, output)
    assert result.returncode != 0
    assert result.stderr
    assert not output.exists()
    assert not (runtime / "engine.json").exists()


@pytest.mark.parametrize("version", ["", "v1.4.5", "../1.4.5", "1.4.5-beta"])
def test_cli_invalid_override_never_reads_default_or_publishes(cli_project, runtime, version):
    (cli_project / "Cargo.toml").unlink()
    output = cli_project / "dist"
    result = run_package_cli(cli_project, runtime, output, "--version", version)
    assert result.returncode != 0
    assert "invalid engine version" in result.stderr
    assert not output.exists()
    assert not (runtime / "engine.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [[], ["runtime"], ["runtime", "dist", "--version"], ["runtime", "dist", "--unknown"]],
)
def test_cli_argument_errors_never_publish(cli_project, arguments):
    result = run_package_cli(cli_project, *arguments)
    assert result.returncode == 2
    assert "usage:" in result.stderr and "error:" in result.stderr
    assert not (cli_project / "dist").exists()
    assert not (cli_project.parent / "dist").exists()


def test_cli_incomplete_runtime_never_publishes(cli_project, runtime):
    (runtime / "mlxl3").unlink()
    output = cli_project / "dist"
    result = run_package_cli(cli_project, runtime, output)
    assert result.returncode != 0
    assert "missing or linked engine file" in result.stderr
    assert not output.exists()
    assert not (runtime / "engine.json").exists()
