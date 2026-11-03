"""Exercise the production engine build/packaging entry point with isolated tools."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh build entry point")


@pytest.fixture
def project(tmp_path):
    for name in [
        "scripts/build-engine.sh",
        "scripts/package_engine.py",
        "Cargo.toml",
        "LICENSES/MTPLX-NOTICE.txt",
        "LICENSES/Apache-2.0.txt",
        "apps/MLXL3Studio/Vendor/PonyExl3/NOTICE",
    ]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tmp_path / "target/release/mlxl3-rs"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "print(os.environ.get('FIXTURE_RUNTIME', json.dumps({'backend':'metal',"
        "'mlx_enabled':False,'chat_enabled':True,'architecture':'aarch64'})))\n"
        "print('dyld: /System/Library/Frameworks/Metal.framework/Metal '+"
        "os.environ.get('FIXTURE_LOADED',''),file=sys.stderr)\n"
    )
    executable.chmod(0o755)
    tool = tools / "tool.py"
    tool.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "name=Path(sys.argv[0]).name\n"
        "if name=='uname': print('arm64')\n"
        "elif name=='cargo': Path(os.environ['FIXTURE_CARGO_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
        "elif name=='otool': print('/System/Library/Frameworks/Metal.framework/Metal '+os.environ.get('FIXTURE_LINKED',''))\n"
    )
    tool.chmod(0o755)
    for name in ["cargo", "uname", "codesign", "lipo", "otool"]:
        (tools / name).symlink_to(tool.name)
    env = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        MLXL3_ENGINE_BACKEND="metal",
        MLXL3_MLX_ROOT="/missing/mlx",
        MLXL3_BUILD_PYTHON=os.sys.executable,
        MLXL3_CARGO=str(tools / "cargo"),
        CARGO_TARGET_DIR=str(tmp_path / "target"),
        FIXTURE_CARGO_LOG=str(tmp_path / "cargo.json"),
    )
    return tmp_path, env


def run(project, **overrides):
    root, env = project
    return subprocess.run(
        ["zsh", str(root / "scripts/build-engine.sh")],
        env=dict(env, **overrides),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def test_native_builder_uses_chat_without_loading_mlx_and_packages_embedded_shaders(project):
    root, _ = project
    result = run(project)
    assert result.returncode == 0, result.stderr
    args = json.loads((root / "cargo.json").read_text())
    assert "--no-default-features" in args
    assert args[args.index("--features") + 1] == "direct-metal,chat"
    runtime = root / "build/rust-runtime"
    assert {p.name for p in runtime.iterdir()} == {"mlxl3", "engine.json"}
    assert json.loads((runtime / "engine.json").read_text())["backend"] == "metal"
    assert len(list((root / "dist").glob("*.tar.gz"))) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "cpu"),
        ("mlx_enabled", True),
        ("chat_enabled", False),
        ("architecture", "x86_64"),
    ],
)
def test_native_builder_rejects_mismatched_runtime_before_publication(project, field, value):
    runtime = {
        "backend": "metal",
        "mlx_enabled": False,
        "chat_enabled": True,
        "architecture": "aarch64",
    }
    runtime[field] = value
    result = run(project, FIXTURE_RUNTIME=json.dumps(runtime))
    assert result.returncode != 0
    assert not (project[0] / "dist").exists()


@pytest.mark.parametrize("variable", ["FIXTURE_LINKED", "FIXTURE_LOADED"])
@pytest.mark.parametrize("name", ["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"])
def test_native_builder_rejects_mlx_dependencies_before_publication(project, variable, name):
    result = run(project, **{variable: name})
    assert result.returncode != 0
    assert not (project[0] / "dist").exists()


def test_unknown_backend_fails_before_build_or_runtime_replacement(project):
    root, _ = project
    result = run(project, MLXL3_ENGINE_BACKEND="unknown")
    assert result.returncode != 0
    assert not (root / "cargo.json").exists()
    assert not (root / "build").exists()


def test_legacy_builder_preserves_the_explicit_mlx_bundle(project):
    root, _ = project
    mlx = root / "reference/lib"
    mlx.mkdir(parents=True)
    for name in ["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"]:
        (mlx / name).write_bytes(name.encode())
    result = run(project, MLXL3_ENGINE_BACKEND="mlx", MLXL3_MLX_ROOT=str(mlx.parent))
    assert result.returncode == 0, result.stderr
    args = json.loads((root / "cargo.json").read_text())
    assert args[args.index("--features") + 1] == "mlx,chat"
    data = json.loads((root / "build/rust-runtime/engine.json").read_text())
    assert data["schema"] == 1 and "backend" not in data
    assert set(data["files"]) == {"mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"}


def test_legacy_builder_requires_reference_libraries_before_compiling(project):
    root, _ = project
    result = run(project, MLXL3_ENGINE_BACKEND="mlx")
    assert result.returncode != 0 and "MLX natif introuvable" in result.stderr
    assert not (root / "cargo.json").exists()
