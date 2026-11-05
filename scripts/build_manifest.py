"""Record the verified bundled engine and the separate converter environment."""

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path

from package_engine import manifest


def runtime_backend(schema: int, backend: str | None) -> str:
    """Classify the actual archive contract, independently of build flags.

    raises: ValueError
    post: __return__ in ("metal", "mlx")
    post: (__return__ == "metal") == (schema == 2 and backend == "metal")
    """
    if type(schema) is int:
        if schema == 1 and backend is None:
            return "mlx"
        if schema == 2 and backend == "metal":
            return "metal"
    raise ValueError("unknown packaged engine schema/backend")


def engine_snapshot(runtime: Path) -> dict:
    source = runtime / "engine.json"
    if (
        runtime.is_symlink()
        or source.is_symlink()
        or not source.is_file()
        or source.stat().st_size > 65536
    ):
        raise ValueError("missing or invalid packaged engine manifest")
    actual = json.loads(source.read_text())
    if not isinstance(actual, dict):
        raise TypeError("invalid packaged engine manifest")
    backend = runtime_backend(actual.get("schema"), actual.get("backend"))
    if (
        not isinstance(actual.get("version"), str)
        or type(actual.get("bridge_protocol")) is not int
        or not isinstance(actual.get("files"), dict)
        or any(
            not isinstance(record, dict) or type(record.get("size")) is not int
            for record in actual["files"].values()
        )
    ):
        raise ValueError("invalid packaged engine identity")
    expected = manifest(runtime, actual.get("version"), backend)
    if actual != expected:
        raise ValueError("packaged engine manifest or payload changed")
    return {
        "backend": backend,
        "version": actual["version"],
        "sha256": actual["files"]["mlxl3"]["sha256"],
    }


def build_info(root: Path, runtime: Path, sdk: str) -> dict:
    engine = engine_snapshot(runtime)
    info = plistlib.loads((root / "apps/MLXL3Studio/Resources/Info.plist").read_bytes())
    return {
        "version": info["CFBundleShortVersionString"],
        "build": info["CFBundleVersion"],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tracked_changes": bool(
            subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root)
        ),
        "runtime": "Rust + Metal" if engine["backend"] == "metal" else "Rust + MLX 0.32.2",
        "engine": engine,
        "quantizer_runtime": "PonyExl3 + MLX",
        "signing": "ad-hoc; not notarized",
        "swift_sdk": sdk,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sdk = (
        os.environ.get("MLXL3_MACOS_SDK")
        or subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    )
    # This is the runtime already copied into the disposable application's bundle.
    payload = build_info(root, args.destination.parent / "runtime", sdk)
    args.destination.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
