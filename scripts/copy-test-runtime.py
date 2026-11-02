"""Copy a verified installed runtime into the disposable Desktop test bundle."""

import hashlib
import json
import os
import plistlib
import re
import shutil
import sys
from pathlib import Path


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError("Invalid runtime version")
    return tuple(map(int, value.split(".")))


def copy_runtime(bundle):
    bundle = Path(bundle)
    info = plistlib.loads((bundle / "Contents/Info.plist").read_bytes())
    engines = Path.home() / "Library/Application Support/io.mlxl3.desktop/Engines"
    pointer = engines / "current.json"
    current = None
    if pointer.exists():
        current = json.loads(pointer.read_text())["version"]
        version(current)
        source = engines / ("v" + current)
        if (source / "disabled").exists():
            raise ValueError("The active runtime is disabled; select a working engine first")
    else:
        source = Path("/Applications/MLXL3 Desktop.app/Contents/Resources/runtime")
    if source.is_symlink():
        raise ValueError("Runtime directory must not be a symlink")
    manifest = json.loads((source / "engine.json").read_text())
    version(manifest["version"])
    if current is not None and manifest["version"] != current:
        raise ValueError("Runtime version does not match the active pointer")
    names = {"mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"}
    if (
        manifest.get("schema") != 1
        or manifest.get("bridge_protocol") != 1
        or manifest.get("architecture") != "arm64"
        or set(manifest.get("files", {})) != names
        or version(manifest["minimum_app_version"]) > version(info["CFBundleShortVersionString"])
    ):
        raise ValueError("Installed runtime is incompatible with this Desktop build")
    for name in sorted(names):
        path = source / name
        record = manifest["files"][name]
        if type(record.get("size")) is not int or not 0 < record["size"] <= 256 * 1024 * 1024:
            raise ValueError("Invalid runtime file size: " + name)
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(
            r"[a-f0-9]{64}", record["sha256"]
        ):
            raise ValueError("Invalid runtime file digest: " + name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError("Invalid runtime file: " + name)
        if name == "mlxl3" and not os.access(path, os.X_OK):
            raise ValueError("Runtime executable is not executable")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError("Runtime hash mismatch: " + name)
    destination = bundle / "Contents/Resources/runtime"
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(names | {"engine.json"}):
        shutil.copy2(source / name, destination / name)
    for notice in source.glob("*NOTICE*"):
        if notice.is_file() and not notice.is_symlink():
            shutil.copy2(notice, destination / notice.name)
    print("Verified test runtime:", manifest["version"], "from", source)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: copy-test-runtime.py TEST_BUNDLE")
    copy_runtime(sys.argv[1])
