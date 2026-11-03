"""Disposable archives for the production Swift updater, including hostile headers."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
names = ["mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"]
files = {name: (name + " disposable test\n").encode() for name in names}
manifest = {
    "schema": 1,
    "version": "1.2.0",
    "bridge_protocol": 1,
    "architecture": "arm64",
    "minimum_macos": "26.2.0",
    "minimum_app_version": "1.2.0",
    "files": {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    },
}
files["engine.json"] = json.dumps(manifest).encode()
for variant in [
    "valid",
    "duplicate",
    "traversal",
    "extra",
    "symlink",
    "hardlink",
    "hash",
    "size",
    "protocol",
    "huge",
    "native",
    "native-extra",
    "native-duplicate",
    "native-hash",
    "native-schema",
    "native-backend",
    "native-missing",
    "native-old-minimum",
    "native-symlink",
    "native-manifest-duplicate",
]:
    entries = dict(files)
    if variant.startswith("native"):
        entries = {"mlxl3": files["mlxl3"]}
        changed = dict(
            manifest,
            schema=2,
            backend="metal",
            version="2.0.2",
            minimum_app_version="2.0.0",
            files={"mlxl3": manifest["files"]["mlxl3"]},
        )
        if variant == "native-schema":
            changed["schema"] = 3
        if variant == "native-backend":
            changed["backend"] = "mlx"
        if variant == "native-old-minimum":
            changed["minimum_app_version"] = "1.2.0"
        if variant == "native-hash":
            changed["files"] = {"mlxl3": dict(changed["files"]["mlxl3"], sha256="0" * 64)}
        if variant == "native-extra":
            entries["libmlx.dylib"] = files["libmlx.dylib"]
        if variant == "native-missing":
            del entries["mlxl3"]
        entries["engine.json"] = json.dumps(changed).encode()
    if variant in ["hash", "size", "protocol"]:
        changed = json.loads(files["engine.json"])
        if variant == "protocol":
            changed["bridge_protocol"] = 99
        elif variant == "hash":
            changed["files"]["mlxl3"]["sha256"] = "0" * 64
        else:
            changed["files"]["mlxl3"]["size"] = 1
        entries["engine.json"] = json.dumps(changed).encode()
    if variant == "huge":
        entries["engine.json"] = b" " * 100000
    if variant == "extra":
        entries["untrusted.sh"] = b"extra"
    if variant == "traversal":
        entries["../mlxl3"] = entries.pop("mlxl3")
    with tarfile.open(
        folder / (variant + ".tar.gz"), "w:gz", format=tarfile.USTAR_FORMAT
    ) as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            if name == "mlxl3" and variant in ["symlink", "hardlink", "native-symlink"]:
                member.type = (
                    tarfile.SYMTYPE if variant in ["symlink", "native-symlink"] else tarfile.LNKTYPE
                )
                member.linkname = "libmlx.dylib"
                member.size = 0
            archive.addfile(member, io.BytesIO(data))
        if variant in ["duplicate", "native-duplicate", "native-manifest-duplicate"]:
            member = tarfile.TarInfo(
                "engine.json" if variant == "native-manifest-duplicate" else "mlxl3"
            )
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
