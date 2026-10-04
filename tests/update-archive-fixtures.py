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
]:
    entries = dict(files)
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
            if name == "mlxl3" and variant in ["symlink", "hardlink"]:
                member.type = tarfile.SYMTYPE if variant == "symlink" else tarfile.LNKTYPE
                member.linkname = "libmlx.dylib"
                member.size = 0
            archive.addfile(member, io.BytesIO(data))
        if variant == "duplicate":
            member = tarfile.TarInfo("mlxl3")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
