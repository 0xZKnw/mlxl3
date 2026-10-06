"""Build a reproducible, independently updatable native engine archive."""

import argparse
import gzip
import hashlib
import json
import plistlib
import re
import tarfile
from pathlib import Path

FILES = ("mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib")
MAX_FILE = 256 * 1024 * 1024


def valid_version(version: str) -> bool:
    """A version used in an output filename must be a canonical numeric triplet.

    post: not __return__ or (len(version.split('.')) == 3 and '/' not in version and '\\' not in version and '..' not in version)
    post: not __return__ or all(0 <= int(p) <= 9223372036854775807 for p in version.split('.'))
    """
    return bool(
        re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version)
    ) and all(len(p) <= 19 and int(p) <= 9223372036854775807 for p in version.split("."))


def manifest(runtime: Path, version: str) -> dict:
    """Read only regular runtime files; fail before publishing incomplete bundles."""
    if not valid_version(version):
        raise ValueError("invalid engine version")
    records = {}
    for name in FILES:
        source = runtime / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"missing or linked engine file: {name}")
        size = source.stat().st_size
        if not 0 < size <= MAX_FILE:
            raise ValueError(f"invalid engine file size: {name}")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records[name] = {"size": size, "sha256": digest}
    return {
        "schema": 1,
        "version": version,
        "bridge_protocol": 1,
        "architecture": "arm64",
        "minimum_macos": "26.2.0",
        # Older Desktop versions do not pass a target to the MTP downloader.
        "minimum_app_version": "1.4.0"
        if tuple(map(int, version.split("."))) >= (1, 4, 0)
        else "1.2.0",
        "files": records,
        "notices": {
            name: (Path(__file__).resolve().parents[1] / path).read_text()
            for name, path in {
                "MTPLX": "LICENSES/MTPLX-NOTICE.txt",
                "Apache-2.0": "LICENSES/Apache-2.0.txt",
            }.items()
        },
    }


def package(runtime: Path, output: Path, version: str) -> Path:
    payload = manifest(runtime, version)
    if (runtime / "engine.json").is_symlink():
        raise ValueError("linked runtime manifest")
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    (runtime / "engine.json").write_bytes(data)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"MLXL3-Engine-v{version}-arm64.tar.gz"
    temporary = destination.with_suffix(".partial")
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
        ):
            for name in ("engine.json", *FILES):
                source = runtime / name
                member = tarfile.TarInfo(name)
                member.size = source.stat().st_size
                member.mode = 0o644 if name in ("engine.json", "mlx.metallib") else 0o755
                with source.open("rb") as stream:
                    archive.addfile(member, stream)
        # A concurrently changed runtime must not be released with stale hashes.
        if payload != manifest(runtime, version):
            raise ValueError("runtime changed while packaging")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    info = Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Resources/Info.plist"
    version = args.version or plistlib.loads(info.read_bytes())["CFBundleShortVersionString"]
    print(package(args.runtime, args.output, version))


if __name__ == "__main__":
    main()
