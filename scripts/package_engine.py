"""Build a reproducible, independently updatable native engine archive."""

import argparse
import gzip
import hashlib
import json
import re
import tarfile
import tomllib
from pathlib import Path

FILES = ("mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib")
NATIVE_FILES = ("mlxl3",)
MAX_FILE = 256 * 1024 * 1024


def valid_version(version: str) -> bool:
    """A version used in an output filename must be a canonical numeric triplet.

    post: not __return__ or (len(version.split('.')) == 3 and '/' not in version and '\\' not in version and '..' not in version)
    post: not __return__ or all(0 <= int(p) <= 9223372036854775807 for p in version.split('.'))
    """
    return bool(
        re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version)
    ) and all(len(p) <= 19 and int(p) <= 9223372036854775807 for p in version.split("."))


def manifest(
    runtime: Path, version: str, backend: str = "mlx", release_tag: str | None = None
) -> dict:
    """Read only regular runtime files; fail before publishing incomplete bundles.

    post: __return__["schema"] == (2 if backend == "metal" else 1)
    post: set(__return__["files"]) == set(NATIVE_FILES if backend == "metal" else FILES)
    """
    if not valid_version(version):
        raise ValueError("invalid engine version")
    if backend not in ("mlx", "metal"):
        raise ValueError("invalid engine backend")
    names = NATIVE_FILES if backend == "metal" else FILES
    if backend == "metal" and any(p.name not in (*names, "engine.json") for p in runtime.iterdir()):
        raise ValueError("unexpected file in native runtime")
    records = {}
    for name in names:
        source = runtime / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"missing or linked engine file: {name}")
        size = source.stat().st_size
        if not 0 < size <= MAX_FILE:
            raise ValueError(f"invalid engine file size: {name}")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records[name] = {"size": size, "sha256": digest}
    if release_tag is not None:
        match = re.fullmatch(
            r"engine-v([0-9]+\.[0-9]+\.[0-9]+)(?:-beta\.([1-9][0-9]*))?", release_tag
        )
        if (
            match is None
            or match[1] != version
            or (match[2] and (len(match[2]) > 19 or int(match[2]) > 9223372036854775807))
        ):
            raise ValueError("release tag must match canonical runtime version")
    minimum_app = "1.4.0" if tuple(map(int, version.split("."))) >= (1, 4, 0) else "1.2.0"
    if backend == "metal":
        minimum_app = "2.0.0"
    return {
        "schema": 2 if backend == "metal" else 1,
        **({"backend": "metal"} if backend == "metal" else {}),
        "version": version,
        **({"release_tag": release_tag} if release_tag is not None else {}),
        "bridge_protocol": 1,
        "architecture": "arm64",
        "minimum_macos": "26.2.0",
        # Older Desktop versions do not pass a target to the MTP downloader.
        "minimum_app_version": minimum_app,
        "files": records,
        "notices": {
            name: (Path(__file__).resolve().parents[1] / path).read_text()
            for name, path in {
                "MTPLX": "LICENSES/MTPLX-NOTICE.txt",
                "TensorFold": "LICENSES/TensorFold-NOTICE.txt",
                "Apache-2.0": "LICENSES/Apache-2.0.txt",
                "MLX-VLM": "LICENSES/MLX-VLM-MIT.txt",
                "PonyExl3": "apps/MLXL3Studio/Vendor/PonyExl3/NOTICE",
            }.items()
        },
    }


def package(
    runtime: Path, output: Path, version: str, backend: str = "mlx", release_tag: str | None = None
) -> Path:
    payload = manifest(runtime, version, backend, release_tag)
    if (runtime / "engine.json").is_symlink():
        raise ValueError("linked runtime manifest")
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    (runtime / "engine.json").write_bytes(data)
    output.mkdir(parents=True, exist_ok=True)
    release_version = release_tag.removeprefix("engine-v") if release_tag else version
    destination = output / f"MLXL3-Engine-v{release_version}-arm64.tar.gz"
    temporary = destination.with_suffix(".partial")
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
        ):
            for name in ("engine.json", *payload["files"]):
                source = runtime / name
                member = tarfile.TarInfo(name)
                member.size = source.stat().st_size
                member.mode = 0o644 if name in ("engine.json", "mlx.metallib") else 0o755
                with source.open("rb") as stream:
                    archive.addfile(member, stream)
        # A concurrently changed runtime must not be released with stale hashes.
        if payload != manifest(runtime, version, backend, release_tag):
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
    parser.add_argument("--release-tag", help="engine-vX.Y.Z[-beta.N] release identity")
    parser.add_argument("--backend", choices=("mlx", "metal"), default="mlx")
    args = parser.parse_args()
    version = args.version
    if version is None:
        cargo = Path(__file__).resolve().parents[1] / "Cargo.toml"
        with cargo.open("rb") as stream:
            version = tomllib.load(stream)["package"]["version"]
    print(package(args.runtime, args.output, version, args.backend, args.release_tag))


if __name__ == "__main__":
    main()
