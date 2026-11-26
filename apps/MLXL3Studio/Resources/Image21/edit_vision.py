"""Fetch the vision tower without duplicating Studio's installed q4 language encoder."""

import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import threading
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "OsaurusAI/Qwen-Image-2.1-mflux-4bit"
REVISION = "235fd28de40d745382e9d7a33f008c7735ec21c2"
URL = f"https://huggingface.co/{SOURCE}/resolve/{REVISION}/"
# Each shard entry gives the full remote size, the size of its safetensors header
# including the length field, and the SHA-256 of those exact header bytes.
SHARDS = {
    "text_encoder/1.safetensors": (
        2145597770,
        63426,
        "ac42ac34191dc3930a4e9767b3fa173e70777cc2cba5535961ce7f67c3b1540d",
    ),
    "text_encoder/2.safetensors": (
        843423144,
        52872,
        "48f1529d9d1a22f9d94ab89760487a98a8e8edf9cf2cc2762b2d56a6a3955263",
    ),
}
CONFIGS = {
    "processor/preprocessor_config.json": (
        507,
        "29cb956472acf03f6ba6846ddcd8686d82c2a36f11cdfcda17845f95bdbd9a79",
    ),
    "text_encoder/config.json": (
        1516,
        "369fe497e51469d3fe4e1c71c45fe98ec6853ea0885176f41cf4a224d113b44a",
    ),
    "transformer/config.json": (
        369,
        "807dc9508f46716e94e96815186b7a1977a6023f382998a86b6eae96f88fa22f",
    ),
    "vae/config.json": (
        2078,
        "fa1ef934cc3bc905c8db73980f161639a0cac48cee4b09ee428d7d52b14412fd",
    ),
}
VISION_BYTES = 519866792
VISION_TENSORS = 532
VISION_FILE = "vision.safetensors"
VISION_FILE_BYTES = 519923440
VISION_SHA256 = "54e6811474b39c2fa78d27e239d975ef9cae575d6413fb735a02631dcfaea446"
CHUNK_SIZE = 1024**2


def _copy_exact(response, out, count):
    while count:
        part = response.read(min(CHUNK_SIZE, count))
        if not part:
            raise OSError("Incomplete Qwen Image vision download")
        out.write(part)
        count -= len(part)
    if response.read(1):
        raise OSError("Oversized Qwen Image vision response")


def _get_range(path, size, start, end, out):
    if not 0 <= start < end <= size:
        raise ValueError("Invalid Qwen Image vision byte range")
    first, last = start, end - 1
    query = Request(URL + path, headers={"Range": f"bytes={first}-{last}"})
    with urlopen(query, timeout=45) as response:
        if response.status != 206 or response.headers.get("Content-Range") != (
            f"bytes {first}-{last}/{size}"
        ):
            raise OSError("Server did not honor the pinned Qwen Image vision byte range")
        _copy_exact(response, out, end - start)


def _plan_vision(shards):
    """Return the new header and contiguous source runs in output order."""
    header = {}
    ranges = []
    cursor = 0
    for source, size, length, details in shards:
        keys = sorted(
            (key for key in details if key.startswith("visual.")),
            key=lambda key: details[key]["data_offsets"][0],
        )
        previous = length
        for key in keys:
            start, end = (length + value for value in details[key]["data_offsets"])
            if key in header or not previous <= start < end <= size:
                raise ValueError("Invalid Qwen Image vision tensor offsets")
            header[key] = {
                **details[key],
                "data_offsets": [cursor, cursor + end - start],
            }
            if ranges and ranges[-1][0] == source and ranges[-1][3] == start:
                ranges[-1] = (source, size, ranges[-1][2], end)
            else:
                ranges.append((source, size, start, end))
            cursor += end - start
            previous = end
    return header, ranges, cursor


def _extract_vision(destination):
    shards = []
    for source, (size, length, digest) in SHARDS.items():
        capture = io.BytesIO()
        _get_range(source, size, 0, length, capture)
        raw = capture.getvalue()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Qwen Image vision shard header checksum mismatch")
        if int.from_bytes(raw[:8], "little") != length - 8:
            raise ValueError("Invalid Qwen Image vision shard header length")
        shards.append((source, size, length, json.loads(raw[8:])))
    header, ranges, payload_size = _plan_vision(shards)
    if len(header) != VISION_TENSORS or payload_size != VISION_BYTES:
        raise ValueError("Incomplete Qwen Image vision tower")
    raw_header = json.dumps(header, separators=(",", ":")).encode("ascii")
    raw_header += b" " * (-len(raw_header) % 8)
    with destination.open("wb") as out:
        out.write(len(raw_header).to_bytes(8, "little"))
        out.write(raw_header)
        for source, size, start, end in ranges:
            _get_range(source, size, start, end, out)


def _valid_asset(path, expected_size, expected_hash):
    try:
        if path.stat().st_size != expected_size:
            return False
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest() == expected_hash
    except OSError:
        return False


def _download_config(name, size, digest, target):
    with urlopen(URL + name, timeout=45) as response, target.open("wb") as output:
        _copy_exact(response, output, size)
    if not _valid_asset(target, size, digest):
        raise ValueError("Qwen Image editing configuration checksum mismatch")


def _download_missing(cache, missing):
    # Stable staging path under the lock. SIGTERM from Studio runs the cleanup
    # below. A later invocation also cleans a staging folder left by SIGKILL.
    staging = cache / ".download"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    handles_signals = threading.current_thread() is threading.main_thread()

    def canceled(sig, frame):
        raise SystemExit(128 + sig)

    if handles_signals:
        previous = signal.signal(signal.SIGTERM, canceled)
    try:
        for name in missing:
            staged = staging / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            if name == VISION_FILE:
                print("Downloading 520 MB of Qwen Image vision weights", flush=True)
                _extract_vision(staged)
                if not _valid_asset(staged, VISION_FILE_BYTES, VISION_SHA256):
                    raise ValueError("Qwen Image vision tower checksum mismatch")
            else:
                size, digest = CONFIGS[name]
                _download_config(name, size, digest, staged)
            final_path = cache / name
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final_path)
    finally:
        if handles_signals:
            signal.signal(signal.SIGTERM, previous)
        shutil.rmtree(staging)


def prepare_edit_vision(cache_dir=None):
    """Return the validated, reusable cache path. No weights are copied on a cache hit."""
    from huggingface_hub.constants import HF_ASSETS_CACHE, HF_HUB_OFFLINE

    base = Path(cache_dir) if cache_dir is not None else Path(HF_ASSETS_CACHE)
    cache = base / "mlxl3-image21-vision" / REVISION
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (cache / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assets = {**CONFIGS, VISION_FILE: (VISION_FILE_BYTES, VISION_SHA256)}
        missing = [
            name
            for name, (size, digest) in assets.items()
            if not _valid_asset(cache / name, size, digest)
        ]
        if missing:
            if HF_HUB_OFFLINE:
                raise FileNotFoundError(
                    "Qwen Image editing components are missing. Connect to the Internet for one download."
                )
            _download_missing(cache, missing)
        elif (cache / ".download").exists():
            shutil.rmtree(cache / ".download")
    return cache
