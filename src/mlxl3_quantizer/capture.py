"""Strict, lazy reader for the native source-forward capture protocol.

This module never imports the optional MLX capture implementation. A complete
manifest attests the exact source/request, captured row prefixes and file hashes.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DTYPES = {"float16", "bfloat16", "float32"}


def captured_rows(observed: int, maximum: int) -> int:
    """Validate the actual first-N row contract.

    raises: ValueError
    post: 0 <= __return__ <= maximum
    post: __return__ <= observed
    post: observed > maximum or __return__ == observed
    """
    if type(observed) is not int or type(maximum) is not int or observed < 0 or maximum <= 0:
        raise ValueError("invalid capture row count")
    return min(observed, maximum)


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant in native capture: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate capture manifest field")
        result[key] = value
    return result


def _payload(root, record, shape, capture_dtype=None):
    if (
        not isinstance(record, dict)
        or record.get("shape") != list(shape)
        or any(type(n) is not int for n in record["shape"])
        or record.get("storage_dtype") != "float32"
        or record.get("original_dtype") not in _DTYPES
    ):
        raise ValueError("invalid native capture shape or dtype")
    name, digest = record.get("path"), record.get("sha256")
    if shape[0] == 0:
        if name is not None or digest is not None:
            raise ValueError("cold capture must not claim a payload")
        return None
    if not isinstance(name, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("invalid native capture path or hash")
    relative = Path(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in ("..", ".") for p in relative.parts)
    ):
        raise ValueError("capture payload escapes its directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in native capture payload")
    if not path.is_file() or path.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError("incomplete native capture payload")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    if sha.hexdigest() != digest:
        raise ValueError("native capture payload hash mismatch")
    # Validate finite rows in chunks, then release this mapping. Consumers open
    # only their active module instead of retaining thousands of descriptors.
    values = np.memmap(path, dtype="<f4", mode="r", shape=shape)
    for start in range(0, shape[0], 256):
        chunk = values[start : start + 256]
        if not np.isfinite(chunk).all():
            raise ValueError("nonfinite native capture payload")
        if capture_dtype == "float16":
            with np.errstate(over="ignore", invalid="ignore"):
                rounded = chunk.astype(np.float16).astype(np.float32)
            if not np.array_equal(chunk.view(np.uint32), rounded.view(np.uint32)):
                raise ValueError("native capture missed its requested FP16 rounding")
    return path


class NativeCapture(Mapping):
    """Validated capture map; each lookup owns only that module's read-only mmap."""

    def __init__(self, root, manifest, files):
        self.root = root
        self.manifest = manifest
        self._files = files

    def __iter__(self):
        return iter(self.manifest["projections"])

    def __len__(self):
        return len(self.manifest["projections"])

    def __getitem__(self, key):
        record = self.manifest["projections"][key]
        path = self._files[key]
        shape = tuple(record["shape"])
        if path is None:
            return np.empty(shape, dtype=np.float32)
        # Recheck the immutable-file contract before serving a module. Captures
        # are private output directories, but later corruption must not pass.
        _payload(self.root, record, shape, self.manifest["request"]["capture_dtype"])
        return np.memmap(path, dtype="<f4", mode="r", shape=shape)

    def teacher_logits(self):
        record = self.manifest.get("logits")
        if record is None:
            return None
        shape = tuple(record["shape"])
        path = _payload(self.root, record, shape)
        return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def read_native_capture(
    directory,
    *,
    source_identity,
    request,
    features,
    segment_lengths,
    vocab_size=None,
):
    """Reject partial/foreign native captures and return a lazy activation map.

    ``features`` and ``segment_lengths`` come from the caller's checkpoint and
    token request, independently of the child manifest. Missing expert rows stay
    explicit zero-row records for the adapter's existing coverage policy.
    """
    root = Path(directory).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid native capture directory")
    manifest_path = root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > 16 * 1024 * 1024
    ):
        raise ValueError("missing or oversized native capture manifest")
    manifest = json.loads(
        manifest_path.read_text(),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema")) is not int
        or manifest["schema"] != 1
        or manifest.get("status") != "complete"
        or manifest.get("source") != source_identity
        or manifest.get("request") != request
    ):
        raise ValueError("incomplete or foreign native capture manifest")
    weights = source_identity.get("weights") if isinstance(source_identity, dict) else None
    if (
        not isinstance(source_identity, dict)
        or not isinstance(source_identity.get("path"), str)
        or not Path(source_identity["path"]).is_absolute()
        or not isinstance(source_identity.get("config_sha256"), str)
        or not _SHA256.fullmatch(source_identity["config_sha256"])
        or not isinstance(weights, list)
        or not weights
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or Path(item["name"]).name != item["name"]
            or not item["name"].endswith(".safetensors")
            or type(item.get("size")) is not int
            or item["size"] <= 0
            or type(item.get("mtime_ns")) is not int
            for item in weights
        )
        or len({item["name"] for item in weights}) != len(weights)
    ):
        raise ValueError("invalid native capture source identity")
    if (
        not isinstance(request, dict)
        or request.get("capture_dtype") not in {"float16", "float32"}
        or type(request.get("logits")) is not bool
        or not all(
            isinstance(request.get(key), str) and _SHA256.fullmatch(request[key])
            for key in ["tokens_sha256", "modules_sha256"]
        )
        or not features
        or any(not isinstance(key, str) or not key for key in features)
        or any(type(width) is not int or width <= 0 for width in features.values())
        or not segment_lengths
        or any(type(n) is not int or n <= 0 for n in segment_lengths)
    ):
        raise ValueError("invalid native capture request")
    maximum = request.get("max_rows")
    captured_rows(0, maximum)
    projections = manifest.get("projections")
    if not isinstance(projections, dict) or projections.keys() != features.keys():
        raise ValueError("native capture module coverage mismatch")
    files = {}
    full = True
    for key, width in features.items():
        record = projections[key]
        if not isinstance(record, dict) or "shape" not in record:
            raise ValueError("invalid native projection record")
        rows = captured_rows(record.get("observed_rows"), maximum)
        files[key] = _payload(root, record, (rows, width), request["capture_dtype"])
        full = full and rows == maximum
    paths = [path for path in files.values() if path is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("native projections alias a capture payload")
    processed = manifest.get("processed_segment_lengths")
    if (
        not isinstance(processed, list)
        or not processed
        or len(processed) > len(segment_lengths)
        or any(type(n) is not int or n <= 0 for n in processed)
        or processed[:-1] != list(segment_lengths[: len(processed) - 1])
        or processed[-1] > segment_lengths[len(processed) - 1]
        or (processed != list(segment_lengths) and (request["logits"] or not full))
    ):
        raise ValueError("native capture segment prefix mismatch")
    if any(record["observed_rows"] > sum(processed) for record in projections.values()):
        raise ValueError("native capture rows exceed processed token count")
    teacher = manifest.get("logits")
    if request["logits"]:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("teacher capture requires the checkpoint vocabulary size")
        path = _payload(root, teacher, (sum(processed), vocab_size))
        if path in paths:
            raise ValueError("teacher logits alias a projection payload")
    elif teacher is not None:
        raise ValueError("unrequested teacher logits")
    return NativeCapture(root, manifest, files)
