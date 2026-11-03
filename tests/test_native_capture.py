"""Native source-capture consumer: independent metadata, boundaries and failures."""

import hashlib
import json

import numpy as np
import pytest

from mlxl3_quantizer.capture import captured_rows, read_native_capture


@pytest.fixture
def sample(tmp_path):
    source = {
        "path": "/checkpoint",
        "config_sha256": "a" * 64,
        "weights": [{"name": "model.safetensors", "size": 16, "mtime_ns": 1}],
    }
    request = {
        "tokens_sha256": "b" * 64,
        "modules_sha256": "c" * 64,
        "max_rows": 3,
        "capture_dtype": "float16",
        "logits": False,
    }
    root = tmp_path / "capture"
    root.mkdir()
    x = np.arange(12, dtype="<f4").reshape(3, 4)
    (root / "a.f32").write_bytes(x.tobytes())
    record = {
        "shape": [3, 4],
        "path": "a.f32",
        "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "storage_dtype": "float32",
        "original_dtype": "bfloat16",
        "observed_rows": 4,
    }
    manifest = {
        "schema": 1,
        "status": "complete",
        "source": source,
        "request": request,
        "projections": {"layer.a": record},
        "processed_segment_lengths": [2, 2],
    }
    args = {
        "source_identity": source,
        "request": request,
        "features": {"layer.a": 4},
        "segment_lengths": [2, 2],
    }
    return root, manifest, args


def save_and_read(sample):
    root, manifest, args = sample
    (root / "manifest.json").write_text(json.dumps(manifest))
    return read_native_capture(root, **args)


def test_complete_capture_is_lazy_readonly_and_preserves_values(sample):
    capture = save_and_read(sample)
    assert list(capture) == ["layer.a"] and len(capture) == 1
    rows = capture["layer.a"]
    assert isinstance(rows, np.memmap) and not rows.flags.writeable
    np.testing.assert_array_equal(rows, np.arange(12).reshape(3, 4))
    assert capture.teacher_logits() is None
    with pytest.raises(KeyError):
        capture["missing"]


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "status",
        "source",
        "request",
        "coverage",
        "width",
        "rows",
        "count",
        "dtype",
        "storage",
        "path",
        "absolute",
        "hash",
        "truncated",
        "nonfinite",
        "rounding",
        "symlink",
        "segments",
        "emptysegments",
        "unrequested_logits",
    ],
)
def test_foreign_incomplete_or_corrupt_captures_are_rejected(sample, fault):
    root, manifest, args = sample
    record = manifest["projections"]["layer.a"]
    if fault == "schema":
        manifest["schema"] = True
    elif fault == "status":
        manifest["status"] = "in_progress"
    elif fault == "source":
        manifest["source"] = {"path": "/other"}
    elif fault == "request":
        manifest["request"] = {**args["request"], "max_rows": 99}
    elif fault == "coverage":
        manifest["projections"]["extra"] = dict(record)
    elif fault == "width":
        record["shape"] = [3, 3]
    elif fault == "rows":
        record["shape"] = [0, 4]
    elif fault == "count":
        record["observed_rows"] = True
    elif fault == "dtype":
        record["original_dtype"] = "float64"
    elif fault == "storage":
        record["storage_dtype"] = "float16"
    elif fault == "path":
        record["path"] = "../a.f32"
    elif fault == "absolute":
        record["path"] = str((root / "a.f32").absolute())
    elif fault == "hash":
        record["sha256"] = "d" * 64
    elif fault == "truncated":
        (root / "a.f32").write_bytes(b"")
    elif fault in {"nonfinite", "rounding"}:
        data = np.arange(12, dtype="<f4")
        data[0] = np.inf if fault == "nonfinite" else np.float32(1.0001)
        (root / "a.f32").write_bytes(data.tobytes())
        record["sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    elif fault == "symlink":
        (root / "a.f32").rename(root / "other.f32")
        (root / "a.f32").symlink_to("other.f32")
    elif fault == "segments":
        manifest["processed_segment_lengths"] = [1, 2]
    elif fault == "emptysegments":
        manifest["processed_segment_lengths"] = []
    elif fault == "unrequested_logits":
        manifest["logits"] = record
    with pytest.raises(ValueError):
        save_and_read(sample)


def test_cold_projection_is_explicit_and_cannot_hide_partial_segments(sample):
    _, manifest, args = sample
    manifest["projections"]["cold.expert"] = {
        "shape": [0, 8],
        "path": None,
        "sha256": None,
        "storage_dtype": "float32",
        "original_dtype": "bfloat16",
        "observed_rows": 0,
    }
    args["features"]["cold.expert"] = 8
    capture = save_and_read(sample)
    assert capture["cold.expert"].shape == (0, 8)
    manifest["processed_segment_lengths"] = [2, 1]
    with pytest.raises(ValueError, match="prefix"):
        save_and_read(sample)


def test_stop_at_row_cap_allows_only_an_input_segment_prefix(sample):
    _, manifest, _ = sample
    manifest["processed_segment_lengths"] = [2, 1]
    manifest["projections"]["layer.a"]["observed_rows"] = 3
    assert save_and_read(sample)["layer.a"].shape == (3, 4)


def test_teacher_logits_require_all_requested_tokens_and_checkpoint_vocab(sample):
    root, manifest, args = sample
    args["request"]["logits"] = True
    args["vocab_size"] = 5
    data = np.arange(20, dtype="<f4").reshape(4, 5)
    (root / "teacher.f32").write_bytes(data.tobytes())
    manifest["logits"] = {
        "path": "teacher.f32",
        "shape": [4, 5],
        "storage_dtype": "float32",
        "original_dtype": "float32",
        "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
    }
    np.testing.assert_array_equal(save_and_read(sample).teacher_logits(), data)
    manifest["processed_segment_lengths"] = [2, 1]
    with pytest.raises(ValueError, match="prefix"):
        save_and_read(sample)


def test_payloads_are_revalidated_after_later_corruption(sample):
    root, _, _ = sample
    capture = save_and_read(sample)
    (root / "a.f32").write_bytes(b"")
    with pytest.raises(ValueError, match="incomplete"):
        capture["layer.a"]


def test_duplicate_json_keys_are_rejected(sample):
    root, manifest, args = sample
    text = json.dumps(manifest).replace('"schema": 1', '"schema": 2, "schema": 1')
    (root / "manifest.json").write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        read_native_capture(root, **args)


def test_aliasing_projection_records_are_rejected(sample):
    _, manifest, args = sample
    args["features"]["alias"] = 4
    manifest["projections"]["alias"] = dict(manifest["projections"]["layer.a"])
    with pytest.raises(ValueError, match="alias"):
        save_and_read(sample)


def test_first_n_rows_count_handles_large_python_integers():
    for observed in [0, 1, 2**64, 10**100]:
        for maximum in [1, 2**32, 10**90]:
            result = captured_rows(observed, maximum)
            assert result <= min(observed, maximum)
    for observed, maximum in [(-1, 1), (0, 0), (True, 2), (1, None)]:
        with pytest.raises(ValueError):
            captured_rows(observed, maximum)


def test_projection_cannot_claim_rows_beyond_actual_processed_tokens(sample):
    _, manifest, _ = sample
    manifest["projections"]["layer.a"]["observed_rows"] = 5
    with pytest.raises(ValueError, match="processed token"):
        save_and_read(sample)


@pytest.mark.parametrize("fault", ["empty", "duplicate", "traversal", "size", "mtime"])
def test_source_identity_is_structured_not_an_empty_attestation(sample, fault):
    _, _, args = sample
    weights = args["source_identity"]["weights"]
    if fault == "empty":
        weights.clear()
    elif fault == "duplicate":
        weights.append(dict(weights[0]))
    elif fault == "traversal":
        weights[0]["name"] = "../model.safetensors"
    elif fault == "size":
        weights[0]["size"] = True
    elif fault == "mtime":
        weights[0]["mtime_ns"] = True
    with pytest.raises(ValueError, match="source identity"):
        save_and_read(sample)


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_rejected_even_in_additional_metadata(sample, constant):
    _, manifest, _ = sample
    manifest["metrics"] = {"loss": constant}
    with pytest.raises(ValueError, match="nonfinite JSON"):
        save_and_read(sample)


@pytest.mark.parametrize("timestamp", [0, -1, 1])
def test_capture_identity_accepts_valid_epoch_and_pre_epoch_file_timestamps(sample, timestamp):
    _, _, args = sample
    args["source_identity"]["weights"][0]["mtime_ns"] = timestamp
    assert len(save_and_read(sample)) == 1
