import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "flash_reference", Path(__file__).parents[1] / "scripts/check-flash-reference.py"
)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def draft_fixture(tmp_path):
    hidden = np.ones((4, 2560), dtype="<f4")
    payload = hidden.tobytes()
    (tmp_path / "hidden.f32").write_bytes(payload)
    data = {
        "status": "complete_cpu_oracle",
        "kind": "dflash2",
        "proposals": 3,
        "residual_norm": "sglang-3c560089-fp32-variance-bf16-residual",
        "arrays": {
            "hidden": {
                "file": "hidden.f32",
                "shape": [4, 2560],
                "count": 10240,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    (tmp_path / "oracle.json").write_text(json.dumps(data))
    return data


def test_draft_head_input_preserves_real_rows_and_dflash_anchor_offset(tmp_path):
    data = draft_fixture(tmp_path)
    _, hidden = oracle.draft_head_input(tmp_path)
    assert hidden.shape == (3, 2560) and np.array_equal(hidden, np.ones((3, 2560)))
    data.update(kind="dspark", proposals=4)
    (tmp_path / "oracle.json").write_text(json.dumps(data))
    _, hidden = oracle.draft_head_input(tmp_path)
    assert hidden.shape == (4, 2560)


@pytest.mark.parametrize("corrupt", ["hash", "size", "empty", "shape", "path", "old_norm", "nan"])
def test_draft_head_input_rejects_corruption_and_incomplete_chain(tmp_path, corrupt):
    data = draft_fixture(tmp_path)
    entry = data["arrays"]["hidden"]
    if corrupt == "hash":
        entry["sha256"] = "0" * 64
    elif corrupt == "size":
        (tmp_path / "hidden.f32").write_bytes(b"\0" * 40961)
    elif corrupt == "empty":
        data["status"] = "failed"
    elif corrupt == "shape":
        entry["shape"] = [4, 2559]
    elif corrupt == "path":
        entry["file"] = "../hidden.f32"
    elif corrupt == "old_norm":
        data["residual_norm"] = "legacy"
    else:
        payload = np.full((4, 2560), np.nan, dtype="<f4").tobytes()
        (tmp_path / "hidden.f32").write_bytes(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
    (tmp_path / "oracle.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        oracle.draft_head_input(tmp_path)


def test_oracle_tokens_bound_work_and_vocabulary():
    assert oracle.parse_tokens("0,248319") == [0, 248319]
    assert len(oracle.parse_tokens(",".join(["1"] * 32))) == 32
    for value in ("", "-1", "248320", "1,", "hello", ",".join(["1"] * 33)):
        with pytest.raises(ValueError):
            oracle.parse_tokens(value)


def test_ngram_decode_uses_circular_predecessors_and_one_half_rounding():
    rng = np.random.default_rng(771)
    words = rng.integers(0, 65536, 41, dtype=np.uint16)
    words[0] = np.float16(0.5).view(np.uint16)
    expected = []
    for i in range(160):
        code = 0
        for previous in range(4):
            symbol = (i - previous) % 160
            code |= ((int(words[1 + symbol // 4]) >> (4 * (symbol % 4))) & 15) << (4 * previous)
        bits = (code * 0x83DCD12D) & 0xFFFFFFFF
        total = 0x6400 + sum((bits >> shift) & 255 for shift in (0, 8, 16, 24))
        h = struct.unpack("<e", struct.pack("<H", total))[0]
        inv = struct.unpack("<e", struct.pack("<H", 0x1EEE))[0]
        bias = struct.unpack("<e", struct.pack("<H", 0xC931))[0]
        rounded = struct.unpack("<e", struct.pack("<e", h * inv + bias))[0]
        expected.append(rounded * 0.5)
    assert np.array_equal(oracle.decode_ngram(words), np.array(expected, dtype=np.float32))
    for bad in (words[:-1], words.astype(np.uint32)):
        with pytest.raises(ValueError):
            oracle.decode_ngram(bad)
    words[0] = np.float16(float("nan")).view(np.uint16)
    with pytest.raises(ValueError, match="nonfinite"):
        oracle.decode_ngram(words)


def tensor_file(path, name="x"):
    header = json.dumps({name: {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(
        struct.pack("<Q", len(header))
        + header
        + np.array([0x3F80, 0x4000, 0x4040, 0x4080], dtype="<u2").tobytes()
    )


def test_store_keeps_inode_reads_bf16_and_rejects_truncation(tmp_path):
    path = tmp_path / "model.safetensors"
    tensor_file(path)
    store = oracle.Store(tmp_path)
    try:
        assert np.array_equal(store.read("x", 1), [3.0, 4.0])
        for row in (-1, 2):
            with pytest.raises(ValueError):
                store.read("x", row)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"bad")
        replacement.replace(path)
        assert np.array_equal(store.read("x", 0), [1.0, 2.0])
    finally:
        store.close()
        store.close()
    tensor_file(path)
    store = oracle.Store(tmp_path)
    try:
        path.write_bytes(b"short")
        with pytest.raises(ValueError, match="truncated"):
            store.read("x")
    finally:
        store.close()


def test_duplicate_tensor_closes_every_open_descriptor(tmp_path, monkeypatch):
    tensor_file(tmp_path / "model-a.safetensors")
    tensor_file(tmp_path / "model-b.safetensors")
    closed = []
    real_close = oracle.os.close

    def close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(oracle.os, "close", close)
    with pytest.raises(ValueError, match="duplicate"):
        oracle.Store(tmp_path)
    assert len(closed) == 2


@pytest.mark.parametrize("width", [2560, 10240])
def test_probe_input_checks_shapes_counts_bytes_and_finiteness(tmp_path, width):
    path = tmp_path / "input.f32"
    entry = {"file": path.name, "count": width, "shape": [1, 1, width]}
    values = np.linspace(-1, 1, width, dtype="<f4")
    path.write_bytes(values.tobytes())
    assert np.array_equal(oracle.probe_input(tmp_path, entry, width).reshape(-1), values)
    for bad in (
        {**entry, "count": 0},
        {**entry, "shape": [1, width]},
        {**entry, "shape": [0, 1, width]},
    ):
        with pytest.raises(ValueError, match="shape/count"):
            oracle.probe_input(tmp_path, bad, width)
    for payload in (b"", values.tobytes()[:-1], values.tobytes() + b"x"):
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="bytes"):
            oracle.probe_input(tmp_path, entry, width)
    values[0] = np.nan
    path.write_bytes(values.tobytes())
    with pytest.raises(ValueError, match="nonfinite"):
        oracle.probe_input(tmp_path, entry, width)


@pytest.mark.parametrize(
    "flags",
    [
        ["--norm-only", "--projections-only"],
        ["--norm-only", "--attention-only", "fixture"],
        ["--projections-only", "--attention-only", "fixture"],
        ["--norm-only", "--mtp-only", "fixture"],
        ["--projections-only", "--mtp-only", "fixture"],
        ["--attention-only", "fixture", "--mtp-only", "fixture"],
    ],
)
def test_probe_modes_reject_conflicts_before_gpu_import(monkeypatch, flags):
    monkeypatch.setattr(oracle.sys, "argv", ["oracle", "model", "output", *flags])
    with pytest.raises(SystemExit) as error:
        oracle.main()
    assert error.value.code == 2
