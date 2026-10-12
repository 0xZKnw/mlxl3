"""Numerical/format regression checks for the disposable Q4 head converter."""

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "draft_head", Path(__file__).parents[1] / "scripts/experimental-draft-head.py"
)
draft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft)


def test_compact_id_padding_and_invalid_subsets():
    assert draft.compact_draft_ids([249, 128, 64], 256) == list(range(61)) + [64, 128, 249]
    for ids, vocab in [
        ([], 256),
        ([-1], 256),
        ([256], 256),
        ([1, 1], 256),
        ([True], 256),
        ([0], 64),
        ([0], 255),
        (list(range(250)), 256),
    ]:
        with pytest.raises(ValueError, match="strict subset"):
            draft.compact_draft_ids(ids, vocab)


def test_pruning_command_rows_failures_and_transaction(tmp_path, monkeypatch):
    import sys

    from safetensors import SafetensorError
    from safetensors.numpy import load_file, save_file

    source = tmp_path / "source"
    source.mkdir()
    metadata = {"input": 128, "output": 256, "bits": 4, "group_size": 64, "mode": "affine"}
    (source / "draft_head_q4.json").write_text(json.dumps(metadata))
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"original MTP fixture")
    original = {
        "lm_head.weight": np.arange(256 * 16, dtype=np.uint32).reshape(256, 16),
        "lm_head.scales": np.ones((256, 2), np.float16),
        "lm_head.biases": np.arange(512, dtype=np.float16).reshape(256, 2),
    }
    payload = source / "draft_head_q4.safetensors"
    save_file(original, payload)
    before = payload.read_bytes()
    vocabulary = tmp_path / "ids.txt"
    vocabulary.write_text("249 128 64")
    output = tmp_path / "compact"
    monkeypatch.setattr(
        sys, "argv", [draft.__file__, str(source), str(output), "--draft-vocab", str(vocabulary)]
    )
    draft.main()
    ids = list(range(61)) + [64, 128, 249]
    compact = load_file(output / "draft_head_q4.safetensors")
    for name in original:
        np.testing.assert_array_equal(compact[name], original[name][ids])
    activation = json.loads((output / "draft_head_q4.json").read_text())
    assert activation["output"] == 64 and activation["token_ids"] == ids
    assert (output / "config.json").resolve() == source / "config.json"
    assert (output / "model.safetensors").resolve() == source / "model.safetensors"
    with pytest.raises(FileExistsError):
        draft.main()
    assert payload.read_bytes() == before
    bad_output = tmp_path / "failed"
    actual_open = Path.open

    def interrupted(path, *args, **kwargs):
        if path == bad_output / "draft_head_q4.json":
            raise KeyboardInterrupt("interrupted activation")
        return actual_open(path, *args, **kwargs)

    with monkeypatch.context() as patches:
        patches.setattr(Path, "open", interrupted)
        with pytest.raises(KeyboardInterrupt, match="activation"):
            draft.prune_projection(source, bad_output, vocabulary)
    assert not list(bad_output.iterdir()) and payload.read_bytes() == before
    (bad_output / "config.json").write_text("preserve me")
    with pytest.raises(FileExistsError):
        draft.prune_projection(source, bad_output, vocabulary)
    assert (bad_output / "config.json").read_text() == "preserve me"
    for kind in ("truncated", "shape", "dtype", "nan", "zero", "extra"):
        data = {name: value.copy() for name, value in original.items()}
        if kind == "truncated":
            payload.write_bytes(before[:-1])
        else:
            if kind == "shape":
                data["lm_head.weight"] = data["lm_head.weight"][:-1]
            if kind == "dtype":
                data["lm_head.weight"] = data["lm_head.weight"].astype(np.float32)
            if kind == "nan":
                data["lm_head.scales"][0, 0] = np.nan
            if kind == "zero":
                data["lm_head.scales"][:] = 0
            if kind == "extra":
                data["extra"] = np.ones(1, np.float16)
            save_file(data, payload)
        with pytest.raises((ValueError, SafetensorError)):
            draft.prune_projection(source, tmp_path / kind, vocabulary)
        assert not (tmp_path / kind).exists()
    payload.write_bytes(before)
    for change in (
        {"token_ids": [0]},
        {"bits": 8},
        {"mode": "symmetric"},
        {"input": 63},
        {"output": 2**32},
    ):
        (source / "draft_head_q4.json").write_text(json.dumps({**metadata, **change}))
        with pytest.raises(ValueError):
            draft.prune_projection(source, tmp_path / "bad-meta", vocabulary)
    assert not (tmp_path / "bad-meta").exists()
    (source / "draft_head_q4.json").write_text(json.dumps(metadata))
    vocabulary.write_text("0 " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        draft.prune_projection(source, tmp_path / "huge-ids", vocabulary)
    assert not (tmp_path / "huge-ids").exists()


def scalar_inner(packed, k, mcg):
    # Independent scalar codec, circular concatenated words and explicit
    # lane ownership; no vectorized indexing/permutation from the converter.
    result = np.zeros((packed.shape[0] * 16, packed.shape[1] * 16), np.float16)
    for tr in range(packed.shape[0]):
        for tc in range(packed.shape[1]):
            raw = packed[tr, tc].view(np.uint16)
            words = [int(raw[j]) + (int(raw[j + 1]) << 16) for j in range(0, 16 * k, 2)]
            for lane in range(32):
                for cg in range(2):
                    for ri in range(4):
                        index = lane * 8 + cg * 4 + ri
                        stop = (index + 1) * k + 256 * k
                        first = (stop - 16) // 32
                        last = (stop - 1) // 32
                        word = (
                            ((words[first % len(words)] << 32) | words[last % len(words)])
                            >> ((last + 1) * 32 - stop)
                        ) & 65535
                        bits = (
                            (word * 0xCBAC1FED) if mcg else (word * 89_226_354 + 64_248_484)
                        ) & 0xFFFFFFFF
                        bits = 0x3B603B60 ^ (bits & 0x8FFF8FFF)
                        halves = np.array([bits & 65535, bits >> 16], dtype=np.uint16)
                        value = halves.view(np.float16).astype(np.float32).sum().astype(np.float16)
                        row = (lane % 4) * 2 + (ri % 2) + (8 if ri >= 2 else 0)
                        col = lane // 4 + 8 * cg
                        result[tr * 16 + row, tc * 16 + col] = value
    return result


@pytest.mark.parametrize("k", range(1, 9))
@pytest.mark.parametrize("mcg", [False, True])
def test_codec_matches_scalar_all_bitrates(k, mcg):
    packed = (
        ((np.arange(2 * 3 * 16 * k, dtype=np.uint32) * 1729 + 391) % 65536)
        .astype(np.uint16)
        .reshape(2, 3, 16 * k)
    )
    expected = scalar_inner(packed, k, mcg)
    assert np.array_equal(
        draft.decode_inner(packed, k, mcg).view(np.uint16), expected.view(np.uint16)
    )
    assert np.array_equal(draft.decode_inner(packed.view(np.int16), k, mcg), expected)


def hadamard():
    # Sylvester matrix definition, independent of the iterative butterfly.
    return np.array(
        [[(-1) ** ((i & j).bit_count()) for j in range(128)] for i in range(128)], dtype=np.float32
    ) / np.sqrt(np.float32(128))


def test_rotations_scales_orientation_and_chunks():
    packed = (
        ((np.arange(8 * 16 * 96, dtype=np.uint32) * 1729 + 391) % 65536)
        .astype(np.uint16)
        .reshape(8, 16, 96)
    )
    su = np.linspace(-1.2, 0.9, 128, dtype=np.float16)
    sv = np.linspace(0.2, 1.1, 256, dtype=np.float16)
    inner = scalar_inner(packed, 6, True).astype(np.float32)
    rotation = hadamard()
    out_rotation = np.kron(np.eye(2, dtype=np.float32), rotation)
    expected = (rotation @ inner @ out_rotation).T * su.astype(np.float32) * sv[:, None]
    full = draft.effective_weight(packed, 6, True, su, sv)
    np.testing.assert_allclose(full, expected, rtol=5e-5, atol=2e-5)
    chunks = [
        draft.effective_weight(packed[:, i : i + 8], 6, True, su, sv[i * 16 : (i + 8) * 16])
        for i in (0, 8)
    ]
    assert np.array_equal(np.concatenate(chunks), full)
    np.testing.assert_allclose(draft.rotate128(draft.rotate128(inner)), inner, atol=2e-5)


def test_chunk_boundaries_and_failures():
    for total in [128, 256, 384, 248320]:
        for chunk in [128, 4096, 65536]:
            ends = [draft.chunk_end(s, total, chunk) for s in range(0, total, chunk)]
            assert ends[-1] == total
            assert all(e % 128 == 0 for e in ends)
    for args in [
        (-128, 256, 128),
        (256, 256, 128),
        (0, 129, 128),
        (0, 128, 0),
        (0, 128, -128),
        (0, 128, 129),
        (1, 256, 128),
    ]:
        with pytest.raises(ValueError):
            draft.chunk_end(*args)
    for shape, dtype, k in [
        ((8, 8, 96), "uint32", 6),
        ((8, 8, 96), "uint16", 0),
        ((8, 8, 96), "uint16", 9),
        ((8, 8, 95), "uint16", 6),
        ((8, 0, 96), "uint16", 6),
        ((8, 96), "uint16", 6),
    ]:
        with pytest.raises(ValueError):
            draft.decode_inner(np.zeros(shape, dtype=dtype), k, True)
    with pytest.raises(ValueError):
        draft.rotate128(np.zeros((2, 127)))
    with pytest.raises(ValueError):
        draft.effective_weight(np.zeros((8, 8, 96), np.uint16), 6, True, np.ones(127), np.ones(128))


def test_exporter_real_artifact_chunks_and_no_overwrite(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from safetensors import SafetensorError

    model, output = tmp_path / "model", tmp_path / "draft"
    model.mkdir()
    packed = (
        ((np.arange(8 * 16 * 96, dtype=np.uint32) * 1729 + 391) % 65536)
        .astype(np.int16)
        .reshape(8, 16, 96)
    )
    su = np.linspace(-1.2, 0.9, 128, dtype=np.float16)
    sv = np.linspace(0.2, 1.1, 256, dtype=np.float16)
    data, header = bytearray(), {}
    for name, array, dtype in [
        ("trellis", packed, "I16"),
        ("suh", su, "F16"),
        ("svh", sv, "F16"),
        ("mcg", np.array(1, np.int32), "I32"),
    ]:
        start = len(data)
        data.extend(array.tobytes())
        header["lm_head." + name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header).encode()
    (model / "head.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        monkeypatch.setattr("sys.argv", ["converter", str(model), str(output), "--chunk", "128"])
        draft.main()
        artifact = output / "draft_head_q4.safetensors"
        original = artifact.read_bytes()
        size = struct.unpack("<Q", original[:8])[0]
        info = json.loads(original[8 : 8 + size])
        assert info["__metadata__"] == {"format": "mlx"}
        assert json.loads((output / "draft_head_q4.json").read_text()) == {
            "input": 128,
            "output": 256,
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }
        expected = mx.quantize(
            mx.array(draft.effective_weight(packed, 6, True, su, sv), dtype=mx.float16),
            group_size=64,
            bits=4,
        )
        actual = mx.load(str(artifact))
        for name, values in zip(("weight", "scales", "biases"), expected):
            assert np.array_equal(np.array(actual["lm_head." + name]), np.array(values))
        with pytest.raises(FileExistsError):
            draft.main()
        assert artifact.read_bytes() == original
        assert not list(output.glob("draft-q4-*"))

        source = model / "head.safetensors"
        valid_source = source.read_bytes()
        failed_output = tmp_path / "failed"
        monkeypatch.setattr(
            "sys.argv", ["converter", str(model), str(failed_output), "--chunk", "128"]
        )
        for invalid in (b"", struct.pack("<Q", 100_000_001), valid_source[:-1]):
            source.write_bytes(invalid)
            with pytest.raises((ValueError, SafetensorError)):
                draft.main()
            assert not (failed_output / "draft_head_q4.json").exists()
        source.write_bytes(valid_source)
        duplicate = model / "duplicate.safetensors"
        duplicate.write_bytes(valid_source)
        with pytest.raises(ValueError, match="duplicate head tensor"):
            draft.main()
        duplicate.unlink()

        def interrupted(*args, **kwargs):
            raise RuntimeError("interrupted conversion")

        with monkeypatch.context() as patches:
            patches.setattr(mx, "quantize", interrupted)
            with pytest.raises(RuntimeError, match="interrupted conversion"):
                draft.main()
        assert not (failed_output / "draft_head_q4.json").exists()
        assert not (failed_output / "draft_head_q4.safetensors").exists()
        assert not list(failed_output.glob("draft-q4-*"))

        dangling = tmp_path / "dangling"
        dangling.mkdir()
        untouched = tmp_path / "must-not-be-created"
        (dangling / "draft_head_q4.safetensors").symlink_to(untouched)
        monkeypatch.setattr("sys.argv", ["converter", str(model), str(dangling), "--chunk", "128"])
        with pytest.raises(FileExistsError):
            draft.main()
        assert not untouched.exists()
        for name in ("draft_head_q4.json", "conversion.json", "reference.json"):
            blocked = tmp_path / name
            blocked.mkdir()
            (blocked / name).write_text("preserve me")
            monkeypatch.setattr(
                "sys.argv", ["converter", str(model), str(blocked), "--chunk", "128"]
            )
            with pytest.raises(FileExistsError):
                draft.main()
            assert (blocked / name).read_text() == "preserve me"
            assert not (blocked / "draft_head_q4.safetensors").exists()
        monkeypatch.setattr("sys.argv", ["converter", str(model), str(model), "--chunk", "128"])
        with pytest.raises(ValueError, match="target checkpoint"):
            draft.main()
        assert source.read_bytes() == valid_source

        late_output = tmp_path / "late-interruption"
        actual_open = Path.open

        def late_interrupted(path, *args, **kwargs):
            if path == late_output / "draft_head_q4.json":
                raise KeyboardInterrupt("interrupted activation")
            return actual_open(path, *args, **kwargs)

        monkeypatch.setattr(
            "sys.argv", ["converter", str(model), str(late_output), "--chunk", "128"]
        )
        with monkeypatch.context() as patches:
            patches.setattr(Path, "open", late_interrupted)
            with pytest.raises(KeyboardInterrupt, match="activation"):
                draft.main()
        assert not list(late_output.iterdir())
    finally:
        mx.set_default_device(previous)
