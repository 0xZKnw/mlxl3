"""CPU contracts for the independent Flash draft oracle; no model download."""

import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_flash_drafts
from check_flash_drafts import (
    CAPTURES,
    Oracle,
    Weights,
    bf16,
    capture,
    check_native_chain,
    rope,
    save_array,
    select_cpu,
)


@pytest.mark.parametrize("scores,expected", [(np.zeros(16), 1 / 16), (np.arange(16) * 1000, 1.0)])
def test_dflash_confidence_is_stable_temperature_one(scores, expected):
    assert check_flash_drafts.candidate_confidence(scores) == pytest.approx(expected)
    assert check_flash_drafts.candidate_confidence(scores + 1e6) == pytest.approx(expected)


@pytest.mark.parametrize(
    "scores", [[], np.ones(15), np.ones(17), np.full(16, np.nan), np.full(16, np.inf)]
)
def test_dflash_confidence_rejects_incomplete_or_nonfinite_scores(scores):
    with pytest.raises(ValueError):
        check_flash_drafts.candidate_confidence(scores)


def native_chain_fixture(tmp_path, kind="dspark", count=1):
    logits = np.ones((count, 248320), dtype="<f4")
    entry = save_array(tmp_path, "native_logits", logits)
    return {
        "status": "complete_native_probe",
        "kind": kind,
        "revision": check_flash_drafts.REVISION[kind],
        "proposals": count,
        "arrays": {"native_logits": entry, "hidden": {"sha256": "a" * 64}},
        "native": {"tokens": [42] * count, "confidence": [0.5] * count},
    }, logits


@pytest.mark.parametrize("kind,count", [("dflash2", 1), ("dspark", 1), ("dspark", 7)])
def test_native_chain_compares_all_logits_and_real_confidences(tmp_path, kind, count):
    data, logits = native_chain_fixture(tmp_path, kind, count)
    confidence = [0.5] * count
    checked = check_native_chain(tmp_path, data, logits, [42] * count, confidence)
    assert checked["status"] == "passed_native_chain"
    assert checked["logit_count"] == count * 248320 and checked["relative_rmse"] == 0
    assert checked["confidence_count"] == len(confidence)
    if confidence:
        data["native"]["confidence"][0] = 0.5001
        check_native_chain(tmp_path, data, logits, [42] * count, confidence)
        data["native"]["confidence"][0] = 0.500101
        with pytest.raises(ValueError, match="confidence diverged"):
            check_native_chain(tmp_path, data, logits, [42] * count, confidence)


@pytest.mark.parametrize(
    "corrupt",
    [
        "status",
        "revision",
        "count",
        "shape",
        "values",
        "hash",
        "short",
        "extra",
        "path",
        "nan",
        "logits",
        "token",
        "bool",
        "empty",
        "confidence",
        "nonfinite",
        "range",
    ],
)
def test_native_chain_rejects_partial_or_divergent_real_outputs(tmp_path, corrupt):
    data, logits = native_chain_fixture(tmp_path)
    entry = data["arrays"]["native_logits"]
    path = tmp_path / entry["file"]
    if corrupt == "status":
        data["status"] = "incomplete"
    elif corrupt == "revision":
        data["revision"] = "foreign"
    elif corrupt == "count":
        data["proposals"] = 0
    elif corrupt == "shape":
        entry["shape"] = [248320]
    elif corrupt == "values":
        entry["count"] = 0
    elif corrupt == "hash":
        entry["sha256"] = "b" * 64
    elif corrupt in ("short", "extra"):
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] if corrupt == "short" else payload + b"x")
    elif corrupt == "path":
        entry["file"] = "../native_logits.f32"
    elif corrupt in ("nan", "logits"):
        payload = np.full_like(logits, np.nan if corrupt == "nan" else 1.01).tobytes()
        path.write_bytes(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
    elif corrupt in ("token", "bool"):
        data["native"]["tokens"] = [43 if corrupt == "token" else True]
    elif corrupt == "empty":
        data["native"]["confidence"] = []
    elif corrupt == "confidence":
        data["native"]["confidence"] = [0.25]
    elif corrupt == "nonfinite":
        data["native"]["confidence"] = [float("nan")]
    else:
        data["native"]["confidence"] = [-0.1]
    with pytest.raises(ValueError):
        check_native_chain(tmp_path, data, logits, [42], [0.5])


def test_bf16_rounding_ties_even_signed_zero_and_idempotence():
    bits = np.array(
        [0, 0x80000000, 0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32
    )
    expected = [0, 0x80000000, 0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000]
    assert bf16(bits.view(np.float32)).view(np.uint32).tolist() == expected
    rng = np.random.default_rng(43)
    for _ in range(100):
        values = rng.uniform(-1024, 1024, 257).astype(np.float32)
        rounded = bf16(values)
        assert np.array_equal(bf16(rounded), rounded)
        assert (np.abs(values - rounded) <= np.abs(values) / 256 + 1e-30).all()


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan, np.finfo(np.float32).max])
def test_bf16_rejects_nonfinite_and_overflow(value):
    with pytest.raises(ValueError):
        bf16([value])


def write_weights(path, *, dtype="BF16", shape=(2, 3), offsets=(0, 12), payload=None):
    header = json.dumps(
        {"weight": {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}}
    ).encode()
    if payload is None:
        payload = np.array([0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0], dtype="<u2").tobytes()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_bf16_weight_reader_uses_rows_and_finite_payload(tmp_path):
    path = tmp_path / "weights.safetensors"
    write_weights(path)
    weights = Weights(path)
    assert weights.finite() == 6
    assert weights("weight").tolist() == [[1, 2, 3], [4, 5, 6]]
    assert weights("weight", [1]).tolist() == [[4, 5, 6]]
    write_weights(path, payload=np.array([0x7FC0] * 6, dtype="<u2").tobytes())
    with pytest.raises(ValueError, match="nonfinite"):
        Weights(path).finite()


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": "F16"},
        {"shape": (0, 3)},
        {"offsets": (2, 14)},
        {"offsets": (0, 10)},
        {"payload": b"\0" * 10},
        {"payload": b"\0" * 13},
    ],
)
def test_weight_reader_rejects_incomplete_or_wrong_layout(tmp_path, options):
    path = tmp_path / "bad.safetensors"
    write_weights(path, **options)
    with pytest.raises(ValueError):
        Weights(path)


def target_reference(root):
    steps = []
    for step in range(5):
        layers = []
        for layer in range(48):
            name = f"s{step}-l{layer}.f32"
            values = np.repeat(np.arange(4, dtype=np.float32) + 16 * layer + step, 2560)
            values.astype("<f4").tofile(root / name)
            layers.append({"file": name, "shape": [1, 1, 10240], "count": 10240})
        steps.append({"token": step, "layers": layers})
    (root / "reference.json").write_text(
        json.dumps({"status": "complete", "mode": "eval", "steps": steps})
    )


@pytest.mark.parametrize("kind", ["dflash2", "dspark"])
def test_capture_selects_completed_layers_and_means_four_streams(tmp_path, kind):
    target_reference(tmp_path)
    actual, tokens = capture(tmp_path, CAPTURES[kind])
    assert tokens == list(range(5)) and actual.shape == (5, 12800)
    for step in range(5):
        for slot, layer in enumerate(CAPTURES[kind]):
            expected = bf16(np.float32(16 * layer + step + 1.5))
            assert np.equal(actual[step, slot * 2560 : (slot + 1) * 2560], expected).all()
    (tmp_path / "s0-l45.f32").write_bytes(b"")
    with pytest.raises(ValueError, match="invalid target capture"):
        capture(tmp_path, CAPTURES[kind])


def test_capture_rejects_missing_or_partial_validation(tmp_path):
    (tmp_path / "reference.json").write_text(
        json.dumps({"status": "incomplete", "mode": "eval", "steps": []})
    )
    with pytest.raises(ValueError, match="incomplete"):
        capture(tmp_path, CAPTURES["dflash2"])


def test_rotary_positions_preserve_energy_and_distinguish_bases():
    values = np.zeros((1, 2, 3, 128), dtype=np.float32)
    values[..., :64] = 1
    assert np.array_equal(rope(values[:, :, :1], 10000, 0), values[:, :, :1])
    for base in [10000, 10000000]:
        rotated = rope(values, base, 17)
        assert np.allclose((rotated * rotated).sum(axis=-1), 64, rtol=0.004)
    assert not np.array_equal(rope(values, 10000, 17), rope(values, 10000000, 17))


def test_two_tap_convolution_resets_at_block_start_and_uses_output_side():
    def weight(name):
        assert name == "conv.base_kernel"
        return np.stack(
            [np.ones((2, 2560), dtype=np.float32), np.full((2, 2560), 2, dtype=np.float32)]
        )

    oracle = Oracle(weight, "dflash2")
    x = np.repeat(np.array([1, 2, 4, 8], dtype=np.float32)[:, None], 2560, axis=1)
    dynamic = np.zeros((4, 640), dtype=np.float32)
    expected = np.repeat(np.array([1, 3, 6, 12], dtype=np.float32)[:, None], 2560, axis=1)
    assert np.array_equal(oracle.convolve(x, dynamic, "conv", 0), expected)
    assert np.array_equal(oracle.convolve(x, dynamic, "conv", 1), expected * 2)
    assert np.array_equal(oracle.convolve(x[:1], dynamic[:1], "conv", 0), x[:1])


def test_residual_norm_uses_unrounded_variance_and_preserves_bf16_residual():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed"
        / "residual-norm-counterexample.json"
    )
    fixture = json.loads(path.read_text())["results"][-1]
    x = np.tile(np.array(fixture["x_bits"], np.uint32).view(np.float32), 160)[None]
    residual = np.tile(np.array(fixture["residual_bits"], np.uint32).view(np.float32), 160)[None]
    expected = np.tile(np.array(fixture["guide_bits"], np.uint32).view(np.float32), 160)[None]
    oracle = Oracle(lambda _: np.ones(2560, np.float32), "dflash2")
    normalized, retained = oracle.residual_norm(x, residual, "norm")
    assert normalized.shape == retained.shape == (1, 2560)
    assert np.array_equal(normalized, expected)
    assert np.array_equal(retained, bf16(x + residual))
    assert np.count_nonzero(normalized != oracle.norm(retained, "norm")) == 320
    zeros, zero_residual = oracle.residual_norm(x * 0, residual * 0, "norm")
    assert np.array_equal(zeros, x * 0) and np.array_equal(zero_residual, residual * 0)


@pytest.mark.parametrize("shape", [(), (2560,), (0, 2560), (9, 2560), (1, 2559)])
def test_residual_norm_rejects_invalid_shapes(shape):
    oracle = Oracle(lambda _: np.ones(2560, np.float32), "dflash2")
    x = np.zeros(shape, np.float32)
    with pytest.raises(ValueError, match="shape"):
        oracle.residual_norm(x, x, "norm")


@pytest.mark.parametrize("kind", ["dflash2", "dspark"])
def test_real_selector_equations_follow_predecessor_and_use_full_edge_scale(kind):
    def weights(name, ids=None):
        if name == "candidate_selector.hidden_projection.weight":
            projection = np.zeros((256, 2560), np.float32)
            projection[0, 0] = 1
            return projection
        if name in ("candidate_selector.predecessor_codebook", "markov_head.markov_w1.weight"):
            values = np.zeros((len(ids), 256), np.float32)
            values[:, 0] = np.where(np.asarray(ids) == 1, -1, 1)
            return values
        if name == "candidate_selector.successor_codebook":
            values = np.zeros((len(ids), 256), np.float32)
            values[:, 0] = np.where(np.asarray(ids) == 1, 2, 0)
            return values
        if name == "markov_head.markov_w2.weight":
            values = np.zeros(248320, np.float32)
            values[1] = 2
            return np.broadcast_to(values[:, None], (248320, 256))
        if name == "confidence_head.proj.weight":
            values = np.zeros((1, 2816), np.float32)
            values[0, 2560] = 1
            return values
        assert name == "confidence_head.proj.bias"
        return np.zeros(1, np.float32)

    hidden = np.zeros((2, 2560), np.float32)
    hidden[:, 0] = 1
    logits = np.zeros((2, 248320), np.float32)
    logits[:, 2] = 1.5
    tokens, confidence = select_cpu(weights, kind, hidden, logits, 7)
    assert tokens == [1, 2]
    if kind == "dspark":
        assert confidence == pytest.approx([0.7310585786300049, 0.2689414213699951])
    else:
        assert confidence == pytest.approx(
            [
                np.exp(2) / (np.exp(2) + np.exp(1.5) + 14),
                np.exp(1.5) / (np.exp(-2) + np.exp(1.5) + 14),
            ]
        )


@pytest.mark.parametrize("bad", ["empty", "width", "hidden", "nan", "kind", "anchor"])
def test_coupled_selector_rejects_invalid_inputs_before_reading_weights(bad):
    hidden = np.ones((1, 2560), np.float32)
    logits = np.zeros((1, 248320), np.float32)
    kind, anchor = "dflash2", 7
    if bad == "empty":
        logits = logits[:0]
    elif bad == "width":
        logits = logits[:, :-1]
    elif bad == "hidden":
        hidden = hidden[:, :-1]
    elif bad == "nan":
        logits[0, 0] = np.nan
    elif bad == "kind":
        kind = "foreign"
    else:
        anchor = 248320

    def weights(*_):
        raise AssertionError("invalid input reached weights")

    with pytest.raises(ValueError):
        select_cpu(weights, kind, hidden, logits, anchor)


@pytest.mark.parametrize("value", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_oracle_never_saves_empty_or_nonfinite_success(tmp_path, value):
    with pytest.raises(ValueError):
        save_array(tmp_path, "bad", value)
    assert not list(tmp_path.iterdir())


def test_saved_arrays_have_exact_shape_count_and_hash(tmp_path):
    actual = np.arange(12, dtype=np.float32).reshape(3, 4)
    entry = save_array(tmp_path, "sample", actual)
    assert entry["shape"] == [3, 4] and entry["count"] == 12 and len(entry["sha256"]) == 64
    assert np.array_equal(np.fromfile(tmp_path / entry["file"], dtype="<f4").reshape(3, 4), actual)


@pytest.mark.parametrize("failure", [ValueError("incomplete model"), KeyboardInterrupt()])
def test_cli_records_failure_or_interruption_without_success(tmp_path, monkeypatch, failure):
    class FakeWeights:
        def __init__(self, _):
            pass

        def finite(self):
            return 603065600

    class InterruptedOracle:
        def __init__(self, *_):
            pass

        def append(self, _):
            return [None] * 5

        def forward(self, *_):
            raise failure

    output = tmp_path / "oracle"
    monkeypatch.setattr(
        sys, "argv", ["oracle", "model", "draft", "reference", str(output), "--kind", "dflash2"]
    )
    monkeypatch.setattr(check_flash_drafts, "Weights", FakeWeights)
    monkeypatch.setattr(check_flash_drafts, "Oracle", InterruptedOracle)
    monkeypatch.setattr(
        check_flash_drafts,
        "capture",
        lambda *_: (np.ones((5, 12800), dtype=np.float32), list(range(5))),
    )
    monkeypatch.setattr(
        check_flash_drafts, "embeddings", lambda *_: np.ones((4, 2560), dtype=np.float32)
    )
    with pytest.raises(type(failure)):
        check_flash_drafts.main()
    report = json.loads((output / "oracle.json").read_text())
    assert report["status"] == "failed_or_interrupted" and report["arrays"] == {}
    assert report["error"] == type(failure).__name__
    assert "parity" not in report


def test_cli_refuses_to_overwrite_a_completed_oracle(tmp_path, monkeypatch):
    output = tmp_path / "oracle"
    output.mkdir()
    previous = '{"status":"complete_cpu_oracle"}\n'
    (output / "oracle.json").write_text(previous)
    monkeypatch.setattr(
        sys, "argv", ["oracle", "model", "draft", "reference", str(output), "--kind", "dflash2"]
    )
    with pytest.raises(FileExistsError):
        check_flash_drafts.main()
    assert (output / "oracle.json").read_text() == previous


def diagnostic_fixture(category):
    names = (
        [name for i in range(5) for name in [f"keys-{i}", f"values-{i}"]]
        + [f"layer-{i}" for i in range(5)]
        + ["hidden", "captured"]
    )
    oracle = {
        "status": "complete_cpu_oracle",
        "arrays": {name: {"count": 100} for name in names},
        "selector_tokens": [25, 18, 32],
        "selector_confidence": [0.5] * 3,
    }

    def comparison(name, tolerance):
        return {"name": name, "count": 100, "relative_rmse": 0.001, "tolerance": tolerance}

    data = {
        "kind": "Dflash2",
        "status": "passed",
        "memory": {"process_lifetime_peak_bytes": 1 << 30},
    }
    if category == "numpy":
        data.update(
            checks=[
                comparison(name, 0.01 if name.startswith(("keys", "values")) else 0.02)
                for name in names[:-1]
            ],
            selector_tokens=[25, 18, 32],
            confidence=[0.5] * 3,
        )
    elif category == "prefix":
        data.update(
            prefix_cases=15,
            widths=[1, 2, 4, 8],
            captures=comparison("captured", 0.004),
            target_logits_and_states="byte_exact",
        )
    else:
        data.update(
            status="passed_quality_not_speed",
            width=3,
            results=[
                {
                    "budget": n,
                    "accepted": 0,
                    "proposed": 0 if n <= 2 else 3,
                    "blocks": 0 if n <= 2 else 1,
                    "target_offset": 64 + n - 1,
                    "draft_seconds": 0.01,
                    "verify_seconds": 0.02,
                    "decode_seconds": 0 if n == 1 else 0.03,
                }
                for n in [1, 2, 3, 17, 64]
            ],
        )
    return data, oracle


def check_diagnostic(category, data, oracle):
    marker = {
        "numpy": "FLASH_DRAFT_NUMPY ",
        "prefix": "FLASH_DRAFT_PREFIX ",
        "greedy": "FLASH_DRAFT_GREEDY ",
    }[category]
    return check_flash_drafts.native_diagnostic(
        marker + json.dumps(data), "dflash2", category, oracle
    )


@pytest.mark.parametrize(
    "bad", [None, "empty", "foreign", "hash", "count", "nan", "logits", "tokens"]
)
def test_coupled_diagnostic_requires_nonempty_real_head_and_proposal_parity(bad):
    data, oracle = diagnostic_fixture("numpy")
    oracle.update(proposals=3, revision=check_flash_drafts.REVISION["dflash2"])
    oracle["arrays"]["hidden"]["sha256"] = "a" * 64
    coupled = {
        "kind": "dflash2",
        "revision": oracle["revision"],
        "hidden_sha256": "a" * 64,
        "tokens": [25, 18, 32],
    }
    data["coupled"] = {
        "tokens": [25, 18, 32],
        "cpu_hidden_tokens": [25, 18, 32],
        "confidence": [0.5] * 3,
        "head": {
            "name": "coupled_logits",
            "count": 3 * 248320,
            "relative_rmse": 0.001,
            "tolerance": 0.004,
        },
    }
    if bad == "empty":
        coupled["tokens"] = data["coupled"]["tokens"] = data["coupled"]["cpu_hidden_tokens"] = []
    elif bad == "foreign":
        coupled["kind"] = "dspark"
    elif bad == "hash":
        coupled["hidden_sha256"] = "b" * 64
    elif bad == "count":
        data["coupled"]["head"]["count"] = 0
    elif bad == "nan":
        data["coupled"]["head"]["relative_rmse"] = float("nan")
    elif bad == "logits":
        data["coupled"]["head"]["relative_rmse"] = 0.005
    elif bad == "tokens":
        data["coupled"]["tokens"][0] = 26

    def check():
        return check_flash_drafts.native_diagnostic(
            "FLASH_DRAFT_NUMPY " + json.dumps(data), "dflash2", "numpy", oracle, coupled=coupled
        )

    if bad is None:
        assert check() == data
    else:
        with pytest.raises(ValueError):
            check()


@pytest.mark.parametrize("category", ["numpy", "prefix", "greedy"])
def test_native_diagnostics_reject_partial_foreign_and_nonfinite_reports(category):
    data, oracle = diagnostic_fixture(category)
    assert check_diagnostic(category, data, oracle) == data
    changes = [
        (["status"], "incomplete"),
        (["kind"], "Dspark"),
        (["memory", "process_lifetime_peak_bytes"], 0),
        (["memory", "process_lifetime_peak_bytes"], (10 << 30) + 1),
        (["memory", "process_lifetime_peak_bytes"], True),
    ]
    if category == "numpy":
        changes += [
            (["checks"], []),
            (["checks", 0, "count"], 0),
            (["checks", 0, "count"], True),
            (["checks", 0, "relative_rmse"], float("nan")),
            (["checks", 0, "relative_rmse"], 0.011),
            (["checks", 0, "tolerance"], 0.1),
            (["selector_tokens"], []),
            (["selector_tokens", 0], 26),
            (["confidence"], [float("nan")]),
        ]
    elif category == "prefix":
        changes += [
            (["prefix_cases"], 0),
            (["prefix_cases"], 14),
            (["prefix_cases"], True),
            (["widths"], [1, 2, 4]),
            (["widths", 0], True),
            (["captures", "count"], 0),
            (["captures", "relative_rmse"], 0.005),
            (["target_logits_and_states"], "text_only"),
        ]
    else:
        changes += [
            (["width"], 7),
            (["results"], []),
            (["results", 0, "budget"], True),
            (["results", 0, "accepted"], 1),
            (["results", 0, "proposed"], -1),
            (["results", 0, "target_offset"], 4097),
            (["results", 0, "draft_seconds"], -1),
            (["results", 0, "verify_seconds"], float("nan")),
            (["results", 0, "decode_seconds"], 0.001),
            (["results", 1, "decode_seconds"], 0),
            (["results", 2, "decode_seconds"], float("nan")),
        ]
    for keys, value in changes:
        mutated = copy.deepcopy(data)
        target = mutated
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        with pytest.raises((ValueError, TypeError, KeyError)):
            check_diagnostic(category, mutated, oracle)
    for log in ["", "FLASH_DRAFT_NUMPY {}\nFLASH_DRAFT_NUMPY {}"]:
        with pytest.raises(ValueError):
            check_flash_drafts.native_diagnostic(log, "dflash2", category, oracle)
    with pytest.raises(ValueError, match="independent oracle"):
        check_diagnostic(category, data, {**oracle, "status": "incomplete"})
