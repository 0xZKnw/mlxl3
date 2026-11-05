from __future__ import annotations

import builtins
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import diagnose_lfm25_reference as reference
import numpy as np
import pytest


def frozen_plan():
    return {"windows": {"window1": [4] * 1024, "window5": [5] * 1024}}


@pytest.fixture(autouse=True)
def synthetic_windows(monkeypatch):
    monkeypatch.setattr(
        reference,
        "WINDOW_HASHES",
        {
            name: hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest()
            for name, tokens in frozen_plan()["windows"].items()
        },
    )


def test_historical_window_validation_needs_no_numpy(monkeypatch):
    original_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    plan = frozen_plan()
    assert reference.validate_windows(plan) == plan["windows"]


def test_historical_input_and_token_domain_are_exact():
    plan = frozen_plan()
    assert reference.validate_windows(plan) == plan["windows"]
    for token in (0, 125016):
        assert reference.valid_token(token) == token
    for token in (-1, 125017, True, 1.0, "1"):
        with pytest.raises(ValueError):
            reference.valid_token(token)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "short", "wrong", "bool", "range", "notlist", "notdict"]
)
def test_incomplete_or_altered_historical_inputs_are_rejected(mutation):
    plan = frozen_plan()
    if mutation == "missing":
        del plan["windows"]["window5"]
    elif mutation == "extra":
        plan["windows"]["window2"] = plan["windows"]["window1"]
    elif mutation == "short":
        plan["windows"]["window1"].pop()
    elif mutation == "wrong":
        plan["windows"]["window1"][0] += 1
    elif mutation == "bool":
        plan["windows"]["window1"][0] = True
    elif mutation == "range":
        plan["windows"]["window1"][0] = 125017
    elif mutation == "notlist":
        plan["windows"]["window1"] = tuple(plan["windows"]["window1"])
    else:
        plan = []
    with pytest.raises(ValueError):
        reference.validate_windows(plan)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=4)
        self.device = "cpu"
        self.implementation = "grouped_mm"
        self.calls = []
        self.failure = None

    def get_experts_implementation(self):
        return {"": self.implementation}

    @contextmanager
    def _optimize_model_for_decode(self):
        before = self.implementation
        self.implementation = "batched_mm"
        try:
            yield
        finally:
            self.implementation = before

    def __call__(self, *, input_ids, past_key_values, use_cache, logits_to_keep):
        import torch

        assert logits_to_keep == 0
        self.calls.append((input_ids.tolist(), past_key_values is None, use_cache))
        logits = torch.arange(4, dtype=torch.bfloat16)[None, None].expand(1, input_ids.shape[1], 4)
        cache = past_key_values or SimpleNamespace(count=0)
        cache.count += input_ids.shape[1]
        cache.get_seq_length = lambda: cache.count
        if self.failure == "nan":
            logits = logits.float() * float("nan")
        elif self.failure == "short":
            logits = logits[..., :3]
        elif self.failure == "nocache":
            cache = None
        elif self.failure == "stale":
            cache.get_seq_length = lambda: 0
        elif self.failure == "crash":
            raise RuntimeError("fixture interrupted")
        return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)


def test_real_capture_entry_preserves_f32_file_and_fresh_decode_state(tmp_path):
    pytest.importorskip("torch")
    model = FakeModel()
    for phase in ("prefill", "decode"):
        path = tmp_path / (phase + ".npy")
        progress = []
        result = reference.capture_window(model, [2, 3, 1], phase, path, progress.append)
        values = np.load(path, allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (3, 4)
        np.testing.assert_array_equal(values, np.tile(np.arange(4), (3, 1)))
        assert result["logits_dtypes"] == ["torch.bfloat16"]
        assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert progress == [3]
    assert model.calls == [
        ([[2, 3, 1]], True, False),
        ([[2]], True, True),
        ([[3]], False, True),
        ([[1]], False, True),
    ]
    assert model.implementation == "grouped_mm"


@pytest.mark.parametrize("failure", ["nan", "short", "nocache", "stale", "crash"])
def test_capture_fails_on_bad_logits_cache_or_interruption_and_restores_backend(tmp_path, failure):
    pytest.importorskip("torch")
    model = FakeModel()
    model.failure = failure
    with pytest.raises((ValueError, RuntimeError)):
        reference.capture_window(model, [2, 1], "decode", tmp_path / "failed.npy", lambda n: None)
    assert model.implementation == "grouped_mm"


def test_capture_rejects_output_overwrite_empty_tokens_and_phase(tmp_path):
    pytest.importorskip("torch")
    model = FakeModel()
    path = tmp_path / "old.npy"
    path.write_bytes(b"keep")
    for tokens, phase, output in (
        ([1], "decode", path),
        ([], "decode", tmp_path / "empty"),
        ([1], "bad", tmp_path / "phase"),
    ):
        with pytest.raises(ValueError):
            reference.capture_window(model, tokens, phase, output, lambda n: None)
    assert path.read_bytes() == b"keep" and not model.calls


def test_real_fp32_bias_restoration_rejects_rounding_missing_bias_and_bad_shape(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    model = torch.nn.Module()
    model.device = "cpu"
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(38)])
    index, tensors = {}, {}
    for i, layer in enumerate(model.layers):
        layer.register_buffer("expert_bias", torch.zeros(64, dtype=torch.bfloat16))
        name = f"layers.{i}.expert_bias"
        index[name] = "biases.safetensors"
        tensors[name] = torch.linspace(-0.170103, 0.220137, 64) + i * 0.00001
    save_file(tensors, str(tmp_path / "biases.safetensors"))
    hashes = reference.restore_biases(model, tmp_path, index)
    assert len(hashes) == 38
    for name, original in tensors.items():
        actual = model.get_buffer(name)
        assert actual.dtype == torch.float32 and torch.equal(
            actual.view(torch.uint8), original.view(torch.uint8)
        )
    with pytest.raises(ValueError, match="38"):
        reference.restore_biases(model, tmp_path, dict(list(index.items())[:-1]))
    tensors["layers.0.expert_bias"] = tensors["layers.0.expert_bias"][:63]
    save_file(tensors, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="Invalid original"):
        reference.restore_biases(model, tmp_path, {name: "bad.safetensors" for name in index})


def test_real_tiny_transformers_bf16_prefill_and_decode_are_causal_and_finite(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

    config = Lfm2MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_dense_layers=1,
        num_hidden_layers=3,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=128,
        layer_types=["conv", "full_attention", "conv"],
        max_position_embeddings=128,
        tie_word_embeddings=True,
        use_expert_bias=True,
    )
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    torch.manual_seed(35)
    model = Lfm2MoeForCausalLM(config).to(torch.bfloat16).eval()
    for layer in model.model.layers[1:]:
        layer.feed_forward.expert_bias = layer.feed_forward.expert_bias.float()
    for phase in ("prefill", "decode"):
        reference.capture_window(
            model, [4, 5, 6, 7], phase, tmp_path / f"{phase}.npy", lambda n: None
        )
    a, b = [np.load(tmp_path / f"{phase}.npy") for phase in ("prefill", "decode")]
    assert np.isfinite(a).all() and np.isfinite(b).all() and a.shape == b.shape == (4, 128)
    np.testing.assert_allclose(a, b, atol=0.01, rtol=0.02)
    assert set(model.get_experts_implementation().values()) == {"eager"}


@pytest.mark.parametrize("failed", [False, True])
def test_real_run_records_complete_or_failed_and_restores_grouped_hook(
    tmp_path, monkeypatch, failed
):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.integrations import moe

    monkeypatch.setattr(reference, "validate_windows", lambda p: {"window1": [1], "window5": [2]})
    monkeypatch.setattr(reference, "load_source", lambda s, r: FakeModel())
    monkeypatch.setattr(moe, "_can_use_grouped_mm", lambda *a: True)
    monkeypatch.setattr(moe, "_grouped_mm", lambda *a: "called")
    original = moe._grouped_mm

    def capture(model, tokens, phase, path, progress):
        if failed:
            raise RuntimeError("interrupted")
        assert moe._grouped_mm(None, None, None) == "called"
        progress(1024)
        return {
            "experts_implementation": {"": "grouped_mm" if phase == "prefill" else "batched_mm"}
        }

    monkeypatch.setattr(reference, "capture_window", capture)
    folder = tmp_path / "run"
    if failed:
        with pytest.raises(RuntimeError, match="interrupted"):
            reference.run(tmp_path, {}, folder)
    else:
        result = reference.run(tmp_path, {}, folder)
        assert result["status"] == "complete" and len(result["windows"]) == 4
    assert moe._grouped_mm is original
    report = json.loads((folder / "report.json").read_text())
    assert (
        report["status"] == ("failed" if failed else "complete") and report["publication"] is False
    )
    with pytest.raises(FileExistsError):
        reference.run(tmp_path, {}, folder)


@pytest.mark.parametrize(
    "error",
    ["timeout", "incomplete", "truncated", "invalidlist", "badshape", "missing", "complete"],
)
def test_modal_worker_runs_bounded_child_and_marks_launch_failures(tmp_path, monkeypatch, error):
    pytest.importorskip("modal")
    import diagnose_lfm25_reference_modal as worker

    assert worker.VOLUME == "mlxl3-lfm25-v3-20261015"

    monkeypatch.setattr(worker, "DATA", tmp_path / "capture")
    monkeypatch.setattr(worker, "validate_windows", lambda p: {})
    commits = []
    monkeypatch.setattr(worker.volume, "commit", lambda: commits.append(True))

    def logged(command, log, timeout):
        assert timeout == 780 and command[2] == "/opt/mlxl3/diagnose_lfm25_reference.py"
        worker.DATA.mkdir()
        path = worker.DATA / "report.json"
        if error == "timeout":
            raise TimeoutError("silent child")
        records = {
            f"{phase}-{name}": {
                "shape": [1024, 128000],
                "storage_dtype": "float32",
                "logits_dtypes": ["torch.bfloat16"],
            }
            for phase in ("prefill", "decode")
            for name in ("window1", "window5")
        }
        if error == "badshape":
            records["prefill-window1"]["shape"] = [1023, 128000]
        elif error == "missing":
            del records["decode-window5"]
        path.write_text(
            "{bad"
            if error == "truncated"
            else "[]"
            if error == "invalidlist"
            else json.dumps(
                {"status": "running" if error == "incomplete" else "complete", "windows": records}
            )
        )

    monkeypatch.setattr(worker.base, "run_logged", logged)
    if error == "complete":
        report = json.loads(worker.diagnose.local({}))
        assert report["status"] == "complete"
    else:
        with pytest.raises((TimeoutError, ValueError, TypeError)):
            worker.diagnose.local({})
        report = json.loads((worker.DATA / "report.json").read_text())
        assert report["status"] == "failed" and report["publication"] is False
    if error in ("truncated", "invalidlist"):
        assert (worker.DATA / "report-first-invalid.json").read_text() == (
            "{bad" if error == "truncated" else "[]"
        )
    assert commits == [True]
    before = (worker.DATA / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        worker.diagnose.local({})
    assert before == (worker.DATA / "report.json").read_bytes()
