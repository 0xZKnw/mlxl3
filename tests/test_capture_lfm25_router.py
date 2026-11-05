from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_lfm25_router as target

torch = pytest.importorskip("torch")


def features(layer, row, shape=(1, 2048)):
    values = torch.arange(2048).remainder(19).to(torch.float16) / 16
    return (values + row + layer / 8).reshape(shape)


def quant_modules():
    modules = {}
    for layer in target.LAYERS:
        module = SimpleNamespace(
            device="cpu",
            cfg=SimpleNamespace(router_logits_bsz1=torch.zeros((1, 64), dtype=torch.float16)),
            result=(
                torch.tensor([[4, 1, 9, 3]], dtype=torch.int64),
                torch.tensor([[0.125, 0.25, 0.125, 0.5]], dtype=torch.float16),
            ),
            calls=[],
            corrupt=None,
        )

        def routing(bsz, cfg, hidden, params, *, module=module):
            module.calls.append((bsz, cfg, hidden, params))
            with torch.no_grad():
                cfg.router_logits_bsz1.copy_(
                    hidden.reshape(-1)[:64].reshape(cfg.router_logits_bsz1.shape) + 0.5
                )
                if module.corrupt is not None:
                    cfg.router_logits_bsz1[0, 0] = module.corrupt
            return module.result

        module.routing_fn = routing
        modules[layer] = module
    return modules


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.logits = torch.zeros((1, 64), dtype=torch.bfloat16)
        self.result = (
            self.logits,
            torch.tensor([[0.125, 0.25, 0.125, 0.5]], dtype=torch.float32),
            torch.tensor([[4, 1, 9, 3]], dtype=torch.int64),
        )

    def forward(self, hidden):
        with torch.no_grad():
            self.logits.copy_(hidden.reshape(1, 2048)[:, :64])
        return self.result


def teacher_gates():
    return {layer: Gate() for layer in target.LAYERS}


def call_quant(module, hidden):
    return module.routing_fn(1, module.cfg, hidden, {"sentinel": object()})


def fill_capture(capture, modules, kind, rows):
    for row in range(rows):
        capture.begin(row)
        for layer, module in modules.items():
            hidden = features(layer, row)
            if kind == "quant":
                assert call_quant(module, hidden) is module.result
            else:
                assert module(hidden.to(torch.bfloat16)) is module.result
        capture.finish()


@pytest.mark.parametrize("shape", [(1, 2048), (1, 1, 2048)])
@pytest.mark.parametrize("capture_inputs", [True, False])
def test_actual_quant_buffers_are_passive_immediate_detached_copies(shape, capture_inputs):
    modules = quant_modules()
    originals = {layer: module.routing_fn for layer, module in modules.items()}
    with target.QuantCapture(modules, 3, capture_inputs=capture_inputs) as capture:
        for row in range(3):
            capture.begin(row)
            for layer, module in modules.items():
                hidden = features(layer, row, shape).requires_grad_()
                ids_before = module.result[0].clone()
                weights_before = module.result[1].clone()
                result = call_quant(module, hidden)
                assert result is module.result
                assert torch.equal(result[0], ids_before)
                assert torch.equal(result[1], weights_before)
                assert module.calls[-1][2] is hidden
                hidden.detach().fill_(-11)
                module.cfg.router_logits_bsz1.fill_(-12)
            capture.finish()
        for layer in target.LAYERS:
            payload = capture.payload(layer)
            assert set(payload) == ({"x", "zq"} if capture_inputs else {"zq"})
            for name, values in payload.items():
                assert values.dtype == torch.float16 and values.device.type == "cpu"
                assert not values.requires_grad
                expected = torch.cat([features(layer, row) for row in range(3)])
                if name == "zq":
                    expected = expected[:, :64] + 0.5
                assert torch.equal(values, expected)
                values.fill_(-100)
                assert torch.equal(capture.payload(layer)[name], expected)
    assert all(module.routing_fn is originals[layer] for layer, module in modules.items())


def test_teacher_hook_copies_actual_raw_gate_output_and_preserves_tuple():
    gates = teacher_gates()
    original_hooks = {layer: dict(gate._forward_hooks) for layer, gate in gates.items()}
    with target.TeacherCapture(gates, 3) as capture:
        for row in range(3):
            capture.begin(row)
            for layer, gate in gates.items():
                hidden = features(layer, row).to(torch.bfloat16)
                result = gate(hidden)
                assert result is gate.result
                assert result[0] is gate.logits
                gate.logits.fill_(-13)
            capture.finish()
        for layer in target.LAYERS:
            payload = capture.payload(layer)
            assert set(payload) == {"zsource"}
            actual = payload["zsource"]
            expected = torch.cat([features(layer, row)[:, :64] for row in range(3)]).to(
                torch.bfloat16
            )
            assert actual.dtype == torch.bfloat16 and actual.device.type == "cpu"
            assert not actual.requires_grad and torch.equal(actual, expected)
            actual.fill_(-100)
            assert torch.equal(capture.payload(layer)["zsource"], expected)
    assert all(dict(gate._forward_hooks) == original_hooks[layer] for layer, gate in gates.items())


@pytest.mark.parametrize("kind", ["quant", "teacher"])
@pytest.mark.parametrize("failure", ["interrupt", "hook_error"])
def test_forward_and_hook_errors_restore_every_original_hook(kind, failure):
    modules = quant_modules() if kind == "quant" else teacher_gates()
    originals = {
        layer: module.routing_fn if kind == "quant" else dict(module._forward_hooks)
        for layer, module in modules.items()
    }
    cls = target.QuantCapture if kind == "quant" else target.TeacherCapture
    capture = cls(modules, 2)
    with pytest.raises((RuntimeError, ValueError)), capture:
        capture.begin(0)
        for layer, module in modules.items():
            if failure == "hook_error" and layer == 20:
                if kind == "quant":
                    module.cfg.router_logits_bsz1 = torch.zeros((1, 64), dtype=torch.float32)
                else:
                    module.result = (module.logits.float(), *module.result[1:])
            if kind == "quant":
                call_quant(module, features(layer, 0))
            else:
                module(features(layer, 0).to(torch.bfloat16))
        raise RuntimeError("injected forward interruption")
    for layer, module in modules.items():
        if kind == "quant":
            assert module.routing_fn is originals[layer]
        else:
            assert dict(module._forward_hooks) == originals[layer]
    with pytest.raises(ValueError):
        capture.payload(2)


@pytest.mark.parametrize("kind", ["quant", "teacher"])
@pytest.mark.parametrize(
    "failure", ["outside", "negative", "bool", "overflow", "skipped", "nested", "missing", "double"]
)
def test_step_alignment_rejects_ambiguous_or_incomplete_capture(kind, failure):
    modules = quant_modules() if kind == "quant" else teacher_gates()
    cls = target.QuantCapture if kind == "quant" else target.TeacherCapture
    with cls(modules, 2) as capture, pytest.raises(ValueError):
        module = modules[2]
        if kind == "quant":
            call = lambda: call_quant(module, features(2, 0))
        else:
            call = lambda: module(features(2, 0).to(torch.bfloat16))
        if failure == "outside":
            call()
        elif failure in {"negative", "bool", "overflow", "skipped"}:
            capture.begin({"negative": -1, "bool": True, "overflow": 2, "skipped": 1}[failure])
        else:
            capture.begin(0)
            if failure == "nested":
                capture.begin(0)
            elif failure == "missing":
                call()
                capture.finish()
            else:
                call()
                call()


@pytest.mark.parametrize("kind", ["quant", "teacher"])
@pytest.mark.parametrize("failure", ["empty", "missing", "extra", "rows_zero", "rows_bool"])
def test_capture_requires_the_complete_expected_panel_and_positive_row_count(kind, failure):
    modules = quant_modules() if kind == "quant" else teacher_gates()
    rows = 1
    if failure == "empty":
        modules.clear()
    elif failure == "missing":
        del modules[39]
    elif failure == "extra":
        modules[40] = modules[2]
    elif failure == "rows_zero":
        rows = 0
    else:
        rows = True
    cls = target.QuantCapture if kind == "quant" else target.TeacherCapture
    with pytest.raises(ValueError), cls(modules, rows):
        pass


@pytest.mark.parametrize("field", ["x", "zq", "zsource"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_payload_refuses_nonfinite_actual_captures(field, nonfinite):
    kind = "teacher" if field == "zsource" else "quant"
    modules = teacher_gates() if kind == "teacher" else quant_modules()
    cls = target.TeacherCapture if kind == "teacher" else target.QuantCapture
    with cls(modules, 1) as capture:
        capture.begin(0)
        for layer, module in modules.items():
            hidden = features(layer, 0)
            if layer == 2 and field in {"x", "zsource"}:
                hidden[0, 0] = nonfinite
            if kind == "teacher":
                module(hidden.to(torch.bfloat16))
            else:
                if layer == 2 and field == "zq":
                    module.corrupt = nonfinite
                call_quant(module, hidden)
        capture.finish()
        with pytest.raises(ValueError):
            capture.payload(2)


@pytest.mark.parametrize(
    "failure", ["input_dtype", "input_shape", "buffer_dtype", "buffer_shape", "batch"]
)
def test_quant_hook_refuses_wrong_input_or_actual_logit_buffer(failure):
    modules = quant_modules()
    module = modules[2]
    with target.QuantCapture(modules, 1) as capture, pytest.raises(ValueError):
        capture.begin(0)
        hidden, batch = features(2, 0), 1
        if failure == "input_dtype":
            hidden = hidden.float()
        elif failure == "input_shape":
            hidden = hidden.reshape(2048, 1)
        elif failure == "buffer_dtype":
            module.cfg.router_logits_bsz1 = module.cfg.router_logits_bsz1.float()
        elif failure == "buffer_shape":
            module.cfg.router_logits_bsz1 = torch.zeros((64,), dtype=torch.float16)
        else:
            batch = 2
        module.routing_fn(batch, module.cfg, hidden, {})


@pytest.mark.parametrize("failure", ["not_tuple", "missing", "dtype", "shape"])
def test_teacher_hook_refuses_changed_raw_gate_signature(failure):
    gates = teacher_gates()
    gate = gates[2]
    if failure == "not_tuple":
        gate.result = gate.logits
    elif failure == "missing":
        gate.result = (gate.logits,)
    elif failure == "dtype":
        gate.result = (gate.logits.float(), *gate.result[1:])
    else:
        gate.result = (gate.logits.reshape(1, 1, 64), *gate.result[1:])
    with target.TeacherCapture(gates, 1) as capture, pytest.raises(ValueError):
        capture.begin(0)
        gate(features(2, 0).to(torch.bfloat16))


def test_payload_requires_all_rows_and_known_layer():
    modules = quant_modules()
    with target.QuantCapture(modules, 2) as capture:
        with pytest.raises(ValueError):
            capture.payload(2)
        fill_capture(capture, modules, "quant", 2)
        with pytest.raises((ValueError, KeyError)):
            capture.payload(1)
        with pytest.raises(ValueError):
            capture.begin(2)


def test_atomic_report_replaces_valid_json_without_temporary_leftovers(tmp_path):
    target.atomic_report(tmp_path, {"status": "running", "rows": 0})
    target.atomic_report(tmp_path, {"status": "failed", "error": "injected interruption"})
    assert json.loads((tmp_path / "report.json").read_text()) == {
        "status": "failed",
        "error": "injected interruption",
    }
    assert not (tmp_path / "report.tmp").exists()


def test_cli_help_imports_without_torch_or_cuda():
    script = Path(target.__file__).resolve()
    command = (
        "import builtins,runpy,sys; original=builtins.__import__; "
        "builtins.__import__=lambda name,*args,**kwargs: "
        "(_ for _ in ()).throw(ModuleNotFoundError('torch forbidden')) "
        "if name == 'torch' or name.startswith('torch.') else original(name,*args,**kwargs); "
        f"sys.argv=[{str(script)!r},'--help']; runpy.run_path({str(script)!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONPATH": str(script.parent)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


class FakeTeacher:
    def __init__(self, failure=None):
        self.device, self.implementation = "cpu", "grouped_mm"
        self.failure, self.calls = failure, []
        self.modules = {}
        for layer, gate in teacher_gates().items():
            gate.weight = torch.nn.Parameter(
                torch.full((64, 2048), layer / 8, dtype=torch.bfloat16)
            )
            self.modules[layer] = SimpleNamespace(
                gate=gate, expert_bias=torch.full((64,), layer / 128, dtype=torch.float32)
            )

    def get_submodule(self, name):
        return self.modules[int(name.split(".")[2])]

    def get_experts_implementation(self):
        return {"": self.implementation}

    def _optimize_model_for_decode(self):
        from contextlib import contextmanager

        @contextmanager
        def optimized():
            original = self.implementation
            self.implementation = "wrong" if self.failure == "backend" else "batched_mm"
            try:
                yield
            finally:
                self.implementation = original

        return optimized()

    def __call__(self, *, input_ids, past_key_values, use_cache, logits_to_keep):
        assert use_cache and logits_to_keep == 0
        token = int(input_ids.item())
        count = 0 if past_key_values is None else past_key_values.get_seq_length()
        self.calls.append((token, count))
        hooked = bool(self.modules[2].gate._forward_hooks)
        for layer, module in self.modules.items():
            if self.failure == "missing_layer" and layer == 20:
                continue
            hidden = features(layer, token).to(torch.bfloat16)
            if self.failure == "capture_nonfinite" and layer == 2:
                hidden[0, 0] = float("nan")
            module.gate(hidden)
        if self.failure == "hook_interrupt" and hooked:
            raise RuntimeError("teacher hook forward interrupted")
        if self.failure == "interrupt":
            raise RuntimeError("teacher interrupted")
        logits = torch.arange(target.VOCABULARY, dtype=torch.bfloat16)[None, None]
        if self.failure == "passive_change" and hooked:
            logits = logits + 1
        if self.failure == "nonfinite":
            logits = logits * float("inf")
        elif self.failure == "shape":
            logits = logits[..., :-1]
        cache = SimpleNamespace(get_seq_length=lambda: count + 1)
        if self.failure == "cache_missing":
            cache = None
        elif self.failure == "cache_stale":
            cache = SimpleNamespace(get_seq_length=lambda: count)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class FakeQuantized:
    def __init__(self, failure=None):
        self.failure, self.calls, self.unloads = failure, [], 0
        self.modules = quant_modules()
        for layer, module in self.modules.items():
            module.routing_cfg = SimpleNamespace(
                gate_tensor=torch.full((2048, 64), layer / 8, dtype=torch.float16)
            )
            module.e_score_correction_bias = torch.full((64,), layer / 128, dtype=torch.float32)
        self.originals = {layer: module.routing_fn for layer, module in self.modules.items()}

    def forward(self, tokens, params):
        assert tokens.shape == (1, 1), "Actual imposed input must remain one token"
        batch_size, maximum = params["batch_shape"]
        assert batch_size >= tokens.shape[0], "Batch size too large for cache"
        assert maximum % 256 == 0, "Cache sequence length must be a multiple of256"
        assert tokens.shape[0] * maximum <= params["cache"].max_num_tokens, (
            "Cache too small for batch shape"
        )
        assert 0 <= params["past_len"] < maximum
        token = int(tokens.item())
        self.calls.append((token, params["past_len"], params["cache"], params["batch_shape"]))
        assert params["attn_mode"] == "flash_attn"
        for layer, module in self.modules.items():
            call_quant(module, features(layer, token))
        if self.failure == "interrupt":
            raise RuntimeError("quantized interrupted")
        logits = torch.arange(target.VOCABULARY, dtype=torch.float16)[None, None]
        if self.failure == "nonfinite":
            logits = logits * float("inf")
        elif self.failure == "shape":
            logits = logits[..., :-1]
        elif self.failure == "f16_overflow":
            logits = logits.float() + 1e10
        elif (
            self.failure == "passive_change" and self.modules[2].routing_fn is not self.originals[2]
        ):
            logits = logits + 1
        return logits

    def unload(self):
        self.unloads += 1


def quant_cache(capacity=256):
    return SimpleNamespace(max_num_tokens=capacity)


def tiny_plan():
    return {
        "train": {"input_ids": [[0, 1], [2]]},
        "validation": {"input_ids": [[3, 4], [5]]},
        "final": {
            "chat-holdout": {"input_ids": [6, 7]},
            "cjk-holdout": {"input_ids": [8]},
        },
    }


@pytest.fixture
def small_vocabulary(monkeypatch):
    monkeypatch.setattr(target, "VOCABULARY", 4)


@pytest.mark.parametrize("kind", ["teacher", "quantized"])
def test_real_decode_entry_uses_fresh_caches_offsets_and_exact_storage(
    tmp_path, small_vocabulary, kind
):
    import hashlib

    import numpy as np

    model = FakeTeacher() if kind == "teacher" else FakeQuantized()
    modules = (
        {layer: module.gate for layer, module in model.modules.items()}
        if kind == "teacher"
        else model.modules
    )
    cls = target.TeacherCapture if kind == "teacher" else target.QuantCapture
    progress = []
    with cls(modules, 3) as capture:
        for offset, tokens in ((0, [1, 2]), (2, [3])):
            output = tmp_path / f"{kind}-{offset}.npy"
            if kind == "teacher":
                record = target.decode_teacher_sequence(
                    model, tokens, output, capture, offset, progress.append, hash_logits=True
                )
            else:
                record = target.decode_quantized_sequence(
                    model, quant_cache(), tokens, output, capture, offset, progress.append, True
                )
            values = np.load(output, allow_pickle=False)
            expected = np.tile(np.arange(4), (len(tokens), 1))
            assert values.shape == (len(tokens), 4) and np.isfinite(values).all()
            assert values.dtype == (np.float32 if kind == "teacher" else np.float16)
            np.testing.assert_array_equal(values, expected)
            assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
            assert record["shape"] == [len(tokens), 4]
            if kind == "quantized":
                assert (
                    record["logits_payload_sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
                )
        key = "zsource" if kind == "teacher" else "zq"
        expected = torch.cat([features(2, token)[:, :64] for token in (1, 2, 3)])
        expected = expected.to(torch.bfloat16) if kind == "teacher" else expected + 0.5
        assert torch.equal(capture.payload(2)[key], expected)
    assert progress == [2, 1]
    assert [call[1] for call in model.calls] == [0, 1, 0]
    if kind == "quantized":
        assert model.calls[0][2] is model.calls[1][2] and model.calls[2][2] is not model.calls[0][2]
        assert [call[3] for call in model.calls] == [(1, 256), (1, 256), (1, 256)]


@pytest.mark.parametrize("kind", ["teacher", "quantized"])
@pytest.mark.parametrize("failure", ["interrupt", "shape", "nonfinite"])
def test_decode_error_keeps_partial_capture_incomplete_and_restores_hooks(
    tmp_path, small_vocabulary, kind, failure
):
    model = FakeTeacher(failure) if kind == "teacher" else FakeQuantized(failure)
    modules = (
        {layer: module.gate for layer, module in model.modules.items()}
        if kind == "teacher"
        else model.modules
    )
    cls = target.TeacherCapture if kind == "teacher" else target.QuantCapture
    capture = cls(modules, 2)
    with pytest.raises((RuntimeError, ValueError)), capture:
        path = tmp_path / "partial.npy"
        if kind == "teacher":
            target.decode_teacher_sequence(model, [0, 1], path, capture)
        else:
            target.decode_quantized_sequence(model, quant_cache(), [0, 1], path, capture)
    with pytest.raises(ValueError):
        capture.payload(2)
    if kind == "teacher":
        assert all(not gate._forward_hooks for gate in modules.values())
    else:
        assert all(module.routing_fn is model.originals[layer] for layer, module in modules.items())


@pytest.mark.parametrize("failure", ["cache_missing", "cache_stale"])
def test_teacher_decode_rejects_missing_or_misaligned_caches(small_vocabulary, failure):
    with pytest.raises(ValueError):
        target.decode_teacher_sequence(FakeTeacher(failure), [0, 1])


def test_quantized_decode_rejects_overflow_when_storing_or_hashing(tmp_path, small_vocabulary):
    model = FakeQuantized("f16_overflow")
    for path, hashing in ((None, True), (tmp_path / "overflow.npy", False)):
        with pytest.raises(ValueError):
            target.decode_quantized_sequence(
                model, quant_cache(), [0], path=path, hash_logits=hashing
            )


@pytest.mark.parametrize("kind", ["teacher", "quantized"])
def test_decode_rejects_existing_output_and_empty_tokens_before_forward(
    tmp_path, small_vocabulary, kind
):
    model = FakeTeacher() if kind == "teacher" else FakeQuantized()
    output = tmp_path / "existing.npy"
    output.write_bytes(b"preserve old attempt")
    for tokens, path in (([0], output), ([], tmp_path / "empty.npy")):
        with pytest.raises(ValueError):
            if kind == "teacher":
                target.decode_teacher_sequence(model, tokens, path)
            else:
                target.decode_quantized_sequence(model, quant_cache(), tokens, path)
    assert output.read_bytes() == b"preserve old attempt" and model.calls == []
    assert not (tmp_path / "empty.npy").exists()


@pytest.mark.parametrize("teacher", [False, True])
def test_routing_parameters_export_exact_existing_names_dtypes_and_orientation(teacher):
    model = FakeTeacher() if teacher else FakeQuantized()
    tensors = target._parameters(model.modules, teacher)
    assert len(tensors) == 76
    for layer in target.LAYERS:
        prefix = f"model.layers.{layer}.feed_forward"
        gate, bias = tensors[prefix + ".gate.weight"], tensors[prefix + ".expert_bias"]
        assert gate.shape == (64, 2048)
        assert gate.dtype == (torch.bfloat16 if teacher else torch.float16)
        assert bias.shape == (64,) and bias.dtype == torch.float32
        assert torch.equal(gate, torch.full_like(gate, layer / 8))
        assert torch.equal(bias, torch.full_like(bias, layer / 128))
    module = model.modules[2]
    if teacher:
        module.gate.weight.data.fill_(-99)
        module.expert_bias.fill_(-98)
    else:
        module.routing_cfg.gate_tensor.fill_(-99)
        module.e_score_correction_bias.fill_(-98)
    prefix = "model.layers.2.feed_forward"
    assert tensors[prefix + ".gate.weight"][0, 0] == 0.25
    assert tensors[prefix + ".expert_bias"][0] == 2 / 128


@pytest.mark.parametrize("failure", ["empty", "nan", "inf", "existing"])
def test_tensor_export_refuses_invalid_payload_and_preserves_existing_file(tmp_path, failure):
    path = tmp_path / "output.safetensors"
    tensors = {"zq": torch.zeros((1, 64), dtype=torch.float16)}
    if failure == "empty":
        tensors.clear()
    elif failure in {"nan", "inf"}:
        tensors["zq"][0, 0] = float(failure)
    else:
        path.write_bytes(b"preserve")
    with pytest.raises(ValueError):
        target._save_tensors(path, tensors)
    assert path.read_bytes() == b"preserve" if failure == "existing" else not path.exists()


def test_frozen_plan_archive_is_accepted_without_model_execution():
    path = Path(target.__file__).resolve().parents[1] / (
        "docs/measurements/quant-lfm25-router38-20261015/input-plan.json"
    )
    plan = json.loads(path.read_text())
    assert target.validate_plan(plan) == plan


def fake_quant_loader(model):
    def load(folder, sequences, phase, report):
        assert not model.calls
        return (
            model,
            {name: quant_cache(2048) for name, _ in sequences},
            [quant_cache(), quant_cache()] if phase == "train" else [],
            model.modules.copy(),
        )

    return load


def test_real_teacher_run_exports_actual_gates_captures_and_all_eval_sequences(
    tmp_path, monkeypatch, small_vocabulary
):
    from safetensors.torch import load_file

    model, plan = FakeTeacher(), tiny_plan()
    monkeypatch.setattr(target, "validate_plan", lambda data: data)
    monkeypatch.setattr(target, "load_source", lambda source, report: model)
    monkeypatch.setattr(target, "_teacher_control", lambda *args: {"bitexact": True})
    output = tmp_path / "teacher"
    result = target.run_teacher(tmp_path / "source", plan, output)
    assert result["status"] == "complete" and not result["publication"]
    assert result["experts_implementation"] == {"": "batched_mm"}
    assert model.implementation == "grouped_mm"
    assert len(result["files"]) == 43
    assert set(result["sequences"]) == {
        "train-00",
        "train-01",
        "validation-00",
        "validation-01",
        "chat-holdout",
        "cjk-holdout",
    }
    assert [count for _, count in model.calls] == [0, 1, 0, 0, 1, 0, 0, 1, 0]
    for layer in target.LAYERS:
        tensors = load_file(str(output / f"layer-{layer}.safetensors"))
        assert set(tensors) == {"zsource"} and tensors["zsource"].shape == (3, 64)
        assert not model.modules[layer].gate._forward_hooks
    assert len(load_file(str(output / "routing-parameters.safetensors"))) == 76
    assert json.loads((output / "report.json").read_text())["status"] == "complete"
    assert not (output / "report.tmp").exists()
    before = (output / "report.json").read_bytes()
    calls_before = len(model.calls)
    with pytest.raises(FileExistsError):
        target.run_teacher(tmp_path / "source", plan, output)
    assert (output / "report.json").read_bytes() == before and len(model.calls) == calls_before


@pytest.mark.parametrize("failure", ["backend", "interrupt", "shape", "cache_stale"])
def test_teacher_run_failure_marks_report_restores_backend_and_hooks(
    tmp_path, monkeypatch, small_vocabulary, failure
):
    model = FakeTeacher(failure)
    modules = model.modules.copy()
    monkeypatch.setattr(target, "validate_plan", lambda data: data)
    monkeypatch.setattr(target, "load_source", lambda source, report: model)
    monkeypatch.setattr(target, "_teacher_control", lambda *args: {"bitexact": True})
    output = tmp_path / "teacher"
    with pytest.raises((RuntimeError, ValueError)):
        target.run_teacher(tmp_path / "source", tiny_plan(), output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and not report["publication"] and report["error"]
    assert model.implementation == "grouped_mm"
    assert all(not module.gate._forward_hooks for module in modules.values())
    assert not any(name.startswith("layer-") for name in report["files"])
    assert not (output / "report.tmp").exists()


@pytest.mark.parametrize("phase", ["train", "validation", "final"])
def test_real_quantized_run_unloads_and_exports_passive_control_or_eval(
    tmp_path, monkeypatch, small_vocabulary, phase
):
    from safetensors.torch import load_file

    model, plan = FakeQuantized(), tiny_plan()
    plan["train"]["input_ids"] = [[0] * 128, [1, 2]]
    monkeypatch.setattr(target, "validate_plan", lambda data: data)
    monkeypatch.setattr(target, "_load_quantized", fake_quant_loader(model))
    output = tmp_path / phase
    result = target.run_quantized(tmp_path / "model", plan, output, phase)
    assert result["status"] == "complete" and result["phase"] == phase and not result["publication"]
    assert model.unloads == 1
    assert all(
        module.routing_fn is model.originals[layer] for layer, module in model.modules.items()
    )
    if phase == "train":
        assert len(load_file(str(output / "routing-parameters.safetensors"))) == 76
        control = result["passive_hook_control"]
        assert control["bitexact"] and control["tokens"] == 128 and control["values"] == 512
        assert (
            control["without_hook"]["logits_payload_sha256"]
            == control["with_hook"]["logits_payload_sha256"]
        )
        assert len(model.calls) == 386
        for layer in target.LAYERS:
            tensors = load_file(str(output / f"layer-{layer}.safetensors"))
            assert set(tensors) == {"x", "zq"}
            assert tensors["x"].shape == (130, 2048) and tensors["zq"].shape == (130, 64)
        assert [model.calls[i][1] for i in (0, 127, 128, 255, 256, 383, 384, 385)] == [
            0,
            127,
            0,
            127,
            0,
            127,
            0,
            1,
        ]
    else:
        assert "passive_hook_control" not in result
        expected = (
            {"validation-00", "validation-01"}
            if phase == "validation"
            else {"chat-holdout", "cjk-holdout"}
        )
        assert set(result["sequences"]) == expected
        assert len(result["files"]) == 2
        assert not (output / "routing-parameters.safetensors").exists()
    assert json.loads((output / "report.json").read_text())["status"] == "complete"
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        target.run_quantized(tmp_path / "model", plan, output, phase)
    assert (output / "report.json").read_bytes() == before and model.unloads == 1


@pytest.mark.parametrize(
    "failure", ["interrupt", "shape", "nonfinite", "passive_change", "short_control"]
)
def test_quantized_run_failure_unloads_restores_hooks_and_never_claims_complete(
    tmp_path, monkeypatch, small_vocabulary, failure
):
    model, plan = FakeQuantized(failure), tiny_plan()
    plan["train"]["input_ids"] = [[0] * (127 if failure == "short_control" else 128)]
    monkeypatch.setattr(target, "validate_plan", lambda data: data)
    monkeypatch.setattr(target, "_load_quantized", fake_quant_loader(model))
    output = tmp_path / "failed"
    with pytest.raises((RuntimeError, ValueError)):
        target.run_quantized(tmp_path / "model", plan, output, "train")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and not report["publication"] and report["error"]
    assert model.unloads == 1
    assert all(
        module.routing_fn is model.originals[layer] for layer, module in model.modules.items()
    )
    if failure == "passive_change":
        assert not report["passive_hook_control"]["bitexact"]
    assert not any(name.startswith("layer-") for name in report["files"])
    assert not (output / "report.tmp").exists()


@pytest.mark.parametrize("kind", ["teacher", "quantized"])
def test_invalid_plan_is_reported_before_any_model_load(tmp_path, monkeypatch, kind):
    def no_load(*args, **kwargs):
        raise AssertionError("model must not be loaded")

    monkeypatch.setattr(target, "load_source", no_load)
    monkeypatch.setattr(target, "_load_quantized", no_load)
    output = tmp_path / kind
    with pytest.raises(ValueError):
        if kind == "teacher":
            target.run_teacher(tmp_path / "source", {}, output)
        else:
            target.run_quantized(tmp_path / "model", {}, output, "train")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["files"] == {}


def test_main_dispatches_the_requested_phase_and_returns_without_loading_models(
    tmp_path, monkeypatch
):
    plan = tiny_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(target, "run_teacher", lambda *args: calls.append(("teacher", *args)))
    monkeypatch.setattr(target, "run_quantized", lambda *args: calls.append(("quantized", *args)))
    target.main(["teacher", str(path), str(output), "--source", str(tmp_path / "source")])
    target.main(
        [
            "quantized",
            str(path),
            str(output),
            "--model",
            str(tmp_path / "model"),
            "--phase",
            "final",
        ]
    )
    assert calls == [
        ("teacher", tmp_path / "source", plan, output),
        ("quantized", tmp_path / "model", plan, output, "final"),
    ]
    assert not output.exists()


@pytest.mark.parametrize(
    ("tokens", "capacity"),
    [(1, 256), (255, 256), (256, 256), (257, 512), (1023, 1024), (1024, 1024), (2048, 2048)],
)
def test_cache_capacity_matches_real_fp16_page_boundaries(tokens, capacity):
    assert target.cache_capacity(tokens) == capacity


@pytest.mark.parametrize("tokens", [0, -1, True, False, 1.0, "1", None])
def test_cache_capacity_rejects_nonpositive_or_noninteger_lengths(tokens):
    with pytest.raises(ValueError):
        target.cache_capacity(tokens)


def test_parameter_export_transposes_nonconstant_actual_gate_without_reordering():
    model = FakeQuantized()
    original = torch.arange(64, dtype=torch.float16)[:, None] + (
        torch.arange(2048).remainder(16).to(torch.float16)[None] / 16
    )
    model.modules[2].routing_cfg.gate_tensor = original.T.contiguous()
    tensors = target._parameters(model.modules, False)
    actual = tensors["model.layers.2.feed_forward.gate.weight"]
    assert torch.equal(actual, original) and actual.is_contiguous()
    assert actual[63, 2031] == 63.9375 and actual[2, 16] == 2


@pytest.mark.parametrize(
    ("kind", "failure"),
    [("teacher", "empty_cache"), ("quantized", "empty_cache"), ("quantized", "unload")],
)
def test_cleanup_error_is_recorded_as_failed_even_after_completed_inference(
    tmp_path, monkeypatch, small_vocabulary, kind, failure
):
    model = FakeTeacher() if kind == "teacher" else FakeQuantized()
    monkeypatch.setattr(target, "validate_plan", lambda data: data)
    monkeypatch.setattr(target, "load_source", lambda source, report: model)
    monkeypatch.setattr(target, "_teacher_control", lambda *args: {"bitexact": True})
    monkeypatch.setattr(target, "_load_quantized", fake_quant_loader(model))

    def cleanup_failure():
        raise RuntimeError("injected cleanup failure")

    if failure == "empty_cache":
        monkeypatch.setattr(torch.cuda, "empty_cache", cleanup_failure)
    else:
        monkeypatch.setattr(model, "unload", cleanup_failure)
    output = tmp_path / kind
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        if kind == "teacher":
            target.run_teacher(tmp_path / "source", tiny_plan(), output)
        else:
            target.run_quantized(tmp_path / "model", tiny_plan(), output, "validation")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and not report["publication"]
    assert "injected cleanup failure" in report["cleanup_error"]
    assert not (output / "report.tmp").exists()


@pytest.mark.parametrize("phase", ["train", "validation"])
@pytest.mark.parametrize(
    "failure",
    [None, "not_fused", "missing", "no_reconstruct", "load", "sha", "config", "config_missing"],
)
def test_real_loader_allocates_page_aligned_caches_before_load_and_unloads_failures(
    tmp_path, monkeypatch, phase, failure
):
    import hashlib
    import importlib.metadata

    folder = tmp_path / "model"
    folder.mkdir()
    shard = folder / "model.safetensors"
    shard.write_bytes(b"disposable checkpoint fixture")
    expected = hashlib.sha256(shard.read_bytes()).hexdigest()
    monkeypatch.setattr(
        target, "V2_SHARDS", {shard.name: expected if failure != "sha" else "0" * 64}
    )
    config_hashes = {}
    for name in target.V2_CONFIGS:
        path = folder / name
        path.write_text(json.dumps({"fixture": name}))
        config_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(target, "V2_CONFIGS", config_hashes)
    if failure == "config":
        (folder / "config.json").write_text('{"changed":true}')
    elif failure == "config_missing":
        (folder / "config.json").unlink()
    events = []
    config = SimpleNamespace(
        infer_params=SimpleNamespace(no_reconstruct=failure == "no_reconstruct")
    )
    mlps = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                key=f"model.layers.{layer}.feed_forward",
                is_quantized=failure != "not_fused",
                bc=object(),
            )
        )
        for layer in target.LAYERS
        if failure != "missing" or layer != 39
    ]

    class Model:
        def __init__(self):
            self.modules = mlps
            self.loaded = False
            self.unloads = 0

        @classmethod
        def from_config(cls, actual):
            assert actual is config
            events.append("model")
            return model

        def load(self, **kwargs):
            assert kwargs == {"device": "cuda:0", "max_output_size": 1}
            events.append("load")
            if failure == "load":
                raise RuntimeError("load interrupted")
            self.loaded = True

        def unload(self):
            self.unloads += 1
            events.append("unload")

    model = Model()

    class Cache:
        def __init__(self, actual, *, max_num_tokens, max_batch_size):
            assert actual is model and not model.loaded
            assert max_num_tokens % 256 == 0 and max_batch_size == 1
            self.capacity = max_num_tokens
            events.append(("cache", max_num_tokens))

    monkeypatch.setitem(
        sys.modules,
        "exllamav3",
        SimpleNamespace(
            Cache=Cache,
            Config=SimpleNamespace(from_directory=lambda directory: config),
            Model=Model,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "quantize_lfm25_modal",
        SimpleNamespace(check_converter_sources=lambda path: {"checked": str(path)}),
    )
    distribution = importlib.metadata.distribution
    version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (
            SimpleNamespace(locate_file=lambda path: folder / path)
            if name == "exllamav3"
            else distribution(name)
        ),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "1.5.4" if name == "exllamav3" else version(name),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "CPU fixture")
    report = {}
    sequences = [("first", [0] * 1023), ("second", [0] * 257)]
    failed = failure is not None and not (failure == "sha" and phase != "train")
    if failed:
        with pytest.raises((RuntimeError, ValueError)):
            target._load_quantized(folder, sequences, phase, report)
        before_model = failure in {"sha", "config", "config_missing"}
        assert model.unloads == (0 if before_model else 1)
        if before_model:
            assert events == []
    else:
        actual, caches, controls, modules = target._load_quantized(folder, sequences, phase, report)
        assert actual is model and tuple(modules) == target.LAYERS
        assert [cache.capacity for cache in caches.values()] == [1024, 512]
        assert [cache.capacity for cache in controls] == ([256, 256] if phase == "train" else [])
        assert events[-1] == "load" and model.unloads == 0
    if events:
        load_index = events.index("load")
        assert all(event[0] == "cache" for event in events[1:load_index])


@pytest.mark.parametrize("kind", ["quant", "teacher"])
def test_capture_context_cannot_be_reentered_and_replaces_no_existing_hooks(kind):
    modules = quant_modules() if kind == "quant" else teacher_gates()
    cls = target.QuantCapture if kind == "quant" else target.TeacherCapture
    if kind == "teacher":
        seen = []
        handle = modules[2].register_forward_hook(lambda module, args, output: seen.append(output))
    capture = cls(modules, 1)
    with capture:
        with pytest.raises(ValueError, match="single-use"), capture:
            pass
        fill_capture(capture, modules, kind, 1)
    with pytest.raises(ValueError, match="single-use"), capture:
        pass
    if kind == "teacher":
        assert seen == [modules[2].result]
        assert len(modules[2]._forward_hooks) == 1
        handle.remove()


@pytest.mark.parametrize("kind", ["quant", "teacher"])
def test_partial_context_install_failure_restores_already_installed_hooks(monkeypatch, kind):
    modules = quant_modules() if kind == "quant" else teacher_gates()
    originals = {
        layer: module.routing_fn if kind == "quant" else dict(module._forward_hooks)
        for layer, module in modules.items()
    }
    if kind == "quant":
        modules[20].device = "invalid-fixture-device"
    else:

        def failure(*args, **kwargs):
            raise RuntimeError("hook registration failed")

        monkeypatch.setattr(modules[20], "register_forward_hook", failure)
    cls = target.QuantCapture if kind == "quant" else target.TeacherCapture
    with pytest.raises((RuntimeError, ValueError)), cls(modules, 1):
        pass
    for layer, module in modules.items():
        if kind == "quant":
            assert module.routing_fn is originals[layer]
        else:
            assert dict(module._forward_hooks) == originals[layer]


@pytest.mark.parametrize(
    "failure",
    ["ids_shape", "ids_dtype", "weights_shape", "weights_dtype", "tuple_short", "tuple_long"],
)
def test_quantized_hook_rejects_changed_actual_selected_output_contract(failure):
    modules = quant_modules()
    module = modules[2]
    ids, weights = module.result
    if failure == "ids_shape":
        module.result = (ids[:, :3], weights)
    elif failure == "ids_dtype":
        module.result = (ids.int(), weights)
    elif failure == "weights_shape":
        module.result = (ids, weights[:, :3])
    elif failure == "weights_dtype":
        module.result = (ids, weights.float())
    elif failure == "tuple_short":
        module.result = (ids,)
    else:
        module.result = (ids, weights, object())
    original = module.routing_fn
    with pytest.raises(ValueError), target.QuantCapture(modules, 1) as capture:
        capture.begin(0)
        call_quant(module, features(2, 0))
    assert module.routing_fn is original


@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("failure", ["gate_shape", "gate_dtype", "bias_shape", "bias_dtype"])
def test_parameter_export_refuses_wrong_existing_gate_or_bias_contract(teacher, failure):
    model = FakeTeacher() if teacher else FakeQuantized()
    module = model.modules[2]
    if failure == "gate_shape":
        if teacher:
            module.gate.weight = torch.nn.Parameter(module.gate.weight[:, :2047])
        else:
            module.routing_cfg.gate_tensor = module.routing_cfg.gate_tensor[:2047]
    elif failure == "gate_dtype":
        if teacher:
            module.gate.weight = torch.nn.Parameter(module.gate.weight.float())
        else:
            module.routing_cfg.gate_tensor = module.routing_cfg.gate_tensor.float()
    else:
        name = "expert_bias" if teacher else "e_score_correction_bias"
        bias = getattr(module, name)
        setattr(module, name, bias[:63] if failure == "bias_shape" else bias.half())
    with pytest.raises(ValueError):
        target._parameters(model.modules, teacher)


def test_atomic_report_refuses_nonfinite_json_and_preserves_previous_record(tmp_path):
    target.atomic_report(tmp_path, {"status": "running"})
    before = (tmp_path / "report.json").read_bytes()
    with pytest.raises(ValueError):
        target.atomic_report(tmp_path, {"status": "complete", "score": float("nan")})
    assert (tmp_path / "report.json").read_bytes() == before
    assert not (tmp_path / "report.tmp").exists()


def test_teacher_passive_control_hashes_exact_f32_logits_and_uses_two_fresh_caches(
    tmp_path, monkeypatch, small_vocabulary
):
    import hashlib
    import struct

    monkeypatch.chdir(tmp_path)
    model = FakeTeacher()
    result = target._teacher_control(model, model.modules, list(range(128)))
    expected = hashlib.sha256(struct.pack("<4f", 0, 1, 2, 3) * 128).hexdigest()
    assert result["tokens"] == 128 and result["values"] == 512 and result["bitexact"]
    for key in ("without_hook", "with_hook"):
        assert result[key]["logits_payload_sha256"] == expected
        assert result[key]["shape"] == [128, 4]
        assert result[key]["compute_dtypes"] == ["torch.bfloat16"]
    assert [count for _, count in model.calls] == list(range(128)) * 2
    assert [token for token, _ in model.calls] == list(range(128)) * 2
    assert all(not module.gate._forward_hooks for module in model.modules.values())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("length", [0, 127, 129])
def test_teacher_passive_control_refuses_wrong_token_count_before_forward(small_vocabulary, length):
    model = FakeTeacher()
    with pytest.raises(ValueError):
        target._teacher_control(model, model.modules, [0] * length)
    assert model.calls == []
    assert all(not module.gate._forward_hooks for module in model.modules.values())


@pytest.mark.parametrize("failure", ["capture_nonfinite", "missing_layer", "hook_interrupt"])
def test_teacher_passive_control_errors_restore_all_gate_hooks(small_vocabulary, failure):
    model = FakeTeacher(failure)
    with pytest.raises((RuntimeError, ValueError)):
        target._teacher_control(model, model.modules, [0] * 128)
    assert all(not module.gate._forward_hooks for module in model.modules.values())


def test_teacher_passive_control_records_divergence_without_claiming_parity(small_vocabulary):
    model = FakeTeacher("passive_change")
    result = target._teacher_control(model, model.modules, [0] * 128)
    assert not result["bitexact"]
    assert (
        result["without_hook"]["logits_payload_sha256"]
        != result["with_hook"]["logits_payload_sha256"]
    )
    assert len(model.calls) == 256
    assert all(not module.gate._forward_hooks for module in model.modules.values())


@pytest.mark.parametrize("failure", [None, "passive_change"])
def test_real_teacher_run_requires_actual_passive_control_before_training_export(
    tmp_path, monkeypatch, small_vocabulary, failure
):
    model, plan = FakeTeacher(failure), tiny_plan()
    plan["train"]["input_ids"] = [[0] * 128]
    monkeypatch.setattr(target, "validate_plan", lambda data: data)
    monkeypatch.setattr(target, "load_source", lambda source, report: model)
    control = target._teacher_control
    calls = []

    def observe_control(actual, modules, tokens):
        assert actual is model and tuple(modules) == target.LAYERS
        assert model.implementation == "batched_mm" and not model.calls
        assert torch.is_inference_mode_enabled()
        calls.append(list(tokens))
        return control(actual, modules, tokens)

    monkeypatch.setattr(target, "_teacher_control", observe_control)
    output = tmp_path / "teacher"
    if failure:
        with pytest.raises(
            ValueError, match="Passive teacher router capture changes normal logits"
        ):
            target.run_teacher(tmp_path / "source", plan, output)
    else:
        assert target.run_teacher(tmp_path / "source", plan, output)["status"] == "complete"
    report = json.loads((output / "report.json").read_text())
    assert calls == [[0] * 128]
    assert report["passive_hook_control"]["bitexact"] == (failure is None)
    assert model.implementation == "grouped_mm"
    assert all(not module.gate._forward_hooks for module in model.modules.values())
    assert [count for _, count in model.calls[:256]] == list(range(128)) * 2
    if failure:
        assert report["status"] == "failed" and report["sequences"] == {}
        assert set(report["files"]) == {"routing-parameters.safetensors"}
        assert len(model.calls) == 256
        assert not list(output.glob("layer-*.safetensors"))
    else:
        assert report["status"] == "complete" and len(report["files"]) == 43
        assert len(model.calls) == 390
    assert not list(output.glob("*control*.npy"))


@pytest.mark.parametrize(
    ("length", "capacity"),
    [
        (1, 256),
        (128, 256),
        (255, 256),
        (256, 256),
        (257, 512),
        (1023, 1024),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_quantized_decode_flash_attention_shape_matches_cache_pages_without_padding_tokens(
    small_vocabulary, length, capacity
):
    model = FakeQuantized()
    model.modules.clear()
    tokens = list(range(length))
    result = target.decode_quantized_sequence(
        model, quant_cache(capacity), tokens, hash_logits=True
    )
    assert result["shape"] == [length, 4]
    assert [call[0] for call in model.calls] == tokens
    assert [call[1] for call in model.calls] == list(range(length))
    assert {call[3] for call in model.calls} == {(1, capacity)}


@pytest.mark.parametrize(("length", "capacity"), [(1, 0), (128, 0), (257, 256), (1023, 768)])
def test_quantized_decode_rejects_flash_attention_shape_exceeding_cache_before_first_token(
    small_vocabulary, length, capacity
):
    model = FakeQuantized()
    model.modules.clear()
    with pytest.raises(AssertionError, match="Cache too small for batch shape"):
        target.decode_quantized_sequence(model, quant_cache(capacity), list(range(length)))
    assert model.calls == []
