import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
path = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/quant-lfm25-v3-20261015/compact-operation.py"
)
spec = importlib.util.spec_from_file_location("compact_operation", path)
operation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operation)


def test_alias_guard_uses_known_half_words_including_signed_zero():
    embedding = torch.tensor([[1.0, -2.0, 0.0, -0.0]], dtype=torch.bfloat16)
    head = torch.tensor([[0x3C00, 0xC000, 0, 0x8000]], dtype=torch.uint16).view(torch.float16)
    operation.require_alias(head, embedding)
    for index in range(4):
        mutant = head.clone()
        mutant.view(torch.int16)[0, index].bitwise_xor_(1)
        with pytest.raises(ValueError):
            operation.require_alias(mutant, embedding)
    head[0, -1] = 0.0
    with pytest.raises(ValueError):
        operation.require_alias(head, embedding)


@pytest.mark.parametrize(
    "head,embedding",
    [
        (torch.empty(0, 2), torch.empty(0, 2)),
        (torch.ones(1, 2), torch.ones(2, 1)),
        (torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 2)),
        (torch.tensor([[float("nan")]]), torch.ones(1, 1)),
        (torch.ones(1, 1), torch.tensor([[float("inf")]])),
        (torch.tensor([[1e20]]), torch.tensor([[1e20]])),
    ],
)
def test_alias_guard_rejects_empty_invalid_and_nonfinite_casts(head, embedding):
    with pytest.raises(ValueError):
        operation.require_alias(head, embedding)


def test_existing_outputs_are_never_overwritten(tmp_path):
    output, report = tmp_path / "output", tmp_path / "report.json"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    report.write_text("keep")
    with pytest.raises(FileExistsError):
        operation.compact(tmp_path / "missing", output, report)
    assert report.read_text() == "keep" and (output / "sentinel").read_text() == "keep"


def test_invalid_model_records_failure_without_creating_output(tmp_path):
    source, output, report = tmp_path / "source", tmp_path / "output", tmp_path / "report.json"
    source.mkdir()
    for name in ("config", "model.safetensors.index", "quantization_config"):
        (source / (name + ".json")).write_text("{}")
    with pytest.raises(ValueError):
        operation.compact(source, output, report)
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and result["parity"] is False and not output.exists()


def test_descriptor_preserves_the_real_raw_conv_and_router_bias_schema():
    info = {"shape": [2], "dtype": "torch.float16", "n_bytes": 4}
    storage = {"linear": {"stored_tensors": {"linear.suh": info}}}
    mapping = {
        "linear.suh": "one.safetensors",
        "model.layers.0.conv.conv.weight": "one.safetensors",
        "model.layers.2.feed_forward.expert_bias": "two.safetensors",
    }
    assert operation.descriptor_entries(storage, mapping) == {"linear.suh": info}
    assert len(mapping) == 3


@pytest.mark.parametrize("case", ["duplicate", "orphan", "unknown_raw", "missing_metadata"])
def test_descriptor_rejects_actual_inconsistent_metadata(case):
    storage = {"linear": {"stored_tensors": {"linear.suh": {}}}}
    mapping = {"linear.suh": "one.safetensors"}
    if case == "duplicate":
        storage["other"] = storage["linear"]
    elif case == "orphan":
        mapping.clear()
    elif case == "unknown_raw":
        mapping["unrecognized.weight"] = "two.safetensors"
    else:
        storage["linear"] = {}
    with pytest.raises(ValueError):
        operation.descriptor_entries(storage, mapping)
