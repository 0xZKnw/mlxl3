import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from quantize_lfm25_moe import (
    capture_lfm_moe,
    check_calibration_recipe,
    expected_projections,
    input_row_stride,
)

PHYSICAL = pytest.mark.skipif(
    os.environ.get("MLXL3_SKIP_METAL_CHECK") == "1", reason="requires a physical Apple GPU"
)


def config():
    return {
        "model_type": "lfm2_moe",
        "tie_word_embeddings": True,
        "layer_types": ["conv", "full_attention", "conv"],
        "num_hidden_layers": 3,
        "num_dense_layers": 1,
        "num_experts": 2,
        "vocab_size": 128,
        "eos_token_id": 127,
        "hidden_size": 128,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1024,
        "use_expert_bias": True,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    }


def test_expected_new_24b_covers_all_experts_and_excludes_router():
    cfg = config()
    cfg.update(
        num_hidden_layers=40,
        num_dense_layers=2,
        num_experts=64,
        layer_types=["full_attention" if i % 4 == 2 else "conv" for i in range(40)],
    )
    keys = expected_projections(cfg)
    assert len(keys) == 7402
    assert "model.layers.39.feed_forward.experts.63.w2" in keys
    assert "model.layers.0.feed_forward.w1" in keys
    assert "model.layers.2.self_attn.q_proj" in keys
    assert not any(key.endswith(("gate", "expert_bias")) for key in keys)


@pytest.mark.parametrize(
    "update",
    [
        {"model_type": "lfm2"},
        {"tie_word_embeddings": False},
        {"num_experts": 0},
        {"num_dense_layers": 4},
        {"num_hidden_layers": 2},
        {"layer_types": ["conv", "other", "conv"]},
    ],
)
def test_bad_config_rejected(update):
    with pytest.raises(ValueError):
        expected_projections({**config(), **update})


@pytest.mark.parametrize("inputs, slots", [(1, 1), (3, 12), (512, 512), (10**30, 4 * 10**30)])
def test_routed_input_alignment(inputs, slots):
    stride = input_row_stride(inputs, slots)
    assert stride >= 1 and inputs * stride == slots
    assert 0 // stride == 0 and (slots - 1) // stride == inputs - 1


@pytest.mark.parametrize("inputs, slots", [(0, 1), (1, 0), (-1, 4), (4, -1), (3, 8), (4, 2)])
def test_routed_input_alignment_rejects_empty_or_misaligned(inputs, slots):
    with pytest.raises(ValueError, match="align"):
        input_row_stride(inputs, slots)


def test_recipe_guard_refuses_mixed_modes_and_mutated_sources(tmp_path):
    work, output = tmp_path / "work", tmp_path / "output"
    all_tokens = {"expert_calibration": "all-experts-v1", "text": "original"}
    routed = {"expert_calibration": "routed-switch-v1", "text": "original"}
    marker = check_calibration_recipe(work, output, routed)
    assert not work.exists() and not output.exists()
    work.mkdir()
    (work / "pipeline_request.json").write_text("{}")
    with pytest.raises(ValueError, match="requires new"):
        check_calibration_recipe(work, output, routed)
    assert check_calibration_recipe(work, output, all_tokens) == marker
    marker.write_text(json.dumps(routed))
    assert check_calibration_recipe(work, output, routed) == marker
    for invalid in (all_tokens, {**routed, "text": "modified"}):
        with pytest.raises(ValueError, match="recipe changed"):
            check_calibration_recipe(work, output, invalid)
    marker.unlink()
    (work / "pipeline_request.json").unlink()
    output.mkdir()
    (output / "weights").write_bytes(b"keep these")
    with pytest.raises(ValueError, match="requires new"):
        check_calibration_recipe(work, output, routed)
    assert (output / "weights").read_bytes() == b"keep these"


@pytest.mark.parametrize("k", range(2, 9))
def test_uniform_plan_ignores_absent_head_but_preserves_real_heads(k):
    pytest.importorskip("ponyexl3.cli.convert")
    from ponyexl3.cli.convert import _fixed_uniform_bits

    for head_bits in range(1, 9):
        assert _fixed_uniform_bits(float(k), head_bits, [k], {}, [0.0]) == k
        result = _fixed_uniform_bits(float(k), head_bits, [k], {"lm_head": [head_bits]}, [0.0])
        assert result == (k if k == head_bits else None)
    assert _fixed_uniform_bits(k + 0.1, 6, [k], {}, [0.0]) is None
    assert _fixed_uniform_bits(float(k), 6, [k, k + 1], {}, [0.0]) is None
    assert _fixed_uniform_bits(float(k), 6, [k], {}, [0.1]) is None


def test_lfm_expert_groups_are_bounded_and_do_not_mix_layers(monkeypatch):
    pytest.importorskip("ponyexl3.convert.driver")
    from ponyexl3.convert import driver

    keys = [
        f"model.layers.{layer}.feed_forward.experts.{expert}.{proj}"
        for layer in (2, 3)
        for expert in range(8)
        for proj in ("w1", "w2", "w3")
    ]
    down = driver._candidate_sibling_group(keys, keys[1], set(), {})
    gates = driver._candidate_sibling_group(keys, keys[0], {keys[0]}, {keys[2]: None})
    assert len(down) == 4 and all(key.endswith("w2") for key in down)
    assert len(gates) == 8 and all(key.endswith(("w1", "w3")) for key in gates)
    assert all(key.startswith("model.layers.2.") for key in down + gates)
    assert keys[0] not in gates and keys[2] not in gates


@pytest.fixture
def source(tmp_path, request):
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.models.lfm2_moe import Model, ModelArgs
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    cfg = {**config(), **getattr(request, "param", {})}
    mx.random.seed(57)
    model = Model(ModelArgs.from_dict(cfg))
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    for layer in model.layers[1:]:
        layer.feed_forward.expert_bias = (
            mx.array([0.10001, -0.050003], dtype=mx.float32)
            if cfg["num_experts"] == 2
            else mx.zeros(cfg["num_experts"], dtype=mx.float32)
        )
    model.eval()
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if ".switch_mlp." in key:
            for expert in range(cfg["num_experts"]):
                name = key.replace(".switch_mlp.", f".experts.{expert}.")
                weights[name] = value[expert]
        else:
            weights[key] = value.transpose(0, 2, 1) if key.endswith("conv.conv.weight") else value
    weights = {
        key.replace("gate_proj.weight", "w1.weight")
        .replace("down_proj.weight", "w2.weight")
        .replace("up_proj.weight", "w3.weight"): value
        for key, value in weights.items()
    }
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(root / "model.safetensors", weights)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: "model.safetensors" for key in weights}})
    )
    vocabulary = {"<unk>": 0, "a": 1, "b": 2, "c": 3, **{f"t{i}": i for i in range(4, 128)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="t127"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(root)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b c " * 256)
    return root, corpus, model, cfg


@PHYSICAL
@pytest.mark.parametrize("seq_len", [16, 129])
def test_layerwise_capture_matches_full_bf16_model_and_own_expert_inputs(source, tmp_path, seq_len):
    import mlx.core as mx
    from mlx_lm.models.activations import swiglu
    from transformers import PreTrainedTokenizerFast

    root, corpus, model, cfg = source
    rows = seq_len * 2
    keys = expected_projections(cfg)
    folder, hidden_file = tmp_path / "capture", tmp_path / "hidden.npy"
    summary = capture_lfm_moe(
        root,
        sorted(keys),
        folder,
        text_path=corpus,
        rows=rows,
        seq_len=seq_len,
        hidden_output=hidden_file,
    )
    assert summary.captured_count == summary.module_count == len(keys) == 23
    assert not summary.missing and not summary.expert_fallback
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root)
    tokens = mx.array(tokenizer.encode(corpus.read_text())[:rows]).reshape(2, seq_len)
    expected = np.asarray(model.model(tokens).astype(mx.float32))
    actual = np.load(hidden_file)
    assert actual.shape == expected.shape and np.isfinite(actual).all() and actual.size > 0
    error = np.sqrt(np.mean((actual - expected) ** 2) / np.mean(expected**2))
    assert error < 0.005
    for layer in (1, 2):
        prefix = f"model.layers.{layer}.feed_forward"
        shared = [
            folder / (key + ".npy")
            for key in keys
            if key.startswith(prefix + ".experts") and key.endswith(("w1", "w3"))
        ]
        assert len({path.stat().st_ino for path in shared}) == 1
        inputs = mx.array(np.load(shared[0])).astype(mx.bfloat16)
        switch = model.layers[layer].feed_forward.switch_mlp
        for expert in range(2):
            expected_down = np.asarray(
                swiglu(
                    inputs @ switch.gate_proj.weight[expert].T,
                    inputs @ switch.up_proj.weight[expert].T,
                ).astype(mx.float16)
            )
            np.testing.assert_array_equal(
                np.load(folder / (prefix + f".experts.{expert}.w2.npy")), expected_down
            )
    for key in keys:
        values = np.load(folder / (key + ".npy"), allow_pickle=False)
        assert values.shape == (rows, 128) and np.isfinite(values).all()


@PHYSICAL
@pytest.mark.parametrize("seq_len, top_k", [(16, 1), (16, 2), (129, 1), (129, 2)])
def test_routed_capture_records_actual_switch_inputs_without_cross_expert_links(
    source, tmp_path, monkeypatch, seq_len, top_k
):
    import mlx.core as mx
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.switch_layers import SwitchGLU

    root, corpus, _model, cfg = source
    cfg["num_experts_per_tok"] = top_k
    (root / "config.json").write_text(json.dumps(cfg))
    weights = mx.load(root / "model.safetensors")
    for layer in (1, 2):
        weights[f"model.layers.{layer}.feed_forward.expert_bias"] = mx.zeros(2, dtype=mx.float32)
    replacement = root / "updated.safetensors"
    mx.save_safetensors(replacement, weights)
    replacement.replace(root / "model.safetensors")
    corpus.write_text((" ".join(f"t{i}" for i in range(4, 128)) + " ") * 16)
    count = 64 if seq_len == 16 else 8
    rows = count * seq_len
    keys = sorted(expected_projections(cfg))
    folder = tmp_path / "capture"
    # Reusing a legacy capture folder must break its cross-expert hard links.
    capture_lfm_moe(root, keys, folder, text_path=corpus, rows=rows, seq_len=seq_len)
    oracle, calls = {}, []
    original = SwitchGLU.__call__

    def observe(switch, x, ids):
        layer = 1 + len(calls) // (count * ((seq_len + 127) // 128))
        calls.append((x.shape, ids.shape))
        actual_ids = np.asarray(ids).reshape(-1, top_k)
        values = np.asarray(x.astype(mx.float16)).reshape(-1, 128)
        assert actual_ids.shape[0] == values.shape[0] and np.isfinite(values).all()
        for expert in range(2):
            mask = np.any(actual_ids == expert, axis=1)
            record = oracle.setdefault((layer, expert), {"inputs": [], "down": []})
            record["inputs"].append(values[mask])
            if mask.any():
                inputs = x.reshape(-1, 128)[mx.array(np.flatnonzero(mask))]
                down = swiglu(
                    inputs @ switch.gate_proj.weight[expert].T,
                    inputs @ switch.up_proj.weight[expert].T,
                )
                record["down"].append(np.asarray(down.astype(mx.float16)))
        return original(switch, x, ids)

    monkeypatch.setattr(SwitchGLU, "__call__", observe)
    summary = capture_lfm_moe(
        root,
        keys,
        folder,
        text_path=corpus,
        rows=rows,
        seq_len=seq_len,
        max_seqs=count,
        routed=True,
    )
    assert summary.captured_count == len(keys) and not summary.expert_fallback
    coverage = json.loads((folder / "coverage.json").read_text())
    assert coverage["status"] == "complete" and coverage["tokens"] == rows
    assert len(coverage["modules"]) == len(keys)
    errors = []
    for (layer, expert), record in oracle.items():
        prefix = f"model.layers.{layer}.feed_forward.experts.{expert}"
        gate, up, down = (np.load(folder / (prefix + f".w{i}.npy")) for i in (1, 3, 2))
        expected = np.concatenate(record["inputs"])
        expected_down = np.concatenate(record["down"])
        assert len(expected) >= 256 and len(expected) <= rows
        if top_k == 1:
            assert len(expected) < rows
        order = np.lexsort(gate.T)
        expected_order = np.lexsort(expected.T)
        np.testing.assert_array_equal(gate[order], expected[expected_order])
        np.testing.assert_array_equal(up, gate)
        assert down.shape == expected_down.shape and np.isfinite(down).all() and down.size > 0
        relative_rms = np.sqrt(
            np.sum(
                (down[order].astype(np.float32) - expected_down[expected_order]) ** 2,
                dtype=np.float64,
            )
            / np.sum(expected_down.astype(np.float64) ** 2)
        )
        assert relative_rms < 1e-4, (prefix, relative_rms)
        errors.append(float(relative_rms))
        paths = [folder / (prefix + f".w{i}.npy") for i in (1, 3)]
        assert paths[0].stat().st_ino == paths[1].stat().st_ino
        other = folder / (f"model.layers.{layer}.feed_forward.experts.{1 - expert}.w1.npy")
        assert paths[0].stat().st_ino != other.stat().st_ino
        assert coverage["modules"][prefix + ".w2"]["observed_rows"] == len(expected)
    print(
        json.dumps(
            {
                "routed_fixture": {
                    "seq_len": seq_len,
                    "top_k": top_k,
                    "tokens": rows,
                    "projections": len(keys),
                    "max_down_relative_rms": max(errors),
                    "observed_rows": {
                        key: value["observed_rows"]
                        for key, value in coverage["modules"].items()
                        if key.endswith("w2") and ".experts." in key
                    },
                }
            }
        )
    )


@PHYSICAL
@pytest.mark.parametrize("source", [{"num_experts": 64, "num_experts_per_tok": 4}], indirect=True)
def test_routed_capture_64_experts_top4_covers_tail_and_all_route_slots(source, tmp_path):
    root, corpus, _model, cfg = source
    corpus.write_text((" ".join(f"t{i}" for i in range(4, 128)) + " ") * 32)
    folder = tmp_path / "capture64"
    keys = sorted(expected_projections(cfg))
    summary = capture_lfm_moe(
        root, keys, folder, text_path=corpus, rows=4, seq_len=129, max_seqs=16, routed=True
    )
    assert summary.captured_count == len(keys) == 395 and not summary.expert_fallback
    report = json.loads((folder / "coverage.json").read_text())
    assert report["status"] == "complete"
    for layer in (1, 2):
        for projection in ("w1", "w2", "w3"):
            counts = [
                report["modules"][
                    f"model.layers.{layer}.feed_forward.experts.{expert}.{projection}"
                ]["observed_rows"]
                for expert in range(64)
            ]
            assert sum(counts) == 16 * 129 * 4 and min(counts) >= 4
        for expert in (0, 31, 63):
            values = np.load(folder / f"model.layers.{layer}.feed_forward.experts.{expert}.w2.npy")
            assert values.shape == (4, 128) and np.isfinite(values).all()


@PHYSICAL
@pytest.mark.parametrize(
    "corruption", ["nan", "finite", "empty", "width", "truncated", "coverage", "empty_file", "npz"]
)
def test_routed_capture_caps_rows_keeps_later_sequences_and_rejects_corrupted_resume(
    source, tmp_path, corruption
):
    import mlx.core as mx

    root, corpus, _, cfg = source
    folder = tmp_path / "capture"
    keys = sorted(expected_projections(cfg))
    options = {"text_path": corpus, "rows": 8, "seq_len": 16, "max_seqs": 16, "routed": True}
    summary = capture_lfm_moe(root, keys, folder, **options)
    assert summary.seqs_run == 16
    coverage = json.loads((folder / "coverage.json").read_text())
    for key in keys:
        assert np.load(folder / (key + ".npy")).shape == (8, 128)
        assert coverage["modules"][key]["observed_rows"] >= 8
        assert coverage["modules"][key]["stored_rows"] == 8
    for layer in (1, 2):
        counts = [
            coverage["modules"][f"model.layers.{layer}.feed_forward.experts.{i}.w1"][
                "observed_rows"
            ]
            for i in range(2)
        ]
        assert sum(counts) == 256
    before = (folder / (keys[0] + ".npy")).stat().st_mtime_ns
    capture_lfm_moe(root, keys, folder, **options)
    assert (folder / (keys[0] + ".npy")).stat().st_mtime_ns == before
    damaged = folder / (keys[0] + ".npy")
    if corruption == "empty_file":
        damaged.write_bytes(b"")
    elif corruption == "npz":
        with damaged.open("wb") as archive:
            np.savez(archive, values=np.zeros((8, 128), dtype=np.float16))
    elif corruption == "truncated":
        damaged.write_bytes(b"\x93NUMPY")
    elif corruption == "coverage":
        (folder / "coverage.json").write_text("[]")
    else:
        values = np.load(damaged)
        if corruption == "nan":
            values[0, 0] = np.nan
        elif corruption == "finite":
            values[0, 0] += 1
        elif corruption == "empty":
            values = values[:0]
        else:
            values = values[:, :1]
        np.save(damaged, values)
    with pytest.raises((ValueError, OSError, TypeError, EOFError)):
        capture_lfm_moe(root, keys, folder, **options)
    assert not (folder / "complete.json").exists()
    assert json.loads((folder / "coverage.json").read_text())["status"] == "failed"
    mx.clear_cache()


@PHYSICAL
@pytest.mark.parametrize("failure", ["cold", "disk", "cancel"])
def test_routed_capture_failure_never_borrows_expert_inputs_or_leaves_hooks(
    source, tmp_path, monkeypatch, failure
):
    import mlx.core as mx
    import quantize_ling
    from mlx import nn
    from mlx_lm.models.switch_layers import SwitchLinear
    from ponyexl3.convert import cancel

    root, corpus, _, cfg = source
    folder = tmp_path / "capture"
    linear, switch = nn.Linear.__call__, SwitchLinear.__call__
    if failure == "cold":
        weights = mx.load(root / "model.safetensors")
        for layer in (1, 2):
            weights[f"model.layers.{layer}.feed_forward.expert_bias"] = mx.array(
                [10, -10], dtype=mx.float32
            )
        replacement = root / "updated.safetensors"
        mx.save_safetensors(replacement, weights)
        replacement.replace(root / "model.safetensors")
    elif failure == "disk":

        def fail(*args, **kwargs):
            raise OSError("injected disk failure")

        monkeypatch.setattr(quantize_ling, "save_rows", fail)
    else:
        cancel.request()
    try:
        with pytest.raises((ValueError, OSError, cancel.ConversionCancelled)):
            capture_lfm_moe(
                root,
                sorted(expected_projections(cfg)),
                folder,
                text_path=corpus,
                rows=8,
                seq_len=16,
                max_seqs=16,
                routed=True,
            )
    finally:
        cancel.clear()
    assert nn.Linear.__call__ is linear and SwitchLinear.__call__ is switch
    assert not (folder / "complete.json").exists()
    report = json.loads((folder / "coverage.json").read_text())
    assert report["status"] == ("interrupted" if failure == "cancel" else "failed")
    if failure == "cold":
        assert report["modules"]["model.layers.1.feed_forward.experts.1.w2"]["stored_rows"] == 0


@PHYSICAL
def test_capture_failure_restores_linear_hook_and_invalidates_completion(
    source, tmp_path, monkeypatch
):
    import quantize_ling
    from mlx import nn

    root, corpus, _, cfg = source
    folder = tmp_path / "capture"
    keys = sorted(expected_projections(cfg))
    options = {"text_path": corpus, "rows": 16, "seq_len": 16}
    capture_lfm_moe(root, keys, folder, **options)
    complete = folder / "complete.json"
    assert complete.is_file()
    before = (folder / (keys[0] + ".npy")).stat().st_mtime_ns
    capture_lfm_moe(root, keys, folder, **options)
    assert (folder / (keys[0] + ".npy")).stat().st_mtime_ns == before
    corpus.write_text(corpus.read_text() + " a")
    original = nn.Linear.__call__

    def fail(*args, **kwargs):
        raise RuntimeError("injected disk failure")

    monkeypatch.setattr(quantize_ling, "capture_experts", fail)
    with pytest.raises(RuntimeError, match="disk failure"):
        capture_lfm_moe(root, keys, folder, **options)
    assert nn.Linear.__call__ is original and not complete.exists()


@PHYSICAL
@pytest.mark.parametrize(
    "options", [{"rows": 0}, {"seq_len": 0}, {"max_seqs": 0}, {"dtype": "float32"}, {"rows": 2048}]
)
def test_capture_rejects_invalid_or_short_inputs(source, tmp_path, options):
    root, corpus, _, cfg = source
    with pytest.raises(ValueError):
        capture_lfm_moe(
            root,
            sorted(expected_projections(cfg)),
            tmp_path / "capture",
            **{"text_path": corpus, "rows": 16, "seq_len": 16, **options},
        )
    assert not (tmp_path / "capture/complete.json").exists()


@PHYSICAL
@pytest.mark.parametrize("routed", [False, True])
def test_cli_converts_complete_checkpoint_and_resumes_without_changing_weights(
    source, tmp_path, routed
):
    from safetensors import safe_open

    from mlxl3_quantizer import list_exl3_modules, validate_checkpoint_files

    root, corpus, model, cfg = source
    if routed:
        corpus.write_text((" ".join(f"t{i}" for i in range(4, 128)) + " ") * 32)
    output, work = tmp_path / "output", tmp_path / "work"
    command = [
        sys.executable,
        str(ROOT / "scripts/quantize_lfm25_moe.py"),
        "--in-dir",
        str(root),
        "--out-dir",
        str(output),
        "--work-dir",
        str(work),
        "--bits",
        "4",
        "--calibration-text",
        str(corpus),
        "--calibration-rows",
        "1024" if routed else "16",
        "--calibration-seq-len",
        "129" if routed else "16",
        "--max-workers",
        "1",
        "--search-backend",
        "metal",
    ]
    if routed:
        command.extend(["--routed-calibration", "--calibration-max-seqs", "16"])
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'apps/MLXL3Studio/Vendor/PonyExl3'}:{ROOT / 'scripts'}",
        "PONYEXL3_FIXED_EXPERT_SCALE": "0.908",
    }
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if routed:
        captures = json.loads((work / "calibration.safetensors/coverage.json").read_text())[
            "modules"
        ]
        counts = {record["stored_rows"] for key, record in captures.items() if ".experts." in key}
        assert len(counts) > 1 and min(counts) >= 256 and max(counts) <= 1024
    validate_checkpoint_files(output)
    assert set(list_exl3_modules(output)) == expected_projections(cfg)
    measurements = json.loads((work / "measurements.json").read_text())
    assert measurements["mode"] == "fixed_uniform" and measurements["candidate_count"] == 0
    contents = {}
    for shard in output.glob("model*.safetensors"):
        with safe_open(shard, framework="numpy") as handle:
            contents.update({key: handle.get_tensor(key) for key in handle.keys()})  # noqa: SIM118
    for key in expected_projections(cfg):
        assert key + ".mcg" in contents
        assert contents[key + ".trellis"].size > 0
        assert (
            np.isfinite(contents[key + ".suh"]).all() and np.isfinite(contents[key + ".svh"]).all()
        )
    for layer in (1, 2):
        bias = contents[f"model.layers.{layer}.feed_forward.expert_bias"]
        assert bias.dtype == np.float32
        np.testing.assert_array_equal(bias, np.array([0.10001, -0.050003], dtype=np.float32))
    reference = tmp_path / "measured"
    reference_command = [
        sys.executable,
        "-c",
        (
            "import runpy,sys; from ponyexl3.cli import convert; "
            "convert._fixed_uniform_bits=lambda *a,**kw: None; "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        ),
        *command[1:],
    ]
    reference_command[reference_command.index(str(output))] = str(reference)
    reference_command[reference_command.index(str(work))] = str(tmp_path / "measured-work")
    result = subprocess.run(
        reference_command, env=env, text=True, capture_output=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reference_contents = {}
    for shard in reference.glob("model*.safetensors"):
        with safe_open(shard, framework="numpy") as handle:
            reference_contents.update({key: handle.get_tensor(key) for key in handle.keys()})  # noqa: SIM118
    assert set(reference_contents) == set(contents)
    for key, value in contents.items():
        np.testing.assert_array_equal(reference_contents[key], value, err_msg=key)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("model*.safetensors")
    }
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("model*.safetensors")
    }
    changed_mode = command.copy()
    if routed:
        changed_mode.remove("--routed-calibration")
    else:
        changed_mode.append("--routed-calibration")
    result = subprocess.run(
        changed_mode, env=env, text=True, capture_output=True, timeout=20, check=False
    )
    assert result.returncode != 0 and "recipe changed" in result.stderr
    corpus.write_text(corpus.read_text() + " a")
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=20, check=False
    )
    assert result.returncode != 0 and "recipe changed" in result.stderr
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("model*.safetensors")
    }
    engine = os.environ.get("MLXL3_QUANT_TEST_ENGINE")
    if engine:
        import mlx.core as mx
        from native_json_process import JsonProcess

        model.model = None
        mx.clear_cache()
        deadline = time.monotonic() + 60
        with (
            (tmp_path / "bridge.stderr").open("w") as stderr,
            JsonProcess(
                [
                    engine,
                    "--registry",
                    str(tmp_path / "registry.json"),
                    "bridge",
                    str(output),
                    "--context-length",
                    "1024",
                ],
                stderr=stderr,
            ) as bridge,
        ):
            while (event := bridge.receive(deadline))["type"] != "ready":
                assert event["type"] != "error", event
            bridge.send(
                {
                    "type": "generate",
                    "request_id": "quant-smoke",
                    "messages": [{"role": "user", "content": "a b c"}],
                    "max_tokens": 4,
                    "temperature": 0.0,
                    "top_k": 1,
                    "repetition_penalty": 1.0,
                },
                deadline,
            )
            text = ""
            while True:
                event = bridge.receive(deadline)
                assert event["type"] != "error", event
                if event["type"] == "delta":
                    text += event["text"]
                if event["type"] == "complete":
                    assert event["stats"]["generated_tokens"] == 4
                    assert text.strip()
                    print(json.dumps({"native_bridge_text": text, "stats": event["stats"]}))
                    break
