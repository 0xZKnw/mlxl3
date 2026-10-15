from __future__ import annotations

import copy
import hashlib
import io
import json
import os

import pytest
from pilot_lfm25_calibration import choose_pilot, select_experts
from quantize_lfm25_v3_modal import convert_candidate, fetch_trace_sources, make_mix, ngrams


@pytest.mark.parametrize("tokens", [[7, 124894], [], [-1], [125017], [True]])
def test_chat_preflight_uses_pinned_source_and_rejects_invalid_tokens(monkeypatch, tokens):
    import sys
    from types import SimpleNamespace

    import quantize_lfm25_v3_modal as module

    class Tokenizer:
        def __len__(self):
            return 125017

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Bonjour."}]
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return tokens

    def load(source, revision, trust_remote_code):
        assert source == module.base.SOURCE and revision == module.base.REVISION
        assert trust_remote_code is False
        return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    if tokens == [7, 124894]:
        module.trace_runtime_smoke()
    else:
        with pytest.raises(ValueError):
            module.trace_runtime_smoke()


def test_real_count_selection_covers_cold_median_hot():
    assert select_experts([i * 100 for i in range(64)]) == [6, 35, 63]
    assert select_experts([512] * 64) == [0, 32, 63]


@pytest.mark.parametrize("counts", [[0] * 64, [512] * 63, [-1] + [512] * 63, [True] * 64])
def test_invalid_or_cold_selection_fails(counts):
    with pytest.raises(ValueError):
        select_experts(counts)


def matrix():
    panels = []
    for corpus in ("original", "trace"):
        for mode in ("all", "routed", "affinity", "squared"):
            rms = 0.1 if (corpus, mode) == ("original", "all") else 0.09
            if (corpus, mode) == ("trace", "affinity"):
                rms = 0.07
            for layer in (2, 20, 39):
                for expert in (0, 31, 63):
                    result = {
                        "relative_rms": rms,
                        "validation": {
                            str(i): {"error": rms**2, "energy": 1, "rows": 16} for i in range(12)
                        },
                    }
                    panels.append(
                        {
                            "layer": layer,
                            "expert": expert,
                            "corpus": corpus,
                            "mode": mode,
                            "projections": {
                                p: copy.deepcopy(result) for p in ("gate", "up", "down")
                            },
                        }
                    )
    return {"status": "complete", "proxy_only": True, "panels": panels}


def test_full_matrix_selection_rejects_a_local_or_domain_regression():
    report = matrix()
    selected = choose_pilot(report)
    assert (selected["corpus"], selected["mode"]) == ("trace", "affinity")
    assert selected["geomean_ratio"] == pytest.approx(0.7)
    assert selected["trunk8"] is False
    case = next(p for p in report["panels"] if p["corpus"] == "trace" and p["mode"] == "affinity")
    case["projections"]["up"]["relative_rms"] = 0.12
    assert choose_pilot(report)["mode"] != "affinity"
    case["projections"]["up"]["relative_rms"] = 0.07
    case["projections"]["up"]["validation"]["0"]["error"] = 100
    assert choose_pilot(report)["mode"] != "affinity"


@pytest.mark.parametrize("error", ["empty", "missing", "duplicate", "nan", "failed", "domain"])
def test_incomplete_pilot_never_selects_a_recipe(error):
    report = matrix()
    if error == "empty":
        report["panels"] = []
    elif error == "missing":
        report["panels"].pop()
    elif error == "duplicate":
        report["panels"].append(report["panels"][0])
    elif error == "nan":
        report["panels"][0]["projections"]["gate"]["relative_rms"] = float("nan")
    elif error == "failed":
        report["status"] = "failed"
    else:
        report["panels"][0]["projections"]["gate"]["validation"]["0"]["rows"] = 0
    with pytest.raises(ValueError):
        choose_pilot(report)


def test_trunk_requires_two_real_comparisons_and_bounded_size():
    report = matrix()
    report["trunk"] = [
        {"key": name, "corpus": "trace", "bits": bits, "relative_rms": rms}
        for name, base in (("conv.out", 6), ("feed_forward.w2", 5))
        for bits, rms in ((base, 0.1), (8, 0.05))
    ]
    assert choose_pilot(report)["trunk8"] is True
    report["trunk_budget"] = 2_000_000_000
    assert choose_pilot(report)["trunk8"] is False


def test_overlap_check_sees_order_and_ignores_short_rows():
    a = list(range(40))
    assert not ngrams([a]).isdisjoint(ngrams([a[4:20]]))
    assert ngrams([a]).isdisjoint(ngrams([a[::-1]]))
    assert ngrams([[1, 2]]) == set()
    with pytest.raises(ValueError):
        ngrams([a], 0)


@pytest.mark.parametrize("corrupt", [False, True])
def test_trace_sources_are_checked_before_writing(tmp_path, monkeypatch, corrupt):
    import quantize_lfm25_v3_modal as module

    data = b"pinned script\n"
    monkeypatch.setattr(
        module, "TRACE_SOURCES", {"eval/script.py": hashlib.sha256(data).hexdigest()}
    )
    urls = []

    def request(url, timeout):
        urls.append(url)
        assert timeout == 60
        return io.BytesIO(b"wrong" if corrupt else data)

    monkeypatch.setattr("urllib.request.urlopen", request)
    if corrupt:
        with pytest.raises(ValueError, match="pinned"):
            fetch_trace_sources(str(tmp_path))
        assert not list(tmp_path.iterdir())
    else:
        fetch_trace_sources(str(tmp_path))
        assert (tmp_path / "eval/script.py").read_bytes() == data
    assert module.base.CONVERTER in urls[0]


def test_unknown_variant_is_rejected_before_network_or_gpu():
    with pytest.raises(ValueError):
        convert_candidate("unknown", "trace")
    with pytest.raises(ValueError):
        convert_candidate("affinity", "trace", "not-a-boolean")


@pytest.mark.parametrize("audit", ["complete", "missing", "duplicate"])
def test_conversion_boundary_audits_coverage_and_preserves_failure_status(
    tmp_path, monkeypatch, audit
):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    import quantize_lfm25_v3_modal as module

    folder = tmp_path / "probe"
    (tmp_path / "config.json").write_text('{"vocab_size":128}')
    (folder / "output").mkdir(parents=True)
    (folder / "output/config.json").write_text(
        json.dumps({"num_experts": 2, "num_hidden_layers": 3, "num_dense_layers": 1})
    )
    committed = []
    monkeypatch.setattr(module, "volume", SimpleNamespace(commit=lambda: committed.append(True)))
    monkeypatch.setattr(module.base, "volume", module.base.volume)
    monkeypatch.setenv("MLXL3_MOE_COVERAGE", "before-test")
    monkeypatch.setenv("MLXL3_MOE_CALIBRATION", "before-test")
    monkeypatch.setenv("MLXL3_MOE_MINIMUM", "before-test")
    old_command = module.base.conversion_command

    def fake_convert(probe_files, candidate):
        assert probe_files == {"fixture": b"test"} and candidate == "v3"
        command = module.base.conversion_command(
            tmp_path, tmp_path / "work", tmp_path / "output", True
        )
        assert command[2] == "/opt/mlxl3/exl3_moe_calibration.py" and "-m" not in command
        from safetensors.torch import load_file

        tokens = load_file(command[command.index("--cal_data") + 1])["input_ids"]
        assert tokens.shape == (16, 128) and tokens[0].tolist() == list(range(128))
        rows = [{"key": str(i), "count": 64, "effective_rows": 32.0} for i in range(8)]
        if audit == "missing":
            rows.pop()
        elif audit == "duplicate":
            rows[-1] = rows[0]
        module.Path(os.environ["MLXL3_MOE_COVERAGE"]).write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
        return json.dumps({"status": "complete", "folder": str(folder), "publication": False})

    # Keep disposable coverage paths inside this test, even though the remote mount is /data.
    original_path = module.Path
    monkeypatch.setattr(
        module,
        "Path",
        lambda value: (
            tmp_path / "coverage.jsonl"
            if str(value).startswith("/data/coverage-")
            else original_path(value)
        ),
    )
    monkeypatch.setattr(module.base, "convert_model", fake_convert)
    if audit == "complete":
        result = json.loads(convert_candidate("affinity", "trace", False, {"fixture": b"test"}))
        assert result["status"] == "complete" and result["coverage"]["hessians"] == 8
    else:
        with pytest.raises(ValueError, match="coverage"):
            convert_candidate("affinity", "trace", False, {"fixture": b"test"})
        assert json.loads((folder / "report.json").read_text())["status"] == "failed"
    assert committed and module.base.conversion_command is old_command


@pytest.mark.parametrize("failure", [None, "short_trace", "wrong_keys", "negative", "vocabulary"])
def test_mix_preserves_211_real_rows_and_replaces_only_39_random(tmp_path, failure):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    original = {f"tensor.{i}": torch.full((1, 2048), i, dtype=torch.int64) for i in range(250)}
    if failure == "wrong_keys":
        original.pop("tensor.249")
    elif failure in ("negative", "vocabulary"):
        original["tensor.0"][0, 0] = -1 if failure == "negative" else 125017
    save_file(original, str(tmp_path / "ids.safetensors"))
    count = 38 if failure == "short_trace" else 39
    trace = {
        "rows": [
            {"input_ids": [1000 + i] * 1024, "response_ids": [2000 + i] * 1024}
            for i in range(count)
        ]
    }
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    if failure:
        with pytest.raises(ValueError):
            make_mix(tmp_path / "ids.safetensors", tmp_path / "trace.json", tmp_path)
        assert not (tmp_path / "trace-mix.safetensors").exists()
    else:
        old, new = make_mix(tmp_path / "ids.safetensors", tmp_path / "trace.json", tmp_path)
        assert old[:211] == new[:211] and old[211:] != new[211:]
        assert len(new) == 250 and all(len(row) == 2048 for row in new)
        actual = load_file(str(tmp_path / "trace-mix.safetensors"))["input_ids"]
        assert actual.shape == (250, 2048) and actual.dtype == torch.int64
