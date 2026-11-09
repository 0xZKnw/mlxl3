import array
import json
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/replay_native_route_swap.py")
)
choose = MODULE["choose_swaps"]


def test_protects_highest_scores_not_first_positions_and_keeps_inputs():
    ids = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    probabilities = [1 / 512] * 512
    weights, starved = [0.1] * 10, [0] * 512
    assert choose(ids, probabilities, weights, {10, 11}, 0, starved)[0] == []
    assert choose(ids, probabilities, weights, {10, 11}, 0.1, starved)[0] == [(0, 10)]
    assert ids == list(reversed(range(10))) and weights == [0.1] * 10 and starved == [0] * 512
    starved[9] = 3
    picks, updated = choose(ids, probabilities, weights, {10}, 0.1, starved)
    assert picks == [(1, 10)] and updated[9] == 0 and updated[8] == 1
    assert choose(ids, probabilities, [0.11] * 10, {10}, 0.4, starved)[0] == []


def test_only_misses_are_replaced_and_hits_keep_recovery_history():
    ids, probabilities, weights = list(range(10)), [1 / 512] * 512, [0.1] * 10
    state = [0] * 512
    state[3] = 2
    picks, updated = choose(ids, probabilities, weights, {3, 10}, 0.2, state)
    assert picks == [(4, 10)] and updated[3] == 2 and updated[4] == 1
    assert state[3] == 2 and state[4] == 0
    for count in range(3):
        state[4] = count
        picks, updated = choose(ids, probabilities, weights, {3, 10}, 0.2, state)
        assert (4, 10) in picks and updated[4] == count + 1
    state[4] = 3
    picks, updated = choose(ids, probabilities, weights, {3, 10}, 0.2, state)
    assert all(rank != 4 for rank, _ in picks) and updated[4] == 0


def test_multi_swap_candidates_are_ready_unique_and_never_original_ids():
    ids = list(range(10))
    picks, _ = choose(ids, [1 / 512] * 512, [0.1] * 10, {3, 10, 11, 12}, 0.4, [0] * 512, limit=3)
    assert picks == [(4, 10), (5, 11), (6, 12)]
    assert len({e for _, e in picks}) == 3


def test_probability_threshold_and_invalid_tolerances():
    probabilities = [0.0] * 512
    for expert in range(10):
        probabilities[expert] = 1.0 - expert * 0.05
    probabilities[10] = probabilities[9] * 0.8
    args = (list(range(10)), probabilities, [0.1] * 10, {10})
    assert choose(*args, 0.1, [0] * 512)[0] == []
    assert choose(*args, 0.3, [0] * 512)[0]
    for invalid in [float("nan"), float("inf"), -1, 0.401]:
        with pytest.raises(ValueError):
            choose(*args, invalid, [0] * 512)


def test_lfu_matches_history_recomputed_oracle_under_churn():
    import random

    rng = random.Random(16504)
    for capacity in (10, 11, 53):
        cache = MODULE["Cache"](capacity)
        keys, history = [None] * capacity, []
        for tick in range(1, 81):
            ids = rng.sample(range(80), 10)
            expected = sum(e not in keys for e in ids)
            counts, last = {}, {}
            # Independent slow oracle reconstructs frequency and last use from history.
            for t, prior in enumerate(history, 1):
                if t % 16 == 0:
                    counts = {e: n // 2 for e, n in counts.items()}
                for e in prior:
                    counts[e] = counts.get(e, 0) + 1
                    last[e] = t
            if tick % 16 == 0:
                counts = {e: n // 2 for e, n in counts.items()}
            for e in sorted(ids):
                if e not in keys:
                    victim = min(
                        (i for i, k in enumerate(keys) if k not in ids),
                        key=lambda i: (
                            keys[i] is not None,
                            counts.get(keys[i], 0),
                            last.get(keys[i], 0),
                            i,
                        ),
                    )
                    keys[victim] = e
            assert cache.reserve(ids) == expected
            assert cache.keys == keys and set(ids) <= cache.ready()
            history.append(ids)


def fixture(folder):
    rows = []
    for position in range(17):
        for layer in range(48):
            ids = list(range(10)) if position < 16 else [0, 1, 2, 16, 17, 18, 19, 20, 21, 22]
            ready = [] if position == 0 else list(range(10))
            hits = sum(e in ready for e in ids)
            rows.append(
                {
                    "document": 0,
                    "layer": layer,
                    "position": position,
                    "ids": ids,
                    "weights": [0.1] * 10,
                    "ready": ready,
                    "actual_hits": hits,
                    "actual_read_bytes": (10 - hits) * 1248012,
                }
            )
    trace = {
        "status": "complete_diagnostic_only",
        "sample_count": len(rows),
        "documents": [{"name": "test", "input_rows": 17}],
        "samples": rows,
        "timing_is_benchmark": False,
    }
    (folder / "trace.json").write_text(json.dumps(trace))
    (folder / "probabilities.f32").write_bytes(
        array.array("f", [1 / 512] * (512 * len(rows))).tobytes()
    )
    return trace


def test_end_to_end_replay_counts_real_opportunities(tmp_path):
    fixture(tmp_path)
    result = MODULE["analyse"](tmp_path)
    assert result["route_samples"] == 816 and result["decode_layer_rows"] == 48
    assert result["observed_cold_reads"] == 336
    assert len(result["variants"]) == 14
    assert result["baseline_ready_sets_exact"]
    pick = next(v for v in result["variants"] if v["max_swaps"] == 1 and v["tau"] == 0.4)
    assert pick["swaps"] == pick["net_modeled_saved_reads"] == 48
    assert pick["net_modeled_cold_read_fraction"] == pytest.approx(1 / 7)
    assert all(
        v["swaps"] == v["net_modeled_saved_reads"] == 0 for v in result["variants"] if v["tau"] == 0
    )


def test_false_baseline_never_yields_a_validated_replay(tmp_path):
    trace = fixture(tmp_path)
    trace["samples"][-1]["ready"].append(400)
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="LFU baseline mismatch"):
        MODULE["analyse"](tmp_path)


@pytest.mark.parametrize(
    "error",
    [
        "incomplete",
        "empty",
        "duplicate",
        "bad_id",
        "nonfinite_weight",
        "bad_hits",
        "bad_bytes",
        "truncated",
        "extra",
        "nonfinite_probability",
    ],
)
def test_invalid_trace_never_becomes_a_complete_result(tmp_path, error):
    trace = fixture(tmp_path)
    if error == "incomplete":
        trace["status"] = "running"
    elif error == "empty":
        trace["samples"] = []
    elif error == "duplicate":
        trace["samples"][1] = trace["samples"][0]
    elif error == "bad_id":
        trace["samples"][0]["ids"][0] = 512
    elif error == "nonfinite_weight":
        trace["samples"][0]["weights"][0] = float("nan")
    elif error == "bad_hits":
        trace["samples"][0]["actual_hits"] = 10
    elif error == "bad_bytes":
        trace["samples"][0]["actual_read_bytes"] = 1
    else:
        file = tmp_path / "probabilities.f32"
        if error == "truncated":
            file.write_bytes(file.read_bytes()[:-4])
        elif error == "extra":
            file.write_bytes(file.read_bytes() + b"xxxx")
        else:
            data = bytearray(file.read_bytes())
            data[:4] = array.array("f", [float("nan")]).tobytes()
            file.write_bytes(data)
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError):
        MODULE["load_trace"](tmp_path)


def test_failure_is_recorded_and_previous_output_preserved(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(OSError):
        MODULE["main"]([str(tmp_path / "missing"), str(output)])
    assert json.loads(output.read_text())["status"] == "failed"
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        MODULE["main"]([str(tmp_path), str(output)])
    assert output.read_bytes() == before


def test_recovery_can_cancel_all_initial_read_savings(tmp_path):
    trace = fixture(tmp_path)
    last = trace["samples"][-48:]
    for position in range(17, 20):
        for row in last:
            current = dict(
                row,
                position=position,
                ready=list(range(10)) + list(range(16, 23)),
                actual_hits=10,
                actual_read_bytes=0,
            )
            trace["samples"].append(current)
    trace["documents"][0]["input_rows"] = 20
    trace["sample_count"] = len(trace["samples"])
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    (tmp_path / "probabilities.f32").write_bytes(
        array.array("f", [1 / 512] * (512 * len(trace["samples"]))).tobytes()
    )
    result = MODULE["analyse"](tmp_path)
    variant = next(v for v in result["variants"] if v["max_swaps"] == 1 and v["tau"] == 0.4)
    assert variant["swaps"] == 3 * 48
    assert variant["net_modeled_saved_reads"] == 0
