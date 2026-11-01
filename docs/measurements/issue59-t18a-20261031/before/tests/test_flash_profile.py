"""The dataset cannot turn incomplete, divergent or overlapping records into timings."""

import gzip
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))


def event(name, parent, begin, end, data=None):
    return {"name": name, "parent": parent, "start_ns": begin, "end_ns": end, "data": data}


def trace():
    return [
        event("decode", None, 0, 1000),
        event("layer_0", 0, 1, 999),
        event(
            "moe_forward_rows",
            1,
            2,
            998,
            {
                "rows": 1,
                "capacity": 53,
                "ids": list(range(10)),
                "slots_hit": [[i, False] for i in range(10)],
                "hot_rows": 0,
                "cold_rows": 10,
            },
        ),
        event(
            "disk_read_batch",
            2,
            3,
            900,
            {
                "ok": True,
                "jobs": [[i, i] for i in range(10)],
                "workers": 4,
                "timings_job_worker_start_end_bytes_failed": [
                    [i, i % 4, (i // 4) * 20, (i // 4) * 20 + 10, 1248012, 0] for i in range(10)
                ],
            },
        ),
    ]


def test_worker_overlap_and_nested_times_are_not_added_as_exclusive_gpu():
    out = API["validate_events"](trace())
    assert len(out["disk_jobs"]) == 10 and sum(j["bytes"] for j in out["disk_jobs"]) == 12480120
    root = next(c for c in out["components"] if c["component"] == "decode")
    assert root["wall_ns"]["total"] == 1000 and root["host_self_ns"]["total"] == 2
    assert out["disk_batches"][0]["last_completion_ns"] == 50
    assert out["disk_batches"][0]["worker_finish_spread_ns"] == 20
    assert "not exclusive GPU" in out["limits"]


def test_shared_physical_slots_allow_2543_but_keep_expert_ids_bounded():
    source = trace()
    source[2]["data"].update(capacity=2544, slots_hit=[[2534 + i, False] for i in range(10)])
    assert API["validate_events"](source)["routes"][0]["capacity"] == 2544
    for field, value in [
        ("capacity", 2545),
        ("ids", [512] + list(range(1, 10))),
        ("slots_hit", [[2544, False]] + [[2535 + i, False] for i in range(9)]),
    ]:
        bad = trace()
        bad[2]["data"].update(source[2]["data"])
        bad[2]["data"][field] = value
        with pytest.raises(AssertionError):
            API["validate_events"](bad)


def rich_trace():
    return trace() + [
        event(
            "expert_kernel_signature",
            2,
            901,
            902,
            {"id": 0, "stable": False, "key": ["name", "header", "source"]},
        ),
        event("expert_kernel_request", 2, 903, 904, {"signature": 0}),
        event("expert_kernel_request", 2, 905, 906, {"signature": 0}),
        event(
            "host_readback",
            2,
            910,
            990,
            {
                "shape": [10],
                "dtype": "UInt32",
                "sample": {
                    "wait_ns": 10,
                    "copy_ns": 5,
                    "contiguous_build_ns": 0,
                    "bytes": 40,
                    "ready_before": False,
                    "direct": True,
                    "total_ns": 40,
                },
            },
        ),
    ]


def test_readback_components_and_kernel_requests_preserve_distinct_meanings():
    out = API["validate_events"](rich_trace())
    assert len(out["kernel_signatures"]) == 1 and len(out["kernel_requests"]) == 2
    assert out["kernel_signatures"][0]["source_bytes"] == 6
    sample = out["readbacks"][0]
    assert sample["layer"] == 0 and sample["bytes"] == 40 and sample["wait_ns"] == 10
    assert sample["copy_ns"] == 5 and sample["ready_before"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_signature",
        "bool_identity",
        "repeated_identity",
        "duplicate_key",
        "no_requests",
        "overflow",
        "backwards_sample",
        "copy_on_direct",
        "wrong_bytes",
        "wrong_shape",
        "invalid_dtype",
        "integer_flag",
        "outside_host_interval",
        "missing_sample",
    ],
)
def test_rich_profile_rejects_corrupt_or_incomplete_records(fault):
    rows = rich_trace()
    sample = rows[-1]["data"]["sample"]
    if fault == "unknown_signature":
        rows[5]["data"]["signature"] = 1
    elif fault == "bool_identity":
        rows[4]["data"]["id"] = False
    elif fault == "repeated_identity":
        rows[5] = event("expert_kernel_signature", 2, 903, 904, rows[4]["data"].copy())
    elif fault == "duplicate_key":
        rows[5] = event("expert_kernel_signature", 2, 903, 904, {**rows[4]["data"], "id": 1})
    elif fault == "no_requests":
        rows = [r for r in rows if r["name"] != "expert_kernel_request"]
    elif fault == "overflow":
        sample["total_ns"] = 1 << 64
    elif fault == "backwards_sample":
        sample["total_ns"] = 14
    elif fault == "copy_on_direct":
        sample["contiguous_build_ns"] = 1
    elif fault == "wrong_bytes":
        sample["bytes"] = 44
    elif fault == "wrong_shape":
        rows[-1]["data"]["shape"] = [True]
    elif fault == "invalid_dtype":
        rows[-1]["data"]["dtype"] = "unknown"
    elif fault == "integer_flag":
        sample["ready_before"] = 0
    elif fault == "outside_host_interval":
        sample["total_ns"] = 81
    elif fault == "missing_sample":
        del rows[-1]["data"]["sample"]["copy_ns"]
    with pytest.raises((AssertionError, KeyError)):
        API["validate_events"](rows)


@pytest.mark.parametrize("width", range(1, 9))
def test_real_multi_token_routes_reuse_experts_between_rows(width):
    rows = trace()
    data = rows[2]["data"]
    data.update(
        rows=width,
        ids=list(range(10)) * width,
        slots_hit=[[i, False] for i in range(10)] * width,
        cold_rows=10 * width,
    )
    out = API["validate_events"](rows)
    assert out["routes"][0]["pairs"] == 10 * (width - 1)
    assert out["routes"][0]["unique"] == 10


@pytest.mark.parametrize("width", [0, 9, True, -1])
def test_route_width_rejects_outside_production_contract(width):
    rows = trace()
    rows[2]["data"]["rows"] = width
    with pytest.raises(AssertionError):
        API["validate_events"](rows)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "negative",
        "backward",
        "parent",
        "nesting",
        "siblings",
        "booltime",
        "routes",
        "rowdup",
        "slot",
        "hits",
        "short",
        "bytes",
        "failed",
        "worker",
        "job",
        "busyworker",
    ],
)
def test_trace_rejects_corruption(fault):
    rows = trace()
    if fault == "empty":
        rows = []
    elif fault == "negative":
        rows[0]["start_ns"] = -1
    elif fault == "backward":
        rows[0]["end_ns"] = -1
    elif fault == "parent":
        rows[1]["parent"] = 1
    elif fault == "nesting":
        rows[2]["end_ns"] = 1001
    elif fault == "siblings":
        rows.append(event("overlap", 0, 500, 800))
    elif fault == "booltime":
        rows[0]["start_ns"] = False
    elif fault == "routes":
        rows[2]["data"]["ids"] = []
    elif fault == "rowdup":
        rows[2]["data"]["ids"][0] = 1
    elif fault == "slot":
        rows[2]["data"]["slots_hit"][0][0] = 53
    elif fault == "hits":
        rows[2]["data"]["hot_rows"] = 1
    elif fault == "short":
        rows[3]["data"]["timings_job_worker_start_end_bytes_failed"].pop()
    else:
        field = {"bytes": 4, "failed": 5, "worker": 1, "job": 0, "busyworker": 2}[fault]
        rows[3]["data"]["timings_job_worker_start_end_bytes_failed"][
            4 if fault == "busyworker" else 0
        ][field] = 1 if fault in ("failed", "worker", "job") else 0
    with pytest.raises((AssertionError, ValueError, TypeError)):
        API["validate_events"](rows)


def test_distribution_keeps_zero_and_tail_samples():
    assert API["distribution"]([0, 1, 2, 3, 100])["p99"] == 100
    assert API["distribution"]([0, 1])["min"] == 0
    for invalid in ([], [float("nan")], [-1], [float("inf")]):
        with pytest.raises(AssertionError):
            API["distribution"](invalid)


def test_compression_round_trip_is_deterministic_and_never_overwrites(tmp_path):
    raw = b'[{"name":"decode"}]'
    files = []
    for name in ("a", "b"):
        path = tmp_path / name
        path.write_bytes(raw)
        API["compress"](path)
        assert not path.exists()
        files.append(Path(str(path) + ".gz"))
    assert (
        files[0].read_bytes() == files[1].read_bytes()
        and gzip.decompress(files[0].read_bytes()) == raw
    )
    original = files[0].read_bytes()
    (tmp_path / "a").write_bytes(b"other")
    with pytest.raises(AssertionError):
        API["compress"](tmp_path / "a")
    assert files[0].read_bytes() == original and (tmp_path / "a").read_bytes() == b"other"


@pytest.mark.parametrize(
    "error", [RuntimeError("crashed"), TimeoutError("silent"), KeyboardInterrupt()]
)
def test_runner_failure_never_claims_parity_and_is_saved(tmp_path, monkeypatch, error):
    plan = tmp_path / "run.json"
    plan.write_text('[ {"profile_directory":"unused"} ]')
    api = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
    namespace = api["main"].__globals__
    namespace["ROOT"] = tmp_path
    (tmp_path / "opti.md").write_text("")

    def run_path(path):
        def failed(plan):
            raise error

        return {"main": failed}

    monkeypatch.setattr(namespace["runpy"], "run_path", run_path)
    with pytest.raises(type(error)):
        api["main"](plan)
    result = json.loads((tmp_path / "dataset-result.json").read_text())
    assert result["status"] == "failed_or_interrupted" and result["parity"] is False
    assert "failed_or_interrupted" in (tmp_path / "opti.md").read_text()
    with pytest.raises(AssertionError):
        api["main"](plan)


def test_vm_events_are_not_mislabeled_as_compressed_bytes():
    raw = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" + "\n".join(
        f"{k}: 7."
        for k in [
            "Compressions",
            "Decompressions",
            "Swapins",
            "Swapouts",
            "Pages occupied by compressor",
        ]
    )
    assert API["vm_counts"](raw)["Compressions"] == 7
    for bad in ("", raw + "\nCompressions: 8.", raw.replace("Swapouts:", "missing:")):
        with pytest.raises(AssertionError):
            API["vm_counts"](bad)


def summary_fixture():
    vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" + "\n".join(
        f"{k}: 7."
        for k in [
            "Compressions",
            "Decompressions",
            "Swapins",
            "Swapouts",
            "Pages occupied by compressor",
        ]
    )
    snap = {
        "memory": {
            "process_footprint_bytes": 12_000_000_000,
            "process_lifetime_peak_bytes": 12_000_000_000,
        },
        "vm_stat": vm,
    }
    return {
        "full_logits_target_head_states_byte_exact": True,
        "logit_count": 248320,
        "tokens": [0] * 17,
        "token_hash": "e7d16c6b24f5c8f5",
        "text": "known prefix",
        "steps": [{"step": i, "token": 0, "seconds": 0.1} for i in range(1, 17)],
        "prefill_seconds": 1.0,
        "decode_seconds": 2.0,
        "diagnostic_tps": 8.0,
        "before_counts": [[0, 0, 0]] * 48,
        "prefill_counts": [[10, 0, 0]] * 48,
        "after_counts": [[170, 0, 0]] * 48,
        "before": snap,
        "prefill_boundary": snap,
        "after": snap,
    }


def test_summary_full_shape_memory_boundary_and_prefix_are_checked():
    data = summary_fixture()
    result = API["validate_summary"](
        data, "quality", {"cache_context": "known prefix continued"}, [0] * 128
    )
    assert result["decode_vm_counter_pages_delta"]["Compressions"] == 0
    assert "not model-exclusive" in result["limits"]


def draft_summary_fixture():
    data = summary_fixture()
    data.update(
        mtp=False,
        dflash=True,
        full_draft_states_byte_exact=True,
        dflash_stats=[8, 8, 8, 1.5],
        dflash_profile={
            "draft_seconds": 0.1,
            "verification_seconds": 1.2,
            "commit_cache_seconds": 0.1,
            "verified_rows": 16,
            "drafted_rows": 16,
            "expert_demand_bytes": 0,
            "verified_widths": [0, 8, 0, 0, 0, 0, 0, 0],
            "drafted_widths": [0, 8, 0, 0, 0, 0, 0, 0],
        },
    )
    for row in data["steps"]:
        block = row["step"] % 2
        row.update(
            mtp_proposed=0,
            mtp_accepted=0,
            mtp_blocks=0,
            dflash={
                "proposed": block,
                "accepted": block,
                "blocks": block,
                "verified_rows": 2 * block,
                "drafted_rows": 2 * block,
                "expert_demand_bytes": 0,
            },
        )
    return data


def test_draft_summary_validates_full_kv_states_counters_and_width_histograms():
    API["validate_summary"](
        draft_summary_fixture(), "quality", {"cache_context": "known prefix continued"}
    )


@pytest.mark.parametrize(
    "fault",
    [
        "kv",
        "missing",
        "accepted",
        "verified",
        "width",
        "zero_width",
        "bytes",
        "bool",
        "time",
        "nan",
        "mode",
    ],
)
def test_draft_summary_rejects_corrupt_or_missing_verification(fault):
    data = draft_summary_fixture()
    if fault == "kv":
        data["full_draft_states_byte_exact"] = False
    elif fault == "missing":
        del data["steps"][0]["dflash"]
    elif fault == "accepted":
        data["steps"][0]["dflash"]["accepted"] = 2
    elif fault == "verified":
        data["dflash_profile"]["verified_rows"] = 15
    elif fault == "width":
        data["dflash_profile"]["verified_widths"][1] = 7
    elif fault == "zero_width":
        data["dflash_profile"]["drafted_widths"][0] = 1
    elif fault == "bytes":
        data["steps"][0]["dflash"]["expert_demand_bytes"] = 1
    elif fault == "bool":
        data["steps"][0]["mtp_proposed"] = False
    elif fault == "time":
        data["dflash_profile"]["verification_seconds"] = 2.0
    elif fault == "nan":
        data["dflash_stats"][3] = float("nan")
    elif fault == "mode":
        data["mtp"] = True
    with pytest.raises((AssertionError, KeyError)):
        API["validate_summary"](data, "quality", {"cache_context": "known prefix continued"})


def test_context_summary_requires_128_outputs_and_full_external_golden():
    data = summary_fixture()
    with pytest.raises(AssertionError):
        API["validate_summary"](data, "context", {"cache_context": "known prefix continued"})
    data["tokens"] = [0] * 128
    h = 0xCBF29CE484222325
    for _ in range(128 * 4):
        h = (h * 0x100000001B3) % (1 << 64)
    data.update(
        token_hash=f"{h:016x}",
        steps=[{"step": i, "token": 0, "seconds": 0.1} for i in range(1, 128)],
        diagnostic_tps=63.5,
    )
    golden = {"cache_context": "known prefix", "token_hash": data["token_hash"]}
    API["validate_summary"](data, "context", golden, [0] * 128)
    with pytest.raises(AssertionError):
        API["validate_summary"](data, "context", golden, [1] * 128)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "id",
        "hash",
        "text",
        "parity",
        "shape",
        "steps",
        "token",
        "nan",
        "tps",
        "counter",
        "layer",
        "peak",
        "footprint",
        "vm",
        "expected",
    ],
)
def test_summary_rejects_divergence_incomplete_nonfinite_and_memory_overrun(fault):
    data = summary_fixture()
    oracle = None
    if fault == "empty":
        data["tokens"] = []
    elif fault == "id":
        data["tokens"][0] = 248320
    elif fault == "hash":
        data["token_hash"] = "0" * 16
    elif fault == "text":
        data["text"] = "wrong"
    elif fault == "parity":
        data["full_logits_target_head_states_byte_exact"] = False
    elif fault == "shape":
        data["logit_count"] = 0
    elif fault == "steps":
        data["steps"].pop()
    elif fault == "token":
        data["steps"][0]["token"] = 1
    elif fault == "nan":
        data["decode_seconds"] = float("nan")
    elif fault == "tps":
        data["diagnostic_tps"] = 0
    elif fault == "counter":
        data["after_counts"][0] = [1, 0, 0]
    elif fault == "layer":
        data["prefill_counts"].pop()
    elif fault == "peak":
        data["after"]["memory"]["process_lifetime_peak_bytes"] += 1
    elif fault == "footprint":
        data["after"]["memory"]["process_footprint_bytes"] = 0
    elif fault == "vm":
        data["after"]["vm_stat"] = "invalid"
    elif fault == "expected":
        oracle = [1] * 128
    with pytest.raises(AssertionError):
        API["validate_summary"](
            data, "quality", {"cache_context": "known prefix continued"}, oracle
        )


@pytest.mark.parametrize("fails", [False, True])
def test_cumulative_guard_uses_campaign_vm_but_restores_per_case_snapshot(fails):
    before = {"Swapins": 200, "Swapouts": 30, "other": 7}
    record = {"before": {"vm_bytes": before}}
    baseline = {"Swapins": 20, "Swapouts": 30}

    def monitor(child, record, parse):
        assert record["before"]["vm_bytes"] == {"Swapins": 20, "Swapouts": 30, "other": 7}
        if fails:
            raise TimeoutError("guard exceeded")
        return 0

    wrapped = API["cumulative_monitor"](monitor, baseline)
    baseline["Swapouts"] = 999
    if fails:
        with pytest.raises(TimeoutError):
            wrapped(None, record, None)
    else:
        assert wrapped(None, record, None) == 0
    assert before == {"Swapins": 200, "Swapouts": 30, "other": 7}
    assert record["live_guard_basis"]["Swapouts"] == 30


@pytest.mark.parametrize("counter", [31, 29, True, None])
def test_cumulative_guard_rejects_any_live_swapout_or_invalid_counter(counter):
    baseline = {"Swapins": 20, "Swapouts": 30}
    before = {"Swapins": 200, "Swapouts": 30, "other": 7}
    record = {"before": {"vm_bytes": before}}

    def monitor(child, record, parse):
        parse("one live sample")
        pytest.fail("the invalid live sample escaped the guard")

    wrapped = API["cumulative_monitor"](monitor, baseline)
    with pytest.raises(AssertionError, match="live Swapouts"):
        wrapped(None, record, lambda _: {"Swapins": 200, "Swapouts": counter})
    assert before == {"Swapins": 200, "Swapouts": 30, "other": 7}


@pytest.mark.parametrize(
    "baseline",
    [{}, {"Swapouts": 0}, {"Swapins": True, "Swapouts": 0}, {"Swapins": -1, "Swapouts": 0}],
)
def test_cumulative_guard_rejects_invalid_baselines(baseline):
    with pytest.raises(AssertionError):
        API["cumulative_monitor"](lambda *_: 0, baseline)


def test_future_vm_baseline_rejected_without_mutation():
    record = {"before": {"vm_bytes": {"Swapins": 1, "Swapouts": 1}}}
    with pytest.raises(AssertionError):
        API["cumulative_monitor"](lambda *_: 0, {"Swapins": 2, "Swapouts": 0})(None, record, None)
    assert record["before"]["vm_bytes"] == {"Swapins": 1, "Swapouts": 1}


def test_loader_intercepts_actual_monitor_without_mutating_global_runpy(tmp_path):
    api = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
    g = api["run_sampler"].__globals__
    g["ROOT"] = tmp_path
    base = tmp_path / "docs/measurements/flash-next-ssd-impl"
    sample = base / "opt-49-variance-diagnostic/sample-system.txt"
    supervisor = base / "opt-45-audit-prerequisites/run-supervisor.txt"
    monitor = base / "opt-43-mtp-route-head/speed-heads/bench-supervisor.txt"
    for p in [sample, supervisor, monitor]:
        p.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        f"import runpy\ndef main(plan):return runpy.run_path({str(supervisor)!r})['main'](plan)\n"
    )
    supervisor.write_text(
        f"import runpy\ndef main(plan):return runpy.run_path({str(monitor)!r})['monitor'](None,plan,None)\n"
    )
    monitor.write_text(
        "def monitor(child,record,parse):return record['before']['vm_bytes'].copy()\nLIMIT = 256 << 20\n"
    )
    original = runpy.run_path
    before = {"Swapins": 100, "Swapouts": 20}
    r = {"before": {"vm_bytes": before}}
    out = api["run_sampler"](r, {"Swapins": 10, "Swapouts": 20})
    assert out == {"Swapins": 10, "Swapouts": 20} and before == {"Swapins": 100, "Swapouts": 20}
    assert runpy.run_path is original and monitor.read_text().startswith("def monitor")


@pytest.mark.parametrize(
    "counter,delta",
    [
        (counter, delta)
        for counter in ("Swapins", "Swapouts")
        for delta in (0, 1, 256 << 20, (256 << 20) + 1, -1, "missing", "bool")
    ]
    + [("Swapouts", "live")],
)
def test_real_sampler_rejects_exhausted_budget_before_starting_child(
    tmp_path, monkeypatch, counter, delta
):
    import shutil
    import subprocess
    import sys
    import threading
    from types import SimpleNamespace

    api = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
    base = Path("docs/measurements/flash-next-ssd-impl")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    paths = {
        "sample": base / "opt-49-variance-diagnostic/sample-system.txt",
        "supervisor": base / "opt-45-audit-prerequisites/run-supervisor.txt",
        "monitor": base / "opt-43-mtp-route-head/speed-heads/bench-supervisor.txt",
        "parser": Path("benchmarks/benchmark_flash_memory.py"),
        "process": Path("scripts/native_json_process.py"),
        "bridge": Path("scripts/benchmark-flash-bridge.py"),
    }
    for path in paths.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    baseline = {"Swapins": 100, "Swapouts": 100}
    current = baseline.copy()
    if delta == "missing":
        del current[counter]
    elif delta == "bool":
        current[counter] = True
    elif delta != "live":
        current[counter] += delta

    class FastEvent(threading.Event):
        def wait(self, timeout=None):
            return super().wait(min(timeout, 0.001) if timeout is not None else timeout)

    original = runpy.run_path

    def load(path):
        namespace = original(path)
        selected = Path(path).resolve()
        if selected == (tmp_path / paths["sample"]).resolve():
            glob = namespace["main"].__globals__
            glob["snapshot"] = lambda: {"errors": [], "test_fixture": True}
            glob["threading"] = SimpleNamespace(Event=FastEvent, Thread=threading.Thread)
        if selected == (tmp_path / paths["monitor"]).resolve():
            namespace["snapshot"] = lambda _: {
                "vm_bytes": current.copy(),
                "pmset -g batt": "Now drawing from 'AC Power'",
            }
            if delta == "live":

                def vm_stat(command, **kwargs):
                    assert command == ["vm_stat"]
                    assert kwargs == {"text": True, "timeout": 5}
                    return "synthetic VM counters"

                namespace["monitor"].__globals__["subprocess"] = SimpleNamespace(
                    TimeoutExpired=subprocess.TimeoutExpired,
                    check_output=vm_stat,
                )
        if selected == (tmp_path / paths["parser"]).resolve() and delta == "live":
            namespace["parse_vm"] = lambda _: {"Swapins": 100, "Swapouts": 101}
        return namespace

    glob = api["run_sampler"].__globals__
    monkeypatch.setitem(glob, "ROOT", tmp_path)
    monkeypatch.setitem(glob, "runpy", SimpleNamespace(run_path=load))
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "child-started"
    plan = tmp_path / "run.json"
    plan.write_text(
        json.dumps(
            [
                {
                    "name": "preflight",
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "import time; from pathlib import Path; Path('child-started').touch(); "
                            f"time.sleep({30 if delta == 'live' else 0})"
                        ),
                    ],
                    "env": {},
                    "expected_exit": 0,
                    "timeout_seconds": 5,
                }
            ]
        )
    )
    maximum = 0 if counter == "Swapouts" else 256 << 20
    allowed = type(delta) is int and 0 <= delta <= maximum
    if allowed:
        api["run_sampler"](plan, baseline)
    else:
        with pytest.raises((AssertionError, KeyError)):
            api["run_sampler"](plan, baseline)
    result = json.loads(plan.with_name("run-result.json").read_text())
    record = result["commands"][0]
    assert record["reaped"] and marker.exists() == (allowed or delta == "live")
    assert ("pid" in record) == (allowed or delta == "live")
    assert record["status"] == (
        "completed_expected_outcome" if allowed else "failed_or_interrupted"
    )
    assert record["before"]["vm_bytes"] == current
    assert record["preflight_guard"]["status"] == (
        "passed" if allowed or delta == "live" else "failed"
    )
    if delta == "live":
        assert record["maximum_live_swapout_bytes"] == 1
        assert record["last_live_vm_bytes"] == {"Swapins": 100, "Swapouts": 101}
        assert "live Swapouts changed" in record["error"]
    assert json.loads(plan.with_name("system-samples.json").read_text())["sampler_joined"]
    assert runpy.run_path is original


def test_preflight_budget_boundaries_are_exact_for_large_counters():
    import copy

    for start in (0, (1 << 64) - 1):
        for limit in (0, 1, 256 << 20):
            for counter in ("Swapins", "Swapouts"):
                for delta in (-1, 0, limit, limit + 1):
                    baseline = {"Swapins": start, "Swapouts": start}
                    before = baseline.copy()
                    before[counter] += delta
                    record = {"before": {"vm_bytes": before}}
                    original = copy.deepcopy(record)
                    if 0 <= delta <= (0 if counter == "Swapouts" else limit):
                        API["cumulative_preflight"](record, baseline, limit)
                        assert record["preflight_guard"]["status"] == "passed"
                    else:
                        with pytest.raises(AssertionError):
                            API["cumulative_preflight"](record, baseline, limit)
                        assert record["preflight_guard"]["status"] == "failed"
                    assert record["before"] == original["before"]
                    assert baseline == {"Swapins": start, "Swapouts": start}


@pytest.mark.parametrize("limit", [-1, True, None, 1.5])
def test_preflight_rejects_invalid_limits(limit):
    baseline = {"Swapins": 0, "Swapouts": 0}
    with pytest.raises(AssertionError):
        API["cumulative_preflight"]({"before": {"vm_bytes": baseline.copy()}}, baseline, limit)
