"""Disposable four-arm contract checks; no model, child process or GPU."""

import contextlib
import copy
import io
import json
import runpy
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / "benchmark-inspection.py"
F02 = BASE.parent / "post-p08-dense-epilogue-20261031"
fixtures = runpy.run_path(str(F02 / "inspection_selftest.py"))


def fixture(output, context, mode, budget, position):
    order = "AGFC" if budget == 17 else "AGFCCFGA"
    arm = order[position - 1]
    data = fixtures["fixture"](output, context, mode, budget, "C" if arm in "FC" else "A")
    entry, manifest, identity, plan, report, supervisor, _, _ = data
    entry.update(arm=arm, position_within_mode=position, gdn_flag="1" if arm in "GC" else "0")
    manifest["experiment"] = identity["experiment"] = context["EXPERIMENT"]
    manifest["plans"] = []
    for index, planned_arm in enumerate(order, 1):
        planned = copy.deepcopy(entry)
        planned.update(
            arm=planned_arm,
            gdn_flag="1" if planned_arm in "GC" else "0",
            f02_flag="1" if planned_arm in "FC" else "0",
            position_within_mode=index,
        )
        if index != position:
            planned["name"] += "_reserved_" + str(index)
        manifest["plans"].append(planned)
    for env in (plan[0]["env"], report["environment"], supervisor["commands"][0]["env"]):
        env["MLXL3_FLASH_PREPARED_CONV"] = entry["gdn_flag"]
    fixtures["add_swap32"](output, context, manifest, identity, supervisor)
    for path in (F02 / "benchmark-inspection.py", F02 / "bounded-gpu-runner.py"):
        identity["sources"][str(path)] = context["file_sha"](path)
    return data


def execute(context, output, mode="AR", budget=17, position=4, mutation=None):
    output.mkdir()
    data = fixture(output, context, mode, budget, position)
    entry, manifest, identity, plan, report, supervisor, _, _ = data
    values = dict(
        zip(
            ("entry", "manifest", "identity", "plan", "report", "supervisor", "source", "cli"), data
        )
    )
    if mutation:
        mutation(values)
    manifest_path, identity_path = output / "manifest.json", output / "identity.json"
    plan_path = Path(entry["plan"])
    for path, value in ((manifest_path, manifest), (plan_path, plan)):
        path.write_text(json.dumps(value))
        identity["sources"][str(path)] = context["file_sha"](path)
    identity_path.write_text(json.dumps(identity))
    report_path = Path(entry["expected_report"])
    if mutation is None:
        assert not report_path.exists()
        context["preflight"](entry, manifest_path, identity_path)
    if report is not None:
        report_path.write_text(json.dumps(report))
    plan_path.with_name(plan_path.stem + "-result.json").write_text(json.dumps(supervisor))
    error = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            context["inspect"](entry, manifest_path, identity_path)
    except (AssertionError, ValueError, KeyError, OSError) as raised:
        error = raised
    result_path = output / "external-inspection-v2.json"
    result = json.loads(result_path.read_text())
    if error:
        assert result["status"] == "failed_external_inspection", result
        assert result["external_golden"] is False and result.get("parity") is not True
    else:
        assert result["status"] == "passed_external_oracle_mode_and_memory_not_gain"
        assert result["external_golden"] is True and len(result["rows"]) == 4
        previous = result_path.read_bytes()
        try:
            context["inspect"](entry, manifest_path, identity_path)
        except AssertionError:
            pass
        else:
            raise AssertionError("existing inspection overwritten")
        assert result_path.read_bytes() == previous
    return error, result


def replace(target, path, value):
    return lambda values: fixtures["set_value"](values[target], path, value)


def main():
    context = runpy.run_path(str(SCRIPT))
    assert context["inspect"].__globals__["preflight"] is context["preflight"]
    assert context["inspect"].__globals__["validate"] is context["upstream"]["validate"]
    cases = []
    for mode in ("AR", "MTP", "DF"):
        for budget, positions in ((17, 4), (128, 8)):
            for position in range(1, positions + 1):
                cases.append((f"positive_{mode}_{budget}_{position}", mode, budget, position, None))
    negative = [
        ("wrong_gdn_report", replace("report", ["environment", "MLXL3_FLASH_PREPARED_CONV"], "0")),
        (
            "missing_gdn_report",
            lambda d: d["report"]["environment"].pop("MLXL3_FLASH_PREPARED_CONV"),
        ),
        ("wrong_f02_report", replace("report", ["environment", "MLXL3_FLASH_DENSE_EPILOGUE"], "0")),
        (
            "wrong_prefill_report",
            replace("report", ["environment", "MLXL3_FLASH_PREFILL_HEADS"], "0"),
        ),
        ("missing_gdn_plan", lambda d: d["plan"][0]["env"].pop("MLXL3_FLASH_PREPARED_CONV")),
        ("wrong_gdn_plan", replace("plan", [0, "env", "MLXL3_FLASH_PREPARED_CONV"], "0")),
        ("wrong_f02_plan", replace("plan", [0, "env", "MLXL3_FLASH_DENSE_EPILOGUE"], "0")),
        ("arm_flags", replace("manifest", ["plans", 0, "gdn_flag"], "1")),
        ("order", lambda d: d["manifest"]["plans"].reverse()),
        ("order_position", replace("manifest", ["plans", 0, "position_within_mode"], 9)),
        ("duplicate_name", lambda d: d["manifest"]["plans"][0].update(name=d["entry"]["name"])),
        ("wrong_campaign", replace("identity", ["experiment"], "POSTP08-F02-01")),
        ("no_sources", replace("identity", ["sources"], {})),
        ("changed_source", lambda d: d["source"].write_text("changed")),
        ("changed_binary", lambda d: d["cli"].write_bytes(b"changed")),
        (
            "unfrozen_upstream",
            lambda d: d["identity"]["sources"].pop(str(F02 / "benchmark-inspection.py")),
        ),
        ("wrong_mode_command", lambda d: d["plan"][0]["command"].append("--mtp")),
        ("wrong_mode_report", replace("report", ["mode"], "mtp")),
        ("quiet_disabled", replace("plan", [0, "require_quiet"], False)),
        ("memory_admission", replace("plan", [0, "minimum_available_bytes"], 2_000_000_000)),
        ("timeout_nan", replace("plan", [0, "timeout_seconds"], float("nan"))),
        ("incomplete", replace("report", ["status"], "incomplete")),
        ("rows_missing", replace("report", ["runs"], [])),
        ("reader_unjoined", replace("report", ["reader_joined"], False)),
        ("supervisor_failed", replace("supervisor", ["commands", 0, "status"], "failed")),
        ("supervisor_unjoined", replace("supervisor", ["commands", 0, "reaped"], False)),
        (
            "supervisor_flag",
            replace("supervisor", ["commands", 0, "env", "MLXL3_FLASH_PREPARED_CONV"], "0"),
        ),
        ("golden_diverged", replace("report", ["runs", 1, "token_hash"], "0" * 16)),
        ("partial_tokens", replace("report", ["runs", 0, "stats", "generated_tokens"], 16)),
        (
            "zero_memory",
            replace("report", ["runs", 0, "stats", "memory", "process_footprint_bytes"], 0),
        ),
        (
            "over_rss",
            replace("supervisor", ["commands", 0, "children_maximum_rss_bytes"], (10 << 30) + 1),
        ),
        (
            "swap_over_budget",
            replace("supervisor", ["commands", 0, "maximum_live_swapout_bytes"], (32 << 20) + 1),
        ),
        ("swap_budget_missing", lambda d: d["identity"].pop("maximum_new_swapout_bytes")),
        ("swap_reset", replace("supervisor", ["commands", 0, "after", "vm_bytes", "Swapouts"], 0)),
    ]
    cases.extend((name, "AR", 17, 4, change) for name, change in negative)
    for mode, prefix in (("MTP", "mtp"), ("DF", "dflash")):
        cases.append(
            (
                "invalid_counters_" + mode,
                mode,
                17,
                4,
                replace("report", ["runs", 0, "stats", prefix + "_accepted_tokens"], 100),
            )
        )
    results = []
    with (
        tempfile.TemporaryDirectory(prefix="compose-inspection-") as temporary,
        contextlib.chdir(temporary),
    ):
        for name, mode, budget, position, mutation in cases:
            error, result = execute(
                context, Path(temporary) / name, mode, budget, position, mutation
            )
            assert (error is not None) == (mutation is not None), (name, repr(error))
            results.append(
                {
                    "name": name,
                    "expected_rejection": mutation is not None,
                    "status": result["status"],
                }
            )
        # Remove the real GDN report comparison in a disposable copy.
        original = SCRIPT.read_text()
        clause = 'report["environment"]["MLXL3_FLASH_PREPARED_CONV"] == entry["gdn_flag"]'
        assert original.count(clause) == 1
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="mutant-", dir=BASE
        ) as source:
            source.write(original.replace(clause, "True"))
            source.flush()
            mutant = runpy.run_path(source.name)
            error, _ = execute(
                mutant, Path(temporary) / "mutant_wrong_gdn", mutation=negative[0][1]
            )
            assert error is None, "mutation did not bypass GDN guard"
        assert SCRIPT.read_text() == original
    print(
        json.dumps(
            {
                "status": "passed_synthetic_selftests_not_benchmark",
                "engine_executed": False,
                "controller_sha256": context["file_sha"](SCRIPT),
                "positive_cases": 36,
                "negative_cases": len(cases) - 36,
                "mutation": "missing GDN report check detected; production unchanged",
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
