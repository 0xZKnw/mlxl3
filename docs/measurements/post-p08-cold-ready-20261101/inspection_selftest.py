"""Synthetic Cold bridge admission and durable-failure tests; no engine or GPU."""

import contextlib
import copy
import json
import runpy
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / "benchmark-inspection.py"
F02 = BASE.parent / "post-p08-dense-epilogue-20261031"
fixtures = runpy.run_path(str(F02 / "inspection_selftest.py"))
reuse = runpy.run_path(str(BASE.parent / "post-p08-compose-20261101/inspection_selftest.py"))


def fixture(output, context, mode, budget, position):
    order = "AB" if budget == 17 else "ABBA"
    arm = order[position - 1]
    data = fixtures["fixture"](output, context, mode, budget, "A")
    entry, manifest, identity, plan, report, supervisor, _, _ = data
    entry.update(
        arm=arm,
        position_within_mode=position,
        gdn_flag="0",
        cold_ready_flag={"A": "0", "B": "1"}[arm],
    )
    manifest["experiment"] = identity["experiment"] = context["EXPERIMENT"]
    manifest["plans"] = []
    for index, planned_arm in enumerate(order, 1):
        planned = copy.deepcopy(entry)
        planned.update(
            arm=planned_arm,
            cold_ready_flag={"A": "0", "B": "1"}[planned_arm],
            position_within_mode=index,
        )
        if index != position:
            planned["name"] += "_reserved_" + str(index)
        manifest["plans"].append(planned)
    for env in (plan[0]["env"], report["environment"], supervisor["commands"][0]["env"]):
        env["MLXL3_SSD_COLD_READY"] = entry["cold_ready_flag"]
    anchor_path = (output / "anchor-gpu1.json").resolve()
    policy_path = (output / "policy-gpu1.json").resolve()
    runner = context["gpu_runner"]
    runner["load_baseline"].__globals__.update(ANCHOR_PATH=anchor_path, POLICY_PATH=policy_path)
    vm = {"Swapouts": 75119296512, "Swapins": 30750539776}
    anchor_path.write_text(
        json.dumps(
            {
                "experiment": context["EXPERIMENT"],
                "timestamp": "2026-11-01T13:00:00+00:00",
                "vm_bytes": vm,
            }
        )
    )
    metadata = {
        "maximum_new_swapout_bytes": 32 << 20,
        "swap_anchor_path": str(anchor_path),
        "swap_anchor_sha256": context["file_sha"](anchor_path),
    }
    policy_path.write_text(
        json.dumps(
            {
                "experiment": context["EXPERIMENT"],
                "authorized_by_user": runner["AUTHORIZATION"],
                "anchor_path": str(anchor_path),
                "anchor_sha256": metadata["swap_anchor_sha256"],
                "maximum_new_swapout_bytes": 32 << 20,
                "maximum_new_swapin_bytes": 256 << 20,
                "no_reanchor": True,
                "previous_anchor": runner["PREVIOUS_ANCHOR"],
            }
        )
    )
    manifest.update(**metadata, policy_amendment=str(policy_path))
    identity.update(metadata)
    identity["binaries"]["libtest"] = dict(identity["binaries"]["final_cli"])
    for moment in ("before", "after"):
        supervisor["commands"][0][moment] = {"vm_bytes": dict(vm)}
    for path in (
        anchor_path,
        policy_path,
        *context["DEPENDENCIES"],
        F02 / "benchmark-inspection.py",
    ):
        identity["sources"][str(path)] = context["file_sha"](path)
    return data


execute = reuse["execute"]
execute.__globals__["fixture"] = fixture


def replace(target, path, value):
    return lambda values: fixtures["set_value"](values[target], path, value)


def main():
    context = runpy.run_path(str(SCRIPT))
    cases = [
        (f"positive_{mode}_{budget}_{position}", mode, budget, position, None)
        for mode in ("AR", "MTP", "DF")
        for budget, positions in ((17, 2), (128, 4))
        for position in range(1, positions + 1)
    ]
    negative = [
        ("wrong_cold_report", replace("report", ["environment", "MLXL3_SSD_COLD_READY"], "0")),
        ("missing_cold_report", lambda d: d["report"]["environment"].pop("MLXL3_SSD_COLD_READY")),
        ("wrong_cold_plan", replace("plan", [0, "env", "MLXL3_SSD_COLD_READY"], "0")),
        ("missing_cold_plan", lambda d: d["plan"][0]["env"].pop("MLXL3_SSD_COLD_READY")),
        ("wrong_gdn_report", replace("report", ["environment", "MLXL3_FLASH_PREPARED_CONV"], "1")),
        ("wrong_f02_report", replace("report", ["environment", "MLXL3_FLASH_DENSE_EPILOGUE"], "1")),
        ("wrong_arm_flag", replace("manifest", ["plans", 0, "cold_ready_flag"], "1")),
        ("order", lambda d: d["manifest"]["plans"].reverse()),
        ("reader_unjoined", replace("report", ["reader_joined"], False)),
        ("supervisor_unjoined", replace("supervisor", ["commands", 0, "reaped"], False)),
        ("golden_diverged", replace("report", ["runs", 0, "token_hash"], "0" * 16)),
        ("rows_missing", replace("report", ["runs"], [])),
        ("wrong_mode", replace("report", ["mode"], "mtp")),
        ("wrong_command", lambda d: d["plan"][0]["command"].append("--mtp")),
        ("incomplete", replace("report", ["status"], "incomplete")),
        ("wrong_anchor", replace("identity", ["swap_anchor_path"], "/another/anchor.json")),
        ("wrong_budget", replace("identity", ["maximum_new_swapout_bytes"], (32 << 20) + 1)),
        ("unfrozen_dependency", lambda d: d["identity"]["sources"].pop(str(context["COMPOSE"]))),
        ("missing_libtest", lambda d: d["identity"]["binaries"].pop("libtest")),
        (
            "swap_exceeded",
            replace("supervisor", ["commands", 0, "maximum_live_swapout_bytes"], (32 << 20) + 1),
        ),
    ]
    cases.extend((name, "AR", 17, 2, change) for name, change in negative)

    # Keep this policy mutation independent of source-hash rejection.
    def unauthorized(values):
        path = Path(values["manifest"]["policy_amendment"])
        policy = json.loads(path.read_text())
        policy["authorized_by_user"] = "not authorized"
        path.write_text(json.dumps(policy))
        values["identity"]["sources"][str(path)] = context["file_sha"](path)

    cases.append(("wrong_policy", "AR", 17, 2, unauthorized))
    results = []
    with (
        tempfile.TemporaryDirectory(prefix="cold-inspection-") as temporary,
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
        original = SCRIPT.read_text()
        report_check = 'report["environment"]["MLXL3_SSD_COLD_READY"] == entry["cold_ready_flag"]'
        inherited_check = 'ENV={**ENV, "MLXL3_SSD_COLD_READY": entry.get("cold_ready_flag")},'
        assert original.count(report_check) == original.count(inherited_check) == 1
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="mutant-", dir=BASE
        ) as source:
            source.write(
                original.replace(report_check, "True").replace(inherited_check, "ENV=ENV,")
            )
            source.flush()
            mutant = runpy.run_path(source.name)
            error, _ = execute(
                mutant, Path(temporary) / "mutant_cold_report", position=2, mutation=negative[0][1]
            )
            assert error is None, "mutation failed to bypass the actual report controls"
        assert SCRIPT.read_text() == original
    print(
        json.dumps(
            {
                "status": "passed_synthetic_selftests_not_benchmark",
                "engine_executed": False,
                "positive_cases": 18,
                "negative_cases": len(cases) - 18,
                "mutation": "disabled actual Cold report controls detected; production unchanged",
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
