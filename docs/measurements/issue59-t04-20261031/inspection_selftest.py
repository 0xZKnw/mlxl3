"""Exercise the real inspector with disposable synthetic copies, never an engine."""

import contextlib
import hashlib
import io
import json
import runpy
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "docs/measurements/issue59-t04-20261031"


def set_value(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def main():
    with contextlib.chdir(ROOT):
        context = runpy.run_path(str(FOLDER / "benchmark-inspection.txt"))
    inspector = context["inspect"]
    assert inspector.__globals__["validate"] is context["validate"]
    assert inspector.__globals__["golden_check"] is context["golden_check"]
    row = ["warmup", 0]
    cases = [("positive_" + mode, mode, None, [], None, "") for mode in ["AR", "MTP", "DF"]] + [
        ("incomplete", "AR", "report", ["status"], "incomplete", ""),
        ("empty_text", "AR", "report", row + ["text"], "", "empty output"),
        (
            "wrong_hash",
            "AR",
            "report",
            row + ["token_hash"],
            "0000000000000000",
            "external token oracle",
        ),
        (
            "mtp_null",
            "MTP",
            "report",
            row + ["stats", "mtp_proposed_tokens"],
            None,
            "invalid MTP counters",
        ),
        (
            "df_null",
            "DF",
            "report",
            row + ["stats", "dflash_proposed_tokens"],
            None,
            "invalid DFlash counters",
        ),
        (
            "zero_footprint",
            "AR",
            "report",
            row + ["stats", "memory", "process_footprint_bytes"],
            0,
            "process_footprint_bytes",
        ),
        (
            "peak_limit",
            "DF",
            "report",
            row + ["stats", "memory", "process_lifetime_peak_bytes"],
            12_000_000_000,
            "process_lifetime_peak_bytes",
        ),
        (
            "nan_duration",
            "MTP",
            "report",
            row + ["stats", "decode_seconds"],
            float("nan"),
            "invalid timing decode_seconds",
        ),
        ("missing_prompt", "AR", "report", ["prompts"], {}, ""),
        (
            "mtp_inactive",
            "MTP",
            "report",
            row + ["generation_mode", "mtp_active"],
            False,
            "engine mode mismatch",
        ),
        (
            "df_inactive",
            "DF",
            "report",
            row + ["generation_mode", "dflash_active"],
            False,
            "engine mode mismatch",
        ),
        (
            "supervisor_unreaped",
            "AR",
            "supervisor",
            ["commands", 0, "reaped"],
            False,
            "",
        ),
    ]
    results = []
    with (
        tempfile.TemporaryDirectory(prefix="t04-inspection-selftest-") as temporary,
        contextlib.chdir(temporary),
    ):
        for label, mode, target, path, replacement, reason in cases:
            output = Path(temporary) / label
            output.mkdir()
            source = (
                ROOT
                / "docs/measurements/issue59-t17-20261029/prepared/runs"
                / f"t17-bridge17-{mode}-2-C/report.json"
            )
            report = json.loads(source.read_text())
            # These remain SYNTHETIC fixtures, not observations of this CLI.
            report["selftest_synthetic_not_measurement"] = True
            report["engine"] = context["expected_binary"]["path"]
            report["binary_sha256"] = context["expected_binary"]["sha256"]
            report["environment"].update(
                MLXL3_FLASH_SWIGLU_DOWN="1",
                MLXL3_FLASH_GDN_EPILOGUE="0",
                MLXL3_SSD_WEIGHTED_EPILOGUE="0",
            )
            report["runtime"]["runtime_executable"] = report["engine"]
            plan = output / "synthetic-plan.json"
            plan.write_text(
                json.dumps(
                    [
                        {
                            "env": {
                                "MLX_MAX_OPS_PER_BUFFER": "40",
                                "MLX_MAX_MB_PER_BUFFER": "40",
                            }
                        }
                    ]
                )
            )
            supervisor = {
                "status": "completed_expected_outcomes_not_speed",
                "commands": [
                    {
                        "status": "completed_expected_outcome",
                        "exit_code": 0,
                        "reaped": True,
                        "swapouts_bytes": 0,
                    }
                ],
            }
            if target:
                set_value({"report": report, "supervisor": supervisor}[target], path, replacement)
            report_path = output / "report.json"
            report_path.write_text(json.dumps(report))
            plan.with_name(plan.stem + "-result.json").write_text(json.dumps(supervisor))
            entry = {
                "name": "SYNTHETIC_SELFTEST_" + label,
                "phase": "selftest",
                "mode": mode,
                "arm": "C",
                "t04_flag": "1",
                "output": str(output),
                "expected_report": str(report_path),
                "plan": str(plan),
                "tokens_per_generation": 17,
            }
            raised = None
            try:
                # Suppress inherited measured-looking stdout from fake fixtures.
                with contextlib.redirect_stdout(io.StringIO()):
                    inspector(entry)
            except (AssertionError, ValueError) as error:
                raised = error
            inspected = json.loads((output / "external-inspection-v2.json").read_text())
            if target:
                assert raised is not None, label
                assert reason in str(raised), (label, repr(raised))
                assert inspected["status"] == "failed_external_inspection", label
                assert inspected.get("error") and inspected.get("parity") is not True
            else:
                assert raised is None, (label, repr(raised))
                assert inspected["status"] == "passed_external_oracle_mode_and_memory_not_gain"
                assert len(inspected["rows"]) == 4
            results.append(
                {
                    "case": label,
                    "mode": mode,
                    "expected_rejection": bool(target),
                    "status": inspected["status"],
                }
            )
        assert Path("opti.md").read_text().count("SYNTHETIC_SELFTEST_") == len(cases)
    summary = {
        "status": "passed_synthetic_selftests_not_benchmark",
        "engine_executed": False,
        "synthetic_fixtures": True,
        "controller_sha256": hashlib.sha256(
            (FOLDER / "benchmark-inspection.txt").read_bytes()
        ).hexdigest(),
        "positive_cases": 3,
        "negative_cases": 12,
        "cases": results,
        "journaling": "temporary cwd only; fixtures removed",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
