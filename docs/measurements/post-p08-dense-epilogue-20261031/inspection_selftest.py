"""Disposable F02 checks derived from T18a's 17 synthetic inspector cases.

No private reports, model, compiled CLI or inference are required. Synthetic
fixtures cannot be confused with observations of the future F02 executable.
"""

import contextlib
import copy
import io
import json
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent


def set_value(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def fixture(output, context, mode, budget, arm):
    cli = output / "synthetic-cli-not-executable"
    cli.write_bytes(b"F02 SYNTHETIC bytes; never execute")
    source = output / "synthetic-source.txt"
    source.write_text("frozen synthetic source")
    binary = {"path": str(cli), "sha256": context["file_sha"](cli)}
    model, draft = output / "synthetic-model", output / "synthetic-draft"
    model.mkdir()
    draft.mkdir()
    goldens = {}
    for label, start, digest in [
        ("MAIN", 0, "709f2462c272f725"),
        ("HOLD", 128, "fda94c7e5c47db25"),
    ]:
        path = output / f"gold-{label}.json"
        path.write_text(
            json.dumps(
                {
                    "tokens": list(range(start, start + 128)),
                    "token_hash": digest,
                    "cache_context": f"{label} synthetic complete output",
                }
            )
        )
        goldens[label] = str(path)
    plan = output / "plan.json"
    manifest_path = output / "manifest.json"
    flag = {"A": "0", "C": "1"}[arm]
    entry = {
        "name": "SYNTHETIC_SELFTEST_" + output.name,
        "phase": "bridge17" if budget == 17 else "pilot128",
        "mode": mode,
        "arm": arm,
        "f02_flag": flag,
        "output": str(output),
        "expected_report": str(output / "report.json"),
        "plan": str(plan),
        "tokens_per_generation": budget,
        "position_within_mode": 1 if arm == "A" else 2,
    }
    manifest = {
        "experiment": "POSTP08-F02-01",
        "model": str(model),
        "dflash_draft": str(draft),
        "external_golden_files_for_root_validation": goldens,
        "plans": [copy.deepcopy(entry)],
    }
    manifest["plans"] = []
    order = ["A", "C"] if budget == 17 else ["A", "C", "C", "A"]
    for position, planned_arm in enumerate(order, 1):
        planned = copy.deepcopy(entry)
        planned.update(
            arm=planned_arm,
            f02_flag={"A": "0", "C": "1"}[planned_arm],
            position_within_mode=position,
        )
        if position != entry["position_within_mode"]:
            planned["name"] += "_reserved_" + str(position)
        manifest["plans"].append(planned)
    mode_args = {
        "AR": [],
        "MTP": ["--mtp"],
        "DF": ["--dflash-draft", str(draft), "--dflash-mode", "1"],
    }[mode]
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark-flash-bridge.py"),
        str(cli),
        str(model),
        str(output),
        "--cache-gib",
        "3",
        "--tokens",
        str(budget),
        "--repeats",
        "1",
        "--timeout",
        "180",
        *mode_args,
    ]
    env = {**context["ENV"], "MLXL3_FLASH_DENSE_EPILOGUE": flag}
    plan_data = [
        {
            "name": entry["name"],
            "command": command,
            "env": env,
            "require_quiet": True,
            "expected_exit": 0,
            "minimum_available_bytes": 10_500_000_000,
            "timeout_seconds": 900,
        }
    ]
    plan.write_text(json.dumps(plan_data))
    manifest_path.write_text(json.dumps(manifest))
    frozen = [
        Path(context["__file__"]),
        ROOT / "scripts/benchmark-flash-bridge.py",
        ROOT / "docs/measurements/issue58-20261029/pool-speed-driver.py",
        manifest_path,
        plan,
        source,
        *(Path(value) for value in goldens.values()),
    ]
    identity = {
        "experiment": "POSTP08-F02-01",
        "sources": {str(p): context["file_sha"](p) for p in frozen},
        "binaries": {
            "final_cli": binary,
            "python_runner": {
                "path": sys.executable,
                "sha256": context["file_sha"](sys.executable),
            },
        },
    }
    report = {
        "selftest_synthetic_not_measurement": True,
        "status": "validated",
        "reaped": True,
        "reader_joined": True,
        "engine": str(cli),
        "binary_sha256": binary["sha256"],
        "model": str(model),
        "tokens": budget,
        "expert_cache_gib": 3,
        "repeats": 1,
        "prompts": context["bridge"]["PROMPTS"],
        "mode": {"AR": "baseline", "MTP": "mtp", "DF": "dflash2"}[mode],
        "environment": env.copy(),
        "runtime": {"runtime_executable": str(cli)},
        "dflash_draft": str(draft) if mode == "DF" else None,
        "warmup": [],
        "runs": [],
    }
    pool = runpy.run_path(str(ROOT / "docs/measurements/issue58-20261029/pool-speed-driver.py"))
    for group, repeat in [("warmup", 0), ("runs", 1)]:
        for label in ("MAIN", "HOLD"):
            golden = json.loads(Path(goldens[label]).read_text())
            prompt_tokens = 87 if label == "MAIN" else 82
            stats = {
                "generated_tokens": budget,
                "decode_tokens": budget - 1,
                "decode_seconds": 2.0,
                "decode_tps": (budget - 1) / 2.0,
                "prompt_tokens": prompt_tokens,
                "prefill_seconds": 1.5,
                "prefill_tps": prompt_tokens / 1.5,
                "ttft_seconds": 1.6,
                "memory": {
                    "process_footprint_bytes": 512 << 20,
                    "process_lifetime_peak_bytes": 768 << 20,
                },
            }
            for prefix, active in (("mtp", mode == "MTP"), ("dflash", mode == "DF")):
                stats.update(
                    {
                        prefix + "_proposed_tokens": 2 if active else 0,
                        prefix + "_accepted_tokens": 1 if active else 0,
                        prefix + "_blocks": 2 if active else 0,
                    }
                )
            gm = {
                "mtp_active": mode == "MTP",
                "mtp_requested": mode == "MTP",
                "mtp_depth": 1 if mode == "MTP" else 0,
                "dflash_active": mode == "DF",
                "dflash_requested": mode == "DF",
                "dflash_mode": 1 if mode == "DF" else 0,
                "dflash_proposals": 1 if mode == "DF" else 0,
                "dflash_draft_path": str(draft) if mode == "DF" else "",
            }
            report[group].append(
                {
                    "prompt": label,
                    "repeat": repeat,
                    "mode": report["mode"],
                    "stats": stats,
                    "generation_mode": gm,
                    "text": golden["cache_context"] if budget == 128 else label,
                    "token_hash": pool["token_hash"](golden["tokens"][:budget]),
                }
            )
    supervisor = {
        "status": "completed_expected_outcomes_not_speed",
        "commands": [
            {
                "name": entry["name"],
                "command": command,
                "env": env.copy(),
                "status": "completed_expected_outcome",
                "exit_code": 0,
                "reaped": True,
                "swapouts_bytes": 0,
                "swapins_bytes": 0,
                "maximum_live_swapout_bytes": 0,
                "maximum_live_swapin_bytes": 0,
                "children_maximum_rss_bytes": 1 << 30,
            }
        ],
    }
    return entry, manifest, identity, plan_data, report, supervisor, source, cli


def add_swap32(output, context, manifest, identity, supervisor):
    """Disposable original anchor and authorization; no campaign files are changed."""
    anchor = output / "original-anchor.json"
    anchor.write_text(json.dumps({"vm_bytes": {"Swapouts": 262144, "Swapins": 0}}))
    metadata = {
        "maximum_new_swapout_bytes": 32 << 20,
        "swap_anchor_path": str(anchor),
        "swap_anchor_sha256": context["file_sha"](anchor),
    }
    amendment = output / "policy-amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "original_anchor": str(anchor),
                "original_anchor_sha256": metadata["swap_anchor_sha256"],
                "maximum_new_swapout_bytes": 32 << 20,
                "baseline_swapout_bytes": 262144,
                "no_reanchor": True,
            }
        )
    )
    manifest.update(metadata, policy_amendment=str(amendment))
    identity.update(metadata)
    identity["sources"].update(
        {str(path): context["file_sha"](path) for path in (anchor, amendment)}
    )
    process = supervisor["commands"][0]
    process.update(
        before={"vm_bytes": {"Swapouts": 262144 + (512 << 10), "Swapins": 0}},
        after={"vm_bytes": {"Swapouts": 262144 + (512 << 10), "Swapins": 0}},
        maximum_live_swapout_bytes=512 << 10,
    )


def main():
    context = runpy.run_path(str(FOLDER / "benchmark-inspection.py"))
    inspector = context["inspect"]
    assert inspector.__globals__["validate"] is context["validate"]
    assert inspector.__globals__["golden_check"] is context["golden_check"]
    row = ["warmup", 0]
    # The first 17 retain T18a's three positives and fourteen rejection scenarios.
    cases = [
        ("positive_" + mode, mode, 17, "C", None, [], None, "") for mode in ("AR", "MTP", "DF")
    ] + [
        ("incomplete", "AR", 17, "C", "report", ["status"], "incomplete", ""),
        ("empty_text", "AR", 17, "C", "report", row + ["text"], "", "empty output"),
        (
            "wrong_hash",
            "AR",
            17,
            "C",
            "report",
            row + ["token_hash"],
            "0" * 16,
            "external token oracle",
        ),
        (
            "mtp_null",
            "MTP",
            17,
            "C",
            "report",
            row + ["stats", "mtp_proposed_tokens"],
            None,
            "invalid MTP counters",
        ),
        (
            "df_null",
            "DF",
            17,
            "C",
            "report",
            row + ["stats", "dflash_proposed_tokens"],
            None,
            "invalid DFlash counters",
        ),
        (
            "zero_footprint",
            "AR",
            17,
            "C",
            "report",
            row + ["stats", "memory", "process_footprint_bytes"],
            0,
            "process_footprint_bytes",
        ),
        (
            "peak_limit",
            "DF",
            17,
            "C",
            "report",
            row + ["stats", "memory", "process_lifetime_peak_bytes"],
            12_000_000_000,
            "process_lifetime_peak_bytes",
        ),
        (
            "nan_duration",
            "MTP",
            17,
            "C",
            "report",
            row + ["stats", "decode_seconds"],
            float("nan"),
            "invalid timing decode_seconds",
        ),
        ("missing_prompt", "AR", 17, "C", "report", ["prompts"], {}, ""),
        (
            "mtp_inactive",
            "MTP",
            17,
            "C",
            "report",
            row + ["generation_mode", "mtp_active"],
            False,
            "engine mode mismatch",
        ),
        (
            "df_inactive",
            "DF",
            17,
            "C",
            "report",
            row + ["generation_mode", "dflash_active"],
            False,
            "engine mode mismatch",
        ),
        ("supervisor_unreaped", "AR", 17, "C", "supervisor", ["commands", 0, "reaped"], False, ""),
        (
            "f02_flag_mismatch",
            "AR",
            17,
            "C",
            "report",
            ["environment", "MLXL3_FLASH_DENSE_EPILOGUE"],
            "0",
            "report F02 flag",
        ),
        (
            "t04_not_disabled",
            "AR",
            17,
            "C",
            "report",
            ["environment", "MLXL3_FLASH_SWIGLU_DOWN"],
            "1",
            "MLXL3_FLASH_SWIGLU_DOWN",
        ),
    ]
    cases += [
        ("positive128_" + mode, mode, 128, "C", None, [], None, "") for mode in ("AR", "MTP", "DF")
    ]
    cases += [
        ("positiveA_" + mode, mode, 17, "A", None, [], None, "") for mode in ("AR", "MTP", "DF")
    ]
    cases += [
        ("positiveA128_" + mode, mode, 128, "A", None, [], None, "") for mode in ("AR", "MTP", "DF")
    ]
    cases += [
        (
            "t18_not_disabled",
            "AR",
            17,
            "C",
            "report",
            ["environment", "MLXL3_SSD_DYNAMIC_READERS"],
            "1",
            "MLXL3_SSD_DYNAMIC_READERS",
        ),
        ("report_binary_hash", "AR", 17, "C", "report", ["binary_sha256"], "0" * 64, ""),
        ("wrong_mode", "AR", 17, "C", "report", ["mode"], "mtp", ""),
        ("missing_raw", "AR", 17, "C", "missing_raw", [], None, ""),
        ("malformed_raw", "AR", 17, "C", "malformed_raw", [], None, ""),
        (
            "partial_tokens",
            "AR",
            17,
            "C",
            "report",
            row + ["stats", "generated_tokens"],
            16,
            "incomplete generation",
        ),
        ("partial_rows", "AR", 17, "C", "report", ["runs"], [], ""),
        (
            "late_divergence",
            "AR",
            128,
            "C",
            "report",
            ["runs", 1, "token_hash"],
            "0" * 16,
            "external token oracle",
        ),
        (
            "nan_prefill",
            "AR",
            17,
            "C",
            "report",
            row + ["stats", "prefill_seconds"],
            float("nan"),
            "prefill duration",
        ),
        ("reader_unjoined", "AR", 17, "C", "report", ["reader_joined"], False, ""),
        ("report_error", "AR", 17, "C", "report", ["error"], "late error", ""),
        ("supervisor_nonzero", "AR", 17, "C", "supervisor", ["commands", 0, "exit_code"], 1, ""),
        ("swapout", "AR", 17, "C", "supervisor", ["commands", 0, "swapouts_bytes"], 16384, ""),
        (
            "swapin_limit",
            "AR",
            17,
            "C",
            "supervisor",
            ["commands", 0, "swapins_bytes"],
            (256 << 20) + 1,
            "",
        ),
        (
            "rss_limit",
            "AR",
            17,
            "C",
            "supervisor",
            ["commands", 0, "children_maximum_rss_bytes"],
            (10 << 30) + 1,
            "",
        ),
        ("source_hash", "AR", 17, "C", "source_hash", [], None, "source identity"),
        ("binary_changed", "AR", 17, "C", "binary_changed", [], None, "binary identity"),
        (
            "old_manifest",
            "AR",
            17,
            "C",
            "manifest",
            ["experiment"],
            "ISSUE59-T18A-01",
            "campaign identity",
        ),
        (
            "old_identity",
            "AR",
            17,
            "C",
            "identity",
            ["experiment"],
            "ISSUE59-T18A-01",
            "campaign identity",
        ),
        (
            "old_cli",
            "AR",
            17,
            "C",
            "identity",
            ["binaries", "final_cli", "sha256"],
            context["OLD_T18_CLI"],
            "historical T18 CLI",
        ),
        (
            "plan_flag",
            "AR",
            17,
            "C",
            "plan",
            [0, "env", "MLXL3_FLASH_DENSE_EPILOGUE"],
            "0",
            "plan F02 flag",
        ),
        ("plan_args", "AR", 17, "C", "plan", [0, "command", 6], "4", "bridge command"),
        ("arm_flag", "AR", 17, "C", "entry", ["arm"], "A", "arm flag"),
        ("plan_order", "AR", 17, "C", "order", [], None, "arm order"),
        ("unfrozen_golden", "AR", 17, "C", "unfrozen_golden", [], None, "unfrozen"),
        ("invalid_golden", "AR", 17, "C", "invalid_golden", [], None, "golden identity"),
    ]
    process_path = ["commands", 0]
    cases += [
        ("s32_incident_already_consumed", "AR", 17, "C", None, [], None, ""),
        ("s32_exact_total", "AR", 17, "C", None, [], None, ""),
        ("s32_exact_per_process", "AR", 17, "C", None, [], None, ""),
        ("s32_swapin_exact_total", "AR", 17, "C", None, [], None, ""),
        ("explicit_zero", "AR", 17, "C", None, [], None, ""),
    ]
    for label, path, value, reason in [
        (
            "total_plus_one",
            ["after", "vm_bytes", "Swapouts"],
            262144 + (32 << 20) + 1,
            "cumulative Swapouts",
        ),
        (
            "maximum_plus_one",
            ["maximum_live_swapout_bytes"],
            (32 << 20) + 1,
            "maximum cumulative Swapouts",
        ),
        ("process_plus_one", ["swapouts_bytes"], (32 << 20) + 1, "per-process Swapouts"),
        ("process_delta_mismatch", ["swapouts_bytes"], 1, "process/VM delta mismatch"),
        ("counter_reset", ["after", "vm_bytes", "Swapouts"], 262144, "VM counter reset"),
        ("before_origin", ["before", "vm_bytes", "Swapouts"], 262143, "VM counter reset"),
        ("sample_reset", ["maximum_live_swapout_bytes"], (512 << 10) + 1, "cumulative Swapouts"),
        ("negative_delta", ["swapouts_bytes"], -1, "per-process Swapouts"),
        ("boolean_delta", ["swapouts_bytes"], False, "per-process Swapouts"),
        ("boolean_maximum", ["maximum_live_swapout_bytes"], True, "maximum cumulative Swapouts"),
        ("boolean_before", ["before", "vm_bytes", "Swapouts"], True, "VM counter types"),
        ("float_after", ["after", "vm_bytes", "Swapouts"], 786432.0, "VM counter types"),
        (
            "swapin_total_plus_one",
            ["after", "vm_bytes", "Swapins"],
            (256 << 20) + 1,
            "cumulative Swapins",
        ),
        (
            "swapin_maximum_plus_one",
            ["maximum_live_swapin_bytes"],
            (256 << 20) + 1,
            "maximum cumulative Swapins",
        ),
        ("swapin_negative", ["swapins_bytes"], -1, "per-process Swapins"),
        ("missing_before", ["before"], {}, "vm_bytes"),
        ("missing_after", ["after"], {}, "vm_bytes"),
    ]:
        cases.append(
            ("s32_" + label, "AR", 17, "C", "supervisor", process_path + path, value, reason)
        )
    for value in (-1, 1, (32 << 20) + 1, True, 33554432.0):
        cases.append(
            (
                "s32_invalid_budget_" + repr(value),
                "AR",
                17,
                "C",
                "both_budgets",
                [],
                value,
                "swap budget",
            )
        )
    cases += [
        (
            "s32_disagreement",
            "AR",
            17,
            "C",
            "identity",
            ["maximum_new_swapout_bytes"],
            0,
            "swap budget agreement",
        ),
        (
            "s32_missing_identity_budget",
            "AR",
            17,
            "C",
            "missing_budget",
            [],
            None,
            "swap budget agreement",
        ),
        ("s32_missing_anchor_sha", "AR", 17, "C", "missing_sha", [], None, "swap_anchor_sha256"),
        (
            "s32_anchor_disagreement",
            "AR",
            17,
            "C",
            "identity",
            ["swap_anchor_path"],
            "/different",
            "swap anchor agreement",
        ),
        ("s32_unfrozen_anchor", "AR", 17, "C", "unfrozen_anchor", [], None, "unfrozen swap anchor"),
        ("s32_anchor_wrong_sha", "AR", 17, "C", "anchor_sha", [], None, "unfrozen swap anchor"),
        (
            "s32_reanchored_baseline",
            "AR",
            17,
            "C",
            "anchor_baseline",
            [],
            None,
            "original swap baseline",
        ),
        (
            "s32_unfrozen_amendment",
            "AR",
            17,
            "C",
            "unfrozen_amendment",
            [],
            None,
            "unfrozen swap amendment",
        ),
        (
            "s32_amendment_wrong_origin",
            "AR",
            17,
            "C",
            "amendment_origin",
            [],
            None,
            "original swap anchor",
        ),
        ("s32_amendment_reanchor", "AR", 17, "C", "amendment_reanchor", [], None, ""),
    ]
    results = []
    with (
        tempfile.TemporaryDirectory(prefix="f02-inspection-selftest-") as temporary,
        contextlib.chdir(temporary),
    ):
        for label, mode, budget, arm, target, path, replacement, reason in cases:
            output = Path(temporary) / label
            output.mkdir()
            entry, manifest, identity, plan, report, supervisor, source, cli = fixture(
                output, context, mode, budget, arm
            )
            if label.startswith("s32_"):
                add_swap32(output, context, manifest, identity, supervisor)
                process = supervisor["commands"][0]
                if label in ("s32_exact_total", "s32_total_plus_one"):
                    process["after"]["vm_bytes"]["Swapouts"] = 262144 + (32 << 20)
                    process["swapouts_bytes"] = (32 << 20) - (512 << 10)
                    process["maximum_live_swapout_bytes"] = 32 << 20
                if label == "s32_total_plus_one":
                    process["swapouts_bytes"] += 1
                elif label == "s32_exact_per_process":
                    process["before"]["vm_bytes"]["Swapouts"] = 262144
                    process["after"]["vm_bytes"]["Swapouts"] = 262144 + (32 << 20)
                    process["swapouts_bytes"] = process["maximum_live_swapout_bytes"] = 32 << 20
                elif label in ("s32_swapin_exact_total", "s32_swapin_total_plus_one"):
                    # No new Swapin in this process; the campaign has consumed its budget.
                    process["before"]["vm_bytes"]["Swapins"] = 256 << 20
                    process["after"]["vm_bytes"]["Swapins"] = 256 << 20
                    process["maximum_live_swapin_bytes"] = 256 << 20
                if label == "s32_swapin_total_plus_one":
                    process["swapins_bytes"] = 1
            elif label == "explicit_zero":
                manifest["maximum_new_swapout_bytes"] = identity["maximum_new_swapout_bytes"] = 0
            if target in ("entry", "manifest", "identity", "plan", "report", "supervisor"):
                set_value(
                    {
                        "entry": entry,
                        "manifest": manifest,
                        "identity": identity,
                        "plan": plan,
                        "report": report,
                        "supervisor": supervisor,
                    }[target],
                    path,
                    replacement,
                )
            elif target == "source_hash":
                identity["sources"][str(source)] = "0" * 64
            elif target == "binary_changed":
                cli.write_bytes(b"different synthetic bytes")
            elif target == "order":
                manifest["plans"].reverse()
            elif target == "unfrozen_golden":
                del identity["sources"][
                    manifest["external_golden_files_for_root_validation"]["MAIN"]
                ]
            elif target == "invalid_golden":
                golden_path = Path(manifest["external_golden_files_for_root_validation"]["MAIN"])
                golden = json.loads(golden_path.read_text())
                golden["token_hash"] = "0" * 16
                golden_path.write_text(json.dumps(golden))
                identity["sources"][str(golden_path)] = context["file_sha"](golden_path)
            elif target == "both_budgets":
                manifest["maximum_new_swapout_bytes"] = identity["maximum_new_swapout_bytes"] = (
                    replacement
                )
            elif target == "missing_budget":
                del identity["maximum_new_swapout_bytes"]
            elif target == "missing_sha":
                del identity["swap_anchor_sha256"]
            elif target in ("unfrozen_anchor", "unfrozen_amendment"):
                key = "swap_anchor_path" if target == "unfrozen_anchor" else "policy_amendment"
                del identity["sources"][manifest[key]]
            elif target == "anchor_sha":
                manifest["swap_anchor_sha256"] = identity["swap_anchor_sha256"] = "0" * 64
            elif target in ("anchor_baseline", "amendment_origin", "amendment_reanchor"):
                changed = Path(
                    manifest[
                        "swap_anchor_path" if target == "anchor_baseline" else "policy_amendment"
                    ]
                )
                data = json.loads(changed.read_text())
                if target == "anchor_baseline":
                    data["vm_bytes"]["Swapouts"] = 786432
                else:
                    data["original_anchor" if target == "amendment_origin" else "no_reanchor"] = (
                        "/different" if target == "amendment_origin" else False
                    )
                changed.write_text(json.dumps(data))
                identity["sources"][str(changed)] = context["file_sha"](changed)
                if target == "anchor_baseline":
                    manifest["swap_anchor_sha256"] = identity["swap_anchor_sha256"] = context[
                        "file_sha"
                    ](changed)
                    amendment_path = Path(manifest["policy_amendment"])
                    amendment = json.loads(amendment_path.read_text())
                    amendment["original_anchor_sha256"] = manifest["swap_anchor_sha256"]
                    amendment_path.write_text(json.dumps(amendment))
                    identity["sources"][str(amendment_path)] = context["file_sha"](amendment_path)
            if target == "entry":
                manifest["plans"][entry["position_within_mode"] - 1] = copy.deepcopy(entry)
            manifest_path, identity_path = output / "manifest.json", output / "identity.json"
            manifest_path.write_text(json.dumps(manifest))
            plan_path = Path(entry["plan"])
            plan_path.write_text(json.dumps(plan))
            for changed in (manifest_path, plan_path):
                identity["sources"][str(changed)] = context["file_sha"](changed)
            identity_path.write_text(json.dumps(identity))
            if target != "missing_raw":
                Path(entry["expected_report"]).write_text(
                    "{" if target == "malformed_raw" else json.dumps(report)
                )
            plan_path.with_name(plan_path.stem + "-result.json").write_text(json.dumps(supervisor))
            raised = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    inspector(entry, manifest_path, identity_path)
            except (AssertionError, ValueError, KeyError, OSError) as error:
                raised = error
            inspected_path = output / "external-inspection-v2.json"
            inspected = json.loads(inspected_path.read_text())
            if target:
                assert raised is not None, label
                assert reason in str(raised), (label, repr(raised))
                assert inspected["status"] == "failed_external_inspection", label
                assert inspected["external_golden"] is False and inspected.get("parity") is not True
            else:
                assert raised is None, (label, repr(raised))
                assert inspected["status"] == "passed_external_oracle_mode_and_memory_not_gain"
                assert inspected["external_golden"] is True and len(inspected["rows"]) == 4
                if label.startswith("s32_"):
                    process = supervisor["commands"][0]
                    assert inspected["maximum_new_swapout_bytes"] == 32 << 20
                    assert inspected["swap_anchor_vm_bytes"] == {"Swapouts": 262144, "Swapins": 0}
                    assert inspected["swapouts_bytes"] == process["swapouts_bytes"]
                    for counter, origin in (("Swapouts", 262144), ("Swapins", 0)):
                        before_vm = process["before"]["vm_bytes"][counter] - origin
                        after_vm = process["after"]["vm_bytes"][counter] - origin
                        assert (
                            inspected["cumulative_" + counter.lower() + "_before_bytes"]
                            == before_vm
                        )
                        assert (
                            inspected["cumulative_" + counter.lower() + "_after_bytes"] == after_vm
                        )
                        assert (
                            inspected["maximum_cumulative_" + counter.lower() + "_bytes"]
                            == after_vm
                        )
                before = inspected_path.read_bytes()
                try:
                    inspector(entry, manifest_path, identity_path)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("existing inspection overwritten")
                assert inspected_path.read_bytes() == before
            results.append(
                {
                    "case": label,
                    "mode": mode,
                    "budget": budget,
                    "arm": arm,
                    "expected_rejection": bool(target),
                    "status": inspected["status"],
                }
            )
        assert Path("opti.md").read_text().count("SYNTHETIC_SELFTEST_") == len(cases)
    print(
        json.dumps(
            {
                "status": "passed_synthetic_selftests_not_benchmark",
                "engine_executed": False,
                "synthetic_fixtures": True,
                "controller_sha256": context["file_sha"](FOLDER / "benchmark-inspection.py"),
                "positive_cases": sum(not row["expected_rejection"] for row in results),
                "negative_cases": sum(row["expected_rejection"] for row in results),
                "cases": results,
                "journaling": "temporary cwd only; fixtures removed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
