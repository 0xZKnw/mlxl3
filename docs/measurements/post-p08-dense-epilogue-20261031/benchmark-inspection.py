"""F02 external bridge inspector, derived from T18a f223e1cf7f940504.

No T18a measurements or identities are accepted as this campaign. The bridge
runner and independent golden validator remain unchanged. Root owns launching.
"""

import hashlib
import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
EXPERIMENT = "POSTP08-F02-01"
OLD_T18_CLI = "1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436"
ENV = {
    "MLXL3_ALLOCATOR_CACHE_MIB": "64",
    "MLXL3_SSD_GPU_HOT": "1",
    "MLXL3_SSD_GPU_HOT_WIDE": "1",
    "MLXL3_FLASH_MULTI_TOKEN": "1",
    "MLXL3_SSD_UNIQUE_COHORT": "1",
    "MLXL3_SSD_CACHE_GIB": "3",
    "MLXL3_SSD_READERS": "4",
    "MLXL3_SSD_OVERLAP": "1",
    "MLXL3_SSD_LOOKAHEAD_DEPTH": "0",
    "MLXL3_SSD_CACHE_POLICY": "auto",
    "MLXL3_SSD_SHARED_EXL3": "0",
    "MLXL3_SSD_DYNAMIC_READERS": "0",
    "MLXL3_FLASH_SWIGLU_DOWN": "0",
    "MLXL3_FLASH_GDN_EPILOGUE": "0",
    "MLXL3_SSD_WEIGHTED_EPILOGUE": "0",
    "MLXL3_FLASH_DRAFT_WIDTH": "1",
    "MLXL3_FLASH_DRAFT_ADAPTIVE": "0",
    "MLXL3_FLASH_DRAFT_Q4": "1",
    "MLXL3_FLASH_DRAFT_RING": "1",
    "MLX_MAX_OPS_PER_BUFFER": "40",
    "MLX_MAX_MB_PER_BUFFER": "40",
}
bridge = runpy.run_path(str(ROOT / "scripts/benchmark-flash-bridge.py"))
validate = bridge["validate"]
pool = runpy.run_path(str(ROOT / "docs/measurements/issue58-20261029/pool-speed-driver.py"))
golden_check = pool["check_golden"]


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def swap_policy(manifest, identity, frozen):
    limit = manifest.get("maximum_new_swapout_bytes", 0)
    recorded = identity.get("maximum_new_swapout_bytes", 0)
    assert type(limit) is int and type(recorded) is int and limit == recorded, (
        "swap budget agreement"
    )
    assert limit in (0, 32 << 20), "swap budget must be zero or explicitly authorized 32MiB"
    result = {"maximum_new_swapout_bytes": limit, "swap_anchor_vm_bytes": None}
    if limit == 0:
        return result
    path = manifest["swap_anchor_path"]
    digest = manifest["swap_anchor_sha256"]
    assert path == identity["swap_anchor_path"] and digest == identity["swap_anchor_sha256"], (
        "swap anchor agreement"
    )
    assert type(path) is str and Path(path).is_absolute() and type(digest) is str, (
        "swap anchor types"
    )
    anchor_path = Path(path).resolve()
    amendment_path = Path(manifest["policy_amendment"])
    assert amendment_path.is_absolute(), "swap amendment path"
    assert frozen.get(str(anchor_path)) == digest and file_sha(anchor_path) == digest, (
        "unfrozen swap anchor/SHA"
    )
    assert str(amendment_path.resolve()) in frozen, "unfrozen swap amendment"
    amendment = json.loads(amendment_path.read_text())
    assert amendment["original_anchor"] == path and amendment["original_anchor_sha256"] == digest, (
        "original swap anchor"
    )
    assert amendment["no_reanchor"] is True
    assert (
        type(amendment["maximum_new_swapout_bytes"]) is int
        and amendment["maximum_new_swapout_bytes"] == limit
    )
    assert (
        type(amendment["baseline_swapout_bytes"]) is int
        and amendment["baseline_swapout_bytes"] == 262144
    )
    vm = json.loads(anchor_path.read_text())["vm_bytes"]
    baseline = {key: vm[key] for key in ("Swapouts", "Swapins")}
    assert all(type(value) is int and value >= 0 for value in baseline.values()), (
        "swap anchor counters"
    )
    assert baseline == {"Swapouts": 262144, "Swapins": 0}, "original swap baseline"
    result.update(swap_anchor_path=path, swap_anchor_sha256=digest, swap_anchor_vm_bytes=baseline)
    return result


def check_swap(process, policy):
    result = {}
    for counter, limit in (
        ("Swapouts", policy["maximum_new_swapout_bytes"]),
        ("Swapins", 256 << 20),
    ):
        field = counter.lower() + "_bytes"
        maximum_field = "maximum_live_" + counter.lower()[:-1] + "_bytes"
        delta, maximum = process[field], process[maximum_field]
        assert type(delta) is int and 0 <= delta <= limit, "per-process " + counter
        assert type(maximum) is int and 0 <= maximum <= limit, "maximum cumulative " + counter
        baseline = policy["swap_anchor_vm_bytes"]
        if baseline is not None:
            before = process["before"]["vm_bytes"][counter]
            after = process["after"]["vm_bytes"][counter]
            assert type(before) is int and type(after) is int, "VM counter types"
            assert baseline[counter] <= before <= after, "VM counter reset"
            assert after - before == delta, "process/VM delta mismatch"
            cumulative = after - baseline[counter]
            assert cumulative <= limit and maximum <= cumulative, "cumulative " + counter
            result["cumulative_" + counter.lower() + "_before_bytes"] = before - baseline[counter]
            result["cumulative_" + counter.lower() + "_after_bytes"] = cumulative
            result["maximum_cumulative_" + counter.lower() + "_bytes"] = max(maximum, cumulative)
    return result


def preflight(entry, manifest_path=None, identity_path=None):
    """Root calls this before PID creation; inspect repeats it after completion."""
    manifest_path = Path(manifest_path or BASE / "prepared/manifest.json")
    identity_path = Path(identity_path or BASE / "final-identity.json")
    manifest = json.loads(manifest_path.read_text())
    identity = json.loads(identity_path.read_text())
    assert manifest["experiment"] == identity["experiment"] == EXPERIMENT, "campaign identity"
    assert Path(manifest["model"]).is_absolute() and Path(manifest["dflash_draft"]).is_absolute()
    assert entry in manifest["plans"], "unregistered entry"
    assert entry["phase"] in ("bridge17", "pilot128")
    assert entry["tokens_per_generation"] == {"bridge17": 17, "pilot128": 128}[entry["phase"]]
    assert entry["mode"] in ("AR", "MTP", "DF")
    assert entry["arm"] in ("A", "C") and entry["f02_flag"] == {"A": "0", "C": "1"}[entry["arm"]], (
        "arm flag"
    )
    ordered = [
        row
        for row in manifest["plans"]
        if row["phase"] == entry["phase"] and row["mode"] == entry["mode"]
    ]
    expected_order = ["A", "C"] if entry["phase"] == "bridge17" else ["A", "C", "C", "A"]
    assert [row["arm"] for row in ordered] == expected_order and [
        row["position_within_mode"] for row in ordered
    ] == list(range(1, len(expected_order) + 1)), "arm order"
    assert Path(entry["output"]).is_absolute() and Path(entry["plan"]).is_absolute()
    assert Path(entry["expected_report"]) == Path(entry["output"]) / "report.json"
    assert identity["sources"], "empty source identity"
    frozen = {str((ROOT / name).resolve()): digest for name, digest in identity["sources"].items()}
    policy = swap_policy(manifest, identity, frozen)
    golden_files = manifest["external_golden_files_for_root_validation"]
    assert set(golden_files) == {"MAIN", "HOLD"}
    for required in (
        manifest_path,
        Path(entry["plan"]),
        Path(__file__),
        ROOT / "scripts/benchmark-flash-bridge.py",
        ROOT / "docs/measurements/issue58-20261029/pool-speed-driver.py",
        *(Path(value) for value in golden_files.values()),
    ):
        assert str(required.resolve()) in frozen, "unfrozen controller/manifest/plan"
    for name, digest in frozen.items():
        assert file_sha(name) == digest, "source identity: " + name
    for path in golden_files.values():
        golden = json.loads(Path(path).read_text())
        assert (
            len(golden["tokens"]) == 128
            and pool["token_hash"](golden["tokens"]) == golden["token_hash"]
        ), "golden identity"
        assert isinstance(golden["cache_context"], str) and golden["cache_context"], "empty golden"
    binaries = identity["binaries"]
    expected_binary = binaries["final_cli"]
    assert expected_binary["sha256"] != OLD_T18_CLI, "historical T18 CLI is not F02"
    for value in binaries.values():
        assert Path(value["path"]).is_absolute() and file_sha(value["path"]) == value["sha256"], (
            "binary identity"
        )
    plan = json.loads(Path(entry["plan"]).read_text())
    assert len(plan) == 1 and plan[0]["name"] == entry["name"]
    item = plan[0]
    assert item["require_quiet"] is True and item["expected_exit"] == 0
    assert item["minimum_available_bytes"] >= 10_500_000_000 and 0 < item["timeout_seconds"] <= 900
    assert all(item["env"].get(key) == value for key, value in ENV.items()), "plan controls"
    assert item["env"].get("MLXL3_FLASH_DENSE_EPILOGUE") == entry["f02_flag"], "plan F02 flag"
    mode_args = {
        "AR": [],
        "MTP": ["--mtp"],
        "DF": ["--dflash-draft", manifest["dflash_draft"], "--dflash-mode", "1"],
    }[entry["mode"]]
    expected_command = [
        binaries["python_runner"]["path"],
        str(ROOT / "scripts/benchmark-flash-bridge.py"),
        expected_binary["path"],
        manifest["model"],
        entry["output"],
        "--cache-gib",
        "3",
        "--tokens",
        str(entry["tokens_per_generation"]),
        "--repeats",
        "1",
        "--timeout",
        "180",
        *mode_args,
    ]
    assert item["command"] == expected_command, "bridge command"
    return {
        "manifest": manifest,
        "identity": identity,
        "binary": expected_binary,
        "plan": plan,
        "swap_policy": policy,
    }


def inspect(entry, manifest_path=None, identity_path=None):
    report_path = Path(entry["expected_report"])
    output = Path(entry["output"]) / "external-inspection-v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), "preserve previous inspection"
    result = {
        "status": "incomplete_external_inspection",
        "name": entry["name"],
        "phase": entry["phase"],
        "mode": entry["mode"],
        "arm": entry["arm"],
        "report": str(report_path),
        "rows": [],
        "external_golden": False,
        "numerical_parity": "not_evaluated_by_bridge_inspector",
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    try:
        configuration = preflight(entry, manifest_path, identity_path)
        manifest = configuration["manifest"]
        expected_binary = configuration["binary"]
        plan = configuration["plan"]
        result.update(configuration["swap_policy"])
        raw = report_path.read_bytes()
        report = json.loads(raw)
        result["report_sha256"] = hashlib.sha256(raw).hexdigest()
        assert (
            report["status"] == "validated"
            and report["reaped"] is True
            and report["reader_joined"] is True
        )
        assert not report.get("error")
        assert (
            report["engine"] == expected_binary["path"]
            and report["binary_sha256"] == expected_binary["sha256"]
        )
        assert (
            hashlib.sha256(Path(expected_binary["path"]).read_bytes()).hexdigest()
            == expected_binary["sha256"]
        )
        assert (
            report["model"] == manifest["model"]
            and report["tokens"] == entry["tokens_per_generation"]
        )
        assert (
            report["expert_cache_gib"] == 3
            and type(report["repeats"]) is int
            and report["repeats"] == 1
        )
        assert report["prompts"] == bridge["PROMPTS"]
        mode = {"AR": "baseline", "MTP": "mtp", "DF": "dflash2"}[entry["mode"]]
        assert report["mode"] == mode
        assert report["environment"]["MLXL3_FLASH_DENSE_EPILOGUE"] == entry["f02_flag"], (
            "report F02 flag"
        )
        assert report["runtime"]["runtime_executable"] == expected_binary["path"]
        for name, value in ENV.items():
            if name.startswith("MLXL3_"):
                assert report["environment"][name] == value, name
        for group, repeat in [("warmup", 0), ("runs", 1)]:
            rows = report[group]
            assert len(rows) == 2 and [r["prompt"] for r in rows] == ["MAIN", "HOLD"]
            for row in rows:
                assert row["repeat"] == repeat and row["mode"] == mode
                stats = row["stats"]
                gm = row["generation_mode"]
                duration = stats["prefill_seconds"]
                assert (
                    type(duration) in (int, float) and math.isfinite(duration) and duration > 0
                ), "prefill duration"
                assert math.isclose(
                    stats["prefill_tps"], stats["prompt_tokens"] / duration, rel_tol=1e-5
                ), "prefill timing"
                event = {
                    "stats": stats,
                    "token_hash": row["token_hash"],
                    "cache_context": row["text"],
                }
                validate(event, gm, entry["mode"] == "MTP", report["tokens"], entry["mode"] == "DF")
                golden = json.loads(
                    Path(
                        manifest["external_golden_files_for_root_validation"][row["prompt"]]
                    ).read_text()
                )
                golden_check(event, golden, report["tokens"])
                assert stats["prompt_tokens"] == {"MAIN": 87, "HOLD": 82}[row["prompt"]]
                assert gm["mtp_requested"] is (entry["mode"] == "MTP") and gm[
                    "dflash_requested"
                ] is (entry["mode"] == "DF")
                assert gm["mtp_depth"] == (1 if entry["mode"] == "MTP" else 0)
                assert gm["dflash_mode"] == (1 if entry["mode"] == "DF" else 0)
                assert gm["dflash_proposals"] == (1 if entry["mode"] == "DF" else 0)
                if entry["mode"] == "DF":
                    assert (
                        gm["dflash_draft_path"] == manifest["dflash_draft"]
                        and report["dflash_draft"] == manifest["dflash_draft"]
                    )
                counters = {}
                for prefix, active in [
                    ("mtp", entry["mode"] == "MTP"),
                    ("dflash", entry["mode"] == "DF"),
                ]:
                    values = [
                        stats.get(prefix + "_" + key, 0)
                        for key in ["proposed_tokens", "accepted_tokens", "blocks"]
                    ]
                    proposed, accepted, blocks = values
                    if active:
                        assert all(type(v) is int and v >= 0 for v in values)
                        assert (
                            proposed == blocks > 0
                            and accepted <= proposed
                            and proposed + accepted <= report["tokens"] - 1
                        )
                    else:
                        assert all(v is None or (type(v) is int and v == 0) for v in values)
                    counters[prefix] = {
                        "proposed": proposed,
                        "accepted": accepted,
                        "blocks": blocks,
                    }
                memory = stats["memory"]
                result["rows"].append(
                    {
                        "group": group,
                        "repeat": repeat,
                        "prompt": row["prompt"],
                        "token_hash": row["token_hash"],
                        "generated_tokens": stats["generated_tokens"],
                        "decode_tokens": stats["decode_tokens"],
                        "prompt_tokens": stats["prompt_tokens"],
                        "decode_seconds": stats["decode_seconds"],
                        "decode_tps": stats["decode_tps"],
                        "prefill_tps": stats["prefill_tps"],
                        "prefill_seconds": duration,
                        "ttft_seconds": stats["ttft_seconds"],
                        "counters": counters,
                        "footprint_bytes": memory["process_footprint_bytes"],
                        "lifetime_peak_bytes": memory["process_lifetime_peak_bytes"],
                    }
                )
        supervisor_path = Path(entry["plan"]).with_name(Path(entry["plan"]).stem + "-result.json")
        supervisor = json.loads(supervisor_path.read_text())
        assert (
            supervisor["status"] == "completed_expected_outcomes_not_speed"
            and len(supervisor["commands"]) == 1
        )
        process = supervisor["commands"][0]
        assert (
            process["status"] == "completed_expected_outcome"
            and process["exit_code"] == 0
            and process["reaped"] is True
        )
        assert process["command"] == plan[0]["command"] and process["env"] == plan[0]["env"], (
            "supervisor plan"
        )
        result.update(check_swap(process, configuration["swap_policy"]))
        assert (
            type(process["children_maximum_rss_bytes"]) is int
            and 0 < process["children_maximum_rss_bytes"] <= 10 << 30
        )
        result.update(
            external_golden=True,
            status="passed_external_oracle_mode_and_memory_not_gain",
            supervisor=str(supervisor_path),
            swapouts_bytes=process["swapouts_bytes"],
            reaped=True,
            reader_joined=True,
        )
    except BaseException as error:
        result.update(status="failed_external_inspection", error=repr(error))
        raise
    finally:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with Path("opti.md").open("a") as stream:
            stream.write(
                f"\n**F02 inspection {entry['name']} : {result['status']}.** Preuve {output} ; aucune validation de gain issue de cette seule réponse.\n"
            )
    print(
        json.dumps(
            {
                "name": entry["name"],
                "status": result["status"],
                "measured": [
                    {
                        k: r[k]
                        for k in ["prompt", "decode_tps", "prefill_tps", "ttft_seconds", "counters"]
                    }
                    for r in result["rows"]
                    if r["group"] == "runs"
                ],
            }
        ),
        flush=True,
    )
    return result
