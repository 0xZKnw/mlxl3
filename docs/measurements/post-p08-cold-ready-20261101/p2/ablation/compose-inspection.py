"""Four-arm campaign adapter; reuse F02's golden, mode and memory inspection."""

import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "docs/measurements/post-p08-compose-20261101"
F02 = BASE.parent / "post-p08-dense-epilogue-20261031"
upstream = runpy.run_path(str(F02 / "benchmark-inspection.py"))
EXPERIMENT = "POSTP08-COMPOSE-01"
FLAGS = {"A": ("0", "0"), "G": ("1", "0"), "F": ("0", "1"), "C": ("1", "1")}
ORDER = {"bridge17": list("AGFC"), "pilot128": list("AGFCCFGA")}
ENV = {**upstream["ENV"], "MLXL3_FLASH_PREFILL_HEADS": "1"}
file_sha = upstream["file_sha"]
bridge = upstream["bridge"]
pool = upstream["pool"]


def preflight(entry, manifest_path=None, identity_path=None):
    """Check the frozen plan before launch, and report flags during inspection."""
    manifest_path = Path(manifest_path or BASE / "manifest.json")
    identity_path = Path(identity_path or BASE / "identity.json")
    manifest = json.loads(manifest_path.read_text())
    identity = json.loads(identity_path.read_text())
    assert manifest["experiment"] == identity["experiment"] == EXPERIMENT, "campaign identity"
    assert Path(manifest["model"]).is_absolute() and Path(manifest["dflash_draft"]).is_absolute()
    assert entry in manifest["plans"], "unregistered entry"
    assert len({row["name"] for row in manifest["plans"]}) == len(manifest["plans"]), "names"
    assert entry["phase"] in ORDER and entry["mode"] in ("AR", "MTP", "DF"), "phase/mode"
    assert entry["tokens_per_generation"] == {"bridge17": 17, "pilot128": 128}[entry["phase"]]
    ordered = [
        row
        for row in manifest["plans"]
        if row["phase"] == entry["phase"] and row["mode"] == entry["mode"]
    ]
    assert [row["arm"] for row in ordered] == ORDER[entry["phase"]] and [
        row["position_within_mode"] for row in ordered
    ] == list(range(1, len(ordered) + 1)), "arm order"
    for row in ordered:
        assert (row["gdn_flag"], row["f02_flag"]) == FLAGS[row["arm"]], "arm flags"
    assert Path(entry["output"]).is_absolute() and Path(entry["plan"]).is_absolute()
    assert Path(entry["expected_report"]) == Path(entry["output"]) / "report.json"
    assert identity["sources"], "empty source identity"
    frozen = {str((ROOT / name).resolve()): digest for name, digest in identity["sources"].items()}
    policy = upstream["swap_policy"](manifest, identity, frozen)
    assert policy["maximum_new_swapout_bytes"] == 64 << 20, "composition swap budget"
    golden_files = manifest["external_golden_files_for_root_validation"]
    assert set(golden_files) == {"MAIN", "HOLD"}
    for required in (
        manifest_path,
        Path(entry["plan"]),
        Path(__file__),
        F02 / "benchmark-inspection.py",
        F02 / "bounded-gpu-runner.py",
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
    binary = binaries["final_cli"]
    for value in binaries.values():
        assert Path(value["path"]).is_absolute() and file_sha(value["path"]) == value["sha256"], (
            "binary identity"
        )
    plan = json.loads(Path(entry["plan"]).read_text())
    assert len(plan) == 1 and plan[0]["name"] == entry["name"]
    item = plan[0]
    assert (
        item["require_quiet"] is True
        and type(item["expected_exit"]) is int
        and item["expected_exit"] == 0
    )
    assert (
        type(item["minimum_available_bytes"]) is int
        and item["minimum_available_bytes"] == 10_500_000_000
    )
    assert (
        type(item["timeout_seconds"]) in (int, float)
        and math.isfinite(item["timeout_seconds"])
        and 0 < item["timeout_seconds"] <= 900
    )
    assert all(item["env"].get(key) == value for key, value in ENV.items()), "plan controls"
    for flag, key in (
        ("gdn_flag", "MLXL3_FLASH_PREPARED_CONV"),
        ("f02_flag", "MLXL3_FLASH_DENSE_EPILOGUE"),
    ):
        assert item["env"].get(key) == entry[flag], "plan " + key
    mode_args = {
        "AR": [],
        "MTP": ["--mtp"],
        "DF": ["--dflash-draft", manifest["dflash_draft"], "--dflash-mode", "1"],
    }[entry["mode"]]
    assert item["command"] == [
        binaries["python_runner"]["path"],
        str(ROOT / "scripts/benchmark-flash-bridge.py"),
        binary["path"],
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
    ], "bridge command"
    report_path = Path(entry["expected_report"])
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report["environment"]["MLXL3_FLASH_PREPARED_CONV"] == entry["gdn_flag"], (
            "report GDN flag"
        )
        assert report["environment"]["MLXL3_FLASH_PREFILL_HEADS"] == "1", "report prefill flag"
    return {
        "manifest": manifest,
        "identity": identity,
        "binary": binary,
        "plan": plan,
        "swap_policy": policy,
    }


# Keep failures inside the upstream inspector's durable failed-result boundary.
inspect = upstream["inspect"]
inspect.__globals__["preflight"] = preflight
