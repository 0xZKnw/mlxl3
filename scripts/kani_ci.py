"""Run every compiled CPU Kani harness once, with finite budgets and durable verdicts."""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

SHARDS = ("core", "flash", "speculative", "allocator")
VERSION = "0.68.0"
API_HARNESS = "openai::proofs::validated_usage_respects_budget_without_total_overflow"
INVALID_BUDGET_HARNESS = (
    "flash_cache_profile::verification::cache_allocation_rejects_every_invalid_two_layer_budget"
)
MEMORY_BYTES = 6 * 1024**3
OBLIGATIONS = Path(__file__).with_name("kani_obligations.json")


def required_names():
    names = json.loads(OBLIGATIONS.read_text())["required_harnesses"]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate obligation manifest")
    return names


def check_obligations(names):
    missing = set(required_names()) - set(names)
    if missing:
        raise ValueError("missing proof obligations: " + ", ".join(sorted(missing)))


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 5400:
        raise argparse.ArgumentTypeError("duration must be finite and in (0, 5400]")
    return value


def inventory_names(payload):
    if payload.get("kani-version") != VERSION or payload.get("file-version") != "0.1":
        raise ValueError("unexpected Kani inventory version")
    if payload.get("contract-harnesses") != {} or payload.get("contracts") != []:
        raise ValueError("contract harnesses require an explicit orchestration update")
    groups = payload.get("standard-harnesses")
    if not isinstance(groups, dict) or not all(isinstance(v, list) for v in groups.values()):
        raise ValueError("invalid harness inventory")
    names = [name for group in groups.values() for name in group]
    if not names or not all(isinstance(n, str) and re.fullmatch(r"[\w:]+", n) for n in names):
        raise ValueError("empty or invalid harness inventory")
    if len(names) != len(set(names)):
        raise ValueError("duplicate library harness")
    if payload.get("totals") != {
        "standard-harnesses": len(names),
        "contract-harnesses": 0,
        "functions-under-contract": 0,
    }:
        raise ValueError("incomplete harness inventory")
    return sorted(names)


def shard_for(name: str) -> str:
    """post: __return__ in SHARDS"""
    module = name.split("::", 1)[0]
    if module == "flash_cache_profile":
        return "allocator"
    if module in ("flash_support", "flash_draft_budget"):
        return "flash"
    if module == "speculative":
        return "speculative"
    return "core"


def select_names(names, shard):
    selected = [n for n in names if shard == "all" or shard_for(n) == shard]
    if not selected:
        raise ValueError(f"shard {shard} selected zero harnesses")
    return selected


def check_verdict(output, name, expected):
    checked = re.findall(r"Checking harness ([\w:]+)\.\.\.", output)
    summaries = re.findall(
        r"Complete - (\d+) successfully verified harnesses, (\d+) failures, (\d+) total\.",
        output,
    )
    covers = re.findall(r"\*\* (\d+) of (\d+) cover properties satisfied", output)
    stubs = re.findall(r"^\s+- Stub: (.+)$", output, re.MULTILINE)
    allowed_stubs = (
        ["allocate_valid_levels -> reject_unreachable_dp"] if name == INVALID_BUDGET_HARNESS else []
    )
    return (
        checked == [name] * expected
        and summaries == [(str(expected), "0", str(expected))]
        and output.count("VERIFICATION:- SUCCESSFUL") == expected
        and "VERIFICATION:- FAILED" not in output
        and all(found == total for found, total in covers)
        and stubs == allowed_stubs
    )


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def stop_group(child):
    # Also kill descendants if the direct child already exited.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_process(command, log, timeout):
    started = time.monotonic()
    result = {"command": command, "status": "process_error", "returncode": None}
    with log.open("w") as output:
        child = subprocess.Popen(
            ["prlimit", f"--as={MEMORY_BYTES}", "--", *command],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            child.wait(timeout=timeout)
            result["status"] = "completed" if child.returncode == 0 else "process_error"
            if child.returncode < 0:
                result["status"] = "signal"
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
        except (KeyboardInterrupt, InterruptedError):
            result["status"] = "interrupted"
        finally:
            stop_group(child)
        result["returncode"] = child.returncode
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def run_proofs(records, directory, report, report_path, timeout, deadline):
    for index, record in enumerate(records):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for pending in records[index:]:
                pending["status"] = "budget_exhausted"
            break
        log = directory / f"{index:03d}-{record['name'].replace('::', '-')}.log"
        record["log"] = log.name
        record["status"] = "running"
        save_report(report_path, report)
        result = run_process(record["command"], log, min(timeout, remaining))
        record.update(result)
        output = log.read_text(errors="replace")
        if record["status"] == "completed":
            record["status"] = (
                "success"
                if check_verdict(output, record["name"], record["expected_instances"])
                else "invalid_output"
            )
        # Backend failures are not automatically counterexamples (e.g. allocation failure).
        print(f"{record['status']}: {record['name']} ({record['seconds']}s)", flush=True)
        if record["status"] != "success":
            # Keep the full log artifact, and expose a bounded diagnostic in the
            # job log even if artifact download is unavailable.
            print("\n".join(output.splitlines()[-35:]), flush=True)
        save_report(report_path, report)
        if record["status"] == "interrupted":
            break
    report["status"] = (
        "success" if records and all(r["status"] == "success" for r in records) else "failure"
    )
    save_report(report_path, report)
    return report["status"] == "success"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=(*SHARDS, "all"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=positive_seconds, default=600)
    parser.add_argument("--budget", type=positive_seconds, default=2700)
    args = parser.parse_args(argv)
    directory = args.report_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.json"
    if path.exists():
        parser.error("report already exists; use a fresh report directory")
    report = {
        "status": "incomplete",
        "shard": args.shard,
        "address_space_limit_bytes": MEMORY_BYTES,
        "per_harness_seconds": args.timeout,
        "total_budget_seconds": args.budget,
        "proofs": [],
    }
    save_report(path, report)
    deadline = time.monotonic() + args.budget
    try:
        # Remove stale output: a successful listing must create its own complete inventory.
        inventory = Path("kani-list.json")
        inventory.unlink(missing_ok=True)
        command = ["cargo", "kani", "--lib", "--no-default-features"]
        listing = run_process(
            command + ["list", "--format", "json", "-Z", "stubbing"],
            directory / "inventory.log",
            min(600, args.budget),
        )
        report["listing"] = listing
        if listing["status"] != "completed":
            raise ValueError(f"inventory failed: {listing['status']}")
        payload = json.loads(inventory.read_text())
        names = inventory_names(payload)
        check_obligations(names)
        inventory.replace(directory / "inventory.json")
        report["discovered"] = len(names)
        report["assignment"] = {shard: select_names(names, shard) for shard in SHARDS}
        selected = select_names(names, args.shard)
        for name in selected:
            report["proofs"].append(
                {
                    "name": name,
                    "status": "pending",
                    "expected_instances": 1,
                    "command": command
                    + [
                        "-Z",
                        "stubbing",
                        "--exact",
                        "--harness",
                        name,
                        "--jobs",
                        "1",
                        "--output-format",
                        "regular",
                        "--verbose",
                    ],
                }
            )
        if args.shard in ("core", "all"):
            report["proofs"].append(
                {
                    "name": API_HARNESS,
                    "status": "pending",
                    "expected_instances": 2,
                    "command": [
                        "cargo",
                        "kani",
                        "--no-default-features",
                        "--tests",
                        "-Z",
                        "stubbing",
                    ]
                    + [
                        "--exact",
                        "--harness",
                        API_HARNESS,
                        "--jobs",
                        "1",
                        "--output-format",
                        "regular",
                        "--verbose",
                    ],
                }
            )
        save_report(path, report)
        return (
            0
            if run_proofs(report["proofs"], directory, report, path, args.timeout, deadline)
            else 1
        )
    except (OSError, ValueError, KeyboardInterrupt, InterruptedError) as error:
        report["status"] = "failure"
        report["error"] = str(error)
        save_report(path, report)
        print(f"Kani orchestration failed: {error}", flush=True)
        return 1


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
