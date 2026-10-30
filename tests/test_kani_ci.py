"""Exercise the real Kani orchestration without installing a verifier or using a GPU."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import kani_ci as ci

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("prlimit") is None, reason="Linux proof runner only")


def inventory(names):
    return {
        "kani-version": "0.68.0",
        "file-version": "0.1",
        "standard-harnesses": {"real.rs": names},
        "contract-harnesses": {},
        "contracts": [],
        "totals": {
            "standard-harnesses": len(names),
            "contract-harnesses": 0,
            "functions-under-contract": 0,
        },
    }


def success(name, count=1):
    return (
        f"Checking harness {name}...\n ** 1 of 1 cover properties satisfied\n"
        "VERIFICATION:- SUCCESSFUL\n"
    ) * count + f"Complete - {count} successfully verified harnesses, 0 failures, {count} total.\n"


def test_real_baseline_inventory_has_complete_disjoint_shards():
    payload = json.loads(
        (ROOT / "docs/measurements/kani-ci-repair/baseline-library-inventory.json").read_text()
    )
    names = ci.inventory_names(payload)
    assert len(names) == 140
    selected = [n for shard in ci.SHARDS for n in ci.select_names(names, shard)]
    assert len(selected) == len(set(selected)) == len(names)
    assert set(selected) == set(names)
    assert ci.select_names(names, "all") == names
    assert ci.shard_for("new_module::new_proof") == "core"
    assert ci.shard_for("flash_cache_profile::proof") == "allocator"
    with pytest.raises(ValueError, match="zero"):
        ci.select_names(["only::core"], "flash")


@pytest.mark.parametrize("change", ["version", "duplicate", "empty", "total", "contract", "name"])
def test_inventory_rejects_missing_or_unsupported_obligations(change):
    payload = inventory(["core::proof"])
    if change == "version":
        payload["kani-version"] = "next"
    elif change == "duplicate":
        payload["standard-harnesses"]["real.rs"] *= 2
    elif change == "empty":
        payload["standard-harnesses"] = {}
    elif change == "total":
        payload["totals"]["standard-harnesses"] = 2
    elif change == "contract":
        payload["contract-harnesses"] = {"new": ["new::proof"]}
    else:
        payload["standard-harnesses"]["real.rs"] = [None]
    with pytest.raises(ValueError):
        ci.inventory_names(payload)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Complete - 0 successfully verified harnesses, 0 failures, 0 total.",
        success("other::proof"),
        success("core::proof") + success("core::proof"),
        success("core::proof").replace("1 of 1", "0 of 1"),
        success("core::proof").replace("SUCCESSFUL", "FAILED"),
        success("core::proof").replace("0 failures", "1 failures"),
    ],
)
def test_zero_partial_foreign_and_failed_verdicts_never_pass(output):
    assert not ci.check_verdict(output, "core::proof", 1)


def test_verdict_requires_both_api_instances_and_keeps_expected_unreachable_assertions():
    assert ci.check_verdict(success(ci.API_HARNESS, 2), ci.API_HARNESS, 2)
    assert not ci.check_verdict(success(ci.API_HARNESS), ci.API_HARNESS, 2)
    assert ci.check_verdict(success("p") + " ** 0 of 10 failed (2 unreachable)\n", "p", 1)


def test_address_space_limit_is_really_applied(tmp_path):
    log = tmp_path / "limit.log"
    result = ci.run_process(
        [sys.executable, "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_AS))"],
        log,
        5,
    )
    assert result["status"] == "completed"
    assert log.read_text().strip() == f"({ci.MEMORY_BYTES}, {ci.MEMORY_BYTES})"


@pytest.mark.parametrize("outcome", ["error", "signal", "timeout"])
def test_real_process_failures_keep_their_reason(tmp_path, outcome):
    source = {
        "error": "raise SystemExit(23)",
        "signal": "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        "timeout": "import time; time.sleep(30)",
    }[outcome]
    result = ci.run_process([sys.executable, "-c", source], tmp_path / "child.log", 0.3)
    assert (
        result["status"]
        == {"error": "process_error", "signal": "signal", "timeout": "timeout"}[outcome]
    )
    assert result["returncode"] is not None
    assert result["seconds"] < 4


def test_timeout_kills_descendant_even_if_parent_exits_and_child_ignores_term(tmp_path):
    child_pid = tmp_path / "pid"
    source = (
        "import os, signal, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f" open({str(child_pid)!r}, 'w').write(str(os.getpid()))\n"
        " time.sleep(30)\n"
        "else:\n"
        " time.sleep(30)\n"
    )
    result = ci.run_process([sys.executable, "-c", source], tmp_path / "tree.log", 0.5)
    assert result["status"] == "timeout"
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 1
    while status.exists() and status.read_text().split()[2] != "Z" and time.monotonic() < deadline:
        time.sleep(0.01)  # SIGKILL delivery to a descendant is asynchronous.
    assert not status.exists() or status.read_text().split()[2] == "Z"


@pytest.fixture
def fake_cargo(tmp_path, monkeypatch):
    names = ["core::a", "core::b", "flash_support::p", "speculative::p", "flash_cache_profile::p"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_INVENTORY", json.dumps(inventory(names)))
    cargo = tmp_path / "cargo"
    cargo.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "if 'list' in sys.argv:\n"
        " if '-Z' not in sys.argv[sys.argv.index('list') + 1:]: sys.exit(31)\n"
        " if not os.environ.get('FAKE_NO_INVENTORY'):\n"
        "  Path('kani-list.json').write_text(os.environ['FAKE_INVENTORY'])\n"
        " sys.exit(0)\n"
        "name = sys.argv[sys.argv.index('--harness') + 1]\n"
        "if os.environ.get('FAKE_SLEEP'):\n"
        " Path('started').write_text(str(os.getpid()))\n"
        " time.sleep(30)\n"
        "if name == os.environ.get('FAKE_EMPTY'): sys.exit(0)\n"
        "count = 2 if '--tests' in sys.argv else 1\n"
        "for _ in range(count):\n"
        " print(f'Checking harness {name}...')\n"
        " print('VERIFICATION:- SUCCESSFUL')\n"
        "print(f'Complete - {count} successfully verified harnesses, 0 failures, {count} total.')\n"
        "sys.exit(23 if name == os.environ.get('FAKE_FAIL') else 0)\n"
    )
    cargo.chmod(0o755)
    return tmp_path


@pytest.mark.parametrize("failure", [None, "FAKE_FAIL", "FAKE_EMPTY"])
def test_cli_runs_remaining_proofs_but_aggregate_rejects_any_failure(
    fake_cargo, monkeypatch, failure
):
    if failure:
        monkeypatch.setenv(failure, "core::a")
    result = ci.main(["--shard", "core", "--report-dir", "results"])
    report = json.loads(Path("results/report.json").read_text())
    assert (result == 0) == (failure is None)
    assert report["discovered"] == 5
    assert len(report["proofs"]) == 3
    assert report["proofs"][-1]["status"] == "success"  # Both API instances really ran.
    if failure:
        assert report["status"] == "failure"
        assert report["proofs"][0]["status"] != "success"


def test_stale_inventory_cannot_turn_missing_listing_into_success(fake_cargo, monkeypatch):
    Path("kani-list.json").write_text(os.environ["FAKE_INVENTORY"])
    monkeypatch.setenv("FAKE_NO_INVENTORY", "1")
    assert ci.main(["--shard", "core", "--report-dir", "results"]) == 1
    report = json.loads(Path("results/report.json").read_text())
    assert report["status"] == "failure"
    assert report["proofs"] == []


def test_whole_job_deadline_marks_pending_proofs_failed(tmp_path):
    records = [{"name": "p", "status": "pending"}]
    report = {"status": "incomplete", "proofs": records}
    assert not ci.run_proofs(records, tmp_path, report, tmp_path / "report.json", 1, 0)
    assert records[0]["status"] == "budget_exhausted"


def test_termination_saves_incomplete_verdict_and_reaps_child(fake_cargo, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "1")
    runner = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/kani_ci.py"),
            "--shard",
            "core",
            "--report-dir",
            "results",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not Path("started").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert Path("started").exists()
        pid = int(Path("started").read_text())
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=5) == 1
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        report = json.loads(Path("results/report.json").read_text())
        assert report["status"] == "failure"
        assert report["proofs"][0]["status"] == "interrupted"
        assert report["proofs"][1]["status"] == "pending"
    finally:
        if runner.poll() is None:
            runner.kill()
        runner.wait()


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "5401"])
def test_invalid_time_budgets_rejected_before_work(duration):
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        ci.positive_seconds(duration)


def test_only_the_documented_panicking_unreachable_stub_is_accepted():
    notice = "  - Stub: allocate_valid_levels -> reject_unreachable_dp\n"
    assert ci.check_verdict(
        success(ci.INVALID_BUDGET_HARNESS) + notice, ci.INVALID_BUDGET_HARNESS, 1
    )
    assert not ci.check_verdict(success(ci.INVALID_BUDGET_HARNESS), ci.INVALID_BUDGET_HARNESS, 1)
    assert not ci.check_verdict(success("other::proof") + notice, "other::proof", 1)
