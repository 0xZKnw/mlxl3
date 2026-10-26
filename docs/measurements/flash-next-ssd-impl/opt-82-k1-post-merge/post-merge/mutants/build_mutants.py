"""Build six frozen shader mutants sequentially; never execute a test binary."""

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import shutil
import signal
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(raw):
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    finished = [r for r in records if r.get("reason") == "build-finished"]
    files = {
        r["executable"]
        for r in records
        if r.get("reason") == "compiler-artifact"
        and r.get("target", {}).get("name") == "mlxl3_native"
        and "lib" in r.get("target", {}).get("kind", [])
        and r.get("profile", {}).get("test") is True
        and r.get("executable")
    }
    if len(finished) != 1 or finished[0].get("success") is not True or len(files) != 1:
        raise ValueError("missing, ambiguous or failed Cargo lib-test artifact")
    return Path(files.pop())


def restore(target, original, mutant_sha):
    current = target.read_bytes()
    if current != original:
        if hashlib.sha256(current).hexdigest() != mutant_sha:
            raise RuntimeError(f"refusing to overwrite an unexpected edit: {target}")
        target.write_bytes(original)
    if target.read_bytes() != original:
        raise RuntimeError(f"byte-exact restore failed: {target}")


def stop(child):
    if child is None:
        return
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=3)


def tree_rss(root):
    raw = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True, timeout=5)
    rows = [tuple(map(int, line.split())) for line in raw.splitlines() if line.strip()]
    owned = {root}
    while True:
        more = {pid for pid, parent, _ in rows if parent in owned}
        if more <= owned:
            return sum(rss * 1024 for pid, _, rss in rows if pid in owned)
        owned |= more


class BoundedChild:
    def __init__(self, child, record):
        self.child, self.record = child, record

    def wait(self, timeout):
        try:
            return self.child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rss = tree_rss(self.child.pid)
            self.record["maximum_sampled_tree_rss_bytes"] = max(
                self.record.get("maximum_sampled_tree_rss_bytes", 0), rss
            )
            if rss > 4 << 30:
                raise RuntimeError("build process tree exceeds 4GiB RSS")
            raise


def main(plan_path):
    plan = json.loads(plan_path.read_text())
    if len(plan["variants"]) != 6 or len({v["name"] for v in plan["variants"]}) != 6:
        raise ValueError("expected exactly six distinct frozen variants")
    work = Path(plan["workspace"])
    destination = plan_path.parent
    result_path = destination / "build-result.json"
    if result_path.exists():
        raise FileExistsError("refusing to overwrite build results")
    if digest(plan["source_manifest"]) != plan["source_manifest_sha256"]:
        raise ValueError("normal source manifest changed")
    sources = json.loads(Path(plan["source_manifest"]).read_text())

    def verify_sources():
        for name, expected in sources.items():
            if digest(work / name) != expected:
                raise ValueError(f"source differs from normal freeze: {name}")

    verify_sources()
    if digest(plan["normal_binary"]) != plan["normal_binary_sha256"]:
        raise ValueError("normal test binary changed")
    monitor = runpy.run_path(plan["monitor"])
    parse_vm = runpy.run_path(plan["parse_vm_source"])["parse_vm"]
    result = {"status": "incomplete", "plan_sha256": digest(plan_path), "builds": []}

    def save():
        result_path.write_text(json.dumps(result, indent=2) + "\n")

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("build orchestration interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        for variant in plan["variants"]:
            verify_sources()
            target = work / variant["file"]
            original = target.read_bytes()
            mutant = Path(variant["mutant_source"])
            binary = Path(variant["native_test_binary"])
            if digest(mutant) != variant["mutant_source_sha256"] or binary.exists():
                raise ValueError("mutant source changed or output binary already exists")
            record = {
                "name": variant["name"],
                "command": plan["cargo_command"],
                "timeout_seconds": plan["build_timeout_seconds"],
                "status": "building",
                "before": monitor["snapshot"](parse_vm),
            }
            result["builds"].append(record)
            save()
            child = None
            try:
                target.write_bytes(mutant.read_bytes())
                with (
                    (destination / (variant["name"] + ".jsonl")).open("x") as stdout,
                    (destination / (variant["name"] + ".log")).open("x") as stderr,
                ):
                    child = subprocess.Popen(
                        plan["cargo_command"],
                        cwd=work,
                        env={**os.environ, **plan["environment"]},
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    record["pid"] = child.pid
                    save()
                    record["exit"] = monitor["monitor"](
                        BoundedChild(child, record), record, parse_vm
                    )
                if record["exit"] != 0:
                    raise RuntimeError("Cargo build failed; not a numerical mutant detection")
                built = artifact((destination / (variant["name"] + ".jsonl")).read_text())
                binary.parent.mkdir(parents=True, exist_ok=True)
                with binary.open("xb") as output, built.open("rb") as source:
                    shutil.copyfileobj(source, output)
                binary.chmod(built.stat().st_mode & 0o777)
                record.update(
                    binary=str(binary), sha256=digest(binary), bytes=binary.stat().st_size
                )
                record["status"] = "built_not_executed"
            finally:
                try:
                    stop(child)
                finally:
                    restore(target, original, variant["mutant_source_sha256"])
                    verify_sources()
                    record["source_restored_byte_exact"] = True
                    record["reaped"] = child is None or child.poll() is not None
                    save()
        verify_sources()
        result["status"] = "six_mutants_built_normal_sources_restored_no_GPU"
    except BaseException as error:
        result["status"] = "failed_or_interrupted"
        result["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    with (args.plan.parent / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.plan)
