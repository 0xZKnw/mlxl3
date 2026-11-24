#!/usr/bin/env python3
"""AGENT-2026-11-22-HARNESS-LATENCY: fixed cost of one `javascript` call and cost per host
operation, through the production MCP entry of an MLXL3 Desktop binary (`--agent-tools`).

usage: latency.py <app-binary> <out.json> [rounds]

Two servers stay alive for the whole run (structured tools, JavaScript), on one disposable
project. Every response is checked before its time is kept; a wrong or missing answer stops
the run and the report says `"status": "failed"`. Both servers are killed on any exit.
"""
import json
import os
import selectors
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BINARY, OUT = sys.argv[1], Path(sys.argv[2])
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
WARMUP = 2
TIMEOUT = 30.0


class Server:
    def __init__(self, root: Path, permissions: Path):
        self.process = subprocess.Popen(
            [BINARY, "--agent-tools", str(root), str(permissions)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.next_id = 1

    def call(self, name: str, arguments: dict) -> tuple[float, dict]:
        identifier = self.next_id
        self.next_id += 1
        line = json.dumps({"jsonrpc": "2.0", "id": identifier, "method": "tools/call",
                           "params": {"name": name, "arguments": arguments}}).encode() + b"\n"
        started = time.perf_counter()
        self.process.stdin.write(line)
        self.process.stdin.flush()
        deadline = started + TIMEOUT
        while b"\n" not in self.buffer:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"no answer within {TIMEOUT} s")
            if not self.selector.select(remaining):
                continue
            chunk = os.read(self.process.stdout.fileno(), 1 << 20)
            if not chunk:
                raise RuntimeError(f"server ended (status {self.process.poll()})")
            self.buffer += chunk
        elapsed = time.perf_counter() - started
        raw, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(raw)
        if response.get("id") != identifier or "result" not in response:
            raise RuntimeError(f"unexpected response: {raw[:300]!r}")
        return elapsed * 1000, response["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)


def text(result: dict) -> str:
    return "".join(block.get("text", "") for block in result.get("content", []))


def main() -> int:
    report = {"status": "failed", "binary": BINARY, "rounds": ROUNDS, "warmup": WARMUP,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_before": os.getloadavg()}
    servers = []
    try:
        with tempfile.TemporaryDirectory(prefix="mlxl3-agent-latency-") as directory:
            root = Path(directory) / "project"
            (root / "src").mkdir(parents=True)
            for index in range(200):
                body = ("// needle\n" if index % 10 == 0 else "") + f"export const v{index} = {index};\n" * 60
                (root / "src" / f"f{index:03}.js").write_text(body)
            (root / "big.txt").write_text("b" * 204_800)
            small = "small-content\n" * 20
            (root / "small.txt").write_text(small)
            (root / "edit.txt").write_text("value v0 end\n")

            def policy(harness: str) -> Path:
                path = Path(directory) / f"{harness}.json"
                path.write_text(json.dumps({"permissions": {"read": True, "write": True, "run": True,
                                                            "commandSeconds": 5, "harness": harness}}))
                return path

            tools = Server(root, policy("tools"))
            js = Server(root, policy("javascript"))
            servers += [tools, js]
            edits = iter(range(10_000))

            def edit_code():
                step = next(edits)
                return f"edit('edit.txt', 'v{step}', 'v{step + 1}'); return 1"

            scenarios = {
                "T-read": (tools, "read_file", lambda: {"path": "small.txt"},
                           lambda out: out.startswith(small) and out.rstrip().endswith("Exit status: 0")),
                "J-return": (js, "javascript", lambda: {"code": "return 1"}, lambda out: out == "Result:\n1"),
                "J-read10": (js, "javascript",
                             lambda: {"code": "let n = 0; for (let i = 0; i < 10; i++) n += read('small.txt').length; return n"},
                             lambda out: out == f"Result:\n{10 * len(small)}"),
                # Counted on disk at check time: J-write adds out/copy.txt after the first round.
                "J-list": (js, "javascript", lambda: {"code": "return list().length"},
                           lambda out: out == f"Result:\n{sum(1 for p in root.rglob('*') if p.is_file())}"),
                "J-search": (js, "javascript", lambda: {"code": "return search('needle').length"},
                             lambda out: out == "Result:\n20"),
                "J-write": (js, "javascript",
                            lambda: {"code": "write('out/copy.txt', 'a'.repeat(204800)); return 1"},
                            lambda out: out == "Result:\n1"
                            and (root / "out/copy.txt").stat().st_size == 204_800),
                "J-edit": (js, "javascript", lambda: {"code": edit_code()}, lambda out: out == "Result:\n1"),
                "J-run": (js, "javascript", lambda: {"code": "return run('true').status"},
                          lambda out: out == "Result:\n0"),
            }
            samples = {name: [] for name in scenarios}

            def run(name: str, keep: bool):
                server, tool, arguments, valid = scenarios[name]
                milliseconds, result = server.call(tool, arguments())
                output = text(result)
                if result.get("isError") or not valid(output):
                    raise RuntimeError(f"{name}: wrong answer {output[:300]!r}")
                if keep:
                    samples[name].append(round(milliseconds, 2))

            names = list(scenarios)
            for name in names:
                for _ in range(WARMUP):
                    run(name, keep=False)
            for round_index in range(ROUNDS):
                for name in (names if round_index % 2 == 0 else list(reversed(names))):
                    run(name, keep=True)

            summary = {}
            for name, values in samples.items():
                ordered = sorted(values)
                summary[name] = {"median_ms": round(statistics.median(ordered), 1),
                                 "p90_ms": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 1),
                                 "min_ms": ordered[0], "max_ms": ordered[-1]}
            fixed = summary["J-return"]["median_ms"]
            summary["derived"] = {
                "js_call_fixed_ms": fixed,
                "read_per_op_ms": round((summary["J-read10"]["median_ms"] - fixed) / 10, 1),
                "list_op_ms": round(summary["J-list"]["median_ms"] - fixed, 1),
                "search_op_ms": round(summary["J-search"]["median_ms"] - fixed, 1),
                "write_200k_op_ms": round(summary["J-write"]["median_ms"] - fixed, 1),
                "edit_op_ms": round(summary["J-edit"]["median_ms"] - fixed, 1),
                "run_true_op_ms": round(summary["J-run"]["median_ms"] - fixed, 1),
            }
            report.update(status="passed", samples=samples, summary=summary)
    except Exception as error:  # the report keeps the failure
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        for server in servers:
            server.close()
        report["load_after"] = os.getloadavg()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("summary", report), indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
