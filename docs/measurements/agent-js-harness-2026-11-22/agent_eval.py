#!/usr/bin/env python3
"""AGENT-2026-11-22-JS-HARNESS: Agent mode with structured tools vs one JavaScript tool.

Drives the real `mlxl3 bridge` with the real sandbox MCP helper (`--agent-tools`, built from
the production AgentSandbox.swift / AgentJavaScript.swift) on disposable workspaces. Each
task has an oracle that does not trust the model: a test script run afterwards, file
contents, or facts the final answer must contain.

usage: agent_eval.py ENGINE MODELS_JSON MODEL HELPER OUT.json [REPS]

One bridge process (one GPU model) for the whole campaign; the harness changes through
`set_mcp` between runs, as the Desktop app does. Order alternates per repetition. A
timeout, a bridge error or a missing `complete` event is recorded as a failed run, never
as a pass; the report status is `complete` only when every run finished.
"""
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

READY_SECONDS = 900
EVENT_SECONDS = 600
RUN_SECONDS = 900
# Qwen3.6 model card, "thinking · precise coding" preset of the Desktop app.
SAMPLING = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.0,
            "presence_penalty": 0.0}
MAX_TOKENS = 16384

TASKS = {
    "bugfix": {
        "prompt": "The test script test.sh fails. Find the bugs in the project and fix them so "
                  "that `sh test.sh` prints PASS. Run the test to confirm.",
        "files": {
            "src/stats.pl": (
                "use strict;\nuse warnings;\n\n"
                "sub mean {\n    my @values = @_;\n    return 0 unless @values;\n"
                "    my $sum = 0;\n    $sum += $_ for @values;\n"
                "    return $sum / (scalar(@values) - 1);\n}\n\n"
                "sub median {\n    my @sorted = sort @_;\n    my $middle = int(@sorted / 2);\n"
                "    return $sorted[$middle];\n}\n\n1;\n"),
            "test.sh": (
                "#!/bin/sh\n"
                "out=$(perl -e 'require \"./src/stats.pl\"; print mean(1, 2, 3), \" \", "
                "median(10, 9, 2), \"\\n\"')\n"
                "if [ \"$out\" = \"2 9\" ]; then echo PASS; else echo \"FAIL: got $out, "
                "expected 2 9\"; exit 1; fi\n"),
            "README.md": "# Stats\n\nSmall Perl helpers. Run `sh test.sh`.\n",
        },
    },
    "rename": {
        "prompt": "Rename the function fetchUser to loadUser everywhere in the project "
                  "(definition, every call and the docs). Then confirm with `sh check.sh`.",
        "files": {
            "src/api.js": "export function fetchUser(id) {\n  return request(`/users/${id}`);\n}\n",
            "src/profile.js": "import { fetchUser } from './api.js';\n\nexport async function "
                              "profile(id) {\n  const user = await fetchUser(id);\n  return user.name;\n}\n",
            "src/admin.js": "import { fetchUser } from './api.js';\n\nexport const ban = "
                            "async (id) => (await fetchUser(id)).ban();\n",
            "src/cache.js": "const seen = new Map();\n// fetchUser results are cached here.\n"
                            "export const remember = (user) => seen.set(user.id, user);\n",
            "README.md": "# Users\n\nCall `fetchUser(id)` to get a user.\n",
            "check.sh": ("#!/bin/sh\nif grep -rn fetchUser src README.md; then echo FAIL; exit 1; fi\n"
                         "n=$(grep -rno loadUser src README.md | wc -l | tr -d ' ')\n"
                         "if [ \"$n\" = 7 ]; then echo PASS; else echo \"FAIL: $n loadUser\"; exit 1; fi\n"),
        },
    },
    "question": {
        "prompt": "How many TODO comments are in this project, and which files contain them? "
                  "Answer with the total and the list of files. Do not change any file.",
        "files": {
            "src/main.c": "int main(void) {\n  // TODO: parse arguments\n  return run();\n}\n",
            "src/run.c": "// TODO: handle errors\nint run(void) {\n  /* TODO: log */\n  return 0;\n}\n",
            "src/util/strings.c": "// helpers\n// TODO: unicode\n",
            "docs/notes.md": "# Notes\n\n- TODO: write the manual\n- TODO: add examples\n- done: setup\n",
            "src/empty.c": "// nothing to do here\n",
        },
    },
}


def oracle(task, root, answer):
    """Independent verdict: never the model's own claim."""
    def sh(script):
        result = subprocess.run(["/bin/sh", script], cwd=root, capture_output=True, text=True,
                                timeout=30)
        return result.returncode == 0 and result.stdout.strip().endswith("PASS")
    if task == "bugfix":
        return sh("test.sh") and (root / "test.sh").read_text() == TASKS["bugfix"]["files"]["test.sh"]
    if task == "rename":
        return sh("check.sh") and (root / "check.sh").read_text() == TASKS["rename"]["files"]["check.sh"]
    if task == "question":
        unchanged = all((root / path).read_text() == text
                        for path, text in TASKS["question"]["files"].items())
        names = ["main.c", "run.c", "strings.c", "notes.md"]
        mentions_total = re.search(r"\b6\b|\bsix\b", answer, re.IGNORECASE) is not None
        return unchanged and mentions_total and all(name in answer for name in names) \
            and "empty.c" not in answer.replace("empty.c contains no", "")
    raise ValueError(task)


def instructions(harness):
    """The production Agent prompt, read from AgentSandbox.instructions(for:)."""
    source = (Path(__file__).resolve().parents[3]
              / "apps/MLXL3Studio/Sources/MLXL3Studio/AgentSandbox.swift").read_text()
    limits = re.search(r'let limits =\s*"([^"]+)"', source).group(1)
    case = re.search(r"case \.%s:\s*return\s*\"((?:[^\"\\]|\\.)*)\"" % harness, source).group(1)
    return case.replace("\\(limits)", limits).replace('\\"', '"')


class BridgeError(RuntimeError):
    pass


def main():
    engine, models_json, model, helper, out = sys.argv[1:6]
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    work = Path(tempfile.mkdtemp(prefix="mlxl3-agent-eval-"))
    root = work / "files"
    project = work / "project.json"
    registry = work / "registry" / "models.json"
    registry.parent.mkdir()
    entry = json.loads(Path(models_json).read_text())["models"][model]
    registry.write_text(json.dumps({"version": 1, "models": {model: entry}}))
    (registry.parent / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
        "sandbox": {"enabled": True, "command": str(Path(helper).resolve()),
                    "args": ["--agent-tools", str(root), str(project)]}}}))
    report = {"id": "AGENT-2026-11-22-JS-HARNESS", "engine": engine, "model": model,
              "model_path": entry.get("path"), "helper": helper, "sampling": SAMPLING,
              "max_tokens": MAX_TOKENS, "reps": reps, "status": "running", "runs": []}

    def save():
        Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))

    stderr = open(Path(out).with_suffix(".stderr.log"), "w")
    process = subprocess.Popen(
        [engine, "--registry", str(registry), "bridge", model, "--context-length", "32768"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    def read_event(seconds):
        deadline = time.monotonic() + seconds
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(timeout=left):
                raise BridgeError(f"no event within {seconds} s")
            line = process.stdout.readline()
            if not line:
                raise BridgeError(f"bridge exited (code {process.poll()})")
            try:
                return json.loads(line)
            except json.JSONDecodeError as error:
                raise BridgeError(f"invalid event line: {line[:200]!r}") from error

    def send(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    try:
        while True:
            event = read_event(READY_SECONDS)
            if event.get("type") == "error":
                raise BridgeError(f"bridge error before ready: {event.get('message')}")
            if event.get("type") == "ready":
                report["ready"] = {k: event.get(k) for k in ("model", "runtime", "version") if k in event}
                break
        order = []
        for rep in range(reps):
            harnesses = ["tools", "javascript"] if rep % 2 == 0 else ["javascript", "tools"]
            for task in TASKS:
                for harness in harnesses:
                    order.append((rep, task, harness))
        for index, (rep, task, harness) in enumerate(order):
            if root.exists():
                shutil.rmtree(root)
            for path, text in TASKS[task]["files"].items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            project.write_text(json.dumps({"permissions": {
                "read": True, "write": True, "run": True, "commandSeconds": 5, "harness": harness}}))
            send({"type": "set_mcp", "enabled": True, "request_id": f"mcp-{index}"})
            while True:
                event = read_event(EVENT_SECONDS)
                if event.get("type") == "mcp_status" and event.get("request_id") == f"mcp-{index}":
                    break
            catalog = [tool["function"]["name"] for tool in event.get("mcp_tool_catalog", [])]
            run = {"rep": rep, "task": task, "harness": harness, "tools": catalog,
                   "mcp_errors": event.get("mcp_errors")}
            expected = ["sandbox.javascript"] if harness == "javascript" else \
                ["sandbox.list_files", "sandbox.read_file", "sandbox.run", "sandbox.write_file"]
            if sorted(catalog) != sorted(expected):
                run.update(status="setup-failed", error=f"unexpected tool catalog {catalog}")
                report["runs"].append(run)
                save()
                continue
            request_id = f"agent-{index}"
            send({"type": "generate", "request_id": request_id,
                  "conversation_id": f"eval-{index}-{time.time_ns()}", "reuse_prompt_cache": False,
                  "messages": [{"role": "system", "content": instructions(harness)},
                               {"role": "user", "content": TASKS[task]["prompt"]}],
                  "max_tokens": MAX_TOKENS, "mcp_enabled": True, **SAMPLING})
            started = time.monotonic()
            answer, calls, results, error, complete = "", [], [], None, None
            timed_out = False
            while complete is None and error is None:
                if time.monotonic() - started > RUN_SECONDS and not timed_out:
                    process.send_signal(signal.SIGUSR1)  # the Desktop app's cancellation
                    timed_out = True
                event = read_event(EVENT_SECONDS)
                if event.get("request_id") not in (None, request_id):
                    continue
                kind = event.get("type")
                if kind == "delta" and event.get("phase") == "answer":
                    answer += event.get("text", "")
                elif kind == "tool_start":
                    calls.append(event.get("tool_name"))
                elif kind == "tool_result":
                    results.append({"tool": event.get("tool_name"), "is_error": event.get("is_error"),
                                    "text": (event.get("text") or "")[:1500]})
                elif kind == "error":
                    error = event.get("message")
                elif kind == "cancelled":
                    error = f"cancelled after {RUN_SECONDS} s"
                elif kind == "complete":
                    complete = event
            stats = (complete or {}).get("stats", {})
            # An engine error (e.g. the 5-round limit) is the model's outcome; a timeout is not.
            run.update(
                status="timeout" if timed_out else "finished", error=error,
                passed=bool(complete) and oracle(task, root, answer),
                seconds=round(time.monotonic() - started, 1),
                tool_rounds=(complete or {}).get("tool_rounds"), tool_calls=len(calls),
                tool_errors=sum(1 for r in results if r["is_error"]), calls=calls,
                generated_tokens=stats.get("generated_tokens"),
                stop_reason=(complete or {}).get("stop_reason"),
                answer=answer[-3000:], results=results)
            report["runs"].append(run)
            save()
            print(f"[{index + 1}/{len(order)}] {task}/{harness} rep{rep}: "
                  f"{'PASS' if run['passed'] else 'FAIL'} rounds={run['tool_rounds']} "
                  f"calls={run['tool_calls']} errors={run['tool_errors']} "
                  f"tokens={run['generated_tokens']} {run['seconds']} s {error or ''}", flush=True)
        report["status"] = "complete" if all(r["status"] == "finished" for r in report["runs"]) \
            else "incomplete"
    except BridgeError as error:
        report["status"] = "failed"
        report["error"] = str(error)
    finally:
        save()
        if process.poll() is None:
            try:
                send({"type": "shutdown"})
                process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        stderr.close()
        shutil.rmtree(work, ignore_errors=True)
    summary = {}
    for run in report["runs"]:
        key = run["harness"]
        bucket = summary.setdefault(key, {"runs": 0, "passed": 0, "rounds": [], "calls": [], "tokens": [], "seconds": []})
        bucket["runs"] += 1
        bucket["passed"] += int(bool(run.get("passed")))
        for name, field in (("rounds", "tool_rounds"), ("calls", "tool_calls"),
                            ("tokens", "generated_tokens"), ("seconds", "seconds")):
            if run.get(field) is not None:
                bucket[name].append(run[field])
    report["summary"] = summary
    save()
    print(json.dumps({"status": report["status"], "summary": summary}, indent=1))
    sys.exit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
