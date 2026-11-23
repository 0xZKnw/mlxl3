#!/usr/bin/env python3
"""AGENT-2026-11-23-GPU-TRIAL: Agent mode on a real model after the harness series.

Drives the real bridge (engine built from the release source) with the real sandbox MCP
helper (`--agent-tools ... long-calls`, built from the production Swift files) on disposable
workspaces, with the Desktop app's Agent request (32,768 tokens, 24 tool rounds). Tasks and
oracles are the ones pre-registered in AGENT-2026-11-22-JS-HARNESS, plus a live `fetch` task
(Internet on, GitHub hosts only) and a tool-round-limit scenario. The verdict never trusts
the model: test scripts run afterwards, file contents, fetched text in the tool results.

usage: agent_trial.py ENGINE REGISTRY MODEL HELPER OUT.json PLAN   (PLAN: inkling | qwen)

One bridge (one GPU model) per invocation. A timeout, a bridge error or a missing `complete`
is a failed run, never a pass; `status` is `complete` only when every run finished.
"""
import json
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
MAX_TOKENS = 32768          # StudioModel.agentTokenBudget
TOOL_ROUNDS = 24            # StudioModel.agentToolRounds
HOSTS = ["raw.githubusercontent.com", "github.com", "codeload.github.com"]
SAMPLING = {
    # Inkling 2 Mini generation_config: temperature 1, nothing else.
    "inkling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "presence_penalty": 0.0},
    # Qwen model card, "thinking, precise coding".
    "qwen": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.0, "presence_penalty": 0.0},
}

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



TASKS["fetch"] = {
    "prompt": "Read https://raw.githubusercontent.com/octocat/Hello-World/master/README with the "
              "fetch function and tell me exactly what the file says.",
    "files": {"README.md": "# Empty project\n"},
}


def passed(task, root, answer, results):
    if task == "fetch":
        fetched = any("Hello World!" in (r["text"] or "") and not r["is_error"] for r in results)
        return fetched and "hello world" in answer.lower()
    return oracle(task, root, answer)


def plan(name):
    """(rep, task, harness, internet, rounds), order alternated per repetition."""
    runs = []
    if name == "inkling":
        for rep in range(2):
            for task in (["bugfix", "rename", "question", "fetch"] if rep == 0 else ["fetch", "question", "rename", "bugfix"]):
                runs.append((rep, task, "javascript", task == "fetch", TOOL_ROUNDS))
        for task in ("bugfix", "rename", "question"):
            runs.append((0, task, "tools", False, TOOL_ROUNDS))
        runs.append((0, "bugfix", "javascript", False, 1))
    elif name == "qwen":
        # Reduced at the user's request ("il est lent, accélère") from 5 runs: the loop on
        # another call format, and its round limit.
        runs.append((0, "bugfix", "javascript", False, TOOL_ROUNDS))
        runs.append((0, "bugfix", "javascript", False, 1))
    else:
        raise SystemExit(f"unknown plan {name}")
    return runs


def instructions(harness, internet):
    """The production Agent prompt, read from AgentSandbox.instructions(for:)."""
    source = (Path(__file__).resolve().parents[3]
              / "apps/MLXL3Studio/Sources/MLXL3Studio/AgentSandbox.swift").read_text()
    def literal(name):
        text = re.search(r"static let %s =\s*\"((?:[^\"\\]|\\.)*)\"" % name, source).group(1)
        return text.replace('\\"', '"')
    if harness == "javascript":
        return literal("javascriptInstructions") + (" " + literal("internetInstructions") if internet else "")
    limits = re.search(r'let limits =\s*"([^"]+)"', source).group(1)
    case = re.search(r"case \.tools:\s*return\s*\"((?:[^\"\\]|\\.)*)\"", source).group(1)
    return case.replace("\\(limits)", limits).replace('\\"', '"')


class BridgeError(RuntimeError):
    pass


def main():
    engine, registry_source, model, helper, out, plan_name = sys.argv[1:7]
    sampling = SAMPLING[plan_name]
    work = Path(tempfile.mkdtemp(prefix="mlxl3-agent-trial-"))
    root = work / "files"
    studio = work / "studio"
    studio.mkdir()
    project = studio / "project.json"
    registry = work / "registry" / "models.json"
    registry.parent.mkdir()
    entry = json.loads(Path(registry_source).read_text())["models"][model]
    registry.write_text(json.dumps({"version": 1, "models": {model: entry}}))
    # The app's registry once the engine announced long calls (StudioModel.prepareAgentRegistry).
    (registry.parent / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
        "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
        "sandbox": {"enabled": True, "command": str(Path(helper).resolve()),
                    "args": ["--agent-tools", str(root), str(project), "long-calls"],
                    "callTimeoutSeconds": 43}}}))
    report = {"id": "AGENT-2026-11-23-GPU-TRIAL", "plan": plan_name, "engine": engine, "model": model,
              "model_path": entry.get("path"), "helper": helper, "sampling": sampling,
              "max_tokens": MAX_TOKENS, "status": "running", "runs": []}

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
                report["ready"] = {k: event.get(k) for k in (
                    "model", "runtime_commit", "tool_rounds_supported", "mcp_call_timeout_supported",
                    "dspark_supported") if k in event}
                break
        for index, (rep, task, harness, internet, rounds) in enumerate(plan(plan_name)):
            for folder in (root,):
                if folder.exists():
                    shutil.rmtree(folder)
            (studio / "js-state.json").unlink(missing_ok=True)
            for path, text in TASKS[task]["files"].items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            project.write_text(json.dumps({"permissions": {
                "read": True, "write": True, "run": True, "commandSeconds": 5, "harness": harness,
                "internet": internet, "internetHosts": HOSTS}}))
            send({"type": "set_mcp", "enabled": True, "request_id": f"mcp-{index}"})
            while True:
                event = read_event(EVENT_SECONDS)
                if event.get("type") == "mcp_status" and event.get("request_id") == f"mcp-{index}":
                    break
            catalog = [tool["function"]["name"] for tool in event.get("mcp_tool_catalog", [])]
            run = {"rep": rep, "task": task, "harness": harness, "internet": internet, "rounds_limit": rounds,
                   "tools": catalog, "mcp_errors": event.get("mcp_errors")}
            expected = ["sandbox.javascript"] if harness == "javascript" else \
                ["sandbox.list_files", "sandbox.read_file", "sandbox.run", "sandbox.write_file"]
            if sorted(catalog) != sorted(expected):
                run.update(status="setup-failed", error=f"unexpected tool catalog {catalog}")
                report["runs"].append(run)
                save()
                continue
            request_id = f"agent-{index}"
            send({"type": "generate", "request_id": request_id,
                  "conversation_id": f"trial-{index}-{time.time_ns()}", "reuse_prompt_cache": False,
                  "messages": [{"role": "system", "content": instructions(harness, internet)},
                               {"role": "user", "content": TASKS[task]["prompt"]}],
                  "max_tokens": MAX_TOKENS, "tool_rounds": rounds, "mcp_enabled": True, **sampling})
            started = time.monotonic()
            answer, calls, results, error, complete = "", [], [], None, None
            shown_arguments, timed_out = 0, False
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
                    arguments = event.get("arguments")
                    shown_arguments += int(isinstance(arguments, dict) and bool(arguments))
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
            run.update(
                status="timeout" if timed_out else "finished", error=error,
                passed=bool(complete) and passed(task, root, answer, results),
                seconds=round(time.monotonic() - started, 1),
                tool_rounds=(complete or {}).get("tool_rounds"), tool_calls=len(calls),
                unreadable_calls=calls.count("invalid_tool_call"), arguments_shown=shown_arguments,
                tool_errors=sum(1 for r in results if r["is_error"]), calls=calls,
                generated_tokens=stats.get("generated_tokens"), decode_tps=stats.get("decode_tps"),
                stop_reason=(complete or {}).get("stop_reason"),
                markup_in_answer="<|content_invoke_tool_json|>" in answer or "<tool_call>" in answer,
                answer=answer[-3000:], results=results)
            report["runs"].append(run)
            save()
            print(f"[{index + 1}] {task}/{harness} rep{rep} rounds<={rounds}: "
                  f"{'PASS' if run['passed'] else 'FAIL'} rounds={run['tool_rounds']} calls={run['tool_calls']} "
                  f"unreadable={run['unreadable_calls']} errors={run['tool_errors']} "
                  f"stop={run['stop_reason']} tokens={run['generated_tokens']} {run['seconds']} s {error or ''}",
                  flush=True)
        report["status"] = "complete" if all(r["status"] == "finished" for r in report["runs"]) else "incomplete"
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
        key = f"{run['harness']}/{run['task']}" + ("/limit" if run.get("rounds_limit") == 1 else "")
        bucket = summary.setdefault(key, {"runs": 0, "passed": 0})
        bucket["runs"] += 1
        bucket["passed"] += int(bool(run.get("passed")))
    report["summary"] = summary
    save()
    print(json.dumps({"status": report["status"], "summary": summary}, indent=1))
    sys.exit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
