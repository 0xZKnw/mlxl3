#!/usr/bin/env python3
"""AGENT-2026-11-23-BOUCLE: the engine's Agent loop on a real model (behaviour, not speed).

Derived from docs/measurements/agent-harness-2026-11-23/agent_trial_harness.py. Drives the
real bridge (engine built from this branch) with the real sandbox MCP server (the app binary's
`--agent-tools ... long-calls`, persistent JS kernel) on disposable workspaces, with the new
Agent request: no total budget (`max_tokens` -1), 256 tool rounds and a per-step
`reasoning_budget`. Records forced ends of thinking, pruning statuses and stop reasons.

usage: agent_loop_trial.py ENGINE REGISTRY MODEL_DIR MODEL_NAME HELPER OUT.json PLAN   (PLAN: loop | prune)

REGISTRY is the app's agent-runtime/models.json; its first entry is copied with MODEL_DIR's
path, name and size (the installed folder may have been renamed since).

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
RUN_SECONDS = 1200
TOOL_ROUNDS = 256            # StudioModel.agentLoopToolRounds
REASONING_BUDGET = 8192      # StudioModel.agentReasoningBudget
FORCED_NOTE = "Considering the limited time, I have to act now"
# Inkling 2 Mini generation_config: temperature 1, nothing else.
SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0}

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



TASKS["snake"] = {
    "prompt": "Write snake.py: a complete terminal Snake game in a single Python file using the "
              "curses module (arrow keys, food, growing snake, score, game over). Then check "
              "that it compiles with `python3 -m py_compile snake.py`.",
    "files": {"README.md": "# Games\n"},
}

WORDS = ["maple", "cobalt", "saffron", "quartz"]
TASKS["prune"] = {
    "prompt": "Use one separate tool call per file, in this order: print the whole content of "
              "big1.txt with read(), then big2.txt, then big3.txt, then big4.txt. After the four "
              "calls, tell me the END WORD written on the last line of each file.",
    "files": {f"big{n + 1}.txt": "".join(f"row {i}: the quick brown fox jumps over the lazy dog {i * 7 % 13}\n"
                                          for i in range(240)) + f"END WORD: {word}\n"
              for n, word in enumerate(WORDS)},
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
    if task == "snake":
        path = root / "snake.py"
        if not path.exists():
            return False
        result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True,
                                timeout=30)
        return result.returncode == 0 and "curses" in path.read_text()
    if task == "prune":
        return all(word in answer.lower() for word in WORDS)
    raise ValueError(task)




def plan(name):
    """(task, reasoning_budget, context)"""
    if name == "loop":
        return 32768, [("bugfix", REASONING_BUDGET), ("bugfix", 256), ("snake", REASONING_BUDGET)]
    if name == "prune":
        return 4096, [("prune", REASONING_BUDGET)]
    raise SystemExit(f"unknown plan {name}")


def instructions():
    """The production Agent prompt (JavaScript harness, Internet off)."""
    source = (Path(__file__).resolve().parents[3]
              / "apps/MLXL3Studio/Sources/MLXL3Studio/AgentSandbox.swift").read_text()
    text = re.search(r"static let javascriptInstructions =\s*\"((?:[^\"\\]|\\.)*)\"", source).group(1)
    return text.replace('\\"', '"')


class BridgeError(RuntimeError):
    pass


def main():
    engine, registry_source, model_dir, model_name, helper, out, plan_name = sys.argv[1:8]
    context, runs = plan(plan_name)
    work = Path(tempfile.mkdtemp(prefix="mlxl3-agent-loop-"))
    root = work / "files"
    studio = work / "studio"
    studio.mkdir()
    project = studio / "project.json"
    registry = work / "registry" / "models.json"
    registry.parent.mkdir()
    entry = dict(next(iter(json.loads(Path(registry_source).read_text())["models"].values())))
    entry.update(path=model_dir, name=model_name,
                 size_bytes=sum(f.stat().st_size for f in Path(model_dir).rglob("*") if f.is_file()))
    registry.write_text(json.dumps({"version": 1, "models": {model_name: entry}}))
    # The app's Agent registry (StudioModel.prepareAgentRegistry): the default Exa server off.
    (registry.parent / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
        "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
        "sandbox": {"enabled": True, "command": str(Path(helper).resolve()),
                    "args": ["--agent-tools", str(root), str(project), "long-calls"],
                    "callTimeoutSeconds": 43}}}))
    report = {"id": "AGENT-2026-11-23-BOUCLE", "plan": plan_name, "engine": engine, "model": model_name,
              "model_path": model_dir, "helper": helper, "sampling": SAMPLING, "context": context,
              "status": "running", "runs": []}

    def save():
        Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))

    stderr = open(Path(out).with_suffix(".stderr.log"), "w")
    process = subprocess.Popen(
        [engine, "--registry", str(registry), "bridge", model_name, "--context-length", str(context)],
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
                    "model", "runtime_commit", "agent_loop_supported", "tool_rounds_supported",
                    "context_limit") if k in event}
                if event.get("agent_loop_supported") is not True:
                    raise BridgeError("engine does not announce agent_loop_supported")
                break
        for index, (task, budget) in enumerate(runs):
            if root.exists():
                shutil.rmtree(root)
            (studio / "js-state.json").unlink(missing_ok=True)
            for path, text in TASKS[task]["files"].items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            project.write_text(json.dumps({"permissions": {
                "read": True, "write": True, "run": True, "commandSeconds": 5,
                "harness": "javascript", "internet": False}}))
            send({"type": "set_mcp", "enabled": True, "request_id": f"mcp-{index}"})
            while True:
                event = read_event(EVENT_SECONDS)
                if event.get("type") == "mcp_status" and event.get("request_id") == f"mcp-{index}":
                    break
            catalog = [tool["function"]["name"] for tool in event.get("mcp_tool_catalog", [])]
            run = {"task": task, "reasoning_budget": budget, "tools": catalog,
                   "mcp_errors": event.get("mcp_errors")}
            if catalog != ["sandbox.javascript"]:
                run.update(status="setup-failed", error=f"unexpected tool catalog {catalog}")
                report["runs"].append(run)
                save()
                continue
            request_id = f"agent-{index}"
            send({"type": "generate", "request_id": request_id,
                  "conversation_id": f"loop-{index}-{time.time_ns()}",
                  "messages": [{"role": "system", "content": instructions()},
                               {"role": "user", "content": TASKS[task]["prompt"]}],
                  "max_tokens": -1, "tool_rounds": TOOL_ROUNDS, "reasoning_budget": budget,
                  "mcp_enabled": True, **SAMPLING})
            started = time.monotonic()
            answer, calls, results, error, complete = "", [], [], None, None
            forced, statuses, timed_out = 0, [], False
            thinking_steps, thinking = [], 0
            while complete is None and error is None:
                if time.monotonic() - started > RUN_SECONDS and not timed_out:
                    process.send_signal(signal.SIGUSR1)
                    timed_out = True
                event = read_event(EVENT_SECONDS)
                if event.get("request_id") not in (None, request_id):
                    continue
                kind = event.get("type")
                if kind == "delta" and event.get("phase") == "answer":
                    answer += event.get("text", "")
                elif kind == "delta" and event.get("phase") == "thinking":
                    text = event.get("text", "")
                    if FORCED_NOTE in text:
                        forced += 1
                    thinking += len(text)
                elif kind == "generation_status":
                    text = event.get("text") or ""
                    if any(key in text for key in ("Acting on", "Freeing context", "Context full", "Reading context")):
                        statuses.append(text)
                elif kind == "tool_start":
                    calls.append(event.get("tool_name"))
                    thinking_steps.append(thinking)
                    thinking = 0
                elif kind == "tool_result":
                    results.append({"tool": event.get("tool_name"), "is_error": event.get("is_error"),
                                    "text": (event.get("text") or "")[:1500]})
                elif kind == "error":
                    error = event.get("message")
                elif kind == "cancelled":
                    error = f"cancelled after {RUN_SECONDS} s"
                elif kind == "complete":
                    complete = event
            thinking_steps.append(thinking)
            stats = (complete or {}).get("stats", {})
            run.update(
                status="timeout" if timed_out else "finished", error=error,
                passed=bool(complete) and oracle(task, root, answer),
                seconds=round(time.monotonic() - started, 1),
                tool_rounds=(complete or {}).get("tool_rounds"), tool_calls=len(calls),
                unreadable_calls=calls.count("invalid_tool_call"),
                tool_errors=sum(1 for r in results if r["is_error"]),
                forced_closes=forced, thinking_chars_per_step=thinking_steps,
                statuses=statuses, generated_tokens=stats.get("generated_tokens"),
                decode_tps=stats.get("decode_tps"), stop_reason=(complete or {}).get("stop_reason"),
                context_full=(complete or {}).get("context_full"),
                markup_in_answer="<|content_invoke_tool_json|>" in answer or "<tool_call>" in answer,
                answer=answer[-3000:], results=results)
            report["runs"].append(run)
            save()
            print(f"[{index + 1}] {task} budget={budget}: {'PASS' if run['passed'] else 'FAIL'} "
                  f"rounds={run['tool_rounds']} calls={run['tool_calls']} forced={forced} "
                  f"unreadable={run['unreadable_calls']} errors={run['tool_errors']} "
                  f"stop={run['stop_reason']} tokens={run['generated_tokens']} {run['seconds']} s "
                  f"statuses={len(statuses)} {error or ''}", flush=True)
        report["status"] = "complete" if all(r["status"] == "finished" and not r.get("error")
                                               for r in report["runs"]) else "incomplete"
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
    save()
    print(json.dumps({"status": report["status"],
                      "runs": [(r["task"], r.get("passed"), r.get("forced_closes")) for r in report["runs"]]}))
    sys.exit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
