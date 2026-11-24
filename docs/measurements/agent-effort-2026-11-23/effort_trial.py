#!/usr/bin/env python3
"""AGENT-2026-11-23-EFFORT: Agent mode on Inkling 2 Mini at several reasoning levels.

One bridge (one GPU model) for the whole campaign; the only variable is the request's
`reasoning_effort` (absent = the template default 0.9, as the app sends it). Production
Agent request otherwise: `max_tokens` -1, 256 tool rounds, no `reasoning_budget`, real
sandbox MCP server (`--agent-tools ... long-calls`, persistent JS kernel), Internet off.
Oracles never trust the model: scripts run afterwards, file contents, fixed test boards.

usage: effort_trial.py ENGINE REGISTRY MODEL_DIR MODEL_NAME HELPER OUT.json [REPS]

A timeout, a bridge error or a missing `complete` is a failed run, never a pass; the
report is saved after every run, `status` is `complete` only when every run finished.
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
CONTEXT = 32768
TOOL_ROUNDS = 256            # StudioModel.agentLoopToolRounds
LEVELS = ["none", "low", "medium", None]   # None = field absent = template default (0.9)
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




TASKS["feature"] = {
    "prompt": "Implement the missing methods of `Inventory` in inventory.py so that "
              "`python3 test_inventory.py` prints OK. Do not modify the test file.",
    "files": {
        "inventory.py": (
            "class Inventory:\n"
            "    \"\"\"Stock of named items: name -> (quantity, unit_price).\"\"\"\n\n"
            "    def __init__(self):\n        self.items = {}\n\n"
            "    def add_item(self, name, quantity, unit_price):\n"
            "        if quantity <= 0:\n            raise ValueError(\"quantity must be positive\")\n"
            "        current, _ = self.items.get(name, (0, unit_price))\n"
            "        self.items[name] = (current + quantity, unit_price)\n\n"
            "    def remove_item(self, name, quantity):\n"
            "        \"\"\"Remove quantity; KeyError if unknown, ValueError if more than in stock;\n"
            "        an item that reaches 0 disappears.\"\"\"\n"
            "        raise NotImplementedError\n\n"
            "    def total_value(self):\n"
            "        \"\"\"Sum of quantity * unit_price, rounded to 2 decimals.\"\"\"\n"
            "        raise NotImplementedError\n\n"
            "    def low_stock(self, threshold):\n"
            "        \"\"\"Sorted names whose quantity is below threshold.\"\"\"\n"
            "        raise NotImplementedError\n"),
        "test_inventory.py": (
            "from inventory import Inventory\n\n"
            "inv = Inventory()\n"
            "inv.add_item('apple', 10, 0.5)\ninv.add_item('pear', 3, 1.25)\ninv.add_item('fig', 1, 2.1)\n"
            "inv.remove_item('apple', 4)\nassert inv.items['apple'][0] == 6\n"
            "inv.remove_item('fig', 1)\nassert 'fig' not in inv.items\n"
            "try:\n    inv.remove_item('pear', 4)\n    raise SystemExit('expected ValueError')\n"
            "except ValueError:\n    pass\n"
            "try:\n    inv.remove_item('kiwi', 1)\n    raise SystemExit('expected KeyError')\n"
            "except KeyError:\n    pass\n"
            "assert inv.total_value() == 6.75, inv.total_value()\n"
            "inv.add_item('kiwi', 2, 0.333)\nassert inv.total_value() == 7.42, inv.total_value()\n"
            "assert inv.low_stock(4) == ['kiwi', 'pear'], inv.low_stock(4)\n"
            "assert inv.low_stock(1) == []\n"
            "print('OK')\n"),
    },
}

TASKS["generate"] = {
    "prompt": "Write tictactoe.py: a complete two-player terminal tic-tac-toe game in one Python "
              "file. It must define `winner(board)`, where board is a list of 9 strings ('X', 'O' "
              "or ' ') read row by row, returning 'X' or 'O' for a winner, 'draw' when the board "
              "is full with no winner, and None while the game goes on. The game loop must run "
              "only under `if __name__ == '__main__':`. Then check winner() yourself.",
    "files": {"README.md": "# Games\n"},
}

WINNER_CASES = [
    (["X", "X", "X", " ", "O", "O", " ", " ", " "], "X"),
    (["O", "X", "X", "O", "X", " ", "O", " ", " "], "O"),
    (["X", "O", "O", " ", "X", " ", " ", " ", "X"], "X"),
    (["X", "O", "X", "X", "O", "O", "O", "X", "X"], "draw"),
    (["X", "O", " ", " ", " ", " ", " ", " ", " "], None),
    ([" "] * 9, None),
]


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
        mentions_total = re.search(r"\b6\b|\bsix\b", answer, re.IGNORECASE) is not None
        names = ["main.c", "run.c", "strings.c", "notes.md"]
        # empty.c may only appear in a sentence (or line) saying it has no TODO
        pieces = re.split(r"(?<=[.!?])\s+|\n", answer)
        wrong_empty = [p for p in pieces if "empty.c" in p
                       and not re.search(r"\b(no|none|zero|0|without|nothing)\b", p, re.IGNORECASE)]
        return unchanged and mentions_total and all(name in answer for name in names) and not wrong_empty
    if task == "feature":
        test = root / "test_inventory.py"
        if test.read_text() != TASKS["feature"]["files"]["test_inventory.py"]:
            return False
        # -s -E (not -I): the test imports inventory.py from its own folder
        result = subprocess.run([sys.executable, "-s", "-E", "test_inventory.py"], cwd=root,
                                capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and result.stdout.strip().endswith("OK")
    if task == "generate":
        path = root / "tictactoe.py"
        if not path.exists():
            return False
        probe = ("import importlib.util, json, sys\n"
                 "spec = importlib.util.spec_from_file_location('ttt', sys.argv[1])\n"
                 "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
                 "cases = json.loads(sys.argv[2])\n"
                 "print(json.dumps([module.winner(list(board)) == expected for board, expected in cases]))\n")
        try:
            result = subprocess.run([sys.executable, "-I", "-c", probe, str(path), json.dumps(WINNER_CASES)],
                                    cwd=root, capture_output=True, text=True, timeout=20,
                                    stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
        try:
            return result.returncode == 0 and all(json.loads(result.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return False
    raise ValueError(task)


def plan(reps):
    order = ["bugfix", "rename", "question", "feature", "generate"]
    runs = []
    for rep in range(reps):
        tasks = order if rep % 2 == 0 else list(reversed(order))
        levels = LEVELS if rep % 2 == 0 else list(reversed(LEVELS))
        for task in tasks:
            for level in levels:
                runs.append((rep, task, level))
    return runs


def instructions():
    """The production Agent prompt (JavaScript harness, Internet off)."""
    source = (Path(__file__).resolve().parents[3]
              / "apps/MLXL3Studio/Sources/MLXL3Studio/AgentSandbox.swift").read_text()
    text = re.search(r"static let javascriptInstructions =\s*\"((?:[^\"\\]|\\.)*)\"", source).group(1)
    return text.replace('\\"', '"')


class BridgeError(RuntimeError):
    pass


def main():
    engine, registry_source, model_dir, model_name, helper, out = sys.argv[1:7]
    reps = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    work = Path(tempfile.mkdtemp(prefix="mlxl3-agent-effort-"))
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
    report = {"id": "AGENT-2026-11-23-EFFORT", "engine": engine, "model": model_name,
              "model_path": model_dir, "helper": helper, "sampling": SAMPLING, "context": CONTEXT,
              "reps": reps, "status": "running", "runs": []}

    def save():
        Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))

    stderr = open(Path(out).with_suffix(".stderr.log"), "w")
    process = subprocess.Popen(
        [engine, "--registry", str(registry), "bridge", model_name, "--context-length", str(CONTEXT)],
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
                    "model", "runtime_commit", "agent_loop_supported", "reasoning_effort_supported",
                    "context_limit") if k in event}
                if event.get("agent_loop_supported") is not True:
                    raise BridgeError("engine does not announce agent_loop_supported")
                break
        for index, (rep, task, level) in enumerate(plan(reps)):
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
            run = {"rep": rep, "task": task, "effort": level or "default", "tools": catalog}
            if catalog != ["sandbox.javascript"]:
                run.update(status="setup-failed", error=f"unexpected tool catalog {catalog}",
                           passed=False)
                report["runs"].append(run)
                save()
                continue
            request_id = f"agent-{index}"
            request = {"type": "generate", "request_id": request_id,
                       "conversation_id": f"effort-{index}-{time.time_ns()}",
                       "messages": [{"role": "system", "content": instructions()},
                                    {"role": "user", "content": TASKS[task]["prompt"]}],
                       "max_tokens": -1, "tool_rounds": TOOL_ROUNDS, "mcp_enabled": True, **SAMPLING}
            if level is not None:
                request["reasoning_effort"] = level
            send(request)
            started = time.monotonic()
            answer, calls, results, error, complete = "", [], [], None, None
            thinking, statuses, timed_out = 0, [], False
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
                elif kind == "delta" and event.get("phase") == "thinking":
                    thinking += len(event.get("text", ""))
                elif kind == "generation_status":
                    text = event.get("text") or ""
                    if text.startswith(("Freeing context", "Context full")):
                        statuses.append(text)
                elif kind == "tool_start":
                    calls.append(event.get("tool_name"))
                elif kind == "tool_result":
                    results.append({"is_error": event.get("is_error"),
                                    "text": (event.get("text") or "")[:600]})
                elif kind == "error":
                    error = event.get("message")
                elif kind == "cancelled":
                    error = f"cancelled after {RUN_SECONDS} s"
                elif kind == "complete":
                    complete = event
            stats = (complete or {}).get("stats", {})
            seconds = round(time.monotonic() - started, 1)
            try:
                verdict = bool(complete) and not timed_out and oracle(task, root, answer)
            except Exception as failure:  # an oracle crash is a failed run, never a pass
                verdict, error = False, error or f"oracle error: {failure}"
            run.update(
                status="timeout" if timed_out else "finished", error=error, passed=verdict,
                seconds=seconds, tool_rounds=(complete or {}).get("tool_rounds"),
                tool_calls=len(calls), unreadable_calls=calls.count("invalid_tool_call"),
                tool_errors=sum(1 for r in results if r["is_error"]), thinking_chars=thinking,
                generated_tokens=stats.get("generated_tokens"), decode_tps=stats.get("decode_tps"),
                stop_reason=(complete or {}).get("stop_reason"), pruning=statuses,
                answer=answer[-1500:], results=results[-6:])
            report["runs"].append(run)
            save()
            print(f"[{index + 1}/{len(plan(reps))}] rep{rep} {task} effort={run['effort']}: "
                  f"{'PASS' if verdict else 'FAIL'} {seconds} s rounds={run['tool_rounds']} "
                  f"calls={run['tool_calls']} tokens={run['generated_tokens']} thinking={thinking} "
                  f"stop={run['stop_reason']} {error or ''}", flush=True)
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
        bucket = summary.setdefault(run["effort"], {"runs": 0, "passed": 0, "seconds": []})
        bucket["runs"] += 1
        bucket["passed"] += int(bool(run.get("passed")))
        bucket["seconds"].append(run.get("seconds") if run.get("status") == "finished" else RUN_SECONDS)
    report["summary"] = summary
    save()
    print(json.dumps({"status": report["status"],
                      "summary": {k: {"passed": v["passed"], "runs": v["runs"]} for k, v in summary.items()}}))
    sys.exit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
