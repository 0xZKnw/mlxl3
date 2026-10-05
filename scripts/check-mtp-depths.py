"""Production bridge regressions for MTP1/2/3, budgets, prefix, cancellation and tuning.

Requires local checkpoints and an Apple GPU. Timings are diagnostics, not an
isolated performance campaign. Never downloads weights or changes a registry.
"""
import argparse
import json
import queue
import signal
import subprocess
import tempfile
import threading
from pathlib import Path


def run(engine, model, head, tune=False):
    with tempfile.TemporaryDirectory(prefix="mtp-depth-check-") as directory:
        process = subprocess.Popen([engine, "--registry", str(Path(directory)/"models.json"),
                                    "bridge", model, "--context-length", "4096"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        events = queue.Queue()
        def read():
            try:
                for line in process.stdout:
                    event = json.loads(line)
                    print(json.dumps(event), flush=True)
                    events.put(event)
            finally:
                events.put({"type": "error", "message": "bridge exited"})
        threading.Thread(target=read, daemon=True).start()
        def until(kind, request=None, cancel_at=None, error=False):
            cancelled = False
            while True:
                event = events.get(timeout=60)
                if request and event.get("request_id") not in (None, request):
                    continue
                if event["type"] == "error":
                    if error:
                        return event
                    raise RuntimeError(event)
                if cancel_at == event["type"] and not cancelled:
                    process.send_signal(signal.SIGUSR1)
                    cancelled = True
                if event["type"] == kind:
                    return event
        count = 0
        def request(kind="generate", depth=0, budget=17, cancel_at=None, error=False, **options):
            nonlocal count
            count += 1
            data = dict(type=kind, request_id=str(count), conversation_id="depth-check",
                        messages=[dict(role="user", content="Write complete Python code to multiply two matrices with comments and examples.")],
                        max_tokens=budget, temperature=0, top_k=1, repetition_penalty=1,
                        mtp=depth > 0, mtp_depth=depth, mtp_head_path=head,
                        reuse_prompt_cache=False)
            data.update(options)
            process.stdin.write(json.dumps(data)+"\n")
            process.stdin.flush()
            return until("cancelled" if cancel_at else "mtp_tune_complete" if kind == "tune_mtp" else "complete",
                         str(count), cancel_at, error)
        def equal(a, b):
            assert a["token_hash"] == b["token_hash"], "target IDs changed"
            assert a["cache_context"] == b["cache_context"], "history changed"
            assert a["stats"]["generated_tokens"] == b["stats"]["generated_tokens"], "budget changed"
        try:
            ready = until("ready")
            assert ready["mtp_max_depth"] == 3 and ready["mtp_tune_supported"]
            for budget in (1, 2, 3, 4, 17, 64):
                baseline = request(budget=budget)
                for depth in (1, 2, 3):
                    actual = request(depth=depth, budget=budget)
                    equal(baseline, actual)
                    assert actual["stats"]["generated_tokens"] <= budget
            for depth in (0, 4):
                assert request(depth=depth, mtp=True, error=True)["type"] == "error"
            golden = request(budget=17)
            for depth in (2, 3):
                for phase in ("context_usage", "delta"):
                    request(depth=depth, budget=256, cancel_at=phase)
                    equal(golden, request(depth=depth))
            # Same exact prefill cache can safely be reused across draft depths.
            long_prompt = "Explain this table in detail.\n" + "1 2 3 4 5 6 7 8 9 10\n"*18
            messages = [dict(role="user", content=long_prompt)]
            cold = request(depth=2, messages=messages)
            request(depth=2, messages=messages, reuse_prompt_cache=True)
            warm = request(depth=3, messages=messages, reuse_prompt_cache=True)
            equal(cold, warm)
            assert warm["stats"]["cached_prompt_tokens"] >= 256
            miss = request(depth=3, messages=messages, reuse_prompt_cache=True, conversation_id="other")
            equal(cold, miss)
            assert miss["stats"]["cached_prompt_tokens"] == 0
            if tune:
                request(kind="tune_mtp", cancel_at="mtp_tune_progress")
                equal(golden, request(depth=3))
                result = request(kind="tune_mtp")
                assert result["tuning_key"] == ready["mtp_tuning_key"]
                rows = result["rows"]
                assert [row["depth"] for row in rows] == [0, 1, 2, 3]
                assert all(row["token_hashes"] == rows[0]["token_hashes"] for row in rows)
                scores = [int(row["decode_tps"]*1000) if row["eligible"] else None for row in rows]
                best = 0
                for depth in range(1, 4):
                    if scores[depth] is not None and scores[depth] > scores[best] and scores[depth]*100 > scores[0]*103:
                        assert rows[depth]["accepted_tokens"] > 0
                        best = depth
                assert result["best_depth"] == best
                equal(golden, request(depth=best))
            process.stdin.write('{"type":"shutdown"}\n'); process.stdin.flush()
            assert process.wait(timeout=10) == 0
            print(json.dumps({"all_depths_parity": True, "prefix_parity": True,
                              "cancel_recovery": True, "tuner_verified": tune}), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("head")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    run(args.engine, args.model, args.head, args.tune)
