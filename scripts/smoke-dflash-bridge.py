"""Compare Desktop's ordinary and DFlash2 greedy paths in one resident bridge.

Usage: python3 scripts/smoke-dflash-bridge.py ENGINE MODEL DRAFT \
    [--tokens 48,128,256] [--repeats 2] [--prompt-file PATH]
"""

import argparse
import json
import queue
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def run(engine, model, draft, budgets, repeats, context_length, prompt, check_prefix_cache=False, check_mcp=False):
    config = tempfile.TemporaryDirectory(prefix="mlxl3-bridge-check-")
    registry = Path(config.name) / "models.json"
    (Path(config.name) / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
        "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
        "fixture": {"enabled": True, "command": sys.executable,
                    "args": [str(Path(__file__).resolve().parents[1] / "tests/fake-desktop-engine.py"), "mcp-fixture"]},
    }}))
    process = subprocess.Popen(
        [engine, "--registry", str(registry), "bridge", model, "--context-length", str(context_length)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    events = queue.Queue()

    def read():
        try:
            for line in process.stdout:
                events.put(json.loads(line))
        except (ValueError, OSError) as error:
            events.put({"type": "error", "message": str(error)})
        finally:
            events.put({"type": "error", "message": "bridge exited"})

    threading.Thread(target=read, daemon=True).start()

    def until(kind, request_id=None, cancel_at=None):
        mode = None
        first_text_at = None
        cancellation_sent = False
        while True:
            event = events.get(timeout=300)
            if (event["type"] == "cancelled" and cancellation_sent and
                event.get("request_id") == request_id):
                return event, mode
            if event["type"] == "error":
                raise RuntimeError(event)
            if request_id is not None and event.get("request_id") != request_id:
                continue
            if event["type"] == "generation_mode":
                mode = event
            if event["type"] == cancel_at and not cancellation_sent:
                process.send_signal(signal.SIGUSR1)
                cancellation_sent = True
            if event["type"] == "delta" and event.get("text") and first_text_at is None:
                first_text_at = time.perf_counter()
            if event["type"] == kind:
                event["first_text_received_at"] = first_text_at
                return event, mode

    counter = 0

    def generate(enabled, budget, *, cancel_at=None, **options):
        nonlocal counter
        counter += 1
        request_id = str(counter)
        request = {
            "type": "generate", "request_id": request_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": budget, "temperature": 0, "top_k": 1,
            "repetition_penalty": 1, "dflash2": enabled,
            "dflash_draft_path": draft,
            **options,
        }
        submitted = time.perf_counter()
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        result, mode = until("complete", request_id, cancel_at)
        if mode is None or mode.get("dflash_active") is not enabled:
            raise RuntimeError(f"engine mode mismatch: {mode}")
        if cancel_at:
            if result["type"] != "cancelled":
                raise RuntimeError("generation completed before cancellation was tested")
            return result
        result["stats"]["bridge_client_total_seconds"] = time.perf_counter() - submitted
        first = result.pop("first_text_received_at")
        result["stats"]["bridge_client_ttft_seconds"] = first - submitted if first else None
        return result

    try:
        ready, _ = until("ready")
        warmup = [generate(False, budgets[0]), generate(True, budgets[0])]
        print(json.dumps({"warmup": [result["stats"] for result in warmup], "runtime": ready}), flush=True)
        for budget in ([] if check_prefix_cache or check_mcp else budgets):
            runs = {False: [], True: []}
            for index in range(repeats):
                for enabled in ([False, True] if index % 2 == 0 else [True, False]):
                    runs[enabled].append(generate(enabled, budget))
            plain, speculative = runs[False], runs[True]
            hashes = {result.get("token_hash") for result in plain + speculative}
            outputs = {result["cache_context"] for result in plain + speculative}
            counts = {result["stats"]["generated_tokens"] for result in plain + speculative}
            parity = (None not in hashes and len(hashes) == len(outputs) == len(counts) == 1)
            comparable = parity and next(iter(counts)) == budget
            print(json.dumps({
                "budget": budget,
                "comparable": comparable,
                "early_eos": next(iter(counts)) < budget if len(counts) == 1 else None,
                "runtime": {key: ready.get(key) for key in
                            ("runtime_commit", "runtime_profile", "mlx_version", "runtime_executable")},
                "model": model,
                "draft": draft,
                "context_length": context_length,
                "prompt": prompt,
                "plain": [item["stats"] for item in plain],
                "dflash2": [item["stats"] for item in speculative],
                "token_hashes": sorted(hashes, key=str),
                "decode_ratio": (statistics.median(item["stats"]["decode_tps"] for item in speculative)
                                 / statistics.median(item["stats"]["decode_tps"] for item in plain)
                                 if comparable and budget > 1 else None),
            }), flush=True)
            if not parity:
                raise RuntimeError("greedy / DFlash token or text parity failed (or missing token hash)")
        if check_prefix_cache:
            base = [
                {"role": "system", "content": "You are a helpful assistant. " +
                 "Reference notes: KV cache stores keys and values for earlier tokens. " * 40},
                {"role": "user", "content": "Say hello briefly."},
            ]
            dialogue = base + [{"role": "assistant", "content": "Hello!"},
                               {"role": "user", "content": "Explain how this cache reduces the latency of a second chat turn."}]
            for enabled in [False, True]:
                # Compile the larger prefill and the cached tail before comparing
                # their steady-state latency (the short decode warmup isn't enough).
                generate(enabled, 1, messages=dialogue, reuse_prompt_cache=False)
                generate(enabled, 1, messages=base, conversation_id="cache-a")
                generate(enabled, 1, messages=dialogue, conversation_id="cache-a")
                results = {False: [], True: []}
                for reuse in [False, True, True, False]:
                    if reuse:
                        generate(enabled, 1, messages=base, conversation_id="cache-a")
                    result = generate(enabled, budgets[0], messages=dialogue,
                                      conversation_id="cache-a", reuse_prompt_cache=reuse)
                    results[reuse].append(result)
                    cached = result["stats"]["cached_prompt_tokens"]
                    if (cached > 0) != reuse:
                        raise RuntimeError(f"wrong prefix cache state: enabled={reuse}, cached={cached}")
                all_results = results[False] + results[True]
                if (None in {item.get("token_hash") for item in all_results} or
                    len({item.get("token_hash") for item in all_results}) != 1 or
                    len({item["cache_context"] for item in all_results}) != 1):
                    raise RuntimeError("cache changed generated tokens or text")
                print(json.dumps({"prefix_cache": True, "dflash2": enabled,
                                  "prompt_messages": dialogue,
                                  "runtime": ready,
                                  "off": [item["stats"] for item in results[False]],
                                  "on": [item["stats"] for item in results[True]],
                                  "token_hash": all_results[0]["token_hash"]}), flush=True)
                for phase in ["context_usage", "delta"]:
                    generate(enabled, 1024, messages=dialogue, conversation_id="cache-a", cancel_at=phase)
                    resumed = generate(enabled, budgets[0], messages=dialogue, conversation_id="cache-a")
                    if (resumed["token_hash"] != all_results[0]["token_hash"] or
                        resumed["cache_context"] != all_results[0]["cache_context"]):
                        raise RuntimeError(f"prefix state changed after cancellation at {phase}")
                print(json.dumps({"prefix_cancellation": True, "dflash2": enabled,
                                  "phases": ["context_usage", "delta"]}), flush=True)
                # An edited system prompt and another conversation must both miss.
                edited = [{"role": "system", "content": "Changed system instruction."}] + dialogue[1:]
                result = generate(enabled, 1, messages=edited, conversation_id="cache-a")
                if result["stats"]["cached_prompt_tokens"] != 0:
                    raise RuntimeError("edited system reused stale state")
                generate(enabled, 1, messages=base, conversation_id="cache-a")
                result = generate(enabled, 1, messages=base, conversation_id="cache-b")
                if result["stats"]["cached_prompt_tokens"] != 0:
                    raise RuntimeError("conversation switch reused another conversation")
        if check_mcp:
            messages = [{"role": "system", "content": "You are a helpful assistant. " +
                         "Reference notes: this is a local test, never make external requests. " * 40},
                        {"role": "user", "content": "Call fixture.echo exactly once with message='cache-check'. Then repeat its result and finish. Do not guess the result."}]
            for enabled in [False, True]:
                results = []
                for reuse in [False, True]:
                    result = generate(enabled, 256, messages=messages, mcp_enabled=True,
                                      conversation_id=f"mcp-{enabled}-{reuse}", reuse_prompt_cache=reuse)
                    rounds = result["round_stats"]
                    stats = result["stats"]
                    if len(rounds) != 2 or stats["tool_rounds"] != 1:
                        raise RuntimeError(f"expected exactly one fixture tool call: {result}")
                    if stats["generated_tokens"] != sum(item["generated_tokens"] for item in rounds):
                        raise RuntimeError("MCP generation statistics did not aggregate both rounds")
                    if stats["tool_seconds"] < 0.19:
                        raise RuntimeError("MCP duration omitted the fixture's 200 ms delay")
                    if (rounds[1]["cached_prompt_tokens"] > 0) != reuse:
                        raise RuntimeError("MCP round did not obey the prefix cache mode")
                    results.append(result)
                if (results[0]["cache_context"] != results[1]["cache_context"] or
                    results[0]["token_hash"] != results[1]["token_hash"]):
                    raise RuntimeError("MCP prefix cache changed the final response")
                print(json.dumps({"mcp_check": True, "dflash2": enabled,
                                  "off": results[0]["stats"], "on": results[1]["stats"],
                                  "rounds_off": results[0]["round_stats"], "rounds_on": results[1]["round_stats"]}), flush=True)
        process.stdin.write('{"type":"shutdown"}\n')
        process.stdin.flush()
        if process.wait(timeout=10) != 0:
            raise RuntimeError("bridge exited with an error")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        config.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("draft")
    parser.add_argument("--tokens", default="48,128,256")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--check-prefix-cache", action="store_true",
                        help="Run counterbalanced cache OFF/ON multi-turn checks instead of the decode matrix")
    parser.add_argument("--check-mcp", action="store_true", help="Check real MCP rounds against an isolated local echo fixture")
    args = parser.parse_args()
    budgets = [int(value) for value in args.tokens.split(",")]
    if not budgets or any(value < 1 for value in budgets) or args.repeats < 1:
        parser.error("tokens and repeats must be positive")
    prompt = (args.prompt_file.read_text() if args.prompt_file else
              "Explain lossless speculative decoding in one paragraph.")
    run(args.engine, args.model, args.draft, budgets, args.repeats, args.context_length, prompt, args.check_prefix_cache, args.check_mcp)
