"""Compare ordinary and DFlash2 greedy output through the Desktop bridge.

Usage: python3 scripts/smoke-dflash-bridge.py ENGINE MODEL DRAFT
"""

import json
import queue
import subprocess
import sys
import threading


def run(engine, model, draft):
    process = subprocess.Popen(
        [engine, "bridge", model, "--context-length", "4096"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    events = queue.Queue()

    def read():
        for line in process.stdout:
            events.put(json.loads(line))
        events.put({"type": "error", "message": "bridge exited"})

    threading.Thread(target=read, daemon=True).start()

    def until(kind):
        while True:
            event = events.get(timeout=180)
            if event["type"] == "error":
                raise RuntimeError(event)
            if event["type"] == kind:
                return event

    def generate(enabled):
        request = {
            "type": "generate", "request_id": str(enabled),
            "messages": [{"role": "user", "content":
                          "Explain lossless speculative decoding in one paragraph."}],
            "max_tokens": 48, "temperature": 0, "top_k": 1,
            "repetition_penalty": 1, "dflash2": enabled,
            "dflash_draft_path": draft,
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        return until("complete")

    try:
        until("ready")
        plain = generate(False)
        speculative = generate(True)
        warmed = generate(True)
        for result in (speculative, warmed):
            assert plain["cache_context"] == result["cache_context"], (
                plain["cache_context"], result["cache_context"]
            )
            assert plain["stats"]["generated_tokens"] == result["stats"]["generated_tokens"]
        print(json.dumps({
            "equal": True,
            "plain_tps": plain["stats"]["decode_tps"],
            "dflash2_tps": speculative["stats"]["decode_tps"],
            "dflash2_warmed_tps": warmed["stats"]["decode_tps"],
            "tokens": plain["stats"]["generated_tokens"],
        }))
        process.stdin.write('{"type":"shutdown"}\n')
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    run(*sys.argv[1:])
