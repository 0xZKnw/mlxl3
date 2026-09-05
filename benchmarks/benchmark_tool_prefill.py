"""Bounded post-tool R&D: real Exa schema/result, replayed without network noise.

Capture public evidence once with --capture; all timed runs are offline. No
user conversations are read. JSONL records timings, prompt hashes and tokens.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import sys
from pathlib import Path
import time

import mlx.core as mx

import mlxl3.cli as cli
from benchmark_audit_runtime import manifest, command


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def capture(path):
    from mlxl3.mcp import MCPManager, MCPServerConfig
    manager = MCPManager([MCPServerConfig(name="exa", url="https://mcp.exa.ai/mcp")])
    try:
        manager.connect()
        arguments = {"query": "site.ml-explore.github.io mlx prompt cache memory lazy evaluation", "numResults": 3}
        started = time.perf_counter()
        result = manager.call("exa.web_search_exa", arguments)
        if result.is_error:
            raise RuntimeError(result.text)
        fixture = {"tools": manager.chat_tools, "result": result.text, "arguments": arguments,
                   "network_seconds": time.perf_counter() - started}
        path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
        emit({"captured": str(path), "result_characters": len(result.text),
              "network_seconds": fixture["network_seconds"]})
    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))
    parser.add_argument("--fixture", type=Path, default=Path("benchmarks/data/exa_prefill_public.json"))
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--real-loop", action="store_true")
    parser.add_argument("--runs", default="1:baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, default=Path("benchmarks/results/tool_prefill_rd_01.jsonl"))
    args = parser.parse_args()
    if args.capture:
        capture(args.fixture)
        return
    if args.output:
        sys.stdout = args.output.open("a", buffering=1)
    fixture = json.loads(args.fixture.read_text())
    emit(manifest(args.model))
    model, tokenizer, *_ = cli._load_model(args.model)
    emit({"kind": "loaded", "active_gb": mx.get_active_memory() / 1e9})
    initial = [{"role": "user", "content": "Utilise exa.web_search_exa pour chercher la documentation officielle du cache MLX. Résume les résultats en français."}]
    if args.real_loop:
        original_stream = cli._stream_response
        def traced_stream(*a, **kw):
            response, stats = original_stream(*a, **kw)
            emit({"kind": "round", "messages": a[2], "response": response, **asdict(stats)})
            return response, stats
        cli._stream_response = traced_stream
        from types import SimpleNamespace
        class ReplayMCP:
            chat_tools = fixture["tools"]
            tools = {t["function"]["name"]: SimpleNamespace(server="exa") for t in chat_tools}
            def call(self, name, arguments):
                emit({"kind": "replay_tool", "name": name, "arguments": arguments})
                return SimpleNamespace(text=fixture["result"], is_error=False)
        session = cli.GenerationSession()
        cli._bridge_generate(model, tokenizer, initial, request_id="rd-real-1",
            max_tokens=768, temperature=0, top_k=0, repetition_penalty=1,
            mcp=ReplayMCP(), session=session)
        return
    import numpy as np
    import mlx_lm.sample_utils as sample_utils
    from mlx_lm.models.cache import make_prompt_cache
    import mlxl3.moe as moe
    import mlxl3.linear as linear
    import mlxl3.kernels.qmv as qmv
    # Keep snapshot construction pinned to the pre-promotion reference too.
    moe._USE_PREFILL_GATHER_SIMD = False

    rounds = [json.loads(line) for line in args.trace.read_text().splitlines() if json.loads(line).get("kind") == "round"]
    messages = rounds[1]["messages"]
    prefix_text = cli._render_generation_prompt(tokenizer, initial, fixture["tools"]) + rounds[0]["response"]
    prefix = list(tokenizer.encode(prefix_text, add_special_tokens=False)) + [tokenizer.eos_token_id]
    full_prompt = cli._render_generation_prompt(tokenizer, messages, fixture["tools"])
    full = tokenizer.encode(full_prompt, add_special_tokens=False)
    assert full[:len(prefix)] == prefix

    # All variants start from the very same evaluated snapshot. Its construction
    # is excluded, like an already-completed tool-call generation in the app.
    base_cache = make_prompt_cache(model)
    model(mx.array(prefix)[None], cache=base_cache)
    mx.eval([c.state for c in base_cache])
    class Probe:
        last_logits = None
        def __getattr__(self, name):
            return getattr(model, name)
        def __call__(self, *a, **kw):
            out = model(*a, **kw)
            self.last_logits = out[:, -1, :]
            return out
    probe = Probe()

    def digest(value):
        if isinstance(value, mx.array):
            raw = np.asarray(mx.contiguous(value).view(mx.uint8)).tobytes()
            return (str(value.dtype), tuple(value.shape), hashlib.sha256(raw).hexdigest())
        if isinstance(value, (list, tuple)):
            return [digest(v) for v in value]
        if isinstance(value, dict):
            return {str(k): digest(v) for k, v in value.items()}
        return value

    original_sampler = sample_utils.make_sampler
    reference = None
    forced = None
    def run(number, variant, warmup=False):
        nonlocal reference, forced
        cli._PREFILL_STEP_SIZE = {"chunk512":512,"chunk1024":1024,"chunk4096":4096}.get(variant, 0)
        cli._PREFIX_CACHE_BLOCK_SIZE = 0 if variant == "chunk4096" else 2048
        cli._COMPACT_CONV_STATES = variant == "compact"
        cli._RETAIN_PREFILL_ALLOCATOR = variant in ("retain", "combined")
        moe._USE_PREFILL_GATHER_HADAMARD = variant in ("gather", "combined")
        moe._USE_PREFILL_GATHER_SIMD = variant in ("simd", "simd_locality")
        qmv._USE_SEGMENTED_LOCALITY = variant in ("locality", "simd_locality")
        linear._USE_STRIDED_PREFILL = variant in ("strided", "combined")
        qmv._SEGMENTED_TENSOR_ROWS = {"bm8":8,"bm16":16}.get(variant, 32)
        qmv._USE_SEGMENTED_BUCKETS = variant == "buckets"
        sampled = []
        def factory(**kw):
            sampler = original_sampler(**kw)
            def sample(logits):
                value = sampler(logits) if forced is None else mx.array([forced[len(sampled)]], dtype=mx.uint32)
                sampled.append(value)
                return value
            return sample
        sample_utils.make_sampler = factory
        session = cli.GenerationSession()
        session.exact_cache = cli.GenerationSession._clone_cache(base_cache)
        session.exact_tokens = list(prefix)
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        emit({"kind":"start", "experiment":number, "variant":variant, "warmup":warmup,
              "power":command("pmset", "-g", "batt"), "thermal":command("pmset", "-g", "therm")})
        text, stats = cli._stream_response(probe, tokenizer, messages, max_tokens=16,
            temperature=0, top_k=0, repetition_penalty=1, on_text=lambda _:None,
            tools=fixture["tools"], session=session)
        mx.eval(probe.last_logits)
        state = digest([(c.state, c.meta_state) for c in session.exact_cache])
        logits = digest(probe.last_logits)
        mx.eval(sampled)
        sampled = [int(v.item()) for v in sampled]
        if reference is None:
            reference = (state, logits)
            forced = sampled
        emit({"kind":"result", "experiment":number, "variant":variant, "warmup":warmup,
              **asdict(stats), "state_exact":state == reference[0], "logits_exact":logits == reference[1],
              "forced_tokens":sampled, "logits_digest":logits,
              "prompt_sha256":hashlib.sha256(json.dumps(full).encode()).hexdigest(),
              "text_sha256":hashlib.sha256(text.encode()).hexdigest(),
              "active_gb":mx.get_active_memory()/1e9, "allocator_gb":mx.get_cache_memory()/1e9})
        session.reset()
        probe.last_logits = None
        gc.collect()
        mx.clear_cache()
    run(0, "baseline", warmup=True)
    for spec in args.runs.split(","):
        number, variant = spec.split(":")
        run(int(number), variant)
    sample_utils.make_sampler = original_sampler


if __name__ == "__main__":
    main()
