"""Generate calibration inputs from BF16 source weights using native grouped MoE GEMMs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import time
from pathlib import Path

SOURCE_HASHES = {
    "models/lfm2_moe/modeling_lfm2_moe.py": "89a3227654dc699c2aab3fe848f56a037673656ba52b3dbce83041b41d875812",
    "integrations/moe.py": "2c8894f6d1392980a61ff265f90f7a99fa90678e0eec461d958b4d32fda9628c",
}


def source_prompts(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CONVERSATIONS" for t in node.targets)
    ]
    if len(values) != 1:
        raise ValueError("Expected one pinned conversation list")
    conversations = ast.literal_eval(values[0])
    prompts = [row[0] for row in conversations]
    if len(prompts) < 39 or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Insufficient valid source prompts")
    return prompts


def generated_rows(
    prompts, sequences, prefix_width: int, eos_ids: list[int], offset: int, vocabulary: int
):
    if len(prompts) != len(sequences) or prefix_width < 1 or not prompts:
        raise ValueError("Mismatched generated trace batch")
    rows = []
    for index, (prompt, sequence) in enumerate(zip(prompts, sequences, strict=True)):
        if (
            len(prompt) > prefix_width
            or sequence[prefix_width - len(prompt) : prefix_width] != prompt
        ):
            raise ValueError("Generated output changed its input prompt")
        response = sequence[prefix_width:]
        stop = next((i + 1 for i, token in enumerate(response) if token in eos_ids), len(response))
        response = response[:stop]
        if (
            not prompt
            or not response
            or any(type(t) is not int or not 0 <= t < vocabulary for t in [*prompt, *response])
        ):
            raise ValueError("Invalid or empty source trace")
        rows.append(
            {
                "conversation": offset + index,
                "turn": 0,
                "epoch": 0,
                "input_ids": prompt,
                "response_ids": response,
                "eos_reason": "stop_token" if response[-1] in eos_ids else "max_new_tokens",
            }
        )
    return rows


def relative_rms(actual, reference) -> float:
    import torch

    if (
        actual.shape != reference.shape
        or not actual.numel()
        or not torch.isfinite(actual).all().item()
        or not torch.isfinite(reference).all().item()
    ):
        raise ValueError("Invalid expert parity vectors")
    energy = reference.double().square().sum().item()
    if not math.isfinite(energy) or energy <= 1e-20:
        raise ValueError("Empty expert reference energy")
    error = ((actual.double() - reference.double()).square().sum().item() / energy) ** 0.5
    if not math.isfinite(error):
        raise ValueError("Nonfinite expert parity error")
    return error


def run(source: Path, output: Path):
    import importlib.metadata

    import torch
    import transformers
    from safetensors import safe_open
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from transformers.integrations import moe

    package = Path(transformers.__file__).parent
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Unexpected Transformers MoE source")
    report = {
        "status": "running",
        "rows": [],
        "backend": "transformers-bf16",
        "experts_implementation_requested": "grouped_mm",
        "decode_optimization": "Transformers 5.16.1 switches to batched_mm on CUDA",
        "source": str(source),
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")
        },
    }
    started = time.monotonic()
    model = None

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, allow_nan=False) + "\n")
        temporary.replace(output)

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(
            str(source), local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = source_prompts(Path("/opt/exl3-source/sc_trace.py"))
        order = list(range(len(prompts)))
        random.Random(20261015).shuffle(order)
        prompts = [prompts[i] for i in order]
        report["prompt_order"] = order
        token_rows = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=True,
                reasoning_effort="high",
            )
            for p in prompts
        ]
        if any(not row or len(row) > 2048 for row in token_rows):
            raise ValueError("Unexpected source prompt length")
        model, loading = AutoModelForCausalLM.from_pretrained(
            str(source),
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
            experts_implementation="grouped_mm",
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if (
            loading.get("unexpected_keys")
            or loading.get("mismatched_keys")
            or set(loading.get("missing_keys", [])) - {"lm_head.weight"}
        ):
            raise ValueError(f"Source checkpoint did not load completely: {loading}")
        if "lm_head.weight" in loading.get("missing_keys", []) and (
            not model.config.tie_word_embeddings
            or model.get_input_embeddings().weight.data_ptr()
            != model.get_output_embeddings().weight.data_ptr()
        ):
            raise ValueError("Missing source head is not a tied alias")
        model.eval()
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        preserved = 0
        for name in index:
            if not name.endswith("expert_bias"):
                continue
            with safe_open(source / index[name], framework="pt", device="cpu") as handle:
                original = handle.get_tensor(name)
            if (
                original.dtype != torch.float32
                or original.shape != (64,)
                or not torch.isfinite(original).all().item()
            ):
                raise ValueError("Invalid original routing bias")
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, original.cuda())
            if not torch.equal(model.get_buffer(name).cpu(), original):
                raise ValueError("Source routing bias was rounded")
            preserved += 1
        if preserved != 38 or any(
            p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in model.parameters()
        ):
            raise ValueError("Source precision/device contract failed")
        report["preserved_fp32_router_biases"] = preserved
        native_calls = 0
        original_mm = moe._grouped_mm

        def guarded_mm(x, weight, offs):
            nonlocal native_calls
            if not moe._can_use_grouped_mm(x, weight, offs):
                raise RuntimeError("Refusing the slow grouped_mm fallback")
            native_calls += 1
            return original_mm(x, weight, offs)

        moe._grouped_mm = guarded_mm
        expert = next(
            m for m in model.modules() if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")
        )
        torch.manual_seed(20261015)
        with torch.inference_mode():
            x = torch.randn((128, expert.hidden_dim), device="cuda", dtype=torch.bfloat16)
            selected = (
                torch.arange(128, device="cuda")[:, None] + torch.arange(4, device="cuda")
            ) % 64
            affinity = torch.full((128, 4), 0.25, device="cuda", dtype=torch.bfloat16)
            model.set_experts_implementation("eager")
            reference = expert(x, selected, affinity)
            model.set_experts_implementation("grouped_mm")
            actual = expert(x, selected, affinity)
            error = relative_rms(actual, reference)
            if error > 0.02 or native_calls == 0:
                raise ValueError("Grouped BF16 expert preflight diverges")
            report["expert_preflight"] = {
                "shape": list(actual.shape),
                "relative_rms": error,
                "tolerance": 0.02,
                "native_calls": native_calls,
            }
            preflight = model(
                input_ids=torch.tensor([token_rows[0][:32]], device="cuda"), logits_to_keep=0
            ).logits
            if (
                preflight.shape != (1, min(32, len(token_rows[0])), 128000)
                or not torch.isfinite(preflight).all().item()
            ):
                raise ValueError("Invalid full source BF16 logits")
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        if not eos:
            raise ValueError("Missing source EOS IDs")
        report["sampling"] = {
            "seed": 20261015,
            "batch_size": 64,
            "max_new_tokens": 3072,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "suppressed_padding_ids": [len(tokenizer), model.config.vocab_size],
            "minimum_full_rows": 39,
        }
        save()

        class Progress(StoppingCriteria):
            def __init__(self, prefix, batch):
                self.prefix, self.batch = prefix, batch

            def __call__(self, input_ids, scores, **kwargs):
                steps = input_ids.shape[1] - self.prefix
                if time.monotonic() - started > 880:
                    raise TimeoutError("Bounded BF16 trace generation expired")
                if steps % 128 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "bf16_trace",
                                "batch": self.batch,
                                "steps": steps,
                                "max_steps": 3072,
                                "completed_rows": len(report["rows"]),
                            }
                        ),
                        flush=True,
                    )
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for start in range(0, len(token_rows), 64):
            batch = token_rows[start : start + 64]
            inputs = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt").to(
                "cuda"
            )
            width = inputs["input_ids"].shape[1]
            torch.manual_seed(20261015 + start)
            with torch.inference_mode():
                sequences = model.generate(
                    **inputs,
                    max_new_tokens=3072,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    logits_to_keep=1,
                    suppress_tokens=list(range(len(tokenizer), model.config.vocab_size)),
                    stopping_criteria=StoppingCriteriaList([Progress(width, start // 64)]),
                )
            report["rows"] += generated_rows(
                batch, sequences.cpu().tolist(), width, eos, start, len(tokenizer)
            )
            save()
            print(
                json.dumps(
                    {
                        "phase": "trace_batch_complete",
                        "completed_rows": len(report["rows"]),
                        "total_rows": len(token_rows),
                    }
                ),
                flush=True,
            )
            if (
                sum(len(r["input_ids"]) + len(r["response_ids"]) >= 2048 for r in report["rows"])
                >= 39
            ):
                break
        if sum(len(r["input_ids"]) + len(r["response_ids"]) >= 2048 for r in report["rows"]) < 39:
            raise ValueError("Fewer than 39 complete long source traces")
        report.update(
            status="complete",
            native_grouped_mm_calls=native_calls,
            meta={
                "input_tokens": sum(len(r["input_ids"]) for r in report["rows"]),
                "output_tokens": sum(len(r["response_ids"]) for r in report["rows"]),
            },
        )
        save()
        return report
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("source_directory trace.json")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
