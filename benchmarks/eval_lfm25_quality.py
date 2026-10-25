"""Compare held-out BF16 and native EXL3 logits without loading both models on GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from native_json_process import JsonProcess, positive_seconds


def window_count(tokens: int, window: int) -> int:
    """Reject an empty or incomplete quality evaluation.

    raises: ValueError
    post: __return__ > 0 and __return__ * window == tokens
    """
    if tokens <= 0 or window <= 0 or tokens % window:
        raise ValueError("tokens must be a positive multiple of window")
    return tokens // window


def log_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("expected a nonempty finite logits vector")
    values = values - values.max()
    return values - np.log(np.exp(values).sum())


def losses(logits, teacher, target):
    actual, reference = log_probabilities(logits), log_probabilities(teacher)
    if actual.shape != reference.shape or type(target) is not int or not 0 <= target < len(actual):
        raise ValueError("invalid label or mismatched vocabulary")
    return (
        float(-actual[target]),
        float(-reference[target]),
        float(np.sum(np.exp(reference) * (reference - actual))),
    )


def native_logits(row, token, vocabulary):
    if not isinstance(row, dict) or type(row.get("token")) is not int or row["token"] != token:
        raise ValueError("native forward returned the wrong imposed token")
    bits = row.get("logits")
    if (
        not isinstance(bits, list)
        or len(bits) != vocabulary
        or not bits
        or not all(type(value) is int and 0 <= value <= 65535 for value in bits)
    ):
        raise ValueError("native forward returned invalid logits bits or shape")
    values = np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("native forward returned nonfinite logits")
    return values


def scored_count(total: int, first: int, last: int) -> int:
    """Count an inclusive scoring range inside a nonempty imposed window.

    raises: ValueError
    post: 0 < __return__ <= total
    post: first + __return__ - 1 < total
    """
    if total <= 0 or first < 0 or last < first or last >= total:
        raise ValueError("invalid scoring range")
    return last - first + 1


def native_window(
    binary, model, tokens, labels, teacher, stderr, timeout, *, score_first=0, score_last=None
):
    if (
        teacher.ndim != 2
        or teacher.shape[0] != len(tokens)
        or len(labels) != len(tokens)
        or not tokens
        or teacher.shape[1] == 0
        or not np.issubdtype(teacher.dtype, np.floating)
        or not np.isfinite(teacher).all()
        or any(type(t) is not int or not 0 <= t < teacher.shape[1] for t in [*tokens, *labels])
    ):
        raise ValueError("invalid native quality window")
    if score_last is None:
        score_last = len(tokens) - 1
    if type(score_first) is not int or type(score_last) is not int:
        raise ValueError("score bounds must be integers")
    count = scored_count(len(tokens), score_first, score_last)
    command = [str(binary), "forward", str(model), "--tokens", ",".join(map(str, tokens))]
    deadline = time.monotonic() + timeout
    total = np.zeros(3, dtype=np.float64)
    matches = 0
    with stderr.open("w") as errors, JsonProcess(command, stderr=errors) as process:
        for index, token in enumerate(tokens):
            values = native_logits(process.receive(deadline), token, teacher.shape[1])
            if score_first <= index <= score_last:
                total += losses(values, teacher[index], labels[index])
                matches += int(np.argmax(values) == np.argmax(teacher[index]))
        if process.process.wait(timeout=min(5, timeout)) != 0:
            raise RuntimeError("native forward exited with an error")
        remaining = bytes(process.buffer) + (process.process.stdout.read() or b"")
        if remaining.strip():
            raise RuntimeError("native forward emitted more rows than requested")
    return {
        "tokens": count,
        "nll": total[0] / count,
        "bf16_nll": total[1] / count,
        "kl": total[2] / count,
        "greedy_matches": matches,
    }


def teacher_windows(source, data, tokens, window, destination):
    import mlx.core as mx
    from ponyexl3.convert.fixtures import SafetensorIndex
    from quantize_lfm25_moe import capture_lfm_moe

    count = window_count(tokens, window)
    vocabulary = json.loads((source / "config.json").read_text())["vocab_size"]
    with tempfile.TemporaryDirectory(prefix="mlxl3-lfm25-teacher-") as directory:
        temporary = Path(directory)
        hidden_path = temporary / "hidden.npy"
        capture_lfm_moe(
            source,
            ["model.layers.0.conv.in_proj"],
            temporary / "capture",
            text_path=data,
            rows=tokens,
            seq_len=window,
            max_seqs=count,
            hidden_output=hidden_path,
        )
        hidden = np.load(hidden_path, allow_pickle=False)
    if hidden.ndim != 3 or hidden.shape[:2] != (count, window) or not np.isfinite(hidden).all():
        raise ValueError("invalid layerwise BF16 teacher hidden states")
    weight = mx.array(SafetensorIndex(source).read_tensor("model.embed_tokens.weight")).astype(
        mx.bfloat16
    )
    for number, sequence in enumerate(hidden):
        logits = mx.array(sequence).astype(mx.bfloat16) @ weight.T
        values = np.array(logits.astype(mx.float32))
        if values.shape != (window, vocabulary) or not np.isfinite(values).all():
            raise ValueError("invalid BF16 teacher logits")
        np.save(destination / f"teacher-{number:02d}.npy", values)
        del logits, values
        mx.clear_cache()
    del weight
    mx.clear_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("quantized", type=Path)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--timeout", type=positive_seconds, default=600.0)
    args = parser.parse_args(argv)
    count = window_count(args.tokens, args.window)
    if args.output.exists() or (args.work_dir.exists() and any(args.work_dir.iterdir())):
        parser.error("output/work-dir must be new; preserve previous attempts")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "tokens": args.tokens,
        "window": args.window,
        "windows": [],
        "execution": "layerwise BF16 teacher; native single-token forward, reset each window",
        "source": str(args.source),
        "model": str(args.quantized),
        "native_sha256": hashlib.sha256(args.native_binary.read_bytes()).hexdigest(),
    }
    started = time.monotonic()
    try:
        from transformers import PreTrainedTokenizerFast

        corpus = args.data.read_bytes()
        report["dataset_sha256"] = hashlib.sha256(corpus).hexdigest()
        ids = PreTrainedTokenizerFast.from_pretrained(args.source).encode(corpus.decode())[
            : args.tokens + 1
        ]
        vocabulary = json.loads((args.source / "config.json").read_text())["vocab_size"]
        if len(ids) != args.tokens + 1 or not all(0 <= token < vocabulary for token in ids):
            raise ValueError("evaluation corpus is too short or has invalid token IDs")
        report["token_sha256"] = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
        report["stage"] = "BF16 teacher"
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        teacher_windows(args.source, args.data, args.tokens, args.window, args.work_dir)
        import mlx.core as mx

        mx.clear_cache()
        report["teacher_mlx_active_bytes_after_clear"] = mx.get_active_memory()
        for number in range(count):
            report["stage"] = f"native window {number + 1}/{count}"
            start = number * args.window
            teacher = np.load(
                args.work_dir / f"teacher-{number:02d}.npy", mmap_mode="r", allow_pickle=False
            )
            result = native_window(
                args.native_binary,
                args.quantized,
                ids[start : start + args.window],
                ids[start + 1 : start + args.window + 1],
                teacher,
                args.work_dir / f"native-{number:02d}.stderr",
                args.timeout,
            )
            report["windows"].append(result)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"window": number + 1, **result}), flush=True)
        if sum(item["tokens"] for item in report["windows"]) != args.tokens:
            raise ValueError("incomplete quality evaluation")
        for label, field in [("native", "nll"), ("bf16", "bf16_nll")]:
            nll = sum(item[field] * item["tokens"] for item in report["windows"]) / args.tokens
            report[label] = {"mean_nll": nll, "perplexity": math.exp(nll)}
        report["mean_kl_bf16_to_native"] = (
            sum(item["kl"] * item["tokens"] for item in report["windows"]) / args.tokens
        )
        report["greedy_match_fraction"] = (
            sum(item["greedy_matches"] for item in report["windows"]) / args.tokens
        )
        report["ppl_change_percent"] = (
            report["native"]["perplexity"] / report["bf16"]["perplexity"] - 1
        ) * 100
        report.update(status="complete", stage="done")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
