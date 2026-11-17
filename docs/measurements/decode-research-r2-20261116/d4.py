#!/usr/bin/env python3
"""D4 LOGIT-GAP-20261116: two-stage certified screening on real LFM3 logits (CPU analysis).

`collect` replays prompt + greedy output through `mlxl3 forward --tokens`
and stores the F16 logits of the generated positions. `analyze` rebuilds the
final hidden state by least squares from the real F16 head, then counts the
rows a certified screen would have to recompute exactly for several proxy
precisions, with the delivered one-stage threshold and a two-stage one.
No kernel is exercised and nothing here is a speed measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TEMPLATE = "<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>"


def collect(args):
    spec = importlib.util.spec_from_file_location("d2", args.d2_script)
    d2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d2)
    report = json.loads(Path(args.d2).read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    rows = {}
    for run in report["runs"]:
        if run["label"] != args.label:
            continue
        for row in run["rows"]:
            if row["arm"] == "G" and row["case"] not in rows:
                rows[row["case"]] = row
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "cases": []}
    try:
        for name in args.cases.split(","):
            row = rows[name]
            prompt = tokenizer.encode(
                TEMPLATE.format(prompt=d2.CASES[name]), add_special_tokens=False
            ).ids
            output = tokenizer.encode(row["cache_context"], add_special_tokens=False).ids
            output = output[: args.positions]
            ids = prompt + output
            process = subprocess.Popen(
                [args.engine, "forward", "--tokens", ",".join(map(str, ids)), args.model],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            kept = []
            try:
                for index, line in enumerate(process.stdout):
                    # Row `index` holds the logits predicting ids[index + 1].
                    if index < len(prompt) - 1 or index >= len(ids) - 1:
                        continue
                    value = json.loads(line)
                    if value["token"] != ids[index]:
                        raise RuntimeError("forward echoed an unexpected token")
                    logits = np.asarray(value["logits"], dtype=np.uint16)
                    if logits.shape != (args.vocab,):
                        raise RuntimeError(f"unexpected logits shape {logits.shape}")
                    kept.append(logits)
                code = process.wait(timeout=args.timeout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if code != 0 or len(kept) != len(output):
                raise RuntimeError(f"forward failed for {name}: exit {code}, rows {len(kept)}")
            bits = np.stack(kept)
            targets = np.asarray(output, dtype=np.int64)
            agree = float(np.mean(bits.view(np.float16).astype(np.float32).argmax(1) == targets))
            np.savez_compressed(out / f"{name}.npz", bits=bits, targets=targets)
            manifest["cases"].append(
                {
                    "name": name,
                    "prompt_tokens": len(prompt),
                    "positions": len(output),
                    "greedy_agreement_with_emitted_text": agree,
                }
            )
            print(json.dumps(manifest["cases"][-1]), flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        (out / "collect.json").write_text(json.dumps(manifest, indent=2))


def load_head(model):
    path = Path(model) / "model.safetensors"
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    entry = header["lm_head.weight"]
    if entry["dtype"] != "F16":
        raise SystemExit("expected an F16 head")
    start, end = entry["data_offsets"]
    raw = np.fromfile(path, dtype=np.float16, count=(end - start) // 2, offset=8 + size + start)
    return raw.reshape(entry["shape"]).astype(np.float32)


def proxy_spec(name):
    """`intB_row` (one scale per row) or `intB_gG` (one F16 scale per G weights)."""
    bits, layout = name[3:].split("_")
    return int(bits), None if layout == "row" else int(layout[1:])


def quantized_error(weights, name):
    bits, group = proxy_spec(name)
    levels = 2 ** (bits - 1) - 1
    if group is None:
        scale = np.abs(weights).max(1, keepdims=True) / levels
        scale[scale == 0] = 1
        return weights - np.round(weights / scale) * scale
    groups = weights.reshape(weights.shape[0], -1, group)
    scale = np.abs(groups).max(2, keepdims=True) / levels
    scale[scale == 0] = 1
    return (groups - np.round(groups / scale) * scale).reshape(weights.shape)


def proxy_bytes(name, width):
    """Bytes per row: packed codes plus two bytes per group scale."""
    bits, group = proxy_spec(name)
    return width * bits // 8 + (0 if group is None else 2 * width // group)


def analyze(args):
    out = Path(args.out)
    manifest = json.loads((out / "collect.json").read_text())
    if manifest["status"] != "complete":
        raise SystemExit("collection is not complete")
    weights = load_head(args.model)
    rows, width = weights.shape
    logits, names = [], []
    for case in manifest["cases"]:
        data = np.load(out / f"{case['name']}.npz")
        values = data["bits"].view(np.float16).astype(np.float32)
        logits.append(values)
        names += [case["name"]] * len(values)
    logits = np.concatenate(logits)  # [P, V]
    names = np.asarray(names)
    # Least squares with an intercept: logits ~ W x + b.
    gram = np.zeros((width + 1, width + 1), dtype=np.float64)
    gram[:width, :width] = weights.T.astype(np.float64) @ weights.astype(np.float64)
    column = weights.sum(0, dtype=np.float64)
    gram[:width, width] = gram[width, :width] = column
    gram[width, width] = rows
    rhs = np.concatenate(
        [logits.astype(np.float64) @ weights.astype(np.float64), logits.sum(1, keepdims=True)], 1
    )
    solution = np.linalg.solve(gram, rhs.T).T  # [P, width + 1]
    states = solution[:, :width].astype(np.float32)
    intercept = solution[:, width]
    rebuilt = states @ weights.T + intercept[:, None].astype(np.float32)
    residual = np.abs(rebuilt - logits).max(1)
    keep = residual <= args.residual
    report = {
        "positions": int(len(logits)),
        "kept": int(keep.sum()),
        "excluded_for_residual": int((~keep).sum()),
        "residual_max_median": float(np.median(residual)),
        "residual_max_max": float(residual.max()),
        "intercept_abs_max": float(np.abs(intercept).max()),
        "state_norm_median": float(np.median(np.linalg.norm(states, axis=1))),
        "proxies": {},
        "sampling": {},
    }
    logits, states, names = logits[keep], states[keep], names[keep]
    state_norm = np.linalg.norm(states.astype(np.float64), axis=1)
    best = logits.max(1)
    order = np.sort(logits, axis=1)[:, ::-1]
    report["logit_std_median"] = float(np.median(logits.std(1)))
    report["top1_minus_mean_median"] = float(np.median(best - logits.mean(1)))
    report["top1_minus_top2_median"] = float(np.median(order[:, 0] - order[:, 1]))
    full_bytes = rows * width * 2
    for name in args.proxies.split(","):
        error = quantized_error(weights, name)
        error_norm = np.linalg.norm(error.astype(np.float64), axis=1)
        approx = logits - states @ error.T  # A_r = (W - D) x, on the real logits
        del error
        # Rounded up to f32 with a little slack; the kernel adds float terms too.
        bound = (state_norm[:, None] * error_norm[None, :] * 1.002 + 1e-3).astype(np.float32)
        upper, lower = approx + bound, approx - bound
        one_stage = (upper >= lower.max(1, keepdims=True)).sum(1)
        provisional = np.argpartition(approx, -8, axis=1)[:, -8:]
        exact_best = np.take_along_axis(logits, provisional, 1).max(1)
        two_stage = (upper >= exact_best[:, None]).sum(1)
        hit = exact_best == best
        violations = int((np.abs(logits - approx) > bound + 0.02).sum())
        proxy = rows * proxy_bytes(name, width)

        def describe(counts, extra_rows=0):
            total = proxy + (counts + extra_rows) * width * 2
            return {
                "candidates_median": float(np.median(counts)),
                "candidates_p90": float(np.quantile(counts, 0.9)),
                "candidates_p99": float(np.quantile(counts, 0.99)),
                "candidates_max": int(counts.max()),
                "bytes_median_MB": float(np.median(total) / 1e6),
                "bytes_p90_MB": float(np.quantile(total, 0.9) / 1e6),
                "bytes_mean_MB": float(total.mean() / 1e6),
                "bytes_mean_x4_groups_MB": float(
                    (proxy + (4 * counts + extra_rows) * width * 2).mean() / 1e6
                ),
                "fraction_over_10pct_rows": float(np.mean(counts > rows // 10)),
            }

        report["proxies"][name] = {
            "proxy_MB": proxy / 1e6,
            "relative_error_median": float(
                np.median(error_norm / np.linalg.norm(weights.astype(np.float64), axis=1))
            ),
            "bound_median_logits": float(np.median(bound)),
            "bound_violations": violations,
            "provisional_top8_contains_argmax": float(hit.mean()),
            "one_stage": describe(one_stage),
            "two_stage": describe(two_stage, extra_rows=8),
            "vs_full_F16_MB": full_bytes / 1e6,
        }
        print(name, json.dumps(report["proxies"][name]), flush=True)
    wide = logits.astype(np.float64)
    probabilities = np.exp(wide - wide.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    ranked = np.sort(probabilities, axis=1)[:, ::-1]
    cumulative = np.cumsum(ranked, axis=1)
    tail80 = 1 - cumulative[:, 79]
    need = lambda mass: (cumulative < 1 - mass).sum(1) + 1  # noqa: E731
    sharp = np.exp((np.log(ranked[:, :80]) - np.log(ranked[:, :1])) / 0.2)
    sharp /= sharp.sum(1, keepdims=True)
    report["sampling"] = {
        "top1_probability_median": float(np.median(ranked[:, 0])),
        "top1_probability_p10": float(np.quantile(ranked[:, 0], 0.1)),
        "mass_outside_top80_median": float(np.median(tail80)),
        "mass_outside_top80_p90": float(np.quantile(tail80, 0.9)),
        "mass_outside_top80_max": float(tail80.max()),
        "rows_for_tail_1e-3_median": float(np.median(need(1e-3))),
        "rows_for_tail_1e-3_p90": float(np.quantile(need(1e-3), 0.9)),
        "rows_for_tail_1e-4_median": float(np.median(need(1e-4))),
        "rows_for_tail_1e-4_p90": float(np.quantile(need(1e-4), 0.9)),
        "T0.2_top80_top1_share_median": float(np.median(sharp[:, 0])),
        "T0.2_top80_top1_share_p10": float(np.quantile(sharp[:, 0], 0.1)),
        "T0.2_top80_top1_share_mean": float(sharp[:, 0].mean()),
    }
    print("sampling", json.dumps(report["sampling"]))
    (out / args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("proxies", "sampling")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "analyze"))
    parser.add_argument("--engine")
    parser.add_argument("--model", required=True)
    parser.add_argument("--d2")
    parser.add_argument("--d2-script")
    parser.add_argument("--label", default="A")
    parser.add_argument("--cases", default="fr,code,chat,story")
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=128000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--residual", type=float, default=0.05)
    parser.add_argument("--proxies", default="int8_row,int6_row,int5_row,int4_row,int4_g64")
    parser.add_argument("--report", default="analysis.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    sys.exit(main())
