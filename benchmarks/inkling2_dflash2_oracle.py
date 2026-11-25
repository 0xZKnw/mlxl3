"""DFLASH2-2026-11-24-INK2M-P0: independent numpy (float64) oracle of the Inkling 2 Mini DFlash2 drafter.

Recomputes, from the BF16 checkpoint and a case exported by the engine test
`inkling2_dflash2::tests::export_oracle_cases`, the context keys/values, the block forward, the final hidden rows,
the top-16 candidates, the selector lattice and its greedy walk, following SGLang `f4284f29ba56`
`srt/models/dflash.py` (DFlashGroupedConv, DFlashDecoderLayer, DFlashDraftModel, CandidateSelector) and the Inkling
patch (`selector_softcap_mode` after the correction). Nothing of the engine's arithmetic is reused: only its inputs
(target features, block embeddings, target head logits of rows 1..7) and its outputs for the comparison.

    python -I benchmarks/inkling2_dflash2_oracle.py MODEL_DIR CASE_DIR [CASE_DIR ...] [--out report.json]

Checks per case: hidden rows relative error <= 0.02, the 16 candidates of each row equal (as sets, and the
ordered top-16 of the exported logits), the 7 proposals equal, and the convolution mutation's hidden rows
farther than 0.05. Exit code 1 and `"status": "failed"` when any check fails or a case is missing.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r")

    def tensor(name):
        info = header[name]
        start, end = info["data_offsets"]
        raw = np.asarray(data[base + start: base + end])
        assert info["dtype"] == "BF16", (name, info["dtype"])
        bits = raw.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64).reshape(info["shape"])

    return header, tensor


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions, theta):
    # x [..., rows, d], NeoX (rotate_half) as Qwen3 and MLX fast.rope(traditional=False)
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    ang = positions[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def grouped_conv(x, delta, base, groups, group, taps, mutate=False):
    rows = x.shape[0]
    blocks = x.reshape(rows, groups, group)
    coef = base.reshape(1, taps, groups, group) + delta[..., None]
    out = coef[:, 0] * blocks
    for tap in range(1, taps):
        if mutate:
            out = out + coef[:, tap] * blocks
            continue
        shifted = np.zeros_like(blocks)
        shifted[tap:] = blocks[:-tap]
        out = out + coef[:, tap] * shifted
    return out.reshape(rows, groups * group)


def forward(cfg, t, features, embeds, mutate=False):
    h, eps, theta = cfg["hidden_size"], cfg["rms_norm_eps"], cfg["rope_parameters"]["rope_theta"]
    heads, kv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    dc = cfg["dflash_config"]
    taps, group = dc["conv_kernel_size"], dc["conv_group_size"]
    groups = h // group
    window = cfg["sliding_window"]
    n = features.shape[0]
    rows = embeds.shape[0]
    ctx = rms(features @ t("fc.weight").T, t("hidden_norm.weight"), eps)
    ctx_pos = np.arange(n, dtype=np.float64)
    blk_pos = np.arange(n, n + rows, dtype=np.float64)
    residual = embeds.copy()
    hidden = rms(residual, t("layers.0.input_layernorm.weight"), eps)
    pending = None
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}"
        if pending is not None:
            out, coef_m, prev = pending
            residual = residual + grouped_conv(out, coef_m[:, 1], t(f"layers.{prev}.mlp_conv.base_kernel")[1],
                                               groups, group, taps, mutate)
            hidden = rms(residual, t(f"{p}.input_layernorm.weight"), eps)
        coef = (hidden @ t(f"{p}.attention_conv.kernel_projection.weight").T).reshape(rows, 2, taps, groups)
        base = t(f"{p}.attention_conv.base_kernel")
        x = grouped_conv(hidden, coef[:, 0], base[0], groups, group, taps, mutate)
        q = rms((x @ t(f"{p}.self_attn.q_proj.weight").T).reshape(rows, heads, d), t(f"{p}.self_attn.q_norm.weight"), eps)
        k = rms((x @ t(f"{p}.self_attn.k_proj.weight").T).reshape(rows, kv, d), t(f"{p}.self_attn.k_norm.weight"), eps)
        v = (x @ t(f"{p}.self_attn.v_proj.weight").T).reshape(rows, kv, d)
        ck = rms((ctx @ t(f"{p}.self_attn.k_proj.weight").T).reshape(n, kv, d), t(f"{p}.self_attn.k_norm.weight"), eps)
        cv = (ctx @ t(f"{p}.self_attn.v_proj.weight").T).reshape(n, kv, d)
        q = rope(q.transpose(1, 0, 2), blk_pos, theta)          # [heads, rows, d]
        k = rope(k.transpose(1, 0, 2), blk_pos, theta)          # [kv, rows, d]
        ck = rope(ck.transpose(1, 0, 2), ctx_pos, theta)        # [kv, n, d]
        keys = np.concatenate([ck, k], axis=1)
        values = np.concatenate([cv.transpose(1, 0, 2), v.transpose(1, 0, 2)], axis=1)
        positions = np.concatenate([ctx_pos, blk_pos])
        attended = np.zeros((rows, heads, d))
        for head in range(heads):
            g = head // (heads // kv)
            s = q[head] @ keys[g].T / np.sqrt(d)                 # [rows, n + rows]
            visible = positions[None, :] > (blk_pos[:, None] - window)
            s = np.where(visible, s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(axis=-1, keepdims=True)
            attended[:, head] = w @ values[g]
        out = attended.reshape(rows, heads * d) @ t(f"{p}.self_attn.o_proj.weight").T
        residual = residual + grouped_conv(out, coef[:, 1], base[1], groups, group, taps, mutate)
        hidden = rms(residual, t(f"{p}.post_attention_layernorm.weight"), eps)
        coef_m = (hidden @ t(f"{p}.mlp_conv.kernel_projection.weight").T).reshape(rows, 2, taps, groups)
        xm = grouped_conv(hidden, coef_m[:, 0], t(f"{p}.mlp_conv.base_kernel")[0], groups, group, taps, mutate)
        gate = xm @ t(f"{p}.mlp.gate_proj.weight").T
        up = xm @ t(f"{p}.mlp.up_proj.weight").T
        mlp = (gate / (1 + np.exp(-gate)) * up) @ t(f"{p}.mlp.down_proj.weight").T
        pending = (mlp, coef_m, i)
    out, coef_m, prev = pending
    residual = residual + grouped_conv(out, coef_m[:, 1], t(f"layers.{prev}.mlp_conv.base_kernel")[1],
                                       groups, group, taps, mutate)
    return rms(residual, t("norm.weight"), eps)


def select(cfg, t, hidden_pred, logits, anchor):
    dc = cfg["dflash_config"]
    k = dc["selector_top_k"]
    n = hidden_pred.shape[0]
    order = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    unary = np.take_along_axis(logits, order, axis=-1)
    cap = dc.get("final_logit_softcapping")
    after = dc.get("selector_softcap_mode") == "after_correction"
    if cap and not after:
        unary = cap * np.tanh(unary / cap)
    proj = hidden_pred @ t("candidate_selector.hidden_projection.weight").T     # [n, r]
    succ_table = t("candidate_selector.successor_codebook")
    pred_table = t("candidate_selector.predecessor_codebook")
    scores = np.zeros((n, k, k))
    for e in range(n):
        preds = [anchor] * k if e == 0 else list(order[e - 1])
        a = pred_table[preds] * proj[e][None, :]                                # [k(p), r]
        b = succ_table[order[e]]                                                # [k(c), r]
        scores[e] = unary[e][None, :] + a @ b.T
    if cap and after:
        scores = cap * np.tanh(scores / cap)
    index = int(np.argmax(scores[0, 0]))
    proposals = [int(order[0, index])]
    for e in range(1, n):
        index = int(np.argmax(scores[e, index]))
        proposals.append(int(order[e, index]))
    return order, scores, proposals


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    model = Path(args.model)
    cfg = json.loads((model / "config.json").read_text())
    _, t = read_safetensors(model / "model.safetensors")
    report = {"status": "failed", "cases": []}
    ok = True
    for case in args.cases:
        case = Path(case)
        entry = {"case": str(case), "problems": []}
        try:
            meta = json.loads((case / "meta.json").read_text())
            h = cfg["hidden_size"]
            features = np.fromfile(case / "features.f32", dtype=np.float32).astype(np.float64).reshape(meta["features_shape"])
            embeds = np.fromfile(case / "embeds.f32", dtype=np.float32).astype(np.float64).reshape(-1, h)
            engine_hidden = np.fromfile(case / "hidden.f32", dtype=np.float32).astype(np.float64).reshape(-1, h)
            mutated = np.fromfile(case / "hidden_mutated.f32", dtype=np.float32).astype(np.float64).reshape(-1, h)
            rows = embeds.shape[0]
            logits = np.fromfile(case / "head_logits.f32", dtype=np.float32).astype(np.float64).reshape(rows - 1, -1)
            assert logits.shape[1] == meta["valid_vocab"], "logits width"
            for a in (features, embeds, engine_hidden, logits):
                assert np.isfinite(a).all(), "non-finite input"
            hidden = forward(cfg, t, features, embeds)
            row_rel = [rel(engine_hidden[i], hidden[i]) for i in range(rows)]
            mutated_rel = rel(mutated, hidden)
            order, scores, proposals = select(cfg, t, hidden[1:], logits, meta["anchor"])
            k = cfg["dflash_config"]["selector_top_k"]
            engine_cand = np.array(meta["candidates"]).reshape(rows - 1, k)
            same_sets = all(set(order[e]) == set(engine_cand[e]) for e in range(rows - 1))
            engine_scores = np.array(meta["scores"]).reshape(rows - 1, k, k)
            entry.update({
                "row": meta["row"], "hidden_rel_max": max(row_rel), "hidden_rel": row_rel,
                "mutated_rel": mutated_rel, "candidates_equal_sets": same_sets,
                "candidates_equal_order": bool((order == engine_cand).all()),
                "scores_rel": rel(engine_scores, scores) if same_sets and (order == engine_cand).all() else None,
                "proposals_oracle": proposals, "proposals_engine": meta["proposals"],
                "proposals_equal": proposals == meta["proposals"], "next_tokens": meta["next_tokens"],
            })
            if max(row_rel) > 0.02:
                entry["problems"].append(f"hidden relative error {max(row_rel):.4f} > 0.02")
            if not same_sets:
                entry["problems"].append("candidate sets differ")
            if proposals != meta["proposals"]:
                entry["problems"].append("proposals differ")
            if mutated_rel <= 0.05:
                entry["problems"].append(f"mutation not detected ({mutated_rel:.4f})")
        except Exception as error:  # noqa: BLE001
            entry["problems"].append(f"error: {error!r}")
        ok &= not entry["problems"]
        report["cases"].append(entry)
        print(json.dumps({key: entry.get(key) for key in (
            "row", "hidden_rel_max", "mutated_rel", "candidates_equal_sets", "candidates_equal_order",
            "scores_rel", "proposals_oracle", "proposals_engine", "proposals_equal", "problems")}))
    report["status"] = "passed" if ok and report["cases"] else "failed"
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    print("status", report["status"])
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
