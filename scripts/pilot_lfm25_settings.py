"""Compare official quantizer settings on projections and a partial-model KL panel."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

PROFILES = ("mcg-standard", "mcg-scaled", "mul1-standard", "mul1-scaled")
SEEDS = (25, 26)
EXPERTS = {2: [58, 24, 44], 20: [8, 39, 43], 39: [21, 11, 18]}


def settings_args(codebook: str, scaled: bool) -> list[str]:
    """Arguments shared by the pilot and production converter.

    raises: ValueError
    post: __return__[:2] == ['-cb', codebook]
    post: ('--scaled_hessian' in __return__) == scaled
    """
    if codebook not in ("mcg", "mul1") or type(scaled) is not bool:
        raise ValueError("Expected an official MCG/MUL1 codebook and boolean Hessian option")
    return ["-cb", codebook] + (["--scaled_hessian"] if scaled else [])


def profile_options(profile: str) -> tuple[str, bool]:
    if profile not in PROFILES:
        raise ValueError("Unknown registered settings profile")
    codebook, mode = profile.split("-")
    return codebook, mode == "scaled"


@contextmanager
def applied_weights(originals, replacements):
    """Copy into the existing allocations so compiled source operators see the change."""
    import torch

    if not originals or set(originals) != set(replacements):
        raise ValueError("A complete nonempty replacement panel is required")
    for key, live in originals.items():
        value = replacements[key]
        if (
            live.shape != value.shape
            or live.dtype != value.dtype
            or not live.numel()
            or not torch.isfinite(live).all().item()
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Invalid replacement weight")
    saved = {key: value.detach().cpu().clone() for key, value in originals.items()}
    try:
        for key, live in originals.items():
            live.copy_(replacements[key].to(live.device))
        yield
    finally:
        for key, live in originals.items():
            live.copy_(saved[key].to(live.device))
            if not torch.equal(live.view(torch.uint8).cpu(), saved[key].view(torch.uint8)):
                raise ValueError("Source weight restoration failed")


def kl_rows(teacher, candidate) -> dict:
    import torch

    if (
        teacher.ndim != 2
        or teacher.shape != candidate.shape
        or not teacher.numel()
        or not torch.isfinite(teacher).all().item()
        or not torch.isfinite(candidate).all().item()
    ):
        raise ValueError("Expected matching nonempty finite logit rows")
    p = torch.log_softmax(teacher.double(), dim=-1)
    q = torch.log_softmax(candidate.double(), dim=-1)
    divergence = (p.exp() * (p - q)).sum(dim=-1)
    if not torch.isfinite(divergence).all().item() or divergence.min().item() < -1e-10:
        raise ValueError("Nonfinite or negative KL")
    return {
        "kl": max(0.0, divergence.mean().item()),
        "rows": teacher.shape[0],
        "vocabulary": teacher.shape[1],
    }


def choose_settings(report: dict) -> dict | None:
    if report.get("status") != "complete" or not report.get("restoration_verified"):
        raise ValueError("The settings experiment must be complete and restored")
    rows = {}
    for row in report["validation"]:
        key = row["profile"], row["seed"], row["window"]
        if (
            key in rows
            or row["profile"] not in PROFILES
            or row["seed"] not in SEEDS
            or type(row["window"]) is not int
            or not 0 <= row["window"] < 12
            or row["rows"] != 128
            or row["vocabulary"] != 128000
            or not math.isfinite(row["kl"])
            or row["kl"] <= 0
        ):
            raise ValueError("Duplicate, incomplete or invalid partial-model KL")
        rows[key] = row["kl"]
    if len(rows) != len(PROFILES) * len(SEEDS) * 12:
        raise ValueError("Incomplete registered KL matrix")
    projections = {}
    for row in report["projections"]:
        key = row["profile"], row["seed"], row["key"]
        if (
            key in projections
            or row["profile"] not in PROFILES
            or row["seed"] not in SEEDS
            or not math.isfinite(row["rms"])
            or row["rms"] <= 0
        ):
            raise ValueError("Invalid projection metric")
        projections[key] = row["rms"]
    keys = {k for p, seed, k in projections if (p, seed) == (PROFILES[0], SEEDS[0])}
    if len(keys) != 29 or set(projections) != {
        (p, seed, k) for p in PROFILES for seed in SEEDS for k in keys
    }:
        raise ValueError("Incomplete projection matrix")
    scores = []
    for profile in PROFILES:
        domains = [
            math.fsum(rows[profile, seed, w] for seed in SEEDS for w in (2 * d, 2 * d + 1))
            / math.fsum(rows[PROFILES[0], seed, w] for seed in SEEDS for w in (2 * d, 2 * d + 1))
            for d in range(6)
        ]
        ratios = [
            projections[profile, seed, k] / projections[PROFILES[0], seed, k]
            for seed in SEEDS
            for k in keys
        ]
        proxy = math.exp(math.fsum(math.log(r) for r in ratios) / len(ratios))
        scores.append(
            {
                "profile": profile,
                "mean_partial_kl": math.fsum(rows[profile, s, w] for s in SEEDS for w in range(12))
                / 24,
                "domain_ratios": domains,
                "projection_geomean_ratio": proxy,
                "worst_projection_ratio": max(ratios),
                "eligible": max(domains + ratios) <= 1.10,
            }
        )
    choices = [s for s in scores[1:] if s["eligible"]]
    return (
        {"selected": min(choices, key=lambda s: s["mean_partial_kl"]), "scores": scores}
        if choices
        else None
    )


def run(source: Path, calibration: Path, validation_file: Path, output: Path) -> dict:
    import time

    import torch
    from exllamav3 import Config, Model
    from exllamav3.modules.mlp import capture_out_sensitivity, finalize_out_sensitivity
    from exllamav3.modules.quant import LinearEXL3
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from safetensors.torch import load_file

    output.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "partial_model_only": True,
        "calibration_rows": 250,
        "expert_panel_origin": "QUANT-2026-10-15-LFM25-30",
        "projections": [],
        "validation": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    model = None
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        train = load_file(str(calibration))["input_ids"]
        valid = load_file(str(validation_file))["input_ids"]
        if train.shape != (250, 2048) or valid.shape != (12, 1024):
            raise ValueError("Changed registered token shapes")
        if any(t.dtype != torch.int64 or t.min() < 0 or t.max() >= 125017 for t in (train, valid)):
            raise ValueError("Invalid source token IDs")
        train_rows = len(train)
        config = Config.from_directory(str(source))
        model = Model.from_config(config)
        model.load(device="cuda:0", max_output_size=2048)
        blocks = {i: model.modules[i + 1].mlp for i in (2, 20, 39)}
        trunk = {
            m.key: m
            for m in model
            if m.key in ("model.layers.0.conv.out_proj", "model.layers.0.feed_forward.w2")
        }
        if len(trunk) != 2:
            raise ValueError("Missing registered trunk")
        states = {i: [] for i in blocks}
        trunk_states = {k: [] for k in trunk}
        originals = []
        try:
            for identifier, module in [*blocks.items(), *trunk.items()]:
                original = module.forward
                originals.append((module, original))

                def capture(x, params, *args, identifier=identifier, original=original, **kwargs):
                    values = x.reshape(-1, x.shape[-1]).detach().half().cpu()
                    if not values.numel() or not torch.isfinite(values).all().item():
                        raise ValueError("Invalid source activation")
                    (states if isinstance(identifier, int) else trunk_states)[identifier].append(
                        values
                    )
                    return original(x, params, *args, **kwargs)

                module.forward = capture
            with torch.inference_mode():
                for number, row in enumerate([*train, *valid]):
                    y = model.forward(
                        row[None], {"attn_mode": "flash_attn_nc", "last_tokens_only": 1}
                    )
                    if y.numel() != 128000 or not torch.isfinite(y).all().item():
                        raise ValueError("Invalid source forward")
                    print(
                        json.dumps(
                            {
                                "phase": "capture",
                                "done": number + 1,
                                "total": train_rows + len(valid),
                            }
                        ),
                        flush=True,
                    )
        finally:
            for module, original in originals:
                module.forward = original

        panels = []
        for layer, mlp in blocks.items():
            counts = torch.zeros(64, dtype=torch.long)
            for x in states[layer][:train_rows]:
                picks, _ = mlp.routing_fn(len(x), mlp.routing_cfg, x.cuda(), {})
                counts += picks.flatten().cpu().bincount(minlength=64)
            if counts.sum().item() != train_rows * train.shape[1] * 4:
                raise ValueError("Incomplete full-calibration routing coverage")
            chosen = EXPERTS[layer]
            if any(counts[expert].item() < 512 for expert in chosen):
                raise ValueError("Frozen expert panel lacks training coverage")
            report.setdefault("expert_selection", {})[str(layer)] = {
                "counts": counts.tolist(),
                "chosen": chosen,
            }
            for expert in chosen:
                linears = {
                    "gate": mlp.gates[expert],
                    "up": mlp.ups[expert],
                    "down": mlp.downs[expert],
                }
                captures = {p: linear.init_H_data(True) for p, linear in linears.items()}
                sensitivity = {}
                validation_inputs = {p: [] for p in linears}
                for n, values in enumerate(states[layer]):
                    x = values.cuda()
                    picks, affinity = mlp.routing_fn(len(x), mlp.routing_cfg, x, {})
                    token, slot = torch.where(picks == expert)
                    g = linears["gate"].forward(x, {})
                    u = linears["up"].forward(x, {})
                    a = torch.empty_like(u)
                    mlp.activation_fn_call(g, u, a, mlp.act_limit)
                    if not torch.isfinite(a).all().item():
                        raise ValueError("Invalid source expert output")
                    for p in linears:
                        features = a if p == "down" else x
                        if n < train_rows:
                            h = captures[p]
                            h["H"].addmm_(features.float().T, features.float())
                            h["count"] += len(features)
                            h["num_total"] += features.numel()
                        elif token.numel():
                            validation_inputs[p].append(
                                (features[token].cpu(), affinity[token, slot].cpu())
                            )
                    if n < train_rows and token.numel():
                        capture_out_sensitivity(
                            sensitivity,
                            linears["gate"],
                            linears["up"],
                            linears["down"],
                            g[token],
                            u[token],
                            mlp.activation_fn,
                            mlp.act_limit,
                            affinity[token, slot],
                        )
                out_sens = finalize_out_sensitivity(sensitivity)
                for p, linear in linears.items():
                    panels.append(
                        (linear, 4, captures[p], validation_inputs[p], out_sens.get(linear.key))
                    )
        for name, linear in trunk.items():
            h = linear.init_H_data(True)
            for values in trunk_states[name][:train_rows]:
                x = values.cuda().float()
                h["H"].addmm_(x.T, x)
                h["count"] += len(x)
                h["num_total"] += x.numel()
            panels.append(
                (
                    linear,
                    5 if "feed_forward" in name else 6,
                    h,
                    [(x, torch.ones(len(x))) for x in trunk_states[name][train_rows:]],
                    None,
                )
            )
        if len(panels) != 29 or len({p[0].key for p in panels}) != 29:
            raise ValueError("Incomplete physical projection panel")
        replacements = {(p, seed): {} for p in PROFILES for seed in SEEDS}
        with torch.inference_mode():
            for linear, bits, captured, inputs, sensitivity in panels:
                weight = linear.inner.get_weight_tensor().float()
                if sum(len(x) for x, _ in inputs) < 32:
                    raise ValueError("Insufficient routed validation coverage")
                for profile in PROFILES:
                    cb, scaled = profile_options(profile)
                    for seed in SEEDS:
                        h = {
                            k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in captured.items()
                        }
                        qa = {
                            "K": bits,
                            "seed": seed,
                            cb: True,
                            "scaled_hessian": scaled,
                            "devices": [0],
                            "device_ratios": None,
                            "apply_out_scales": None,
                            "out_sensitivity": sensitivity,
                        }
                        _, _, packed = quantize_exl3(weight.clone(), h, qa, False)
                        decoded = LinearEXL3(
                            config, linear.in_features, linear.out_features, **packed
                        ).get_weight_tensor()
                        if (
                            qa["q_fallback"]
                            or decoded.shape != weight.shape
                            or not torch.isfinite(decoded).all().item()
                        ):
                            raise ValueError("Invalid packed quantization")
                        error = energy = 0.0
                        for x, affinity in inputs:
                            x = x.cuda().float()
                            scale = affinity.cuda().double()[:, None]
                            error += (
                                ((x @ (decoded.float() - weight)).double() * scale)
                                .square()
                                .sum()
                                .item()
                            )
                            energy += ((x @ weight).double() * scale).square().sum().item()
                        if energy <= 0 or not math.isfinite(error + energy):
                            raise ValueError("Empty or nonfinite projection metric")
                        replacements[profile, seed][linear.key] = decoded.cpu()
                        report["projections"].append(
                            {
                                "profile": profile,
                                "seed": seed,
                                "key": linear.key,
                                "rms": math.sqrt(error / energy),
                            }
                        )
                        save()
                print(
                    json.dumps(
                        {"phase": "projection", "done": len(report["projections"]), "total": 232}
                    ),
                    flush=True,
                )
            live = {linear.key: linear.inner.get_weight_tensor() for linear, *_ in panels}

            def logits(row):
                y = model.forward(
                    row[None], {"attn_mode": "flash_attn_nc", "last_tokens_only": 128}
                ).float()
                if y.numel() != 128 * 128000 or not torch.isfinite(y).all().item():
                    raise ValueError("Invalid partial-model logit shape or values")
                return y.reshape(128, 128000).cpu()

            teachers = [logits(row) for row in valid]
            canary = trunk["model.layers.0.conv.out_proj"].inner.get_weight_tensor()
            with applied_weights({"canary": canary}, {"canary": canary * 1.1}):
                changed = kl_rows(teachers[0], logits(valid[0]))
                if changed["kl"] <= 1e-8:
                    raise ValueError("The real source forward did not observe the weight mutation")
            report["negative_control"] = changed
            for (profile, seed), weights in replacements.items():
                with applied_weights(live, weights):
                    for number, row in enumerate(valid):
                        result = kl_rows(teachers[number], logits(row))
                        report["validation"].append(
                            {"profile": profile, "seed": seed, "window": number, **result}
                        )
                        save()
                if not torch.equal(
                    logits(valid[0]).view(torch.uint8), teachers[0].view(torch.uint8)
                ):
                    raise ValueError("Source logits did not restore bit exactly")
                print(
                    json.dumps({"phase": "partial_kl", "profile": profile, "seed": seed}),
                    flush=True,
                )
        report.update(status="complete", restoration_verified=True)
        report["selection"] = choose_settings(report)
        report["input_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (calibration, validation_file)
        }
        save()
        return report
    except BaseException as error:
        report.update(status="failed", error=repr(error), selection=None)
        save()
        raise
    finally:
        if model is not None:
            model.unload()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        raise SystemExit("source calibration.safetensors validation.safetensors output")
    run(*(Path(p) for p in sys.argv[1:]))
