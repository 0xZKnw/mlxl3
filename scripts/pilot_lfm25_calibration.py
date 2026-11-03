"""H100 proxy ablation on source states; this does not measure full-model KL."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from exl3_moe_calibration import capture_weighted, check_coverage


def select_experts(counts: list[int]) -> list[int]:
    if len(counts) != 64 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Expected 64 nonnegative expert counts")
    eligible = sorted((n, i) for i, n in enumerate(counts) if n >= 512)
    if len(eligible) < 3:
        raise ValueError("Insufficient covered experts for the pilot")
    return [eligible[i][1] for i in (0, len(eligible) // 2, len(eligible) - 1)]


def choose_pilot(report: dict) -> dict:
    if report.get("status") != "complete" or not report.get("proxy_only"):
        raise ValueError("A complete proxy report is required")
    variants = {}
    domains = {}
    for panel in report["panels"]:
        variant = variants.setdefault((panel["corpus"], panel["mode"]), {})
        for projection, result in panel["projections"].items():
            key = (panel["layer"], panel["expert"], projection)
            value = result["relative_rms"]
            if key in variant or not math.isfinite(value) or value <= 0:
                raise ValueError("Duplicate, empty or invalid pilot result")
            variant[key] = value
            totals = domains.setdefault(
                (panel["corpus"], panel["mode"]), [[0.0, 0.0] for _ in range(6)]
            )
            for number, measured in result["validation"].items():
                index = int(number)
                error, energy = measured["error"], measured["energy"]
                if (
                    not 0 <= index < 12
                    or measured["rows"] <= 0
                    or not math.isfinite(error + energy)
                    or error < 0
                    or energy <= 0
                ):
                    raise ValueError("Invalid domain metric")
                totals[index // 2][0] += error
                totals[index // 2][1] += energy
    expected = {
        (corpus, mode)
        for corpus in ("original", "trace")
        for mode in ("all", "routed", "affinity", "squared")
    }
    if set(variants) != expected or any(len(v) != 27 for v in variants.values()):
        raise ValueError("Incomplete registered pilot matrix")
    baseline = variants[("original", "all")]
    if any(error <= 0 or energy <= 0 for sums in domains.values() for error, energy in sums):
        raise ValueError("Incomplete validation domain coverage")
    scores = []
    for (corpus, mode), values in variants.items():
        if values.keys() != baseline.keys():
            raise ValueError("Pilot variants exercise different projections")
        ratios = [value / baseline[key] for key, value in values.items()]
        domain_ratios = [
            math.sqrt((error / energy) / (old_error / old_energy))
            for (error, energy), (old_error, old_energy) in zip(
                domains[(corpus, mode)], domains[("original", "all")], strict=True
            )
        ]
        score = math.exp(math.fsum(math.log(r) for r in ratios) / len(ratios))
        scores.append(
            {
                "corpus": corpus,
                "mode": mode,
                "geomean_ratio": score,
                "worst_projection_ratio": max(ratios),
                "domain_ratios": domain_ratios,
                "eligible": max(ratios + domain_ratios) <= 1.10,
            }
        )
    selected = min((s for s in scores if s["eligible"]), key=lambda s: s["geomean_ratio"])
    trunk = [r for r in report.get("trunk", []) if r["corpus"] == selected["corpus"]]
    trunk_ratios = []
    for name in {r["key"] for r in trunk}:
        entries = {r["bits"]: r["relative_rms"] for r in trunk if r["key"] == name}
        base = 5 if "feed_forward" in name else 6
        if set(entries) != {base, 8} or any(
            not math.isfinite(v) or v <= 0 for v in entries.values()
        ):
            raise ValueError("Incomplete trunk comparison")
        trunk_ratios.append(entries[8] / entries[base])
    estimated_bpw = 8 * (12614197321 + report.get("trunk_budget", 0)) / 23971587712
    return {
        **selected,
        "all_variants": scores,
        "trunk_ratios": trunk_ratios,
        "estimated_compact_bpw": estimated_bpw,
        "trunk8": len(trunk_ratios) == 2 and max(trunk_ratios) <= 0.8 and estimated_bpw <= 4.35,
    }


def run(source: Path, old_path: Path, new_path: Path, validation_path: Path, output: Path) -> dict:
    import torch
    from exllamav3 import Config, Model
    from exllamav3.modules.mlp import capture_out_sensitivity, finalize_out_sensitivity
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from safetensors.torch import load_file

    output.mkdir(exist_ok=False)
    report = {"status": "running", "proxy_only": True, "panels": [], "selection": None}
    model = None

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def key(tokens):
        return hashlib.sha256(tokens.numpy().astype("<i8").tobytes()).hexdigest()

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        old = load_file(str(old_path))["input_ids"]
        new = load_file(str(new_path))["input_ids"]
        validation = load_file(str(validation_path))["input_ids"]
        if old.shape != (250, 2048) or new.shape != old.shape or validation.shape != (12, 1024):
            raise ValueError("Pilot token plan shape mismatch")
        indices = np.linspace(0, 249, 32).astype(int).tolist()
        trains = {"original": old[indices], "trace": new[indices]}
        sequences = {key(row): row for rows in [*trains.values(), validation] for row in rows}
        config = Config.from_directory(str(source))
        model = Model.from_config(config)
        model.load(device="cuda:0", max_output_size=2048)
        blocks = {layer: model.modules[layer + 1] for layer in (2, 20, 39)}
        trunk = {
            linear.key: linear
            for linear in model
            if linear.key in ("model.layers.0.conv.out_proj", "model.layers.0.feed_forward.w2")
        }
        if len(trunk) != 2:
            raise ValueError("Missing two registered trunk projections")
        trunk_states = {name: {} for name in trunk}
        states = {layer: {} for layer in blocks}
        originals = {}
        current_key = None
        for name, linear in trunk.items():
            original = linear.forward
            originals[name] = original

            def capture_linear(x, params, *args, name=name, original=original, **kwargs):
                if current_key is None:
                    raise RuntimeError("Missing trunk capture sequence")
                trunk_states[name][current_key] = x.reshape(-1, x.shape[-1]).detach().half().cpu()
                return original(x, params, *args, **kwargs)

            linear.forward = capture_linear
        for layer, block in blocks.items():
            mlp = block.mlp
            original = mlp.forward
            originals[layer] = original

            def capture(x, params, *, layer=layer, original=original):
                if current_key is None:
                    raise RuntimeError("Missing capture sequence")
                values = x.reshape(-1, x.shape[-1]).detach().half().cpu()
                if values.shape[1] != 2048 or not torch.isfinite(values).all().item():
                    raise ValueError("Invalid source MLP inputs")
                states[layer][current_key] = values
                return original(x, params)

            mlp.forward = capture
        with torch.inference_mode():
            for current_key, row in sequences.items():
                result = model.forward(
                    row[None], {"attn_mode": "flash_attn_nc", "last_tokens_only": 1}
                )
                if result.numel() != 128000 or not torch.isfinite(result).all().item():
                    raise ValueError("Invalid source forward")
                print(
                    json.dumps(
                        {"phase": "source_states", "done": len(states[2]), "total": len(sequences)}
                    ),
                    flush=True,
                )
        for layer, block in blocks.items():
            block.mlp.forward = originals[layer]
        for name, linear in trunk.items():
            linear.forward = originals[name]
        current_key = None

        def routed(mlp, x):
            selected, weights = mlp.routing_fn(x.shape[0], mlp.routing_cfg, x, {})
            if selected.shape != (x.shape[0], 4) or not torch.isfinite(weights).all().item():
                raise ValueError("Invalid top4 routing")
            return selected, weights

        def features(mlp, expert, x):
            u = mlp.ups[expert].forward(x, {})
            g = mlp.gates[expert].forward(x, {})
            a = torch.empty_like(u)
            mlp.activation_fn_call(g, u, a, mlp.act_limit)
            if not torch.isfinite(a).all().item():
                raise ValueError("Invalid source down input")
            return g, u, a

        with torch.inference_mode():
            for layer, block in blocks.items():
                mlp = block.mlp
                counts = torch.zeros(64, dtype=torch.long)
                for row in trains["original"]:
                    x = states[layer][key(row)].cuda()
                    selected, _ = routed(mlp, x)
                    counts += selected.flatten().cpu().bincount(minlength=64)
                chosen = select_experts(counts.tolist())
                report.setdefault("expert_selection", {})[str(layer)] = {
                    "counts": counts.tolist(),
                    "chosen": chosen,
                }
                for expert in chosen:
                    linears = {
                        "up": mlp.ups[expert],
                        "gate": mlp.gates[expert],
                        "down": mlp.downs[expert],
                    }
                    original_maps = {p: linear.qmap for p, linear in linears.items()}
                    for p, linear in linears.items():
                        linear.qmap = "input" if p != "down" else "down"
                    weights = {
                        p: linear.inner.get_weight_tensor().float() for p, linear in linears.items()
                    }
                    valid = []
                    for number, row in enumerate(validation):
                        x = states[layer][key(row)].cuda()
                        selected, affinity = routed(mlp, x)
                        token, slot = torch.where(selected == expert)
                        if not token.numel():
                            continue
                        xr = x[token]
                        _, _, a = features(mlp, expert, xr)
                        valid.append((number, xr.cpu(), a.cpu(), affinity[token, slot].cpu()))
                    if sum(len(x) for _, x, _, _ in valid) < 32:
                        raise ValueError("Insufficient held-out routed rows")
                    for corpus, rows in trains.items():
                        captures = {mode: {} for mode in ("all", "routed", "affinity", "squared")}
                        sensitivity = {}
                        for row in rows:
                            x = states[layer][key(row)].cuda()
                            selected, affinity = routed(mlp, x)
                            token, slot = torch.where(selected == expert)
                            g, u, a = features(mlp, expert, x)
                            for mode, captured in captures.items():
                                for p in ("up", "down"):
                                    linear = linears[p]
                                    values = x if p == "up" else a
                                    if mode == "all":
                                        if linear.qmap not in captured:
                                            captured[linear.qmap] = linear.init_H_data(True)
                                        data = captured[linear.qmap]
                                        vf = values.float()
                                        data["H"].addmm_(vf.T, vf)
                                        data["count"] += len(values)
                                        data["num_total"] += values.numel()
                                    elif token.numel():
                                        capture_weighted(
                                            linear,
                                            values[token],
                                            {
                                                "capture": captured,
                                                "mlxl3_moe_mode": mode,
                                                "mlxl3_expert_weights": affinity[token, slot],
                                            },
                                        )
                            if token.numel():
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
                        for mode, captured in captures.items():
                            panel = {
                                "layer": layer,
                                "expert": expert,
                                "corpus": corpus,
                                "mode": mode,
                                "projections": {},
                            }
                            for p, weight in weights.items():
                                data = captured[linears[p].qmap]
                                if mode != "all":
                                    check_coverage(
                                        data["count"],
                                        data["affinity_mass"],
                                        data["affinity_mass2"],
                                        256,
                                    )
                                qa = {
                                    "K": 4,
                                    "seed": 25,
                                    "mcg": True,
                                    "devices": [0],
                                    "device_ratios": None,
                                    "apply_out_scales": None,
                                    "out_sensitivity": out_sens.get(linears[p].key),
                                }
                                quantized, proxy, _packed = quantize_exl3(
                                    weight.clone(), data, qa, True
                                )
                                if (
                                    qa["q_fallback"]
                                    or quantized.shape != weight.shape
                                    or not torch.isfinite(quantized).all().item()
                                ):
                                    raise ValueError("Invalid or uncalibrated pilot quantization")
                                numerator = denominator = 0.0
                                per_domain = {}
                                for number, x, a, affinity in valid:
                                    values = (a if p == "down" else x).cuda().float()
                                    scale = affinity.cuda().double()[:, None]
                                    difference = (values @ (quantized - weight)).double() * scale
                                    reference = (values @ weight).double() * scale
                                    error, energy = (
                                        difference.square().sum().item(),
                                        reference.square().sum().item(),
                                    )
                                    numerator += error
                                    denominator += energy
                                    per_domain[str(number)] = {
                                        "error": error,
                                        "energy": energy,
                                        "rows": len(x),
                                    }
                                if denominator <= 0 or not math.isfinite(numerator + denominator):
                                    raise ValueError("Empty or nonfinite pilot metric")
                                panel["projections"][p] = {
                                    "relative_rms": math.sqrt(numerator / denominator),
                                    "proxy": float(proxy),
                                    "validation": per_domain,
                                }
                            report["panels"].append(panel)
                            save()
                            print(
                                json.dumps(
                                    {
                                        "phase": "panel",
                                        "layer": layer,
                                        "expert": expert,
                                        "corpus": corpus,
                                        "mode": mode,
                                    }
                                ),
                                flush=True,
                            )
                    for p, linear in linears.items():
                        linear.qmap = original_maps[p]
            report["trunk"] = []
            for name, linear in trunk.items():
                weight = linear.inner.get_weight_tensor().float()
                baseline = 5 if "feed_forward" in name else 6
                for corpus, rows in trains.items():
                    for bits in (baseline, 8):
                        data = linear.init_H_data(True)
                        for row in rows:
                            x = trunk_states[name][key(row)].cuda().float()
                            if not torch.isfinite(x).all().item():
                                raise ValueError("Invalid trunk activations")
                            data["H"].addmm_(x.T, x)
                            data["count"] += len(x)
                        qa = {
                            "K": bits,
                            "seed": 25,
                            "mcg": True,
                            "devices": [0],
                            "device_ratios": None,
                            "apply_out_scales": None,
                        }
                        quantized, _, _ = quantize_exl3(weight.clone(), data, qa, True)
                        if (
                            qa["q_fallback"]
                            or quantized.shape != weight.shape
                            or not torch.isfinite(quantized).all().item()
                        ):
                            raise ValueError("Invalid trunk quantization")
                        error = energy = 0.0
                        for row in validation:
                            x = trunk_states[name][key(row)].cuda().float()
                            error += (x @ (quantized - weight)).double().square().sum().item()
                            energy += (x @ weight).double().square().sum().item()
                        if not math.isfinite(error + energy) or energy <= 0:
                            raise ValueError("Invalid trunk metric")
                        report["trunk"].append(
                            {
                                "key": name,
                                "corpus": corpus,
                                "bits": bits,
                                "relative_rms": math.sqrt(error / energy),
                            }
                        )
                        save()
                        print(
                            json.dumps(
                                {"phase": "trunk", "key": name, "corpus": corpus, "bits": bits}
                            ),
                            flush=True,
                        )
            report["trunk_budget"] = sum(
                linear.weights_numel() * (8 - (5 if ".feed_forward." in linear.key else 6)) // 8
                for linear in model
                if getattr(linear, "qbits_key", None) == "bits"
                and linear.qmap
                and ".experts." not in linear.key
            )
        report["status"] = "complete"
        save()
        return report
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        if model is not None:
            model.unload()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        raise SystemExit("source old.safetensors trace.safetensors validation.safetensors output")
    run(*(Path(p) for p in sys.argv[1:]))
