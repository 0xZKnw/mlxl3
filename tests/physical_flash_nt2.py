"""Qualify native resident NT2 against independent stock and same-binary NT1."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from physical_projection_group import save_checkpoint

REAL = [
    ("model.language_model.layers.0.mlp.shared_expert.down_proj", 640, 2560),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv", 2560, 10240),
    ("model.language_model.layers.0.linear_attn.in_proj_z", 2560, 6144),
    ("model.language_model.layers.0.linear_attn.out_proj", 6144, 2560),
    ("model.language_model.layers.3.self_attn.q_proj", 2560, 12288),
    ("model.language_model.layers.3.self_attn.o_proj", 6144, 2560),
    ("lm_head", 2560, 248320),
    ("mtp.fc_embedding", 2560, 2560),
]


def fixtures(root, model):
    rng = np.random.default_rng(75112)
    native, stock = [], []

    def add(folder, prefix, width, cols, nt, *, generic=False, batch=1):
        x = rng.normal(0, 0.0625, (batch, width)).astype(np.float16)
        for fp32 in (False, True):
            packet = {
                "path": str(folder),
                "prefix": prefix,
                "x": x.view(np.uint16).ravel().tolist(),
            }
            native.append(
                dict(
                    packet,
                    fp32=fp32,
                    generic=generic,
                    candidate_nt=nt,
                    control_nt=4 if generic else 1,
                    words=batch * cols,
                )
            )
            stock.append(
                dict(packet, op="metal-exl3-linear" if generic else "metal-checkpoint-linear")
            )

    for i, (k, cb, cols, dtype, generic) in enumerate(
        [(3, 2, n, d, False) for n in (2432, 2560, 2688) for d in (0, 1, 2)]
        + [(k, cb, 2560, 2, False) for k, cb in ((2, 2), (4, 0), (4, 1), (7, 2), (8, 2))]
        + [(4, 2, 2560, 2, True), (4, 2, 2560, 0, False)]
    ):
        folder = root / f"synthetic-{i}"
        folder.mkdir()
        width = 4096 if k == 4 and cb == 2 and not generic else 128
        tensors = {
            "p.trellis": rng.integers(0, 65536, (width // 16, cols // 16, 16 * k), np.uint16)
        }
        bf16 = []
        for suffix, size in (("suh", width), ("svh", cols), ("bias", cols)):
            key = "p." + suffix
            if dtype == 1:
                tensors[key] = np.resize(
                    np.array([0x3F00, 0x3F40, 0xBF00, 0x3F80], np.uint16), size
                )
                bf16.append(key)
            else:
                tensors[key] = rng.normal(0.6, 0.1, size).astype(
                    np.float32 if dtype == 0 else np.float16
                )
        if cb:
            tensors["p.mul1" if cb == 2 else "p.mcg"] = np.array([1], np.int32)
        save_checkpoint(folder, tensors, {"p": {"quant_format": "exl3"}}, bf16)
        (folder / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5_moe" if generic else "qwen4_exp"})
        )
        nt = 4 if generic else 2 if k in (3, 4) and cb == 2 and cols >= 2560 else 1
        add(folder, "p", width, cols, nt, generic=generic, batch=3 if i % 2 else 1)
    # Keep a real checkpoint last: the native fixture also checks fc_hidden exclusions.
    for prefix, width, cols in REAL:
        add(model, prefix, width, cols, 2)
    (root / "cases.json").write_text(json.dumps(stock))
    (root / "native-cases.json").write_text(json.dumps(native))
    return native


def process(command, env, log):
    try:
        child = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=240, check=False
        )
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error) + "\n" + str(error.stdout) + "\n" + str(error.stderr))
        raise
    log.write_text(child.stdout + child.stderr)
    assert child.returncode == 0 and "1 passed; 0 failed" in child.stdout, log
    return child.stderr


def words(actual: list[int], expected: list[int]) -> int:
    """Fail closed on empty, truncated, invalid or divergent numerical evidence.

    raises: AssertionError
    post: __return__ == len(actual) == len(expected) > 0
    post: all(a == b and 0 <= a < 65536 for a, b in zip(actual, expected))
    """
    assert len(actual) == len(expected) > 0
    assert all(type(v) is int and 0 <= v < 65536 for v in actual + expected)
    a, b = np.asarray(actual, np.uint16), np.asarray(expected, np.uint16)
    assert np.isfinite(a.view(np.float16)).all() and np.isfinite(b.view(np.float16)).all()
    assert np.any(b.view(np.float16) != 0)
    assert np.array_equal(a, b)
    return len(actual)


def verify(args, root, report):
    cases = fixtures(root, args.model.resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    process(
        [
            str(args.stock),
            "flash_projection_stock_oracle",
            "--exact",
            "--ignored",
            "--test-threads=1",
        ],
        {**env, "MLXL3_PROJECTION_FIXTURES": str(root)},
        root / "stock.log",
    )
    refs = json.loads((root / "stock.json").read_text())
    assert len(refs) == len(cases) > 0
    compared = 0
    report["modes"] = []
    for enabled in (0, 1):
        stderr = process(
            [
                str(args.native),
                "gpu::projection::physical_flash_resident_nt2",
                "--exact",
                "--ignored",
                "--test-threads=1",
            ],
            {
                **env,
                "MLXL3_NATIVE_FLASH_NT2": str(enabled),
                "MLXL3_NT2_FIXTURES": str(root),
                "DYLD_PRINT_LIBRARIES": "1",
            },
            root / f"native-{enabled}.log",
        )
        images = [line.lower() for line in stderr.splitlines() if line.startswith("dyld[")]
        assert images and not any(
            key in line
            for line in images
            for key in ("/libmlx", "/libjaccl", "/site-packages/mlx/")
        )
        result = json.loads((root / f"native-{enabled}.json").read_text())
        assert result["parity"] is True and result["unequal"] == 0
        assert len(result["cases"]) == len(cases)
        submissions = 0
        for i, (case, ref, item) in enumerate(zip(cases, refs, result["cases"], strict=True)):
            nt = case["candidate_nt"] if enabled else case["control_nt"]
            count = int(bool(enabled and not case["generic"] and nt == 2))
            assert item["case"] == i and item["nt"] == nt and item["nt2_submissions"] == count
            assert item["words"] == case["words"]
            submissions += count
            actual = np.fromfile(root / f"native-{enabled}-{i}.bin", dtype="<u2").tolist()
            assert len(actual) == case["words"] > 0
            compared += words(actual, ref)
            if enabled:
                words(actual, np.fromfile(root / f"native-0-{i}.bin", dtype="<u2").tolist())
        assert result["nt2_submissions"] == submissions and (submissions > 0) == bool(enabled)
        report["modes"].append(result)
    process(
        [
            str(args.native),
            "gpu::projection::physical_flash_resident_nt2",
            "--exact",
            "--ignored",
            "--test-threads=1",
        ],
        {**env, "MLXL3_NATIVE_FLASH_NT2": "true", "MLXL3_NT2_FIXTURES": str(root)},
        root / "invalid-flag.log",
    )
    assert json.loads((root / "invalid-flag.json").read_text()) == {
        "rejected": True,
        "fc_hidden_nt": 1,
        "generic_unchanged": True,
    }
    report["invalid_flag_recovery"] = True
    report.update(status="verified", parity=True, compared=compared, mlx_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "stock", "model", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = {
        "status": "started",
        "parity": False,
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "stock_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as sink:
        try:
            root = args.report.with_suffix("")
            root.mkdir()
            verify(args, root, report)
        except BaseException as error:
            report.update(status="failed", parity=False, error=repr(error))
            raise
        finally:
            json.dump(report, sink, indent=2)
            sink.write("\n")


if __name__ == "__main__":
    main()
