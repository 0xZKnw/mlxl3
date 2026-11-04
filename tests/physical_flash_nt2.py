"""Qualify native resident NT2 and static-K defaults against independent stock."""

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


def fixtures(root, model, *, fixed_dims=False):
    rng = np.random.default_rng(75112)
    native, stock = [], []

    def add(folder, prefix, width, cols, nt, *, generic=False, batch=1, fixed_key=None):
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
                    fixed_key=fixed_key,
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
    if fixed_dims:
        for name, width, cols, key in [
            ("model.language_model.layers.0.mlp.experts.0.gate_proj", 2560, 640, 0),
            ("model.language_model.layers.0.mlp.experts.0.up_proj", 2560, 640, 0),
            ("model.language_model.layers.0.mlp.experts.0.down_proj", 640, 2560, 1),
            ("model.language_model.layers.3.self_attn.indexer.index_qk_proj", 2560, 640, 0),
        ]:
            add(model, name, width, cols, 1, batch=3, fixed_key=key)
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


def static_counts(result: dict, enabled: bool) -> list[int]:
    """Require a complete actual-submission vector for each selected static PSO.

    raises: AssertionError, KeyError, TypeError
    post: len(__return__) == 5
    post: all(n >= 0 for n in __return__)
    """
    assert result["static_k"] is enabled and result["cases"]
    assert len(result["static_submissions"]) == 5 and all(
        type(n) is int and n >= 0 for n in result["static_submissions"]
    )
    total = [0] * 5
    for case in result["cases"]:
        key, counts = case["static_key"], case["static_submissions"]
        assert key is None or (enabled and type(key) is int and 0 <= key < 5)
        assert len(counts) == 5 and all(type(n) is int and n >= 0 for n in counts)
        assert counts == [int(key == index) for index in range(5)]
        total = [a + b for a, b in zip(total, counts, strict=True)]
    assert result["static_submissions"] == total
    return total


def fixed_counts(result: dict, cases: list[dict], enabled: bool) -> list[int]:
    """Require real fixed-PSO submissions, including nonempty real indexer evidence.

    raises: AssertionError, KeyError, TypeError
    post: len(__return__) == 2
    post: all(n >= 0 for n in __return__)
    """
    assert result["fixed_dims"] is enabled and len(result["cases"]) == len(cases) > 0
    total = [0, 0]
    indexer = False
    for item, case in zip(result["cases"], cases, strict=True):
        expected = case.get("fixed_key") if enabled else None
        assert expected is None or (type(expected) is int and 0 <= expected < 2)
        counts = item["fixed_submissions"]
        assert item["fixed_key"] == expected
        assert len(counts) == 2 and all(type(n) is int and n >= 0 for n in counts)
        assert counts == [int(expected == i) for i in range(2)]
        if expected is not None:
            assert item["static_key"] is None and item["static_submissions"] == [0] * 5
            indexer |= "index_qk_proj" in case.get("prefix", "")
        total = [a + b for a, b in zip(total, counts, strict=True)]
    assert result["fixed_submissions"] == total
    assert len(result["fixed_submissions"]) == 2 and all(
        type(n) is int and n >= 0 for n in result["fixed_submissions"]
    )
    assert all(n > 0 for n in total) and indexer if enabled else total == [0, 0]
    return total


def verify(args, root, report):
    fixed_dims = getattr(args, "fixed_dims", None)
    cases = fixtures(root, args.model.resolve(), fixed_dims=fixed_dims is not None)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    static_k = getattr(args, "static_k", None)
    if static_k is not None:
        env["MLXL3_NATIVE_FLASH_STATIC_K"] = str(static_k)
    if fixed_dims is not None:
        env["MLXL3_NATIVE_FLASH_FIXED_DIMS"] = str(fixed_dims)
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
    static_total = [0] * 5
    for setting in (0, 1, None):
        enabled = 1 if setting is None else setting
        label = "default" if setting is None else str(setting)
        mode_root = root / f"mode-{label}"
        mode_root.mkdir()
        (mode_root / "stock.json").write_text(json.dumps(refs))
        (mode_root / "native-cases.json").write_text(json.dumps(cases))
        native_env = {**env, "MLXL3_NT2_FIXTURES": str(mode_root), "DYLD_PRINT_LIBRARIES": "1"}
        if setting is not None:
            native_env["MLXL3_NATIVE_FLASH_NT2"] = str(setting)
        stderr = process(
            [
                str(args.native),
                "gpu::projection::physical_flash_resident_nt2",
                "--exact",
                "--ignored",
                "--test-threads=1",
            ],
            native_env,
            root / f"native-{label}.log",
        )
        images = [line.lower() for line in stderr.splitlines() if line.startswith("dyld[")]
        assert images and not any(
            key in line
            for line in images
            for key in ("/libmlx", "/libjaccl", "/site-packages/mlx/")
        )
        result = json.loads((mode_root / f"native-{enabled}.json").read_text())
        assert result["parity"] is True and result["unequal"] == 0
        assert len(result["cases"]) == len(cases)
        submissions = 0
        for i, (case, ref, item) in enumerate(zip(cases, refs, result["cases"], strict=True)):
            nt = case["candidate_nt"] if enabled else case["control_nt"]
            count = int(bool(enabled and not case["generic"] and nt == 2))
            assert item["case"] == i and item["nt"] == nt and item["nt2_submissions"] == count
            assert item["words"] == case["words"]
            submissions += count
            actual = np.fromfile(mode_root / f"native-{enabled}-{i}.bin", dtype="<u2").tolist()
            assert len(actual) == case["words"] > 0
            compared += words(actual, ref)
            if enabled:
                words(
                    actual, np.fromfile(root / "mode-0" / f"native-0-{i}.bin", dtype="<u2").tolist()
                )
            if setting is None:
                words(
                    actual, np.fromfile(root / "mode-1" / f"native-1-{i}.bin", dtype="<u2").tolist()
                )
        assert result["nt2_submissions"] == submissions and (submissions > 0) == bool(enabled)
        if fixed_dims is not None:
            fixed_counts(result, cases, bool(fixed_dims))
        counts = static_counts(result, static_k != 0)
        static_total = [a + b for a, b in zip(static_total, counts, strict=True)]
        report["modes"].append({**result, "setting": setting})
    assert all(n > 0 for n in static_total) if static_k != 0 else static_total == [0] * 5
    report["static_submissions"] = static_total
    if fixed_dims is not None:
        process(
            [
                str(args.native),
                "gpu::projection::physical_flash_resident_nt2",
                "--exact",
                "--ignored",
                "--test-threads=1",
            ],
            {**env, "MLXL3_NATIVE_FLASH_FIXED_DIMS": "true", "MLXL3_NT2_FIXTURES": str(root)},
            root / "fixed-invalid-flag.log",
        )
        assert json.loads((root / "fixed-invalid-flag.json").read_text()) == {
            "rejected": True,
            "fc_hidden": True,
            "fallback": True,
        }
        report["fixed_invalid_flag_recovery"] = True
    if static_k is not None:
        process(
            [
                str(args.native),
                "gpu::projection::physical_flash_resident_nt2",
                "--exact",
                "--ignored",
                "--test-threads=1",
            ],
            {
                **env,
                "MLXL3_NATIVE_FLASH_NT2": "1",
                "MLXL3_NATIVE_FLASH_STATIC_K": "true",
                "MLXL3_NT2_FIXTURES": str(root),
            },
            root / "static-invalid-flag.log",
        )
        assert json.loads((root / "static-invalid-flag.json").read_text()) == {
            "rejected": True,
            "fc_hidden": True,
            "fallbacks": True,
        }
        report["static_invalid_flag_recovery"] = True
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
    parser.add_argument(
        "--static-k", type=int, choices=(0, 1), help="override native enabled default"
    )
    parser.add_argument(
        "--fixed-dims",
        type=int,
        choices=(0, 1),
        help="qualify both exact K2 shapes and real indexer",
    )
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
