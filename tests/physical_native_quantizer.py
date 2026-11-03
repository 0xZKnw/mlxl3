"""Compare the production native tile converter with a separate stock Metal process."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, **options):
    result = subprocess.run(args, capture_output=True, check=False, timeout=180, **options)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-4096:])
    return result


def capture(root):
    import mlx.core as mx
    from ponyexl3.convert import metal_search as oracle

    expected = REPO / "apps/MLXL3Studio/Vendor/PonyExl3/ponyexl3/convert/metal_search.py"
    assert Path(oracle.__file__).resolve() == expected.resolve()
    values = np.fromfile(root / "tiles.f32", dtype="<f4").reshape(-1, 256)
    for k in range(2, 9):
        for cb in range(3):
            decoded, states = oracle.quantize_tiles_mlx(values, k, cb)
            mx.eval(decoded, states)
            decoded = np.asarray(decoded, dtype="<f4")
            states = np.asarray(states, dtype="<u2")
            assert (
                decoded.shape == values.shape
                and states.shape == values.shape
                and np.isfinite(decoded).all()
            )
            from ponyexl3.convert.mlx_trellis import pack_trellis_mlx

            packed = np.asarray(pack_trellis_mlx(mx.array(states), k), dtype="<u2")
            for name, data in [
                ("decoded.f32", decoded),
                ("states.u16", states),
                ("packed.u16", packed),
            ]:
                (root / f"oracle-{k}-{cb}-{name}").write_bytes(data.tobytes())
    (root / "oracle.json").write_text(
        json.dumps(
            {
                "mlx": importlib.metadata.version("mlx"),
                "source_sha256": digest(expected),
                "source": str(expected),
            }
        )
    )


def verify(cli, python, root, reference=None):
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / "verification.json"
    report = {"status": "in_progress", "cases": [], "cli_sha256": digest(cli)}
    report_path.write_text(json.dumps(report, indent=2))
    try:
        i = np.arange(256)
        values = np.stack(
            [np.zeros(256), ((i % 23) - 11) / 7, np.where(i % 2, -0.5, 0.5), np.full(256, 1e10)]
        ).astype("<f4")
        source = root / "tiles.f32"
        source.write_bytes(values.tobytes())
        if reference is None:
            env = dict(os.environ, PYTHONPATH=str(REPO / "apps/MLXL3Studio/Vendor/PonyExl3"))
            run([str(python), str(Path(__file__).resolve()), "--capture", str(root)], env=env)
            report["oracle"] = json.loads((root / "oracle.json").read_text())
        else:
            frozen = json.loads((reference / "reference.json").read_text())
            assert (
                frozen["status"] == "complete"
                and frozen["tile_shape"] == [4, 256]
                and len(frozen["cases"]) == 21
            )
            assert digest(source) == frozen["files"]["tiles.f32"]
            for name, expected in frozen["files"].items():
                assert Path(name).name == name and digest(reference / name) == expected
                if name != "tiles.f32":
                    shutil.copy2(reference / name, root / name)
            report["oracle"] = frozen["oracle"]
            report["reference_sha256"] = digest(reference / "reference.json")
        for k in range(2, 9):
            for cb in range(3):
                output = root / f"native-{k}-{cb}"
                scratch = 2 * 516 * (1 << (16 - k))
                result = run(
                    [
                        str(cli),
                        "quantize-tiles",
                        str(source),
                        str(output),
                        "--k",
                        str(k),
                        "--codebook",
                        str(cb),
                        "--max-scratch-bytes",
                        str(scratch),
                    ]
                )
                metadata = json.loads(result.stdout)
                assert (
                    metadata["schema"] == 1
                    and metadata["status"] == "complete"
                    and metadata["values"] == 1024
                    and metadata["launch_tiles"] == 2
                )
                case = {"k": k, "codebook": cb, "fields": {}}
                for name, words, dtype in [
                    ("decoded.f32", 1024, "<f4"),
                    ("states.u16", 1024, "<u2"),
                    ("packed.u16", 4 * 16 * k, "<u2"),
                ]:
                    expected, actual = root / f"oracle-{k}-{cb}-{name}", output / name
                    ref, out = np.fromfile(expected, dtype=dtype), np.fromfile(actual, dtype=dtype)
                    assert ref.size == out.size == words and words > 0
                    assert np.isfinite(ref).all() and np.isfinite(out).all()
                    unequal = int(
                        np.count_nonzero(
                            ref.view("u4" if dtype == "<f4" else "u2")
                            != out.view("u4" if dtype == "<f4" else "u2")
                        )
                    )
                    case["fields"][name] = {
                        "words": words,
                        "unequal": unequal,
                        "oracle_sha256": digest(expected),
                        "native_sha256": digest(actual),
                    }
                    if unequal:
                        report["cases"].append(case)
                        raise ValueError(
                            f"native trellis differs at K{k}/CB{cb}/{name}: {unequal}/{words}"
                        )
                report["cases"].append(case)
        # Exercise the exported Python converter entry after the complete independent CLI comparison.
        from mlxl3_quantizer.native import quantize_tiles_native

        decoded, states = quantize_tiles_native(
            values[:, ::-1][:, ::-1], 4, 1, engine=cli, max_scratch_bytes=516 * (1 << 12)
        )
        assert decoded.tobytes() == (root / "oracle-4-1-decoded.f32").read_bytes()
        assert states.tobytes() == (root / "oracle-4-1-states.u16").read_bytes()
        report["python_entry"] = "exact decoded/states with one-tile scratch split"
        # The signal can arrive during compilation or a launch. Both must leave no published result.
        cancel = root / "cancel"
        many = root / "cancel-tiles.f32"
        many.write_bytes(np.tile(values, (32, 1)).tobytes())
        process = subprocess.Popen(
            [
                str(cli),
                "quantize-tiles",
                str(many),
                str(cancel),
                "--k",
                "2",
                "--max-scratch-bytes",
                str(516 * (1 << 14)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(0.3)
            process.send_signal(signal.SIGUSR1)
            out, error = process.communicate(timeout=15)
            assert process.returncode != 0 and not cancel.exists()
            report["cancelled"] = {
                "exit": process.returncode,
                "error": error.decode("utf-8", errors="replace"),
                "published": False,
            }
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        report["status"] = "verified"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--reference", type=Path, default=REPO / "tests/fixtures/native-quantizer")
    args = parser.parse_args()
    if args.capture:
        capture(args.capture.resolve())
    elif args.cli and args.output:
        verify(
            args.cli.resolve(),
            args.python.absolute(),
            args.output.resolve(),
            args.reference.resolve(),
        )
    else:
        parser.error("provide --cli and --output")
