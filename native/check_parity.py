"""Compare the native executable against the unchanged Python codec oracles.

CPU checks need NumPy only. --metal additionally uses the repository's MLX
environment and a real Apple GPU; it never loads an entire language model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def source_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/mlxl3/codec" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/mlxl3-rs")
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--mlx", action="store_true")
    args = parser.parse_args()
    process = subprocess.Popen([str(args.binary), "codec"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cases = 0

    def request(op, **payload):
        process.stdin.write(json.dumps({"op": op, **payload}) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        assert line, "native codec process exited before replying"
        value = json.loads(line)
        assert "error" not in value, value
        return value["data"]

    try:
        trellis = source_module("trellis")
        codebook = source_module("codebook")
        perm = source_module("perm")
        np.testing.assert_array_equal(request("permutation"), perm.permutation())
        cases += 1
        words = np.arange(65536, dtype=np.uint16)
        for mode in range(3):
            actual = np.asarray(request("codebook", mode=mode, data=words.tolist()), dtype=np.uint16)
            expected = codebook.decode_codewords(words, mode).view(np.uint16)
            np.testing.assert_array_equal(actual, expected)
            cases += 1
        for k in range(1, 9):
            rng = np.random.default_rng(0xE13 + k)
            encoded = rng.integers(0, 65536, size=(6, 256), dtype=np.uint16)
            packed = trellis.pack_trellis(encoded, k)
            native = request("pack", k=k, data=encoded.ravel().tolist())
            np.testing.assert_array_equal(native, packed.ravel())
            states = trellis.unpack_trellis(packed, k).ravel()
            np.testing.assert_array_equal(request("unpack", k=k, data=native), states)
            cases += 2
            if args.metal:
                np.testing.assert_array_equal(request("metal-pack", k=k, data=encoded.ravel().tolist()), native)
                cases += 1
                for mode in range(3):
                    result = request("metal-decode", k=k, mode=mode, data=native)
                    np.testing.assert_array_equal(result["states"], states)
                    np.testing.assert_array_equal(result["values"], codebook.decode_codewords(states, mode).view(np.uint16))
                    cases += 1
        if args.metal:
            import mlx.core as mx
            from mlxl3.kernels.qmv import _qmv_inner_kernel

            for rows, cols in [(32, 48), (128, 128), (256, 64)]:
                for k in range(1, 9):
                    rng = np.random.default_rng(rows + cols + k)
                    encoded = rng.integers(0, 65536, size=(rows // 16, cols // 16, 256), dtype=np.uint16)
                    packed = trellis.pack_trellis(encoded, k)
                    x = rng.normal(size=rows).astype(np.float16)
                    for mode in range(3):
                        expected = _qmv_inner_kernel(k, mode, rows, cols)(
                            inputs=[mx.array(x), mx.array(packed)],
                            template=[("K", k), ("CB", mode), ("INPUT_DIMS", rows), ("TILES_N", cols // 16)],
                            grid=(cols * 128, 1, 1), threadgroup=(128, 1, 1),
                            output_shapes=[(cols,)], output_dtypes=[mx.float16],
                        )[0]
                        expected = np.asarray(expected).view(np.uint16)
                        actual = request("metal-qmv", k=k, mode=mode, cols=cols,
                                         data=packed.ravel().tolist(), x=x.view(np.uint16).tolist())
                        np.testing.assert_array_equal(actual, expected, err_msg=f"QMV {rows}x{cols} K{k} CB{mode}")
                        cases += 1
        if args.mlx:
            import mlx.core as mx
            from mlxl3.moe import router_topk
            from mlxl3.kernels.qmv import qmv_exl3, qmv_exl3_grouped
            from mlx_lm.models.gated_delta import gated_delta_kernel
            for rows, cols in [(128, 128), (1024, 512), (2048, 128)]:
                for k in range(1, 9):
                    rng = np.random.default_rng(rows + cols + k)
                    packed = trellis.pack_trellis(rng.integers(0, 65536,
                        size=(rows // 16, cols // 16, 256), dtype=np.uint16), k)
                    x = rng.normal(size=rows).astype(np.float16)
                    suh = rng.uniform(-0.2, 0.2, rows).astype(np.float16)
                    svh = rng.uniform(-0.2, 0.2, cols).astype(np.float16)
                    for mode in range(3):
                        expected = np.asarray(qmv_exl3(mx.array(x), mx.array(packed),
                            mx.array(suh), mx.array(svh), k, mode)).view(np.uint16)
                        actual = request("mlx-linear", k=k, mode=mode, cols=cols,
                            data=packed.ravel().tolist(), x=x.view(np.uint16).tolist(),
                            suh=suh.view(np.uint16).tolist(), svh=svh.view(np.uint16).tolist())
                        np.testing.assert_array_equal(actual, expected,
                            err_msg=f"MLX projection {rows}x{cols} K{k} CB{mode}")
                        cases += 1
            for rows, widths in [(2048, (2048, 512, 512)), (1024, (256, 512))]:
                for k in (1, 2, 3, 4, 5, 6, 8):
                    rng = np.random.default_rng(rows + k)
                    packed = [trellis.pack_trellis(rng.integers(0, 65536,
                        size=(rows//16, width//16, 256), dtype=np.uint16), k) for width in widths]
                    x = rng.normal(size=rows).astype(np.float16)
                    suh = rng.uniform(-0.2, 0.2, (len(widths), rows)).astype(np.float16)
                    svh = rng.uniform(-0.2, 0.2, sum(widths)).astype(np.float16)
                    tile_sub = np.repeat(np.arange(len(widths), dtype=np.uint32), np.asarray(widths)//16)
                    for mode in range(3):
                        expected = qmv_exl3_grouped(mx.array(x), mx.array(np.concatenate(packed, axis=1)),
                            mx.array(suh), mx.array(svh), mx.array(tile_sub), output_dims=widths, k=k, mode=mode)
                        actual = request("mlx-group", k=k, mode=mode, widths=widths,
                            data=np.concatenate([p.ravel() for p in packed]).tolist(), x=x.view(np.uint16).tolist(),
                            suh=suh.view(np.uint16).ravel().tolist(), svh=svh.view(np.uint16).tolist())
                        for index, reference in enumerate(expected):
                            np.testing.assert_array_equal(actual[index], np.asarray(reference).view(np.uint16).ravel(),
                                err_msg=f"MLX grouped {rows}x{widths} K{k} CB{mode}, projection {index}")
                        cases += 1
            for seed, initial_state in ((41, False), (42, True)):
                rng = np.random.default_rng(seed)
                key_heads, value_heads, value_dims = 1, 2, 8
                q = rng.normal(size=(1, 1, key_heads, 128)).astype(np.float16)
                key = rng.normal(size=q.shape).astype(np.float16)
                value = rng.normal(size=(1, 1, value_heads, value_dims)).astype(np.float16)
                decay = rng.uniform(0.01, 0.99, size=(1, 1, value_heads)).astype(np.float32)
                beta = rng.uniform(0.01, 0.99, size=(1, 1, value_heads)).astype(np.float16)
                state = (rng.normal(size=(1, value_heads, value_dims, 128)).astype(np.float32)
                         if initial_state else np.zeros((1, value_heads, value_dims, 128), dtype=np.float32))
                expected_output, expected_state = gated_delta_kernel(
                    mx.array(q), mx.array(key), mx.array(value), mx.array(decay),
                    mx.array(beta), state=mx.array(state))
                actual = request("mlx-gdn", key_heads=key_heads, value_heads=value_heads,
                    value_dims=value_dims, q=q.view(np.uint16).ravel().tolist(),
                    x=key.view(np.uint16).ravel().tolist(), v=value.view(np.uint16).ravel().tolist(),
                    g=decay.ravel().tolist(), beta=beta.view(np.uint16).ravel().tolist(),
                    state=state.ravel().tolist())
                np.testing.assert_array_equal(np.asarray(actual["output"], dtype=np.uint16),
                    np.asarray(expected_output).view(np.uint16).ravel(),
                    err_msg=f"Gated DeltaNet output, nonzero state={initial_state}")
                np.testing.assert_array_equal(np.asarray(actual["state"], dtype=np.float32).view(np.uint32),
                    np.asarray(expected_state).astype(np.float32).ravel().view(np.uint32),
                    err_msg=f"Gated DeltaNet state, nonzero state={initial_state}")
                cases += 1
            router_cases = [
                np.random.default_rng(51).uniform(0, 1, 256).astype(np.float16),
                np.array([0.5] * 4 + [0.25] * 4 + [0.0] * 8, dtype=np.float16),
                np.array([np.nan, -1.0, 0.0, -0.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float16),
            ]
            for values in router_cases:
                for normalize in (False, True):
                    top_k = min(8, values.size)
                    expected_indices, expected_scores = router_topk(
                        mx.array(values[None]), top_k, normalize=normalize)
                    actual = request("mlx-router", k=top_k, normalize=normalize,
                        data=values.view(np.uint16).tolist())
                    np.testing.assert_array_equal(np.asarray(actual["indices"], dtype=np.uint32),
                        np.asarray(expected_indices).astype(np.uint32).ravel(),
                        err_msg=f"MoE router indices, normalize={normalize}")
                    np.testing.assert_array_equal(np.asarray(actual["scores"], dtype=np.uint16),
                        np.asarray(expected_scores).view(np.uint16).ravel(),
                        err_msg=f"MoE router scores, normalize={normalize}")
                    cases += 1
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stderr = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
    assert process.returncode == 0, stderr
    print(json.dumps({"status": "passed", "cases": cases, "metal": args.metal, "mlx": args.mlx,
                      "comparison": "exact uint16 / float16 bit patterns", "performance": "not measured"}))


if __name__ == "__main__":
    main()
