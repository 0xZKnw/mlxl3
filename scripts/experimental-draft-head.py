"""Build a disposable full-vocabulary affine4 MTP projection from an EXL3 head.

The target checkpoint is read only. Both Hadamard rotations and scales are
included before quantization; output chunks must contain whole 128-row blocks.
This changes draft proposals, so acceptance and target parity must be tested.
"""

import argparse
import json
import struct
import tempfile
import time
from pathlib import Path

import numpy as np


def chunk_end(start: int, total: int, chunk: int) -> int:
    """Choose the end of a complete output-rotation block.

    pre: 0 <= start < total
    pre: start % 128 == 0 and total % 128 == 0
    pre: chunk > 0 and chunk % 128 == 0
    post: start < __return__ <= total
    post: __return__ % 128 == 0
    post: __return__ - start <= chunk
    """
    if not (
        0 <= start < total and start % 128 == total % 128 == 0 and chunk > 0 and chunk % 128 == 0
    ):
        raise ValueError("chunks must contain whole 128-row blocks")
    return min(start + chunk, total)


def decode_inner(packed, k, mcg):
    """EXL3 trellis to [input, output], without rotations, FP16 codewords."""
    if (
        type(k) is not int
        or not 1 <= k <= 8
        or packed.ndim != 3
        or packed.shape[2] != 16 * k
        or min(packed.shape) <= 0
        or packed.dtype not in (np.dtype("uint16"), np.dtype("int16"))
    ):
        raise ValueError("invalid EXL3 trellis")
    tiles = packed.view(np.uint16).reshape(-1, 16 * k)
    pairs = tiles[:, ::2].astype(np.uint32) | (tiles[:, 1::2].astype(np.uint32) << 16)
    state = np.arange(256)
    b0 = state * k + k + 256 * k - 16
    b1 = b0 + 16
    i0, i1 = b0 // 32, (b1 - 1) // 32
    words = (
        (
            (pairs[:, i0 % (8 * k)].astype(np.uint64) << 32)
            | pairs[:, i1 % (8 * k)].astype(np.uint64)
        )
        >> ((i1 + 1) * 32 - b1).astype(np.uint64)
    ) & 65535
    words = words.astype(np.uint32)
    bits = (
        words * np.uint32(0xCBAC1FED)
        if mcg
        else (words * np.uint32(89_226_354) + np.uint32(64_248_484))
    )
    bits = np.uint32(0x3B603B60) ^ (bits & np.uint32(0x8FFF8FFF))
    values = (bits & 65535).astype(np.uint16).view(np.float16).astype(np.float32) + (
        bits >> 16
    ).astype(np.uint16).view(np.float16).astype(np.float32)
    inverse = np.empty(256, dtype=int)
    for lane in range(32):
        r, c = lane % 4 * 2, lane // 4
        for cg, col in enumerate((c, c + 8)):
            for ri, row in enumerate((r, r + 1, r + 8, r + 9)):
                inverse[row * 16 + col] = lane * 8 + cg * 4 + ri
    return (
        values.astype(np.float16)[:, inverse]
        .reshape(packed.shape[0], packed.shape[1], 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(packed.shape[0] * 16, packed.shape[1] * 16)
    )


def rotate128(matrix):
    """Normalized Sylvester Hadamard over independent final-axis blocks."""
    if matrix.ndim != 2 or matrix.shape[1] == 0 or matrix.shape[1] % 128:
        raise ValueError("rotation requires 128-aligned matrix rows")
    shape = matrix.shape
    result = matrix.astype(np.float32).reshape(-1, 128).copy()
    width = 1
    while width < 128:
        blocks = result.reshape(-1, 128 // (2 * width), 2, width)
        left, right = blocks[:, :, 0].copy(), blocks[:, :, 1].copy()
        blocks[:, :, 0], blocks[:, :, 1] = left + right, left - right
        width *= 2
    return (result / np.sqrt(np.float32(128))).reshape(shape)


def effective_weight(packed, k, mcg, suh, svh):
    inner = decode_inner(packed, k, mcg)
    if (
        suh.shape != (inner.shape[0],)
        or svh.shape != (inner.shape[1],)
        or not np.isfinite(suh).all()
        or not np.isfinite(svh).all()
    ):
        raise ValueError("invalid EXL3 scales")
    return rotate128(rotate128(inner).T) * suh.astype(np.float32) * svh[:, None]


def compact_draft_ids(ids: list[int], vocab: int) -> list[int]:
    """Validate a strict subset and pad to whole 64-row output tiles.

    raises: ValueError
    post: 0 < len(__return__) < vocab and len(__return__) % 64 == 0
    post: all(0 <= i < vocab for i in __return__)
    post: all(__return__[i-1] < __return__[i] for i in range(1, len(__return__)))
    post: all(i in __return__ for i in ids)
    """
    if (
        type(vocab) is not int
        or vocab <= 64
        or vocab % 64
        or not ids
        or len(ids) >= vocab
        or any(type(i) is not int or not 0 <= i < vocab for i in ids)
        or len(set(ids)) != len(ids)
        or (len(ids) + 63) // 64 * 64 >= vocab
    ):
        raise ValueError("draft IDs must be a unique in-range strict subset")
    selected = set(ids)
    candidate = 0
    while len(selected) % 64:
        if candidate not in selected:
            selected.add(candidate)
        candidate += 1
    return sorted(selected)


def prune_projection(source: Path, output: Path, vocab_file: Path):
    """Copy selected Q4 rows on CPU; activate only a completed paired head."""
    from safetensors.numpy import load_file, save_file

    metadata = json.loads((source / "draft_head_q4.json").read_text())
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"input", "output", "bits", "group_size", "mode"}
        or any(type(metadata[k]) is not int for k in ("input", "output", "bits", "group_size"))
        or not 0 < metadata["input"] <= 2**31 - 1
        or not 0 < metadata["output"] <= 2**31 - 1
        or metadata["input"] % 64
        or metadata["bits"] != 4
        or metadata["group_size"] != 64
        or metadata["mode"] != "affine"
    ):
        raise ValueError("requires a complete full-vocabulary affine4/group64 projection")
    if vocab_file.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("draft vocabulary file exceeds 2 MiB")
    ids = compact_draft_ids([int(i) for i in vocab_file.read_text().split()], metadata["output"])
    h, rows = metadata["input"], metadata["output"]
    tensors = load_file(source / "draft_head_q4.safetensors")
    expected = {
        "lm_head.weight": (np.uint32, h // 8),
        "lm_head.scales": (np.float16, h // 64),
        "lm_head.biases": (np.float16, h // 64),
    }
    if set(tensors) != set(expected) or any(
        tensors[name].dtype != dtype
        or tensors[name].shape != (rows, width)
        or not np.isfinite(tensors[name]).all()
        for name, (dtype, width) in expected.items()
    ):
        raise ValueError("invalid full Q4 tensor shapes, dtypes or values")
    if not np.any(tensors["lm_head.scales"][ids] != 0):
        raise ValueError("zeroed compact projection")
    for name in ("config.json", "model.safetensors"):
        if not (source / name).is_file():
            raise ValueError("place the projection beside its original MTP config and weights")
    names = (
        "config.json",
        "model.safetensors",
        "draft_head_q4.safetensors",
        "conversion.json",
        "draft_head_q4.json",
    )
    for name in names:
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError(output / name)
    output.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with tempfile.TemporaryDirectory(prefix="draft-q4-", dir=output) as temporary:
            staged = Path(temporary) / "projection.safetensors"
            save_file({name: value[ids] for name, value in tensors.items()}, staged)
            for name in names[:2]:
                target = output / name
                target.symlink_to((source / name).resolve())
                created.append(target)
            destination = output / names[2]
            destination.hardlink_to(staged)
            created.append(destination)
            report = {
                "status": "converted",
                "source": str(source.resolve()),
                "input": h,
                "output": len(ids),
                "target_vocabulary": rows,
                "bytes": destination.stat().st_size,
                "token_ids_source": str(vocab_file.resolve()),
                "correctness": "unverified",
            }
            for name, data in [
                ("conversion.json", report),
                ("draft_head_q4.json", {**metadata, "output": len(ids), "token_ids": ids}),
            ]:
                target = output / name
                with target.open("x") as stream:
                    created.append(target)
                    stream.write(json.dumps(data) + "\n")
    except BaseException:
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise
    print(json.dumps(report))


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument(
        "--draft-vocab", type=Path, help="prune an existing paired Q4 head using this token-ID list"
    )
    args = parser.parse_args()
    if args.draft_vocab is not None:
        prune_projection(args.model, args.output, args.draft_vocab)
        return
    import mlx.core as mx
    from safetensors import safe_open

    started = time.perf_counter()
    entries = {}
    for path in sorted(args.model.glob("*.safetensors")):
        # Reuse safetensors' header/range validation without reading weights.
        with safe_open(path, framework="numpy"):
            pass
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for name, info in header.items():
            if name.startswith("lm_head."):
                if name in entries:
                    raise ValueError(f"duplicate head tensor: {name}")
                entries[name] = (path, size + 8, info)
    if not {"lm_head.trellis", "lm_head.suh", "lm_head.svh"} <= entries.keys():
        raise ValueError("missing EXL3 head tensors")
    if "lm_head.mul1" in entries or "lm_head.bias" in entries:
        raise ValueError("prototype supports only unbiased Default/MCG EXL3 heads")

    def tensor(name):
        path, offset, info = entries["lm_head." + name]
        dtype = {"I16": "<i2", "U16": "<u2", "F16": "<f2"}[info["dtype"]]
        return np.memmap(
            path,
            dtype=dtype,
            mode="r",
            shape=tuple(info["shape"]),
            offset=offset + info["data_offsets"][0],
        )

    packed, suh, svh = tensor("trellis"), tensor("suh"), tensor("svh")
    vocab = len(svh)
    chunk_end(0, vocab, args.chunk)
    if packed.shape[:2] != (len(suh) // 16, vocab // 16) or len(suh) % 128:
        raise ValueError("EXL3 head dimensions mismatch")
    if args.output.resolve() == args.model.resolve():
        raise ValueError("output must be separate from the target checkpoint")
    for name in (
        "draft_head_q4.safetensors",
        "draft_head_q4.json",
        "conversion.json",
        "reference.json",
    ):
        if (args.output / name).exists() or (args.output / name).is_symlink():
            raise FileExistsError(args.output / name)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "draft_head_q4.safetensors"
    parts = {name: [] for name in ("weight", "scales", "biases")}
    squared_error = squared_weight = count = 0
    reference = None
    with tempfile.TemporaryDirectory(prefix="draft-q4-", dir=args.output) as temporary:
        for start in range(0, vocab, args.chunk):
            end = chunk_end(start, vocab, args.chunk)
            weight = effective_weight(
                packed[:, start // 16 : end // 16],
                packed.shape[2] // 16,
                "lm_head.mcg" in entries,
                suh,
                svh[start:end],
            )
            if not np.isfinite(weight).all():
                raise ValueError("nonfinite reconstructed head")
            if start == 0:
                inputs = np.array(
                    [((np.arange(len(suh)) + seed) % 251 - 125) / 128 for seed in (0, 17, 101)],
                    dtype=np.float16,
                )
                reference = {"inputs": inputs.tolist(), "logits": (inputs @ weight.T).tolist()}
            q, s, b = mx.quantize(mx.array(weight.astype(np.float16)), group_size=64, bits=4)
            mx.eval(q, s, b)
            restored = np.array(mx.dequantize(q, s, b, group_size=64, bits=4)).astype(np.float32)
            if not np.isfinite(restored).all() or not np.any(np.array(s) != 0):
                raise ValueError("invalid/zeroed affine draft head")
            squared_error += float(np.sum((restored - weight) ** 2, dtype=np.float64))
            squared_weight += float(np.sum(weight**2, dtype=np.float64))
            count += weight.size
            chunk_path = Path(temporary) / f"{start}.safetensors"
            mx.save_safetensors(
                str(chunk_path),
                {"lm_head.weight": q, "lm_head.scales": s, "lm_head.biases": b},
                metadata={"format": "mlx"},
            )
            loaded = mx.load(str(chunk_path))
            for name, arrays in parts.items():
                arrays.append(loaded["lm_head." + name])
            del weight, restored, q, s, b
            mx.clear_cache()
        combined = {"lm_head." + name: mx.concatenate(arrays) for name, arrays in parts.items()}
        mx.eval(combined)
        created = []
        try:
            staged = Path(temporary) / "complete.safetensors"
            mx.save_safetensors(str(staged), combined, metadata={"format": "mlx"})
            destination.hardlink_to(staged)
            created.append(destination)
            report = {
                "status": "converted",
                "correctness": "unverified",
                "input": len(suh),
                "output": vocab,
                "bits": 4,
                "group_size": 64,
                "chunk": args.chunk,
                "mse": squared_error / count,
                "relative_rmse": (squared_error / squared_weight) ** 0.5,
                "bytes": destination.stat().st_size,
                "conversion_seconds": time.perf_counter() - started,
            }
            # Activation is last; no existing file or symlink is overwritten.
            for name, data in [
                ("conversion.json", report),
                ("reference.json", reference),
                (
                    "draft_head_q4.json",
                    {
                        "input": len(suh),
                        "output": vocab,
                        "bits": 4,
                        "group_size": 64,
                        "mode": "affine",
                    },
                ),
            ]:
                target = args.output / name
                with target.open("x") as stream:
                    created.append(target)
                    stream.write(json.dumps(data) + "\n")
        except BaseException:
            for target in reversed(created):
                target.unlink(missing_ok=True)
            raise
    print(json.dumps(report))


if __name__ == "__main__":
    main()
