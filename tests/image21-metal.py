"""Physical Metal oracle for the same EXL3 projection used by Studio."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Resources/Image21"))
from render import PROJECTIONS, REVISION, SOURCE, EXL3Linear, load_denoiser

rng = np.random.default_rng(371)
for bits in (3, 4, 5):
    rows, cols = 128, 256
    packed = rng.integers(0, 65536, size=(rows // 16, cols // 16, 16 * bits), dtype=np.uint16)
    w = np.empty((rows, cols), dtype=np.float32)
    # Decode each circular bitstream independently of the kernel's window/address calculations.
    for a, b in np.ndindex(packed.shape[:2]):
        stream = np.unpackbits(packed[a, b].view("<u4").astype(">u4").view("u1"))
        for i in range(256):
            end = (i + 1) * bits
            state = int.from_bytes(
                np.packbits(stream[np.arange(end - 16, end) % len(stream)]).tobytes(), "big"
            )
            code = (state * 0xCBAC1FED & 0x8FFF8FFF) ^ 0x3B603B60
            val = np.array([code], dtype="<u4").view("<f2").sum(dtype=np.float16)
            lane, inner = divmod(i, 8)
            r = 2 * (lane % 4) + (0, 1, 8, 9)[inner % 4]
            c = lane // 4 + 8 * (inner // 4)
            w[a * 16 + r, b * 16 + c] = val
    h = np.array([[1]], dtype=np.float32)
    while len(h) < 128:
        h = np.block([[h, h], [h, -h]])
    h /= np.sqrt(128.0)
    su = rng.uniform(0.15, 1, rows).astype("f2")
    sv = rng.uniform(0.15, 1, cols).astype("f2")
    proj = EXL3Linear(
        {
            "a.suh": mx.array(su),
            "a.svh": mx.array(sv),
            "a.trellis": mx.array(packed.view("i2")),
            "a.mcg": mx.array(-877912083, dtype=mx.int32),
        },
        "a",
        (cols, rows),
        bits,
    )
    for batch in (1, 32, 35, 127, 128, 129, 255, 256, 257):
        x = rng.normal(size=(batch, rows)).astype("f2")
        y = np.asarray(proj(mx.array(x)), dtype="f")
        want = (((x * su) @ h) @ w).reshape(-1, 128) @ h
        want = want.reshape(batch, cols) * sv
        assert np.isfinite(y).all() and y.shape == want.shape
        err = np.linalg.norm(y - want) / np.linalg.norm(want)
        assert err < 0.004, (bits, batch, err)
        if batch >= 128:
            # Splitting forces the original single-SIMD path for an independent dispatch comparison.
            original = mx.concatenate([proj(mx.array(x[i : i + 32])) for i in range(0, batch, 32)])
            np.testing.assert_array_equal(y.view("u4"), np.asarray(original, dtype="f").view("u4"))
        print(bits, batch, err)

# Exercise the production loader before it can access any auxiliary network service.
checks = unittest.TestCase()
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    manifest = root / "manifest.json"
    with checks.assertRaises(FileNotFoundError):
        load_denoiser(None, root)
    manifest.write_text(json.dumps({"source": "wrong", "source_revision": REVISION}))
    with checks.assertRaisesRegex(ValueError, "base Qwen"):
        load_denoiser(None, root)
    inputs = {"source": SOURCE, "source_revision": REVISION, "parts": [], "recipe": {}}
    manifest.write_text(json.dumps(inputs))
    with checks.assertRaisesRegex(ValueError, "Incomplete"):
        load_denoiser(None, root)
    files = [f"block-{i:02}.safetensors" for i in range(32)] + ["protected.safetensors"]
    inputs["parts"] = [{"file": f, "sha256": "0" * 64} for f in files]
    inputs["recipe"] = {
        f"transformer_blocks.{i}.{p}": {"bits": 4} for i in range(32) for p in PROJECTIONS
    }
    manifest.write_text(json.dumps(inputs))
    (root / files[0]).write_bytes(b"corrupted shard")
    with checks.assertRaisesRegex(ValueError, "Checksum mismatch"):
        load_denoiser(proj, root)
    inputs["parts"][0]["file"] = "../outside.safetensors"
    manifest.write_text(json.dumps(inputs))
    with checks.assertRaisesRegex(ValueError, "Incomplete"):
        load_denoiser(None, root)
with checks.assertRaisesRegex(ValueError, "input width"):
    proj(mx.zeros((1, 129)))
print("6 invalid inputs rejected")
