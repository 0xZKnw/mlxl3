# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mflux @ git+https://github.com/mflux-community/mflux.git@d39c9639e6ac22407a725432cc221a83fab19f1b", "mlx==0.32.4", "transformers==5.24.0"]
# [tool.uv]
# exclude-newer = "2026-11-22T20:00:00Z"
# ///
"""Studio's local Qwen-Image-2.1 text-to-image worker on Apple M5 Metal."""

import argparse
import hashlib
import json
import os
import platform
import sys
from functools import lru_cache, partial
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

SOURCE = "Qwen/Qwen-Image-2.1"
REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
AUX = "mlx-community/Qwen-Image-2.1-mflux-q4"
AUX_REVISION = "746a58556820933a2df5c75887a2570f1ad200c0"
PROJECTIONS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.proj",
    "img_mlp.gate_layer",
    "img_mlp.out",
)

# Depth of one TensorOps multiply for the measured 1024-row image projections.
# Keys are (input width, output width, EXL3 bits). Other geometries keep K16.
WINDOW_MATMUL_DEPTH = {
    (4096, 4096, 3): 64,
    (4096, 4096, 4): 64,
    (4096, 12288, 3): 64,
    (4096, 12288, 4): 32,
    (12288, 4096, 3): 32,
    (12288, 4096, 4): 64,
}


@lru_cache(maxsize=32)
def qmm(inputs, outputs, bits, block_rows, group):
    shared = group > 1
    op_k = WINDOW_MATMUL_DEPTH.get((inputs, outputs, bits), 16) if shared else 16
    filename = "_qmm_tensor_window_kernel.metal" if shared else "_qmm_tensor_kernel.metal"
    shader = Path(__file__).with_name(filename)
    if not shader.is_file():
        shader = Path(__file__).resolve().parents[4] / "native/shaders" / filename
    inverse = [0] * 256
    for lane in range(32):
        r, c = 2 * (lane % 4), lane // 4
        for ci, column in enumerate((c, c + 8)):
            for ri, row in enumerate((r, r + 1, r + 8, r + 9)):
                inverse[row * 16 + column] = lane * 8 + ci * 4 + ri
    header = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
inline float mlxl3_decode_codeword(uint state, int unused) {
    uint bits = ((state * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half2 pair = as_type<half2>(bits);
    return float(pair.x + pair.y);
}
"""
    header += "constant ushort mlxl3_perm_inv[256] = {" + ",".join(map(str, inverse)) + "};\n"
    for name, value in {
        "BM": block_rows,
        "STEP_TILES": 4,
        "ROW_GROUP": group,
        "BN": 32,
        "BK": 16,
        "OP_K": op_k,
        "INPUT_DIMS": inputs,
        "OUTPUT_DIMS": outputs,
        "K_BITS": bits,
        "PACKED_U32": 8 * bits,
        "TILES_N": outputs // 16,
        "WEIGHT_TILE_OFFSET": 0,
    }.items():
        header += f"#define {name} {value}u\n"
    return mx.fast.metal_kernel(
        name=f"image21_qmm_{inputs}_{outputs}_{bits}_{block_rows}_{group}_depth{op_k}",
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        header=header,
        source=shader.read_text(),
    )


class EXL3Linear(nn.Module):
    def __init__(self, weights, name, shape, bits):
        super().__init__()
        outputs, inputs = shape
        self.suh = weights[name + ".suh"]
        self.svh = weights[name + ".svh"]
        packed = weights[name + ".trellis"]
        marker = weights[name + ".mcg"]
        if (
            bits not in (3, 4, 5)
            or inputs % 128
            or outputs % 128
            or packed.dtype != mx.int16
            or packed.shape != (inputs // 16, outputs // 16, 16 * bits)
            or self.suh.shape != (inputs,)
            or self.svh.shape != (outputs,)
            or self.suh.dtype != mx.float16
            or self.svh.dtype != mx.float16
            or marker.shape != ()
            or marker.dtype != mx.int32
            or (marker.item() & 0xFFFFFFFF) != 0xCBAC1FED
        ):
            raise ValueError(f"Invalid EXL3 dimensions or codebook in {name}")
        self.words = packed.reshape(-1).view(mx.uint32)
        self.bits = bits

    def __call__(self, x):
        k, n = self.suh.size, self.svh.size
        if x.shape[-1] != k:
            raise ValueError("EXL3 input width mismatch")
        rows = x.size // k
        # The window kernel was qualified for the 512 px image-token geometry.
        block_rows, group = (64, 4) if rows == 1024 else (32, 1)
        blocks = (rows + block_rows - 1) // block_rows
        padded = blocks * block_rows
        rotated = mx.hadamard_transform((x.astype(mx.float16) * self.suh).reshape(-1, 128)).reshape(
            rows, k
        )
        if padded != rows:
            rotated = mx.pad(rotated, [(0, padded - rows), (0, 0)])
        y = qmm(k, n, self.bits, block_rows, group)(
            inputs=[rotated, self.words],
            grid=(n * group, (blocks + group - 1) // group, 1),
            threadgroup=(32 * group, 1, 1),
            output_shapes=[(padded, n)],
            output_dtypes=[mx.float16],
        )[0]
        return (
            mx.hadamard_transform(y[:rows].reshape(-1, 128)).reshape(*x.shape[:-1], n) * self.svh
        ).astype(x.dtype)


def load_denoiser(model, folder):
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

    m = json.loads((folder / "manifest.json").read_text())
    if m["source"] != SOURCE or m["source_revision"] != REVISION:
        raise ValueError("Expected the base Qwen-Image-2.1 EXL3 checkpoint")
    expected = {f"transformer_blocks.{i}.{p}" for i in range(32) for p in PROJECTIONS}
    files = {f"block-{i:02}.safetensors" for i in range(32)} | {"protected.safetensors"}
    if (
        len(m["parts"]) != len(files)
        or {p["file"] for p in m["parts"]} != files
        or set(m["recipe"]) != expected
    ):
        raise ValueError("Incomplete EXL3 checkpoint")
    params = dict(tree_flatten(model.parameters()))
    dense = {}
    replacements = []
    seen = set()
    for part in m["parts"]:
        path = folder / part["file"]
        with path.open("rb") as reader:
            if hashlib.file_digest(reader, "sha256").hexdigest() != part["sha256"]:
                raise ValueError(f"Checksum mismatch in {path.name}")
        values = mx.load(str(path))
        names = part["packed"]
        required = {
            name + "." + suffix for name in names for suffix in ("suh", "svh", "mcg", "trellis")
        }
        if (
            set(values) != required | set(part["dense"])
            or seen.intersection(names)
            or set(dense).intersection(part["dense"])
        ):
            raise ValueError(f"Unexpected or duplicate tensor in {path.name}")
        seen.update(names)
        dense.update({k: values[k] for k in part["dense"]})
        for name in names:
            if name not in expected:
                raise ValueError(f"Unknown projection {name}")
            replacements.append(
                (
                    name,
                    EXL3Linear(
                        values, name, params[name + ".weight"].shape, m["recipe"][name]["bits"]
                    ),
                )
            )
    dense = Qwen21Initializer._normalize_transformer_weights(dense)
    learned = {k for k in params if k.endswith((".weight", ".bias"))}
    if seen != expected or set(dense) != (learned - {k + ".weight" for k in expected}):
        raise ValueError("Checkpoint does not cover all denoiser parameters")
    if any(v.shape != params[k].shape or v.dtype != mx.bfloat16 for k, v in dense.items()):
        raise ValueError("Incorrect protected tensor dimensions or dtype")
    model.load_weights(list(dense.items()), strict=False)
    from mlx.utils import tree_unflatten

    model.update_modules(tree_unflatten(replacements))
    model.use_packed_projections = False


def build(folder):
    from dataclasses import replace

    from huggingface_hub import snapshot_download
    from mflux.models.common.compute_precision import ComputePrecision
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer as Init
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    class AuxWeights(Qwen21WeightDefinition):
        @staticmethod
        def get_components():
            return [
                replace(c, skip_quantization=False)
                for c in Qwen21WeightDefinition.get_components()
                if c.name != "transformer"
            ]

    model = QwenImage21.__new__(QwenImage21)
    nn.Module.__init__(model)
    Init.init_config(model, ModelConfig.qwen_image_21())
    model.small_text_encoder = False
    Init._init_models(model)
    print("Loading EXL3 denoiser", flush=True)
    load_denoiser(model.transformer, folder)
    print("Loading tokenizer, text encoder and VAE", flush=True)
    root = Path(
        snapshot_download(
            AUX,
            revision=AUX_REVISION,
            allow_patterns=["processor/*", "text_encoder/*", "vae/*", "LICENSE"],
        )
    )
    Init._init_tokenizers(model, str(root))
    Init.load_components(model, root, AuxWeights, None, validate=True)
    Init.apply_lora(model, None, None, True)
    model.compute_precision = ComputePrecision(None)
    return model


def require_supported_gpu():
    version = platform.mac_ver()[0].split(".")
    if (
        version[0].isdigit()
        and tuple(map(int, (version + ["0"])[:2])) >= (26, 3)
        and "MLXL3_DISABLE_TENSOR_OPS" not in os.environ
        and mx.metal.is_available()
    ):
        device = mx.device_info()
        name = device["device_name"]
        if name == "Apple M5" or name.startswith("Apple M5 "):
            return device
    raise RuntimeError(
        "Qwen Image requires an Apple M5 GPU and macOS 26.3 or later with TensorOps enabled."
    )


def checked_decode(decode, latent):
    # Check on both sides of the VAE, before MFLUX clips and casts pixels to uint8.
    if not latent.size or not mx.all(mx.isfinite(latent)).item():
        raise ValueError("Image generation produced empty or non-finite latents")
    pixels = decode(latent)
    if not pixels.size or not mx.all(mx.isfinite(pixels)).item():
        raise ValueError("Image generation produced empty or non-finite pixels")
    return pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, choices=range(2, 51), metavar="2..50", default=40)
    parser.add_argument("--size", type=int, default=512, choices=(512, 768, 1024))
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "Limit the free Metal buffer cache to 512 MiB at 1024 px on Apple M5 with "
            "24 GiB of memory. This can increase render time."
        ),
    )
    args = parser.parse_args()
    if not args.prompt.strip() or not 0 <= args.seed < 2**32:
        parser.error("Invalid prompt or seed")
    device = require_supported_gpu()
    print(device["device_name"], flush=True)
    from mflux.callbacks.instances.memory_saver import MemorySaver

    if args.size == 512 or (
        (args.size == 768 or (args.size == 1024 and args.low_memory))
        and device["device_name"] == "Apple M5"
        and device.get("memory_size") == 24 * 1024**3
    ):
        # Limit unused Metal buffers before the worker loads the model.
        mx.set_cache_limit(512 * 1024**2)
    model = build(args.model)
    # Studio launches this worker for one image. Each component can be released after its phase.
    model.callbacks.register(MemorySaver(model, keep_transformer=False, cache_limit_bytes=None))
    model.vae.decode = partial(checked_decode, model.vae.decode)
    im = model.generate_image(
        seed=args.seed,
        prompt=args.prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
    )
    image = im.image
    if image.size != (args.size, args.size):
        raise ValueError("Unexpected generated image dimensions")
    image.save(args.output, format="PNG")


if __name__ == "__main__" and sys.argv[1:] == ["--interpreter"]:
    require_supported_gpu()
    print(sys.executable)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
