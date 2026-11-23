#!/usr/bin/env python3
"""Independent CPU reference for the MLXL3 Inkling 2 Mini vision path.

With Hugging Face Transformers (Inkling 2 support: 5.24.0.dev0 or later) and PyTorch on the CPU, for one
picture (a deterministic PNG written to OUT_DIR, or --image):

- `pixel_values` and `grid` from the checkpoint's own processor (`AutoProcessor`, processor_config.json),
  flattened in the encoder's patch order (`reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3)`);
- `input_ids` of a user message holding the picture then the prompt, rendered by that processor
  (chat template, `add_generation_prompt=True`, one placeholder per image token);
- `features_f32` / `features_bf16`: `Inkling2VisionModel` with the checkpoint's `model.visual.*` tensors
  (renamed as Transformers' conversion mapping does, strict load), run in FP32 and in BF16.

Writes OUT_DIR/inkling2_vision_oracle.json, read by `inkling2_vision::gpu_tests`
(MLXL3_INKLING_VISION_ORACLE). Nothing runs on the GPU.

usage: python -I benchmarks/inkling2_vision_oracle.py MODEL_DIR OUT_DIR [--image PNG] [--prompt TEXT]
"""

import argparse
import json
import math
import os
import re
import sys

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

# checkpoint name -> Inkling2VisionModel name (Transformers conversion_mapping.py, "inkling2")
PREFIX_RULES = [
    (r"^model\.visual\.backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^model\.visual\.backbone\.embeddings\.", "embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "layers."),
    (r"^model\.visual\.proj_in\.", "merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "merger.linear_2."),
]
LAYER_RULES = [
    (r"\.ln_1\.", ".layernorm_before."),
    (r"\.ln_2\.", ".layernorm_after."),
    (r"\.mlp\.c_fc\.", ".mlp.fc1."),
    (r"\.mlp\.c_proj\.", ".mlp.fc2."),
    (r"\.ls_1\.gamma$", ".lambda_1"),
    (r"\.ls_2\.gamma$", ".lambda_2"),
]


def rename(name):
    for pattern, target in PREFIX_RULES:
        if re.search(pattern, name):
            name = re.sub(pattern, target, name)
            break
    else:
        raise SystemExit(f"unexpected vision tensor {name}")
    if name.startswith("layers."):
        for pattern, target in LAYER_RULES:
            name = re.sub(pattern, target, name)
    return name


def vision_state(model_dir):
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    files = {}
    for name, shard in index["weight_map"].items():
        if name.startswith("model.visual."):
            files.setdefault(shard, []).append(name)
    if not files:
        raise SystemExit("the checkpoint has no model.visual.* tensors")
    state = {}
    for shard, names in files.items():
        with safe_open(os.path.join(model_dir, shard), "pt") as f:
            for name in names:
                state[rename(name)] = f.get_tensor(name)
    return state


def test_picture(path):
    """210 x 130 (not whole 40-pixel tiles): a gradient, a red square and a blue disc."""
    width, height = 210, 130
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            r, g, b = (x * 255) // (width - 1), (y * 255) // (height - 1), 96
            if 20 <= x < 80 and 30 <= y < 90:
                r, g, b = 220, 20, 30
            if (x - 150) ** 2 + (y - 65) ** 2 <= 35**2:
                r, g, b = 20, 40, 230
            pixels[x, y] = (r, g, b)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="Describe the shapes and colors in this picture.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    image_path = args.image or os.path.join(args.out_dir, "inkling2_vision_test.png")
    if not args.image:
        test_picture(image_path)
    image = Image.open(image_path)
    image.load()

    processor = AutoProcessor.from_pretrained(args.model_dir)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    grid = inputs["image_grid_thw"]
    if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
        raise SystemExit(f"unexpected image grid {grid.tolist()}")
    pixel_values = inputs["pixel_values"].float()
    tokens = int(grid.prod())
    if pixel_values.shape != (tokens, 40, 40, 3) or not torch.isfinite(pixel_values).all():
        raise SystemExit(f"unexpected pixel values {tuple(pixel_values.shape)}")
    input_ids = inputs["input_ids"][0].tolist()
    if sum(1 for t in input_ids if t == processor.image_token_id) != tokens:
        raise SystemExit("placeholder count differs from the image tokens")

    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    vision_config = Inkling2VisionConfig(**config["vision_config"])
    model = Inkling2VisionModel(vision_config).eval()
    missing, unexpected = model.load_state_dict(vision_state(args.model_dir), strict=True)
    if missing or unexpected:
        raise SystemExit(f"state mismatch: {missing} {unexpected}")

    features = {}
    with torch.no_grad():
        for key, dtype in (("features_f32", torch.float32), ("features_bf16", torch.bfloat16)):
            model.to(dtype)
            out = model(pixel_values=pixel_values.to(dtype), image_grid_thw=grid).pooler_output.float()
            if out.shape != (tokens, config["text_config"]["hidden_size"]) or not torch.isfinite(out).all():
                raise SystemExit(f"unexpected {key} {tuple(out.shape)}")
            features[key] = out.reshape(-1).tolist()
    a, b = torch.tensor(features["features_bf16"]), torch.tensor(features["features_f32"])
    bf16_vs_f32 = float((a - b).norm() / b.norm())

    patches = pixel_values.reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3).reshape(-1).tolist()
    fixture = {
        "source": {
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "model_dir": os.path.abspath(args.model_dir),
        },
        "image": os.path.abspath(image_path),
        "prompt": args.prompt,
        "grid": [int(grid[0, 1]), int(grid[0, 2])],
        "pixel_values": patches,
        "input_ids": input_ids,
        "features_f32": features["features_f32"],
        "features_bf16": features["features_bf16"],
        "bf16_vs_f32_relative_l2": bf16_vs_f32,
    }
    if any(not math.isfinite(v) for v in patches):
        raise SystemExit("non-finite pixel values")
    out_path = os.path.join(args.out_dir, "inkling2_vision_oracle.json")
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(json.dumps({"out": out_path, "grid": fixture["grid"], "tokens": len(input_ids), "bf16_vs_f32": bf16_vs_f32}))


if __name__ == "__main__":
    sys.exit(main())
