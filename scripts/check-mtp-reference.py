"""Create a numerical MTP reference using the official MLX-LM Qwen decoder.

Architecture reference: MTPLX qwen3_5_mtp_patch.py, Apache-2.0.
Only selected target embedding rows are read; no second target model is loaded.
"""

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs


class Head(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [DecoderLayer(args, args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden, embedding, cache, return_residual=False):
        x = self.fc(
            mx.concatenate(
                [self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(hidden)], axis=-1
            )
        )
        residual = self.layers[0](x, mask="causal", cache=cache)
        return residual if return_residual else self.norm(residual)


def embeddings(model, tokens):
    for path in model.glob("*.safetensors"):
        with path.open("rb") as stream:
            header_bytes = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_bytes))
            entry = header.get("model.language_model.embed_tokens.weight")
            if entry is None:
                continue
            assert entry["dtype"] == "BF16"
            width = entry["shape"][1]
            rows = []
            for token in tokens:
                stream.seek(8 + header_bytes + entry["data_offsets"][0] + token * width * 2)
                row = np.frombuffer(stream.read(width * 2), dtype="<u2").astype(np.uint32)
                rows.append((row << 16).view(np.float32).astype(np.float16))
            return mx.array(np.stack(rows)[None], dtype=mx.float16)
    raise ValueError("target embedding tensor missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--recursive", action="store_true", help="Three recursive pre-norm hidden feedback steps"
    )
    args = parser.parse_args()
    config = json.loads((args.head / "config.json").read_text())
    layout = TextModelArgs.from_dict(config["text_config"])
    head = Head(layout)
    nn.quantize(head, group_size=64, bits=4, mode="affine")
    weights = mx.load(str(args.head / "model.safetensors"))
    weights = {
        name: value if value.dtype == mx.uint32 else value.astype(mx.float16)
        for name, value in weights.items()
    }
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    cache = KVCache()
    steps = []
    offset = 0
    previous = None
    for time in [1, 1, 1] if args.recursive else [1, 2, 3, 17, 24]:
        tokens = [1 + (offset + i) % 31 for i in range(time)]
        values = np.array(
            [
                ((offset * layout.hidden_size + i) % 251 - 125) / 128
                for i in range(time * layout.hidden_size)
            ],
            dtype=np.float16,
        ).reshape(1, time, -1)
        if args.recursive and previous is not None:
            values = previous
        result = head(
            mx.array(values), embeddings(args.model, tokens), cache, return_residual=args.recursive
        )
        mx.eval(result)
        output = np.array(result).astype(np.float16)
        assert output.shape == (1, time, layout.hidden_size) and np.isfinite(output).all()
        keys, cache_values = [np.array(x).astype(np.float16) for x in cache.state]
        expected_shape = (1, layout.num_key_value_heads, offset + time, layout.head_dim)
        assert keys.shape == cache_values.shape == expected_shape
        assert np.isfinite(keys).all() and np.isfinite(cache_values).all()
        previous = output
        steps.append(
            {
                "tokens": tokens,
                "hidden": values.view(np.uint16).reshape(-1).tolist(),
                "residual" if args.recursive else "normalized": output.view(np.uint16)
                .reshape(-1)
                .tolist(),
                "keys": keys.view(np.uint16).reshape(-1).tolist(),
                "values": cache_values.view(np.uint16).reshape(-1).tolist(),
            }
        )
        offset += time
    args.output.write_text(
        json.dumps({"source": "mlx_lm.models.qwen3_5.DecoderLayer", "steps": steps})
    )
    print(json.dumps({"reference": str(args.output), "positions": offset, "dtype": "float16"}))


if __name__ == "__main__":
    main()
