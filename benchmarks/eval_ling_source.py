"""The fixed-window BF16 perplexity baseline, loading one Ling layer at a time."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import mlx.core as mx
import numpy as np
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from quantize_ling import capture_ling
from ponyexl3.convert.fixtures import SafetensorIndex
from eval_checkpoint_perplexity import mean_nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--data", type=Path, default=Path("benchmarks/data/wikitext-2-raw/wiki.test.raw"))
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.window < 1 or args.tokens % args.window:
        parser.error("tokens must be a positive multiple of window")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    corpus = args.data.read_bytes()
    ids = tokenizer.encode(corpus.decode())[:args.tokens + 1]
    if len(ids) != args.tokens + 1:
        raise ValueError("Evaluation corpus is too short")
    with tempfile.TemporaryDirectory(prefix="mlxl3-ling-eval-") as directory:
        hidden_path = Path(directory) / "hidden.npy"
        capture_ling(args.model, [], Path(directory) / "capture", text_path=args.data,
                     rows=args.tokens, seq_len=args.window, max_seqs=args.tokens // args.window,
                     hidden_output=hidden_path)
        hidden = np.load(hidden_path)
    mx.clear_cache()
    weight = mx.array(SafetensorIndex(args.model).read_tensor("lm_head.weight")).astype(mx.bfloat16)
    total = 0.0
    for number, sequence in enumerate(hidden):
        logits = mx.array(sequence).astype(mx.bfloat16) @ weight.T
        start = number * args.window
        loss = float(mean_nll(logits, mx.array(ids[start + 1:start + args.window + 1])))
        if not math.isfinite(loss):
            raise ValueError("Non-finite BF16 perplexity")
        total += loss
        del logits
        mx.clear_cache()
    nll = total / len(hidden)
    result = {"model": args.model.name, "dataset": "WikiText-2 test",
              "dataset_sha256": hashlib.sha256(corpus).hexdigest(), "tokens": args.tokens,
              "window": args.window, "mean_nll": nll, "perplexity": math.exp(nll),
              "execution": "layerwise BF16; 128-token chunks, cache reset each window"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
