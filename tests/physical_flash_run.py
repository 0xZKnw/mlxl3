"""Replay the complete Flash AR CLI with and without MLX on a physical Apple GPU."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(cli, args, direct, *, user_input=None, success=True, timeout=600):
    child = subprocess.run(
        [cli, *args],
        input=user_input,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
        env={**os.environ, "DYLD_PRINT_LIBRARIES": "1", "MLXL3_SSD_CACHE_GIB": "0.6"},
    )
    if (child.returncode == 0) != success:
        raise RuntimeError(f"{cli.name} exit {child.returncode}: {child.stderr[-1600:]}")
    images = [line.lower() for line in child.stderr.splitlines() if line.startswith("dyld[")]
    assert images
    if success:
        assert any("/metal.framework/" in line for line in images)
    mlx = any(
        path in line
        for line in images
        for path in ("/libmlx", "/libjaccl", "/site-packages/mlx/", "/mlx.framework")
    )
    assert mlx != direct
    return child.stdout, child.stderr, len(images)


def scored(stdout: str) -> tuple[str, int]:
    """Parse a finished generation.

    raises: AssertionError, ValueError
    post: bool(_[0].strip()) and _[1] > 0
    """
    match = re.fullmatch(
        r"(?s)(.+)\n\[prefill \d+\.\d tok/s · decode \d+\.\d tok/s · TTFT \d+ ms · (\d+) tokens\]\n",
        stdout,
    )
    assert match and match[1].strip() and int(match[2]) > 0
    return match[1], int(match[2])


def verify(args, report):
    import numpy as np

    imposed = [248044, 1, 42, 7]
    cmd = ["forward", args.model, "--tokens", ",".join(map(str, imposed))]

    def forward(cli, direct, more=()):
        out, _, images = execute(cli, cmd + list(more), direct)
        rows = [json.loads(line) for line in out.splitlines()]
        assert [r["token"] for r in rows] == (imposed[-1:] if more else imposed)
        a = np.array([r["logits"] for r in rows], dtype=np.uint16)
        assert a.shape == (len(rows), 248320) and np.isfinite(a.view(np.float16)).all()
        return a, images

    ref, _ = forward(args.stock, False)
    native, images = forward(args.cli, True)
    batched, _ = forward(args.cli, True, ("--batch",))
    report.update(
        logits=int(native.size),
        dyld_images=images,
        logit_sha256=[hashlib.sha256(r.tobytes()).hexdigest() for r in ref],
        logit_mismatches=int(np.count_nonzero(ref != native)),
        batch_mismatches=int(np.count_nonzero(ref[-1] != batched[0])),
    )
    assert report["logit_mismatches"] == report["batch_mismatches"] == 0
    for tokens in ("248320", "4294967295"):
        _, err, _ = execute(
            args.cli, ["forward", args.model, "--tokens", tokens], True, success=False
        )
        assert "invalid native Flash tokens or context" in err
    prompt = "Combien font 2 + 2 ? Réponds en un chiffre. /no_think"
    cmd = ["run", args.model, "--prompt", prompt, "--max-tokens", "256"]
    stock_text = scored(execute(args.stock, cmd, False)[0])
    text = scored(execute(args.cli, cmd, True)[0])
    report.update(
        prompt=prompt,
        generated_tokens=text[1],
        same_text=text == stock_text,
        answer=text[0].partition("— Answer —")[2].strip(),
        text_sha256=hashlib.sha256(text[0].encode()).hexdigest(),
    )
    assert text == stock_text and text[1] < 256 and report["answer"] == "4"
    # Exercise the same frontend in interactive mode and discard the conversation with /clear.
    cmd = ["run", args.model, "--max-tokens", "1"]
    dialog = "Bonjour\n/clear\nBonjour\n/quit\n"
    out, _, _ = execute(args.cli, cmd, True, user_input=dialog)
    rounds = [scored(x) for x in out.split("\n>>> ") if "[prefill " in x]
    assert len(rounds) == 2 and rounds[0] == rounds[1] and rounds[0][1] == 1
    report["interactive_replays"] = 2


def long_forward(args, report):
    import numpy as np

    tokens = [248044, *(1 + 17 * i % 251 for i in range(1, 2053))]
    command = ["forward", args.model, "--batch", "--tokens", ",".join(map(str, tokens))]
    report["tokens_sha256"] = hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest()
    report["positions"] = len(tokens)
    results = []
    for label, cli in [("direct", args.cli), ("stock", args.stock)]:
        out, _, images = execute(cli, command, label == "direct", timeout=1800)
        row = json.loads(out)
        logits = row["logits"]
        assert row["token"] == tokens[-1] and row["states"] == []
        assert len(logits) == 248320 and all(type(x) is int and 0 <= x <= 65535 for x in logits)
        bits = np.asarray(logits, dtype="<u2")
        assert np.isfinite(bits.view("<f2")).all()
        report[label] = {"sha256": hashlib.sha256(bits.tobytes()).hexdigest(), "images": images}
        results.append(bits)
    report["mismatches"] = int(np.count_nonzero(results[0] != results[1]))
    assert report["mismatches"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "stock", "model", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--long-only", action="store_true", help="compare the complete 2053-position sparse path"
    )
    args = parser.parse_args()
    report = {
        "status": "incomplete",
        "cli_sha256": digest(args.cli),
        "stock_sha256": digest(args.stock),
    }
    with args.report.open("x") as output:
        try:
            if args.long_only:
                long_forward(args, report)
            else:
                verify(args, report)
            report["status"] = "verified"
        except Exception as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
