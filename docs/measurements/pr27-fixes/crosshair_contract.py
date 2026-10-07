"""Symbolic work-identity contract calling the actual bridge validator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
from benchmark_smallm_bridge import fingerprint


def signature_preserves_work(accepted: int, proposed: int, blocks: int) -> bool:
    """
    pre: 0 <= accepted <= proposed
    pre: blocks > 0
    post: __return__
    """
    event = {
        "stats": {
            "generated_tokens": 256,
            "decode_tps": 8.0,
            "decode_seconds": 32.0,
            "mtp_accepted_tokens": accepted,
            "mtp_proposed_tokens": proposed,
            "mtp_blocks": blocks,
        },
        "token_hash": "stable",
        "cache_context": "stable history",
    }
    signature = fingerprint(event, "target", 256)
    changed = {**event, "stats": {**event["stats"], "mtp_proposed_tokens": proposed + 1}}
    return signature[-3:] == (accepted, proposed, blocks) and signature != fingerprint(
        changed, "target", 256
    )
