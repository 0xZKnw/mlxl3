"""Frozen data and resource boundaries for registered router trial38."""

from __future__ import annotations

import hashlib
import json

PLAN_SHA256 = "76806e9bc30a0f196c4ac48052030d4d4acfdf726b37462e2d4706794cfdbfbb"
ALPHAS = {"alpha-025": "0.25", "alpha-050": "0.5", "alpha-100": "1"}


def validate_plan(plan: dict) -> dict:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != PLAN_SHA256:
        raise ValueError("Only the frozen, independently checked router38 dataset is allowed")
    return plan


def child_timeout(remaining: int, limit: int) -> int:
    """Reserve cleanup time rather than starting an unbounded or expired child.

    raises: ValueError
    post: 0 < __return__ <= limit
    post: __return__ <= remaining - 30
    """
    if type(remaining) is not int or type(limit) is not int or remaining <= 30 or limit < 1:
        raise ValueError("Insufficient registered time remaining")
    return min(remaining - 30, limit)
