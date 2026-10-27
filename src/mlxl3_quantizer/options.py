"""Explicit projection exclusions shared by the existing conversion adapters."""

import json
import os


def excluded_projection(key: str) -> bool:
    prefixes = json.loads(os.environ.get("MLXL3_QUANT_EXCLUDE_PREFIXES", "[]"))
    if not isinstance(prefixes, list) or not all(isinstance(p, str) and p for p in prefixes):
        raise ValueError("Invalid conversion exclusions")
    return any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)
