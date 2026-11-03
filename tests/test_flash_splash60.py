import runpy
from pathlib import Path

import pytest

qualify = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "docs/measurements/flash-next-ssd-impl/opt-60-splash-dflash/qualify.py"
    )
)["qualify"]


def records():
    return [
        {
            "prompt": p,
            "index": i,
            "variant": v,
            "binary_sha256": "a" * 64,
            "status": "validated",
            "reaped": True,
            "reader_joined": True,
            "peak_bytes": 12_000_000_000,
            "swapouts_bytes": 0,
            "tps": 10.2 if v == "Q" else 10.0,
            "seconds": 127 / (10.2 if v == "Q" else 10.0),
            "token_hash": "0123456789abcdef",
            "text": "exact output",
        }
        for p in ("MAIN", "HOLD")
        for i, v in enumerate("BQQB")
    ]


@pytest.mark.parametrize(
    "case,status",
    [
        ("gain", "validated_gain"),
        ("zero", "rejected_gain"),
        ("negative", "rejected_gain"),
        ("drift", "inconclusive_drift"),
        ("spread", "inconclusive_drift"),
    ],
)
def test_only_stable_gains_in_both_pairs_and_prompts_qualify(case, status):
    rows = records()
    if case == "zero":
        rows[2]["tps"] = 10.0
    if case == "negative":
        for row in rows:
            if row["variant"] == "Q":
                row["tps"] = 9.0
    if case == "drift":
        rows[0]["tps"] = 9.0
    if case == "spread":
        rows[1]["tps"] = 11.0
    for row in rows:
        row["seconds"] = 127 / row["tps"]
    result = qualify(rows, "B", "Q")
    assert result["status"] == status and result["confirmed_gain"] == (case == "gain")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("reaped", False),
        ("reader_joined", False),
        ("peak_bytes", 12_000_000_001),
        ("peak_bytes", True),
        ("swapouts_bytes", 1),
        ("swapouts_bytes", False),
        ("tps", float("nan")),
        ("tps", True),
        ("seconds", 0),
        ("seconds", 1),
        ("token_hash", ""),
        ("text", ""),
        ("variant", "R"),
        ("binary_sha256", "b" * 64),
    ],
)
def test_invalid_partial_or_foreign_evidence_cannot_qualify(field, value):
    rows = records()
    rows[1][field] = value
    with pytest.raises(ValueError):
        qualify(rows, "B", "Q")


def test_empty_partial_and_divergent_outputs_are_rejected():
    for rows in [[], records()[:7], list(reversed(records()))]:
        with pytest.raises(ValueError):
            qualify(rows, "B", "Q")
    rows = records()
    rows[1]["text"] = "wrong"
    with pytest.raises(ValueError):
        qualify(rows, "B", "Q")
