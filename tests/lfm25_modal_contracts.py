"""Pure launch contracts; no claim about CUDA numerical correctness or quality."""

from pathlib import Path

from quantize_lfm25_modal import conversion_command


def quality_contract(probe: bool) -> None:
    """post: __return__ is None"""
    command = conversion_command(Path("source"), Path("work"), Path("output"), probe)
    assert command[command.index("-b") + 1] == "4"
    assert command[command.index("-hb") + 1] == "16"
    assert command[command.index("-cb") + 1] == "mcg"
    assert "--hq" in command
    assert command[command.index("--out_scales") + 1] == "auto"


def calibration_budget_contract(probe: bool) -> None:
    """post: __return__ is None"""
    command = conversion_command(Path("source"), Path("work"), Path("output"), probe)
    rows = int(command[command.index("-cr") + 1])
    columns = int(command[command.index("-cc") + 1])
    assert rows > 0 and columns > 0
    assert rows * columns == (2048 if probe else 512000)
    assert "--override_anyway" not in command
