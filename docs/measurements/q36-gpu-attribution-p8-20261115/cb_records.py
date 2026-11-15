"""Q36-CB-GPU-TIME-P4: read cb_timing records (80-byte little-endian structs)."""
import struct
from pathlib import Path

FORMAT = "<QQddddddqQ"
SIZE = struct.calcsize(FORMAT)
assert SIZE == 80
FIELDS = ("magic", "sequence", "commit", "kernel_start", "kernel_end", "gpu_start", "gpu_end", "handler", "status", "pad")


def read(path):
    data = Path(path).read_bytes()
    if len(data) % SIZE:
        raise ValueError(f"truncated record file: {len(data)} bytes")
    rows = [dict(zip(FIELDS, struct.unpack_from(FORMAT, data, o))) for o in range(0, len(data), SIZE)]
    if any(r["magic"] != 0x31544243 for r in rows):
        raise ValueError("bad magic")
    rows.sort(key=lambda r: r["sequence"])
    if [r["sequence"] for r in rows] != list(range(len(rows))):
        raise ValueError("missing or duplicate sequence numbers")
    return rows
