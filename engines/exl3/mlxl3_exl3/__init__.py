"""MLXL3 EXL3 engine worker for Windows (CUDA) and WSL2 (ROCm).

The worker wraps the upstream ExLlamaV3 library behind the same resident
JSON-lines bridge protocol (version 1) that MLXL3 Desktop uses on macOS, so
the Windows app talks to one protocol whatever GPU backend runs underneath.
"""

__version__ = "0.1.0"
BRIDGE_PROTOCOL = 1
