"""Report what the installed runtime can run, as one JSON object on stdout."""

from __future__ import annotations

import json
import platform
import sys


def probe() -> dict:
    report = {"ok": False, "python": platform.python_version(), "platform": sys.platform,
              "machine": platform.machine(), "devices": []}
    try:
        import torch
    except Exception as error:
        report["error"] = f"torch import failed: {error}"
        return report
    report.update({"torch_version": torch.__version__, "cuda": torch.version.cuda,
                   "hip": getattr(torch.version, "hip", None)})
    report["backend"] = "rocm" if report["hip"] else "cuda" if report["cuda"] else "cpu"
    try:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for index in range(count):
            device = torch.cuda.get_device_properties(index)
            report["devices"].append({
                "index": index, "name": device.name,
                "memory_gb": round(device.total_memory / 1e9, 2),
                "arch": getattr(device, "gcnArchName", None)
                or f"sm_{device.major}{device.minor}",
            })
    except Exception as error:
        report["error"] = f"device query failed: {error}"
        return report
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401  (triton-windows: required by ExLlamaV3 on Windows)
            report["triton"] = True
        except Exception as error:
            report["triton"] = False
            report["error"] = f"triton import failed: {error}"
            return report
    try:
        import exllamav3  # noqa: F401  (imports the compiled extension)
        from exllamav3.version import __version__
        report["exllamav3_version"] = __version__
    except Exception as error:
        report["error"] = f"exllamav3 import failed: {error}"
        return report
    if not report["devices"]:
        report["error"] = "torch sees no GPU"
        return report
    report["ok"] = True
    return report


def main() -> int:
    report = probe()
    sys.stdout.write(json.dumps(report) + "\n")
    return 0 if report["ok"] else 2
