"""Report what the installed runtime can run, as one JSON object on stdout."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROCDXG_PATHS = ("/opt/rocm/lib/librocdxg.so.1", "/opt/rocm/lib/librocdxg.so", "/usr/lib/librocdxg.so",
                "/usr/local/lib/librocdxg.so")


def in_wsl(proc_version: str | None = None) -> bool:
    if proc_version is None:
        try:
            proc_version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    return "microsoft" in proc_version.lower()


def wsl_report(exists=os.path.exists, environ=os.environ) -> dict:
    """What ROCm needs inside WSL2: the DXG device, DXCore and AMD's librocdxg bridge."""
    return {
        "dxg_device": exists("/dev/dxg"),
        "dxcore": exists("/usr/lib/wsl/lib/libdxcore.so"),
        "rocdxg": next((p for p in ROCDXG_PATHS if exists(p)), None),
        "dxg_detection": environ.get("HSA_ENABLE_DXG_DETECTION") == "1",
    }


def no_gpu_hints(backend: str, wsl: dict | None) -> list[str]:
    """Actionable causes for 'torch sees no GPU', most likely first."""
    if backend == "cuda":
        return ["Update the NVIDIA driver to 570.65 or newer and restart Windows.",
                "Check that the GPU appears in nvidia-smi."]
    if backend != "rocm":
        return ["The installed torch has no GPU support; reinstall the engine runtime."]
    hints = []
    if wsl is not None:
        if not wsl["dxg_device"]:
            hints.append("/dev/dxg is missing: this distribution is not using WSL2 GPU support. "
                         "Run 'wsl --update', convert it with 'wsl --set-version <distro> 2', "
                         "then 'wsl --shutdown'.")
        if not wsl["dxcore"]:
            hints.append("libdxcore.so is missing from /usr/lib/wsl/lib: update WSL ('wsl --update').")
        if not wsl["rocdxg"]:
            hints.append("AMD's WSL bridge librocdxg is not installed; reinstall the engine runtime "
                         "so MLXL3 installs rocdxg-roct.")
        if not wsl["dxg_detection"]:
            hints.append("HSA_ENABLE_DXG_DETECTION=1 is not set for the engine.")
    hints.append("Install the latest AMD Software: Adrenalin Edition on Windows (WSL support needs "
                 "a recent driver), then run 'wsl --shutdown' and retry.")
    hints.append("WSL2 ROCm supports Radeon RX 9000 / RX 7000 series, Radeon PRO W7000 / R9700 and "
                 "Ryzen AI Max; RX 6000 and older GPUs are not supported in WSL.")
    return hints


def probe() -> dict:
    report = {"ok": False, "python": platform.python_version(), "platform": sys.platform,
              "machine": platform.machine(), "devices": []}
    wsl = wsl_report() if sys.platform.startswith("linux") and in_wsl() else None
    if wsl is not None:
        report["wsl"] = wsl
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
    if not report["devices"]:
        report["error"] = "torch sees no GPU"
        report["hints"] = no_gpu_hints(report["backend"], wsl)
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
    report["ok"] = True
    return report


def main() -> int:
    report = probe()
    sys.stdout.write(json.dumps(report) + "\n")
    return 0 if report["ok"] else 2
