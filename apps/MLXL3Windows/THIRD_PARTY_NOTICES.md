# Third-party notices — MLXL3 Desktop for Windows

MLXL3 Desktop for Windows is a thin shell around upstream projects. The
installer bundles the Tauri shell, the MLXL3 worker (`mlxl3_exl3`, MIT) and the
`uv` binaries. The GPU runtime (Python, PyTorch, ExLlamaV3, …) is **not**
bundled: the app downloads it on first use, from the pinned sources listed in
`runtime/cuda-windows.txt` and `runtime/rocm-wsl.txt`, into a private
environment. Each of those packages keeps its own license, shown below.

| Component | Use | License | Source |
| --- | --- | --- | --- |
| ExLlamaV3 (turboderp-org) | EXL3 inference engine (CUDA and ROCm wheels) | MIT | https://github.com/turboderp-org/exllamav3 |
| PyTorch | Tensor runtime for ExLlamaV3 | BSD-3-Clause (plus bundled NVIDIA / AMD runtime libraries under their own terms) | https://github.com/pytorch/pytorch |
| triton-windows (woct0rdho) | Triton compiler for Windows, required by ExLlamaV3 on Windows | MIT | https://github.com/woct0rdho/triton-windows |
| AMD ROCm Python libraries (`rocm[libraries]`) | HIP runtime inside WSL2 for the AMD backend | MIT and other licenses listed by AMD | https://github.com/ROCm/ROCm |
| NVIDIA CUDA runtime libraries (pulled in by the PyTorch CUDA wheel) | CUDA runtime | NVIDIA Software License Agreement / CUDA EULA | https://docs.nvidia.com/cuda/eula/ |
| uv (Astral) | Installs a private CPython 3.12 and the pinned packages | MIT or Apache-2.0 | https://github.com/astral-sh/uv |
| CPython (python-build-standalone) | Private Python interpreter downloaded by uv | PSF License Version 2 | https://github.com/astral-sh/python-build-standalone |
| Jinja2 (Pallets) | Renders the model's chat template | BSD-3-Clause | https://github.com/pallets/jinja |
| huggingface_hub | Downloads checkpoints from Hugging Face | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| Tauri and tauri-plugin-dialog | Windows application shell | MIT or Apache-2.0 | https://github.com/tauri-apps/tauri |
| Microsoft Edge WebView2 | UI runtime (installed by the Tauri bootstrapper when missing) | Microsoft Software License Terms | https://developer.microsoft.com/microsoft-edge/webview2/ |
| serde, serde_json, anyhow, sha2 and the other Rust crates in `Cargo.lock` | Shell and core library | MIT or Apache-2.0 (per crate) | https://crates.io |

Model checkpoints are downloaded on request and keep the license chosen by
their authors; MLXL3 does not redistribute them.
