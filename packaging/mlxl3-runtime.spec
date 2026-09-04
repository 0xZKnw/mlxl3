"""PyInstaller recipe for the self-contained MLXL3 inference runtime."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


root = Path(SPECPATH).parent
datas = []
binaries = []
hiddenimports = []

# MLX ships its Metal library and native dylibs as package data. MLX-LM loads
# model implementations from config.json at runtime, so every architecture has
# to remain discoverable in the frozen application.
for package in (
    "mlx",
    "mlx_lm",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "transformers",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("mlx_lm.models")
hiddenimports += collect_submodules("mlx_lm.tool_parsers")

analysis = Analysis(
    [str(root / "src/mlxl3/cli.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "jax",
        "matplotlib",
        "pandas",
        "tensorflow",
        "torch",
        "torchvision",
    ],
    noarchive=False,
    optimize=1,
)

python_modules = PYZ(analysis.pure)

executable = EXE(
    python_modules,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="mlxl3",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    target_arch="arm64",
    codesign_identity=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="runtime",
)
