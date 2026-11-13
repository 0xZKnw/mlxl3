# `mlxl3_quantizer` package

- `checkpoint.py` and `options.py`: checkpoint inspection and conversion options.
- `metal.py`, `packing.py`, and `pack_trellis.metal`: quantization and packing with MLX/Metal.
- `native.py` and `process.py`: launching and communicating with native workers.
- `capture.py`, `transforms.py`, and `feedback.py`: captures, transformations, and computations used during conversion.

Model-specific adapters are in [scripts/](../../scripts/README.md) and regression tests in [tests/](../../tests/README.md). Dependencies and packaged data are defined in [pyproject.toml](../../pyproject.toml).
