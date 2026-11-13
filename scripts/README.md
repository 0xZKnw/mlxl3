# Development and delivery scripts

- `build-*.sh`, `build_manifest.py`, `package_engine.py`: build and package Desktop, engine, or quantizer artifacts.
- `check-*.sh` and `check_*.py`: targeted checks for Desktop, bridge, model, and native coverage paths.
- `profile_*`, `analyze_*`, `benchmark*`, and `replay_*`: Flash or engine profiling, diagnostics, and measurements.
- `quantize_*`, `calibrate_*`, `capture_*`, `evaluate_*`: quantization conversion and qualification.
- `kani_ci.py` and `kani_obligations.json`: CI proof orchestration and inventory.
- [patches/](patches/README.md): adaptations applied to PonyExl3.

Check each script's arguments and effects before running it: some build artifacts, download models, or run GPU workloads. Optimization protocols are in [opti.md](../opti.md).
