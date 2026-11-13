# Paquet `mlxl3_quantizer`

- `checkpoint.py` et `options.py` : inspection des checkpoints et options de conversion.
- `metal.py`, `packing.py` et `pack_trellis.metal` : quantification et empaquetage côté MLX/Metal.
- `native.py` et `process.py` : lancement et dialogue avec les workers natifs.
- `capture.py`, `transforms.py` et `feedback.py` : captures, transformations et calculs utilisés pendant la conversion.

Les adaptations par modèle sont dans [scripts/](../../scripts/README.md) et les régressions dans [tests/](../../tests/README.md). Les dépendances et données empaquetées sont définies dans [pyproject.toml](../../pyproject.toml).
