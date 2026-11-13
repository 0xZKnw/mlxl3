# Kernels Metal

Sources Metal incluses ou compilées par le moteur : opérations EXL3 et matrices (`exl3`, `_qmv*`, `_qmm*`), Flash (`flash_*`), Qwen, LFM3, DFlash, quantification et primitives communes. Les fichiers `.h` partagent des définitions entre kernels.

Les appelants et la sélection des chemins sont dans [native/src/](../src/README.md), notamment [gpu/](../src/gpu/README.md) et `linear.rs`. Une compilation seule ne valide pas les résultats numériques sur GPU ; consulter les [consignes](../../AGENTS.md) et le [journal](../../opti.md) avant optimisation.
