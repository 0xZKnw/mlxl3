# Scripts de développement et de livraison

- `build-*.sh`, `build_manifest.py`, `package_engine.py` : construction et assemblage Desktop, moteur ou quantizer.
- `check-*.sh` et `check_*.py` : contrôles ciblés des parcours Desktop, bridge, modèles et couverture native.
- `profile_*`, `analyze_*`, `benchmark*` et `replay_*` : profilage, diagnostic et mesures Flash ou moteur.
- `quantize_*`, `calibrate_*`, `capture_*`, `evaluate_*` : conversion et qualification de quantification.
- `kani_ci.py` et `kani_obligations.json` : orchestration et inventaire des preuves de CI.
- [patches/](patches/README.md) : adaptations appliquées à PonyExl3.

Vérifier les arguments et effets de chaque script avant exécution : certains construisent des artefacts, téléchargent des modèles ou lancent des calculs GPU. Les protocoles d’optimisation sont dans [opti.md](../opti.md).
