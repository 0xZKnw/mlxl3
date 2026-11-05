# Audit qualité LFM2.5 EXL3 v1

[Rapport et portée des conclusions](../../lfm25-quality-audit-2026-10-14.md), [journal préenregistré](../../../opti.md).

- `real-projections-plan.json` : sélection des 12 projections, captures hashées, paramètres et deadline.
- `real-projections.json` : comparaison bit à bit entre les deux hooks Pony originaux, les hooks MLXL3 et les tenseurs exportés v1, sources identifiées.
- `activation-diagnostic.json` : distribution des experts, diagonales conditionnelles, sensibilité du routage à la précision sur entrées fixées.
- `conditional-ablation.json` : trois variantes down en mémoire sur captures de test distinctes ; tous les contrôles train/test et routés/tous-tokens, sans PPL globale.
- `conditional-ablation.first-capture-failed.json` : échec de chemin avant calcul, conservé.

Les poids, les fichiers source et les captures volumineuses restent hors Git. Captures originales : `/Users/justin/Documents/mix-stq1_0/build/lfm25-24b-4bpw/calibration.safetensors`. Captures test : `/Users/justin/Documents/mix-stq1_0/build/lfm25-eval/quality-audit-heldout`. Les sources numériques et entrées comparées sont identifiées par SHA-256 dans les rapports. Aucun nouveau checkpoint n'a été écrit.
