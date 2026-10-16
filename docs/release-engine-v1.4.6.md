# MLXL3 moteur 1.4.6 — compagnon Desktop 1.4.6

Cette sous-version aligne le moteur avec Desktop 1.4.6 build 28 pour permettre
la migration des anciens updaters. Le correctif de syntaxe LLVM lipo se trouve
dans Desktop et le packager ; le moteur natif conserve les calculs de la 1.4.5.

## Correctifs et compatibilité

Le numéro Cargo/manifeste/runtime-info passe à **1.4.6**. L’archive garde les
mêmes cinq fichiers attendus et le protocole bridge 1. Apple Silicon/macOS26.2+,
MLX0.32.2, signature ad-hoc sans notarisation. Publication avec **`latest=false`**
pour que le DMG Desktop reste proposé par l’updater.

Les corrections API, le profil mémoire LFM et les limites de la 1.4.5 sont
conservés ; consulter son [changelog détaillé](release-engine-v1.4.5.md).

## Performances et qualité

Pas de nouvelle quantification, ni de changement des poids V2 ou de précision.
Le préfill MoE expérimental reste désactivé par défaut ; aucun gain de vitesse
ou de fidélité supplémentaire annoncé. Le build livré est contrôlé selon son
identité propre, sans attribuer les résultats d’un ancien binaire au nouveau.

## Vérification et limites

Le manifeste **MLXL3-v1.4.6-validation.json** joint publie le SHA source, les
noms/tailles/SHA-256 des assets, les vérifications réelles du paquet et les
résultats CI inspectés. La comparaison aux oracles numériques historiques, si
exécutée, y est distinguée des tests CPU et des doubles signés d’installation.
Les sources Rust fonctionnelles restent inchangées : les preuves CPU bornées
et les timeouts historiques de la 1.4.5 ne constituent pas une preuve Metal,
concurrence ou génération complète. Aucun remplacement de l’app utilisateur.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), intégration et qualification des releases ;
  [@HENK0O](https://github.com/HENK0O), bases mémoire/contexte des PR31/32.
- [LiquidAI](https://huggingface.co/LiquidAI), modèles LFM ;
  [MLX/mlx-lm](https://github.com/ml-explore/mlx-lm), runtime et référence ;
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3), format EXL3 ;
  [PonyExl3](https://github.com/beamivalice/PonyExl3), conversion/inférence MLX.
