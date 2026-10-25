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


## Artefacts

Source : `be030721c982942b7ed179ac57d9254941f25cbb`. Signature ad-hoc, sans notarisation.

| Fichier | Octets | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v1.4.6-b28-Apple-Silicon.dmg | 73400205 | `802916abc14e50affda4f18ec654e1b01cb7b9fd2db5bad5434a4533c4b287f1` |
| MLXL3-Engine-v1.4.6-arm64.tar.gz | 66957364 | `0047f7d935129cbfca78f48fca5fe6fdfdd11d975c6d6abe7636881aea2bfc44` |
| MLXL3-Engine-v1.4.6-arm64.json | 13587 | `5be214e495347e7c199513848e814e90b2ac7cb2f04d2aa4b6dfec5f026fdbd9` |

## Résultats du paquet final

41 contrôles paquet/DMG réussis ; suite Desktop complète réussie (379,21s), avec les vrais DMG et TAR1.4.6,15cas signés et les deux frontières d’installation factices. Le vrai préflight1.4.3 accepte le DMG sur CLT26.5. 45tests packaging passent. Les60sources natives et les bibliothèquesMLX/metallib sont identiques1.4.5 ; seul le numéro/identité du moteur change.

Contrôle numérique additionnel du nouveau binaire **non validé** : première tentative timeout120s pendant une campagne GPU externe, deuxième bloquée avant chargement par la garde de concurrence. Sorties partielles/échecs conservés ; pas de parité ni gain déduits. Aucun poids ni calcul numérique modifié, aucune nouvelle mesure KL/PPL. Swift non prouvé formellement ; limites Kani historiques conservées. Aucun remplacement de l’app utilisateur par nos outils.

CI du commit be030721 : [Regression/Desktop](https://github.com/0xZKnw/mlxl3/actions/runs/38910111789) **réussie**, dont15cas signés et les deux frontières installmoteur/copieapp ; **772Python passés/115sautés** (dépendances facultatives/Metal physique exclus). [Rust Linux/macOS](https://github.com/0xZKnw/mlxl3/actions/runs/38910111797) fonctionnel **réussi**,118tests CPU/3ignorés et7oracles CPU featureMLX sur macOS. Kani usageAPI ciblé réussi ; suiteKani complète encore en cours, pas de preuve globale. [Conversion/packing](https://github.com/0xZKnw/mlxl3/actions/runs/38910111662) **réussis**. [PR41](https://github.com/0xZKnw/mlxl3/pull/41) fusionnée.

Manifeste de validation : **MLXL3-v1.4.6-validation.json**, 38275octets, SHA-256 `6cf7f9e6f2622ee2edbf67acf521c3432c3693b2cc0c4979102ec67b5a67383d`. [Preuves/commandes](https://github.com/0xZKnw/mlxl3/tree/main/docs/measurements/updater-v1.4.6), archive verification-evidence.tar.gz/548203octets/SHA-256 `668457a6235389b5b140618ee7d8efee0f3c6cbc0b9b3e4d2501efd61c132983`. Les limites numériques et Kani ci-dessus restent explicites.
