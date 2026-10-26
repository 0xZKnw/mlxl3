# MLXL3 Engine 2.0.0-beta.2

Préversion `engine-v2.0.0-beta.2`, latest=false, version interne2.0.0. Source construite : [`673f779f7323fd1c90a88ac95d16ee458d27cb59`](https://github.com/0xZKnw/mlxl3/commit/673f779f7323fd1c90a88ac95d16ee458d27cb59).

## Nouveautés et correctifs

La commande dflash-draft reconnaît le checkpoint Flash Next et prépare son draftRAW compatible, distinct des paquets35B/27B. Elle valide config,81tenseursBF16, tailles etSHA256 épinglés avant publication atomique du cache. Réutilisation du cache, reprise et annulation sont conservées. La géométrie CPU est partagée avec le chargeur existant ; aucun calculArray ou kernel changé.

## Performances

Aucun gainTPS nouveau ni nouvelle campagneGPU. La copie locale qualifiée a pu être réutilisée avec zéro téléchargement réseau, après hash intégral.

## Installation et compatibilité

Apple Silicon/macOS26.2+, MLX0.32.2. Utiliser Desktop2.0.0-beta.2/build32 pour recevoir aussi le bouton et la préférence du moteur embarqué. Le numéro interne2.0.0 reste numérique : un moteur bêta1 déjà2.0.0 n’est pas proposé comme mise à jour autonome ; le Desktopbêta2 embarque le correctif. Aucun modèle inclus dans l’archive.

## Vérification et limites

30tests ciblés réussis/2ignorés, fmt, ClippyCPU/MLX et compilationMLX ; vraieCLI sur draftlocal, refus de fichiers/cible invalides et cache géré intégralement hashé. TentativeKani de deux contrats interrompue par erreur interne du compilateur avant obligations : non vérifiée. Aucun nouveau test numériqueGPU ; les calculs existants sont conservés. Paquet/CI : signatures, arm64, quatre hashes runtime, CLI relocalisée/cache etDMG readonly conformes ; CI duSHA inspectée, conversion réussie, autres jobs encoreen cours àcettephotographie. RPATHabsoluhérité présentmais nonutilisé parleslibrairies chargées du paquet.

## Artefacts

| Fichier | Octets | SHA-256 |
|---|---:|---|
| MLXL3-Engine-v2.0.0-beta.2-arm64.json | 13588 | `8fa7e7fc9ac0853d29463cf8e3e388edb42891fea0e42d1705623abd56a87889` |
| MLXL3-Engine-v2.0.0-beta.2-arm64.tar.gz | 67254379 | `a424138b53d6c7a9dbc632c06af946523e1dd8f7630e2ba63d4dd086c74fb036` |

## Remerciements

Merci à [0xZKnw](https://github.com/0xZKnw) pour l’intégration, à [IncoAI](https://github.com/incoai/splash) pour le draftDFlash2, à [ExLlamaV3](https://github.com/turboderp-org/exllamav3) et [PonyExl3](https://github.com/beamivalice/PonyExl3) pour EXL3, et à [MLX](https://github.com/ml-explore/mlx) pour le runtime.
