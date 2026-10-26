# MLXL3 Engine 2.0.0-beta.2

Préversion `engine-v2.0.0-beta.2`, latest=false, version interne2.0.0. Source construite : SOURCE_SHA_BETA2.

## Nouveautés et correctifs

La commande dflash-draft reconnaît le checkpoint Flash Next et prépare son draftRAW compatible, distinct des paquets35B/27B. Elle valide config,81tenseursBF16, tailles etSHA256 épinglés avant publication atomique du cache. Réutilisation du cache, reprise et annulation sont conservées. La géométrie CPU est partagée avec le chargeur existant ; aucun calculArray ou kernel changé.

## Performances

Aucun gainTPS nouveau ni nouvelle campagneGPU. La copie locale qualifiée a pu être réutilisée avec zéro téléchargement réseau, après hash intégral.

## Installation et compatibilité

Apple Silicon/macOS26.2+, MLX0.32.2. Utiliser Desktop2.0.0-beta.2/build32 pour recevoir aussi le bouton et la préférence du moteur embarqué. Le numéro interne2.0.0 reste numérique : un moteur bêta1 déjà2.0.0 n’est pas proposé comme mise à jour autonome ; le Desktopbêta2 embarque le correctif. Aucun modèle inclus dans l’archive.

## Vérification et limites

30tests ciblés réussis/2ignorés, fmt, ClippyCPU/MLX et compilationMLX ; vraieCLI sur draftlocal, refus de fichiers/cible invalides et cache géré intégralement hashé. TentativeKani de deux contrats interrompue par erreur interne du compilateur avant obligations : non vérifiée. Aucun nouveau test numériqueGPU ; les calculs existants sont conservés. Paquet/CI : PACKAGE_STATUS_BETA2.

## Artefacts

ARTIFACTS_ENGINE_BETA2

## Remerciements

Merci à [0xZKnw](https://github.com/0xZKnw) pour l’intégration, à [IncoAI](https://github.com/incoai/splash) pour le draftDFlash2, à [ExLlamaV3](https://github.com/turboderp-org/exllamav3) et [PonyExl3](https://github.com/beamivalice/PonyExl3) pour EXL3, et à [MLX](https://github.com/ml-explore/mlx) pour le runtime.
