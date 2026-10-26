# MLXL3 Desktop 2.0.0-beta.2 — build 32

Préversion : `v2.0.0-beta.2`, latest=false ; version interne Desktop/moteur2.0.0. Source construite : [`673f779f7323fd1c90a88ac95d16ee458d27cb59`](https://github.com/0xZKnw/mlxl3/commit/673f779f7323fd1c90a88ac95d16ee458d27cb59). Les modèles et conversations restent séparés de l’application.

## Nouveautés

- Bouton **Télécharger le draft compatible** pour Flash Next : environ1,2Go, révision IncoAI épinglée, intégrité SHA256, cache, reprise, progression et annulation. Le choix local reste disponible ; le chemin téléchargé est enregistré après confirmation du moteur.
- Choix **Stable / Bêta** dans Réglages → Mises à jour. Stable est le défaut ; Bêta permet de recevoir les préversions. Changer de canal efface la proposition précédente ; Rechercher lance la vérification, sans rétrogradation automatique.
- Quatre cartes carrées dans Modèles → Découvrir : Qwen3.6 35B-A3B2.49bpw, Qwen3.8 27B2.00bpw, Flash Next2.05bpw et LFM2.5 24B-A2B v2 publique. Les liens utilisent les révisions retrouvées dans les téléchargements locaux. Les conversions locales LFM v3/CUDA sont distinctes de cette v2.

## Correctifs

Tune indisponible sur Flash Next n’est plus présenté comme une mise à jour manquante ; Auto/2/7 restent sélectionnables. Le moteur embarqué est préféré à un ancien moteur géré de même version, afin que bêta1 ne masque pas le correctif bêta2.

## Performances

Aucun nouveau gain de débit annoncé, aucun benchmarkGPU répété pour cette mise à jour. Les calculs d’inférence et réglages expérimentaux de bêta1 sont conservés.

## Installation et compatibilité

Apple Silicon, macOS26.2+, signature ad-hoc sans notarisation. Installer ce DMG, ou depuis Desktop1.5.1 choisir Bêta puis Rechercher. Le moteur corrigé arrive avec le Desktopbêta2 : les mises à jour autonomes du moteur ne distinguent pas deux tags partageant le numéro interne2.0.0. Le checkpoint cible et le draft restent à télécharger séparément si absents.

## Vérification et limites

Contrôles ciblés :30tests natifs réussis/2ignorés, format/Clippy CPU etMLX, vraieCLI+cacheRAW SHA256 ;9scénarios téléchargement et8locaux/legacy ; deux suites updater dont15cas d’applications signées jetables ; compilationSwift6 stricte et229contrôles workspace. Pas de nouvelle générationGPU. Kani tenté sur deux contrats réels mais erreur interne de compilation avant obligations : non vérifiés. Ces tests ne constituent pas une preuve complète. Paquets etCI : signatures, arm64, quatre hashes runtime, CLI relocalisée/cache etDMG readonly conformes ; CI duSHA inspectée, conversion réussie, autres jobs encoreen cours àcettephotographie. RPATHabsoluhérité présentmais nonutilisé parleslibrairies chargées du paquet.

## Artefacts

| Fichier | Octets | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v2.0.0-beta.2-b32-Apple-Silicon.dmg | 74116543 | `c2526196ac92c4027c5fc77593a42a5fc9874a8ebbd338bc627bfd8fb398336c` |
| MLXL3-v2.0.0-beta.2-validation.json | 2859 | `af84b78ab8935a19dbc21620a00f5036bbdcb50bd353a07e59f066278329de31` |
| MLXL3-v2.0.0-beta.2-verification.tar.gz | 10016 | `e5b347e2a32cf1bb4b81d8197d626609edccad0840f336fa439ac325dfea9fd5` |

## Remerciements

Merci à [0xZKnw](https://github.com/0xZKnw) pour MLXL3 et ses conversions LFM, à [IncoAI/Splash](https://github.com/incoai/splash) pour DFlash2, à [UnstableLlama](https://huggingface.co/UnstableLlama) et [turboderp](https://huggingface.co/turboderp) pour les checkpoints EXL3, et à [MLX](https://github.com/ml-explore/mlx) pour le runtime Apple Silicon.
