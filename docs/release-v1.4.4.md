# MLXL3 Desktop 1.4.4 — build 26

Cette version embarque le moteur **1.4.5** et corrige l'inspecteur des modèles LFM
ainsi que la fiabilité de l'API locale.

## Nouveautés

- L'inspecteur distingue la préférence MTP/DFlash enregistrée du mode compatible
  avec le modèle chargé. Sur LFM sans tête MTP, le bouton OFF et les réglages de
  sampling restent accessibles.
- L'estimation de contexte utilise désormais le profil des caches LFM. Pour
  LFM2.5-24B-A2B : **20 480 octets/token + 245 760 octets fixes**.
- Le moteur expose `/ready` et `/v1/capabilities` ; les fonctions réellement
  acceptées par l'API sont décrites dans les [notes moteur](release-engine-v1.4.5.md).

## Correctifs

Le passage Qwen→LFM ne verrouille plus température, Top K et répétition à cause
d'une ancienne préférence spéculative. Les valeurs libres restent conservées
lorsqu'on active puis désactive un mode compatible.

L'API annonce correctement les fins EOS, budget de sortie et contexte plein.
Elle refuse une requête supplémentaire lorsque le moteur est occupé, récupère
une déconnexion avant le premier delta et borne les attentes et écritures HTTP.
L'arrêt du serveur pendant le chargement termine et attend le processus moteur.

Le packaging utilise la version Cargo du moteur, indépendamment de la version
Desktop ; l'option explicite de version reste prioritaire.

## Performances et mémoire

Un préfill MoE par microblocs est disponible en option pour la géométrie EXL3
K4/MCG qualifiée de LFM2.5-24B-A2B V2. Il conserve les formes M1 canoniques pour
attention, convolution, normes, gates et head. Activation explicite :
`MLXL3_LFM_MOE_PREFILL=1` dans l'environnement du moteur. **Il reste désactivé par
défaut** : les campagnes de vitesse ont été perturbées par la veille puis une
charge graphique concurrente. Aucun boost de vitesse n'est annoncé comme validé.

Les contrôles numériques du chemin actif comparent 128 000 logits F16 terminaux,
50 états et les routes des 38 MoE, puis 17 continuations forcées. La matrice
historique comprend 32 cas jusqu'à 266 tokens ; les contrôles affectés ont aussi
été répétés jusqu'à 600 tokens. Les résultats sont bitidentiques sur ces entrées,
sans preuve de toutes les positions ou de tout le contexte maximal. Le prototype
KV à capacité a été retiré faute de gain utile. [Rapport et limites](measurements/audit-api-lfm-20261016/lfm-final.md).

L'estimation mémoire décrit les états persistants logiques ; elle exclut poids,
backings supplémentaires, temporaires, cache allocateur et RAM processus. Aucun
poids ni paramètre de quantification n'est modifié ; aucune baisse KL/PPL n'est
revendiquée.

## Installation et compatibilité

Artefact : **MLXL3-Desktop-v1.4.4-b26-Apple-Silicon.dmg**. Apple Silicon, macOS
**26.2+**, moteur **1.4.5** et MLX **0.32.2** embarqués. Les poids sont séparés.
Distribution **signée ad-hoc, sans notarisation Apple**.

La publication moteur conserve `latest=false`, pour que le canal Desktop propose
le DMG. L'app actuellement installée est conservée, selon la décision utilisateur ;
la production des artefacts ne vaut pas remplacement dans `/Applications`.

## Vérification et limites

La source conservée passe **118 tests Rust CPU / 3 ignorés**, format et Clippy
strict, dont **33 tests API inclus**. La suite intégrée Python/Swift passe
**495 tests Python / 69 sautés**, puis les contrôles Desktop ; le sampling compte
**821 assertions / 180 états**, et l'oracle mémoire 240 cas. Les sources Swift et
Python sont inchangées depuis ce contrôle intégré ; les changements Rust finaux
ont leurs contrôles propres.

Kani du sélecteur opt-in : **33 obligations / 5 couvertures**, booléens symboliques
et chaînes de 0 à 4 octets, unwind 6, sans hypothèse. Les preuves CPU ne prouvent
ni Metal, ni toutes les interactions HTTP. Les tests GPU restés ignorés, le long
contexte non qualifié et les timeouts formels sont identifiés dans le rapport.
Les autres fonctionnalités/R&D de l'audit restent dans la
[matrice des 40 propositions](audit-api-lfm-status.md).

Les SHA source, état des jobs CI du commit exact, signatures, noms, tailles et
SHA-256 des fichiers livrés sont publiés dans **MLXL3-v1.4.4-validation.json**,
joint aux deux releases. Les résultats du paquet livré sont distingués des
anciens binaires de recherche ; un job interrompu n'est pas déclaré réussi.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), intégration et validation API/LFM/UI dans
  [PR #39](https://github.com/0xZKnw/mlxl3/pull/39).
- [@HENK0O](https://github.com/HENK0O), fondations mémoire/contexte des
  [PR #31](https://github.com/0xZKnw/mlxl3/pull/31) et
  [PR #32](https://github.com/0xZKnw/mlxl3/pull/32).
- [MLX / mlx-lm](https://github.com/ml-explore/mlx-lm), exécution Apple Silicon
  et architecture LFM ; les autres projets amont du moteur sont crédités dans
  [son changelog](release-engine-v1.4.5.md).
