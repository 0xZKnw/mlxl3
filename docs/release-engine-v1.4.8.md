# MLXL3 moteur 1.4.8 — activer le gain Qwen27B dans l'app

## Correctif et comportement

Le moteur 1.4.7 contenait la disposition optimisée des activations EXL3 mais la laissait désactivée par défaut. Le Desktop ne définissant pas son option d'environnement, installer cette version ne suffisait pas à obtenir le gain qualifié.

Le moteur 1.4.8 active désormais ce chemin automatiquement sur les formes déjà qualifiées **M5 / EXL3 Mul1 du Qwen3.8-27B, M1/M2 par projection**. Le regroupement MLP M2/M4 reste automatique ; M3 conserve son parcours en série. Les gardes, shaders, poids, sampler et calculs numériques sont inchangés.

`MLXL3_EXL3_LANE_ORDER=0` restaure le chemin stock. La valeur `1` active aussi l'optimisation ; les valeurs explicites invalides, vides ou non Unicode la désactivent. La clé Tune emploie la même interprétation que le dispatch et sépare les profils 0/1. Les anciennes calibrations ne sont pas attribuées à la nouvelle version.

## Performances et autres options

La qualification Q27-19 mesurait **10,97 → 13,65 tok/s en français et 10,77 → 13,38 tok/s en code**, environ **+24 %**, en activant explicitement la même optimisation. Conditions : MacBook Pro M5 standard, 10 cœurs GPU, 24 Go, batterie, MLX 0.32.2, Qwen3.8-27B EXL3 et tête MTP 4 bits d'origine, MTP1 / MLP automatique, contexte 4096, greedy, cache prompt désactivé. Huit processus ABBA puis BAAB, warmup de 64 tokens, 130 tokens français jusqu'à EOS et 256 tokens code. Variation max/min du candidat : 2,83 % / 2,49 %.

Ces chiffres historiques ne constituent pas une mesure du nouveau paquet. Celui-ci est soumis à une comparaison de son **défaut sans variable** contre `0` explicite, après les contrôles de correction. Les résultats inspectés et les conditions du paquet livré sont joints au manifeste de validation. **15 tok/s ne sont pas établis** ; un message court quelconque ne reproduit pas ce protocole.

L'[audit des autres réglages](qwen27-defaults-audit-20261016.md) ne trouve aucun autre gain de décodage qualifié laissé désactivé. NT4, TensorOps/préfill et MLP sont déjà automatiques. MB3, pipeline, normes fusionnées, lookup et MTP adaptatif n'ont pas établi de boost supplémentaire du 27B. Le compactage des embeddings a une qualification mémoire séparée, sans gain TPS établi. Les gains de campagnes différentes ne s'additionnent pas.

## Installation et compatibilité

Moteur uniquement : **Desktop 1.4.6 / build 28 reste inchangé**, et son DMG reste latest. Utiliser la recherche de mise à jour du moteur indépendant dans le Desktop. Cette publication conserve `latest=false`.

Apple Silicon arm64, macOS 26.2+, Desktop 1.4.0+, MLX 0.32.2. Archive `MLXL3-Engine-v1.4.8-arm64.tar.gz`, manifeste `MLXL3-Engine-v1.4.8-arm64.json`, validation et archive des preuves. Distribution signée ad-hoc, sans notarisation Apple. Le manifeste publié indique le SHA source exact, les tailles et les SHA-256 vérifiés. Les modèles restent séparés.

## Vérification et limites

- Régression reproduite sur l'ancienne fonction Rust avant correction. Le test du vrai helper couvre absence, 1, 0, vide, valeurs malformées, Unicode et NUL.
- **179 tests Rust passés / 85 ignorés, 442 tests Python réussis**, format, Clippy strict et build `mlx,chat` réussis.
- Quatre harnais Kani ciblés réussis : **276 obligations satisfaites, 17 couvertures atteintes**. Le nouveau contrat explore présence/absence et octets symboliques de longueur 0 à 4, unwind 10, avec assertions de déroulement conservées. Sept branches internes de la bibliothèque standard sont inatteignables ; les assertions fonctionnelles et couvertures sont atteintes.
- Les contrôles du nouveau défaut distinguent vrais processus bridge, oracles historiques, valeurs d'environnement invalides, installation Desktop dans un store jetable et mesures appariées. Les résultats du paquet et du SHA exact de CI sont joints séparément. Aucun test ignoré ni job en cours n'est présenté comme réussi.
- Les oracles physiques Q27-19 sur projections, MLP, logits, hidden et états restent attachés à leurs identités ; leurs shaders et gardes n'ont pas changé. Le model checking CPU borné ne prouve ni MLX, ni Metal, ni la concurrence ou la génération complète.

Limites conservées : vraie API Array non vérifiée après erreur interne Kani, MSL non prouvé, CrossHair indisponible pour les nouveaux contrôleurs. Long contexte, autres Mac et qualité globale sur corpus non requalifiés ; températures et fréquences non mesurées.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), intégration et validation sur le Mac physique.
- [Sushi / beamivalice](https://github.com/beamivalice/sushi/tree/70dc0e807e4113179389377253fe8270d5a3333b), disposition des activations et lectures directes par lane, adaptées au codec et aux oracles MLXL3.
- [MLX](https://github.com/ml-explore/mlx), runtime et opérateurs de référence ; [ExLlamaV3](https://github.com/turboderp-org/exllamav3) et [PonyEXL3](https://github.com/beamivalice/PonyExl3), format et travaux EXL3.
- [MTPLX](https://github.com/youssofal/MTPLX), travaux MTP ; [Qwen](https://github.com/QwenLM), architecture et modèles ; [Kani](https://github.com/model-checking/kani), vérification des contrats CPU.
