# Qwen3.8-27B : audit des optimisations activées dans le Desktop

Audit Q27-22 du 16 octobre 2026, à partir du moteur 1.4.7 (source957b1946). Révision après publication du moteur **1.4.8**, source26987a2c : le défaut d'activation identifié est corrigé et vérifié. Le Desktop1.4.6 transmet son environnement au bridge ; il ne définit aucune des options de performance ci-dessous.

**Un seul gain de décodage qualifié était resté coupé : lane order, Q27-19.** Les autres gains retenus sont déjà automatiques. Les options encore coupées n'ont pas établi de gain de débit du27B, ou concernent la mémoire ou un autre modèle. L'audit n'a relancé aucun ancien benchmark et n'a activé aucun autre paramètre.

| Mécanisme / option | Défaut réel | Preuve et décision sur le27B/M5 |
|---|---|---|
| `MLXL3_EXL3_LANE_ORDER` | OFF dans1.4.7 ; **ON dans1.4.8** | Q27-19 :13,65/13,38tok/s contre10,97/10,77, ABBA+BAAB sur batterie. Seul défaut de livraison confirmé. Garde M5/Mul1,16formes,M1/M2 conservée |
| `MLXL3_EXPERIMENTAL_DENSE_MLP_BATCH` | **ON**, M2/M4 | Q27-03 :11,08/10,94 contre9,18/9,01tok/s sur secteur, avantlane. L'app atteint déjà ce chemin ; gains non additionnables |
| `MLXL3_DENSE_DECODE_NT4` | **ON**, sauf0 | Environ5% sur forward à tokens imposés ; bridge historique non concluant. Déjà intégré aux projections qualifiées |
| TensorOps M5 / `MLXL3_DISABLE_TENSOR_OPS` | **Automatique** | Détection AppleM5 et QMM déjà livrées. La variable sert à désactiver ; aucune bascule manquante dans le Desktop |
| `MLXL3_DENSE_PREFILL_M64` | **ON**, sauf0 | BM64 pour les grandes formes admissibles. Concerne le préfill ; aucune seconde accélération cachée du decode |
| `MLXL3_EXPERIMENTAL_GROUPED_MB3` | **OFF** | Essais historiques non concluants. Q27-00/02 conservent M3 en série après régression du batchMLP3 |
| `MLXL3_QWEN_PIPELINE` | **OFF sur le dense27B** | Q27-05 :11,09→11,32FR et11,03→11,10code, sous le seuil3%. Le défaut automatique duMoE qualifié ne s'étend pas au27B |
| `MLXL3_QWEN_FUSED_NORM` / `MLXL3_MTP_FUSED_NORM` | **OFF** | Qualité bitexacte ; −3,03% à+1,80% et dérive. Gain non établi, activation par défaut déjà rejetée |
| `MLXL3_MTP_ADAPTIVE` | **OFF** | Contrôleur, états et budgets validés ; vitesse non qualifiée. MTP1 reste la meilleure profondeur mesurée du27B |
| `MLXL3_MTP_LOOKUP` | **OFF** | Gain historique non reproduit, débit dense non mesuré. Vérifier des propositions du contexte a aussi un coût |
| `MLXL3_EMBEDDINGS_PACKED` | **OFF** | Économie historique de301465600octets MLX pour l'embedding27B, exactitude validée. Objectif mémoire ; aucun gainTPS qualifié |
| `MLXL3_ALLOCATOR_CACHE_MIB` | **Défaut MLX conservé** | Débit Q27-03/19 qualifié ainsi ;256Mio servait aux grosses fixtures de qualité. Limiter ou couper la réutilisation n'est pas un gain établi |
| Cache préfixe, compaction et nettoyage au chargement | **Actifs par défaut** | Budget256Mio, réutilisation d'un préfixe capturé. `0/57` sur un court prompt n'indique pas un boost de decode oublié |
| Chaîne draftGPU, projections groupées, fusionsGDN retenues | **Chemins automatiques existants** | Déjà exercés par Q27-03/19. Les prototypes GDNHv48, tête réduite et funnel rejetés/retirés ne sont pas des options prêtes à activer |

## Câblage et qualification du correctif

`MLXL3Bridge.start` transmet `ProcessInfo.processInfo.environment`. `StudioModel` et le bridge choisissent MTP1 par défaut. La clé Tune inclut modèle, contexte, version, commit et options pertinentes ; les anciennes calibrations ne sont pas attribuées à la nouvelle configuration.

Le correctif atteint les deux appelants de la décision lane : dispatch dans `linear.rs` et clé Tune dans `mtp_tune.rs`. Absence et1 activent ;0, vide, malformé et non Unicode désactivent. Le shader et les gardes sont inchangés. Le paquet signé passe six processus pour ces variantes, six sorties FR130/code256 exactes aux références, puis budgets, préfixes, annulations et reprises du bridge.

La comparaison récente du paquet observe11,21FR/11,22code contre9,17/8,62tok/s, avec une dérive des témoins dépassant3% : **non concluante en vitesse**, BAAB conditionnelle non lancée. Les13,65/13,38 restent les mesures historiques Q27-19 ;15tok/s ne sont pas établis. [Changelog et limites1.4.8](release-engine-v1.4.8.md).

## Sources et portée

[Dispatch](../native/src/linear.rs), [Qwen/MLP](../native/src/qwen35.rs), [contrats](../native/src/contracts.rs), [Tune](../native/src/mtp_tune.rs), [détectionM5/cache](../native/src/array.rs), [bridgeDesktop](../apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift), [sélectionMTP](../apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift).

[Q27-03](qwen27-optimization-20261016.md), [pipelineQ27-05](qwen27-vers-15-20261016.md), [Q27-19 et34pistes](qwen27-max-optimisations-20261016.md), [MB3/batch historiques](qwen27-speed-20261008.md), [NT4](qwen27-m5-round2.md), [préfill](qwen27-m5-optimization.md), [normes fusionnées](mtp-dense-1.4.0-validation.md), [lookup](engine-v1.4.2-research.md), [journal](../opti.md).

Audit statique de14mécanismes, sans nouveau gain additionnel ni preuveGPU générale. Les débits d'un message réel dépendent aussi du contenu, de l'acceptation, du contexte et des conditions de la machine. [Validation publiée](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.8).
