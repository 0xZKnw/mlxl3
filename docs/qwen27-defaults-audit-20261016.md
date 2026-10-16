# Qwen3.8-27B : optimisations activées dans le Desktop

Audit Q27-22 du 16 octobre 2026, moteur publié 1.4.7 / source 957b1946, correctif 1.4.8 en qualification. Le Desktop 1.4.6 lance le bridge avec son environnement hérité ; il ne définit aucune des options de performance ci-dessous. Le moteur installé est bien 1.4.7 (`Engines/current.json`, lecture seule). Aucun autre moteur natif ne tournait au relevé ; batterie 52 %, explicitement autorisée.

**Un seul gain de décodage qualifié est resté coupé par erreur de livraison : lane order, Q27-19.** Les autres optimisations de débit retenues sont déjà automatiques. Les options encore coupées ont une qualification insuffisante, un autre objectif ou un autre modèle. L'audit ne relance pas les anciens benchmarks et n'active aucune option supplémentaire.

| Mécanisme / option | Défaut réel1.4.7 | Preuve et décision27B/M5 |
|---|---|---|
| `MLXL3_EXL3_LANE_ORDER` | **OFF** en absence de variable | Q27-19 : 13,6508FR/13,3806code contre10,9712/10,7673 tok/s, ABBA+BAAB sur batterie. Défaut de livraison confirmé ; corrigé séparément sousQ27-21. Garde M5/Mul1,16formes,M1/M2 par projection conservée |
| `MLXL3_EXPERIMENTAL_DENSE_MLP_BATCH` | **ON** absent ou1, M2/M4 seulement | Q27-03 : 11,0765/10,9362tok/s contre9,176/9,0106 sur secteur, avantlane. App atteint le même `forward_verification`; aucun flag manquant |
| `MLXL3_DENSE_DECODE_NT4` | **ON** sauf0, formes qualifiées | Round2 :≈5% sur forward àtokens imposés ; bridgehistorique nonconcluant. Déjà intégré dans `Exl3Linear`/`Exl3Group`. Ne pas réadditionner ce gain àQ19 |
| TensorOps M5 / `MLXL3_DISABLE_TENSOR_OPS` | **Automatique** ; variable de désactivation absente normalement | `Array::is_m5_gpu` détecte AppleM5 ; QMM préfill déjà livré. Pas de bascule supplémentaire côtéGUI |
| `MLXL3_DENSE_PREFILL_M64` | **ON** sauf0, grandes formes/préfill admissibles | Sélection BM64 déjà dans `tensor_qmm`. Agit sur le préfill et ses limites, pas une seconde accélération cachée du decode |
| `MLXL3_EXPERIMENTAL_GROUPED_MB3` | **OFF** | K1/K2 historiques nonconcluants ; Q27-00/02 gardent M3 série après régression du batchMLP3. Ce n'est pas un gain validé laissé coupé |
| `MLXL3_QWEN_PIPELINE` | **OFF pourdense27B** ; autoMTP pourMoE qualifié | Q27-05 :FR11,0907→11,3218/code11,0280→11,0986tok/s, sous3% malgré contrôlesstables. Ne pas généraliser le boostMoE au27B |
| `MLXL3_QWEN_FUSED_NORM` et `MLXL3_MTP_FUSED_NORM` | **OFF** | Campagnes1.4.0 :qualité bitexacte, −3,03% à+1,80% et dérive ; gain nonétabli. Activationpardéfaut déjà rejetée |
| `MLXL3_MTP_ADAPTIVE` | **OFF** | Contrôleur/états/budgets validés, vitesse nonqualifiée. MTP1 reste la meilleure profondeur mesurée du27B ; davantage de propositions ne garantit pas davantage de tok/s |
| `MLXL3_MTP_LOOKUP` | **OFF** | Qualité contrôlée, gain historique nonreproduit et débitdense nonmesuré. Les propositions issues du contexte sont vérifiées mais peuvent coûter plus qu'elles rapportent |
| `MLXL3_EMBEDDINGS_PACKED` | **OFF** | Économie de301465600octets MLX pour l'embedding27B dans le contrôlehistorique, exactitude validée. Objectifmémoire, débit nonqualifié ; aucune hausseTPS déduite de cette économie |
| `MLXL3_ALLOCATOR_CACHE_MIB` | **DéfautMLX conservé**, aucune limiteforcée | Q27-03/Q19débit qualifié sous cedéfaut ;256Mio servait àborner les grossesfixtures dequalité. Désactiver le cache peut supprimerlaréutilisation ; aucune hausseTPS qualifiée |
| Cachepréfixe / compaction / nettoyage après chargement | **Actifs par défaut**, budget256Mio | Réutilisation conditionnée àunpréfixe déjàcapturé ;0/57 sur un premier court message n'indique pas unboostdedecodeoublié. Le cachepréfixe accélère surtout la reprise duprompt ; ne seconfond pasavec lecacheallocateur |
| Chaîne draftGPU / projectionsgroupées / fusionsGDN déjàretenues | **Chemins automatiques existants** | Session/projections réelles déjà exercées parQ27-03/19. Les prototypes GDNHv48, tête réduite etfunnel ne sont pasdesflags prêtsàactiver : rejetés/retirés |

## Câblage vérifié

`MLXL3Bridge.start` transmet `ProcessInfo.processInfo.environment` au moteur indépendant. `StudioModel` et le bridge choisissent MTP1 par défaut ; la clé de calibration inclut version, commit, modèle, contexte, batchMLP, lane, pipeline et autresoptions décrites. Une miseàjourmoteur ne doit donc pas reprendre une ancienne mesure comme silaconfiguration était inchangée. L'interface affiche bien 1.4.7 : la panne décrite n'est pas une ancienne version encoreinstallée.

Le correctif deQ27-21 atteint les deuxappelants de la décision lane :préparation/dispatch dans `linear.rs`, puis `runtime_key` dans `mtp_tune.rs`. Absence devientON,1resteON,0/valeursinvalides ycomprisnonUnicode restentOFF ; profils0/1 restentdistincts. Le shader et la garde des formes sont inchangés. Les autresparamètres ci-dessus restent telsquequalifiés.

## Sources locales et limites

- [Source du dispatch](../native/src/linear.rs), [Qwen etbatchMLP](../native/src/qwen35.rs), [contrats](../native/src/contracts.rs), [Tune](../native/src/mtp_tune.rs), [détectionM5/cache](../native/src/array.rs), [bridgeDesktop](../apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift), [sélectionMTP](../apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift).
- [Q27-03/MLP](qwen27-optimization-20261016.md), [Q27-05/pipeline](qwen27-vers-15-20261016.md), [Q27-19/lanes et34pistes](qwen27-max-optimisations-20261016.md), [MB3/batchhistoriques](qwen27-speed-20261008.md), [NT4](qwen27-m5-round2.md), [préfill](qwen27-m5-optimization.md), [fusionsnorme](mtp-dense-1.4.0-validation.md), [lookup](engine-v1.4.2-research.md), [journalcomplet](../opti.md).

Audit statique du code et des preuves conservées : aucun nouveau gainadditif, aucune nouvelle preuve GPU, aucunbenchmark relancé pourcet inventaire. Les 14 lignes ne sont pas 14 gains. Les résultats de vitesse du défaut 1.4.8 seront joints àla release aprèsqualification ;13,65/13,38 restent les mesures historiquesQ19 tantquecettenouvellecampagne n'est pas terminée. Les débits d'un message court réel dépendent aussi du contenu, de l'acceptation, du contexte et desconditionsmatérielles.
