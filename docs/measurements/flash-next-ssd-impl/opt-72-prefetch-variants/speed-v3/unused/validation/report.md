# 72 unused — confirmation B2 C-H

Écran diagnostique terminé. Condition CH≥1 % sur les deux prompts : False. Un point mesuré par bras/prompt ne confirme pas un gain reproductible. Les limites de compression77B restent échouées. Aucun gain promu.

| Prompt | H legacy | C candidat | Écart | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 8.770837130 tok/s | 8.614660406 tok/s | -1.780636 % | 13.576041/13.847063 s |
| HOLD | 8.389887197 tok/s | 8.412918053 tok/s | +0.274507 % | 14.131012/14.126680 s |

DFlash2fixeB2 confirmé, WIDTH1/ADAPTIVE0, MTPinactif, GPUHot/HotWide et prélecturepipeline1/lookahead2. Seule PREFETCH_VARIANT legacy→unused diffère entre H/C. Cache3Gio53slots/allocateur64Mio/contexte4096/128sorties127decode/warmup128+mesure128. Les165sources de préparation sont rehashées : seule la CI a changé comme consigné ; sources arithmétiques, CLI77v1, observer et driver identiques. SHA CLI 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. HEADGit89174fcc est une publication, pas un nouveau build utilisé.

Qualité : huit réponses warmup+mesure exactement égales au golden60 OFF pour texte complet non vide, token_hash et comptes128/127. Aucun tableau brut d’IDs persisté ni nouveau contrôle logits/états par cet écran. Quatre rapports validated/reaped/reader_joined ; six étapes pré20/4enfants/post30 code0/reaped ; samplerjoint et secteur à toutes les frontières. Le bridge43 spécifique du candidat72v3 reste préparé/nonexécuté : aucune validation43 transférée depuis packed.

| Bras | Acceptés/proposés/blocs | Lignes vérifiées | Demande experts | Prélectures submitted/completed/used | Réutilisées / timeouts | Attente urgente ns |
|---|---:|---:|---:|---:|---:|---:|
| MAIN C | 58/69/69 | 138 | 26747393184 B | 6476/6476/4433 | 7/1 | 2391334 |
| MAIN H | 58/69/69 | 138 | 26915874804 B | 6453/6453/4801 | 7/2 | 3196208 |
| HOLD C | 56/70/70 | 140 | 28735476300 B | 6573/6573/4668 | 6/2 | 3754833 |
| HOLD H | 56/70/70 | 140 | 28891477800 B | 6566/6566/5057 | 18/0 | 5573165 |

Les compteurs nommés proviennent de qwen4.rs::prefetch_stats dans son ordre [submitted,completed,used,demand_reused,demand_timeouts,demand_wait_ns,bytes,late_needed]. Les octets sont des compteurs moteur de demandes/enregistrements lus ; ils ne mesurent pas les octets physiques SSD ni les écritures swap. Les temps de vérification sont inclusifs : ne pas les additionner à leurs sous-phases ou au decode total. Toute différence de compteurs d’acceptation/largeur est conservée dans pairs[].same_dflash_acceptance_and_width_counters.

Mémoire : picphysique 7852595464 B ; deltas ledger_swapins depuis ready [0, 0, 0, 0] B. Swapouts système parphase [0, 0, 0, 0, 0, 0] B ; Swapins [0, 0, 0, 0, 0, 0] B, sous la borne stock256Mio. Ces compteurs globaux ne sont pas attribués au modèle. Les analyses77B restent failed avec source_errorsvide ; limites historiques inchangées, aucun seuil futur appliqué rétrospectivement.

| Bras | Δlifetime compression warm→mesure | Δjauge compression | Δdécompressions | Δledger_swapins |
|---|---:|---:|---:|---:|
| MAIN C | 7733248 B | +2768896 B | 30553 | 0 B |
| MAIN H | 8830976 B | +3063808 B | 30681 | 0 B |
| HOLD C | 8028160 B | +2277376 B | 23491 | 0 B |
| HOLD H | 9895936 B | +3112960 B | 30463 | 0 B |

Deltas warm→mesure descriptifs ; ils ne requalifient pas les échecs de vie77B. Ledger_swapins sans hausse ne prouve pas l’absence d’écritures swap privées. Aucun GPU/build/proveur/test lourd lancé par l’analyste, aucun code/plan/seuil modifié.

Preuves : summary.json, quatre report.json/memory-analysis.json, run-result.json, system-samples.json, run.json et identity.json dans ce dossier.
