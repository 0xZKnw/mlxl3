# 72 unused — écran B2 H-C

Écran diagnostique terminé. Condition CH≥1 % sur les deux prompts : True. Un point mesuré par bras/prompt ne confirme pas un gain reproductible. Les limites de compression77B restent échouées. Aucun gain promu.

| Prompt | H legacy | C candidat | Écart | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 8.854470125 tok/s | 9.075447698 tok/s | +2.495661 % | 13.482766/13.171529 s |
| HOLD | 8.737266910 tok/s | 8.882685205 tok/s | +1.664345 % | 13.533843/13.352556 s |

DFlash2fixeB2 confirmé, WIDTH1/ADAPTIVE0, MTPinactif, GPUHot/HotWide et prélecturepipeline1/lookahead2. Seule PREFETCH_VARIANT legacy→unused diffère entre H/C. Cache3Gio53slots/allocateur64Mio/contexte4096/128sorties127decode/warmup128+mesure128. Les165sources de préparation sont rehashées : seule la CI a changé comme consigné ; sources arithmétiques, CLI77v1, observer et driver identiques. SHA CLI 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. HEADGit89174fcc est une publication, pas un nouveau build utilisé.

Qualité : huit réponses warmup+mesure exactement égales au golden60 OFF pour texte complet non vide, token_hash et comptes128/127. Aucun tableau brut d’IDs persisté ni nouveau contrôle logits/états par cet écran. Quatre rapports validated/reaped/reader_joined ; six étapes pré20/4enfants/post30 code0/reaped ; samplerjoint et secteur à toutes les frontières. Le bridge43 spécifique du candidat72v3 reste préparé/nonexécuté : aucune validation43 transférée depuis packed.

| Bras | Acceptés/proposés/blocs | Lignes vérifiées | Demande experts | Prélectures submitted/completed/used | Réutilisées / timeouts | Attente urgente ns |
|---|---:|---:|---:|---:|---:|---:|
| MAIN H | 58/69/69 | 138 | 26925858900 B | 6453/6453/4799 | 11/0 | 2823499 |
| MAIN C | 58/69/69 | 138 | 26744897160 B | 6476/6476/4433 | 17/0 | 4638624 |
| HOLD H | 56/70/70 | 140 | 28892725812 B | 6567/6567/5058 | 20/0 | 3803710 |
| HOLD C | 56/70/70 | 140 | 28732980276 B | 6573/6573/4672 | 15/0 | 3019543 |

Les compteurs nommés proviennent de qwen4.rs::prefetch_stats dans son ordre [submitted,completed,used,demand_reused,demand_timeouts,demand_wait_ns,bytes,late_needed]. Les octets sont des compteurs moteur de demandes/enregistrements lus ; ils ne mesurent pas les octets physiques SSD ni les écritures swap. Les temps de vérification sont inclusifs : ne pas les additionner à leurs sous-phases ou au decode total. Toute différence de compteurs d’acceptation/largeur est conservée dans pairs[].same_dflash_acceptance_and_width_counters.

Mémoire : picphysique 7852300600 B ; deltas ledger_swapins depuis ready [0, 0, 0, 0] B. Swapouts système parphase [0, 0, 0, 0, 0, 0] B ; Swapins [0, 0, 0, 0, 0, 0] B, sous la borne stock256Mio. Ces compteurs globaux ne sont pas attribués au modèle. Les analyses77B restent failed avec source_errorsvide ; limites historiques inchangées, aucun seuil futur appliqué rétrospectivement.

| Bras | Δlifetime compression warm→mesure | Δjauge compression | Δdécompressions | Δledger_swapins |
|---|---:|---:|---:|---:|
| MAIN H | 184107008 B | +177537024 B | 30575 | 0 B |
| MAIN C | 8650752 B | +3801088 B | 29811 | 0 B |
| HOLD H | 7143424 B | +2523136 B | 30642 | 0 B |
| HOLD C | 8519680 B | +3194880 B | 30599 | 0 B |

Deltas warm→mesure descriptifs ; ils ne requalifient pas les échecs de vie77B. Ledger_swapins sans hausse ne prouve pas l’absence d’écritures swap privées. Aucun GPU/build/proveur/test lourd lancé par l’analyste, aucun code/plan/seuil modifié.

Preuves : summary.json, quatre report.json/memory-analysis.json, run-result.json, system-samples.json, run.json et identity.json dans ce dossier.
