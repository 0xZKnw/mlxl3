# 72 late — écran B2 H-C

Écran diagnostique terminé. Condition CH≥1 % sur les deux prompts : False. Un point mesuré par bras/prompt ne confirme pas un gain reproductible. Les limites de compression77B restent échouées. Aucun gain promu.

| Prompt | H legacy | C candidat | Écart | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 8.550512518 tok/s | 8.460787148 tok/s | -1.049357 % | 13.968476/13.987112 s |
| HOLD | 8.213442670 tok/s | 8.964298402 tok/s | +9.141791 % | 14.422631/13.151654 s |

DFlash2fixeB2 confirmé, WIDTH1/ADAPTIVE0, MTPinactif, GPUHot/HotWide et prélecturepipeline1/lookahead2. Seule PREFETCH_VARIANT legacy→late diffère entre H/C. Cache3Gio53slots/allocateur64Mio/contexte4096/128sorties127decode/warmup128+mesure128. Les165sources de préparation sont rehashées : seule la CI a changé comme consigné ; sources arithmétiques, CLI77v1, observer et driver identiques. SHA CLI 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. HEADGit89174fcc est une publication, pas un nouveau build utilisé.

Qualité : huit réponses warmup+mesure exactement égales au golden60 OFF pour texte complet non vide, token_hash et comptes128/127. Aucun tableau brut d’IDs persisté ni nouveau contrôle logits/états par cet écran. Quatre rapports validated/reaped/reader_joined ; six étapes pré20/4enfants/post30 code0/reaped ; samplerjoint et secteur à toutes les frontières. Le bridge43 spécifique du candidat72v3 reste préparé/nonexécuté : aucune validation43 transférée depuis packed.

| Bras | Acceptés/proposés/blocs | Lignes vérifiées | Demande experts | Prélectures submitted/completed/used | Réutilisées / timeouts | Attente urgente ns |
|---|---:|---:|---:|---:|---:|---:|
| MAIN H | 58/69/69 | 138 | 26919618840 B | 6453/6453/4800 | 10/1 | 4020916 |
| MAIN C | 58/69/69 | 138 | 26915874804 B | 6454/6454/4801 | 8/0 | 2285334 |
| HOLD H | 56/70/70 | 140 | 28898965872 B | 6566/6566/5057 | 16/1 | 3466583 |
| HOLD C | 56/70/70 | 140 | 28900213884 B | 6567/6567/5055 | 7/1 | 3227999 |

Les compteurs nommés proviennent de qwen4.rs::prefetch_stats dans son ordre [submitted,completed,used,demand_reused,demand_timeouts,demand_wait_ns,bytes,late_needed]. Les octets sont des compteurs moteur de demandes/enregistrements lus ; ils ne mesurent pas les octets physiques SSD ni les écritures swap. Les temps de vérification sont inclusifs : ne pas les additionner à leurs sous-phases ou au decode total. Toute différence de compteurs d’acceptation/largeur est conservée dans pairs[].same_dflash_acceptance_and_width_counters.

Mémoire : picphysique 7852300552 B ; deltas ledger_swapins depuis ready [0, 0, 0, 0] B. Swapouts système parphase [0, 0, 0, 0, 0, 0] B ; Swapins [0, 0, 0, 0, 0, 0] B, sous la borne stock256Mio. Ces compteurs globaux ne sont pas attribués au modèle. Les analyses77B restent failed avec source_errorsvide ; limites historiques inchangées, aucun seuil futur appliqué rétrospectivement.

| Bras | Δlifetime compression warm→mesure | Δjauge compression | Δdécompressions | Δledger_swapins |
|---|---:|---:|---:|---:|
| MAIN H | 33619968 B | +28098560 B | 30711 | 0 B |
| MAIN C | 198721536 B | +185778176 B | 31262 | 0 B |
| HOLD H | 164397056 B | +156811264 B | 30737 | 0 B |
| HOLD C | 7667712 B | +2752512 B | 30605 | 0 B |

Deltas warm→mesure descriptifs ; ils ne requalifient pas les échecs de vie77B. Ledger_swapins sans hausse ne prouve pas l’absence d’écritures swap privées. Aucun GPU/build/proveur/test lourd lancé par l’analyste, aucun code/plan/seuil modifié.

Preuves : summary.json, quatre report.json/memory-analysis.json, run-result.json, system-samples.json, run.json et identity.json dans ce dossier.
