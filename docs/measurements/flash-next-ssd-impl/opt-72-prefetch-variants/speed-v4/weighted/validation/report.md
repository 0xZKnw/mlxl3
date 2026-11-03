# 72 weighted — confirmation B2 C-H

**Décision actuelle — retrait utilisateur72W-WITHDRAW.** Le vrai A/B contreB2sansprélecture estdéfavorable danssonHC : MAIN−1,557172 %, HOLD−18,111644 %. L’utilisateurannule72W-KEEP ; aucunevarianteweightedàpublier, CHannulé/nonlancé. Lesanciensgainsvslegacyet décisionsci-dessoussont historiques ; lesdonnéesnumériqueset caps77Bfailed restentinchangées.

Écran diagnostique terminé. Condition CH≥1 % sur les deux prompts : True. Un point mesuré par bras/prompt ne confirme pas un gain reproductible. Les limites de compression77B restent échouées. Aucun gain promu.

| Prompt | H legacy | C candidat | Écart | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 7.995750038 tok/s | 9.316084220 tok/s | +16.512950 % | 14.941037/12.750488 s |
| HOLD | 9.597052874 tok/s | 10.772261672 tok/s | +12.245518 % | 12.346993/10.910490 s |

DFlash2fixeB2 confirmé, WIDTH1/ADAPTIVE0, MTPinactif, GPUHot/HotWide et prélecturepipeline1/lookahead2. Seule PREFETCH_VARIANT legacy→weighted diffère entre H/C. Cache3Gio53slots/allocateur64Mio/contexte4096/128sorties127decode/warmup128+mesure128. Les165sources de préparation sont rehashées : seule la CI a changé comme consigné ; sources arithmétiques, CLI77v1, observer et driver identiques. SHA CLI 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. HEADGit89174fcc est une publication, pas un nouveau build utilisé.

Qualité : huit réponses warmup+mesure exactement égales au golden60 OFF pour texte complet non vide, token_hash et comptes128/127. Aucun tableau brut d’IDs persisté ni nouveau contrôle logits/états par cet écran. Quatre rapports validated/reaped/reader_joined ; six étapes pré20/4enfants/post30 code0/reaped ; samplerjoint et secteur à toutes les frontières. Le bridge43 spécifique du candidat72v3 reste préparé/nonexécuté : aucune validation43 transférée depuis packed.

| Bras | Acceptés/proposés/blocs | Lignes vérifiées | Demande experts | Prélectures submitted/completed/used | Réutilisées / timeouts | Attente urgente ns |
|---|---:|---:|---:|---:|---:|---:|
| MAIN C | 58/69/69 | 138 | 27063140220 B | 6177/6177/4554 | 6/2 | 2645707 |
| MAIN H | 58/69/69 | 138 | 26915874804 B | 6453/6453/4802 | 12/0 | 2746209 |
| HOLD C | 56/70/70 | 140 | 29235929112 B | 6306/6306/4713 | 6/2 | 2806626 |
| HOLD H | 56/70/70 | 140 | 28900213884 B | 6566/6566/5054 | 11/3 | 5442583 |

Les compteurs nommés proviennent de qwen4.rs::prefetch_stats dans son ordre [submitted,completed,used,demand_reused,demand_timeouts,demand_wait_ns,bytes,late_needed]. Les octets sont des compteurs moteur de demandes/enregistrements lus ; ils ne mesurent pas les octets physiques SSD ni les écritures swap. Les temps de vérification sont inclusifs : ne pas les additionner à leurs sous-phases ou au decode total. Toute différence de compteurs d’acceptation/largeur est conservée dans pairs[].same_dflash_acceptance_and_width_counters.

Mémoire : picphysique 7853136184 B ; deltas ledger_swapins depuis ready [0, 0, 0, 0] B. Swapouts système parphase [0, 0, 0, 0, 0, 0] B ; Swapins [0, 0, 131072, 0, 0, 0] B, sous la borne stock256Mio. Ces compteurs globaux ne sont pas attribués au modèle. Les analyses77B restent failed avec source_errorsvide ; limites historiques inchangées, aucun seuil futur appliqué rétrospectivement.

| Bras | Δlifetime compression warm→mesure | Δjauge compression | Δdécompressions | Δledger_swapins |
|---|---:|---:|---:|---:|
| MAIN C | 19644416 B | +6340608 B | 31249 | 0 B |
| MAIN H | 166445056 B | +156467200 B | 30898 | 0 B |
| HOLD C | 11943936 B | +9617408 B | 27340 | 0 B |
| HOLD H | 15335424 B | +11501568 B | 17354 | 0 B |

Deltas warm→mesure descriptifs ; ils ne requalifient pas les échecs de vie77B. Ledger_swapins sans hausse ne prouve pas l’absence d’écritures swap privées. Aucun GPU/build/proveur/test lourd lancé par l’analyste, aucun code/plan/seuil modifié.

Preuves : summary.json, quatre report.json/memory-analysis.json, run-result.json, system-samples.json, run.json et identity.json dans ce dossier.

## Révision de décision72W-KEEP — 23octobre2026

L’utilisateur demande explicitement de conserver ce gain malgré sa variabilité. Les verdictsstricts ci-dessus décrivent les critères72S originaux et restent inchangés ; ils ne rejettent plus l’intégration. Weighted est désormais **retenu à la demande utilisateur, validation bridge43 en attente**, avec les quatre gainsvslegacy positifs et leur plage conservée (MAIN+16,512950 à+19,951846 %, HOLD+6,780078 à+12,245518 %). Aucuneaddition, aucuneprésomptiondeboostvsnofetch/MTPnonappariés, aucun15tok/sannoncé. Les capscompression77B échoués restent échoués ; l’app installée n’est pas modifiée. La publication du chemin opt-in reste conditionnée au bridge43weighted, après les témoins d’encadrement récents.

Actualisation après bridgeweightedv4 :43/43 validés, paritétrue etpost30joint. La mention «bridgeenattente» ci-dessus décrit la décision initiale72W-KEEP ; la qualificationfonctionnelle propreaucandidat estdésormaisclose. Publicationopt-inenattenteduparent, nouvelleA/B72W-Ren cours séparément. Lesverdictsstrictshistoriqueset77Bfailedrestentinchangés.
