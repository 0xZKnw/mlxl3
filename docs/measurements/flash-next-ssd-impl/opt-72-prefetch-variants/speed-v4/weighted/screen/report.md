# 72 weighted — écran B2 H-C

**Décision actuelle — retrait utilisateur72W-WITHDRAW.** Le vrai A/B contreB2sansprélecture estdéfavorable danssonHC : MAIN−1,557172 %, HOLD−18,111644 %. L’utilisateurannule72W-KEEP ; aucunevarianteweightedàpublier, CHannulé/nonlancé. Lesanciensgainsvslegacyet décisionsci-dessoussont historiques ; lesdonnéesnumériqueset caps77Bfailed restentinchangées.

Écran diagnostique terminé. Condition CH≥1 % sur les deux prompts : True. Un point mesuré par bras/prompt ne confirme pas un gain reproductible. Les limites de compression77B restent échouées. Aucun gain promu.

| Prompt | H legacy | C candidat | Écart | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 7.993086484 tok/s | 9.587854769 tok/s | +19.951846 % | 14.930680/12.347447 s |
| HOLD | 8.740291796 tok/s | 9.332890413 tok/s | +6.780078 % | 13.567380/12.601526 s |

DFlash2fixeB2 confirmé, WIDTH1/ADAPTIVE0, MTPinactif, GPUHot/HotWide et prélecturepipeline1/lookahead2. Seule PREFETCH_VARIANT legacy→weighted diffère entre H/C. Cache3Gio53slots/allocateur64Mio/contexte4096/128sorties127decode/warmup128+mesure128. Les165sources de préparation sont rehashées : seule la CI a changé comme consigné ; sources arithmétiques, CLI77v1, observer et driver identiques. SHA CLI 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. HEADGit89174fcc est une publication, pas un nouveau build utilisé.

Qualité : huit réponses warmup+mesure exactement égales au golden60 OFF pour texte complet non vide, token_hash et comptes128/127. Aucun tableau brut d’IDs persisté ni nouveau contrôle logits/états par cet écran. Quatre rapports validated/reaped/reader_joined ; six étapes pré20/4enfants/post30 code0/reaped ; samplerjoint et secteur à toutes les frontières. Le bridge43 spécifique du candidat72v3 reste préparé/nonexécuté : aucune validation43 transférée depuis packed.

| Bras | Acceptés/proposés/blocs | Lignes vérifiées | Demande experts | Prélectures submitted/completed/used | Réutilisées / timeouts | Attente urgente ns |
|---|---:|---:|---:|---:|---:|---:|
| MAIN H | 58/69/69 | 138 | 26910882756 B | 6453/6453/4800 | 21/3 | 7181084 |
| MAIN C | 58/69/69 | 138 | 27064388232 B | 6177/6177/4552 | 2/0 | 448541 |
| HOLD H | 56/70/70 | 140 | 28893973824 B | 6566/6566/5056 | 3/0 | 1510750 |
| HOLD C | 56/70/70 | 140 | 29233433088 B | 6306/6306/4713 | 5/0 | 1566499 |

Les compteurs nommés proviennent de qwen4.rs::prefetch_stats dans son ordre [submitted,completed,used,demand_reused,demand_timeouts,demand_wait_ns,bytes,late_needed]. Les octets sont des compteurs moteur de demandes/enregistrements lus ; ils ne mesurent pas les octets physiques SSD ni les écritures swap. Les temps de vérification sont inclusifs : ne pas les additionner à leurs sous-phases ou au decode total. Toute différence de compteurs d’acceptation/largeur est conservée dans pairs[].same_dflash_acceptance_and_width_counters.

Mémoire : picphysique 7853824288 B ; deltas ledger_swapins depuis ready [0, 0, 0, 0] B. Swapouts système parphase [0, 0, 0, 0, 0, 0] B ; Swapins [0, 0, 32768, 0, 0, 0] B, sous la borne stock256Mio. Ces compteurs globaux ne sont pas attribués au modèle. Les analyses77B restent failed avec source_errorsvide ; limites historiques inchangées, aucun seuil futur appliqué rétrospectivement.

| Bras | Δlifetime compression warm→mesure | Δjauge compression | Δdécompressions | Δledger_swapins |
|---|---:|---:|---:|---:|
| MAIN H | 220512256 B | +187072512 B | 32671 | 0 B |
| MAIN C | 175472640 B | +163299328 B | 31211 | 0 B |
| HOLD H | 9404416 B | +3063808 B | 30703 | 0 B |
| HOLD C | 103219200 B | +97714176 B | 30620 | 0 B |

Deltas warm→mesure descriptifs ; ils ne requalifient pas les échecs de vie77B. Ledger_swapins sans hausse ne prouve pas l’absence d’écritures swap privées. Aucun GPU/build/proveur/test lourd lancé par l’analyste, aucun code/plan/seuil modifié.

Preuves : summary.json, quatre report.json/memory-analysis.json, run-result.json, system-samples.json, run.json et identity.json dans ce dossier.

## Révision de décision72W-KEEP — 23octobre2026

L’utilisateur demande explicitement de conserver ce gain malgré sa variabilité. Les verdictsstricts ci-dessus décrivent les critères72S originaux et restent inchangés ; ils ne rejettent plus l’intégration. Weighted est désormais **retenu à la demande utilisateur, validation bridge43 en attente**, avec les quatre gainsvslegacy positifs et leur plage conservée (MAIN+16,512950 à+19,951846 %, HOLD+6,780078 à+12,245518 %). Aucuneaddition, aucuneprésomptiondeboostvsnofetch/MTPnonappariés, aucun15tok/sannoncé. Les capscompression77B échoués restent échoués ; l’app installée n’est pas modifiée. La publication du chemin opt-in reste conditionnée au bridge43weighted, après les témoins d’encadrement récents.

Actualisation après bridgeweightedv4 :43/43 validés, paritétrue etpost30joint. La mention «bridgeenattente» ci-dessus décrit la décision initiale72W-KEEP ; la qualificationfonctionnelle propreaucandidat estdésormaisclose. Publicationopt-inenattenteduparent, nouvelleA/B72W-Ren cours séparément. Lesverdictsstrictshistoriqueset77Bfailedrestentinchangés.
