# T04 — audit final des données, 48 réponses

31 octobre 2026. **Audit données terminé : aucun défaut concret nouveau identifié avant le retrait conditionnel.** Les 12 rapports finalisés et leurs inspections externes v2 sont cohérents : 24 chauffes + 24 mesures, 48 réponses. Cette revue ne lance aucun moteur, GPU, build, test ou preuve et ne modifie ni code, ni protocole, ni journal. L’analyse initiale du premier couple AR est conservée intégralement plus bas et dans le JSON.

## Provenance et travail comparé

Rapports pris dans les 12 entrées `pilot128` du manifeste préparé : AR/MTP/DF, ordre A–C–C–A par mode, même CLI `81bc3a14d509224d77fd6d98d8dc64f9b840c0094f7376f5f004e924849732cd`, seules variables T04=0/1 différentes. Métadonnées runtime identiques hors chargement/mémoire. Les 12 hashes de rapports correspondent à leurs inspections ; processus et lecteurs joints. Les 48 réponses satisfont les contrôles externes texte/hash128, 127 transitions, modes et compteurs homologues. Pic natif de vie 7902009944 octets ; zéro nouveau Swapout selon les inspections et zéro Swapin logique reporté pour ces processus.

Contexte 4096, MAIN 87/HOLD 82 entièrement évalués, contextes finaux 215/210, aucune réutilisation de prompt, cache experts 3 Gio/53 slots/4 lecteurs/lookahead 0 ; mêmes conditions renforcées d’admission 11 Go et ancre T04-w1. Deux mesures seulement par prompt et bras ; les modes ne sont pas des réplications indépendantes du même traitement.

## Débits du pilote complet

Agrégats = somme des tokens / somme des durées ; chauffes exclues. TTFT = moyenne des réponses. Les lignes ALL mélangent MAIN/HOLD uniquement au sein du mode.

| Mode/prompt | Décode A → C (tok/s) | Delta | Préfill A → C (tok/s) | Delta | TTFT A → C (s) |
|---|---:|---:|---:|---:|---:|
| AR/MAIN | 10.617541 → 10.392496 | -2.1196% | 10.243316 → 10.237761 | -0.0542% | 8.493709 → 8.498383 |
| AR/HOLD | 10.323755 → 10.202781 | -1.1718% | 10.357713 → 10.272002 | -0.8275% | 7.917163 → 7.983242 |
| AR/ALL | 10.468587 → 10.296765 | -1.6413% | 10.298505 → 10.254346 | -0.4288% | 8.205436 → 8.240813 |
| MTP/MAIN | 11.042913 → 10.958111 | -0.7679% | 9.816176 → 9.625407 | -1.9434% | 8.863577 → 9.039282 |
| MTP/HOLD | 10.088019 → 10.244776 | +1.5539% | 9.786731 → 9.842481 | +0.5697% | 8.379288 → 8.331866 |
| MTP/ALL | 10.543890 → 10.589444 | +0.4320% | 9.801867 → 9.729524 | -0.7380% | 8.621432 → 8.685574 |
| DF/MAIN | 11.497301 → 11.566876 | +0.6051% | 10.017137 → 10.058470 | +0.4126% | 8.685626 → 8.649942 |
| DF/HOLD | 10.894035 → 11.031666 | +1.2634% | 10.023698 → 10.017059 | -0.0662% | 8.181122 → 8.186724 |
| DF/ALL | 11.187542 → 11.292933 | +0.9420% | 10.020320 → 10.038335 | +0.1798% | 8.433374 → 8.418333 |

AR est négatif sur les deux prompts. MTP : MAIN négatif/HOLD positif ; DF : petits signaux positifs. Aucun gain général sans régression décode n’est qualifié. La dérive du premier au dernier OFF atteint :

| Mode | MAIN | HOLD |
|---|---:|---:|
| AR | -3.1974% | -1.8809% |
| MTP | +2.7552% | +4.3818% |
| DF | +1.2079% | -3.5466% |

Ces dérives, jusqu’à +4,3818% et −3,5466%, empêchent une attribution exclusive des petits deltas au flag. Elles ne sont pas à elles seules une preuve d’inefficacité universelle ; le critère utilisateur n’autorise cependant pas une promotion générale avec ce résultat AR.

## I/O et phases DF

AR/MTP ne reportent aucun compteur de lecture, hit/miss ou route Hot/Cold : aucune égalité ou hausse de lectures ne peut être affirmée pour ces modes. DF conserve exactement le même travail logique à chaque phase/prompt, quel que soit le bras :

| Phase/prompt | Demande logique décode (octets) | Lignes draftées/vérifiées |
|---|---:|---:|
| warmup/MAIN | 23664803544 | 138/138 |
| warmup/HOLD | 25647894612 | 140/140 |
| runs/MAIN | 23889445704 | 138/138 |
| runs/HOLD | 25808888160 | 140/140 |

DF MAIN : 69 propositions/58 acceptées/69 blocs ; HOLD 70/56/70, toutes largeurs 1, prélecture SSD nulle. MTP MAIN 72/54/72 et HOLD 78/48/78. Les compteurs sont identiques A/C, chauffes et mesures homologues comprises. Ces demandes décode ne mesurent ni l’I/O préfill, ni les octets NVMe physiques, ni les lectures critiques ou masquées.

| Phase DF mesurée | A MAIN → C MAIN (s) | A HOLD → C HOLD (s) |
|---|---:|---:|
| draft_seconds | 0.502449458 → 0.508432311 | 0.520153353 → 0.506059919 |
| verification_seconds | 10.309319099 → 10.239644291 | 10.800897292 → 10.670232459 |
| commit_cache_seconds | 0.232891917 → 0.229970623 | 0.236590751 → 0.236766813 |

Vérification DF légèrement plus rapide en C ; draft/commit proches. Horloges inclusives hôte/attente GPU existantes : ne pas les additionner comme des temps exclusifs, et ne pas les utiliser pour expliquer AR ou le préfill.

## Mémoire et compression : aucune cause isolée

| Mode/bras | Empreinte moyenne (octets) | Active MLX moyenne (octets) | Compression logique moyenne (octets) | Somme Δdécompressions des 4 mesures |
|---|---:|---:|---:|---:|
| AR/A | 7148050396 | 6579452032 | 1142132736 | 315 |
| AR/C | 7141365736 | 6579450240 | 1240936448 | 73 |
| MTP/A | 7807130164 | 7293993440 | 2198626304 | 115508 |
| MTP/C | 7805352536 | 7293993824 | 1854152704 | 128071 |
| DF/A | 7738624354 | 7144274048 | 1525198848 | 143344 |
| DF/C | 7740143988 | 7144280704 | 1644855296 | 127692 |

La mémoire active MLX reste quasi identique par mode ; les empreintes moyennes changent de quelques Mo seulement. Les niveaux de compression et leurs deltas ne suivent pas un effet uniforme du flag : en AR, C est plus lent malgré moins de nouvelles décompressions mesurées (73 contre 315 pages A), en MTP C en a plus, en DF moins. Les processus partent de pressions mémoire différentes. Aucun temps de blocage n’est mesuré par ces compteurs ; les différences entre complétions incluent le travail entre requêtes et l’ensemble préfill/décode.

## Conclusion avant nettoyage

**Aucune anomalie concrète nouvelle** de formes, compteurs, charge logique, identité ou budget n’est révélée par les 48 réponses. Les données ne réfutent pas l’audit source déjà effectué : précision/casts/H128/splits/fallbacks conformes, pas de readback/synchronisation/lecture ajoutée identifiée ; les coûts éventuels de troisième entrée, metadata, vue supplémentaire ou codegen ne sont pas isolés. Le contrôleur ne fournit pas de temps route-wait, copie/donation, cache de compilation ou GPU idle.

Le retrait sélectif prévu par `ISSUE59-T04-PILOT-CLOSE` peut donc suivre sa condition « aucun défaut concret trouvé », avec archive exacte et vérification du code restauré. Ce constat n’est ni une autorisation nouvelle de publication ni un résultat de vitesse du moteur final. Les preuves du prototype PR61 correctement adapté, la capture réelle 3→2 dispatchs, les qualités et résultats négatifs doivent rester conservés. Aucun mélange avec T17 ni addition des gains AR/MTP/DF.

Provenance complète des 12 rapports et 12 inspections, recomputation indépendante des agrégats (concordants avec `pilot128-inspection.json`), 48 statistiques et analyse initiale : `audit-performance-data.json`.

---

## Analyse initiale du premier couple — conservée intégralement

Les paragraphes suivants décrivent le périmètre historique A1/C2, avant lecture du reste de la matrice ; leurs limites et formulations prospectives ne remplacent pas la synthèse finale ci-dessus.

# T04 — données du premier couple AR, audit partiel

31 octobre 2026, `T04-AUDIT-PERF-01`. Lecture seule des deux rapports **finalisés** AR128 A1 puis C2 et de leurs inspections v2 : 4 chauffes + 4 mesures, 8 réponses. Aucun autre processus, rapport incomplet ou mode n’est lu ; aucune nouvelle mesure/exécution. **Ce document ne clôt pas la campagne ACCA et ne prédit pas un gain ou rejet général.**

Les hashes de rapports correspondent aux inspections. Même CLI `81bc3a14d509224d77fd6d98d8dc64f9b840c0094f7376f5f004e924849732cd`, environnements identiques sauf T04=0/1 et mêmes métadonnées runtime hors load/mémoire. T17/T05 OFF, contexte 4096, cache experts 3 Gio, 4 lecteurs, lookahead 0. MAIN 87 tokens/contexte final 215, HOLD 82/contexte 210 ; prompts entièrement évalués, 128 sorties/127 transitions, aucun outil. Les processus et lecteurs sont joints, inspections externes passées, zéro nouveau Swapout reporté et zéro Swapin logique des processus.

| Phase/prompt | Décode A → C (tok/s) | Préfill A → C (tok/s) | TTFT A → C (s) |
|---|---:|---:|---:|
| warmup/MAIN | 10.576044 → 10.404743 | 10.004273 → 9.858081 | 8.697264 → 8.826341 |
| warmup/HOLD | 10.462378 → 10.184910 | 10.430704 → 9.917732 | 7.862232 → 8.268999 |
| runs/MAIN | 10.792894 → 10.486501 | 10.402336 → 10.191980 | 8.363853 → 8.536570 |
| runs/HOLD | 10.422705 → 10.229971 | 10.460417 → 10.295194 | 7.839430 → 7.965291 |

Une seule mesure par prompt et bras : pas d’intervalle statistique. Décode mesuré MAIN : +0,343806 s C (TPS −2,839 %) ; HOLD : +0,229566 s (TPS −1,849 %). Ces variations décrivent seulement le premier couple ; la dérive du dernier OFF n’est pas connue dans ce périmètre.

## Mémoire : niveaux et variations

| Mesure | Empreinte A → C (octets) | Active MLX A → C (octets) | Compression logique A → C (octets) | Décompressions depuis réponse précédente A/C |
|---|---:|---:|---:|---:|
| MAIN | 7140923368 → 7149688808 | 6579516032 → 6579513984 | 625442816 → 1226080256 | 0 / 9 |
| HOLD | 7141152744 → 7149197288 | 6579386496 → 6579386496 | 628441088 → 1226080256 | 0 / 0 |

Le niveau de compression C est supérieur, mais les processus démarrent déjà avec des situations différentes : compression au ready A 0, C 169 869 312 octets. Après les chauffes, les mesures ajoutent très peu de décompressions : MAIN A 0/C 9 pages et HOLD A 0/C 0. La compression logique augmente au MAIN mesuré de 55 181 312 octets A contre 262 144 C ; HOLD de 2 998 272 A contre 0 C. Ces deltas concernent l’intervalle entre complétions, avec tout travail entre réponses, préfill et décode ; ils ne donnent ni durée ni attribution de phase. Le niveau comprimé ne peut pas être présenté comme la cause du coût décode.

Empreinte maximale observée aux complétions : A 7 141 152 744 octets, C 7 149 688 808 ; mémoire active et pics MLX restent proches. Cache MLX : A 48 129 986–48 767 346 octets, C 48 224 194–48 801 218. Aucune hausse importante de mémoire active n’explique directement le ralentissement ; cette observation n’exclut ni les attentes ni les copies transitoires.

## Postes absents et causalité

Aucun hit/miss, octet lu logique/physique, nombre de routes Hot/Cold, temps d’attente de routes ou lecture SSD critique/masquée n’est présent dans ces rapports AR. `dflash_profile` est null ; les compteurs spéculatifs sont inactifs. Impossible donc d’affirmer une hausse ou une égalité des lectures à partir de ces données. Aucun temps CPU actif, GPU utile/idle, synchronisation, copie ou cache de compilation n’est mesuré.

**Cause non attribuée.** Le flag, l’ordre des processus, les conditions mémoire initiales et l’état système restent confondus dans ce seul A→C. Continuer à préserver les contrôles prévus ; aucune extrapolation MTP/DF, aucune conclusion avant la suite finalisée, aucune correction proposée sur cette seule corrélation.

Provenance exacte et 8 statistiques complètes : `audit-performance-data.json`, avec SHA-256 des 2 rapports et 2 inspections. Les artefacts source restent inchangés.

