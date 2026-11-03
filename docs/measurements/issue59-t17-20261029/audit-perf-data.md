# T17 — audit des données après pilote (31 octobre 2026)

Analyse en lecture seule des 12 rapports et 12 inspections v2 : 48 réponses (24 chauffes + 24 mesures), aucune nouvelle exécution. Les empreintes des rapports correspondent aux inspections. Tous les rapports sont `validated`, processus et lecteurs joints, inspections externes passées avec zéro nouveau Swapout. Cet audit ne réexécute pas les oracles.

Environnements reportés identiques sauf le flag T17 ; métadonnées runtime identiques hors temps de chargement et mémoire. Même CLI SHA-256 `202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa`, MLX 0.32.2, contexte 4096, commit `7978fceac30d-dirty`. MAIN : 87 tokens, contexte final 215 ; HOLD : 82 tokens, contexte final 210. Aucun token de prompt réutilisé ; 128 sorties, 127 transitions, un round sans outils. Les options `MLX_MAX_*` figurent dans les plans figés, hors du filtre environnement des rapports.

## Mesures existantes, chauffes exclues

Débits agrégés = nombre de tokens / somme des durées. Deux mesures par prompt et bras ; aucun intervalle statistique estimé.

| Mode/prompt | Décode A → C (tok/s) | Delta | Préfill A → C (tok/s) | Delta |
|---|---:|---:|---:|---:|
| AR/MAIN | 10.413438 → 10.461347 | +0.460% | 10.150843 → 10.099876 | -0.502% |
| AR/HOLD | 10.232231 → 10.207064 | -0.246% | 10.069126 → 10.235128 | +1.649% |
| AR/ALL | 10.322039 → 10.332642 | +0.103% | 10.111028 → 10.165052 | +0.534% |
| MTP/MAIN | 11.050266 → 11.138336 | +0.797% | 9.863628 → 9.869019 | +0.055% |
| MTP/HOLD | 10.178883 → 10.237115 | +0.572% | 9.856849 → 9.896377 | +0.401% |
| MTP/ALL | 10.596691 → 10.668727 | +0.680% | 9.860338 → 9.882274 | +0.222% |
| DF/MAIN | 11.735541 → 11.713367 | -0.189% | 10.224518 → 10.182861 | -0.407% |
| DF/HOLD | 11.188145 → 11.254292 | +0.591% | 10.178491 → 8.923625 | -12.329% |
| DF/ALL | 11.455307 → 11.479242 | +0.209% | 10.202133 → 9.530330 | -6.585% |

La dérive des références reste présente. Entre le premier et le dernier OFF, les débits décode MAIN/HOLD changent respectivement :

- AR : -4.375% / -2.730%.
- MTP : +1.788% / +1.963%.
- DF : +2.295% / +1.695%.

## Particularité DF/HOLD

Ordre fixe : chauffe MAIN, chauffe HOLD, mesure MAIN, mesure HOLD. Le ralentissement concerne la quatrième réponse des deux processus ON ; leurs chauffes HOLD restent proches des OFF.

| Processus | Chauffe HOLD (tok/s) | Mesure HOLD (tok/s) | Préfill mesuré (s) | Décode mesuré (s) | Décompressions depuis la réponse précédente |
|---|---:|---:|---:|---:|---:|
| DF 1 A | 10.053667 | 10.080991 | 8.134121 | 11.446715 | 28709 |
| DF 2 C | 10.056719 | 8.911257 | 9.201845 | 11.287581 | 50577 |
| DF 3 C | 10.188873 | 8.936028 | 9.176336 | 11.281585 | 39355 |
| DF 4 A | 10.190808 | 10.277896 | 7.978287 | 11.255887 | 27445 |

Le préfill passe en moyenne de 8.056204 s à 9.189091 s (+1.132887 s). Le TTFT passe de 8.056665 à 9.189600 s. Le chargement draft aux quatre mesures HOLD vaut 34.583 / 35.208 / 36.459 / 37.292 µs : aucune recharge coûteuse observée. Ces durées viennent des rapports existants.

## I/O et phases DFlash

Aucun compteur hits/misses ou I/O préfill dans ces rapports. Les seules demandes experts disponibles concernent le décode DFlash : compteur logique, sans mesure des octets physiques ni des lectures critiques/masquées. Les valeurs sont identiques A/C et aux deux occurrences de chaque bras :

| Phase/prompt | Demandes logiques experts (octets) |
|---|---:|
| Chauffe MAIN | 23664803544 |
| Chauffe HOLD | 25647894612 |
| Mesure MAIN | 23889445704 |
| Mesure HOLD | 25808888160 |

Tous les DF MAIN ont 69 propositions, 58 acceptées, 69 blocs, 138 lignes draftées/vérifiées. HOLD : 70 propositions, 56 acceptées, 70 blocs, 140 lignes. Largeur 1 exclusive ; tous les compteurs de prélecture SSD valent zéro. Aucun changement de travail spéculatif détecté.

| DF/HOLD mesuré | Moyenne A (s) | Moyenne C (s) |
|---|---:|---:|
| draft_seconds | 0.498224438 | 0.507277294 |
| verification_seconds | 10.525191411 | 10.451495351 |
| commit_cache_seconds | 0.231252311 | 0.229852293 |

Ces horloges couvrent le décode, avec frontières inclusives CPU/GPU/I/O ; elles ne décomposent pas le préfill. La vérification ne montre pas la pénalité préfill : OFF 10.613244 / 10.437139 s contre ON 10.446718 / 10.456273 s. Aucun temps de lecture SSD critique ou masquée ne peut être reconstruit.

## Mémoire et limites de causalité

| DF/HOLD mesuré | Moyenne A | Moyenne C |
|---|---:|---:|
| process_footprint_bytes (octets) | 7727061340 | 7734319452 |
| process_compressed_logical_bytes (octets) | 1623220224 | 1715503104 |
| mlx_active_bytes (octets) | 7144222848 | 7144222848 |
| mlx_cache_bytes (octets) | 67491962 | 67569487 |
| mlx_peak_bytes (octets) | 7289337194 | 7289275284 |

L’empreinte moyenne augmente de 7258112 octets, sans saut important de mémoire active MLX. La compression change aussi entre processus OFF ; ses niveaux ne prouvent pas un temps de blocage.

Les incréments de décompressions entre fins de réponses sont plus grands sur HOLD ON : 50 577 / 39 355 contre 28 709 / 27 445 OFF. Ce sont des pages du processus, pas des lectures SSD ni des millisecondes. L’intervalle inclut tout travail entre réponses, préfill et décode ; aucun snapshot ne marque la fin du préfill.

Contre-exemple à une causalité simple : le MAIN mesuré du troisième processus ON ajoute 47746 décompressions tout en conservant 10.260480 tok/s de préfill. Le nombre de pages décompressées ne suffit donc pas à isoler la pénalité HOLD.

Les 48 valeurs `process_ledger_swapins_logical_bytes` valent zéro ; les inspections rapportent zéro nouveau Swapout. Cela n’exclut ni la compression/décompression ni les lectures du modèle ni une attente CPU/GPU. La mémoire compresseur n’est pas un compteur SSD.

Cause connue : aucune. Le signal préfill DF/HOLD se répète dans les deux ON après trois réponses antérieures ; ordre et état résiduel restent confondus avec le flag. Les données ne tranchent pas cache de factories/JIT, copies/donation, exp precise, fragmentation, coût du graphe ou pression système. Une hypothèse source concrète ou un diagnostic ciblé préenregistré doit isoler le poste avant correction.

Autres postes non mesurés : hits/misses et I/O préfill, I/O décode AR/MTP, I/O physique, chemin critique SSD, GPU idle, temps kernel/CPU exclusifs, nombre d’allocations, cache factory hits/évictions et fréquences thermiques. Le petit échantillon et la dérive des références restent visibles. Aucun gain global établi par cet audit.

`audit-perf-data.json` conserve les 48 statistiques, métadonnées comparées, deltas avec leur portée, agrégats et empreintes des 24 fichiers sources. Aucun exécutable ni `opti.md` modifié.
