# 82F06 — écran H-C, audit indépendant

Quatre rapports clos et `validated`, enfants et lecteurs joints, post30 et sampler terminés. Les huit réponses warmup/mesure ont le texte **et** le hash gold60 exacts, 128 tokens générés / 127 decode, DFlash2 fixe B2 actif et MTP inactif. Les cinq compteurs MTP sont **JSON null**, conformément au schéma inactif ; zéro entier n’est pas l’oracle.

| Prompt | H tok/s | C tok/s | C/H |
|---|---:|---:|---:|
| MAIN | 11.206068 | 11.170360 | -0.3186% |
| HOLD | 10.585691 | 10.711404 | +1.1876% |

Chaque phase a les mêmes acceptations, largeurs et octets SSD logiques entre H et C : MAIN 58/69, HOLD 56/70 ; 138/140 lignes drafted et vérifiées. Les histogrammes ont huit cases : index 1 = bloc de deux lignes, ancre + une proposition, avec 69/70 blocs et les autres cases à zéro. La prélecture reste nulle. Il ne s’agit pas de trafic SSD physique.

Dérives warmup→mesure :

- MAIN-0-H : -1.025% ; décompressions propres +30960, compression logique +165986304 octets, footprint +3031040 octets.
- MAIN-1-C : +0.216% ; décompressions propres +30730, compression logique +130596864 octets, footprint -2916352 octets.
- HOLD-0-H : -0.847% ; décompressions propres +30848, compression logique +174522368 octets, footprint +147432 octets.
- HOLD-1-C : +2.396% ; décompressions propres +30455, compression logique +155648000 octets, footprint -16384 octets.

Les six phases sont jointes et secteur AC. Swap-out système : zéro dans chaque phase et dans le monitor. Swap-in système : **1 294 336 octets au total**, dont 65 536 pour MAIN-C et 1 228 800 pour HOLD-H ; ces deltas sont également observés en direct. Les quatre autres phases ont zéro delta swap-in. L’origine de ces pages n’est pas attribuée au modèle. Sampler : 120 observations sans erreur. Sur douze snapshots ready/warmup/mesure, pic physique de durée de vie **7.851 Go**, footprint courant maximal **7.758 Go**, compression logique courante maximale **3.089 Go**, ledger swap-in propre zéro. Les anciens caps compression 77B restent non qualifiés ; les octets logiques compressés ne sont pas le volume physique du compresseur. Ledger nul ne prouve pas absence de swap-out privé.

Les durées de vérification/draft/commit sont conservées en secondes et millisecondes par token dans le JSON, comme intervalles inclusifs côté hôte. Ne pas les additionner aux durées decode/block ou les appeler temps GPU exclusifs. Aucune cause matérielle n’est déduite des dérives ou de la compression. Les courbes VM système sont archivées séparément et n’attribuent pas leurs variations au modèle.

**Confirmation v1 exclue intégralement.** Son préflight attendait à tort des compteurs MTP zéro et lisait mal l’index des histogrammes ; le shell sans errexit a poursuivi le lancement. La phase a été interrompue puis jointe, et le rapport partiel est incomplet. Le sampler est joint ; aucun chiffre de ce répertoire n’est réutilisé ici. Seul le nouveau `confirm-v2` peut qualifier C-H. Les échecs historiques restent visibles.

Les 85 sources, le runtime `cbc03630` et le contrôleur `8935` concordent avec l’identité gelée au moment de cette revue. Aucun gain promu sur cet écran seul. Aucun test, benchmark, build, proveur, GPU ou fichier W modifié par cet audit. Détails exacts dans `cross-review-screen.json`.
