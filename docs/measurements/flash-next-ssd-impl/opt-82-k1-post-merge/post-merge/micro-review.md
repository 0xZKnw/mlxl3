# K1PM : contre-revue des deux microbenchmarks

H128 (704 échantillons, 32 cellules) et SwiGLU (192 échantillons, 8 cellules) sont complets, avec leurs pré/post-observations et samplers joints. La qualité physique et les mutants numériques sont traités séparément dans quality-review.md. Ces micros ne qualifient aucun gain en tok/s.

| Composant | Cellules | Médianes C plus rapides / plus lentes | Étendue des ratios H/C − 1 | A/A max/min maximal |
|---|---:|---:|---:|---:|
| H128 | 32 | 17 / 15 | -5.70 à +45.42 % | 1,598 |
| SwiGLU | 8 | 6 / 2 | -1.40 à +7.14 % | 2,903 |

Ces nombres comptent des cellules ; ils ne sont ni une moyenne du modèle ni une prédiction de débit. Les huit fixtures utilisent des masques de résidence réels figés (plus deux contrôles synthétiques), mais des activations QA/synthétiques. Les quantiles ne sont pas des poids de fréquence.

H128 : le +45,42 % de MAIN p05 GU-input compare 35,292 à 24,268 µs/appel amorti. La même géométrie donne une baseline HOLD p05 de 10,717 µs, et les A/A MAIN p50 varient de 37,43 %. Plusieurs petites géométries régressent. Le contrôle tous Hot est plus lent sur 3/4 étapes. HOLD p05 GU-input reste positif dans 5/5 blocs (+10,32 à +12,56 %), mais cela ne suffit pas à une économie globale.

SwiGLU : MAIN p05 −1,40 %, p50 +2,74 %, p95 +1,57 % ; HOLD p05 −0,80 %, p50 +3,77 %, p95 +7,14 %. HOLD p50/p95 ont 5/5 blocs positifs. Tous Hot +4,11 % et tous Cold +4,59 %. A/A HOLD p95 max/min 2,903 ; certains blocs atteignent +44,37 % ou −43,65 %. Les valeurs brutes restent toutes incluses, sans attribution aux clocks, à la donation ou au thermique.

Le temps par appel divise une chaîne de 32 par 32. Ce n’est pas une latence GPU exclusive : construction hôte, dispatch et completion restent dans l’horloge. H128 n’imite pas les quatre transformations entrelacées avec QMV/scales ; SwiGLU ajoute 31 récurrences communes. Les oracles de chaîne et lectures hôte sont hors chronométrage, et les 8 warmups/bras sont des appels simples.

Mémoire : les six phases des deux micros montrent 0 nouvelle entrée/sortie de swap système aux endpoints, AC aux endpoints, enfants et samplers joints. Le compteur live manque pendant les phases sous-seconde ; null ne devient pas zéro. Aucun pic physique de modèle ou volume de sorties de swap privées n’est déduit. Les chiffres de compression système sont des observations non attribuées.

Identité : les 100 SHA source actuels correspondent au manifeste 2b4eb648 ; les deux plans et les fixtures 50bf33e3 sont exacts. Les micros utilisent le natif Release 041b3b09. Le vrai HC/CH compare les CLI d8757fa6 et cc9669ad séparément ; ses résultats ne sont pas réutilisés dans cette revue.

Décision : conserver les micros comme diagnostic, attendre les vraies mesures DFlash2 HC/CH. Aucun gain du modèle promu et aucun pourcentage additionné. Aucun build, proveur, test ou GPU lancé pour cette analyse.

Détails reproductibles : plans/h128-micro/analysis.{json,md}, plans/h128-micro/analysis-cells.csv et plans/activation-micro/analysis.{json,md}, plans/activation-micro/analysis-cells.csv ; chaque JSON conserve les cinq blocs, A/A et dérives. micro-review.json fixe les empreintes et limites.
