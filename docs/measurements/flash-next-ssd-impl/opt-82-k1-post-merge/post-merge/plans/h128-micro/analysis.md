# H128 après merge — 704 échantillons, diagnostic du composant

**32 cellules contrôlées, aucun gain en tok/s déduit.** Cette matrice comporte trois phases pre20/micro/post30, toutes exit0/jointes et sampler joint. Le rapport contient8plans/32setups/704échantillons exacts, parité single+32-chain Hot aux bits/finitude, Cold+0 candidat (zéro signé accepté côté natif).

Chaque cellule a deux observations native/native AA puis cinq blocs A-B-B-A, donc10temps par bras. Les médianes ci-dessous excluent AA. Les µs/appel sont le temps mural de32transformations chaînées divisé par32, avec host/dispatch/completion ; pas une latence GPU exclusive ni32observations indépendantes. Huit warmups d’un appel par bras, setup exclu. Aucune pondération et aucune addition entre cellules.

| Masque | Hot/20 | Étape | Natif µs/appel | Candidat µs/appel | Débit composant Δ | AA max/min | Blocs positifs/5 |
|---|---:|---|---:|---:|---:|---:|---:|
| MAIN-p05 | 7 | GU-input | 35.292 | 24.268 | +45.42% | 1.063 | 5 |
| MAIN-p05 | 7 | GU-output | 20.732 | 17.918 | +15.71% | 1.049 | 3 |
| MAIN-p05 | 7 | down-input | 18.754 | 18.434 | +1.74% | 1.155 | 4 |
| MAIN-p05 | 7 | down-output | 23.946 | 19.898 | +20.34% | 1.012 | 5 |
| MAIN-p50 | 14 | GU-input | 25.576 | 22.255 | +14.93% | 1.598 | 4 |
| MAIN-p50 | 14 | GU-output | 15.068 | 15.420 | -2.28% | 1.081 | 2 |
| MAIN-p50 | 14 | down-input | 15.384 | 15.058 | +2.17% | 1.004 | 1 |
| MAIN-p50 | 14 | down-output | 13.812 | 13.699 | +0.82% | 1.025 | 2 |
| MAIN-p95 | 19 | GU-input | 15.571 | 15.325 | +1.61% | 1.004 | 4 |
| MAIN-p95 | 19 | GU-output | 9.596 | 10.085 | -4.85% | 1.025 | 1 |
| MAIN-p95 | 19 | down-input | 9.105 | 9.300 | -2.10% | 1.011 | 2 |
| MAIN-p95 | 19 | down-output | 9.549 | 9.704 | -1.60% | 1.026 | 2 |
| HOLD-p05 | 7 | GU-input | 10.717 | 9.584 | +11.82% | 1.021 | 5 |
| HOLD-p05 | 7 | GU-output | 8.490 | 8.648 | -1.83% | 1.024 | 2 |
| HOLD-p05 | 7 | down-input | 8.196 | 8.265 | -0.84% | 1.015 | 0 |
| HOLD-p05 | 7 | down-output | 9.117 | 8.899 | +2.44% | 1.008 | 4 |
| HOLD-p50 | 14 | GU-input | 10.607 | 10.020 | +5.86% | 1.039 | 4 |
| HOLD-p50 | 14 | GU-output | 8.546 | 8.667 | -1.40% | 1.020 | 1 |
| HOLD-p50 | 14 | down-input | 8.119 | 8.260 | -1.70% | 1.025 | 1 |
| HOLD-p50 | 14 | down-output | 9.204 | 9.391 | -1.98% | 1.074 | 2 |
| HOLD-p95 | 19 | GU-input | 10.561 | 10.540 | +0.20% | 1.035 | 3 |
| HOLD-p95 | 19 | GU-output | 8.473 | 8.666 | -2.23% | 1.025 | 0 |
| HOLD-p95 | 19 | down-input | 8.161 | 8.462 | -3.55% | 1.034 | 3 |
| HOLD-p95 | 19 | down-output | 9.171 | 9.445 | -2.90% | 1.094 | 0 |
| synthetic-all-hot | 20 | GU-input | 10.720 | 10.624 | +0.90% | 1.001 | 2 |
| synthetic-all-hot | 20 | GU-output | 8.386 | 8.893 | -5.70% | 1.069 | 0 |
| synthetic-all-hot | 20 | down-input | 8.142 | 8.403 | -3.11% | 1.031 | 0 |
| synthetic-all-hot | 20 | down-output | 9.104 | 9.490 | -4.07% | 1.010 | 1 |
| synthetic-all-cold-nonempty-map | 0 | GU-input | 10.691 | 9.094 | +17.56% | 1.002 | 4 |
| synthetic-all-cold-nonempty-map | 0 | GU-output | 8.589 | 8.117 | +5.82% | 1.029 | 5 |
| synthetic-all-cold-nonempty-map | 0 | down-input | 8.251 | 7.923 | +4.15% | 1.023 | 5 |
| synthetic-all-cold-nonempty-map | 0 | down-output | 9.180 | 8.421 | +9.01% | 1.011 | 5 |

17médianes candidat plus rapides et15plus lentes ne forment pas un gain global. L’effet temporel est manifeste : même géométrie GU-input, natif MAINp05 35,292µs contre HOLDp05 10,717µs ; MAINp50 AA baisse de37,43% entre deux observations natives. Les données changent aussi entre masques/couches ; ni cause horloge, ni donation, ni thermique affirmée. Le +45,42% du premier masque ne doit pas être transposé au modèle.

HOLDp05 GU-input a cinq blocs positifs (+10,32…+12,56%), mais GU-output/down de nombreuses cellules sont négatifs ou mélangés. Le contrôle all-hot régresse sur3/4médianes (-5,70/-3,11/-4,07%), GU-input+0,90% seulement et2/5blocs positifs. Un dispatch sélectif géométrique serait une nouvelle expérience à préenregistrer, pas un gain tiré de cette matrice.

Les plages parbloc et dérives début/fin figurent dans analysis.json ; ce sont des étendues observées, pas des intervalles de confiance. InputsQA-derived et masquesréelsB2, aucune activation GU/down réellement capturée ; Coldinitialisé0. La donation du natif peut différer du custom, mais son coût n’a pas été isolé. Aucun rapprochement chiffré vers15tok/s.

Mémoire : trois phases endpoints swap-in/out0 et samplers35joints sans erreur ; compression physique système maximum4300652544octets, sans attribution. Pas de mesure exhaustive du pic physique processus des composants.
