# K1PM : analyse indépendante du micro SwiGLU

192 échantillons, 8 cellules, parité complète byte à byte et finitude vérifiées par le test. Trois phases et sampler joints. Ce sont des temps muraux de composant, pas des tok/s.

Chaque cellule contient 4 A/A natives, puis 5 blocs A–B–B–A. Les médianes ci-dessous excluent A/A et divisent le temps des 32 activations par 32. Les 31 récurrences communes, la construction du graphe, le dispatch et l’attente finale restent inclus. Les 8 warmups par bras sont des appels simples. Aucun agrégat pondéré par trace.

| Fixture | Hot/20 | Natif µs/appel | Candidat µs/appel | Ratio H/C − 1 | Blocs positifs | A/A max/min |
|---|---:|---:|---:|---:|---:|---:|
| MAIN-p05 | 7 | 31.757 | 32.209 | -1.40 % | 4/5 | 1.319 |
| MAIN-p50 | 14 | 32.318 | 31.456 | +2.74 % | 4/5 | 1.032 |
| MAIN-p95 | 19 | 20.748 | 20.427 | +1.57 % | 2/5 | 1.340 |
| HOLD-p05 | 7 | 17.400 | 17.540 | -0.80 % | 1/5 | 1.072 |
| HOLD-p50 | 14 | 15.588 | 15.021 | +3.77 % | 5/5 | 1.054 |
| HOLD-p95 | 19 | 14.376 | 13.417 | +7.14 % | 5/5 | 2.903 |
| synthetic-all-hot | 20 | 14.087 | 13.531 | +4.11 % | 5/5 | 1.021 |
| synthetic-all-cold-nonempty-map | 0 | 14.171 | 13.549 | +4.59 % | 5/5 | 1.077 |

Six médianes sont plus rapides, deux plus lentes : ce comptage ne représente aucun gain global. HOLD p50 et p95 ont 5/5 blocs positifs. Le contrôle tous Hot est aussi plus rapide (+4,11 %), alors qu’aucune ligne Cold n’y est évitée ; le test mesure donc aussi les différences de fusion, dispatch et graphe. Leur coût exclusif reste non mesuré.

L’hétérogénéité reste visible : A/A HOLD p95 va de 41,229 à 14,202 µs/appel (max/min 2,903). Les cinq blocs MAIN p05 vont de −16,03 à +44,37 %, ceux de HOLD p05 de −43,65 à +3,97 %. Aucun échantillon n’a été filtré. Les intervalles sont des étendues observées, pas des intervalles de confiance. Ni clock, ni donation, ni thermique ne sont établis comme cause.

Les trois phases ont 0 nouvelle entrée/sortie de swap système aux endpoints ; 34 observations système et 23 VM, sampler joint. Compression système physique observée 4157390848–4164976640 octets. Pas de pic physique du modèle ni de compteur de sorties de swap privées. La phase sous-seconde n’a pas d’échantillon VM live : null reste absence d’observation.

Sources et fixtures : manifeste de 100 fichiers inchangé, SHA du plan et des fixtures contrôlés. Binaire natif déclaré 041b3b09, déjà contrôlé dans la revue qualité ; pas de rehash du gros binaire pendant les vrais HC. Les limites formelles sont dans quality-review.md. Aucun nouveau test, build, preuve ou GPU lancé par cette analyse.

Preuves : component.json, run-result.json, system-samples.json, vm-curve.json, analysis.json (tous les blocs et dérives), analysis-cells.csv.
