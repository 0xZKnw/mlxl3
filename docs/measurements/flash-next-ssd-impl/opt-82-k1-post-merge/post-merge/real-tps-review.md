# K1PM : contre-revue des débits H-C puis C-H

Aucun gain K1 observé dans ces deux ordres. Les deux prompts sont légèrement plus lents avec le candidat. Ce résultat ne démontre ni un bug d’implémentation ni une cause exclusive du ralentissement.

| Prompt | H-C : H → C tok/s | Écart C/H | C-H : H → C tok/s | Écart C/H | Géomoyenne des ratios |
|---|---:|---:|---:|---:|---:|
| MAIN | 11.399755 → 11.254234 | -1.2765 % | 11.576340 → 11.439044 | -1.1860 % | -1.2313 % |
| HOLD | 10.903516 → 10.896760 | -0.0620 % | 11.128820 → 10.946912 | -1.6346 % | -0.8514 % |

La géomoyenne résume deux ratios appariés ; aucun pourcentage additionné et aucun intervalle de confiance. Les témoins H progressent de 1,549 % entre ordres sur MAIN et 2,066 % sur HOLD. Une mesure par bras après un warmup reste une portée limitée.

Protocole : 8 processus de benchmark, chacun un warmup de 128 tokens puis une génération mesurée de 128 (127 décodés). Les 16 textes et hashes non vides égalent exactement le gold60 externe. B2 fixe (une proposition), adaptatif 0, draft Q4/ring, cache 3 Gio, allocateur 64 Mio, Hot Wide/UNIQUE/63 actifs ; prélecture, lookahead et shared désactivés. MTP inactif publie des compteurs null, pas zéro.

Compteurs identiques : MAIN 69 propositions/blocs, 58 acceptations, 138 lignes ; HOLD 70/56/140, plus un token de queue séquentielle. Les huit cases des histogrammes n’utilisent que l’index 1. Demande logique des experts mesurée : 23 930 630 100 octets MAIN et 25 474 420 944 HOLD, identique entre bras et ordres. Ce compteur ne mesure pas les lectures physiques SSD.

| Ordre/phase | Warmup tok/s | Mesure tok/s | Variation warm → mesure | Vérification ms/token décodé |
|---|---:|---:|---:|---:|
| HC MAIN-0-H | 10.714169 | 11.399755 | +6.399 % | 81.735 |
| HC MAIN-1-C | 11.439212 | 11.254234 | -1.617 % | 82.859 |
| HC HOLD-0-H | 10.804659 | 10.903516 | +0.915 % | 85.000 |
| HC HOLD-1-C | 10.700635 | 10.896760 | +1.833 % | 85.047 |
| CH MAIN-0-C | 11.371622 | 11.439044 | +0.593 % | 81.601 |
| CH MAIN-1-H | 11.377783 | 11.576340 | +1.745 % | 80.447 |
| CH HOLD-0-C | 10.915374 | 10.946912 | +0.289 % | 84.633 |
| CH HOLD-1-H | 10.940942 | 11.128820 | +1.717 % | 83.273 |

Vérification, draft et commit/cache restent séparés dans le JSON. La vérification inclut backbone, MoE, routage et attentes SSD : elle n’est pas le temps exclusif des nouveaux kernels et ne doit pas s’ajouter à ses sous-phases.

Mémoire : 12 phases attendues terminées, enfants et 2 samplers joints sans erreur (120 + 118 observations système ; 108 + 107 points VM). Secteur aux endpoints ; 0 nouvelle sortie de swap système dans les 12 phases et leurs maximums live. Les entrées système sont de 65 536 octets pendant le post30 H-C (maximum live identique), de 0 pendant les 8 processus mesurés et de 0 sur tout C-H. Pic physique lifetime maximal publié : 7,851,071,800 octets, inférieur à 12 Go. Ledger d’entrées privées : 0 sur les 24 snapshots ready/complete. Aucun compteur de sorties privées ; ces zéros ne prouvent pas l’absence de pression mémoire.

Compression logique courante des snapshots : 0–3,929,571,328 octets (ready inclus à zéro). Une différence nette existe par exemple entre MAIN H-C H ≈3,93 Go et C ≈1,68 Go ; origine non attribuée. Compression physique système HC 4,116,742,144–6,869,106,688 octets, CH 4,723,965,952–6,772,686,848. Les critères historiques 77B ne sont pas requalifiés par ces observations.

Identités : 100 SHA source candidat inchangés, 7 drivers et 2 plans exacts ; CLI réhashées d8757fa6 (baseline) et cc9669ad (candidat). La metadata baseline e551-dirty est historique : ses 91 sources runtime avaient été comparées byte à byte à la base fusionnée 9c8474ca. Manifeste et SHA binaire fixent l’identité testée.

Portée : textes/hashes et compteurs ne remplacent pas les logits/états de la qualité physique passée. Six mutants numériques détectés séparément. Le bridge K1 de 43 requêtes et les débits OFF/MTP ne font pas partie de cette campagne. Contrats CPU non équivalents à une preuve MLX/Metal/concurrence.

Conclusion : aucune raison d’intégrer K1 pour un gain de débit sur ces données. Les opportunités locales des micros ne se retrouvent pas dans les deux vrais ordres. Aucune preuve ici que K1 a été mal implémenté ; aucun coût de donation, clock ou dispatch n’est isolé pour expliquer le résultat. Aucun modèle, test, build ou proveur lancé par cette analyse.

Preuves : plans/screen et plans/confirm (bruts, rapports, VM, supervision), plans/confirm/summary.json ; real-tps-review.json conserve les 16 réponses résumées, profils, mémoires et empreintes ; quality-review.* et micro-review.* fixent les contrôles distincts.

Correction avant clôture : la première prose indiquait à tort zéro entrée système partout, alors que les données JSON conservaient déjà les 65 536 octets de post30 H-C. Ancienne prose/JSON archivés dans review-versions ; aucun rerun.
