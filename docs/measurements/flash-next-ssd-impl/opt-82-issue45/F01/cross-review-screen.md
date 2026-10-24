# 82F01 — écran H-C, revue documentaire indépendante

Les quatre processus et leur post30 sont clos, joints et `validated`. Les huit réponses (warmup puis mesure) correspondent exactement au texte **et** au hash de l’oracle externe 60, avec 128 tokens / 127 decode, DFlash2 actif, MTP inactif et une proposition (B2). Aucun nouveau GPU, test, build ou proveur lancé par cette revue.

| Prompt | H tok/s | C tok/s | Écart C/H |
|---|---:|---:|---:|
| MAIN | 11.13848 | 10.63319 | -4.536% |
| HOLD | 10.16987 | 10.32175 | +1.493% |

Il n’y a pas de gain global qualifié : MAIN est négatif et HOLD ne dispose que d’une paire positive. La confirmation C-H prévue reste nécessaire pour distinguer l’effet du candidat de l’ordre des processus.

Les compteurs sont identiques entre H et C dans chaque phase : MAIN 58 acceptés / 69 proposés, HOLD 56/70 ; les deux lignes de vérification/draft sont systématiquement utilisées et la prélecture reste nulle. Les octets de demande urgente SSD sont également identiques par phase entre bras. Ils mesurent des lectures logiques, pas le trafic physique SSD.

Variations warmup→mesure :

- MAIN-0-H : +3.314% ; décompressions propres supplémentaires 30 747.
- MAIN-1-C : -4.876% ; décompressions propres supplémentaires 30 636.
- HOLD-0-H : +5.723% ; décompressions propres supplémentaires 31 500.
- HOLD-1-C : +0.808% ; décompressions propres supplémentaires 30 579.

Secteur AC avant/après les six phases, aucun changement détecté en direct. Zéro delta global swap-in/swap-out pour toutes ces phases ; sampler joint, 123 observations sans erreur. Pic physique de durée de vie parmi les douze snapshots ready/warmup/mesure : **7.825 Go**, ledger swap-in propre 0. Les caps compression 77B restent non qualifiés ; ce ledger ne prouve pas une absence de swap-out privé. Le résultat présent n’efface pas les 255,75 Mio de swap-out global observés dans le bridge précédent.

Les 83 sources et le CLI 19c9e1f2 / driver 8935 ont été rehashés : aucune divergence avec le gel v3. Les deltas de temps des phases sont inclusifs côté hôte, pas des temps GPU exclusifs. Les variations warmup/mesure et la compression sont consignées sans attribution causale. Tous les bruts, chiffres exacts, SHA et limites figurent dans `cross-review-screen.json`.
