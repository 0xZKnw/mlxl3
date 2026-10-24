# 82F01 — confirmation C-H et clôture indépendante

Les huit réponses C-H correspondent exactement au texte et au hash gold60, avec 128 tokens / 127 decode, DFlash2 fixe B2, compteurs et demandes logiques SSD H=C par phase. Les 16 réponses des deux ordres sont donc vérifiées.

| Prompt | C-H H tok/s | C-H C tok/s | C-H | H-C | Géomoyenne HCCH |
|---|---:|---:|---:|---:|---:|
| MAIN | 11.08425 | 11.35597 | +2.451% | -4.536% | -1.104% |
| HOLD | 10.41794 | 10.04134 | -3.615% | +1.493% | -1.094% |

Le signe change sur chaque prompt lorsque l’ordre est inversé ; aucun gain reproductible. Les deux géomoyennes restent négatives. Le retrait du prototype est cohérent ; aucun débit OFF/MTP supplémentaire n’est justifié. Ce constat n’établit pas une cause matérielle ou une régression universelle.

Warmup→mesure :

- MAIN-0-C : +0.114%.
- MAIN-1-H : +1.817%.
- HOLD-0-C : -4.762%.
- HOLD-1-H : +1.597%.

Pré20, quatre processus et post30 : exits 0, joints, secteur AC ; sampler joint, 122 observations sans erreur. Les six phases ont zéro delta swap-in/out global. Douze snapshots mémoire : pic physique 7.845 Go et ledger swap-in propre 0. Les caps compression 77B restent non qualifiés ; l’ancien swap-out bridge 255,75 Mio est conservé.

Sources/binaire/driver et tous les détails sont dans `cross-review-confirm.json`. Aucun test, build, proveur ou GPU lancé par cette revue.
