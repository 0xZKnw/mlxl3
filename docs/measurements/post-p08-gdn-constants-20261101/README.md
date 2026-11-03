# Coefficients de convolution GDN préparés une fois

1 novembre 2026 — POSTP08-GDN-CONST-01. **Candidat conservé pour composition**,
option locale `MLXL3_FLASH_PREPARED_CONV=1`, défaut 0. Qualité composant v1
validée ; gain modèle non mesuré. Aucun changement de l'app installée.

## Changement et portée

Le moteur Rust prépare au chargement le coefficient immuable F32 dans la forme
attendue par `Gdn::convolve`, en conservant d'abord l'arrondi F16 du chemin
stock. Mono-token et vérification multi-token utilisent ce même helper.
Ni l'ordre arithmétique, ni les états, ni le cache experts ne sont changés.
L'ancien poids F16 est gardé pour le comparatif ; payload F32 supplémentaire :
5,625 Mio pour 36 couches. Ce chiffre n'est pas la RAM processus mesurée.

[Audit d'adaptation v1](adaptation-audit.md) : pas de défaut numérique trouvé.
Un wrapper C++ supplémentaire était cloné à chaque appel ON ; R2 remplace ce
clone par un emprunt. Ce n'était pas une copie GPU et son coût n'a pas été
isolé. La capture Xcode proposée reste différée, facultative pour conserver
ce candidat ; aucun dispatch supprimé n'est revendiqué.

## Résultats v1 conservés

[96 échantillons bruts](micro.json), [résumé](micro-summary.json).
Même GDN complet, B1/2/4/8, 12 rounds OFF/ON alternés, 16 forwards par bras,
24 chauffes, aucun échantillon exclu. Construction + soumission + completion
mesurées ; ce ne sont pas des tok/s du modèle ou des durées GPU exclusives.

| Lignes | OFF µs/forward | ON µs/forward | Ratio de débits agrégés |
| --- | ---: | ---: | ---: |
| 1 | 762,376 | 734,113 | +3,85 % |
| 2 | 801,189 | 799,403 | +0,22 % |
| 4 | 1310,059 | 1307,509 | +0,20 % |
| 8 | 2323,042 | 2310,500 | +0,54 % |

B1 fortement dispersé : ratios appariés −21,19 à +71,43 %, médiane +0,94 %.
Le +3,85 % n'est pas un gain stable établi. Les petits signaux B2/4/8 justifient
la conservation, sans addition entre formes ni extrapolation aux 36 couches.

## Vérification et décision

V1 : 36 poids / 1 474 560 coefficients face à un oracle CPU indépendant ;
5 positions et stages externes ; cohortes 1..8, sorties, histoires, états,
préfixes, continuation, reset et erreurs contrôlés ; mutation détectée puis
restaurée. Qualité physique 1/1, flag invalide 1/1. CPU 174 passés / 8 ignorés ;
MLX 202 passés / 187 ignorés. Format, Clippy strict et build réel `mlx,chat`
passent. Les tests ignorés ne sont pas couverts par ces suites ordinaires.

Kani réel `prepared_conv_preserves_symbolic_half_endpoints_through_real_mlx`
tenté : un u16 symbolique répété sur 40 960 coefficients, valeurs F32 aux deux
extrémités, forme/dtype, unwind 3. Échec interne du compilateur avant obligations,
[commande et résultat](kani-real-mlx-result.json) : **non vérifié**, aucune preuve
Metal ni remplacement par un modèle CPU présenté comme preuve.

La [nouvelle procédure](../../../opti.md) permet de conserver ce petit candidat
et de qualifier un ensemble compatible. Qualité et gain global de la composition
restent à tester avant activation. F02 est disponible comme autre candidat,
avec ses résultats mixtes/incomplets conservés séparément. Aucune somme de gains.
Les gardes de ressources sont inchangées, dont 32 Mio de nouveaux Swapout au
total depuis l'ancre originale F02 ; aucun réancrage.

## R2 : qualification terminée, candidat conservé

Seul delta exécutable : emprunter le coefficient préparé au lieu de cloner son
wrapper. Binaire SHA256
`605f18cd29014964ba4781d64a88fb1af29432ee09429d59b7e3667dcd1e6005`,
[identité et 644 empreintes](identity-v2.json). Format, Clippy strict et build
release `mlx,chat` passent. Les limites de la tentative Kani v1 sur la préparation
inchangée restent applicables ; l'emprunt FFI/Metal n'est pas formellement prouvé.

Qualité ciblée 1/1, mêmes oracles et domaines que v1 ; micro 1/1 ; suite MLX
202 passés / 187 ignorés. Trois processus joints, zéro nouveau Swapout pendant
ces tests, cumul original 14,6875 Mio sur 32 Mio. Tests CPU sans MLX, parseur du
flag et fixtures inchangés : leurs preuves v1 sont réutilisées, pas réexécutées.

| Lignes | OFF µs/forward | ON µs/forward | Ratio de débits agrégés | Médiane des ratios appariés |
| --- | ---: | ---: | ---: | ---: |
| 1 | 744,018 | 673,144 | +10,53 % | +1,11 % |
| 2 | 802,313 | 796,553 | +0,72 % | +0,70 % |
| 4 | 1312,991 | 1305,971 | +0,54 % | +0,51 % |
| 8 | 2334,288 | 2313,351 | +0,91 % | +0,89 % |

[Brut R2](micro-v2.json), [résumé R2](micro-summary-v2.json). Les 96 échantillons
et l'ordre sont vérifiés indépendamment ; aucune exclusion. B1 reste dispersé
(−2,37 à +80,86 % par paire ; latence OFF moyenne +33,89 % entre moitiés) ;
ni +10,53 % stable ni gain modèle annoncé. Les deux processus v1/v2 ne permettent
pas d'attribuer leur différence au retrait du clone.

**Décision : conserver R2 pour composition**, option locale disponible, défaut
OFF. Aucune répétition pour lisser les chiffres. Capture Xcode non exécutée ;
modèle/bridge et composition non testés avec ce candidat. Disponibilité avant
qualité légère 8,75 Go, sous la marge modèle de 10,5 Go ; aucun gros chargement
tenté. App installée et CI distante du code local non publié inchangées.

Commandes R2 (racine du dépôt, `CARGO_BUILD_JOBS=1`,
`MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx`) :

```sh
cargo fmt --check
cargo clippy --locked --release --features mlx,chat --all-targets -- -D warnings
cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json
```

Exécutions physiques via `../post-p08-dense-epilogue-20261031/bounded-gpu-runner.py`
avec l'ancre originale et `identity-v2.json` :
[qualité](gdnconst-quality-v2-plan.json), [micro](gdnconst-micro-v2-plan.json),
[suite MLX](ordinary-mlx-v2-plan.json). Chaque plan contient la commande, les
options, le modèle, les fixtures et les gardes ; reçus `*-plan-result.json`
associés inspectés. Aucune CI ne remplace ces oracles physiques.
