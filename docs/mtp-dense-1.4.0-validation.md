# MTP automatique et kernels Qwen — 1.4.0

## Comportement livré

La sélection repose sur la configuration réelle du modèle, pas son nom de
dossier. Qwen3.8-27B et Qwen3.6-35B-A3B disposent chacun d'une tête MTP épinglée
et vérifiée en taille/SHA-256. Le switch MTP global conserve ON/OFF au changement
de modèle ; l'app annule les téléchargements obsolètes, arrête l'ancien moteur
et prépare la tête correspondante avant d'autoriser une génération.

Le bridge partage le chargeur entre configuration, génération et tuner. OFF,
tête étrangère et chemin absent libèrent la tête et les caches ; un second ON
réutilise les poids. Une barrière MLX à cette frontière rend les compteurs et
la libération des temporaires déterministes. Elle n'est pas dans la boucle de
génération. Les réglages Tune par modèle restent disponibles, mais leur
restauration ne peut écraser un ON/OFF explicitement choisi.

Versions app/moteur 1.4.0, Desktop build 23 ; bridge protocole 1 avec capacités
additives. Le moteur exige Desktop 1.4.0 pour le téléchargement selon la cible.
Apple Silicon/macOS ≥26.2. Le checkout principal et ses prototypes MTP locaux
sont préservés ; le travail est sur la [PR24](https://github.com/0xZKnw/mlxl3/pull/24).

## Sources, protocoles et preuves

Préenregistrements : `MTP-2026-10-05-DENSE-AUTO`, `MTP-2026-10-05-NORM-FUSION`,
`QWEN38-2026-10-05-NORM-FUSION` dans [opti.md](../opti.md). Tous les résultats,
y compris les échecs précédents, sont dans
[measurements/mtp-dense-1.4.0](measurements/mtp-dense-1.4.0/).

MacBook Air M5/24 Gio, macOS27.2, MLX0.32.2, MLX-LM0.32.0,
Rust1.98.1/Kani0.68.0/CBMC6.11.0, Python3.12.14, Swift6.4/SDK26.5.
Les empreintes des binaires, dépendances, fichiers checkpoint et 1 548 fichiers
de référence sont dans `campaign-metadata.json` et les patches associés.
Les références brutes cible (1,8 Gio) restent dans
`build/qwen38-norm-reference` du checkout principal.

La fusion addition/RMSNorm conserve les arrondis FP16, l'ordre de réduction
MLX et la normalisation de chaque ligne. Elle réutilise le wrapper Metal et
garde un repli stock pour dtype, dimensions et epsilon non éligibles.
Source étudiée : [MTPLX fused_norm.py à 9882703](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kernels/fused_norm.py).
Licence/attribution conservées dans `LICENSES/MTPLX-NOTICE.txt`, le manifeste
moteur et les liens de l'app/CLI. Aucun gain annoncé en amont n'est repris.

## Correction et mémoire physique

- Normes : 72 cas, **2 256 384 mots FP16 présents, finis et égaux en bits**,
  plus replis FP32 et formes non supportées (`fusion-quality.log`).
- Tête dense : référence indépendante MLX-LM sur 47 positions puis trois
  étapes récursives, sorties et K/V exacts (`reference-dense*.json/log`).
  Tête MoE : 47 positions et K/V exacts (`reference-moe*.json/log`).
- Cible dense : préfixes23/128/256 puis trois tokens imposés, **12 points**
  de248320logits finis et128états chacun, tous octets exacts à la baseline
  (`target-fusion-finite-states.log`).
- Sessions : **24 blocs D1..3 par modèle**, tokens greedy, états et caches
  exacts. Réparation K/V sans projection/MLP égale à la tête complète, y
  compris la prédiction suivante ; lignes1/2/3/17/23/24/255 et erreurs.
- Rollback dense : préfixes1/23/24/129/256, vérification2..4, tous préfixes
  retenus, limites du contexte et erreurs (`dense-rollback.log`).
- Parcours réel CLI/bridge : **dense→MoE→dense, 36 cas** D0..3 et budgets
  1/3/17, token hashes et historiques non vides exacts ; ON/reuse, têtes
  étrangères, chemins absents, OFF avant/après génération et enfants rejoints
  (`model-switch-after-generation.json`).

| Modèle | Baseline/OFF MLX actif (octets) | ON | Tête ajoutée |
|---|---:|---:|---:|
| 27B EXL3 2,0bpw | 9 172 153 212 | 9 411 084 156 | 238 930 944 |
| 35B-A3B EXL3 2,49bpw | 12 078 473 664 | 12 553 599 552 | 475 125 888 |

ON répété garde exactement l'allocation ; OFF revient exactement à la baseline,
y compris après génération. Il s'agit de mémoire MLX active, distincte de la
RAM du processus. Les rapports conservent aussi son empreinte. Aucun débit
n'est déduit des durées de tests.

## Mesures de vitesse

Microbenchmark terminé, **non concluant** : 8 formes2048/5120×1..4, 10 warmups
et60paires alternées, wall ms/op host+GPU ; variations de temps −2,08 % à
+3,33 %. Pas de mesure GPU seule ni de gain établi
(`norm-microbench.log`, `norm-microbench-summary.json`).

Quatre campagnes bridge ABBA D0/D1/D2/D3 terminées, deux prompts français/code,
warmup8tokens, mesure128tokens/prompt/passe, contexte4096/cacheOFF, repos15s,
timeout600s. A sans fusion, B avec les deux fusions ; hashes/binaires/prompts
dans `speed-preflight.json`. Batterie76% en décharge au départ, température
non disponible. Aucun build/prover/autre modèle GPU pendant les mesures.
Batterie76→65% en décharge sur la campagne complète. **32 générations/4096tokens**
mesurés et32warmups/256tokens, toutes sorties exactes entre binaires et profondeurs.

| Mode | Sans fusion français/code (tok/s) | Fusion français/code (tok/s) | Variation decode français/code |
|---|---:|---:|---:|
| D0 | 8,653 / 8,502 | 8,721 / 8,655 | +0,78% / +1,80% |
| D1 | 8,769 / 8,337 | 8,506 / 8,084 | −3,00% / −3,03% |
| D2 | 7,036 / 6,912 | 6,979 / 6,833 | −0,81% / −1,15% |
| D3 | 6,770 / 6,866 | 6,838 / 6,752 | +1,00% / −1,66% |

Médianes de deux passages par binaire/prompt, préfill exclu du decode. Acceptation
D1~79–84%, D2~67–69%, D3~61–62% ; l'acceptation n'implique pas une accélération.
La dérive du contrôle A atteint −4,98..+7,70%, sans température/fréquences GPU.
Les profondeurs ont été mesurées dans des campagnes distinctes ; leur comparaison
est descriptive. Aucune accélération générale ni gain MTPLX établi sur ce M5.

**Décision : fusions OFF par défaut**, disponibles seulement par activation
explicite `MLXL3_MTP_FUSED_NORM=1` et `MLXL3_QWEN_FUSED_NORM=1`. Les replis stock
restent le chemin normal. `speed-summary.json` vérifie aussi les hashes inter-mode ;
les quatre `norm-abba-dN/results.json` conservent les mesures/options/binaires.
Aucun benchmark en cours. Tune MTP reste le contrôle prévu pour un autre workload.

## Vérification source, tests et limites

- `cargo fmt --check`, Clippy strict MLX/chat et CPU/chat : réussis.
  `cargo test --release --locked --features mlx,chat` : **60 passés/54 ignorés** ;
  CPU/chat : **50 passés/3 ignorés**. Les tests physiques ci-dessus sont
  sélectionnés séparément, un par processus ; les autres ignorés ne sont pas
  attribués à une exécution GPU.
- Suite Python finale : **192 passés/4 skips** (ponyexl3 absent, modèle Ling absent),
  après190/4 avant les deux contre-exemples de démarrage lent.
  Ruff/format/py_compile réussis. Contrôleur de switch : **76 cas** de parité,
  budgets, sorties vides/incomplètes, JSON invalide, enfant silencieux,
  annulation et nettoyage. Comparateur : régressions MTP/warmup et faux rapports.
  Un démarrage artificiel600ms reproduit l'échec CI du délai400ms ; délai2s
  et watchdog15s restent bornés. Chaque scénario exige son erreur spécifique,
  donc un timeout ne peut masquer une régression mémoire ou parité.
- `scripts/check-desktop.sh` : succès complet, lifecycle, transport CLI,
  bridge, téléchargements, rendu, MCP, annulation, tuner et updater. Contre-
  exemples ON/OFF avec anciens profils Tune et OFF livré sur erreur conservés.
- `cargo kani --lib --no-default-features` : **33/33 harnais, 4 826 obligations,
  0 échec, 77/77 covers** ; 70 checks inatteignables existants inspectés.
  Nouvelle classification :181obligations/3covers ; garde de lancement :
  109obligations/3covers. Scalari32/usize/f32/bool/Option symboliques complets,
  aucune hypothèse supplémentaire ni boucle dans ces deux nouveaux harnais.
  Bornes/unwinding des harnais existants conservés ; synthèse `kani-summary.json`.
- CrossHair0.0.101 : trois postconditions,30s/condition et5s/chemin,
  **Not confirmed**, aucun contre-exemple ; ce n'est pas une preuve.
- CBMC sur la vraie source Metal : extension non supportée. Vrai bridge C++
  préprocessé par ClangC++20 avec headers MLX et SDK : parsing libc++ Apple
  refusé par CBMC6.11.0 ; ESBMC absent. GPU/FFI/barrière non vérifiés
  formellement, aucun stub ajouté (`synchronize-cbmc.json`).

Les tests échantillonnés et le model checking CPU borné ne prouvent pas
MLX/Metal, allocateur, concurrence ou génération complète. Swift est contrôlé
par compilation Swift6 et les parcours E2E, sans preuve formelle de concurrence.
CI du SHA final de livraison, paquets signés/updater réel et publication restent
à compléter.

## Échecs conservés

Compilation d'un harnais modifié simultanément, import absent dans une fixture,
anciens profils Tune écrasant ON/OFF, écart de512octets de temporaires GPU au
second ON (corrigé par synchronisation), garde inutilisée Clippy CPU (cfg
corrigé), filtre cache MoE sélectionnant zéro test puis relancé correctement,
tentatives CrossHair/CBMC incompatibles. Les rapports partiels gardent un état
failed explicite ; aucune tolérance numérique ou mémoire n'a été élargie.

Première CI fc41314 : Desktop réussi, native Linux/macOS échoués sur le cfg,
Kani annulé sans runner acquis ; correction8629157 puis pin Ubuntu22.04 poussés
séparément. **4/4 jobs PR réussis sur6e79de6** :
[Native/Kani](https://github.com/0xZKnw/mlxl3/actions/runs/37375917937),
[Desktop/E2E](https://github.com/0xZKnw/mlxl3/actions/runs/37375917879).
Logs/status inspectés dans `ci-pinned-*`, mêmes4826obligations et77covers.
Aucune app personnelle remplacée.
