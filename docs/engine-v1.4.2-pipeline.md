# Moteur v1.4.2 — recherche supplémentaire du 7 octobre 2026

Suite de la [PR draft #26](https://github.com/0xZKnw/mlxl3/pull/26), après
`79ec53586697e5f00092355415a8e10457da9a78`. Aucun merge, release ou remplacement
de l'app. Les modifications du checkout personnel sont préservées. Les résultats
de [phase2](engine-v1.4.2-research.md) restent historiques et ne sont pas effacés.

## Soumission progressive des couches cible

La source primaire [TensorFold cb2ebf054, row_forward.py](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/kernels/qwen/dense/v1/row_forward.py)
soumet la première couche puis des groupes de quatre pour permettre au GPU de
travailler pendant que le CPU construit la suite du graphe. Notre implémentation
indépendante réutilise MLX `async_eval` et les wrappers CABI existants ; aucune
opération arithmétique, précision ou vérification cible ne change. Ce mécanisme
est distinct du prototype « async par niveau draft » retiré en phase2.

Le défaut final concerne les deux boucles MTP de Qwen sur **M5**, H2048,
40 couches, 256 experts/top8. Les autres chemins gardent le défaut précédent.
`MLXL3_QWEN_PIPELINE=0` désactive la soumission ; `1` l'étend expérimentalement
aux six boucles Qwen, y compris capture/vérification DFlash. La clé Tune contient
la politique `off`, `all` ou `mtp-m5-v1` pour invalider les anciens résultats.
Le draft stock reste sélectionné par défaut ; aucun poids n'est fourni dans la PR.

Les mesures finales comparent **le même binaire**
`4e3af9f76516f253a9adc0d8de0e6ff4020983c864eceeb5d33a2653de8c56d0`, A override0
contre B sans override, lookup0 dans les deux. Qwen3.6-35B-A3B EXL3 2,49 bpw,
tête MTP4bit, MLX0.32.2, M5/24 GiB/macOS27.2, contexte4096/cacheOFF,
warmup128 puis128 tokens mesurés, deux répétitions et settle2s. Une seule
inférence GPU à la fois, aucune compilation/preuve pendant les mesures. Les
probes indiquent secteur100%, aucun warning publié ; température et fréquence
GPU non mesurées. Swap et mémoire processus sont conservés par passage.

| Mesure finale | Decode FR | Decode code | Temps complet code | Dérive maximale decode |
| --- | ---: | ---: | ---: | ---: |
| ABBA | +7,98% | +7,80% | −6,80% | 0,79% |
| BAAB de confirmation | +7,11% | +7,16% | −6,27% | 2,58% |
| Contrôle D0 ABBA, chemin inchangé | −0,49% | +0,29% | −0,33% | 0,55% |
| Compact MTP2 ABBA | +9,62% | +7,53% | −6,32% | 1,32% |
| Compact MTP2 BAAB | +9,40% | +9,80% | −8,22% | 2,26% |

Les sorties/hashes/budgets sont exacts dans toutes ces passes. Le gain stock D2
est confirmé sur ces deux prompts, également avec la tête compacte. D0 est un contrôle sans gain établi. Les gains
ne s'additionnent ni au compact draft contre1.4.1, ni au lookup expérimental.

Avec1968 tokens de contexte, le final stock ABBA donne63,34tok/s, +6,94%
decode et −3,16% complet, contrôle≤1,15%. Sa confirmation BAAB a un contrôle
FR A dérivant de−9,23% : **gain long non confirmé**. Les +16,24% affichés par
cette fenêtre ne sont pas retenus comme gain. Les deux fenêtres restent
archivées, sans moyenne entre protocoles ni extrapolation à tout contexte.

Le prototype global précédent, binaire `c3071d06…`, avait lui aussi un signal
D2 confirmé mais son contrôle D0 dérivait fortement (−13,20%/ +19,51%). Son
long avait un contrôle FR dérivant de5,14%. Ces fenêtres restent **non concluantes**
et ne valident pas la politique finale. D3 prototype n'a pas de confirmation
finale ; aucune vitesse D3 ou dense n'est revendiquée. Sources/options de chaque
variante sont sauvegardées séparément.

## Autre vocabulaire draft : MTPLX64k

La liste primaire [MTPLX 9882703, qwen38_code_ranked_64k.npy](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/data/qwen38_code_ranked_64k.npy)
contient réellement65536 IDs int32 uniques/triés dans0..248319 ; SHA256
`922a4d0570ce0a79e03c1e1ecb25e6c8c1e9cae943b2c0d5ce11874d42d74a17`.
Elle diffère de la liste TensorFold79591 et n'est pas une troncature arbitraire.
Les deux tokenizers locaux partagent le vocabulaire de base248044 ; sept tokens
audio ajoutés diffèrent et aucun n'est sélectionné. Les tokenizers ne sont pas
présentés comme identiques.

Le convertisseur CPU existant produit une projection+map de **72,25 MiB**, soit
**15,52 MiB de moins** que le compact79616. C'est une différence de tenseurs,
pas une mesure de RAM processus. Les poids et la liste restent locaux/non fournis.
Qualité : trois fois65536 logits finis exacts contre les lignes du Q4 complet,
24 blocs récursifs D1..3/80 états/hidden/KV exacts,20 rejets du loader et bridge
39 requêtes/33 completions/4 annulations/2 erreurs attendues. Le Tune ABBA
sur baseline79ec, pipeline0/lookup0,32warmup/deux96tokens donne D2+0,78%,
D3+0,71%, D1−1,32%, contrôlesD0≤1,26% : **aucun gain vitesse établi**.
Le seuil3% n’est pas franchi ; aucun BAAB ou long supplémentaire lancé.
Cette variante mémoire demeure un artefact local facultatif, sans changement
de défaut ni recommandation de vitesse.

## Dernière fusion MoE : exacte, sans gain retenu

À la demande d'une dernière optimisation visant environ10%, une variante
indépendante a gardé le Hadamard MLX et fusionné uniquement le gather des scales,
les deux produits FP16 et la réduction top8. Elle différait de PERF68 et
PERF116/D36, dont les résultats négatifs/interrompus sont conservés dans `opti.md`.
L'ordre FP16 suit les sources primaires [MLX v0.32.2 reduce.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/reduce.cpp),
[reduce_col.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/reduction/reduce_col.h)
et [ops.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/reduction/ops.h).

Sur rows1..4/H2048/256experts/top8,20480 valeurs finies sont bitexactes contre
MLX et un oracle CPU half::f16 ; routes dupliquées/permutées, zéros signés,
subnormales, produits et scores signés sont exercés. Huit metadata invalides
sont refusées et deux routes hors borne retournent NaN sans lecture hors borne.
Une mutation de l'index de score est détectée puis restaurée. Le modèle réel
passe51 positions/logits/hidden/80 états exacts. Kani CPU126 checks et5 covers
satisfaits vérifie les domaines i32/u32 complets des lancements et indices sans
assume ; cette vérification ne prouve pas le kernel. La vraie API MLX provoque
l'ICE Kani antérieur, Metal CLI manque et CBMC ne reconnaît pas le langageMetal.

Le microbenchmark graph+sync ABBA puisBAAB pour quatre nombres de lignes,
8 warmups et40 échantillons de240 invocations, reste autour de260µs :
**−0,27% à +0,19%**, sous le seuil3%. Le plancher de synchronisation peut masquer
un effet GPU ; ce n'est ni un timerGPU isolé, ni un gain en génération.
L'alimentation et la thermique pendant ce micro n'ont pas été relevées ; la
probe secteur ultérieure est conservée sans rétroattribuer ses conditions.
**Prototype retiré intégralement** ; aucun timing génération supplémentaire
ni boost10% revendiqué. Sources/tests/mutation/échec sont archivés, et les trois
sources restaurées correspondent bitàbit au candidat pipeline précédemment
mesuré. Décision : `last-reduce-decision.json`.

## Vérification et limites

- Contrôles physiques finaux sélectionnés explicitement, seuls : 11 exécutions
  réussies, puis deux répétitions du test renforcé hidden stock/dense. Trois
  prefills23/24/257 et16 décodes comparent51 positions avec248320 logits,
  formes/count/finitude du hidden et80 états MoE ou128 états dense bitexactement.
  Récursion stock/compact/64k D1..3, rollback T1..8 pour les prefills
  1/23/24/129/256 et tous les préfixes engagés, erreurs0/9/OOV/reset couvertes.
  DFlash force1 exerce capturelong2825, largeurs1..8 et commit sélectif.
- Le vrai bridge final/Tune passe43 requêtes,35 completions,5 annulations et
  2 erreurs attendues. Chaque mode Tune decode190 tokens avec les mêmes deux
  hashes ; clé réelle `pipeline=mtp-m5-v1:lookup=0` conforme au ready.
  Changement dense→MoE→dense :36 completions exactes, reuse/OFF/erreurs/cleanup.
  Smoke DFlash64/128 puis préfixeOFF/ON et annulation : réussis sous force1.
- L'API Array physique vérifie multi-sorties, rétention après libération des
  parents, vide et refus des tableaux/éléments NULL. Mutation de la garde NULL
  détectée puis restaurée ; omission de la garde M5 également détectée par test CPU.
- Checks locaux finaux : fmt, Clippy strict CPU et `mlx,chat`, compilation C++20
  stricte et build release `mlx,chat` réussis ; suites Rust68 passés/60 ignorés
  avec MLX et56 passés/2 ignorés CPU. Les ignorés GPU sont sélectionnés
  séparément comme décrit ci-dessus ; tous les ignorés ne sont pas exécutés.
- Suite Kani CPU complète : `cargo kani --lib --no-default-features -Z unstable-options
  --harness-timeout 600s --export-json kani-phase-3.json` :39/39 harnais réussis,
  5463 checks passés/98 covers satisfaits,0 échec/timeout,484,18s commande
  (471,92s vérification). Les72 obligations inatteignables correspondent toutes
  à celles inspectées de phase2 ; aucune nouvelle assertion inaccessible.
- Kani ciblé vérifie le vrai contrat `pipeline_layer` sur usize/bool complets,
  sans assume :13 assertions/4 covers,0 inatteignable. Le vrai helper d'éligibilité
  utilise i32/usize/bool complets sans assume :1 assertion/4 covers satisfaits,0 inatteignable. Les domaines
  du matcher lookup restent20 IDs symboliques/unwind33 ; map compacte0..8/unwind10.
- Kani de la vraie API Array MLX : **non vérifié**, ICE `intrinsics.rs:243` avant
  obligations. CBMC du vrai pont préprocessé C++20 : **non vérifié**, parseur
  C++11 refusant libc++/SDK, exit6 ; ESBMC absent. Pas de stub GPU. Une preuve
  scalaire CPU ne prouve ni MLX/Metal, ni allocations/concurrence/génération entière.
- La CI PR/push ajoute Clippy/build de la variante livrée `mlx,chat` sur
  macos-26/MLX0.32.2 pinné, sans modèle ni credential. Actionlint1.7.12 officiel
  vérifié par checksums passe les deux workflows. Le [runner standard macos-26](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
  est arm64 ; ses compilations ne remplacent pas les contrôles GPU physiques.
  La CI du nouveau SHA sera inspectée après push et liée dans la description de PR.

Les commandes exactes et sorties sont conservées dans [l’archive phase3](measurements/engine-v1.4.2/phase-3-raw.tar.gz)
et son [manifest SHA256](measurements/engine-v1.4.2/phase-3-manifest.json). Extraire
dans le répertoire de mesures : `pipeline-*`, `mtplx-*`, `last-reduce-*`,
`phase-3-*` et `kani-phase-3*`. Sources locales finales/prototypes indépendants,
empreintes binaires/modèles et échecs sont inclus ; aucun poids, binaire,
code amont ou liste de vocabulaire n’est fourni. Les archives des phases1 et2 restent immuables. Le journal `opti.md`
préenregistre chaque essai et distingue prototype, variante livrée et publication.
