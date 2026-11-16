# Prefill Qwen3.6-35B-A3B : recherche du 16 novembre 2026

Recherche documentaire et statique, avec un modèle CPU. **Aucun prototype, build, preuve, benchmark GPU ni chargement de modèle.** Aucun gain n'est mesuré ici. Les estimations sont des modèles, à confirmer ou réfuter par les contrôles proposés. Entrée de journal : `RESEARCH-2026-11-16-Q36-PREFILL` dans [opti.md](../opti.md).

Source analysée : `origin/main` `15218328d422a9e438492d61b939d0568e1e6a36`, Desktop 2.0.3, contenant le candidat `3de208e0` avec plan de routes parallèle et dernière ligne segmentée. Le checkout personnel (main `1ac910d0`, modifications locales) n'a pas été modifié, hormis les fichiers de cette recherche. Modèle : `models/Qwen3.6-35B-A3B-EXL3-2.49bpw`. Architecture lue dans `config.json` : 40 couches dont 10 d'attention complète (`full_attention_interval` 4) et 30 GDN, hidden 2048, 256 experts top-8, `moe_intermediate_size` 512, expert partagé 512, attention 16 têtes Q / 2 têtes KV en `head_dim` 256, GDN 16 têtes K / 32 têtes V en 128×128.

## Résumé

1. **Piste principale, nouvelle sous cette forme : des lots MoE plus grands, sans toucher aux formes des mélangeurs de séquence.** À 256 tokens, chaque expert voit environ 8 lignes. Le kernel segmenté calcule pourtant des tuiles de 32 lignes et décode le trellis une fois par sous-bloc de 32 lignes. Environ 75 % des lignes MMA sont donc du padding. Regrouper quatre chunks de 256 pour le seul MoE (routeur, experts routés, expert partagé, normes), tout en appelant l'attention et le GDN sur les mêmes tranches de 256 qu'aujourd'hui, réduirait le travail expert par token à **≈0,36–0,41×** selon le modèle CPU. Par Amdahl, avec les experts à ≈60 % du cumul shader, cela donnerait un ordre de grandeur de **+40 à +60 % de prefill** sur les parties de 1024 tokens. C'est une estimation, pas une mesure.
2. **Cause probable du rejet « bloc 1024 » de l'issue 44 (P12), à vérifier.** Dans le SDPA de MLX 0.32.2 copié dans `docs/measurements/issue44-20261017/mlx0322-sdpa.cpp.txt`, pour `head_dim` 256 et plus de 8 queries, `use_fallback` choisit le chemin **non fusionné** (matmul, softmax, matmul). À partir de **1024 queries** causales sur NAX, il passe au kernel fusionné `sdpa_full_self_attention_nax`. Un chunk complet de 1024 change donc l'arithmétique de l'attention. Ce changement suffit à expliquer des logits différents. La piste 1 évite ce changement en gardant l'attention sur 256 queries.
3. **Indice d'invariance déjà acquis.** `Q36-FINAL-ROW` a montré que la dernière ligne passée seule dans `forward_segmented` (`rows = 1`) donne des logits F16 bit-identiques à la même ligne au sein du lot de 73. Le résultat MMA du kernel segmenté ne dépend donc pas, sur ce cas, des autres lignes ni de la position dans la tuile. Cela ne suffit pas comme preuve générale. C'est pourquoi la piste commence par un oracle bit à bit.
4. **Pistes secondaires.** Une fois les segments de plus de 32 lignes devenus courants, partager le décodage entre les deux sous-blocs de 32 d'un segment. C'est une continuation de P03 côté MoE, qui avait donné −0,345 % à T256, faute de segments assez longs. Pour les projections denses (≈20 %), décodées `rows/64` fois par chunk, il n'y a pas de nouvelle raison suffisante après la régression de QMM3 dense (−16 %), sauf preuve amont d'un mécanisme coopératif différent.
5. **Hors contrat exact.** GDN chunkwise/WY, attention fusionnée NAX à 256 queries, autre quantification. Ces voies changent les arrondis. Elles ne sont listées que comme contrat numérique distinct.

## Point de départ

| Élément | Valeur et source |
| --- | --- |
| Prefill long 1097 tokens (FR) | 497,2 → 502,8 tok/s ABBA, 489,5 → 494,8 BAAB après `Q36-FINAL-ROW` (binaire 8caeae84, M5 24 Gio, MLX 0.32.2, secteur) |
| Attribution | Profil M1, prefill 1097 MTP1 : experts segmentés ≈60 %, denses ≈20 %, plan de routes ≈4,3 % du **cumul shader échantillonné**, pas de la latence |
| Découpage | `Qwen => 256` dans `prefill_chunk_size` (`native/src/main.rs`), `dflash::PREFILL_CHUNK = 256`, `prefill_mtp` `CHUNK = 256` ; checkpoints du cache de préfixe aux multiples de 256 |
| Kernel expert | `native/shaders/_expert_qmm_tensor_kernel.metal` : BM32/BN32/BK16, `matmul2d` en `execution_simdgroup`, un simdgroup par threadgroup, grille `[(N/32)·32, max_blocks, 2]`. Segments de 64 lignes coupés en deux sous-blocs de 32, dont le second retourne s'il est vide. Chaque sous-bloc actif décode toute la bande de poids de son expert |
| Sélection | `SwitchGlu::forward` : segmenté si `rows ≥ 64`, experts ≤ 256, K ≠ 7, TensorOps EXL3 disponibles ; sinon QMV mappé (queues courtes) |
| Attention | `Array::sdpa(..., causal = time > 1)` sur K/V concaténés ; D256 et 256 queries donnent le chemin non fusionné MLX |
| GDN | récurrence séquentielle `gated_delta_packed.metal` (corps mlx-lm), état FP32 transmis entre chunks ; préparation fusionnée si `time ≤ 8`, sinon `conv1d` MLX |

Historique pertinent, à ne pas refaire tel quel :

- **RUST-PERF-13 (septembre)** : 512 rejeté contre 256 (−6,7 %). C'était **avant** le MoE segmenté (PERF-15), avec des paliers de fréquence instables. L'argument « lignes par expert » n'existait pas encore.
- **tool-prefill-rd (5 septembre)** : chunks 512 et 1024 plus lents et non exacts, sur l'ancien runtime Python et `gather_qmm`. Ce n'est pas comparable.
- **Issue 44 P12 (17 octobre)** : bloc complet 1024 contre 4×256 divergent dès les logits, sur le 35B comme sur le 27B (`p12-model35.log`). La cause n'a pas été localisée et aucune vitesse n'a été mesurée.
- **Issue 44 P03** : partage de fragments QMM, MoE −0,345 % et dense −16,04 % ; P05 segments +0,641 % ; P10 queues courtes divergentes.
- **Q36-POST209** : segments de 32 (C) non confirmé, plan parallèle (P) retenu. **Q36-FINAL-ROW** : dernière ligne segmentée retenue, +1,1 %.
- **Audit 2026-09-23** : 128 contre 256 change les logits. La largeur de chunk fait partie du contrat numérique actuel.

## Piste 1 : grands lots MoE, mélangeurs aux formes actuelles

### Mécanisme

Le coût d'un sous-bloc expert actif dépend peu du nombre de lignes utiles tant que celui-ci ne dépasse pas 32. Il comprend le décodage trellis de `K×32` poids par pas BK pour chaque bande de 32 colonnes, plus une MMA 32×32×16 complète. À 256 tokens, 2048 routes se répartissent sur 256 experts, soit ≈8 lignes par expert. Presque tous les experts sont actifs : chaque poids expert de chaque couche est décodé environ une fois par chunk, pour ≈8 lignes utiles. À 1024 tokens, la moyenne passe à ≈32 lignes. Le travail par chunk n'augmente que pour les experts qui dépassent 32 lignes, alors que le nombre de tokens est multiplié par quatre.

### Modèle CPU (pas une mesure)

`benchmarks/q36_prefill_subblock_model.py` donne le résultat suivant, sauvegardé dans [subblock-model.json](measurements/q36-prefill-research-20261116/subblock-model.json). Le script tire 20 lots par cas, top-8 distincts, popularités uniformes (σ = 0) ou log-normales (σ = 0,5 et 1). Il compte `Σ ceil(n_e/32)` sous-blocs actifs, comme le kernel.

| Tokens par lot MoE | Sous-blocs actifs / token, relatif à 256 (σ 0 / 0,5 / 1) | Lignes MMA utiles |
| --- | --- | --- |
| 73 (queue) | 3,17 / 2,97 / 2,60 | 8–10 % |
| 256 (actuel) | 1 | ≈25 % |
| 512 | 0,50 / 0,53 / 0,59 | 44–50 % |
| 1024 | 0,36 / 0,37 / 0,41 | 64–69 % |
| 2048 | 0,31 / 0,31 / 0,33 | ≈80 % |

Les hypothèses à falsifier sont les suivantes. Le coût est supposé proportionnel aux sous-blocs actifs, en négligeant les threadgroups vides, qui retournent tôt (`launched_subblocks` les compte). Le routage réel du 35B n'a pas été capturé : il est plus concentré que le tirage uniforme, et la corrélation entre tokens voisins n'est pas modélisée. Le gain marginal au-delà de 1024 est faible, alors que la mémoire et la granularité d'annulation se dégradent. **1024 est donc la cible naturelle, et 512 le premier palier.**

Ordre de grandeur, **non mesuré** : si les experts représentent 60 % du temps et passent à 0,4×, le temps par token tombe à environ 0,64 du temps actuel, soit **≈+55 %** de prefill sur un lot de 1024. Avec 512, on obtiendrait environ 0,7, soit ≈+40 %. Le cumul shader échantillonné n'est pas une latence. Les opérations proportionnelles aux lignes (`routed_inputs`, `glu_down_input`, `finish_and_reduce`, denses) restent inchangées par token. La fourchette réaliste est donc plus basse. Seul le contrôle E2 ci-dessous peut la fixer.

### Pourquoi le contrat exact est plausible

Pour une longueur L, le découpage actuel est `⌊L/256⌋` chunks de 256, puis une queue de `L mod 256`. La proposition garde **exactement** ce découpage pour tout ce qui mélange la séquence, et ne regroupe que les opérations ligne par ligne :

- **Attention (10 couches)** : RoPE, concaténation K/V et SDPA appelés sur chaque tranche de 256, avec les mêmes `offset`, `qL` et `kL` qu'aujourd'hui. Le chemin non fusionné reste sélectionné et le basculement NAX ≥1024 est évité.
- **GDN (30 couches)** : préparation `conv1d` et récurrence appelées sur chaque tranche de 256, état conv/FP32 transmis comme aujourd'hui.
- **Lignes indépendantes, sur 1024 lignes** : RMSNorm, projections denses, routeur, experts routés et partagé, résiduels. Les tuiles BM64 du dense restent alignées, puisque 256 est un multiple de 64. Le routeur SIMD admet 1..256 lignes, son repli au-delà est bit-exact selon le contrôle `router-prefill` de `Q36-DECODE-PERF`, ou il peut aussi être appelé par tranches. Les experts segmentés sont invariants par ligne, d'après l'indice `FINAL-ROW`, à confirmer.
- **Queue `L mod 256`** : traitée seule comme aujourd'hui, sur le même chemin (segmenté ou QMV), donc sans risque P10.
- **Cache de préfixe, MTP, DFlash** : mêmes checkpoints aux multiples de 256. Un méga-lot ne doit jamais traverser `checkpoint_offset`. `prefill_mtp` et la capture DFlash consomment déjà tous les hidden et gardent leurs formes par tranche.

Pour 1097 tokens, cela donne 1024 + 73 au lieu de 4×256 + 73.

### Contrôles falsifiables proposés (à préenregistrer séparément)

- **E1 (exactitude, avant toute vitesse).** Sur `origin/main`, avec les vrais poids et les prompts 1097 et 2825 retokenisés, procéder dans cet ordre. (a) Reproduire la divergence P12 du chunk complet 1024. (b) Comparer couche par couche les hidden F16 pour localiser le premier opérateur divergent. L'attente est la couche 3, première attention complète, via la sélection SDPA ; en cas de divergence antérieure dans le GDN `conv1d` ou le MoE, l'hypothèse tombe pour ce composant. (c) Pour le candidat découplé, exiger des logits 248320 et des 80 états bit-identiques à 4×256, la même continuation, les checkpoints de préfixe et les sessions MTP1/2/3. Les largeurs 512, 768 et 1024, avec des queues de 0, 1, 63, 64, 65 et 255, entrent dans l'oracle. Une seule différence de bit suffit à rejeter.
- **E2 (composant).** Appeler `forward_segmented` sur les poids réels d'une couche, avec des hidden réels, à 256, 512, 1024 et 2048 lignes. Mesurer le temps par ligne et vérifier l'égalité bit à bit de chaque ligne contre son calcul dans un lot de 256. Ce contrôle valide ou réfute le modèle des sous-blocs.
- **E3 (bout en bout).** Passer par le bridge, AR puis MTP1, sur les prompts 1097, 2825 et ≈4300 tokens. Comparer en ABBA puis BAAB, puis relever TTFT, prefill, decode témoin, pic MLX et RSS séparément, ainsi que la latence d'annulation. FR27 et code43 servent de témoins inchangés, puisque les prompts de moins de 512 tokens ne passent pas par le nouveau chemin.

Risques : pic mémoire (≈+70 Mo de `routed_inputs` et quelques dizaines de Mo d'intermédiaires à 1024 lignes, à mesurer) ; annulation moins fine (un méga-lot ≈1,5–2 s au débit actuel) ; plus grand graphe MLX par évaluation ; couplage avec `dflash::PREFILL_CHUNK`, les assertions Kani `prefill_capture_start` et la fenêtre draft 2048, qui reste divisible.

## Piste 2 : un seul décodage pour les deux sous-blocs d'un segment

À 1024 lignes, les modèles σ 0 à 1 donnent 373–399 sous-blocs actifs pour 256 experts. Une partie des segments dépasse donc 32 lignes, et chaque sous-bloc redécode la même bande de poids. Plusieurs mécanismes sont possibles : un threadgroup de deux simdgroups qui partage la bande décodée en mémoire threadgroup, ou un `matmul2d` BM64 en `execution_simdgroups<2>`. Dans les deux cas, le décodage par segment serait divisé par deux pour les experts chargés. llama.cpp `mul_mm_id` fait exactement cela, avec un bloc déquantifié partagé par quatre simdgroups (voir les sources). **Il s'agit d'une continuation de P03 MoE**, qui donnait −0,345 % à T256, là où les segments de plus de 32 lignes étaient rares. La condition nouvelle est la piste 1. À ne préenregistrer qu'après E2, et seulement si les segments longs pèsent vraiment, avec la même exigence bit à bit sur l'ordre `operation.run` par pas BK.

## Piste 3 : projections denses (≈20 %)

À 256 lignes en BM64, chaque poids dense est décodé quatre fois par chunk. Le regroupement de la piste 1 ne change rien à ce rapport. Le partage de fragments entre blocs de lignes (P03, QMM3) a régressé de 16 % en dense. Sans source amont montrant un mécanisme coopératif à registres contenus, différent de P03, cette piste reste basse priorité. Elle n'est pas relancée sous un autre nom.

## Hors contrat exact (contrat numérique distinct)

| Méthode | Effet attendu | Pourquoi exclu du chemin exact |
| --- | --- | --- |
| SDPA fusionné NAX dès 256 queries (D256) | L'attention ne matérialise plus les scores `[16, 256, kL]` | Arithmétique différente du chemin non fusionné actuel |
| Chunk complet 1024 | Atteint le NAX D256 natif | Rejeté P12, logits divergents |
| GDN chunkwise/WY (FLA `chunk_gated_delta_rule`) | Parallélisme intra-chunk de la récurrence (≈quelques % du prefill selon M1) | Réassociation des sommes ; antécédent docs/recherche-prefill-decode-2026-10-17 |

## Sources amont vérifiées

Les sources ont été lues via l'API GitHub et le web le 16 novembre 2026, sans build ni exécution. Révisions : mlx `fc344c9` (6 novembre 2026), mlx-lm `56230fa`, llama.cpp `a946eb5`, exllamav3 `8c593e4`, vllm `c418202`, FLA `9c10273`. Les chiffres amont sont ceux des auteurs. Ils ne constituent pas des gains MLXL3.

| Source | Apport | Transposable au contrat exact ? |
| --- | --- | --- |
| [MLX #3925](https://github.com/ml-explore/mlx/issues/3925), [PR #4023](https://github.com/ml-explore/mlx/pull/4023) (fusionnée le 10 août 2026) | Mesure sur M5 de base de `gather_qmm_rhs_nax` à E=256, K=2048, N=512 : à 8 lignes par expert, 10,5 → 15,3 % du pic de 15,4 TFLOPS en choisissant BM selon les lignes par expert ; Qwen3.6-35B-A3B 4 bits, chunks de 512, prefill 523,6 → 557,9 tok/s selon l'auteur | **Confirme le régime de la piste 1** : à 8 lignes par expert, le MMA est très sous-utilisé sur M5. Changer BM est un autre sujet, à tester bit à bit |
| [MLX #4567](https://github.com/ml-explore/mlx/pull/4567), [#4572](https://github.com/ml-explore/mlx/pull/4572) ; [llama.cpp #30373](https://github.com/ggml-org/llama.cpp/pull/30373) | Tuiles alignées sur les experts, liste compacte de descripteurs, grille bornée (llama.cpp : ≈94 % de threadgroups vides supprimés) | Déjà fait chez nous (plan de routes, segments par expert) ; seuls les threadgroups vides restent (voir `launched_subblocks`). Faible levier |
| [llama.cpp `mul_mm.metal`](https://github.com/ggml-org/llama.cpp/blob/a946eb541a6d3051496bdcc451a0fa85772b8dfd/ggml/src/ggml-metal/kernels/mul_mm.metal), `mul_mm_id` | Tuile de 64 poids × 32 tokens, 4 simdgroups ; bloc déquantifié une fois par pas K en mémoire threadgroup et partagé ; `matmul2d<…, execution_simdgroups<4>>` sur Metal 4 | Mécanisme de référence pour la **piste 2**. Exact seulement si chaque simdgroup garde la forme MMA et l'ordre K actuels ; `execution_simdgroups<4>` ou une autre forme `matmul2d` doivent être validés bit à bit (ordre interne non documenté par Apple) |
| [ExLlamaV3 `block_sparse_mlp.py`](https://github.com/turboderp-org/exllamav3/blob/8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0/exllamav3/modules/block_sparse_mlp.py) | Kernel fusé à décodage trellis jusqu'à `MOE_FUSED_ROWS` lignes par expert, avec tuiles de 16/32/64 lignes selon l'expert ; au-delà, reconstruction FP16 puis GEMM dense ; accumulation déterministe | CUDA/ROCm seulement. La reconstruction ne serait exacte que si le GEMM reproduit l'ordre K et les arrondis du kernel fusé. Non prioritaire |
| llama.cpp [`common.h`](https://github.com/ggml-org/llama.cpp/blob/a946eb541a6d3051496bdcc451a0fa85772b8dfd/common/common.h) (`n_batch` 2048, `n_ubatch` 512) ; mlx-lm `generate_step` (`prefill_step_size` 2048, 512 en spéculatif) | Les moteurs amont préfillent par lots de 512 à 2048 tokens, contre 256 chez nous | Argument d'ordre de grandeur pour la piste 1. Le contrat exact impose le découplage, pas un simple chunk plus grand |
| [vLLM `fused_moe.py`](https://github.com/vllm-project/vllm/blob/c4182020b39aa30fe985cc38404a43f077f7d052/vllm/model_executor/layers/fused_moe/fused_moe.py), [`batch_invariant_configs.py`](https://github.com/vllm-project/vllm/blob/c4182020b39aa30fe985cc38404a43f077f7d052/vllm/model_executor/determinism/batch_invariant_configs.py) | `BLOCK_SIZE_M` selon M ; en mode invariant, une configuration fixe indépendante de M | Même principe que notre exigence : une seule forme de kernel quelle que soit la taille du lot |
| [Thinking Machines, 10 septembre 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [SGLang déterministe](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic) | Opérations non invariantes par lot : split-K, changement d'instruction à petit M, réductions de normes fractionnées, split-KV de l'attention, cache KV traité à part | Liste de contrôle pour E1 : chez nous, les risques sont la sélection SDPA, `conv1d` et les queues ≤ 8 ou < 64 |
| [MLX `gated_delta_update`](https://github.com/ml-explore/mlx/blob/fc344c986c4169d99f76b5d95ae6b4a58a359a63/mlx/backend/metal/gated_delta_update.cpp) ([#4020](https://github.com/ml-explore/mlx/pull/4020)) ; [FLA `chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/9c102739e348968753d86bb729c57dd6efa1c413/fla/ops/gated_delta_rule/chunk.py) | GDN par chunks C = 8 ou 16 (NAX), validé par la perplexité et non bit à bit ; C dépend de T | **Hors contrat exact.** Le kernel compacté exact ([mlx-lm #1559](https://github.com/ml-explore/mlx-lm/pull/1559)) est déjà celui du moteur |
| [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/) | Accélérateur neuronal destiné au calcul dense du prefill ; exemple de matmul quantifié en `execution_simdgroups<4>` ; conseil de garder les données déquantifiées en cooperative tensors | Orientation pour la piste 2. Aucun chiffre ni taille de tuile publiés |

**Version de MLX : point à vérifier.** Le moteur épingle MLX **0.32.2** (workflows CI, `scripts/build-engine.sh`). Dans la copie 0.32.2 de `use_fallback`, le chemin D256 à plus de 8 queries est non fusionné, sauf à partir de 1024 queries causales sur NAX. La révision `fc344c9` de MLX lue par la recherche amont envoie au contraire toute requête de plus de 8 queries en D256 vers `sdpa_full_self_attention_nax`, sans seuil à 1024. L'hypothèse de cause P12 vaut donc pour 0.32.2. Une montée de version MLX changerait l'arithmétique de l'attention prefill même sans toucher aux chunks, et demanderait un oracle complet. Dans les deux versions, une tranche ou une queue de 8 tokens ou moins passe aux kernels vectoriels.

## Limites

Aucun chiffre de cette page n'est un gain mesuré. Le routage réel n'a pas été capturé. L'invariance par ligne du kernel segmenté n'est établie que pour le cas de la dernière ligne (`FINAL-ROW`). La sélection SDPA vient de la copie MLX 0.32.2 conservée dans les preuves de l'issue 44, pas d'une trace des kernels réellement atteints. Kani, CBMC et les tests ne sont pas concernés, puisqu'aucun code moteur n'a été modifié.
