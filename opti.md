# MLXL3 — journal des optimisations

À lire **avant** toute optimisation ; à mettre à jour **avant et après chaque
essai**, y compris les essais ratés. Voir [AGENTS.md](AGENTS.md).

## Format des nouvelles entrées

```text
### OPT-AAAA-MM-JJ-NN — Nom — statut
- Hypothèse / changement :
- Antécédents consultés / raison de retester, le cas échéant :
- Baseline / candidat (commit + diff ou options) :
- Environnement : matériel, OS, dépendances, alimentation, conditions connues.
- Protocole : modèle/bpw, shapes ou contexte, tokens, warmup, répétitions.
- Commandes et preuves :
- Résultats avant → après, unités et dispersion :
- Qualité / exactitude vérifiée ; contrôles non effectués :
- Conclusion / limites / prochaine condition de réexamen :
- Intégration : prototype / code local / app / publication, selon vérification.
```

## Historique à consulter avant de proposer une piste

Journal initialisé le 10 septembre 2026 à partir des rapports existants.
**Ce résumé n'est pas une transcription exhaustive des anciens essais.**
Pour les domaines concernés, lire également ces archives et rechercher la
piste dedans ; elles restent la source des protocoles et résultats détaillés.

| Domaine | Rapports existants |
| --- | --- |
| Decode général, Qwen, LFM | [10 septembre](docs/decode-investigation-2026-09-10.md), [R&D du 7 septembre](docs/general-performance-2026-09-07.md), [roadmap decode](docs/decode-roadmap-2026-09-04.md), [audit runtime](docs/audit-runtime-2026-09-04.md) |
| Qwen3.8 dense | [Essais decode Qwen3.8](docs/qwen38-decode-local.md) |
| Gemma | [Decode](docs/gemma-decode-investigation.md), [SDPA512](docs/gemma-sdpa512-investigation.md) |
| Prefill après outils/MCP | [R&D prefill](docs/tool-prefill-rd-2026-09-05.md) |
| Quantification EXL3 Metal | [Optimisations quantification](docs/metal-quantization-optimization.md), [LFM2.6](docs/lfm26-local-quantization.md), [Ling](docs/ling-local-quantization.md) |
| UI, streaming, sessions | [R&D du 7 septembre](docs/general-performance-2026-09-07.md), [audit v1](docs/audit-v1-2026-09-06.md), [validation v1](docs/v1-validation.md) |

Les statuts ci-dessous décrivent les conclusions des rapports à leur date,
pas une vérification de la version actuellement installée ou publiée.
Les chemins `build/` sont des preuves locales temporaires, non garanties dans Git.

## 10 septembre 2026 — recherche decode générale

Source commune : [rapport complet et preuves](docs/decode-investigation-2026-09-10.md).
M5, 24 GiB, macOS 27.0 (26A428), MLX 0.32.2. Trois paires alternées,
96 tokens générés, warmup exclu. Batterie puis secteur, thermique non contrôlée.
Les différences ci-dessous sont les médianes des variations appariées.
Texte identique dans les paires ; pas de nouvelle validation bit-à-bit des
logits/caches. **Aucun changement de production retenu dans cette série.**

### OPT-2026-09-10-01 — Chargement TensorOps — bloqué

Le chargement normal de Qwen échoue au warmup sur
`get_destination_cooperative_tensor` (contrainte de template dans les headers
MetalPerformancePrimitives). Cause exacte non établie. Pour les essais 02–06,
les deux côtés utilisent `MLXL3_TENSOR_QMM=0 MLXL3_TENSOR_SEGMENTED_QMM=0`.
Ce contournement de benchmark n'est pas activé dans l'app ; les mesures ne
valident pas le chemin prefill TensorOps normal. Réexaminer après correction
de compatibilité macOS 27.

### OPT-2026-09-10-02 — Budgets command-buffer, Qwen — rejeté

- Candidat : `MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000` vs défauts.
- Qwen3.6-35B-A3B EXL3 2.49 bpw : decode non caché **−0,38 %**.
- Pic d'allocation MLX : **12,422 → 13,302 GB**. Un fort gain apparent sur une
  paire provient d'une baseline subitement lente, pas d'un gain fiable.
- Preuve : `build/decode-command-buffers-qwen-fallback-20260910.jsonl`.

### OPT-2026-09-10-03 — Mêmes budgets, LFM8 — rejeté

- LFM2.5-8B-A1B EXL3 3.10 bpw : decode **−2,41 %**, caché **−1,59 %**.
- Preuve : `build/decode-command-buffers-lfm8-20260910.jsonl`.

### OPT-2026-09-10-04 — Compiler 24 feed-forward LFM8 — non concluant

- Réutilisation du compilateur stateless existant : decode **+0,11 %**, bruit.
- Preuve : `build/decode-ff-compile-lfm8-all-20260910.jsonl`.
- Le pilote limité à deux blocs n'a pas établi de gain non plus ; ne pas
  additionner les résultats de ce pilote et de la série complète.

### OPT-2026-09-10-05 — Compiler 30 feed-forward LFM2.6 — non concluant

- LFM2.5-2.6B EXL3 4 bpw : decode **+0,30 %**, bruit.
- Preuve : `build/decode-ff-compile-lfm26-20260910.jsonl`.

### OPT-2026-09-10-06 — Compiler les entrées QMV partagées — rejeté

- `mx.compile` des fonctions denses/groupées/experts réellement appelées.
- LFM8, série séparée sur secteur : decode **−3,06 %** médian.
- Preuve : `build/decode-qmv-compile-lfm8-ac-20260910.jsonl` ; runner local :
  `build/bench_decode_ff_compile.py` (`--qmv` pour cette variante).
- Attention : l'ancien `work/compiled_qmv_bench.py` cible des alias obsolètes.
  Ne pas utiliser ses résultats comme validation du chemin actuel.

## 7 septembre 2026 — résultats antérieurs à ne pas redécouvrir

Source et protocoles : [rapport général](docs/general-performance-2026-09-07.md).
Les validations rapportées ici ne lèvent pas le blocage macOS 27 découvert
le 10 septembre. Les gains portent uniquement sur le périmètre indiqué.

| ID | Essai | Résultat documenté | Conclusion historique |
| --- | --- | --- | --- |
| OPT-2026-09-07-01 | Sortir les calculs d'adresse QMM de la boucle | Prefill LFM2.6 +6,84 %, Qwen +6,09 % ; decode dans le bruit ; logits/caches forcés bit-à-bit | Validé sur le chemin TensorOps de l'époque |
| OPT-2026-09-07-02 | Évincer les snapshots avant la session entière | TTFT second tour 5,1517 → 0,1673 s ; 3 500 tokens réutilisés dans le scénario contraint | Validé pour ce scénario, pas un gain decode universel |
| OPT-2026-09-07-03 | Préparation incrémentale des blocs de code UI | À 1 MiB : 52,482 → 1,640 ms par ajout | Validé en microbenchmark CPU, pas FPS/tok/s |
| OPT-2026-09-07-04 | Comptages de chaînes bornés | Grande ligne ASCII 3,561 → 2,946 ms ; Unicode 54,674 → 29,909 ms | Validé séparément, ne pas multiplier les gains |
| OPT-2026-09-07-05 | Validation UUID du bridge sans attendre la queue IO | Délai synthétique 53,877 → 0,034 ms quand IO occupée 50 ms | Validé pour la réactivité du bridge, pas le decode |
| OPT-2026-09-07-06 | Omettre les diagnostics quant non utilisés | Temps −1,51 % médian, quatre projections K=4 ; fingerprints exacts | Mesure limitée, remplacée par la comparaison combinée suivante |
| OPT-2026-09-07-07 | Quatre changements de préparation quant combinés | Temps −2,46 % médian ; fingerprints/scores exacts | Validé sur quatre projections, pas conversion complète ; ne pas ajouter −1,51 % |
| OPT-2026-09-07-08 | Supprimer copie gate/up MoE prefill | Prefill Qwen environ −3 à −6 % malgré contrôles numériques réussis | Rejeté |
| OPT-2026-09-07-09 | Fusion gather/Hadamard decode | LFM8 +0,46 % non caché, +0,02 % caché ; Qwen instable | Retiré, gain non établi |
| OPT-2026-09-07-10 | Extraction QMM 64 bits → funnel 32 bits | Plus lent, pourcentage non précisé dans ce rapport | Rejeté |
| OPT-2026-09-07-11 | Préchargement des mots QMV | Résultats synthétiques mixtes | Non retenu |
| OPT-2026-09-07-12 | Grouper les projections LFM w1/w3 | Decode +1,33 %, prefill −6,59 %, environ +171 MB pic requête cachée | Rejeté ; résoudre les copies compactantes avant réessai |
| OPT-2026-09-07-13 | Cacher les métadonnées récurrentes vides | Wrapper CPU ~580–600 → 403–434 ns ; pas de gain modèle mesuré | Retiré, impact estimé négligeable |

## Pistes ouvertes, non validées

- Corriger la compatibilité TensorOps macOS 27 avant les nouvelles validations
  bout en bout du chemin normal.
- Étudier des régions decode compilées plus larges : attention, normalisation,
  RoPE et cache ensemble ; les blocs Qwen récurrents et plusieurs MLP sont
  déjà compilés. Les essais isolés 04–06 n'ont pas montré de gain.
- Projections groupées mixtes (bits/codebooks) et suppression des copies
  compactantes : vérifier les formes réellement exclues avant tout prototype.
- Profiler les dispatchs GPU et les lectures réelles avant une nouvelle variante
  QMV ; taille du fichier / tok/s n'est pas une mesure de bande passante GPU.

## Nouvelle série du 10 septembre 2026

### OPT-2026-09-10-07 — Masquage top-k par les tokens conservés — non concluant

- Hypothèse : conserver exactement `argpartition(-logprobs, kth=k-1)`, mais
  remplacer le scatter des V-k tokens rejetés par un tableau rempli de -inf
  et un scatter des k valeurs conservées, économise des écritures irrégulières.
- Antécédents : D07 du roadmap/audit proposait le sampler ; aucun essai de ce
  changement précis trouvé. Pas de changement du greedy, du RNG ou de top-k.
- Baseline : `mlx_lm.sample_utils.apply_top_k` installé, MLX 0.32.2.
- Protocole : parité des sorties FP16/BF16/FP32, ties/NaN/Inf et seeds ; puis
  microbenchmark vocab 65 536/248 320/262 144, k=40. Si favorable, trois paires
  modèle à température 0,7, top-k 40, seeds identiques, 128 tokens.
- M5 24 GiB, macOS 27, secteur (82 %), thermique non contrôlée ; workaround
  TensorOps de l'essai 01 identique des deux côtés pour les essais modèle.
- Commande/prototype : `.venv/bin/python benchmarks/bench_topk_scatter.py` ;
  preuves prévues `build/topk-scatter-20260910.jsonl` et fichiers modèle séparés.
- Résultat : 252 cas de masques bit-à-bit, sampling avec seed contrôlé réussi.
  Microbenchmark favorable mais Qwen decode +0,70 % médian, paires
  [+0,70 ; −0,86 ; +16,27] %, trop instables pour promotion.
  Preuve modèle : `build/topk-scatter-qwen-20260910.jsonl`.
- Reproduction modèle avec le runner actuel : ajouter
  `--model models/Qwen3.6-35B-A3B-EXL3-2.49bpw --temperature 0.7 --top-k 40`.
  Les défauts ultérieurs du runner sont ceux de l'essai 08 (0,2 / 80).
- Intégration : diagnostic uniquement, pas l'app.

### OPT-2026-09-10-08 — Top-k hiérarchique exact — non concluant

- Nouvelle hypothèse : dans le backend Metal MLX installé, ArgPartition appelle
  un tri complet. Trier des blocs, retenir leurs k meilleurs puis trier ces
  candidats évite les grandes fusions du tri global. Aucun token du top-k
  global ne peut être exclu par un top-k local. Les ties doivent conserver
  l'ordre initial via les tris stables du backend actuel ; contrôler NaN/Inf.
- Différence avec 07 : réduit le travail du tri, pas seulement le masquage.
- Prototype : `benchmarks/bench_topk_scatter.py --hierarchical`. Bloc 1024,
  fallback référence si petit vocabulaire ou k >= bloc. Padding NaN en fin,
  comme les sentinelles du tri MLX. Non destiné aux backends non validés.
- Contrôles prévus : masques bit-à-bit FP16/BF16/FP32, ties, NaN/Inf, bords de
  blocs, k=1/40/80/V−1 ; sampling avec même seed ; puis modèle température
  0,2/top-k 80 (défauts GUI), trois paires de 128 tokens si parité réussie.
- Mêmes matériel/OS et workaround que 07. Prototypes seulement, non intégrés.
- Preuves : `build/topk-hierarchical-20260910.jsonl` (1 020 cas bit-à-bit,
  sampling seed identique), `build/topk-hierarchical-qwen-20260910.jsonl`.
  Qwen : decode +0,38 % médian, paires [−0,36 ; +1,54 ; +0,38] %, bruit.
  LFM1.2 Thinking : −0,96 % médian, paires [−0,96 ; −6,53 ; +6,39] %,
  textes identiques ; `build/topk-hierarchical-lfm12-20260910.jsonl`.
  Pas de gain modèle fiable, pas de promotion malgré le gain du filtre isolé.
- Microbenchmark synchronisé (temps CPU+GPU, pas temps GPU pur) : réduction
  médiane du temps top-k de **18,99 % / 33,71 % / 33,60 %** pour les vocabulaires
  65 536 / 248 320 / 262 144. Ne pas appliquer ces pourcentages au decode.

### OPT-2026-09-10-09 — Compiler la pénalité de répétition bornée — rejeté

- Hypothèse : compiler gather/arithmétique/scatter de la pénalité existante,
  avec la fenêtre de tokens bornée avant l'entrée compilée pour ne pas créer
  un graphe par longueur d'historique. Ne change pas la pénalité ni son ordre.
- Antécédent D01 : borne déjà l'historique stable du préfixe, pas cette fusion
  du processeur. Les tests QMV/FFN 04–06 concernaient d'autres opérations.
- Référence : factory MLX-LM installée. Candidat : même closure sous
  `mx.compile`, entrée `tokens[-20:]`, cache borné des factories dans le prototype.
- Protocole : comparaison bit-à-bit sur dtypes, tokens répétés, signes et
  longueurs de fenêtre ; microbenchmark puis trois paires greedy modèle si
  exact. Même environnement/workaround que 08 ; pas de top-k modifié simultanément.
- Prototype/proofs : `benchmarks/bench_topk_scatter.py --penalty`,
  `build/penalty-compile-20260910.jsonl` puis fichiers modèle séparés.
- Premier contrôle de parité échoue : FP16, vocab 257, historique 21 tokens,
  pénalité 0,8. Arrêt avant benchmark modèle. Diagnostic des différences en
  FP16 confirme aussi un écart avec la pénalité par défaut 1,05 :
  0,65185546875 → 0,65234375 (un logit fini). Preuve :
  `build/penalty-compile-diagnostic-20260910.jsonl`. Aucun gain revendiqué.
  Non intégré dans le moteur/l'app.

### OPT-2026-09-10-10 — Types explicites TensorOps macOS 27 — validé en prototype

- Suite de 01 justifiée par l'inspection des nouveaux headers MPP : les
  contraintes de types n'effacent pas l'espace d'adressage `thread` porté
  par `decltype(variable)`. Tester les mêmes types tensor/cooperative_tensor
  via leurs alias explicites sans ce qualificatif ; ne pas changer le calcul.
- Ce n'est pas un boost decode annoncé : rétablir un chemin normal de prefill
  est nécessaire aux mesures complètes. Pas de modification des headers Apple.
- Prototype : intercepteur local de la factory Metal dans un runner de test,
  puis suite QMM existante. Même MLX/macOS, aucun TensorOps désactivé.
- Commande : `.venv/bin/python build/check_tensor_types.py` ; sortie
  `build/tensor-types-check-20260910.log` (premiers 24 cas), puis
  `build/tensor-types-full-20260910.log` : **87 tests QMM réussis**.
- Chargement + warmup + 32 tokens Qwen avec TensorOps activé réussis :
  `.venv/bin/python build/check_tensor_types.py --model models/Qwen3.6-35B-A3B-EXL3-2.49bpw` ;
  `build/tensor-types-qwen-20260910.jsonl`, stderr vide. Une seule génération,
  pas un A/B de performance. Parité macOS 26/27 non mesurée ; pas de promesse
  bit-à-bit inter-OS ni validation exhaustive de tous les modèles.
- Correctif testé dans les factories `_qmm_tensor_kernel` et
  `_segmented_expert_qmm_tensor_kernel` : remplacer les arguments
  `decltype(first_left), decltype(right), float` de l'accumulateur par
  `tensor<device half, dextents<int, 2>, tensor_inline>`,
  `tensor_ops::matmul2d<descriptor, execution_simdgroup>::cooperative_tensor_right_input_t<half, half, float>`,
  `float`. Les calculs, tuiles et poids ne changent pas.
- Intégration : intercepteur du runner de diagnostic uniquement. Sources de
  production, app installée et dépendances non modifiées. Avant intégration :
  ajouter un contrôle de source durable, vérifier l'autre OS supporté et le
  chemin segmenté sur une matrice dédiée. Pas de boost decode revendiqué.

## 11 septembre 2026 — objectif : au moins 10 % sur une métrique complète

### OPT-2026-09-11-01 — Borner les chunks de prefill en vol — non concluant

- Hypothèse : `_prepare` soumet tous les chunks par `async_eval` avant
  l'attente terminale ; borner cette file peut réduire le pic des temporaires
  sans changer taille des chunks, poids, cache ou précision.
- Antécédents : les changements de taille de chunks du 5 septembre ont été
  rejetés pour dérive numérique. Ici seul l'ordonnancement change. Aucun
  essai de backpressure prefill trouvé dans les archives consultées.
- Protocole : chemin réel `_stream_response`, prompts fixes ~8k puis ~16k
  selon mémoire, 16 tokens générés ; A/B synchronisation de chaque chunk
  stable, puis pipeline de deux chunks si utile. Comparer pic MLX, temps,
  texte et empreintes des caches ; confirmer en processus séparés si favorable.
- M5 24 GiB, macOS 27.0/26A428, MLX 0.32.2, batterie 46 %, thermique non
  contrôlée. Appliquer le correctif de types de 2026-09-10-10 aux deux côtés
  du benchmark, uniquement dans le prototype, pour garder TensorOps normal.
- Préserver les autres changements locaux ; aucune refonte/changement de
  langage, aucun push ni remplacement d'app sans validation supplémentaire.
- Preuves prévues : `build/prefill-flight-20260911*.jsonl`, runner dans
  `benchmarks/bench_prefill_flight.py`.
- Pilote LFM1.2 Thinking, 6 154 tokens : même pic MLX **1,851846224 GB**,
  TTFT 3,461 → 3,497 s ; texte et états finaux identiques. Pas de gain RAM ;
  ne pas poursuivre cette synchronisation systématique sur ce résultat.
  `build/prefill-flight-20260911-lfm12.jsonl`.

### OPT-2026-09-11-02 — Partager les KV du préfixe après génération — non concluant

- Hypothèse : dans `GenerationSession.finish`, le préfixe stable KV dupliqué
  est identique au début des KV de la génération terminée. Remplacer seulement
  ces copies KV par des forks/vues du cache final peut économiser la RAM
  retenue entre les tours, en conservant les états récurrents de chaque date.
- Différence avec COW au début de génération : pas de copie immédiatement
  déclenchée par les tokens générés. Réutilise `SharedKVCache` existant ; pas
  de pagination ni changement de langage. Mesurer séparément RAM active au
  repos et pic pendant la génération, ne pas confondre les deux.
- Prototype : option `--share-finished-kv` du runner 01, LFM1.2 Thinking,
  ~16k tokens, 16 générés ; baseline inchangée contre partage post-finish.
  Si favorable : trois paires, qualité préfixe/final, conversation suivante,
  branche au préfixe et snapshots indépendants ; vérification deuxième modèle.
- Mêmes environnement et correctif de types de benchmark que 01. Logs
  `build/finished-kv-20260911*.jsonl`. Aucun résultat mesuré ni intégration.
- Pilote 16 394 tokens : allocation MLX retenue **1 483 087 368 → 1 278 647 816
  octets (−13,78 %)** ; texte + préfixe + cache final identiques. Pic pendant
  génération inchangé (2,301 GB). Une seule paire, pas encore une validation.
- Étape suivante : vider aussi les allocations libérées après partage pour
  restituer la RAM au système, et mesurer `proc_pid_rusage.ri_phys_footprint`
  comme le widget. Contrôler séparément pool allocateur/MLX actif/empreinte OS.
  Candidat combiné partage + restitution ; ne pas additionner des pourcentages.
- Trois paires LFM16k : −13,79 % d'allocation MLX active retenue, mêmes textes
  et caches. **Pas de baisse nette d'empreinte physique macOS** (−1,95 %, +0,08 %,
  +0,05 % de réduction), pic inchangé. Ne pas présenter cela comme un gain
  de RAM système ou de pic. `build/finished-kv-20260911-lfm12-final.jsonl`.
  Non intégré ; contrôle des tours suivants encore requis pour promotion.
- Recherche suspendue sur ce candidat : pas de gain RAM système établi,
  aucune modification de `GenerationSession.finish` conservée en production.

### OPT-2026-09-11-03 — Étendre BM64 dense aux autres matrices — non concluant

- Antécédent : M64 existe déjà pour grandes matrices MUL1 à entrée >=4096.
  Nouvelle couverture testée : entrées 2048 et autres codebooks, uniquement
  lorsque les lignes sont multiples de 64, >=128, hors head >=65536 sorties.
  Garder BK16/BN32, donc même ordre de réduction ; parité à vérifier.
- Hypothèse : amortir la déquantification EXL3 sur 64 plutôt que 32 lignes,
  en évitant de modifier chunks/padding/poids. Pas de nouveau langage/refonte.
- Runner 01 option `--dense-m64`, LFM2.6 4 bpw, ~4k tokens, 16 générés ;
  comparaisons suivantes seulement si caches et texte exacts. Les deux côtés
  utilisent la correction de types TensorOps, sans autre candidat RAM combiné.
- M5/macOS27, batterie, thermique non mesurée. Logs
  `build/dense-m64-20260911*.jsonl`. Aucun résultat ni intégration.
- Pilote 4 107 tokens : 789,57 → 811,95 tok/s prefill (+2,84 %), TTFT
  5,227 → 5,071 s ; pic identique, texte/préfixe/cache final exacts.
  Une seule paire : non concluant pour promotion, pas de gain généralisé.
  `build/dense-m64-20260911-lfm26.jsonl`. Prototype seulement.

### OPT-2026-09-11-04 — BM64 TensorOps segmenté MoE — validé sur M5, code local

- Hypothèse : déquantifier chaque tuile de poids une fois pour 64 lignes au
  lieu de deux fois pour 32, dans les blocs experts déjà paddés à 64 lignes.
  Antécédents : BM8/16/32 testés, BM64 pas dans la matrice historique.
- Prototype : `_SEGMENTED_TENSOR_ROWS=64` en mémoire contre 32 ; buckets et
  locality désactivés, BK16/BN32 inchangés. Pas de refonte ni de nouveau langage.
- Qwen3.6 35B A3B EXL3 2.49 bpw, prompt fixe, 16 tokens, pilote puis trois
  paires alternées si texte et empreintes de tous les caches restent exacts.
  Même correctif TensorOps 2026-09-10-10 des deux côtés ; M5/macOS27 sur batterie.
- Runner `bench_prefill_flight.py --segmented-m64 --repeats 512`, preuves
  `build/segmented-m64-20260911*.jsonl`. Aucun résultat ni intégration.
- Pilote interrompu avant baseline : compilation Metal du chemin segmenté
  refuse la référence au descripteur local dans l'alias imbriqué utilisé par
  le correctif 10-10. Le warmup court précédent ne couvrait pas ce chemin.
  `build/segmented-m64-20260911-qwen.stderr`. Pas de mesure de performance.
- Révision du protocole : nommer les alias d'opération/opérande avant leur
  usage comme arguments template, sans changer leurs types ni les calculs.
  Réessai de compilation puis A/B dans un nouveau log `*-qwen-alias.jsonl`.
- Alias seul : même erreur, avant mesure. Le segmenté est inutilement instancié
  avec `T=half` alors que sa source utilise exclusivement `half`. Retirer ce
  paramètre template inutilisé dans le runner pour éviter la contrainte sur
  le descripteur local ; calcul inchangé. Log `*-qwen-no-template.jsonl`.
- Compilation rétablie. Pilote 4 106 tokens : prefill 414,33 → 517,90 tok/s
  (+25,00 %), TTFT 9,967 → 7,951 s ; cache/préfixe/texte exacts. Les compilations
  propres à la forme ne sont pas exclues : pas encore de gain validé.
- Confirmation prévue : une paire complète exclue (`pair=-1`), puis trois
  paires AB/BA, mêmes 4 106 tokens, options `--warmup-pairs 1 --pairs 3`,
  log `build/segmented-m64-20260911-qwen-warm.jsonl`. Contrôle numérique
  dédié : tous les codebooks et bits supportés, segments courts/64/65/129 et
  experts vides ; extension à LFM8 si les résultats restent favorables.
- Trois paires après warmup : prefill **+7,89 / +10,46 / +9,43 %** (médiane
  +9,43 %), TTFT **−7,26 / −9,41 / −8,64 %** ; caches et texte exacts.
  Le pilote +25 % surestimait le gain chaud. Pic ~14,05–14,06 GB inchangé.
  Gain prometteur mais objectif 10 % pas encore confirmé ; prototype seulement.

### OPT-2026-09-11-05 — Adresses invariantes dans le QMM segmenté — validé en combinaison, code local

- Antécédent : le hoisting du 7 septembre ne couvre que le QMM dense.
  Le QMM segmenté recalcule encore permutation/positions/shift à chaque BK16.
  Même technique appliquée aux adresses du bloc expert, sans changer les
  lectures ni l'ordre des additions. Garde de capacité coopérative conservée.
- Prototype dans la factory du runner : préparer offsets/shifts avant la boucle
  K, puis incrémenter uniquement la base de profondeur ; aucun poids déquantifié
  matérialisé. Comparer candidat combiné BM64 + hoisting à baseline BM32 normale,
  sans additionner les gains. Mêmes Qwen4k, warmup exclu, trois paires si pilote
  exact, puis matrice numérique dédiée. Logs `build/segmented-hoist-20260911*`.
- Aucune mesure ni intégration à ce stade.
- Pilote combiné chaud Qwen4k : 483,68 → 532,94 tok/s (+10,18 %), TTFT
  8,511 → 7,727 s (−9,21 %) ; pic 14,0523 GB identique, texte et caches exacts.
  Confirmation nécessaire. Avant cela, exécuter les 24 cas de
  `tests/test_segmented_m64.py` avec l'intercepteur compatible macOS27 et
  le hoisting activé : `build/segmented-m64-quality-20260911.log`.
- Matrice 24 cas réussie : BM32/BM64 avec hoisting actif des deux côtés,
  sorties finies et bit-à-bit. Cette matrice ne compare pas encore hoisting
  actif/inactif ; la comparaison combinée modèle vérifie ce dernier.
- Validation externe de forme : LFM8 A1B 3.10 bpw, même prompt répété 512,
  paire de warmup exclue puis trois paires, candidat BM64 + hoisting,
  log `build/segmented-hoist-20260911-lfm8-warm.jsonl`.
- LFM8, 4 106 tokens : trois gains prefill **+31,43 / +32,37 / +29,40 %**,
  TTFT **−23,80 / −24,40 / −22,67 %** ; textes et caches exacts. Pic ~4,923 GB
  inchangé. Candidat combiné en prototype uniquement ; objectif de mesure
  atteint sur ce scénario, sans conclusion llama.cpp ni decode universel.
- Ablation avant intégration : BM64 seul sur LFM8, mêmes trois paires chaudes,
  pour éviter de conserver un hoisting inutile ; log
  `build/segmented-m64-20260911-lfm8-warm.jsonl`.
- Intégrer maintenant uniquement le correctif de compilation macOS27 commun
  (alias de types explicites et suppression du template T inutilisé segmenté),
  puis contrôler les suites existantes. Cela n'est pas compté comme un boost.
- Ablation BM64 seul : LFM8 prefill **+22,97 / +23,67 / +22,98 %**,
  TTFT **−18,59 / −19,03 / −18,68 %**, texte/caches exacts, pic inchangé.
  Les séries ne sont pas directement additionnables ; l'essai combiné reste
  la preuve du gain cumulé. Garder les deux candidats pour validation locale.
- Intégration prévue dans le kernel commun : BM64 automatique à partir de
  4 096 lignes routées, BM32 en dessous, surcharge manuelle préservée (0=auto).
  Hoisting segmenté distinct, désactivable pour A/B et garde de capacité.
  Aucun chemin decode modifié. Petits prompts et suites numériques à vérifier
  avant de qualifier le code local ; ni app reconstruite ni publication.
- Code local intégré pour vérification. Le runner utilise désormais les vraies
  options de production (baseline BM32/hoist off, candidat auto/hoist on),
  sans réécriture des sources Metal. Les anciennes commandes utilisant
  `tensor_types_factory()` décrivent le prototype retiré du runner.
- Suite lancée : `.venv/bin/python -m pytest -q -x tests/test_segmented_m64.py
  tests/test_qmm_address_hoist.py tests/test_dense_prefill.py tests/test_audit_candidates.py`.
  Log `build/segmented-production-quality-20260911.log`. Matrice segmentée compare
  maintenant BM32 sans hoist à BM64 avec hoist. Ajout de contrôles durables
  capacité insuffisante et dispatch auto ; à exécuter après la suite en cours.
- **175 tests réussis** sans intercepteur Metal. Tests complémentaires ajoutés
  ensuite encore à exécuter ; compatibilité autre version macOS non mesurée.
- Confirmation finale du code réellement intégré : Qwen4k et LFM8 4k,
  `--segmented-m64 --segmented-hoist --warmup-pairs 1 --pairs 3 --max-tokens 96`,
  puis prompts courts `--repeats 16 --pairs 3 --max-tokens 32`. Journaux
  `build/segmented-final-20260911-*.jsonl`. Garder la comparaison à 96 tokens
  distincte de celles à 16 ; vérifier aussi decode et pic sans revendiquer un
  gain decode sur ce kernel de prefill. Aucun benchmark llama.cpp effectué.
- Holdout prévu après validation initiale : document technique non répétitif
  `docs/audit-runtime-2026-09-04.md`, option `--prompt-file`, warmup puis trois
  paires à 32 tokens sur LFM8 et Qwen. Log `build/segmented-heldout-20260911-*`.
  Ce contrôle évite de limiter la conclusion à une phrase répétée qui peut
  concentrer le routing sur quelques experts ; conserver les résultats séparés.
- Production Qwen4k/96 générés : gains prefill **+14,42 / +10,70 / +12,35 %**
  (médiane +12,35 %), TTFT **−12,57 / −9,69 / −11,02 %**. Decode
  −1,73 / +2,74 / −0,48 % : pas de gain établi, fluctuation de batterie/thermique.
  Pic identique 14,0523 GB ; empreinte OS candidat +~6–8 MB, pas un gain RAM.
  Caches et texte exacts. `build/segmented-final-20260911-qwen.jsonl`.
- Production LFM8 4k/96 générés : prefill **+34,86 / +36,84 / +30,85 %**
  (médiane +34,86 %), TTFT **−25,74 / −26,88 / −23,54 %**. Decode +1,08 / +1,55 /
  +2,41 %, sans changement decode direct ni revendication indépendante. Pic
  4,92316 GB identique ; texte/caches exacts. Empreinte OS ~4,70 GB, sans baisse.
  `build/segmented-final-20260911-lfm8.jsonl`. Mesures sur batterie, pas de
  garantie de ces pourcentages sur tous les prompts ou tous les Mac.
- Prompts courts 138 tokens / 32 générés : LFM8 prefill +16,14 / +12,99 /
  +17,46 %, TTFT −12,94 / −10,96 / −15,13 % ; Qwen prefill +11,84 / +8,95 /
  +6,95 %, TTFT −10,58 / −7,98 / −6,35 %. Caches/textes exacts, pics inchangés.
  Decode court : LFM médiane −2,49 %, Qwen −1,31 % (31 tokens chronométrés,
  petites fluctuations opposées à certaines paires de 96 tokens) ; ne pas
  cacher cette limite ni présenter une accélération decode.
  `build/segmented-final-20260911-{lfm8,qwen}-short.jsonl`.
- Holdout LFM8, document varié 2 746 tokens / 32 générés : prefill **+24,77 /
  +25,82 / +25,59 %** ; TTFT **−19,84 / −20,45 / −20,40 %**. Texte/caches exacts,
  pic ~4,72155 GB stable. Decode −3,79 / −3,89 / +0,32 % : petite régression
  dans deux des trois courts échantillons, à conserver dans le bilan.
  `build/segmented-heldout-20260911-lfm8.jsonl`.
- Holdout Qwen, 2 842 tokens / 32 générés : prefill +6,63 / +14,12 / +9,37 %,
  TTFT −6,32 / −12,18 / −8,63 %, decode médiane −0,62 %. Caches et texte exacts,
  pic ~13,91–13,92 GB sans gain établi. `build/segmented-heldout-20260911-qwen.jsonl`.
- **177 tests réussis** après ajout des contrôles capacité/auto-tile, sans
  intercepteur. `build/segmented-production-quality-final-20260911.log`.
  Dernier smoke prévu du chemin non-TensorOps :
  `MLXL3_TENSOR_SEGMENTED_QMM=0 .venv/bin/python -m pytest -q tests/test_audit_candidates.py -k device_side_route_buckets`,
  log `build/segmented-fallback-quality-20260911.log`. Pas un test sur un vrai M1.
- Rapport durable : [prefill-investigation-2026-09-11.md](docs/prefill-investigation-2026-09-11.md).
  Mesures finales/caches/manifestes copiés par patch dans
  [prefill-segmented-20260911.json](benchmarks/results/prefill-segmented-20260911.json),
  donc conservables dans Git même si les logs `build/` sont ensuite nettoyés.
- Intégration : `src/mlxl3/kernels/qmv.py` local, chemin partagé du moteur/CLI ;
  pas d'app reconstruite/installée, de push ou de publication. Aucun changement
  de langage/refonte. Aucun gain RAM/decode ni supériorité sur llama.cpp annoncé.
  Les candidats RAM/tuilage dense étendu non concluants restent hors production.
- Contrôle final : **8 tests non-TensorOps réussis**, 52 désélectionnés ;
  `git diff --check` et compilation Python des fichiers modifiés réussis.
  Le lanceur `~/.local/bin/mlxl3` utilise bien la `.venv` de ce dépôt ; import
  vérifié vers ce `qmv.py`, auto-tile=0 et hoisting actif. Aucun benchmark ne
  reste en cours. L'objectif >=10 % est atteint en prefill et TTFT sur les
  scénarios précisés ; les limites decode/RAM et de généralisation restent
  celles du rapport, sans prétendre avoir optimisé tout modèle/tout contexte.

### Publication demandée le 11 septembre 2026

- L'utilisateur demande de terminer puis pousser sur GitHub. Préparer le lot
  validé sur `main` : kernel, tests, runner, preuves, rapports et consignes du
  journal. Les modifications Ling/quantification restent locales hors du commit.
- Aucune release/DMG ni reconstruction de l'app n'est demandée par ce push.
  Le résultat du push sera consigné après vérification du commit distant.
- Push du code **réussi** : `f7825f49986b31c44a44b225faef3b9586eb0d0d`
  sur `https://github.com/0xZKnw/mlxl3`, branche `main`. Commit distant confirmé
  avec `git ls-remote origin refs/heads/main`. Tests/préuves/journal inclus ;
  changements Ling préservés hors commit. App installée et DMG inchangés.
