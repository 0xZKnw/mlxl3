# Audit runtime — premier lot vérifié

Source : audit de 31 pages « MLXL3 sur Apple M5 », fourni le 4 septembre
2026. Branche `codex/audit-exact-runtime`, base production `1334391`.
Le rapport prescrit une progression par validations, pas l'activation simultanée
de toutes les pistes. Ce lot ne prétend pas terminer le backlog.

## Changements

| Point | Implémentation | Contrat et limites |
| --- | --- | --- |
| D01 | L'historique stable donné aux pénalités est borné par leur fenêtre explicite, 20 pour la répétition actuellement configurée. | Q0 : ordre et multiplicité conservés ; processeur inconnu = historique complet. L'accumulation interne de MLX-LM reste inchangée. |
| P01 | Inversion de la permutation de routage par scatter entier Metal, au lieu d'un second argsort. | Q0 : le premier argsort, les slots experts et la réduction ne changent pas. Précondition : une permutation issue d'argsort, pas des IDs experts bruts. |
| M01 | Estimation et admission des snapshots avant copie ; éviction des anciens blocs avant allocation. | Q0 pour les valeurs. Un snapshot refusé peut modifier les futurs cache hits, pas le contexte logique. |
| M02 | Fork KV de génération avec réserve arrondie au pas du cache ; snapshots compacts. | Q0 sur le préfixe utile, stockage indépendant. Compte la réserve réelle des KVCache ; les autres alias restent estimés conservativement. |
| T01 | Annulation et garde mémoire aux frontières de chunks, y compris le callback de prefill MLX-LM. Cache partiel invalidé. | Pas de synchronisation GPU supplémentaire ; les chunks déjà soumis ne sont pas interrompus. Le scheduler peut réduire son chunk à mesure que la mémoire augmente. |
| V01/T01 | Benchmark sans liste de réponses/logprobs, manifeste, lignes brutes, mesures template / prefill stable / fork / premier texte séparées. | Temps muraux, pas temps GPU. Le débit prefill agrégé comprend encore le premier pas de decode mesuré par MLX-LM pour le suffixe. |

Les anciens benchmarks Gemma CLI et LFM ne conservent plus les vecteurs de
logprobs de chaque réponse. Ils conservent texte/IDs et la dernière réponse.

## Validation

Suite complète : **278 tests réussis, 3 ignorés** (fixtures locales absentes).
Nouveaux tests : permutations exhaustives jusqu'à 5 éléments ; routage à IDs
experts répétés jusqu'à 8192 slots ; forks KV autour de 256/512, indépendance,
réserve, cache vide ; admission avant allocation ; pénalités répétition,
présence et fréquence bit-exactes ; fenêtres et préfixes vides/longs ; annulation
et garde mémoire de prefill.

Qwen3.6-35B-A3B EXL3 2.49 bpw : 394 tokens de prompt, 32 générés, trois
scénarios par processus. Référence = mêmes changements runtime, sauf inversion
avec ancien argsort. **Ce n'est pas un A/B de tout le lot contre main.**

| Scénario | Ancien inverse : TTFT / decode | Scatter : TTFT / decode |
| --- | --- | --- |
| Froid | 965 ms / 55.09 tok/s | 986 ms / 54.79 tok/s |
| Chaud sans cache | 950 ms / 52.52 tok/s | 949 ms / 54.89 tok/s |
| Préfixe réutilisé | 196 ms / 54.42 tok/s | 189 ms / 55.09 tok/s |

Les six textes ont le même SHA256
`6883a451b7d19c81dfd16927640408438390201fea5981d40c7f2ee39d4cdfa5`.
Cela ne prouve pas l'égalité de tous les logits/états de tous les modèles.
Une seule paire de processus, prompt synthétique court et état thermique non
mesuré : **aucun gain global de decode validé**. Le prefill chaud est pratiquement
identique (771 contre 773 ms de prefill stable).

Un premier microbenchmark synchronisé de l'inversion seule donnait un facteur
de débit médian +10%, +28%, +36% à 512/4096/16384 slots. Ces valeurs incluent
la soumission/synchronisation CPU, ne sont ni des gains modèle ni une mesure de
bande passante DRAM. Les nouvelles exécutions brutes sont conservées séparément.

Résultats reproductibles :

```sh
.venv/bin/python benchmarks/benchmark_audit_runtime.py \
  --model models/Qwen3.6-35B-A3B-EXL3-2.49bpw \
  --tokens 32 --prompt-repeats 32 --output /tmp/audit-scatter.jsonl
# Refaire en processus séparé avec --reference-routing, dans l'ordre inverse
# sur les paires suivantes. Le fichier de sortie doit ne pas déjà exister.
```

Fichiers bruts : `benchmarks/results/audit-qwen-scatter-20260904.jsonl` et
`benchmarks/results/audit-qwen-reference-20260904.jsonl`.
Les poids ne sont pas hashés ; les métadonnées le sont. Temps GPU et état
thermique non instrumentés sont explicitement marqués non mesurés.

## Restant du rapport — état après le premier lot (historique)

- V02 : harness complet de continuation forcée comparant logits, routage et
  états récurrents à chaque étape ; corpus long et matrice de modèles.
- D02/D03 : groupement Q/K Gemma, épilogue expert pondéré et GeGLU 704/768.
  Ne pas réutiliser aveuglément une réduction FP16 si les scores sont FP32.
- D04 : compilation ShortConv LFM et étude de la rétention de vues de prefill.
  Modèle LFM absent localement, aucune performance réelle revendiquée.
- D05/D06/D08 : traces matérielles, fusion de la sortie Hadamard et primitive
  C++ seulement si l'occupation/les trous CPU mesurés le justifient.
- D07/D09/D10 : top-k, exposition des logprobs et attention longue ; préserver
  ordre, ties, NaN, RNG et arrondis. Ne pas remplacer le sampler par un argmax
  brut sous couvert d'équivalence stricte.
- P02–P05 : buckets de routage selon occupation, vues stridées des poids,
  autotuning TensorOps sur shapes retenues à part, gather de prefill fusionné.
- M03/M04 : comptabilité par propriétaire d'allocation et KV paginé/CoW :
  changements structurels non remplacés par une simple somme de nbytes.
- Manifeste/mesures : empreinte processus, traces GPU, poids hashés en option,
  intervalles de confiance et A/B multi-modèles longs à ajouter.

Aucun speculative decoding, nouveau KV quantifié, pruning, changement du
checkpoint ou prototype LUT/window/gather de la branche de recherche précédente
n'a été activé. Aucun binaire Desktop ni release n'a été remplacé par ce lot.

## Mise à jour du 5 septembre — livraison v0.3.2

Le lot suivant ajoute le cache des métadonnées immuables de détokénisation,
activé par défaut. Chaque génération garde un état de texte indépendant. Le
fallback est conservé pour les détokéniseurs inconnus ; le cache se désactive
avec `MLXL3_CACHED_DETOKENIZER=0`. Il ne modifie ni les tokens ni le sampler.

Qwen 35B-A3B, trois paires de processus alternées, warmup sur le même chemin
de préfixe : TTFT médian sans cache 290 → 191 ms, avec préfixe réutilisé
176 → 83 ms. Construction du détokéniseur environ 95 → moins de 1 ms.
Ce sont des temps muraux. Aucun gain global de decode n'est revendiqué.
Les résultats sont dans `benchmarks/results/audit-v2-detokenizer.jsonl`.

Les autres candidats sont présents pour reproduction, **désactivés par défaut** :

| Option | Contrôle et décision |
| --- | --- |
| `MLXL3_QK_GROUPS` | Gemma : écart maximal de logits 1.6171875 sur continuation forcée. Non conforme au contrat strict. |
| `MLXL3_GEMMA_EXPERT_REDUCE` | Scores FP32 préservés ; Gemma 16 pas logits/états identiques. Gain de decode non établi. |
| `MLXL3_GELU_DOWN_PREP` | Prototype GeGLU avec padding : sorties non finies sur Gemma réel. Ne pas activer pour utilisation. |
| `MLXL3_COMPILED_SHORTCONV` | LFM 64 pas logits/états identiques ; essai long synthétique 1000 pas. Pas de gain de decode stable. |
| `MLXL3_BITONIC_ROUTER` | Ties/NaN/Inf et Qwen 16 pas contrôlés ; pas de gain stable. |
| `MLXL3_QMV_OUTPUT_FUSION` | Qwen 16 pas exacts ; régression de vitesse dans les essais. |
| `MLXL3_STRIDED_PREFILL` | Vues de trellis sans copie sur TensorOps, K2/3/4/7 et 513 lignes testés ; Qwen exact. Gain global non établi. |
| `MLXL3_PREFILL_GATHER_HADAMARD` | Qwen 16 pas exacts ; prefill +8.9% médian mais dispersion forte. Reste expérimental. |
| `MLXL3_SEGMENTED_BUCKETS` | Branches GPU BM8/16/32 selon occupation, experts vides testés. Débit trop variable pour promotion. |
| `MLXL3_COW_KV` | Forks partageant le stockage, détachement à l'écriture et contrôle des snapshots. Pas de KV paginé ; baisse du pic global non établie. |
| `MLXL3_COMPACT_CONV_STATES` | Détache les petites vues convolutionnelles de leur grand parent ; mesure modèle interrompue avant achèvement. |

Autotuner hors ligne : `benchmarks/tune_tensorops.py`. Profils liés au matériel,
à MLX, à macOS et à l'ABI TensorOps. Activation exige `MLXL3_TENSOR_PROFILE`
et `MLXL3_ALLOW_Q1_TUNING=1` ; un test exact sur échantillons ne constitue pas
une preuve pour toutes les entrées. Le +34% mesuré sur une shape isolée ne
constitue pas un gain modèle. Les profils antérieurs sans ABI doivent être refaits.

Outils ajoutés : `compare_runtime.py` (paires alternées, données brutes et
intervalles bootstrap), `validate_forced_decode.py` (continuation imposée,
logits/dtypes/états complets, refus des non-finis), `capture_decode.py`
(capture Metal locale et profil CPU), `diagnose_gelu.py` (diagnostic uniquement).
Les artefacts détaillés avec états/logits/traces restent locaux, non publiés.
Les tests de performance V1 avaient un warmup différent du chemin mesuré :
ils sont archivés, pas utilisés pour annoncer un gain global. Le fichier
`audit-v2-fastsync.jsonl` est incomplet, arrêt demandé par l'utilisateur.

Les chantiers conditionnés aux compteurs matériels (D05, primitive C++ D08b,
nouvelle attention D10 et véritable pagination M04) ne sont pas livrés.
La capture Metal de contrôle fonctionne, mais Xcode complet/xctrace manque
pour l'analyse des compteurs. Aucune bande passante DRAM n'est inférée des
temps Python. Aucun speculative decoding ni nouvelle quantification.
