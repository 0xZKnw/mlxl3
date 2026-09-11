# Prefill EXL3 segmenté — 11 septembre 2026

Deux optimisations du kernel partagé MoE, sans changer le modèle, les poids,
la précision, le sampling, la taille des chunks ou l'ordre des réductions :

- Tuiles TensorOps de 64 lignes à partir de 4 096 lignes routées ; 32 en dessous.
  Chaque déquantification sert davantage de lignes. Ce n'est pas une règle
  propre à LFM ou Qwen. Le chemin TensorOps reste limité aux GPU M5 supportés.
- Calcul des adresses/permutations/shifts EXL3 une fois avant la boucle K,
  au lieu de les recalculer à chaque tuile BK16. Fallback conservé si la capacité
  du tenseur coopératif dépasse l'espace prévu par lane.

Le correctif de compilation macOS 27 utilise des types d'opérandes explicites
et retire le paramètre template `T` inutilisé du kernel segmenté. Il est appliqué
aux **deux** côtés de la comparaison ; sa restauration de TensorOps n'est pas
comptée comme un gain d'optimisation.

## Mesures bout en bout

Apple M5, 24 GiB, macOS 27.0 build 26A428, MLX 0.32.2, MLX-LM 0.32.0.
Batterie, thermique non contrôlée. Une paire complète d'échauffement exclue,
puis trois paires alternées AB/BA. Nouvelle session pour chaque requête.
Pourcentages : médianes des variations **appariées**, pas sommes de microbenchmarks.

| Modèle / entrée | Tokens entrée / sortie | Prefill | Réduction TTFT | Decode |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 35B A3B 2.49 bpw / phrase répétée | 4 106 / 96 | +12,35 % | 11,02 % | −0,48 % |
| LFM2.5 8B A1B 3.10 bpw / phrase répétée | 4 106 / 96 | +34,86 % | 25,74 % | +1,55 % |
| Qwen / prompt court | 138 / 32 | +8,95 % | 7,98 % | −1,31 % |
| LFM8 / prompt court | 138 / 32 | +16,14 % | 12,94 % | −2,49 % |
| Qwen / document technique non répétitif | 2 842 / 32 | +9,37 % | 8,63 % | −0,62 % |
| LFM8 / document technique non répétitif | 2 746 / 32 | +25,59 % | 20,40 % | −3,79 % |

Le pic MLX reste pratiquement identique (~14,05 GB Qwen et ~4,92 GB LFM8
dans le test 4k). L'empreinte physique macOS ne baisse pas ; quelques MB
supplémentaires selon la série. **Pas de boost RAM ni decode revendiqué.**
Les baisses decode sur certains échantillons courts sont conservées :
leur origine thermique/ordonnancement n'est pas isolée.

Les textes générés, les états des préfixes stables et les caches finaux sont
identiques bit à bit dans toutes ces comparaisons. Les tests numériques couvrent
les trois codebooks, 1 à 8 bits, les experts vides et les fins de blocs ;
la garde de capacité et le dispatch automatique disposent aussi de tests.
177 tests passent, plus 8 contrôles exécutés avec TensorOps segmenté désactivé.
Ce n'est pas une preuve de parité universelle sur tous les modèles/OS.

## Reproduction et état

Les mesures et empreintes sont conservées dans
[`prefill-segmented-20260911.json`](../benchmarks/results/prefill-segmented-20260911.json).
Le journal [`opti.md`](../opti.md) conserve aussi pilotes, échecs et ablations.

```sh
.venv/bin/python benchmarks/bench_prefill_flight.py \
  models/fixtures/LFM2.5-8B-A1B-EXL3-3.10bpw \
  --segmented-m64 --segmented-hoist --repeats 512 \
  --warmup-pairs 1 --pairs 3 --max-tokens 96
```

Pour le holdout : remplacer `--repeats 512` par
`--prompt-file docs/audit-runtime-2026-09-04.md` et utiliser 32 tokens générés.
Le runner compare BM32 sans hoisting à l'auto-tile avec hoisting de production.
Rollback diagnostic : `MLXL3_SEGMENTED_TENSOR_ROWS=32 MLXL3_SEGMENTED_ADDRESS_HOIST=0`.

Code moteur local, utilisé par le CLI de ce dépôt. Pas de reconstruction de
l'app installée, de push ni de release. Pas de refonte/changement de langage.
Gemma, les autres Mac/OS et une comparaison équitable à llama.cpp restent
non mesurés ; aucune supériorité sur llama.cpp n'est annoncée.
