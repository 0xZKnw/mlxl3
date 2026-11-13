# Exa → prefill : 20 essais bornés

Branche `codex/tool-prefill-rd`, référence main `357f23e` (v0.4.0).
Apple M5, 24 Gio, macOS 26.6.2, Qwen3.6-35B-A3B EXL3 2.49 bpw.
Pas de speculative decoding, nouvelle quantification, suppression de résultats
MCP, changement du sampler de production ou modification du prompt.

## Diagnostic

L'essai 1 est une vraie génération de l'appel outil, suivie du rejeu d'un
résultat Exa public capturé une fois (3 résultats, 13 916 caractères).
L'appel HTTP de capture dure 1,236 s. Le deuxième passage du modèle reçoit
4 692 tokens : **950 en cache, 3 742 à évaluer**, dont 3 737 en prefill stable.
Le cache réutilise donc bien le premier passage, raisonnement et appel compris.
La deuxième TTFT est 10,705 s, dont 10,305 s de prefill stable ; le template
et la tokenisation représentent moins de 8 ms. Le goulot est bien la lecture
du nouveau contexte, pas le réseau dans cette capture.

L'essai 1 a une limite de diagnostic de 768 tokens par passage et termine le
second passage sur cette limite. **La limite de l'app n'est pas modifiée.**
Les essais suivants rejouent exactement les mêmes tokens, hors réseau.

## Interface livrée

- Blocs chronologiques à identifiants stables : réflexion → outil → traitement
  des résultats → nouvelle réflexion → réponse. La deuxième réflexion ne se
  rattache plus au premier bloc.
- Pendant le prefill : libellé explicite, tokens nouveaux/en cache et compteur
  de secondes. Aucune fausse barre de progression GPU : les chunks stables
  sont soumis de manière asynchrone et ne fournissent pas de pourcentage terminé.
- Les marqueurs non-delta traversent la barrière existante du bridge, qui vide
  les fragments en attente avant de livrer le marqueur.
- Chronologie persistée, migration des anciens chats, arrêt du statut en cas
  d'annulation. L'ordre historique des vieux chats n'était pas enregistré :
  il ne peut pas être reconstitué a posteriori.
- Timer local à la ligne active, pas de reconstruction Markdown à chaque seconde.

## Protocole

`benchmarks/benchmark_tool_prefill.py` capture le schéma Exa et un résultat public,
puis construit un snapshot évalué unique des 950 tokens communs. Chaque variante
part d'un fork de ce même snapshot. Ce setup est exclu des temps, comme l'appel
outil déjà terminé dans l'app. Les essais mesurent `_stream_response`, template,
tokenisation, prefill, fork et premier token inclus. La synchronisation/hachage
de validation intervient **après** le chronométrage.

16 tokens de continuation sont imposés, avec le même token de lookahead MLX-LM.
Les contrôles comparent les octets des logits finaux, des tableaux des caches
et leurs métadonnées. Ce n'est pas une comparaison de tous les logits
intermédiaires ni une preuve sur toutes les entrées/modèles.
Les warmups sont marqués séparément (numéro 0), hors des 20 essais.

Le Mac est resté sur batterie. `pmset` et le manifeste sont enregistrés. Aucune
alerte thermique n'a été rapportée, mais **les temps de référence ont beaucoup
varié**. Les compteurs GPU/fréquences ne sont pas instrumentés : ne pas attribuer
la baisse de 10 à 6 secondes au code et ne pas extrapoler en bande passante DRAM.

## Mesures brutes

TTFT en secondes ; débit en tokens/s ; pic MLX en GB décimaux (pas RSS totale).
« Exact » = contrôle final logits + cache sur continuation imposée.

| Essai | Variante | TTFT | Prefill | Pic | Exact |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Boucle outil réelle, deuxième passage | 10.705 | 357.1 | 13.99 | Diagnostic |
| 2 | Référence | 9.298 | 405.6 | 14.08 | oui |
| 3 | Chunk 512 | 12.049 | 311.3 | 13.28 | non |
| 4 | Référence | 9.586 | 393.1 | 14.08 | oui |
| 5 | Chunk 1024 | 10.490 | 359.1 | 13.47 | non |
| 6 | Chunk 4096 sans frontière de snapshot | 8.880 | 427.5 | 15.06 | non |
| 7 | Référence | 9.775 | 385.7 | 14.08 | oui |
| 8 | Ancien gather/Hadamard fusionné | 9.631 | 393.4 | 14.16 | oui |
| 9 | Vues stridées | 9.666 | 390.0 | 14.08 | oui |
| 10 | États conv compacts | 9.593 | 392.7 | 14.08 | oui |
| 11 | Conservation du cache allocateur | 9.505 | 397.2 | 14.08 | oui |
| 12 | Référence | 9.487 | 397.1 | 14.08 | oui |
| 13 | Nouveau gather SIMD | 6.098 | 616.2 | 14.16 | oui |
| 14 | Localité des dispatchs experts | 6.650 | 565.4 | 14.08 | oui |
| 15 | SIMD + localité | 6.114 | 615.5 | 14.16 | oui |
| 16 | Référence | 6.469 | 582.0 | 14.08 | oui |
| 17 | Référence | 8.276 | 471.5 | 14.08 | oui |
| 18 | SIMD | 6.083 | 617.3 | 14.16 | oui |
| 19 | SIMD | 6.129 | 614.6 | 14.16 | oui |
| 20 | Référence | 6.475 | 581.2 | 14.08 | oui |

## Décision

**Promu : gather + multiplication + Hadamard dans un seul SIMD group de 32
threads**, sans tableaux temporaires d'activations routées/scales, mémoire
threadgroup ni barrières. Les étapes radix-16 puis radix-8 et les arrondis FP16
intermédiaires de MLX restent identiques. Pas de modification du routage, de
l'ordre des réductions, des poids ou des scores.

La dernière paire adjacente 19/20 donne 6,129 contre 6,475 s : **−5,3 % de
TTFT**, et 614,6 contre 581,2 tok/s : **+5,7 % de débit prefill**. Les essais
13 et 18 sont cohérents avec cela. Les références 16 et 20 sont proches, mais
17 est plus lent : série courte et non stationnaire, pas d'intervalle de
confiance ni revendication d'un gain universel. Ne pas utiliser la médiane
17/20 pour gonfler artificiellement le gain.

Le pic augmente d'environ **81 Mo** (14,08 → 14,16 GB) malgré la suppression
d'intermédiaires, du fait des durées de vie/allocations du graphe. Ce lot ne
revendique donc **aucune baisse du pic mémoire** ni gain de decode.

Activation sur le chemin MoE SwiGLU seulement (Qwen validé). Gemma/GeGLU garde
le chemin existant. Désactivation : `MLXL3_PREFILL_GATHER_SIMD=0`.
Tests unitaires : tailles 128/256/2048, activations FP16/FP32/BF16, routes
répétées, scales signées, zéros et sous-normaux.

Les chunks alternatifs modifient l'arithmétique du prefill et ne satisfont pas
le contrôle strict : non promus. Les autres options n'ont pas de gain établi.
`MLXL3_SEGMENTED_LOCALITY=1` et `MLXL3_RETAIN_PREFILL_ALLOCATOR=1` restent des
expériences désactivées. Les options préexistantes restent inchangées.

## Reproduction et limites

```sh
.venv/bin/python benchmarks/benchmark_tool_prefill.py --capture
.venv/bin/python benchmarks/benchmark_tool_prefill.py --real-loop \
  --output benchmarks/results/tool_prefill_rd_01.jsonl
.venv/bin/python benchmarks/benchmark_tool_prefill.py \
  --runs 17:baseline,18:simd,19:simd,20:baseline \
  --output benchmarks/results/tool_prefill_rd_17_20.jsonl
```

Utiliser de nouveaux fichiers de sortie : les journaux sont append-only.
La capture publique locale est ignorée par Git (`benchmarks/data/`). Une
nouvelle capture réseau peut changer : le SHA256 du prompt est enregistré
dans chaque résultat. L'empreinte de cette série est
`58f6b8db722ce9f443d620aeefa498daff75b05e11e35e73ee29d0da1622a977`.
Les résultats sont dans les quatre fichiers `benchmarks/results/tool_prefill_rd_*.jsonl`.

347 tests Python réussis, 2 ignorés (fixtures locales absentes). Contrôles Swift
sans modèle : `--check-chat-timeline`, `--check-mcp-preferences`.
Le schéma de tests suit le skill inference-engineering : séparation réseau /
template / GPU, entrée figée, référence répétée, refus des candidats non exacts.

Sources primaires consultées : [génération MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py),
[cache MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py),
[évaluation différée MLX](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html),
[clear_cache](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clear_cache.html).
Le comportement effectivement utilisé a été vérifié dans la version locale.
