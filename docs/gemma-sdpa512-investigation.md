# Gemma d=512 decode — expérimentation CLI

## Portée

Branche `codex/gemma4-sdpa512`, Apple M5 24 GiB, MLX 0.32.2.
Modèle local : Gemma 4 26B A4B EXL3, 3.54 bpw.
Desktop active désormais `grouped` au lancement de son moteur embarqué,
sauf si `MLXL3_GEMMA_SDPA512` est explicitement défini. Le CLI reste opt-in.

Les cinq couches globales de ce modèle utilisent 16 têtes Q, deux têtes KV
et une dimension de tête de 512. La version installée de MLX n'a pas de SDPA
fusionné pour cette dimension. Son fallback traite les têtes GQA comme des
lots de produits matrice-vecteur séparés.

Deux chemins expérimentaux réutilisent les KV entre les huit têtes Q :

- `grouped` : reformulation en matrices de huit lignes, kernels MLX natifs,
  graphe compilé sans spécialisation sur la longueur KV.
- `matrix` : kernel Metal QK dédié, matrices SIMD 8×8, puis softmax précis
  et produit probabilités-V groupé via MLX. Ce n'est pas un FlashAttention
  entièrement fusionné ; les scores sont encore matérialisés.

Le premier prototype à softmax en ligne était plus lent et a été retiré.
Le chemin groupé est préférable au custom dans les mesures réalisées.

## Garanties et limites

Aucune spéculation, aucun élagage du contexte, aucune quantification KV ou
modification des poids. Les scores et probabilités restent FP16, avec softmax
précis comme dans le fallback. L'ordre des réductions change cependant :
**pas de garantie d'identité bit à bit ni de sorties identiques sur tout prompt**.
Les tests de tenseurs vérifient une tolérance numérique, pas une preuve
d'équivalence stricte. Le CLI conserve son chemin de référence par défaut.
Desktop active le chemin groupé sur demande de l'utilisateur ; le kernel
custom reste optionnel.

Le routage n'affecte que les instances Gemma d=512 en inférence, avec GQA=8,
FP16, un token et au moins 2048 tokens KV. Les autres cas (masque explicite,
cache quantifié, sinks, prefill, entraînement) gardent le chemin amont.
Les modèles Qwen et LFM ne sont pas modifiés.

## Reproduction

Depuis le checkout et son environnement de développement :

```sh
MLXL3_GEMMA_SDPA512=grouped .venv/bin/mlxl3 run gemma-4-26B-A4B-it-exl3
MLXL3_GEMMA_SDPA512=matrix .venv/bin/mlxl3 run gemma-4-26B-A4B-it-exl3
```

Sans variable, ou avec `MLXL3_GEMMA_SDPA512=off`, le chemin normal est conservé.

```sh
.venv/bin/python benchmarks/benchmark_gemma_attention.py \
  '/Users/justin/Library/Application Support/io.mlxl3.desktop/Models/gemma-4-26B-A4B-it-exl3' \
  --context 16384 --tokens 128 --pairs 3
.venv/bin/python benchmarks/benchmark_sdpa512.py
.venv/bin/pytest tests/test_attention.py -q
```

Le benchmark modèle prépare un cache commun avec le chemin de référence,
le clone pour chaque essai et alterne l'ordre des variantes. Température zéro,
128 tokens, trois comparaisons ; le préremplissage n'est pas inclus dans le
débit decode. Le prompt répétitif sert à fixer la longueur de contexte et ne
constitue pas une évaluation de qualité. Les trois répétitions d'un même prompt
ne sont pas trois exemples indépendants.

## Premières mesures

Avant la compilation sans spécialisation KV :

| Contexte | Référence tok/s | Groupé tok/s | Metal custom tok/s |
|---|---:|---:|---:|
| 2048 | 40.80 | 40.86 | 39.82 |
| 16384 | 32.69 | 35.73 | 35.10 |

Médianes sur trois essais. À 16k, gains appariés médians de +9.37 % et +7.29 %.
À 2k, aucun gain significatif. Tous les essais produisaient les mêmes 128 tokens
que leur référence ; cela ne garantit pas l'identité sur d'autres générations.
Pic MLX du processus de benchmark : 15.14 GB à 16k (inclut le cache de référence
et ses clones, ce n'est pas la RAM d'un chat unique).

## Validation de la version finale

Après compilation sans spécialisation de longueur, nouveau passage 16k :

| Essai | Référence tok/s | Groupé tok/s | Metal custom tok/s |
|---|---:|---:|---:|
| 1 | 33.674 | 36.891 | 36.355 |
| 2 | 32.949 | 36.043 | 36.255 |
| 3 | 33.574 | 36.807 | 36.291 |

Gains appariés médians : **+9.56 % groupé**, **+8.09 % custom**.
128 tokens identiques dans chaque comparaison, pic MLX 15.144 GB.
Ce nouveau passage confirme le bénéfice à long contexte ; la différence entre
les deux passages ne permet pas d'attribuer un gain isolé à la compilation.

Voir aussi [les autres pistes decode](decode-roadmap-2026-09-04.md).
