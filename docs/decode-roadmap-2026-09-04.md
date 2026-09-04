# Decode MLXL3 : pistes sans spéculation

Recherche du 4 septembre 2026, après fusion de la PR #12 dans `main`
(`470f936b09cd4b159fd1d558d7cc92271bb52727`). M5 de base, 24 Gio,
MLX 0.32.2, MLX-LM 0.32.0. Périmètre : génération interactive à un token
par passe, Gemma 4 26B-A4B, Qwen3.6 35B-A3B et LFM2.5 8B-A1B.
Pas de MTP, DFlash, draft model, n-gram speculation, requantification,
activation INT8, réduction du nombre d'experts ou attention approximative.

Ce document est une investigation, pas l'annonce de gains déjà implémentés.
Les nouveaux contrôles sont des inspections du code/des métadonnées et un
test de disponibilité SDPA sur le GPU local. Aucun moteur GUI n'a été remplacé.
LFM n'est plus présent localement : ses pistes ne sont pas benchmarkées ici.

## Résultat principal

Trois axes méritent un vrai chantier : **disposition mémoire EXL3 adaptée au
decode**, **épilogues MoE/Hadamard qui évitent les intermédiaires**, et
**attention Metal d=512 pour Gemma**. Le coût de construction/soumission des
graphes reste une quatrième piste commune. Un nouvel autotuner ne doit pas
simplement rejouer les variantes déjà perdantes.

### Ce qui existe déjà — ne pas le compter comme nouveau gain

- QMV direct sur trellis, sans reconstruction dense pour le decode normal.
- Codebooks spécialisés ; fenêtres de décodage K=3 pour les experts.
- Projections QKV, gate/up, Qwen qkv/z regroupées lorsqu'elles sont compatibles.
- Router top-k et réduction pondérée fusionnés pour Qwen/LFM.
- Préparation SwiGLU/down fusionnée pour les experts compatibles.
- Compilation des blocs récurrents Qwen ; MLP/router Gemma compilés.
- Prefix cache par blocs, LRU multiconversation et annulation coopérative.
- Génération MLX-LM avec `async_eval` : le GPU calcule déjà le prochain token
  pendant que le token précédent est traité côté CPU. Ce chevauchement ne
  prédit pas plusieurs tokens et n'est pas du speculative decoding.
- Cache attention glissant déjà circulaire en decode dans MLX-LM.
- Coalescence des deltas côté Swift déjà présente.

Sources locales : `src/mlxl3/{cli,linear,moe,recurrent}.py`,
`src/mlxl3/kernels/qmv.py`, `MLXL3Bridge.swift` et dépendances installées.

## 1. Gemma : manque de SDPA fusionnée d=512 confirmé

Le checkpoint a 30 couches : 25 attentions glissantes d=256, fenêtre 1024,
et 5 attentions globales d=512. Les globales ont 16 têtes Q / 2 têtes KV.
Test local de `mx.fast.scaled_dot_product_attention(..., force_fused=True)`
avec FP16, Q=(1,16,1,d), KV long de 1024 :

| Dimension | Résultat |
| --- | --- |
| 128 | noyau fusionné disponible |
| 256 | noyau fusionné disponible |
| 512 | erreur explicite : aucun noyau fusionné disponible |

Le [code MLX v0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp)
confirme la restriction. L'[issue #3885](https://github.com/ml-explore/mlx/issues/3885)
est fermée, mais cela ne signifie pas que notre version supporte d=512 ; ses
commentaires signalent aussi des régressions de variantes fusionnées selon
le GPU/GQA/contexte. Ne pas importer les chiffres M4 Max comme prédiction M5.

**Implémentation proposée :** noyau de decode `M=1, D=512`, FP16 avec
accumulations adaptées au calcul de référence. Lire les K/V par blocs,
réutiliser chaque bloc pour les têtes Q partageant une tête KV, calculer la
normalisation softmax et la sortie sans matrice de scores globale. Deux
variantes, courte et longue séquence, avec réduction de blocs si nécessaire.
Conserver `scale=1.0`, le RoPE partiel et les normalisations propres à Gemma.
On ne peut pas confondre K et V finaux : ils partent parfois de la même
projection mais subissent ensuite des traitements différents.

**Alternative conservatrice :** fusionner d'abord softmax et accumulation V
en conservant les scores QK et leur précision de référence. Cela économise
moins de mémoire mais facilite l'analyse des différences numériques.

**Réserve majeure :** une attention tiled/online est exacte au sens algorithmique,
mais son ordre de réduction peut différer. Exiger les contrôles de parité avant
activation ; ne pas la vendre comme bit-identique par construction.
Pas de promesse de gros gain à contexte court. À contexte long, cette piste
devient plus importante. Référence mathématique :
[FlashAttention](https://arxiv.org/abs/2205.14135).

## 2. Lire EXL3 dans une disposition faite pour nos kernels

Dans `_qmv_tile_kernel` / `_qmv_mapped_tile_kernel`, l'adresse contient
`(tile_k * TILES_N + tile_n) * PACKED_U32`. Chaque groupe parcourt K et ne
possède que quelques tuiles N. Les expert weights sont empilés dans N : les
groupes ne lisent que les experts choisis mais font de grands sauts d'adresse.
Cela ne prouve pas un mauvais débit DRAM : les autres groupes peuvent compléter
les accès adjacents, et les caches peuvent amortir la disposition actuelle.

**Piste prioritaire :** repacker uniquement les octets des tuiles, au chargement,
en `[expert/projection, groupe_N, tuile_K, N_local, mots]`. Pour les matrices
non-MoE, omettre expert/projection. Garder exactement les mêmes codewords,
scales, produits et ordre d'accumulation ; adapter seulement l'adressage.
Comparer cette disposition à l'originale, sans requantification ni entraînement.

- Débuter par le head et une couche d'experts, pas par tout le modèle.
- Vérifier inversion du repacking octet par octet et sorties exactes.
- Ne pas conserver deux copies complètes en RAM. Prévoir QMM compatible avec
  la nouvelle disposition ou limiter le prototype à quelques matrices.
- Mesurer aussi chargement, pic mémoire et prefill : un gain decode peut
  masquer un mauvais compromis global.

Le principe d'une disposition contiguë est également utilisé dans
[ExLlamaV3](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md),
mais sa mesure de swizzle concerne le CPU : elle ne quantifie aucun gain Metal.

### Autres changements du coeur QMV à tester

1. **Extraction par bitrate** : comparer mots 32 bits + shuffles, lectures
   directes et staging partagé. K=3 a déjà sa spécialisation ; viser surtout
   K=4/5/6. Tester des manipulations 32 bits équivalentes aux fenêtres `ulong`.
   Le compilateur peut déjà les produire : seule la mesure tranche.
2. **Autotuner limité et persistant** : clé GPU/version/kind/shape/bits/codebook,
   comparer tile N, déroulage, stratégie de chargement et split-K. Les variantes
   modifiant le découpage de la somme passent un contrôle numérique séparé.
3. **Pipeline de lecture** : précharger la prochaine petite fenêtre dans les
   registres pendant le calcul courant, sans changer la somme. Risque : plus
   de registres, donc moins de groupes résidents.
4. **Codebook lookup** : une table exacte de 65536 valeurs FP16 coûterait
   128 Kio. C'est un échange calcul contre accès irréguliers/cache, pas une
   amélioration évidente. Priorité basse ; garder le calcul MCg comme témoin.
5. **K=7** : le decode a déjà un kernel sérialisé, mais un chemin moins efficace
   et des groupes désactivés. Amélioration de couverture, pas priorité pour les
   checkpoints actuels dominés par d'autres bitrates.

[QTIP](https://arxiv.org/abs/2406.11235) fournit le contexte trellis/calcul versus
lookup. Les [compteurs M5 Apple](https://developer.apple.com/videos/play/tech-talks/111431/)
permettent de distinguer pression registres, pression L1 et attente mémoire.
Davantage de threads n'est donc pas automatiquement mieux.

## 3. Le head mérite un kernel dédié, sans shortlist approximative

Calcul à partir des `n_bytes` des trellis dans les deux fichiers locaux
`quantization_config.json` :

| Poids sérialisés | Gemma 26B-A4B | Qwen3.6 35B-A3B |
| --- | ---: | ---: |
| Head seul, 6 bits | 553 648 128 o | 381 419 520 o |
| Autres matrices EXL3 non-expert | 969 605 120 o | 702 545 920 o |
| Tous les experts | 10 796 138 496 o | 9 764 339 712 o |
| Experts actifs, moyenne uniforme | 674 758 656 o | 305 135 616 o |
| Total actif indicatif | 2 198 011 904 o | 1 389 101 056 o |

La moyenne utilise 8/128 et 8/256 ; ce n'est pas une trace de routage. Les
scales, poids non-EXL3, états et KV ne sont pas inclus. Ce sont des volumes
logiques de poids, pas des mesures de trafic DRAM : caches/relectures changent
le trafic réel. Le head représente environ 25% / 27% de ce volume indicatif,
pas nécessairement de la latence.

Actions : repacking ciblé ; kernel K=6/MCg grandes sorties ; tuning d'occupation
distinct des experts ; fusion de la transformation de sortie et de son scale.
Ensuite seulement, sampler spécialisé. Il faut calculer les 262144 / 248320
candidats, pas faire une projection grossière puis espérer que le bon token
soit dans un petit top-k. Ne pas reconstruire le head entier FP16 en cache :
cela augmente fortement les octets à lire.

## 4. MoE : gains encore accessibles autour des multiplications

### Gemma : trois trous précis dans notre intégration

1. `gemma4_text.Experts` appelle `switch_glu(x, indices)` puis multiplie et
   somme. Notre `EXL3SwitchGLU` sait aussi recevoir `scores`, mais Gemma ne lui
   transmet pas. **Adapter l'épilogue à Gemma** pour éviter cet intermédiaire.
   Ne pas réutiliser aveuglément la conversion FP16 des scores du helper actuel :
   préserver le dtype et l'ordre de réduction de Gemma.
2. `fuse_compatible_linear_groups` cherche Q/K/V tous présents. Les cinq couches
   K=V n'ont pas de `v_proj`, donc leur Q/K n'est pas regroupé. Ajouter un motif
   Q/K à deux projections, avec contrôle bits/codebook et absence de double-fusion.
3. Le router Gemma reste sur `argpartition`, gather, softmax des huit scores,
   puis scale par expert. Un kernel dédié peut les combiner, mais doit préserver
   les égalités, l'ordre des experts et les dtypes ; trier les experts autrement
   peut changer la somme finale.

Base à comparer :
[implémentation Gemma MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py).

### Tous MoE : gather des scales et fusions ciblées

Le chemin courant prépare `gu_suh[selected]`, `gu_svh[selected]`,
`down_suh[selected]`, `down_svh[selected]`, et une vue/broadcast des entrées.
Déplacer le gather des scales dans les kernels de transformation évite des
opérations intermédiaires. Une vue broadcast n'est pas forcément une copie :
compter les vrais buffers dans la trace avant de revendiquer une économie.
Ne pas partager un Hadamard entre experts ayant des `suh` différents.

Pour Gemma, créer **GeGLU + masque 704/768 + préparation down** fusionnés.
La fusion actuelle ne couvre que SiLU et exclut les dimensions logiques paddées.
Respecter les arrondis intermédiaires du Hadamard et de GELU. Ne pas supprimer
les 64 canaux sérialisés au motif qu'ils sont paddés : la rotation les mélange.

Piste suivante : sortie QMV par blocs de 128 + Hadamard/scale intégré. Les
groupes actuels possèdent 16–64 sorties ; agrandir peut augmenter la pression
registres ou réduire le parallélisme. Commencer sans split-K. Le
[prototype PonyExl3](https://github.com/beamivalice/PonyExl3/blob/master/ponyexl3/mlx/gemv_metal.py)
est instructif, pas une preuve de parité : distribuer un Hadamard arrondi sur
des sommes partielles n'est pas identique à transformer leur somme arrondie.

## 5. Qwen et LFM : viser les petits opérateurs encore séparés

**Qwen** : les blocs récurrents sont déjà compilés, et le noyau Gated Delta
packed existe déjà dans notre dépendance (`MLX_GDN_PACKED=1`). La suite utile
est une convolution decode spécialisée incluant concat/état/SiLU, puis une
fusion des normalisations Q/K avec la préparation récurrente qui conserve
les arrondis. Ne pas annoncer une nouvelle fusion récurrente sans comparer au
packed actuel. État FP32 théorique des 30 blocs : environ 62,9 Mo ; read+write
environ 125,8 Mo/token, hors autres opérateurs. Il n'explique donc pas à lui
seul le volume total de poids.

[ZMLX documente](https://github.com/Hmbown/ZMLX/blob/main/src/zmlx/patch/patterns/deltanet.py)
des fusions et leurs risques de dérive de RMSNorm ; copier leurs kernels sans
preuve de parité ne respecte pas notre contrainte. Une exécution périodique
de référence à partir d'un état déjà modifié ne rétablit pas magiquement l'état
qu'aurait produit toute la trajectoire de référence.

**LFM** : `ShortConv` enchaîne B*x, concat de l'état, convolution depthwise,
puis C*sortie. Faire un kernel un-token dédié et un état de taille fixe est
plus précis comme cible que « fusionner tout LFM ». La compilation récurrente
de MLXL3 ne couvre actuellement que Qwen ; tester un chemin compilé explicite
pour LFM avant de descendre plus bas. Garder les masques/biais et mettre à jour
correctement le cache. Notre top-k LFM est déjà fusionné.
Voir [ShortConv MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py).

## 6. Exécution et mémoire : pistes communes hors kernels mathématiques

- **Command buffers** : sweep contrôlé de `MLX_MAX_OPS_PER_BUFFER` et
  `MLX_MAX_MB_PER_BUFFER`, avant import MLX, sans modifier les valeurs globales
  du Mac. Les [réglages documentés](https://ml-explore.github.io/mlx/build/html/usage/environment_variables.html)
  peuvent réduire les soumissions, mais aussi retarder le travail, retenir
  davantage de temporaires et dégrader annulation/TTFT.
- **Primitive C++ multikernel** : si la trace montre un coût CPU significatif,
  regrouper la séquence EXL3 transform/QMV/épilogue en une primitive MLX, tout en
  utilisant son [command encoder et sa gestion des temporaires](https://ml-explore.github.io/mlx/build/html/dev/extensions.html).
  Ne pas contourner les dépendances de streams ni détruire un buffer en vol.
- **Plan d'exécution réutilisable** : distinguer compilation de graphe MLX et
  replay de commandes GPU. ExLlamaV3 dispose de blocs C++ capturés CUDA ; ce
  n'est pas un bouton portable. Les argument tables/allocateurs de
  [Metal 4](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
  donnent une piste structurelle, avec un coût d'intégration important.
- **Sampler et logprobs inutilisés** : MLX-LM calcule toujours logsumexp et un
  vecteur logprobs. Notre UI ne l'utilise pas. Mesurer ce coût sur grand vocab.
  Pour du greedy, argmax avant/après normalisation est équivalent en réel,
  mais les arrondis peuvent créer des égalités. Préserver leur sémantique ou
  garder la référence dans ces cas. Ne pas changer top-p/température/RNG.
- **Émission terminal/bridge** : les flushes Python restent synchrones. Comparer
  moteur seul, callback vide, terminal et bridge ; regrouper les deltas côté
  producteur avec latence bornée si cela aide. La coalescence Swift seule
  n'enlève pas le travail Python.
- **Scratch et cache mémoire** : profiler les allocations réelles. MLX possède
  déjà un allocateur cache ; une arène ne garantit rien. Préallouer seulement
  les buffers réellement recréés et coûteux. Les états partagés entre chats
  nécessitent du copy-on-write, jamais une mutation aveugle.
- **Contexte long** : garder les KV globales complets, optimiser leur croissance
  par capacité/blocs. Les 25 fenêtres Gemma représentent environ 209,7 Mo en
  FP16 à saturation ; les cinq couches globales ajoutent 20480 octets par
  position. Les fenêtres sont déjà circulaires : pas de faux nouveau gain.

## 7. Ordre d'attaque et preuves attendues

| Priorité | Travail | Ce qui décide de la suite |
| --- | --- | --- |
| P0 | Capture GPU courte, kernel labels, temps CPU/GPU séparés | Latence réellement dominante, registres/stalls |
| P1 | Gemma Q/K + épilogue MoE compatible dtype | Parité complète, gain CLI répété |
| P1 | Prototype repacking head + experts | Temps GPU et CLI ; pic RAM/prefill inchangés |
| P1 | SDPA d=512 custom Gemma | Gain surtout à 4k/16k/32k ; contrôle numérique |
| P2 | GeGLU/Hadamard/down + gather des scales | Gain de dispatch sans arrondis modifiés |
| P2 | Conv decode Qwen/LFM | Cache état identique sur longues générations |
| P2 | Paramètres de soumission / sampler / bridge | Gain end-to-end sans hausse TTFT/RAM |
| P3 | Primitive C++ / plan Metal réutilisable | Coût CPU prouvé suffisamment élevé |

Mesures : prompts identiques, alternance AB/BA, état thermique/alimentation,
plusieurs runs, tokens imposés pour le coût moteur et greedy réel pour la
trajectoire. Contextes court, 4k, 16k, 32k selon mémoire disponible ; sorties
longues pour détecter la dérive. Séparer TTFT/prefill du decode. Contrôler
logits, routes, états, caches et présence de NaN/Inf ; quelques tokens identiques
ne suffisent pas à prouver l'équivalence de toutes les entrées.

La [capture MLX/Metal](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
se fait avec `MTL_CAPTURE_ENABLED=1`, après warmup, sur peu de tokens. Ne pas
utiliser une capture pour annoncer le débit normal. Le profiler synchronisé
existant est utile pour localiser, mais ses 186 appels de transformée d'entrée
et 186 de sortie ne sont ni un compte exact de dispatchs ni une mesure GPU pure.

Les expériences précédentes restent dans `gemma-decode-investigation.md` :
compilation globale FFN et baisse des SIMD groups n'ont pas établi de gain
acceptable ; le Hadamard SIMD exact reste opt-in (+2,22% médian apparié dans
un test très variable). Cette recherche ne transforme pas ces résultats en
promesse de x2. Les sources et le code donnent maintenant des expériences
discriminantes, notamment un manque d=512 vérifié localement, plutôt que de
nouveaux réglages au hasard.
