# Inkling 2 Mini — revue du chemin de décodage (2026-11-22)

Revue en lecture seule de la source de `main` à `442625ec0` (moteur 2.0.24). Aucun code modifié, aucune
mesure lancée : les chiffres « mesuré » viennent du journal (`opti.md`, entrées citées), les autres sont des
estimations tirées des tailles de tenseurs et du compte d'opérations, et sont marqués comme tels.

Référence de vitesse (R3 / fichiers publiés, M5 10 cœurs GPU, contexte court) : 38,4–38,7 ms par token en
glouton au harnais (SPEED-INK2M-P12), ≈ 42–45 ms par token à T = 1 au bridge (DSPARK-INK2M-P10a, P12).

## 1. Où part le temps d'un pas

| Poste | Temps | Nature | Source |
|---|---:|---|---|
| Experts EXL3 (5 kernels × 48 visites) | ≈ 12,5 ms | calcul : 3,06 G poids décodés par token | P11 D1 |
| Projections denses 8 bits | ≈ 9,5 ms | bande passante : 1,29 Go | P8 (50,2 → 40,8 ms) |
| Petits kernels (≈ 4 500 par token) | ≈ 10–11 ms | ≈ 2,3 µs par kernel | déduit, cohérent avec P4 |
| Tête 8 bits | ≈ 3,3 ms | bande passante : 0,46 Go | P3 |
| Construction du graphe (CPU, GPU à l'arrêt) | 2,35–2,44 ms | CPU | P12 D2 |
| Sélection du token à T = 1 (CPU, GPU à l'arrêt) | ≈ +2 à +6 ms | CPU | écart bridge glouton / T = 1, jamais isolé |

Les experts décodent 245 G poids/s contre 136 G poids/s pour les denses 8 bits : les denses sont au plafond de
bande passante, les experts au plafond de calcul. Aucun des deux n'a de levier sans perte bon marché.

## 2. Le GPU attend le CPU à chaque token

### 2.1 Échantillonnage sur CPU sur tout le vocabulaire (mode par défaut de l'app)

- Le préréglage Desktop d'Inkling est T = 1, `topK: 0` (`SamplingPresets.swift:63`).
- `select_token` (`main.rs:3756`) passe par la relecture top-k, mais `certified_k` (`sampling.rs:134`) ne
  l'accepte que pour `top_k` entre 1 et 256. Avec `top_k` = 0 chaque token prend le repli complet :
  - `log_prob_rows` (`sampling.rs:536`) relit les 200 058 log-probabilités (800 Ko) et les recopie deux fois
    (`to_vec`, puis `clone` dans `topk_readback.rs:317`) ;
  - `Sampler::select` (`sampling.rs:170`) construit un `Vec<(usize, f32)>` de 3,2 Mo, calcule 200 058
    exponentielles en `f64`, puis deux autres vecteurs ;
  - `canonical` (`sampling.rs:616`) refait une paire de vecteurs et un tri par token.
- Pendant ce temps le GPU ne fait rien. Coût estimé 2 à 4 ms par token ; l'écart déjà mesuré au bridge entre
  AR glouton (≈ 38,8 ms) et AR T = 1 (≈ 42,4–44,6 ms) est de 3,6 à 5,8 ms (P10a), jamais attribué.
- Le même coût s'applique à chaque ligne vérifiée de DSpark (`speculative_accept_logits`,
  `sampling.rs:505` : toutes les lignes relues, une loi complète par ligne consultée).
- Piste sans changement de loi : tirage Gumbel-max sur GPU dans le graphe du pas, quand les pénalités sont
  neutres, `top_k` = 0 et `top_p` = 1 (exactement le préréglage). `gumbel_draw` existe et est testé par
  khi-deux (`inkling2_draft.rs:715`, DSPARK-INK2M-P3). Le flux aléatoire change, mais le bridge ignore déjà le
  champ `seed` et se sème sur l'horloge (`main.rs:5292`) : aucune reproductibilité n'est perdue.
- Variante produit, avec changement de loi : un `top_k` entre 1 et 256 dans le préréglage active le chemin
  rapide existant.

### 2.2 Tout le graphe est construit avant la moindre soumission

- `Inkling2::run` construit les 48 visites puis appelle `Array::eval_many` une fois (`inkling2.rs:2521`) :
  2,35–2,44 ms de CPU par token pendant lesquels le GPU est à l'arrêt (6 % du pas).
- La soumission progressive (`async_eval` toutes les quelques visites) a un précédent validé sur LFM3
  (OPT-2026-11-11-LFM3-UPSTREAM, +3,6 à +7 % en décodage). Jamais essayée sur Inkling : P9-C5 n'a évalué que
  le pipeline entre tokens.

### 2.3 Les kernels préparés de P5 ne sont pas sur `main`

- Seule la convolution courte passe par `array::prepared_kernel` (`inkling2.rs:801`).
- Les autres kernels personnalisés reformatent leur source et recopient des chaînes à chaque appel, soit 6
  appels par visite :
  - `routed_inputs_small` (`moe.rs:306`) : `format!` de la source ;
  - experts gate/up et down (`linear.rs:3216–3294`) : en-tête et nom reformatés, source de 11 Ko recopiée en
    `CString` (`array.rs:2142`) ;
  - `glu_down_input` (`moe.rs:519`) ;
  - routeur (`router.rs:288`) : deux sources formatées, dont une jetée quand le chemin parallèle est pris.
- P5-E1 avait mesuré −0,86 ms de construction par token (2,63 → 1,77 ms) en local, bit à bit. Le port sur la
  branche de release avait été écarté par prudence (clé de cache à compléter pour les kernels partagés).

### 2.4 Une seconde évaluation pour choisir le token

- Après `eval_many`, `chat_greedy_ids` (`array.rs:1808`) lance un second graphe (`log_probs`, `argmax`) et une
  seconde synchronisation. L'intégrer au graphe du pas en économise une par token (estimé 0,1–0,3 ms).

### 2.5 Inconnue principale : le coût d'encodage CPU dans `eval`

- Un pas compte 8 511 primitives, dont ≈ 4 850 kernels. MLX les encode sur le thread appelant.
- P4 a mesuré ≈ 0,8 µs de temps mur par cast ou vue retiré. Une vue n'a aucun kernel GPU : cela suggère que le
  pas est en partie limité par le CPU dans les zones de petits kernels (non prouvé).
- Si c'est le cas, un vrai pipeline (graphe du token suivant construit et encodé pendant que le GPU calcule le
  token courant) vaut bien plus que les 2,4 ms de construction.
- Diagnostic proposé avant tout choix : harnais en « teacher forcing » (les tokens d'un passage glouton connu
  sont rejoués, deux pas soumis d'avance par `async_eval`), empreinte des logits identique exigée, plus le
  temps CPU du thread pendant `eval`. Cela donne le plafond du pipeline sans écrire d'échantillonneur GPU.

## 3. Petits kernels : ≈ 4 500 par token, ≈ 10–11 ms

MLX insère une barrière dès qu'un kernel lit un tampon écrit depuis la précédente
(`device.cpp`, `set_input_array`), et il évalue en profondeur d'abord : les chaînes dépendantes se paient
kernel par kernel. Les kernels indépendants ne coûtent presque rien (P9-C3 : 144 kernels retirés, gain nul).

Inventaire par lecture du code (kernels par token, estimation ; temps à ≈ 2,3 µs) :

| Bloc | Kernels | Temps estimé | Où | État |
|---|---:|---:|---|---|
| Colle de l'attention (clips L2, casts, masque, porte de sortie) | ≈ 1 200 | ≈ 2,8 ms | `inkling2.rs:1305–1440` | P1 partiel |
| Portes de message (96 portes × 10) | ≈ 960 | ≈ 2,2 ms | `inkling2.rs:1615`, `1859` | P1 partiel |
| Attention en profondeur (96 + 24 messages) | ≈ 576 | ≈ 1,3 ms | `inkling2.rs:648–695` | non fusionnée |
| Normes, casts et résidus autour | ≈ 480 | ≈ 1,1 ms | `Layer::forward` | — |
| Routeur (cast, GEMV, sigmoïde, top-8, somme, division, échelle) | ≈ 336 | ≈ 0,8 ms | `inkling2.rs:1664`, `router.rs:409–421` | non fusionné |
| Combinaison des experts (`take`, cast, Hadamard, 2 produits, somme) | ≈ 288 | ≈ 0,7 ms | `moe.rs:609–632`, `1046` | non fusionnée |
| Convolutions dynamiques (8 visites pleines × 2 × ≈ 14) | ≈ 224 | ≈ 0,5 ms | `inkling2.rs:894–931` | non fusionnée |
| Convolutions statiques | 176 | ≈ 0,4 ms | `inkling2.rs:781` | fait (P4-S2) |
| Mises à jour K/V | ≈ 150 | ≈ 0,3 ms | `inkling2.rs:1499` | fait (P12) |

Ces estimations recoupent les ablations de P4 (portes 2,3 ms, attention hors projections 2,2 ms, profondeur
1,4 ms, convolutions 0,9 ms).

Candidats, du moins risqué au plus lourd (chacun sous la règle de l'empreinte bit à bit) :

1. **Combinaison des experts en un kernel** : opérations élément par élément, Hadamard 128 déjà reproduit au
   bit dans `routed_inputs_small`, somme de 8 termes. ≈ −240 kernels.
2. **Convolution dynamique en un kernel** : produits et additions FP32 dans l'ordre des taps, comme la
   convolution statique de P4. ≈ −200 kernels.
3. **Queue du routeur en un kernel** (sigmoïde, top-8, normalisation) et suppression de l'aller-retour
   FP32 → FP16 → FP32 avant le GEMV par un cast compilé. ≈ −240 kernels.
4. **Casts dans les clips L2 compilés** (q, k, r), masque (gather + puits + cast) en un kernel.
   ≈ −240 kernels.
5. **Messages d'une passe en un seul `depth_attend`** (48 requêtes au lieu de 12 appels de 4) et un seul
   `rms` pour les 4 messages d'une visite. ≈ −150 kernels ; identité au bit à vérifier (forme du matmul).
6. **Porte de message et attention en profondeur en kernels dédiés** : il faut reproduire l'ordre
   d'accumulation des GEMV de MLX. Gain le plus gros (≈ −900 kernels), risque le plus élevé.

Plafond si tout aboutit : ≈ −2 000 kernels, soit ≈ −4 à −5 ms (non mesuré).

## 4. Attention : chaque ligne K/V est lue trois fois

- Le kernel `sdpa_vector` de MLX 0.32.2 lance un threadgroup par tête de requête
  (`kv_head_idx = q_head / gqa_factor`, `sdpa_vector.h`). Inkling a 24 têtes de requête pour 8 têtes K/V :
  chaque ligne de clés et de valeurs est lue par 3 threadgroups.
- MLX ne bascule sur son kernel GQA en deux passes qu'à partir de 4 096 clés. Les 40 visites glissantes
  (≤ 1 024 clés) restent donc toujours sur le kernel qui lit trois fois.
- Estimation à ≥ 1 024 tokens de contexte : 40 × 4 Mo × 3 = 0,48 Go par token contre 0,16 Go en une lecture,
  soit jusqu'à ≈ 2 ms. La pente mesurée en place (38,99 → 40,80 ms de 256 à 1 024, P12) tombe entre les deux
  modèles : gain réel probable 1 à 2 ms, à confirmer par un microbenchmark.
- Piste : kernel vecteur qui traite les 3 têtes d'un groupe en lisant K/V une fois, avec les mêmes opérations
  par tête dans le même ordre.

## 5. Propre à la boucle

- Les passes 2–3 coûtent ≈ 18,8 ms (49 % du pas, P12 D1), dont ≈ 9 ms au-delà de la lecture de leurs poids.
- Leur part propre de petits kernels : portes (≈ 960) et messages (≈ 96), soit ≈ 2,5 ms. C'est la cible des
  points 5 et 6 ci-dessus.
- Calculs strictement répétés entre passes : l'entrée de profondeur de la couche 6 est identique aux trois
  passes (`pre_hidden` et `pre_blocks`), ≈ 12 kernels.

## 6. Autour du décodage

- **Aucun cache de prompt pour Inkling.** `prefill_round` ne restaure un préfixe que pour Qwen
  (`main.rs:2870`) et le chemin DSpark renvoie `cached = 0` (`main.rs:4209`). Chaque tour réévalue toute la
  conversation (`cached_prompt_tokens: 0` dans tous les rapports du bridge) : à ≈ 180 tok/s, 4 000 tokens
  d'historique font ≈ 22 s avant le premier token.
- **DSpark à T = 1 est au niveau de l'AR, pas au-dessus** (P10c). Par bloc : le draft est relu avant de
  construire la vérification (`inkling2_draft.rs:926`), puis vérification, choix sur CPU, et un `commit` qui
  resynchronise (`inkling2_draft.rs:971`). Le GPU attend entre chaque étape.
- **Le drafter recopie ses caches** à chaque ajout de contexte et à chaque proposition
  (`inkling2_draft.rs:189`, `220`), et ajoute son contexte token par token même quand aucun bloc n'est tiré
  (≈ 40 kernels par token en largeur 0).
- **Le champ `seed` est ignoré** (`main.rs:5292`) : les contrôles échantillonnés ne sont pas rejouables.
- `MaskPlan::new` (`inkling2.rs:956`) reconstruit et téléverse trois tableaux par type d'attention à chaque
  token ; le plan glissant est pourtant identique d'un token à l'autre une fois la fenêtre pleine. Coût
  faible aujourd'hui, linéaire en contexte.

## 7. Leviers avec perte (décision requise)

- Denses 6 bits : −25 % d'octets sur ≈ 9,5 ms si le kernel MLX 6 bits tient le débit du 8 bits (non mesuré) ;
  le coût en qualité n'a jamais été mesuré (4 bits : écarté en P3).
- `top_k` dans le préréglage (voir 2.1).

## 8. Vérifié, sans problème

- Le cast FP16 de `Linear::forward` est un no-op quand l'entrée est déjà FP16.
- Les crochets de capture (`capture_dense_input`, `capture_routes`) ne coûtent qu'un test en production.
- Le cache de fonctions du pont (`mlx_bridge.cpp:1037`) retrouve bien ses kernels.
- Le découpage en command buffers : les banques d'experts forcent des commits par le seuil en éléments, mais
  P9-C0 a déjà réduit le nombre de commits (seuil d'opérations de 40 à 400) sans effet mesurable.
- Ne pas refaire : budgets de command buffers (P9-C0), q|r|gate|sink fusionnés (P9-C3), kernel d'experts à
  première occurrence (P11-K1), NT/SG des experts (P5), denses 4 bits (P3).

## 9. Ordre proposé

1. Diagnostic « teacher forcing » + temps CPU dans `eval` (2.5) : il décide entre pipeline et fusions.
2. Tirage Gumbel sur GPU pour le préréglage (2.1) : seul gain sûr dans le mode réel de l'app.
3. Port des kernels préparés (2.3), puis soumission progressive ou pipeline selon le diagnostic (2.2).
4. Trio de fusions à faible risque : combinaison des experts, convolution dynamique, queue du routeur (3).
5. Microbenchmark GQA, puis kernel si le gain se confirme (4).
6. Portes et attention en profondeur (3, point 6).
7. Cache de prompt (6), pour le temps avant le premier token.
