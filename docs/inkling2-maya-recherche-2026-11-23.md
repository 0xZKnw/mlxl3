# Inkling 2 Mini : ce que fait Project Maya, et ce qui peut servir à MLXL3 (2026-11-23)

Recherche en lecture seule, demandée par l'utilisateur (« pour inkling 2 mini pour améliorer la vitesse, vas voir maya »).
**Aucun code modifié, aucune mesure, aucun prototype, aucun code repris.** Entrée de journal :
`RESEARCH-2026-11-23-INK2M-MAYA` dans `opti.md`.

## Sources lues

- Dépôt GitHub `mw00/project-maya`, branche `main`, commit `fd601f2988aaacc0d3e39feec41a765912810a9c`
  (2026-11-22, version 1.1.74, licence MIT), cloné en lecture seule hors du dépôt. Fichiers lus :
  `docs/INKLING2.md` (§ 6 et § 8, Inkling-2-Mini), `CHANGELOG.md` (v1.1.72 → v1.1.74),
  `src/ik2/ik2_main.cu` (`SpecCtl`, `sample_dist`, sortie `kCand`), `src/ik2/ik2_kernels.cu` (`cand_topk`).
  Rien n'a été compilé ni exécuté.
- Hugging Face `peasantsmith/Inkling-2-Mini-Maya-GGUF@7ab64bac7090c9f77df551ac0fac98119494a2a5` : contient seulement
  la tour de vision `mmproj` BF16. Le modèle utilisé par Maya est `unsloth/Inkling-2-Mini-GGUF` UD-Q4_K_XL
  (17,6 Go : experts des couches 0–19 en NVFP4 d'origine, le reste en Q8_0).
- Comparé à notre journal canonique (`opti.md` de `origin/main` `7060704bc`) : SPEED-INK2M-P1 à P17, DSPARK-INK2M-P0
  à P10c, RESEARCH-INK2M-EXO, RESEARCH-INK2M-DECODE-REVIEW, SPEED-INK2M-TOTAL, DECISION-INK2M-DSPARK-OFF.

## Ce que fait Maya pour Inkling-2-Mini

Moteur CUDA dédié (`maya-ik2`, ≈ 22 000 lignes dans `src/ik2/`), NVIDIA (sm_70 et plus) et AMD RDNA 3/4.

- **Chiffres publiés** : RTX 3090 (936 Go/s), UD-Q4_K_XL, cache FP8, 124,8 tok/s à 47 tokens de profondeur,
  120,7 à 4 096, 89,1 après 128 k ; llama.cpp b12590 sur la même carte : 57 tok/s. Échantillonné T = 1 : 110 tok/s.
  Prompt : ≈ 2 500 tok/s. Un token lit ≈ 3,7 Go de poids (moitié experts, moitié denses Q8_0).
- **Chemin d'un token entièrement sur GPU** : aucune synchronisation entre l'embedding et les logits, position lue
  sur le GPU, un graphe CUDA rejoué par token. Lancements réduits de 1 404 à 757 par token : convolutions courtes
  fusionnées dans les produits qui les précèdent, convolution du MoE dans la combinaison des experts, puits dans le
  lancement des clips, q/k/v d'une couche bouclée en un lancement, trois normes en un, quatre messages en un,
  routeur (choix + tables de pointeurs) en un, messages des 12 couches du cœur en un lancement par passe.
- **Experts NVFP4 sur les cœurs tenseurs** : un warp seul sur 16 lignes était plus lent que le noyau scalaire
  (il attend les octets d'un bloc avant de demander les suivants) ; huit warps sur les mêmes 16 lignes, chacun sur
  une tranche de la ligne, lisent 745 Go/s. L'entrée FP32 passe en deux « membres » FP16 pour rester exacte.
- **Attention d'un token** sur les cœurs tenseurs, directement depuis le cache vers les registres (846 Go/s FP16).
- **Échantillonnage** : candidats extraits sur GPU (`cand_topk`, histogramme puis seuil, ≈ 1 000 logits relus au
  lieu de 200 058). Avec `top_k = 0`, `sample_dist` tronque à ces 1 000 candidats : **la loi n'est pas exacte**.
- **Spéculatif** : drafter officiel DSpark + **recherche dans le prompt** (`suffix_drafter` : les tokens qui suivaient
  la dernière copie antérieure de la fin du texte, sans poids) ; un contrôleur `SpecCtl` choisit à chaque pas entre
  token simple, recherche et drafter, et la largeur, en maximisant (1 − aᴮ)/(1 − a) divisé par le temps mesuré du
  bloc (a = acceptation récente, décroissance 0,85 ; un essai toutes les 8 étapes simples). Chiffres (RTX 3090,
  glouton, contre décodage simple) : prose 131 / 127 tok/s, nouveau code 162 / 126, édition d'un document 213 / 124.
- **Fenêtre exacte** : la vérification d'un bloc donne à chaque token les valeurs du chemin d'un token **au bit**
  (produits denses avec les poids lus une fois mais la somme de chaque colonne du GEMV d'un token, experts groupés
  par expert sur le GPU, lus une fois pour les tokens qui les ont choisis : 5 tokens coûtent 2,7 fois les experts
  d'un token). Donc glouton identique avec et sans spéculation.
- **Cache de la conversation gardé pour le tour suivant** ; prompts courts (≤ 96 tokens, un tour de chat) : experts
  lus par groupes de tokens au lieu d'un GEMM, 12 nouveaux tokens en 56 ms au lieu de 118.
- **Cache K/V FP8** par défaut sur Mini (avec perte : KL 1,9·10⁻³ contre FP16 sur 199 tokens forcés ; FP16 reste
  le défaut sur Inkling-2-Small parce que FP8 change des choix du routeur).

## Mise en regard avec MLXL3

| Idée de Maya | État chez nous | Référence |
| --- | --- | --- |
| Token choisi sur GPU, pas de synchronisation, graphe rejoué | Fait autrement : tirage Gumbel exact sur GPU + soumission un pas d'avance ; le pas est limité par le GPU | P13, P14 |
| Échantillonnage sur candidats top-1000 | **Ne pas reprendre** : tronque la loi à `top_k = 0` ; notre Gumbel est exact et déjà sur GPU | P14 |
| Moins de lancements (portes, profondeur, routeur, masque, conv) | Fait en partie, sans perte : F1, F2, F3, F5 validés ; F4, F6 sous le seuil | P1, P4, P17 |
| q/k/v en un lancement, 12 messages en un, normes en un | Kernels indépendants déjà gratuits (pas limité par les chaînes dépendantes) | P9-C3, P13 |
| K/V lus à la vitesse de la mémoire | Plafond ≤ 0,48 ms sur les 40 visites glissantes | P16 |
| K/V sans recopie | Fait (en place, bit à bit) | P12 |
| Denses Q8_0 | Équivalent : denses affines 8 bits g64 par défaut | P8 |
| Cache K/V FP8 | **Ne pas reprendre** : avec perte | — |
| Experts groupés dans la fenêtre de vérification | Essayé, rejeté (occupation du GPU effondrée avec MB = T) | P11-K1 |
| Fenêtre exacte (spéculatif = AR au bit) | Absent : nos blocs DSpark ne sont pas identiques à l'AR (quasi-égalités) | DSPARK-P0/P2 |
| **Cache de conversation entre tours** | **Absent pour Inkling** : `PromptCache` ne gère que Qwen (`main.rs` `PromptCache::capture`) ; chaque tour et chaque tour d'outil du mode Agent re-préremplit toute la conversation | DECODE-REVIEW n° 7 |
| **Recherche dans le prompt + contrôleur par pas** | **Absent** (aucun drafter sans poids dans le moteur) | — |
| Routeur entier (choix + tables) en un kernel | Seul le produit du routeur est fusionné (F3) ; la queue (sigmoïde, biais, top-8, normalisation × 8) jamais lancée | P4-S5, P17-F3 |
| Convolution du MoE dans la combinaison des experts | Absent (`finish_and_reduce` puis convolution) | P4-S2 |
| Experts : plusieurs groupes sur les mêmes lignes pour avoir plus de lectures en vol | Non étudié ; experts ≈ 11,2 ms/token (≈ 103 Go/s contre ≈ 150 pour les denses) | P5, P17-D0 |

**Comparaison des débits (estimation, non mesurée)** : Maya ≈ 3,7 Go/token × 124,8 tok/s ≈ 460 Go/s, soit ≈ 49 % des
936 Go/s de la RTX 3090. Nous : ≈ 3,3–3,4 Go/token (denses 8 bits ≈ 1,38, experts R3 ≈ 1,27, tête 8 bits ≈ 0,49,
routeurs/portes ≈ 0,25) en ≈ 33,8 ms au harnais (un pas d'avance) ≈ 100 Go/s, soit ≈ 65 % des ≈ 153 Go/s du M5.
**L'écart de ×4–5 vient surtout de la bande passante du matériel**, pas d'une technique qui nous manquerait sur l'AR.

## Pistes nouvelles, par intérêt (aucune lancée ; chacune à préenregistrer)

1. **Cache de prompt pour Inkling** (temps jusqu'au premier token des tours 2+ et des tours d'outils). Aujourd'hui
   un tour après 8 192 tokens de conversation repaye ≈ 24–25 s de prefill (P12), ≈ 3,8 s à 1 200 tokens (P1).
   État à sauver : tampons K/V en place des 48 visites (8 pleines ≈ 32 Kio/token, 40 glissantes ≈ 160 Mio fixes),
   historiques des convolutions, logits de la dernière ligne. Le gabarit est stable d'un tour à l'autre (effort de
   réflexion émis une fois en tête, `reasoning_content` passé conservé), donc le préfixe de tokens se prolonge.
   Deux contrats possibles : (a) celui de Qwen — reprise seulement aux frontières de morceaux de prefill du prompt
   précédent, logits **identiques au bit** à un prefill à froid ; (b) reprise de tout l'état, réponse générée
   comprise (état exact de la génération, mais pas identique au bit à un prefill à froid). Gain attendu (estimé) :
   le prefill d'un tour se réduit à la réponse précédente + le nouveau message en (a), au nouveau message seul en (b).
   Attention au pas en vol du pipeline P14 (abandonné en fin de réponse : longueur logique à reculer, ce que P12 rend
   possible) et au budget mémoire (`MLXL3_PROMPT_CACHE_MIB`).
2. **Diagnostic du kernel d'experts EXL3** (11,2 ms/token, le plus gros poste restant). Maya a trouvé que son noyau
   était limité par la latence mémoire (trop peu de lectures en vol), pas par la bande passante. Chez nous, le
   décodage du treillis peut aussi être limité par le calcul. Microbenchmark proposé aux formes d'Inkling : routes
   réelles, routes toutes sur un même expert (données en cache), et copie pure du même volume. Si c'est la latence :
   préchargement de la tuile suivante avant de décoder la courante, **sans changer l'ordre des sommes** (bit à bit).
   Les variantes de Maya qui partagent une ligne entre groupes changent l'ordre des sommes : exclues. Ordre de
   grandeur si l'on atteignait ≈ 140 Go/s : −2 à −3 ms/token (estimé, non mesuré).
3. **Queue du routeur en un kernel** (sigmoïde, biais de sélection, top-8 avec le même départage que MLX,
   normalisation × 8, tables), comme Maya et Exo. Après F3, ≈ 1,2 ms de routeur estimé ; gain attendu −0,3 à
   −0,6 ms (estimé), sous la règle de l'empreinte bit à bit.
4. **Convolution du MoE dans la combinaison des experts** : un kernel dépendant de moins par visite (48 par token),
   −0,15 à −0,3 ms estimés ; F6 (analogue) est resté sous le seuil de 0,2 ms : priorité basse.
5. **Recherche dans le prompt + contrôleur par pas** (pas de l'AR, mais sans drafter ni RAM) : utile sur les copies
   (édition, réécriture, code cité, sorties d'outils du mode Agent), neutre en prose si le contrôleur reste en AR.
   Nos fenêtres de vérification coûtent 51 / 62 / 70 ms pour T = 2 / 3 / 4 (P11, sans les gains P14/P17 qui ne
   couvrent que les pas d'une ligne) contre ≈ 34 ms le pas AR : il faut ≥ 1,5–2 tokens acceptés par bloc. Exact en
   loi à T = 1 (`speculative_accept_logits` avec brouillon ponctuel) ; glouton non identique à l'AR sans fenêtre
   exacte. Décision utilisateur nécessaire au vu de DECISION-INK2M-DSPARK-OFF (« prochains boosts sur l'AR »).

## Limites

- Lecture de code et de documentation seulement ; les chiffres de Maya sont ceux de son auteur sur du matériel
  NVIDIA, non reproduits ici. Les gains indiqués pour MLXL3 sont des estimations, aucune n'est mesurée.
- Le modèle de Maya (UD-Q4_K_XL, experts NVFP4) n'est pas notre R3 EXL3 3,33 bpw : les leçons sur les experts ne se
  transfèrent pas telles quelles (décodage du treillis contre lecture directe de blocs FP4).
