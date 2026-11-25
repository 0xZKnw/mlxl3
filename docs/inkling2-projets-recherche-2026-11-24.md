# Inkling 2 Mini — pistes tirées des PR et issues des autres projets (2026-11-24)

Entrée du journal : `RESEARCH-2026-11-24-INK2M-PROJECTS` (`opti.md`). Lecture seule, rien exécuté :
quatre recherches parallèles (sous-agents en lecture seule, `gh api` / pages web) sur Maya, MTPLX et
l'écosystème MLX, llama.cpp / Ollama / ExLlamaV3, vLLM / SGLang / Hugging Face. Chiffres = ceux des
auteurs, non reproduits. Notes brutes dans le bloc-notes de session (temporaire, non conservé :
`research-{maya,mlx,cpp,serve}/`). [lu] = lu dans le code ou le texte d'une PR ; [inféré] = déduction.

## Ce qui confirme nos résultats

- **Pas de vérification MoE 2–8 lignes « presque gratuite » à 128 experts top-8** : exllamav3 `58d4d7322`
  (MMA m16n8k16 par groupe de routes d'un expert, coût quasi linéaire : 20,2 / 55,1 / 113,1 µs à bsz
  1 / 4 / 8, RTX 5090) ; llama.cpp #30047 (MUL_MAT_ID ×9,2 à n = 16 sur M4) ; Sushi
  (`beamivalice/sushi`, EXL3 sur Metal) : GEMV d'experts limité par les FMA et le chemin d'entrée, la
  déduplication des experts ne récupère que 4–7 % à 3–4 lignes (= notre B rejeté) ; Delphi n'active ses
  experts tensoriels qu'à partir de 24 lignes et avec perte (= notre C1). [lu]
- **Acceptation du drafter officiel par contenu** : llama.cpp #32773 (code 68–81 %, raisonnement
  30–55 %, prose FR 9–18 %, prose EN 18–21 %) ; Maya (code 0,78 en 1ʳᵉ position) ; vLLM #69081 (2,3 / 2,7
  / 3,7 tokens par pas en prose / explicatif / code) ; HF #36 (K = 4, T = 0 : prose −4 %, code +128 %).
  Nos 0,48–0,60 en FR (1ʳᵉ position) sont plutôt hauts. [lu]
- **Conventions DSpark** (taps aux visites 5 / 23 / 41 + sortie normée, fusion `fuse_weight` +
  `dflare_norm`, K du contexte `k_norm_ctx` + RoPE θ 1e6, V non normé, ancre `embed_norm(embed)` + 7
  `mask_embed`, masque bidirectionnel, couches 0 et 2 à fenêtre 1 024 ancrée au début du bloc, couche 1
  globale, tête Markov à état nul, softcap 30, ids ≥ 200 058 masqués, bloc toujours calculé sur 8 lignes) :
  cohérentes avec notre implémentation validée contre la référence mlx-vlm (P0/P1) ; mlx-serve #1296 (couche
  globale traitée comme fenêtrée : acceptation 25 → 34 % à 23 k) est déjà géré chez nous
  (`inkling2_draft.rs`, `local_layer_ids`). Point à confirmer : SGLang/vLLM décrivent chaque tap comme le
  `partial` de l'AttnRes (somme attn + MLP depuis la dernière frontière de bloc de 6 couches). [lu / à
  vérifier]

## Pistes nouvelles (aucune lancée ; chacune à préenregistrer ; gains estimés, non mesurés)

| # | Piste | Source | Exactitude | Estimation chez nous |
|---|---|---|---|---|
| R1 | **Drafter DFlash2 réentraîné** `incoai/Inkling-2-Mini-DFlash2` (990 Mo BF16, 2026-11-24) : longueur acceptée 4,80–4,97 contre 3,47–3,66 DSpark (code/maths, T = 1), 3,90 contre 2,95 (MT-Bench) ; +27–40 % à concurrence 1 (H200/SGLang) | carte HF, sglang | exacte (vérification standard) | plus gros levier d'acceptation trouvé ; architecture différente (sélecteur de chemin, convolutions dynamiques) ; nous avons une infrastructure DFlash2 pour Qwen ; licence et format à vérifier ; aucune mesure sur Mac |
| R2 | **Rejet standard avec q complet** (propositions tirées de q, min(1, p/q), résidu) : optimal pour une position (Σ min(p, q) = 1 − TV ≥ couplage) ; pièges corrigés ailleurs : q stocké tel qu'échantillonné en FP32 (sglang #42710), q > 1 par arrondi toujours rejeté (#45819), résidu nul en FP32 (vllm #65176), comparaison en log (#51692) | vLLM / SGLang | exacte | notre P3 n'avait gagné que ≈ 1 point (bridge, FR) ; **mesurer d'abord Σ min(p, q) dans D2-DIAG** (même texte, sans code de production) ; si ≫ 0,563 (couplé), décision P3 sur GPU |
| R3 | **Tête du drafter sur un sous-vocabulaire** (≈ 32 768 tokens fréquents + spéciaux) : MTPLX `fast/draft_table.py` (couvre 99,3–100 % code et prose, 1/6 des octets), Delphi (draft 1,9 → 0,70 ms, M5 Pro), exllamav3 `47d42fead` | MTPLX, Delphi, exllamav3 | exacte (seules les propositions changent) | tête de base 8 bits ≈ 0,46 Go et repli Markov FP16 103 Mo par proposition : ≈ −3 à −4 ms sur ≈ 8 ms de drafter par bloc ; couverture FR à mesurer ; ids tiktoken ≈ rang de fusion ≈ fréquence |
| R4 | **Recherche pilotée par la longueur du match** : acceptance suivie par classe de longueur (4 classes), nombre de propositions selon la profondeur (7 si ≥ 8, sinon 3), un match ≥ 16 prend la suite, un match ≥ 3 prolonge le bloc du drafter là où ils concordent (+2,4 à +6 %) | Maya `draft_policy.hpp`, `4cee820e79`, #1166 (ouverte) | exacte (politique) | suite directe de D5 (variabilité observée : 58 à 140 blocs de recherche selon le texte tiré) |
| R5 | **Largeur selon la confiance du drafter** (q_max par position, seuil = coût d'une ligne × débit récent, borné ; drafter surconfiant) : TensorFold (Inkling 2 Small, M5 Ultra : 57,1 → 61,4 tok/s), megakernel-zoo, estimateur SGLang (q_max seulement : exact) | MLX, SGLang | exacte si la décision ne dépend que du drafter | une ligne coûte 35 % d'un pas chez nous (15–24 % sur M5 Max/Pro) : le filtrage vaut plus ; coût : lire q_max avant la vérification (perd une partie du recouvrement A3) |
| R6 | **Produit « panel » des denses et de la tête à 2–8 lignes** reproduisant l'arithmétique du `qmv` à une ligne (poids lus une fois) : MTPLX `fast/qproduct.py` (tête 201 024 lignes à 4 lignes 856 µs contre 1 474 µs pour 4 passes ; sortie à 2 lignes 19,2 → 10,3 µs, M5 Max) ; llama.cpp Metal #29869 (dense 2–16 lignes sur simdgroup 8×8, activé sur M5) | MTPLX, llama.cpp | identique au bit au pas AR pour ces produits (MTPLX) | borne : la part hors experts d'une ligne vérifiée (≈ 23 % de ≈ 11–12,5 ms, P11-D1) ≈ −2 à −2,5 ms par ligne ; rendrait aussi la vérification ≡ AR pour les denses |
| R7 | **Entrée EXL3 préparée une fois par (ligne, expert)** (`suh·H128(x)` et le milieu SwiGLU dans un dispatch dédié, rangés dans l'ordre des voies) : 0,885× à 3 lignes, 0,881× à 4, octets identiques (M1 Max) | Sushi PR #120 | identique au bit | à auditer : notre `routed_inputs_small` / `glu_down_input` le font peut-être déjà ; le gain viendrait de la disposition lue par le GEMV |
| R8 | **Coût par largeur enregistré seulement si la largeur ne change pas d'un tour à l'autre**, effet de chaque échantillon borné (Ollama `speculate_depth.go`) ; premier pas d'une largeur ignoré (Maya) | Ollama, Maya | exacte (politique) | vise nos sondes de largeur 1 à 81 ms (A1F) ; famille D3 |

Également noté : MLX ≥ 0.32.4 non publié (#4781 `affine_qmv_nax` M = 6–16 : 8 bits ×1,14 à M = 6, ×1,28 à
M = 8 sur M5 Pro ; rien pour M = 2–5) ; la vérification par blocs (Sun et al. 2403.10444, sglang #42297 /
#43012) n'aide pas avec des propositions ponctuelles et peu à largeur ≤ 3 ; un drafter à 0,5 × la
température de la cible (exllamav3 #423, règle standard) ; bug Ollama #19843 (vérification de 8 tokens à
4 k de contexte différente du décodage un par un ; NaN dans `route_topk` quand tous les scores valent −100) :
nos oracles D5 (rejeu AR : seulement des quasi-égalités communes à DSpark) couvrent ≈ 300 tokens de
contexte, pas 4 k ; MTPLX #710 (décalage d'une ligne des visites bouclées à la frontière prefill/décodage) :
nos oracles D1/D5 le détecteraient. Variantes inexactes écartées (relaxed acceptance, FLy).

## Ordre conseillé

1. **R2-mesure** (Σ min(p, q) ajouté à D2-DIAG : quelques lignes de test, une exécution) et **R4**
   (politique, suite de D5).
2. **R3** (sous-vocabulaire du drafter) et **R8** (coûts).
3. **R1** (DFlash2) : étude de l'architecture et de la licence avant tout portage.
4. **R6 / R7** : audit puis prototype de noyau, identiques au bit.
