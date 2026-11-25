# Inkling 2 Mini — ce que Maya v1.1.75–1.1.78 apporte pour la vitesse et DSpark (2026-11-24)

Entrée du journal : `RESEARCH-2026-11-24-INK2M-MAYA2` (`opti.md`). Lecture seule : rien exécuté,
compilé ni mesuré, aucun code repris. Répétition datée de `RESEARCH-2026-11-23-INK2M-MAYA`
(Maya v1.1.74, `fd601f29`), justifiée par quatre versions de Maya publiées depuis, dont le
drafting d'Inkling 2 Mini (#1154).

## Sources

- GitHub `mw00/project-maya` `main` `1016876a387e` (v1.1.78, 2026-11-24, MIT), clone superficiel
  temporaire dans le bloc-notes de session (non conservé) : `CHANGELOG.md` v1.1.75–1.1.78,
  `docs/INKLING2.md` § 8, `src/ik2/ik2_draft.{hpp,cu}`, `src/ik2/ik2_main.cu` (`SpecCtl`,
  `gumbel_pick`, `spec_step`), `src/ik2/ik2_kernels.cuh` (`gumbel_noise`).
- Notre code : `origin/main` `6d1b8df16a60` (moteur 2.0.31, `native/src/main.rs`
  `prefill_round` et branche DSpark du bridge) ; branche `claude/inkling2-dspark-fast`
  `00b8e97e9` (DSpark A + A3, poussée, **non fusionnée** dans `main`) : `native/src/inkling2_draft.rs`
  (`WidthPolicy`, `propose_with`), `native/src/inkling2.rs` (`DrawSeeds`, `decide_block`),
  `native/src/moe.rs` (`expert_segmented_qmm`), `native/shaders/_qmv_mapped_tile_kernel.metal`,
  `native/shaders/_expert_qmm_tensor_kernel.metal`, `native/src/linear.rs` (`codebook_header`).
- Journal relu : P1, P3, P10, P11, P14, PC/LK (SPEED-…-MAYA), DECISION-LOOKUP-ON,
  AGENT-…-VITESSE, RESEARCH-…-DRAFTERS/MTPLX, DSPARK-…-MTPLX-A/A3/B0/B/A1F/C0, TENSOR-C1.

## Ce que Maya a changé (chiffres de l'auteur, RTX 3090, UD-Q4_K_XL, non reproduits)

- **#1154 (v1.1.75, @needmorevram)** : drafter DSpark officiel en GGUF Q8_0 ; pas de fenêtre de 5
  lignes 25,5 → 17,6 ms (propositions 2,5 → 1,7 ; fenêtre 22,2 → 15,7 ; reste 0,8 → 0,15), logits
  de chaque token identiques au bit au chemin à un token. Glouton, drafting / sans : code neuf
  201 / 124 tok/s, appel d'outil 201 / 124, réécriture 210 / 124, appel d'outil de 900 tokens
  237 / 124, prose 135 / 124. Pas simple ≈ 8,06 ms → fenêtre de 5 ≈ 1,95 pas, pas complet ≈ 2,18.
- **Couplage du drafter et de la cible (T > 0)** : le token de la cible à la position i est
  `argmax(log p_i / T + g(salt, i, token))` (Gumbel-max, bruit fonction du sel de la requête, de la
  position et du token) ; le drafter tire sa proposition de la même position avec **le même bruit**
  sur ses propres logits ; la proposition est gardée si elle est égale au token de la cible. Loi de la
  cible exacte quelle que soit la proposition. Code neuf à T = 1 : **58 → 68 %** de propositions
  gardées (commentaire du code : proposition = meilleur token du drafter, gardée avec la probabilité
  p(x) : 60 %, glouton : 65 %). `IK2_DRAFT_COUPLE=0` revient à la règle ponctuelle.
- **Contrôleur `SpecCtl`** : acceptation par position conditionnelle (code : 0,78 en tête, 0,8–0,9
  ensuite), **coût de chaque largeur mesuré en ligne** (moyenne glissante 0,8 / 0,2, premier pas de
  chaque largeur ignoré, largeurs non mesurées au prorata des mesurées), jusqu'à 8 de large ;
  **garde par profondeur** : en contexte long, les rounds sont chronométrés contre les pas simples
  à cette profondeur et le drafting s'arrête tant qu'il perd.
- **`--drafting auto`** : mesure une fois par PC aucun drafting / recherche dans le prompt / drafter
  / les deux, garde le plus rapide ; en mode `on`, drafter et recherche dans le même contrôleur.
- **Fenêtre exacte** : denses de la fenêtre (`mv_q8p_cols`, poids lus une fois, ordre des
  multiplications-additions du chemin à un token), attention des couches sans fenêtre glissante
  (requêtes côte à côte sur une ligne de clés lue une fois), experts sur les cœurs tensoriels avec les
  tokens en colonnes (même découpage, même ordre que le chemin à un token, lui-même sur les cœurs
  tensoriels depuis v1.1.72) → drafting actif = inactif au bit.
- **Contexte long** : après un document de 65 K tokens, code neuf 114 → 154 tok/s (simple 102),
  proposition 6,1 → 2,0 ms, fenêtre de 5 23,2 → 17,4 ms (clés lues une fois pour la fenêtre,
  attention du drafter sur les cœurs tensoriels).
- **Cache conservé entre les tours avec le drafter actif**, à tout contexte jusqu'à 1 M.
- v1.1.76–1.1.78 : corrections (drafter perdu avec les images sur les cartes de 24 Go, réserve de
  VRAM, arrêt), rien de neuf pour la vitesse d'Inkling.

## Comparaison avec nous

| | Maya (3090) | Nous (M5, journal) |
|---|---|---|
| Pas simple | 8,06 ms | ≈ 36 ms harnais (C0-b), 37,3 ms bridge (A1F) |
| Bloc à 3 propositions / pas simple | (5 lignes : 2,18) | 2,40 harnais (A3), 2,46 bridge (A1F) |
| Ligne vérifiée en plus | ≈ 1,9 ms ≈ 24 % d'un pas | ≈ 12,5 ms ≈ 35 % (B0), dont ≈ 10 ms d'experts (C0) |
| Drafter / pas simple | 1,7 ms ≈ 21 % | ≈ 8 ms ≈ 22 % (A3) |
| Acceptation code T = 1 | 68 % couplé (60 % ponctuel) | 65 % ponctuel (A1F, 1 861 / 2 878) |
| Gain code | ×1,62 glouton | ×1,003 à T = 1 (A1F) |
| Cache de prompt avec DSpark | oui | **non** (voir D1) |

Lecture : la part du drafter est la même ; l'écart vient (1) d'une ligne vérifiée moins chère
relativement (24 % contre 35 %), ce qui permet des fenêtres de 5–8, et (2) d'une acceptation plus
haute, en partie grâce au couplage à T > 0.

## Pistes nouvelles (aucune lancée ; chaque essai à préenregistrer ; gains estimés, non mesurés)

### D1 — cache de prompt avec DSpark (prérequis produit, sans perte)

Constat dans `origin/main` `6d1b8df16` (`native/src/main.rs`, préremplissage du bridge) : quand
DSpark est actif, `session.prefill_with_images(..)` repréremplit toute la conversation et renvoie
`cached = 0` ; `prefill_round` (qui reprend `CacheState::Inkling`) n'est pas appelé, rien n'est
capturé. La campagne PC mesure ce que cela coûte : tour 2 à 3 082 tokens, prefill 2,39–2,45 s avec
reprise contre 17,35–18,03 s sans. Activer DSpark dans l'app coûte donc ≈ +15 s de TTFT par tour à
3 k tokens, bien plus que tout gain de décodage plausible (×1,1 sur 1 200 tokens ≈ −4 s). Changement
: capturer à la même frontière de morceau l'état du drafter (caches K/V de ses 5 couches et son
`f_rnn`) avec `InklingPrefix`, ou le recalculer depuis les taps des seuls nouveaux tokens.
Oracle : logits et état du drafter repris = prefill à froid, au bit (même protocole que PC).

### D2 — couplage par bruit de Gumbel partagé (T > 0, chemin de l'app ; loi exacte)

Notre décision de bloc à T > 0 (`decide_block`, A) est la règle ponctuelle `U_i < p_i(x_i)` sur
les meilleurs tokens du drafter ; P3 (propositions tirées de q, règle min(1, p/q), bruit
indépendant) n'a gagné qu'≈ 1 point. Le couplage n'a jamais été essayé. Le drafter sait déjà tirer
par Gumbel-max sur GPU (`propose_with` avec `sampled`, graine + position) ; il faut (a) le même
générateur de bruit `g(graine du bloc, position, token)` dans le tirage du drafter et dans la
décision, (b) la décision « token de la cible = `argmax(log p_i / T + g)`, proposition gardée si
égale », plus simple que l'actuelle (pas de résidu). Une graine neuve par bloc suffit à l'exactitude :
le préfixe commité au rejet en position j ne dépend que du bruit des positions ≤ j.
Contrôles : loi jointe (acceptés, token) par khi-deux comme A, mutation (bruit de la cible réutilisé
entre blocs, ou résidu) détectée, glouton inchangé. Estimation avec les coûts A1F : si l'acceptation
gagne ≈ 10 points comme chez Maya (≈ 0,65 → 0,75), un bloc de 3 donne ≈ 2,73 tokens pour 2,40–2,46
pas → **code ≈ ×1,1** (au lieu de ×1,0) ; rédaction FR inchangée (≈ 0,37 en tête).

### D3 — coûts mesurés en ligne + garde par profondeur (sans perte, politique seule)

`WidthPolicy` (branche dspark-fast) est déjà par position, mais ses coûts sont une table figée
(P10c : 1 / 1,56 / 1,83 / 2,17) alors qu'A1F mesure 1 / 2,18 / 2,12 / 2,46 au bridge. À la façon de
`SpecCtl` : coût de chaque largeur appris des `DeviceRoundTimes` (déjà relevés depuis A3), premier
bloc de chaque largeur ignoré ; garde par profondeur, utile aussi à la recherche dans le prompt en
mode Agent (contexte jusqu'à 32 k). À vérifier dans le même essai : les sondes de largeur 1 coûtent
81 ms au bridge contre 61 au harnais (hypothèse A1F non vérifiée : contexte du drafter rattrapé au
bloc suivant après une longue suite de largeur 0).

### D4 — experts sur unités matricielles, tokens en colonnes (gros chantier, décision utilisateur)

C'est le levier structurel de Maya (ligne vérifiée à 24 % d'un pas) et celui que C0 désignait.
Deux constats du code corrigent ce qu'on pourrait transposer :

- L'astuce de Maya « entrée FP32 en deux limbes FP16 » **ne s'applique pas** chez nous : le chemin
  par route (`_qmv_mapped_tile_kernel.metal`) et le chemin TensorOps (`_expert_qmm_tensor_kernel.metal`)
  lisent déjà tous deux `xhat` en FP16, les poids décodés sont exacts en FP16 (`codebook_header`
  additionne en `half`) et les deux accumulent en FP32. L'écart de qualité de C1 vient donc de
  l'ordre d'accumulation et de sa propagation dans le modèle bouclé, pas de la précision des entrées.
- C1 rapporte des moyennes (≈ 1e-3 sur 10 lignes sur 12, 0,0112 au total, cascades sur chat0 / fr0),
  Maya une médiane (5,7e-5) : les chiffres ne sont pas comparables.

Ce qui rend la forme de Maya exacte, c'est que **le chemin à un token est lui-même sur les unités
matricielles** : la fenêtre ajoute des colonnes au même calcul, donc drafting actif = inactif au bit.
Chez nous cela voudrait dire changer aussi les experts de l'AR (changement numérique de tout le
modèle par rapport à aujourd'hui, contraire à la règle « noyaux sans perte » sauf décision explicite),
avec un contrôle de qualité (KL médiane et queue, accord d'argmax, perplexité, une évaluation) puis
de vitesse. Maya note aussi qu'un seul simdgroup par tuile était plus lent que le noyau scalaire tant
que plusieurs ne gardaient pas des lectures en vol. Gain non estimable sans un microbenchmark d'un
prototype (T = 1, 2, 4, 8, vraies banques) ; ED (77 % du temps des experts reste quand les données sont
en cache) et C0 (coût par route) disent seulement que le cœur, pas la bande passante, limite
aujourd'hui.

### D5 — drafter et recherche dans le prompt dans le même contrôleur

Aujourd'hui la recherche est coupée quand DSpark est demandé (`dspark.is_none()` dans la condition
de `inkling_lookup`) : en mode DSpark on perd les gains de copie (réécriture +7 à +60 %, code +20 à +34 %
à T = 1 d'après LK). Maya propose la plus longue des deux à chaque pas. Utile seulement si DSpark
devient utilisable (D1, D2).

### V1 — coût d'une ligne vérifiée selon la profondeur (diagnostic d'abord)

Nos mesures de lignes vérifiées sont à contexte court (harnais) ou ≈ 1–2,5 k (bridge). L'attention
d'Inkling passe par `sdpa` de MLX avec un masque additif FP16 `[1, têtes, T, clés + 1]` construit par
visite ; à 32 k (mode Agent, recherche active) on ne sait pas ce que coûte une ligne en plus ni le
drafter. Maya a gagné 25 % sur la fenêtre et 67 % sur le drafter à 65 K. Mesure proposée :
`verify_cost_by_rows` et le coût du drafter à 1 k / 8 k / 32 k ; ne viser un noyau (requêtes côte à
côte, même ordre par requête) que si la part de l'attention grandit nettement.

## Déjà couvert chez nous (ne pas refaire)

Drafter quantifié et repli Markov 8 bits (P1, 8/8 retenu) ; acceptation par position (`WidthPolicy`) ;
round décidé sur GPU et drafter recouvert (A + A3, branche `claude/inkling2-dspark-fast`, non fusionnée) ;
décodage partagé des experts dans la vérification (B rejeté, C0-c) ; chemin TensorOps existant pour la
vérification (C1 rejeté) ; cache de prompt hors DSpark (PC) ; recherche dans le prompt, y compris en mode
Agent (LK, AGENT-…-VITESSE V1) ; tirage GPU un pas d'avance (P14) ; K/V en place (P12). À ne pas
reprendre : cache K/V FP8 (avec perte), candidats top-1000 (loi tronquée), regroupement des experts du
prefill court (même mécanisme que B ; P6b n'a rien gagné à 32 tokens).

## Ordre conseillé

1. **D1** (sans lui DSpark reste inutilisable en conversation, quel que soit le décodage).
2. **D2** puis **D3** (petits, sans perte de loi, ciblent le chemin de l'app à T = 1) ; une mesure au
   bridge code + FR avec cache actif tranche alors DSPARK-OFF.
3. **V1** (diagnostic court) ; **D4** seulement sur décision explicite (changement numérique de l'AR).

## Limites

Chiffres de Maya non reproduits, autre matériel (RTX 3090, ≈ 6× notre bande passante) et autre
quantification (GGUF NVFP4/Q8_0 contre EXL3 3,33 bpw). Estimations de gain calculées avec les coûts
A1F et une acceptation supposée, non mesurées. Clone de Maya temporaire, non conservé.
