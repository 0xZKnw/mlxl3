# Recherche decode LFM3 et Qwen3.6-35B-A3B — 16 novembre 2026 (R2)

Entrée de journal : `RESEARCH-2026-11-16-DECODE-R2` dans `opti.md`.
Preuves et scripts : `docs/measurements/decode-research-r2-20261116/`.

Portée : lecture du journal et du code (`origin/main` 8c893fd38, moteur 2.0.13),
recherche web, et des diagnostics préenregistrés (D1, D2, D4, D4b, D4c, D5 et
une simulation). **Aucun code moteur modifié, aucun kernel écrit, aucune
optimisation mesurée.** Les pourcentages des sections 3 et 4 sont des
estimations, sauf mention « mesuré ». Les mesures bridge ont été prises avec
une charge machine de 6 à 18 (compilations d'autres sessions) : précision
≈ ±5 % au mieux, et plusieurs séries de débit sont déclarées non concluantes.

## Classement des pistes

| Rang | Piste | Modèle | Gain estimé | Sans perte ? | Appui |
|---|---|---|---|---|---|
| 1 | L1 — tête certifiée top-k pour l'échantillonnage | LFM3 | +10 à +12 % sur le chemin par défaut de l'app | Loi quasi identique, pas bit à bit : décision à prendre | Marge mesurée au bridge (D2), écart de loi mesuré (D4b) |
| 2 | L2 — crible int4 g64 à deux étapes | LFM3 | +5 à +7 % de plus en greedy ; à T>0 seulement avec une fenêtre de logits | Oui | Logits réels (D4, D4c) |
| 3 | L4 + L3 — lookup v2 composé avec la tête certifiée | LFM3 | +25 à +40 % en édition et résumé, +3,5 à +8 % en génération libre | Oui | Simulation calibrée |
| 4 | Q1 — relire seulement le top-K des lignes échantillonnées | 35B | ≤ +5 à +6 % sur MTP3 avec les défauts app | Oui, en loi | Écart mesuré au bridge (D2), sous charge |
| 5 | Q2 — dédoublonnage des experts dans le kernel | 35B | +3 à +7 % (majorant) | Oui | Statique ; deux variantes déjà rejetées |
| 6 | Q3 — draft MTP tiré | 35B | ≤ +2 à +4 %, à T 1,0 seulement | Oui, en loi | Acceptation mesurée aux préréglages (D5) |

## 1. Point de départ

| | LFM3-1.7B EXL3 8 bpw | Qwen3.6-35B-A3B EXL3 2,49 bpw |
|---|---|---|
| Octets de poids lus par token | ≈1,99 Go : FFN 1134 Mo, tête F16 524 Mo, conv 219, attention 116 | ≈1,45 Go : in-proj GDN 368 Mio, tête K6 364, experts routés 296, out-proj GDN 120, QKV 90, expert partagé 61 |
| Decode AR greedy | ≈50 tok/s ici, avec la tête certifiée (moteur 2.0.13) | ≈54–58 tok/s (pipeline P6) |
| Distance au mur mémoire | 93–98 % (plafond pratique ≈115 Go/s, DIAG-20261116-M5-BW) | ≈60 % : 17 ms GPU pour un plancher ≈10,3 ms (P8) |
| Spéculatif existant | prompt lookup opt-in : copie +79 %, édition +27 %, reste neutre | MTP1 ≈73–81 tok/s ; DFlash2 mode 1 ≈63–77 tok/s (code local) |

Ce que cela impose :

- **LFM3** : les kernels ne sont plus le sujet. Un gain sans perte vient soit de
  lire moins d'octets par token sans changer le résultat, soit de produire
  plusieurs tokens par lecture des poids. Une vérification de 2 à 4 lignes ne
  coûte que 1,24 à 1,43 pas (µbench LFM3-PLD).
- **35B** : deux mois de travail kernel (P4–P24, GROUPED-R9, DECODE-GLUE, T1,
  P13–P15) ont fermé presque toutes les pistes AR. Une ligne de vérification
  coûte ≈0,37–0,41 pas AR (≈6–8 ms, dont ≈3 ms d'experts routés) : c'est ce qui
  plafonne MTP et DFlash2.

## 2. Ce que les diagnostics de cette session établissent

1. **L'app n'utilise jamais la tête certifiée LFM3.** Les défauts Desktop
   (T 0,2, top-k 80, pénalité 1,05) et tous les préréglages recommandés du
   catalogue sont à T>0, or la tête certifiée exige le greedy. Mesuré au bridge
   (D2) : défauts app 44,6 tok/s contre 50,2 en greedy, soit ≈ −10 à −12 %.
2. **Le sampler CPU n'est pas le problème sur LFM3, mais le chemin échantillonné
   pèse sur MTP3.** Le sampler coûte 0,18 ms par ligne à 128 000 logits et
   0,35 ms à 248 320 (D1). Sur LFM3, tête certifiée coupée, greedy et défauts
   app vont à la même vitesse (D2, +0,2 %). Sur le 35B en MTP3, les défauts app
   sont 6,4 % plus lents que le greedy à acceptation égale (D2, 4 lignes relues
   en entier par bloc).
3. **Un crible int4 devient viable avec un seuil exact à deux étapes.** Sur 512
   pas réels (D4), int4 par groupes de 64 ne laisse que 16 candidats en médiane
   (p99 1 941) et lirait 140 Mo au lieu de 262 Mo. À une étape, comme le
   kernel livré, il en laisse 1 542 en médiane et 72 293 au p90 : la conclusion
   « int4 inutilisable » du journal ne vaut que pour une étape.
4. **Le crible int8 déjà livré suffit pour l'échantillonnage.** Certifier le
   top-80 exact ne demande que 122 lignes à recalculer en médiane, 170 au pire
   (D4c). Et normaliser le sampler sur ce top-K ne déplace en moyenne que 4e-5
   de la masse de probabilité par pas, sans jamais changer le token le plus
   probable (D4b).
5. **Le prompt lookup actuel est trop strict.** Sa règle (8 IDs identiques dans
   les 1024 derniers) ne propose rien sur un résumé de document de 2 248 tokens.
   En simulation calibrée (le simulateur retrouve exactement les blocs mesurés
   par LFM3-PLD sur 5 cas), une correspondance de 2 IDs sur tout le contexte
   donne une estimation de +25 % sur ce résumé, +26 à +40 % en édition, et
   +3,5 à +8 % en génération libre. Sur le 35B la même idée ne paie pas.
6. **Sur le 35B, l'acceptation MTP tient aux préréglages officiels.** 0,853 en
   greedy, 0,836 à T 0,6 et 0,813 à T 1,0 en MTP1 (D5) : un draft tiré au lieu
   de l'argmax n'aurait que quelques pour cent à reprendre.

## 3. Pistes LFM3, par ordre de priorité

### L1 — Tête certifiée top-k pour l'échantillonnage (chemin par défaut de l'app)

- **Ce que c'est** : étendre la tête certifiée du greedy au top-k. Le crible
  donne des bornes sur toutes les lignes ; on recalcule exactement les lignes
  candidates et on certifie que l'ensemble des k meilleures est complet. Le
  sampler ne voit plus que ces logits exacts.
- **Gain attendu** : ramener le chemin par défaut au niveau du greedy, soit
  ≈ +10 à +12 % (majorant mesuré par D2).
- **Pas de nouveau crible à écrire** : avec le crible int8 déjà livré, certifier
  le top-80 exact ne demande que 122 lignes à recalculer en médiane, 170 au pire
  (D4c), soit moins de 1 Mo en plus des 262 Mo du crible.
- **Le point dur** : le sampler actuel travaille sur `log_probs`, donc sur le
  `logsumexp` de tout le vocabulaire, arrondi en F16 par un kernel MLX en
  `fast::exp`. Ce nombre n'est pas reproductible sans lire toute la tête : sur
  ≈10 % des pas il faudrait recalculer la moitié du vocabulaire pour borner la
  queue à 1e-4 (D4). Deux façons d'en sortir :
  - **S2, nouveau contrat du sampler (recommandé)** : normaliser sur les logits
    exacts du top-K certifié, en f64 sur CPU, avec un ordre canonique des
    candidats. La loi change légèrement : seuls les tokens pénalisés voient leur
    poids relatif bouger, d'un facteur `exp((p−1)·δ/T)` où δ est la masse hors
    top-K, et l'arrondi F16 des log-probas disparaît. Mesuré sur 512 pas réels
    (D4b) avec les défauts app : distance en variation totale par pas médiane
    ≈0, p99 3,3e-4, max 0,7 %, moyenne 4e-5 ; le token le plus probable ne
    change jamais. Rien n'est reproductible à graine fixe aujourd'hui (le bridge
    sème avec l'horloge), donc aucun contrat visible ne casse. **C'est une
    décision utilisateur** : ce n'est pas bit-identique à l'ancien sampler.
  - **S1, loi exacte** : choisir le token avec une normalisation approchée,
    puis vérifier par lots de M pas avec la vraie tête multi-lignes (une seule
    lecture pour M lignes) et revenir en arrière en cas d'écart. Exact, mais
    demande un retour arrière d'état et gagne moins.
- **Antécédents** : CSV-Decode (certification top-k par bornes de
  Cauchy-Schwarz sur des clusters) ; l'échantillonnage exact « lazy Gumbel »
  pour top-k = 0.
- **Premier contrôle** : l'écart de loi est déjà mesuré sur CPU (D4b). Reste
  l'oracle GPU « top-k certifié = top-k de la tête complète », avec repli sur
  les égalités à la frontière, avant toute mesure de vitesse.

### L2 — Crible int4 g64 à seuil exact en deux étapes

- **Ce que c'est** : remplacer la copie int8 (262 Mo) par une copie int4 à une
  échelle par groupe de 64 poids (139 Mo). Étape 1 : crible int4 et meilleur
  provisoire. Étape 2 : logit exact de ce provisoire, qui sert de seuil ; seules
  les lignes dont la borne haute le dépasse sont recalculées.
- **Preuve de faisabilité** (D4, logits réels) : 16 candidats en médiane, 315 au
  p90, 6 307 au pire, 0 violation de borne ; 140 Mo lus contre 262 Mo.
  Des groupes de 128 donnent 136 Mo avec une queue deux fois plus longue ;
  int3 est rejeté (199 à 241 Mo lus en moyenne).
- **Gain attendu** : ≈ −0,9 à −1,1 ms par token, soit ≈ +5 à +7 % en plus de
  la tête certifiée actuelle, si un kernel int4 tient le débit mémoire. Bonus :
  −123 Mo de mémoire GPU.
- **Risques** : décodage des nibbles plus coûteux par octet ; la borne int4
  (9,3 logits) est douze fois plus large que la borne int8, donc les termes
  d'erreur flottante doivent être refaits proprement.
- **Sans perte** : oui par construction (recalcul exact + certificat + repli),
  comme la tête livrée.
- **Limite à T>0** : int4 ne certifie pas un top-80 exact à bas coût (20 088
  candidats en médiane, 265 Mo lus en moyenne, D4c). Combiné à L1, il ne reste
  intéressant que si le sampler se contente d'une fenêtre de logits sous le
  maximum (158 Mo en moyenne à T 0,2, 143 Mo à T 0,1). En greedy il n'y a pas
  cette contrainte.

### L3 — Lookup v2 : tout le contexte, correspondance courte

- **Ce que c'est** : garder `verify_rows` tel quel (R ≤ 4, bit-exact) et changer
  le drafter : plus long suffixe trouvé dans **tout** le contexte (table de
  hachage ou automate de suffixes), longueur minimale 2 au lieu de 8, largeur 3.
- **Estimation par simulation** (coût pessimiste, préfixe 2000) :

  | Cas | Règle actuelle (8 IDs, fenêtre 1024) | m=2, tout le contexte | m=4, tout le contexte |
  |---|---|---|---|
  | copie | +59 % (mesuré +79 %) | +69 % | +67 % |
  | édition de code | +23 % (mesuré +27 %) | +40 % | +29 % |
  | résumé d'un document de 2 248 tokens | 0 % | +25 % | +19 % |
  | génération libre (fr, code, chat, howto, story) | −0,2 à +3,6 % | +3,5 à +8,1 % | +0,2 à +7,2 % |

- **Lecture** : gros gain dès que la réponse s'appuie sur un texte fourni ou sur
  son propre raisonnement ; gain modeste en génération libre. Le critère
  préenregistré (+8 % hors copie) est passé pour l'édition et le résumé, pas
  pour la génération libre.
- **Prérequis** : L4, sinon activer le lookup fait perdre la tête certifiée.
- **Antécédents** : SAM-Decoding (automate de suffixes statique + dynamique),
  SuffixDecoding (arbres de suffixes sur les sorties précédentes), `ngram-mod`
  de llama.cpp (cite justement les modèles à raisonnement et le résumé).
  llama.cpp recommande des clés longues (12–24 IDs) parce qu'un rejet y coûte
  cher ; ici la deuxième ligne coûte 24–27 % d'un pas et les suivantes presque
  rien, d'où l'intérêt des clés courtes.

### L4 — Composer tête certifiée et lookup

- Aujourd'hui les deux sont exclusifs (`lfm3_cert_enabled(…, lookup, …)`) : le
  lookup lit la tête F16 complète à chaque pas, proposition ou non.
- Faire passer les lignes de vérification et les pas sans proposition par le
  crible rend les deux gains cumulables. Le crible lit ses poids une seule fois
  pour R lignes.
- Sans perte, même oracle que les deux fonctions existantes.

### M1 — Mémoire : embedding LFM3 lu à la demande

- L'embedding BF16 [128 000 × 2048] est chargé en entier sur GPU (524 Mo, soit
  ≈20 % de l'empreinte du modèle) alors qu'un pas n'en lit qu'une ligne.
- Le moteur sait déjà lire une ligne d'embedding par `pread` avec un petit cache
  pour Flash Next. L'appliquer à LFM3 libérerait ≈524 Mo sans toucher au calcul
  (mêmes octets, même conversion). Vitesse attendue neutre ; non mesuré.
- Avec L2, cela ferait ≈ −650 Mo au total par rapport au moteur 2.0.13.

### L5 — Pistes écartées par le calcul

- **Drafter = même modèle à bas bpw** (4,5 bpw) : un pas de draft coûte ≈0,6 pas
  cible ; avec 90 % d'acceptation le gain plafonne à ≈ +5 à +7 % pour +1,3 Go.
- **Tête différée à sous-vocabulaire chaud** : à 97 % d'accord le retour
  arrière mange le gain ; elle ne bat la tête certifiée qu'au-delà de 99 %.
- **Plus de 4 lignes par vérification** : utile seulement en copie pure
  (≈ +100 % contre +69 %, coût par ligne supposé, non mesuré).
- **Tête sur CPU en parallèle du GPU** : le bus mémoire est déjà saturé, le
  total d'octets par token ne baisse pas.
- **Baisser le bpw du FFN** (8 → 6) : seul levier sur les 57 % d'octets du FFN,
  mais avec perte ; décision utilisateur, hors de cette recherche.

## 4. Pistes Qwen3.6-35B-A3B

### Q1 — Lignes de vérification échantillonnées : ne relire que le top-K

- **Mesuré** (D2, MTP3, deux processus, deux ordres) : avec les défauts app, MTP3
  tourne à 67,8 tok/s contre 72,5 en greedy, soit −6,4 % (−4,7 à −8,8 % selon
  le cas), pour une acceptation identique (0,81) et le même nombre de tokens par
  bloc. Cela fait ≈2,5 ms de plus par bloc de 4 lignes.
- **Cause probable** : chaque ligne passe par `log_probs` sur 248 320 valeurs,
  une conversion F32, une relecture de 1 Mo, deux copies, puis 0,35 ms de
  sampler CPU. Seuls ces 0,35 ms sont mesurés isolément (D1).
- **Piste** : extraire sur GPU les K meilleurs logits de chaque ligne (plus les
  tokens pénalisés) et ne relire que cela. Demande le même ordre canonique des
  candidats que L1 ; avec le `logsumexp` MLX conservé, la loi ne change pas.
- **Gain attendu** : au mieux ≈ +5 à +6 % sur MTP3 échantillonné ; moins sur
  MTP1 (2 lignes). En AR, l'effet n'est pas visible (mesure trop bruitée).
- **À confirmer d'abord** sur machine calme : la mesure a été prise sous charge.

### Q2 — Dédoublonnage des experts dans le kernel, sans tri

- **Constat du journal** : en vérification, l'union des experts distincts vaut
  68–81 % des créneaux pour 2 à 4 lignes (P16) et ≈61 % pour 6 lignes
  (PERF-94) ; chaque créneau redécode pourtant son expert.
- **Déjà rejeté** : chemin segmenté à M=6 (PERF-95, −41 %) et kernel par paires
  après tri/gather GPU (PERF-96, −16 %). La session DFlash2 la liste comme
  « piste suivante » : elle a déjà deux échecs.
- **Variante non essayée** : garder la grille de dispatch dense actuelle et
  décider *dans* le kernel. Le créneau (ligne r, expert e) cherche si une ligne
  suivante route vers e ; si oui il accumule les deux lignes sur le même décodage
  et le créneau jumeau sort tôt. Pas de tri, pas de gather, pas de dispatch
  clairsemé : ce sont les trois coûts cités par PERF-96.
- **Majorant** : ≈0,7–1,0 ms sur un bloc MTP1 de ≈22,7 ms (+3 à +4,5 %),
  ≈4–5 ms sur un bloc DFlash2 de largeur 5 (≈ +7 %). Risque élevé (pression
  registres des kernels à deux lignes). Sans perte si chaque route garde ses
  accumulateurs et son ordre.

### Q3 — Draft MTP tiré au lieu de l'argmax : faible enjeu, mesuré

- **Constat source** : en mode échantillonné, le draft MTP reste l'argmax de la
  tête (`mtp.rs`) et la vérification accepte x avec probabilité p(x)
  (`sampling.rs`, draft « masse ponctuelle »). C'est exact en loi, mais
  l'acceptation ne peut pas dépasser la probabilité du token le plus probable.
- **Hypothèse de départ** : les préréglages Qwen3.6 du catalogue sont à T 1,0
  et T 0,6. Sur les logits réels de D4 (LFM3, à titre indicatif), la part du
  top-1 vaut en moyenne 0,76 à T 1,0 / top-k 20 / top-p 0,95 : une tête parfaite
  y perdrait un quart de ses propositions. Le 35B se révèle bien plus piqué
  que cela (voir l'enjeu mesuré ci-dessous).
- **Ce que c'est** : l'échantillonnage spéculatif standard. Tirer x dans la loi
  q de la tête MTP (après les mêmes transformations de sampler, avec une
  température de draft réglable), accepter avec probabilité min(1, p(x)/q(x)),
  sinon tirer dans le résidu max(0, p − q) normalisé. La loi de sortie reste
  exactement p, quelle que soit q. L'acceptation devient Σ min(p, q) : elle
  vaut p(argmax) quand la température de draft tend vers 0 (comportement
  actuel) et 1 − TV(p, q) à température égale.
- **Attention** : si la tête est moins confiante que la cible, tirer fait
  perdre. D'où la température de draft, à régler sur traces.
- **Indice favorable** : la confiance de la tête est déjà bien calibrée contre
  l'accord greedy (T0A du 15 novembre : 44 % d'accord sous 0,5 de confiance,
  59 % entre 0,5 et 0,8, 86 % entre 0,8 et 0,95, 99,6 % au-dessus de 0,99).
- **Mise en œuvre compatible avec la chaîne actuelle** : le tirage peut rester
  sur GPU (argmax de log-probas + bruit de Gumbel), donc toujours une seule
  chaîne paresseuse et un seul transfert par cycle ; q(x) voyage avec les IDs,
  comme le top-1 d'EVICT aujourd'hui.
- **Enjeu mesuré (D5)** : aux préréglages officiels, l'acceptation baisse peu.
  MTP1 : 0,853 en greedy, 0,836 à T 0,6, 0,813 à T 1,0. MTP3 : 0,807, 0,805,
  0,762. En tokens par bloc cela fait −0,8 à −0,9 % à T 0,6 et −2,3 à −3,7 %
  à T 1,0. Un draft tiré ne peut donc reprendre que quelques pour cent, et
  seulement à T 1,0 : **piste de faible priorité**.
- **Premier contrôle** : compteur de diagnostic par position, p(x*) contre
  Σ min(p, q_τ) pour quelques τ, sans changer aucune sortie.
- **Antécédents** : Leviathan et al. 2023, Chen et al. 2023.

### Q4 — Lookup v2 sur le 35B : rejeté au crible

- Simulé sur les sorties greedy du 35B avec le coût réel d'une ligne
  (≈0,40 pas) : fr −0,2 %, code +4,0 %, chat −4,5 %, édition +2,7 % (m=2) ;
  −2,0 / +0,8 / +2,2 / +2,1 % (m=4).
- Une ligne de vérification coûte trop cher pour des propositions acceptées à
  27–52 %, et la tête MTP fait mieux (0,81). Reste utile seulement pour de
  longues copies, déjà couvertes par le lookup MTP existant.

### Q5 — Ce qui ne marche pas sur le 35B

- **Tête certifiée** : la tête est déjà à 6 bpw. Un crible sous 6 bpw a une
  borne de plusieurs logits, et le recalcul exact d'une ligne exige de décoder
  son bloc Hadamard de 128 lignes (format EXL3, à confirmer dans le code avant
  tout essai) : quelques centaines de candidats dispersés
  touchent presque tous les blocs. Variante avec copie FP16 des lignes
  candidates : ≈ +9 % en AR greedy seulement, pour +1,25 Go. Écartée.
- **Expert partagé sur CPU** : 40 allers-retours CPU↔GPU par token, plus chers
  que les ≈1,3 ms à gagner.

## 5. Résultats détaillés

### 5.1 D1 — coût CPU du sampler (microbenchmark)

Vrai `Sampler::new(0.2, 80, 1.0, 1.05, 0.0)` du moteur, 2 000 itérations après
200 chauffes, trois passes concordantes :

| Vocabulaire | Conversion octets → `Vec<f32>` | `distribution` + `sample` | Total par ligne |
|---|---|---|---|
| 128 000 (LFM3) | 8,6–9,2 µs | 171–174 µs | 179–183 µs |
| 248 320 (35B) | 16,9–17,5 µs | 332–342 µs | 349–360 µs |

### 5.2 D2 — bridge réel, greedy contre défauts app

LFM3-1.7B-EXL3--8bpw, moteur 2.0.13 publié (SHA-256 644c2e08…8e00), 4 processus
A B B A, 256 tokens, cache OFF, secteur, charge 6,7–6,9.

| Processus | Greedy (médiane) | Défauts app (médiane) | Écart moyen par cas |
|---|---|---|---|
| A : tête certifiée active en greedy | 50,18 tok/s | 44,63 tok/s | −9,5 % (−13,3 … +0,1) |
| B : `MLXL3_LFM3_CERT_HEAD=0` | 43,06 tok/s | 42,33 tok/s | +0,2 % (−3,7 … +6,9) |

Le même bras « défauts app » vaut 44,63 (A) et 42,33 (B) pour un code identique :
la dérive entre processus est de ≈5 %.

Qwen3.6-35B-A3B EXL3 2,49 bpw + tête MTP 4 bits, même binaire, 4 processus
AR / MTP3 / MTP3 / AR, 256 tokens, charge 6,3 → 14,1 :

| Mode | Greedy (moyenne) | Défauts app (moyenne) | Écart par cas | Acceptation MTP |
|---|---|---|---|---|
| AR | 41,4 tok/s | 42,3 tok/s | −6,6 … +13,9 % : non concluant | — |
| MTP3 | 72,5 tok/s | 67,8 tok/s | −4,7 … −8,8 %, moyenne −6,4 % | 0,807 / 0,812 |

L'AR est très sous son niveau habituel (≈54–58 tok/s) avec des passages à
22–33 tok/s : la machine était chargée. L'écart MTP3 garde le même signe dans
les deux processus (−6,8 % et −6,2 %).

### 5.3 D4 — criblage sur logits réels

512 positions réelles (fr, code, chat, story × 128), état final reconstruit par
moindres carrés sur la tête F16 (résidu max 0,015 logit).

| Crible | Octets | Borne médiane (logits) | Candidats, une étape : médiane / p90 / max | Candidats, deux étapes : médiane / p90 / p99 / max | Octets lus, deux étapes (moyenne) |
|---|---|---|---|---|---|
| int8 ligne (livré) | 262 Mo | 0,78 | 1 / 3,9 / 9 | 1 / 2 / 4 / 6 | 262 Mo |
| int6 ligne | 197 Mo | 3,21 | 5 / 35 / 533 | 2 / 8 / 17 / 24 | 197 Mo |
| int5 ligne | 164 Mo | 6,62 | 50 / 3 192 / 66 951 | 6 / 48 / 244 / 650 | 164 Mo |
| int4 ligne | 131 Mo | 14,19 | 84 485 / 127 995 / 128 000 | 396 / 13 754 / 62 490 / 86 564 | 151 Mo |
| **int4 g64** | 139 Mo | 9,26 | 1 542 / 72 293 / 127 975 | **16 / 315 / 1 941 / 6 307** | **140 Mo** |

Statistiques d'échantillonnage (T=1) : probabilité du top-1 médiane 0,835
(p10 0,33) ; masse hors top-80 médiane 6,3e-5, p90 1,8 %, max 32 % ; lignes à
garder pour laisser ≤ 1e-4 de masse : médiane 51, p90 69 158.

Candidats à recalculer pour l'échantillonnage (D4c, seuil exact à deux étapes),
médiane / p90 / max, et octets lus en moyenne :

| Ensemble à certifier | int8 par ligne | int4 g64 |
|---|---|---|
| argmax (greedy) | 1 / 2 / 6 — 262 Mo | 16 / 315 / 6 307 — 140 Mo |
| fenêtre de 3 logits, k ≤ 50 (T 0,1) | 2 / 11 / 35 — 262 Mo | 55 / 2 022 / 39 363 — 143 Mo |
| fenêtre de 6 logits, k ≤ 80 (T 0,2) | 7 / 41 / 156 — 262 Mo | 269 / 11 129 / 79 150 — 158 Mo |
| top-80 exact | 122 / 143 / 170 — 263 Mo | 20 088 / 68 150 / 107 060 — 265 Mo |

Écart de loi d'un sampler normalisé sur le top-K (D4b), distance en variation
totale par pas :

| Réglage | médiane | p99 | max | moyenne | mode changé |
|---|---|---|---|---|---|
| défauts app (T 0,2 / k 80 / rép. 1,05) | ≈0 | 3,3e-4 | 6,6e-3 | 4,1e-5 | 0 / 512 |
| LFM3 « Général » (T 0,1 / k 50 / rép. 1,1) | ≈0 | 6,6e-4 | 2,7e-2 | 7,3e-5 | 0 / 512 |

### 5.4 Simulation des drafters sans réseau

Calibration : la règle actuelle simulée retrouve exactement les blocs mesurés
par LFM3-PLD (copie 39 blocs 112/115, édition 22 blocs 58/66, code 2, fr 2,
long 0). Les débits simulés sont prudents : copie +59 à +64 % simulé contre
+78,9 % mesuré.

Débit estimé contre l'AR, coût pessimiste. « Actuel » = 8 IDs dans les 1024
derniers ; m = longueur minimale de correspondance, sur tout le contexte,
largeur 3.

| Modèle | Cas | Actuel | m=2 | m=4 | Acceptation à m=2 |
|---|---|---|---|---|---|
| LFM3 | copie | +59,4 % | +69,0 % | +67,4 % | 0,84 |
| LFM3 | édition (PLD) | +23,4 % | +40,2 % | +29,1 % | 0,66 |
| LFM3 | édition (D2) | +15,5 % | +26,2 % | +23,3 % | 0,45 |
| LFM3 | résumé long | 0 % | +24,8 % | +18,8 % | 0,41 |
| LFM3 | fr | +1,2 % | +4,7 % | +2,8 % | 0,32 |
| LFM3 | code | +1,2 % | +6,5 % | +4,6 % | 0,27 |
| LFM3 | chat | −0,2 % | +3,5 % | +0,2 % | 0,24 |
| LFM3 | howto | +1,9 % | +8,1 % | +3,5 % | 0,40 |
| LFM3 | story | +3,6 % | +6,5 % | +7,2 % | 0,28 |
| 35B | fr | 0 % | −0,2 % | −2,0 % | 0,39 |
| 35B | code | +0,5 % | +4,0 % | +0,8 % | 0,52 |
| 35B | chat | +1,7 % | −4,5 % | +2,2 % | 0,27 |
| 35B | édition | +2,3 % | +2,7 % | +2,1 % | 0,44 |

### 5.5 D5 — MTP aux préréglages Qwen3.6

Même binaire, 35B + tête MTP 4 bits, 4 cas × 256 tokens, ordre G T C C T G par
cas. G = greedy ; C = T 0,6 / top-k 20 / top-p 0,95 ; T = T 1,0 / top-k 20 /
top-p 0,95 / présence 1,5.

| Profondeur | Bras | Acceptés / proposés | Acceptation | Tokens par bloc |
|---|---|---|---|---|
| MTP1 | G | 936 / 1 098 | 0,853 | 1,865 |
| MTP1 | C | 926 / 1 108 | 0,836 | 1,848 |
| MTP1 | T | 914 / 1 124 | 0,813 | 1,822 |
| MTP3 | G | 1 302 / 1 614 | 0,807 | 2,790 |
| MTP3 | C | 1 297 / 1 612 | 0,805 | 2,768 |
| MTP3 | T | 1 272 / 1 670 | 0,762 | 2,688 |

Les débits de ce run ne sont pas exploitables : charge machine 13,6 au début,
passages de 9 à 83 tok/s dans un même bras. Seuls les comptes d'acceptation
sont retenus.

## 6. À ne pas refaire sans raison nouvelle

Lecteurs K2/K4/K6, LUT, FP16 prédécodé, épilogues fusionnés (T1, P15), budgets
de command buffer (P13/P14), second stream GPU (P11), dense wide M5–8 (B1/B1b),
chemin expert segmenté et paires après tri (PERF-95/96), sampler top-k côté GPU
de l'ancien moteur Python (OPT-2026-09-10-07/08), draft neuronal DSpark sur LFM3,
arbres de draft (R7).

## 7. Sources

- Token Recycling : https://arxiv.org/abs/2408.08696
- SAM-Decoding : https://arxiv.org/abs/2411.10666
- SuffixDecoding : https://arxiv.org/abs/2411.04975
- CSV-Decode : https://arxiv.org/abs/2511.21702
- Lazy Gumbel sampling (Mussmann, Levy, Ermon, UAI 2017) : https://arxiv.org/abs/1707.03372
- Speculative decoding (Leviathan et al., 2023) : https://arxiv.org/abs/2211.17192
- Speculative sampling (Chen et al., 2023) : https://arxiv.org/abs/2302.01318
- llama.cpp, `docs/speculative.md` (copie locale `references/llama.cpp` b81c99b)
- MLX, `mlx/backend/metal/kernels/logsumexp.h` (copie locale `references/mlx`)
