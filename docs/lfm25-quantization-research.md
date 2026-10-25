# LFM2.5-24B-A2B K4 : recherche pour viser 2–4 heures

Recherche et contrôles des 13–14 octobre 2026, Apple M5 / 24 Gio. La
[PR #35](https://github.com/0xZKnw/mlxl3/pull/35) contient le convertisseur
et le packing Metal ; cette campagne n'ajoute **aucun gain de vitesse validé**.
Cinq variantes K4 exactes ont été retirées parce qu'elles ralentissaient les
groupes d'experts ; un changement de purge du cache reste non concluant et a
aussi été retiré. La durée totale et la qualité du 24B restent non mesurées.

**Révision après conversion et mesures, 14 octobre au soir :** la conversion
complète avec le code conservé prend 3 h 58 min 39 s, sans comparaison de durée
complète amont. Sur 8 192 tokens WikiText-2 en fenêtres de 1 024, PPL+4,71 %
contre le teacher BF16 ; un protocole commun avec le GGUF UD-Q4_K_XL fourni
donne PPL18,3656/18,3467, pour 12,44/14,88 Go de poids. La proximité observée
ne certifie pas une égalité générale de qualité. Ces résultats sont postérieurs
aux recherches ci-dessous et n'attribuent aucun gain aux prototypes retirés.
[Rapport et limites](lfm25-measurements-2026-10-14.md).

## Ce que le profil permet de conclure

Le profil exerce `hessian.ldlq_quantize_group` avec les vrais groupes du
convertisseur : huit gate/up de forme interne 2048×1536 ou quatre down de
forme 1536×2048, poids synthétiques BF16, 2 048 lignes de calibration FP16,
K4/MCG, sigma 0,025, seed 139. La préparation, la recherche des scales,
la factorisation, la récursion et le packing sont inclus ; la calibration du
modèle et l'export des shards sont exclus.

Une chauffe par forme est exclue, puis trois répétitions alternent les formes.
Les huit passages couvrent les douze projections, non vides, finies et validées ;
leurs hashes de treillis et scales sont identiques par forme. Source de
référence : PR `80c55746ed6972e686ac0466235e8159ef4725e2`, PonyExl3
`8e7fa6b1556f59fc669e25087903b279b9b0346f` et patch consolidé du dépôt.

| Poste, médiane | Gate/up ×8 | Down ×4 |
|---|---:|---:|
| Groupe complet | 13,226199 s | 6,637325 s |
| Préparation, incluant scales/Hessienne/LDL | 3,909907 s | 1,952349 s |
| Récursion LDLQ, incluant search et feedback | 9,257691 s | 4,656806 s |

La récursion représente **70,05 %** du temps complet ; la préparation environ
29,5 %. Le champ `batched_g_scale_s` est quasiment nul sur LFM parce que sa
recherche scalaire est comptée dans la préparation, pas parce qu'elle est gratuite.
Les spans parallèles internes ne sont pas additionnés comme des temps muraux.

Les 38 couches MoE ont chacune 128 gate/up et 64 down : cela donne
**608 groupes de chaque forme**. L'extrapolation linéaire vaut
`608 × (13,226199 + 6,637325) / 3600 = 3,3547 h`, soit environ **3 h 21 pour
les experts synthétiques seuls**. Elle suppose le même coût à chaque groupe
et exclut calibration, 106 autres projections, lecture/export et chauffe prolongée.
Elle ne constitue pas une estimation validée de la conversion complète.

- Pour finir en quatre heures, cette extrapolation laisse environ **39 minutes**
  à tous les autres postes. C'est une condition à vérifier sur le modèle réel.
- Deux heures exigent déjà une réduction d'au moins **40,4 %** du temps des
  experts, même en supposant les autres postes gratuits. Supprimer toute leur
  préparation laisserait **2 h 21 de LDLQ** ; il faut donc accélérer la récursion.

Le pic actif MLX du profil est 2 152 857 616 octets. Il ne mesure ni la RAM du
processus ni le pic du modèle complet. Les conditions connues sont batterie
98→96 % en décharge, MLX 0.32.2, MLX-LM 0.32.0, NumPy 2.5.2, Python 3.12,
macOS 27.2, pmset sans avertissement ; température physique non mesurée.
Aucun autre travail GPU, compilation ou proveur lancé par cette tâche pendant
les mesures. [Bruts](measurements/quant-lfm25-research-20261013/profile-baseline.json),
[calculs dérivés](measurements/quant-lfm25-research-20261013/decisions.json).

## Sources examinées et décisions

Les versions sont épinglées pour distinguer une idée amont d'un gain sur M5.

| Source primaire | Ce qu'elle apporte | Décision pour cette PR |
|---|---|---|
| [EXL3, conversion](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/doc/convert.md) | Distribution du trellis sur plusieurs GPU CUDA ; Hessiennes précomputées | Une piste sur une machine CUDA disponible, pas une accélération démontrée du M5 |
| [EXL3, quantize.py](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/modules/quant/exl3_lib/quantize.py) | Groupes de même K, scales groupés, caches par entrée/device ; `scaled_ldl` tient compte des scales propres | Batching déjà présent ; ne pas partager naïvement un facteur LDL entre experts |
| [QTIP, LDLQ](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/lib/algo/ldlq.py), [papier](https://arxiv.org/abs/2406.11235) | Récursion par blocs, cache du produit et encodage trellis | Le feedback dépend des blocs déjà quantifiés ; les paralléliser librement changerait les poids |
| [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp/tree/1d8f873e9a2a8b86b12bb1064c312c5689b77d98), [papier](https://arxiv.org/abs/2402.04396) | Autre codebook/recette | Pas un remplacement compatible du kernel EXL3 K4 |
| [MLX, kernels Metal](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [cache de noms](https://github.com/ml-explore/mlx/issues/3832) | Kernels personnalisés et précautions de cache des variantes | Kernels nommés distinctement dans le second prototype ; aucune mise à jour implicite de MLX |

L'amont EXL3 courant inclut aussi des changements de recette. Ses chiffres CUDA
et ses choix de scales ne prouvent ni la vitesse ni la qualité de notre recette
LFM. La comparaison ici conserve scales, calibration, arithmetic FP32, tie-break
et deux passes tail-biting de la référence Pony épinglée.

Avant tout nouvel essai, les historiques [Metal](metal-quantization-optimization.md)
et [opti.md](../opti.md) ont été relus. Ils évitent de refaire les pistes suivantes :

- Les deux passes trellis sont déjà dans un seul dispatch de notre search.
  Le LDLQ groupé et le batching des matmuls sont déjà intégrés.
- GSS vectorisé adaptatif exact : déjà rejeté, 15,459 contre 14,839 s sur
  18 projections Ling, même nombre d'échantillons ; aucune reprise ici.
- Plus gros groupes, scratch augmenté, fences retardées, branchement/pruning K4
  et compactage après sélection ont déjà échoué ou donné un gain insuffisant.
- Grille de scales et scale fixe Ling 0,908 modifient les poids LFM. Ils ne
  deviennent pas un réglage par défaut pour atteindre une durée annoncée.
- Math relâchée ou erreurs half : les contrôles numériques historiques échouent.
  La précision des comparaisons n'est pas abaissée.

## Nouvel essai K4 : suivi direct des labels gagnants

Le précédent compactage formait un parent 16 bits puis le décalait au store.
Le nouvel essai suit directement le label gagnant 0..15 et un sentinel 128,
restaure le parent à la traceback et stocke un octet au lieu de deux. À budget
scratch inchangé de 256 Mio, le lot maximal passe de 128 à 256 tiles.
DEFAULT/MCG sont concernés ; MUL1 conserve son kernel ushort de référence.

La variante A sélectionne des registres `uchar2` à chaque transition. B garde
des `ushort2` et convertit au store. La seconde teste l'hypothèse du coût des
conversions répétées ; son échec ne permet pas d'attribuer la régression à une
cause unique. La taille des lots change aussi dans les deux variantes.

Avant les chronos : tests indépendants Pony sur les trois codebooks, zéro,
NaN/Inf/saturation et chunks 255/256/257, puis vrai CLI miniature et bridge.
Le helper original de parent passe CBMC et une mutation shift4→5 est détectée.
Les warmups groupés vérifient les hashes avant les observations mesurées.

Chaque comparaison comprend un warmup par forme/variante et trois paires
R/C, C/R, R/C, avec ordre des formes alterné : seize passages au total.

| Variante | Gate/up, référence → candidat | Down, référence → candidat | Décision |
|---|---|---|---|
| A, sélections uchar2 | 13,404438 → 18,189841 s (+35,70 %) | 6,638937 → 7,967387 s (+20,01 %) | Rejetée |
| B, sélections ushort2 | 13,355165 → 18,906856 s (+41,57 %) | 6,647861 → 7,826419 s (+17,73 %) | Rejetée |

Les treillis/scales des seize passages de chaque campagne sont exacts entre
variantes. La dérive du candidat dépasse le seuil de 3 % dans au moins une forme
(A gate/up −6,24 % ; B gate/up −3,65 %, down +5,21 %). Les pourcentages ne sont
donc pas des estimations précises qualifiées, mais toutes les observations sont
nettement plus lentes : aucune confirmation BAAB ni aucun gain publié.
Les pics actifs MLX baissent dans le prototype, sans justifier son coût en temps ;
aucun gain de RAM processus n'est revendiqué.

Conditions A : batterie 96 % au départ puis 93 % au relevé de campagne ; B :
91 % au départ, en décharge. pmset sans avertissement, température non mesurée.
Le journal conserve et corrige la première transcription 93 % pour B.
Bruts : [A](measurements/quant-lfm25-research-20261013/k4-compare-abba.json),
[B](measurements/quant-lfm25-research-20261013/k4-b-compare-abba.json).
Les [patches et helper rejetés](measurements/quant-lfm25-research-20261013/rejected-k4/identity.json)
restent archivés. **Le search de production est restauré byte exact au commit
80c557 ; le helper expérimental est absent du package.**

## Seconde recherche : Viterbi et allocation MLX

La recherche supplémentaire du 14 octobre conserve la recette et tous les
candidats. Les sources ci-dessous distinguent une optimisation transposable
d'un changement d'algorithme ou de quantification.

| Source primaire | Application examinée | Décision |
|---|---|---|
| [FLASH-Viterbi v2](https://arxiv.org/html/2510.19301v2), [code des auteurs](https://github.com/Dzh-16/FLASH-Viterbi) | Propager un ancêtre au lieu de conserver toute la traceback ; la branche full-state garde les candidats | Transposition locale limitée au pin de la première passe, essais11/12 exacts mais plus lents. La variante beam change les chemins ; elle est exclue |
| [Viterbi GPU, convolutional codes](https://arxiv.org/html/2011.09337v1) | Frames parallèles avec chevauchement/traceback et compromis BER | Le découpage approximatif ne satisfait pas l'égalité des treillis EXL3. Aucun prototype |
| [Allocateur MLX 0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp), [limite du cache](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html) | Réutiliser les buffers libres, éviction native contre purge complète | Essai09 du seuil de purge ; moins de purges mais aucune accélération stable. L'allocateur natif avait déjà été essayé dans l'historique, aucune répétition ici |
| [Apple, shader performance](https://developer.apple.com/videos/play/tech-talks/111373/), [guide Metal 2026](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) | Coalescence, ressources partagées/registres, occupation et latence | Les essais11/12 testent deux empreintes partagées. `xctrace` absent : la cause matérielle de leur régression reste une hypothèse |
| [BCJR-QAT](https://arxiv.org/abs/2605.10655) | Kernel Triton d'une relaxation différentiable du trellis et distillation | Le chiffre du papier concerne BCJR/QAT ; ce n'est pas un remplacement exact de notre conversion EXL3. Aucun changement de recette |
| [XOR-Trellis, septembre 2026](https://arxiv.org/abs/2610.00432) | Mapping de déquantification structuré et objectif dans les coordonnées d'origine | Autre mapping/recette ; pas de substitution compatible validée sur LFM2.5. Aucun prototype |

Les prototypes historiques ont aussi été recherchés dans
`build/quantization-research/` du checkout personnel, sans les modifier. Ils
contiennent déjà argmin SIMD, lookups préchargés, vec4, swizzles, pruning,
pipeline, convergence et lots équilibrés. La propagation d'ancêtre testée ici
ne fait aucun arrêt anticipé et n'utilise aucun beam. Elle conserve les 256
étapes des deux passes et la même arithmétique des coûts.

Chaque essai suivant comprend seize groupes complets : un warmup par
forme/variante puis trois paires R/C, C/R, R/C. Mêmes BF16 seed139, 2 048
activations, K4/MCG et sigma0,025 ; correction avant vitesse. Les hashes de
treillis et scales sont identiques dans les **64 passages** des quatre essais.
Sources numériques de référence : be143a/80c557, Hessienne/cache512 identiques.

| Essai | Gate/up ×8, référence → candidat | Down ×4, référence → candidat | Décision |
|---|---|---|---|
| 09, seuil purge512→1024 Mio | 15,657205→15,808092 s (+0,96 %) | 8,107580→7,708394 s (−4,92 %) | Non concluant : toutes les dérives premier→dernier dépassent3 % |
| 10, parents octets de07B, lot128 identique | 13,560475→15,614997 s (+15,15 %) | 6,860568→7,708429 s (+12,36 %) | Rejeté ; dérives≤3 %, toutes les paires plus lentes |
| 11, ancêtres partagés sur deux rangées,32 Kio | 13,923169→20,361610 s (+46,24 %) | 6,902473→9,963916 s (+44,35 %) | Rejeté ; dérives≤3 %, toutes les paires plus lentes |
| 12, une rangée et barrière avant écriture,24 Kio | 13,903491→18,417505 s (+32,47 %) | 6,951936→8,957061 s (+28,84 %) | Rejeté ; dérives≤3 %, toutes les paires plus lentes |

Le seuil512 Mio est un **point de purge complète**, pas un plafond instantané
du cache de l'allocateur. Avec1024, les purges passent de16 à3–4 par gate/up et
de12 à2 par down ; des libérations ultérieures peuvent encore dépasser1 Gio.
Ce changement n'est pas retenu : dérives R/C gate+11,74/+8,55 %, down+4,49/+13,79 %.
Le RSS est un relevé après chaque groupe, distinct du pic actif MLX et du
cache libre ; aucun pic de RAM processus ni gain de RAM n'est revendiqué.

L'essai10 découple enfin lot et représentation des parents, confondus dans07.
Conserver128 tiles au lieu de256 ne suffit pas à rentabiliser le compactage.
11/12 éliminent les écritures de parents de la première passe, mais ajoutent
des gathers et de la mémoire partagée ;12 réutilise les registres parents et
une barrière uniforme. Sans compteurs matériels, attribuer les ralentissements
à l'occupation serait une **inférence**, pas une mesure. Les écarts de campagnes
différentes ne sont pas additionnés.

Conditions avant09/10/11/12 : batterie62/50/45/40 % en décharge, pmset sans
avertissement, température non mesurée, dépendances identiques au profil06.
Aucun autre GPU/build/proveur lancé par cette tâche pendant les mesures.
Tous les enfants sont terminés et rejoints. [Plans, bruts, patches et décisions](measurements/quant-lfm25-research-20261014/)
conservent les échecs, sources/hashes et budgets. Le driver mesuré est archivé
verbatim dans `warmup-origin-k4/profile-measured.py` ; les plans09..12 en
identifient la même empreinte.

Les copies complètes redondantes de Metal/Hessienne ne sont pas conservées :
les patches appliqués au baselinebe143a/Ponyépinglé+patchconsolidé reconstruisent
byte exactement les sources mesurées, vérifiées avant leur retrait.
[Contrôle de reconstruction](measurements/quant-lfm25-research-20261014/prototype-source-reconstruction.log).
Les helpers, harnais, driver mesuré, hashes et diagnostics restent archivés.

Les helpers rejetés ont des contrôles de source utiles, limités à leur modèle
host :2 obligations/3 couvertures pour le parent de10 ;9 pour le helper
ancêtre4096states/ri0..127/NULLri0 ; récurrence bornée4states×4étapes/unwind5,
22 obligations pour11 et21 pour12. Mutationri0→ri1 détectée ;4 cartes de128
étapes×4096états, tous16 384états finaux comparés à une traceback indépendante
sous ASan/UBSan. **Ce ne sont pas des preuves de Metal, de concurrence ou de
coûts flottants.** Les refus du frontend CBMC et le mauvais modèle initial du
cast tableau→pointeur sont conservés ; les preuves finales utilisent une
adresse explicite, les assertions de sécurité et les mêmes domaines.

Après archivage, `metal.py` et la Hessienne de production sont restaurés byte
exacts àbe143a ; les helpers expérimentaux sont absents du package. Seules les
frontières127/128/129 supplémentaires et le durcissement du profiler restent
intégrés. La calibration, les scales et les poids de la recette n'ont pas été
modifiés pour annoncer une durée plus courte.

## Pistes restantes, avec leur condition de validation

1. **Mesurer une couche MoE réelle et la calibration complète.** Le pilote doit
   couvrir ses 192 projections et leurs vraies activations, puis attribuer
   lecture, scales, Hessienne/LDL, search/feedback et export séparément. Le profil
   synthétique ne dit pas si la marge de 39 minutes suffit. Observer les
   conditions thermiques d'une campagne prolongée et vérifier logits/PPL/KL.
2. **Profiler matériellement le kernel K4 et le feedback avant une nouvelle
   réécriture.** Réduire les états ou stocker moins d'octets ne suffit pas.
   Il faut identifier occupation, trafic et attentes sur les formes 768/512
   tiles. Un nouveau kernel doit garder un oracle numérique, puis passer les
   groupes complets ; le packing seul ne retire pas une heure de conversion.
3. **Étudier la réutilisation de la covariance dans la base d'origine.** Des
   gate/up partagent l'entrée, mais leurs scales imposent des facteurs distincts.
   Réutiliser une covariance brute puis la transformer par les scales propres
   est une piste amont ; cela réordonne les opérations flottantes de notre
   chemin actuel. Comparer Hessiennes, LDL, scales et treillis ; si les poids
   changent, une validation du modèle réel est nécessaire. Gain non mesuré.
4. **Pour une cible stricte de deux heures, comparer du matériel plus puissant.**
   La distribution CUDA d'EXL3 est une possibilité documentée. Vérifier d'abord
   support LFM, recette/codebook et sortie native ; aucune machine distante,
   dépense cloud ou conversion CUDA n'a été lancée ici.

Réduire les lignes de calibration, supprimer le feedback ou fixer les scales
peut changer le résultat ; aucun gain de qualité équivalente n'est établi pour
ces options sur le nouveau modèle. Les hardlinks, la calibration couche par
couche, les caches bornés, la reprise et le contournement de la passe de mesure
en K entier sont déjà dans la PR, pas de nouveaux gains de cette campagne.

## Reproduction et vérification du résultat conservé

Utiliser les dépendances et le patch épinglés dans le
[guide de conversion](lfm25-moe-quantization.md). Aucun poids 24B n'est requis
pour le profil synthétique ; ses données temporaires sont supprimées.

```sh
PYTHONPATH=src:references/PonyExl3:scripts python \
  docs/measurements/quant-lfm25-research-20261013/profile-baseline.py \
  --output build/lfm25-group-profile.json --repeats 3
```

Le point d'entrée final supervise son enfant à 240 s, le termine/rejoint sur
timeout ou interruption, refuse une sortie silencieuse/malformée/incomplète
et écrit `failed`, `parity=false`. Il remplace d'abord l'ancien rapport pour
ne pas conserver une validation d'un essai précédent. Un JSON n'annonce
`complete` qu'après tous les passages et leurs contrôles. Le parent possède
le dossier temporaire englobant, supprimé même après un arrêt forcé du worker.

Pour reproduire une variante rejetée, repartir du search80c557 dans un checkout
jetable, appliquer son patch et copier le helper archivé à côté de `metal.py`.
Conserver une copie du search80c557 et la passer avec `--reference-source`.
Ne pas modifier le kernel pendant la campagne. `--reverse-order` sert seulement
à une confirmation préenregistrée ; aucune confirmation n'a été nécessaire ici.

| Contrôle final | Résultat et portée |
|---|---|
| Suite dépôt, moteur natif activé | 476 réussis / 1 sauté, modèle Ling absent ; [log](measurements/quant-lfm25-research-20261013/python-full-final.log) |
| Driver de profil et superviseur | 16 réussis : couverture/hashes, données invalides, silence, JSON/compte incomplets, erreur, annulation et timeout avec PID rejoint ; dossiers supprimés sur tous les six chemins d'erreur ; [log](measurements/quant-lfm25-research-20261013/profile-cleanup-final.log) |
| Sélection CPU de la CI | 47 réussis / 43 tests GPU explicitement sautés ; [log](measurements/quant-lfm25-research-20261013/ci-host-cleanup-final.log) |
| CLI final physique après retrait et correction du nettoyage | 4 passages / 12 projections, hashes identiques au profil06 ; [validation](measurements/quant-lfm25-research-20261013/profile-cli-cleanup-validation-check.log) |
| CrossHair sur les deux gardes Python réelles | Rejet des résultats vides et répétitions ≤0, conditions confirmées par recherche symbolique ; [log](measurements/quant-lfm25-research-20261013/profile-crosshair-cleanup.log) |
| CBMC du helper archivé | 2 obligations, edge0..4095 et fresh0..15/128 symboliques, parent exact/borné ; 3/3 couvertures séparées ; [preuve](measurements/quant-lfm25-research-20261013/k4-parent-relocated-cbmc.log), [couverture](measurements/quant-lfm25-research-20261013/k4-parent-relocated-cover.log) |
| ASan/UBSan, helper archivé | 69 632 entrées exhaustives finies ; [log](measurements/quant-lfm25-research-20261013/k4-parent-relocated-host.log) |

La répétition CLI du 14 octobre vérifie le nouveau superviseur, avec un seul
passage mesuré par forme. Ses chronos diffèrent des médianes du 13 octobre ;
ce contrôle n'est pas une nouvelle qualification de vitesse. Batterie et état
connu sont dans [validation-conditions.json](measurements/quant-lfm25-research-20261013/validation-conditions.json).
CrossHair ne prouve pas Python ; CBMC du helper rejeté ne prouve ni le GPU ni
le kernel conservé. La preuve du packing conservé et ses bounds53/17/513
restent décrits dans le guide ; le Rust, le binaire et le shader livré n'ont
pas changé dans cette campagne. Aucun modèle 24B, qualité finale ou durée
2–4 h n'est présenté comme validé.

La suite complète précède le dernier durcissement du nettoyage temporaire du
profiler ; les 16 tests affectés, la sélection CPU de 47 tests et le vrai CLI
sont relancés après ce changement d'orchestration. Le calcul numérique et les
sources de production n'ont pas changé.

La version finale du driver ajoute la comparaison de variantes et la supervision
du CLI. Les révisions initiales du driver ne sont pas conservées verbatim ; les
seeds/options, sorties et hashes restent dans les bruts. Les patches et hashes
des kernels mesurés sont conservés. Les poids synthétiques temporaires ont été
supprimés ; le [manifeste](measurements/quant-lfm25-research-20261013/source-manifest.json)
identifie les sources finales et les limites de cette provenance.

La CI du précédent SHA80c557 avait EXL3 et Regression/Desktop réussis et les
deux jobs Native réussis. Son job Kani a été interrompu par un shutdown du runner
pendant `dflash_copy_len_9` ; ce n'est pas une suite Kani complète ni une CI
globale verte. [État inspecté](measurements/quant-lfm25-research-20261013/ci-previous-head-native.json),
[diagnostic](measurements/quant-lfm25-research-20261013/ci-previous-head-kani-tail.log).
Les contrôles du nouveau SHA suivent dans la PR ; aucune source Rust n'a changé.

## Vérification du 14 octobre après la seconde recherche

Le superviseur vérifie maintenant chaque identité de passe et les douze clés,
les hashes SHA256 et leur égalité, les quatre phases et les compteurs mémoire.
Un bon nombre de lignes ne suffit plus à déclarer `parity=true`. Deux
contre-exemples reproduisent les défauts précédents : huit dictionnaires vides
acceptés, puis JSON NaN empêchant l'écriture de `failed` et laissant le résultat
complet de l'enfant. Les deux échecs sont conservés. Le lecteur JSON refuse
également `Infinity`, `-Infinity` et `1e999`, même dans un champ ignoré, avant
la récupération du rapport d'erreur.

| Contrôle du code conservé | Résultat |
|---|---|
| Suite dépôt, bridge natif activé | **505 réussis /1 sauté**, modèle Ling absent,99,54s ; [log](measurements/quant-lfm25-research-20261014/python-full-final.log) |
| Driver et superviseur | **42 réussis** : contenu/compte, répétitions extrêmes, hashes divergents, NaN/Inf, neuf enfants en erreur/silencieux/incomplets/interrompus, trois réponses CLI positives ; temp et PID nettoyés ; [log](measurements/quant-lfm25-research-20261014/profile-validation-final-gates.log) |
| Sélection CPU du workflow | **73 réussis /46 sauts GPU explicites**,2,47s ; [log](measurements/quant-lfm25-research-20261014/ci-host-final.log) |
| CrossHair sur la source réelle | Trois gardes signalées «Confirmed» : résultats vides, répétitions≤0, faux rapport complet. Recherche symbolique, **pas une preuve déductive** ; [log](measurements/quant-lfm25-research-20261014/profile-crosshair-final.log) |
| Point d'entrée CLI physique final | **4 passages /12 projections**, hashes identiques06, enfant240s rejoint et fichiers supprimés ; [contrôle](measurements/quant-lfm25-research-20261014/profile-cli-final-validation-check.log) |
| Rapports09..12 avec le validateur final | **64 passages distincts**, hashes cohérents et champs valides ; [log](measurements/quant-lfm25-research-20261014/reports-final-validation.log) |

La répétition CLI est un contrôle de06 après modification du superviseur, avec
un seul passage mesuré par forme. Ses chronos ne qualifient aucun gain. Elle
commence cette fois sur secteur, batterie31 % en charge, pmset sans avertissement ;
température non mesurée. [Conditions et empreintes](measurements/quant-lfm25-research-20261014/final-cli-validation-conditions.json).
Les comptes sont inspectés, les sources numériques et le patch Pony restent
byte identiques àbe143a. Format/lint/compileall passent ; les trois YAML et
déclencheurs PR/pushmain sont contrôlés. `actionlint` absent ; la première
vérification YAML supposait à tort que tous les triggers push portaient une
listebranches, puis la validation est corrigée pour les pushes non filtrés.

Le workflow ajoute CrossHair0.0.101/Z35.1.0.0 épinglés, vérifie les trois
résultats présents et refuse une exploration incomplète. Les jobs du SHA
publié doivent être inspectés séparément des contrôles locaux. La CI Native
du précédent be143a a ses deux jobs natifs réussis mais Kani annulé ; ce
n'est pas une CI globale verte. Le packing conservé garde sa preuve53/6 déjà
décrite dans le guide ; aucun Rust n'a changé dans cette seconde recherche.

Le modèle24B, ses logits/PPL/KL, le pic de RAM complet et la durée2–4h restent
**non vérifiés**. Aucun checkpoint, app installée ou release n'a été remplacé.
