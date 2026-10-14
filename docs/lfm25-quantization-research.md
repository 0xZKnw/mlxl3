# LFM2.5-24B-A2B K4 : recherche pour viser 2–4 heures

Recherche et contrôles des 13–14 octobre 2026, Apple M5 / 24 Gio. La
[PR #35](https://github.com/0xZKnw/mlxl3/pull/35) contient le convertisseur
et le packing Metal ; cette campagne n'ajoute **aucun gain de vitesse validé**.
Deux variantes numériques exactes ont été retirées parce qu'elles ralentissaient
les groupes d'experts. La durée totale et la qualité du 24B restent non mesurées.

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
