# T17 — audit source de la divergence micro, 30 octobre 2026

## Révision DIAG-02 / FIX-01 — défaut isolé sur le contre-exemple

Le [log physique DIAG-02](t17-diag-02-boundaries.log) contient exactement
**1 test passé, 0 échec, 0 ignoré**, cinq variantes avec 6 144 valeurs et trois
sorties par variante. Il répète le **même couple** norme/gate du
contre-exemple ; ce n'est pas une exploration de 6 144 couples différents.
Le [résultat de supervision](diag-02-gpu-plan-result.json) confirme exit0,
processus joint et zéro nouveau Swapout. Binaire diagnostic :
`59cdf76957669fd529f9a02c8ba990fce1ff8044277ca72c207d567b3d5590c1`.

| Variante | Sigmoid FP32 | Produit FP32 | Sortie FP16 | Écarts p/produit/sortie sur 6 144 valeurs |
|---|---|---|---|---|
| Original | `387c4074` | `387c5ffe` | `03f1` | 6144 / 6144 / 6144 |
| `precise_exp` | `387c4076` | `387c6000` | `03f2` | 0 / 0 / 0 |
| Sans `volatile` | `387c4074` | `387c5ffe` | `03f1` | 6144 / 6144 / 6144 |
| `precise_exp` sans `volatile` | `387c4076` | `387c6000` | `03f2` | 0 / 0 / 0 |
| Sigmoid stock passé en entrée | `387c4076` | `387c6000` | `03f2` | 0 / 0 / 0 |

Les sorties du capteur original reproduisent le vrai `apply` : le diagnostic
n'a pas perdu le défaut observé. La probabilité originale est inférieure de
**2 ULP FP32** à la référence. L'intervention isolée sur `exp` rétablit la
probabilité, le produit au milieu d'arrondi et la sortie stock dans ce cas.
Enlever `volatile` seul ne corrige rien ; recevoir le sigmoid stock ne
produit pas de divergence de multiplication/conversion ici.

Conclusion bornée : la différence est localisée au **chemin de calcul de
l'exponentielle pour ce contre-exemple**, et non au choix du buffer normed
ou à l'absence d'un facteur. Ces données ne déterminent pas la cause interne
exacte du compilateur/metallib, ne prouvent pas tous les inputs et ne
constituent aucune mesure de vitesse.

FIX-01 est préenregistré dans `opti.md`. Relecture du correctif effectué par
l'orchestrateur : le diff entre `fix-01-shader.metal.before` et
`native/shaders/flash_gdn_epilogue.metal` contient **une seule substitution**,
ligne32, `metal::exp` → `metal::precise::exp`. Les deux intermédiaires
`volatile float`, la sélection des lignes, les dtypes et la conversion finale
sont inchangés. Aucune table, nouvelle allocation, synchronisation ou
modification de budget n'est introduite.

- Original archivé : SHA256
  `3576a9962102f860aa7f642712abc34caf1e85151ca152b5e786abb1ed402636`.
- Shader FIX-01 : SHA256
  `1fa5b6ae214104186021ff2aa43a5e7b083e838fb0a410cf886326aaf8f9ca44`.

**La LUT n'est pas retenue.** Le contrôle du contre-exemple justifie d'abord
ce correctif d'une ligne. La qualification du shader réel corrigé exige
encore le smoke et l'exhaustif original, puis les contrôles affectés selon
le protocole. Aucun nouveau build/test/GPU/proveur n'a été exécuté par
l'auteur de cette mise à jour ; seule la documentation est modifiée.

## Analyse initiale conservée — avant le résultat DIAG-02

Audit documentaire après le smoke physique réussi et le premier échec de
`exhaustive_finite_half_gates_and_fp32_products_match_stock`. Il ne remplace pas
le contre-exemple attendu de T17-DIAG-01. Aucun test, build, GPU ou proveur lancé
pour cet audit ; aucun fichier exécutable modifié.

Le candidat et la référence expriment le même sigmoid FP32 avec la même
symétrie. L'écart le plus plausible reste une différence dans le calcul
transcendantal/division entre le shader JIT et le metallib stock, révélée par
une frontière d'arrondi FP16. **C'est une hypothèse, pas une attribution
établie.** Un écart de produit, FTZ ou conversion finale reste à distinguer.

Actualisation après transmission du contre-exemple DIAG-01 par
l'orchestrateur : B1, phase2, batch8, index1244 ; norme FP32
`0x3f801001` = 1,0004884004592896 ; gate FP16 `0xc8dc` = −9,71875 ;
sortie fusion `0x03f1`, stock `0x03f2` ; sigmoid stock FP32
`0x387c4076` = 0,00006014151585986838. Le shader de production reste
inchangé. Ces champs sont confirmés dans [diag-01-inspection.json](diag-01-inspection.json),
statut `failure_reproduced_not_candidate_parity`, processus joint et zéro
nouvelle écriture swap.

Un calcul documentaire sur ces bits, avec `struct` et `Fraction` Python,
donne le produit arrondi FP32 **`0x387c6000`**, exactement le milieu entre
FP16 `0x03f1` et `0x03f2`. L'arrondi au plus proche, pair en cas d'égalité,
choisit `0x03f2`. Le produit rationnel exact est seulement
1,0307987102775584×10⁻¹³ au-dessus de ce milieu ; une très petite différence
du sigmoid peut donc changer la sortie FP16. Ce calcul CPU ne mesure pas
les intermédiaires du candidat et ne remplace pas DIAG-02.

Les opérandes FP32 de ce contre-exemple sont normaux, loin de la frontière
subnormale FP32 : un FTZ FP32 n'est pas une explication suffisante de ce cas.
La sortie est subnormale FP16 ; son arrondi doit rester contrôlé. DIAG-02
prévu doit comparer les probabilités et produits FP32 du candidat, puis un
produit recevant directement le sigmoid stock pour isoler la conversion.
Tester `precise::exp` dans ce diagnostic est plus petit qu'ajouter une table,
mais ne devient une correction que si les résultats stock sont reproduits.
La variante sans `volatile` sert à détecter l'effet du compilateur ; réussir
ce seul cas ne justifierait pas de supprimer les frontières FP32.

## Faits vérifiés dans les sources

1. `native/src/gdn_epilogue.rs:58–100` reçoit chaque norme FP32 séparément,
   garde `z` FP16, exige les formes de production et retourne FP16
   `[rows, 6144]`. Il ne concatène pas les normes FP32, n'évalue pas les
   entrées et ne lit pas leurs données sur le CPU.
2. `native/shaders/flash_gdn_epilogue.metal:31–36` convertit `z` en float,
   calcule `y = 1 / (1 + exp(abs(z)))`, sélectionne `z < 0 ? y : 1-y`, puis
   multiplie la norme en FP32 et convertit le résultat en FP16. Aucun facteur
   `z` supplémentaire : ce n'est pas SiLU.
3. Le fichier **installé**
   `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/unary_ops.h:308–313`
   contient exactement cette formule dans `Sigmoid`. Son blob Git recalculé
   est `1337f6b52edb510f4fb199dee5e443e48e02b675`, identique au fichier du tag
   MLX `v0.32.2`.
4. MLX `unary.cpp:9–12,137` route `Sigmoid` vers le backend unary GPU ;
   `kernels/unary.metal:29–32,67` instancie aussi la variante float32.
   Le fichier installé `kernels/unary.h:3–18` applique l'opération à chaque
   valeur et écrit une sortie FP32. Le travail par thread ne transforme pas
   cette opération en une réduction entre valeurs.
5. Notre FFI `native/ffi/mlx_bridge.cpp:702–703` transmet
   `mx::CompileOptions{}`. Le header installé
   `backend/common/metal_kernel.h:7–16` initialise `math_mode` à `Safe`.
   MLX `device.cpp:37–62` mappe ce choix vers `MTL::MathModeSafe` sur ce Mac.
6. La recette amont `kernels/CMakeLists.txt:12–38` compile les kernels stock
   avec **`-fno-fast-math`**. Elle ne prouve pas à elle seule les options
   historiques exactes du metallib binaire installé. Il serait donc faux de
   présenter un écart « stock Fast / candidat Safe » comme un fait établi.
7. `device.cpp:65–79,661–668` choisit Metal language version **4.1** sur macOS27
   et l'applique à la compilation runtime. Les versions exactes du compilateur, du langage et
   des bibliothèques mathématiques ayant produit le metallib installé ne
   sont pas établies par ce header. Le même texte `metal::exp` ne constitue
   donc pas une garantie de mêmes bits entre les deux chaînes.

La référence de `tests.rs:10–34` conserve une sortie FP32 distincte pour le
sigmoid, une autre pour le produit, puis la conversion FP16. Les deux
intermédiaires `volatile float` du shader limitent les contractions et
réassociations au travers de ces frontières. Ils ne rendent pas `exp` ou la
division identiques au metallib, et ne prouvent pas l'identité de FTZ.
Leur coût potentiel en instructions ou stockage privé n'est pas mesuré ; les
retirer maintenant confondrait un changement de coût et un changement
numérique.

## Ce que le diagnostic doit encore séparer

- Le premier écart doit conserver les bits de `normed` FP32, de `z` FP16 et
  des deux résultats FP16, avec rows/phase/batch/index. Les 16 normes de
  l'exhaustif incluent ±zéro, subnormaux FP32, minimum normal et frontières
  d'arrondi FP16 ; le smoke n'en couvre pas le produit cartésien.
- Le sigmoid FP32 stock au même `z` permet de vérifier le calcul de produit
  et d'arrondi attendu. Il ne révèle pas à lui seul le sigmoid FP32 calculé
  dans le candidat : une attribution exclusive à `exp` demanderait un
  contrôle intermédiaire dédié, préenregistré, si le premier contre-exemple
  ne suffit pas.
- FTZ et signe de zéro sont des hypothèses distinctes. Les très petites
  normes ne doivent pas être remplacées par zéro dans l'oracle pour obtenir
  la parité. De même, une tolérance ou une comparaison du texte ne répare
  pas le contrat bit à bit de l'entrée de `out`.
- La géométrie Rust ne prouve ni les fonctions mathématiques Metal ni les
  conversions FP32/FP16. Selon le résultat transmis par l'orchestrateur,
  Kani a désormais passé 148 obligations et 9 couvertures avec unwind17 ;
  la première borne7 était insuffisante et reste conservée dans les logs.

## Variante de table, conditionnelle et non implémentée

Option historique examinée avant DIAG-02 ; **non retenue dans FIX-01**,
comme précisé en tête. Aucun travail de table ou de budget supplémentaire
n'est lancé.

Si le diagnostic identifie un écart du sigmoid, une table **FP32** indexée
par les 16 bits de `z` supprimerait cette divergence de transcendantale sans
modifier la norme. La construire avec la véritable chaîne stock :
65 536 patterns FP16 bruts → `astype(Float32)` → **sigmoid non compilé**.
Une table calculée par CPU, par une formule réécrite ou par le shader JIT
défaillant ne serait pas cet oracle.

- Taille finale : `65536 × 4 = 262144 B`, soit **256 Kio**, sans diminution
  du cache target de 3 Gio ni assimilation de la table à ce cache.
- La préparation peut être un graphe MLX lazy transmis comme entrée du
  kernel ; l'ordonnanceur en porte la dépendance. Aucune nécessité d'ajouter
  `eval`, `synchronize` ou un readback au parcours de chaque token. Le
  premier calcul du graphe reste un coût réel, à exclure des mesures chaudes
  seulement par une chauffe explicitement prévue.
- Il faut **réutiliser une table**, pas une par couche ou appel. `Array`
  contient `PhantomData<Rc<()>>` (`array.rs:505–512`) : aucun cache global
  `OnceLock<Array>` ni nouveau `Send/Sync` non justifié. Un propriétaire de
  modèle avec clones MLX légers donne une durée de vie explicite. Un cache
  thread-local est une autre possibilité, mais sa rétention après unload et
  sa multiplication entre threads doivent être comptées, pas masquées.
- Préparation naïve : entrée FP16 de 128 Kio, cast FP32 de 256 Kio et sortie
  de 256 Kio, soit au plus **640 Kio de buffers logiques** simultanés avant
  libération/donation, plus le vecteur CPU temporaire de 128 Kio. La donation
  MLX pourrait réduire cette somme ; aucun pic physique n'a été mesuré.
- Le kernel lirait `table[as_type<ushort>(z[index])]`, puis garderait le
  produit FP32 et la conversion FP16. Il ajoute une entrée et une lecture
  de table : le gain de temps n'est pas garanti. Si l'écart vient du produit
  ou du FTZ plutôt que du sigmoid, la table ne suffit pas.
- Les patterns non finis peuvent également figurer dans la table, mais
  cela ne promet pas une identité des payloads NaN à travers le produit.
  Les contrats finis et non finis restent ceux des oracles existants.

Ce mécanisme n'est pas nouveau par son dtype : HENK0O-QWEN27-09 avait déjà
une table de valeurs FP32 indexée par FP16, mais de **SiLU**, avec une norme
fusionnée à arrondis FP16. Voir `benchmarks/benchmark_gdn_norm_gate.py:48–61,118`.
La différence proposée ici serait sigmoid seul et norme native FP32 gardée.
Une telle variante exige encore son préenregistrement, son propriétaire et
son budget avant toute modification.

## Provenance

Sources MLX épinglées à `v0.32.2`, consultées par le connecteur GitHub en
lecture seule :

| Fichier | Blob Git | Lignes utilisées |
|---|---|---|
| `mlx/backend/metal/kernels/unary_ops.h` | `1337f6b52edb510f4fb199dee5e443e48e02b675` | 308–313 |
| `mlx/backend/metal/unary.cpp` | `354f199c68a99c14b0cccb90ced23b2381c5b42c` | 9–12, 98–112, 137 |
| `mlx/backend/metal/kernels/unary.metal` | `54a0f566c8236927aab8d65a60c8a1e931288b6d` | 29–32, 67 |
| `mlx/backend/metal/kernels/CMakeLists.txt` | `ecabbda550780ae2e18ee8e002914f03d12bca4d` | 12–38 |
| `mlx/backend/metal/device.cpp` | `2f25f894e494a4328886148bb6e3332d9f9b298a` | 37–79, 661–668 |
| `mlx/backend/metal/compiled.cpp` | `e95d7b8d5f1ea879bb12a067bfe04cde7739fd9b` | 205–220, 265–267 |

Le dernier fichier montre que le JIT de graphes utilise également des
temporaires typés et les opérateurs de `unary_ops`. T17 utilise actuellement
un **custom kernel**, pas cette fusion de graphe ; cela ne constitue donc
pas une validation numérique de son binaire.

Vérifications d'identité CPU exécutées pour cet audit uniquement : lectures,
`shasum -a 256`, `git hash-object` sans écriture. Le fichier `device.cpp`
archivé sous `build/flash-strata57` est bien identique au blob amont ci-dessus.

| Source locale/installation | SHA-256 au relevé |
|---|---|
| `native/shaders/flash_gdn_epilogue.metal` | `3576a9962102f860aa7f642712abc34caf1e85151ca152b5e786abb1ed402636` |
| `native/src/gdn_epilogue.rs` | `ab0e52de616869f2aa14aa1dd6cdbc7e8dcb12550ae03a5793b7060e5fe2b2e4` |
| `native/ffi/mlx_bridge.cpp` | `efc08fcbe039e24a5cc0dac7e9483ae612a8a09fef2e285559262c13aab4ccc8` |
| MLX installé `kernels/unary_ops.h` | `0a5492b65ae39ecb6d8b04e64ea5007e0a4ff60d8d9428559bfbfd03387ece2a` |
| MLX installé `backend/common/metal_kernel.h` | `b64ca4cebcba530d8ec1e0a0f23b57313507a37d05e3615ded3796b563cc83fc` |

Aucune correction ni option alternative n'est validée à ce stade. Le
contre-exemple physique et les contrôles affectés restent à examiner.
