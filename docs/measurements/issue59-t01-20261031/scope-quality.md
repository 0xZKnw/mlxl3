# T01 — tests existants et contrat d'un filtre CPU après hit

31 octobre 2026. Lecture seule après `ISSUE59-T01-SCOPE-01` dans
[opti.md](../../../opti.md). Aucun code, build, test, preuve, benchmark ou GPU
lancé. Le moteur conservé est revenu au CLI `202f96dc…`, MLX 0.32.2 ; T04
est retiré, ses sources/qualités archivées ne valident pas un futur T01.

**Le filtre utile doit viser le travail restant après le hit, pas refaire
le cache factory128 ni la construction du préfixe déjà optimisée par 80M.**
Un gain CPU de construction de graphes non évalués ne sera ni un gain GPU
ni un gain de tokens/s. Le plafond propre à T01 reste non mesuré.

## Couverture réellement présente dans le code

| Entrée existante, non exécutée pour cette revue | Propriété et limite |
| --- | --- |
| [kernel_cache_key.cpp](../../../native/tests/kernel_cache_key.cpp) | Appelle le vrai comparateur de [kernel_cache_key.h](../../../native/ffi/kernel_cache_key.h). 256 fixtures déterministes, **65 536 paires ×4 combinaisons owned/view**, oracle tuple historique ; chaînes vides, octets hauts, listes ordonnées. Compare deux maps sous éviction lexicographique à capacité128, modifie/détruit une clé temporaire après insertion, vérifie les résultats et exige zéro allocation pour 128 hits empruntés. Les valeurs de map sont des **int**, pas de vrais graphes/pipelines MLX. |
| Même exécutable, option `--bench` | Micro all-hit de lookup de clé, chauffe et ABBA/BAAB ; option distincte de la qualité. Il compare notamment l'ancien chemin qui copie les clés au chemin **déjà intégré**. Le relancer ne mesurerait pas la suppression de signature/source après hit. |
| [kernel_cache_key_proof.cpp](../../../native/tests/kernel_cache_key_proof.cpp) | Harnais autour du vrai comparateur : textes symboliques de longueur0..3, listes0..2, oracle tuple et quatre covers. En CI il est seulement **vérifié syntaxiquement**, pas exécuté par CBMC ; sa présence n'est pas une preuve obtenue. |
| `array::tests::metal_cache_rejects_null_key_metadata_before_gpu_dispatch` | Sept cas FFI réels de pointeurs de clé/noms nuls, refusés avant les chemins GPU. Ne prouve pas tous les guards Rust de métadonnées ni la sécurité de pointeurs étrangers arbitraires. [Source](../../../native/src/array.rs:2002). |
| `array::tests::metal_cache_full_key_and_eviction_preserve_outputs` | Test Apple ignoré explicitement : **320 clés /330 requêtes /1 320 valeurs finies exactes**, variations header/source et revisites après >128 clés. **Chaque sortie est évaluée immédiatement** ; pas de graphe pending traversant éviction/libération, pas de clear de pipeline sous travail asynchrone. [Source](../../../native/src/array.rs:2079). |
| `array::tests::async_group_retains_parents_and_rejects_null` | Vues et parents retenus lors de soumission asynchrone ; utile pour la frontière Array, mais ne démontre pas la durée de vie d'un nouveau handle de kernel ou d'une pipeline. |
| `codec::tests::static_reference_header_preserves_legacy_bytes_and_every_position` et `codec::verification::reference_header_is_static_and_exactly_gated` | Préfixe80M réellement servi : bytes historiques, 256positions/permutation, réutilisation du pointeur ; harnais booléen sur la vraie fonction. Le micro `reference_header_cpu_microbench` est distinct et ignoré. Ces contrôles ne portent pas sur la signature complète MLX après hit. [Source](../../../native/src/codec.rs:142). |

Le [workflow actuel](../../../.github/workflows/rust.yml:119) exécute le test
C++ clé sous ASan/UBSan, sans `--bench`, puis compile syntaxiquement le
harnais. Aucun run CI distant du futur T01 n'existe à ce stade.

## Coûts et propriétés à ne pas confondre

[Array::metal_kernel](../../../native/src/array.rs:1402) valide puis recrée
CString name/header/source/noms, vecteurs de pointeurs, formes aplaties et
types **à chaque appel**. Le [pont C++](../../../native/ffi/mlx_bridge.cpp:681)
réutilise déjà une clé empruntée et un `CustomKernelFunction` thread-local,
bornés à128 ; les cinq champs sont name, inputs ordonnés, outputs ordonnés,
header et source. Un hit économise la copie de clé et la factory, pas
nécessairement ce qui suit son appel.

L'[audit primaire T01 antérieur](../issue59-20261029/T01-read.md) localise la
signature/source reconstruite dans la closure MLX, puis le hash/lookups dans
`CustomKernel::eval_gpu`. La source MLX0.32.2 archivée `custom_kernel.cpp`
conserve allocations des sorties, copies de contiguïté nécessaires,
`add_temporaries`, bindings, barrières, clamp de threadgroup et dispatchs.
Préparer ces éléments immuables sans refaire l'encodeur est un périmètre
distinct d'une pipeline préparée complète.

Historique19 : 58 misses initiaux, zéro éviction sur819270 appels ; il n'a
pas chronométré tout `write_signature` après hit. 80M a déjà remplacé le
préfixe1006octets par un littéral et obtenu sa propre qualification : ne pas
le présenter comme nouveau T01. POSTPROFILE03 a échoué fonctionnellement
en supprimant une soumission Hot précoce ; **préserver les deux événements
de disponibilité et les tests d'overlap**, sans convertir les12,142ms du
profilage en budget récupérable du nouveau filtre.

## Plus petit contrat à préenregistrer avant implémentation

Fixer une **signature Hot GU réellement capturée** et garder la délégation
à l'encodeur stock. Préparer uniquement description/variante immuables ;
chaque appel doit consommer les entrées courantes, produire des sorties
distinctes et créer un nouveau nœud de graphe. Aucun Array d'activation,
sortie ou slot bancaire ne doit être caché comme configuration. Le handle
Rust doit conserver les restrictions de thread de `Array`, pas introduire
un `Send/Sync` injustifié.

Contrôles à prévoir, via la vraie entrée de production proposée :

- **Identité :** anciens cinq champs complets, formes/dtypes/classes
  scalaire/constant/device, seuil7/8, templates/options, stream/device et
  flags de contiguïté pertinents. Un changement de variante doit produire
  un miss ou un refus explicite ; ne pas réutiliser un ancien résultat.
- **Ownership :** détruire les chaînes/configurations de création, varier
  les entrées après chaque hit, garder des sorties pending pendant >128
  insertions/évictions, libérer le handle avant eval puis contrôler toutes
  les sorties. Le test actuel à valeurs `int` ne couvre pas cette frontière.
- **Métadonnées :** ajouter des cas ciblés au vrai `metal_kernel` pour
  comptes de noms/outputs, dimensionsgrid/threadgroup nulles/négatives,
  formes impossibles/overflow et NUL intérieur. Plusieurs refus peuvent
  être testés avec **zéro Array**, avant `initialize`, donc sans contexte
  GPU. Ne pas affirmer «avant FFI» sur le seul `Err` sans compteur adapté.
- **Clear et erreurs :** erreur de construction/MSL et reprise, unload,
  clear de bibliothèque/cache, éviction avec consommateurs encore vivants.
  Si T01A délègue à CustomKernel stock, ne pas garder de pipeline empruntée.
  Si une phase ultérieure retient une pipeline, protéger sa résolution,
  son propriétaire et son usage jusqu'à complétion ; le seul lifetime du
  primitive ne suffit pas à supposer cette propriété asynchrone.

Le filtre CPU doit distinguer chauffe/miss et all-hit, vérifier une borne
de variantes/mémoire, compter génération/signature/octet source/allocations
aux postes réellement changés, et détruire les graphes non évalués à chaque
itération pour éviter leur accumulation. Les chemins d'inputs minuscules
peuvent changer leur classe de signature : toute lecture implicite doit
être exclue ou explicitement observée dans un test dit «CPU seulement».

Pour la vérification source, réutiliser ASan/UBSan et le comparateur réel ;
tenter CBMC sur le vrai nouveau helper C++ avec contexte réel et bornes
explicites. L'ancien parseur libc++ incompatible est une limite connue,
pas une preuve : conserver le diagnostic si répété. Kani peut couvrir un
garde Rust réel de formes/tailles/indices, pas une réécriture du comparateur
C++ présentée comme preuve de l'original. Ni preuve CPU ni succès de graphe
non évalué ne prouvent MLX/Metal ou la concurrence.

Après un éventuel signal CPU et seulement sous nouveau préenregistrement,
la qualification physique devra comparer sorties complètes finies/bytes
avec le chemin stock, préserver Hot/Cold/strides/erreurs et états MTP/DF,
puis bridge et mesures release alternées. **Aucune mesure nouvelle ni aucun
gain T01 n'est établi par cette revue.**
