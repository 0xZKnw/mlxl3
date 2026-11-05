# ISSUE59-T01-SCOPE-01 — frontière Rust/FFI après hit

31 octobre2026. Relecture après préenregistrement en tête d'opti.md,
historiques19/67/80M/POSTPROFILE03 et
[T01-read](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-20261029/T01-read.md).
Lecture seulement : aucun code, build, test, proveur, GPU ou benchmark lancé.

**Filtre minimal proposé : une seule variante Hot GU AR, chronométrage borné
de la construction du graphe après hit, puis éventuellement un handle qui
prépare sa signature/source et délègue l'évaluation au CustomKernel stock.**
Le coût récupérable est non mesuré ; les12ms d'encodeur ne sont pas son budget.

## Baseline et chemin observé dans les sources

Worktree exclusif : `/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0`.
Les cinq sources relues correspondent byte/SHA au manifeste190 pré-T04 :

| Source | SHA256 |
|---|---|
| `native/src/array.rs` | `e19a7c0929231b60fd944ee5907724c34208995dcbdf82eeb0a88d17bb788378` |
| `native/src/linear.rs` | `42106754300edc10d4b73b24a63720681626805acd0d97e27737642b973be18e` |
| `native/ffi/mlx_bridge.cpp` | `efc08fcbe039e24a5cc0dac7e9483ae612a8a09fef2e285559262c13aab4ccc8` |
| `native/ffi/kernel_cache_key.h` | `05f1deaea931c29b9340a72a747f966fe732c999bb64a9fb4ee8e9fb5d27965f` |
| `native/src/ssd_moe.rs` | `4aa0b893e3a6c53ee6c5348a1def4d6dce040ffb00ce9c8fdb61990d56f31e0b` |

CLI retenu202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa,
MLX0.32.2, T04 retiré, T05/T17 OFF. Aucun binaire T01 n'existe dans cet audit.

`prepare_resident_hot` → `calculate_slots` → `linear::expert_slots` →
`expert_mapped_layout` → `array::metal_kernel` → `mlxl3_metal_kernel` →
closure MLX → CustomKernel::eval_gpu. Le `slots.indices().async_eval()`
antérieur au calcul puis `output.async_eval()` restent indispensables au
contrat d'overlap testé. Aucun déplacement/suppression proposé.

## Ce qui reste payé malgré un hit

| Frontière | Travail réellement présent | Préparable sans modifier le calcul |
|---|---|---|
| [linear.rs:1479](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/native/src/linear.rs:1479) | Strings du masque, stride, codebook, concat du préfixe de référence et suffixes, nom complet, Vec de forme de sortie. | Nom/header/corps et géométrie de la variante fixe. Ne pas toucher aux vues/H128/entrées. |
| [array.rs:1402](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/native/src/array.rs:1402) | Validation ; CString name/header/source/noms ; Vec des noms/pointeurs/dims/ranks/dtypes/outputs. Tout cela précède le lookup C++. | Descriptions/CString/pointeurs de noms et shapes/dtypes immuables ; conserver les pointeurs d'entrées actuels. |
| [mlx_bridge.cpp:693](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/native/ffi/mlx_bridge.cpp:693) | Lookup emprunté de clé5champs ; après hit, nouveaux vectors shapes/dtypes, byte_count, copies de handles d'entrées, invocation de closure, nouveaux handles de sorties. | Métadonnées fixes et lookup par handle ; chaque appel doit garder ses propres arrays/outputs. |
| MLX closure `metal_kernel.cpp:295–374` | Nom d'après dtype/classe scalaire/constant/device ; `write_signature` réécrit la source ; nouvelle primitive et graphe. | Signature/source/options/shape_infos fixes ; créer néanmoins une nouvelle identité de primitive par appel. |
| MLX `CustomKernel::eval_gpu:20–106` | Allocation des sorties, éventuelles copies de contiguïté, hash source, lookup lib/pipeline, bindings/ressources/barrières/dispatch. | Une préparation de source seule ne retire pas ces coûts. Un launcher à pipeline connue serait une seconde étape distincte. |

Le cache de factories ne possède pas seulement le nom : il compare les cinq
champs, sans copie de clé sur hit. L'analyse des attributs et `*_shape` lors
de la création de factory n'est déjà plus répétée sur hit ; **ne pas l'inclure
dans un temps de travail post-hit évitable**. En revanche le nom typé et
`write_signature` le sont bien. La génération de source est avant l'évaluation
GPU ; le hash est dans eval_gpu. Ces postes ne peuvent être additionnés aux
12,142ms de soumission comme s'ils formaient la même fenêtre.

`Array::owned` lit une fois la métadonnée C++ et copie la shape dans un Vec
Rust (`array.rs:549–577`) ; `shape()`/`dtype()` sont ensuite des accès locaux.
Aucune lecture de valeurs GPU n'est introduite par cette étape. Les copies
de `mx::array` de la FFI portent les handles/refcounts, pas une copie démontrée
du contenu tensoriel. Les reshapes ne sont pas présumés être des kernels.

## Une vraie variante Hot GU bornée

AR, une ligne cible,10routes physiques masquées, cache53 :

- nom : `mlxl3_rs_expert_2560_640_2_2_1_4_1_slots1_resident53` ;
- K2/MUL1, projectionsGU2, slot_major1, masked1, groupes absents,
  stable_shapes0 ; NT1, SG4, splits1 ;
- banque GU UInt16 `[53,2,160,40,32]`, vue trellis UInt32 aplatie ;
- xhat F16 `[51200]`, tile_map et tile_sub UInt32 `[10]` ; ces deux derniers
  arguments sont le **même Array**, alias actuel à préserver ;
- sortie du kernel QMV F32 `[1,20,640]` ; grid `[102400,1,1]`, threadgroup
  `[128,1,1]` ; GU complet garde ses finish stock puis `[10,2,640]` F16 ;
- en-tête `LOCAL_OUTPUT_DIMS=12800`, capacity53 ; aucun index de couche ni
  adresse de poids dans la configuration immuable. Les inputs restent frais.

Sources :
[calculate_slots](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/native/src/ssd_moe.rs:344),
[Hot](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/native/src/ssd_moe.rs:551),
[nom/grille](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/native/src/linear.rs:1522).
Le nombre physique est10 même si seulement certains experts sont dans le
cache. Une route masquée ne permet pas de réduire n au nombre de hits.

B2 donne20routes/40lignesGU, xhat102400, sortie `[1,40,640]`, grid204800,
`LOCAL_OUTPUT_DIMS=25600` : **même nom, autre header/géométrie**. Le premier
filtre peut refuser cette variante et reprendre le stock ; ne pas réutiliser
une clé basée uniquement sur le nom. Les inputs U32 sont tous ≥8 éléments,
donc device dans cette variante ; le seuilMLX7/8 et scalaires reste une
frontière à tester avant toute généralisation.

## Observation CPU proposée, sans nouvelle exécution ici

Préenregistrer un écran séparé qui ne construit que des graphes non évalués
pour cette signature fixe, sans modèle/SSD, `eval`, `async_eval`, synchronize,
MetalCapture ou soumission. Préparer des entrées lazy de shapes/types exacts,
les réutiliser, détruire chaque graphe avant le suivant ; par exemple quelques
chauffes puis8lots×32constructions, deadline et nombre de graphes bornés.
L'initialisation du contexte Metal/MLX reste possible : ce n'est pas un test
Rust pur ni une mesure GPU, même sans kernel volontairement évalué.

Instrumenter la vraie voie, avec compteurs agrégés : temps Rust de formatage,
CStrings/Vec ; temps lookup/hit et shapes/handles C++ ; temps closure incluant
write_signature ; appels/misses/évictions et octets de source. Comptabiliser
la première création séparément des hits, conserver tous les lots et un
contrôle du coût des timers. Les sorties structurelles attendues doivent être
non vides, shape/type exacts et indépendantes par appel. Ne pas appeler cela
parité numérique : aucune valeur n'a été calculée.

Si la construction post-hit constitue un poste utile, le plus petit candidat
suivant est **T01A**, déjà décrit dans T01-read : handle immutable pour cette
unique variante, nouvelle primitive adaptateur/graphe à chaque appel,
délégation à CustomKernel stock. Cela retire marshalling immuable/source,
sans changer hash/copies/allocation/encode/soumissions. La pipeline préparée
type HostGpu n'est pas nécessaire pour ce premier filtre ; elle engage les
contrats supplémentaires de clear_library/stream et rétention jusqu'à la
complétion. Aucun prototype de ces deux étapes n'est appliqué ici.

Le handle doit être choisi **avant** les formats de header/nom et les
CStrings pour supprimer aussi ce travail Rust. L'ajouter seulement après
`array::metal_kernel` laisse ces coûts payés et ne teste que la génération
MLX après hit ; nommer précisément le sous-périmètre évite un port incomplet.

## Antécédents et limites

19 :819270appels,58misses initiaux/0éviction ; lookup+factory≈0,40% des
177,95s instrumentées. Ce résultat ne mesurait pas write_signature/hash et
ne justifie pas un deuxième cache de factories. 67 :64→12clés metadata,
aucun gain qualifié ; ne pas refaire uniquement la stabilisation des headers.
80M a déjà rendu statique le préfixe1006octets ; ce travail et son gain
historique ne s'additionnent pas à T01. POSTPROFILE03 a retiré une soumission
précoce, puis échoué à l'overlap avant tout timing : ne pas la supprimer.

MLX sources primaires v0.32.2 :
[metal_kernel.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/metal_kernel.cpp)
blob`7a2eef51fcfa04d1e6828ca72be100235fded40f` ;
[CustomKernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp)
copie primaire SHA256`92548fba86690996a30725ef96bb8e0e299b69a4559e71c60340e7e11c1cac5e`.
Launcher mlx-serve épinglé déjà relu :
`0f424fb44757643a0ffb5147b9e92b793818b6fe`,
`mlx_launch_cache.inc` blob`0598aa7b1f9c1ca907b89b0f7d3e5fbd8bb093bf`.
Son microbench host-only et pipeline factice ne prouvent aucune accélération
sur notre modèle. Durées par poste, plafond récupérable et gain tok/s :
**non mesurés** dans ce rapport.
