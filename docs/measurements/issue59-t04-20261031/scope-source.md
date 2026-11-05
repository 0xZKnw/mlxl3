# ISSUE59-T04-SCOPE-01 — frontière SwiGLU → préparation down

31 octobre 2026. Lecture seule après le préenregistrement placé en tête
d'`opti.md`, historique POSTPROFILE05-C et revue T04 du29 octobre relus.
Aucun code exécutable modifié, aucun build/test/proveur/GPU lancé.

**Premier candidat précis : reprendre sélectivement la fonction compilée à
trois entrées de PR61, corrigée en40f45f80, uniquement sur les slots non
masqués et non groupés.** Garder H128 natif et tout le calcul EXL3. Il existe
une frontière de graphe distincte ; son nombre de lancements effectivement
supprimés et son gain restent à capturer/mesurer après qualification numérique.

## Chaîne réelle et contrat

`native/src/ssd_moe.rs:344–406` : GU EXL3 commun → vues gate/up →
`Array::swiglu` → down. Le SwiGLU est déjà une closure `mx::compile(..., true)`
qui calcule `(gate * sigmoid(gate)) * up`
(`native/src/array.rs:1328–1329`, `native/ffi/mlx_bridge.cpp:569–573`).
Refusionner son intérieur seul ne supprime pas une nouvelle frontière.

`expert_slots` conserve capacity, slot_major, masked et groupes
(`linear.rs:1322–1355`). Le down effectue ensuite :

1. `astype(F16)` de l'activation ;
2. produit avec `suh.astype(F16)`, résultat F16 ;
3. reshape `[n,5,128]`, H128 MLX avec le facteur historique, flatten ;
4. trellis K2/MUL1, réduction des splits stock, cast F16, H128 de sortie,
   produit avec `svh` F16, puis restauration du dtype d'activation.

Sources : `linear.rs:1423–1433,1441–1445,1608–1624`.
Les banques sont GU `[capacity,2,160,40,32]`, down
`[capacity,1,40,160,32]` UInt16 ; scales F16
(`ssd_moe.rs:198–205`). Les splits slot-major viennent uniquement de
`reference_splits(input_tiles,output_tiles)`, sans dépendance au nombre de
routes : GU160→40 donne1split, down40→160 donne1split sur ces dimensions
(`flash_support.rs:286–294`). Ne pas déplacer leurs réductions pour ce candidat.

Les slices GU et reshapes sont des expressions de vues, pas une preuve de
copie/dispatch. La frontière candidate est **la sortie de la closure SwiGLU
puis le produit d'entrée down extérieur à cette closure**. En F16 le premier
cast est un no-op ; le produit reste extérieur. Étendre la closure peut
éviter de matérialiser son activation seule, sans intégrer H128.

Les callers réels utilisent `routed_input(...).astype(F16)` :
Hot anticipé `ssd_moe.rs:551–582`, Hot compact917 et Cold1012/noHot1060
via le même `routed_x` obtenu830–831. Donc un x MoE extérieur F32 ne suffit
pas à conclure GU F32 : ces routes sont converties F16 avant GU. Les appels
directs/composants peuvent néanmoins exercer F32 ; le helper doit conserver
ce contrat et le dtype de sortie au lieu de l'abandonner silencieusement.

## Raccord minimal recommandé

Une closure C++ dédiée reçoit gate, up et down.suh déjà rassemblés. Elle
reprend exactement la chaîne SwiGLU actuelle dans le dtype GU (F16 ou F32),
la convertit explicitement en F16, puis calcule un produit F16 avec suh F16.
La seule sortie est ce produit. Aucun changement de gather, ordre des routes,
Hadamard, cache, banques, `last`, soumission Hot ou attente I/O.

Le helper Rust fait ensuite le H128 natif, puis appelle
`expert_mapped_layout` avec `input_pretransformed=true`, `slot_major=true`,
`return_raw=false`, projections1/K2/MUL1, flags stables préservés. Restaurer
le dtype GU après le résultat down. **Ne pas utiliser `expert_mapped` public** :
ce wrapper force slot_major=false et perd les métadonnées SSD
(`linear.rs:1157–1192`). La voie correcte est déjà écrite dans PR61.

Avant de demander la closure, prévalider entièrement : n1..80, capacity
valide et égale au trellis, slots UInt32 validés, `!masked && groups.is_none()`,
GU `[n,2,640]` F16/F32, suh `[n,640]` F16, svh `[n,2560]` F16,
trellis UInt16 `[capacity,1,40,160,32]`. Ne pas exiger n≤capacity : les
occurrences peuvent répéter un expert. Le helper PR61 ne vérifie initialement
que rang/capacity du trellis ; les autres gardes trellis/svh arrivent dans
`expert_mapped_layout` **après** la demande SwiGLU. Les remonter avant cette
requête est une petite correction de frontière de validation utile au port.

Les Hot masqués restent stock : leur hauteur physique peut être n10/n20
même avec zéro hit logique ; `resident_scales` fournit les zéros pour les
slots absents, et le kernel trellis saute leur banque avant lecture
(`linear.rs:1268–1317`, `_qmv_mapped_tile_kernel.metal:110–117`).
Les groupes partagés restent stock également : leur restauration par
origins/restore est distincte (`linear.rs:1584–1602`). Le prédicat non
masqué/non groupé inclut aussi l'ancien Hot compact et le cas sans Hot ;
ne pas étiqueter tous ses appels « Cold » sans préciser ce sens physique.

## Implémentation existante à reprendre sélectivement

Archive locale : `docs/measurements/issues-triage-20261030/pr-61/changes.patch`,
tête archivée `a36ba5077bf386466660d6b478058fbe6880245c`. Ce fichier concatène
des patches sans noms : **ce n'est pas un patch applicable globalement**.

| Bloc local | Contenu |
|---|---|
| 3007–3020 | `mlxl3_array_swiglu_scaled`, closure3entrées |
| 3060–3084 | `Array::swiglu_scaled`, garde/FFI |
| 3510–3570 | prédicat slots et `expert_slots_swiglu_down` slot-major |
| 3728–3744 | branche `calculate_slots`, gathers down inchangés |
| 3278–3380 | fixtures numériques + vraie projection down, non exécutées sur M5 dans cette PR |

Comparaison primaire fichier par fichier entre les refs4d34e61c et40f45f80
(pas de récupération de PR entière) :

| Fichier | Blob avant | Blob corrigé | Différence T04 pertinente |
|---|---|---|---|
| `native/ffi/mlx_bridge.cpp` | `112bc7ada8f9f37d0119e76c23e1f9a5648e27e5` | `dbf982c5727dc98ee9a426edf49015940537d322` | Formule identique ; commentaire dtype précisé. |
| `native/src/array.rs` | `ede20123020cb46e7c73bc91e7f8395d355c9167` | `262f1a2626e6a046f3760c777a753a63ea019484` | Garde F16 seulement → F16/F32, up.dtype=gate.dtype ; scale reste F16. |
| `native/src/linear.rs` | `80af6af3f6aa5cdc3c35c88ea890439eeffbd2be` | `565972edd77fc90288a566a43c9865021d2b8f6b` | Ajout `.and_then(|output| output.astype(gu.dtype()))` : le xhat prétransformé est F16, il ne doit pas imposer ce dtype à l'API down. |

Sources : [FFI corrigée](https://github.com/0xZKnw/mlxl3/blob/40f45f80/native/ffi/mlx_bridge.cpp#L607),
[garde corrigée](https://github.com/0xZKnw/mlxl3/blob/40f45f80/native/src/array.rs#L1313),
[helper corrigé](https://github.com/0xZKnw/mlxl3/blob/40f45f80/native/src/linear.rs#L1324).
Ne prendre ni le T17 de cette PR (norme F16 incompatible au raccord local),
ni son T18, workflow ou journal global.

## Différence historique et risques à vérifier

POSTPROFILE05-C fusionnait gather/scale dans un kernel banké, avait corrigé
huit reshapes ajoutés, puis restait négatif en release après audit. T04 garde
ces gathers et élargit la closure SwiGLU : frontière et adaptation différentes.
Ne pas relancer scale_rows, remplacer H128, ou renommer l'ancien port.

Turbo345bee9e35920ee77292055d873fca891146529c sert d'inspiration de frontière,
pas d'oracle arithmétique : `fin7_4.metal` blob
`1c4a31c391004807f1ce8e5cd9b40b22efdb5254` utilise d'autres placements de
scales/casts/H128 ; ses variantes de sigmoid diffèrent entre decode/préfill
(KERNELS blob `875e6cb836d4c4d8bfdd8c569f66301536fbbdef`, lignes61–69).

Privilégier la closure MLX à un nouveau shader custom : elle réutilise les
opérateurs et types compilés actuels au lieu de recopier sigmoid et les
arrondis half. Cela ne prouve pas l'identité du JIT élargi : vérifier les
produits **avant H128**, puis H128/down complets, sorties non vides/finies,
F16/F32, strides, doublons, n1/2/10/20/80 et frontières d'arrondi. Une
mutation qui retarde le cast F16 doit être détectée. Capturer ensuite la
frontière stock/candidate réellement évaluée avant d'annoncer un lancement
supprimé ; aucune compilation/GPU ni preuve de ces propriétés dans cette revue.
