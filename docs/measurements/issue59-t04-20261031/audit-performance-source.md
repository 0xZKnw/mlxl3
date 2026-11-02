# T04-AUDIT-PERF-01 — adaptation et coûts de source

31 octobre 2026. Audit lecture seule préenregistré dans `opti.md`, pendant
la campagne isolée du parent. Aucun test, build, preuve, profilage, GPU ou
inférence supplémentaire lancé. Seul ce document est écrit.

**Aucun défaut concret d'adaptation retrouvé.** La frontière supprimée est
réelle ; son bénéfice global n'est pas acquis. Les deux premiers processus
AR128 sont des observations intermédiaires, pas une décision de performance.

## Identité et chemin réellement configuré

Les rapports finalisés `prepared/runs/t04-pilot128-AR-{1-A,2-C}/report.json`
annoncent le même CLI
`81bc3a14d509224d77fd6d98d8dc64f9b840c0094f7376f5f004e924849732cd`,
le même chemin `build/issue59-t04-20261031/engine/prototype-engine`,
T04=0 puis1, GDN=0, T05=0, SHARED_EXL3=0, Hot/Wide/Unique=1 et cache3Gio.
MAIN87/HOLD82,128 sorties/127 transitions ; leurs débits ne sont pas
utilisés ici comme mesure de temps de closure, copie ou allocation.

Les quatre fichiers actuels ont été rehashés et correspondent au gel
`final-identity-r3.json` :

| Source | SHA256 |
|---|---|
| `native/src/linear.rs` | `10563360ef1201f6704410514ee95a83d5f91831d016b1a27ae4fc68a4bcc74a` |
| `native/src/ssd_moe.rs` | `e051599998ad1e54f8711c9e771a77dcb6f032581522edbc22cace9dca5fc3e6` |
| `native/src/array.rs` | `3268277f79e32b61b94fe1f0aba1dd2161f76353b857130e26dbfa573c26d931` |
| `native/ffi/mlx_bridge.cpp` | `836f0ffd0718df2fa050d11d9ad0fedf0865545640ce648398465297cf524fcb` |

`ssd_moe.rs:138–140` lit strictement le flag une fois par chargement, défaut0.
`calculate_slots:387–400` appelle le helper si flag1 et slots non masqués/non
groupés. Il n'y a pas de fallback silencieux à l'intérieur du helper : les
préconditions refusées donnent une erreur.

Les vraies routes sont F16 via `routed_input`, même si le x MoE extérieur est
F32. Les routes Cold et le chemin sans Hot créent des `ExpertSlots::new` avec
SHARED_EXL3=0 ; ils passent donc dans le candidat. Le Hot GPU anticipé est
masqué et reste stock ; les groupes partagés restent stock. L'ancien Hot
compact non masqué peut aussi passer dans le candidat : « Cold » désigne ce
périmètre de slots, pas un test de cache miss à chaque invocation.

Les compteurs de passage sont `cfg(test)` et absents du CLI. Leur non-vacuité
et les fallbacks ont été qualifiés par les contrôles physiques conservés,
dont le MoE isolé (`model-quality-inspection.json`), et non déduits du seul
texte greedy. Le pilote release ne fournit pas de comptage par appel T04.

## Fidélité à PR61

Patches individuels archivés `pr61-linear.patch`,
`pr61-native_src_ssd_moe.rs.patch` et `pr61-native_ffi_mlx_bridge.cpp.patch`,
tête `a36ba5077bf386466660d6b478058fbe6880245c`, correction dtype
`40f45f8016ef68881bad9b6344fc6022b28fc038`.

- La formule C++ locale (`mlx_bridge.cpp:625–634`) est identique au bloc T04
  corrigé : SwiGLU dans le dtype GU, cast F16 de l'activation, produit avec
  suh F16, sortie F16. Seul son commentaire a été raccourci.
- Le helper (`linear.rs:1325–1383`) garde H128 MLX, K2/MUL1,
  `slot_major=true`, `input_pretransformed=true`, `return_raw=false`, les
  splits/réductions stock et `astype(gu.dtype())` final de la correction.
  Aucune rotation doublée ou supprimée ; pas de wrapper `expert_mapped`
  non slot-major substitué.
- Les différences locales utiles sont des gardes avant la requête : n1..80,
  capacity1..512, indices UInt32, formes exactes, scales F16 et trellis16bits.
  Les tableaux Int16/UInt16 restent compatibles avec la vue UInt32 stock.
- Les gathers down.suh/down.svh, banques, mapping, ordre de réduction,
  rétention `last`, soumissions Hot et protections I/O restent ceux du stock.
  Aucun T17/T18 de la PR n'a été importé par ce raccord.

La portée diffère de POSTPROFILE05-C : ce dernier avait fusionné
gather/scale, corrigé ses huit vues supplémentaires, puis échoué à qualifier
un gain. T04 conserve ces gathers et étend la closure SwiGLU. Le résultat
historique reste négatif ; il ne peut être recyclé en résultat T04.

## Coûts nouveaux, supprimés ou encore inconnus

La capture physique existante R20 (`xcode-dispatch-inspection.json`) montre
stock3 → candidat2 dispatchs : SwiGLU compilé puis MultiplyF16 puis H128,
contre SwiGLU+scale compilé puis **même H128**. Les deux kernels compilés
sont `strided_2`, même grille[640,20,1] et threadgroup[64,16,1]. Elle ne
mesure pas leurs durées ni le coût complet par token.

Le candidat ajoute un troisième buffer d'entrée à une closure `mx::compile`
au lieu des deux entrées SwiGLU, tout en retirant le `mul(suh)` séparé. Cela
modifie lecture/indexation et graphe généré. Il ne démontre pas à lui seul
un coût supérieur. La capture ne montre pas un passage de vectorisé à stridé :
le kernel SwiGLU stock était déjà stridé. Le travail CPU de tracing/cache
`mx::compile`, ses variantes et ses durées ne sont pas comptés ici.

Un reshape de plus existe, également dans PR61 : H128 → `[n,640]` pour
entrer dans `expert_mapped_layout`, puis flatten dans la branche
prétransformée (`linear.rs:1353–1356,1481–1482`). C'est un appel de vue et de
métadonnées supplémentaire, **pas une copie physique démontrée**. Le cast
F16 de xhat et le cast final vers GU F16 sont des identités dans le parcours
F16 ; le cas GU F32 conserve la restauration exigée.

Les autres gardes sont des lectures de métadonnées Rust déjà possédées.
Le compteur test-only n'ajoute aucun travail dans le CLI. La FFI passe un
vector3inputs au lieu de vector2inputs ; les vues et le gather de scale
existaient déjà. Aucun tableau de poids ou buffer résident nouveau, aucune
lecture CPU, `eval`, `async_eval`, fence ou synchronisation ajoutée.

### Contiguïté et donation : distinguer Compiled et CustomKernel

Le chemin T04 utilise **Compiled**, pas le CustomKernel avec
`ensure_row_contiguous=true` du cache de factories. MLX0.32.2
[`metal/compiled.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/compiled.cpp),
blob `e95d7b8d5f1ea879bb12a067bfe04cde7739fd9b`, lignes360–418,
calcule les strides, lie directement les tableaux et leurs strides, puis
alloue les sorties. Aucune matérialisation contiguë obligatoire des slices
gate/up n'est introduite par ce code.

[`common/compiled.cpp:108–165`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/compiled.cpp),
blob `9dc5a4c54cf4c99403a10d4416bb4961dfb2b75b`, autorise la donation
dans les chemins contigus et stridés. Dans le cas stridé, un input de même
taille/type, row-contiguous et donatable peut fournir le buffer de sortie ;
le troisième input scale peut donc être un donneur potentiel. Le stock
Multiply possède aussi ses possibilités de donation. Les refcounts et
donations effectivement choisis ne sont pas observés : **aucune perte forcée
de donation ni allocation de sortie supplémentaire n'est établie**.

Les dépendances et durées de vie changent mécaniquement : scale
devient une dépendance du même kernel que gate/up, et le produit n'a plus
son lancement propre. Leur effet sur disponibilité GPU, donation et seuils
40ops/40MB reste à mesurer ; ce n'est pas une nouvelle barrière CPU.

## Décision de cet audit

Aucune correction numérique, de flag, fallback, banque, arrondi, H128 ou
synchronisation n'est requise par les constats relus. L'extra reshape et la
troisième entrée sont des coûts structurels identifiés, pas la cause mesurée
du premier écart AR. Les supprimer ou changer la frontière constituerait
un nouvel essai à préenregistrer, avec ses contrôles affectés ; cet audit
n'autorise ni modification immédiate ni répétition du même candidat pour bruit.
Attendre la matrice prévue et son analyse avant conservation/retrait ; ne
pas attribuer les anciens12ms d'encodage ni les pages compressées à T04.
