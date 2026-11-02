# T04 — revue du raccord avant qualification physique

31 octobre 2026. Lecture seule du port courant repris sélectivement de PR61 (`a36ba5077bf386466660d6b478058fbe6880245c`, correction T04 `40f45f8016ef68881bad9b6344fc6022b28fc038`). **Aucun défaut concret de raccord identifié dans les sources lues.** Aucun build, test, GPU ou proveur lancé pour cette revue ; la parité et la réduction de dispatch restent à qualifier physiquement.

| Frontière | Contrat vérifié en source |
|---|---|
| `array.rs:1343–1364` | Gate/up de mêmes forme et dtype F16 ou F32 ; scale F16, même forme ; vraie garde CPU rank 2, rows 1..80, largeur 640, avant FFI. Le compteur test compte des requêtes, pas des dispatchs. |
| `mlx_bridge.cpp:625–635` | Même sigmoid et deux produits que le SwiGLU stock, dans le dtype GU, puis cast activation F16, produit scale F16, sortie F16. Pas de promotion F32 ajoutée ni formule sigmoid réécrite. Closure compilée thread-local, erreurs propagées par `protect`/`Array::output`. |
| `linear.rs:1325–1376` | Slots UInt32/capacité 1..512, GU `[n,2,640]`, scales F16 `[n,640]`/`[n,2560]`, trellis I16/U16 `[capacity,1,40,160,32]` ; routes non masquées/non groupées, n≤80. H128 MLX exécuté une fois avant `input_pretransformed=true`. |
| `ssd_moe.rs:387–403` | Flag strict 0/1 lu au chargement, défaut OFF ; nouvelle branche seulement si le plan CPU est compatible. Chemin Hot masqué et plans groupés gardent leur référence. « Cold » décrit cette admissibilité, pas uniquement les misses : des hits CPU peuvent y passer. |

Les slices gate/up gardent les valeurs et strides du GU réel. Le helper applique le même H128 F16 avec le même facteur, puis réutilise `expert_mapped_layout` slot-major, K2/MUL1, sans masque/groupe, `return_raw=false` et même `stable_shapes`. Les dimensions et le calcul des splits de référence ne changent pas. La réduction des plans, le H128 de sortie et le produit `svh` restent stock. L’entrée prétransformée est F16 ; le cast final vers `gu.dtype()` restaure correctement la sortie F16 **ou F32** du chemin antérieur. Il n’y a ni H128 supprimé/doublé ni seconde multiplication par `suh`.

Les indices viennent de `ExpertSlots::new/grouped` validés CPU ; aucun readback d’indices n’est ajouté. Les deux gathers down.suh/down.svh existent déjà, avec les mêmes banques et routes. La nouvelle branche les construit avant la fermeture compilée, sans soumission/attente explicite. Aucun `eval`, synchronize, lecture disque ou changement de cache/lease/ordonnancement Hot/Cold n’est ajouté. Les tableaux C++ du graphe possèdent leurs dépendances : les slices temporaires peuvent disparaître à la sortie Rust.

## Coûts et différences avec PR61

Le port ajoute des validations de métadonnées déjà présentes côté Rust et une entrée scale à la closure compilée. Un coût concret supplémentaire apparaît : après H128, la vue `[n,640]` nécessaire au helper est ensuite aplatie par le chemin prétransformé, soit **un reshape intermédiaire de plus** que le stock. Une vue n’est pas une copie GPU démontrée. À n=20, un buffer logique `[n,640]` vaut 25 600 octets F16 ou 51 200 octets F32 ; le nombre réel de buffers économisés, donations et copies de contiguïté n’est pas mesuré. Le cache de compilation/traçage et les strides doivent être observés avant attribution d’un gain.

La formule FFI est celle du T04 corrigé de PR61. Le local renforce les gardes PR : borne 80 via la vraie fonction CPU, métadonnées complètes avant toute soumission, lecture sûre de la première dimension et compteur de passage. Il conserve son T17 séparé OFF et n’importe pas T18/dynamic readers. Cette revue ne constate aucun changement accidentel de la précision ou du budget experts.

La correction source des casts ne prouve toutefois pas leur matérialisation dans le kernel compilé. Le vrai candidat mono-sortie doit être comparé au stock `swiglu → astypeF16 → mul(suh)`, notamment avec GU F32 aux frontières d’arrondi, scales non unitaires, strides et saturation. Un oracle instrumenté à plusieurs sorties ne suffit pas à qualifier ce codegen.

Preuves déjà obtenues par root : Kani sur la garde CPU réelle, **166 obligations sans échec, 12 couvertures, unwind 17, rangs 0..5**, sans hypothèse ajoutée ; cela ne prouve pas MLX/Metal. CBMC de la FFI réelle échoue au parsing libc++, exit 6 avant obligations : **FFI non vérifiée**. Aucune conclusion de débit ni de dispatch n’est tirée ici.

Sources relues (SHA-256 au moment de cette revue, pas une nouvelle identité de binaire) :

- `mlx_bridge.cpp` : `836f0ffd0718df2fa050d11d9ad0fedf0865545640ce648398465297cf524fcb`.
- `array.rs` : `3268277f79e32b61b94fe1f0aba1dd2161f76353b857130e26dbfa573c26d931`.
- `linear.rs` : `10563360ef1201f6704410514ee95a83d5f91831d016b1a27ae4fc68a4bcc74a`.
- `ssd_moe.rs` : `e051599998ad1e54f8711c9e771a77dcb6f032581522edbc22cace9dca5fc3e6`.
