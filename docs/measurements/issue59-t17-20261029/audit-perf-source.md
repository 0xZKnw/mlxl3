# T17-AUDIT-PERF-01 — cache de factories et coût du raccord

31 octobre 2026. Revue de source seulement, après lecture de `opti.md`
et du pilote128. Aucun build, test, proveur, GPU ou mesure supplémentaire
lancé ; aucun fichier exécutable modifié. Cette note examine une cause
possible du prefill DFlash HOLD plus lent, sans lui attribuer cette cause.

**Aucune saturation ni reconstruction répétée de factory n'est démontrée
par les artefacts du pilote.** Le mécanisme d'éviction existe ; les variantes
réellement possibles sont beaucoup plus limitées qu'une variante par expert,
couche, token ou position. T17 ajoute deux clés atteignables dans ce protocole
B1/B2, pas une clé à chaque appel. Le cache Metal interne de MLX est distinct
du cache128 de factories. Aucune correction de production n'est proposée sur
la seule base de cette revue.

## Périmètre et identités

Pilote : même CLI `202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa`,
flags T17 0/1, DFlash fixe WIDTH1/ADAPTIVE0/Q4/RING, cible
`Qwen3.8-Flash-Next-exl3-205-65c89531`, draft `a22a1744`, cache3Gio/53slots,
Hot/HotWide/Unique actifs, SHARED_EXL3=0, WEIGHTED_EPILOGUE=0,
LOOKAHEAD_DEPTH=0. Les plans ne définissent pas STABLE_EXPERT_KERNELS ; son
défaut est0 (`native/src/ssd_moe.rs:127–131`).

Sources relues, SHA256 recalculés :

| Source | SHA256 |
|---|---|
| `native/ffi/mlx_bridge.cpp` | `efc08fcbe039e24a5cc0dac7e9483ae612a8a09fef2e285559262c13aab4ccc8` |
| `native/ffi/kernel_cache_key.h` | `05f1deaea931c29b9340a72a747f966fe732c999bb64a9fb4ee8e9fb5d27965f` |
| `native/src/array.rs` | `e19a7c0929231b60fd944ee5907724c34208995dcbdf82eeb0a88d17bb788378` |
| `native/src/linear.rs` | `42106754300edc10d4b73b24a63720681626805acd0d97e27737642b973be18e` |
| `native/src/flash_draft.rs` | `ef63dc83ba9e5c74c76d26bda7b286e1a717bdadcf90b6496835d0d40131b4d0` |
| `native/src/flash_draft_runtime.rs` | `73b978a4d0f51384b70fa19e906525c5ccfdaf5472cabf6015cfc2ec2061a41c` |
| `native/src/gdn_epilogue.rs` | `ab0e52de616869f2aa14aa1dd6cdbc7e8dcb12550ae03a5793b7060e5fe2b2e4` |
| `native/shaders/flash_gdn_epilogue.metal` | `1fa5b6ae214104186021ff2aa43a5e7b083e838fb0a410cf886326aaf8f9ca44` |

Les résultats de qualité/capture et le détail numérique restent dans les
rapports précédents ; cette revue ne les étend pas. Les comparaisons de
temps du pilote sont descriptives et ne sont pas utilisées comme mesure
du temps lookup/factory/JIT.

## Cache128 : mécanisme réel et limite de l'explication

`native/ffi/mlx_bridge.cpp:681–717` contient un `thread_local std::map`
borné à128 entrées. Sa clé contient exactement nom, noms des entrées,
noms des sorties, header et corps source. Elle ne contient ni adresses
des poids ni index de couche ni grille ni formes des sorties.

`native/ffi/kernel_cache_key.h:29–54` compare d'abord le nom lexicalement,
puis les autres champs dans cet ordre. Sur miss avec128 entrées,
`cache.erase(cache.begin())` retire donc la plus petite clé, sans consulter
sa fréquence ou son ancienneté. Si le working set dépasse128, une famille
de petit nom peut effectivement être reconstruite répétitivement. Dans
ce cas les noms `mlxl3_flash_draft_top16...` précèdent notamment
`mlxl3_flash_gdn...`, `mlxl3_flash_ring...`, `mlxl3_resident...` et
`mlxl3_rs...`.

Cela fournit un mécanisme conditionnel, **pas sa condition d'activation** :
le pilote n'enregistre ni cardinalité maximale, ni hits/misses, ni évictions,
ni durée de construction. `reset()` du modèle ne vide pas cette map
(`qwen4.rs:1680–1690`, `flash_draft_runtime.rs:82–92`) ; elle peut donc garder
les clés de MAIN pendant HOLD. Cependant, le fait que HOLD passe après MAIN
ne prouve ni que128 clés aient été atteintes ni que T17 change les victimes.

Le cache est par thread appelant. Le bridge traite ses requêtes dans sa
boucle stdin (`native/src/main.rs:3122`), tandis que la map est consultée
à la construction du graphe dans `Array::metal_kernel`, et non dans les
workers de lecture SSD. Les count totals d'un processus ne seraient pas
automatiquement la cardinalité d'une seule map en présence d'autres
threads appelants ; aucune instrumentation correspondante n'est ici ajoutée.

## Variantes utiles pour cette campagne

WIDTH1 produit deux lignes de draft et deux lignes vérifiées ; la fin de
budget peut revenir à `target.forward(anchor)` sans appeler le drafter
(`flash_draft_runtime.rs:115–146`). Les profils du pilote confirment uniquement
la case drafted_widths[1]/verified_widths[1] non nulle.

| Famille | Variation de clé externe dans ce protocole | Source |
|---|---|---|
| T17 | 2 clés, ROWS1 et ROWS2 ; les noms d'inputs et header changent avec ROWS. Aucun offset, couche ou valeur dans la clé. 8 clés au maximum sur tout le domaine API1..8. | `gdn_epilogue.rs:75–100` |
| Ring split/reduce | 2 clés ici, R2 ; 16 sur tout domaine1..8. La position est dans `params=[position,2048]`, et non dans header/name. Les cinq couches réutilisent la même paire. | `flash_draft.rs:744–808` |
| Top16/merge draft | 2 clés constantes ; les headers ne contiennent pas rows/position. Les formes/grilles varient au lancement. Ici logits[1,248320]. | `flash_draft.rs:559–576,811–849` |
| Projections Q4 draft | 0 clé dans cette map : `Draft::linear` appelle `Array::affine4`, puis `mx::quantized_matmul`. Les kernels/caches natifs MLX correspondants sont un autre mécanisme. | `flash_draft.rs:339–351`, `mlx_bridge.cpp:626–634` |
| Experts Cold | En-tête non stable : LOCAL_OUTPUT_DIMS dépend de n. Pour B1/B2, n∈1..20 et deux projections GU/down : au plus40 combinaisons de cette seule dimension. Pas512×48 variantes : poids, IDs et couche ne font pas partie de la clé. | `ssd_moe.rs:344–401`, `linear.rs:1487–1537` |
| Experts Hot masqués | Hauteur physique10 ou20, GU/down : au plus4 clés pour ces formes ; capacity53 constante entre couches et appels. Le nombre de hits logiques ne modifie pas cette hauteur physique. | `ssd_moe.rs:551–582`, `linear.rs:1479–1537` |
| Projections EXL3 résidentes | `MLXL3_MATRIX_ROWS` et nom changent avec le batch ; les dimensions des poids restent constantes. Inventaire métadonnées du checkpoint :9 géométries EXL3 hors experts/vision pour le backbone+tête. B1/B2 donnent au plus18 combinaisons de cette dimension, pas une par couche. | `linear.rs:368–430,497–516,605–645` |
| RoPE cible | Spécialisation par dimensions/heads/width ; offset transmis par tableau position. Pas de clé par token. | `qwen4.rs:554–597` |
| QSA scores | Une variante par BLOCKS existe réellement, mais seulement après `(offset+1)/4>512`. Non atteinte avec prompts87/82 et128 sorties ; ne peut expliquer ce pilote. | `qwen4.rs:630–654` |

Les compteurs40/4/18 ci-dessus sont des bornes de familles issues du code,
pas une collecte exhaustive de toutes les clés d'un processus. Ils ne
démontrent ni une occupation globale précise ni une absence d'éviction.
Les autres familles (HC, router, GDN récurrent, resident_slots/scales)
ont aussi des spécialisations ; aucune nouvelle variante illimitée liée
à la position du drafter n'a été trouvée. Les sources ne fournissent donc
pas un motif concret affirmant une saturation provoquée par T17.

Le paramètre STABLE_EXPERT_KERNELS déplace précisément le stride dépendant
de n vers les métadonnées runtime (`linear.rs:1487–1508,1543–1558`). Ce
n'est pas une nouvelle optimisation :67S l'a déjà implémenté et comparé,
64→12 triplets metadata dans sa fixture, sans mesure de misses ; écran
OFF contradictoire puis interruption mémoire MTP, aucun gain qualifié.
`opti.md:15333–15400` conserve ce résultat et l'option par défaut0. Une
reprise identique ne serait pas justifiée par ce seul soupçon.

## Une factory reconstruite n'est pas une compilation Metal démontrée

Source primaire MLX0.32.2 :

- [`mlx/backend/common/metal_kernel.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/metal_kernel.cpp),
  blob `7a2eef51fcfa04d1e6828ca72be100235fded40f`, lignes222–375 : la factory
  analyse les noms/attributs ; chaque invocation de sa closure génère encore
  la signature et un objet `CustomKernel`. Les dtypes, le cas scalaire et
  les petites entrées de moins de8 éléments peuvent modifier le nom Metal
  généré (310–338), sans créer une nouvelle entrée du cache C++ externe.
- [`mlx/backend/metal/custom_kernel.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp),
  SHA256 de la copie historique relue
  `92548fba86690996a30725ef96bb8e0e299b69a4559e71c60340e7e11c1cac5e`,
  lignes48–56 : lib_name dépend du nom, hash de source générée et options ;
  appel séparé à Device::get_library puis get_kernel.
- [`mlx/backend/metal/device.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp),
  blob `2f25f894e494a4328886148bb6e3332d9f9b298a`, lignes800–818 et843–898 :
  library_map_ et library_kernels_ mémorisent bibliothèques/pipelines avec
  lookup avant création. La limite128 précédente ne leur est pas appliquée.

Une éviction de la closure externe peut donc coûter de l'analyse/copie CPU
tout en réutilisant sa bibliothèque et son pipeline. Aucun compteur ne
permet d'attribuer à cela une partie des secondes de prefill HOLD.
L'inverse est aussi important : un hit externe n'évite pas tout le travail
de génération de signature et de construction de graphe.

Antécédent19 (`opti.md:133–154`) :819270 appels,58 misses initiaux,
0 éviction,58/128 entrées ; lookup715,40ms+factories4,23ms sur177,95s
instrumentées, environ0,40% agrégé. Parcours historique OFF/MTP/cache5Gio,
avant HotWide et DFlash : cette mesure ne prouve pas le même résultat ici,
mais interdit de présenter la map128 comme un coût critique déjà connu.

## Copies, allocations, donation et frontières

- Le raccord T17 ne contient aucun `eval`, `async_eval`, `synchronize`,
  readback ou accès SSD (`gdn_epilogue.rs:58–101`). Les attentes Hot, routes,
  lecteur et drain sont dans les chemins communs inchangés. La réduction
  de dispatchs ne supprime pas automatiquement ces attentes ni les seuils
  MLX40ops/40MB ; leur effet de regroupement est une hypothèse à profiler.
- B2 garde une norme native FP32 par ligne et passe directement ses deux
  buffers ; aucune concaténation FP32 ajoutée (`qwen4.rs:412–438`).
  Le stock créait ses deux sorties F16 puis les concaténait ; T17 produit
  directement [2,6144]. Les dépendances des deux normes arrivent désormais
  au même kernel ; cela peut changer la planification/lifetime sans nouvelle
  barrière hôte explicite. Son coût n'est pas mesuré séparément.
- La FFI demande `ensure_row_contiguous=true` (`mlx_bridge.cpp:702–703`).
  MLX CustomKernel::eval_gpu copie effectivement une entrée non row-contiguous
  avant exécution (lignes31–45 du source primaire ci-dessus). Les résultats
  de norme et le cast F16 de z sont les mêmes calculs dans les deux bras ;
  aucune nouvelle copie de production n'est démontrée par les formes seules.
  Les tests stridés valident la correction, pas l'absence ou la durée des copies.
- Chaque appel Rust alloue encore CString/Vec/pointeurs et métadonnées
  (`array.rs:1432–1476`), la FFI crée ses vectors et handles
  (`mlx_bridge.cpp:706–715`), puis MLX écrit la signature. Le lookup emprunté
  du cache évite la copie permanente de clé sur hit, pas tout ce travail.
  T17 remplace plusieurs appels d'opérations par un tel appel custom ; le
  différentiel CPU réel n'a pas été isolé dans le pilote.
- CustomKernel alloue sa sortie via allocator::malloc, sans donation
  (`custom_kernel.cpp:24–29`). **Il serait faux d'en déduire une allocation
  finale ajoutée** : le stock finit lui aussi par un cast F32→F16, et
  `is_donatable` exige la même itemsize. Source
  [`common/utils.h:218–223`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/utils.h),
  blob `c6d7820619411998637bebc614fa8e674c7f4041` ;
  [`common/copy.h:25–47`](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/copy.h),
  blob `859ce0410e853d4d28d3fa2a5ca06c1d7f1d3d78`.
  Les intermédiaires stock F32 peuvent réutiliser un buffer selon propriété
  de donation/refcount, ce qui ne rend pas gratuite la sortie F16 finale.
- `volatile probability/product` et `precise::exp` restent dans le shader
  (lignes32–35). Leur code machine/occupation/éventuels spills ne sont pas
  connus ici. DIAG-02 a montré que retirer volatile seul gardait le défaut
  numérique et que precise::exp corrigeait le contre-exemple ; son coût
  peut être étudié, mais aucun retrait numérique n'est justifié à l'aveugle.

## Conclusion bornée

Le cache externe peut théoriquement churner au-delà128 clés et son éviction
n'est pas LRU. Ni les profils existants ni une source avec des headers
variables ne prouvent ce churn. Le drafter ne crée pas une clé par position,
ses Q4 ne passent pas par cette map, et T17 ajoute au plus deux clés dans le
pilote. Aucun fallback silencieux, concatF32, sync/readback supplémentaire
ou nouvelle allocation finale obligatoire n'a été trouvé dans le raccord.

S'il faut encore départager cette hypothèse, la plus petite observation
utile serait des compteurs opt-in par thread de calls/misses/evictions/
max_entries et durées lookup/factory, segmentés avant/après MAIN puis HOLD,
sans changer clés/éviction/shader/budget. Ce serait un nouvel essai à
préenregistrer, pas une modification autorisée par cette note ; ses temps
instrumentés resteraient exclus d'une qualification de gain. Une capture
GPU ou mesure de génération de graphe serait nécessaire pour distinguer
le coût du kernel précis/volatile, les copies réelles et la planification
des40ops. Le ralentissement HOLD demeure **observé, causalité non isolée**.
