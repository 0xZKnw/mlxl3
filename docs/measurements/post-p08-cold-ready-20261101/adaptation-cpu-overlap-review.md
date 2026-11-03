# Cold Ready — préparation CPU de W2 avant jointure

Audit de source en lecture seule, 1 novembre 2026, pendant GPU1. Aucun changement moteur, compilation, preuve, test, capture ni nouvelle mesure. Relit le P1 et ses résultats de qualification ; les premiers chiffres AR17 ne permettent aucune conclusion de gain.

**Conclusion : le chemin release actuel permet de construire le graphe de W2 après sélection réelle W1/rest et avant `batch.finish()`, puis de conserver strictement toute soumission/évaluation W2 après la jointure réussie.** Cette conclusion concerne ce chemin précis ; elle ne rend pas sûre une évaluation anticipée. L'intérêt est le recouvrement de construction du graphe CPU avec la fin des lectures, sans nouvelle vague ni nouveau kernel. Coût masquable non mesuré, possiblement nul si les lecteurs ont déjà fini pendant `W1.async_eval()`.

## Appels et effets observés

| Chemin actuel | Pendant la construction de W2 | Lecture de poids / exécution différée |
| --- | --- | --- |
| `ssd_moe.rs:383` `calculate_cold` | Crée les petits indices CPU de sélection puis un `take` de `routed_x`. `calculate` crée `ExpertSlots::new` (`linear.rs:1260`), qui vérifie les IDs CPU et crée un autre tableau d'indices. | Aucun readback de routes ni de banques. |
| `ssd_moe.rs:475` `calculate_slots` | Reshape/broadcast, quatre gathers de scales, graphe GU, SwiGLU, graphe Down. | `gather_scales` (`linear.rs:1344`, non masked ici) appelle `bank.take`; il construit le Gather, sans lire les coefficients. |
| `linear.rs:1410` `expert_mapped_layout` | Valide shapes/dtypes; construit casts, produits, H128, reshape/view du trellis et les deux CustomKernel GU/Down; forme les signatures/source. | Les poids et scales ne sont lus par le GPU qu'à l'évaluation des primitives. `is_m5_gpu` (`array.rs:524`) consulte appareil/cache/option, pas les poids. |
| `array.rs:639/1217/1561`, pont `mlx_bridge.cpp:257/582/651/776` | Métadonnées/handles, CString et vecteurs; factory de kernel déjà cachée ou création de factory, puis appel de closure et création de sorties différées. | Pas d'appel eval/async/data sur les banques dans ce parcours. `Array::owned` ne demande que rang/shape/dtype. |
| `mlx_bridge.cpp:88/263` et `Array::from_u32` | **Allocation immédiate** de petits buffers d'indices et memcpy des indices CPU. Ce ne sont pas des nœuds entièrement paresseux. | Ces copies ne lisent aucune banque; l'allocateur peut prendre du temps ou attendre une ressource. Les coûts et la contention avec les lecteurs ne sont pas quantifiés. |

Sources primaires MLX **v0.32.2** : [ops.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/ops.cpp) `astype`303–312 (même dtype = retour de l'entrée), `reshape`458–467, H128566–592, `multiply`3108–3117, `gather/take`3547–3707 et `view`6626–6651 construisent des primitives à partir de métadonnées. [metal_kernel.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/metal_kernel.cpp), blob déjà archivé `7a2eef51fcfa04d1e6828ca72be100235fded40f`, renvoie `array::make_arrays(CustomKernel, inputs)` après la génération de signature/source. La copie primaire locale de `custom_kernel.cpp`, SHA256 `92548fba86690996a30725ef96bb8e0e299b69a4559e71c60340e7e11c1cac5e`, montre les allocations de sorties, copies conditionnelles de contiguïté, hash source, lookup bibliothèque/PSO et bindings/dispatch dans **`CustomKernel::eval_gpu`**, donc après la construction.

Le SwiGLU compilé du pont (`mlx_bridge.cpp:665`) ne change pas ce constat : [compile.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)420–437 trace des placeholders sans données, 1143–1187 recherche/trace/fusionne/remplace le graphe; `compile_replace`1047–1082 crée des tableaux avec les vrais inputs. La simplification peut lire des **constantes scalaires déjà disponibles** (595–623), pas les banques ni les activations GU transmises comme paramètres. Aucun callback de cette closure ne fait eval/item/readback.

## Conditions de sûreté à garder

- **Exception explicite cfg(test)** : `flash_profile::stage` (`flash_profile.rs:401–407`) appelle `value.eval()` lorsque le profil est fenced. Les stages de `calculate_slots` et `expert_mapped_layout` sont sur ce parcours. Déplacer W2 avant finish serait donc incorrect dans ce mode : ses scales/poids pourraient être lus pendant leur écriture. Le port doit garder ce mode sur l'ancien placement, ou refuser/neutraliser explicitement le placement anticipé; ne pas prétendre que tout libtest est intrinsèquement lazy. Les CLI release et tests sans trace fenced ne prennent pas cette branche.
- Construire seulement **après** `wait_ready(2)` et la partition effectivement observée (`ssd_moe.rs:430–440`). Conserver les mêmes occurrences, l'ordre W1+rest puis le `take` canonique déjà en place, et les mêmes kernels/arrondis. Les réservations demeurent exclues des nouveaux fills/prefetch jusqu'à la fin de cette transaction.
- Ne faire aucun async/eval/to_bytes/compact/retained_bytes sur W2 ou ses dépendances avant le succès de finish. Les banques ont été évaluées et leur stockage retenu avant le lancement des workers; ceux-ci ne changent que les octets des destinations, pas les métadonnées ni les pointeurs.
- Garder le batch local actif pendant la construction. En cas d'erreur de construction, son Drop doit rejoindre les lecteurs avant l'invalidation d'ExpertFill et avant le drainage externe des consommateurs. L'objet FFI retient les banques (`PreparedDiskBatch`, pont133) et le wrapper retient les fichiers. Le nouveau graphe lazy peut simplement être détruit sur erreur; ce n'est pas un consommateur GPU tant qu'il n'a pas été soumis.
- Conserver séparément les propriétaires/guard de Hot et W1 (`ColdConsumers`, `ssd_moe.rs:33`); ne pas écraser un consommateur soumis par un W2 encore lazy. Un stockage de W2 dans un guard dont Drop fait eval exige la même jointure préalable sur **toutes** les sorties d'erreur. Le cas le plus simple est un résultat local W2 non soumis, retourné seulement après finish, puis utilisé par l'assemblage.
- Le raccourcissement de la jointure n'autorise pas d'avancer `fill.commit()` (`ssd_moe.rs:1154`) avant la réussite du lot, ni de perdre les erreurs canoniques des jobs non prêts. Les échecs GPU irrécupérables restent la limite existante du guard; cet ordre ne la résout pas.

## Ce qui peut être gagné, et ce qui reste payé

Aujourd'hui W1 est construit/soumis dans `fill_ready`441–444, puis `batch.finish`460 attend tout; W2 n'est construit qu'au retour, dans le forward1172–1173. Le changement minimal porterait le résultat lazy W2 à côté de W1 jusqu'au retour réussi de fill_ready, et supprimerait cette reconstruction après la jointure. Il ne change ni nombre de vagues ni lectures/bytes/cache ni ordre de réduction. Le chemin « tout est déjà prêt » peut rester stock, sans créer de graphe W2 supplémentaire.

Le plafond propre est au plus le **travail de construction W2 qui tombe dans la fin I/O encore non masquée après W1.async_eval**, pas toute la durée I/O et pas les ~12,142 ms historiques dans async CPU actif ∩ GPU idle. Les allocations de sorties/copies contiguïté, hash/PSO, encodage/barrières et la plupart de la visite/évaluation MLX restent au moment de la soumission après finish. La mesure T01-P ne concernait qu'un GU Hot et trouvait environ5,529 µs/appel de boucle pré-eval; elle ne borne pas l'ensemble de W2 (gathers/GU/SwiGLU/Down), mais interdit d'identifier les 12 ms à la simple construction de source. Ce sont des postes différents et inclusifs.

Un contrôle falsifiable ultérieur, après le pilote courant et un préenregistrement distinct, comparerait le placement actuel/anticipé sur le même CLI, avec la qualité composant non contiguë + erreur/reprise et les oracles/modes déjà touchés. Pour qualifier le mécanisme, relever séparément début/fin de construction W2 et attente restante de finish; éviter d'attribuer ces intervalles inclusifs à un gain exclusif. Le test de job retenu devrait en plus empêcher une évaluation anticipée de W2, sans transformer ce délai synthétique en preuve de vitesse.

Préparer un préfixe fixe **avant** wait_ready est une autre politique : elle exige d'attendre les jobs imposés ou de jeter/reconstruire un graphe quand les premiers prêts sont non contigus. Elle perd la fenêtre du crible first-ready2 par rapport au préfixe2 et n'est pas requise pour ce petit déplacement CPU. Aucun tel changement recommandé ici.

## Identité du code relu

- `native/src/ssd_moe.rs` SHA256 `d3b65041cc6b0fb3affaf76eee9fad6828eeadadb7f9096a4785a2db6ae08cd1`.
- `native/src/array.rs` SHA256 `3938aec8c7eb44fc16f82cea670f67a1d700a5f339efc36f00384fb670f21224`.
- `native/src/linear.rs` SHA256 `529d1947fb9a027868d4e6f013ff1b138831b6e4dca70a806c071ef6a51b2440`.
- `native/ffi/mlx_bridge.cpp` SHA256 `b713228b11f9b5b2174ef99065e7c9dce9ae58bff542dbd685add28652344d8c`.
- `native/src/flash_profile.rs` SHA256 `136b318793e74cf59e94e5a59a7516223b873e2219faaf17382bcd02bd99bfe2`.
