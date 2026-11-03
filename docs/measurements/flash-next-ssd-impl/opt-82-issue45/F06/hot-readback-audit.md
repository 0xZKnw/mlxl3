# GPU Hot et lecture des routes — audit statique 2026-10-25T00:08:26.371073+00:00

**Dans le chemin Hot actuel, `selected.to_u32()` n’attend pas systématiquement la fin de tout le stream MLX ni celle du Hot soumis après ses routes.** La distinction est déjà documentée et physiquement exercée par56B ; aucune nouvelle panne n’est établie ici. Lecture uniquement, aucun code modifié, modèle, GPU, benchmark, build ou prouveur lancé.

## Chemin exact de production

1. `forward_unique_rows` et `forward_rows_prepared` construisent les IDs GPU, puis appellent `prepare_resident_hot` **avant** `selected.to_u32()` (`ssd_moe.rs:582–595`, `759–790`).
2. Le helper construit le mapping expert→slot par `ExpertSlots::residents(selected, map, capacity)` ; son kernel lit `selected.reshape([-1])` (`linear.rs:1159–1186`). Il soumet **d’abord** `slots.indices().async_eval()` puis construit et soumet `output.async_eval()` pour les experts Hot (`ssd_moe.rs:549–558`). L’évaluation du premier graphe programme donc les routes dont il dépend, avant le graphe Hot.
3. `Array::to_u32` appelle `to_bytes`, puis `mlxl3_array_copy_bytes` (`array.rs:966–1005`). La FFI actuelle a le correctif56B : si la source est déjà soumise et `row_contiguous`, **`source.eval()` puis memcpy direct**, sans nouvelle primitive `Contiguous` et sans appel à `mx::synchronize` (`mlx_bridge.cpp:436–449`). Les IDs top10/concaténations de ce chemin sont compacts ; ils ont déjà été programmés par l’évaluation du mapping.

## Ce que MLX0.32.2 attend

Les sources primaires déjà archivées lors de78 proviennent du tag v0.32.2, commit **`1f8e74e3f12f31365464a6867c6579f0e9b29d85`**. Les empreintes `array.cpp`, `array.h`, `transforms.cpp`, `ops.cpp` correspondent au manifeste78. Le `array.h` installé est byte-identique à l’archive (`5c90807aa44226c557c5bda740bdc1732109810e8862697ad5d374092669db65`) et `version.h` installé déclare 0.32.2. Ce contrôle établit la correspondance des en-têtes/source tag, pas une reconstruction byte à byte de la bibliothèque installée.

- [array.cpp155–162](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/array.cpp#L155) : `array::eval` n’appelle `core::eval` que si l’array est `unscheduled`. Sinon il fait `wait()`.
- [array.cpp132–153](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/array.cpp#L132) : `wait()` contrôle puis attend **l’événement attaché à cet array**, et le détache quand disponible. Pas de synchronisation explicite du stream entier.
- [transforms.cpp232–246 et337–344](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/transforms.cpp#L232) : un `async_eval` attache son événement aux nœuds qu’il programme, puis le signale et finalise les streams touchés. Le second `async_eval` Hot n’ajoute pas rétroactivement son attente à l’événement du premier graphe mapping/routes. Le premier événement couvre néanmoins tout le premier graphe et les travaux antérieurs nécessaires dans le même stream, pas uniquement le kernel top10 isolé.
- [array.h373–376](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/array.h#L373) : `data()` récupère le pointeur, sans `eval` caché dans ce corps. Le header prévient qu’un backend peut copier un tampon non accessible au CPU ; sur ce parcours Metal partagé, les tests historiques ci-dessous exercent le chemin réel.

La soumission/finalisation d’un stream n’est donc pas équivalente à attendre sa fin sur CPU. `async_eval` peut malgré son nom attendre sous pression de scheduler/mémoire (`transforms.cpp:268–283`, `wait_for_one`) ; c’est une autre frontière, **pendant la soumission**, et son coût exclusif actuel n’est pas mesuré. La branche d’erreur du même fichier synchronise les streams pour réparer les événements ; elle ne décrit pas le readback normal.

## Le vrai ancien défaut, et l’exception à ne pas confondre

`mx::contiguous(source)` crée une primitive même si la source est déjà contiguë ([ops.cpp6757–6767](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/ops.cpp#L6757)). Construire cette primitive au readback la place après les travaux Hot déjà soumis au même stream ; attendre sa sortie peut alors attendre le Hot par ordre de file. **C’était le défaut56 corrigé**, pas une propriété générale de `to_u32`.

La FFI conserve volontairement le chemin `contiguous + eval` pour une source non soumise ou stridée.72`packed` a identifié un cas différent : prédiction `[2,16]` obtenue par slice à stride512, matérialisée tardivement derrière Hot. Son microdiagnostic distinguait bien legacy/packed, mais ses mesures complètes et son retrait restent dans l’historique72/73. La prédiction n’est pas le tableau compact `selected` du chemin top10 actuel ; ne pas réintroduire ce prototype ni le compter comme nouvelle découverte. Si le mapping n’a aucun résident, `prepare_resident_hot` renvoieNone et ne soumet aucun Hot : le fallback éventuel du readback ne peut pas annuler un recouvrement Hot absent.

## Preuves existantes relues, sans répétition

- `opt-56-hybrid-hot/mechanism-v2/56B-readback.log` : test réel `scheduled_route_readback_does_not_wait_for_later_gpu_work`,1 passé, `later_gpu_pending=true`,4 IDs exacts.
- `opt-56-hybrid-hot/mechanism-v2/56B-overlap.log` : demande SSD à2,963875ms, Hot encore en vol au retour des routes et au début de la demande ;1 test passé, `ready_at_routes=false`, `ready_at_demand=false`,5 hits/5 misses. Retard GPU artificiel : ces durées ne sont pas un gain tok/s.
- Le test courant `ssd_moe/hybrid_tests.rs:35–61` conserve ces assertions ; le chemin Wide/UNIQUE les exerce aussi sur B8 avec slots pending (`:148–171`). Le test73 réutilisait56 et conservait Hot pending même avec shared anticipé. Ces anciens passages ne sont pas présentés comme une nouvelle exécution F06.

**Décision : pas de correctif proposé pour une attente globale supposée de `selected.to_u32`.** Les frontières restant possibles sont l’attente réelle du graphe routes/mapping, la contention/pression dans la soumission, les autres readbacks stridés et le dernier drain nécessaire avant écriture des banques. Leur poids dans le débit F06 n’est pas isolé ici. Aucun gain, nouvelle preuve formelle ou validation numérique intégrale n’est revendiqué par cette lecture.

Sources locales consultées : `/Users/justin/.codex/worktrees/flash-splash60/mix-stq1_0/native/src/{ssd_moe.rs,linear.rs,array.rs}`, `native/ffi/mlx_bridge.cpp`, archives primaires `/Users/justin/Documents/mix-stq1_0/build/flash-opt-78-gdn-constants/sources/`, historiques56/64H/72/73/79/80 du journal.
