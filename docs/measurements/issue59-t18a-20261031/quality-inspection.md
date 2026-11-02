# T18a — inspection qualité du 31 octobre 2026

**17 exécutions physiques réussies : 12 légères + 5 modèle, correspondant à 13 sélecteurs distincts.** Chaque invocation exacte imprime `running 1 test` puis `1 passed; 0 failed; 0 ignored`, avec 374 tests filtrés non exécutés. Les marqueurs requis sont présents, tous les processus joints ; aucun gain modèle déduit de leurs durées. [Inspection détaillée](quality-inspection.json), [stockage](storage-plan-result.json), [composants](component-plan-result.json), [modèle r3](model-quality-r3-plan-result.json).

## Identité, options et mémoire

Les **531 empreintes** de [final-identity-r3.json](final-identity-r3.json) ont été relues par SHA en streaming : zéro manquant/écart. Les trois binaires concordent, dont CLI `1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436` et libtest r2 `de3616e3d2e0c3d9507e075cf9af4dfe8b3290f0b291ac704b3e1df537f8d910`. Toutes les sélections physiques utilisent ce libtest. Le CLI release est réutilisé : les corrections natives ultérieures concernent les régressions/diagnostics cfg(test), provenance explicitée dans l'identité, pas une identité source r1 antérieure inventée.

T04/T17/T05 sont OFF, cache target3 Gio/53×48, quatre lecteurs, mêmes checkpoint/référence indépendante/draft et contrôles MLX40/40. Le stockage et trois oracles composants sont exercés OFF/ON ; les autres composants et cinq modèles sont ON. Le cas d'option invalide vérifie un **rejet sans écriture**, pas une parité numérique.

Sous l'[ancre T18A-w1](vm-anchor-t18a-w1.json), Swapouts absolus **140865945600 B**, Swapins **53831401472 B**. Les 17 reçus déclarent chacun zéro nouveau Swapout/Swapin, leurs snapshots avant/après gardent l'ancre Swapout et leurs processus sont joints. Le plus grand RSS enfant wait4 est 7490732032 B ; ce n'est pas le footprint. Le footer DFlash observé donne footprint7347132584 B/pic vie7385077976 B, sous12 Go ; aucun capteur continu universel de footprint natif n'est revendiqué.

## Couverture réellement exercée

| Groupe | Observation inspectée |
| --- | --- |
| Stockage, 3 invocations | Deux bras : **18 cas/1644 records/52608 octets payload chacun**, canaris et affectations vers vrais buffers Metal. OFF :0 Dynamic/18 Static ; ON :9/9, donc vraie garde et fallback exercés jusqu'à512 jobs. Option invalide :5 records,40 octets inchangés. |
| Composants, 9 invocations | Limites des records, alias/prévalidation, MoE indépendant froid/chaud/paires/rows1..8/petits caches/échec/reprise dans les deux bras ; Hot pending, Wide Hot et fenêtres uniques en ON. Le marqueur d'invalidation des anciennes timings après refus pré-FFI est présent dans les deux MoE. |
| Hot/Wide/Unique | Hot imprime5 hits/5 misses et `ready_at_routes=false`, `ready_at_demand=false` ; Wide imprime12 cas rows3..8, byte-exact/asynchrone ; Unique imprime32 cas rows6..8, capacités20/24/53, pending exclu et reprise I/O. Ces compteurs ne sont pas transformés en chronos GPU. |
| Modèle indépendant | `flash_matches_independent_reference_states_and_logits` imprime les cinq positions s0..s4 avec **248320 logits chacune**, RMSE/maxabs affichés nuls ; source : finitude/nombre exact, égalité byte des logits et entrées récurrentes comparées, routes/argmax contrôlés. |
| Composants modèle | `flash_components_match_reference_on_identical_inputs` passe ses comparaisons attention/GDN/HC/MoE aux entrées indépendantes ; garder les tolérances de chaque comparaison, sans inventer un nombre de sorties non imprimé. |
| Paires/MTP | `flash_pair_verification_preserves_serial_logits_and_all_states` passe les transactions/commits/continuations et refus. MTP budgets1/2/3/17 imprime `(proposés,acceptés,blocs)` = `(0,0,0)/(1,0,1)/(2,0,2)/(12,5,12)` ; acceptation et rejet sont donc exercés, états target/head et invalides vérifiés par le test réel. |
| DFlash | `flash_draft_captures_and_all_prefix_commits_match_serial` imprime **36 préfixes**, largeurs1..8, logits/états target byte-exact ; capture indépendante64000 valeurs, `byte_equal=true`, tolérance0,004 conservée. Abandon/commit invalide/rollback et continuation sont dans l'entrée réelle. |

`check` refuse tableaux vides, mauvais comptes et valeurs non finies des deux côtés ; `tensor_state_bytes` refuse les états présents vides/non finis et conserve leur forme. Cela qualifie les domaines des tests, pas toutes les trajectoires ni tous les états à chaque token d'une génération128. Les sélecteurs et hashes de logs complets sont dans le JSON lié.

## Admissions et correction du superviseur préservées

[R1](model-admission-r1.json) refuse avant modèle dès le premier snapshot :10961993728 B, soit38,0 Mo sous11 Go ; aucune observation20s complète. Amendement prospectif à10,8 Go avant modèle/vitesse, sans reset d'ancre. [R2](model-admission-r2.json) passe20s, puis le [lanceur refuse avant PID](model-quality-plan-result.json) à10774691840 B<10,8 Go : aucun résultat numérique ni débit dans ce refus.

Le [diagnostic CPU du parent](controller-memory-comparison.json) rapporte les mêmes529 hashes dans deux processus : RSS après/pic bulk225,034 Mo contre streaming19,399 Mo, avant≈19,1 Mo. Cela établit un coût mémoire évitable, sans prouver la cause exclusive du refus OS. `gpu-runner.py` et ses tests **ont changé** ; originaux conservés dans [controller-before/](controller-before/). Sept régressions ajoutées, [44 tests et lint/format/compilation Python passés](controller-correction-checks-plan-result.json). La garde10,8 Go et0 nouveau Swapout reste inchangée ; r1/r2 conservées, nouvelle identité r3.

[Observation r3](model-admission-r3.json) :3 snapshots sur20,19s à10,975/10,948/10,952 Go, puis les cinq oracles passent. La réduction mémoire du superviseur n'est pas un boost moteur.

## Contrôles CPU et limites

[Reçus CPU](cpu-qualification-plan-result.json) : **308 cas lecteur** sous ASan+UBSan puis TSan ; **4084 cas queue concurrents +4096 politiques effectives** sous chacun, pas exploration exhaustive des interleavings. Mutation CAS+2 isolée [détectée](queue-mutation-plan-result.json), production inchangée. **233 tests Python profiling** passent ; les44 tests superviseur post-correction sont une portée séparée, sans addition comme tests uniques. [Contrôleur externe](inspection-checks-plan-result.json) : Ruff/format/py_compile passent ; **17 auto-tests synthétiques =3 positifs/14 refus**, aucune inférence. Les refus finissent `failed_external_inspection`, sans `parity:true`.

CBMC échoue au parsing avant propriétés, Kani sur vraie entrée MLX rencontre l'ICE `intrinsics.rs:243`, CrossHair est absent : ces propriétés source restent **non vérifiées**. Les sanitizers et tests ne prouvent pas le moteur ou sa concurrence. Suites finales pertinentes, bridge17 et mesures release restent des étapes distinctes coordonnées par le parent ; leur état n'est pas anticipé ici. Aucun code, plan, manifeste ou journal modifié par cette inspection, aucun build/test/proof/GPU lancé par cet agent, aucune app ou publication.
