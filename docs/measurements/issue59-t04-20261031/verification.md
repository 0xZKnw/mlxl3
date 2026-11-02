# T04 — vérification du candidat archivé et du code restauré

31 octobre 2026. **T04 rejeté pour intégration après audit, puis retiré du
code courant. Vérifications finales du code restauré PASSÉES.** Le candidat
archivé a passé ses qualités numériques, **10 sélections GPU uniques**, le
bridge17 **6/6** et le pilote128 **12/12**. Le pilote ne satisfait pas la
demande d'un gain général sans ralentissement du décode ; les petits écarts
spéculatifs ne sont pas promus en boost. L'audit n'a trouvé aucun défaut
concret nouveau justifiant une correction de ce portage.

**Les résultats candidat ci-dessous appartiennent aux sources et binaires
archivés, pas au moteur actuellement restauré.** L'application installée
reste inchangée ; aucun push, PR ou release n'a publié ce candidat.

Préenregistrement : `ISSUE59-T04-COLD-01` en tête d'
[opti.md](../../../opti.md). L'[audit de périmètre](scope-quality.md) décrit
les antécédents et les tests d'intégration à reprendre. La présente rédaction
n'a lancé aucun test, build, proveur, capture ou GPU.

## Code et identité du candidat réellement qualifié

Le portage reprend sélectivement le T04 de PR61, tête
`a36ba5077bf386466660d6b478058fbe6880245c`, et son raccord corrigé
`40f45f8016ef68881bad9b6344fc6022b28fc038`, sans merge de la PR ni import de
T17/T18. Il élargit la closure SwiGLU compilée pour inclure le cast de
l'activation F16 et le produit avec `suh` F16. Il garde le dtype gate/up
**F16 ou F32**, les arrondis de leurs primitives, le H128 MLX, EXL3
K2/Mul1, les splits et sommes, puis restaure le dtype GU en sortie down.

Le domaine servi par le nouveau helper est slot-major, routes CPU
**non masquées et non groupées**, GU `[R,2,640]`, R=1..80. Le Hot masqué et
les routes groupées gardent le chemin stock ; « Cold » ne signifie pas que
chaque appel du domaine ne pourrait contenir que des misses.

Ces descriptions s'appliquent désormais aux [neuf sources exactes archivées](candidate-source-identity.json),
notamment [Array](candidate-source/native/src/array.rs), la
[FFI](candidate-source/native/ffi/mlx_bridge.cpp), le
[helper down](candidate-source/native/src/linear.rs), le
[raccord SSD](candidate-source/native/src/ssd_moe.rs), la
[garde CPU](candidate-source/native/src/flash_support.rs) et les
[oracles T04](candidate-source/native/src/ssd_moe/swiglu_down_tests.rs).
Le [patch local complet](candidate-complete-local.patch) et toutes les preuves
historiques sont conservés ; le nouveau fichier de tests n'existe plus dans
le code actif.

Le [gel source initial](initial-source-identity.json) identifie **191
fichiers** avant compilation ; l'[identité du smoke](smoke-r1-identity.json)
étend le manifeste à 192 éléments pour le plan. Le libtest immuable des
quatre premières sélections physiques est :

`build/issue59-t04-20261031/libtest-r1/prototype-engine`

SHA-256 **`22073b88203fa4db0a3e2792f03f99c0e52f1a131c27f2f6d37c535ac10b384d`**.

L'[identité finale r2](final-identity-r2.json) identifie **218 éléments** après
les suites et les contrôles du lanceur. Les 191 éléments du gel initial,
dont les sources natives, sont inchangés. Le libtest final copié dans
`build/issue59-t04-20261031/libtest-final/prototype-engine` est **byte-identique
au r1 ci-dessus** ; l'oracle MoE l'utilise également. Le CLI T04 final,
`build/issue59-t04-20261031/engine/prototype-engine`, a pour SHA-256
**`81bc3a14d509224d77fd6d98d8dc64f9b840c0094f7376f5f004e924849732cd`**.
Les métadonnées des checkpoints sont identifiées ; les fichiers de poids ne
sont pas intégralement hachés. L'inspecteur Python a été seulement reformatté
depuis le gel précédent, avec AST de `inspect` contrôlé identique ; cela
n'a changé ni le code numérique natif ni ses binaires.

L'[identité r3](final-identity-r3.json), **219 éléments**, conserve ces mêmes
CLI/libtest et les 191 éléments initiaux inchangés. La seule nouvelle
révision concerne l'admission future : **20 fichiers de configuration**
(18 plans bridge/pilote, leur manifeste et le plan des cinq qualités modèle)
modifiés avant exécution, avec [relevé des différences](admission11-revision.json).
Aucun code natif, prompt, ordre de test ou booléen candidat n'a changé.

Les [identités de matrice](boundary-down-r1-identity.json) et
[de capture](capture-r1-identity.json) sont conservées séparément. Matériel :
Apple M5, 24 Gio, macOS 27.2, MLX 0.32.2. La capture charge une seule couche
SSD du vrai checkpoint `Qwen3.8-Flash-Next-exl3-205-65c89531` pour produire GU ;
ce n'est pas une qualification du modèle complet. L'ancien CLI T17 ne
valide pas ce nouvel exécutable T04.

## Contrôles locaux et vérification source du candidat archivé

Les commandes ci-dessous sont celles des plans archivés, exécutées depuis
la racine du worktree contenant `Cargo.toml`.

| Contrôle | Résultat effectivement lu |
| --- | --- |
| `cargo fmt --all -- --check` | Exit 0, processus joint ; [plan initial](initial-cpu-plan-result.json). |
| `cargo test --locked --release --no-default-features --lib swiglu_down_elements -- --test-threads=1` | **2 passés, 0 échec, 0 ignoré, 178 filtrés** ; c'est une sélection, pas une suite complète. [Log](t04-cpu-geometry-tests.log). |
| `cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json` | Libtest r1 construit, exit 0 et joint ; [résultat initial](initial-cpu-plan-result.json). |
| Clippy strict `--locked --all-targets --no-default-features -- -D warnings` | Passé, exit 0, joint ; [log](t04-final-clippy-cpu.log). |
| Clippy strict `--locked --all-targets --features mlx,chat -- -D warnings` | Passé, exit 0, joint ; [log](t04-final-clippy-mlx.log). |
| `cargo test --locked --release --no-default-features -- --test-threads=1` | **258 passés, 0 échec, 8 ignorés**, cibles agrégées ; [log](t04-final-cpu-suite.log). |
| `cargo build --locked --release --features mlx,chat --bin mlxl3-rs` | Passé, exit 0, joint ; CLI Apple réel, [log](t04-final-cli-build.log). |
| `cargo test --locked --release --features mlx,chat -- --test-threads=1` | **295 passés, 0 échec, 189 ignorés**, cibles agrégées ; [log](t04-final-mlx-suite.log). |

Le [plan complet](full-cpu-plan-result.json) est maintenant terminé et son
[inspection](full-cpu-inspection.json) conserve la ventilation : CPU
`172/0/8 + 35/0/0 + 16/0/0 + 35/0/0 + 0/0/0 + 0/0/0` ; MLX
`200/0/179 + 44/0/8 + 16/0/0 + 35/0/0 + 0/0/2 + 0/0/0`
(passés/échecs/ignorés). **Les ignorés restent non exécutés par ces suites** ;
les dix sélections physiques sont comptées séparément. Les processus de
contrôle ont tous été joints, sans nouvelle écriture swap observée.

La garde de production `flash_support::swiglu_down_elements`, réellement
appelée par `Array::swiglu_scaled`, a été vérifiée par :

`cargo kani --lib --no-default-features --harness swiglu_down_elements_checks_symbolic_shapes_and_bounds --solver kissat`

Résultat : **0/166 obligations en échec, 12/12 couvertures satisfaites,
unwind 17**, assertions d'unwinding conservées, sans `kani::assume`.
Le harnais explore des dimensions `i32` et rangs 0..5 symboliques ; il vérifie
l'acceptation exacte de `[R,640]`, R=1..80, le nombre d'éléments 640..51 200,
les indices `u32` et les bornes d'adressage F16/F32. Il couvre R1/R2/R80,
le dernier indice 51 199 et les refus de rangs/dimensions extrêmes. Les rangs
au-delà de 5 ne sont pas dans ce domaine de preuve. Voir
[plan/résultat](geometry-kani-plan-result.json),
[identité de la source prouvée](geometry-kani-source-identity.json) et
[log complet](t04-geometry-kani.log).

La tentative `cargo kani --lib --tests --features mlx,chat --harness
actual_swiglu_scaled_mlx_entry` appelle la vraie entrée Array/FFI, sans stub.
Elle a échoué **avant obligations**, ICE
`kani-compiler/src/intrinsics.rs:243:17`, exit 1 ;
[log](t04-actual-mlx-kani.log). **MLX/FFI non vérifiés.** Le statut du
superviseur `completed_expected_outcome` signifie ici que cet échec attendu
a été recueilli, pas qu'une preuve a réussi.

CBMC 6.11 sur la vraie `mlxl3_array_swiglu_scaled` dans le fichier FFI entier,
avec les headers MLX installés, a échoué au parsing libc++
`integral_constant.h`, **exit 6, avant obligations**. Le vérificateur est
limité ici à C++11 tandis que le build réel utilise C++20 ; aucune traduction
ni stub n'a été utilisé pour obtenir du vert. Voir
[commande/résultat](cbmc-ffi-result.json) et [diagnostic](cbmc-ffi.log).
**FFI non vérifiée.** La preuve CPU de géométrie ne prouve ni les arrondis
MLX/Metal, ni H128, ni l'allocateur, ni le SSD ou la concurrence.

La [revue CI du candidat](ci-inspection.json) confirme le workflow PR/main et les nouvelles
sélections CPU/géométrie ; son YAML a été accepté par Ruby. `actionlint`
était absent. Les tests Apple physiques restent des sélections locales
explicites. **Aucun run GitHub de cet arbre local non publié n'est déclaré
exécuté ou vert.**

## Contrôleur externe Python

L'inspecteur du pilote a été réutilisé et paramétré pour T04, puis contrôlé
avant les futurs chronos. Ruff lint et format passent pour l'inspecteur et
son selftest ; les refus de style antérieurs restent archivés. Les vraies
fonctions `inspect`, `validate` et `check_golden` ont exercé **3 cas positifs
AR/MTP/DF17 et 12 refus**, avec fixtures explicitement **synthétiques**, aucun
moteur lancé, cwd/journal temporaires supprimés. Les cas invalides conservent
`failed_external_inspection` et ne déclarent pas de parité. Ces cas CPU ne
sont ni des réponses réelles du bridge ni des mesures de performance. Voir
[contrôles](inspection-python-checks.json) et
[résumé des 15 cas](inspection-selftest-summary.json).

Le [sondage de disponibilité CrossHair](inspection-crosshair-result.json)
confirme le module absent. Aucune analyse symbolique du contrôleur n'a pu
être exécutée ; **Python non formellement vérifié**. Une préparation du gel
avait aussi refusé le nom absent `inspection-selftest.py` avant PID ; le
fichier existant `inspection_selftest.py` a ensuite été correctement
identifié dans r2. Ces erreurs de préparation ne sont pas des échecs du
modèle et ne sont pas comptées comme des tests GPU.

## Composants et MoE : cinq sélections physiques terminées

Chaque sélection exacte a annoncé **1 passé, 0 échec, 0 ignoré**, avec 378
autres tests filtrés, puis a été jointe. Les nombres ci-dessous proviennent
des marqueurs de complétion des logs, pas seulement du statut du lanceur.

| Sélection (préfixe `ssd_moe::swiglu_down_tests` pour les quatre premières) | Couverture validée |
| --- | --- |
| `t04_smoke_matches_stock_and_independent_literals` | **8 cas / 7 680 valeurs F16**, R1/R2, gate/up F16 et F32, ancres indépendantes gate 0/16 ; vraie FFI appelée, aucune lecture hôte pendant construction, cardinalités/finitude/bytes complets et sources immuables. [Log](t04-smoke-r1.log), [résultat](smoke-r1-plan-result.json). |
| `t04_swiglu_down_preserves_both_half_roundings_and_h128` | **56 cas / 640 000 valeurs avant H128 + 640 000 après**, R1/2/4/8/10/20/80, deux dtypes, deux layouts et deux phases. F32 inclut des entrées vérifiées non représentables en F16. **Deux R80 supplémentaires / 102 400 valeurs avant et autant après H128** couvrent ensemble les **63 488 patterns de gate F16 finis**, avec up/scales bornés et non triviaux. **13 refus Array avant FFI/readback**. [Log](t04-boundary-r1.log), [résultat](boundary-down-r1-plan-result.json). |
| `t04_down_projection_preserves_production_dtype_and_bits` | Vrai helper down slot-major K2/Mul1, **56 cas / 2 560 000 valeurs**, mêmes R, F16/F32, strides, slots répétés et `stable_shapes` 0/1 ; shape/dtype/finitude/bytes complets et immutabilité. **16 refus avant FFI**, dont slots masqués/groupés. Ces refus directs ne qualifient pas encore le fallback du modèle complet. [Log](t04-down-r1.log), [résultat](boundary-down-r1-plan-result.json). |
| `t04_captures_stock_and_fused_real_gu20_preparation` | Deux traces seulement depuis le vrai GU `[20,2,640]` d'une couche SSD ; **12 800 valeurs F16 par capture**, finies, bit à bit et sources inchangées. [Log](t04-capture-r1.log), [résultat](capture-r1-plan-result.json). |
| `ssd_moe::tests::two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads` | Vraie couche MoE SSD et oracle indépendant complet ; **9 capacités ×6 options** : 10/19/20/24/53/71/80/89/512 slots, overlap OFF/ON et 1/2/4 lecteurs. Cold/Hot, paires, lignes 1..8, formes invalides, échec lecture et récupération ; compteur T04 ON effectivement non nul pour le chemin éligible, garde d'éligibilité conservée. Les refus masqué/groupé sont contrôlés séparément par le helper ci-dessus. **1 passé, 0 échec, 0 ignoré**, joint en 10,350 s superviseur, zéro nouveau Swapout. [Log](t04-moe-component-r1.log), [résultat](moe-component-plan-result.json). |

Le plan initial de six qualités a été **scindé prospectivement** : l'oracle
MoE ne charge qu'une couche et a été exécuté comme composant avec la garde
micro **2 Go**, sous [son plan distinct](moe-component-plan.json). Les cinq
autres oracles chargent le modèle complet ; leur admission initiale 12 Go
a ensuite été révisée prospectivement à 11 Go comme décrit plus bas. Le MoE
n'a pas été relancé dans ce plan : le bilan distingue **1 composant MoE et
5 tests complets**, sans les renommer six tests du modèle complet.

L'oracle de frontière est le vrai graphe stock
`swiglu → astype(F16) → multiply(suh F16)`, puis le **même H128 MLX**.
L'oracle down appelle le chemin `expert_slots` de référence, avec le même
H128/QMV/ordre numérique. Les nouvelles APIs sont contrôlées sans lecture
de données à la construction et par compteurs d'entrée FFI ; ces compteurs
ne sont pas des compteurs de dispatchs GPU.

Le contrôle négatif recalcule volontairement le graphe en différant les
arrondis F16 jusqu'après le produit d'échelle. Le mutant est vérifié **fini
et complet**, puis la divergence est détectée pour chacun des deux dtypes.
Il s'agit d'un **graphe de contrôle volontairement différent**, pas d'une
mutation temporaire de la source de production dont on revendiquerait ici
la restauration.

Les sondes subnormales montrent pourquoi il faut garder le dtype : les deux
premières sorties F16 stock sont `0000/0000` pour gate F16, contre
`0001/0000` pour gate F32 ; le candidat suit chaque stock. Le premier
mutant F16 produit `0001` et est détecté. Le document ne remplace pas le
stock GPU par une hypothèse CPU aveugle de FTZ. L'exploration est exhaustive
des gates F16 finies **pour les up/scales choisis**, pas du produit cartésien
gate/up/scale ni de toutes les valeurs F32. Les non-finis généraux et payloads
NaN ne sont pas qualifiés par cette matrice finie.

## Modèle complet : cinq qualités passées

Le [plan modèle restant](model-quality-remaining-plan-result.json) est
terminé : chaque sélection exacte annonce **1 passé, 0 échec, 0 ignoré**,
processus joint et zéro nouveau Swapout depuis T04-w1. L'
[inspection modèle](model-quality-inspection.json) conserve le décompte
**1 MoE isolé +5 modèle complet**, sur le même libtest `22073b…`, avec seul
T04 actif :

| Sélection `qwen4::…` | Contrat observé |
| --- | --- |
| `tests::flash_matches_independent_reference_states_and_logits` | Fixture indépendante, **5 positions ×248 320 logits complets**, routes et états récurrents ; égalité bit à bit pour les logits et états explicitement contrôlés, cardinalités/finitude et reset/refus. [Log](t04-final-model-oracle.log). |
| `tests::flash_components_match_reference_on_identical_inputs` | Composants sur entrées indépendantes identiques, avec leurs seuils propres ; ne pas généraliser toute cette batterie en une preuve universelle bit à bit. [Log](t04-final-model-components.log). |
| `tests::flash_pair_verification_preserves_serial_logits_and_all_states` | Logits/états des paires, commits de préfixe 1/2, continuation et refus atomiques. [Log](t04-final-model-pairs.log). |
| `tests::flash_mtp_session_preserves_greedy_sequence_target_and_head_states` | Budgets **1/2/3/17**, séquence greedy et états cible/tête exacts, acceptation et rejet réellement couverts, alignement et erreurs. [Log](t04-final-model-mtp.log). |
| `draft::flash_draft_captures_and_all_prefix_commits_match_serial` | **36 préfixes**, largeurs 1..8, commits/rollback et continuations, logits/états cible exacts ; 64 000 valeurs capturées `byte_equal=true`. [Log](t04-final-model-dflash.log). |

Les durées de superviseur 3,199 /3,392 /9,973 /9,059 /26,571 s sont des durées
de tests et non des mesures de débit. Avec les quatre qualités du composant
et le MoE isolé, il y a **10 sélections physiques uniques**, sans preuve
formelle du GPU ni qualification de toute entrée possible.

## Capture et comptage Xcode

Les inputs et les deux graphes ont été chauffés avant capture, avec comparaison
hors capture. Chaque capture suit `start → synchronize → nouveau graphe/eval
→ finish`, ce qui évite le défaut de buffer préexistant rencontré par les
anciens essais. Le helper de contrôle filesystem déjà qualifié est réutilisé,
sans nouveau walker.

Les **deux traces cumulées occupent 521 317 700 octets**, soit moins que le
plafond **cumulatif 1 073 741 824 octets (1 Gio)**. Cette campagne de deux
traces respecte son plafond au contrôle après écriture ; ce contrôle n'est
pas une limitation active de la taille pendant l'enregistrement.

L'[inspection physique Xcode 27.2](xcode-dispatch-inspection.json) observe :

| Chemin R20 | Dispatchs |
| --- | --- |
| Stock | **3** : SwiGLU compilé stridé, `vv_Multiplyfloat16`, `n2hadamard_128_float16`. |
| Fusion | **2** : SwiGLU+scale compilé stridé, **même fonction native H128** `n2hadamard_128_float16`. |

Les deux graphes ont 2 command buffers et 1 encodeur compute ; aucun encodeur
render/draw observé. Le second command buffer n'a pas été développé : aucun
compteur universel de blits n'est affirmé. Le profilage après replay était
**OFF** ; le `0 ns` affiché n'est **pas une mesure**. Le toolchain shader debug
optionnel était absent, son téléchargement a été décliné et le replay a
fonctionné. Les workloads ont été stoppés et Xcode quitté, `isRunning=false`
vérifié avant la phase de compilation suivante.

Cela confirme **un dispatch retiré à cette frontière**, sans démontrer de
temps GPU gagné, GPU idle supprimé, ni débit du modèle accru.

## Admission révisée avant résultats et conditions conservées

Le premier prévol restait insuffisant à environ 10,210 Go ; après libération
RAM, l'[observation r1](model-admission-observation-r1.json) a encore refusé
**11 244 371 968 <12 000 000 000 octets**, après 0,042 s, **avant PID**.
Elle n'a ni observé 60 secondes stables ni lancé un modèle. Ce refus et
les [plans originaux à 12 Go](admission12-original/) sont conservés.

Après fermeture de Brave, environ 11,4 Go étaient disponibles. L'amendement
**T04-ADMISSION11-01**, en tête d'[opti.md](../../../opti.md), a changé la
marge **heuristique volontaire** de 12 à **11 Go avant tout modèle complet
ou chrono T04**, après avis indépendant. Ce n'est ni une propriété numérique
requise par AGENTS ni une garantie anti-swap : l'incident T17 à +1 Mio
malgré 11,952 Go estimés reste explicitement conservé. Aucun résultat
préexistant n'a été requalifié sous le nouveau seuil.

L'[observation r2](model-admission-observation-r2.json) passe **4 instantanés
sur 60,289 s**, disponibles 11,404 /11,392 /11,358 /11,402 Go, minimum exact
**11 358 322 688 octets**, secteur/quiet. La même ancre **T04-w1** est
strictement conservée, Swapouts absolus **140 865 945 600 octets** ; aucun
reset. Les dix sélections terminées ont été jointes sans nouveau Swapout.
Cela ne préjuge ni de la suite ni du swap historique avant l'ancre.

Le seuil de **11 000 000 000 octets avant chaque processus complet**,
60 secondes stables avant reprise, Swapin ≤256 Mio, RSS supervisé ≤10 Gio,
footprint natif <12 Go aux observations, cache experts **3 Gio**,
53 slots ×48 couches et 4 lecteurs restent les conditions de cette phase.
La garde micro/composant reste 2 Go. Aucun changement du budget experts ou
des limites RSS/footprint ; arrêt au premier nouveau Swapout, même au
chargement, sans réancrage automatique. Builds/proveurs/replays sont séparés
du GPU. Les footers natifs ne constituent pas un suivi continu universel
de tous les pics de footprint.

## Bridge, pilote et audit avant retrait

Le [bridge17](bridge17-inspection.json) a terminé **6/6 processus**,
24 réponses complètes (12 chauffes et 12 mesures), 17 tokens et 16
transitions chacune. Les goldens externes, modes, compteurs actifs et budgets
ont été inspectés ; processus et lecteurs joints, zéro nouveau Swapout
depuis T04-w1. Ce bridge ne constitue pas un gain de vitesse à 128 tokens.

Le [pilote128](pilot128-inspection.json) a ensuite terminé **12/12 processus**,
24 chauffes +24 mesures, **48 réponses** de 128 tokens/127 transitions,
texte/hash indépendants complets et compteurs actifs homologues A/C
identiques. Même CLI candidat `81bc…`, flag0/1, admission11 Go et cache3 Gio ;
processus/lecteurs joints, zéro nouveau Swapout depuis la même ancre jamais
réinitialisée. Pic natif de vie rapporté : **7 902 009 944 octets**.

| Mode, MAIN/HOLD agrégés | Décode A→C (tok/s) | Variation décode | Variation préfill |
| --- | --- | --- | --- |
| AR | 10,468587 →10,296765 | −1,641318 % | −0,428787 % |
| MTP | 10,543890 →10,589444 | +0,432035 % | −0,738049 % |
| DFlash | 11,187542 →11,292933 | +0,942043 % | +0,179787 % |

Le débit est calculé par somme des transitions /somme des durées, séparément
par mode, sans additionner leurs écarts. Les [résultats détaillés](pilot-results.md)
conservent MAIN/HOLD, les plages et les dérives de contrôle, notamment
+4,3818 % en MTP/HOLD et −3,5466 % en DF/HOLD. Les petits signaux positifs
ne démontrent pas un gain causal général ; AR régresse dans ce pilote.

L'[audit source](audit-performance-source.md) et l'
[audit des 48 réponses](audit-performance-data.md) ont été relus avant le
retrait : formule fidèle à PR61, dtypes/H128/splits/fallbacks cohérents,
flags/binaire/tokens/compteurs vérifiés, demandes logiques homologues A/C
identiques ; aucune copie, synchronisation ou lecture hôte nouvelle
identifiée. Les coûts propres du troisième input, des vues et des métadonnées
n'ont pas été isolés ; aucune causalité exclusive compression/SSD n'est
affirmée. **Rejet pour intégration de cette adaptation**, sans prétendre que
l'idée amont est universellement inefficace.

## Retrait sélectif appliqué ; contrôles du code conservé passés

Le [résultat de retrait](retirement-result.json) établit **8 fichiers
restaurés**, le seul nouveau fichier `native/src/ssd_moe/swiglu_down_tests.rs`
supprimé après archivage byte-exact, **190/190 SHA identiques au pré-T04**,
zéro divergence et autres modifications locales préservées. Les neuf
sources candidates exactes, leurs tests/proofs, le CLI `81bc…`, le libtest
`22073b…`, les captures et résultats restent des artefacts historiques.
Ils ne sont pas l'identité du code conservé. T17/T05 n'ont pas été modifiés
par ce retrait ; aucune application n'a été remplacée.

Le [plan du code restauré](retained-checks-plan-result.json) est terminé ;
l'[inspection finale](retained-checks-inspection.json) établit : format,
Clippy strict CPU/MLX et build CLI passés, **CPU 256 passés /0 échec /8 ignorés**,
**MLX 293 passés /0 échec /185 ignorés**, cibles agrégées et processus joints.
Les ignorés restent non exécutés. Les **190 sources pré-T04 sont identiques**.

CLI normal restauré :
`202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa`,
**byte-identique au CLI pré-T04** ; libtest normal :
`a751cf592d9e28622d1c062a9f93213f4c55d8a6dcce0fa8b1c035be16c6b862`,
également revenu à son identité pré-T04. **Aucun nouveau test GPU ni chrono
n'a été lancé après le retrait.** Les 258/295 tests, 166 obligations/
12 couvertures, dix qualités GPU, six processus bridge et douze processus
pilote ci-dessus restent ceux du candidat archivé `81bc…/220…`, pas ces
nouveaux binaires normaux. CI distante de l'arbre local non publié toujours
non exécutée. **T04 clos et retiré**, sans gain intégré ni mise à jour de
l'application.
