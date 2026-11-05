# T01-P-01 — vérification du diagnostic hôte

31 octobre 2026. Synthèse des artefacts réellement inspectés, sous le préenregistrement `ISSUE59-T01-P-01`. Le diagnostic construit puis détruit des graphes Hot GU différés ; il ne porte pas encore le launcher préparé. **Aucun modèle, calcul de valeurs, parité numérique ou débit tok/s n’est validé.**

## Identité et périmètre

Bibliothèque de tests Apple figée : `build/issue59-t01-20261031/libtest-r1/prototype-engine`, SHA256 **`20a0006f54e00bbae2e937b9c86f73fcc24dfe8262d36b7c5d1179984cae8e0c`**,11036592 octets. MLX0.32.2, Rust1.98.1, arbre `7978fceac30d991a8c06d72114fc932b2443f555` avec modifications locales préservées. [Identité](instrumented-identity-r1.json) :500 empreintes source/dépendances ; [inspection indépendante](host-inspection.json) :0 manquante/0 écart. Sources instrumentées sauvegardées dans [instrumented-source](instrumented-source/), distinctes du futur code conservé.

Le collecteur réel `Stats::checked_record` possède sept compteurs nanosecondes et un compteur d’appels. Un overflow garde tous les compteurs précédents et rend `valid=false` de façon persistante. Les modes/cible sont locaux au thread ; Off n’enregistre rien, Coarse conserve uniquement le total, Fine conserve les six phases et leur total. L’instrumentation Rust est limitée aux tests ; le diagnostic C++ reste désactivé par défaut.

## Vérification source et limites

**Kani0.68.0/CBMC6.11.0, contrôle borné ciblé réussi** : `cargo kani --lib --no-default-features --harness kernel_build_profile_records_checked_counters_atomically --solver kissat`. Commande et résultat dans [plan détaillé](collector-kani-detail-plan.json) et [log complet](t01-collector-kani-detail.log).

- Entrées `valid:bool`, `calls:u64`, `ns:[u64;7]`, premier delta et delta ultérieur symboliques, **sans `assume`**. Assertions champ par champ contre additions `u128`, tout-ou-rien et invalidité persistante ; implémentation réelle appelée, sans stub. Unwind129 avec assertions d’unwinding conservées ; boucles fonctionnelles de sept éléments.
- Résultat : **257 obligations,0 échec,1 inatteignable ;4/4 covers satisfaits**. La lecture indépendante du détail confirme que l’unique obligation inatteignable est Check55, `std::fmt::Arguments::from_str.assertion.1`, « attempt to shift left with overflow », `core/src/fmt/mod.rs:820:38`. Ce n’est pas une assertion de contrat rendue vacuement vraie.
- Les quatre covers concernent succès, état déjà invalide, overflow appels et overflow d’un champ. Le cover placé dans une boucle produit **une seule propriété** : aucune revendication de sept couvertures indépendantes. Les assertions restent universelles sur les sept compteurs ; les tests CPU isolent chaque champ.

**Chemin Array/MLX non vérifié formellement** : tentative réelle `cargo kani --lib --tests --features mlx,chat --harness actual_hot_gu_kernel_builder_mlx_entry`, précondition d’ID symbolique<512, unwind129 ; ICE `kani-compiler/src/intrinsics.rs:243` avant obligations. [Plan](actual-mlx-kani-plan.json), [diagnostic](t01-actual-mlx-kani.log). Aucun succès de preuve MLX/Metal/GPU déduit du contrôle CPU.

**C++ non vérifié formellement** : CBMC sur le vrai `native/ffi/mlx_bridge.cpp`, fonction `mlxl3_kernel_build_profile_copy`, includes installés, `--cpp11 --unwind 129 --unwinding-assertions --bounds-check --pointer-check --signed-overflow-check --unsigned-overflow-check`. Parsing libc++ refusé : **0 propriété vérifiée**, ni succès ni contre-exemple fonctionnel. Le contexte exact et la limite C++11/headers modernes restent dans [plan](cbmc-ffi-plan.json) et [log](t01-cbmc-ffi.log). Aucun modèle réécrit ailleurs pour remplacer ce contrôle.

Ce n’est **pas une preuve du package complet**, de l’horloge, du TLS concurrent, des allocations, de la FFI, de MLX ou du calcul numérique.

## Tests et compilation inspectés

| Contrôle | Résultat et preuve |
| --- | --- |
| `cargo test --locked --release --lib --no-default-features kernel_build_profile::tests:: -- --test-threads=1` | **4 passés/0 échec/0 ignoré**,178 filtrés. Limites exactes, overflow appels+chaque champ, sticky/reset/modes/nom exact et isolation threads. [Plan](collector-tests-plan.json), [log](t01-collector-tests.log). |
| Exécutable Apple figé, filtre `kernel_build_profile`, puis `linear::launch_profile_tests::launch_profile` | **7+2 passés/0 échec/0 ignoré**,376 puis381 filtrés. Inclut les quatre tests du collecteur réexécutés, les gardes métadonnées/FFI et les rejets du validateur de rapport : ne pas additionner ces recouvrements comme tests uniques. [Plan](metadata-tests-plan.json), logs [collecteur](t01-ffi-collector.log)/[rapport](t01-report-validation.log). |
| Suite bibliothèque Apple figée, `--test-threads=1` | **207 passés/0 échec/176 ignorés/0 filtré**,18,52s. Les tests physiques ignorés restent non exécutés. [Plan](instrumented-lib-suite-plan.json), [log](t01-instrumented-lib-suite.log). |
| Format, Clippy strict CPU et MLX, build réel libtest MLX/chat | **Passés** : `cargo fmt --all -- --check`, `cargo clippy --locked --all-targets --no-default-features -- -D warnings`, variante `--features mlx,chat`, puis `cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json`. [CPU](clippy-cpu-plan-result.json), [r2](build-smoke-r2-plan-result.json). |

L’échec r1 Clippy `collapsible_if` du cleanup a précédé le build et la première mesure. Correction stylistique ciblée puis r2 réussi ; [échec r1](t01-clippy-mlx.log) et [résultat r1](build-smoke-plan-result.json) conservés.

## Filtre hôte exécuté et audit avant retrait

Sélection exacte ignorée explicitement exécutée : `linear::launch_profile_tests::hot_gu_post_hit_construction_profile --exact --ignored --nocapture --test-threads=1`, via [plan hôte](host-r1-plan.json),180s maximum. **1 passé/0 échec**,8 lots×3 modes×256 constructions = **6144 appels mesurés**,64 chauffes,24 échantillons. Les2048 appels Fine ont0 miss/0 éviction ; toutes sorties différées,0 nouveau readback. [Rapport](host-profile-r1.json), [inspection](host-inspection.json), [supervision](host-r1-plan-result.json) : processus joint, **0 nouvelle écriture swap/0 swapin** sous ancre T01-w1. Ce sont des temps de construction hôte, sans `eval`, `async_eval`, synchronisation ni lecture de valeurs.

La revue avant mesure a trouvé un retour possible de timer après FFI avant adoption des handles : corrigé **avant la première mesure**, tous les handles sont adoptés avant propagation de cette erreur. Fin/adoption ultérieures bénéficient de RAII. Off/Coarse/Fine utilisent0/2/7 lectures d’horloge ; les six différences Fine partitionnent le total, mais le bookkeeping charge la phase suivante. C++ est inclus dans FFI ; accumulation et validations externes ajoutent un coût d’observation. Aucun poste inclusif additionné, aucun microtemps converti en gain modèle. Les autres audits sont conservés séparément avec leurs limites.

Workflow PR/main inspecté, harnais Kani ciblé temporaire ajouté. **Aucun push ni run CI distant du code local** ; le workflow seul ne constitue pas une CI verte.

## Code conservé — contrôles finaux terminés

L'instrumentation a été retirée après les deux audits : cinq fichiers restaurés
depuis les snapshots antérieurs vérifiés, deux nouveaux modules archivés puis
retirés ;493 autres sources/dépendances figées inchangées. Les190 sources du
moteur conservé sont identiques à l'état pré-T01. La procédure documentaire PR61
et les preuves restent conservées. [Reçu](retirement-result.json).

Les six processus du [plan final](retained-checks-plan.json) terminent avec
exit0, tous joints, zéro nouvelle écriture swap :

- `cargo fmt --all -- --check` et Clippy strict tous targets CPU/MLX passent.
- `cargo build --locked --release --features mlx,chat --bin mlxl3-rs` passe.
- `cargo test --locked --release --no-default-features -- --test-threads=1` :
  **256 passés,0 échec,8 ignorés**.
- `cargo test --locked --release --features mlx,chat -- --test-threads=1` :
  **293 passés,0 échec,185 ignorés**. Les tests physiques ignorés restent non
  exécutés dans cette vérification finale ; ne pas sommer ces suites comme
  des tests uniques ni qualifier leurs sélections vides comme contrôle GPU.

[Inspection finale](retained-checks-inspection.json) et
[identité](retained-engine-identity.json) : CLI SHA256
`202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa`,
libtest `a751cf592d9e28622d1c062a9f93213f4c55d8a6dcce0fa8b1c035be16c6b862`,
tous deux **byte-identiques avant T01**. Le binaire20a et le contrôle Kani du
collecteur décrivent le prototype instrumenté archivé, pas ce moteur restauré.
Le harnais temporaire a aussi été retiré du workflow PR/main ; aucun contrôle
CI distant de modifications locales non poussées n'est revendiqué.

**Tous les essais T01 lancés sont clos.** T01-P est un diagnostic hôte,
T01A source-seule est différé, T01B hash/pipeline/encodeur n'a pas été mesuré.
Aucun modèle chronométré, parité numérique ou gain tok/s établi ; aucune app
remplacée, publication, commit ou push effectué.
