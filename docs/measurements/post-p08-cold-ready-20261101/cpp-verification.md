# Cold Ready — vérification du lecteur C++ asynchrone

État au 1 novembre 2026, 12:08 UTC : implémentation locale terminée ; tests CPU avec sanitizers et syntaxe du harnais passés. Les essais GPU et la validation du parcours MLX ne sont pas couverts par ces contrôles. Aucun résultat de vitesse.

## Contrat réellement exercé

`DiskReadBatch` utilise au plus quatre workers GCD statiques. Un job devient prêt après tous ses segments, publication release / lecture acquire. Les masques excluent les erreurs. `finish()` et le destructeur libèrent le retard de test puis rejoignent tous les workers ; les erreurs sont propagées par index de job canonique et restent répétables. Le propriétaire FFI conserve les tableaux MLX et le wrapper Rust conserve les fichiers jusqu’à cette jointure, sur le thread d’inférence. Aucun worker n’appelle MLX.

`Array::begin_read_records` réutilise la validation du lecteur synchrone ; C++ vérifie aussi les destinations entre records avant tout lancement. La sécurité GPU dépend encore du contrat unsafe de l’appelant : ne lire que les slots prêts et drainer tous les consommateurs avant invalidation/réutilisation. Le wrapper Rust est non Send / non Sync. Le test de retard est un setter Rust cfg(test), sans variable d’environnement de production.

## Contrôles terminés

| Contrôle | Résultat | Durée |
| --- | --- | --- |
| `cpp-asan-build-r3` | exit 0 | 1.238 s |
| `cpp-asan-tests-r2` | exit 0 | 0.280 s |
| `cpp-tsan-build` | exit 0 | 0.622 s |
| `cpp-tsan-tests` | exit 0 | 2.704 s |
| `cpp-proof-syntax` | exit 0 | 0.285 s |

Les 41 mêmes cas passent sous ASan+UBSan puis TSan : ce sont 41 cas distincts, pas 82. Domaine : 1–512 records, 1–4 workers, tailles 1/2/3/6/20/512, ordre statique, masques non contigus avec job retenu, succès avant la fin du lot, finish répétable, destructeur qui rejoint, délai expiré, erreurs simultanées canonisées après drainage, lecture courte réelle, injection EINTR / lectures partielles dans le syscall seulement, bornes invalides et destinations chevauchantes rejetées avant écriture. Les données sont des fichiers temporaires jetables ; aucune inférence.

Commandes exactes (également dans `cpp-checks-r2.json`) :

```sh
/usr/bin/c++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -I native/ffi native/tests/disk_reader_async.cpp -o build/post-p08-cold-ready-20261101/cpp/disk-reader-async-asan
```

```sh
build/post-p08-cold-ready-20261101/cpp/disk-reader-async-asan
```

```sh
/usr/bin/c++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=thread -I native/ffi native/tests/disk_reader_async.cpp -o build/post-p08-cold-ready-20261101/cpp/disk-reader-async-tsan
```

```sh
build/post-p08-cold-ready-20261101/cpp/disk-reader-async-tsan
```

```sh
/usr/bin/c++ -std=c++17 -Wall -Wextra -Werror -I native/ffi -fsyntax-only native/tests/disk_reader_async_proof.cpp
```

Deux échecs de préparation sont conservés : `cpp-asan-build.log` (comparaison signed/unsigned dans le test), puis `cpp-asan-tests.log` (le cas d’offset UINT64_MAX produisait overflow_error, alors que le test n’attrapait que invalid_argument). Correction des tests uniquement ; les relances figurent dans `cpp-asan-build-r3.log` et `cpp-asan-tests-r2.log`.

## Vérification source : limite et arrêt explicite

Le harnais `native/tests/disk_reader_async_proof.cpp` appelle la vraie classe C++ avec deux destinations symboliques bornées, sans remplacement des atomiques, de GCD ni du lecteur. Sa syntaxe C++17 stricte passe. **CBMC n’a pas été lancé et aucune propriété formelle de ce lecteur n’est vérifiée.** La mutation temporaire du masque d’erreur, prévue pour qualifier l’oracle, n’a pas été exécutée.

À 12:06 UTC le parent a arrêté les compilations et proveurs après une pression mémoire extrême observée pendant un autre contrôle Kani. Le plafond global Swapout est dépassé ; aucune nouvelle compilation, preuve, mutation ou mesure GPU n’est autorisée jusqu’à résolution du protocole. Nos propres commandes étaient toutes jointes à 12:05:36.493 UTC ; le snapshot de 12:07:59 UTC ne trouve aucun processus disk-reader-async / cbmc / kani / rustc / clang / c++. Les tests sanitizers ne prouvent ni tous les entrelacements, ni MLX/Metal, ni la transaction SsdMoe, ni la génération entière. Le harnais formel et sa tentative restent à qualifier lorsque le parent autorisera une reprise.

## Identité des sources à la fin de cette tâche

Archive avant édition : `before-async/identity.json` et copies associées. Aucun changement des fichiers existants disk_reader.h / disk_reader.cpp / disk_reader_proof.cpp.

| Source | SHA-256 |
| --- | --- |
| `native/ffi/disk_read_batch.h` | `1d9da90a1e2b9a61cb926d1dd17c4f172be2f2f39f4d20f17d8cf20bb4ab26df` |
| `native/ffi/mlx_bridge.cpp` | `b713228b11f9b5b2174ef99065e7c9dce9ae58bff542dbd685add28652344d8c` |
| `native/src/array.rs` | `3938aec8c7eb44fc16f82cea670f67a1d700a5f339efc36f00384fb670f21224` |
| `native/tests/disk_reader_async.cpp` | `dc7d87e559cb730e22be7e715dc49cea747a3d8755aabd06ed923126d441848d` |
| `native/tests/disk_reader_async_proof.cpp` | `6ef43e89360134218e2798e4231391ade434de703a918f092a073d98423623f9` |

La compilation Rust/FFI MLX, les suites ordinaires et les tests physiques de cohortes/erreurs restent coordonnés par le parent. La CI doit ajouter les deux sanitizers pour le nouveau test et la syntaxe du nouveau harnais ; aucun workflow n’a été modifié par cet agent.
