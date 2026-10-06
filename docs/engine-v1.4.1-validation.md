# Moteur 1.4.1 — intégration PR25

Source livrée : `71588ab9c758ad95bb6b0041bed6c061fb7a70c7`, combinaison du main
1.4.0 `22a53bc` et de PR25 `c6406a8`. Les seuls conflits concernaient CI/opti ;
les deux historiques, le runner Kani Ubuntu22.04 et les contrôles MTP ont été
conservés. Cette source est poussée sur la branche originale de PR25 dans le
fork de l'auteur, en fast-forward. Desktop reste 1.4.0/build23.

## Contrôles locaux

| Contrôle | Résultat |
| --- | --- |
| Rust release MLX/chat, format, Clippy strict, build | 62 passés, 55 ignorés, aucun échec |
| Rust CPU/chat, Clippy strict, build | 51 passés, 3 ignorés, aucun échec |
| Python complet, Ruff 14 fichiers, compilation 8 scripts | 208 passés, 4 skips ; lint/format réussis |
| Desktop E2E, SDK26.5 | Réussi : cycle de vie, bridge, rendu, imports, CLI, MTP, updater |
| Kani0.68.0/CBMC6.11.0, CPU sans features | 34 harnais, 4 763 SUCCESS +70 UNREACHABLE, 80 covers satisfaites |
| Checkpoint dense physique | 12 positions, 248320 logits finis et128 états finis/non vides par position, octets identiques |
| Bridge du moteur signé, dense→MoE→dense | 36 cas D0..3/budgets1,3,17, parité non vide, nettoyage et processus rejoints |
| Archive/updater Swift de production | Signatures/arm64, 5 fichiers/hashes, installation/move/exécution/repli dans stores jetables réussis |

Le premier contrôle Desktop a échoué avec le SDK27.0 implicite du CLT : plugin
SwiftUIMacros absent. La relance avec SDK26.5 a réussi sans changement de code.
Le log négatif est conservé. Les quatre skips Python concernent ponyexl3 et le
checkpoint Ling absents. Un test GPU ignoré a ensuite été sélectionné seul ;
les 54 autres tests ignorés de la variante MLX ne sont pas déclarés exécutés.

La garde NT4 est vérifiée sur tous les `i32/i32/usize/bool/bool`, sans `assume`,
boucle ni stub : 7 obligations et 3 covers. Les bornes et assertions d'unwinding
des 33 autres harnais sont inchangées. Les 70 obligations inatteignables restent
préexistantes ; aucun échec/timeout. Ceci est du model checking Rust CPU borné,
sans preuve Metal/MLX/FFI/allocateur/concurrence/Swift/génération complète.
La tentative CrossHair de revue sur les microfiltres n'a trouvé aucun contrat
analysable : ces chemins Python restent non vérifiés formellement.

Les 27 fichiers de checkpoint/têtes et 1548 références sauvegardées conservent
leurs SHA-256 historiques. La tête MoE réellement utilisée dans Drafts conserve
les hashes de la copie historique dans models. Le test dense lit les références
existantes et ne les réécrit jamais. M5/GPU10/macOS27.2/MLX0.32.2, batterie48%
en décharge au relevé ; compilations/provers terminés avant les contrôles GPU,
un seul modèle chargé à la fois. Aucun nouveau gain vitesse/RAM mesuré.

## Commandes

Depuis le worktree isolé, PATH inclut `/Users/justin/.cargo/bin` et Python3.12
provient du venv du projet. Les commandes de build MLX utilisent
`MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx`
et `MACOSX_DEPLOYMENT_TARGET=26.2`.

```sh
cargo fmt --check
cargo clippy --release --locked --all-targets --features mlx,chat -- -D warnings
cargo test --release --locked --features mlx,chat
cargo build --release --locked --features mlx,chat
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo test --locked --features chat
cargo build --locked --features chat
MLXL3_TEST_PYTHON=<venv>/bin/python MLXL3_SKIP_METAL_CHECK=1 scripts/check-e2e.sh
MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  MLXL3_TEST_PYTHON=<venv>/bin/python MLXL3_SKIP_METAL_CHECK=1 scripts/check-desktop.sh
cargo kani -Z unstable-options --lib --no-default-features \
  --harness dense_decode_tile_never_crosses_unmeasured_shapes_or_output_boundaries \
  --harness-timeout 120
cargo kani -Z unstable-options --lib --no-default-features --harness-timeout 120
MLXL3_QWEN_TEST_MODEL=<dense> MLXL3_QWEN_REFERENCE_DIR=<références historiques> \
  target/release/deps/mlxl3_native-fb2fec27ffba9a03 --exact \
  qwen35::tests::checkpoint_optimization_matches_saved_logits_and_states \
  --ignored --nocapture --test-threads=1
python scripts/package_engine.py build/engine-v1.4.1/runtime dist --version 1.4.1
build/engine-v1.4.1/updater-runtime dist/MLXL3-Engine-v1.4.1-arm64.tar.gz 71588ab9c758
python scripts/check-mtp-model-switch.py build/engine-v1.4.1/runtime/mlxl3 \
  <dense> <moe> --timeout 180 --output <preuves>/model-switch-packaged.json
```

Le test Rust physique réutilise le binaire exact de la suite release précédente
(SHA-256 dans checkpoint-parity.json) pour éviter toute compilation concomitante.
Les trois Mach-O sont copiés puis signés ad hoc (`codesign --force --sign -`),
contrôlés par `lipo -verify_arch arm64` et `codesign --verify --strict`.
Les deux harnais de livraison Swift sont compilés depuis UpdateManager,
EngineRuntimeStore et Localization réels avec Swift6, warnings-as-errors et SDK26.5.

## CI et publication

CI de la tête exacte PR25 : **quatre jobs réussis**, logs inspectés et arbre du
merge synthétique d1ca14f identique à71588ab :
[Native37509900320](https://github.com/0xZKnw/mlxl3/actions/runs/37509900320),
[Desktop37509900331](https://github.com/0xZKnw/mlxl3/actions/runs/37509900331).
PR25 est fusionnée dans main par69d22b7, qui conserve exactement le même arbre.
Native :51testsRust parOS (2ignorésLinux/3macOS),206Python parOS ; Desktop208/4skips
et E2E ; Kani34harnais. Les premiers runs sur la branche de préparation sont
distingués dans les preuves. Les quatre jobs du merge69d22b7 et les quatre du tag71588ab ont également
réussi ; leurs SHA/jobs sont inspectés et conservés séparément (ci-main-*.json,
ci-tag-*.json). Aucun succès des mêmes tests n’est additionné comme une nouvelle
propriété de correction.

Archive locale : `MLXL3-Engine-v1.4.1-arm64.tar.gz`, **66 866 870 octets**,
SHA-256 `e220ae2e4fccae7515f9601a49ec0007730798cf2bf764177de96a09a3a81d0b`.
Version1.4.1, protocole1, arm64, minimum macOS26.2/Desktop1.4.0, MLX0.32.2.
Binaire construit depuis une source propre71588ab ; seuls journal/preuves ont
été modifiés après le build. Aucune app personnelle remplacée.

[Release publique](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.1)
vérifiée par téléchargement HTTPS sans authentification : taille/SHA-256 identiques
au paquet local testé et au digest GitHub. La sélection UpdateManager réelle
retrouve Desktop1.4.0/build23 et moteur1.4.1 ; latest reste v1.4.0. Les preuves
sont publication-proof.json et updater-selection.json.

Les commandes/options, conditions, rapports et logs sont dans
[les preuves de livraison](measurements/engine-v1.4.1/) et
[la revue indépendante](measurements/pr25-review/), avec résumé dans opti.md.

Les arguments complets des20commandes, variables, bornes Kani et résultats sont
dans [verification-summary.json](measurements/engine-v1.4.1/verification-summary.json).
Les logs cités sont archivés sans perte avec suffixe `.log.gz` ; archive-index.json
conserve les SHA-256 et tailles décompressées. Les originaux restent dans
`build/engine-v1.4.1/raw-proof-logs/` du worktree isolé.
