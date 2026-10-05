# Reproduire la deuxième campagne

Ne lancer qu'un modèle à la fois sur un GPU Apple physique. Séparer les mesures
GPU des compilations et autres inférences. Utiliser des nouveaux chemins de
sortie : les scripts refusent d'écraser les campagnes précédentes.

## Build et contrôles CPU

Avec Rust 1.99.0 et un `MLXL3_MLX_ROOT` pointant vers MLX 0.32.2 :

```sh
cargo fmt --check
cargo clippy --release --locked --all-targets --features mlx,chat -- -D warnings
cargo test --release --locked --features mlx,chat
cargo build --release --locked --features mlx,chat
python -m pytest -q tests/test_bridge_benchmark.py tests/test_package_engine.py tests/test_qmm_tiles.py tests/test_compare_native.py tests/test_native_json_process.py tests/test_decode_microbenchmarks.py
```

Le build final archivé correspond au source de `468276a` ; les commits
documentaires suivants ne changent pas son calcul. Les empreintes des fichiers
exécutables et du binaire sont dans `provenance.json`.

## Filtres synthétiques

Ces commandes importent MLX et demandent un GPU Apple :

```sh
python benchmarks/benchmark_decode_tiles.py --iterations 40 --output /tmp/new-tiles.json
python benchmarks/benchmark_gdn_norm_gate.py --iterations 20 --output /tmp/new-norm-gate.json
```

Les anciens fichiers `tiles-screen.json`, `tiles-confirm.json` et
`gdn-screen-final.json` sont produits avant le durcissement du rapport CLI.
Les matrices, calculs et frontières des mesures sont identiques ; les nouveaux
rapports ajoutent l'état de campagne et conservent aussi les erreurs.
Les logs d'échecs de compilation de la fusion GDN sont volontairement présents.

## Checkpoint exact et paires natives

La référence numérique a été sauvegardée avec la baseline de la première
campagne. Elle n'est pas incluse dans Git (environ 1,5 Gio). Ne pas la réécrire
pendant un contrôle candidat.

```sh
export MLXL3_QWEN_TEST_MODEL=/path/to/Qwen3.8-27B-exl3-6a9ca9d0
export MLXL3_QWEN_REFERENCE_DIR=/path/to/baseline-qwen-reference
unset MLXL3_QWEN_WRITE_REFERENCE
cargo test --release --locked --features mlx,chat --lib checkpoint_optimization_matches_saved_logits_and_states -- --ignored --nocapture --test-threads=1
cargo test --release --locked --features mlx,chat --lib checkpoint_dense_decode_tiles_paired -- --ignored --nocapture --test-threads=1
```

Le test apparié restaure le même snapshot préfill, impose les mêmes tokens et
compare toutes les sorties finales. Le premier JSON peut être préfixé du nom
du test Rust dans le log, ce qui ne supprime pas la première paire.

## Bridge

Utiliser le même binaire final pour A et B. A remet le dispatch précédent avec
`MLXL3_DENSE_DECODE_NT4=0`; B utilise la sélection conservée. Le script prévoit
une requête warmup puis deux requêtes mesurées par prompt et par passe.

```sh
python benchmarks/compare_native.py "$MLXL3_QWEN_TEST_MODEL" \
  --baseline target/release/mlxl3-rs --candidate target/release/mlxl3-rs \
  --baseline-env MLXL3_DENSE_DECODE_NT4=0 \
  --tokens 24 --repeats 2 --order BAAB --settle-seconds 25 \
  --prompt-file docs/measurements/qwen27-m5/prompt-document.txt \
  --output /tmp/new-nt4-baab
```

Le contexte est4096, cache/MTP/DFlash/MCP désactivés, température0. Le résultat
bridge conservé est numériquement fonctionnel mais son débit est non concluant
à cause des dérives de performance, dont celles du contrôle préfill inchangé.
La répétition ABBA utilise les mêmes paramètres avec `--order ABBA`.

## Preuve et limites

La commande Kani ciblée appelle le vrai contrat Rust, sans hypothèse réduisant
son domaine :

```sh
cargo kani --lib --no-default-features --harness dense_decode_tile_never_crosses_unmeasured_shapes_or_output_boundaries
cargo kani --lib --no-default-features
```

Les premières tentatives Kani, CBMC et CrossHair faute d'outil sont conservées
dans `*-local.log`. Kani0.68.0 a ensuite été installé dans l'outillage isolé,
avec `KANI_HOME=../work/kani` et Cargo/Rustup dans `../work/toolchain`. Le contrat
ciblé puis les32harnaisCPU passent surARMApple ; détail dans
`kani-arm-results.json`, logs ciblés et bruts complets`kani-arm-full.log.gz`.
La CI Ubuntu prévoit aussi Kani0.68.0 ; distinguer son statut réel de la preuve
locale. Ce contrôle CPU ne prouve pas Metal/MLX ni l'intelligence du modèle.
Les tests GPU exacts et les traces bridge sont des preuves expérimentales
séparées. La sonde C++ du cache est retirée ; seul son patch de diagnostic est
conservé. Les logs bruts gardent les espaces et fins de ligne des outils.
