# Commandes et artefacts small-M

Base `5de518e0fbe7d3367416b9bc5999c7bd1f92c74d`, checkout `mlxl3-smallm`, branche `optimize/qwen27-smallm-verify`. Les commandes ci-dessous sont des recettes de reproduction : les mesures originales des 6–7 octobre sont conservées, y compris échecs et interruptions. Ne pas écraser les chemins de résultats existants ; les CLI le refusent.

```sh
source ../work/build-env.sh
PYTHON=../work/venv/bin/python
SMALLM_MODEL='/Users/henko/Library/Application Support/io.mlxl3.desktop/Models/Qwen3.8-27B-exl3-6a9ca9d0'
SMALLM_HEAD='/Users/henko/Library/Application Support/io.mlxl3.desktop/Drafts/Qwen3.8-27B-MTP-4bit'
SMALLM_OUT=docs/measurements/qwen27-smallm
```

Les GPU doivent être exécutés seuls, modèle Desktop éjecté pendant la campagne et rechargé ensuite ; aucun build, proveur, test CPU ou autre inférence pendant les mesures. La première campagne commence sur batterie et passe sur secteur au cinquième passage. La tentative A-AC commence sur secteur, puis revient sur batterie au cinquième passage ; les relevés réels figurent par passage dans les JSON. Une répétition sur secteur doit enregistrer ses propres conditions et rester distincte.

## Instrumentation et filtres GPU

```sh
"$PYTHON" benchmarks/benchmark_smallm.py --checkpoint "$SMALLM_MODEL" --inventory-only --output "$SMALLM_OUT/inventory.json"
"$PYTHON" benchmarks/benchmark_smallm.py --checkpoint "$SMALLM_MODEL" --iterations 40 --output "$SMALLM_OUT/grouped-mb3-screen.json"
"$PYTHON" benchmarks/benchmark_smallm.py --checkpoint "$SMALLM_MODEL" --iterations 40 --output "$SMALLM_OUT/grouped-mb3-confirm.json"
"$PYTHON" benchmarks/benchmark_smallm_gdn.py --checkpoint "$SMALLM_MODEL" --iterations 40 --output "$SMALLM_OUT/gdn-columns-screen.json"
MLXL3_EXPERIMENTAL_SMALLM_TENSOR=1 "$PYTHON" benchmarks/benchmark_smallm_tensor.py --checkpoint "$SMALLM_MODEL" --iterations 40 --output "$SMALLM_OUT/tensor-bm16-screen.json"
"$PYTHON" benchmarks/benchmark_smallm_inventory.py --checkpoint "$SMALLM_MODEL" --iterations 20 --output "$SMALLM_OUT/inventory-timings.json"
```

Logs de même stem `.log`, issus de stdout/stderr réunis. Warmups3 pour les microfiltres, 40 paires AB/BA pour A/B/C (C s’arrête avant tout timing), 20 paires de contrôles pour l’inventaire. Chaînes de8 projections, temps divisé par8. Scales/transforms/codebooks/trellis originaux, fixtures déterministes avec seeds27600/27601/27602/27603 selon script ; les sources exactes et leurs hashes sont enregistrés dans `campaign-identity.json`.

`inventory-initial-sg-diagnostic.json` conserve l’inventaire initial erroné SG4 pour les K2 separate ; les fichiers finaux utilisent SG8. `inventory.json` est statique, `parity:null` ; seul `inventory-timings.json` porte les contrôles numériques exécutés et `parity:true`.

## Prototype natif A, retiré

Les commandes suivantes nécessitent **le patch archivé**, absent du runtime final. Il contient le sélecteur, son test/harnais Kani, la garde de dispatch et le test apparié physique. Reconstituer ce patch dans un checkout distinct si l’on reprend l’essai ; ne pas le réactiver silencieusement dans main.

```sh
cargo fmt --check
cargo clippy --locked --release --all-targets --features mlx,chat -- -D warnings
cargo build --locked --release --features mlx,chat
cargo test --locked --release --features mlx,chat
KANI_HOME=../work/kani cargo kani --lib --no-default-features --harness grouped_mb3_preserves_rows_and_all_fallbacks
KANI_HOME=../work/kani cargo kani --lib --no-default-features

MLXL3_EXPERIMENTAL_GROUPED_MB3=1 MLXL3_MTP_TEST_MODEL="$SMALLM_MODEL" cargo test --locked --release --features mlx,chat qwen35::tests::mtp_prefill_verification_and_rollback_match_target -- --exact --ignored --nocapture --test-threads=1
MLXL3_EXPERIMENTAL_GROUPED_MB3=1 MLXL3_MTP_TEST_MODEL="$SMALLM_MODEL" MLXL3_MTP_TEST_HEAD="$SMALLM_HEAD" cargo test --locked --release --features mlx,chat qwen35::tests::mtp_recursive_sessions_match_greedy_and_exact_caches -- --exact --ignored --nocapture --test-threads=1
MLXL3_QWEN_TEST_MODEL="$SMALLM_MODEL" cargo test --locked --release --features mlx,chat qwen35::tests::checkpoint_grouped_mb3_verify_paired -- --exact --ignored --nocapture --test-threads=1
```

Kani0.68.0, CBMC6.11.0, nightly2026-08-21 aarch64 ; sélecteur7 entrées symboliques, aucune hypothèse, 164 obligations et6 covers. Logs `kani-grouped.log` / `kani-all.log` ; cette preuve CPU concerne le patch retiré. Le benchmark natif alterne l’override du test, quel que soit le flag environnement ; quatre warmups, huit paires, snapshot69tokens, huit blocs verify M3/24IDs forcés par passage. `mb3-verify-paired.log` : première fenêtre invalide pour performance à cause des tests auxiliaires concurrents ; `mb3-verify-isolated.log` et résumé JSON : répétition seule après terminaison de ces tests, sans changement source.

## Débit bridge A

```sh
"$PYTHON" benchmarks/benchmark_smallm_bridge.py --engine ../work/smallm-mb3-prototype-rs --model "$SMALLM_MODEL" --head "$SMALLM_HEAD" --tokens 256 --output "$SMALLM_OUT/mb3-bridge-256.json"
```

Binaire SHA256 `232f8c3026b2eccddbb7ea721b6e4d27f27533a574e2b935c6b62f33d9f2317f`, flag `MLXL3_EXPERIMENTAL_GROUPED_MB3=0/1`, ordre ABBA/BAAB, warmup64 puis mesure256, MTP2/température0/top_k1/repetition1/cacheOFF/context4096. Prompt exact dans le JSON : code complet de multiplication de matrices avec commentaires, exemples, tests et explication. Readiness120 s, génération180 s, shutdown5 s et wait10 s, enfants fermés/rejoints via le transport existant. Le cinquième passage expire ; `status:failed`, `parity:false` conservés. Quatre sorties complètes validées. Pas de comparaison globale de débit considérée concluante.

Ce driver n’inspecte pas la présence du flag dans un moteur arbitraire : **main final ne possède pas le flag MB3**. Utiliser le binaire archivé ou une reconstruction explicitement identifiée du patch, jamais présenter un A/B sans flag effectif comme validation d’une optimisation.

## Capture et contrôles finaux

Capture via MLX avec `MTL_CAPTURE_ENABLED=1`, seed27604, première forme qkv/z5120→10240+6144/K2, M3/dispatch stock. Trois warmups avec oracle M1/NT1 et synchronisation ; `mx.metal.start_capture(path)` puis un appel du kernel, `mx.eval`, `mx.synchronize`, `mx.metal.stop_capture`, vérification exacte et présence de l’artefact. `.gputrace`204 309 985octets conservé hors Git, métadonnées/log `metal-capture.{json,log}`. Pas d’analyse du nombre de dispatchs ni des compteurs : `xctrace` indisponible.

La compilation finale revient aux sources natives de la base et s’exécute après la fin de toutes les mesures GPU : fmt, Clippy release all-targets strict MLX/chat, build release et tests release. Logs `final-native-{fmt,clippy,build,tests}.log`. Ensuite, seule, la suite Python du workflow natif :

```sh
../work/venv/bin/ruff format --check benchmarks/benchmark_smallm*.py tests/test_smallm_benchmark.py
../work/venv/bin/ruff check benchmarks/benchmark_smallm*.py tests/test_smallm_benchmark.py
"$PYTHON" -m py_compile benchmarks/benchmark_smallm*.py tests/test_smallm_benchmark.py
"$PYTHON" -m pytest -q tests/test_bridge_benchmark.py tests/test_package_engine.py tests/test_qmm_tiles.py tests/test_compare_native.py tests/test_native_json_process.py tests/test_mtp_model_switch.py tests/test_smallm_benchmark.py tests/test_decode_microbenchmarks.py
"$PYTHON" -m crosshair check benchmarks/benchmark_smallm.py --per_condition_timeout 30
"$PYTHON" -m crosshair check benchmarks/benchmark_smallm_bridge.py --per_condition_timeout 30
"$PYTHON" -m crosshair check benchmarks/benchmark_smallm_tensor.py --per_condition_timeout 30
```

258 tests Python passés, 62 tests Rust passés/55 ignorés ; CrossHair absent, diagnostics `crosshair*.log`, **aucune preuve Python/Metal revendiquée**. Le workflow `.github/workflows/rust.yml` couvre les cinq nouveaux drivers et leurs régressions sans MLX sur runners hébergés. Aucun run de cette branche locale inspectable avant push, aucun statut GitHub inventé.

`campaign-identity.json` contient les hashes des16 fichiers modèle/tête, des sources finales, l’identité du moteur installé inchangé et la restauration de l’app. `prototype-provenance.json` distingue le patch/binaire expérimental retiré. Les données de référence historiques n’ont pas été écrasées.

## Répétition A-AC — close, conditions variables

Même commande bridge avec output `mb3-bridge-ac-256.json`, même binaire et options, préenregistrée le7octobre après observation de secteur76%. Huit passages complets, sorties exactes ; starts0..3 secteur,4..7 batterie, relevé final batterie78%. `mb3-bridge-ac-summary.json` applique les critères préenregistrés (contrôle max/min≤1,10 ; deux gains de fenêtre≥3% ; secteur partout). Tous les critères ne passent pas : nonconcluant, pas de promotion du runtime. Les512tokens de warmup et2048tokens mesurés n’ont pas changé les poids/génération/app. Aucune autre campagne en cours.

## Audit final des validateurs

Trois mutations (booléen en timing, hash nontexte, contexte vide) sont reproduites rouges avant durcissement (`bridge-validator-red.log`). La source utilisée pour les mesures est conservée en `bridge-campaign-source.py.txt`, et les12 sorties complètes passent le nouveau comparateur à partir du vrai texte sauvegardé, vérifié contre leur SHA256 original (`bridge-validator-raw-audit.json`). Timers et parcours GPU/génération identiques.

Les suites257cas avant correction du testQMM montrent255passes/2timeouts de fixtures300ms, même sans modèle ; logs `final-python-tests-validator{,-isolated}.log` et `final-python-tests-no-model.log`. Septcas passent seuls. Nouveau contre-exemple explicite : démarrage400ms puis JSON invalide, rouge avec300ms (`qmm-slow-start-red.log`). Budget de ces seules fixtures1s, watchdog10s, mêmes messages spécifiques et tousPID rejoints : huitcas verts (`qmm-slow-start-green.log`), puis258/258 (`final-python-tests-complete.log`). Format/lint/py_compile incluent aussi `tests/test_qmm_tiles.py`. CrossHairabsent, diagnostic enregistré ; aucune preuve d’ordonnanceur/subprocess. App rechargée après cette dernière suite, aucun processus d’essai restant.
