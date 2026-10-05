# Livraison du moteur 1.3.1 — 5 octobre 2026

Source du paquet : `69fa9283ff0dba153f897f67d21193e618a2f3ef`, après la
fusion `f43118519d6af6cb15496e23520d4f61bd47e913` de la
[PR23](https://github.com/0xZKnw/mlxl3/pull/23). Depuis cette fusion, seuls
Cargo/lock, les notes et le journal de livraison ont changé. Le binaire a été
compilé sur cette source propre ; ses métadonnées confirment la révision exacte.
Le checkout principal et ses travaux MTP locaux ont été préservés.

## Paquet et périmètre

- Archive : `MLXL3-Engine-v1.3.1-arm64.tar.gz`, **66 851 669 octets**.
- SHA-256 : `4e60f147d67fb10d588954a8063ff03102444906d8651ee34f5e1e44dd771853`.
- Version1.3.1, protocole bridge1, MLX0.32.2, fonctions MLX/chat activées,
  arm64, macOS≥26.2, Desktop≥1.2.0. Le canal Desktop conserve1.3.0/build22.
- Les quatre fichiers du runtime ont une taille et une empreinte dans
  `engine.json`. Signatures ad hoc vérifiées strictement, sans notarisation.
- Inclut lecture Darwin bornée, libération du cache après chargement et
  BM64 M5 éligible, ainsi que les trois corrections d'outillage de PR23.
  Aucun nouveau modèle, algorithme, benchmark de vitesse ou gain annoncé.

## Commandes et résultats locaux

Les préenregistrements et mises à jour sont dans
`ENGINE-2026-10-05-1.3.1` du [journal](../opti.md). Les logs et métadonnées
sont conservés dans [les preuves](measurements/engine-v1.3.1/).

- `cargo fmt --all -- --check`, `git diff --check`,
  `cargo metadata --locked --no-deps --format-version 1` : réussis.
- Avec `MLXL3_MLX_ROOT` vers le paquet MLX0.32.2 et
  `MACOSX_DEPLOYMENT_TARGET=26.2`,
  `cargo build --release --locked --features mlx,chat` : réussi en38,63s.
  `runtime-info` :1.3.1, commit69fa9283ff0d sans suffixe dirty, profilrelease,
  MLX/chat/protocole1 corrects. SDK natif27.0/minOS26.2 ; dylibsSDK26.5.
- `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`
  et `cargo test --release --locked --features mlx,chat` : réussis,
  **59 tests passés/53 ignorés**, aucun échec. Avertissement futur préexistant
  `block0.1.6`, sans refus de Clippy.
- `codesign --force --sign -`, `codesign --verify --strict`,
  `lipo -verify_arch arm64`, `xcrun vtool -show-build` : réussis pour
  `mlxl3`, `libmlx.dylib`, `libjaccl.dylib`, chacun minOS26.2.
- `python scripts/package_engine.py build/engine-v1.3.1/runtime dist --version 1.3.1`
  et `python -m pytest -q tests/test_package_engine.py` : réussis,
  **15 tests** de déterminisme/empreintes, versions/fichiers invalides,
  liens, limites et modification concurrente.
- `swiftc -swift-version 6 -warnings-as-errors -parse-as-library -sdk SDK26.5`
  sur `UpdateManager.swift`, `EngineRuntimeStore.swift`, `Localization.swift`
  de production et le harnais existant adapté à1.3.1 : réussi.
  `updater-check ARCHIVE_FIXTURES ARCHIVE` : installation signée, activation,
  identité du runtime, `list --json` sur registre jetable, compatibilité
  Desktop1.2.0 et1.3.0, déplacement du store, exécution, rejet/repli,
  archives hostiles, canaux et limites des helpers réussis.
  Le premier essai échouait sur ma fixture `{}` sans version ; schéma corrigé
  à `{"version":1,"models":{}}`, puis relance réussie. Les sources/traces
  négatives sont conservées et aucune source de production n'a été modifiée.
- `python native/check_qmm_tiles.py BM32_WRAPPER BM64_WRAPPER --output qmm-release.json --request-timeout 120`
  avec le même binaire signé packagé et `MLXL3_DENSE_PREFILL_M64=0/1` :
  **64 cas, 1 887 232 mots FP16 présents, finis et bit-identiques**.
  K1..8, trois codebooks, projections simples/groupées et dimensions de repli
  sont exercés. Nombre attendu calculé indépendamment et forme de chaque cas
  vérifiée. 14,285s de contrôle fonctionnel, pas un benchmark ; aucun modèle
  chargé, compilation/prover local terminé avant exécution.

## CI et publication

Les workflows PR/push existants ont été inspectés, permissions `contents:read`,
sans credential de production. Sur le commit source exact69fa9283ff0d :
[Native Rust/Kani](https://github.com/0xZKnw/mlxl3/actions/runs/37362724535)
et [Desktop/Python/E2E](https://github.com/0xZKnw/mlxl3/actions/runs/37362724561).
Les deux jobs natifs sont réussis. Desktop/E2E est désormais réussi :
**103 tests Python passés/4 skips**, hardening, updater, imports, bridge,
streaming/rendu, transport CLI, téléchargements, tuner et MCP réussis.
Les [statuts](measurements/engine-v1.3.1/ci-desktop.json) et
[logs](measurements/engine-v1.3.1/ci-desktop.log) ont été inspectés.
Kani est terminé et réussi : **31/31 harnais, 4 536 obligations, 0 échec,
71/71 covers**, 70 checks inatteignables déclarés et conservés. La suite
`cargo kani --lib --no-default-features` vérifie les contrats Rust CPU
symboliques avec les bornes/unwinding existants, sans changer les domaines.
La garde BM64 couvre tous i32/bool symboliques sans `assume`, 61 obligations
et2 covers ; les autres contrats et bornes sont détaillés dans la revue et
les harnais. Cette exécution ne prouve pas le GPU/MLX/FFI.

**4/4 jobs réussis** sur69fa928 avant publication. Linux/macOS : fmt,
Clippy/build, chacun49tests Rust (2 ignorés Linux/3 macOS), chacun101tests
Python d'outillage/packaging, Ruff et compilation des scripts réussis.
Les [statuts](measurements/engine-v1.3.1/ci-native.json),
[logs complets](measurements/engine-v1.3.1/ci-native.log) et
[synthèse des obligations](measurements/engine-v1.3.1/verification-summary.json)
ont été inspectés. Aucun code exécutable n'a changé après ces contrôles.

L'archive est téléversée dans une release **brouillon** `engine-v1.3.1` visant
la source exacte. État `uploaded`, taille et digest GitHub identiques au paquet
local vérifié ; [preuve](measurements/engine-v1.3.1/draft-release.json).
À ce premier relevé, publication/téléchargement public et sélection des canaux
étaient encore à vérifier. Clôture : [moteur1.3.1 publié](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.3.1)
à21:34:16Europe/Paris (19:34:16UTC), après les4jobs CI réussis. Tag exact69fa928,
`draft=false`/`prerelease=false`, moteur non-latest ; Desktop1.3.0 reste latest.
Le téléchargement HTTPS public sans authentification est réussi, HTTP200,
taille/SHA-256 identiques au paquet local installé pendant le contrôle.
L'updater de production sélectionne moteur1.3.1/asset/digest exact et
Desktop1.3.0/build22 sur l'inventaire public sans authentification.
Preuves : [publication](measurements/engine-v1.3.1/publication.json),
[téléchargement](measurements/engine-v1.3.1/download-proof.json),
[sélection](measurements/engine-v1.3.1/updater-selection.json),
[réponse publique](measurements/engine-v1.3.1/published-releases.json).
La release est publiée et proposée par l'updater ; aucun essai local en cours.

## Portée et limites

Cette livraison répète les contrôles nécessaires pour le paquet release signé,
après les contrôles debug de la revue. Les tests numériques finis et la
vérification Kani CPU bornée ne prouvent ni Metal/MLX/FFI ni la génération
complète. Les 53 tests ignorés ne sont pas exécutés dans la suite standard ;
la parité du checkpoint Qwen complet fourni par l'auteur n'est pas rejouée.
CrossHair antérieur reste inconclusif. Débit, gain, température et RAM nouveaux
non mesurés ; les limites de performance de la revue restent applicables.

Installation de vérification dans des dossiers jetables supprimés ensuite.
L'app de l'utilisateur n'est pas remplacée. Après mise à jour moteur dans
l'app, redémarrer Desktop pour charger le nouveau runtime ; le défaut antérieur
d'activation à chaud reste différé. Les compilations/tests locaux sont terminés.
