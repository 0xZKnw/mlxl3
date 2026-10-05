# Livraison du moteur 1.2.1 — 5 octobre 2026

[PR #20](https://github.com/0xZKnw/mlxl3/pull/20) fusionnée à la demande de l'utilisateur avant la fin de la CI. [Release publiée](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.2.1) à 07:00:09 UTC. Le tag pointe sur le commit source `954cfe37167f9f877b04dc840394d303106be687`, inclus dans le merge `1f9ba848d6ea58478d9aa490e38404d01034706f`. Aucun changement exécutable supplémentaire après les tests MTP-03 ; seuls le numéro Cargo et les documents de livraison ont été ajoutés.

## Paquet livré

- `MLXL3-Engine-v1.2.1-arm64.tar.gz` : **66 827 147 octets**.
- SHA-256 : `ffa0300d89c03562c56b3a05b4eca8c1fc29d245b20bf3a2169b7f611f147902`.
- Le digest GitHub et celui du téléchargement distant sont identiques au paquet installé pendant le contrôle local.
- Manifeste : version1.2.1, bridge1, arm64, macOS≥26.2, Desktop≥1.2.0, quatre fichiers avec taille et empreinte. Aucun poids de modèle.
- Le code réel de sélection de l'app reconnaît moteur1.2.1 et Desktop1.2.0 dans la réponse GitHub publique. La release engine est publiée avec `latest=false` ; `/releases/latest` reste `v1.2.0`.
- Aucun moteur installé dans l'app personnelle pendant cette tâche. Installation de vérification dans un dossier temporaire supprimé ensuite.

## Commandes et résultats

`PATH=/Users/justin/.cargo/bin:$PATH MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx MACOSX_DEPLOYMENT_TARGET=26.2 cargo build --release --locked --features mlx,chat` : réussi en32,18 s. `runtime-info` confirme version1.2.1, commit954cfe37167f, profilrelease, MLX/chat activés et protocole1. `cargo fmt --all -- --check` réussi sur la version préparée.

`codesign --force --sign -`, puis `codesign --verify --strict`, `lipo -verify_arch arm64` et `vtool -show-build` sur `mlxl3`, `libmlx.dylib` et `libjaccl.dylib` : réussis ; minOS26.2 dans chaque Mach-O. Bibliothèques MLX0.32.2 reprises du paquet1.2.0.

`.venv/bin/python scripts/package_engine.py build/engine-v1.2.1/runtime dist --version 1.2.1` : réussi. `.venv/bin/python -m pytest -q tests/test_package_engine.py` : **15 tests réussis**, couvrant déterminisme/empreintes, versions invalides, fichiers absents/vides/liens/trop volumineux et modification concurrente du runtime.

`tests/updater-check.swift` copié dans `build/engine-v1.2.1/` et adapté uniquement pour la release optionnelle1.2.1, avec un mode supplémentaire de sélection des releases publiques. Compilation Swift6 avec SDK26.5 contre les sources de production inchangées `UpdateManager.swift`, `EngineRuntimeStore.swift` et `Localization.swift`. Commande `build/engine-v1.2.1/updater-check build/engine-v1.2.1/archives dist/MLXL3-Engine-v1.2.1-arm64.tar.gz` : réussie. Tests de canaux, versions/URLs/hashes, archives hostiles, activation/repli, copie d'app échouée, limites du helper ; archive signée installée, déplacée, exécutée (`runtime-info`, `list --json`), puis rejetée avec repli dans un store jetable, appVersion1.2.0. Aucun modèle chargé.

`build/engine-v1.2.1/updater-check --select build/engine-v1.2.1/published-releases.json` : sélection moteur1.2.1/Desktop1.2.0 réussie. `gh release download engine-v1.2.1 --pattern MLXL3-Engine-v1.2.1-arm64.tar.gz --dir build/engine-v1.2.1/downloaded`, puis taille/SHA-256 : identiques au paquet local.

## Vérification du code et limites

Les tests MTP terminés avant livraison sont conservés dans [opti.md](../opti.md) et `docs/measurements/mtp-03-*` : Clippy strict avec/sans MLX, **54 tests Rust réussis**, **26 harnais Kani réussis**, trois contrôles physiques ciblés, référence MLX-LM et bridge réel. Le contrat nouveau utilise offset/rows i32 entièrement symboliques : refus des lignes vides/négatives et des offsets négatifs, exactitude de la somme et absence de dépassement ; 12 propriétés et4 couvertures dans le contrôle ciblé. Il s'agit d'une vérification bornée du contrat Rust, pas d'une preuve du GPU/FFI. Les 47 tests ignorés de la suite standard, autres GPU/architectures et branches non ciblées restent hors de la couverture exécutée.

La [CI Rust/Kani de la PR](https://github.com/0xZKnw/mlxl3/actions/runs/37274850532) est réussie. La [CI Desktop/Python de la PR](https://github.com/0xZKnw/mlxl3/actions/runs/37274850631) et les runs déclenchés par le merge restent **en cours au relevé après publication** ; aucune validation complète de ces runs n'est revendiquée. Aucun nouveau benchmark/inférence lancé. L'avertissement futur préexistant de `block0.1.6` persiste sans échec de compilation.

Les [empreintes et le relevé CI](measurements/engine-v1.2.1-proof.json), [log build](measurements/engine-v1.2.1-build.log), [log installation](measurements/engine-v1.2.1-install.log), [log sélection](measurements/engine-v1.2.1-selection.log) et [tests packaging](measurements/engine-v1.2.1-package-tests.log) sont conservés dans le dépôt. Les fichiers temporaires supplémentaires restent sous `build/engine-v1.2.1/`, hors Git.

## Activation dans l'app ouverte — à corriger plus tard

Retour utilisateur après installation : une fermeture/réouverture de l'app est nécessaire pour utiliser le nouveau moteur ; débit rapporté56tok/s avant relance puis62tok/s après. Observation déclarative, sans comparaison contrôlée ni attribution numérique supplémentaire à MTP-03. Cause non investiguée pendant la livraison ; correction différée explicitement à sa demande.

Comportement attendu pour une correction ultérieure : après installation du moteur, arrêter/recréer le subprocess avec le runtime sélectionné et recharger le modèle dans l'app ouverte. Vérifier la version réellement exécutée, les requêtes suivantes et la récupération en cas d'échec, sans redémarrage manuel. Ajouter un test d'intégration de cette transition dans l'app déjà ouverte : le test actuel valide l'installation/l'exécution du paquet en store jetable, mais pas l'activation dans une session GUI existante.
