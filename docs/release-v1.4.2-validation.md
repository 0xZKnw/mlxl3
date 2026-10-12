# Desktop et moteur 1.4.2 — validation de livraison

Les changements de PR26 sont fusionnés dans main en
`1de60760081db9b0682abee6b554443a818b9bb2`. Son arbre est identique à la tête
PR testée `a20d45f8bad40d7c961164b4e8d1211bffe712c5`. Les métadonnées Desktop
1.4.2/build24, changelogs et règle `AGENTS.md` sont préparés en
`607d6412859d2c996008c6d859415203ee763e5d`. Aucun code exécutable, test,
kernel, dépendance ou script de packaging n'a changé entre ces sources ; seul
Info.plist et les documents évoluent. Le checkout personnel reste préservé.

## Source et CI

Les huit jobs PR/push de la tête a20d45f réussissent, avec inspection des étapes
et logs : [Native PR](https://github.com/0xZKnw/mlxl3/actions/runs/37664823400),
[Native push](https://github.com/0xZKnw/mlxl3/actions/runs/37664815532),
[Desktop PR](https://github.com/0xZKnw/mlxl3/actions/runs/37664823359),
[Desktop push](https://github.com/0xZKnw/mlxl3/actions/runs/37664816248).
Le checkout synthétique c1a8392, parents e26630e/a20d45f, a un arbre entier
identique à la tête. La CI du commit final de release et les identités de ses
artefacts sont suivies séparément dans les releases GitHub ; ce rapport ne leur
attribue pas rétroactivement les résultats d'une autre tête.

- Natifs Linux/macOS : 56 tests Rust par OS, 2 ignorés Linux/3 macOS ;
  293 tests Python et 1 saut ; format, Clippy strict, build et variante release
  `mlx,chat` avec MLX0.32.2 sur macOS réussis.
- Desktop : 296 tests Python/4 sauts, 49 cas composer, 114 assertions
  idle/récupération MTP et E2E complète dans les deux jobs.
- Kani0.68.0 : `cargo kani --lib --no-default-features`, 39/39 harnais par job,
  5463 SUCCESS, 98 SATISFIED, 72 UNREACHABLE internes, zéro échec/indéterminé.
  Les signatures inatteignables correspondent à4a7daa2 ; aucune assertion
  applicative n'est inaccessible. MTP≤3/unwind6, préfixes≤8/unwind34 et lookup
  history≤20/unwind33 sont des vérifications CPU bornées, avec les hypothèses
  et domaines des harnais conservés.

## Paquets locaux et parcours réels

Mac M5Air24GiB, macOS27.2, SDK26.5, MLX0.32.2 ; app1.4.2/build24,
moteur1.4.2/protocole1. Le candidat initial a volontairement conservé
`tracked_changes=true` et l'identité `1de60760081d-dirty` : il est **non
publiable**. Reconstruction propre depuis607d641 : les deux identités
annoncent le bon commit, build-info `tracked_changes=false`.

| Contrôle | Résultat |
| --- | --- |
| Packaging Python | 20 tests passés, aucun saut ; versions/erreurs, liens, tailles, reproductibilité, hashes et changement concurrent couverts. |
| Build DMG | Release Rust `mlx,chat`, release Swift, signatures ad-hoc/arm64 et DMG réussis. |
| E2E avec archive1.4.2 | 296 Python/4 sauts ; 49 cas composer, 114 assertions idle/MTP, vrai updater signé : installation, relocation, exécution, rejet et fallback. |
| App release | Commandes réelles `--check-chat-timeline` et `--check-mcp-preferences` réussies. |
| Archive moteur | Cinq fichiers réguliers attendus, manifest/version/protocole, tailles et hashes complets ; les quatre fichiers runtime sont identiques à ceux du bundle. Desktop minimum1.4.0. |
| DMG en lecture seule | `hdiutil verify`, codesign strict, montage sans ouverture UI, 14 fichiers du bundle byte exact ; volume détaché et données temporaires nettoyées. |
| Bridge physique signé | 43 requêtes : 35 completions non vides/finies, 5 annulations, 2 erreurs attendues, 1 Tune ; parité IDs/historique/budgets MTP1/2/3, préfixe, reprise et tuner vérifiés. |

Commandes principales, depuis le worktree isolé :

```sh
MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  MLXL3_MLX_ROOT=/absolute/path/to/mlx \
  MLXL3_BUILD_PYTHON="$PWD/.venv/bin/python" scripts/build-macos-dmg.sh

MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  MLXL3_TEST_PYTHON="$PWD/.venv/bin/python" MLXL3_SKIP_METAL_CHECK=1 \
  MLXL3_TEST_ENGINE_ARCHIVE="$PWD/dist/MLXL3-Engine-v1.4.2-arm64.tar.gz" \
  MLXL3_TEST_ENGINE_VERSION=1.4.2 scripts/check-e2e.sh

MLXL3_MTP_LOOKUP=0 .venv/bin/python scripts/check-mtp-depths.py \
  'dist/MLXL3 Desktop.app/Contents/Resources/runtime/mlxl3' \
  models/Qwen3.6-35B-A3B-EXL3-2.49bpw build/engine-v1.4.2/frspec-head --tune
```

Le bridge utilise un registre jetable et un délai global180s avec nettoyage
du groupe de processus sur erreur. MTP2 compact/Qwen3.6 EXL3 2.49bpw,
profondeurs0..3, budgets1/2/3/4/17/64 et contexte4096 ; aucun autre modèle,
compilation ou proveur local simultané. Alimentation relevée sur batterie98%
pour ce contrôle de correction. Ses57,837s sont un diagnostic, **pas un débit**
ni une campagne comparant des performances. Pas d'installation personnelle,
changement de registre utilisateur ni téléchargement de poids.

## Limites et preuves conservées

La preuve CPU bornée ne couvre pas MLX, Metal, FFI, Swift, concurrence ou
génération entière. Pas de vérificateur Swift source exploitable ; l'essai
CrossHair du correctif Python reste non exécuté (module absent). Le contrôle
physique Metal de la suite Desktop est explicitement sauté ; le bridge physique
ci-dessus est distinct. Aucun gain mémoire processus, déchargement physique
après veille ou nouveau tok/s n'est mesuré ici.

Diagnostics non fatals conservés : chemins linker CLT absents, dépréciations
CoreText du SwiftMath vendored/hdiutil, et PDF hostile de test. Le premier audit
final d'archives utilisait un filtre de noms incomplet ; les neuf archives ont
ensuite toutes passé hashes/manifests/blobs committés et relecture gzip.

[11 preuves brutes](measurements/release-v1.4.2/validation-raw.tar.gz) et
[manifeste vérifié](measurements/release-v1.4.2/validation-manifest.json)
conservent candidats, identités, checks et CI a20d45f. Pas de poids ou binaire
dans cette archive. Les paquets finaux sont reconstruits depuis le commit de
release propre ; leurs contrôles et SHA-256 exacts sont publiés avec
[Desktop1.4.2](https://github.com/0xZKnw/mlxl3/releases/tag/v1.4.2) et
[engine1.4.2](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.2).
Les anciens résultats négatifs et les neuf archives d'optimisation restent
intacts ; aucun prototype retiré n'est réintégré.
