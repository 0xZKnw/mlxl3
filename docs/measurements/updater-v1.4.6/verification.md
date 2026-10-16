# Vérification de l’updater 1.4.6

État final : correctif updater validé, PR41 fusionnée, Desktop/moteur1.4.6 publiés.
La requalification numérique additionnelle du nouveau binaire reste **non validée**.
[Manifeste complet](MLXL3-v1.4.6-validation.json) et
[preuves brutes vérifiées](verification-evidence.tar.gz), 83 entrées,
548203 octets/SHA-256 `668457a6235389b5b140618ee7d8efee0f3c6cbc0b9b3e4d2501efd61c132983`.
Aucun remplacement de l’app utilisateur.

## Défaut et domaine

Le job Desktop de la source `62b16b751a3173e2abf9e3cf2bf29b7debad9173`,
[run38907063506](https://github.com/0xZKnw/mlxl3/actions/runs/38907063506),
refuse le chemin placé après `-verify_arch arm64` avec LLVM lipo/Xcode26.6.
CLT26.5 accepte cet ordre, expliquant le succès local de la 1.4.5. Les quatre
appels de production sont corrigés en plaçant le fichier avant la commande.

La portée comprend préflight app, installateur moteur, script de copie app et
packager. Les quinze gardes de versions, compatibilité, signature et intégrité
existantes restent actives. Les doubles signés exercent les vraies fonctions et
le vrai script dans un dossier temporaire, avec une identité d’app distincte et
un exécutable sans effet. Ce n’est pas une validation de MLX/Metal par les doubles.

## Contrôles locaux initiaux

- Compilation ciblée `swiftc -swift-version 6 -parse-as-library -warnings-as-errors`
  de `EngineRuntimeStore.swift`, `Localization.swift` et concaténation de
  `UpdateManager.swift` avec `tests/updater-application-check.swift` : succès.
- Exécution du checker : **15 cas signés**, installation/activation du TAR Mach-O
  factice, vrai script de remplacement/lancement/sauvegarde app factice : succès.
- `python -m pytest -q tests/test_package_engine.py` : **45 passés**, 1,05s.
- `/bin/zsh -n scripts/build-macos-app.sh scripts/check-desktop.sh` : succès.
- Trois workflows chargés via PyYAML BaseLoader, événements PR et push présents : succès.
- `git diff --check` : succès. Aucune dépendance ajoutée.

## Résultats finaux

- Source construite/testée : `be030721c982942b7ed179ac57d9254941f25cbb` ;
  [PR41](https://github.com/0xZKnw/mlxl3/pull/41) fusionnée
  `d09f51f705c7481130403ae5b5cc35a0baa47ada`, arbre Git identique.
- `MLXL3_BUILD_PYTHON=…/.venv/bin/python MLXL3_MLX_ROOT=…/site-packages/mlx
  MLXL3_MACOS_SDK=…/MacOSX26.5.sdk SDKROOT=…/MacOSX26.5.sdk
  caffeinate -i /bin/zsh scripts/build-macos-dmg.sh` : succès, **116,47s**,
  release `mlx,chat`/MLX0.32.2. Chemins complets et environnement dans les preuves.
- **41 contrôles** bundle/signatures/architecture/manifeste/SHA/DMG/runtime déplacé
  passent ; footer absent du binaire. Volume détaché. Même DMG accepté par les
  vraies sources préflight1.4.3 avec CLT26.5 en **3,25s** ; aucun installateur
  utilisateur lancé par ce contrôle.
- `MLXL3_TEST_PYTHON=…/.venv/bin/python MLXL3_SKIP_METAL_CHECK=1
  MLXL3_TEST_APP_DMG=…/MLXL3-Desktop-v1.4.6-b28-Apple-Silicon.dmg
  MLXL3_TEST_APP_VERSION=1.4.6
  MLXL3_TEST_ENGINE_ARCHIVE=…/MLXL3-Engine-v1.4.6-arm64.tar.gz
  MLXL3_TEST_ENGINE_VERSION=1.4.6 caffeinate -i /bin/zsh scripts/check-desktop.sh` :
  **succès/exit0,379,21s**, chemins SDK/home jetable dans `desktop.json` archivé.
  Sampling821assertions/180états, mémoire240oracles, lifecycle114assertions,
  LaTeX9formules×2styles/63préparations, imports150frontières/256encodages,
  streaming1194+4845fragments, CLI/MCP et tous les contrôles updater sélectionnés.
- [CI Desktop du commit exact](https://github.com/0xZKnw/mlxl3/actions/runs/38910111789) :
  **success**, dont15cas et deux frontières d’installation sous LLVM lipo ;
  Python **772 passés/115 sautés** (dépendances facultatives/Metal physique exclus).
- [CI Rust](https://github.com/0xZKnw/mlxl3/actions/runs/38910111797) : Linux/macOS
  fonctionnels **success**, **118CPU/3ignorés**, plus **7oracles CPU featureMLX**
  sur macOS. Kani usageAPI ciblé réussi ; suite Kani complète encore en cours au
  relevé, aucune réussite globale proclamée.
- [CI conversion/packing](https://github.com/0xZKnw/mlxl3/actions/runs/38910111662) :
  **success**. Publication réellement bloquée jusqu’au succès Desktop exact-head.
- Les deux tags publics pointent be030721. DMG **73400205octets**, TAR **66957364**,
  manifeste moteur **13587** ; téléchargements publics sans authentification
  comparés aux SHA-256 du paquet local, tous identiques. Dernière stable
  `v1.4.6`, canal moteur `latest=false`. Empreintes complètes dans le manifeste
  et les [changelogs Desktop](../../release-v1.4.6.md)/[moteur](../../release-engine-v1.4.6.md).

## Limites et contrôles numériques interrompus

Pas de vérificateur source Swift/AppKit adapté disponible : compilation stricte
et tests d’intégration, aucune preuve formelle proclamée. Le défaut LLVM n’était
pas reproductible avec le seul lipo CLT local ; le job hébergé est cette fois
inspecté et réussi avant publication. Rust ne change que de numéro de package :
**60sources natives** et les trois bibliothèques MLX/metallib sont byte-identiques
au moteur1.4.5 publié. Les limites Kani historiques restent inchangées.

Le contrôle numérique additionnel du nouveau runtime signé9cb167af utilise
`forward` sur la V2 inchangée, positions101,174,…,612 (8pas), puis batch65
avec opt-in0/1, logitsF16/50états et oracles historiques indépendants. Validation
Metal activée, limite120s/processus, arrêt/join sur erreur, aucun benchmark.
La première tentative **timeout120,35s**, sortie partielle conservée,
processus tué−9/rejoint, rapportfailed/parityfalse. Une campagne GPU externe
exécutait `dense_smallm_dispatch_matches_serial_prefixes --ignored` en même temps ;
elle n’a pas été arrêtée. La garde initiale ne détectait pas les binaires Rust tests.
Reprise préenregistrée avec garde étendue : **bloquée avant chargement**, zéro
processus modèle lancé, aucun test qualifiant. Les deux échecs sont archivés.
Pas de nouvelle parité, baisse KL/PPL ou vitesse déduite ; les validations
numériques historiques sont distinguées du nouveau binaire non requalifié.

L’app personnelle1.4.5/b27 garde les mêmes empreintes Info/exécutable avant/après.
Un ancien client déjà affecté par LLVM peut nécessiter une installation manuelle
du DMG ; le préflight1.4.3 n’a été validé ici que sur CLT26.5.

Le footer MTPLX est supprimé des réglages à la demande utilisateur, sans supprimer
les licences. Compilation Desktop et contrôle du binaire livré réussis ; pas de
nouveau test dédié à cette suppression de texte statique.
