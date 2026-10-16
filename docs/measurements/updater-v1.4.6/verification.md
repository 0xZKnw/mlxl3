# Vérification de l’updater 1.4.6

État initial : contrôles ciblés locaux réussis ; paquet final et CI en attente.
Le manifeste joint à la release conservera les résultats finaux, leurs sources
et les empreintes des assets. Aucun remplacement de l’app utilisateur.

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

## Limites et suite prévue

Pas de vérificateur source Swift/AppKit adapté disponible : compilation stricte
et tests d’intégration, aucune preuve formelle proclamée. Le défaut LLVM n’est
pas reproductible avec le seul lipo CLT local ; le job Desktop du commit exact
doit réussir avant publication stable. Rust ne change que de numéro de package ;
les sources numériques et les limites Kani historiques restent inchangées.

Le footer MTPLX est supprimé des réglages à la demande utilisateur, sans supprimer
les licences. Compilation Desktop et contrôle du binaire livré prévus ; pas de
nouveau test dédié à cette suppression de texte statique.
