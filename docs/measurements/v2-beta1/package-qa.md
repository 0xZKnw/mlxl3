# V2 bêta 1 — contrôle des paquets et relocalisation CPU

**Validé sur le paquet final e04c6558959609443dda362f2e8509b998adaa6d**, version interne 2.0.0, Desktop build31. Les noms externes portent bêta.1. Aucun modèle ni inférence chargé par ce contrôle ; aucun build, installation système, remplacement d’app, changement de signature ou de quarantaine.

- `codesign --verify --deep --strict` passe sur `dist/MLXL3 Desktop.app` ; les trois Mach-O extraits (`mlxl3`, `libmlx.dylib`, `libjaccl.dylib`) passent aussi `codesign --verify --strict`.
- `hdiutil verify` passe sur le DMG bêta final. Les tailles et SHA-256 des trois artefacts, ainsi que leurs fichiers `.sha256`, correspondent à artifacts.json.
- Info.plist : identifiant `io.mlxl3.desktop`, version2.0.0, build31, minimum macOS26.2. build-info.json indique le SHA e04c6558 complet et `tracked_changes:false`, SDK26.5.
- L’archive contient exactement cinq fichiers réguliers : engine.json et les quatre fichiers runtime. Aucun lien, traversée, nom dupliqué ou extra ; tailles bornées, uid/gid/mtime nuls, modes sans bits spéciaux. Extraction explicite vers `relocated-runtime`, aucun extractall.
- Les trois manifestes (JSON publié, archive extraite, app embarquée) sont byte à byte identiques. Les quatre fichiers runtime ont les tailles/SHA attendus et égalent les fichiers embarqués.
- Architecture unique arm64 pour l’exécutable Desktop et les trois Mach-O runtime. `otool -L` montre des dépendances système ou relatives ; pas de dépendance LC_LOAD_DYLIB absolue vers un environnement de développement.
- Manifestes : version2.0.0, schéma1, bridge1, architecturearm64, minimum macOS26.2.0 et app1.4.0. Les notices MTPLX et Apache2.0 attendues sont présentes et non vides ; cela ne signifie pas que le registre complet THIRD_PARTY_NOTICES est embarqué.

## Exécution du moteur relocalisé

Les commandes terminent avec code0, sans `MLXL3_MLX_ROOT` et sans aucune variable `DYLD_*`, avec `MLXL3_HOME` et registre v1 vide dans un dossier jetable ROOT :

```text
relocated-runtime/mlxl3 --version
relocated-runtime/mlxl3 --registry package-qa-state/models.json runtime-info
relocated-runtime/mlxl3 --registry package-qa-state/models.json list --json
```

Résultats : `mlxl3-native 2.0.0` ; runtime-info version2.0.0 / architectureaarch64 / bridge1 / mlx_enabledtrue / chat_enabledtrue / profilrelease / commite04c65589596 ; catalogue exactement `[]`.
Aucun modèle utilisateur enregistré, historique ou store moteur utilisateur n’a été modifié. Le snapshot préalable de noms de processus n’a trouvé ni moteur natif ni application MLXL3 active ; ce snapshot n’est pas une preuve d’absence de toute activité GPU du système.

## Chemin RPATH restant et vérification de résolution réelle

Le premier contrôle exigeant que **tous** les LC_RPATH soient portables s’est arrêté sur `mlxl3` : `@executable_path`, puis `/Users/justin/Documents/localqwen3.8flash/.venv/lib/python3.12/site-packages/mlx/lib`. Cet arrêt est conservé dans package-qa-v1-rpath-rejection.json ; aucun binaire modifié et aucune signature refaite.
Ce chemin absolu est un fallback après le chemin adjacent. Le critère de livraison est la résolution réelle des bibliothèques empaquetées, et ne suppose pas qu’aucun fallback absolu ne figure dans Mach-O.
Après autorisation explicite du parent, une commande runtime-info supplémentaire utilise **uniquement** `DYLD_PRINT_LIBRARIES=1` (aucun autre DYLD, aucun MLX_ROOT). La trace confirme les deux chargements :

```text
.../build/v2-beta1/relocated-runtime/libmlx.dylib
.../build/v2-beta1/relocated-runtime/libjaccl.dylib
```

Aucun chargement de ces deux bibliothèques depuis l’environnement de développement observé. Les trois probes normaux précédents restent exécutés sans aucun DYLD. **La relocalisation est validée sur ce parcours CPU, avec le fallback absolu présent et inutilisé dans la trace** ; ne pas publier « aucun RPATH absolu ».

## Preuves et limites

31 commandes terminées et jointes ; résultats/argv/durées et fichiers de logs dans [package-qa.json](package-qa.json), package-qa-logs et package-qa-logs-v2. L’état avant le diagnostic dyld est conservé dans package-qa-before-dyld-resolution.json. Le snapshot brut `process-preflight.stdout` contient les noms d’autres applications et doit rester local ; le rapport expose seulement le résultat filtré vide.
Les paquets d’origine n’ont pas changé ; aucun `install_name_tool`, codesign force, bypass de quarantaine, masquage du MLX original ou installation système. Le dossier relocalisé et le registre de test sont jetables et conservés pour audit.
Ces probes ne valident pas un kernel, la génération, l’annulation, l’UI, la notarisation Apple ou l’updater réel du paquet. La signature reste ad-hoc. Les contrôles physiques/bridge et stores jetables du parent restent des preuves distinctes ; aucune CI ni performance globale n’est déduite de ce rapport.
