# Bêta 2 — QA delta du paquet figé

**PASS**, source `0842599c24461b3395b4d2d2bd9bf617eb739798`, Desktop **2.0.0/build 32**. `build-info.json` porte ce SHA et `tracked_changes:false` ; le vrai runtime répond exactement `commit:0842599c2446`, sans marqueur dirty. L’échec 93d654cc reste conservé séparément et ne devient pas une validation rétroactive.

- Les quatre tailles/SHA-256 runtime correspondent au manifeste moteur 2.0.0 et aux cinq entrées régulières exactes de l’archive. Trois ressources MLX/JACCL/metallib sont identiques à la première QA ; seul le CLI a changé pour porter la nouvelle identité de build. Le diff des sources natives 673f779f→0842599c est vide.
- Trois probes CPU du runtime extrait/relocalisé : runtime-info, --version et list avec registre jetable vide. Toutes passent ; MLX/chat actifs, bridge 1, architecture aarch64, profil release. Aucun modèle ou cache exécuté.
- Signature app `codesign --verify --deep --strict` valide avant montage et depuis le DMG. Architecture arm64 vérifiée pour l’app et le nouveau CLI. Signature ad-hoc, sans notarisation revendiquée.
- DMG monté readonly/nobrowse : inventaire complet du bundle byte à byte identique (14 fichiers réguliers), puis démontage réussi. Onze commandes terminées et rejointes.

## Empreintes finales

- `MLXL3-Desktop-v2.0.0-b32-Apple-Silicon.dmg` : **74117391 octets**, SHA-256 `83999f5e83d051ab55503d604e8804617d5a0eae84b2116b2572a545f140096d`.
- `MLXL3-Engine-v2.0.0-arm64.tar.gz` : **67254401 octets**, SHA-256 `8922476474341ffd3db73f586067b29ea04f56bb81e216fe6c4e96e25b2d6b9a`.

## Limites conservées

Le RPATH absolu hérité demeure après @executable_path et égale celui de la première QA. Aucune nouvelle trace dyld n’est exécutée ; la trace historique de résolution locale reste datée et associée à son premier paquet. Aucun GPU, génération, suite native, nouveau téléchargement/cache, installation, modification source ou Git. L’interface, les quatre cartes et le texte SSD ont été contrôlés séparément par le parent. La CI, l’upload et l’installation restent distincts de cette QA delta.

Preuves : `package-qa-frozen.json` et `package-qa-frozen-logs/`. Échec précédent : `package-qa-final-runtime-identity-rejection.json` et `package-qa-final.json`. Les noms numériques de M/dist seront renommés en assets bêta par le parent en conservant ces octets/SHA.
