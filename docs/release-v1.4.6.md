# MLXL3 Desktop 1.4.6 — build 28

Cette version corrige la vérification des mises à jour avec le `lipo` LLVM des
Xcode récents. Elle retire aussi la mention « Powered by MTPLX » des réglages :
l’application utilise le moteur MLXL3. Les licences et attributions restent conservées.

## Correctifs

Les vérifications arm64 placent désormais le fichier avant `-verify_arch`.
L’ancien ordre fonctionnait avec les Command Line Tools du Mac de validation,
mais échouait avec LLVM lipo de Xcode 26.6 en CI. Les trois parcours concernés
sont le préflight du DMG, l’installation du moteur et le script de remplacement
de l’app ; le packager utilise la même syntaxe portable. Signature, empreintes,
compatibilité et sauvegarde de l’ancienne app restent vérifiées.

Le correctif de versions indépendantes de la 1.4.5 est conservé. Desktop et moteur
portent tous deux le numéro 1.4.6 pour permettre la migration des anciens clients
qui exigent encore leur égalité. La 1.4.4 reste une préversion retirée du canal stable.

## Performances et qualité

Les corrections API, LFM et LaTeX des PR37–39 sont conservées. Aucun poids, calcul
numérique, réglage de quantification ou défaut d’inférence n’est modifié par cette
version. Aucun nouveau gain de vitesse ou de KL/PPL n’est revendiqué. Le préfill
MoE expérimental reste OFF par défaut.

## Installation et compatibilité

**MLXL3-Desktop-v1.4.6-b28-Apple-Silicon.dmg**, moteur **1.4.6**, MLX **0.32.2**,
Apple Silicon/macOS **26.2+**. Signature ad-hoc, sans notarisation Apple.
Le canal moteur reste publié avec `latest=false`.

Un ancien client utilisant déjà LLVM lipo peut échouer avant d’installer le
correctif ; installer alors le DMG manuellement. Un binaire installé ne peut pas
être réparé par la seule publication d’une nouvelle release. Les modèles et
données sont conservés ; nos contrôles ne remplacent jamais l’app utilisateur.

## Vérification et limites

Le défaut LLVM est reproduit dans le job Desktop du commit 62b16b75 (run
38907063506). Les 15 cas du préflight signé restent obligatoires. Deux contrôles
exercent aussi les vraies installations dans des dossiers jetables : activation
d’un moteur Mach-O signé factice, et copie/lancement d’une app sans effet avec
sauvegarde de l’ancienne. Ces doubles ne valident ni MLX ni Metal.

La publication stable exige le succès du job Regression checks du commit source
exact. La suite Desktop, les vrais DMG/TAR, leurs signatures, identités et
empreintes sont contrôlés séparément. Résultats, commandes, SHA source, tailles
et SHA-256 figurent dans **MLXL3-v1.4.6-validation.json** joint à la release.
Swift/AppKit n’est pas prouvé formellement ; aucun remplacement de l’app réelle
n’est exécuté par les tests. Les limites historiques Kani/Metal restent celles
des [notes moteur 1.4.5](release-engine-v1.4.5.md).

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), signalements, correctifs et validation
  du parcours de mise à jour, après la [PR40](https://github.com/0xZKnw/mlxl3/pull/40).
- [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique ;
  [MLX](https://github.com/ml-explore/mlx), exécution Apple Silicon. Les crédits
  amont du moteur sont conservés dans ses notes et les licences du dépôt.
