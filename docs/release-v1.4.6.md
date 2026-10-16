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


## Artefacts

Source : `be030721c982942b7ed179ac57d9254941f25cbb`. Signature ad-hoc, sans notarisation.

| Fichier | Octets | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v1.4.6-b28-Apple-Silicon.dmg | 73400205 | `802916abc14e50affda4f18ec654e1b01cb7b9fd2db5bad5434a4533c4b287f1` |
| MLXL3-Engine-v1.4.6-arm64.tar.gz | 66957364 | `0047f7d935129cbfca78f48fca5fe6fdfdd11d975c6d6abe7636881aea2bfc44` |
| MLXL3-Engine-v1.4.6-arm64.json | 13587 | `5be214e495347e7c199513848e814e90b2ac7cb2f04d2aa4b6dfec5f026fdbd9` |

## Résultats du paquet final

41 contrôles paquet/DMG réussis ; suite Desktop complète réussie (379,21s), avec les vrais DMG et TAR1.4.6,15cas signés et les deux frontières d’installation factices. Le vrai préflight1.4.3 accepte le DMG sur CLT26.5. 45tests packaging passent. Les60sources natives et les bibliothèquesMLX/metallib sont identiques1.4.5 ; seul le numéro/identité du moteur change.

Contrôle numérique additionnel du nouveau binaire **non validé** : première tentative timeout120s pendant une campagne GPU externe, deuxième bloquée avant chargement par la garde de concurrence. Sorties partielles/échecs conservés ; pas de parité ni gain déduits. Aucun poids ni calcul numérique modifié, aucune nouvelle mesure KL/PPL. Swift non prouvé formellement ; limites Kani historiques conservées. Aucun remplacement de l’app utilisateur par nos outils.

CI du commit be030721 : [Regression/Desktop](https://github.com/0xZKnw/mlxl3/actions/runs/38910111789) **réussie**, dont15cas signés et les deux frontières installmoteur/copieapp ; **772Python passés/115sautés** (dépendances facultatives/Metal physique exclus). [Rust Linux/macOS](https://github.com/0xZKnw/mlxl3/actions/runs/38910111797) fonctionnel **réussi**,118tests CPU/3ignorés et7oracles CPU featureMLX sur macOS. Kani usageAPI ciblé réussi ; suiteKani complète encore en cours, pas de preuve globale. [Conversion/packing](https://github.com/0xZKnw/mlxl3/actions/runs/38910111662) **réussis**. [PR41](https://github.com/0xZKnw/mlxl3/pull/41) fusionnée.

Manifeste de validation : **MLXL3-v1.4.6-validation.json**, 38275octets, SHA-256 `6cf7f9e6f2622ee2edbf67acf521c3432c3693b2cc0c4979102ec67b5a67383d`. [Preuves/commandes](https://github.com/0xZKnw/mlxl3/tree/main/docs/measurements/updater-v1.4.6), archive verification-evidence.tar.gz/548203octets/SHA-256 `668457a6235389b5b140618ee7d8efee0f3c6cbc0b9b3e4d2501efd61c132983`. Les limites numériques et Kani ci-dessus restent explicites.
