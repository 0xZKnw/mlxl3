# MLXL3 Desktop 1.5.1 — build 33

## Nouveautés

Un choix **Stable / Bêta** est ajouté dans les réglages de mise à jour. Le canal Stable est le réglage initial ; le choix est enregistré pour les prochaines recherches. Le canal Bêta rend accessibles les préversions compatibles publiées par MLXL3.

Les releases Desktop et moteur restent sélectionnées séparément. Cette stable met à jour uniquement le Desktop ; aucun moteur 1.5.1 n’est publié. Les noms des fichiers, les URL GitHub et les empreintes SHA-256 restent contrôlés avant une installation.

## Correctifs

Les résultats d’une ancienne recherche ne peuvent pas remplacer ceux du canal courant après un changement de canal ou une annulation. Un moteur externe ayant exactement la même version que le moteur embarqué ne masque plus celui du nouveau bundle.

## Performances

Aucun nouveau gain de débit ou de mémoire n’est revendiqué. Cette livraison part de la stable1.5.0 et porte uniquement les canaux de mise à jour et leurs gardes ; le moteur stable 1.5.0 est conservé sans modification.

## Installation et compatibilité

Artefact prévu : **MLXL3-Desktop-v1.5.1-b33-Apple-Silicon.dmg**, Apple Silicon arm64, macOS **26.2+**, moteur embarqué **1.5.0**, MLX **0.32.2**. Signature ad-hoc, sans notarisation Apple.

Le choix d’un canal ne déclenche pas une installation et ne rétrograde pas l’application. Pour revenir d’une bêta à une stable de version inférieure, utiliser une installation manuelle et conserver une sauvegarde de ses données. Les modèles et historiques ne sont pas déplacés par ce choix.

## Vérification et limites

Les deux compilations ciblées Swift 6 avec avertissements traités comme erreurs passent, ainsi que les suites du sélecteur/updater et de l’installateur : choix et persistance du canal, tags/assets/URLs/hashes invalides, annulation, archives hostiles, activation/fallback et 15 cas signés de compatibilité/intégrité. Un premier arrêt du test installateur dans la sandbox est conservé ; le même binaire réussit hors sandbox, sans changement de code.

Le [rapport ciblé](desktop-v1.5.1-verification.md) distingue ces contrôles du paquet final. Au gel de cette source, le DMG final, ses signatures/empreintes et sa CI ne sont pas encore qualifiés. Aucune nouvelle génération, mesure de vitesse ou preuve formelle Swift n’est annoncée.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), demande du choix Stable/Bêta et validation du produit sur Mac.
- [MLX](https://github.com/ml-explore/mlx), moteur de calcul local conservé ; [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique conservé. Les licences et autres attributions de la stable restent dans le dépôt.
