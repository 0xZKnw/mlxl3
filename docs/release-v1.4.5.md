# MLXL3 Desktop 1.4.5 — build 27

Cette version corrige la mise à jour depuis Desktop 1.4.3. Elle conserve le moteur
1.4.5 déjà publié et les améliorations API, LFM, sampling et LaTeX des PR37–39.

## Correctifs

L’installateur confondait le numéro de l’app avec celui du moteur embarqué.
Desktop1.4.4 avec moteur1.4.5 échouait donc avant le remplacement de l’app, avec
un message trompeur « moteur incompatible ». Cette version lit le numéro du
moteur dans son manifeste et vérifie sa compatibilité avec l’app cible. Les
signatures, empreintes, fichiers, architecture, protocole et versions minimales
restent contrôlés.

Le numéro Desktop1.4.5 permet aussi aux anciens updaters, qui exigent encore
l’égalité des deux versions, d’installer cette mise à jour. Desktop1.4.4 est
conservé en préversion pour l’historique et exclu du canal stable.

## Nouveautés conservées

Les PR37,38,39 sont fusionnées : API locale corrigée, sampling accessible sur LFM
sans tête MTP, profil mémoire logique LFM et rendu de `\dots`/`\boxed`. Les outils
Python de calibration/diagnostic restent séparés de l’app ; aucun candidat
régressif de recalibration n’est promu. V2 reste le modèle publié.

## Performances et qualité

Aucune nouvelle quantification ni mesure de vitesse. Le moteur1.4.5 publié est
réutilisé sans changement de fichiers dans le nouveau paquet Desktop. Les poids
V2 restent inchangés. Le préfill MoE expérimental reste OFF par défaut ; aucun
gain de vitesse non qualifié n’est annoncé.

## Installation et compatibilité

**MLXL3-Desktop-v1.4.5-b27-Apple-Silicon.dmg**, Apple Silicon/macOS26.2+, moteur
1.4.5/MLX0.32.2. La recherche de mises à jour depuis1.4.3 propose cette version.
Les données et modèles sont conservés. Aucun remplacement de l’app utilisateur
n’est effectué par les outils de validation.

Signature ad-hoc, sans notarisation Apple. Le canal moteur garde `latest=false`.

## Vérification et limites

Le défaut est reproduit par le préflight de production1.4.3 sur le vrai DMG1.4.4.
Le correctif passe15cas signés : versions indépendantes, même version/build,
app trop ancienne, OS, architecture, schéma/protocole, version invalide,
empreinte moteur fausse et signature app altérée. Le vrai DMG est monté en
lecture seule, vérifié, puis détaché ; aucun installateur utilisateur n’est lancé.

Les résultats du paquet final, du préflight1.4.3 sur le nouveau DMG, de la suite
Desktop et des jobs CI sont consignés dans le manifeste de livraison. Les
sources Swift/AppKit ne sont pas prouvées formellement. La validation avant
remplacement ne constitue pas un test de remplacement automatique de l’app
réelle ni une nouvelle qualification numérique du modèle.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), signalement utilisateur, correctif et
  validation du parcours de mise à jour ; intégration des
  [PR37](https://github.com/0xZKnw/mlxl3/pull/37),
  [PR38](https://github.com/0xZKnw/mlxl3/pull/38) et
  [PR39](https://github.com/0xZKnw/mlxl3/pull/39).
- [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique ;
  [MLX/mlx-lm](https://github.com/ml-explore/mlx-lm), exécution Apple Silicon.
  Les autres contributions amont sont créditées dans les
  [notes du moteur1.4.5](release-engine-v1.4.5.md).
