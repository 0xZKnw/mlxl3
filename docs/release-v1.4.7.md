# MLXL3 Desktop 1.4.7 — build 29

## Nouveautés

Le texte reçu en streaming apparaît avec un fondu continu de 120 ms. L’arrivée
d’un nouveau fragment conserve la progression des précédents. Seul un suffixe
borné à 64 caractères estimés est animé ; aucun délai de type machine à écrire
ni timer permanent n’est ajouté.

Le défilement suit la hauteur réellement préparée avec une courte animation.
Le suivi se suspend quand on commence à remonter, pendant la décélération et
pendant la lecture des anciens messages. « Dernier message » reprend le suivi.
La préférence macOS de réduction des animations est respectée.

## Dessin et correctifs

Les portions de texte de même opacité sont dessinées ensemble. Une fois le
texte opaque, le renderer dessine directement les lignes. Les correctifs récents
LaTeX, import, mises à jour et réglages du Desktop 1.4.6 sont conservés.

Le transport et son regroupement de deltas de 50 ms restent inchangés. Le
nombre de caractères source reste une estimation des glyphes formés, notamment
pour les écritures complexes ; le contrat conserve le rendu et les textes.

## Performances

Les mesures préliminaires APP-02, sur Apple M5/24 Go et Qwen3.6-35B-A3B EXL3
2,49 bpw avec MTP1, donnent des médianes de débit stables : 56,84 → 56,85 tok/s
français et 59,28 → 59,41 tok/s code, huit passages ABBA + BAAB et 128 tokens
par requête. La variance empêche de démontrer une marge stricte de 1 %.

Le coût UI dépend du cas : la prose 64 Kio à 50 ms ajoute 0,331 s CPU sur une
relecture de 24 appends et environ 51 Mio d’empreinte processus. Le CPU n’est donc
pas gratuit partout. FPS, énergie et latence de présentation de chaque pixel ne
sont pas mesurés. Ces essais antérieurs au portage sur main restent historiques ;
les résultats du paquet final sont consignés séparément avant publication.

## Installation et compatibilité

**MLXL3-Desktop-v1.4.7-b29-Apple-Silicon.dmg**, Apple Silicon arm64,
macOS **26.2+**, MLX **0.32.2**. Signature ad-hoc, sans notarisation Apple.
Les modèles et conversations existants sont conservés.

Le bundle réutilise le moteur **1.4.8 déjà publié**, sans recompilation ni
modification de ses fichiers. Les versions Desktop et moteur sont indépendantes.
Le canal moteur [engine-v1.4.8](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.8)
reste publié avec `latest=false` ; cette publication Desktop fournit le DMG stable.
Aucun remplacement de l’app utilisateur n’est exécuté par nos tests.

## Vérification et limites

Les tests de régression ajoutent 786 cas finis de fondu, 63 comparaisons natives
de dessin, l’ancrage et l’interpolation du scroll, ainsi que la politique de
tracking, lecture et reprise. Une mutation qui redémarre les âges précédents
est rejetée. Les tests existants de LaTeX, copie complète, bridge, annulation et
historique long sont conservés.

Les résultats locaux et la CI du SHA source exact, l’identité du runtime publié,
les vérifications du DMG/updater, les tailles et les SHA-256 seront inscrits dans
la validation jointe. Swift strict et le vérificateur SIL contrôlent la compilation,
sans constituer une preuve de comportement SwiftUI. Les gestes physiques, les
autres Mac, les autres modèles et les contextes longs ne sont pas requalifiés.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), demandes, intégration et validation sur Mac.
- [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique conservé ;
  [MLX](https://github.com/ml-explore/mlx), runtime Apple Silicon du moteur embarqué.
- [Qwen](https://github.com/QwenLM), modèles utilisés pour les contrôles. Les
  attributions et licences amont du moteur restent dans sa release et le dépôt.
