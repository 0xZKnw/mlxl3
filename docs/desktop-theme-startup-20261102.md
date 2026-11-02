# Thèmes Desktop et chargement au démarrage

Branche : `codex/desktop-theme-startup`, base `72f4744` (`codex/qwen27-lossless-ram`).
Cette base conserve la version utilisée dans ce chat ; aucune intégration du main
amont plus récent n’est incluse. Le changement est limité au Desktop et aux tests.
Les modifications locales antérieures d’opti.md, relatives au nettoyage du disque,
sont conservées séparément et ne font pas partie de cette fonctionnalité.

## Comportement

- Réglages → Apparence : Système, Clair, Sombre. Préférence persistée et application
  immédiate dans la fenêtre, les réglages et le panneau de barre de menus.
  Les surfaces, textes, bordures, boutons et couleurs du code suivent l’apparence.
  Les formules sont des images modèles teintées selon le thème, cache partagé sûr.
- Réglages → Charger le modèle au démarrage : à l’ouverture de MLXL3, charge le
  dernier modèle sélectionné ou le premier disponible. OFF conserve la sélection
  mais attend une sélection manuelle ; ni l’activation ni la désactivation de cette
  préférence n’interrompt un modèle actif. Pas de service de lancement macOS.
- Compatibilité : thème sombre et chargement automatique activé par défaut,
  conformément au comportement antérieur. Préférences absentes ou invalides :
  repli explicite. Le mode Économie de mémoire n’est pas modifié.
- Un changement de thème ne recrée pas le StudioModel, ne réinitialise ni le
  brouillon ni la conversation et ne redémarre pas le moteur.

## Frontières et contrats

`AppSettingsView` → `StudioModel.setTheme` → UserDefaults / `@Published` →
`preferredColorScheme` des trois surfaces. `.system` transmet nil pour hériter
l’apparence de l’hôte. `StudioTheme.adaptiveColor` choisit la palette claire ou
sombre ; textes à contraste inverse sur les boutons remplis. Les couleurs
syntaxiques sont dynamiques, y compris les AttributedString déjà mises en cache.

`MLXL3StudioApp.onAppear` → `StudioModel.start` (idempotent, interdit en preview)
→ liste via subprocess → sélection restaurée ou première disponible → chargement
conditionnel. La préférence est relue dans le callback de démarrage : désactiver
pendant la liste empêche un chargement tardif. Activer après un démarrage OFF
attend le prochain lancement. `selectModel` et les imports restent indépendants.
La bibliothèque vide et les réponses JSON invalides ne démarrent aucun moteur.

## Vérification

Les nouveaux tests réutilisent les sources Swift réelles, le bridge et le moteur
Python factice. Préférences à UUID, historique temporaire, mise à jour pointant sur
un fichier local inexistant, aucun téléchargement ou modèle GPU. Attentes bornées
à six secondes, moteurs éjectés et données jetables nettoyées en sortie.
Le test est raccordé à `scripts/check-desktop.sh`, déjà appelé par la CI macOS
PR/push via `scripts/check-e2e.sh`. Aucune nouvelle dépendance de vérification.

Skill `formal-proof-skill` chargé ainsi que sa référence Python ; le skill n’a pas
de référence Swift. Aucun vérificateur formel Swift disponible (`swiftverify`
absent). Les propriétés SwiftUI/AppKit/Foundation restent **non prouvées** ;
compilation Swift 6, tests de cycle de vie et rendu ne constituent pas une preuve
formelle. Aucun code Rust/Metal ou calcul d’inférence modifié, donc pas de reprise
de Kani ni des campagnes numériques historiques.

Résultats finaux et logs : `docs/measurements/desktop-preferences-20261102/`.
Aucune installation ou release demandée ; l’app installée reste inchangée.


Résultat final : **52 assertions ciblées passent**, build debug et release passent,
ainsi que la suite Desktop complète (dont 114 assertions de déchargement, imports
150 cas finis/256 encodages, bridge, rendu, transport CLI, Tune 64 grilles et MCP).
Le contrôle négatif détecte un défaut de valeur par défaut dans une copie du vrai
code. Captures des réglages et du rendu riche inspectées en clair/sombre ; le test
Système simule le changement d’apparence de NSApplication. Les essais initiaux
avec chemin SwiftPM obsolète et simulation de fenêtre inadéquate restent dans les
logs. Aucun échec fonctionnel final. Le helper de rendu final impose une surface
fixe pour éviter de ne capturer qu’un fragment d’une LazyVStack sans ScrollView.
Les appels natifs de clic ne sont pas automatisés ; les vrais setters, vues et
sous-processus sont exercés. Le contrôle Metal est explicitement sauté pour ces
tests d’interface. Aucun gain de vitesse/mémoire ou preuve universelle revendiqué.


## Ajout Bleu nuit — 2 novembre 2026

Quatrième choix « Bleu nuit » (Midnight blue), palette bleu sombre inspirée des
éditeurs de code : canvas#1A2534, sidebar#111824, saisie#213045, accent bleu.
Le thème impose le mode sombre pour garder les textes, le code et les formules
lisibles. La sélection voyage dans une petite valeur d’environnement SwiftUI,
sans état global de palette et sans recréer le modèle de conversation. Les trois
choix précédents gardent leur palette et le sombre reste le défaut.

Les62assertions ciblées passent : persistance des quatre choix, rendu du même
panneau clair→sombre→bleu→clair, composante bleue vérifiée indépendamment sur le
bitmap, mode Système et parcours de démarrage. Fenêtre principale et réglages
inspectés visuellement. Build release et reconstruction/signature du bundle de
test réussis. Suite Desktop/Python complète du commit précédent non répétée pour
ce changement de palette ; tous les sources UI sont recompilés et la sélection
/rendu sont exercés par le harnais ciblé. Aucune vérification formelle Swift.
Preuves : [résumé et captures](measurements/desktop-blue-theme-20261102/verification-summary.json).
