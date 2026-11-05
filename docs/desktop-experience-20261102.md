# Desktop : lecture, épingles et MTP mémorisé

Branche `codex/desktop-pins-layout-mtp`, base `codex/desktop-theme-startup`
au commit `bacee9d99f6fc1251b45a42172d117e3e4a13c54`. Travail commencé
le 2 novembre, vérification finale le 3 novembre 2026. Les anciennes notes
locales de nettoyage de `opti.md` ne font pas partie de cette PR.

## Comportement

- Largeur des réponses **Centrée / Large**, disponible dans la barre du chat
  et les réglages, mémorisée au redémarrage. La taille des caractères reste
  identique ; messages, sélection et brouillon sont conservés.
- Épingles accessibles par le bouton de chaque chat ou son menu contextuel.
  Les chats épinglés apparaissent en premier, avec un ordre stable dans chaque
  groupe. Recherche, suppression/annulation, sauvegarde et export conservent
  leur état. Les historiques existants restent lisibles sans migration destructive.
- Nouveau bouton **MTP** dans la barre du chat et panneau commun à l’inspecteur :
  activation, état, profondeur, optimisation et tête externe avancée.
  Les erreurs et la préparation sont visibles ; le bouton distingue ON, OFF,
  AUTO et erreur. Les résultats Tune restent accessibles dans l’inspecteur.
- MTP **OFF pour un nouveau modèle**. L’activation et la désactivation sont
  mémorisées séparément pour chaque chemin de modèle, puis restaurées après
  `ready`, au rechargement et au démarrage automatique. Une erreur temporaire
  conserve l’intention ON et permet une nouvelle tentative ; OFF l’annule.
  L’ancien booléen global peut migrer vers un seul premier modèle, jamais tous.
  Les calibrations gardent leur clé modèle/tête/révision/runtime distincte.
- Flash Next `qwen4_exp` utilise sa tête intégrée dans le checkpoint cible.
  Le moteur 2.0.2 la prend en charge, mais son ancien sous-programme `mtp-head`
  refuse cette architecture : le Desktop évite désormais ce chemin incompatible.

## Parcours et contrats

`WorkspaceHeader` et `AppSettingsView` → `setReadingWidth` → préférences et
environnement SwiftUI → `MessagesView` / `ComposerView`. Aucun remontage de la
vue par identifiant de largeur, aucun changement du contenu.

`SidebarConversationItem` → `setConversationPinned` → `Conversation` et
`ConversationSnapshot` → `ConversationStore`. L’identifiant doit appartenir à
l’historique ; un identifiant absent ne modifie rien. Le champ optionnel
`isPinned` reste compatible avec les snapshots de schéma 1 existants.

`selectModel` / démarrage → `loadSelectedModel` → `restoreMTPPreference` →
événement `ready` → `setMTPEnabled` → préparation cible → `configureMTP` →
acquittement `mtp_status`. Les gardes de génération, identifiants d’opération,
annulation et timeout restent ceux du chemin de production. Le chargement de
Flash est validé par le moteur ; aucune tête étrangère n’est substituée.

MTP conserve le décodage déterministe déjà utilisé et la vérification des
propositions par le modèle cible. Aucun kernel, poids, précision du KV,
quantification ou politique mémoire n’est modifié. Aucun gain de vitesse ou
RAM n’est mesuré ou revendiqué.

## Vérification

Les commandes, journaux, captures et identités de sources sont conservés dans
[`measurements/desktop-experience-20261102`](measurements/desktop-experience-20261102).

- Nouveau harnais du vrai `StudioModel` et du bridge sur moteur factice :
  **35 assertions** (largeur, brouillon, épingles, ancien historique,
  préférences invalides, migration, OFF/ON, reload, démarrage, isolation entre
  modèles, échec, opt-out et tête Flash intégrée).
- Copie mutée du vrai `StudioModel`, sans adaptateur Flash : le harnais échoue
  sur l’oracle « Flash Next incorrectly used the external head installer ».
  Le code de production n’a pas été muté.
- Python : **358 passés, 4 ignorés**, identiques aux exclusions historiques
  (`ponyexl3` absent, modèle Ling absent).
- Suite Desktop complète passée : mémoire, compositeur49états, inspecteur MTP,
  préparation/annulation, idle114assertions, préférences62assertions,
  notifications14assertions, nouveaux contrôles35assertions, updater, imports
  (150frontières/256encodages), timeline, MCP, bridge, rendu4845fragments,
  réponse56cas, transport CLI, progression300cas et Tune64grilles. Les sorties
  des groupes ne sont pas additionnées en un nombre artificiel de tests.
  Après la dernière retouche qui évite de retrier la liste par ligne, le
  harnais expérience35assertions, les cinq captures et le bundle release ont
  été répétés ; les autres chemins exécutables sont identiques.
- Moteur réel Flash Next 2.0.2, contexte4096 : baseline OFF8tokens, ON8tokens,
  nouveau processus et ON8tokens. Texte, comptes et hash `30716b66c3e75491`
  identiques ; chaque génération ON a proposé3tokens, tous3acceptés. Deux
  cycles activation/désactivation/réactivation acquittés, deux enfants fermés
  et rejoints. Aucune compilation ni autre inférence GPU simultanée.
  Le contrôleur a aussi rejeté six corruptions de complétion et vérifié le
  nettoyage d’un sous-processus invalide (`failed`, `parity:false`).
- Build release et signature stricte du bundle réussis ; les quatre fichiers
  runtime correspondent toujours au manifeste après signature. Lint Python
  F/E9, syntaxe zsh et whitespace du diff passent.

La première suite Desktop s’est arrêtée sur un ancien oracle MCP qui exigeait
encore MTP global dès l’initialisation. L’oracle a été adapté à l’activation
différée par modèle et vérifie aussi que le sampling n’est pas changé avant
la sélection. Le journal d’échec est conservé avec celui de la relance.

Le skill `formal-proof-skill` et sa référence Python ont été lus. Aucun
vérificateur de source Swift n’est configuré ; CrossHair a été tenté mais son
module est absent. Ces contrôles sont des tests déterministes et des
validations d’intégration, **pas une preuve formelle**. Aucun Rust modifié,
donc aucun nouveau contrat Rust/Kani à vérifier. Les preuves numériques
historiques ne sont pas réattribuées à cette modification d’interface.
La comparaison physique porte sur un prompt court et ne prouve pas une parité
numérique générale. Batterie77% au départ, Apple M5/24Gio, macOS27.2,
Swift6.4.2/SDK26.5/MLX0.32.2. Temps, température, fréquences et gain RAM :
non mesurés. Les avertissements locaux du linker CLT et le diagnostic
CoreGraphics du fixture PDF apparaissent dans les journaux, sans échec final.

## Intégration

Le workflow `.github/workflows/check.yml` exécute `scripts/check-e2e.sh`, qui
inclut la suite Desktop et le nouveau harnais via `scripts/check-desktop.sh`,
pour les PR et pushes. La CI distante du nouveau commit doit être inspectée
séparément ; la présence du workflow n’en établit pas le résultat.

Le bundle `dist/MLXL3 Desktop Test.app` est une application de test locale.
La commande suivante construit/signature le bundle puis le lance, après
fermeture de l’application Desktop déjà ouverte :

```sh
cd "/Users/henko/Documents/Code/mlxl3-post-smallm" && ./scripts/build-desktop-test.sh --launch
```

Le script reprend le moteur installé vérifié 2.0.2, compatible Flash Next,
sans reconstruire l’ancien moteur de cette branche. Aucune installation dans
`/Applications`, fusion ou release n’est effectuée.
