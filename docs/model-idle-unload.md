# Déchargement automatique après inactivité

Dans **Réglages → Déchargement automatique du modèle**, choisir **Jamais**,
1, 5, 10, 15 ou 30 minutes, 1 heure ou 2 heures. Le défaut est **15 minutes**.
La préférence est globale et conservée au redémarrage de l'app.

Le délai commence lorsque le modèle est prêt et inutilisé. Une génération,
ses appels d'outils, le tuning MTP, la préparation de la tête ou une
configuration MCP/MTP suspendent la minuterie. Un délai complet repart
quand ces opérations finissent, y compris après annulation. Modifier le
délai applique un nouveau compte à rebours ; Jamais le supprime.

À l'échéance, l'app réutilise `ejectModel()` : arrêt du processus moteur,
libération des caches et de la tête, état « Modèle éjecté ». La sélection,
les conversations et les réglages sont conservés. Sélectionner de nouveau
le modèle le recharge. Les fichiers du modèle ne sont pas supprimés.

Le contrôleur possède une seule tâche annulable, avec une échéance
`ContinuousClock`, qui compte aussi le temps de veille. Chaque annulation
invalide son ticket ; une ancienne échéance ne peut pas décharger un
modèle rechargé. Le callback vérifie à nouveau l'état du moteur avant
d'éjecter. Aucun sondage permanent du GPU ni modification des kernels.
Voir les [horloges Swift présentées par Apple](https://developer.apple.com/videos/play/wwdc2022/110355/).

## Vérification

Base `8f63e6c89e656bb1e582d9a3dc92b1135b275de0`, branche
`feat/model-idle-unload`, Mac arm64, Swift6.4 et SDK macOS26.5.

- Build de l'app réussi. Compilation des sources de l'app et du harnais
  en Swift6, concurrence stricte et warnings-as-errors réussie.
- **77 assertions** sur les délais, valeurs persistées/invalides, callback
  réel, annulation, durée modifiée, destruction du contrôleur et le vrai
  `StudioModel`/bridge avec moteur subprocess jetable. Génération, arrêt,
  tuning, annulation du tuning, configuration MTP ON/OFF, MCP,
  remplacement/rechargement, crash et conservation de l'historique couverts.
- Suite Desktop existante réussie : lifecycle, updater, import PDF/texte,
  bridge, rendu/streaming, transport CLI, préférences et configuration MCP.
  Son contrôle physique Metal est explicitement désactivé ; aucun modèle
  réel n'a été chargé ou déchargé pour ces tests.
- Copie temporairement incorrecte sans garde de tuning rejetée par
  « MTP tuning was interrupted ». Les sources de production sont intactes.
- **273 tests Python** des outils natifs passés, aucun sauté. La suite
  Python complète n'a pas pu collecter deux modules de conversion : Metal
  inaccessible au sandbox et `mlx_lm` absent ; un module `ponyexl3` sauté.
  Cette limitation est conservée, sans changer ces outils non affectés.
- Aperçu isolé contrôlé via l'accessibilité : section visible, huit choix,
  défaut15minutes et sélection effective de Jamais. Aucun réglage personnel
  ni application installée modifié.

Les premiers diagnostics de compilation sont conservés : dépréciations
CoreText dans SwiftMath vendored lorsque warnings-as-errors était appliqué
globalement, avertissement d'une référence faible locale du test et deux
tentatives ciblées avec un ancien chemin SwiftPM. Le build normal du dépôt
et la compilation stricte de l'app avec les objets vendored réussissent.
Le harnais final conserve une référence faible mutable compatible avec les
anciens compilateurs Swift6. Aucune assertion de comportement relâchée.

Il n'existe pas de vérificateur source Swift exploitable dans l'outillage
existant du projet ; compilation et tests dynamiques **ne sont pas une
preuve formelle**. Les horloges des tests lifecycle sont injectées, le
timer réel est vérifié séparément. Aucun chiffre de RAM libérée ou de
performance du modèle réel n'est revendiqué.

Commandes, résultats et empreintes :
[verification-summary.json](measurements/model-idle-unload/verification-summary.json).
La CI existante passe par `scripts/check-desktop.sh`, qui exécute maintenant
le nouveau harnais. La branche locale n'est pas publiée ; aucun run GitHub
de cette fonctionnalité n'est annoncé vert. L'app personnelle reste à sa
version installée ; le réglage nécessite un nouveau build de **Desktop**.

## Récupération MTP — correctif de la revue PR28

Une réponse MTP illisible produit une erreur sans identifiant de requête.
La préparation est maintenant annulée dans ce cas, tout en conservant le
traitement des erreurs de génération et de tuning. Une demande MTP OFF
sans réponse récupère après30secondes ; sa tâche est annulée lors d'un
acquittement, d'une éjection ou d'un rechargement. Le délai idle complet
repart après récupération ou fin de génération, sans interrompre une
génération active àl'échéance MTP.

La régression échoue sur la source originale. Le harnais final exécute
114assertions, avec attente MTP30s réelle et génération libérée par un
signal fichier jetable ; le contrôle indépendant de revue passe3/3cas.
Build et suite Desktop réussis. Commandes, hashes et limites :
[vérification du correctif](measurements/pr28-fixes/verification-summary.json).
La preuve formelle Swift et le déchargement d'un modèle physique restent
non exécutés ; les résultats locaux ne valent pas résultat de CI distante.
