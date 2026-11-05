# Fluidité de MLXL3 Desktop et affichage des tokens

Audit du 16 octobre 2026 sur les sources Desktop de `1ac910d0a40efcad049c6e5a8cf081d770723465`. Le premier candidat est une retouche du fondu existant et du suivi du bas. Le coût presque nul demandé doit être établi par comparaison ; aucun gain de CPU, débit, FPS ou mémoire nouveau n'est mesuré dans cet audit.

## Ce qui fonctionne déjà

Le [bridge Swift](../apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift) regroupe les fragments sur sa queue IO pendant 50 ms et les vide avant un événement non delta. [ChatMessage.append](../apps/MLXL3Studio/Sources/MLXL3Studio/Domain.swift) publie le changement du seul message ; les tokens ne republient déjà pas le modèle global de l'app. La préparation Markdown utilise un acteur, une tâche active et un seul snapshot en attente. Le transcript et les chunks sont paresseux et les parties figées sont équatables.

Ces acquis sont documentés dans UI-113 et AUDIT-05 du [journal](../opti.md), ainsi que dans la [validation Desktop 1.1.3](desktop-v1.1.3-validation.md). Leurs gains historiques ne constituent pas une mesure du candidat proposé ici. Le fondu actuel existe déjà : 15 % d'opacité initiale, puis 120 ms, sur une estimation des 64 derniers caractères au maximum.

## Les trois changements à examiner en premier

| Ordre | Constat dans le code actuel | Plus petit candidat et limite |
| --- | --- | --- |
| 1 | `TokenFadeRenderer.draw`, `MarkdownResponseView.swift:1466–1480`, matérialise les runs, recompte leurs glyphes et dessine individuellement les glyphes du run traversant la frontière animée, y compris son préfixe opaque. Ce chemin reste utilisé à `progress == 1`. | Dessiner directement les runs lorsque le fondu est terminé ou désactivé ; sinon dessiner le préfixe et le suffixe par tranches ayant la même opacité. Cela réduit les appels sans changer la cadence. Un run appartient à une ligne visuelle, pas normalement à tout un paragraphe. Vérifier les glyphes qui se chevauchent et le rendu aux opacités intermédiaires. |
| 2 | Le transcript attend au moins 250 ms entre tentatives de suivi, puis effectue un `scrollTo` sans animation (`ChatComponents.swift:49–55,79–98`). Les déplacements déclenchés par ce follower sont donc au plus à 4 Hz. Il observe la révision du message, avant la préparation asynchrone de sa nouvelle hauteur. | Essayer l'ancrage natif lors des changements de taille, conditionné au suivi et à l'absence de geste utilisateur. Le même pattern existe déjà pour la réflexion à la ligne 368. Ne supprimer le follower qu'après validation du transcript réel. L'ancrage traite le suivi ; il ne garantit pas à lui seul une animation douce. |
| 3 | `SmoothTokenText`, `MarkdownResponseView.swift:1446–1454`, remet le fondu à 15 % à chaque nouvelle source et remplace la taille de sa queue. Le suffixe précédent devient opaque au nouvel arrivage, même si ses 120 ms ne sont pas écoulées. | Conserver une progression continue lors d'une arrivée pendant le fondu ; viser un effet court, initialement 80–120 ms à comparer, sans file d'attente de caractères ni retard artificiel de la réponse. Séparer cet essai du changement de dessin pour attribuer le résultat. |

Le suffixe est estimé à partir de `String.count`, puis appliqué à des glyphes du texte rendu. Markdown, ligatures, emoji et écritures complexes ne donnent pas une correspondance caractère/glyphe/token. Le premier candidat conserve le contrat actuel ; un défilement réellement caractère par caractère demanderait un contrat et des tests distincts.

Le résultat visé est un texte déjà disponible dont seule l'arrivée visuelle s'adoucit. Conserver le batching de 50 ms évite de refaire parsing et publication à chaque image. Une animation SwiftUI peut interpoler le dessin entre deux publications ; elle consomme néanmoins des ressources graphiques. [TextRenderer](https://developer.apple.com/documentation/swiftui/textrenderer) est l'API existante à conserver. Apple expose aussi un [rôle d'ancrage pour les changements de taille](https://developer.apple.com/documentation/swiftui/view/defaultscrollanchor(_:for:)). Leur aptitude au transcript actuel reste à vérifier.

## Autres pistes dans l'app

- **Historique volumineux** : chaque delta appelle `activeMessage()`, qui parcourt conversations et messages (`StudioModel.swift:1112–1118,1207–1215`). Une référence privée au message actif éviterait cette recherche, avec nettoyage à toutes les fins, annulations et erreurs. Le coût réel reste à mesurer.
- **Saisie et recherche** : `SidebarView.filteredConversations` recherche dans les titres et contenus et est évalué pour le compteur puis pour la liste (`StudioView.swift:74–79,132,142`). Les frappes dans le brouillon publient le modèle global. Calculer le filtre une seule fois par rendu est le premier petit candidat ; séparer davantage l'état seulement si un profil le justifie.
- **Rendu encore sur le thread UI** : le Markdown inline, le scan mathématique et les nouvelles images de formules restent dans `InlineLatexText.render` (`MarkdownResponseView.swift:1162–1189`). La réflexion découpe aussi son texte dans `ThinkingBlock.body` (`ChatComponents.swift:354`). Profiler ces coûts avant d'ajouter des caches.
- **Sauvegardes longues** : le snapshot de tout l'historique se construit sur `MainActor` toutes les dix secondes pendant une génération (`StudioModel.swift:1225–1248`). L'encodage et l'écriture sont déjà déportés ; l'ancien fichier est relu, validé et sauvegardé (`ConversationStore.swift:99–110`). Une évolution doit conserver les révisions, l'atomicité et la protection des historiques illisibles.

Une réécriture AppKit, une migration générale de l'observation et un timer permanent à 60/120 Hz ne sont pas justifiés par ces constats. [Apple recommande d'attribuer les mises à jour coûteuses ou trop fréquentes avec Instruments](https://developer.apple.com/documentation/xcode/understanding-and-improving-swiftui-performance) avant d'élargir le changement.

## Vérification du premier candidat

Préenregistrer séparément dessin, scroll et continuité du fondu. Archiver sources et binaires A/B avec leurs SHA-256 et conserver exactement le même moteur. Tester la correction avant la vitesse.

1. **Régression visuelle et fonctionnelle** : source vide, un caractère, limites 63/64/65, arrivées espacées et en rafales, opacités 0,15/0,5/1, fin et annulation en plein fondu, remplacement/troncature, Reduce Motion et changement de conversation. Couvrir emoji ZWJ, caractères combinés, ligatures, arabe, Markdown et code coloré. Vérifier texte/copier/sauver exacts et progression entièrement opaque après la fin. Comparer le rendu réel aux mêmes états de référence, avec une tolérance graphique fixée avant essai.
2. **Scroll réel** : prose, gros fence, réflexion et outils ; nouvelle ligne, dernière préparation après le dernier delta, fenêtre redimensionnée. Remonter à la main doit suspendre le suivi, le bouton de reprise doit fonctionner, aucune tâche d'une ancienne conversation ne doit déplacer la nouvelle.
3. **UI seule** : moteur factice et fenêtre native attachée de taille fixe ; traces déterministes lentes vers 15 tokens/s, rapides et en rafales. Réutiliser les tailles 64/256/1024 Kio de UI-113, avec warmup et ordres A/B/B/A puis B/A/A/B, sans compilation ni modèle simultané. Mesurer le flux complet jusqu'à la fin du dernier fondu : CPU UI/principal, empreinte, retards p50/p95/p99/max, arrivée→publication→préparation et retard final. Le harnais historique `append + layoutSubtreeIfNeeded` ne couvre pas tout le dessin asynchrone ; ses temps ne sont pas des FPS. Réserver une passe de profilage distincte à l'attribution des coûts.
4. **Inférence séparée** : mêmes poids, prompts, contexte, cache, budget, mode MTP et échantillonnage ; une seule app et un seul modèle. Relever alimentation et état thermique, warmup stable, A/B/B/A puis B/A/A/B sans profiler/proveur/compilation. Exiger texte, IDs et compteurs identiques. Comparer débit decode, TTFT moteur et bout en bout, durée et empreintes app/moteur. Une marge proposée de 1 % doit être fixée avant essai ; si le bruit ne permet pas d'exclure une régression supérieure, résultat non concluant.

Les commandes existantes après implémentation sont `scripts/check-desktop.sh`, puis `scripts/check-e2e.sh`, avec `MLXL3_TEST_PYTHON` et `MLXL3_MACOS_SDK` adaptés à l'environnement, et lint Swift ciblé. Les tests actuels couvrent Unicode, parité des chunks, préparation périmée/annulée, ordre du bridge et sauvegarde/relecture d'une réponse de 20 000 lignes. Ils ne couvrent pas directement la trajectoire du fondu ni le suivi du scroll. Swift 6 et ces tests ne constituent pas une preuve formelle SwiftUI.

## État de la vérification

Audit statique uniquement : zéro test applicatif, zéro build, zéro benchmark et zéro propriété formelle exécutés dans cette tâche. Les sources Desktop et checks suivis sont identiques à HEAD ; les changements préexistants du moteur et des autres fichiers sont préservés. [Empreintes et état source](measurements/app-fluidite-2026-10-16/sources.json), [statut CI inspecté](measurements/app-fluidite-2026-10-16/ci.json).

Le [run Desktop du SHA exact](https://github.com/0xZKnw/mlxl3/actions/runs/37306150549) et son job `desktop-and-protocol`, dont l'étape `scripts/check-e2e.sh`, sont réussis. C'est la CI du commit du 5 octobre, pas une validation du checkout moteur modifié ni d'un futur candidat. Les contrôles hébergés sautent explicitement Metal ; aucun FPS ni effet sur l'inférence ne s'en déduit.

Décision : commencer par le dessin du fondu et l'ancrage du bas, puis comparer la continuité de l'animation. Candidats non implémentés et performances non mesurées ; aucune app installée remplacée, aucun essai d'exécution lancé ou restant en cours.
