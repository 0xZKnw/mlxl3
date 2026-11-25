# Colonne d'activité de Desktop : appels d'outils, réflexion et indicateurs (2026-11-24)

## Objectif

Les appels d'outils (MCP en Chat, sandbox en Agent), la réflexion et les indicateurs de génération étaient jugés « ultra
cheap » : une carte pleine largeur par appel avec le nom brut de la fonction (`web_search_exa`, « MCP · exa ») et un
statut isolé à l'autre bout de la ligne, un point orange qui pulse sous la réponse et à côté de « Thinking », une ligne
« Context processed · 0 s » à chaque tour, chaque réflexion dépliée sur 180 pt et une rangée de boutons sous chaque
message. En mode Agent, une tâche de dix étapes remplissait plusieurs écrans de cadres.

## Sources consultées

Avis de praticiens et documentation de composants, pas des études :

- guides anti « AI slop » : referodesign `anti-ai-slop.md`, wiki Laith0003 « Anti-AI-slop ban list » ;
- timetobuildbob, « Making long agent conversations scannable » (2026-03-30) ;
- 21st.dev, « React typing indicator components » (2026-08-21) ;
- composants d'appels d'outils et de shimmer : AI SDK Elements, shinychat, aiuxplayground, MUI X Chat.

Règles retenues : pas de carte quand rien n'est interactif ; la couleur seulement pour un état (échec) ; dire ce qui
tourne plutôt qu'animer des points ; le mouvement seulement tant que l'état change ; une seule famille d'icônes ;
détails repliés par défaut ; Reduce Motion respecté.

## Ce qui change

| Avant | Après |
|---|---|
| carte par appel, `web_search_exa`, « MCP · exa », « Done » | une ligne : icône, action au bon temps, objet (« Read `src/stats.pl` », « Ran `sh test.sh` », « Web search “mlx cache” · exa ») |
| échec : « Error » | icône et issue en `StudioTheme.failure` ; pour une commande, son code (« Exit 1 » / « Code 1 ») |
| point pulsé à côté de « Thinking » et sous la réponse | balayage lumineux sur le libellé en cours (`StudioShimmerText`), rien quand le texte défile déjà |
| « Réflexion en cours » + trois points | ligne « Préparation de la réponse » balayée |
| réflexions toujours dépliées | dépliée pendant le stream, repliée ensuite en « Thought for 4 s » / « Réflexion · 4 s » ; un clic l'épingle |
| « Context processed · 0 s » à chaque tour | lecture de contexte affichée en direct, conservée ensuite seulement si ≥ 1 s ou interrompue |
| 13 pt entre chaque ligne d'activité | lignes consécutives regroupées (3 pt), la réponse garde son espacement |
| boutons Copier/Modifier/Brancher sous chaque message | au survol (la dernière réponse les garde) ; à gauche de la bulle pour l'utilisateur, sans réserver de hauteur |
| bloc de code d'une ligne de 76 pt | hauteur ajustée aux lignes |

Le détail d'un appel (entrée indentée — code JavaScript tel qu'écrit, JSON trié — puis résultat ou erreur) s'ouvre sous
la ligne dans un panneau discret, avec « Tout afficher » et Copier. L'entrée est omise quand la ligne montre déjà
l'unique argument (`read_file {"path": …}`).

La présentation est une fonction pure (`ToolCallPresentation`) du `ToolActivity` enregistré : les conversations
restaurées s'affichent de la même façon. Les durées utilisent les champs `startedAt`/`elapsed` déjà persistés
d'`AssistantPart`, maintenant renseignés aussi pour la réflexion et les outils ; les anciennes conversations n'ont pas
de durée et affichent « Thought » / « Réflexion ».

## Vérifications

Voir l'entrée `UI-2026-11-24-ACTIVITY-ROWS` d'`opti.md`.
