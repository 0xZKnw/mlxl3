# Taille du texte dans le bouton d’affichage

Base `6710bcd503d9ac1944acecc035bc54ec5809b4e7`, même branche
`codex/desktop-pins-layout-mtp`. Le bouton de largeur ouvre maintenant un
panneau avec le choix Centrée/Large, un curseur natif de12à24points, la valeur
courante, un aperçu immédiat et la réinitialisation à15points.

Le réglage est mémorisé au redémarrage et agit en direct sur les messages,
titres, listes, citations, code, tableaux et formules. La réflexion et les
résultats textuels d’outils suivent aussi le facteur typographique. La saisie,
les boutons et les paramètres moteur gardent leurs tailles habituelles.
Les préférences de taille et de largeur sont indépendantes.

## Chemin et contrats

`ConversationDisplayControl` → `ConversationDisplaySettings` → binding du
Slider → `StudioModel.setResponseTextSize` → `ResponseTextSize.normalized`
et préférence `studio.responseTextSize`. Les13positions entières sont permises.
Valeurs finies hors bornes : clamp ; valeurs fractionnaires : arrondi ; NaN,
infini ou préférence mal formée/booléenne : défaut15points.

`StudioView` injecte le facteur dans l’environnement → `MessageView` /
`AssistantMessageView` → rendu Markdown/réflexion. Les vues équatables reçoivent
la taille explicitement et la comparent : une modification de style doit
redessiner le contenu même si aucun nouveau token n’arrive. Les StateObject,
caches de parsing, identifiants, contenu et sources copiées sont conservés ;
aucun remontage par un identifiant de taille et aucun rechargement du modèle.
Les images LaTeX utilisent leur cache existant indexé par taille.

## Vérification

- **51assertions ciblées** :13positions/persistance, bornes, arrondi,
  valeurs non finies et préférences invalides ; brouillon Unicode, messages
  non vides, sélection et état de génération conservés.
- Cinq rendus vivants : prose, titres, code, tableau et formule. Séquence
 15→24→12→15points dans la même vue. Hauteurs prose219→492→161→219points ;
  titres108→137→99→108 ; code136→163→127→136 ; tableau133→160→127→133 ;
  formule65→76→61→65. Ce sont des observations de mise en page, pas un benchmark.
- Le vrai `StudioView` change ses pixels de messages à24points et retrouve
  exactement le rendu initial à15points ; le curseur ne dépend pas d’une
  génération. Les captures du panneau existent aux tailles12/15/24.
- Deux premiers contrôles ont trouvé un retour au défaut incomplet avec
  l’environnement seul, y compris après200ms d’attente. Les journaux sont
  conservés ; la propagation explicite et les égalités ont corrigé ce cas.
- Mutation isolée qui ignore la taille dans le vrai MarkdownResponseView :
  échec attendu sur l’absence de changement vivant (exit133). Le code de
  production est resté intact ; le nouveau test est intégré à la suite Desktop
  existante, exécutée par la CI PR/push.
- Suite Desktop complète passée : hardening, idle114assertions,
  préférences62, notifications14, expérience35, taille51, dispositionMTP16,
  updater/imports, timeline/MCP/bridge, code1194fragments, Markdown4845,
  réponse56cas/copie/streaming, CLI et Tune64grilles. Ces nombres ne sont pas
  additionnés en un compte artificiel de tests. Bundle release signé strict,
  quatre fichiers runtime conformes au manifeste2.0.2 après signature.
  Aucun nouveau test GPU, changement numérique, quantification,
  économie mémoire ou gain de performance revendiqué.

Le skill `formal-proof-skill` a été relu. Swift n’a pas de vérificateur source
configuré : ces tests déterministes, captures et contrôles d’intégration
ne constituent pas une preuve formelle. Les journaux des échecs restent
conservés avec les réussites. La CI distante du commit publié est distincte
des contrôles locaux.
Les octets des deux premières variantes utilisant uniquement l’environnement
ne sont pas archivés ; leurs diagnostics ne valident pas la source finale.
La source finale testée est identifiée par ses empreintes dans les preuves.

Preuves : [`measurements/desktop-text-size-20261103`](measurements/desktop-text-size-20261103).
Application de test locale : `dist/MLXL3 Desktop Test.app`, moteur vérifié2.0.2.
Aucun remplacement de l’application installée, merge ou release.
