# Harness Agent : DeepSeek Harness face au nôtre (2026-11-23)

Étude de lecture seulement : aucun code modifié, aucun essai lancé (donc pas d'entrée
d'essai dans `opti.md`). Question de l'utilisateur : « est-ce qu'on peut pas juste reprendre
le deepseek harness et l'intégrer ? », puis « vas-y, regarde ».

- **dsh** : `deepseek-ai/deepseek-harness`, commit `4b399c39a6d1479cd84ca591b4d7c88e24d38a96`
  (2026-11-22), MIT, lu via l'API GitHub (rien téléchargé ni exécuté). Même commit que les
  lectures d'AGENT-2026-11-23-HARNESS-SOLIDE et AGENT-2026-11-23-BOUCLE.
- **Nous** : `origin/main` `6a243e05d` (Desktop 2.0.22) : `native/src/main.rs`
  (`bridge_generate`), `native/src/agent_loop.rs`, `native/src/tool_call.rs`,
  `native/src/mcp.rs`, `apps/MLXL3Studio/Sources/MLXL3Studio/AgentJavaScript.swift`,
  `AgentSandbox.swift`.

## Verdict

L'intégrer tel quel serait une régression. dsh est un produit complet en Node 24 (boucle,
sessions, interface web/Electron, plugins), sans API intégrable dans un autre processus. Il
remplacerait la boucle du moteur et ses parseurs d'appels par famille de modèle. Sur les
points où l'on pourrait espérer un gain, il fait moins bien que nous :

| Sujet | dsh (base livrée) | Nous |
|---|---|---|
| Édition | `edit` strictement exact (CRLF seul normalisé), aucun « passage le plus proche » ; `apply_patch` en cascade **désactivé** dans la base | cascade exact → `N\| ` → espaces → ponctuation Unicode, lignes les plus proches, remplacement déjà présent, `edits` atomique |
| Lecture avant édition | implémentée (`fs-observation-policy`) mais **désactivée** | inutile : `find` doit correspondre au fichier actuel |
| Sandbox macOS | `sandbox-exec` en `(allow default)` + `(deny file-write*)` : lecture, réseau et exec libres | `jsc` sans fichiers, réseau ni processus ; commandes sous Seatbelt ; copie isolée relue par l'utilisateur |
| Annulation de changements | aucune restauration de fichiers (le « checkpoint » est un flush du journal) | copie isolée, rien n'est appliqué sans revue |
| Élagage du contexte | tout résultat > 8 192 caractères, **récents compris** | anciens résultats d'abord, 20 % les plus récents protégés |
| Appels répétés | rappels à 3/5/8, jamais bloquants | note à 3 (serveur JS), arrêt des outils à 5 (moteur) |
| Code exécuté | `run_code` TypeScript, processus Node + `vm` (pas une frontière), sandbox OS | `javascript`, noyau `jsc` persistant, tâches de fond |
| Petits contextes | `headroomTokens` 65 536 par défaut : la compaction **se désactive** sous ~65 k sans réglage manuel | pensé pour 32 k |

## Ce qui vaut la peine d'être repris

Classé par rapport gain/coût pour de petits modèles locaux. Aucun de ces points n'a été
mesuré ici.

### 1. Tests déterministes de la boucle avec un modèle scripté (priorité haute)

dsh teste la vraie boucle avec un faux adaptateur : une file de réponses scriptées, qui
échoue bruyamment quand le script est épuisé (`packages/core/agent-loop/tests/mock-adapter.ts`).
Il ajoute un rejeu de sessions enregistrées (`packages/test-support/llm-replay`), un serveur
modèle simulé (erreurs, coupures, JSON cassé) et des tests de propriétés.

Chez nous, les règles de la boucle (relance unique d'une étape vide, nouvel essai unique quand
le contexte est plein, arrêt après des appels répétés, appel illisible renvoyé au modèle, note
de dernier tour) vivent dans `bridge_generate`, environ 350 lignes de `main.rs`. Elles ne sont
vérifiées que sur GPU, par quelques essais indicatifs. Seules les fonctions pures de
`agent_loop.rs` ont des tests unitaires.

Piste : sortir la boucle derrière une interface « produire une étape » pour qu'un générateur
scripté la pilote dans des tests CPU, avec des oracles sur les événements émis et les messages
transmis. Aucun changement de comportement, et un vrai gain de solidité.

### 2. Deuxième niveau de contexte : résumé quand l'élagage ne suffit pas

dsh, `packages/compaction/compaction-basic` :
- **Ordre** : l'élagage sans modèle passe d'abord, le résumé par le modèle seulement si
  l'élagage ne suffit pas.
- **Seuil** : `min(0,8 × fenêtre, fenêtre − sortie − marge)`.
- **Queue gardée telle quelle** : 16 % de la fenêtre.
- **Coupe équilibrée** : la frontière ne sépare jamais un appel de son résultat.
- **Résumé** : Markdown en 8 sections fixes (demande, concepts, fichiers, erreurs/correctifs,
  tâches en attente, travail en cours, prochaine étape, contexte critique). Il fusionne le
  résumé précédent et il est rejeté s'il ne raccourcit pas le contexte.
- **Requête de résumé** : le préfixe exact de la conversation, plus une demande finale, pour
  réutiliser le cache.
- **Si le fournisseur déborde** : élagage, puis résumé de la plus grande tête possible, puis
  un seul nouvel essai.

Chez nous, B4 ne touche jamais aux textes système, utilisateur ou assistant. Quand l'élagage ne
suffit plus, l'étape tourne contexte plein et le tour s'arrête (`context_full`).

Pour nous, la requête de résumé en préfixe exact se marie bien avec `prompt_cache`. Le coût à
mesurer : décoder le résumé sur un modèle local, environ 1 à 2 k tokens à 20 tok/s, soit environ
1 à 2 minutes. Ce coût n'est pas mesuré. Il faut des marges dimensionnées pour les petites
fenêtres, pas les 65 536 de dsh.

### 3. Sortie complète enregistrée dans un fichier, avec la façon de la relire

dsh, `packages/spill/spill-policy` :
- **Seuil** : au-delà de 12 500 tokens, le résultat complet va dans un fichier privé (0600).
- **Ce que le modèle voit** : la tête, la queue et une notice qui donne le chemin et dit de
  le relire par plage ou d'y chercher.

Chez nous, `run` garde les 8 premiers et les 24 derniers Kio, et le moteur coupe au milieu
(`min(8192, contexte/8)` tokens). Le milieu est perdu, alors que c'est souvent là qu'est la
première erreur d'une longue compilation ou d'une suite de tests.

Piste : écrire la sortie complète dans `.mlxl3-scratch/`, qui est déjà exclu des changements
relus, et ajouter à la notice « `view(path, offset)` ou `search(regex, path)` ». C'est peu coûteux.

### 4. Erreur d'outil inconnu qui donne le remède (petit)

En mode PTC, dsh répond à un appel direct d'un autre outil en disant de l'appeler depuis
`run_code`. Chez nous (`native/src/mcp.rs`, `call_cancellable`), le modèle reçoit seulement
`Unknown MCP tool: read_file`. Les petits modèles appellent volontiers `read_file` ou `bash`
directement.

Piste : lister les outils disponibles et, si `javascript` en fait partie, rappeler que
`read`, `view`, `run`, etc. sont des fonctions à appeler dans ce programme.

Hors mode Agent, un outil hors de la liste autorisée du projet interrompt tout le tour
(`anyhow::ensure!` dans `bridge_generate`) au lieu d'être renvoyé au modèle. Ce problème est
antérieur et hors du mode Agent : `allowed_tools` y vaut `nil`.

### 5. Idées secondaires (à essayer seulement avec un essai GPU)

- **Contenu de fichier écrit depuis le code** : dsh conseille un tableau de lignes joint par
  `\n` plutôt qu'un seul gabarit (template literal). Cela évite les `\n` et `${` interprétés
  par JavaScript dans du code Python ou du shell. Notre contrôle de syntaxe après `write` ou
  `edit` couvre déjà une partie du problème.
- **Paramètre `description`** (5 à 10 mots) sur l'appel de code : il donne une étiquette
  lisible dans la timeline au lieu de « Code exécuté ». Il coûte quelques tokens par appel.
- **Contexte changeant envoyé à part** : dsh envoie le dossier de travail et les permissions
  comme message séparé, seulement quand ils changent, et annonce les outils ajoutés ou retirés.
  À regarder si l'on permet de changer les permissions en cours de conversation.
- **Message de l'utilisateur pendant un tour** (`steer()` : injecté à l'étape suivante sans
  arrêter le tour). Non vérifié chez nous.

## À ne pas reprendre

- Sandbox, édition et approbations : les nôtres sont plus strictes ou mieux adaptées.
- `llm-retry` : il cible des fournisseurs réseau (429/5xx). Notre équivalent local est la
  relance d'étape vide.
- Absence de plafond d'étapes : nous gardons 256 tours avec une dernière réponse sans outil.
- Marge de compaction de 65 536 tokens : incompatible avec nos fenêtres.
