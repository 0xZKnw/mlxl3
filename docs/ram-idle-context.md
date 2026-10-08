# RAM de MLXL3 au repos — M5 / version 1.4.2

Campagne du 8 octobre 2026. Base : release officielle `v1.4.2`,
`9f2a572e22a1560399ac8d77a3088573ed8e74cb`. Branche locale
`optimize/ram-idle-context`, repositionnée sur main
`d20ca721d808759f07ecf8976274d878e64a1ed3` pour la PR : les changements
amont depuis la release sont uniquement documentaires. App installée et poids préservés ; le candidat est
testé comme moteur séparé, avec un registre jetable. Les autres branches restent
intactes, notamment les anciens essais d'optimisation de kernels.

## Diagnostic

L'utilisateur constate 9–17 Go dans le Moniteur d'activité avec MTP et un PDF.
L'interface installée, avec son historique existant, mesure **119,1 Mio chargée**
puis **83,1 Mio après éjection**. Le moteur disparaît effectivement après
éjection. Ce contrôle ne reproduit pas toutes les tailles d'historique possibles,
mais ne montre pas de consommation de plusieurs Go par l'interface.

Le PDF est extrait en texte, plafonné à 256 Kio par pièce et 512 Kio par message.
Le modèle traite ce texte comme contexte ; aucun document/conversation privé
n'est utilisé dans les essais. Le cache d'images LaTeX est déjà borné à 128 entrées
et 32 Mio de coût estimé. Aucun changement de l'interface, des pièces jointes ou
de la persistance n'est nécessaire pour le problème reproduit côté moteur.

Un même modèle peut occuper des quantités différentes selon les poids chargés,
la tête MTP, le contexte, le checkpoint de préfill et les buffers GPU inutilisés.
Qwen3.8-27B possède environ 9,673 Go de fichiers safetensors ; sa tête MTP,
0,239 Go. Ces tailles sur disque ne sont pas une mesure de l'empreinte RAM.
Le pic MLX mesure les allocations actives et ne comprend pas tout le cache MLX ;
RSS ne remplace pas non plus l'empreinte macOS incluant les allocations Metal.

## Changement

À la fin d'un tour complet du bridge, après la disparition des variables de
génération, le moteur libère les KV/GDN finaux et l'état final de la tête MTP.
Ces états ne sont pas utilisés au prochain envoi : `prefill_round` et
`prefill_mtp` réinitialisent le modèle ou restaurent leur checkpoint distinct.
Ce checkpoint de préfill reste conservé, avec sa limite existante de 256 Mio.

Après synchronisation GPU, les buffers inutilisés sont libérés si le cache MLX
dépasse 512 Mio. Les petits jeux de buffers restent disponibles. Cette opération
se fait après succès, erreur, annulation et Tune ; jamais entre les tokens,
entre les rounds d'outils d'un même tour, ou en modifiant les poids, les kernels,
la quantification, la précision, les tokens du document ou les paramètres MTP.
Un échec de cleanup est consigné sur stderr sans créer un deuxième événement
terminal ni remplacer l'erreur d'origine. Un ping fournit aussi les compteurs
mémoire après traitement de la requête précédente.

## Vérifications

Les vérifications physiques de `clear_cache` et de restauration du snapshot
Qwen passent, chacune avec un vrai test sélectionné explicitement : les tableaux
vivants, logits et états restaurés restent exacts. Les 40 harnais Kani CPU
passent, dont la nouvelle politique de seuil, avec ses quatre couvertures
réalisées. Cela ne prouve pas MLX, Metal, la concurrence ou l'allocateur.
CrossHair n'est pas installé : aucune preuve Python n'est revendiquée.

Le driver vérifie des completions non vides, les hashes de tokens/texte,
les budgets et compteurs de contexte/MTP, le réemploi d'au moins 256 tokens,
le rejet de profondeur 4, et la reprise après annulation. Chaque moteur est
possédé, fermé et rejoint ; échanges bornés à 180 s, campagne à 900 s, contrôleur
externe à 930 s. Les rapports partiels/échecs restent distincts d'une parité validée.

Les premiers lancements des suites Python/Rust sous sandbox ont échoué à
l'initialisation Metal. Les diagnostics sont conservés et les contrôles nécessitant
le vrai GPU sont exécutés séparément hors sandbox. Les tests physiques sont
séparés des compilations et proveurs. La CI ajoute le lint strict et les tests
du nouveau driver ; un résultat local n'est pas un résultat de CI distante.

Suite finale : **319 tests Python passés / 4 sautés** (trois adapters PonyExl3
absents et modèle Ling non installé), **68 Rust passés / 60 ignorés**, dont les
deux tests physiques sélectionnés et passés séparément. Le driver/transport passe
50 tests, y compris oracles invalides, vraie erreur de child, stderr conservé,
rapport partiel et restauration des handlers. Format et Clippy strict couvrent
les variantes `chat` et `mlx,chat`. Une exclusion de compilation a été corrigée
après le diagnostic Clippy de la variante sans MLX : la politique n'y est pas
utilisée, son corps et la variante MLX mesurée restent inchangés. Le rebuild
final produit exactement le même SHA-256 de binaire que celui des huit essais
physiques (`rebuild-verification.json`).

## Résultats

Les données et commandes sont sous
[`measurements/ram-idle-context/`](measurements/ram-idle-context/).
L'empreinte au repos est lue après un ping accusant la fin du traitement,
avec `proc_pid_rusage`, sans confondre cache, active ou pic.

Les deux modèles ont traité le même texte synthétique : **10 138 tokens** pour
Qwen3.8 et **10 096 tokens** pour Qwen3.6, puis 32 tokens de sortie, MTP2,
sans réutilisation effective sur ce contexte long.
Le réemploi est vérifié séparément sur le contexte court, en MTP2 puis MTP3,
y compris après une requête de profondeur4 refusée. Deux ordres indépendants
A→B puis B→A, neuf cas par moteur : 72 cas dans huit processus terminés/rejoints,
56 completions dont les oracles sont exacts, huit annulations et huit erreurs
attendues. Les oracles sont aussi comparés entre les quatre processus par modèle.

| Modèle / ordre | Stock au repos | Candidat au repos | RAM rendue |
|---|---:|---:|---:|
| Qwen3.8-27B / A→B | 18,277 Go | 9,838 Go | 8,439 Go |
| Qwen3.8-27B / B→A | 18,278 Go | 9,836 Go | 8,441 Go |
| Qwen3.6-35B-A3B / A→B | 18,511 Go | 12,947 Go | 5,564 Go |
| Qwen3.6-35B-A3B / B→A | 18,519 Go | 13,006 Go | 5,513 Go |

Sur Qwen3.8, environ **46 % d'empreinte en moins au repos** dans ces cas,
dont 7,825 Go de cache MLX stock inutilisé après le long préfill. Le candidat
garde environ 9,411 Go actifs à ce stade, cache inutilisé nul. Sur Qwen3.6,
le stock gardait environ 5,264 Go de cache, le candidat garde 12,554 Go actifs,
cache nul. Les checkpoints/poids restent chargés ; la différence n'est pas
obtenue en coupant MTP ou en réduisant le document.

Le premier essai dense a été interrompu sur le stock à 180 s, avant tout
candidat. Il reste `failed/parity:false`. Les reprises denses complètes utilisent
300 s par échange, 1200 s global et un contrôleur externe à1230 s ; même workload
et critères. Le premier stderr temporaire n'avait pas été exporté : limite
signalée, corrigée et couverte par un vrai processus de test dans le driver v2.
Les anciens scripts, commandes, diagnostics et mesures restent conservés.

Le changement vise la **RAM conservée après une réponse**. Le pic pendant le
préfill n'est pas réduit par cette stratégie. Les timings restent diagnostiques :
le Mac est sur batterie et les débits/TTFT varient largement entre passages ;
aucune accélération ni absence de coût de réallocation n'est certifiée.

Les métriques de la bulle de conversation décrivent la génération avant cleanup.
Le Moniteur d'activité et le menu mémoire de l'app permettent d'observer la RAM
actuelle après la réponse. Les autres architectures, le pic d'import de PDF et
les parcours DFlash/MCP/Tune complets ne sont pas mesurés par cette matrice ; le
cleanup est placé après l'intégralité du tour et après Tune, sans modifier leurs
calculs ni le cache de préfill. Les checkpoints disponibles sont les deux Qwen.

Les résultats ne généralisent pas à tous les modèles, PDF ou tailles de contexte.
L'app installée reste la release officielle : le changement exige une nouvelle
construction du moteur pour être utilisé dans l'app, sans modification des poids.
