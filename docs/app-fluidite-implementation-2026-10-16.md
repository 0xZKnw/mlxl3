# Fluidité Desktop — implémentation et vérification du 16 octobre 2026

État : **fonctionnel validé dans les cas testés, code local**. Les médianes de
débit sont stables ; **la marge stricte de 1 % reste non concluante**. Aucun gain
CPU universel revendiqué. Aucune installation, publication, modification ou
reconstruction du moteur.

## Comportement livré dans les sources

- Le fondu de 120 ms conserve l’âge des fragments précédents quand un nouveau
  fragment arrive. Le suffixe animé reste borné à 64 caractères estimés ; il
  disparaît immédiatement à la fin du streaming ou avec Reduce Motion.
- Le dessin utilise des tranches de run de même opacité et dessine directement
  les lignes lorsque le texte est opaque. Aucun timer permanent ajouté.
- Le scroll suit la hauteur effectivement préparée avec une animation de 120 ms,
  via une tâche annulable après layout. Le suivi se suspend dès le début du geste,
  pendant la décélération et quand on lit plus haut. Reduce Motion désactive
  l’animation. Le bouton « Dernier message » reprend le suivi.

Sources : `MarkdownResponseView.swift` et `ChatComponents.swift` dans
`apps/MLXL3Studio/Sources/MLXL3Studio/`. Callers conservés : paragraphes et titres
Markdown, texte de réflexion et blocs de code via `SmoothTokenText`, transcript
via `MessagesView`. Le transport et son regroupement de deltas de 50 ms sont
inchangés ; aucun retard de type machine à écrire n’est ajouté.

La correspondance entre caractères source et glyphes formés reste une estimation,
comme auparavant. Le nouveau contrat de dessin utilise l’opacité native : une
égalité pixel à pixel avec l’ancien mélange de chaque glyphe n’est pas revendiquée
pour les glyphes qui se chevauchent.

## Correction avant performance

La suite complète `scripts/check-e2e.sh` a réussi sur le Mac physique, avec SDK
macOS 26.5, Python du dépôt et caches de compilation sous `/tmp` :

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
CLANG_MODULE_CACHE_PATH=/tmp/mlxl3-app-module-cache \
SWIFTPM_MODULECACHE_OVERRIDE=/tmp/mlxl3-app-module-cache \
MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
PYTHONPATH=src scripts/check-e2e.sh
```

- Python : **49 tests réussis, 1 ignoré**.
- Desktop : hardening, updater, imports, timeline, MCP, bridge, transport CLI,
  téléchargements et réglages MTP réussis ; build de l’app réussi.
- Fondu : **786 cas finis** d’append et de bornes, valeurs d’opacité finies,
  monotones et bornées, âges continus, expiration, remplacement, vide et Unicode.
  Les cas incluent des fragments de caractères combinés, ZWJ, drapeaux et CRLF.
- Dessin : **63 comparaisons natives**. Oracle indépendant `Text.opacity` pour
  l’opacité uniforme et les écritures complexes ; oracle par glyphes pour les
  frontières de suffixe monospacé et les fragments d’âges différents. Tolérance
  fixée : écart maximal 32/255 et au plus 0,5 % de canaux avec écart supérieur à 2.
- Scroll natif : ancrage initial/final et positions intermédiaires vérifiés.
  Tracking/drag/décélération/lecture/reprise : tests de la politique de production.
  Les faux événements de molette n’ont pas permis un test fiable du geste physique.
- Intégration conservée : 1 194 fragments de code et 4 845 fragments Markdown,
  56 parités de préparation worker, réponses longues de 20 000 lignes à travers
  bridge/rendu/copie/historique, annulation et rejet des préparations obsolètes.

Une mutation isolée supprimant les âges précédents est rejetée par le test de
continuité. Le code de production n’a pas été muté. Swift 6 strict,
`-warnings-as-errors` et `-Xfrontend -sil-verify-all` réussissent sur les sources et
le test natif ; cela vérifie les invariants du compilateur, sans prouver le
comportement de SwiftUI. Aucun vérificateur de comportement SwiftUI utilisable
n’est configuré. CrossHair a été tenté pour le contrôleur Python, mais le module
est absent. Aucune preuve complète revendiquée ; aucun changement Rust/Kani.

Le lint strict porte sur les blocs ajoutés/modifiés et le nouveau harnais. Les
anciens fichiers contiennent des violations de style préexistantes. Le build
complet garde aussi les avertissements préexistants de SwiftMath et de chemins
de frameworks du SDK CommandLineTools.

Preuves : [suite complète](measurements/app-fluidite-impl-2026-10-16/check-e2e-height.log),
[mutation rejetée](measurements/app-fluidite-impl-2026-10-16/fade-mutant.log),
[vérification SIL](measurements/app-fluidite-impl-2026-10-16/response-height-task-build.log).

## Comparaison préenregistrée

Essai `APP-2026-10-16-02` dans [opti.md](../opti.md). Référence A : sources Desktop
de `1ac910d0a40efcad049c6e5a8cf081d770723465`, propres au début de la tâche et
archivées sous `/tmp/mlxl3-app-fluidite-20261016-baseline`. Candidat B : les deux
fichiers UI ci-dessus. Le checkout moteur avait déjà des modifications personnelles,
conservées. Les empreintes de **48 fichiers moteur/binaire** de ce checkout sont
identiques au relevé initial. Le worktree indépendant `qwen27-smallm` a évolué
dans un autre chat ; ses binaires ne servent pas à cette comparaison.

Machine : Apple M5, 24 Go, macOS 26.5, Swift 6.4. Harnais natif commun compilé
avec `swiftc -O -whole-module-optimization -swift-version 6 -warnings-as-errors`.
La version finale du harnais lance réellement `NSApplication.run`, maintient une
fenêtre flottante de 900 × 700 et exige sa visibilité avant/après chaque cas.

Empreintes des exécutables comparés :

| Artefact | SHA-256 |
| --- | --- |
| A | `fefab0bc65e51b3cad10d63f872721b9088ab2e6443a6090e0514734d38e3446` |
| B | `ae1bc37aaf0cef222e7758a4cf5a34669784d41074db77417b091f31a791400a` |
| Moteur inchangé `target/release/mlxl3-rs` | `248a35c3f2e668a4e157b527769b2103ce390d8d93731f6a6f62f5a19c17b2aa` |

Provenance détaillée, variables MLXL3/MLX pertinentes, configurations et métadonnées
des poids : [campaign-sources.json](measurements/app-fluidite-impl-2026-10-16/campaign-sources.json).
Les gros shards sont identifiés par chemin/taille/mtime et ne sont pas rehashés.

Relecture UI : prose/code/réflexion à 64/256/1 024 Kio, plus prose lente 15/s et
rafales à 64 Kio, soit **11 cas**. Après présentation 400 ms et six appends de
warmup : 24 appends mesurés, puis une fenêtre fixe de 180 ms pour laisser finir
la préparation et le fondu. La fin réelle de présentation de chaque pixel n’est
pas instrumentée. Ordre **ABBA + BAAB**, soit huit processus et 88 cas attendus. CPU total
app/main, délais d’arrivée sur le MainActor, durée et empreinte processus sont
relevés. L’empreinte est le maximum échantillonné aux appends, pas un pic
exhaustif. Ces mesures ne sont pas des FPS, une mesure de GPU ou d’énergie.

Génération réelle : même moteur, checkpoint
`Qwen3.6-35B-A3B-EXL3-2.49bpw`, tête `Qwen3.6-35B-A3B-MTP-4bit`, MTP activé,
contexte 4 096, température 0, top_k 1, pénalité 1, MCP désactivé. Une génération
de 48 tokens de warmup puis deux requêtes FR/code de 128 tokens chacune, avec
identifiants de conversation distincts. Même transcript préchargé de 64 Kio dans
les vues de production. Huit processus en ordre ABBA + BAAB, **un modèle GPU à la
fois**, sans compilation/proveur/inférence concurrente. Cible préenregistrée :
régression de débit ≤1 %, résultat non concluant si le bruit empêche de la vérifier.

Le contrôleur vérifie complétude, visibilité, métriques finies/positives, texte
SHA-256 exact, nombres de tokens et compteurs d’acceptation. Il borne chaque
processus à 180 s, nettoie et rejoint son groupe sur erreur. Le bridge de l’app
ne conserve pas les IDs de tokens ; cette campagne ne prouve ni égalité des IDs,
ni parité numérique des logits. Les tok/s viennent des statistiques du moteur,
pas d’une mesure du premier pixel affiché.

Commandes de campagne :

```sh
python3 docs/measurements/app-fluidite-impl-2026-10-16/run-campaign.py ui
python3 docs/measurements/app-fluidite-impl-2026-10-16/run-campaign.py engine
```

Les modes invalides et les statistiques moteur incomplètes sont rejetés par le
harnais final avec rapport `failed` et enfant rejoint. Le contrôleur accepte
les 11 cas natifs complets et rejette **10 rapports mutés** : partiel, dupliqué,
non fini, vide, divergent ou fenêtre invisible. Les rapports restent incomplets
jusqu’à la réussite de tous les cas.

Après collecte, le contrôleur a été durci pour refuser des compteurs MTP absents
ou invalides et tuer tout le groupe même si le parent quitte avant un descendant
ignorant SIGTERM. **28 contrôles** passent : 10 rapports UI invalides, 15 rapports
moteur invalides et trois parcours de nettoyage/interruption. Une copie mutée avec
l’ancien nettoyage laisse survivre le descendant et est rejetée ; le test force
son nettoyage même en échec. Une interruption du véritable contrôleur sur un
faux exécutable jetable produit un rapport `failed`, jamais `complete`.

Commande : `python3 -B docs/measurements/app-fluidite-impl-2026-10-16/check-campaign.py`.
Les 88 cas UI et 16 générations existants ont été revalidés par les gardes finales.
Ces changements de contrôle n’altèrent ni les binaires mesurés ni les intervalles
CPU/GPU ; le contrôleur de collecte est conservé sous `run-campaign-v3-measured.py`.
`py_compile` des deux scripts réussit ; les propriétés Python restent non prouvées
(CrossHair absent). [Contrôles](measurements/app-fluidite-impl-2026-10-16/controller-final-checks.json),
[mutation rejetée](measurements/app-fluidite-impl-2026-10-16/controller-cleanup-mutant.log).

## Essais négatifs conservés

- Dessin initial : l’oracle « chaque glyphe séparément » échoue sur des italiques
  partiellement opaques (max 46 puis 55/255). Tentative de quantification retirée.
  Le nouveau contrat natif ci-dessus est explicite ; cet échec n’est pas effacé.
- Diagnostics scroll : cas exécutés en sandbox non utilisables pour valider les
  fenêtres natives ; les contrôles finaux sont exécutés hors sandbox. Certains
  diagnostics précoces avaient lancé un ancien binaire après compilation refusée,
  sorties non interprétables, conservées et identifiées dans le journal.
- UI v1 : huit passages complets, mais CPU très variable et présentation non
  maîtrisée ; **non concluante**. [Archive](measurements/app-fluidite-impl-2026-10-16/ui-v1/).
- UI v2 : fenêtre non visible, arrêt avant le premier cas ; **bloquée**.
  [Archive](measurements/app-fluidite-impl-2026-10-16/ui-v2/).
- UI v3 première tentative : A complet, puis compilation Swift concurrente d’un
  autre chat détectée avant B ; **interrompue, non concluante**.
  [Archive](measurements/app-fluidite-impl-2026-10-16/ui-v3-interrupted/).

## Résultats et limites

Les **8 passages UI / 88 cas** et les **8 passages moteur / 16 générations**
sont complets. Texte exact, nombres de tokens, compteurs d’acceptation MTP et
visibilité sont conservés. Tous les processus enfants ont été arrêtés et joints.

| Génération de 128 tokens | A médiane (tok/s) | B médiane (tok/s) | Écart observé |
| --- | ---: | ---: | ---: |
| Français / cache LRU | 56,8436 | 56,8482 | +0,008 % |
| Code / merge_intervals | 59,2828 | 59,4105 | +0,215 % |

Aucune baisse des médianes n’est observée. **Ces petits écarts ne sont pas des
gains établis et le seuil ≤1 % n’est pas démontré.** Le premier passage A français
est à 45,98 tok/s, contre 55,63 à 58,09 pour ses trois autres passages ; B est à
56,20 à 57,29. Les quatre écarts appariés FR sont +22,82/−1,31/−1,48/+1,03 % ;
code +6,05/−0,74/−1,41/−5,28 %. Ni l’outlier initial ni la baisse du dernier cas
code B n’ont été supprimés. La variance interdit une conclusion stricte à 1 %.

La médiane du CPU total des 11 relectures par passage est **8,0768 s → 7,9836 s
(−1,15 %)**, un petit écart descriptif qui ne démontre pas une économie. Le
comportement dépend du scénario :

| Relecture | CPU A → B (ms) | Retard d’arrivée p95 A → B (ms) | Empreinte A → B (Mio) |
| --- | ---: | ---: | ---: |
| Prose 64 Kio / 50 ms | 327,12 → 658,13 | 7,32 → 8,16 | 146,79 → 198,23 |
| Prose 64 Kio / 15 appends/s | 834,72 → 757,42 | 22,72 → 8,20 | 168,57 → 187,29 |
| Prose 64 Kio / rafales | 471,81 → 531,16 | 11,19 → 10,67 | 159,45 → 165,19 |
| Code 64 Kio / 50 ms | 404,21 → 390,55 | 6,76 → 7,06 | 114,33 → 123,91 |
| Réflexion 64 Kio / 50 ms | 866,88 → 860,08 | 6,00 → 8,82 | 256,09 → 252,13 |
| Prose 256 Kio / 50 ms | 805,69 → 602,17 | 7,94 → 38,81 | 219,04 → 175,02 |
| Code 256 Kio / 50 ms | 477,38 → 463,20 | 4,77 → 5,49 | 159,71 → 145,02 |
| Réflexion 256 Kio / 50 ms | 948,05 → 946,52 | 9,89 → 9,62 | 362,09 → 332,92 |
| Prose 1 Mio / 50 ms | 877,61 → 743,65 | 9,66 → 8,38 | 192,93 → 201,06 |
| Code 1 Mio / 50 ms | 1 204,72 → 1 168,76 | 12,85 → 13,21 | 153,13 → 151,59 |
| Réflexion 1 Mio / 50 ms | 866,91 → 823,94 | 5,99 → 8,44 | 279,77 → 274,01 |

Chaque cellule est une médiane de quatre passages. **Le cas prose 64 Kio / 50 ms
augmente de 0,331 s CPU (+101 %) et de 51,44 Mio**, avec des plages CPU disjointes
A 0,279–0,442 s / B 0,553–0,675 s. Le p95 d’arrivée sur le MainActor augmente aussi
de 30,87 ms pour la prose 256 Kio. Ces coûts sont réels dans cette relecture et
restent visibles ; les contributions isolées du scroll, du fondu et du dessin
n’ont pas été séparées. Le candidat n’est donc pas qualifié de gratuit en CPU,
mémoire ou latence UI. Le dessin groupé reste une réduction du nombre d’appels
de dessin, pas un gain isolé mesuré.

Dans les deux générations réelles, le CPU app médian est 1,104 → 1,013 s (FR) et
1,114 → 1,037 s (code), avec des plages qui se chevauchent. L’empreinte app est
170,90 → 167,84 Mio et 143,27 → 149,20 Mio ; il ne s’agit pas de la RAM du moteur
ni de la mémoire GPU allouée. TTFT moteur médian : 160,95 → 158,23 ms et
161,31 → 160,70 ms, sans garantie sur le premier texte effectivement affiché.

Conditions : batterie en décharge **38 → 34 %**, swap utilisé
**4 107,81–6 651,69 M** selon les 16 relevés `sysctl vm.swapusage` (unité
rapportée par cet outil). `pmset` ne signale aucune alerte, mais température,
fréquence et pression thermique réelle ne sont pas mesurées. La machine était
réservée sans compilateur, prouveur ni autre modèle actif ; l’autre chat a repris
ses mesures seulement après libération explicite.

Preuves : [résumé numérique et plages](measurements/app-fluidite-impl-2026-10-16/performance-summary.json),
[88 cas UI](measurements/app-fluidite-impl-2026-10-16/ui-campaign.json),
[16 générations](measurements/app-fluidite-impl-2026-10-16/engine-campaign.json),
[contrôles négatifs](measurements/app-fluidite-impl-2026-10-16/controller-negative-checks.json).
Les rapports par passage et relevés de conditions se trouvent dans le même dossier.
Aucun essai n’a été relancé pour obtenir un résultat favorable.


La CI existante `.github/workflows/check.yml` exécute `scripts/check-e2e.sh` sur
les PR et les pushes, dont ces nouvelles régressions. La CI Desktop du SHA de
référence a été inspectée et réussit ; aucune CI distante du candidat local
non committé n’existe. Les gestes physiques de lecture/reprise, le coût énergie,
les FPS et les autres modèles/contexte long restent non mesurés. L’app installée
n’a pas été remplacée.

Le modèle utilisateur `Qwen3.8-27B-exl3-6a9ca9d0` a été rechargé dans l’instance
initiale `dist/MLXL3 Desktop.app` après les deux campagnes. L’interface confirme
**Ready on Metal**, contexte **128 386**, moteur 1.4.7 déjà présent. L’instance
supplémentaire ouverte pour inspection a été fermée ; un seul bridge natif reste,
celui de l’utilisateur. Aucun message de conversation n’a été envoyé ni supprimé.
Ces opérations de runtime n’installent pas le candidat UI et ne remplacent aucun
binaire de l’app.
