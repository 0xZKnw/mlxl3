# Revue indépendante F06 — qualité v1/v2

Date UTC : 2026-10-24T23:50:52.494491+00:00. Lecture des sources, journaux, plans et empreintes uniquement. Aucun build, test, prouveur, chargement de modèle ou benchmark lancé par cette revue.

Les contrôles F06 terminés ne révèlent aucun nouveau défaut numérique dans le domaine testé. **La qualité v2 passe ; la mémoire 77B ne passe pas ; le bridge et le débit ne sont pas qualifiés par cette revue.**

## Identité et état du runner

- Sources : base `62cae0b3a51db760185d65eee61f1da79c9b19c7` plus F06 local, F01 absent du code compilé ; les **85 empreintes** de sources et les **8 pilotes** correspondent au gel v2. Ce n’est pas une affirmation de binaire issu du seul commit propre.
- CLI : `cbc03630aa08fd8dcdb949b905896ec1dc7f37888a7e984059c12624ec6743e7` ; tests : `f83049e4e77202ef01f7d96b5ed24bc14b8897ab992db501eaca06eb2d927431`.
- Plan v2 : `773bb1fd65b115241072236b84fd04e2ee0a5df503c004242dda3b4a5f29e5cf`, identique au `matrix_sha256` du résultat.
- **V1 reste `failed_or_interrupted`** : la suite MLX a réellement terminé avec 130 tests passés, 0 échec, 150 ignorés, exit0. Le runner a rejeté ses comptes parce que l’ajout de suite conservait les métadonnées d’un test unique. Deux processus seulement exécutés, tous joints, sampler joint ; post30 non exécuté. V2 ne répète pas cette suite déjà passée.
- **V2 : 19/19 étapes attendues**, dont 9 refus CLI exit2, 8 invocations physiques exactement sélectionnées, pré20/post30. Chacune est jointe. Chaque invocation native affiche 1 test passé, 0 échec, 0 ignoré, 279 filtrés. Sampler joint : 59 observations sans erreur ; courbe VM : 39 échantillons.

## Couverture inspectée dans le code et les sorties

| Contrôle | Résultat et portée |
| --- | --- |
| Sélecteur indépendant | 256 cas : B1/B2/B4/B8 × F16/F32 × contigu/stridé × 8 fixtures × normalisation0/1. Oracle CPU de rang indépendant ; IDs des voies historique/0/1 exacts et scores aux mêmes octets. Ties, NaN, infinis, ±0 et valeurs limites exercés. Les fixtures intentionnellement non finies ne sont pas présentées comme finies. |
| Métadonnées et replis | 5 rejets de forme/dtype, 6 géométries hors domaine inchangées, reprise valide non vide. |
| Routes réelles | 384 entrées : 48 couches × 2 captures × F16/F32 × contigu/stridé ; chaque entrée exerce 0 puis1 (768 appels nouveaux). Probabilités et scores finis, IDs/scores exacts face au routeur historique. Chaque entrée réelle est une ligne ; les formes multi-lignes sont couvertes par le composant. |
| Chargement | 3 processus distincts : variable absente (explicitement retirée par env-u),0,1 ; champ false,false,true et route non vide. Aucune mutation d’environnement après initialisation MLX. Lecture unique du champ vérifiée dans le code ; pas de revendication de test dynamique après chargement. |
| CLI invalide | 8 valeurs ASCII invalides et byte0xff rejetés, exit2 et message attendu. |
| Préfixes DFlash | 36 couples largeur/rétention B1–B8, logits248320 et captures/états exacts face au séquentiel, vérification du pas suivant et erreurs/abandon/reset. Capture NumPy : tolérance0,004, résultat observé byte-equal sur64000 valeurs. **Les deux voies cible utilisent le même flag1 : ce n’est pas une comparaison intégrale cible flag0/1.** |
| MTP | Budgets1/2/3/17 ; propositions0/1/2/12, acceptations0/0/0/5. Séquence greedy et états cible/tête face au séquentiel, invalides sans mutation, alignement/budget/pending vérifiés. Acceptation et rejet réellement parcourus, sous flag1 commun. |

La mutation MSL `(tid+1)%512` est détectée au premier cas par l’oracle CPU indépendant : exit101, 0 passé/1 échec, `rows=1 fixture=0`. Elle corrompt aussi le chemin historique ; cette indépendance évite une fausse parité communément corrompue. Le composant normal release passe avant mutation ; le mutant est un binaire debug jetable, sans mesure de débit. La mutation CPU E511 échoue aussi ; test et Kani passent après restauration.

## Vérification source et limites formelles

Les journaux CPU montrent117 tests lib passés/6 ignorés plus16 tests d’intégration. Les échecs Clippy test-only v1/v2 sont conservés ; Clippy MLX v3 passe. Kani sur le vrai prédicat d’admission symbolique (`i32`, `usize`, `bool`) valide5 obligations et satisfait4 couvertures sans `assume` ; fonction sans boucle. Cela prouve uniquement le confinement de la géométrie demandée.

La tentative Kani sur l’entrée MLX réelle échoue avec une erreur interne `intrinsics.rs:243:17`, exit1 après4,020s, pic groupe684474368B, sous les limites180s/4Gio. **MLX/Metal, l’allocateur, les courses et l’inférence complète restent non prouvés.** Aucun stub substitué. Aucun run CI futur F06 inspecté ici.

## Mémoire : faits distincts

Les19 phases et les39 observations VM conservent les compteurs système : Swapins53329920B et Swapouts475791360B, donc **0 nouvel in/out global observé**, pré/post compris. AC observé à chaque frontière. La compression système passe de3780296704B à6554910720B, pic échantillonné8007434240B ; ce compteur système ne donne pas le propriétaire des pages.

Le seul snapshot propre au processus préfixes (PID90130/start1391950878349) rapporte :

| Mesure | Octets |
| --- | ---: |
| Empreinte courante | 7358830712 |
| Pic d’empreinte vie du processus | 7395956880 |
| MLX actif / cache / pic alloué | 6460060960 / 67109168 / 6783453948 |
| Compression logique courante / pic | 3358900224 / 3360112640 |
| Compression logique cumulée vie | 3396861952 |
| Ledger swap-ins cumulés observés | 0 |
| Décompressions cumulées | 198476 |

Le pic d’empreinte observé reste sous12Go. **Les plafonds77B de512Mio pour la compression courante et le pic sont dépassés**, de2822029312B et2823241728B. Ces plafonds restent inchangés et échoués. Le plafond de delta compression vie1Gio ne peut pas être évalué avec un seul snapshot ; le cumul n’est pas un delta. Le ledger cumulatif observé0 n’établit pas l’absence de swap-out privé et n’est pas une série temporelle.

Le processus QA charge un oracle, garde des captures et logits CPU et écrit des états dans des fichiers temporaires. Sa pression ne représente pas les futurs processus de débit. Cette structure ne prouve pas la cause exclusive de la compression : aucune décomposition n’a été mesurée. Empreinte, pic d’empreinte, mémoire MLX allouée et `ru_maxrss` d’enfants restent des métriques différentes. **F06 ne prétend pas corriger la mémoire.** Le bridge réel puis les mesures HC/CH auront leurs propres observations ; succès numérique, critère F06 sous12Go/zéro global et anciennes limites77B doivent rester distincts.

Le bridge43 est lancé par le parent mais hors de ce gel qualité. Aucune mesure tok/s ni accélération n’est revendiquée. Les preuves et empreintes détaillées sont dans [cross-review-quality.json](/Users/justin/Documents/mix-stq1_0/build/flash-opt-82/F06/cross-review-quality.json).
