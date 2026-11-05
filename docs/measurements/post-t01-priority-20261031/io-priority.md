# Priorité I/O après T01 — lecture du 31 octobre 2026

Préenregistrement `POST-T01-PROFILE-01`, lecture seule et calculs sur les traces existantes. Aucun build, test, proveur, GPU, benchmark ou nouveau prototype. **T18a est un petit port plausible, pas le budget dominant du profil ni une promesse de 15 tok/s.** T18b et le rollback Sushi sont plus risqués pour un bénéfice exposé moins directement établi. Le budget experts reste 3 Gio.

## Ce qui bloque réellement

La [fenêtre courte qualifiée](../ultra-profile-20261027/attribution/README.md) contient 20 pas AR MAIN : bloc lectures 38,108 ms/token, union des appels de lecture 37,211 ms, commandes GPU moteur concomitantes 6,685 ms. Sa queue après la dernière activité GPU indépendante vaut **31,409 ms**, dont 30,862 ms d'union de `preadv`. Sur les 2 619 lectures, 244 finissent avant cette frontière, 1 876 la traversent et 499 commencent après. La fenêtre exploratoire de 36 pas donne 27,215/26,988 ms ; sa campagne a été interrompue après la phase validée et ne devient pas une seconde campagne complète conforme.

Cette frontière décrit une attente exposée aux dépendances dans l'ordre actuel des queues, **pas la criticité causale de chaque lecture**. Les jobs de 1 248 012 octets logiquement retournés concordent avec les syscalls ; aucun capteur bloc n'a qualifié le trafic physique, la latence NAND ou la contribution du cache fichier macOS. Ne pas qualifier ces 27–31 ms de temps SSD récupérable. Le GPU moteur peut être actif tout en subissant des stalls.

L'attente routes est un autre poste : 40,056 ms d'enveloppe, dont 39,947 ms de commandes GPU moteur et seulement environ 0,106 ms sans activité GPU observée ; copie des IDs 0,00565 ms. Enlever l'attente ne retire pas ses producteurs. Les 12,142 ms de CPU actif/GPU idle dans les soumissions, et les 22,150 ms sur tout le decode, ne sont pas des budgets I/O additionnels. L'observateur pèse : 113,755 ms/token tracés contre 91,671 ms au contrôle.

## T18a : borne de déséquilibre, distincte de la queue complète

La [source actuelle](../../../native/ffi/disk_reader.h) attribue `i=worker; i+=workers` puis rejoint tous les callbacks GCD. [`fill_many`](../../../native/src/ssd_moe.rs) attend ce retour ; le remplissage est ensuite commis, puis Cold construit. Les erreurs sont mémorisées par job, tous les jobs drainés, puis la première erreur dans l'ordre des indices est rejouée. Il existe donc une différence falsifiable avec une attribution dynamique, sans changer kernels, requêtes, banques ni soumissions.

Recalcul indépendant sur les **traces HOLD conservées**, une trace par mode et 127 transitions/sorties de decode (les pas MTP/DF sans nouveau calcul cible restent dans ce dénominateur). Chaque job est unique, fini sans erreur, retourne 1 248 012 octets et appartient au worker `job % min(4,n)` ; tous ces contrôles passent dans les données. Ces résultats ne sont pas les traces MAIN auxquelles #59 attribue 26,69/60,84/61,89 % et 2,31/4,12/3,81 ms : ne pas fusionner les jeux.

Pour un lot, `d_i=end_i-start_i`, `L=max(max(d_i),sum(d_i)/min(4,n))`. `max(sum(d_i par worker))-L` est une borne idéale du déséquilibre de charge ; `durée_lot-L` ajoute aussi départ GCD, trous et retour. Les durées sont artificiellement figées : une nouvelle concurrence peut les changer et une file FIFO ne réalise pas nécessairement l'optimum.

| HOLD, trace 1 | AR | MTP | DFlash |
| --- | ---: | ---: | ---: |
| Lots / jobs | 5 475 / 19 694 | 3 672 / 22 138 | 3 293 / 20 715 |
| Lots >4 jobs | 29,55 % | 60,97 % | 63,56 % |
| Bloc lectures, ms/sortie | 30,651 | 28,060 | 26,816 |
| Déséquilibre idéal seul, ms/sortie | **2,490** | **4,117** | **4,025** |
| Lot moins L, ms/sortie | 3,313 | 4,629 | 4,467 |
| p95 du lot / maximum, µs | 1 424 / 6 456 | 1 959 / 5 544 | 2 036 / 4 789 |

En AR HOLD, les 48 couches ont 80–800 jobs et 0,219–0,949 ms de bloc par sortie. Les plus grandes contributions à `lot-L` sont couches 0/1/7/22/6 : 0,190/0,161/0,156/0,155/0,135 ms par sortie. Le dernier job appartient aux workers 0/1/2/3 dans 2 382/1 644/939/510 lots : **ce n'est pas la preuve d'un cœur lent**, car le striping donne plus de jobs aux premiers indices lorsque n n'est pas divisible par quatre.

Les deux fenêtres AR MAIN avec calage CPU/GPU permettent aussi de borner la portion exposée. Appariement exact des lots par ordre/durée/nombre de jobs et décalage constant : 816 lots/2 619 jobs puis 1 513/5 244. Le déséquilibre seul vaut 2,462 puis 2,265 ms/token. La somme `min(durée_lot-L, queue_exposée_du_lot)` vaut **3,137 puis 2,921 ms/token** : bornes optimistes incluant du slack, à durées et dépendances figées. Leur effacement total correspondrait à environ **+2,84/+2,91 % dans ces fenêtres instrumentées**, sans prédiction release. Ce n'est pas +35–38 % ; il reste notamment le plus long job obligatoire de chaque lot.

## Ce qui a déjà été filtré

- [14/14B, journal](../../../opti.md) : groupement des records et 1/2/4 lecteurs déjà exercés ; T18a est l'attribution dynamique à **quatre lecteurs inchangés**, pas une nouvelle revendication de cette ancienne parallélisation.
- [Pool global/ARC](../issue58-20261029/README.md) : replay exact de 60 912 appels/609 120 paires ; ARC ajoute 5,78–6,95 % de chargements MAIN et 2,30–2,65 % HOLD. Le vrai runtime qualifié donne −3,48/−2,15 % decode ; banques partagées LFU seules +0,58/+0,75 %, non établis. Audit réalisé avant retrait, surcoût CPU ARC linéaire environ 0,55 ms/token ; O(1) non mesuré. Ne pas rejouer ces politiques inchangées. Les allocations statiques 47/55 sont aussi anciennes : 55 réduit les jobs MAIN mais les augmente HOLD, rejet avant GPU ; [reçu](../flash-next-ssd-impl/opt-55-cache53/closure.json).
- [Sushi specRoute/specMatches/GDN différé](../sushi-20261027/README.md) : GPU Hot partiel, partagé et événement précoce des IDs existent déjà. Le mécanisme amont tout-Hot exige les mêmes IDs/slots ; sinon restauration et reconstruction. Les préfixes éligibles locaux totalisent 2,37–2,72 ms, mais seulement **0,380–0,469 ms de GPU idle** : effacement idéal inférieur à 0,5 % dans ces fenêtres. GDN/conv/HC/PLE/injection/initialisation et cache doivent rester transactionnels ; la garde amont sans PLE et seq=1 ne couvre pas directement MTP/DF. Aucune raison de lui attribuer 38 ms.

## Choix et validation minimale avant tout chronométrage

**T18a est le premier candidat I/O raisonnable**, reprenant sélectivement le port PR61 `a36ba5077bf386466660d6b478058fbe6880245c` : `DiskWorkQueue.claim`, compteur atomique relaxed pour attribuer les indices ; la publication reste assurée par la jointure synchrone. Les tests portables et sanitizers cloud archivés ne qualifient pas GCD, `preadv` réel vers Metal ni le débit M5. [Patch primaire archivé](../issues-triage-20261030/pr-61/changes.patch), régions 2870–2953 ; sa concaténation sans noms de fichiers n'est pas directement applicable comme patch Git. T18a n'est pas porté localement.

Ce choix a moins de risque numérique et de coût d'encodage nouveau que T18b ; **il ne domine pas le budget CPU/GPU**, qui mérite l'attribution parallèle. Une cible de kernel/encodeur avec un coût exposé effectivement identifié peut passer devant. T01-P ne ferme pas T01B. Ni une file dynamique ni une nouvelle fusion ne justifie seule l'objectif 15 tok/s.

Avant vitesse : réutiliser PR61 sans importer T17/T04 ; préserver exactly-once, bornes 1..4/1..512, destinations disjointes, lecture complète/EINTR/troncature, drainage même en erreur et ordre de l'erreur, compteurs worker réels plutôt qu'hypothèse modulo. Pas d'appel MLX depuis les workers. Tenter la vérification C++ réelle applicable avec ses limites, sanitizers et mutation pertinente, puis vrai test I/O Metal, Hot/Cold/overlap/pending/cohortes/échec/reprise, oracles logits/états, MTP/DF et bridge. Même binaire flag0/1, cache 3 Gio/53×48, quatre lecteurs, points de soumission inchangés ; enfin pilote release court alterné MAIN/HOLD, fin/p95 des lots et débit complet. Auditer l'adaptation avant rejet.

**T18b reste conditionnel** : publication par expert seulement après ses douze segments complets, banques et générations protégées tant que GPU en vol, contributions aux positions canoniques, somme stock inchangée, erreurs/annulation/pending exacts. Le premier→dernier retour vaut 15,96/19,07/18,36 ms/sortie dans HOLD ; ce n'est ni une fenêtre entièrement GPU idle ni un gain. Sous-cohortes Cold impliquent allocations/graphes/soumissions supplémentaires précisément dans un poste CPU déjà exposé. Attendre le filtre T18a avant ce changement d'architecture.

## Identités des données inspectées

SHA-256 complets, sans nouvelle acquisition :

| Artefact | SHA-256 |
| --- | --- |
| `ultra-profile-20261027/natural/AR-HOLD/profile/trace_1-decode.json` | `b0cff608659a4bc9065ed7f52e09f950cde08669fdfbe9c5c9bc468cf2234bb1` |
| même chemin `MTP-HOLD` | `940b04602b0ef679cdc8ab8bf490a6698809d64601b77eca11caae51752e044b` |
| même chemin `DF-HOLD` | `e4cc0cf61c3b77e0c8cc6acd305f0af4e597713af7d4274aa26e595d9e613c95` |
| `attribution/short/AR-MAIN/profile/trace_1-decode.json` | `49b0a551b3608c003b4c5e5dfc48b7903a3a056d728320f09e5f466f30201298` |
| `attribution/short/AR-MAIN/io-tail-batches.json` | `66ffc740d09a80e2303f896f0ab01b65a84b993078326b253fb5d83968313d80` |
| `attribution/AR-MAIN/profile/trace_1-decode.json` | `6de2cf687927b4322e79d5aa88ff59a157562afefce96205fe55a522e9aa9c6f` |
| `attribution/AR-MAIN/io-tail-batches.json` | `c8bdda717b63dca8ca00abe69e4454dbb0fb40fed2883a54f26e74ae7764a037` |
| PR61 `changes.patch` | `b4266272f1a71aa846fc2a8630d8929f2d85efda3a5b2e6527480770f5fdc766` |
| source actuelle `native/ffi/disk_reader.h` | `c191ee3c887010df2b007f5a0d68c9ae01d7d080cd77cb897518626e2ef355f8` |

Les chemins `attribution/...` sont relatifs à `ultra-profile-20261027/`. Les snapshots [issue59](issue59-current.md) et [issue60](issue60-current.md) motivent cette lecture ; leurs chiffres historiques restent distincts des recalculs HOLD ci-dessus. Les traces MAIN naturelles initiales attendues ne sont pas disponibles sous leurs noms historiques dans ce checkout ; aucune acquisition n'a été relancée pour les recréer. Les datasets d'attribution MAIN présents ont été utilisés et identifiés séparément.
