# Après DEC-ONE : recherche ciblée dans le backend MLX

1 novembre 2026. Recherche terminée ; aucune optimisation implémentée ou
mesurée dans cette étape. DEC-ONE est abandonné à la demande de l'utilisateur,
reste OFF et conserve ses preuves. Dynamic+ColdU reste ON. Moteur et application
inchangés. HEAD local `d5a776e53425bb4d0ba61c0774c5399ae7987c06`, avec les
modifications locales préexistantes conservées ; M5 24 Go, MLX 0.32.2, experts
3 Gio. [Sources épinglées](sources.json), [extraction et empreintes](evidence.json).

## Premier filtre recommandé : coût de conservation des buffers par opération

Dans le véritable backend MLX 0.32.2, `gpu::eval` crée pour chaque primitive
un `unordered_set<shared_ptr<array::Data>>`, y place ses entrées et sorties
sœurs, exclut la sortie principale donnée par donation, puis attache cette
collection à un callback de fin Metal. En cas de commit automatique, la
collection est capturée par le callback de commit. Un pool autorelease existe
aussi par primitive. Ce mécanisme reste présent dans le main MLX consulté.
Les deux versions de `eval.cpp` sont conservées ici en fichiers texte.

**Hypothèse distincte du lanceur prepared : regrouper le keepalive et les
callbacks au niveau du command buffer.** Conserver les références nécessaires
jusqu'à la même complétion GPU, mais éviter de multiplier les petits objets,
allocations et captures. Les kernels, grilles, arrondis, événements précoces
de routes, frontières de commit et budget experts restent identiques.
T01B accélérait seulement la préparation d'un CustomKernel et ajoutait un
wrapper ; il ne supprimait pas ce travail répété de `gpu::eval`.

Ce n'est pas un patch amont découvert et déjà validé. C'est une hypothèse locale
issue du croisement du code amont et de nos piles. Elle nécessite un build MLX
isolé de la même révision, sélectionnable par le `MLXL3_MLX_ROOT` existant ;
aucune mise à jour globale du package ou des shaders n'est nécessaire au filtre.

### Ce que le profilage permet réellement de dire

P09, fenêtre commune 87–113, 27 tokens, avant Dynamic+ColdU :

| Échantillons CPU actif/GPU idle dans host_async_submit | AR | MTP | DFlash |
|---|---:|---:|---:|
| Total dans ce scope | 329 | 258 | 233 |
| Première frame MLX = gpu::eval | 23 | 25 | 14 |
| Présence de Block_copy dans la pile | 10 | 5 | 3 |
| Présence de unordered_set dans la pile | 8 | 4 | 1 |
| Présence de __hash_table dans la pile | 52 | 27 | 26 |

Malloc/free et captures apparaissent parmi les feuilles des piles dont la
première frame MLX est `gpu::eval`. Aucun échantillon ne porte explicitement
le nom `addCompletedHandler`. Les catégories se chevauchent ; les hash tables
servent aussi au graphe et aux barrières. **Ces comptes ne sont ni une durée
de callback ni une preuve qu'il domine.** Le plafond observé de toute la
soumission CPU exposée est 12,433 / 9,323 / 7,940 ms/token selon le mode ;
le plafond propre au keepalive n'est pas mesuré et ne peut reprendre ces totaux.
La capture ajoute du coût et ne représente pas directement le débit release.

### Contrôle discriminant avant un port plus large

Préenregistrer séparément le prototype et son filtre avant toute édition ou
exécution : même MLX 0.32.2, deux builds isolés, uniquement la rétention modifiée.
D'abord compter les primitives, callbacks, références et commits sur les chaînes
Cold réelles ; mesurer construction, soumission et attente séparément après
correction. Ne pas attribuer une baisse de temps GPU à la gestion des callbacks.
Si le coût mesuré est minuscule, conserver ce constat et éviter un port large.

Contrats à vérifier : pas de libération/réutilisation avant la complétion ;
plusieurs buffers en vol et graphes simultanément vivants ; alias/vues et
donation ; commits automatiques, finalize, erreurs, annulation et destruction.
Un lot global ne doit jamais effacer une référence requise par une primitive
antérieure sous prétexte que la primitive courante donne sa sortie par donation.
Le nombre de références peut influer sur la donation : le nouveau conteneur ne
doit pas être supposé neutre sans contrôle. Garder les handlers d'erreur,
événements et fences, ainsi que le suivi de hazards. Ne pas supprimer le pool
autorelease simultanément : cela brouillerait l'attribution et pourrait gonfler
la mémoire. La RSS et la durée de vie des temporaires restent à mesurer.

Ensuite seulement : oracles composants/logits/états, vraie entrée CLI/bridge et
écran court alterné AR/MTP/DFlash, mêmes prompts/3 Gio/4 lecteurs/Dynamic+ColdU,
avec prefill contrôlé. Aucun changement numérique ne doit être accepté sur la
seule égalité du texte. Les gardes et l'ancre déjà autorisées restent applicables.
Ce plan n'a pas été exécuté et n'est pas une activation par défaut.

## Classement après lecture des sources et des essais antérieurs

| Rang | Piste | Enveloppe et intérêt | Risque / décision |
|---|---|---|---|
| 1 | Keepalive et callbacks groupés par command buffer | Travail CPU répété identifié dans MLX et allocations visibles ; gain propre non mesuré | Durée de vie, donation, erreurs ; filtre local avant port moteur |
| 2 | Une primitive MLX pour une chaîne EXL3 complète, avec les dispatchs canoniques inchangés | Réduirait davantage la traversée et la gestion par nœud ; partage la même enveloppe CPU, non additionnable | Plus intrusif : temporaires, allocation, strides, dépendances et donation ; ne pas confondre avec replay gratuit ou simple mx::compile |
| 3 | Top10 SIMD512 archivé | Travail de classement quadratique réel ; coût GPU isolé non mesuré | OPT28B unique paire AR MAIN −0,474 %, campagne interrompue ; SIMD01 ultérieur préparé mais jamais qualifié GPU, pas une découverte neuve ni un gain établi |

Le rang 1 est le meilleur prochain **filtre de coût/risque**, pas une garantie
de plus grand gain possible. Aucun pourcentage positif n'est mesuré ici.
Le rang 2 ne réduit pas nécessairement les dispatchs ni les barrières Metal :
son bénéfice doit être attribué au travail CPU effectivement supprimé. Les
limites du lanceur T01B et les résultats négatifs DEC-ONE restent conservés.

## Ce que l'internet ne justifie pas de recopier

- **Sushi specRoute/specMatches et deferred GDN** : GPU Hot couvre déjà une
  partie du mécanisme. Le diagnostic historique SUSHI-SYNC trouvait seulement
  11,63–11,81 % de cohortes toutes Hot ; parmi les transitions admissibles vers
  GDN sans PLE, environ 86 % étaient des misses. L'attente conditionnelle des
  succès admissibles était 4,507–4,563 ms/pas, inclusive et non exclusive GPU.
  Le coût de calcul jeté/rollback manque ; les chiffres amont ne se transfèrent
  pas au cache 3 Gio. Aucune nouvelle exécution ni extrapolation à Dynamic.
- **Strata et lectures contiguës** : nos 12 segments de destination ne sont
  pas 12 appels disque. `disk_reader.h` emploie déjà `preadv` depuis un record
  contigu directement vers les banques. Le gain amont « trois lectures vers
  une » ne décrit donc pas notre différence. Les variantes ROUTE/PARTIAL sont
  elles-mêmes documentées sans gain, avec routes périmées sur M1 pour ROUTE.
- **MLX PR4633** : renouvelle la résidence après plusieurs secondes GPU idle.
  Elle concerne surtout la reprise d'une requête ; elle ne démontre pas un
  gain de decode continu. Le wired limit local avait été mesuré à zéro dans
  OPT52D, sans nouvelle sonde ici. Pas de changement de politique mémoire.
- **Thread unsafe stream / FAST_SYNCH** : le premier rend un stream utilisable
  depuis d'autres threads ; le chemin normal consulte déjà un encodeur local
  au thread. Le second modifie les fences inter-streams/devices et ne remplace
  pas l'attente hôte `Event::wait` des routes. Ni l'un ni l'autre ne prouve une
  suppression de nos attentes CPU/GPU. Pas d'activation.
- **mx::compile ou command buffer unique** : ne suppriment pas automatiquement
  l'encodage backend. Strata a une architecture Metal native ; les lectures SSD
  dépendantes des routes chez nous empêchent de transposer aveuglément un seul
  command buffer pour toutes les couches. L'issue MLX3789 est une proposition,
  pas une validation de gains équivalents sur ce moteur Rust.
- **MLX main / nouvelles voies affine ou int8** : EXL3 n'est pas le format
  affine Q4. Une mise à jour globale mélangerait changements de dispatch et
  corrections numériques ; aucun gain de ces PR ne valide notre contrat exact.

## Vérification de la recherche

Lecture de `opti.md`, des rapports P09, T01B, Dynamic, OPT28/SIMD01,
SUSHI-SYNC, et des fonctions natives concernées. Extraction JSON hors GPU avec
Python standard : 27 steps vérifiés dans chacune des trois fenêtres, scopes
et piles non vides, 13 empreintes revérifiées identiques, cinq artefacts non
vides et contrôle de whitespace réussi. Aucune suite moteur,
preuve Kani/CBMC, mesure GPU, compilation, CI distante ou publication lancée :
seuls des documents et snapshots texte ont été ajoutés. La qualité et la
vitesse du candidat sont **non vérifiées / non mesurées**. Aucun essai en cours.
