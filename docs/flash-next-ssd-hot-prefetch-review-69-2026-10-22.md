# 69R — prélecture et GPUHot, revue statique

Lecture réalisée pendant la qualité68, sans modifier les155sources gelées,
sans nouveau modèle, compilation, preuve ou benchmark69. Les mesures68 ont
LES/lookahead désactivés : les constats ci-dessous concernent un futur mode
prélecture et n'expliquent pas directement leurs latences actuelles.

## Réutilisation des lectures encore en vol

Dans ReaderPool, `poll` publie les lectures terminées et `note_late` compte
les experts nécessaires encore en vol. Il ne les attend ni ne les promeut.
ExpertCache::reserve ne cherche l'expert que dans `keys`, alors qu'un record
en cours reste dans `pending`. La demande peut donc réserver un autre slot,
le remplir par une deuxième lecture, puis rejeter la publication de la
première comme doublon. Le test historique
`pending_records_never_hit_alias_or_publish_over_a_demand` décrit ce choix
de sécurité : les destinations n'aliasent pas et un seul expert est publié.

[Strata](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/stream.cpp)
retrouve aussi les experts en vol via `where_`, les épingle avant de réserver
les manquants, puis déplace leurs morceaux encore en queue vers `urgent_q`.
Les lecteurs terminent les morceaux commencés dans leur slot initial.

[Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/ExpertStore.cpp)
conserve les places de staging encore utilisées. Pour un expert demandé,
il reprend les fragments terminés et relit les morceaux non pris ou échoués.
Ce mécanisme ne se transpose pas directement à nos records contigus lus
dans12segments des banques définitives. Un nouveau repackingSSD n'est pas
nécessaire pour étudier leur réutilisation.

Fréquence des doublons, octets évitables et attente réellement exposée :
**non mesurés** dans cette revue. Une attente de prélecture pourrait aussi
coûter plus qu'une nouvelle lecture urgente : ce n'est pas un gain garanti.

## Compatibilité du Hot anticipé

Deux gardes `prefetcher.is_none()` et `depth==0` excluent le chemin Hot
anticipé lorsque la prélecture est active. Le Hot tardif, choisi après le
readback CPU, recouvre déjà des lecturesCold ; l'amélioration à chercher est
le démarrage plus précoce et la réduction des attentes restantes.

Le snapshot expert→slot exclut les destinations `pending`. Les publications
restent sur le thread d'inférence et chaque lecture conserve son lease
jusqu'à complétion/jonction. Ces protections sont des bases utiles, mais
retirer les gardes sans contrôler le chemin de réservation serait insuffisant.
Une complétion entre snapshot et readback doit être calculée dans la partie
Cold si elle n'était pas dans le snapshot, sans contribution manquante ou double.

### Appelant DFlash2 : prédictions absentes pendant la vérification

Le traçage complémentaire du véritable appelant précise la portée de ce
problème. `flash_draft.rs::forward_captured` transmet le MoE suivant et la
profondeur à `forward_lookahead`. En revanche, `verify_draft` parcourt les
couches avec `iter_mut` et appelle `forward_draft_rows`, qui transmet
`next=None, depth=0`. Il ne produit donc pas de prédictions pour la couche
suivante pendant cette vérification. Cela n'exclut pas les prélectures
provenant du préfill ou d'un autre passage séquentiel.

Activer la profondeur de lookahead crée néanmoins le ReaderPool sur chaque
couche, ce qui rend `prefetcher.is_none()` faux : le Hot anticipé et la
réunification conditionnelle de fenêtres y sont alors désactivés. Un futur
test DFlash2 doit vérifier ensemble l'appelant, l'émission effective des
prédictions et la conservation du Hot, plutôt que comparer deux options dont
une n'atteint pas la vérification. Fréquence et coût de cette interaction :
non mesurés ; les mesures68 avec lookahead0 ne la démontrent pas.

## Prochain test discriminant proposé

1. Relever les demandes qui rencontrent le même expert encore en vol, leur
   attente, leurs octets et la fréquence de seconde réservation, sans `eval`
   supplémentaire de profilage.
2. Vérifier qu'une demande peut réutiliser le record en vol, y compris lecture
   échouée, destination protégée, interruption et publication périmée. Garder
   l'oracle numérique et les compteurs de fréquence du cache.
3. Étudier ensuite le Hot anticipé avec ce chemin, d'abord surB2. Isoler
   prédiction, politiqueLFU/LRU et démarrageHot : ils changent aujourd'hui
   ensemble lorsqu'on activeLES.

Les cribles39A interrompus et les diagnostics41 ne qualifient pas un gain
LES actuel. Les timings instrumentés ne s'additionnent pas et le résultat
du shared68 ne doit pas être utilisé comme témoin d'un futur binaire69.

État final69R : revue statique terminée, **aucun prototype ni tok/s69**.
Sources amont épinglées et SHA dans
[research-source.json](measurements/flash-next-ssd-impl/opt-68-shared-expert/research-source.json).
