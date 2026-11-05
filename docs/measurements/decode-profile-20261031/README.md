# Décode Flash : attribution P08 et limites

31 octobre 2026. **Analyse de captures partielles terminée ; campagne GPU interrompue. Aucun nouveau gain moteur validé.**

Le poste CPU de soumission persiste après suppression d'un biais important du
harnais : environ **11,85 ms par sortie pendant lesquelles le CPU travaille et
aucune commande GPU n'est active**. Les piles désignent surtout l'encodage
Metal, le suivi des ressources, les allocations et la traversée du graphe.
Elles ne donnent pas une durée exclusive exacte pour chaque fonction.

## Périmètre et correction du harnais

M5 24 Go, secteur, macOS27.2 build26B123, Xcode27.2(27B52), MLX0.32.2.
Qwen3.8-Flash-Next EXL3 2.05bpw, checkpoint `65c89531`, AR MAIN,
87 tokens d'entrée et128 sorties/127 transitions, cache3Gio/53slots×48couches,
quatre lecteurs. Hot/Wide/Unique/overlap actifs ; lookahead, T04/T05/T17/T18
désactivés. Les plans contiennent les options complètes. Aucun build, proveur
ou autre modèle ne tournait pendant les acquisitions GPU.

HEAD `7978fceac30d991a8c06d72114fc932b2443f555`, modifications locales
préexistantes conservées. Le binaire diagnostique final est
`32cc1c366497b0385e4ea6594211a9bef5e8393e42fb42891edc0ef444de3483`.
Les identités physiques sont [r3](natural-r3/identity.json) et
[r4](natural-r4/identity.json) ; l'[identité de compilation](stages-01/build-identity.json)
détaille les sources. Le CLI archivé `1fea100a…` et l'app installée sont inchangés.

L'audit a trouvé des clones d'Array dans les vecteurs `stages` sous `cfg(test)`,
même sans événements actifs. Ils gardent des références susceptibles d'empêcher
la donation MLX. STAGES01 garde ces références pendant warmup, puis les coupe
dans les cinq phases suivantes. Les comparaisons complètes de tokens, logits et
états restent actives, hors chronométrage. Dans r3/r4, la chauffe a789 stages et
les phases suivantes zéro, avec parité byte-à-byte des phases fermées.
`control_before` valide cette comparaison avant attachement d'Instruments.

Cela rapproche le harnais du chemin normal, sans le rendre identique au CLI :
spans, compteurs, sondes, métadonnées et instrumentation restent présents.
Les tableaux suivants sont des budgets diagnostiques, pas une ventilation
garantie de la latence de l'app. Le préfill AR suit bien le même découpage en
chunks de8 que le CLI ; à l'intérieur, Flash reste séquentiel par token.

## Fenêtre corrigée r3

[Analyse détaillée et preuves](natural-r3/attribution-partial/README.md).
Sur52 transitions complètes disponibles, sélection préenregistrée des40 premières
éligibles : pas76–115. L'alignement utilise5985 retours `preadv`,
7469351820octets, correction−26,989µs et intervalle résiduel±1,375µs.
La partition des états CPU n'a aucun résidu dans la fenêtre.

En revanche, environ10,24ms/sortie de Running∩GPUidle se trouvent en dehors
des deux soumissions. Les tableaux I/O décrivent une partie de ce travail,
mais ce rapport ne prétend pas attribuer exhaustivement chaque fonction de
ce reste. Une instrumentation source MLX disjointe serait nécessaire pour
des durées exclusives exactes ; elle n'a pas été ajoutée ici.

Les colonnes CPU/GPU se recouvrent et **ne se somment pas**. Running inclut le
travail utilisateur, noyau et tracing ; GPU actif signifie commandes en cours,
pas calcul ALU utile établi.

| Poste | Wall ms/sortie | CPU Running | CPU bloqué | GPU moteur actif | Running ∩ GPU idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transition entière |104,570|38,435|65,410|58,292|22,090|
| Deux soumissions async |22,385|22,261|0,008|10,176|11,854|
| Readback des routes |36,756|0,329|36,161|36,626|0,0036|
| Lot de lectures |32,576|7,990|24,314|5,396|4,843|

Dans le readback, la copie vaut seulement **0,00431 ms/sortie** ; l'enveloppe
sans aucune activité GPU est environ0,106ms. Les36–39ms d'attente ne sont donc
pas un trou GPU supprimable en retirant le readback. Accélérer ses producteurs
GPU est un autre changement.

Les soumissions ont deux rôles distincts :

| Appel | Appels sur40sorties | Wall ms/sortie | Running ∩ GPU idle |
| --- | ---: | ---: | ---: |
| `slots.indices().async_eval()` |1920|18,649|11,768|
| async du résultat Hot |1920|3,737|0,0862|

La première soumission évalue aussi les ancêtres du graphe : Cold précédent,
combinaisons/HC, GDN/QSA, routeur et cache-map. Son nom n'implique pas que les
indices eux-mêmes coûtent18,6ms. Le second point de disponibilité reste utile
au recouvrement Hot/lectures ; l'essai historique POSTPROFILE03 a déjà réfuté
leur réunion aveugle.

## Appels CPU identifiés

R3 contient un seul provider standard1ms :1543 points du thread d'inférence,
tous périodiques, dont497 dans les soumissions éligibles (494indices/3Hot).
Voici la première frame MLX rencontrée par point ; ce sont des comptes
statistiques exclusifs selon cette règle, pas des nombres d'appels ou des ms.

| Frame MLX | Points |
| --- | ---: |
| `CommandEncoder::maybeInsertBarrier` |73|
| `CommandEncoder::dispatch_threads` |68|
| `CommandEncoder::get_command_encoder` |62|
| `gpu::eval` |56|
| `eval_impl` |51|
| `CommandEncoder::set_input_array` |26|
| `CommandEncoder::commit` |17|

La feuille utilisateur `__kdebug_trace64` apparaît123fois. Une fraction du
profil implique donc le driver/tracing ; elle n'est pas une mesure du temps
récupérable sans Instruments. `maybeInsertBarrier` peut aussi gérer ses
ensembles sans émettre de barrière : points et appels ne comptent pas les
barrières réellement exécutées.

La [capture r2 haute fréquence](natural-r2/attribution-partial/README.md)
apporte4830 points périodiques éligibles et des piles noyau. Ses providers HF
et standard contiennent exactement les mêmes15645 timestamps/TID principaux,
avec poids100µs contre1ms. Ce ne sont pas deux observations indépendantes et
les poids ne sont pas des durées CPU exactes. R2 conserve aussi l'ancien biais
de rétention. Les callbacks exécutés sur d'autres threads restent séparés du
chemin critique du thread d'inférence.

## Lectures exposées et masquées

Dans r3, l'union des `preadv` vaut31,960ms/sortie, contre32,576ms pour le lot
complet. Le GPU moteur recouvre5,396ms des lots. Selon la frontière fixe
«dernière commande indépendante avant retour», le tail exposé est27,166ms,
dont26,930ms recouvrent des `preadv` :457 lectures terminent avant cette
frontière,4133 la traversent,1395 commencent après.

Cette classification décrit la file POSIX observée. Elle ne prouve ni le temps
NAND, ni que toutes les lectures traversantes sont causales. Le cache fichier
macOS peut servir des données au-delà du cache explicite3Gio ; sa contribution
au modèle n'est pas mesurée. Les anciens témoins physiques SSD n'ont pas fourni
d'événements bloc exploitables : ce capteur reste non qualifié.

## Classement et plafonds conditionnels

Les plafonds ci-dessous supposent que le poste indiqué disparaît et que le
reste garde son coût/ordonnancement. Ce sont des limites de modèle de coûts de
la fenêtre104,570ms, **pas des gains attendus sur le CLI**. Ne pas les additionner.

| Priorité | Travail à examiner | Plafond et limite | Risque |
| --- | --- | --- | --- |
|1|T09 : projection dense avec moins de frontières MLX/Metal, mécanisme DEC-ONE adapté|Le budget de toutes les soumissions exposées est11,854ms, soit+12,79% de débit si tout disparaissait. La part propre à T09 et son coût GPU sont non mesurés.|Ordre de réduction/arrondis, publication des partiels, compteurs, travail répété de préparation.|
|2|T12 : gate d'attention→préparation, sorties d'indexeur contiguës|Sous-ensemble du même budget CPU et de travaux GPU ; plafond propre non mesuré.|Strides, sigmoid, RoPE, pooling/contexte2052 et rollback exacts. Notre SDPA n'a pas le merge QSA de l'autre moteur.|
|3|T01B : variante/pipeline réellement préparée pendant eval|Sous-ensemble CPU, non mesuré en ms. T01A construction/source seule offrait seulement0,0335ms projetée pour sa closure ; ne pas lui attribuer12ms.|ABI MLX privée, signature/dtypes, fallbacks, concurrence et rétention.|
|4|Cache/ordonnanceur SSD ciblé à3Gio|Effacer les27,166ms de tail donnerait+35,10% dans le modèle fixe ; aucun cache réalisable3Gio n'est démontré atteindre cela.|Nouvelles lectures, leases GPU en vol, churn et pression RAM ; pool/ARC négatifs historiques et T18 mixte restent conservés.|
|5|Vérification GDN différée / spéculation Sushi|Ancien sous-ensemble éligible exposait seulement~0,38–0,47ms de GPU idle ; pas remesuré ici.|PLE/conv/HC et rollback ; ne pas lui attribuer l'attente routes entière.|
|6|Copie des routes seule|0,00431ms ; moins de0,01% du temps de la fenêtre.|Priorité faible ; les producteurs GPU ne disparaissent pas.|

Les [captures de projections isolées](projection-01/gpu-profiles.json)
identifient le QMV comme hotspot de samples, mais les temps absolus de replay
varient fortement pour la même capture. Elles ne qualifient pas une latence
native par kernel. Le plafond GPU utile par projection reste non mesuré.

La [lecture directe de TurboEXL3](turbo-read-01/README.md) précise ces choix.
Un vrai préfill par blocs est une autre piste importante, encore non mesurée ;
voir l'[audit GPU/ANE](npu-path-audit.md). Aucun facteur d'accélération n'est promis.

## Vérification et clôture

- Rust test-only : format, Clippy release MLX/chat strict tous targets et build
  passent ; libCPU171passés/8ignorés, libMLX physique ordinaire199passés/178ignorés.
  Les tests physiques ignorés ne sont pas déclarés exécutés.
- Kani : `cargo kani --lib --tests --no-default-features --harness
  flash_profile::stage_retention_restores_symbolic_nested_scopes --jobs 1` :
  95obligations réussies,4covers, aucun échec/inatteignable, deux booléens,
  unwind33. Kani remplace le TLS par un état statique : portée séquentielle,
  pas preuve d'isolation thread/MLX/Metal. Panic/thread sont exercés par le test CPU.
- Contrôleur final : `python -m pytest -q tests/test_decode_capture.py`,
  **106passés**, Ruff, format, py_compile et diffcheck passent. Les régressions
  préfix SIGTERM, RSS terminal et contrat de stages sont archivées. CrossHair
  absent ; pas de preuve formelle générale Python.
- Analyse : par dossier r2/r3,10tests,3375requêtes union et3375intersection du
  self-check fini, Ruff/format/compile passent. Une mutation du byte-count est
  refusée. Commandes/SHA/provenance dans les receipts et verification.json.
- CI PR/main : contrôles Python câblés et harnais Kani ajouté à l'étape
  existante ; YAML inspecté. Actionlint absent, aucun run GitHub du code local
  non publié. Ne pas déclarer CI distante verte.

R1 s'arrête à la jointure10s ; r2 à60s ; r3 au faux refus RSS Darwin E pendant
la sortie du collecteur. Les erreurs et nettoyages restent enregistrés.
FIX06 traite l'état E et attend son enfant sur EPERM avec borne5s.
La dernière relance r4 s'arrête après **256Kio de nouvelles écritures swap
globales**, contre la règle zéro. Cible et traceur sont rejoints, aucun processus
restant. **Aucun run natural six phases complet** n'est validé ici ; contrôle
après, MTP/DF et capture CPU-only séparée restent non exécutés. Pas de relance
automatique, de réancrage, de changement de seuil ou de publication.

Les plans exacts, logs, sorties partielles, SHA et décisions sont liés depuis
[opti.md](../../../opti.md). Aucun résultat interrompu n'a été transformé en
succès ; aucun nouveau boost n'est intégré à l'app.
