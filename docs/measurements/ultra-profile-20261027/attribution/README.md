# Attribution CPU/GPU et lectures d’experts — 27 octobre 2026

Les priorités observables sont l’orchestration des soumissions MLX et le chemin de cache/lecture des experts. L’attente des routes accompagne presque entièrement des commandes GPU en cours. La copie des identifiants est négligeable. **Le temps physique SSD n’a pas pu être qualifié : il ne faut pas appeler les temps de `preadv` du temps de service SSD.** Aucun nouveau boost, algorithme ou kernel n’a été intégré à partir de cette analyse ; l’app MLXL3 installée reste inchangée.

## Périmètre et provenance

- M5 standard 24 Gio, secteur, macOS 27.2(26B5112a), MLX 0.32.2, Xcode/xctrace 27.1(27A9275). Pas de compilation/proveur/autre modèle durant les acquisitions GPU.
- Qwen3.8-Flash-Next EXL3, checkpoint local `Qwen3.8-Flash-Next-exl3-205-65c89531`, MAIN,87 tokens d’entrée,128 sorties, décodage AR sans MTP/DFlash2, cache experts cible 3 Gio/53 slots par couche/48 couches,4 lecteurs, lookahead 0, GPUHot/overlap actifs. Les options complètes figurent dans les plans.
- HEAD `7978fceac30d991a8c06d72114fc932b2443f555` avec modifications locales de profilage déjà présentes. Libtest figé SHA256 `d4bb73099490a6908db442fac5337c44feefeab2910746af69aed5ec844c87b2` ; [identité sources/options](../v1-source.json). Aucun changement natif depuis cette identité pour l’attribution fine.
- Référence historique :37,97–38,81 ms de source-wait routes,17,03–18,51 ms de soumission hôte et 29,00–30,29 ms de bloc lectures par sortie. Ces valeurs ne sont pas des mesures GPU exclusives. Voir [rapport initial](../first-AR-breakdown.json).

Le protocole ciblé exécute warmup/control_before/trace_1 puis termine volontairement **notre** processus après la validation numérique et la fermeture des JSON de trace_1. Il ne déclare pas passé le test natif complet à 6 phases : trace_2/trace_3/control_after sont délibérément non exécutés. [Plan](short/AR-MAIN-plan.json), [supervision](short/AR-MAIN-plan-result.json), [capture](short/AR-MAIN/capture.json).

La fenêtre courte a réussi : zéro nouveau Swapout du début à la fin, empreinte physique native aux frontières au plus 7,554 Go (<12 Go),128 tokens/texte/hash golden et 248320 logits finis avec états complets byte-exact dans chacune des 3 phases. Traceur/cible/superviseur rejoints, SIGTERM-15 planifié et consigné. Les captures lourdes antérieures interrompues après 104,19 Mio puis 5,44 Mio de nouveau swap restent des échecs, conservés séparément.

## Horloges et couverture

L’origine des événements vient de `time-info` et du timebase 125/3, **pas du seul start-date du TOC**. La capture GamePerformance initiale expose notamment un TOC décalé par rapport à son origine réelle. Un mauvais calage d’environ 24 us change sensiblement le partage CPU pour les petits spans répétés 48 fois/token.

Le calage constant est contraint indépendamment du GPU : chaque syscall de lecture doit être inclus dans son bloc natif parent, avec nombre de retours et octets identiques aux jobs. Sur la courte capture :2619 lectures,3268543428 B, décalage central+6666 ns, intervalle compatible de demi-largeur 1292 ns. Sur la première :5244 lectures,6544574928 B, correction-23624 ns et demi-largeur 1396 ns. Les 297 lots à un job de cette première capture donnent un temps de fonction utilisateur 583–10958 ns plus long que son syscall, cohérent avec l’emboîtement. C’est une calibration sous hypothèse d’offset constant sur la fenêtre, pas une preuve de toutes les horloges système.

La courte acquisition demandait 2 s ; les exports couvrent 20 pas complets 48–67. La première fournit 36 pas complets 84–119. Les marges incomplètes sont exclues ; les deux fenêtres ne sont pas regroupées pour produire une moyenne. Les états CPU du thread réel d’inférence sont intégrés : Running/Blocked/Runnable/Interrupted/Preempted conservent exactement le temps de chaque groupe dans ces exports. Les travailleurs I/O opèrent en parallèle : leurs latences ne sont pas additionnées.

## Décomposition mesurée

Toutes les valeurs suivantes sont des **ms par token décodé dans les 20 pas capturés**, et non une nouvelle baseline de vitesse. Le thread CPU est celui de l’inférence, pas la somme du travail de tous les cœurs. Les axes CPU et GPU sont simultanés : **ne pas additionner une colonne CPU et une colonne GPU**.

| Groupe hôte | Durée murale | CPU actif | CPU bloqué | Runnable/interruption/préemption | GPU du moteur actif | GPU sans activité observée, tous processus |
|---|---:|---:|---:|---:|---:|---:|
| Lecture des routes, enveloppe complète |40,056|0,302|39,473|0,281|39,947|0,106|
| Soumission MLX |22,033|21,912|0,010|0,110|9,710|12,195|
| Bloc lectures experts |38,108|7,387|30,147|0,574|6,685|31,055|

[Résultats complets](short/AR-MAIN/partial-attribution.json), [corroboration 36 pas](AR-MAIN/partial-attribution.json). Les bornes de calibration sont importantes pour les très petits résidus ; «GPU actif» signifie exécution de commandes attribuées au processus, **pas taux d’occupation ALU ni temps purement arithmétique**. Les délais mémoire peuvent se trouver à l’intérieur d’une commande active.

- **Routes :** source-wait C++ 40,002 ms, copie 0,00565 ms, enveloppe 40,056 ms. Le chevauchement avec les commandes GPU explique l’essentiel de l’attente. Sur 36 pas, source-wait 35,937 ms et enveloppe 36,006 ms, dont 35,929 ms d’activité GPU du moteur. Retirer un appel d’attente ne supprime pas le calcul qui produit les routes. Le résidu GPU inactif est d’environ 0,08–0,11 ms/token dans ces fenêtres, sous réserve du calage.
- **Soumission :**96 appels `async_eval`/token, et non 96 dispatches GPU. CPU actif pendant un GPU inactif :12,142 ms/token ; corroboration 11,803 ms. Environ 1,166 ms de CPU actif est à l’intérieur des syscalls, le reste (~20,746 ms) hors de ces intervalles : MLX/encodage Metal/allocations/hooks côté utilisateur restent à distinguer plus finement. La soumission est largement du travail CPU, pas une grande attente GPU cachée sous ce nom.
- **Métadonnées :**5258 fstat du thread principal sur 20 pas ne totalisent que 0,113 ms/token. Ils viennent notamment des contrôles de troncature/bornes de `TensorInfo::source` puis `Array::read_records`. Leur suppression serait peu rentable et risquerait d’enlever une protection utile.

## Quels appels occupent les 12,14 ms exposées ?

La séparation des deux appels est mesurée depuis les spans et les états CPU/GPU, sans partage arbitraire des piles. Chaque couche a exactement deux soumissions dans `SsdMoe::prepare_resident_hot` :

| Appel Rust puis même FFI `mlxl3_array_async_eval` | Appels sur 20 tokens | Durée totale/token | CPU actif avec GPU inactif/token |
|---|---:|---:|---:|
| `slots.indices().async_eval()` — ssd_moe.rs:560 |960|18,323 ms|**11,897 ms (98,0%)**|
| `output.async_eval()` — ssd_moe.rs:569, résultat des experts Hot |960|3,710 ms|**0,245 ms (2,0%)**|

Le chemin est `Array::async_eval` → `mlxl3_array_async_eval` → `mlx::core::async_eval` → `mlx::core::eval_impl(..., true)` → évaluation/encodage des primitives GPU. «Async» dispense d’attendre la fin GPU ; la préparation CPU et l’encodage restent exécutés avant le retour. Le premier appel entraîne les dépendances du tenseur d’indices : il ne mesure pas seulement le calcul du petit kernel cache-map.

Sur **249 points CPU à 1 kHz** appartenant à ces fenêtres exposées, 243 sont dans le premier appel et6 dans le second. Choisir une seule frame MLX, la plus proche de la feuille, donne les comptes exclusifs suivants :

| Frame MLX la plus proche | Points /249 | Part des points |
|---|---:|---:|
| `metal::CommandEncoder::maybeInsertBarrier()` |39|15,7%|
| `metal::CommandEncoder::get_command_encoder()` |37|14,9%|
| `metal::CommandEncoder::dispatch_threads(...)` |29|11,6%|
| `gpu::eval(array&)` |23|9,2%|
| `eval_impl(vector<array>, bool)` |21|8,4%|
| `metal::CommandEncoder::commit(...)` |18|7,2%|
| `metal::CommandEncoder::set_input_array(...)` |10|4,0%|

Les fonctions `CommandEncoder` et leurs callbacks représentent au total 143/249 points (57,4%). C’est surtout de l’encodage Metal dans cette vue, avec allocations, suivi des ressources, émission des barrières/dispatches et commits ; la simple traversée `eval_impl` n’est pas la majorité. Dans [MLX v0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp), `maybeInsertBarrier` encode une barrière de buffers et maintient les ensembles d’entrées/sorties ; `get_command_encoder` peut créer un encodeur concurrent et une fence. Cela ne décrit pas une attente CPU de fin GPU, et ces dépendances mémoire doivent être préservées.

**Limite essentielle :** la feuille `__kdebug_trace64` apparaît48/249fois. Le tracing peut donc peser sur cette capture ; il n’est pas établi que tout le budget exposé soit récupérable hors Instruments. Les allocations/libérations à la feuille chevauchent les catégories MLX ci-dessus, donc ne doivent pas y être ajoutées. Les pourcentages sont statistiques : ils ne fournissent pas des durées exactes par fonction. Tous les points sélectionnés ont un contexte MLX résolu ; les feuilles anonymes restent dans le dénominateur. Le compteur1,166ms dans les syscalls observés n’est pas une partition exhaustive user/kernel, notamment face à ce tracing.

Le filtre initial «Running à l’instant même de l’échantillon» retournait zéro et a été rejeté. Les757points du thread d’inférence dans la fenêtre sélectionnée se trouvent dans des intervalles Interrupted ; ils ont tous un contexte Running juste avant l’interruption. Le PC échantillonné est rattaché à ce contexte, puis filtré sur la soumission et le GPU inactif. Aucun point non résolu n’est silencieusement normalisé dans cette analyse. [Comptes et durées](short/AR-MAIN/async-callstacks.json), [piles sélectionnées](short/AR-MAIN/async-eligible-stacks.json), [source de l’analyse](short/async-stack-analysis-source.txt).

## Lectures critiques, masquage et limite physique

Les 2619 retours `preadv` sont chacun de 1248012 B et concordent exactement avec les jobs natifs. Leur union temporelle vaut 37,211 ms/token, à comparer aux 38,108 ms du bloc complet. Le reste inclut préparation, banques/MLX éventuels, marshalling, ordonnanceur, retour et instrumentation ; ce n’est pas un compteur de CPU pur.

Pendant ces blocs,6,685 ms/token de commandes GPU du moteur sont observées. Dans 816 lots, le résultat Hot est encore indisponible au début et prêt au retour dans 751 cas ; encore indisponible au retour dans 61 cas ; déjà prêt aux deux bornes dans 4 cas. Ces observations décrivent l’ordre des producteurs ; elles ne prouvent pas que chaque octet est ou n’est pas critique.

Avec l’ordre actuel des queues et le fork Hot/lectures existant, la référence «dernière activité GPU indépendante avant retour I/O» donne 31,409 ms/token de queue exposée, dont 30,862 ms d’union de `preadv`.244 lectures finissent avant cette référence,1876 la traversent et 499 commencent après. C’est un **budget de chemin de service/copie/OS**, pas 31 ms de NAND, ni une preuve qu’avancer toutes ces lectures avance autant le token final. La fenêtre 36 pas confirme la structure, avec une queue d’environ 27 ms/token ; valeurs exactes dans son JSON.

`DiskIoRoutines` contient **zéro ligne** dans la capture modèle ; les compteurs de shader et les labels d’encodeurs sont également absents. Pour ne pas transformer une absence de données en résultat, deux témoins ont été effectués avec fichiers jetables, données déterministes, `F_NOCACHE=48`, `F_FULLFSYNC=51`, retours byte-exact et nettoyage :

1. TimeProfiler + mêmes instruments,128 Mio écrits et lus. Les 4 pwrite/4 pread/fsync/fcntl sont bien observés, mais aucune requête bloc. iostatdisk 0 augmente 1241,80 MB pendant la fenêtre, ce qui inclut les écritures du traceur et ne s’attribue pas au helper.
2. SystemTrace +DiskIOLatency,3 itérations de 128 Mio écrits/lus, toujours zéro événement bloc. Cibles et traceurs sortent 0 et sont rejoints ; fichiers supprimés. [Témoin 1](io-positive/inspection.json), [témoin2](io-system/inspection.json).

Le capteur physique reste donc **non qualifié dans cette session**. La cause exacte n’est pas établie ; ne pas inventer une explication d’autorisation ou de version. Les temps NAND, file du contrôleur, octets physiques par requête et leur portion critique restent **non mesurés**. Les compteurs `ru_inblock/oublock=0` du helper ne sont pas utilisés pour déduire zéro I/O.

Le moteur lit des fichiers ordinaires par `preadv` sans `F_NOCACHE` ; le cache fichier macOS peut servir une partie des requêtes en plus des banques explicites 3 Gio. Son volume propre au modèle n’est pas mesuré. Le compteur `read_bytes` représente les octets logiquement retournés/importés, pas les octets physiques SSD. Ne pas recommander un SSD plus rapide ni attribuer toute la queue au stockage sur cette base.

## Plafonds conditionnels et classement

Les plafonds ci-dessous sont des **budgets optimistes du modèle de coûts observé**, à durées de travail restantes et ordre de queues comparables. Ils ne sont pas des gains validés sur la baseline. Les optimisations changent les recouvrements et parfois les autres budgets : **ne pas additionner leurs plafonds**. La fenêtre instrumentée est plus lente que son contrôle :113,755 ms/token contre 91,671 ms sur les mêmes indices ; la première donne 103,204 contre91,993 ms. L’écart mélange instrumentation, état des caches et conditions ; il interdit une projection précise de ces plafonds vers la vitesse livrée.

| Priorité | Piste | Budget/plafond observé | Risques et décision |
|---|---|---|---|
|1|Réduire le coût CPU des graphes/soumissions MLX en conservant les points de disponibilité utiles|~11,8–12,1 ms/token de CPU actif avecGPU inactif, soit ~12–13% de débit dans ces fenêtres si tout ce budget disparaissait|Ne pas fusionner aveuglément les fins de buffer : cela peut retarder les routes ou le Hot. Séparer encodage/allocations/traversée MLX avant modification, puis oracle et A/B production. Gain réel non mesuré.|
|2|Cache global d’experts/redistribution des slots, piste TensorFold, **à 3 Gio**|Le chemin logique de lecture offre ~27–31 ms de queue exposée dans les captures. L’effacer intégralement donnerait une borne conditionnelle ~35–38%, **pas un gain compatible 3 Gio démontré**|La borne «lectures gratuites» n’est pas une politique de cache réalisable. Le gain à budget 3 Gio reste non mesuré : replay de routes/politiques nécessaire. Protéger les lectures GPU encore en vol, les leases/offsets et les pics RAM.|
|3|Spéculation cache-map et vérification GDN différée de Sushi|Sous-ensemble AllHot 12,4–15%; préfixes éligibles avant prochaine lecture de routes :2,37–2,72 ms/token de durée totale, seulement 0,38–0,47 ms deGPU inactif|Le préfixe mesuré inclut plus que le GDN : c’est déjà une borne lâche. PLE est exécuté dans le modèle ; le chemin amont sansPLE ne se porte pas tel quel. Rollback exact GDN/conv/HC/injection/initialisation requis, avec coût d’une éventuelle soumission supplémentaire. Faible budget exposé face au risque.|
|4|Optimiser un kernel GPU producteur identifié|63,013 ms/token de commandes GPU actives dans la courte capture,57,233 dans l’autre ; **plafond par kernel utile non mesuré**|Ces temps incluent les stalls et plusieurs opérations. Labels/counters manquants : ne pas choisir un portage depuis ces seules durées. Les essais Mia déjà rejetés ne sont pas relancés sans forme/kernel/protocole nouveau.|
|5|Dédupliquer les contrôles de métadonnées par lot|Au plus~0,11–0,12 ms/token de fstat observés|Conserver les erreurs de fichier tronqué et de bornes. Priorité basse, aucun code modifié.|
|6|Accélérer seulement la copie des routes ou supprimer le readback sans changer les producteurs|Copie~0,005–0,006 ms/token ; ~0,08–0,11 ms de GPU inactif dans l’enveloppe d’attente|Gain minuscule dans l’ordre actuel. La spéculation qui déplace du travail futur a un autre budget, évalué séparément ci-dessus.|

La piste de pool est pertinente à tester parce que les défauts sont hétérogènes : sur 20 pas chaque couche reçoit 200 sélections, mais ses misses vont de 12 à 129 malgré la même capacité 53. Cela **ne prouve pas** que retirer des slots à la couche froide conserve ces résultats sur le contexte suivant. Les essais historiques statiques 47/55 ne valident pas un pool global. Les slots du head MTP entièrement résident ne doivent pas être confondus avec ceux du target 3 Gio.

Pour situer l’objectif : passer de 11,5 à 15 tok/s exige de gagner 20,29 ms/token (23,33% de latence). Les chiffres ci-dessus ne démontrent pas que cette somme soit récupérable ; aucun boost n’est annoncé. Le cache de préfixes SSD concerne surtout le TTFT et ne reçoit pas la priorité pour ce decode natif.

## Vérification, état des outils et suite non mesurée

L’[analyse reproductible](analyze.py) refuse les tables vides, références inconnues/dupliquées ou de type incompatible, bornes invalides, étapes absentes, sorties vides et discordances des nombres/octets de lectures. Son self-check compare 3375 requêtes d’union et 3375 d’intersection sur deux intervalles et fenêtres de grille 0..4 à un oracle indépendant par ensembles ; il couvre aussi 3 entrées de bornes invalides, un XML valide et 5 XML corrompus/vides. C’est de l’**exploration exhaustive finie de ce domaine**, pas une preuve générale des intervalles ou du moteur.

Commandes CPU : `python analyze.py --self-check`, `python -m py_compile analyze.py`, `ruff format --check analyze.py`, `ruff check analyze.py`, puis `python analyze.py <capture-directory>` sur chacune des 2 captures. 230tests des exports/supervision/profilage passent en2,72s. Une mutation temporaire max→min de la fusion des intervalles est détectée par le self-check, puis la version conservée repasse. La vérification YAML et le câblage des commandes CI passent localement. Les vérifications self-check sont ajoutées à la CI PR/push main existante ; la version locale non publiée n’a pas de run GitHub inspectable. `python -m crosshair check analyze.py --analysis_kind asserts` a été tenté : module absent, propriété source **non vérifiée formellement**. Les résultats ne prouvent ni les transitions Metal ni les horloges externes.

Les MTP/DFlash2 détaillés CPU/GPU, les contextes longs et les largeurs restantes ne sont pas extrapolés depuis AR ; la matrice complète demeure incomplète. Les 9 datasets natifs antérieurs et les campagnes CLI restent documentés dans le [rapport parent](../README.md). Pas de push/release/remplacement de l’app. Xcode est installé, le consentement de licence est reçu ; son enregistrement est maintenant confirmé : après action utilisateur, xcodebuild -checkFirstLaunchStatus et xcrun xctrace version sortent0. Le CLI public utilisé ici fonctionne avec les droits préexistants, sans modifier xcode-select, DevToolsSecurity ou les préférences de licence.

Révision après acceptation de Xcode : `xcodebuild -checkFirstLaunchStatus` et `xcrun xctrace version` passent. Le témoin disque identique post-initialisation fournit 0 événements bloc ; la latence physique du modèle reste non mesurée. [Contrôle post-setup](io-post-xcode/inspection.json). [Vérifications finales](verification.json).
