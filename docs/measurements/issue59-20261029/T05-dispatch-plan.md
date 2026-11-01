# T05 — vérifier les vrais dispatchs, 29 octobre 2026

État : plan et test de capture préparés, **aucune nouvelle capture/GPU/compilation/proof exécutée par cet agent**. Seuls l’inventaire des outils et l’aide CLI ont été interrogés. Le nombre de dispatchs retirés reste **non mesuré**.

## Chemin le plus court et déjà disponible

Réutiliser `array::MetalCapture` (`native/src/array.rs:395`), son pont C++ `mlxl3_start_capture/stop_capture` et MLX0.32.2 `metal::start_capture/stop_capture`. La garde existante a déjà été exercée physiquement lors de MEMORY-2026-10-09-PROFILE-CAPTURE : finish/drop, refus d’écrasement et artefacts non vides. Elle synchronise les travaux avant le stop et tente toujours le nettoyage lors d’une erreur.

Un test physique séparé est maintenant préparé dans `native/src/weighted_epilogue/capture_tests.rs` :

```text
weighted_epilogue::capture_tests::captures_real_legacy_and_fused_epilogue_commands
```

La commande à placer dans le superviseur GPU après gel du nouveau libtest est :

```text
<frozen-libtest> weighted_epilogue::capture_tests::captures_real_legacy_and_fused_epilogue_commands --exact --ignored --nocapture --test-threads=1
```

Environnement ajouté : `MTL_CAPTURE_ENABLED=1` dès le démarrage du processus et `MLXL3_WEIGHTED_CAPTURE_OUTPUT=<répertoire neuf existant>`. Les deux noms `legacy.gputrace` et `fused.gputrace` doivent être absents **avant** toute capture. Pas de chargement du modèle complet : seulement les tableaux littéraux du composant.

Le test fixe Hot10×2560, Cold5×2560, dix rangs canoniques intercalant cinq occurrences de chaque buffer, valeurs FP16 distinctes et poids FP32 non uniformes/signés, activation FP32. Inputs/poids sont matérialisés ; les deux chemins sont chauffés et synchronisés. Il construit ensuite un **nouveau graphe** legacy et l’évalue à l’intérieur de la première capture, puis un nouveau graphe fused dans la seconde. Les readbacks, formes/types, finitude de25600valeurs et égalité des102400octets complets sont vérifiés **après** les captures. Le résultat ne réutilise pas la sortie du warmup.

Le compteur interne `kernel_requests_for_test()==1` ne sert qu’à confirmer que le helper fused est atteint ; ce n’est pas un compteur de dispatch. Le stdout l’annonce explicitement : `captured_exact_outputs_dispatch_inspection_pending`.

Les deux chemins créent leur propre tableau GPU d’indices depuis la même slice CPU à chaque appel, comme leurs chemins de production. Les éventuelles commandes d’upload sont donc conservées symétriquement dans les captures ; les créations de ressources CPU ne sont pas des dispatchs.

## Lecture qui autorise une conclusion

Ouvrir les deux `.gputrace` terminées dans **Xcode GPU Frame Debugger** après la fin des tests GPU natifs. Inspecter la liste de commandes et les fonctions des pipelines pour le seul intervalle capturé. Compter séparément :

- appels Metal de compute (`dispatchThreads`, `dispatchThreadgroups`, y compris les éventuels indirects) ;
- commandes de copie/blit ;
- command buffers/encoders, qui ne constituent **pas** des dispatchs.

Conserver capture/nom du fichier, liste ou capture d’écran des commandes, noms des fonctions, numéros des commandes et comptes par catégorie. Le kernel candidat attendu s’appelle `mlxl3_flash_weighted_epilogue_v1` (nom MLX généré pouvant contenir préfixe/types). Déterminer depuis le fichier ce que le backend legacy a réellement matérialisé pour concat/take/cast/mul ; ne pas supposer «quatre opérations donc quatre dispatchs». Sum/sharedadd sont volontairement exclus de ce petit consommateur et restent identiques en production.

Une capture non vide et une sortie exacte sont nécessaires mais ne suffisent pas à valider le comptage : si la liste de commandes n’est pas exploitable, marquer **dispatchs non mesurés**. L’ouverture/replay de la capture peut utiliser le GPU ; elle doit rester séparée de toute campagne de vitesse modèle. Les temps observés sous capture ne sont pas des résultats release.

## Pourquoi les anciens exports Instruments ne suffisent pas

Le jeu déjà validé `ultra-profile-20261027/attribution/short/AR-MAIN` contient5542lignes `metal-gpu-intervals`, mais ce sont des enveloppes «GPU Execution», avec labels absents et IDs d’encodeurs. Son `metal-application-encoders-list.xml` contient **zéro ligne**. Le rapport d’attribution les utilise légitimement pour activité GPU/CPU et union temporelle, jamais comme nombre de kernels. Les requêtes/signatures MLX du profileur Rust sont encore une troisième métrique : du graphe lazy.

L’inventaire actuel de Xcode27.1 confirme les templates `Metal System Trace`, `Game Performance` et `Time Profiler`. La CLI exporte des tables `.trace` par `--toc`/`--xpath` ; aucun exporteur `.gputrace` en liste de dispatchs n’a été qualifié. `xctrace export --help` retourne une erreur de parsing46 tout en imprimant l’usage : ce n’est pas une qualification d’export. Le lancement Game Performance historique `metal-smoke` avait fini en54 et target.log vide ; ne pas le recycler comme preuve de dispatchs.

Si une automatisation CLI devient indispensable, faire un essai séparé et borné avec `xctrace record --template 'Metal System Trace' --launch -- <frozen-libtest> <test-exact> …`, puis `xctrace export --input <trace> --toc`. **Qualifier d’abord** une table fournissant réellement une ligne par appel de compute (et pas par encoder), avec identité du processus et nom de pipeline, sur un témoin connu ; sinon abandonner ce comptage. Le template existe, sa capacité à fournir ces données pour notre CLI headless n’est pas démontrée. Ne pas relancer un profil modèle lourd pour résoudre cette question.

## Bornes et nettoyage

Le root doit placer ce seul test dans le superviseur déjà utilisé : sélection exacte1test, timeout120s proposé, processus/groupe joints sur succès ou erreur, logs durables, secteur/quiet, ancre cumulative à zéro nouveau Swapout, Swapin≤256Mio, RSS≤10Gio et footprint≤12Go. Aucun compileur/proveur/autre modèle simultané. Surveiller le total des deux captures et arrêter à1Gio si leur collecte dérive. L’absence, interruption, timeout, parité échouée, capture incomplète ou inspection impossible conserve un état explicite ; aucun fallback n’est renommé «dispatchs validés».

## Revue indépendante du contrat T05

Le chemin proposé conserve correctement la sélection d’une seule source avant lecture, Hotphysique (et non nombre de hits), cast versTx puis promotionTx/Tw, somme10rangs et sharedadd inchangés ; les sorties du graphe conservent les sources vivantes comme le consommateur legacy. Aucune fusion FMA/somme par arrivée n’est introduite dans le shader lu.

Un défaut concret de frontière a été trouvé et transmis : une weight scalaire(rank0) pourR1 passe le compte d’éléments, mais MLX génère une référence scalaire MSL et `weights[row]` ne compile pas. Le root a choisi le garde minimal explicite `!weights.shape().is_empty()` avant le lancement ; le test d’erreur scalaireR1 est demandé à l’agent des oracles. Les rangs1/2 de production restent admis sans reshape ajouté. Ces observations sont une revue de source, pas une preuve numérique ou concurrente ; compilation et tests physiques restent à exécuter par le root.
