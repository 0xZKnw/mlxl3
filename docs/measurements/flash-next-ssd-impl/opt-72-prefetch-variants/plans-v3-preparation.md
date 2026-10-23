# Préparation72/73 v3 — aucun lancement

Six nouveaux bridges43 sont préparés : late,bounded,unused,lfu,weighted,early-shared. Leurs run-v2 sont conservés ; seules la CLI77v1 et les destinationsv3 changent. Chaque plan garde pré20/checker840s(parent900s)/post30, cache3Gio53slots, allocateur64Mio, options exactes72T/73T.

L’écran packedB2 prépare MAIN puisHOLD, H-C pour chaque prompt. H=legacy/pipeline1/lookahead2 ; C=packed, unique différence d’environnement. WIDTH1/ADAPTIVE0,128sorties/127decode, unwarmup128+unemesure128 parenfant. Les oracles/goldens et paramètres initiaux sont inchangés. Pas de CH ajouté ni nouvellevariante.

Le wrapper77fresh-controls/observe-stock.py est référencé telquel, sans copie ni modification de codeexécutable. Les launch_argv du manifeste ne sont pas exécutés. Le parent garde l’exclusivitéGPU et décide de chaque lancement après les43contrôles valides du mode concerné ; packedv3 déjàen cours n’est ni relancé ni modifié.

Statuts séparés :43numérique peut être validé ; les plafondscompression77B, déjàéchoués sur témoins, restent échoués. Le stock garde256MioSwapins/Swapouts système parphase. Les débits sont diagnostiques/nonpromus tant que la qualificationmémoire ne passe pas. Les métriques propres doivent être analysées et les violations conservées ; absence de hausseledger ne prouve pas absence de swap-outsmodèle. Aucun nouveau seuil ou baselineeffaçantl’incident72.

Les sources exactes, changements attendus75/77 et deuxbinaires sont dans plans-v3-source-identity.json. CLI SHA69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c ; wrapper SHAad28af767cec59cbf1a763cf0aeb12b4d0970219bc44da06cda0f1c68e4fb5ac. Aucun modèle/benchmark/test/build/prouveur lancé par cette préparation.

Manifest : plans-v3-prepared.json,7plans avec chemins, SHA et commande de lancement éventuelle. Ne pas utiliser le contrôleur strict75 pour ces plans77B ; ils sont explicitement destinés au wrapperstock. Le stock enchaîne les items sans analyse mémoire parprocessus intermédiaire : le parent conserve cette limite dans le diagnostic et inspecte les rapports avant une phase dépendante.
