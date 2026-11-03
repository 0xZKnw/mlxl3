# 75M — admission mémoire du superviseur Flash

État : code local, correction CPU vérifiée ; nouvelle campagne modèle non lancée. Aucun gain de RAM moteur ou de débit revendiqué.

## Diagnostic

Le bridge72 packed échoue après34/43contrôles : les Swapouts système passent de157155328à206635008B (+49479680B). Cet incident reste échoué. Les26événements `complete` comparables au bridge69 réussi ne montrent pas d’augmentation importante du moteur : maximum de l’écart MLXactif+264704B, empreinte physique+19709904B. Àrequête33, pics de vie7 935 351 288contre7 935 662 584B. Les slots restent53/couche, cacheexpert3Gio. Ceci ne démontre pas l’absence de fuite ailleurs.

En revanche notre superviseur démarrait avec seulement2 995 339 264B libres contre7 492 829 184B auparavant. Il contrôlait les compteurs swap avant Popen, mais aucun budget de travail ; ensuite il attendait une écriture swap pour couper. L’anonyme avait augmenté3 815 636 992B avant chargement, sans attribution exclusive à un processus. La quantité file-backed ne constitue pas une réserve de RAM garantie.

## Correction minimale

`scripts/profile_flash.py` réutilise le hook et le moniteur existants, sans modifier les scripts historiques :

- Une étape annonçant `MLXL3_FLASH_MODEL` doit déclarer troisentiers strictement positifs : `load_headroom_bytes`, `minimum_live_free_bytes`, `maximum_compressor_bytes`. Une politique partielle est refusée. Les sleeps pré/post restent possibles. Les autres commandes chargeant un modèle doivent déclarer explicitement cette politique ; aucun budget n’est deviné depuis leur ligne de commande.
- `load_headroom_bytes` est la mémoire libre minimale exigée **avant chargement pour toute la phase**. La future75T doit choisir le working-set mesuré (~7,94Go pour ce bridge)+réserve, et pas seulement le picstartup3,74Go. Aucune valeur universelle4Gio/512Mio/7Go n’est ajoutée au produit. Les valeurs de tests sont des fixtures.
- Les seuils live arrêtent et joignent l’enfant possédé via le nettoyage existant ; un arrêt reste `failed_or_interrupted`, jamais une parité. Les échantillons VM ordonnés et horodatés `epoch` sont conservés. L’erreur d’admission arrive avant Popen. Une baseline cumulative explicite est obligatoire pour une politique modèle, y compris partielle.
- Une dernière sonde après `wait()` contrôle Swapouts et Swapins cumulatifs avec la limite originale du vrai moniteur, même si l’enfant termine en moins de2s. Les snapshots initiaux et l’ancre ne sont pas modifiés.

Cette garde conservatrice peut refuser une machine qui pourrait récupérer du cache. Elle ne réduit pas les7,94Go du moteur, ne garantit pas zéro swap entre deux sondes et ne remplace pas un budget natif. Aucun cacheexpert réduit, fermeture d’app, purge, reboot ou nouvelle origine modèle.

## Vérifications

- Fmt/Ruff/py_compile passent ; YAML CI parsé, checks ajoutés au job Python existant sur push et PR. Run GitHub non exécuté pour ce code local.
-168tests admission/profilage passent6,67s sur la version finalev4. Ils couvrent les frontières exactes, entiers0/2^64−1, bool/négatifs/incomplets, cachefile-backed ignoré, politiques partielles, refus avant enfant et vrais enfants jetables arrêtés/joints.
- Trois mutations isolées du vrai source détectées : retirer l’admission, retirer la sonde finale, retirer la limite Swapins finale. Chaque mutation fait échouer1test ; normal/restauré3/3passent. Copies supprimées, production inchangée. Bruts dans `mutation-v4/`.
- Première suitePythonv2 :1047passés/4ignorés107,67s. Suitev4 finale : **1072passés/4ignorés91,74s**, processus joint, sourceSHA identiques avant/après ; voir `tests-full-v4.log` et son résultat de processus. Échantillonnage CPU : zéro Swapout additionnel aucompteur206635008B (ancienincident49479680B conservé, pas de nouvelle qualification modèle). Le fichier65P préexistant non lié est exclu, aucune modification de ses données.
- CrossHair tenté sur la vraie `memory_headroom` : module absent/exit1 ; aucune preuve formelle Python. Aucun Rust modifié dans75M, donc pas de Kani75M. Les tests ne prouvent ni macOS ni MLX/Metal ni les courses mémoire système.

Les deux échecs de fixtures/lint intermédiaires et les résultatsv1/v2 restent dans le journal. Aucun benchmark GPU ni test numérique modèle n’a été relancé pour75M.

## Révision23octobre — v5 après revue indépendante

Une frontière manquait en v4 : `load_headroom_bytes:null`, avec `env:{}` et sans les deux autres clés, était confondu avec une politique absente et pouvait retourner sans garde. Les commandes CLI bridge n’exigent pas `MLXL3_FLASH_MODEL` ; une politique explicite ne doit donc pas dépendre de ce flag. La correction distingue présence et valeur par une seule ligne, sans changer les seuils ni inventer une politique.

Six régressions vérifient chacune des trois clés nulles individuellement, avec envvide, avant lancement et en contrôle live. Un mutant isolé rétablissant l’ancienne condition produit2échecs/4passages ; normal/restauré6/6, production jamais mutée. Résultats dans `mutation-v5/`. Les trois mutants historiques restent conservés.

Version finalev5 : **174tests ciblés passent6,99s ;1078passés/4ignorés dans la suite complète107,57s**. Les quatre skips et le fichier65P exclu ont les mêmes raisons quev4. Fmt/Ruff/py_compile passent. QuatreSHA sources inchangés avant/après ; `source-identity-v5.json` et `sources-v5.tar.gz`. La suiteCPU a été jointe et échantillonnée sous1GioRSS ; métriques exactes dans `summary-v5.json`. Résultatsv4, échecs et archives restent intacts ; `summary-v4.json` préserve aussi l’ancien résumé.

CrossHair a été retenté sur la vraie fonction v5 : module absent,exit1. Aucune preuvePython/OS/GPU annoncée. La CI comprend les tests dans le jobPythonPR/push existant, mais aucun runGitHub n’a encore testé ces sources non publiées. Le workflow local archivé contient aussi trois harnaisRust72 préexistants hors périmètre75 ; leurs preuves ne sont pas revendiquées ici.

L’utilisateur autorise désormais du swap global si son attribution hors modèle est comprise.75M conserve son protocole strict existant ; ce n’est pas une interdiction utilisateur absolue. La recherche77 traite séparément l’instrumentation par processus et une éventuelle future politique. Aucun modèle/GPU/TPS relancé pour75M et aucun budget modèle rebasé.
