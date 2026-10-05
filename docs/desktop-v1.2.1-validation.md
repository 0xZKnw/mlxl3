# MLXL3 Desktop1.2.1 — validation du 5 octobre 2026

## Périmètre

Correction de la frontière catalogue CLI/GUI : `ModelSummary.gated` Rust expose la valeur JSON de Hugging Face, alors que le décodeur Swift attendait strictement un Bool. Accepter absent/null comme accès ouvert, booléens inchangés et modes auto/manual comme accès restreint ; chaînes legacy true/false également acceptées. Types et modes inconnus restent refusés. La [documentation ModelInfo](https://huggingface.co/docs/huggingface_hub/v0.30.2/en/package_reference/hf_api) confirme les modes d'approbation.

Sampling : à la dernière instruction de l'utilisateur, conserver le verrouillage MTP et les valeurs greedy0/1/1 existants ; ajouter uniquement une explication FR/EN. Les modifications expérimentales de repli/persistance ont été retirées avant publication. Aucun changement des algorithmes du moteur, de la quantification ou des poids des modèles.

## Reproduction et régression

`build/engine-v1.2.1/runtime/mlxl3 hub search qwen3.6 --limit 60` :60 résultats, plusieurs `gated:null`, stderr vide, aucune inférence. Type Swift `HubModel` original extrait des sources et compilé Swift6, puis décodage du payload réel : échec `DecodingError.typeMismatch`, chemin `[0].gated`, Bool attendu/null reçu. Même compilation et mêmes données avec le décodeur corrigé :60 modèles décodés. Logs et réponse conservés dans `docs/measurements/desktop-v1.2.1/`.

Premier prototype refusé à la compilation pour placement de `try` dans une expression `||` lançant une erreur. Syntaxe corrigée avant validation ; log négatif conservé, aucun résultat tiré de cet échec.

`swiftc -swift-version 6 -warnings-as-errors -parse-as-library -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk -module-cache-path build/desktop-v1.2.1/modules -I apps/MLXL3Studio/.build/out/Products/Debug <sources Desktop hors MLXL3StudioApp.swift> apps/MLXL3Studio/.build/out/Products/Debug/SwiftMath.o tests/studio-hardening-check.swift -o build/desktop-v1.2.1/hardening-check`, puis `build/desktop-v1.2.1/hardening-check "$PWD/tests/fake-desktop-engine.py"` : réussis.

Couverture :8 formes d'accès légitimes,5 valeurs/types invalides ; transport CLI réel et état `ModelLibrary`, résultats/détails/variante, recherche vide, subprocess en erreur, JSON invalide, recherche lente annulée et nouveau résultat conservé, reprise après échec, cache, plus et rafraîchissement. Régressions lifecycle/historique/outils/MTP et Markdown existantes également exécutées. Fixtures et préférences/données de test jetables, aucun poids chargé.

`python3 -m py_compile tests/fake-desktop-engine.py`, smoke-test `hub search qwen3.6 --limit 60`, `git diff --check` : réussis. Aucune modification Rust/Metal/FFI ; Kani ne vérifie pas le décodeur Swift. Aucun vérificateur formel de Swift/JSONDecoder disponible dans l'outillage du projet : compilation Swift6 et tests déterministes finis, sans revendication de preuve complète. Pas de nouveaux benchmark ni mesure de consommation/thermique.

## Progression des téléchargements

`ModelDownloadProgress` consomme les événements natifs de progression avec `ProcessInfo.systemUptime` ; première observation servant de référence, débit calculé à partir de 0,5 seconde entre observations, remise à zéro du calcul si compteurs ou temps reculent. Les valeurs non finies/négatives sont refusées, la fraction affichée est bornée à0…1, la taille inconnue reste indéterminée. `ModelLibrary` réinitialise le débit à chaque reprise et associe les callbacks au transfert courant pour ignorer des événements retardataires.

`swiftc -swift-version 6 -warnings-as-errors -parse-as-library -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk -module-cache-path build/desktop-v1.2.1/modules apps/MLXL3Studio/Sources/MLXL3Studio/ModelDownloadProgress.swift tests/download-progress-check.swift -o build/desktop-v1.2.1/download-progress-check`, puis exécution : **réussis**. Cas attendus indépendants du calcul : reprise avec4Go déjà conservés,200Mo/s attendus pour100Mo en0,5s, stagnation à0, reset, temps inversé, taille inconnue, NaN/infini/négatifs, dépassement numérique. Matrice de300 combinaisons finies de compteurs/totaux/temps vérifiant fraction bornée et débit fini positif ou nul ; aucune preuve formelle de Swift revendiquée.

Contrôle négatif : copie temporaire du calcul incluant à tort les octets conservés dans le premier débit, test **échouant comme attendu** ; sources de production jamais mutées, puis test réel relancé et réussi (`download-progress-mutation.log`).

Hardening-check final, mêmes sources/commande complète Swift6 ci-dessus : **réussi**, incluant téléchargement NDJSON via vrai subprocess factice, apparition de la progression et du débit, pause conservant les octets, reprise avec débit réinitialisé, succès à100% et message d’erreur après subprocess en échec. Pas de poids ni de registre utilisateur modifié. Ces contrôles sont ajoutés à `scripts/check-desktop.sh` exécuté par la CI complète.

[Prévisualisation des vues SwiftUI de production](measurements/desktop-v1.2.1/download-progress-preview.png) : transfert, pause/reprise et fin, sans découpage visible du contenu. Les86,4Mo/s,72% et volumes de cette image sont **des données de fixture**, pas une mesure réseau. Deux erreurs du harnais de rendu (source Theme dépendant de EngineState absente, puis clé d’environnement SwiftUI non modifiable) corrigées dans le harnais uniquement ; compilation finale avec toutes les sources et rendu réussis. Le flux réel de téléchargement de fichiers lourds et les variations réseau prolongées restent non exécutés localement à la demande d’éviter de chauffer le Mac ; aucune inférence ni campagne de benchmark.

## Livraison

Préparation Desktop1.2.1/build20, moteur1.2.1/protocole1/macOS≥26.2/arm64. Les suites CI complètes, le bundle/DMG, leurs signatures/empreintes et la publication restent **en attente à cette étape**. Le défaut d'activation à chaud du moteur reste explicitement différé ; les tests de cette correction ne prétendent pas le couvrir.
