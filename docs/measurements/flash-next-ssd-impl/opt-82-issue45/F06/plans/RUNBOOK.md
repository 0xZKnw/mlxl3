# 82F06 — plans futurs, non exécutés

25 octobre2026. Reprise après interruption : ces plans n’existaient pas encore. Ils sont préparés à partir des plans82F01 et des noms/impressions du patch82F06, sans exécuter de test, preuve, build, modèle ou benchmark. Aucun W/index/opti modifié.

Les quatre fichiers de lancement sont quality/run.json, bridge/run.json, screen/run.json et confirm/run.json. commands.json conserve les commandes observe-stock45/49 ; elles ne doivent pas être lancées tant que le parent n’a pas satisfait gates-before-gpu.json et figé identity-template.json. Les SHA source intégrée, CLI et tests restent null. Ne pas reprendre les identités19c9/8165 propres àF01.

## Base expérimentale

Le flag mesuré est uniquement MLXL3_SSD_TOP10_EARLY_EXIT : H=0, C=1. Le contrôleur de débit82B publié est identifié par son SHA actuel8935daf3bd69cea3eb1a48aaab5ba5c06d9869cd3e30f614d525c429e980fd76 ; il doit être recontrôlé au gel.

F01 n’est pas activé par cette préparation : sa variable est absente du JSON et explicitement supprimée de l’environnement de chaque enfant avec /usr/bin/env -u. Même protection pour80Q,80G et82F05. Cela évite qu’un shell conservant un flag transforme le test en combinaison. Le parent décidera après la clôtureF01 : retirer F01, le garder désactivé, ou conserver un gain validé comme base commune explicitement décrite. Dans ce dernier cas, mettre à jour les deux bras ET leurs préfixes env avant tout lancement, puis refaire les identités. Aucun résultatF01 ne préjuge du gain F06.

## Préconditions CPU et mutations

Format/lint/build et contrôlesCPU pertinents appartiennent à l’intégration réelle du parent ; cette préparation n’en lance aucun. Le testCPU du domaine et Kani sur le vrai garde doivent passer. La tentativeMLX réelle doit avoir été effectuée et son résultat inspecté avantGPU ; ICE/FFI/timeout restent NON VÉRIFIÉS, sans bloquer par une prétendue preuve ou créer un stub. Les commandes et bornes180s/4Gio sont reprises de commands.json voisin ; le chemin du runner borné reste null jusqu’à identification. Arrêter et joindre tous les proveurs/compilateurs avantGPU.

La mutationCPU doit être détectée puis restaurée. La mutationGPU est une étape séparée : composant seul sur candidat normal, binaire jetable muté, échec attendu de l’oracle indépendant, restauration puis gel du vrai binaire. Ne pas lancer tout quality/run.json pour insérer une mutation au milieu : la matrice complète suppose ces étapes préparatoires terminées. Son premier composant normal est le contrôle requis après restauration. Aucun de ces résultats n’existe encore.

## Qualité minimale affectée

quality : pré20,256cas top10 contre triCPU indépendant/scores anciens, erreurs et six domaines inchangés,384routes réelles48couches, loaderdéfaut/0/1 dans des processus distincts, neufCLIinvalides externes (huitASCII,unbyte0xffPOSIX),36préfixes/logits/captures/états DFlash, sessionMTP aux quatre budgets, post30. Exactement8 invocations natives sélectionnant chacune1test, plus neufCLIinvalides attendant exit2 et le message requis. Aucune fixture ne modifie l’environnement. Le caractère loadonce est établi par lecture du call graph (load seul lit la variable, route lit son champ), sans prétendue validation dynamique post-load. --test-threads=1 ne neutralise pas les workers MLX.

Les casts253962/240QMV/64MoE spécialisés F01 et l’oracle176 de calculEXL3 ne sont pas répétés : F06 ne change ni poids, QMV, H128, scales ou réductions expertes. Les suitesCPU nouvelles sont celles de l’intégration ; ce plan n’ajoute pas une répétition physique de la suite native historique. Les régressions I/O/leases historiques restent disponibles dans commands.json voisin si une modification de portée les affecte ; elles ne sont pas artificiellement comptées comme de nouveaux contrôles exécutés.

bridge :43requêtes réelles avec flag1/ADAPTIVE1, WIDTH absent du JSON ET supprimé de l’environnement hérité. Exiger checks.json status=complete, parity=true, checks=43, annulation/reprise et erreurs attendues ; vérifier aussi enfants/sampler joints, VMcourbes et mémoire des événements. Le JsonProcess actuel contrôle ces requêtes mais ne certifie pas tous les événements tardifs de fermeture : conserver cette limite de l’outil, exit0 ne la supprime pas. Aucun débit avant qualité entière.

## Débits

screen : MAIN H→C puis HOLD H→C. confirm : MAIN C→H puis HOLD C→H, conditionnel au signal et à la décision parent ; ce n’est pas un lancement automatique. Un CLI immuable commun aux bras, seul le flagF06 diffère. DFlash fixe B2, WIDTH1/ADAPTIVE0, Q4/ring,128warm+128mesure/127decode, cache3Gio/53slots, alloc64Mio, Hot/Wide/63/UNIQUE1, prélecture/LES/shared/stable0.

Comparer les rapports validated/reaped/reader_joined et leurs réponses non vides à golden60 (path+SHA dans l’identité), nombres128/127, finitude, modes, acceptation, largeurs et SSDlogique. Les expériences ne s’additionnent pas. Conserver toutes les paires/dérives ; contrôlesOFF/MTP1 frais requis par82T avant promotion, non préparés comme une nouvelle campagne automatique ici.

Gardes45/49 inchangées : alimentation stable, pré20/post30, un modèleGPU, aucune compilation/proof concurrente, délais bornés et enfants/samplers joints. DeltasVM système et RSSRUSAGE_CHILDREN ne sont pas un footprint ni une attribution au modèle. Inspecter séparément mémoireprocess≤12Go, ledger, compression77 ; caps77B historiques restent FAILED s’ils sont dépassés. Conserver run-result/logs/system-samples/vm-curve et échecs partiels. Aucun seuil n’est relevé par ces plans.

Seule la cohérence statique des données a été contrôlée ici : JSON lisible, noms présents dans les copies sources, marqueurs référencés, hashes des outils actuels, brasH/C ne différant que par le flag. Aucun lancement, aucune parité numérique ni vitesse nouvellement validée.

Révision sécurité25octobre : les fixtures mutantes d’environnement ont été retirées avant toute exécution. Défaut/0/1 etinvalides sont désormais des processus distincts dont env est fixé au démarrage. Le byte0xff est représenté dans leJSON par \udcff, décodé parPython puis encodé POSIXsurrogateescape lors de Popen ; cette branche reste non exécutée et doit conserver tout échec de lancement explicite. ProductionF06 inchangée.
