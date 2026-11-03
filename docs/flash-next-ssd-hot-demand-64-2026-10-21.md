# Demande Hot perdue dans le repli — expérience64L

21 octobre2026. **Défaut reproduit et corrigé ; qualité physique validée en v6, gain de débit non concluant.** Suite de [64H](flash-next-ssd-early-hot-64-2026-10-21.md), dont le gain de débit est rejeté. Le correctif est publié dans la PR34 au commit `aa5bc75f` ; UNIQUE reste désactivé par défaut.

Le repli termine le calcul Hot de toute la fenêtre puis réutilise ses valeurs, même si leurs anciens slots sont réaffectés. Il réserve uniquement Cold. Cette optimisation omettait un effet de `ExpertCache::reserve` : mettre à jour la fréquence et l'ancienneté des experts effectivement demandés. Des experts Hot utilisés pouvaient donc être évincés à la place d'experts moins utilisés.

Le nouveau test sur la production v4 inchangée échoue précisément sur cette rétention LFU : un test sélectionné,0passé/1échoué, processus joint1,696s, postcontrôle30s passé et0nouveauSwapout. [Reproduction et log](measurements/flash-next-ssd-impl/opt-64-unique-cohort/lfu-reproduction/), [identité source/binaire](measurements/flash-next-ssd-impl/opt-64-unique-cohort/identity-lfu-repro.json). Seul le test change par rapport au manifeste v4 ; la reproduction conserve `parity:false`.

La correction utilise la réservation existante sur les seuls experts Hot sélectionnés, encore présents dans le snapshot stable, avant que Cold choisisse ses victimes. Toutes ces réservations doivent être des hits ; la liste vide est ignorée. Le chemin où l'union entière tient réservait déjà tous ses experts et ne reçoit pas ce deuxième appel. Aucune lecture SSD ni écriture de banque supplémentaire, aucun nouveau kernel, cache toujours3Gio/53slots.

Le test est étendu à32cas : rétention Hot quand des victimes moins utilisées existent, éviction imposée par la capacité avec réutilisation exacte des valeurs terminées, snapshot non vide sans expert Hot sélectionné, F32 contrôlé/F16 avec routes réelles, pending0/4, UNIQUE0/1, erreur I/O et reprise. Les comptes Cold attendus sont55records avec cinq experts Hot sélectionnés,60si les résidents sont tous inutilisés. Les sorties sont comparées à un oracle séquentiel indépendant, et les formes/nombres de valeurs sont contrôlés.

La première campagne v5 échoue dans la préparation du cas sans Hot sélectionné : le test choisissait comme prélecture un expert déjà primé hors fenêtre, puis faisait `unwrap()` sur le refus. L'échec et le postcontrôle30s restent conservés. v6 exclut ces cinq experts de la liste des candidats pending, sans changer la production ni les assertions. La CLI v5/v6 est identique ; seul le fichier du test change, avec Clippy et compilation native repassés.

La qualité v6 passe :32cas MoE en1,786s,36préfixes du vérificateur DFlash avec logits complets/états en25,971s,43contrôles du vrai bridge en305,204s. Trois processus joints et postcontrôle30s passé,0nouveauSwapout sur l'origine qualitév3 conservée. [Résultat complet](measurements/flash-next-ssd-impl/opt-64-unique-cohort/quality-lfu-v6/campaign-result.json). Les contrôles numériques MTP v4 sont réutilisés : à53slots,B1/B2 ne traverse pas le repli modifié. Les benchmarks de débit utiliseront cependant des témoins OFF/MTP neufs.

[Identité v6](measurements/flash-next-ssd-impl/opt-64-unique-cohort/identity-v6.json) :146sources/deux artefacts,archive `ROOT/build/flash-opt-64/lfu-v6/source.tar.gz` ; CLI `58b76586709d04028e05517b5dd827d4306f2063f398e59e328ae5b1d96d359a`,natif `7d89e963267fe8390fdc6cec19724abd67b62a523b3400f351c693db6662ec7f`.

Les contrôles locaux déjà exécutés passent : rustfmt,Clippy strict,115tests CPU/3ignorés et144tests MLX/chat/144ignorés ; les tests physiques sont sélectionnés séparément. Kani retenté sur le vrai chemin Rust échoue avant obligations dans `intrinsics.rs:243`, chemin GPU **non prouvé**. Le helper CPU inchangé garde sa vérification bornée20IDs0..511/capacité0..512/unwind21 ; cela ne prouve ni80IDs, niMLX/Metal/concurrence.

Le protocole préenregistré est DFlash2Q4/ring/adaptatif7,MAIN/HOLD,128sorties/127decode,HCCH UNIQUE0/1 dans le même binaire ;63/Hot/Wide1,cache3Gio,4lecteurs,allocateur512Mio,secteur/applis ouvertes. Seuils≥1%chaque paire, dérive≤5%, étendue≤3points. Empreinte physique≤12milliards d'octets,0nouveauSwapout depuis l'origine qualitév3 conservée, incoming256Mio par fenêtre courte,pré20/post30.

Les huit processus DFlash passent les contrôles de sortie et le postcontrôle30s, avec un pic physique de8 259 656 256octets et aucun nouveau Swapout. Les seize réponses (chauffes comprises) sont exactes. Le débit reste **non concluant** :

| Prompt | H1 UNIQUE0 | C1 UNIQUE1 | C2 UNIQUE1 | H2 UNIQUE0 | Gains appariés |
| --- | ---: | ---: | ---: | ---: | --- |
| MAIN | 10,649 | 10,619 | 10,743 | 10,424 | −0,283% ; +3,059% |
| HOLD | 9,544 | 8,433 | 9,849 | 9,584 | −11,635% ; +2,768% |

Unités tok/s. La dérive des témoins est2,155%MAIN/0,418%HOLD ; l'étendue des gains dépasse3points (3,341/14,403). L'étiquette automatique `inconclusive_drift` désigne ici la dispersion des gains, pas une dérive des seuls témoins supérieure à5%. Le8,433 est conservé. Ce bras a67cycles/59acceptés contre55/71pour l'autre candidat HOLD, malgré140lignes cible dans les deux : l'ordonnancement adaptatif change le nombre de cycles. Ces compteurs ne prouvent pas une cause GPU exclusive.

[Débits et contrôles mémoire](measurements/flash-next-ssd-impl/opt-64-unique-cohort/speed-result-lfu-v6.json), [tous les bras](measurements/flash-next-ssd-impl/opt-64-unique-cohort/speed-A-lfu-v6/), [clôture](measurements/flash-next-ssd-impl/opt-64-unique-cohort/closure-lfu-v6.json).

Les témoins frais128 sont OFF10,268→10,725 MAIN et10,110→10,281 HOLD ; MTP1 11,086→11,152 MAIN et10,101→10,310 HOLD. Les quatre premiers et les deux MTP finaux passent. Le premier OFF final a été interrompu par `BlockingIOError: [Errno35] Resource temporarily unavailable` lors d'un lancement de processus ; le rapport et son postcontrôle échoué restent `parity:false`. Après récupération sans modèle, la relance explicitement demandée des **deux OFF manquants seulement** passe, avec l'origine Swapout initiale conservée et postcontrôle30s. Aucun témoin ne remplace un résultat DFlash défavorable ; aucun gain n'est qualifié.

L'[analyse64D des hits, octets et largeurs](flash-next-ssd-cache-width-64-2026-10-21.md) observe environ2,3–2,4% de records demandés en moins, sur des traces64sorties distinctes. Elle ne reconstruit pas les histogrammes absents des statistiques CLI128 et ne valide pas un gain de débit. L'application installée reste inchangée.

L'audit de portée64A confirme que les240/144splits historiques appartiennent au décodage. Ils représentent5grandes fenêtres sur17–18passages cible MAIN et3sur22HOLD ; leurs durées de vérification hôte inclusives pèsent55,5–56,2%/35,4–35,8%. Ces fractions ne représentent ni du GPU exclusif, ni du surcoût intégralement supprimable. [Histogrammes, durées et SHA des quatre traces](measurements/flash-next-ssd-impl/opt-64-unique-cohort/scope-audit-64A.json). Le chiffre90%reste conditionnel aux seuls splits, pas à toutes les vérifications.
