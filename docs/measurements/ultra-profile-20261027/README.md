# Profilage Flash Next M5 — résultats et limites

Objectif : localiser les coûts avant tout nouveau portage Sushi/TensorFold. Modèle EXL3 2,05 bpw, M5 standard 24 Gio, MLX 0.32.2, cache expert 3 Gio / 53 slots par couche. Aucun nouveau boost intégré ; app installée inchangée.

Les trois qualifications courtes AR/MTP/DFlash2 passent (17 sorties, logits et états complets finis/byte-exacts, y compris les KV draft). MAIN est terminé dans les trois modes sur 128 sorties / six phases. HOLD est également terminé dans les trois modes. Les trois mesures synchronisées des couches sont également terminées. Le reste de la matrice est préparé et reste non mesuré. Voir [matrice](full-matrix.json), [CLI](cli-matrix.json) et [journal](../../../opti.md).

## Premier résultat : MAIN AR128

Trois traces naturelles, 6 096 cohortes par trace. Attente d’évaluation/amont des routes : 37,97–38,81 ms par sortie ; copie CPU : 0,00558–0,00591 ms. Lots de lectures SSD : 29,00–30,29 ms de temps propre hôte par sortie ; soumissions MLX : 17,03–18,51 ms. Ce sont des horloges hôte : ni l’attente GPU ni les lectures ne représentent du GPU exclusif ou une inactivité du GPU. Ne pas sommer les durées imbriquées ou les lectures de travailleurs simultanées. Détail : [répartition](first-AR-breakdown.json).

709–720 cohortes entièrement Hot sur 6 096 (11,63–11,81 %). Ce taux ne valide pas à lui seul un portage de la spéculation complète Sushi ; la réutilisation partielle Hot existe déjà.

Les contrôles AR avant/après donnent 11,96 et 14,24 s de decode : dérive conservée, aucun gain causal déduit. Les oracles complets en RAM et l’instrumentation influencent la campagne ; les mesures CLI sans ces oracles seront distinguées.

## Preuves et limites actuelles

[Export MAIN seulement](MAIN-only-export-v2/result.json) : 2 828 169 événements, 77 982 readbacks et 1 143 pas ; le statut ne couvre pas la matrice entière. L’export v1 échoué est conservé. JSON bruts archivés en gzip avec SHA-256 du contenu vérifié.

La première fenêtre a passé les trois modes MAIN sans nouveau swap. Le lancement HOLD suivant a été arrêté avant mesure après 218,4 Mio de swap apparus entre les essais ; garde et données conservées. La fenêtre w2 est explicitement séparée, après libération de RAM et fin des travaux CPU. Les gardes restent secteur, absence de build/prover/autre inférence, zéro nouveau Swapout et mémoire physique native sous 12 Go.

Vérification : suites Rust MLX/chat 278 passés / 167 ignorés, CPU/chat 248 / 6 ; Python 384 passés avant le dernier correctif d’attribution, 214 après, puis 101 exports après correction lint. Kani vérifie le contrat CPU des huit compteurs u64/usize arbitraires sans hypothèse (4 couvertures) ; interning interrompu à 4 Gio RSS, MLX réel bloque sur un ICE, CBMC ne parse pas les en-têtes C++ Apple, CrossHair absent. Aucun de ces résultats ne prouve Metal, la concurrence ou la génération complète. CI distante du code non commité non exécutée.

## Nouvelle priorité : attribution CPU/GPU et chemin critique

Les CLI release AR/MTP/DFlash2 sont terminés : six réponses mesurées et deux chauffes par mode, 128 tokens, hashes/textes/comptes conformes, processus/lecteurs/samplers joints, zéro Swapout dans leur fenêtre. AR : 10,77–10,99 tok/s. Les mesures ne constituent pas un gain d’optimisation livré. Les piles CPU MTP et DF ont été capturées sur la première chauffe ; les tentatives AR ont échoué et restent documentées.

L’utilisateur demande maintenant la séparation CPU actif, synchronisation, travail GPU et GPU idle, puis lectures critiques/masquées et plafonds. Les contextes longs et largeurs supplémentaires sont donc explicitement non lancés, la matrice globale reste incomplète.

Sur les trois traces AR, l’union des fonctions de lecture entre travailleurs vaut 28,52–29,77 ms par sortie, contre 29,00–30,29 ms pour le bloc I/O complet ; le reste du bloc vaut 0,48–0,52 ms. Ce reste inclut préparation, évaluations éventuelles, ordonnancement, retour et journalisation. L’union des lecteurs ne mesure pas le recouvrement GPU, et les fonctions de lecture incluent l’OS. Aucun plafond de gain réel n’est déduit de leur seule durée.

Xcode 27.1 est installé depuis l’App Store officiel et l’utilisateur a explicitement accepté son accord. Après confirmation de l’utilisateur, xcodebuild -checkFirstLaunchStatus et xcrun xctrace version passent : Xcode est maintenant initialisé et la licence enregistrée. Les captures antérieures par le CLI public restent documentées. Une capture CPU/GPU ciblée a passé les contrôles numériques et zéro nouveau swap. Aucun algorithme ni kernel n’a changé pour cette nouvelle attribution. Les métriques Metal rétrospectives ne fournissent pas de remplacement au profilage headless.

L’[attribution détaillée et le classement des optimisations](attribution/README.md) séparent les axes CPU/GPU sur20tokens complets, avec36tokens d’une première capture en corroboration. Résultat principal : routes presque entièrement concomitantes à de l’activité GPU ; environ12ms/token de CPU actif pendant un GPU inactif à l’intérieur des soumissions ; chemin des lectures d’experts largement exposé. **La part physique SSD reste non mesurée** : les événements bloc sont absents même sous deux témoins positifs. Les plafonds conditionnels ne sont pas des gains validés. Les échecs de collecte et les anciennes conclusions restent dans le journal et les fichiers bruts.
