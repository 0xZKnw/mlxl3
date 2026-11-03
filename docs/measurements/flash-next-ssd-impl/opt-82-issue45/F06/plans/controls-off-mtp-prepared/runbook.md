> Clôture du 25 octobre : ces quatre plans n’ont jamais été exécutés. Aucun gain DFlash F06 qualifié ; runtime retiré. Conserver comme préparation historique, ne pas lancer sur l’identité périmée.

# 82F06 — contrôles OFF puis MTP1 préparés, non lancés

**Tous les plans de ce dossier sont conditionnels.** Ne lancer aucune commande ci-dessous tant que DFlash2 H-C/C-H n’a pas confirmé un gain F06, que le parent n’a pas approuvé et préenregistré ces contrôles, puis revérifié l’identité gelée. Cette préparation n’est ni une mesure ni une qualification. Aucune écriture W, opti ou index. Aucun modèle, build, test ou preuveur lancé.

## Matrice et commandes disponibles

Quatre plans réutilisent le contrôleur 82B `8935` et la chaîne stock observer → sampler49 → supervisor45 → monitor43. Ordre prévu : OFF H-C, OFF C-H, puis MTP1 H-C, MTP1 C-H. Chaque plan contient pré20, quatre processus MAIN/HOLD séparés et post30. Chaque processus produit un warmup 128 puis une mesure 128, dont 127 tokens decode. Un seul modèle physique à la fois. Les ordres inverses sont préparés ; leur lancement reste une décision du parent, jamais automatique.

Les argv de lancement exacts sont dans `manifest.json`, champ `plans[].launch_argv`, à exécuter avec le cwd W indiqué. Ils appellent seulement l’observer stock existant sur le `run.json` concerné. Les commandes internes sont toutes consignées dans les quatre `run.json`. Aucune de ces commandes n’a été exécutée. Les dossiers de sortie sont distincts du DFlash screen/confirm ; ne pas écraser un résultat ou relancer silencieusement.

L’argparse actuel a été lu : OFF n’ajoute aucun drapeau de mode ; MTP1 ajoute uniquement `--mtp`. Aucun `--compare`, `--dflash-draft` ou `--dflash-mode`. Le contrôleur envoie `mtp_depth=1` lui-même. Tous les paramètres `MLXL3_FLASH_DRAFT*`, dont WIDTH, sont retirés du dictionnaire et explicitement supprimés par `/usr/bin/env -u` avant Python : ils ne peuvent pas être hérités. Les flags de prototypes F01/QSA/GDN/batched routing restent également supprimés, comme dans le screen F06.

## Identité et conditions communes

CLI F06 `cbc03630aa08fd8dcdb949b905896ec1dc7f37888a7e984059c12624ec6743e7` ; contrôleur `8935daf3bd69cea3eb1a48aaab5ba5c06d9869cd3e30f614d525c429e980fd76`. Le manifeste reprend les 85 sources gelées et les SHA de tous les drivers, du gold60 et du binaire de tests de référence ; le binaire de tests n’est exécuté par aucun de ces plans. Les contrôles statiques de préparation ne remplacent pas le rehash immédiatement avant/après les éventuelles mesures. Le runner stock ne fait pas respecter automatiquement les champs `engine_sha256`/`benchmark_sha256`.

Même modèle Flash Next 2,05 bpw `65c89531`, cache 3 Gio / 53 slots, allocateur 64 Mio, secteur AC, lookahead 0 / prefetch 0 ; GPU Hot/Wide/UNIQUE/MULTI_TOKEN et tous les flags de backbone conservés du screen. La seule différence H/C est `MLXL3_SSD_TOP10_EARLY_EXIT=0|1`. Aucun draft DFlash n’est demandé ou chargé par ces requêtes.

## Lecture obligatoire après chaque phase

Vérifier la clôture du pré20, des quatre enfants et du post30, puis le join du sampler. Ne pas conclure à partir de quatre rapports alors que la phase est encore active. Dans chaque rapport : `validated`, enfant et lecteur joints, exactement un warmup / une mesure, 128/127 tokens, texte **et** hash gold60 exacts pour toutes les réponses, SHA du vrai CLI exact.

OFF doit avoir MTP inactif/profondeur 0 ; MTP1 actif/profondeur 1. DFlash doit rester inactif, propositions 0 et mode 0 dans les deux cas. MTP1 : travail réellement exercé (`blocks>0`, `proposed>0`), compteurs entiers non négatifs, `proposed==blocks`, `accepted<=proposed`, `blocks+accepted<=127`, profondeur `[0,blocks,0,0]`, lookup 0 ; recouper les compteurs H/C de la même phase. Les éventuels compteurs DFlash sont nuls ou absents, avec aucun travail drafted/verified. Les compteurs de modes différents ne doivent pas être comparés comme s’ils décrivaient le même travail.

Conserver les octets SSD logiques, les durées inclusives, pics physiques, compteurs propres, compression 77B et deltas système séparément. Un ledger swap-in nul ne prouve pas l’absence de swap-out privé ; un swap système ne donne pas son propriétaire. Ne changer aucun seuil stock ni requalifier les échecs de compression. Reporter aussi les dérives warmup→mesure et H-C/C-H sans causalité inventée.

Ces contrôles servent à vérifier OFF et MTP1 avec H 0 / C 1 sur leurs propres témoins frais. Ils ne constituent ni une addition de gains ni une preuve que DFlash2 bat MTP1 ; une comparaison entre modes demande de citer les conditions et leurs trajectoires distinctes. Si DFlash F06 n’est pas qualifié, conserver ces plans en état **non exécuté**.
