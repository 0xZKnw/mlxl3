# Sushi : routes GPU, vérification GDN et cache experts

27 octobre 2026. Baseline locale : Qwen3.8 Flash Next EXL3 2,05 bpw, M5 standard/24 Gio, cache experts **3 Gio**, HEADONLY conservé après retrait du kernel Mia. Recherche préenregistrée SUSHI-2026-10-27-READ, puis diagnostic SUSHI-2026-10-27-SYNC demandé par l'utilisateur. La priorité est le decode ; le cache de préfixes reste une piste distincte pour le TTFT des conversations répétées.

## Sources et portée

[Sushi fc811352](https://github.com/beamivalice/sushi/tree/fc811352cad5a9993fb57e12b786d3862cbd84af), commit du 27 octobre à 14:37:23Z. Comparaison avec notre ancien pin ad562c4078a62d9d0a562add7e109cbc330de96b : 144 commits d'avance, arbres complets, liste API de fichiers plafonnée à 300. Treize fichiers collectés, régions pertinentes relues : [URLs, blobs Git et SHA-256](sources.json). Les sources complètes sont dans build/sushi-review/fc811352cad5a9993fb57e12b786d3862cbd84af/, hors Git. Les treize blobs actuels et trois anciens fichiers de code sont vérifiés octet par octet. Le connecteur renvoyait un contenu vide pour transformer.zig (2,75 Mo) ; lecture brute à la même révision et vérification de son blob. Ce n'est pas un audit complet du dépôt.

Les mécanismes de spéculation Qwen ci-dessous étaient **déjà présents au pin ad562**. Ce qui reste à évaluer est leur différence avec notre chemin actuel, après les optimisations 56/63, pas un nouveau pourcentage amont à ajouter. Les chiffres M5 Max/128 Go de Sushi ne décrivent pas notre matériel, notre quantification ni notre cache de 3 Gio. Leur budget résident comprend aussi tronc, tête MTP et espaces de travail. Aucun moteur Sushi exécuté ici.

## Comparaison des chemins réels

| Étape | Sushi | MLXL3 actuel |
| --- | --- | --- |
| Carte de cache | `Engine.specRoute` rafraîchit une carte expert→slot et fournit les banques ; les experts non prêts sont marqués absents. | `prepare_resident_hot` capture les slots prêts, construit le lookup GPU et soumet Hot avant `selected.to_u32()`. |
| Calcul pendant le retour des routes | Experts de la carte GPU et expert partagé soumis avant l'attente hôte. | Hot et partagé sont déjà soumis tôt. Le correctif FFI56 attend l'événement propre aux IDs contigus, sans primitive `contiguous()` tardive derrière Hot. |
| Acceptation du calcul anticipé | `specMatches` exige mêmes nombres d'IDs/slots, aucune union workspace, et **chaque** ID résolu dans le même slot que la carte. Sinon reconstruction exacte. | Les résultats Hot restent utilisables même avec des misses ; seuls les Cold sont lus/calculés, puis le résultat est remis dans l'ordre original du top-k. Ce n'est pas un chemin « tout ou rien ». |
| Couche suivante | Qwen peut laisser une couche non résolue et soumettre la couche GDN suivante avant de vérifier la précédente. | Construction encore séquentielle : retour des routes, réservation LFU, lecture des misses, assemblage exact, puis couche suivante. Pas de rollback GDN entre couches pour ce motif. |
| Mise à jour du cache | Les routes du successeur spéculatif ne remplissent ni ne touchent LRU tant que le prédécesseur n'est pas validé. | `reserve` et les écritures de banques sont actuels et exacts ; les déplacer avant validation nécessiterait une nouvelle transaction, pas seulement un appel `async_eval`. |

Sources primaires : [specRoute](https://github.com/beamivalice/sushi/blob/fc811352cad5a9993fb57e12b786d3862cbd84af/src/expert_stream.zig#L2213), [specMatches](https://github.com/beamivalice/sushi/blob/fc811352cad5a9993fb57e12b786d3862cbd84af/src/expert_stream.zig#L1654), [ordonnanceur Qwen](https://github.com/beamivalice/sushi/blob/fc811352cad5a9993fb57e12b786d3862cbd84af/src/qwen4_forward.zig#L1967), [contrats documentés](https://github.com/beamivalice/sushi/blob/fc811352cad5a9993fb57e12b786d3862cbd84af/docs/engine-expert-streaming.md#L187). Côté local : native/src/ssd_moe.rs:513–560 et :727–1028 ; [historique du correctif56](../../performance/flash-dflash/flash-next-ssd-strata-56-2026-10-20.md).

### Ce que protège réellement le rollback

Le garde de Sushi impose batch=1 et seq=1 ; captures, profilage, imatrix et diagnostics forcent la vérification synchrone. Le successeur doit être GDN + MoE **sans PLE**. Les frontières QSA/PLE vérifient d'abord. Ce mécanisme ne s'applique donc pas directement à une vérification MTP/DFlash de plusieurs lignes ; leur support DFlash2 Qwen est distinct et possède [son propre test](https://github.com/beamivalice/sushi/blob/fc811352cad5a9993fb57e12b786d3862cbd84af/tests/test_dflash2_qwen.sh).

En cas de miss, `verifyQwenMoe` rétablit le flux HC et la gate d'injection de la couche précédente avec son résultat exact, détruit l'écriture HC différée invalide, puis le caller restaure le successeur et recommence ce même indice de couche. [StreamGdnUndo](https://github.com/beamivalice/sushi/blob/fc811352cad5a9993fb57e12b786d3862cbd84af/src/transformer.zig#L56168) conserve les handles convolution/état récurrent et le drapeau initialized ; la restauration transfère leur propriété. Ce sont des handles à préserver correctement, pas une nouvelle exécution supposée équivalente. Les banques du successeur restent distinctes de celles du prédécesseur ; la résolution de routes incorrectes ne doit pas contaminer le cache.

## Diagnostic local : état

Le premier prévol a attendu la fin d'un build Swift. La première tentative GPU a été arrêtée à 57,621 s : **1 279 918 080 octets** de nouvelles écritures swap, enfant joint, deux phases sans trace seulement, aucun rapport complet. L'utilisateur a ensuite signalé jouer simultanément avec Qwen3.6 35B A3B. Aucun temps de cette tentative n'est retenu. [Clôture r1](sync-r1-closure.json).

Après son signal « c'est bon », reprise r2 au même binaire/source/options ; [identité](sync-identity.json), [nouveau prévol](sync-r2-preflight.json), plans et données dans sync-r2/. Test natif existant, sans nouveau code : MAIN/HOLD, 128 sorties ×6 phases par prompt (warmup, contrôle avant, trois traces naturelles sans fences supplémentaires, contrôle après). Chaque phase compare IDs, 248 320 logits et tous les états complets non vides/finis bit à bit à la première ; les IDs/texte/hash sont aussi comparés au golden historique externe128. Une seule instance du modèle, secteur, quatre lecteurs, LFU, allocateur64 Mio, zéro nouveau Swapout.

**MAIN r2 terminé et validé.** Un test sélectionné passe /336 filtrés,153,132s, enfant joint et zéro nouvelle écriture swap. Le validateur53 contrôle aussi les six phases/golden128, les compteurs par couche, les octets lus et l'imbrication des événements. HOLD r2 a été interrompu au chargement par un build Swift ; aucune mesure. La préparation de reprise r3 rencontre ensuite un autre build, avant tout lancement. Les échecs et l'erreur de chaînage du wrapper sans ancre sont conservés dans sync-r2-hold-closure.json et sync-r3-hold/preflight-blocked.json. Pas de relance MAIN pour bruit.

| MAIN, trois traces naturelles | Valeurs observées |
| --- | --- |
| Decode instrumenté,127 pas émis | 14,267–15,155 s |
| Attente des IDs, total par pas émis | 52,534–52,620 ms |
| Cohortes entièrement prêtes | 709–720 /6096, soit11,63–11,81% |
| Entièrement prêtes + successeur GDN sans PLE | 575–589 /6096 |
| Attente des IDs de ces seuls cas | 4,507–4,563 ms par pas émis |
| Ce poste rapporté au temps de la trace | 3,777–4,061% |
| Appels `disk_read_batch` par pas émis | 26,54–28,30 ms |

[Calculs par trace](main-sync-opportunity.json), analyses standards et résumés sous sync-r2/AR-MAIN/. Le test vérifie tous les états non vides/finis et leur égalité, pas seulement le texte. Pour le calcul conjoint, chaque `route_readback_inclusive` est relié à son parent routeur puis à `moe_forward_rows` ; rows=1/capacité53/dix IDs et cohérence hits carte/CPU sont assertés. Le successeur est lu dans le config local : linear_attention, sans PLE, indice<48. Les6096cohortes et les comptes de la validation53 sont croisés.

Les attentes sont **inclusives du travail GPU amont restant**, pas exclusivement du transfert ni du GPU inactif. La suppression idéale du seul poste éligible représenterait environ4% de ces traces ; ce n'est ni une prédiction de gain ni une borne universelle pour un autre ordonnanceur. Il faut encore construire/soumettre le travail suivant, protéger les états et payer les reconstructions. Parmi les4318transitions vers GDN admissibles par trace, seuls13,3–13,6% sont entièrement Hot ; un port qui tenterait toutes ces transitions rencontrerait donc des misses dans environ86% des cas observés. Une politique sélective serait une autre expérience. Les opérations QSA/PLE restent hors garde. Ne pas multiplier un hit moyen par expert pour inventer une probabilité de cohorte prête.

L'instrumentation a un coût : contrôles non tracés MAIN13,964–14,119s, traces14,267–15,155s. Tous les échantillons sont conservés ; pas de pourcentage de débit de production qualifié. Les5376–5387sondes d'I/O par trace voient Hot encore en vol au départ dans5336–5359cas, et parfois encore en vol au retour dans723–842cas. Les lectures n'attendent donc pas systématiquement sa fin ; ces sondes ne chronomètrent pas son exécution GPU exclusive. Les attentes terminales de tête/drain et la préparation CPU existent aussi ; les scopes parents/enfants ne sont pas additionnés. Le scope fallback `previous_bank_users_wait` est absent sur ce parcours Hot ; `prepare_resident_hot` conserve néanmoins un `last.eval()` non isolé avant son propre scope. Le temps propre restant du parent routeur, incluant ce prélude et la construction du routeur, est0,269–0,354ms/pas : c'est une enveloppe, pas la mesure isolée de ce `eval`. On ne conclut pas « zéro synchronisation de banques » de ce seul scope absent. Le driverAR préremplit ligne par ligne ; aucun TTFT de production n'est déduit de ces mesures.

**Décision avant portage :** la différence GDN mérite d'être distinguée du Hot déjà présent, mais ces données ne justifient ni une reprise aveugle, ni15tok/s promis. Le mécanisme upstream seq=1 ne couvre pas directement MTP/DFlash multiligne. Terminer le contrôle HOLD lorsque la machine est libre ; aucun gain decode et aucun prototype de rollback ne sont déclarés validés. Le pool global reste à simuler séparément, à budget fixe.

## Le pool TensorFold reste une piste différente

Le [pool CLOCK/leases TensorFold](https://github.com/ashhart/TensorFold/blob/b30d5b9d36786ff778b4bb7a7e1c4b5d26de126e/zig/src/core/fsr/cache.zig) partage dynamiquement les banques entre couches. Notre répartition reste 48×53=2544 slots ; le profil statique47/55 rejeté n'est pas ce mécanisme. Sushi utilise ici des caches LRU par couche ; son prêt/restauration de pools pour le prefill DeepSeek n'est pas un cache global interchangeable pour Flash Next.

Le premier filtre utile pour TensorFold serait un replay des routes au même budget et avec une politique inchangée, puis une comparaison séparée de CLOCK et LFU. Compter octets, demandes dédupliquées, cohortes prêtes et leases ; expliciter l'état initial et contrôler HOLD. Pas de nouvelle calibration du profil statique. Les traces r2 peuvent servir à cette étude, mais un replay ne prouverait ni correction GPU ni gain de decode.

**Interaction à traiter avant un port combiné :** la sûreté du rollback Sushi repose explicitement sur les banques distinctes. Un pool global peut réattribuer un slot lu par un calcul Hot encore en vol. Garder un handle MLX sur la banque ne protège pas son contenu contre une écriture CPU. Il faut protéger les slots/GPU consumers, les générations de maps et la publication après lecture complète, y compris quand les routes CPU ne sont pas encore connues. Ne pas introduire les deux mécanismes ensemble sans isoler leur effet. [Revue TensorFold et antécédents](../tensorfold-20261027/README.md).

## Autres différences repérées, hors priorité decode actuelle

- Lectures directes Metal, workers et protection des cohortes existent déjà chez nous. Sushi fusionne les offsets exactement contigus d'un même fichier, puis choisit destination directe ou bounce aligné ; native/ffi/disk_reader.h lit déjà les douze segments d'un record par preadv, sans fusion de records voisins. Piste déjà identifiée, intérêt dépendant de leur fréquence réelle ; aucun retour automatique à F_NOCACHE12F/G.
- Prélecture DeepSeek nouvelle par rapport à ad562 : profondeur3, seulement à partir de2lignes, un expert non caché parmi les top12 par ligne ; elle attend les jobs avant toute mutation de son pool. Ce n'est pas une prévision exacte des futurs experts Qwen, ni une raison de relancer72/73 sans différence concrète.
- Préfixes SSD : états Qwen GDN + QSA et sidecars spéculatifs, writer borné avant allocation, libération des vieux checkpoints après écriture, manifeste publié en dernier. Échec d'écriture invalide l'entrée. L'écriture atomique ne fait pas de fsync ; le fingerprint repose en partie sur taille/mtime. Chez nous le parcours Flash/DFlash vide encore PromptCache ; ancienS3 exact mais dernière admission mémoire non qualifiée avant retrait. Intérêt pour tours répétés/Agent Studio, aucun boost du decode froid établi.

Aucun nouveau kernel, cache, portage Sushi/TensorFold ou réglage de RAM intégré. Le kernel Mia rejeté reste retiré ; HEADONLY et son binaire de production local sont inchangés. Aucun push, publication ou remplacement de l'app installée.
