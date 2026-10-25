# 82F06 — arrêt du comptage des perdants top-10

**Clôturé le 25 octobre : qualité locale passée sur le domaine testé, débit non concluant, runtime effectivement retiré. Aucun gain F06 livré.** Après archive et contrôle des 85 sources, seuls les hunks F06 des quatre fichiers suivis ont été inversés par le patch exact vérifié. Deux fichiers de tests sont préservés sans référence de module et ne sont plus compilés. L’application installée est inchangée. F05 est un essai isolé ultérieur : F06 ne constitue pas sa base commune.

## Mécanisme testé

Le prototype demandait `MLXL3_SSD_TOP10_EARLY_EXIT=0|1`, lu strictement une seule fois au chargement, défaut 0. Sur le seul routeur legacy Flash E512 / top10, la boucle des perdants s’arrêtait après dix clés strictement supérieures. Les gagnants parcouraient tous les concurrents ; clés/ties, scores, ordre de réduction et barrières restaient inchangés. Flag 0 : source et factory historiques. Aucun SIMD E256/top8, top16 draft, GEMV, softmax ou ordonnanceur SSD modifié. Le code source économisé ne prouvait pas un gain GPU : une lane gagnante peut maintenir la SIMD dans la boucle.

## Débits réels : aucun gain qualifié

M5 10 cœurs / 24 Gio, Flash Next 2,05 bpw `65c89531`, draft Q4 `a22a1744`, MLX 0.32.2. Secteur, cache 3 Gio / 53 slots, allocateur 64 Mio, DFlash2 fixe B2 ; un warmup de 128 tokens puis une mesure de 128 tokens / 127 decode par processus. GPU Hot/Wide, UNIQUE et multi-token actifs, prélecture/LES/shared/stable désactivés. Même CLI `cbc03630` et contrôleur 82B `8935` ; seule la valeur F06 0/1 diffère entre H et C.

| Prompt | H puis C (tok/s) | C puis H v2 (tok/s) | Géomoyenne brute C/H |
|---|---|---|---|
| MAIN | 11,206068 → 11,170360 (−0,3186 %) | 11,441572 → 11,486666 (−0,3926 %) | −0,3556 % |
| HOLD | 10,585691 → 10,711404 (+1,1876 %) | 10,223068 → 10,762146 (−5,0090 %)* | −1,9597 %* |

\* **HOLD C-H n’est pas qualifié au critère zéro swap-out système** : sa phase C a enregistré 27 197 440 octets (25,9375 Mio), sous la garde stock de 256 Mio. Origine inconnue ; ces valeurs brutes ne prouvent pas une régression causale du candidat. La géomoyenne HOLD est purement descriptive. MAIN n’établit aucun gain, ni une régression universelle de 0,36 %.

Les seize réponses H-C / C-H v2, textes et hashes, sont exactement gold60, avec 128/127, modes, acceptations et largeurs identiques. Demandes SSD logiques H=C par phase ; aucune mesure du trafic SSD physique. MTP inactif a des compteurs **JSON null**, pas zéro. Les histogrammes ont huit cases : index 1 compte les blocs ancre + une proposition. Les profils vérification/draft/commit sont inclusifs côté hôte, pas des durées GPU exclusives ; ils ne sont pas additionnés aux totaux block/decode. Warmup/mesure et conditions sont conservés sans causalité matérielle inventée.

## Qualité, preuve et limites

- CPU : 117 lib réussis / six ignorés, seize intégrations ; format/Clippy CPU+MLX/builds inspectés. Suite native initiale : 130 réussis / 150 ignorés. Les tests physiques concernés ont ensuite été sélectionnés explicitement.
- Qualité v2 : 19 processus attendus clos et joints, neuf refus CLI stricts dont un environnement non Unicode, loaders absent/0/1 réels, 256 cas composant, six domaines fallback/erreurs, 384 cas de routes sur les 48 gates réelles, F16/F32 et strides. IDs contre tri CPU indépendant, scores contre la voie historique ; non-finis intentionnels distingués des sorties finies attendues. Aucun changement d’environnement après initialisation MLX.
- 36 préfixes DFlash : captures externes exactes observées, logits/états/commits exacts par rapport au serial sous flag 1 commun. Quatre budgets MTP 1/2/3/17, quinze propositions et cinq acceptations, séquence et états cible/tête exacts. Ces tests ne sont pas un différentiel intégral cible flag 0/1 ; le différentiel direct porte sur composants et routes. Bridge : 43 requêtes, parité, erreurs attendues, annulation/reprise. Son transport ne certifie pas tous les événements tardifs de fermeture ; aucun incident tardif observé. Le contrôleur TPS 82B contrôle son EOF normal et sa file finale.
- Kani sur le vrai garde CPU : cinq obligations réussies, quatre couvertures satisfaites, domaines symboliques i32/usize/bool complets sans hypothèse, contrat de géométrie E512/K10. Ne prouve ni classement MSL, ni MLX, concurrence ou inférence. La tentative sur l’entrée MLX réelle échoue par ICE `intrinsics.rs:243`, après environ 4,02 s et 684 Mo de RSS groupe, sous limites 180 s / 4 Gio ; propriété Metal **non vérifiée**, aucun stub.
- Mutant CPU E511 détecté, garde restauré puis test/Kani repassés. Mutant GPU d’ID `(tid+1)%512` détecté par l’oracle CPU indépendant, même si les deux voies partagent la corruption. Normal release et mutant debug sont des profils différents : aucune revendication de débit tirée de ce contrôle.

## Échecs conservés

Les deux premiers Clippy MLX échouent sur des fixtures/signatures de tests, puis sont corrigés sans divergence de production observée. Quality-v1 reste **FAILED** : la suite de 130 tests passe, mais un champ de métadonnées de test unique était conservé à tort pour la suite entière. La v2 retire seulement cette étape déjà passée et poursuit les contrôles non exécutés ; elle ne requalifie pas v1.

Le premier préflight de vitesse attendait à tort `mtp_blocks==0` au lieu de `null` et lisait mal les histogrammes. Le shell sans errexit a lancé la confirmation malgré l’échec. Cette confirmation v1 a été interrompue, enfant et sampler joints ; son rapport partiel est **exclu intégralement** des résultats. Le contrôle corrigé et le dossier neuf `confirm-v2` sont les seuls utilisés pour C-H. Tous les bruts et erreurs sont conservés.

## Mémoire

| Phase | Swap-in système | Swap-out système | Pic physique processus |
|---|---:|---:|---:|
| Bridge | 131 072 octets | 0 | 7,938 Go |
| H-C | 1 294 336 octets | 0 | 7,851 Go |
| C-H v2 | 0 | 27 197 440 octets | 7,851 Go |

H-C : 65 536 octets d’entrée sur MAIN-C et 1 228 800 sur HOLD-H. C-H : sortie sur HOLD-C, observée au compteur de fin de phase malgré un dernier monitor live à zéro. Ne pas effacer ces deltas par les zéros des autres phases. Pré/post, enfants, lecteurs et samplers sont joints ; secteur contrôlé. Les snapshots du ledger swap-in propre valent zéro, mais ne prouvent pas l’absence de swap-out privé ou le propriétaire des pages système.

Compression logique courante maximale des processus de débit : environ 3,089 Go H-C et 3,158 Go C-H, différente de la taille physique du compresseur. Les caps compression 77B de 512 Mio restent FAILED ; aucune fuite, copie double permanente ou cause exclusive de latence n’est prouvée. Les durées et allocations des oracles de qualité ne sont pas celles d’un benchmark. Les notes statiques `memory-followup` et `hot-readback-audit` sont distinctes de gains mesurés.

## Archives et retrait effectif

`evidence/source-snapshot.tar.gz` contient les 85 sources du gel `identity-frozen-v2.json`, toutes concordantes. `integrated-source.patch` inclut production, CI et deux nouveaux tests ; `tracked-integrated.patch` décrit les quatre fichiers suivis inversés. `source.patch`, `ci.patch` et les versions de préparation restent historiques. Les copies complètes `original/` et `changed/` ne sont pas livrées : base Git + patchs + snapshot couvrent leur contenu utile.

Les deux tests natifs détachés sont préservés dans W mais non référencés et hors stage. Aucun privé écrasé. Les modifications F05 éventuelles du worktree après retrait ne changent pas les sources mesurées F06 et ne doivent pas être confondues avec cette archive. `withdrawal-doc-audit.json` conserve les SHA et la constatation statique du retrait.

Les plans OFF/MTP1 ont été préparés puis **clos sans exécution**, puisqu’aucun gain DFlash n’est qualifié. Aucun remplacement de l’application ni push d’un gain F06. Publication proposée : documents et preuves historiques uniquement ; exclure `runtime/`, tout binaire, copies de préparation et poids. Les commandes/options, identités, logs, erreurs et contre-revues sont conservés dans ce dossier.

La CI de la base et des archives F01 ne valide pas un moteur F06 publié : F06 n’a pas été livré et aucune CI verte F06 n’est revendiquée. Les README/manifeste de préparation et contre-revues gardent leurs états historiques ; le présent README et `summary.json` indiquent l’état final **clôturé, retiré**. Aucun gain cumulé ni 15 tok/s annoncé.
