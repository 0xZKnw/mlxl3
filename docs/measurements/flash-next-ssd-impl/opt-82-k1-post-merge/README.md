# K1 sur Flash Next : sources et résultats post-fusion

Lire le [rapport complet](post-merge/final-report.md), la [revue de conception](post-merge/performance-design-review.md) et la [mise à jour CI finale](post-merge/ci-42c0958/README.md).

K1 n’a apporté aucun gain de débit observé dans les deux ordres testés. Cette archive conserve un prototype de masquage H128/SwiGLU des occurrences Cold, ses contrôles et ses résultats ; elle ne livre pas d’optimisation runtime ni de mise à jour de l’app.

| Prompt | H-C : baseline → candidat | Écart | C-H : baseline → candidat | Écart |
|---|---:|---:|---:|---:|
| MAIN | 11,399755 →11,254234tok/s | −1,2765% | 11,576340 →11,439044tok/s | −1,1860% |
| HOLD | 10,903516 →10,896760tok/s | −0,0620% | 11,128820 →10,946912tok/s | −1,6346% |

Huit processus, chacun128tokens de warmup puis128sorties/127decode mesurés, sur M5 10GPU/24Gio, secteur, Qwen3.8 Flash Next EXL3 2,05bpw65c89531, draft DFlash2Q4/ring a22a1744, fixe B2, cache3Gio/53slots et alloc64MiB. Les16textes/hashes égalent gold60 ; compteurs d’acceptation/histogrammes/demande logique SSD identiques entrebras. Un passage par bras et par ordre ne démontre pas une régression causale universelle.

Qualité séparée : H1283144cas+240bords+10replis, activation960+2nonfinis+1frontièreIDs et28replis/10refus,528slots,36préfixes/logits/états et MTP passent. SixmutantsMSL provoquent la divergence numérique attendue puis les normaux passent. Kani CPU vérifie804obligations/18covers dans les bornes documentées ; la tentative réelle MLX échoue surICEintrinsics243. Ni MSL, ni MLX, ni concurrence n’est prouvé formellement. Les704samplesH128 et192activation mesurent un composant wall avec coûts communs, pas des gains de tokens/s ; ils ne s’additionnent pas.

Zéro nouvelle sortie de swap système observée dans les12phases de débit ;65 536B d’entrées pendant le post30H-C,0sur les processus mesurés/C-H. Pic physique processus maximal publié≈7,851Go. Ledger privé d’entrées0 ne prouve pas zéro swap-out privé. Compression/VM système restent non attribuées et les anciens caps77BFAILED ne sont pas requalifiés. Les corrections de prose et échecs de préparation sont conservés.

## Provenance et reconstruction

Base exacte : `9c8474ca1292e29bae3c7d4d6383bd1820549349` (fusion PR34). Patch dix fichiers : `post-merge/source.patch`, SHA256 `9782c74c49613acc520a38975376eefac5a52217941cfc254909702521dd358b`.

Le contrôle `provenance/reconstruction-preflight.json` a extrait les95sources existantes du commit puis appliqué le patch sur copies isolées, sans fuzz. Les cinq nouveaux modules sont apportés par le patch. **100/100SHA** égalent `post-merge/candidate-source-sha256.json` (manifesteSHA `2b4eb648ff2e0c4362bb88b1705c2022a6acb6d1896f1f2766923ed8d1ca98c4`) ; les dix fichiers de delta sont également exacts. Base+patch+manifeste suffisent : le tar source redondant reste local et n’est pas publié. Les sixpatchsmutants reconstruisent aussi leurs sixSHA sources gelées, sans compilation supplémentaire.

CLI baseline `d8757fa6d9f62648d66f7a2d58617d6d54ff33905fefd408475519edd85105a6`, candidat `cc9669ad864dd73dadbed0bb503a440fefcec2c736e065d61f7ec0936639e9a4`, natifRelease `041b3b092383f30bdc50f9f51c737bfef0b52114f3ef7ee07cfa37fe0dcdd3dd`. La metadata embarquée e551-dirty de la baseline est historique ; ses91sources runtime avaient été comparées à la base fusionnée. Les binaires ne sont pas publiés dans cette archive. Reconstruction des octets source n’implique pas un build identique à l’octet ou une nouvelle validation numérique.

Pour inspecter le prototype dans un checkout isolé, partir du commit exact, vérifier le SHA du patch, exiger `git apply --check` depuis la racine puis appliquer le patch ; comparer tous les fichiers au manifeste avant de construire. Ne pas appliquer à une branche de travail ayant des changements non sauvegardés. Le code actif de main n’a pas besoin de ce patch pour lire les résultats.

## Chemins locaux dans les preuves

Les plans et bruts sont conservés tels qu’exécutés, avec leurs chemins absolus historiques :

| Préfixe local historique | À fournir pour une répétition |
|---|---|
| `/Users/justin/.codex/worktrees/v2-merge/mix-stq1_0` | racine du checkout reconstruit exact ; pilotes et source doivent être rehashés |
| `/Users/justin/Documents/mix-stq1_0/build/flash-opt-82/K1/post-merge` | nouveau dossier jetable de résultats et binaires compilés ; `post-merge/` dans cette archive contient les preuves historiques |
| `/Users/justin/Documents/mix-stq1_0/models/...` | checkpoints exacts65c89531/a22a1744, absents de l’archive |
| `/Users/justin/Documents/mix-stq1_0/.venv/...` | environnement MLX0.32.2/Python/outils identifié séparément |
| `.../oracle-07`, `.../numpy-dflash2-3`, `.../flash-mtp-oracle-held-02` | oracles externes exacts, avec leurs identités ; pas inventer de succès s’ils sont absents |

Ne pas exécuter directement les argv absolus archivés. Copier les plans vers des destinations neuves, adapter explicitement les chemins/checkpoints/outils et figer une nouvelle identité ; conserver les anciens bruts et leurs SHA. Les résultats historiques ne qualifient pas automatiquement un autre modèle, matériel, environnement ou binaire.

## Contenu et limites

`post-merge/real-tps-review.*` et les rapports HC/CH fixent les16réponses, profils et VM. `quality-review.*`, `micro-review.*`, plans/componentJSON/CSV, preuvesKani/ICE, sixpatchs et logsmutants, leurs runners/tests et les sixCargoJSONL bruts sont conservés. Les CargoJSONL ne sont pas byteidentiques ; aucun extrait ou normalisation ne remplace leurs octets publiés. Les garde-fous de supervision stock45/49/43 et échecsv1 restent visibles.

Le bridgeK1 de43requêtes et les débits OFF/MTP de promotion n’ont pas été exécutés dans cette campagne. Pas de gain15tok/s revendiqué, pas d’addition des micros. Les poids, binaires, target, caches, copiePony, rapports utilisateur privés et historique W/opti sont exclus. Seules les notes K1PM publiques sélectionnées par le parent accompagnent la PR documentaire.

Retrait effectif consigné dans `post-merge/withdrawal-result.json` :95fichiers de base byteexacts,5modules conservés détachés, binaire de développement d875 restauré ; app installée inchangée, aucun build/GPU répété. Les snapshotsCI `post-merge/ci-exact/` (22:00:32UTC) et `post-merge/ci-snapshot-1.json` (22:38UTC) sont historiques et distincts ; Kani global y reste en cours, aucuneCI globalement verte annoncée.
