# T05 — consommateur Hot/Cold pondéré exact

**Phase close — 29 octobre 2026. Qualité numérique : VALIDÉE dans les contrôles décrits. Capture de dispatchs : VALIDÉE sur le composant. Gain global AR : NON CONCLUANT.** La comparaison release128 est terminée8/8 : petit gain sur MAIN, résultat légèrement négatif sur HOLD et préfill hétérogène. **Prototype LOCAL CONSERVÉ, flag désactivé par défaut ; aucun retrait ni promotion.**

Ce document synthétise les preuves acquises sous `ISSUE59-T05-WEIGHTED-01` et `T05-CAPTURE-02` dans [opti.md](../../../opti.md). Il n’a lancé aucun test, build, prover ou nouveau benchmark. Le bridge17 r2 est clos4/4 et le comparatif release128 clos8/8. Les petits chronos17 restent des contrôles du bridge ; la conclusion de débit repose sur les24mesures128, décrites séparément ci-dessous.

## Périmètre et contrat conservé

Le candidat remplace, uniquement dans le parcours physiquement mixte, la concaténation Hot/Cold, la restauration `take`, le cast vers le dtype d’activation et la multiplication des scores par un kernel consommateur. Le flag `MLXL3_SSD_WEIGHTED_EPILOGUE=0|1` vaut **0 par défaut**. Tout Hot et tout Cold sans buffer Hot conservent le chemin stock. Un buffer Hot physiquement présent mais entièrement masqué reste un cas mixte admissible : le kernel sélectionne alors les seules contributions Cold.

En production Flash : `R = 10 × tokens`, de 10 à 80 occurrences, largeur 2560 ; les sources Hot/Cold sont F16. Les poids ont la forme `[tokens,10]`, en F16/F32, et l’activation est F16/F32. Le helper accepte plus généralement R=1..80 et des largeurs positives dont l’arithmétique est validée. La hauteur Hot utilisée est physique : R pour le buffer early Hot, ou le nombre d’occurrences Hot compactées pour l’autre parcours. Le mapping et les doublons restent canoniques `(token, rang top10)`, y compris après découpage en cohortes.

Chaque produit reproduit `sourceF16 → cast<Tx> → promotion<Tp> × cast<Tp>(poids) → store<Tp>`. F16×F16 produit F16 ; les autres combinaisons supportées produisent F32. La réduction `reshape([tokens,10,2560]).sum(1,false)` et l’ajout de l’expert partagé sont inchangés. Aucune FMA accumulatrice, atomique flottante, somme par ordre d’arrivée ou réorganisation de la réduction n’est introduite.

Il s’agit du petit consommateur canonique de T05, inspiré du mécanisme documenté dans [l’issue59 archivée](issue59.md). Ce n’est pas le port complet de `dfin7` : EXL3, Hadamard, SwiGLU, projections down, routeur et lectures SSD restent inchangés. La documentation primaire TurboEXL3 est épinglée à `345bee9e35920ee77292055d873fca891146529c` ; `docs/KERNELS.md`, blob Git `875e6cb836d4c4d8bfdd8c569f66301536fbbdef`, SHA-256 `9f868e123a69eb0813403b3b99195e1098e6887aae1d29e83814b0b81ba8abb7`, est conservée avec LICENSE/NOTICE dans [upstream/](upstream/root-manifest.json).

## Conditions et identités

M5 standard 24 Gio, MLX0.32.2, moteur Rust/FFI MLX/Metal, Qwen3.8 Flash-Next EXL3 2,05bpw `65c89531`. Draft DFlash2 `a22a1744`, oracle target `flash-next-ssd-impl/oracle-07` et oracle draft `build/flash-opt-44/numpy-dflash2-3` inchangés. Cache target3Gio/53slots par couche, 4 lecteurs, overlap/Hot/Wide/Unique actifs, lookahead0, cache allocateur64Mio, limites MLX40opérations/40MB. Aucun pool runtime ni routeur SIMD expérimental n’est joint à T05.

HEAD `7978fceac30d991a8c06d72114fc932b2443f555` avec modifications locales ; ce SHA seul ne décrit pas les exécutables. Les manifestes [v1](t05-identity-v1.json), [v2 capture](t05-identity-v2-capture.json) et [v3 bridge r2](t05-identity-v3-bridge-r2.json) fixent les sources, options, scripts et exécutables.

| Artefact | SHA-256 |
| --- | --- |
| CLI baseline A | `a60eb4ada3a948110fc057d9551032b6d03598ab404201145f5a5410f26f8e75` |
| CLI candidat C, inchangé lors de la correction du test capture | `2fb12d79eed089b603b5ceceb6ce59d996923cfe7e3d0d744b6c1f672209baab` |
| Libtest ayant exécuté les 18 qualités | `9ea659c2f881608c8d7999d440ce567163b6e7701bab3a4af5f5ee94335c009c` |
| Libtest capture v2 | `96a0cdfc5839075d5757b8dab78da2987b2b6d04d6a580512ecdb940113dd6bc` |
| Libtest mutant sans pondération, contrôle négatif seulement | `ddffe4a320f9c2cd2ffe194ca7e138562054086267738031a7b88daace237c60` |
| `native/src/weighted_epilogue.rs` | `9f86025b9533b5b754a8e1a2a77a230487160ffaa7acf8dd311ba4ab86ce9f1b` |
| `native/shaders/flash_weighted_epilogue.metal` | `ad4e58575c56d4bcbf91821efd77b5460c4090fdbafa0cf476a528e5cdfa0f8b` |
| `native/src/ssd_moe.rs` | `4aa0b893e3a6c53ee6c5348a1def4d6dce040ffb00ce9c8fdb61990d56f31e0b` |
| Test capture v2, seule source modifiée après les 18 qualités | `78e1579d82c009a801b082182a73f2901d125fca11cd5884d564cccc3f15f4d9` |

Les GPU sont isolés des builds/provers/autres inférences, sur secteur, avec processus et lecteurs rejoints. L’ancre cumulative [T05-w1](t05-gpu-anchor-w1.json) part de Swapouts=`140864897024B`, Swapins=`53725347840B`. Les qualités et captures inspectées ci-dessous n’ajoutent aucun Swapout. Gardes : Swapin cumulé≤256Mio, RSS≤10Gio, footprint natif<12Go. L’admission disponible libre+fichiers est une heuristique : 8,5Go pour le modèle complet ; une révision prospective documentée la fixe à2Go pour les seuls petits oracles sans checkpoint et la capture. Les autres gardes n’ont pas été relâchées.

## Correction physique acquise

Toutes les sélections ci-dessous sont individuelles, `--exact --ignored --nocapture --test-threads=1`, avec exactement1test passé,0échec,0ignoré pour chaque sélection positive. Le succès d’une suite ordinaire n’est pas utilisé pour compter un test physique ignoré comme exécuté.

| Étape | Résultat inspecté | Preuves |
| --- | --- | --- |
| 5 tests composants | 5464cas positifs :4096partitions Hot/Cold,1288formes/strides/frontières,16littéraux,64frontières finies F16/petits poids F32 ;39refus avant requête kernel | [inspection micro](t05-micro-inspection-v1.json), [résultat natif](t05-normalized/t05-micro-light-v1-plan-result.json) |
| Mutation de la vraie source MSL | Suppression de la pondération seule ; exit101 et échec de l’oracle pondéré attendu, pas crash ni erreur JIT ; bonne version qualifiée avant et après dans la matrice | [identité mutant](t05-mutant-identity.json), mêmes résultats micro |
| 8 tests MoE/SSD réels | Parcours fusionné atteint ; Hot physique10/Cold5/order10 et5hits/5misses ; contrôles tout Hot/tout Cold stock ; Hot en attente pendant I/O, erreurs/reprise, slots pending, Wide, cohortes6+2/4+4 et éviction future ; petits caches et2routes | [inspection SSD](t05-ssd-quality-inspection.json), [résultat natif](t05-normalized/t05-ssd-quality-plan-result.json) |
| 5 tests modèle | Oracle indépendant5positions/248320logits, états GDN, composants sur entrées identiques, paires, séquence MTP/états cible+tête et DFlash2/36préfixes | [inspection modèle](t05-model-quality-inspection.json), [résultat natif](t05-normalized/t05-model-quality-plan-result.json) |

Les composants comparent des valeurs canoniques indépendantes au kernel et au chemin stock concat/take/cast/mul : formes et dtypes attendus, sorties non vides/finies, bytes exacts, réduction MLX inchangée, aliases d’entrée inchangés. Ils exercent les poids F16/F32, activations F16/F32, largeurs frontières, R1..80, H2560, strides, doublons, inversions, zéros signés et poisons inutilisés. Les sous-normaux/petits poids utilisent le GPU stock comme oracle du contrat FTZ ; les littéraux normaux CPU indépendants restent présents.

Au niveau modèle, les assertions exactes de trajectoire/logits/états et celles de séries/paires/MTP/DFlash sont conservées ; les comparaisons indépendantes de composants et de captures ont leurs tolérances explicites. Ce résumé ne transforme pas toutes ces vérifications en une égalité universelle de toute génération ou de toutes les valeurs futures.

## Dispatchs réels : capture v2 qualifiée

Un seul composant synthétique : Hot physique10×2560F16, Cold5×2560F16, R10, poids F32 et activation F32 ; entrées évaluées et deux chemins chauffés avant capture. Chacun construit un index `order` neuf. Les comparaisons sont effectuées hors capture : **25600valeurs F32,102400octets, sorties complètes finies et byte-exactes**.

Le test v2 appelle `synchronize()` immédiatement après chaque début de capture, puis construit et évalue le nouveau graphe. Ce flush est propre au capteur ; il n’est pas ajouté au moteur. Inspection Xcode27.1 de chaque commande/pipeline, pas conversion des requêtes lazy en dispatchs :

| Chemin | Commande | Pipeline GPU observé |
| --- | --- | --- |
| Stock | 9 | `gg1_copyfloat16float16`, grille25600×1×1 |
| Stock | 15 | `gg1_copyfloat16float16`, grille12800×1×1 |
| Stock | 23 | `gather_frontfloat16_uint32_int_2`, grille1280×10×1 |
| Stock | 29 | `v_copyfloat16float32`, grille25600×1×1 |
| Stock | 37 | `g2_Multiplyfloat32`, grille2560×10×1 |
| Fusionné | 13 | `custom_kernel_mlxl3_flash_weighted_epilogue_v1_float16_t_float16_t_uint32_t_float_float`, grille25600×1×1 |

**Résultat composant :5dispatchs compute stock contre1fusionné**, chacun2command buffers,1compute encoder,0blit encoder. Ce n’est pas une division par5 du temps GPU, ni un gain TPS, ni un comptage du modèle complet. Les conseils Xcode sur la création tardive de fence/buffer ne sont pas un diagnostic de coût chronométré. Xcode et ses replays sont quittés avant les lancements modèle/chrono.

Preuves : [inspection Xcode v2](t05-capture-v2-xcode-inspection.json), [plan/résultat exact](t05-capture-v2-plan-result.json), traces [legacy](prepared-t05-gpu/t05-capture-v2/legacy.gputrace) et [fused](prepared-t05-gpu/t05-capture-v2/fused.gputrace). Le test exact a terminé en0,724s, tous processus joints et0nouveau Swapout ; cette durée de processus n’est pas un microbenchmark du kernel.

## Bridge réel17 r2 terminé :4/4

L’inspection du29octobre à21:21:24UTC valide les quatre processus AR **MAIN-A, MAIN-C, HOLD-A, HOLD-C**, chacun1chauffe et1mesure :4chauffes+4générations mesurées de17tokens, soit16transitions de decode par génération. Les exécutables sont exactement A=`a60eb4ad…` et C=`2fb12d79…`, avec environnements identiques hors flag pondéré. Chaque rapport est `validated`, tous les processus et lecteurs sont joints.

MAIN conserve le hash de préfixe externe `4b46ba2824c9f47c` et un texte non vide de84caractères ; HOLD conserve `bf1f7927af918558` et69caractères, dans A et C, chauffe et mesure. Les textes sont les préfixes attendus des goldens128. Ce contrôle bridge ne constitue pas une preuve numérique par position des logits/états ; les qualités natives précédentes restent distinctes.

Sur l’ancre cumulativeT05-w1 : **0nouveau Swapout**, Swapin maximal **48 218 112B**, pic natif maximal **7 126 161 408B**, sous les gardes. Le refus d’admission initial avant PID est conservé. **Les quatre petits chronos17 qualifient le bridge et les oracles, pas un gain de vitesse.** Preuves : [inspection bridge17 r2](t05-bridge17-r2-inspection.json), [plan/résultat](t05-normalized/t05-bridge17-r2-plan-result.json), [identité v3](t05-identity-v3-bridge-r2.json).

## Comparaison release AR128 terminée : gain global non concluant

Ordre **ACCA×MAIN/HOLD**,8processus terminés et validés,1chauffe+3mesures par processus : **24générations mesurées+8chauffes**,128tokens de sortie et127transitions de decode par génération. Chaque prompt/bras représente6mesures dans2processus, pas6processus indépendants. Toutes les mesures sont conservées ; les chauffes sont exclues des agrégats. Texte, hash et comptes complets sont conformes aux goldens externes. Superviseur, processus et lecteurs joints ;123sources et6exécutables de l’identité inspectés.

Les débits sont `somme des tokens / somme des secondes` ; le TTFT est la moyenne des mesures. Les pourcentages C/A restent séparés par prompt :

| Prompt | Decode A→C (tok/s) | Δ decode | Préfill A→C (tok/s) | Δ préfill | TTFT moyen A→C (s) | Δ TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAIN | 10,144611→10,351204 | +2,03648% | 9,955162→10,109127 | +1,54658% | 8,739832→8,606639 | −1,52397% |
| HOLD | 10,064316→9,991023 | −0,72825% | 10,050932→9,702572 | −3,46595% | 8,158987→8,451867 | +3,58967% |

Sur l’ancre cumulativeT05-w1 : **0nouveau Swapout**, Swapin maximal **48 218 112B**, pic natif maximal **7 155 652 608B** (7,15565Go). Même cache3Gio, options, identités et conditions prévues ; aucun Xcode/build/prover/autre modèle concurrent. Preuves complètes : [inspection128](t05-speed128-inspection.json), [résultat superviseur](t05-normalized/t05-speed128-plan-result.json), [plan figé](t05-normalized/t05-speed128-plan.json).

L’utilisateur accepte la variation : **la variance n’est pas employée comme motif automatique de rejet**. Les dérives des processus de référence A sont documentées (MAIN−0,71576%, HOLD+2,24112%) sans imposer un nouveau veto. Le résultat est néanmoins modeste et hétérogène : le gain MAIN ne se retrouve pas sur HOLD, où le préfill recule et le TTFT augmente. Aucun pourcentage global n’est fabriqué en additionnant ou en moyennant ces gains par prompt.

La comparaison porte sur l’ancien CLI A et le nouveau CLI C. Un A/B avec **le même CLI C, flag0 contre flag1, n’a pas été mesuré** ; l’attribution exclusive du petit écart à la fusion seule reste limitée. La réduction composant5→1 est réelle, mais ne mesure ni la baisse de GPU idle T05 ni sa contribution au temps de token complet. **Décision : gain global non concluant, prototype conservé localement avec défaut0 ; aucun retrait ni promotion.**

## Vérification CPU, preuves et portée

Fmt, Clippy strict CPU et `mlx,chat` all-targets, builds CLI réels et suites passent : **CPU254passés/0échec/8ignorés ; MLX290passés/0échec/177ignorés**, agrégés sur les cibles applicables. Après la correction du seul test capture, fmt/Clippy/buildCLI/suiteMLX/inventaire repassent ; le CLI reste byte-identique. [Contrôles bons v1](t05-good-cpu-plan-result.json), [contrôles capture v2](t05-capture-v2-cpu-plan-result.json). Les177ignorés incluent les besoins physiques sélectionnés séparément ; ils ne sont pas déclarés tous exécutés.

Kani0.68.0 sur la vraie fonction CPU `geometry`, solveur Kissat :

| Harnais | Domaine et propriété | Résultat |
| --- | --- | --- |
| `weighted_geometry_bounds_every_symbolic_address`, unwind3 | Hot/Cold/largeur/colonne `usize` symboliques et source `u32` ; refus invalides, conversions sans perte, bornes de chaque adresse acceptée et grille | 169obligations,0échec,1inatteignable ;6/6covers ;124,80s superviseur |
| `weighted_geometry_checks_all_eighty_canonical_rows`, unwind82 | 81indices `u32` symboliques, longueur `u8`, appel quand longueur≤81 ; toute acceptation a1..80lignes et chaque source dans les bornes | 172obligations,0échec,1inatteignable ;4/4covers ;99,56s superviseur |

Aucun `assume` n’a été ajouté. La correction du harnais remplace l’oracle arithmétique u128 par u64 après assertions de conversions sans perte ; elle ne réduit pas le domaine de production. Les assertions d’unwinding restent actives. Commandes et diagnostics : [plan Kani v2](t05-proof-v2-plan.json), [résultats](t05-proof-v2-plan-result.json).

Ces vérifications CPU ne prouvent pas Metal, les casts matériels, MLX/FFI, l’allocateur, les durées de vie asynchrones ou la concurrence I/O. La vraie entrée MLX a été tentée et reste **NON VÉRIFIÉE**, ICE `intrinsics.rs:243` avant obligations. CBMC6.11 ne reconnaît pas le vrai fichier `.metal` : [diagnostic](t05-msl-cbmc.log). Aucune traduction de substitution n’est présentée comme preuve MSL. Les deux harnais CPU sont câblés à la CI existante PR/main ; aucun run distant de cet arbre local non publié n’est déclaré vert.

## Échecs et corrections conservés

| Étape | Constat et suite donnée |
| --- | --- |
| Première revue API, avant GPU | Poids rank0 incompatibles avec `weights[row]` dans l’ABI scalaire MLX ; refus explicite avant upload/kernel, R1rank1/rank2 positifs et scalaire négatif ajoutés. Pas de reshape supplémentaire production. |
| Kani v1 | CaDiCaL expire à90s ; aucun contre-exemple ni preuve, commandes suivantes non démarrées. [Résultat conservé](t05-proof-plan-result.json). V2 décrit ci-dessus réussit avec oracle équivalent et Kissat. |
| Kani MLX / CBMC MSL | ICE MLX en5,74s/exit1 avant obligations ; MSL non parsé. Résultats restent non vérifiés malgré le succès CPU. |
| Premier prévol GPU | Disponible8 327 462 912B<8 500 000 000B ; refus avant création d’ancre/PID ; tentative suivante refusée par ancre absente. [Log](t05-micro-v1-runner.log). Admission légère2Go ensuite préenregistrée pour les seules petites matrices/capture, jamais pour le modèle complet. |
| Métadonnées du premier plan léger | Le wrapper historique exige le membre exact `1 passed; 0 failed`, absent malgré la chaîne plus stricte `1 passed; 0 failed; 0 ignored` ; refus avant test/PID. [Log](t05-micro-light-v1-runner.log). Copies normalisées ajoutent le membre attendu et gardent l’assertion stricte ; aucun validateur affaibli. |
| Mutation sans poids | Échec numérique intentionnel détecté, source correcte restaurée et bonne version retestée. Ce n’est pas un succès numérique du mutant. |
| Capture v1 | Test de valeurs réussi, mais Xcode montre0computeencoder/0dispatch dans les deux fichiers : **capteur invalide**, pas zéro calcul réel. MLX avait créé le command buffer avant `start`. [Inspection et sources Apple/MLX](t05-capture-v1-xcode-inspection.json), [premier résultat](t05-capture-inspection.json) conservés. |
| Capture v2, extraction CPU | Extraction attendait à tort290tests sur une seule ligne au lieu d’agréger les cibles ; identité non créée et lancement GPU refusé avant PID. Les comptes ont été inspectés, sans masquer le refus. Seul le test capture change parmi113sources figées ; CLI/shader/production identiques, v1 archivée. |
| Premier bridge17 | A MAIN refusé avant PID : disponible7 702 003 712B, manque797 996 288B au seuil8,5Go. [Résultat conservé](t05-normalized/t05-bridge17-plan-result.json). Aucun checkpoint/chrono lancé par cet essai. |
| Bridge17 r2 | Reprise préenregistrée après libération de RAM : disponible8 865 038 336B, même ancreT05-w1 et Swapout inchangé. Mêmes exécutables/configurations ; nouveaux noms de logs et [identité v3](t05-identity-v3-bridge-r2.json). Clôturé4/4 validés ; voir la section bridge et [l’inspection](t05-bridge17-r2-inspection.json). |
| Première commande d’analyse128 | Commande en mémoire refusée au parsing avant exécution ou création du fichier ; corrigée avant l’inspection finale. Aucun processus GPU ni mesure affecté. Constat conservé dans [l’inspection128](t05-speed128-inspection.json). |

Les anciennes captures, refus et rapports incomplets restent conservés. Une relance propre n’efface pas un échec d’admission, un capteur invalide ou une tentative de vérification non aboutie.

## Audit après vitesse, décision et limites restantes

Les revues indépendantes [initiale](t05-code-review.md) et actualisée dans le journal ne trouvent aucun autre défaut concret de forme, dtype, mapping physique, fallback, copie/synchronisation ou ownership après la correction rank0. Les tests physiques pending/cohortes/éviction/erreurs ont ensuite passé. Les deux `async_eval`, les fences des banques et les évaluations nécessaires aux erreurs ne sont pas supprimés.

Après les mesures128, le rapprochement des sources figées, des formes/dtypes de production, du parcours mixte réellement atteint et des contrôles physiques ne met en évidence **aucun nouveau défaut concret d’adaptation** à corriger. Les sources de production et le shader sont conformes à l’identité testée ; la correction capture v2 n’a touché que le test. Cela borne le constat au candidat essayé : ce n’est ni une preuve d’optimalité, ni une attribution temporelle exclusive des écarts. Aucun retrait n’est décidé sur la seule variation ou sur le résultat HOLD.

Le graphe mixte retire les intermédiaires exprimés concat/take/cast/mul et la vue des poids. Le kernel possède sa sortie ; les entrées non row-contiguous peuvent toujours nécessiter les copies prévues par l’API MLX, exercées par les oracles. Les vues Hot par lignes complètes, Cold et poids des cohortes de production sont contiguës. Aucun gain RSS/footprint n’est déduit du seul nombre de buffers logiques : donation, lazy evaluation et cache allocateur comptent.

Restent payés : calcul CPU de l’ordre, validation supplémentaire, formatage header et arguments/CString, factory MLX cachée mais génération de signature/source après hit, hash et recherche de pipeline à l’évaluation, allocation de sortie, bindings, barrières, dispatch et commits. T05 ne supprime pas l’encodeur MLX ; les12,142ms du profilage antérieur ne constituent pas son plafond récupérable mesuré.

**Aucun essai de cette phase n’est encore actif.** Qualité numérique, capture v2, bridge17 et matrice AR128 sont closes ; les refus et tentatives non abouties restent visibles. Un contrôle même CLI C/flag0–1, un profil attribuant le GPU idle exclusif à T05, les performances MTP/DFlash et les contextes longs restent **NON EXÉCUTÉS/NON MESURÉS**. Ce sont des suivis possibles distincts, pas des résultats acquis ni des essais implicitement lancés. Une future mesure ciblée devra préciser le mécanisme et les contrôles affectés avant exécution ; elle ne peut pas reprendre le plafond12ms comme gain attendu.

**Intégration : prototype LOCAL CONSERVÉ derrière flag désactivé par défaut ; gain global NON CONCLUANT, aucun retrait ni promotion, aucun gain livré, aucune mise à jour d’app, push, merge ou release T05.** La publication séparée de [l’issue60 de profilage](https://github.com/0xZKnw/mlxl3/issues/60) ne publie ni ce candidat ni ses artefacts. Cette phase T05 est close avec les limites ci-dessus.
