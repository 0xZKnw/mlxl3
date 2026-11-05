# Flash Next SSD63 — vérificateur multi-token exact

21 octobre2026. Base publiée `540ff4806b033ae58a1bff9f512ae8b4b50f185b`, code63 publié `ba7068fbbc9b60bbf0fecfdf16117135e4d39100`, suite des expériences61/62. **63 clos : correction physique et microbench validés ; gain MTP de6,18–6,87% qualifié dans les fenêtres appariées. Débit DFlash63seul non concluant selon les seuils préenregistrés ; ajoutWide à63 qualifié en adaptatif, +1,12–3,29%, sans additionner les gains.** Options expérimentales `MLXL3_FLASH_MULTI_TOKEN=1` et `MLXL3_SSD_GPU_HOT_WIDE=1`, chacune défaut0 ; app installée inchangée.

## Calcul réellement groupé

Le vérificateur DFlash2 puis MTP1 préparent les entrées GDN indépendantes ensemble, exécutent les projections EXL3 qkv/z/out et la tête sur plusieurs lignes, et utilisent le scan GDN multi-token avec sa tape compacte existante. Le kernel EXL3 décode chaque tuile une fois pour deux accumulations indépendantes : NT1/MB2/SG4, splits et ordre de réduction de référence, mêmes Hadamard/conversionsFP16. B3 masque la quatrième ligne de sa dernière paire ; K7 et formes hors2..8 gardent le chemin séquentiel. Le FC caché MTP conserve son contrat spécialisé antérieur à quatre streams.

Les convolutions/historiques, PLE et HC conservent leur ordre par position ; QSA reste causal et séquentiel. Les projections nonEXL3 gardent leur réduction mono-ligne. La tape GDN partage l’état initial, les clés, delta et decay ; un commit partiel rejoue le préfixe retenu et restaure son historique convolution/PLE. Un commit total conserve l’état final. Les erreurs, abandon et annulation gardent les mécanismes de reset ; les barrières de durée de vie des experts SSD restent en place.

L’adaptation ne reproduit pas tout le runtime Splash. Leurs graphe/scratchs persistants, kernel GDN fusionné, commit reporté dans le scan suivant, batchHC/PLE/QSA et stockage GGUF/Table16 sont distincts. La [comparaison précise avec Splash](flash-next-ssd-verifier-63-splash-map-2026-10-21.md) identifie ces écarts. HEAD Splash vérifié avec URLexplicite :`1482f6bbc2abf12375332725302019594150b6cb`, blobs/SHA256 archivés. L’erreur initiale de provenance Git (dossier de blobs remontant au dépôt parent) est corrigée et conservée dans `opti.md`.

Aucun calcul LFM2 n’a été modifié. Le plan MoE reste celui du moteur SSD ; partage des experts réellement répétés58 et autres fusions ne sont pas revendiqués ici.

## Profilage du vrai bridge

`FlashNext::dflash2_profile` expose dans `NativeStats.dflash_profile` les durées hôte draft/sélection, vérification cible et commit/cache, les lignes vérifiées et octets demandés des experts. Aucun `eval()` de profilage ajouté. Les timings incluent les attentes déjà présentes ; le replay GDN paresseux peut être exécuté lors de la vérification suivante. Ce ne sont pas des temps GPU exclusifs et il ne faut pas additionner des intervalles GPU supposés.

Les rapports v5 dérivent :

- `verification_seconds/(accepted+blocks)` pour les sorties des blocs vérifiés ;
- millisecondes par ligne cible effectivement vérifiée ;
- octets demandés et équivalents records experts par token effectivement décodé ;
- zéro ou un token terminal ordinaire, séparé des sorties des blocs.

Les octets sont les demandes experts réussies vers les buffers, hors préfill/chargement statique et hors prélecture (lookahead0 dans cette campagne). Ce ne sont ni des appels système ni des octets réellement lus sur le device : le cache fichier macOS peut satisfaire les lectures. Les identitésv3/v4 ne comptaient que les blocs ; v5 corrige le dernier token ordinaire. Les anciens fichiers ne sont pas réécrits. La régression physique voit430 564 140octets sur un vrai fallback, et l’exportCLI deux tokens MAIN/HOLD contient148–200Mo au lieu de0.

## Sources et vérification

Données :[`opt-63-verifier/`](measurements/flash-next-ssd-impl/opt-63-verifier/). Les identitésv1/v3/v4/v5 fixent146sources et les SHA256 des binaires/archives : v1 projections+scan, v3 tête/DFlash+profil, v4 branchementMTP, v5 compteurterminal. Binaires immuables dans `build/flash-opt-63/`, source parent+modifications locales distingués du commit publié. Les suites et preuves CPU ont été exécutées hors campagnes GPU.

| Contrôle | Résultat inspecté |
| --- | --- |
| CPU v5 `cargo test --locked --no-default-features` |114passés :98lib+16contrats ;3ignorés |
| Livré v5 `cargo test --locked --release --features mlx,chat` |143passés :115lib+12CLI+16contrats ;143ignorés, distincts des sélections physiques ci-dessous |
| Clippy strict, fmt, build CLI livré | Succès ; avertissement de compatibilité future préexistant `block0.1.6` |
| Python complet |915passés/4sautés : `ponyexl3`/modèleLing absents ; sources Python inchangées après ce contrôle |
| Projections GPU |144cas qkv/z/out/tête, B1..9 dont repli9, F16/F32/strides, valeurs finies et contrôle mauvais ordre ; bit à bit vs mono-ligne |
| GDN GPU |T1..8,16têtes clés/48valeurs/128dim et variantes historiques ; sortie, étatfinal et replay de chaquepréfixe bit à bit ; préfixes invalides refusés |
| Cible complète DFlash v3 |36commits B1..8, logits248320/captures/136états, continuation/erreurs/abandon exacts ; oracle captures indépendant64000valeurs, RMSE0 |
| Runtime DFlash v3 |20configurations BF16/Q4/ring/adaptatif×budgets1/2/3/17/64,136états/cas ; annulation et reprise |
| Bridge DFlash v3 |43contrôles, paritétrue ;19profils exportés nonvides |
| MTP v4 |10transactions/préfixes1/2 et sessions budgets1/2/3/17, sousflags0et1 ; logits/étatscible/tête exacts, acceptation et rejet couverts |
| Bridge MTP v4 |Budgets1/2/3/17 HCCH, modes/limites/erreurs, annulationscontext_usage/delta et reprise, fallbackpenalty validés |
| Compteur v5 |Test physique fallback nonvide et4réponsesCLI deux tokens, reset et export vérifiés ; compteursproposés/blocs/lignes0 attendus |

Les campagnes physiques v1/v3/v4/v5 sont closes avec0nouveauSwapout et postflight30s, tous processus joints. Origine63 conservée :Swapins85 102 100 480octets,Swapouts120 147 050 496octets. Pic du test préfixes v3 :7 463 508 184octets ; runtime20configurations :8 737 414 224octets. Il s’agit de pics physiques de tests, pas d’une économie universelle pour toute génération.

Kani0.68.0, contrats CPU réels : layoutMB2/i32 symbolique69obligations/2couvertures ; préfixesGDN66/5 ; commitpair71/4 ; commitlarge71/3 ; compteurdemandes73/4 avec troisu64symboliques et oracleu128 indépendant. Aucun échec ; domaines et assertions d’unwinding conservés. Une mutation qui omettait l’accumulation est détectée, puis source restaurée et suites repassées. Le collecteur cherchait initialement lepanic dansstdoutseul ; erreur conservée, inspection du logcombiné confirmée.

Tentatives Kani réelles MLX dense et pair : ICE `intrinsics.rs:243` avant vérification. GPU, MLX, concurrence et génération entière restent nonprouvés. La CI ajoute les trois contrats pertinents au jobFlash PR/push. Les CI de la base540ff et du code63 ba7068fb passent régressions/Linux/macOSMLX/KaniFlash. Kani global échoue sur l’ancien harnais allocation ; pourba706 le log exact confirme un shutdown du runner/exit143, sans verdict de ce harnais. Snapshots et logs des jobs exacts sont archivés dans `opt-63-verifier/ci-ba706-*`, aucune CI globale verte annoncée.

## Microbench du vérificateur seul v3

M5/10cœursGPU/24Gio, macOS27.2, MLX0.32.2, secteur100%, applications laissées ouvertes. Cible65c89531 EXL3 2,05bpw,53slots/couche/3Gio,4lecteurs, allocateur512Mio, Hot1/WIDE0/overlap1/lookahead0/LES0. Une séquence fixe avec cache réchauffé,1warmup+5mesures par processus, HCCH pour chaque largeur,16processus/80échantillons. Mêmebinaire, seuleoption63 varie, parité de tous les logits et de la continuation. Pas de compilation/proveur/autre modèle concurrent.0Swapout jusqu’au postflight30s.

| Largeur | Latence témoin, deux bras | Latence63, deux bras | Réduction de latence | Gain de débit debloc |
| --- | --- | --- | --- | --- |
| B2 |102,370/102,449ms |92,491/92,703ms |9,651/9,513% |10,682/10,514% |
| B3 |174,819/175,262ms |167,306/167,370ms |4,297/4,503% |4,490/4,715% |
| B4 |230,122/229,710ms |205,659/205,415ms |10,630/10,577% |11,895/11,827% |
| B8 |465,597/464,978ms |408,831/406,565ms |12,192/12,562% |13,885/14,367% |

Dérive témoins≤0,254%, étendue des gains≤0,483point ; les quatre largeurs passent les seuils préenregistrés. Réchauffé ne signifie pas entièrement résident : demandes par bloc identiques entrebras,0Mo B2,2,496Mo B3,13,728Mo B4,307,011Mo B8. La répétition d’une trace réduit les misses et diffère d’une génération réelle. EnB8, pic physique témoin7,847–7,852Go contre7,173–7,178Go avec63 ; ce résultat ne prédit pas le pic d’un runtimeDFlash adaptatif complet.

[Résultats exacts](measurements/flash-next-ssd-impl/opt-63-verifier/micro-speed-v3/analysis.json). Le débit debloc `H/C−1` et la réduction de latence `1−C/H` sont distincts. Aucun de ces pourcentages n’est présenté comme un gain de tok/s de génération.

## Campagne de génération v5

OFF puis MTP1 puis DFlash Q4/ring fixe3 et adaptatif7, MAIN/HOLD128sorties/127decode,1warmup+1mesure/processus, HCCH,32bras ; cache3Gio et WIDE0 inchangés. Seulflag63 varie dans un mode, SHA unique de CLIv5 `b49cc73c6ea18ab83cc43878be3b64d9d06773cbb3a212ff5f44c22df49de4e1`. Min1% dans chaque paire, dérive témoin≤5%, étendue≤3points, textes/empreintes identiques au témoinOFF historique exact, pic≤12e9octets et0nouveauSwapout jusqu’au postflight30s.

Les32processus sont joints,64réponses exactes warmups compris. Postflight16échantillons/30,005s :0nouveauSwapout,13 074 432octetsSwapin cumulatifs depuis origine63, secteur100%. Les fichiers initiaux ne sont pas réécrits : la [clôture séparée](measurements/flash-next-ssd-impl/opt-63-verifier/speed-v5-closure.json) confirme le contrôle mémoire final. Le statut global `confirmed_gain:true` signifie qu’un mode est qualifié, pas que tous le sont.

| Mode | Tok/s63 MAIN, deux bras | Tok/s63 HOLD, deux bras | Gains appariés MAIN | Gains appariés HOLD | Décision |
| --- | --- | --- | --- | --- | --- |
| OFF |10,767/10,763 |9,330/10,378 |−0,673/+0,675% |−9,675/+3,696% | B1 inchangé, aucun gain ; outlierHOLD conservé |
| MTP1 |11,146/11,225 |10,259/10,257 |+6,182/+6,526% |+6,343/+6,871% | **Gain validé dans ces fenêtres** |
| DFlash fixe3 |9,762/9,842 |9,580/9,503 |+8,108/+11,138% |+11,714/+9,880% | **Non concluant** : étendueMAIN3,030points>3 |
| DFlash adaptatif7 |10,867/10,935 |9,627/9,767 |+7,036/+7,203% |+6,199/+12,111% | **Non concluant** : étendueHOLD5,912points>3 |

Les témoins MTP sont10,497/10,538 MAIN et9,647/9,598 HOLD, dérive0,391/0,512%, étendue0,344/0,527point. Le candidat MTP atteint7,920–7,926Go physiques ; le maximum des témoins est8,201Go mais les autres témoins sont aussi proches7,920Go : pas d’économie mémoire MTP universelle revendiquée. Le contrôleOFF indépendant comporte une fenêtreHOLD instable, ce qui limite les comparaisons entre modes ; le gain MTP est celui des paires internes au mode, pas un gain extrapolé depuis un autre moment. Les15tok/s ne sont pas atteints.

Le pic candidat fixe est8,222Go contre8,429Go pour les témoins ; adaptatif8,263Go contre8,934Go. L’adaptatif décide selon le temps mesuré : MAIN vérifie136lignes côté témoin et138 côté candidat, HOLD136–137 ; les routes/demandes ne sont donc pas fixes même si le texte est identique. Les comparaisons exactes par bras sont dans [speed-v5-result.json](measurements/flash-next-ssd-impl/opt-63-verifier/speed-v5-result.json).

Sur fixeMAIN, les182lignes et28 849 045 392octets demandés sont identiques entre bras : vérification13,312/13,546s→12,263/12,159s. Cela correspond à95,742–96,558ms/sortie debloc et66,809–67,379ms/ligne avec63. FixeHOLD :186lignes,29 511 739 764octets,98,452–99,280ms/sortie et67,222–67,788ms/ligne. AdaptatifMAIN :81,546–82,093ms/sortie et75,046–75,549ms/ligne ; HOLD :89,933–91,215ms/sortie et83,320–83,891ms/ligne, avec1tokenterminalordinaire séparé. Ces métriques hôte inclusives ne mesurent pas le coûtGPU exclusif du replay ou des lectures.

Le palier63C ci-dessous compare séparément la combinaison63+WIDE, sans additionner les gains. **Le nouvel essai de cache vers10Go est annulé à la demande de l’utilisateur**, avant lancement ; garder3Gio et poursuivre les mécanismes Splash après63. Aucun nouvel essai de cache agrandi ni installation d’app n’a été lancé.

## Interaction63C avec GPU Hot Wide

Mêmes sources146 et4artéfacts de l’identitév5, mêmes3Gio/53slots/Q4/ring etCLI, flag63=1 partout ; WIDE0/1 seul varie. Les36préfixes passent, puis un bref changement d’alimentation interrompt le runtime. La reprise limitée aux20configurationsruntime/136états et43contrôlesbridge passe, avec annulation/reprise, états exacts et postflight30s. Le premier `failed_postflight` reste conservé, sans réécriture en succès. Une erreur d’affichage ad hoc `len(checks)` sur un compteurentier est corrigée avant vitesse ; elle n’a relancé aucun test ni changé le résultat du bridge.

L’utilisateur avait ouvert puis referméMLXL3 **après** la fin de63V :1 277 624 320octetsSwapout supplémentaires sont observés hors mesure, aucun moteur restant. La cause exclusive système n’est pas prouvée et les résultats63V antérieurs ne sont pas réécrits. Une origine63C distincte est explicitement enregistrée après cet incident ; les compteurs et horodatages sont conservés.

La première campagne de vitesse termine les8brasfixes, tous joints et exacts. GainsWideMAIN−0,346/+2,826%,HOLD+0,558/+0,017%, étendueMAIN3,172points>3 : **non concluant**, pas de gain additionnel établi. Sur l’adaptatif, troisbrasMAIN terminent puis le dernier témoin est arrêté par la limite cumulée de256MioSwapin, avantHOLD. Le résultat reste `failed_postflight/parity:false/confirmed_gain:false`, même si tous les bras terminés sont exacts et aucun nouveauSwapout n’est observé.

Après nettoyage :365 510 656octetsSwapin cumulés depuisorigine63C,0nouveauSwapout. Une répétition bornée des8brasadaptatifs seulement, avec nouveaux témoinsHCCH MAIN/HOLD, est préenregistrée après30s sansmodèle. OrigineSwapin distincte, mêmeborne256Mio et0Swapout sur l’ensemble, pré20/post30, mêmes source/binaire/limites ; ni les3brasA initiaux ni un seuil modifié ne sont mélangés à cette reprise. La fixe non concluante n’est pas relancée.

**Cette reprise est terminée et qualifiée** :8processusjoints/16réponses exactes ; postflight16échantillons/30,005s, secteur100%,0nouveauSwapout sur l’ensemble63C et851 968octetsSwapin dans la nouvellefenêtre courte. Le cumulSwapin depuisorigine63C initiale reste366 362 624octets, donc l’échec long n’est pas effacé ni transformé en succès sous256Mio. [Clôture](measurements/flash-next-ssd-impl/opt-63-verifier/combined-wide-v5/adaptive-short-v5/closure.json), [résultat brut](measurements/flash-next-ssd-impl/opt-63-verifier/combined-wide-v5/adaptive-short-v5/speed-result.json).

| Prompt adaptatif |63seul, WIDE0 tok/s |63+Wide, tok/s | Gains appariés |
| --- | --- | --- | --- |
| MAIN |10,7223/10,7237 |11,0127/10,8989 |+2,708/+1,634% |
| HOLD |9,6593/9,4279 |9,7673/9,7384 |+1,118/+3,293% |

Dérive témoinsMAIN0,014%,HOLD2,454% ; étendue des gains1,074/2,175points, sous les limites5%/3points, chaque paire≥1%. Pic physique maximal8 287 033 872octets, sans économie mémoireWide revendiquée. MAIN garde138lignes/77acceptées/50blocs ; HOLD139–140lignes et64–66acceptées/60–62blocs : le gain inclut les décisions adaptatives influencées par le temps. Il ne mesure pas isolément unkernel ou les seules lecturesSSD. Aucun nouveau gainOFF/MTP attribué àWIDE, inactifB1/B2. Ce n’est pas une comparaison contemporaine DFlashcontreMTP : les chiffresMTP63V ont leur propre fenêtre. **15tok/s non atteints.**

63V et63C sont clos ; aucune autre relanceWide prévue, défauts0 conservés. Sources exécutables identiques àba7068fb, pas de nouvelle compilation ni de nouveau kernel pendant63C. Les interruptions d’alimentation et de swap-in, les réponses incomplètes et les diagnostics corrigés restent archivés.

Les mécanismes encore absents et leur priorité conditionnée par le profil sont détaillés dans la [rechercheSplash64R](flash-next-ssd-splash-64-research-2026-10-21.md). Les87–89% de vérificationDFlash ne constituent pas une partGDN ; aucun kernel64 ni profilGPU exclusif nouveau n’est revendiqué.
