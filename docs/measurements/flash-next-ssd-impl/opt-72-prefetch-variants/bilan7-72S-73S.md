# Bilan des sept candidats 72S/73S

Les sept candidats ont leurs mesures terminées :48rapports et48warmups,96réponsesgolden60exactes. Après le vrai A/B contreB2sansprélecture, l’utilisateur retire weighted et annule la confirmationCH. Le bridge43weighted reste validé ; le retrait concerne le gain attendu et l’intégration. Les8référencesavant,8après et8réponsesduvraiHC sont séparées. Les anciennes pairesvslegacy et la consigne72W-KEEP restent historiques. Publication documentaire seulement, aucun15tok/sannoncé.

## Débits par paire

Chaque cellule donne H→C en tok/s, puis C/H. Pour72,H=legacy avec prélecture ; pour73,H=EARLY_SHARED0. C ne change que la variante. Les plagesHCCH décrivent les paires observées, pas des intervalles de confiance. Aucun gain n’est additionné. Une baisse isolée ne démontre pas une régression causale.

| Candidat / mode | MAIN HC | MAIN CH | HOLD HC | HOLD CH | Décision |
|---|---:|---:|---:|---:|---|
| packed/B2 | 9.1224→10.3060 (+12.97%) | 8.7143→9.8095 (+12.57%) | 8.6952→9.6261 (+10.71%) | 7.8341→9.5230 (+21.56%) | Ancien critère non satisfait |
| late/B2 | 8.5505→8.4608 (-1.05%) | non lancé | 8.2134→8.9643 (+9.14%) | non lancé | Écran unique, non promu |
| bounded/B2 | 8.6713→8.4094 (-3.02%) | non lancé | 8.6195→8.4673 (-1.77%) | non lancé | Écran unique, non promu |
| unused/B2 | 8.8545→9.0754 (+2.50%) | 8.7708→8.6147 (-1.78%) | 8.7373→8.8827 (+1.66%) | 8.3899→8.4129 (+0.27%) | Ancien critère non satisfait |
| lfu/B2 | 8.8104→9.1018 (+3.31%) | 8.6272→8.7462 (+1.38%) | 8.3728→8.7143 (+4.08%) | 8.4740→8.5452 (+0.84%) | Ancien critère non satisfait |
| weighted/B2 | 7.9931→9.5879 (+19.95%) | 7.9958→9.3161 (+16.51%) | 8.7403→9.3329 (+6.78%) | 9.5971→10.7723 (+12.25%) | Retiré après vraiA/B sansprélecture |
| early-shared/OFF | 8.4617→8.0781 (-4.53%) | non lancé | 8.1190→8.4665 (+4.28%) | non lancé | Écran unique, non promu |
| early-shared/B2 | 9.5427→10.1179 (+6.03%) | non lancé | 9.0688→8.6831 (-4.25%) | non lancé | Écran unique, non promu |

Weighted donne quatre paires positives vslegacy : MAIN+16,512950 à+19,951846 %, HOLD+6,780078 à+12,245518 %. L’utilisateur avait accepté cette variation sous72W-KEEP, puis a retiré la piste après le vraiA/B72W-R contre sansprélecture ; le lotHC est maintenant terminé etCH annulé.

## Critères historiques et mémoire

| Candidat | MAIN dériveH / étendue gains | HOLD dériveH / étendue gains | Échecs stricts conservés |
|---|---:|---:|---|
| packed | 4.6830% /0.4070pt | 10.9914% /10.8530pt | HOLD: H drift >5% (10.991) ; HOLD: gain spread >3 points (10.853) |
| unused | 0.9535% /4.2763pt | 4.1405% /1.3898pt | MAIN: pair gain <1% (-1.781) ; HOLD: pair gain <1% (0.275) ; MAIN: gain spread >3 points (4.276) |
| lfu | 2.1235% /1.9278pt | 1.2092% /3.2393pt | HOLD: pair gain <1% (0.840) ; HOLD: gain spread >3 points (3.239) |
| weighted | 0.0333% /3.4389pt | 9.8024% /5.4654pt | MAIN: gain spread >3 points (3.439) ; HOLD: H drift >5% (9.802) ; HOLD: gain spread >3 points (5.465) |

Seuils stricts72S d’origine : chaque paire≥1 %, dériveH≤5 %, étendue≤3points, surplus physiqueC≤256Mio. L’enum historique inconclusive_drift inclut une dispersion excessive des gains : unused/LFU passent les deux dérivesH, et échouent sur certaines paires/étendues. Weighted conserve ses anciens échecs stricts et l’historique72W-KEEP, désormais révoqué. Le retrait actuel vient de la vraie comparaison sansprélecture défavorable, pas des seuils de variation.

| Candidat / mode | Pic physique | Swapouts système sommés | Ledgerdelta propre | Compression77B |
|---|---:|---:|---:|---|
| packed/B2 | 7853988128B | 458752B | 0B | failed conservé |
| late/B2 | 7852300552B | 0B | 0B | failed conservé |
| bounded/B2 | 7852661048B | 0B | 0B | failed conservé |
| unused/B2 | 7852595464B | 0B | 0B | failed conservé |
| lfu/B2 | 7853807904B | 0B | 0B | failed conservé |
| weighted/B2 | 7853824288B | 0B | 0B | failed conservé |
| early-shared/OFF | 7153358752B | 0B | 0B | failed conservé |
| early-shared/B2 | 7850121528B | 0B | 0B | failed conservé |

Pic maximal des débits 7853988128B, sous12Go. Ledger_swapins logique du moteur sans hausse. Swapouts système totaux 458752B, dont458752B dans packedMAINH, sous le plafond stock256Mio par phase ; ils ne sont pas attribués au modèle. Les caps77B de compression restent échoués : actuel/pic logique512Mio et lifetime depuisready1Gio dépassés. Les48analyses ont leur hashsource revérifié, source_errorsvide et3snapshots valides ; toutes sontfailed. Aucun seuil n’a changé et aucun échec n’est requalifié. Les deltas warm→mesure descriptifs restent dans les résumés de phase. Ledgerstable ou RSSstable ne prouvent pas l’absence d’écritures swap privées.

## Références fraîches séparées

| Mode sans prélecture | MAIN avant | HOLD avant | MAIN après | HOLD après |
|---|---:|---:|---:|---:|
| B2 | 10.222852654 | 9.486925854 | 11.021135664 | 10.269771465 |
| MTP | 9.833896963 | 8.907714222 | 10.799178310 | 9.903072863 |

Ce sont des références non appariées à chaque candidat. Le point weightedHOLD10,772262 dépasse l’ancien B2HOLD9,486926, sans démontrer un gain causal contre le no-prefetch. Les chiffres70S2 sont historiques et ne sont pas mélangés à cette campagne. Les8réponses avant et les8après sont comptées séparément des96réponses candidats. Les références après sont+7,81/+8,25 % pourB2 MAIN/HOLD et+9,82/+11,17 % pourMTP : variation descriptive des extrémités, pas gain d’optimisation. [Rapport après](../opt-77-process-memory/fresh-controls-after-v1/report.md) :8oracles/modes confirmés, pré20/post30AC/enfants/lecteurs/sampler joints, pic7,888149Go, globalin/out0 etledger0. Quatreanalyses77Bfailed conservées avec donnéesvalides. La nouvelle comparaison72W-R contrelenofetch sera appariée ; ces anciens points n’en tiennent pas lieu.

## Qualité et portée

[Composants](component-v1/summary.json) :112casMoE B1–B8, capacités24/53, HotCold/poison ;96formes EarlyShared supplémentaires. [Qualité nativev2](quality-native-v2/summary.json) et sixpréfixesv1 inchangés totalisent288contrôles préfixes/logits/captures/états : huit configurations×36, largeurs1–8 ; sessionMTP et10pairesMTP. Captures observées byte_equaltrue/count64000, garde de tolérance0,004 explicitée ; logits/états cibles byte_exact. Ces domaines finis ne prouvent pas tous les contextes ni le QSA long4096.

Deux candidats ont leur bridge43validé : [packedv3](bridge-packed-v3/summary.json) et [weightedv4](bridge-weighted-v4/summary.json), chacun33completeevents etannulations/erreurs/reprises prévues. Les cinqautres candidats restent non exécutés au niveaubridge43. Weighted a été retiré après sa validationfonctionnelle, pour la décision de performance. Aucun transfert des43contrôles à tous les candidats. Les48enfants mesurés, lecteurs et samplers sont joints, pré20/post30 sur secteur. Toutes les96réponses ont texte complet non vide, token_hash et comptes128sorties/127decode exacts golden60. Les arrays bruts d’IDs ne sont pas persistés ; les débits ne constituent pas un nouveau contrôle direct logits/états.

## Commandes, preuves et CI

Les run.json de chaque phase contiennent les commandes exactes, variables et timeouts. Lancement réel par observe-stock.py PLAN, réutilisant les superviseurs45/49 et la courbeVM2s. Chaque enfant utilise benchmark-flash-bridge.py CLI MODEL OUTPUT, tokens128/repeats1/cache-gib3/promptMAIN ouHOLD, optiondraft/mode1 pourB2. Un modèle physique à la fois, sans compilation/proveur concurrent. L’analyseur de mémoire77B est réutilisé sans nouveaueval ni modèle. [Manifest](screens-v3-72S-prepared.json), puis weightedv4 sous speed-v4/weighted.

[ContrôlesCPU77](../opt-77-process-memory/summary-cpu.json) : fmt/Clippystrict/buildmlx,chat réussis ;159testspassés/159ignorés, lesignorés ne sont pas des tests physiques exécutés. SDKC++32combinaisons et frontières zéro/signé/max sousASan/UBSan. [Analyseur](../opt-77-process-memory/analyzer-cpu/summary.json) :220testspassés, mutationPID détectée/restaurée ; CrossHair tenté, module absent, pas de preuvePython. Aucun test lourd relancé par cet analyste pendantGPU.

[Kani72](kani-full-width.json), Kani0.68 :1643checks passés,9covers comptées séparément,20checks de bibliothèque inatteignables. Helper : ids/budgetusize et bool arbitraires sansassume, keep≤min(ids,budget,4). Victim : cache10, slots occupied/pending/prefetch/predicted et compteursu32/âgesu64 symboliques, une route issue deu8, unwind12. Politique : cache10, deuxu32 et deuxbools arbitraires, IDs valides<512, unwind514. Commande exacte dans[kani-full-width-process](kani-full-width-process.json) : cargo kani --lib --no-default-features --jobs 1, trois --harness ciblés, timeout180s et exportJSON. Model checking CPU borné, pas de preuveMetal/MLX/concurrence. [Vraie voieMLX](kani-actual-mlx-v3.log) tentée : ICEintrinsics243 avant obligations, non prouvée.

[Kani77](../opt-77-process-memory/verification-summary.json) : vrai décodeur12u64 arbitraires+PIDu32 sansassume, unwind13,127obligations passées et3covers séparées. Commande cargo kani --lib --features mlx,chat --harness memory_stats_decoder_preserves_validity_and_values, timeout180s/exportJSON. CBMC6.11 sur le vrai encodeurSDK : cpp11/unwind13/unwindingassertions/bounds/pointers/signed+unsignedoverflow ; parseApplelibc++ unsupported avant obligations. Aucun stubSDK ni réécriturePython présentés comme preuve. APIOS, attributionSwapout, FFI/SDKC++ etGPU restent non prouvés.

[NativeCI89174final](../opt-77-process-memory/ci-89174fcc/run-39958506522-final.json) :completed/failure ; Linux/macOS/FlashKani success, globalKani runner shutdown après31harnais successful et32e sansverdict. Cause inconnue, aucuncontre-exemple observé, aucunexit143 dans ce log77. [Regressionfinal](../opt-77-process-memory/ci-89174fcc/run-39958506412-final.json) :successDesktop/protocole. La CI du commit77 publié ne valide pas les calculs72/73 locaux. Aucune CIentière verte annoncée.

## Provenance, incidents et intégration

MacM5 utilisateur, secteur aux frontières. CibleQwen3.8-Flash-Next-exl3-205-65c89531, révision65c895314393431c09050b2e04e250836b3a6eb4 ; draftQwen3.8-Flash-Next-DFlash2-a22a1744, révisiona22a1744e85c17364dcc54a6a420af7135d4c7f6, Q4/ring. Cache3Gio53slots, allocateur64Mio, contexte4096, greedy, B2WIDTH1/ADAPTIVE0 ; pipeline1/lookahead2 pour72, pipeline0/lookahead0 pour73. CLI77v1 SHA69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c, runtime1.4.3/250488c11f1c-dirty/release/MLX0.32.2. Binaire localmixte, non reconstruit depuis le HEAD publié89174fcc.Lors des contrôles de mesure,165sources étaient rehashées, avecseuleCI modifiée. Après lesessais, les6sourcesRustprototypes etlaCI sont restaurées auHEAD89174fcc ; cetétatactueln’estpaslasourceduCLI69gelé etne réécritpaslesrésultatshistoriques. Révisions checkpoint archivées ; pas de nouveauSHA de tout le payload>60Go.

Échecs conservés : packedbridgev2failed34/43/+49479680Bouts selon son ancien protocole ; weightedv3interrompu sur livepowerchanged avant première mesure, settlementpost30 séparé. Weightedv4 utilise son propre témoin sans recycler les donnéesv3. La fixture weightedv1 a échoué sur le compte prévisions, corrigé uniquement souscfgtest ; oracles et production inchangés, anciennes preuves conservées.

71P/78T préparés non appliqués,65Ppréexistant exclu. App installée inchangée. Weighted retiré par l’utilisateur ; publication documentaire seulement, sourcesprototypes retirées aprèsarchivage par le parent ; autres candidats expérimentaux/nonpromus. Le vraiHC72W-R contre no-prefetch est terminé : MAIN−1,557172 %, HOLD−18,111644 %. La confirmationCH est annulée/nonlancée à la demande utilisateur. Aucune conclusion ancienne vslegacy ne valide cette nouvelle comparaison.

Validation propre àweighted : [résumé43](bridge-weighted-v4/summary.json),33completeevents/6erreurs/2annulations attendues,43requêtes terminées etactiveNone. Mémoire et préfetcheffectif conservés,77Bfailed nonrequalifié. La qualificationfonctionnelle estclose, mais la dernière consigne utilisateur annule la publication du codeopt-in : documents etarchivesseuls.

## Décision finale —72W-WITHDRAW

[VraiA/BHC](speed-72W-R/screen/summary.json) :MAIN H11,301206/C11,125227tok/s(−1,557172 %),HOLD H10,716598/C8,775646(−18,111644 %). Huitréponsesexactes,joints/AC/globalouts0/ledger0 ;77Bfailedconservé. L’utilisateur retire weighted, car il voulait un gain contre la meilleure référence. Aucun gainvsmeilleur établi ; unseulHC n’estpas une preuve de régressionuniverselle. [CHnonlancé](speed-72W-R/confirmation/not-run.json) annulé explicitement. AncienKEEP reste historique, variantesprototypes retirées aprèsarchive parparent, appinchangée.

Étatfinal vérifié : [manifest de retrait](withdrawn-runtime/manifest.json) et [patcharchivé](withdrawn-runtime/prototype-and-test-cfg.patch), SHA938b2e9c541b43f8af3c72894dc442edaa73e752c6712cd8bdf12783cb9e7498. Les7hashesrestaurés correspondent auHEAD89174fcc, CLI69inchangé. Lesprototypesne sont plusactifsdanslessources, aucun codeweightedàpublier ; preuvesdocumentairesetarchivesuniquement.
