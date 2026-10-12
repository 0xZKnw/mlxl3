# Flash Next EXL3 2,05 bpw avec streaming SSD

État : parcours texte et MTP1 implémentés, contrôlés et mesurés sur M5 ; aucun gain de MTP établi sur les deux prompts testés, défaut OFF conservé. Aucun remplacement de l'app installée.

## Lectures SSD parallèles, 12 octobre

Les experts absents sont désormais lus par quatre lecteurs bornés, directement dans les banques existantes. Toutes les lectures se terminent avant le retour, y compris après erreur. Le LFU et le budget RAM adaptatif sont conservés ; `MLXL3_SSD_READERS=1` active les lectures séquentielles (valeurs permises : 1 à 4).

Le [rapport des lecteurs](flash-ssd-readers-2026-10-12.md) conserve une première campagne non concluante et une confirmation distincte préenregistrée. Sur ce M5, à budget experts fixe de 5 Gio, les douze nouvelles paires donnent un gain de 5,5–10,4 %, avec sorties exactes et zéro nouvelle écriture de swap système. Médianes candidat : OFF 7,29/7,05 tok/s, MTP1 7,13/6,89 tok/s pour MAIN/HOLD. Cela ne valide ni 15 tok/s ni un débit constant dans d'autres conditions.

## Correction du budget RAM, 12 octobre

La capacité fixe de8Gio est remplacée par un cache automatique plafonné à6Gio, avec une réserve estimée au chargement pour les autres usages. Le LFU garde les experts fréquents ; les autres sont lus sur le SSD. Le cache de buffers MLX inutilisés vise512Mio. Les variables explicites conservent la priorité. La tête MTP reste résidente lorsqu'elle est active, pour environ0,665Gio supplémentaires.

Comparaison A/G/G/A sur le M5/24Gio avec les applications utilisateur ouvertes, quatre réponses MAIN/HOLD × OFF/MTP64 par processus, même contexte4096 et mêmes sorties :

| Passage | Experts en RAM | Pic processus | Nouvelles écritures swap système |
|---|---:|---:|---:|
| A, ancien défaut | 7,98Gio,143slots/couche | 12,98Gio | 0Mio |
| G, automatique | 4,63Gio,83slots/couche | 9,31Gio | 0Mio |
| G, automatique | 5,91Gio,106slots/couche | 10,60Gio | 0Mio |
| A, ancien défaut | 7,98Gio,143slots/couche | 12,98Gio | 262,88Mio |

Réduction d'empreinte processus observée de2,38à3,67Gio. Les compteurs swap/compression sont globaux ; ils ne prouvent pas à eux seuls la causalité ni l'absence de swap futur. L'estimation est faite au chargement et peut changer entre deux processus. Secteur100% et absence d'alertepmset relevés pour les trois derniers passages ; alimentation non consignée séparément pour le premier. Les timings échantillonnés de ce diagnostic ne valident pas un gain de vitesse. [Matrice et limites](measurements/flash-next-ssd-impl/opt-12-memory/G-memory-summary.json).

Le binaire G `938e2aa7b3532c608607c852810313bf9fe6630b021d8560cffb992c64b1478e` vient deee114bf6 avec les modifications source identifiées dans [G-identity.json](measurements/flash-next-ssd-impl/opt-12-memory/G-identity.json).19contrôles physiques passent : oracles logits/états/composants, MTP/T2/commits, cohortes froides/chaudes, erreurs de lecture, plafond MLX et overrides. Le bridge passe16générations ABBA aux budgets1/2/3/17 et deux annulations/reprises. Format, Clippystrict,84testsCPU/2ignorés et104testsrelease `mlx,chat`/94ignorés passent. Les contrôles physiques sont sélectionnés séparément. [Commandes build/tests](measurements/flash-next-ssd-impl/opt-12-memory/G-build.json), [qualité physique](measurements/flash-next-ssd-impl/opt-12-memory/G-quality.json), [bridge](measurements/flash-next-ssd-impl/opt-12-memory/G-bridge.json).

Kani vérifie les deux nouveaux contrats du budget sur le domaineu64 complet sans hypothèse :187obligations,6couvertures,0inatteignable ; la suite ciblée de huit contrats passe1024obligations/27couvertures/3inatteignables. Les chemins réels OS/MLX et le package complet échouent en compilation du vérificateur (`intrinsics.rs:243`), sans propriété GPU prouvée. CrossHair ne confirme pas le parseur VM. Le test initial de l'allocateur demandait à tort une limite immédiate au free ; échec conservé et test corrigé suivant le contrat MLX de récupération à l'allocation suivante. Le prototypeF_NOCACHE et l'échec initial du superviseur de bridge restent archivés ; ils ne qualifient pas le code G. [Tentatives formelles](measurements/flash-next-ssd-impl/opt-12-memory/formal.json), [journal](../opti.md).

La comparaison [Sushi/Strata](flash-ssd-upstream-2026-10-12.md) retient les lectures manquantes parallèles comme piste suivante. NT2 et le regroupement MTP restent différés ; aucun gain n'est annoncé pour ces prototypes. L'objectif15tok/s reste non atteint.

La CI macOS26 a détecté un refus incorrect des révisions plus courtes de `host_statistics64` (commit7fcea298). La correctionH accepte une réponse couvrant les champs lus et rejette toute taille hors capacité ; le calcul du budget reste identique. La régression échoue avant correction puis passe, avec le vrai snapshot système et les frontières. Kani du prédicat réel couvre tous les comptesu32 sans hypothèse :66obligations/4couvertures, aucune inatteignable. La tentative de l'appelOS réel reste bloquée par l'ICE du vérificateur.

Le rebuildH `5e423506d0e39b8d71c0e31048951c3a73c018eb5a9b66b7a6b3da1109fe71a2` passe fmt/Clippy,86testsCPU (2ignorés),106testsrelease (94ignorés) et les neuf contrats CPU ciblés (1090obligations/31couvertures/3inatteignables). Les deux contrôles physiques affectés refont les16positions exactes ; huit générations bridge OFF/MTP de16tokens passent avec mêmes textes/hashes et préfixes identiques au binaireG. Aucun timing de ce contrôle fonctionnel n'est utilisé comme débit. Les mesures RAM et vitesse restent rattachées au binaireG, dont H ne modifie que la garde de format au chargement et le diagnostic d'erreur. [Sources](measurements/flash-next-ssd-impl/opt-12-memory/H-identity.json), [build](measurements/flash-next-ssd-impl/opt-12-memory/H-build.json), [qualité](measurements/flash-next-ssd-impl/opt-12-memory/H-quality.json), [vérificateurs](measurements/flash-next-ssd-impl/opt-12-memory/H-checks.json).

La CI du commit de code `fe998151df222be3bdc3541b0c59e90de64b0b68` passe [Linux/macOS avec build MLX et Kani ciblé](https://github.com/0xZKnw/mlxl3/actions/runs/38249129330), ainsi que [Desktop/protocole](https://github.com/0xZKnw/mlxl3/actions/runs/38249129264). Le job Kani complet est encore en cours à l'inspection et n'est pas qualifié. Le compte rendu final est une modification documentaire ; les runs de ce nouveau SHA restent distincts. [État des jobs inspectés](measurements/flash-next-ssd-impl/opt-12-memory/ci-fe998151-inspection.json), [régression réparée sur le vrai runner macOS26](measurements/flash-next-ssd-impl/opt-12-memory/ci-fe998151-macos.log).

## Débit du cache automatique, 12 octobre

Le même binaireG a été mesuré séparément du diagnostic mémoire, avec le benchmark `--compare --auto-cache` : cache effectif5,24Gio/94slots par couche, contexte4096, greedy, overlap1, quatre warmups puis trois répétitions par prompt/mode,64tokens/63decode. Aucun échantillonnage VM pendant le décodage, aucun autre modèle/compilateur/proveur. Les12mesures conservent les hashes et textes historiques exacts.

| Prompt | Sans MTP, médiane (étendue) | MTP1, médiane (étendue) |
|---|---:|---:|
| MAIN | 6,927tok/s (6,794–6,947) | 6,956tok/s (6,767–6,970) |
| HOLD | 6,965tok/s (6,790–6,980) | 6,764tok/s (6,579–6,779) |

Secteur100%, aucun avertissementpmset. Compteurs système relevés avant/après seulement : aucune nouvelle écriture swap, compresseur+622,08Mio ; les autres applications restent ouvertes. Ces mesures décrivent le compromis du nouveau défaut automatique, sans gain causal revendiqué face à la campagne historique à8Gio. MTP reste OFF par défaut et15tok/s ne sont pas atteints. [Mesures brutes](measurements/flash-next-ssd-impl/opt-12-memory/S-speed/report.json), [conditions et contrôle des sources](measurements/flash-next-ssd-impl/opt-12-memory/S-execution.json).

## Débit historique à cache8Gio, base 1.4.3

Source `2162c5f64a16558e1443758b4dbf22ba09566a78`, release `mlx,chat` SHA256 `cbff15f63ae1343344b4cd7b1530d09e180c780adfb153e12f227552564b39fa`, M5/24Gio/MLX0.32.2. La composition avec 1.4.3 est requalifiée avant mesure. Même protocole OFF/MTP1 que ci-dessous : contexte4096, cache8Gio/overlap1, greedy, quatre warmups puis trois répétitions de64tokens/63decode par couple, modes alternés dans un seul bridge. Aucun compilateur, prouveur ou autre modèle pendant cette campagne. [Rapport brut](measurements/flash-next-ssd-impl/mtp-speed-143/report.json), [résumé et conditions](measurements/flash-next-ssd-impl/mtp-speed-143/summary.json), [empreintes avant/après](measurements/flash-next-ssd-impl/mtp-speed-143/identity.json).

| Prompt | Sans MTP | MTP1 | Écart médian | Acceptation |
| --- | ---: | ---: | ---: | ---: |
| MAIN, cache LFU | **7,60 tok/s** (7,599–7,610) | **7,54 tok/s** (7,501–7,552) | −0,79% | 28/34 (82,35%) |
| HOLD, merge_intervals | **7,57 tok/s** (7,537–7,579) | **7,31 tok/s** (7,285–7,332) | −3,43% | 27/35 (77,14%) |

Les16générations, warmups inclus, ont les mêmes hashes/textes pour chaque prompt entre modes, répétitions et campagne06. MTP est inférieur à OFF dans chacune des six paires ; les étendues relatives OFF sont0,14%/0,55%. **Gain MTP rejeté sur ces deux cas, défaut OFF conservé.** La tête intégrée et ses commits restent fonctionnellement validés. Ces chiffres ne sont pas une comparaison causale avec la campagne06, dont la source et les conditions diffèrent.

TTFT médian MAIN OFF12,21s/MTP12,35s, HOLD11,44s/13,38s ; prefill MAIN7,13/7,04tok/s, HOLD7,17/6,62tok/s ; temps complet MAIN20,48/20,71s, HOLD19,76/22,02s. Les réactivations MTP incluent le rechargement de sa tête. Actif MLX≈11,147/11,812Gio (+0,665Gio), pic maximal11,207/12,042Gio ; RAM processus≈12,98Gio dans les deux modes, mesurée séparément. Secteur, batterie94→96% en charge, swap4246,38→4241,12Mio, aucune alerte pmset ; températures et octets SSD physiques non mesurés. Aucun remplacement de l'app installée ni release.

## Comparaison historique corrigée avant l'intégration 1.4.3

Source publiée `d506cc78243e680668aaf73961c56824852b4798`, release SHA256 `e89da70ad32906743e332702575053ead81922a42d85d3f33a65e277bf434dfb`, M5 avec 24 Gio et MLX 0.32.2. Même processus et binaire, contexte de 4096 tokens, cache experts de 8 Gio, overlap1 et greedy ; MCP, DFlash et cache préfixe OFF. Quatre warmups puis trois passages de 64 tokens (63 en decode) par couple prompt/mode, ordre alterné. [Résultats bruts](measurements/flash-next-ssd-impl/mtp-speed-resume-20261012/report.json), [médianes, étendues, mémoire et conditions](measurements/flash-next-ssd-impl/mtp-speed-resume-20261012/summary.json), [identités](measurements/flash-next-ssd-impl/mtp-speed-resume-20261012/identity.json).

| Prompt | Sans MTP | MTP1 | Acceptation MTP |
| --- | ---: | ---: | ---: |
| MAIN, cache LFU | **7,21 tok/s** (6,62–7,54) | **7,20 tok/s** (7,15–7,48) | 28/34 (82,35%) |
| HOLD, merge_intervals | **7,15 tok/s** (6,92–7,50) | **6,89 tok/s** (6,80–7,15) | 27/35 (77,14%) |

Les 16 générations, warmups inclus, donnent les mêmes hashes et textes dans les deux modes pour chaque prompt. L’écart médian MAIN de −0,17% est dans le bruit ; HOLD atteint −3,70%, avec MTP inférieur au contrôle dans les trois paires. Aucun gain MTP établi ; le mode reste disponible pour activation explicite. TTFT médian MAIN : 12,74 s/OFF contre 13,12 s/MTP ; HOLD : 12,07 s contre 15,88 s. Temps complet MAIN : 21,48/21,80 s et HOLD : 20,88/25,08 s ; les passages qui réactivent MTP incluent le rechargement de sa tête.

Actif MLX : environ 11,147 Gio/OFF contre 11,812 Gio/MTP (+0,665 Gio), pic maximal 11,207/12,042 Gio. RAM processus médiane : environ 12,977 Gio dans les deux modes, mesurée séparément ; ce processus conserve des allocations et caches. Sur secteur, batterie de 22 à 27% en charge, swap de 1931,38 à 4531,31 Mio, aucune alerte pmset. Températures et octets SSD physiques non mesurés ; deux prompts et trois répétitions ne démontrent pas un gain général. La précédente campagne sur batterie a été interrompue à 11/12 passages : [rapport incomplet conservé](measurements/flash-next-ssd-impl/mtp-speed/report.json), [état d’interruption](measurements/flash-next-ssd-impl/mtp-speed/interruption.json). Ses chiffres ne sont pas combinés avec cette reprise.

## Première mesure sans MTP, avant correction de la divergence

Binaire baseline `011dc68`, SHA256 `986a6a307a914e0be2cb8e429149a2b7c3a9f8f5d58771653a6315b7f0f175bc`, M5/24Gio, MLX0.32.2, contexte4096, cache8Gio et overlap1. Deux prompts fixes,64tokens chacun, warmup puis3répétitions alternées, greedy/MTPoff/MCPoff/préfixeoff. [Rapport complet](measurements/flash-next-ssd-impl/baseline-speed/report.json).

| Prompt | Decode médian | Étendue | TTFT médian | Prefill médian |
| --- | ---: | ---: | ---: | ---: |
| MAIN, cache LFU | 7,08tok/s | 6,77–7,59tok/s | 12,78s | 6,81tok/s |
| HOLD, merge_intervals | 7,03tok/s | 6,61–7,51tok/s | 12,50s | 6,56tok/s |

Six générations de64tokens,63decode chacune ; mêmes hashes et textes dans chaque prompt, y compris warmup. RAM processus11,70Gio ; actif MLX11,15Gio ; picMLX11,21Gio. Batterie45→40% en décharge, swap utilisé5736,94Mio en fin de mesure, aucune alerte pmset. Ce sont les performances du premier binaire sans MTP, pas un gain établi. La correction numérique ci-dessous a ensuite été validée ; cette première mesure reste attachée au binaire antérieur.

## Modèle et architecture

Checkpoint `turboderp/Qwen3.8-Flash-Next-exl3`, branche `2.05bpw_h4_ng4`, révision immuable `65c895314393431c09050b2e04e250836b3a6eb4`. Les sept fichiers LFS ont leurs tailles et SHA-256 vérifiés : [empreintes](measurements/flash-next-ssd-impl/checkpoint-hashes.json). Payload poids 58,3587 Gio, dont 24,4380 Gio de n-grammes. Le modèle reste sur le disque.

Le moteur couvre les 48 couches texte, les quatre flux HyperConnections, 36 blocs Gated DeltaNet, 12 blocs QSA et le PLE/n-grammes de la couche 1. Les embeddings BF16 sont lus par ligne. Les poids communs et la tête EXL3 restent résidents ; les experts K2/MUL1 utilisent des banques persistantes, un cache LFU borné et des lectures `preadv` directes. La cohorte entière est protégée avant toute éviction. Tous les dix experts sélectionnés sont calculés ; pas de suppression approximative d'expert. Le travail GPU sur les hits peut recouvrir la lecture des misses.

Sans `MLXL3_SSD_CACHE_GIB`, le cache experts est automatique : plafond6Gio, réduit selon la mémoire libre/fichiers estimée au chargement, avec8Gio réservés pour les poids résidents, MTP, les buffers et les autres usages. Le chargement échoue explicitement si même10slots par couche ne tiennent pas dans cette politique. C'est une estimation au chargement, pas une garantie contre le swap si les autres applications grossissent ensuite. Une valeur explicite de0,6 à12Gio conserve le contrôle manuel et peut dépasser ce budget prudent. Les anciennes mesures ci-dessus employaient8Gio.

Flash fixe aussi une cible de512Mio pour les buffers MLX inutilisés (`MLXL3_ALLOCATOR_CACHE_MIB` explicite reste prioritaire). MLX récupère l'excédent à l'allocation suivante : libérer un gros buffer peut donc dépasser temporairement cette cible ; le cleanup existant libère aussi le cache inutilisé au-delà de512Mio après une réponse. C'est le contrat de [l'allocateur MLX0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp). Les descripteurs restent attachés à l'inode inspecté ; le lecteur conserve le cache fichiers macOS. Le prototype F_NOCACHE a été retiré après un contrôle montrant un coût I/O important sans baisse claire de l'empreinte moteur. `MLXL3_SSD_OVERLAP=0` désactive le recouvrement pour les contrôles. Le cache n-grammes est borné à16384lignes. La réduction du cache experts peut augmenter les lectures SSD ; les résultats RAM et leurs limites sont consignés dans l'essai12 de [opti.md](../opti.md). Le nombre2,05bpw décrit le checkpoint publié, pas une nouvelle quantification faite ici.

## Correction de la divergence avant MTP

Les normes et opérations HC compilées, l’ordre d’accumulation EXL3 (ligne puis SIMD), la partition des experts et le cast avant les matrices denses GDN suivent désormais la référence. PLE utilise la sigmoid stable, la convolution native et la SiLU compilée. QSA calcule les fréquences comme MLX-VLM et utilise le même MRoPE FP32 avec cast final FP16 ; les positions restent des données, sans compilation par token.

Un défaut du premier oracle a aussi été corrigé : MLX démarre les modules en mode entraînement, qui choisissait une récurrence GDN différente. Le nouvel oracle appelle `.eval()` sur les véritables couches avant les wrappers d’observation et refuse un mauvais mode. Les anciennes fixtures et leurs échecs sont conservés.

Référence indépendante : MLX-VLM0.7.3, PonyExl3 `8e7fa6b1556f59fc669e25087903b279b9b0346f`, MLX0.32.2, même checkpoint. Les cinq tokens originaux et onze tokens tenus à part (français/chinois/code/EOS/dernier ID vocabulaire) couvrent16positions. **Les3 973 120logits, les32états GDN0/1 de786432valeurs chacun, les768cohortes et les16argmax sont identiques**, avec égalité bit à bit pour logits/états. Le seuil initial1% reste dans le test, renforcé par cette égalité. Les4 608sorties de composants passent leurs contrôles, dont les mélanges HC et sorties attention en bits ;40normes,20projections,6injections et200stades isolés sont exacts.

[Contrôle final des octets, original](measurements/flash-next-ssd-impl/divergence/byte-parity-original.log), [séquence tenue à part](measurements/flash-next-ssd-impl/divergence/byte-parity-held-02.log), [slots](measurements/flash-next-ssd-impl/divergence/slots-final-02.log), [sources et qualification](measurements/flash-next-ssd-impl/divergence/quality.json). Les probes détectaient effectivement les anciens arrondis ; leurs échecs sont conservés avec tous les essais retirés dans [opti.md](../opti.md). Cette validation finie ne prouve pas tous les prompts/contextes ni la qualité générale du checkpoint.

La première qualification historique [baseline](measurements/flash-next-ssd-impl/baseline-quality.json) avait échoué à1%, jusqu’à2,6487%RMSE, puis employé une borne5%. Elle est **remplacée par la qualification d’inférence corrigée**, sans effacer les chiffres ni leur protocole initial. Les premières mesures de débit précèdent cette correction et ne mesurent pas le code corrigé.

## MTP natif intégré

La tête `mtp.*` et son patch HyperConnections proviennent du même checkpoint. Elle utilise une pré-norme cachée globale, une projection distincte de chacun des quatre flux et une seule couche QSA sans PLE. Les512experts de cette couche restent résidents (0,595Gio de payload d'experts). La vérification cible calcule deux lignes dans une cohorte SSD protégée, conserve le premier état après rejet et les deux après acceptation. Le cache historique de la tête est prolongé par les seuls calculs K/V/index nécessaires.

Profondeur1, mode greedy et pénalité1 ; les autres réglages passent en décodage ordinaire. Le Desktop utilise le dossier du modèle comme tête intégrée, garde MTP OFF pour un nouveau modèle et sauvegarde le choix ON/OFF par modèle et moteur. Aucun téléchargement de tête supplémentaire ni tuner de profondeur n'est nécessaire pour Flash Next.

Dix positions tenues à part concordent en octets avec la vraie classe MTP de MLX-VLM : fusion, six composants, couche, mixer,248320logits par position, argmax, K/V et tail. Le cache-only est identique au cache du passage complet natif. Les blocs cible T2/commit1/2 reproduisent les logits, tous les états GDN/QSA/PLE/HC et le token suivant du T1. Les sessions budgets1/2/3/17 couvrent acceptation et rejet avec mêmes tokens et états cible/tête. Trente paires SSD contre l'oracle indépendant couvrent caches10/20/143, froid/chaud, overlap ON/OFF, doublons, lecture corrompue et reprise. Les16positions du tronc corrigé restent exactes après intégration. [Preuves numériques et échecs conservés](measurements/flash-next-ssd-impl/mtp-impl/).

Le bridge release passe16générations A/B/B/A aux budgets1/2/3/17, mêmes hashes/texte/historique/counts. Il refuse les cinq requêtes invalides, reprend exactement après annulation au début du prefill et au premier delta, et conserve le fallback avec pénalité1,1. Le bridge Qwen3.6 existant passe aussi12générations, les erreurs et annulations. [Flash](measurements/flash-next-ssd-impl/mtp-impl/bridge-flash-01.jsonl), [Qwen3.6](measurements/flash-next-ssd-impl/mtp-impl/bridge-qwen-legacy-01.jsonl). Ces temps courts sont fonctionnels, pas une comparaison de débit.

## Vérification source et limites

La PR intègre la base 1.4.3 `b4dfd3915e262b2d33581102ae3320042acee5c9` ; source publiée `2162c5f64a16558e1443758b4dbf22ba09566a78`. Les évolutions du cache Metal, du bridge, de la mémoire, de DFlash et du Desktop sont conservées. La préparation native de la tête intégrée est raccordée à `set_mtp` ; les contrôles Desktop couvrent ON/OFF sauvegardé, configuration et éjection immédiate avant préparation. Une perte d'activation sauvegardée détectée après composition est corrigée, avec son échec initial conservé.

Cette composition passe 102 tests Rust `mlx,chat` (74 bibliothèque, 12 main, 16 contrats ; 92 ignorés par cette suite), 431 tests Python (4 sauts), le Desktop complet puis le typecheck et hardening Swift6 stricts. Les 15 tests physiques affectés sont exécutés séparément et passent : 16 positions du tronc, 10 de la tête, états et commits T2, sessions, cohortes SSD et slots. Les bridges release Flash/Qwen3.6 passent aussi configuration ON/OFF, générations ABBA, erreurs, annulations et reprise. Les six harnais Kani conservent les mêmes domaines et résultats ci-dessous ; ASan/UBSan du cache C++ passe. CBMC refuse toujours le vrai contexte libc++, et CrossHair n'analyse pas ce checker I/O. [Manifestes, commandes, diagnostics initiaux et limites de l'intégration](measurements/flash-next-ssd-impl/integration-143/quality.json). Ces vérifications finies ne constituent pas une preuve de l'inférence complète.

Kani 0.68 : contrat d'offsets u64, 12 obligations et deux couvertures satisfaites ; hash n-grammes u64, boucle au plus trois éléments, 87 obligations (une inatteignable) et deux couvertures satisfaites. Le harnais de réservation sur deux routes symboliques et dix slots a été tenté sur la fonction réelle ; après correction d'une ICE liée aux erreurs anyhow, l'exploration a été interrompue après environ cinq minutes. **Cache non prouvé** ; 2 000 cohortes déterministes couvrent vieillissement, évictions, doublons, erreurs et invalidation après lecture échouée.

Kani du calcul de partitions réel : domaines u32 complets, unwind26,37obligations réussies et3couvertures satisfaites ; vérification bornée CPU, sans preuve des réductions flottantes Metal. [Preuve](measurements/flash-next-ssd-impl/divergence/kani-partition.log).

CBMC a été tenté sur la FFI C++ réelle préprocessée : parsing libc++/MLX incompatible. CrossHair a exploré le parseur de tokens pendant huit secondes : deux propriétés non confirmées. Aucun de ces résultats ne prouve la concurrence GPU, MLX, Metal ou l'inférence complète. Le contexte QSA au-delà de 2 048 tokens n'est pas encore contrôlé physiquement ; le parcours multimodal n'est pas implémenté dans ce port texte.

Pour MTP : six harnais Kani réussis,837obligations/21couvertures,3obligations inatteignables dans les contrats d'acceptation existants. Intervalle tensoriel u64 complet et commit i32 complet sans hypothèses ; partition u32 complète/unwind26 ; préfixe greedy au plus7propositions/unwind9 ; profondeur1/unwind4 et profondeurs0..3/unwind6 avec budgets symboliques. Ce sont des vérifications bornées de helpers CPU de production. La suite `mlx,chat` passe43tests bibliothèque+7main+16contrats,65ignorés ; les14contrôles physiques ciblés de cette étape sont sélectionnés séparément. Python68pass/4skips ; Desktop E2E et sources app complètes Swift6/warnings-as-errors réussis. Le build strict du package entier échoue sur les dépréciations CoreText préexistantes de SwiftMath ; le vendor n'est pas modifié. CBMC/Swift/Metal/concurrence et cache symbolique restent non prouvés. [Commandes, bornes, sources et limites](measurements/flash-next-ssd-impl/mtp-impl/quality.json).

CI du commit MTP d506cc7 inspectée : [Desktop/Python/E2E réussis](https://github.com/0xZKnw/mlxl3/actions/runs/38166474586) ; [Rust Linux/macOS et Kani ciblé réussis, Kani complet annulé](https://github.com/0xZKnw/mlxl3/actions/runs/38166474556). Le harnais de cache ajouté avec le streaming Flash dépasse30minutes d’exploration dans la suite complète ; il reste non prouvé. Les logs des jobs sont conservés, sans déclarer cette CI intégralement verte.

Pour la correction :11contrôles GPU ciblés,39tests CPU+15contrats et39tests Python réussis,3tests CPU ignorés ; fmt/Ruff et Clippy strict avec/sans MLX, build release `mlx,chat` réussis. Le bridge release vérifie16tokens par prompt MAIN/HOLD avec warmup+1répétition et hash/texte stables ; la CLI génère4tokens. Le rebuild final conserve exactement l’empreinte du binaire testé. Ces timings courts sont des contrôles fonctionnels, sans gain revendiqué. CBMC reste incompatible avec libc++ et CrossHair ne confirme pas les deux contrats du parseur en8s. CI précédente894c796 : Desktop/Rust Linux/macOS réussis, Kani annulé ; statut du nouveau SHA à inspecter après push.

Qualification du premier commit, conservée : au worktree isolé, format/Ruff, Clippy strict et compilation release `mlx,chat` réussis ; 38 tests CPU et 15 contrats passent, trois tests sont ignorés ; 20 tests Python passent. Le mutant de l’adresse du slot est détecté puis restauré. Cinq contrôles GPU passent. Le bridge du release isolé reproduit les 16 IDs de la baseline ([log](measurements/flash-next-ssd-impl/pr-baseline-bridge.jsonl)). Ces contrôles sont consignés dans le répertoire des mesures. Les GPU sont sélectionnés explicitement et exécutés seuls. La CI hébergée couvre les contrats CPU et le script de référence ; elle n'a ni le modèle ni un GPU Apple physique. Son statut doit être vérifié au SHA de la PR.

## Sources et intégration

Sushi `beamivalice/sushi` à `eaaca12c5508e6a24a01671384a9a41089370058` a été inspecté : kernels HC/QSA/EXL3 et format n-grammes. Son format de banques compilées diffère du checkpoint brut ; aucun corps de kernel Sushi n'est copié et sa sélection tolérante d'experts n'est pas utilisée. Le moteur réutilise les kernels EXL3 de MLXL3/Pony et les idées de caches directs du projet SSD local. [Recherche et antécédents](recherche-flash-next-ssd-205-2026-10-11.md), [attributions](../THIRD_PARTY_NOTICES.md).

Les modifications locales préexistantes restent dans le checkout principal, préservé. La PR est construite dans un worktree séparé ; elle intègre maintenant la base 1.4.3 et ses fonctionnalités publiées. Son diff contre cette base n'ajoute aucune modification propre à `mtp.rs` ou `qwen35.rs`.

## Utilisation locale

```sh
MLXL3_MLX_ROOT=/chemin/vers/site-packages/mlx cargo build --locked --release --features mlx,chat
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 65c895314393431c09050b2e04e250836b3a6eb4 --local-dir models/flash-next-205
target/release/mlxl3-rs run models/flash-next-205 --prompt "Bonjour" --max-tokens 64
```

Pour MTP, activer **Génération → MTP** dans le Desktop construit avec ce moteur, ou utiliser le bridge JSON-lines :

```sh
cat <<'JSON' | target/release/mlxl3-rs bridge models/flash-next-205 --context-length 4096
{"type":"generate","request_id":"demo","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":64,"temperature":0,"top_k":1,"repetition_penalty":1,"mtp":true,"mtp_depth":1,"mtp_head_path":"models/flash-next-205","reuse_prompt_cache":false}
{"type":"shutdown"}
JSON
```

La fixture indépendante peut être recréée avec `scripts/check-flash-reference.py` dans un environnement contenant MLX-VLM 0.7.3 et le checkout PonyExl3 épinglé ; `--mtp-only` prend une fixture du tronc et exporte la tête seule. Les fichiers binaires des oracles restent locaux ; le script, leurs empreintes, les mesures et les logs sont fournis dans la PR. La commande `run` garde le décodage ordinaire ; MTP utilise le bridge/Desktop.
