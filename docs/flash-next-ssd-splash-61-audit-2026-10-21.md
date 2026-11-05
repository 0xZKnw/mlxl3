# Flash Next SSD : écarts Splash, Strata et campagne60 — revue61R

21 octobre 2026. **Recherche statique terminée ; aucune nouvelle inférence, compilation ou mesure GPU.** Le bilan60 a été poussé d'abord dans la [PR34](https://github.com/0xZKnw/mlxl3/pull/34), commit `95d08366e3ac1085b80a78ad2bbc3bd16251d6bb`. Le code moteur reste celui de `dc33e3e4`, identitév7. Aucune application installée remplacée.

**La mission fournie a raison de prioriser GPU Hot B3–B8 et le vérificateur cible. Le port60 reprend Q4, le ring et le budget probabiliste, mais pas toute l'organisation de calcul de Splash.** En particulier, la cible reste largement séquentielle entre positions et les kernels Q4 spécialisés ne sont pas ceux exécutés par Flash.

La revue ajoute des constats plus précis : deuxième borne à20 routes dans le helper GPU Hot, fractionnement avant déduplication réelle des experts, expert partagé préparé après les lectures, label supplémentaire inexploité par l'adaptatif, et prélecture absente du vérificateur DFlash malgré une configuration globale qui peut désactiver son hybride. Ce sont des constats de code ; **leur effet en tok/s est non mesuré**.

## Révisions et périmètre

| Source | Révision effectivement examinée | Portée |
| --- | --- | --- |
| MLXL3 | `95d08366e3ac1085b80a78ad2bbc3bd16251d6bb` | Code runtime identique à `dc33e3e4`, bilan60 terminé ; prototypes57/58 absents de ce moteur. |
| Splash HEAD GitHub | `1482f6bbc2abf12375332725302019594150b6cb` | Inchangé depuis59 ;19 fichiers sources locaux contrôlés contre les blobs du tree Git épinglé, lecture des fonctions et appelants concernés. |
| Strata HEAD GitHub | `011c9e600a93bbc9fd451c563501b5831c32bbca` |238 commits depuis `9de04cd4` ;15 fichiers Metal modifiés, aucun supprimé.13 sources collectées/empreintes vérifiées, cinq fichiers relus aux régions utiles et par diff. Pas un audit intégral des238 commits. |
| Mission utilisateur | SHA256 `4e6b7d5624dc908144572618c2d7d3d76a2f34dc4f2fc8048768dfd8bcabf6e6` |9581 octets, lue après le push ; ses instructions sont traitées comme propositions à évaluer. |

Le compare REST Strata renvoie300 fichiers, donc il n'est pas supposé complet. L'inventaire Metal est établi par comparaison de deux arbres Git complets, dont le nouveau compte5200 entrées et `truncated:false`. Les fichiers collectés et ceux réellement relus sont distingués dans [sources](measurements/flash-next-ssd-impl/opt-61-splash-audit/sources.json), [lectures](measurements/flash-next-ssd-impl/opt-61-splash-audit/read-set.json) et [fichiers Strata](measurements/flash-next-ssd-impl/opt-61-splash-audit/strata-source-files.json). Aucun code amont exécuté.

Cible inchangée : Qwen3.8 Flash Next EXL3 2,05bpw `65c89531`,48 couches dont36 GDN et12 QSA ; draft de 5 couches `a22a1744`, M5/24Gio. Baseline60 : secteur/apps ouvertes, cache experts3Gio/53slots par couche, quatre lecteurs, allocateur512Mio, overlap/GPU Hot1, lookahead0, contexte4096, greedy128sorties. OFF frais10,46–10,48tok/s descriptif ; MTP1 9,59–10,55 ; DFlash2 A8,99–10,01 selon prompt/bras. Zéro nouveau Swapout dans60, garde et30s postflight inclus. [Rapport et limites60](flash-next-ssd-dflash-splash-60-2026-10-21.md).

## Matrice fonctionnelle

Les liens locaux ci-dessous fixent la révision publiée95d08366. « Partiel » décrit l'implémentation, pas un gain déjà mesuré. Chaque ligne correspond aux sources et fonctions indiquées ; aucune couverture exhaustive de tous les kernels des deux dépôts n'est revendiquée.

| Mécanisme | Splash / Strata | MLXL3 exécuté | État et implication |
| --- | --- | --- | --- |
| Quantification47projections | Splash `DraftCheckpoint::Builder::projection`, AF4G64. | [Draft::load_options/linear](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L342), conversion une fois au chargement et QMM MLX. | **Présent mathématiquement, différent en exécution.** Gain mémoire fixe qualifié, pas de boost global qualifié. |
| Layout et kernels Q4 | Splash `Linear::decodePlan/addGguf`, choix registre/staging/splits selon forme et nombre de cœurs. | [FFI affine4](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/ffi/mlx_bridge.cpp#L578) appelle `mx::quantized_matmul`. Le kernel packed [Q4Projection](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/dflash.rs#L1106) legacy n'est pas appelé par Flash. | **Absent du chemin Flash.** Adapter les layouts, pas réinterpréter les octets. |
| QKV du draft | Splash `DFlashDraft::addDecode`, projection QKV groupée puis préparation commune. | [Draft::attention](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L600) : trois projections, normes/RoPE/transposes distincts. | **Partiel.** Poids déjà Q4, groupement/préparation absents. |
| Gate/up et convolution | Splash épilogue GateUp et `DraftAttention::addConvolution`, entrées préparées. | [forward/convolution](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L564) : gate/up séparés, convolution par primitives ; résidu+norme déjà fusionnés. | **Partiel.** Le wrapper convolution legacy refuse H2560 et attend8lignes ; ce n'est pas un branchement direct. |
| KV circulaire2048 | Splash `DraftAttention::addDecode` split/reduce et commit des seules lignes retenues. | [append_ring/ring_attention](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L856), MPP porté, offsets runtime, ownership conservé. | **Présent, architecture différente.** Parité qualifiée ; Q/R débit non concluant, fallback concat déjà disponible. |
| Contexte draft | Splash `addContextCommit`, K/V groupés, entrée normalisée préparée commune, commit GPU. | [Draft::append](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L504), FC commun puis dix K/V séparés et deux eval par couche. | **Partiel.** Optimiser le pipeline de commit, pas seulement l'attention ring. |
| Sélecteur Markov | Splash `DraftSelector::add`, trois étapes GPU, tokens/confidences en buffers. | [select_logits/top16](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L731), top16 GPU et transitions MLX, puis deux readbacks. | **Présent pour le calcul, partiel pour la résidence GPU.** Une confidence vide n'est plus un problème du code60. |
| Choix des lignes | Splash `draft_verify_rows` sur GPU, scratch de forme fixe et MoE des seules lignes actives. | [Session::advance](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft_runtime.rs#L80) : choix CPU puis graphes de la largeur retenue. | **Implémenté différemment.** On supprime effectivement des lignes cible, mais on paie sélection/readback et construction variable. |
| Modèle adaptatif | Splash `VerifyBudget`, mêmes prior/EWMA/MAP/exploration. | [VerifyBudget](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft_budget.rs#L85), mono-séquence FP64 CPU, reset à chaque requête. | **Présent avec différences.** Labels en bord de préfixe et durée d'apprentissage détaillés plus bas. |
| Préparation cible | Splash `Qwen4ExpTarget::addVerify`, HC/projections/QSA sur les lignes du bloc. | [verify_draft](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_draft.rs#L1124) appelle `Layer::prepare` par ligne. | **Absent pour le vrai groupement cible.** MoE groupé ne signifie pas bloc cible entièrement groupé. |
| Projections EXL3 cible/tête | Splash plan commun multiligne. | [Exl3Linear::forward](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/linear.rs#L300) resérialise les lignes Flash de référence, sauf `mtp.fc_hidden`. | **Différent volontairement pour l'arrondi.** Concaténer les entrées seul ne résout pas ce coût ; MB2 global15/48 non qualifié. |
| GDN et rollback | Splash tape des entrées/k/v/gates, replay du préfixe retenu au bloc suivant. | [Gdn::forward/Layer::snapshot](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/qwen4.rs#L180), états complets par position ; commit restaure le bon préfixe. | **Correction présente, tape compacte absente.** Potentiel temps/mémoire à mesurer, dépendances causales à conserver. |
| GPU Hot large | Strata organise la fenêtre et sa frontière hôte ; MLXL3 Hot56 déjà qualifié MTP1. | [forward_rows](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/ssd_moe.rs#L452) limite Hot à2lignes ; [residents](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/linear.rs#L1095) limite à20 routes. | **Absent B3–B5, seulement certaines queues B6/B7.** Le recouvrement hot/cold après readback existe encore sur les autres cohortes. |
| Experts répétés | Strata grouped GEMM ; Splash `MoE::addRouting/addExperts`, descripteurs/dispatchs des tuiles vivantes. | `calculate_slots` EXL3 NT1 par occurrence ; déduplication des lectures SSD présente. | **Absent du moteur publié.**58 compiled/CPU seulement ;57 compact seul négatif. |
| Prélecture urgente | Strata `where_` couvre prêt/en vol, `promote_slot` ; Splash conserve les morceaux lus et complète le reste. | [ReaderPool::note_late](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/ssd_prefetch.rs#L181) compte, ne fusionne ni ne promeut ; `fill_many` séparé. | **Absent.** Ne pas additionner LES et Hot : garde explicite d'exclusion. |
| Ordonnance SSD | Splash index partagé, demandes prioritaires, lectures par morceaux/staging. | [read_disk_records](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/ffi/disk_reader.h#L113), quatre workers par indices espacés et attente du batch. | **Implémenté différemment.** Déséquilibre critique à prouver avant une file dynamique ; pas de raison de multiplier les threads. |
| Cache experts global | Splash slots par layout, ARC/historique, admission résistante aux scans ; Strata arènes par classe. | [ExpertCache](https://github.com/0xZKnw/mlxl3/blob/95d08366/native/src/flash_support.rs#L484), LFU/LRU par couche. | **Absent comme cache global.**55 statique rejeté ne teste pas cette architecture. |
| Graphes/buffers/réveil GPU | Strata fenêtre préconstruite ; Splash arènes, commandes et événements hôte, chronos pickup/work/resume. | Graphes MLX construits à chaque bloc, arrays temporaires, barrières aux frontières nécessaires. | **Partiel.**57 n'est pas cette architecture ; reprise GPU exclusive non mesurée localement. |
| Desktop et préfill | Splash intégration famille/checkpoint/cache de préfixes et captures utiles. | Bridge Flash explicite testé ; [StudioModel](https://github.com/0xZKnw/mlxl3/blob/95d08366/apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift#L813) passe encore par l'inspection legacy. | **Partiel.** Interface d'activation Flash non aboutie, aucune validation app ; détail ci-dessous. |

Sources amont principales : [poids du draft](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/DraftCheckpoint.cpp#L28), [exécution draft](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/DFlashDraft.cpp#L246), [sélection](https://github.com/incoai/splash/blob/1482f6bb/runtime/ops/DraftSelector.cpp#L38), [vérificateur Flash](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/Qwen4ExpTarget.cpp#L327), [MoE](https://github.com/incoai/splash/blob/1482f6bb/runtime/ops/MoE.cpp#L185), [plans de projections](https://github.com/incoai/splash/blob/1482f6bb/runtime/ops/LinearGguf.cpp#L333).

## GPU Hot : les trois limites à traiter ensemble

La borne visible `rows<=2` n'est que la première. `ExpertSlots::residents` refuse un nombre de routes hors1..20. Enlever uniquement la garde de `forward_rows` ferait donc échouer B3 au helper, avant tout gain.

Deuxième conséquence : la cohorte est découpée **avant** le calcul des routes selon `rows*10`, et non selon le nombre d'experts uniques effectivement sélectionnés. Pour53slots, les formes sont :

| Lignes cible | Cohortes actuelles | GPU Hot possible avec lookahead0 et overlap1 |
| --- | --- | --- |
| B1/B2 |1 /2 | Oui |
| B3/B4/B5 |3 /4 /5 | Non |
| B6 |5+1 | Seulement la queue1 |
| B7 |5+2 | Seulement la queue2 |
| B8 |5+3 | Non |

Exemple déduit : huit positions routant les mêmes dix experts pourraient tenir dans dix slots, mais passent tout de même par5+3. Cela ne signifie pas que les routes réelles ont cette répétition ; leurs unions B3–B8 ne sont pas enregistrées dans60. Une partition tenant compte des unions doit préserver chaque occurrence et son poids dans la réduction, garder NT1 pour les experts uniques et protéger les banques jusqu'à leur dernier utilisateur.

Troisième limite : la prélecture garde de la place pour20 routes seulement (`claim_prefetch`, `flash_support.rs:593`). Avec53slots, quatre lectures encore en vol laissent49slots non pending ; une cohorte de cinq lignes avec50experts uniques peut alors être refusée par `reserve`. Le pool réel n'a que quatre workers : ne pas inventer33lectures simultanées pour ce scénario. C'est un **risque déduit des contrats CPU**, pas un échec reproduit sur le M5 dans61. L'extension large doit tenir compte des slots utilisables, pas seulement de la capacité nominale.

Enfin, `forward_draft_rows(x)` force `next=None, depth=0`. Les vérifications DFlash ne lancent donc pas les prédictions de la couche suivante. Si `MLXL3_SSD_LOOKAHEAD_DEPTH>0` a installé un pool au chargement, sa simple présence désactive aussi Hot pour B1/B2. Le préfill peut encore prédire, mais ce n'est pas un pipeline decode DFlash LES+Hot intégré.60 était explicitement àlookahead0 et ne valide pas cette composition.

## Autres manques au-delà des deux priorités de la mission

### Un label cible disponible n'alimente pas l'adaptatif

`Session::advance` calcule toutes les propositions, les tronque à `selected`, puis appelle `observe(&confidence[..selected], accepted, seconds)`. Si les `selected` propositions sont toutes acceptées et qu'une proposition suivante existe, le dernier logit cible permet déjà de la comparer. Ce label est perdu. Avec `selected=0`, le passage cible sur l'ancre produit aussi un premier label disponible, mais notre slice de confidence est vide.

Exemple : draft `[a,b,c,…]`, budget2propositions, logits cible donnant `[a,b,c]`. L'émission reste `[a,b,c]` selon le contrat actuel ; pour la calibration, a/b/c sont tous observables. MLXL3 n'enregistre que a/b. Si le troisième logit vautd, c peut fournir un label négatif, sans calcul cible supplémentaire.

Splash conserve les sept confidences/propositions ; son compte d'acceptation est distinct de la rétention et son budget étiquette jusqu'à `min(accepted, verifyRows-1,6)` inclus. [Acceptation GPU](https://github.com/incoai/splash/blob/1482f6bb/runtime/metal/kernels/decode/sampling.metal#L1656), [labels](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/VerifyBudget.cpp#L165), [appel de production](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/Runtime.cpp#L2198). Il faut garder séparés compteur d'émission/commit et observation supplémentaire ; **ne jamais committer une entrée cible qui n'a pas été vérifiée**. L'effet attendu est une calibration mieux informée, pas une accélération chiffrée établie.

### L'expert partagé attend encore les lectures SSD

Dans `SsdMoe::forward_rows`, gate/up/down et la porte de l'expert partagé sont construits aux lignes727–745, après `fill_many` ligne671. Ses poids ne sont pas dans les banques d'experts routés et ses entrées sont déjà disponibles. Strata lance gate/up et porte avant la frontière hôte qui place les experts en arène. [Ordonnancement Strata actuel](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/window.cpp#L882).

Cela justifie un essai distinct : soumettre les projections partagées assez tôt, conserver leurs sorties et l'addition finale exacte, puis observer le recouvrement. Candidat utile à OFF et MTP1 aussi. **Ce manque était déjà identifié dans09**, mais pas intégré ; ce n'est pas une découverte entièrement nouvelle ni une relance MB2 de15. Le profil53 historique ne mesure pas sa part exposée dans DFlash60.

### Le budget et son protocole ne jugent pas encore la même phase d'apprentissage

Les priors, décroissance0,97, cinq cycles de régularisation du coût, MAP16itérations,4096labels, refit64/exploration32 sont bien présents. Ce n'est pas une moitié d'équation oubliée. Le choix CPU en FP64, les tables mono-séquence et le reset par requête sont des adaptations explicites.

Splash construit son `VerifyBudget` dans le runtime ; MLXL3 le remet à zéro à chaque préfill (`flash_draft_runtime.rs:51,184`). Dans60 R/A, MAIN ne compte que50–51cycles : **aucun refit probabiliste avant la fin**, même après un warmup, car ce préfill remet l'apprentissage à zéro. HOLD64–65cycles atteint le premier refit en toute fin. Le modèle de coût apprend dès les premiers cycles ; ne pas dire que l'adaptatif entier est inactif.

Les sept propositions sont calculées en entier avant ce choix. Le budget peut sélectionner zéro, mais cela n'évite ni le draft ni sa sélection : ce n'est pas un retour au coût OFF. Un futur choix de mode sans draft doit gérer la continuité des captures/KV avant de le réactiver ; persister sans discernement les coûts d'un autre prompt n'est pas une correction gratuite.

Autre limite d'attribution : R/A change le draft de4lignes (fixe3propositions) à8lignes (maximum7) **et** son budget. Cela compare deux configurations utiles, sans isoler le seul coût de décision à forme de draft constante. Le témoin scientifique supplémentaire est fixe7 contre adaptatif7, mêmes Q4/ring/kernel, puis ringON/OFF à politique identique.512sorties doivent constituer une campagne séparée, pas être mélangées aux128historiques. Ces mesures ne sont pas lancées ici.

### Les kernels Q4 hérités ne s'appliquent pas directement aux47matrices

La géométrie du checkpoint donne10 matrices dynamiques640×2560,10 KV1024×2560,5 Q4096×2560,5 O2560×4096,10 gate/up7680×2560,5 down2560×7680,1 FC2560×12800 et1 selector256×2560. [Inventaire statique](measurements/flash-next-ssd-impl/opt-61-splash-audit/static-geometry.json).

Le kernel legacy `dflash_q4_mpp.h` attend8lignes et un stockage en tuilesN256. Son wrapper refuse une sortie640. Les poids QMM Flash sont en packing MLX par ligne ; leur ABI n'est pas le packingN256. Il faut une conversion bornée avec oracle des codes/échelles/biais, padding ou kernel gérant640, puis mesurer chaque forme. La convolution legacy ne prend queH2048/H5120 et8lignes, avec un ordre d'arrondi différent des primitives Flash ; l'ajout deH2560 exige ses propres cas numériques.

Il faut aussi regarder **le planificateur Splash actuel**, pas seulement notre ancien kernel copié. `LinearGguf.cpp:144,152` décide registre/staging et active certaines fusions/spreading à partir de16cœurs. Notre M5 en a10. Copier une configuration M5Pro/Max ou compter tous leurs effets GateUp comme applicables ici serait une autre erreur. Les fusions encore manquantes comprennent QKV, préparation norme/RoPE, convolutions, K/V du contexte et réemploi des entrées ; une fusion n'est retenue qu'après mesure.

### Tape GDN, graphes et mémoire

`snapshot` clone des handles MLX ; il ne fait pas directement un memcpy de chaque tenseur. Mais le vérificateur garde les versions d'états de chaque préfixe. Un état GDN `[1,48,128,128]` FP32 représente3145728 octets ; sept préfixes sur 36 couches représentent792723456 octets de payload de versions possibles, hors état final/historiques/QSA/allocateur. **Ce calcul n'est pas un pic physique mesuré ni la mémoire forcément récupérable.**

La tape Splash contient les entrées de convolution, k/v, decay/beta, puis incorpore seulement le préfixe retenu au bloc suivant. [GDN](https://github.com/incoai/splash/blob/1482f6bb/runtime/metal/kernels/decode/gdn.metal#L22). Le port doit préserver les arrondis de notre cible FP16/FP32 et les états observables après commit, annulation et reprise ; copier leur GDN BF16 ne garantit pas cette égalité.

Le ring est déjà2048, les offsets sont des valeurs d'array runtime, et les anciens buffers passent par `cache.take()` puis réassignation de `slice_update` avant eval. Je n'ai pas constaté de recompilation par offset ni prouvé une copie intégrale du ring à chaque append. L'API autorise un chemin de donation ; seule une mesure des allocations/copies permet de dire quand il est pris. Deux eval par couche font des points d'attente possibles, pas nécessairement dix attentes matériel indépendantes. Les supprimer sans organiser les dépendances pourrait masquer une course.

Splash gère de façon commune arènes, sélection GPU et commit ; Strata conserve une fenêtre de dispatchs et ses événements. Les remplacer par une collection de primitives indirectes MLX séparées reproduirait le risque57. Pour le retour d'I/O, Splash distingue pickup/work/resume et utilise un mécanisme awake avec repli ; une partie utilise des extensions Metal internes. Son commentaire de latence amont n'est pas une mesure du M5 local. [Frontière hôte](https://github.com/incoai/splash/blob/1482f6bb/runtime/metal/MetalBackend.mm#L1142).

### Préfill et intégration visibles

`prefill_dflash2` effectue cible/captures/tête/drain puis append draft pour chaque token. Splash organise des captures compactes du contexte utile et projette le contexte par blocs ; une restauration de préfixe n'a pas besoin de toutes les positions draft anciennes. [Plan de contexte](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/DraftContextPlan.cpp#L33), [projection contextuelle](https://github.com/incoai/splash/blob/1482f6bb/runtime/model/DFlashDraft.cpp#L154). C'est surtout une piste TTFT/contexte long ; elle n'explique pas à elle seule les tok/s decode annoncés.

Le bridge annonce `dflash_supported=true` pour Flash (`main.rs:2927`), mais le bouton Desktop et le choix d'un dossier local passent par `dflash-draft --target … --inspect …`. Cette commande appelle `dflash::target_family`, limité aux35B/27B packed, avant le traitement du dossier. Le dossier BF16 Flash accepté par le bridge est donc rejeté par ce parcours d'activation. **C'est un trou d'intégration concret**, déjà signalé comme non étendu dans60, et pas une validation GUI acquise par les43contrôles bridge. L'application installée n'a pas été modifiée ; cette revue ne tente pas de l'installer ni de la lancer.

## Ce qui a réellement changé chez Strata depuis59

Les diffs des chemins pertinents sont conservés avec les données61. Le changement Metal contient notamment le support Gyro/TQ2_T/TQK6/TQK7, ses rotations Hadamard et des tuiles spécifiques. Ces treillis ne sont **pas** nos blocsEXL3MUL1 : blocs128valeurs avec échelleFP16 et codebook de paires, contre nos tuiles16×16, suh/svh et convention de décodage EXL3. Leur support nouveau n'autorise pas de brancher ces kernels sur nos fichiers.

À reprendre : la préparation de l'entrée avant la frontière SSD, le choix des tuiles selon forme/matériel et le vrai partage d'une tuile de poids entre activations du même expert. À ne pas présenter comme nouveauté déjà accélératrice : les nouvelles instanciations du grouped MoE ajoutent des formats à son cheminpréfillv2 ; le code désactive explicitementv3 pour ces treillis. Cela ne fournit pas un kernelEXL3B8 validé.

Le diff `stream.cpp` ajoute surtout l'exclusion de la grande table ngram lors du chargement des poids denses ; promotion des lectures urgentes et cartes prêt/en vol existaient déjà dans la révision59. [Décodeur treillis actuel](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/kernels/quant.metal#L584), [réemploi des lectures](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/stream.cpp#L649). Le ngram Flash local lit déjà ses lignes séparément ; pas de gain équivalent inventé ici.

## Pistes déjà évaluées : décisions conservées

| Piste | Historique réel | Décision61 |
| --- | --- | --- |
| NT2/MB2 systématiques |15/48 retirés, résultats modèle insuffisants ou composants instables/négatifs.48 est strictement `inconclusive_drift`, pas la preuve que tout partage multiligne est mauvais. | Ne pas réactiver globalement. Les vrais doublons58 et une nouvelle formeB3–B8 restent des mécanismes distincts. |
| Layout/décodage54 | Oracles exacts, gains composants trop faibles/instables ; prototype retiré. | Pas de relance identique. |
| Allocation statique55 | Réduction de jobs surMAIN, régression surHOLD à budget égal. | Cache global ARC ou replay multitrace seulement comme hypothèses nouvelles ; pas de réemploi du profil gagnantMAIN. |
| Compact57 isolé | Composants stables mixtes1,78–4,33% plus lents ; pas de vitesse modèle. | Rejet conservé. Une fusion globale de graphe/partage de poids doit être distinguée avant un nouvel essai. |
| Partage EXL3 vrais doublons58 | Compilé, CPU/sanitizer build ; GPU non exécuté, Kani arrêté sous garde mémoire. | Prototype non qualifié, pas du code livré. Les oracles GPU viennent avant le débit. |
| Cache6Gio / F_NOCACHE | Pression/swap ou résultat négatif déjà consignés dans les essais SSD. | Pas de réactivation directe ; les486–490Mo Q4 fixes ne sont pas une marge garantie du modeA. |

[54/55](flash-next-ssd-roadmap-54-55-2026-10-20.md), [57](measurements/flash-next-ssd-impl/opt-57-compact-plan/micro-summary.json), [58](measurements/flash-next-ssd-impl/opt-58-shared-exl3/verification-status.json), [48](measurements/flash-next-ssd-impl/opt-48-paired-experts/closure.json). Les gains d'étages, machines et baselines différentes ne sont pas additionnés.

## Ordre concret pour la prochaine implémentation

1. **Profil DFlash réel puis GPU Hot B3–B8**, à cache53 constant : nombres de routes/uniques/cohortes, résidence au snapshot, attente du readback et fills, sorties/événements de durée de vie. Étendre ensemble helper20 routes, partition et protection ; ne pas supprimer une garde seule. InclureB1/B2 pour contrôler OFF/MTP1 et les régressions.
2. **Vérificateur cible par étapes** : d'abord préparer des projections indépendantes à plusieurs lignes avec accumulations exactes, incluant router et expert partagé ; garder GDN/QSA/PLE causaux. Grouper la tête avec son vrai contrat d'arrondi, pas seulement une concaténation qui resérialise. Puis tape GDN dans un essai séparé, avec instantané sémantique complet au commit.
3. **Correction de l'observation adaptative**, avec un test distinguant label supplémentaire et tokens committés ; comparaison à forme constante fixe7/adaptatif7 et ringON/OFF. La variante zéro doit être explicitement couverte. Campagne512 séparée pour le refit ; ne pas revendiquer un gain du modèle probabiliste à partir deMAIN128.
4. **Draft spécialisé par forme** : oracle de repackingQ4, noyaux adaptés aux10 sorties640, QKV/convolutions/contextKV, puis draft complet et acceptation. Conserver QMM et concat comme témoins tant que les gains ne sont pas qualifiés.
5. **SSD et MoE** : réemploi/promotions des demandes en vol + protection du snapshot, partage des vrais doublons58, puis replay ARC et petite hausse de cache à budget physique égal. Mesurer la latence exposée avant une file dynamique.

La journalisation des temps doit entourer les frontières déjà synchronisantes **sans ajouter d'eval au parcours de débit** : préparation/draft, sélection et ses readbacks, choix, préparation cible, route/readback, fill et lecteurs, MoE, tête, commit cible, append draft. Les profils hôte avec fences restent une campagne diagnostique distincte. Le harnais53 test-only donne une base, mais le bridge60 ne publie que le temps global des blocs : les millisecondes exclusives des phases60 sont **non mesurées**. Ses95 % historiques MTP ne doivent pas être attribués automatiquement à DFlash60.

Avant toute vitesse : formes/non-vacuité/finitude, logits/états et tokens contre oracle cible séquentiel, cohortes sans miss/tout miss/mélange/doublons, slots pending, défaillances I/O, budget et contexte, commit/rejet/EOS/annulation/reprise, véritable CLI/bridge. La couverture physique actuelle du ring dépasse déjà2048 au composant ; une génération complète512 et un contexte cible dépassant le seuil sparseQSA sont deux contrôles différents. Aucun n'est déclaré exécuté par61. Plafond12milliards d'octets et zéro nouveau Swapout jusqu'à30s après, warmup/HCCHMAIN-HOLD/identités exactes, conditions d'alimentation enregistrées.

Les options expérimentales Q4/ring/adaptatif sont111 par défaut quand DFlash est explicitement activé ; le défaut global demeureOFF. Le ring conserve un fallback0. Puisqu'aucun boost ring n'est qualifié, son état doit rester visible et aucune modification de défaut silencieuse n'est justifiée par cette revue.

## Vérification et limites de61

Lecture de la pièce jointe, traçage des appels de production,19 blobs Splash et13 blobs Strata vérifiés contre les arbres épinglés ; inventaire géométrique depuis les en-têtes/configs, aucune lecture des poids dans un modèleGPU. Diffs/identités/matrice conservés. Aucun changement exécutable : les suites et tentatives de preuve60 restent applicables à leur code, mais ne constituent pas de nouveaux tests des pistes61.60 a915 tests Python passés/4skips, RustCPU110/3ignorés, releaseMLX139/138ignorés,43 contrôles bridge et20 cas numériques ; GPU/concurrence restent non formellement prouvés. [Bilan détaillé60](measurements/flash-next-ssd-impl/opt-60-splash-dflash/validation-summary.json).

CI du SHA95d08366 suivie séparément jusqu'à sa fin : **régressions, Linux natif, macOS avec build/lint MLX et Kani Flash passent**. Les premiers relevés encore en cours sont conservés, puis `snapshot2` confirme leurs résultats. Kani global reçoit de nouveau un shutdown runner sur le harnais inchangé `cache_allocation_is_optimal_and_exact_on_two_layers_two_levels`, sans contre-exemple observé. [Workflow natif](https://github.com/0xZKnw/mlxl3/actions/runs/39589281004), [régressions](https://github.com/0xZKnw/mlxl3/actions/runs/39589281223), snapshots et log exact dans les données61. Aucune CI globalement verte annoncée.

**Décision :** retenir les priorités GPU Hot large/vérificateur et les écarts concrets décrits ; ne pas annoncer15tok/s ni un pourcentage de gain avant leurs propres essais. Revue61 close, pistes non implémentées/non mesurées. L'intégration Desktop Flash reste incomplète.
