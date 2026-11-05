## Objectif

Accélérer le vrai parcours **MLXL3 sur M5 / 24 Go**, particulièrement **Qwen3.8 Flash Next EXL3 2.05 bpw avec experts streamés depuis le SSD**, en reprenant les mécanismes utiles de [TurboEXL3](https://github.com/davidtai/turboexl3).

Cette issue compare les sources et propose des expériences. **Aucun portage, nouveau benchmark ou gain MLXL3 n’est validé par cette analyse.** Elle complète #36, #44, #45, #46, #52, #54 et #58 avec les apports précis de TurboEXL3, sans remettre à zéro les essais négatifs.

Sources figées pour rendre la comparaison vérifiable :

- **MLXL3 `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`**, branche publiée `main`, 27 octobre 2026. Les changements locaux ultérieurs ne sont pas visibles ici : vérifier leur diff avant un futur essai.
- **TurboEXL3 `345bee9e35920ee77292055d873fca891146529c`**, 29 octobre 2026. Ses tableaux mesurent des versions antérieures explicitement identifiées dans `RESULTS.md` ; la tête documentaire ne doit pas être présentée comme le binaire chronométré.
- `AGENTS.md`, `opti.md` et les rapports pertinents de MLXL3 ont été consultés. Aucun essai matériel n’a été lancé.

## Ce que les résultats amont montrent réellement

Sur **M5 Max 128 Go**, TurboEXL3 annonce **140 tok/s en K3 à 1K** et **138,2 tok/s en K5 à 4K**, avec MTP. Ce ne sont pas des objectifs directement transposables à notre M5 et à son streaming SSD. Le plus petit repack complet annoncé pèse **63,1 Go**, table n-gram incluse. [T-R]

Les comparaisons les plus utiles pour choisir nos travaux :

| Comparaison amont | Résultat publié | Portée |
| --- | ---: | --- |
| MoE K2, 2 lignes, routage capturé : forme 9 / Sushi | **1,054×** | Microbenchmark pertinent pour notre quantification et les petites fenêtres |
| MoE K2, 4 lignes, routage capturé | **1,159×** | Même réserve ; routé seul |
| MoE K4, 4 lignes, routage capturé | **1,493×** | Le « 1,49× » ne concerne pas notre K2 |
| K5, 4K, MTP désactivé : Turbo / Sushi | **55,1 / 56,2 tok/s** | La fusion ne garantit pas un gain en decode singleton |
| iQ-MLX à 1K, MTP profondeur 3 / défaut 6 | **125,9 / 97,1 tok/s** | La profondeur MTP explique une partie des écarts |

Les microbenchmarks excluent le shared expert et le routeur fusionnés, ainsi que nos I/O SSD. Ils utilisent les mêmes entrées dans un binaire commun, avec tours alternés ; les tableaux serveur ont **une exécution par cellule**, parfois signalée comme bruitée, et des réponses différentes selon le bras. Ne pas additionner ces gains. [T-R]

**Point numérique décisif :** « byte-identical » chez TurboEXL3 signifie identique à **sa chaîne de référence**, pas automatiquement à MLXL3. Ses produits EXL3, sa carte MUL1 différée et ses réductions peuvent différer de Sushi ; le contrôle publié des timings tolère jusqu’à **2,7 % du RMS** de la sortie Sushi. Les formes 5–9 préservent les bits de sa forme 3. Notre portage exact doit conserver **nos** casts, FMA, Hadamard, sigmoid et arbres de réduction. [T-K] [T-R]

## Ce qui existe déjà chez nous

À conserver comme baseline, pas à présenter comme manquant :

- **Décompression EXL3 directement dans les QMV/QMM**, sans matrice FP16 complète systématique.
- **Gate/up en un passage de projection**, routage GPU, banques de slots et recouvrement Hot / lectures Cold.
- **TensorOps M5 dans les QMM**, dont la voie experte segmentée.
- Vérification multi-token exacte, partage EXL3 CPU connu **expérimental**, et cache des fabriques de kernels.
- Sélection/tuning MTP et DFlash2 déjà présents.

Les écarts se situent dans les **frontières entre ces opérations**, leur intégration au parcours SSD et le travail restant à chaque lancement. [M-S] [M-L] [M-QMM] [M-EQMM] [M-FFI]

## Priorité 0 — filtrer les mécanismes les plus isolables

### T01 — Préparer et réutiliser le lancement natif, au-delà du cache de fabrique

**Apport Turbo.** `plugin/sdk.zig` garde les kernels et configurations par géométrie ; son hôte `HostGpu.launch` utilise la pipeline MLX construite sans régénérer et hacher le source à chaque lancement. [T-SDK]

**Écart observé.** Notre cache C++ existe déjà. Après un hit, `mlxl3_metal_kernel` reconstruit encore les descriptions de sorties et appelle `CustomKernelFunction` ; le Rust construit aussi des métadonnées de dispatch. Le coût exact restant à l’intérieur de MLX est à tracer, pas à déduire du seul code. [M-FFI] [M-L]

**Premier essai.** Mesurer séparément construction Rust, FFI, encodage MLX et complétion. Tester un handle natif stable avec configuration réutilisable, borné par géométrie/dtype/source/version. L’API de lancement de mlx-serve n’est pas une API publique directement disponible dans MLXL3 : vérifier la faisabilité avec notre MLX 0.32.2.

**Validation.** Mêmes sorties, clés complètes, invalidation lors d’un changement de source/configuration, erreurs et durée de vie des handles. Comparaison all-hit puis Hot/Cold. Antécédents : cache-factory19 rejeté, métadonnées80M, formes stables67. La nouveauté est le coût du **lancement après hit**, pas un deuxième cache de compilation.

### T02 — Lire les huit codewords d’une lane depuis une fenêtre commune

**Apport Turbo.** `exl3_tile_load` / `exl3_tile_cw` utilisent une fenêtre de 64 bits pour les taux admis jusqu’à `n=108`, puis 96 bits jusqu’à `n=128` ; les décalages des huit extractions sont constants. [T-LANE]

**Écart observé.** Notre QMV mapped possède déjà une spécialisation de fenêtre **K3**, mais son chemin général traite deux groupes de quatre poids ; la voie dense utilise aussi ces groupes. Étudier les taux réellement chauds, d’abord K2 des experts et K4 des projections. [M-MAPPED] [M-TILE]

**Premier essai.** Remplacer seulement l’extraction du bitstream, en conservant valeurs FP16 décodées, ordre des FMA et réduction de référence. Contrôler les accès cycliques/SWAP16 et l’alignement ; comparer les extractions à notre oracle de format sur toutes les lanes et frontières.

**Décision.** Retenir seulement si instructions/lectures ou temps complet diminuent sur nos formes. Les anciens funnels32 et la LUT MUL1 exploratoire ne prouvent rien pour cette fenêtre commune.

### T03 — Supprimer les intermédiaires de scales et d’activation à la préparation

**Apport Turbo.** Ses kernels `prep2` / `gu8` adressent directement les scales de l’expert, et la préparation produit le layout consommé par le GEMV. `gu8` absorbe aussi la préparation au singleton lorsqu’elle tient dans la géométrie admise. [T-DEC]

**Écart observé.** `SsdMoe::calculate_slots` exprime encore expansion gate/up et gathers de `suh/svh` ; `expert_mapped_layout` enchaîne multiplication, Hadamard et QMV. La matérialisation effective de chaque intermédiaire doit être vérifiée. [M-S] [M-L]

**Premier essai.** Commencer par charger les scales au slot dans leur consommateur et éviter l’expansion des entrées quand l’index du token suffit. Garder le Hadamard natif et ses arrondis. Ensuite seulement comparer un layout de sortie de préparation directement adapté aux lanes, sans `take` supplémentaire.

**Validation.** Slots absents masqués **avant** toute lecture, Hot/Cold, strides, plusieurs projections aux `suh` distincts. Flash désactive actuellement le layout lanes qualifié sur d’autres modèles : ce n’est pas une option à activer aveuglément. Ne pas rejouer le remplacement H128 de K1, rejeté ; continuation ciblée de #58/07 et /11. [M-LANE]

## Priorité 1 — réduire le coût des couches du vérificateur

### T04 — Fusionner les frontières gate/up → SwiGLU → préparation down

**Apport Turbo.** La forme servie 9 fusionne finishes gate/up, activation et préparation down (`fin7`), puis finish down et réduction (`dfin7`). Sa chaîne passe de **11 lancements / 8 niveaux** à **4 / 4 au singleton**, **5 / 5 de 2 à 16 lignes**. Ce décompte décrit sa chaîne non fusionnée ; **ce n’est pas un comptage mesuré de MLXL3**. [T-K] [T-DEC]

**Écart observé.** Nous fusionnons déjà la projection gate/up, mais les deux fins, `swiglu` et la préparation suivante restent exprimées séparément. [M-S] [M-L]

**Premier essai.** Une frontière à la fois, en reproduisant exactement les casts et l’activation MLX, avec dump des intermédiaires et contrôle de la mémoire de threadgroup/registres. Les H128 en registres de Turbo ne sont pas présumés interchangeables avec les nôtres.

**Validation.** Entrées non identiques, T1/T2/T3/T4/T8 et sorties par ligne invariantes au regroupement. Distinguer nombre de dispatchs, dépendances et fin réelle du token.

### T05 — Écrire les contributions à leur destination finale et fusionner la réduction routée

**Apport Turbo.** `dfin7` finit les experts et produit directement la somme pondérée du token. Le prefill utilise une table inverse plutôt qu’une restauration générale des résultats. [T-DFIN] [T-PF]

**Écart observé.** Le SSD concatène les sorties Hot/Cold, restaure leur ordre avec `take`, puis caste, multiplie et somme les dix contributions. [M-REDUCE]

**Premier essai.** Conserver un emplacement canonique `(token, rang top10)` pour chaque contribution, quelle que soit sa date de complétion ; fusionner ensuite le consommateur. Garder **notre arbre de somme et les arrondis avant pondération** : la boucle de FMA ordonnée de Turbo ne le remplace pas automatiquement.

**Validation.** Toutes les partitions Hot/Cold, doublons, experts absents, erreurs/annulation, disponibilité des buffers et absence d’atomiques flottantes par ordre d’arrivée. Continuation #58/28 et /42 ; les gains des anciens copieurs ne sont pas présumés.

### T06 — Intégrer le shared expert aux lancements du routé

**Apport Turbo.** `moeFusedShared` ajoute les travaux du shared dans les grilles gate/up/down et leurs finishes, **sans lancements propres aux trois projections du shared**. Sa largeur intermédiaire doit correspondre à celle du routé ; il sert 1–4 lignes. [T-SHARED]

**Écart observé.** Notre `shared_projection` puis `shared` gardent une chaîne séparée et, par défaut, une boucle par ligne. Le regroupement `SHARED_BATCH` existe, mais l’essai68 n’a pas qualifié de gain de génération. [M-SHARED] [H68]

**Premier essai.** Fusionner les travaux réellement compatibles en gardant les propres poids/scales du shared, son multiplicateur scalaire et l’ordre de la somme finale. En Hot/Cold, calculer le shared **une seule fois par token**, avec une dépendance explicite à la contribution qui le combine.

**Validation.** Comparer à notre chemin séquentiel, y compris sigmoid et casts. La table sigmoid/SwiGLU utilisée par Turbo reproduit son hôte ; elle doit être vérifiée contre le nôtre. Un nouvel essai est justifié par la **fusion de lancements**, distincte du batching68 et du simple déplacement de branche73.

### T07 — Partager le décodage des experts répétés avec un plan GPU peu coûteux

**Apport Turbo.** À partir de deux lignes, deux occurrences du même expert peuvent partager lecture et décodage des poids dans le même threadgroup. Dans son routage capturé, 40 occurrences contiennent en moyenne **28,2 experts distincts** ; le bénéfice dépend donc fortement des routes. [T-DEC] [T-R]

**Écart observé.** `ExpertSlots::grouped` partage déjà les poids par groupes réels de 1–4, mais seulement pour les slots CPU connus, avec plusieurs dispatchs/concat/restauration ; il refuse la voie Hot masquée. Les essais65/66 n’ont pas validé un boost global à activer par défaut. [M-GROUP] [H65] [H66]

**Premier essai.** Grouper les occurrences Hot directement sur GPU, ou regrouper les tailles dans un lancement unique avec destinations finales. Mesurer plan + masques + restauration éventuelle, sur **fenêtres naturelles B2/B4/B8** ; les répétitions artificielles sont un contrôle séparé.

**Validation.** Aucun readback supplémentaire, même snapshot de résidence et protection des slots en vol, aucun regroupement entre experts distincts. Garder un repli singleton lorsque le coût du plan dépasse les décodages économisés. Continuation71P, avec nouvelle intégration ; pas relance inchangée de65.

### T08 — Adapter la géométrie des lancements sans changer la partition numérique

**Apport Turbo.** La forme9 distingue singleton et multi-lignes ; tiles par threadgroup et parts de K distribuées aux groupes changent l’occupation tout en gardant la partition/somme de la forme3. `gu8` s’arrête au singleton car la préparation est répétée par tuile de sortie. [T-DEC]

**Écart observé.** Le SSD impose notamment `NT=1` et quatre SIMD groups pour sa référence. Les splits sont déjà indépendants du nombre de routes : **cette invariance existe et doit rester**. [M-L]

**Premier essai.** Explorer seulement regroupement de tiles/SIMD et répartition physique des parts existantes, sur GU 2560→640 et down640→2560, avec compteurs de registres, spills et mémoire threadgroup. Séparer T1, B2 et B4.

**Validation.** Mêmes produits et même arbre par sortie, sorties identiques seules et en cohorte. Changer les vrais splits ou l’ordre d’addition relève d’une autre recette numérique. Les anciennes géométries54/67 restent documentées.

### T09 — Une voie dense du type DEC-ONE

**Apport Turbo.** Pour 1–4 lignes, DEC-ONE rassemble GEMV et finish ; la préparation est parfois intégrée. Le dernier threadgroup qui publie ses partiels effectue le finish, avec compteurs entiers réinitialisés ; il n’attend pas activement les autres groupes. [T-DENSE]

**Écart observé.** Les projections denses Flash gardent préparation, partiels, réduction, H128 et scales en plusieurs opérations ; le batching exact63 ne réalise pas cette fusion complète. [M-DENSE]

**Premier essai.** Une projection dense fréquente et petite, puis les formes réelles du vérificateur. Comparer la variante avec préparation séparée à celle intégrée : cette dernière peut répéter trop de travail.

**Validation.** Publication mémoire correcte, un seul écrivain final, compteurs propres à chaque appel, appels répétés, erreurs, et nos réductions exactes. Aucun spin inter-threadgroups ni addition flottante non déterministe. Inclure le coût du lancement de remise à zéro.

### T10 — Diminuer les registres vivants des GEMV multi-lignes

**Apport Turbo.** `dgemv_lean` parcourt les codewords par élément d’entrée plutôt qu’en gardant quatre valeurs par ligne vivantes ; il conserve les FMA et permet des chunks de 6–8 lignes, puis deux chunks pour 9–16. [T-DENSE]

**Écart observé.** Notre kernel multi-row garde `x_values[MB][4]` avec les accumulateurs ; des voies exactes et politiques de batch existent déjà. [M-TILE] [M-MAPPED]

**Premier essai.** Réordonner uniquement les chargements/la durée de vie des temporaires, sans changer la séquence de FMA de chaque accumulateur. Prioriser les projections réellement exécutées à 6–8 lignes, puis vérifier qu’un changement ne pénalise pas B2.

**Validation.** Spills mesurés, temps du chunk complet, queues et comparaisons à la ligne isolée. Aucun bénéfice supposé simplement parce qu’un batch plus large devient possible.

## Priorité 2 — intégration plus large et temps du premier token

### T11 — Fusion du routeur et de la préparation : respecter la coupure SSD

**Apport Turbo.** `moeFusedSharedRouted` absorbe top-k, softmax/renormalisation et éventuellement logit du shared dans la préparation ; la voie servie concerne **2–4 lignes**, pas le singleton. [T-SHARED]

**Écart observé.** Notre routeur est déjà GPU ; le CPU doit toutefois connaître les misses pour déclencher les lectures. Supprimer ce rendez-vous en copiant la fusion résidente casserait le streaming. [M-ROUTE] [M-S]

**Premier essai.** D’abord un parcours entièrement résident ou le draft éligible. Pour le SSD, produire tôt IDs/scores/plan Hot, préserver le départ des lectures et fusionner seulement les travaux qui ne retardent pas ces sorties.

**Validation.** Top10 exact, ex aequo, dtype/softmax identiques, délai entre routes disponibles et départ I/O. Ne pas supprimer ni retarder la soumission Hot utile. Ce n’est pas une suppression gratuite du readback.

### T12 — Fusionner gate d’attention, préparation de projection et sorties de l’indexeur

**Apport Turbo.** Les merges decode `decodeMergeGated`, `decodeMergeQmergeGated`, `decodeMergeQPlanes` et `decodeMergeIndexer` fusionnent les voisins de la projection : gate→préparation, sorties query/gate contiguës, norm/RoPE et clé poolée. L’indexeur annonce sept lancements retirés dans **sa** chaîne. [T-DECB]

**Écart observé.** `Qsa::forward` garde q_proj, vues, q_norm, RoPE, gate et out séparés ; son indexeur comporte encore des opérations distinctes. [M-QSA]

**Premier essai.** Gate→préparation out, puis écriture directe des plans query/gate. Étudier les sorties de l’indexeur séparément. Notre cœur utilise SDPA : le merge split-K QSA de mlx-serve n’est pas un élément manquant à ajouter sans correspondant local.

**Validation.** Positions/strides, arrondis et états exacts ; contexte avant/après **2052**, anneau de pooling et rollback de chaque préfixe. Différent du simple batching q/out B1, retiré après confirmation non concluante.

### T13 — Préparation Q/K/V/indexeur commune et épilogues attention pour le prefill

**Apport Turbo.** `attnIn` / `attnPrep4` lisent une fois l’entrée pour plusieurs préparations ; `attn_finish` réunit fins des projections, QK-norm et RoPE ; `linearGated` absorbe le produit gated dans la préparation de out. La voie est proposée à **17 lignes et plus**. [T-ATTN]

**Écart observé.** Ces opérations restent séparées dans notre parcours Flash inspecté ; le prefill Flash doit d’abord exposer de vrais chunks. Les formes et kernels MLX locaux ne sont pas identiques à ceux de l’hôte Turbo. [M-QSA] [M-PROMPT]

**Premier essai.** Un chunk de projections indépendantes, avec chaque `suh` propre, puis un épilogue reproduisant les opérations actuelles. Partager la lecture de x ne signifie pas partager un Hadamard entre scales différentes.

**Validation.** Petites queues, normes et RoPE exacts, causalité QSA et états au découpage des chunks. Les économies de lancements amont doivent être recomptées sur notre graphe.

### T14 — Rendre les kernels de prefill atteignables sur Flash SSD512

**Apport Turbo.** Au-delà de16 lignes, ses experts passent à une chaîne GEMM groupée avec décompression dans le loader, mid et finish fusionnés. [T-PF]

**Écart observé.** `FlashNext::forward_tokens` boucle encore sur les tokens. Le `route_plan` générique est limité à **256 experts**, contre **512 pour Flash Next** ; les QMM existants ne constituent donc pas un prefill Flash SSD groupé livré. [M-PROMPT] [M-MOE]

**Premier essai.** Exposer des microchunks natifs, grouper seulement les projections/FFN indépendants et ordonnancer les experts selon la capacité réelle des slots. Préserver les étapes récurrentes/attention causales et leur état.

**Validation.** Logits finaux, captures, reprise/annulation et états identiques au séquentiel ; taille du chunk bornée par cache + scratch + activations. Cible **TTFT et conversations longues**. Dépendances connues #54/P01 et #58/23–29 ; Turbo apporte les kernels candidats, pas le streaming SSD complet.

### T15 — Routes count/scan/scatter et tuiles mixtes adaptées aux comptes par expert

**Apport Turbo.** `pf_t128_route` construit ordre et tables en un lancement. Les groupes ≤32 lignes utilisent une voie registre ; les autres des tiles64/128, avec loader EXL3 double-buffered, layout de fragments NAX et plusieurs blocs de sortie down par threadgroup. [T-PF]

**Écart observé.** Notre voie segmentée utilise argsort, deux recherches binaires par expert et construction de table dans `tid0`. Le QMM expert emploie BM32 avec sous-blocs : l’amélioration recherchée porte sur plan, partage d’une tuile décodée et queues, **pas sur l’ajout de TensorOps déjà présents**. [M-MOE] [M-EQMM]

**Premier essai.** Séparer (a) plan count/scan/scatter512, (b) loader/layout, (c) choix petit/gros groupe. Évaluer sur distributions réelles de routes et en comptant padding, table, chargement de poids et pic mémoire.

**Validation.** Même correspondance token/rang et même ordre des produits par sortie ; si le changement de forme MMA change les bits, le sortir du portage exact. Les seuils17/32/64/128 de Turbo ne sont pas des seuils à importer tels quels. Ne pas refaire BM64 ou les seuils précédemment négatifs sans ce nouveau mécanisme identifié.

### T16 — Exposer préparation / produit / finish pour recouvrir les branches GDN

**Apport Turbo.** `linearPrep`, `linearInnerSplit`, `linearFinish` et les `linearPieces*` permettent à l’hôte d’insérer norme/gate entre projections et de recouvrir des morceaux sur deux streams. Ils gardent les mêmes kernels lorsque les formes restent admissibles. [T-API] [T-DENSE]

**Écart observé.** `Gdn::forward_rows` construit encore certaines étapes par ligne puis concatène avant la projection out ; notre API de projection expose surtout le résultat entier. [M-GDN]

**Premier essai.** Tracer les vrais temps exposés, puis rendre une frontière de projection réutilisable. Comparer le même graphe sur un stream à une variante où le produit dense indépendant recouvre le norm/gate.

**Validation.** États et historiques de chaque préfixe, identité du kernel MLX sélectionné à chaque taille, dépendances/buffers, contention mémoire. Deux streams ne garantissent pas une exécution utile en parallèle. Ne pas rejouer la fusion préparation+récurrence GDN rejetée.

## Pistes conditionnelles — autre budget ou autre contrat numérique

### E01 — Carte affine MUL1 différée

Turbo accumule sur la somme d’octets `t` puis applique `A × acc + C × Σx` une fois par sortie, au lieu de reconstruire le poids FP16 à chaque produit. C’est un mécanisme central de ses GEMV/GEMM. [T-LANE] [T-K]

**Cela change potentiellement les résultats :** supprimer l’arrondi FP16 de chaque poids et réassocier la somme n’est pas bit à bit équivalent à notre chaîne. Notre calcul MUL1 par paires d’octets existe déjà ; ne pas le confondre avec cette carte différée. [M-CB]

À étudier seulement comme backend numérique distinct, désactivé par défaut : mesures de logits/états, KL/PPL et tâches représentatives **par rapport au même checkpoint** avant vitesse. La proximité des scores de code amont ne valide pas la fidélité de notre moteur. Aucun changement silencieux du mode exact.

### E02 — W_hat dense résidente, avec budget strict

Turbo conserve optionnellement la matrice déquantifiée du **tronc dense**, évitant de redéquantifier aux prefills suivants ; les bits restent ceux de sa voie de déquantification par appel. [T-DENSE]

Notre QMM décode déjà en calcul : il n’existe pas partout un coût de déquantification complète à supprimer. Comparer seulement des projections pour lesquelles GEMM dense + reconstruction amortie peut battre le kernel direct. Compter la hausse mémoire, l’éviction et la pression sur le cache d’experts. **Aucun cache FP16 de tous les experts sur 24 Go.** Priorité basse sur la machine cible.

### E03 — Tronc déroté W_std et matmul unique

`trunkUnrot` incorpore H128/scales dans les poids au chargement, puis utilise une matmul ; les projections d’une même entrée peuvent être jointes. [T-DENSE]

Turbo le décrit explicitement comme **opt-in avec ses propres arrondis**. Cela peut déplacer beaucoup de travail vers le chargement, mais augmente les poids résidents et modifie le calcul. À réserver à une étude distincte, avec budget et qualité mesurés ; pas une optimisation exacte prête à activer sur notre parcours SSD.

## Intégration à prévoir

- **Adapter les kernels Metal ciblés à Rust/C++/MLX**, ou prouver l’intérêt d’une couche Zig séparée. Le plugin fourni vise le SDK/ABI de mlx-serve ; il ne se charge pas directement dans MLXL3.
- Préserver notre layout **slot-major**, les dtypes FP16/FP32 utilisés par Flash, le packing, les fenêtres de codewords, les codebooks et les limites par modèle. Turbo publie aussi des taux fractionnaires / fenêtres8–16bits et des refus explicites : ce sont des exigences de compatibilité à examiner, pas une accélération gratuite de checkpoints existants. [T-SPEC]
- Garder l’admission matérielle : NAX = **accélérateurs matriciels du GPU M5**, pas l’ANE/CoreML. Les kernels Turbo exigent leur gate M5/macOS26.2+ ; préserver les fallbacks des autres Macs et épingler les headers MLX0.32.2. [T-NAX]
- Reprendre les vérifications par étage, l’invariance au nombre de lignes et les refus **avant lancement**. Les diagnostics `slot_stats` / captures de routes ajoutent des synchronisations : les exclure des timings normaux. [T-K] [T-API]
- Si du code est repris, conserver **LICENSE + NOTICE**, les attributions David Tai et les portions MLX/Apple, ExLlamaV3, Sushi et mlx-serve ; compléter nos notices. [T-LIC] [T-NOTICE]

## Ordre de travail et critères de réussite

1. **T01/T02/T03** : coûts natifs et extractions/layout, un candidat isolé à la fois.
2. **T04/T05/T06** : fusions autour des kernels existants, avec leur ordre numérique.
3. **T07–T10** : partage des poids et voies multi-lignes uniquement si le profil réel les justifie.
4. **T11/T12**, puis **T13–T16** : intégration routeur/attention et véritable prefill streamé.
5. **E01–E03** : études séparées ; aucune activation implicite dans le backend exact.

Avant chaque essai, suivre `AGENTS.md` : entrée prospective dans `opti.md`, antécédent et raison nouvelle, sources/binaires/modèle/options figés. La présente issue ne prétend pas préenregistrer des mesures déjà lancées.

- [ ] Profiler la version réellement utilisée via **bridge/CLI → vérificateur → SsdMoe → FFI → Metal**, en séparant construction hôte, GPU et attentes SSD. Ne pas additionner les spans inclusifs.
- [ ] Vérifier la correction **avant** vitesse : oracles indépendants, sorties non vides/finies, logits/états, chaque préfixe accepté et rollback, slots absents/en vol, erreurs et annulation. Tester les dtypes et formes de production ; les tables amont et deux textes identiques ne suffisent pas.
- [ ] Pour le code futur, effectuer les contrôles et tentatives de preuve exigés par le dépôt ; distinguer contrats CPU et validation Metal réelle.
- [ ] Mesurer sur notre M5/24Go, même checkpoint EXL3 2.05 bpw et même draft, mémoire/cache/lecteurs fixés, un modèle GPU à la fois. OFF, MTP et DFlash2 **séparément**, profondeurs/largeurs et acceptations explicites.
- [ ] Utiliser routes capturées naturelles, all-hit puis Hot/Cold réaliste ; warmup, ordres alternés, plusieurs répétitions et prompts distincts. Rapporter decode, TTFT/prefill, temps de réponse, octets SSD logiques, répétitions d’experts, pic/compresseur/swap sans confondre logique et trafic NVMe.
- [ ] Promouvoir uniquement un gain de bout en bout reproductible à qualité et budget mémoire conservés. Archiver aussi échec/régression/non-concluant ; retirer les prototypes non retenus et fermer leur entrée `opti.md`.

**Antécédents à conserver :** MTP15/15B, fusions H128/K1, géométries54, grouped65/66, stable67, shared68, prélectures72/73 et batching QSA B1 ont des résultats négatifs ou non concluants. Une référence Turbo ne les annule pas. Un nouveau kernel, une fusion ou une intégration mesurable doit expliquer précisément pourquoi l’essai diffère.

### Références

Documentation Turbo : [architecture des kernels][T-K], [résultats et méthodologie][T-R], [API et intégration][T-API]. Comparaison locale : [kernels experts][M-L], [parcours SSD][M-S], [bridge natif][M-FFI]. Historique : [groupement65][H65], [largeurs66][H66], [shared68][H68]. Tous les liens de code fixent les commits comparés.

[T-K]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/KERNELS.md
[T-R]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/RESULTS.md
[T-DEC]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/decode.zig#L526
[T-SHARED]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/KERNELS.md
[T-DFIN]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/dfin7.metal
[T-DENSE]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/dense.zig
[T-PF]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/prefill.zig
[T-ATTN]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/attn_io.zig
[T-DECB]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/decb.zig
[T-LANE]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/lane/helpers.metal
[T-SDK]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/plugin/sdk.zig
[T-API]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/README.md
[T-SPEC]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/spec.zig
[T-NAX]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/plugin/nax.zig
[T-LIC]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/LICENSE
[T-NOTICE]: https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/NOTICE

[M-S]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/ssd_moe.rs#L339
[M-ROUTE]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/ssd_moe.rs#L399
[M-REDUCE]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/ssd_moe.rs#L985
[M-SHARED]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/ssd_moe.rs#L1054
[M-L]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/linear.rs#L1359
[M-LANE]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/linear.rs#L279
[M-DENSE]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/linear.rs#L383
[M-GROUP]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/linear.rs#L1222
[M-CB]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/linear.rs#L1646
[M-FFI]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/ffi/mlx_bridge.cpp#L646
[M-MAPPED]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/shaders/_qmv_mapped_tile_kernel.metal
[M-TILE]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/shaders/_qmv_tile_kernel.metal
[M-QMM]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/shaders/_qmm_tensor_kernel.metal
[M-EQMM]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/shaders/_expert_qmm_tensor_kernel.metal
[M-MOE]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/moe.rs#L74
[M-QSA]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/qwen4.rs#L577
[M-GDN]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/qwen4.rs#L304
[M-PROMPT]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/qwen4.rs#L1620
[H65]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/docs/flash-next-ssd-grouped-exl3-65-2026-10-21.md
[H66]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/docs/flash-next-ssd-fixed-width-66-2026-10-21.md
[H68]: https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/docs/flash-next-ssd-shared-expert-68-2026-10-22.md

Journal : [opti.md](https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/opti.md) · Consignes : [AGENTS.md](https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/AGENTS.md)


## Compléments de lecture TurboEXL3 — 29 octobre 2026

Deux mécanismes précisent T06/T15 et une réserve complète T07. Ils ne constituent pas trois optimisations indépendantes supplémentaires. Aucun nouvel essai matériel ni gain MLXL3 validé. Sources inchangées : TurboEXL3 `345bee9e35920ee77292055d873fca891146529c`, MLXL3 `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`. Vérifier le diff local avant toute expérience et conserver les essais rejetés.

### Complément T06 — Retirer un niveau dépendant avec le finish au dernier groupe

**Mécanisme.** Avec `fused_fin_fold`, à partir de deux lignes, Turbo intègre `fin7s` dans `gu3s` sous la forme `gu3sf`. Les groupes publient leurs partiels ; le dernier arrivé pour un bloc effectue le finish, SwiGLU et la préparation down. Un lancement indépendant initialise les compteurs. Le total peut rester à cinq lancements, **avec un niveau dépendant de moins**. Cette variante précise les fusions T04/T06 et se distingue du DEC-ONE dense T09. [Implémentation](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/decode.zig#L1562-L1603).

**Écart et portée.** Notre `calculate_slots` exprime gate/up, SwiGLU puis down ; le shared conserve sa chaîne séparée. Cible : vérification B2/B4, **après qualification des fusions simples**, sans gain présumé au singleton. [Routé](https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/ssd_moe.rs#L339-L389), [shared](https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/src/ssd_moe.rs#L1054-L1099).

**Essai.** Comparer la future chaîne fusionnée sans puis avec ce finish au dernier arrivant, à partition numérique identique. Mesurer profondeur dépendante, fin GPU et durée complète du vérificateur, initialisation des compteurs comprise. All-hit d’abord, puis Hot/Cold, sans retarder routes ni lectures.

**Garde-fous.** Préserver nos arrondis. Le protocole de publication amont doit être validé pour notre backend : compteurs/scratch privés par appel, visibilité des partiels, écrivain final unique, aucun spin ni finish sur partiels absents. Couvrir appels en vol, erreurs, annulation et calcul du shared une seule fois malgré Hot/Cold. Compter les copies de sommes/partiels et la mémoire supplémentaire. Retenir seulement si le chemin critique complet raccourcit sans perte d’overlap ni pression mémoire nuisible.

### Complément T15 — Omettre réellement les fragments des queues courtes

**Mécanisme.** L’option `half_rows` amont omet les chargements A et MMA des lignes 16–31 quand le groupe contient au plus seize lignes, en gardant l’ordre MMA des lignes utiles. **Option désactivée par défaut et non activée par `pfOptions` : aucun gain servi démontré.** [Option](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/prefill.zig#L387-L390), [kernel](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/prefill/gemm_small_4.metal), [fragments MMA](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/prefill/small_helpers.metal), [configuration servie](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/root.zig#L793-L801).

**Écart et dépendance.** Notre QMM expert exécute `operation.run` sur BM, puis masque les écritures dépassant `segment_rows` : masquer le store ne supprime pas le calcul. Cible : queues du **futur prefill Flash groupé T14/T15**, pas le parcours SSD token-par-token actuel ni un gain direct MTP/DFlash. [Kernel local](https://github.com/0xZKnw/mlxl3/blob/f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6/native/shaders/_expert_qmm_tensor_kernel.metal#L99-L114).

**Essai et décision.** Après accès au vrai prefill groupé, relever les comptes par expert puis omettre seulement les fragments morts. Tester 1/15/16/17/31/32 lignes et les queues après tuiles pleines. Garder produits, casts et ordre MMA de chaque sortie utile ; remplacer simplement BM32 par BM16 ne prouve pas l’équivalence. Compter préparation, padding et sélection de branche. Protéger les lectures de padding et vérifier que les fragments omis ne sont jamais consommés. Priorité basse tant que fréquence des queues et gain TTFT/temps de couche complet ne sont pas établis à budget identique.

### Complément T07 — Provenance des routes et contrôle négatif

Le « routage capturé » amont vient du **prefill du pack K5/5.05 sur un prompt de code**. Les microbenchmarks decode reprennent les routes de tokens consécutifs d’un chunk de 8192 lignes. Ce ne sont pas des fenêtres de candidats MTP/DFlash2 capturées pendant une vérification K2. Les poids du microbenchmark sont des codes trellis aléatoires et les activations BF16 : cela ne valide pas notre checkpoint et ses dtypes de production. [Méthode](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/RESULTS.md#L210-L229).

En K2/B2, rapport médian publié **1,054× avec ces routes capturées contre 0,993× sur routes synthétiques** ; à B4, **1,159× contre 1,020×**. Capturer nos vraies fenêtres de vérification, multiplicité et résidence comprises, avant de prioriser T07. Aucun rapport amont ne prédit le gain MLXL3. [Table capturée](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/RESULTS.md#L231-L241), [table synthétique](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/RESULTS.md#L267-L274).

