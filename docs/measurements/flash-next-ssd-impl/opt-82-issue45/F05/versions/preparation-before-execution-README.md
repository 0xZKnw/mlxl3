# 82F05 — softmax et top10 FP32 groupés, GEMV scalaires conservées

Préparé isolément depuis `9dc85ef167e97e82bdefb8a9035dc53a0bd07361` (80M publié), après lecture du journal82T, de81I et des anciens43/49/64/79. **Aucun fichier W/index/opti modifié, aucun build, test, proveur, modèle ou benchmark lancé.** Seul rustfmt a été exécuté sur les copies, exit0. Le patch est applicable sur une copie de sa base sans fuzz ; cela ne constitue pas une compilation ou une parité.

## Changement de production

Le flag strict `MLXL3_SSD_BATCH_ROUTES=0|1`, absent=0, est lu et stocké par `SsdMoe::load`. Il n’est pas relu par token. La méthode `route` scalaire reste inchangée. Pour FP32/B2..8, le nouveau `route_rows` calcule les mêmes `gate_logits` sur chaque vue mono-ligne, concatène les logits FP32, puis appelle une seule fois `softmax_precise` et le top10 multi-lignes existant avec normalisation. FP16 et B1 gardent la construction historique par appels de `route` suivis des concaténations d’IDs/poids.

Les **deux sites** sont reliés au helper : la planification de fenêtre unique et le chemin standard/cohortes. Aucun déplacement de `prepare_resident_hot`, de `selected.to_u32`, de `last`, de fence, de réservation ou de protection de slots ; aucune prélecture, banque, politique U, lecture SSD ou réduction experte n’est modifiée. Le Hot est toujours lancé avant le readback des routes. Aucun shader, FFI, LFM2, draft ou tête MTP n’est ajouté/modifié.

Ce n’est pas l’essai43A rejeté : il ne partage pas de conversion de poids, ne matérialise pas de nouvelles gates FP32 et ne remplace aucune GEMV par une GEMM. Il regroupe uniquement les réductions sur les logits déjà calculés. La MLX embarquée pourrait néanmoins choisir une autre réduction de softmax selon le nombre de lignes : la parité exacte reste à établir physiquement. Le top10 par ligne et ses règles de classement restent ceux du kernel existant.

Les compteurs `route_gate_calls` et `route_batch_calls` sont exclusivement `cfg(test)` ; ils vérifient respectivement les GEMV scalaires et la branche réelle. Le chemin CLI release ne les contient pas. Le helper CPU d’éligibilité est appelé par la production et ne qualifie que FP32/[B,2560], 2≤B≤8.

## Artefacts

- `source.patch` : `ssd_moe.rs`, le petit contrat CPU dans `flash_support.rs` et le nouveau fichier de tests ignorés physiques.
- `ci.patch` : ajoute le harnais CPU réel à la liste Kani PR/push existante. Le test CPU est automatiquement inclus dans la suite native.
- `combined.patch` : ces deux patches réunis ; appliquer celui-ci **ou** les deux séparément, sans doublon.
- `base/` et `candidate/` : copies complètes, ne pas les copier sur W. Appliquer les hunks après décision du parent sur82F01, en conservant les autres travaux locaux.
- `manifest.json` : SHA de chaque original/résultat et de tous les patches ; `commands.json` : commandes futures, noms exacts, marqueurs et environnement.
- `mutations/*.patch` : trois mutants préparés seulement, aucun appliqué au candidat ou à W.

## Oracles préparés, non exécutés

1. **1004 cas de gates réelles, 404 branches rapides.** Les48 matrices du checkpoint et deux activations indépendantes capturées par couche ; largeurs1/2/3/4/8 partout,5/6/7 surlayer0 ; F16/F32 et entrées contiguës/transposées après cast. Le chemin resident_gate est aussi exercé surlayer0, avec fallbackF16. Les logits sont comparés aux appels mono-lignes de `Projection::forward`, les probabilités au softmax mono-ligne, les IDs et poids à l’ancien `route`. Formes, dtype, cardinalités, finitude et octets exacts sont obligatoires. Un tri CPU des512 probabilités contrôle indépendamment les dix IDs et leurs égalités. Les compteurs exigent exactement B appels de gate et une seule branche rapide quand elle est admissible ; invalides/rangs/dtypes ne doivent ni calculer la gate ni muter le cache.
2. **28 cas de réductions seules.** Zéros signés, égalités, fortes amplitudes, petites marges, −Inf, +Inf et NaN avec payload, en B2/B3/B4/B8. Ancien softmax/top10 ligne par ligne contre nouvelles opérations groupées, tous les bits des probabilités/IDs/poids comparés, y compris les résultats historiquement non finis. Ce composant teste les opérations de réduction ; le câblage `route_rows` est contrôlé séparément dans les gates réelles. Il n’autorise pas des sorties de génération non finies.
3. **24 cas MoE complets.** Vraies entrées et sorties indépendantes de la couche0, B1/B2/B3/B8, capacités20/53 et quatre slots pending, cohortes legacy/U, cinq experts Hot réellement remplis. Sorties byte exactes, branche groupée non vide, compteurs de soumission et claims pending conservés jusqu’au drain/retrait explicite. La chronologie Hot avant readback est contrôlée sur les **quatre fenêtres U débordant la capacité nominale disponible**, via le compteur existant, sans nouveau readback. Les vingt autres cas ne disposent pas de cette assertion de chronologie et le marqueur le précise.
4. **Loader strict réel, environnement fixé avant Popen.** Sélectionner le test dans neuf processus séparés : absence,0,1 puis six littéraux ASCII invalides. Les trois cas valides contrôlent le champ stocké et deux appels réels de `route_rows` (branche, deux GEMV, IDs/poids exacts de l’oracle scalaire). Les invalides doivent être refusés au chargement. Le test ne modifie et ne restaure **aucune** variable du processus : `set_var/remove_var` après initialisation MLX serait unsafe, même avec un seul thread de tests. La propriété « chargé une seule fois » repose sur le champ et la revue du câblage, sans prétendre à une mutation dynamique sûre. La variante Unix non-Unicode nécessite un lancement avec environnement bytes ; elle reste explicitement non préparée dans cette matrice JSON ASCII.
5. **Oracle U historique obligatoire sous flag1** : `ssd_moe::hybrid_tests::unique_windows_preserve_outputs_pending_slots_and_greedy_fallback`. Il ajoute les cas de fenêtre8 réunifiée ou6+2/4+4, future contribution Hot conservée après éviction, experts résidents non sélectionnés, positions pending et échec de lecture puis récupération. Il ne faut pas déduire ces propriétés du seul composant24.

Les tests3/4 chargent au plus un SsdMoe à la fois ; aucune fixture ne conserve48 banques d’experts. Le composant1004 parcourt les couches séquentiellement. Le cache d’experts du moteur complet reste3Gio. Les statistiques `read_bytes` restent logiques, pas des octets physiques NVMe.

Après ces composants, réutiliser36préfixes/captures/logits/états DFlash byte exacts, la session MTP/budgets1/2/3/17 et le bridge43 réel (annulation/reprise/erreurs) sous flag1. Pour le bridge adaptatif, enlever WIDTH du JSON **et de l’environnement hérité**. Aucun débit avant ces validations.

## Mutations et preuve

- `admit-B1.patch` : admet B1 dans le vrai contrat. Le test CPU et le harnais Kani doivent fournir un échec, puis être relancés après restauration.
- `drop-normalization.patch` : conserve les IDs mais supprime la renormalisation du nouveau top10. Le composant réel doit détecter les poids erronés.
- `replace-scalar-GEMVs.patch` : remplace les B appels par une gate multi-lignes. Le compte réel des appels doit échouer même si certaines valeurs numériques coïncident.

Le harnais `flash_route_batch_preserves_scalar_fallback_domain` appelle le **helper de production** : quatre dimensions i32 symboliques, précision booléenne symbolique, rang0..4, unwind5, cinq couvertures (B2/B8, B1 refusé, FP16 refusé, rang4 refusé). Sa portée est seulement l’admissibilité des formes dans ce domaine ; il ne prouve ni GEMV, softmax, top10, Metal, durée de vie des slots ou concurrence.

Un second harnais préparé appelle le véritable test MLX de gates : `actual_batch_routes_preserve_real_gate_logits_and_weights`. Sa tentative doit rester explicitement non vérifiée si la compilation MLX/Kani rencontre intrinsics243, un harnais non sélectionné, un timeout ou une autre incompatibilité. Aucun stub, réécriture CPU du calcul ou réduction de domaine après échec n’est prévu. Aucune tentative Kani n’a été exécutée pendant la préparation, puisque82F01 GPU est réservé au parent.

## Campagne future du parent

Construire/formatter/Clippy CPU+MLX, tester le contrat CPU et les suites pertinentes, tenter Kani et vérifier les mutations sous surveillance bornée, puis arrêter/join tous les compilateurs/proveurs. Geler sources/binaires/options/plan. Exécuter ensuite seuls les tests physiques et le bridge, avec les gardes45/49 et statistiques processuelles77.

Si la qualité passe : même CLI pour flag0/1, DFlash fixe B2 WIDTH1/ADAPTIVE0, MAIN/HOLD, cache3Gio53slots/alloc64Mio, Hot/Wide1/UNIQUE1/MULTI_TOKEN1, prélecture/LES/shared/stable0,128warm+128mesure/127decode. Ordre H-C puis C-H selon le signal, OFF/MTP1 frais avant promotion. Conserver toutes les paires et dérives, golden60 et comptes/modes/acceptations/largeurs, mémoire et alimentation. Ne pas cumuler ce résultat avec F01 ou F06 : les mesurer isolément avant une combinaison explicitement enregistrée.

Gain, temps GPU exclusif, coût de synchronisation et pic mémoire **non mesurés**. Le batch ajoute une petite concaténation de logits et peut annuler les dispatchs économisés. Les caps compression77B historiques restent des échecs lorsqu’ils sont dépassés ; ni une empreinte stable ni ledger0 ne prouvent l’absence d’écritures système de swap. Les15tok/s restent un objectif, pas un résultat de ce patch préparé.

Révision préparatoire après revue croisée : le test loader exige maintenant six routages après changement d’environnement ; le marqueur MoE annonce quatre contrôles Hot/U débordants, sans généraliser cette chronologie aux24 cas. Ces nouveaux contrôles ne sont pas encore exécutés.

Révision25octobre : la proposition précédente de six mutations d’environnement après load est retirée avant toute exécution, car MLX peut avoir créé des workers. Les patches/manifeste de cette préparation non exécutée restent dans `superseded-v2-post-load-env`. Nouvelle matrice : neuf processus à environnement fixé, aucun appel Rust de mutation d’environnement.
