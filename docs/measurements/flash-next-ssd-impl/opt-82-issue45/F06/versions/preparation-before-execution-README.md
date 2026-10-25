# 82F06 — arrêt du comptage des perdants top-10, prototype isolé

**État : préparé, non appliqué, non construit, non testé, non prouvé, non mesuré.** Base explicite9dc85ef167e97e82bdefb8a9035dc53a0bd07361, 80M publié. Ni80Q ni82F01 ne sont inclus. Le worktree actif, son index et son journal n’ont pas été modifiés. Seul rustfmt2024 a été exécuté sur les copies.

Le préenregistrement parent est82T dans W/opti.md. Ce dossier contient original/, changed/, source.patch, ci.patch séparé, deux mutations et commands.json. Les identités du futur binaire/source intégrée restent null jusqu’au gel réel. Les trois originaux existants ont été confrontés à git show9dc85ef, plutôt qu’à un worktree changeant pendant82F01. La première copie SSD avait capturé cinq hunksF01 ; ils ont été détectés et retirés uniquement dans ce dossier avant le gel.

## Plus petit changement de production

- `MLXL3_SSD_TOP10_EARLY_EXIT=0|1`, défaut0, parseur strict existant et champ `SsdMoe::top10_early_exit` initialisé une seule fois au load, avant allocation des banques. Non-Unicode propagé. Aucun accès à l’environnement par token.
- Les deux branches F16/F32 de `SsdMoe::route` transmettent ce champ à `router::flash_top10`. La brancheF16 garde sa normalisation MLX après sélection ; la brancheF32 garde la somme et division du kernel historique.
- Le helper CPU réellement utilisé `flash_top10_early_exit` limite l’émission du break à E512/K10 demandé. Les autres consommateurs de `router::topk`, le SIMD E256/K8, le top16 draft et les autres modèles gardent leur voie actuelle.
- Le shader flag1 ajoute seulement `if (above >= int(TOP_K)) break;` après l’incrément entier. Aucun retour anticipé, déplacement de barrière, clé64bits, tie par indice, score, ordre de réduction, grille ou scratch modifié. Son nom de factory porte un suffixe explicite.
- Flag0 : le fragment ajouté est vide et le nom de factory reste historique. La comparaison textuelle de la vraie chaîneMSL formatée, après substitution vide, est exacte avec9dc ; cette propriété statique ne remplace pas l’oracle physique.

Callers : Flash `Layer`/vérificateur63→`SsdMoe::forward_rows` ou cohortes uniques→route ligne par ligne→Flash top10→legacy. Le MTP Flash utilise aussi SsdMoe et doit être contrôlé. Les logits/GEMV/softmax, readbacks, GPU Hot, cache/leases, chargements SSD, commit et états ne changent pas.

## Pourquoi le contrat de rang peut être conservé

La clé reste un entier total unique (clé flottante haute, indice bas). Le compteur `above` ne fait qu’augmenter. Une fois10 clés strictement supérieures rencontrées, aucun suffixe ne peut rendre ce candidat gagnant ; il ne doit plus écrire. Les dix gagnants ont un rang strictement inférieur à10 et parcourent toujours tous les512 concurrents, donc gardent leur rang exact et leur emplacement. Toutes les lanes atteignent ensuite la même seconde barrière. Cet argument statique ne prouve ni le compilateur Metal ni la concurrence physique.

Le gain reste inconnu : une lane gagnante peut maintenir sa SIMD dans la boucle. Aucune économie de comparaisons source n’est présentée comme une économie de tempsGPU, de trafic physique ou de tok/s.

## Tests préparés

1. TestCPU sansGPU : `flash_support::flash_top10_early_exit_never_changes_other_router_domains`. Demande0 toujoursinactive ; frontières E511/512/513 etK9/10/11, domaines entiers extrêmes. Le garde de production est exercé.
2. Composant physique `router::flash_top10_tests::flash_top10_early_exit_matches_independent_ranks_and_legacy_scores` :256cas, B1/2/4/8 ×F16/F32 ×contigu/stridé ×8fixtures ×normalisé/non. Tri CPU indépendant par valeurs représentées et indice, NaN classé au-dessus, ±0 égaux comme le contrat historique. IDs indépendants, scores byteexacts contre le vrai ancien kernel, scores bruts finis et zéros signés comparés aux valeurs d’origine ; les non-finis intentionnels ne sont pas qualifiés de sorties finies. Fixtures uniformes, monotones inversées, égalités, non-finis, faibles marges, sous-normaux et extrêmes. Conversion avant transposition pour conserver de vraies vues stridées dans les deux dtypes.
3. Métadonnées et six domaines génériques inchangés : rangs/largeurs/types invalides, puis entrée valide après refus. La fonction de sélection générique demandée avec le flag vrai retombe réellement sur la source historique horsE512/K10.
4. `ssd_moe::top10_early_exit_tests::top10_early_exit_preserves_real_routes_and_normalization` :384cas sur les48gates réelles, deux entrées oracle07, F16/F32 et vues stridées, candidat0/1 comparés à l’ancien `router::topk` et à sa vraie normalisationF16. Chargement séquentiel d’une couche, pas de deuxième modèle cible complet.
5. Câblage du loader : processus séparés avec variable absente,0 puis1, fixée avant le démarrage. Le test lit le champ chargé et exerce une route, sans mutation de l’environnement. Le caractère loadonce est contrôlé par la revue du call graph : `load` seul lit la variable, `route` lit le booléen stocké. Aucune validation dynamique d’un changement post-load n’est revendiquée.
6. Refus strict via CLI enfants distincts : huit littéraux invalides et un environnement non-UnicodePOSIX sont fournis au lancement du processus ; défaut/0/1 sont couverts par les trois loaders natifs. Aucun `set_var/remove_var` après initialisation MLX : `--test-threads=1` ne suffit pas à rendre ces appels sûrs face aux workers du runtime. Ces contrôles externes sont préparés dans plans/quality/run.json.

Les sorties imprimées exactes et les noms complets figurent dans commands.json. Le parent doit vérifier `--list`, sélection exactement1test et résultat1passed/0failed avant de retenir un contrôle. Aucun nombre de tests passés n’est annoncé ici.

## Mutations indépendantes

- `mutation-gpu.patch` tourne l’ID écrit d’un cran dans le vrai shader. Le tri CPU indépendant doit échouer, même si ancien et nouveau chemins partagent accidentellement le même mauvais store. Cette mutation vérifie la sensibilité de l’oracle de sélection ; elle ne prouve pas la position du seuil de sortie.
- `mutation-cpu.patch` autoriseE511/K10 dans le vrai garde. Le testCPU et le harnais du garde doivent détecter la violation de domaine. Restaurer les sources et rerun les contrôles affectés.

Ne pas compter une mutation non appliquée comme détectée. Pas de mutant volontairement divergent autour des barrières, ni d’accès hors buffer.

## Vérification source et CI à exécuter par le parent

Les références formal-proof-skill Rust/Kani et C/C++ ont été lues. Ce skill ne fournit pas de vérificateurMSL ; traiter le shader Metal comme du C++CBMC ordinaire ou réécrire son calcul enRust ne constituerait pas une preuve de la source originale. Les tentatives prévues sont explicitement distinctes :

- `cargo kani --lib --no-default-features --harness flash_top10_early_exit_is_confined_to_requested_flash_geometry` : vrai garde CPU de production, domaines i32/usize/bool complets, aucune hypothèse, quatre covers. Cela ne vérifie pas le classementMSL.
- `cargo kani --lib --features mlx,chat --harness actual_flash_top10_early_exit_preserves_a_symbolic_input` : vraie entréeMLX, un motifFloat32 arbitraire parmi511uns, IDs et octets des scores comparés, unwind513 et trois covers. Une absence d’outil, ICEintrinsics243, FFI non prise en charge ou timeout reste NON VÉRIFIÉ. Aucun stubGPU ni modèleCPU ne remplace ce chemin.
- Bornes préparées180s/4Gio par arbre, jobs1 et jonction obligatoire. Le chemin du lanceur borné reste null jusqu’à identification par le parent ; la commande cargo seule n’impose pas ces bornes.
- Format, ClippyCPU etMLX stricts, builds CLI/tests `mlx,chat`, testCPU ciblé puis suites pertinentes. Les tests physiques ignorés compilent dans la CI macOS mais n’y sont pas exécutés. `ci.patch` propose seulement d’ajouter le nouveau harnaisCPU au job ciblé existant ; à fusionner avec les autres hunksCI par le parent et vérifier sur le SHA final.

## Qualité puis vitesse

Après le composant et les routes réelles : réutiliser les deux régressionsMoE existantes `two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads` et `unique_windows_preserve_outputs_pending_slots_and_greedy_fallback` avec flag1 (UNIQUE1 pour la seconde). Elles couvrent les banques Hot/Cold, les capacités, fenêtres découpées, slots protégés et reprise après lecture échouée ; ce changement n’ajoute pas son propre ordonnanceur.

Ensuite36préfixes DFlash/captures/logits/états, MTPsession/budgets, bridge43 exact annulation/reprise/erreurs. Bridge ADAPTIVE1 avec WIDTH réellement absent, y compris de l’environnement parent hérité. Pas de débit avant cette qualité entière.

Débit isolé82T : même binaire immuable H=flag0/C=flag1, MAIN/HOLD B2fixe WIDTH1/ADAPTIVE0,128warm+128mesure,3Gio/53slots, alloc64Mio, Hot/Wide/63/UNIQUE1, prélecture/LES/shared/stable0. F01 et autres expériences doivent être explicitement figés communs ou absents ; le patch ne les transporte pas. HC puis CH de confirmation, sorties golden60 et128/127, acceptation et octets SSD logiques, conditions d’alimentation, VM et processus archivés. OFF/MTP1 frais avant promotion, aucune addition de gains. Gardes45/49 et post30, footprint≤12Go/ledger propre et capscompression77B historiques conservés dans leur propre statut.

La présente préparation n’autorise ni lancement automatique, ni push, ni remplacement de l’application. Le parent applique, construit, vérifie et sérialise les campagnes.


Révision25octobre avant application : les deux tests initiaux modifiant l’environnement après load ont été retirés sans jamais être exécutés. Le correctif ne concerne que les fixtures/protocole, productionF06 inchangée. Les hashes/manifeste sont renouvelés ; aucune ancienne revendication de test dynamique loadonce ou RAII d’environnement n’est conservée comme résultat.
