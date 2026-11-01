# T18a — périmètre de qualification, 31 octobre 2026

Préenregistrement `ISSUE59-T18A-01` dans [opti.md](../../../opti.md). Ce document décrit les vérifications à exécuter par le coordinateur ; **aucun test, build, preuve ou GPU lancé par cet agent**. Le code et les résultats de T04 restent archivés/retirés ; aucun port de T04 n'est repris.

## Validation des traces et référence PR61

Adaptation minimale du [patch PR61 archivé](../issues-triage-20261030/pr-61/changes.patch), tête `a36ba5077bf386466660d6b478058fbe6880245c`, régions 4130–4205 : `validate_events` accepte `schedule=static|dynamic`, absence équivalente à `static`. Le modulo historique reste obligatoire en Static ; Dynamic accepte tout worker dans `0..min(workers,jobs)` sans exiger qu'ils travaillent tous. Dans les deux cas : 1..4 workers, 1..512 jobs, longueur complète, index exact/unique dans l'ordre, octets complets, erreur nulle, temps valides et intervalles successifs sans recouvrement pour un même worker. La politique est conservée dans `disk_batches`. Le parseur garde le format Dynamic historique de PR61 ; le nouveau fallback jobs<=workers/worker unique est vérifié sur la production native, pas imposé rétroactivement aux anciens fichiers.

Les nouvelles régressions utilisent la vraie entrée Python `validate_events` : un worker draine dix jobs, deux workers seulement sur quatre restent actifs ; les mêmes données doivent être refusées sous Static explicite ou absent. Seize corruptions couvrent politique inconnue/null/entier, nombre de workers, worker hors borne/négatif/borné par le nombre réel de jobs, chevauchement, erreur, index erroné/dupliqué/désordonné, lot incomplet, compte d'octets et temps inversé. Le test legacy existant affirme désormais la politique exportée Static. **18 nouveaux cas paramétrés attendus**, à compter réellement après exécution. Mutation utile au coordinateur : désactiver le modulo Static ou la garde de chevauchement doit faire échouer les refus correspondants ; l'ancien validateur refuse la fixture Dynamic à un seul worker dès le premier job.

Le chemin utilisateur est `profile_flash.run → validate_events → résultats/export`, sans nouvelle API ou dépendance. La [CI existante](../../../.github/workflows/rust.yml) exécute déjà Ruff format/lint et `tests/test_flash_profile.py` sur PR/push ; le workflow seul ne prouve aucun run distant. Charger le skill formal-proof et sa référence Python a été fait. Une tentative CrossHair sur l'implémentation réelle reste à coordonner ; son absence historique ne vaut pas preuve, et aucun nouveau harnais qui réécrit le validateur n'est proposé.

## Petit smoke physique avant modèle

Sélecteur nouveau préparé par le coordinateur : `array::tests::disk_schedule_profile_matches_real_batch_assignment`. Matrice annoncée : n=1/4/5/9/17/512, workers=1/2/4, 18 cas/1 644 records, vrais `Array::read_records` vers Metal, octets et canaris, affectation/intervalles/profil réel. Flag1 doit observer **9 lots Dynamic et 9 Static**, flag0 **0 et 18**. Cela vérifie explicitement la garde et la frontière 512 ; sélection seule, `--exact --ignored --nocapture --test-threads=1`, dans deux processus distincts puisque l'option est fixée au premier lot du thread. Vérifier le compte réellement imprimé, pas seulement l'exit code.

Sélections existantes complémentaires, sources actuelles et noms complets :

| Sélecteur | Frontière couverte / limite |
| --- | --- |
| `ssd_moe::tests::disk_records_fill_only_owned_ranges_and_reject_bad_sources` | Vrai fichier, offsets et segments disjoints, sentinelles, tailles/troncature/overflow. Un seul record : fallback Static. |
| `ssd_moe::tests::disk_batches_reject_aliases_and_validate_every_record_before_writing` | Vrais buffers partagés, alias entre banques, prévalidation de tout le lot et échec sans écriture. **Deux jobs seulement**, donc toujours Static sous la nouvelle garde ; ne qualifie pas Dynamic. |
| `ssd_moe::tests::two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads` | Oracle MoE indépendant, neuf capacités 10/19/20/24/53/71/80/89/512 × overlap0/1/lecteurs1/2/4, paires froid/chaud/répétées, rows1..8, refus et compteurs invariants, lecture invalide/invalidation/reprise. Exiger des lots Dynamic réellement non nuls dans le bras1 et la même sortie complète. |
| `array::tests::scheduled_route_readback_does_not_wait_for_later_gpu_work` | Événement propre des IDs, travail GPU ultérieur encore pending ; protection de l'overlap, distincte du temps d'I/O. |
| `array::tests::readback_profiling_preserves_layouts_errors_and_overlap` | Profiler/readback sur vrais layouts/erreurs/chevauchement, sans confondre disponibilité et attente globale. |
| `ssd_moe::hybrid_tests::demanded_reads_start_while_gpu_resident_experts_are_pending` | Hot masqué réellement en vol, routes disponibles et lectures demandées démarrées avant sa fin ; ne pas retirer l'événement précoce. |
| `ssd_moe::hybrid_tests::wide_hot_is_opt_in_and_overlaps_routes_for_three_to_eight_rows` | Wide Hot rows3..8, garde opt-in et recouvrement routes/I/O. |
| `ssd_moe::hybrid_tests::unique_windows_preserve_outputs_pending_slots_and_greedy_fallback` | Cohortes maximales sûres, slots pending protégés, sorties exactes et fallback/reprise. |

Les tests de prélecture `ssd_moe::tests::completed_prefetches_compute_while_other_experts_are_read`, `completed_previous_layer_reads_do_not_block_next_prefetch_cohort`, `failed_prediction_invalidates_reserved_routes_and_recovers` et `failed_hot_compute_invalidates_reserved_routes_and_recovers` restent utiles pour pending/leases/erreurs quand le chemin de lecteur concerné est partagé ; le baseline lookahead0 ne les rend pas actifs en génération ordinaire. Ne pas les présenter comme une validation de T18b : aucune publication partielle n'est introduite.

## Cinq oracles modèle à réutiliser, sur le nouveau binaire

Les noms et fixtures se reprennent des plans T04 [restants cinq](../issue59-t04-20261031/model-quality-remaining-plan.json) et [MoE isolé](../issue59-t04-20261031/moe-component-plan.json), **pas leurs binaires ni leur flag T04=1**. Créer les futurs plans T18a avec identité/options propres ; T04/T17/T05 restent OFF, Dynamic1 seulement, mêmes checkpoint/référence indépendante/draft, cache experts3 Gio, quatre lecteurs. Chaque sélection physique est explicitement non vide et unique.

1. `qwen4::tests::flash_matches_independent_reference_states_and_logits` : cinq positions, 248 320 logits complets par position et états GDN, finitude/compte/parité.
2. `qwen4::tests::flash_components_match_reference_on_identical_inputs` : composants attention/GDN/MoE sur les mêmes entrées indépendantes.
3. `qwen4::tests::flash_pair_verification_preserves_serial_logits_and_all_states` : vraie paire, logits/états sériels et transitions de vérification.
4. `qwen4::tests::flash_mtp_session_preserves_greedy_sequence_target_and_head_states` : séquence greedy, états target/head, budgets1/2/3/17 et reprises.
5. `qwen4::draft::flash_draft_captures_and_all_prefix_commits_match_serial` : 36 préfixes, largeurs1..8, captures/commits/rollback et états/logits exacts.

Leurs résultats historiques T04 ne valident pas le nouveau transport ; les sélecteurs et données indépendantes sont réutilisables. Avant vitesse, le vrai bridge17 doit encore contrôler AR/MTP/DF dans deux bras du même binaire, golden externe complet et compteurs, empreinte et processus/lecteurs joints. Puis seulement le pilote release ACCA prévu. Le texte CLI seul ne prouve pas les logits/états.

## Preuves inchangées et limites

Les formules EXL3, dtypes, arrondis, H128, réductions, routeur et kernels n'étant pas modifiés, leurs tests de microarithmétique exhaustifs ne sont pas répétés automatiquement. Réutilisation conditionnelle à l'identité des sources effectives et à l'absence d'impact sur ces appelants ; les gardes CPU de destinations réelles gardent leur domaine de preuve si inchangées. Les nouvelles propriétés d'attribution atomique, terminaison, drainage/jointure, erreurs et visibilité vers Metal exigent leurs contrôles propres. Une preuve de queue portable ou les 4 084 cas cloud ne prouvent ni GCD ni MLX/Metal ni concurrence du moteur complet.

Les limites Kani MLX ICE et parsing C++ historique restent explicites ; tenter le vrai chemin applicable selon le protocole root, sans stub d'atomique ni domaine artificiellement réduit pour verdir. Les gardes physiques T18A-w1/0 nouveau Swapout/3 Gio/admission11 Go modèle et2 Go composant, absence de compilations/proveurs/autre modèle, secteur et joints bornés sont celles d'opti.md. Aucun gain, aucune qualité nouvelle passée et aucune app modifiée ne sont annoncés par ce document.
