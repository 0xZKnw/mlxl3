# Audit statique local — Qwen3.8-27B vers 15 tok/s, 16 octobre 2026

Périmètre : lecture seule du worktree /Users/justin/.codex/worktrees/qwen27-smallm/mix-stq1_0, du rapport utilisateur et des historiques. Aucune modification native/journal, aucun benchmark, chargement GPU, build ou preuve lancé par cet agent. Les propositions ci-dessous ne portent aucun gain mesuré.

## Ce qui doit réellement accélérer

La baseline qualifiée Q27-03 est MTP1 + MLP dense batch M2/M4, ~11,0765 tok/s français / 10,9362 code. 15 demande environ 36% de débit en plus, soit environ 27% de temps en moins. Cela vise un gros morceau du tronc, ou une meilleure acceptation à coût inchangé, pas quelques microsecondes de sampler.

Trajet réel :
- bridge_generate_round (native/src/main.rs:2242+) choisit le token, decode et flush les deltas, puis appelle Session::advance ou model.forward.
- MTP Head::draft_chain (mtp.rs:729) construit residual, head complet, log_probs, argmax, puis ramène les IDs une fois.
- Session::advance (mtp.rs:869) appelle verify_mtp([anchor, draft...]), chat_greedy_ids, greedy_accept, commit_dflash_verification, puis répare seulement les KV utiles du MTP.
- Qwen35Moe::verify_mtp (qwen35.rs:1594) garde les normes/résidus par ligne M1, groupe des projections, MLP M2/M4 seulement; réalise logits.eval AVANT chat_greedy_ids.
- Dense MLP (qwen35.rs:262) : ProjectionBundle gate/up -> Array::swiglu FP16 compilé -> Projection down.
- GDN (qwen35.rs:1992+) : grouped qkv/z, deux petites projections a/b, prepare_qkv fusionné, recurrence fusionnée avec gates + tape, RMSNorm128, precise_swiglu F32 puis F16, out_proj EXL3.
- 48 GDN Hk16/Hv48/Dk=Dv128/conv4 et 16 full attentions. Le vieux prototype Rapid avait Hv32, pas48.
- Exl3Linear::forward et Exl3Group::forward (linear.rs:256/719) : FP16 scale + WHT128, QMV, réduction splits FP32, FP16 + WHT128 + scale FP16. Deux shaders distincts actifs, tile et mapped.
- Attention::forward_verification_impl (qwen35.rs:555) garde RMSNorm/RoPE/SDPA par ligne; KV concaténés à chaque ligne; commit recoupe au préfixe retenu.

## Trois premières actions recommandées

1. **Filtre funnels 32 bits limité à quatre codewords, M1/M2 réels K1..4/Mul1.** Parent a déjà choisi ce filtre. Les deux shaders contiennent un ulong pour chaque fenêtre. Garder EXACTEMENT word0/word1, leurs deux fenêtres, extraction, ordre FMA/quad/SG/splits/NT. Tester shift=0 explicitement, éviter décalage par32, et conserver le chemin général K6/8. Ne pas refaire la fenêtre unique huit codewords K2 qui a régressé 3–8%, ni PERF106 K6/MCG (−3,35%). Raison nouvelle : fenêtre quatre codewords, nouveau code amont, H5120/K1..4/Mul1, M1/M2, nouvelle baseline.
2. **Filtre GDN préparation + récurrence, nouvelle Hv48, avec l'archive propre Rapid.** L'ancienne fusion est numériquement bien contrôlée mais retirée après dérive; elle ne constitue pas un gain actuel. Reprendre uniquement les vrais calculs propres existants, Hk16/Hv48/D128/T1..2 d'abord, tape/replay exacts. La différence concrète est ce 27B, 48 heads et MTP1 avec MLP2, pas une répétition aveugle MoE35B/Hv32. Filtrer la frontière avant tout portage par défaut.
3. **Avant une grosse réécriture : profil non perturbant du vrai bridge 27B/MTP1 et AR.** Les profils de MoE35B et chaînes proxy n'attribuent pas le coût dense courant. Utiliser un sample CPU court + compteurs de construction graphes/données, sans barrières ajoutées, puis rapprocher d'un témoin non instrumenté. Décider entre KV target, fusion fin GDN/début EXL3 et AR run-ahead selon les bulles réellement observées. Ni cache C++ ni sampler brut ne sont un choix par défaut.

## Atlas des candidats, ordonné par plausibilité et effort

| Candidat | Frontière / différence nouvelle | Premier prototype utile | Risque / oracle décisif |
|---|---|---|---|
| 1. Funnels32 quatre codewords | tile/mapped QMV de toutes les grosses projections cible; parent sélectionne | Remplacer seulement fusion/décalage entiers, pas fenêtres ni géométrie | Équivalence entière bitbasis/ringwrap/K1..8, cas shift0, puis toutes partielles FP32 finies bitexactes |
| 2. GDN prepare + recurrence Hv48 | prepare_qkv -> step_with_gates(_tape), 48 couches | Adapter fusion propre Rapid archivée Hv32 vers48, T1/2, garde stricte | ÉtatF32/convF16/QKV/tapes/replay; occupancy d'un kernel plus gros peut annuler les dispatchs gagnés |
| 3. Fin GDN -> préparation out_proj | RMSNorm128 + precise_swiglu + suh + WHT128, 6144->5120 | Une frontière alignée exactement sur les heads128; produire xhat d'out_proj | Préserver réduction norme, deux arrondis, gateSiLU F32 compilé, FP16 activation et scale/WHT. TableSiLU du harnais HENK0O09 permet oracle exact, pas exp approximatif |
| 4. Fin gate/up -> entrée down dense | Exl3Group gate/up 34816 -> SwiGLU -> down WHT17408 sur64couches M1/M2 | Exposer raw projections/scales et préparer le down sans buffers gate/up finaux | Ne pas copier moe::glu_down_input : son SiLUF32 et butterflies ne sont pas le contrat dense actuel. Oracle MLX primitive finitude/bitexact tous points, shapes/échelles différentes |
| 5. Append KV cible une fois par bloc | Attention::forward_verification_impl actuellement append par ligne MTP1/2 | Préparer Q/K/RoPE M1 pour les2lignes, concat KV total une fois, SDPA M1 sur vues de préfixes causaux | Strides par tête changent et peuvent changer dispatch/copies SDPA; aucune attention approximative; comparer logits/128états à préfixes remplis256/4k/8k, tous commits |
| 6. AR run-ahead borné 1token | main.rs émet/flush le token AVANT construire le forward suivant | Token device opaque + next forward async pendant decode/flush du précédent, aucune file illimitée | Distinct du seul forward_lazy PERF07. EOS/cancel/stop/budget/backpressure et cache préfill : état spéculé ne doit pas être capturé comme committé; besoin vraie intégration bridge |
| 7. Reconstruire avec MLX0.32.4 | Backend courant0.32.2; compile/alloc/softmax peuvent avoir changé | Venv/build séparés avec ABI correspondante, aucun swap DYLD de lib sous ancien moteur | Métallib/exp/log/projections changent potentiellement les bits; oracle source/dtypes/logits/caches entre versions, mesure seulement après parité; aucun gain promis par changelog |
| 8. Préparer métadonnées Rust des kernels | array::metal_kernel reconstruit CString source/header/noms et Vec à chaque appel; coût C++ seul déjà faible | PROFIL Rust de cette préparation d'abord; si significatif, descriptions immuables par projection/forme | Le profil C++ excluait ce coût, mais 0,111% mesuré ne justifie pas nouveau cache. Clé complète/collision/eviction/ownership/erreurs NUL à préserver |
| 9. GEMV a+b à vecteurs colonnes batchés | GDN verify sérialise deux projections × M ×48couches | Paire96x5120 préparéeauload, weight[None,:,:] @ x[:,:,None], une API dense bornée | SMALLM-B déjà162cas exacts, pas fullmodel; économieproxy~0,84ms/518ms donc faiblepriorité. Ne pas utiliser forward_dense_rows_exact, qui a divergé sur GDN FIX09 |
| 10. Paquet MTP draft+posterior, transfert unique | MTP actuel lit IDsdraft avant target et logits.eval avant argmax | Reprendre mécanisme DFlash SINGLE-TRANSFER avec typefermé + targetgreedy existant + un paquet3IDs pourD1 | Ancien MTPDEVICE conserve logits.eval et n'estpas ce paquet, mais déjà<1,4%; DFlash variante a dérivé et a étéretirée. Nouveaudomaine27B/MTP1 àjustifier; erreursIDs/NaN/reset/commit et lookuphost préservés |
| 11. Draft MTP argmax brut seulement | draft_chain normalise avant argmax; cible reste canonical | Option test-only rawargmax DU DRAFT, mêmes poids/tête complète | Peut changer ties et donc acceptation. Jamais changer chat_greedy_ids cible. Gains probablement petits; E2E/corpus indépendant, pas calibration sur FR/code de mesure |
| 12. Prefill GDN chunkwise et attention partagée | Grandsprompts, pas la cible15tok/s court contexte | Diagnostic TTFT séparé et filtre math exact avant toute adaptation | Chunkwise/softmax partitionné réassocient souvent flottants; souscontratstrict actuel, exclure si ordre changé; ne pas annoncer un gain decode |

Les candidats3/4 sont des fusions plus ambitieuses, pas «activer un flag». Ils doivent être stoppés immédiatement si une frontière numérique échoue. Les candidats8/9/11 ont faible pouvoir de produire 36% seuls. L'absence de gain n'autorise pas à additionner leurs proxies.

## Protocole concret de sélection

### QMV funnels
- Réutiliser benchmarks/benchmark_decode_tiles.py, benchmark_smallm.py::qmv et benchmark_smallm_reuse.py plutôt qu'un nouveau gros banc.
- Oracles indépendants : calcul entier scalaire ring/codeword (bits de base, déterministe), reference MB1/NT1 puis partiellesFP32 stock. Tester les deux shaders, K1..8/3codebooks pour nonrégression de shared source; chronométrer uniquement formesK1..4/Mul1 réellement présentes.
- M1 NTproduction : qkv/z K2 et gate/up K2 NT4, gate/up K1/K3 NT2, headK3 NT4, downNT4. M2 garde MB2/NT2/SG/splits actuels.
- Garder cellules synthétiques petites avec splits1/4/8, vrai trellis/scales/checkpoint; compter formes et valeurs, finitude, exactbits; mutation extraction doit être détectée.
- Après correction seule : 40paires AB/BA isolées et chaines déjà utilisées, 3warmups; mêmecachecommun si fixturepartielle. Gain≥3%, >=32/40gagnantes, AB etBA chacun≥3%, dériveparbras≤3%.
- Si filtrepasse : portage minimal, builds/proofs/tests horsGPU, modèlefull logits248320/hidden5120/128états, prefill23/24/257 et M1/M2/touspréfixes; bridge exact FR/code MTP1 et témoinAR, defaultallocator, max256/warm64, ABBA puis BAAB conditionnel. Le micro ne fournit aucun tok/s.

### Fusion GDN
- Reprendre l'archive docs/measurements/engine-v1.4.2/rapid-followup-raw.tar.gz; ne pas remplacer le code courant complet.
- GardeCPU nouvelle Hk16/Hv48/D128/C4/T1..2; symboliques i32/Kani et refusal horsdomaine, offsets/taille/launch/scratch bornés.
- Utiliser prepare_qkv + step_with_gates(_tape) actuels comme reference indépendante, poids conv du 27B. États nuls/non nuls/extremums/annulation, sorties/nonvides/finitude/FP16-FP32 exactbits.
- TempsT1/T2 et tapes, replay chaquepréfixe, mutation k_scale->q_scale détectée; suite modèle et bridgecommeQMV avanttiming.
- Comparaisonfrontière isolée/chaîne avantmodèle, seuilsfixés; si fusiongrossen'occupepas mieuxGPU, retrait préservé.
- CBMC réelMSL et KaniArray ont historiquement échoué pour incompatibilité/ICE; tenter selon instructions sans stub ni traductions trompeuses, et garder domainesCPU distincts.

### Profil avant host/IO
- Capturer CPU du bridgeMTP1 et AR sur campagnes séparées. Ne pas sommer des sousétapesentouréeschacuned'eval.
- Compter appels de factories, volumesCStringRust, allocation/shapes; compteurs diagnostiques retirés avant vitesse.
- Si I/Odecode+flush est sous1% du total, ne pas écrire ARrun-ahead; si constructionsoumission domine, tester frontièrebornée plutôtque nouveau runtime.

## Régressions et preuves existantes à réutiliser

- qwen35::tests::dense_smallm_dispatch_matches_serial_prefixes et dense_m3_fallback_matches_serial_prefixes.
- checkpoint_optimization_matches_saved_logits_and_states : vrai checkpoint, logits/128états.
- mtp_prefill_verification_and_rollback_match_target; selective_dflash_commit_matches_exact_prefix.
- mtp_recursive_sessions_match_greedy_and_exact_caches; mtp_lookup_accepts_rejects_and_repairs_exact_caches.
- fused_gdn_preparation_matches_mlx; gated_delta::tests::packed_step_with_gates_matches_graph; tape_replay_matches_every_serial_prefix.
- snapshot_restore_replays_logits_and_state_exactly; compact_prefill_snapshot_preserves_logits_states_and_budget.
- main tests dflash_output_budget_matches_target_and_skips_unused_work, bridge_prefix_cache_replays_cold_logits_exactly; scripts/check-mtp-depths.py budgets/cancel/recovery et JsonProcess borné.
- tests/test_smallm_reuse.py et analyseurprofile : refusrapportsvides/partiels/nonfinis/dérive, sourcehashdansclé.
- CI rust.yml PR/push existante; GPU ignoré dans suitegénérale ne devient pas exécuté.

## Pistes à ne pas relancer sans vraie différence

- Q27-05 pipeline dense sur MLPbatch : +2,08% FR/+0,64% code sousseuil.
- Q27-06/07 hoistingdecode/NT/MB4 : aucun filtre utile. Q27-08 LUTcomplet : pasgain stable.
- Q27-10 Q4compact et Q27-12 blocsEXL3 : tête seule68%plusrapide maisrégressiongénération FR; acceptationbaisse.
- HENK0O-QWEN27-02 fenêtreunique8K2 uint32 : régressions3–8%; PERF106 K6/MCG −3,35%.
- docs/qwen38-decode-local.md : N-majortranspose, cooperativeshuffleloads, K3windows, SGdifférents, unrolling, packedcodebook/FP16dotdéjàrejetés. Anciencheckpoint2,75bpw différent n'autorisepasrépétition sansmécanisme/précondition nouvelle.
- PERF111 repackhead[N/2,K,Nlocal,words] exactmais0,76..1,81%microseulement/copiedupoids+QMM àadapter; pas une nouvelleidée.
- PERF07 forward_lazy→argmax : régressionaprèsretrace, retiré. Ne pasleconfondreavecrun-ahead.
- PERF04 compileHadamard/scales : −4,49%; PERF09 compilesortieMoE −0,65%.
- HENK0O09 RMSNorm128+gateSiLU seul : exact40fixtures maispasgainmodèle; norme/gatedense résiduelfusiondéjàexpérimentalnonconcluant.
- GDN gatespartagéesthreadgroup/SIMD PERF42/43, petitsinvariantsPERF36, adressageincrémentalQMV PERF35 : déjàrejetés.
- Argmaxdirectcible PERF32 et head+argmaxPERF102 : rejets; cible log_probs/rounding est contratpréservé.
- KV8long a échouéqualité; aucunequantifcache/poids/dtype, nouveauxordresderéduction/fastmath souslaprésentecampagne.

## Sources lues

opti.md Q27-00..14 et entrées nommées ci-dessus; AGENTS.md; rapport utilisateur téléchargé (sections4/5/6/8); docs/qwen27-vers-15-20261016.md; docs/qwen27-optimization-20261016.md; docs/qwen27-speed-20261008.md; docs/qwen27-smallm-m5.md; docs/qwen27-m5-round2.md; docs/qwen38-decode-local.md; docs/engine-v1.4.2-upstream.md; docs/measurements/qwen27-smallm/inventory-table.md et sources Rust/Metal/FFI/bench ciblées. Lecture ciblée, pas audit intégral. Aucun débit nouveau mesuré.
