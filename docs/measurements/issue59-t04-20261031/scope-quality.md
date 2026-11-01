# T04 — contrat numérique et périmètre de qualité

31 octobre 2026. Revue **lecture seule**, après
`ISSUE59-T04-SCOPE-01` en tête de [opti.md](../../../opti.md). Sources locales
lues exclusivement dans le worktree `slotstream-flash/mix-stq1_0`. Aucun
prototype, test, build, GPU ou proveur lancé ; seul ce rapport est ajouté.
Les contrôles ci-dessous sont **réutilisables/proposés, non exécutés pour T04**.

## Frontière exacte

[SsdMoe::calculate_slots](../../../native/src/ssd_moe.rs:344) reçoit les lignes
routées F16 et produit GU `[R,2,640]` F16. Il prend les vues gate/up `[R,640]`,
appelle [Array::swiglu](../../../native/src/array.rs:1328), puis
`linear::expert_slots` pour down. Le [FFI opération 6](../../../native/ffi/mlx_bridge.cpp:569)
utilise déjà `mx::compile((gate * sigmoid(gate)) * up, shapeless=true)`.
Les primitives sont typées F16 : **sigmoid F16, premier Multiply F16
(SiLU), deuxième Multiply F16 (×up)**. Cela ne désigne pas un SiLU F32
suivi d'un seul cast final.

[expert_mapped_layout](../../../native/src/linear.rs:1359) prépare ensuite
`half(activated) × half(suh)` **en F16**, reshape `[R,5,128]`, puis H128 MLX
normalisé. La frontière minimale proposée conserve H128, QMV K2/Mul1,
splits, ordre des sommes, H128 de sortie et échelles `svh`. Elle doit donc
reproduire **tous les arrondis F16 avant H128**, pas seulement le dtype final.

Ne copier ni `precise_swiglu` (opération 8, SiLU F32), ni
[moe::swiglu_down_input](../../../native/src/moe.rs:505) : ce dernier reçoit
des partielles GU et utilise un autre calcul SiLU F32. La correction
`precise::exp` du sigmoid F32 de T17 n'est pas une recette automatiquement
valable pour ce sigmoid **F16 compilé**. L'oracle est le vrai graphe MLX de
production, pas une formule CPU ou un second shader.

**Deux dimensions de lignes à distinguer :** B1/B2/B4/B8 sont les tokens ;
les experts physiques peuvent représenter 10/20/40/80 routes avant masque,
tandis que Cold peut n'en conserver qu'un sous-ensemble. Tester aussi
R=1/2/4/8, les doublons et les tailles intermédiaires, pas seulement R=B.
Hot garde ses lignes physiques et ses slots sentinelles `u32::MAX` ;
`ExpertSlots::gather_scales` fournit zéro aux absents. Ne pas lire une banque
à l'index sentinelle, réduire les lignes physiques aux seuls hits, ou réordonner
les origines des routes groupées. Les tests doivent prouver que le candidat
est réellement demandé, sans fallback silencieux.

## Oracles et tests existants à réutiliser

| Entrée exacte | Ce qu'elle couvre / limite |
| --- | --- |
| Nouveau micro via **vraie API de production T04** contre `gate.swiglu(up).mul(suh)` F16, puis même H128 MLX | Comparer directement la sortie **avant H128**, puis la préparation et la vraie down complète. Formes, dtypes, cardinalités non nulles, finitude et bytes complets ; ne pas valider seulement un capteur multi-sortie dont le codegen diffère du mono-sortie servi. Ce raccord n'existe pas encore. |
| `linear::tests::flash_slot_shapes_match_resident_reference_for_every_route` | Slot-major GU `(2560→640,2 projections)` et down `(640→2560,1)`, R=1/2/4/10..80, F16/F32, strides, formes stables, groupes/restore et masques vide/partiel/plein. Compare les sorties à des `Exl3Linear` indépendants par route, contrôle mauvais slot/origine et absence de readback. **Ne contient pas SwiGLU entre les deux projections : insuffisant seul pour T04.** [Source](../../../native/src/linear.rs:1750). |
| `linear::tests::gpu_resident_plans_and_scale_gathers_preserve_bounds_and_strides` | 28 cas, 514 IDs dont invalides, banques/IDs stridés, poison, sentinelles et valeurs d'échelles indépendantes ; six refus et mauvais mapping. Utile si l'API touche au gather/layout ; ne prouve pas l'activation. [Source](../../../native/src/linear.rs:2044). |
| `ssd_moe::tests::two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads` | Vrais poids/entrées/sorties MoE indépendantes (positions 0/1/4), Cold puis Hot, 9 capacités, overlap OFF/ON et 1/2/4 lecteurs ; paires distinctes et répétées, lots **1..8** en deux passages, erreurs de formes/lectures et récupération. Batterie large à placer **après** le smoke court. [Source](../../../native/src/ssd_moe.rs:2131). |
| `ssd_moe::hybrid_tests::demanded_reads_start_while_gpu_resident_experts_are_pending` ; `wide_hot_is_opt_in_and_overlaps_routes_for_three_to_eight_rows` ; `unique_windows_preserve_outputs_pending_slots_and_greedy_fallback` | Frontières de l'ordonnanceur réellement traversées : Hot retardé, lecture demandée qui avance, Wide, cohortes 6..8, slots en vol et récupération. Garder les assertions de chemin/compteurs ; la seule parité du texte ne vérifie pas l'overlap. [Source](../../../native/src/ssd_moe/hybrid_tests.rs:85). |
| `ssd_moe::shared_tests::shared_batches_match_single_row_projections_and_scalar_gates` | Contrôle si une API commune peut atteindre le shared : couches 0/23/47, lignes 1..8, F16/F32 et strides ; sériel contre batch, gates scalaires, mutation gate/up détectée. Si T04 est strictement routé, le shared est un contrôle inchangé, pas un gain revendiqué. [Source](../../../native/src/ssd_moe/shared_tests.rs:84). |
| `qwen4::tests::flash_components_match_reference_on_identical_inputs` et `flash_matches_independent_reference_states_and_logits` | Fixture indépendante du modèle : composants, **5 positions ×248 320 logits complets**, routes et états récurrents ; contrats bit à bit là où explicitement imposés, tolérances propres aux autres composants. Réexercer avec T04 réellement activé, T17/T05 OFF. [Source](../../../native/src/qwen4.rs:3806). |
| `qwen4::tests::flash_pair_verification_preserves_serial_logits_and_all_states` ; `flash_mtp_session_preserves_greedy_sequence_target_and_head_states` ; `qwen4::draft::flash_draft_captures_and_all_prefix_commits_match_serial` | Paires/commits/continuations, MTP budgets 1/2/3/17 avec acceptation et rejet, DFlash **36 préfixes de largeurs 1..8**, captures/logits/états exacts et rollback. Compléter par bridge17 AR/MTP/DF même nouveau CLI flag0/1 et golden externe, avant vitesse. [Bilan des contrats déjà exercés pour T17](../issue59-t17-20261029/verification.md). |

Le test `weighted_epilogue_runs_on_real_mixed_hot_cold_with_physical_hot_rows`
nécessite T05=1 : **ne pas l'utiliser tel quel pour qualifier T04 seul**.
Sa stratégie `prime_half` et l'assertion 5 hits/5 misses sont réutilisables
dans un test court mixte avec T05 OFF, sorties complètes et compteur T04.

## Matrice courte puis contre-exemples

Premier smoke : GU `[R,2,640]` non uniforme pour R=1/2 et R=10/20,
échelles F16 distinctes par route/bloc, vue gate/up stridée issue du vrai GU,
une fixture Cold et une mixte réelle. Conserver des ancres littérales
indépendantes, par exemple gate=16, up=1/8, suh=1/4 donnant 1/2 avant H128
quand le vrai sigmoid F16 vaut 1. Aucune valeur attendue calculée par le
nouveau shader lui-même.

Puis couvrir R=1/2/4/8/10/20/40/80, blocs H128 0..4 et frontières 127/128,
255/256, 639 ; strides de GU/scales, masques vide/partiel/plein, origines
répétées/groupées, poison NaN/Inf **non sélectionné**, alias et immutabilité.
Si le nouveau garde est étroit, vérifier explicitement ses refus ou fallbacks
sur rangs, dtypes, tailles, indices et débordements invalides avant requête.

Deux contre-exemples **analytiques proposés, non exécutés** :

- Gate `2^-24` (minimum subnormal F16), up=2, suh=1 : le sigmoid F16 est
  voisin de 1/2 ; l'arrondi SiLU F16 de `2^-25` donne zéro avant ×up. Un
  calcul F32 traversant ce point peut finir à `2^-24`. Vérifier par le vrai
  stock GPU, notamment son FTZ ; ne pas imposer aveuglément un CPU IEEE.
- Gate=16, up=`2^-24`, suh=`2^-5` : activation F16 `2^-20`, puis produit
  pré-H128 `2^-25`, au milieu d'arrondi vers zéro. Omettre ce dernier
  arrondi avant un H128 sur ligne constante peut rendre sa première
  composante non nulle. La comparaison pré-H128 localise mieux l'écart
  qu'une seule down finale, qui peut masquer une très petite erreur.

L'exhaustif utile est **63 488 patterns de gate F16 finis**, croisés avec
quelques up/scales déterministes aux frontières d'arrondi (±0, minimum et
maximum subnormaux, minimum normal, voisinages de ties, signes opposés).
Choisir les autres facteurs pour maintenir les sorties finies ; traiter
séparément overflow/Inf/NaN, avec contrat de classification NaN explicite.
Ce n'est pas un exhaustif de tous les triplets gate/up/suh. Mutation pertinente :
supprimer un cast F16 réel ou décaler de 640 la vue up, exiger l'échec de
l'oracle, restaurer puis requalifier les seuls contrôles affectés.

## Antécédents et effort de preuve

L'antécédent **Q27-2026-10-16-17** est très proche numériquement : fusion
dense SwiGLU→suh→H128, **268 contrôles exacts**, 64 échelles réelles,
63 488 gates finies et mutation d'arrondi détectée ; vitesse non concluante,
aucune promotion. Les [notes de préparation](../qwen27-max-20261016/dense-prepare-notes.md)
étaient prospectives ; le [rapport final](../qwen27-max-20261016/dense-prepare-profile.json)
et le journal portent les résultats ultérieurs. T04 ne doit pas se présenter
comme une idée jamais essayée : nouveauté éventuelle = **raccord Rust SSD
slot-major, H640, routes/masques, frontière pré-H128 gardant H128 MLX**,
et coût de la chaîne réelle. POSTPROFILE05-C fusionnait gather/scale et a
été rejeté après audit/correction ; ses tests prototype ont été archivés
et retirés, pas supposés encore présents dans le runtime.

Pour un nouveau garde CPU de production, Kani peut vérifier réellement la
géométrie et les offsets : R symbolique 1..80, width=640, GU interleavé
`row*1280 + projection*640 + col`, scales/output `row*640+col`, tous les
indices, conversions et produits `checked_*`. Covers aux extrêmes, aux
tailles physiques B1/B2/B4/B8 et aux refus ; conserver les assertions
d'unwinding. Réutiliser le vrai garde appelé par l'API, pas une copie
arithmétique écrite uniquement pour obtenir une preuve. Le harnais existant
`cpu_slot_validation_preserves_bounds_for_real_route_widths` borne ses IDs
à **20**, donc ne qualifie pas à lui seul le nouveau domaine 80.

Conformément au skill formel et à sa référence Rust relus, tenter aussi une
entrée MLX réelle ciblée, sans stub ; les ICE `intrinsics.rs:243` déjà
observés pour T17/slots restent des limites à rapporter, pas des succès.
Kani CPU ne prouve pas les arrondis Metal, H128, MLX, les lectures SSD ou la
concurrence ; CBMC n'a pas accepté le vrai MSL lors du précédent essai.
Les preuves/contrôles devront rester séquentiels et séparés du GPU. Aucun
nouveau résultat de preuve n'est revendiqué par cette revue.
