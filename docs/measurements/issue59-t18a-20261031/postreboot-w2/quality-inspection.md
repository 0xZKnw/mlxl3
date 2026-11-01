# T18a w2 — qualité post-redémarrage, 31 octobre 2026

**Trois oracles physiques sélectionnés passent, sans nouveau Swapout/Swapin ; qualité ciblée confirmée avant les pilotes w2.** Aucun débit ou gain qualifié par cette inspection. [Données/hashes](quality-inspection.json), [plan](quality-plan.json), [reçus](quality-plan-result.json).

Les **541 sources/dépendances** de [l'identité actuelle](identity.json) ont été relues par SHA streaming : zéro écart/manquant. Les trois binaires concordent : CLI `1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436`, libtest `de3616e3d2e0c3d9507e075cf9af4dfe8b3290f0b291ac704b3e1df537f8d910`, Python `30b2a6bf9706029b609327ca5db722cd458239df96efc87e1f8e37beed49db2e`. Les tests utilisent ce libtest, pas celui produit par une autre suite.

Chaque commande est le sélecteur exact avec `--exact --ignored --nocapture --test-threads=1`, imprime son nom/`running 1 test` puis **1 passé/0 échec/0 ignoré**. Tous les processus ont exit0 et sont joints ; tests filtrés conservés dans le JSON, pas comptés exécutés. Dynamic1, T04/T17/T05=0, cache3 Gio et quatre lecteurs sont contrôlés.

| Oracle | Marqueurs numériques inspectés |
| --- | --- |
| `flash_matches_independent_reference_states_and_logits` | s0..s4 : **248320 logits chacun**, RMSE/maxabs affichés0. Dix entrées imprimées `gdn_state_0/1`, chacune786432 valeurs. L'entrée réelle refuse comptes vides/incohérents et non-finitude, impose logits/états récurrents comparés byte-exact, routes et argmax ; pas de dénombrement inventé de tous les états48 couches. |
| `flash_mtp_session_preserves_greedy_sequence_target_and_head_states` | Budgets1/2/3/17 ; proposés/acceptés/blocs : **0/0/0, 1/0/1, 2/0/2, 12/5/12**, identiques au reçu w1. Séquence greedy et états target/head face au sériel, pending, invalides, acceptation et rejet restent le contrat du test. |
| `flash_draft_captures_and_all_prefix_commits_match_serial` | `FLASH_DRAFT_PREFIX` passé : **36 préfixes**, largeurs1..8, logits/états target byte-exact, captures64000 valeurs, `byte_equal=true`/RMSE0. La tolérance indépendante0,004 des captures reste distincte du contrat byte-exact cible ; rollback, abandon, invalides et continuation exercés par le vrai test. |

[Admission r5](../postreboot-admission-r5.json) passée : trois instantanés à **10620764160/10643554304/10670194688 B**, secteur/quiet et compteurs swap0. L'amendement prospectif `W2-ADMISSION105` fixe **10,5 Go** pour les trois qualités **et les huit pilotes**, contrôlé dans chaque plan ; les plans10,8 Go et refus antérieurs restent archivés dans [postreboot-w2-admission108-original](../postreboot-w2-admission108-original/). Aucune ancienne mesure w2 à10,8 Go n'existe à mélanger.

[Nouvelle ancre w2](vm-anchor.json) : Swapouts/Swapins0, compteurs de reboot distincts de w1. Les trois reçus déclarent0 delta et0 maximum observé. Le footer DFlash donne footprint**7340447864 B**, pic vie**7375132816 B**, sous12 Go ; pas de capteur natif continu universel revendiqué pour les trois tests.

Les durées supervisées6,995/10,597/27,634s ne sont pas des mesures de gain. Les huit pilotes MTP/DF ACCA restent une phase séparée, exclusivement w2 ; l'AR et le MTP incomplet w1 ne leur sont pas agrégés. Les limites de preuve Kani MLX/CBMC/CrossHair antérieures persistent. Aucun test/build/proof/GPU ou moteur lancé par cet audit ; seuls ces documents JSON/Markdown sont écrits.
