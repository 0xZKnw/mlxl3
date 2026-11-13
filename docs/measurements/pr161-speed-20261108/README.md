# PR161 : mesures et correctifs locaux

La composition finale et la nouvelle comparaison avec TurboEXL3/Regolith sont dans [le rapport final](../../issues/reviews/pr161-fixes-2026-11-08.md). Dernier diagnostic MAIN64 chaud, 53 slots/couche : R1 7,832 / final **8,311** / R1 7,627 / AR 7,671 tok/s, sorties identiques. Signal +6,11 à +8,98 % contre R1, sans gain stable établi sur AR. 359 tests release passent / 129 ignorés ; 6 invocations GPU sur la composition finale et bridge AR/MTP8 passent. Quatre harnais CPU bornés Kani passent, sans preuve Metal ou concurrence. Aucun push ni application installée modifiée.

[Patch final code + inventaire](fixall-final-delivery.patch), [identités et vérification](fixall-final-verification.json), [plan final](fixall-r3-speed-plan.json), [résultats bruts](fixall-r3-speed-summary.json), [analyse et limites](fixall-r3-speed-analysis.json), [sources TurboEXL3](turbo-upstream-r3/manifest.json), [nouveaux audits Regolith](regolith-r3-sources.json). Les fichiers `fixall-*` conservent aussi les étapes R1/R2, preuves, échecs et logs physiques. Leurs exécutables figés restent dans le répertoire `build/pr161-speed-20261108` du worktree indiqué dans les manifestes, hors de cette archive documentaire.

## Diagnostic historique interrompu

Source runtime : `ebacd5ecdec128c249607eac264c28a4ba836ec2`. M5/24 Gio, EXL3 moyenne 2,05 bpw, contexte4096,128 sorties/127 decode,90 slots/couche fixes. Campagne interrompue sur demande :14 mesures MAIN conservees, deux rotations completes et quatre bras de la troisieme ; aucun HOLD128. Tous les tokens sauvegardes concordent avec AR. Aucun swapin/out systeme supplementaire observe.

La comparaison principale ci-dessous utilise les deux rotations completes : deux points par mode, pas une campagne statistique achevee.

| Mode | Decode median (tok/s) | Plage (tok/s) |
|---|---:|---:|
| ar | 7.571 | 7.350 - 7.793 |
| d1 | 7.121 | 7.079 - 7.162 |
| b2 | 6.062 | 6.039 - 6.086 |
| b2_ssd | 5.826 | 5.806 - 5.847 |
| b2_ssd_shared | 6.011 | 5.820 - 6.201 |

Limite de protocole decouverte a la revue : `clear()` vide aussi les experts du draft, puis seul AR rechauffe la cible. Les tenseurs fixes du draft restent charges, ses experts sont froids au debut de chaque mesure. Sur le premier tour, les quatre bras MTP allouent247,267Mo pendant le decode ; AR reste stable. Le champ historique `draft_resident_for_all_arms=true` ne qualifie donc pas un draft entierement resident. Ces chiffres sont un diagnostic avec experts du draft froids, pas une preuve du debit a chaud.

La reference AR a les options incompatibles avec les cohortes desactivees ; ce n est pas le meilleur AR livre. Les octets reader ne sont pas des octets SSD physiques. Les maxima memoire sont echantillonnes aux frontieres des tokens. Le champ runtime_base830bb38a du harnais est historique ; les empreintes des binaires et le manifeste source identifient le code effectivement execute.

Les14points, cycles et options sont preserves dans les JSON voisins. La qualification avant mesure comprend354tests bibliotheque passes/127ignores,7invocations physiques reussies apres correction de preparation du test SSD (echec initial conserve dans le build), bridge reel et20smokes8tokens. Le contrat CPU Kani de reutilisation passe1assertion/5couvertures ; aucun resultat Kani ne prouve Metal ou la concurrence.

Audit et correctif local : [analyse Regolith](../../issues/reviews/pr161-regolith-audit-2026-11-08.md). Aucun push, merge, paquet ou application installee modifie.

## Correctif HC/routeur et diagnostic court

Le correctif local restaure le producteur HC/routeur deja present en AR. Le test de regression echoue sur l'ancien runtime puis passe sur le nouveau :96soumissions de routeur separees evitees par bloc B2 sur48couches quand la fusion est eligible. La validation finale passe354tests bibliotheque/127ignores,6tests inventaire/2sous-tests et5invocations physiques. Kani ne peut pas exposer cette fixture GPU : aucune preuve formelle du raccordement Metal.

Le bridge de production A1/B/A2,32sorties/31decode, cache explicite3Gio/53slots, avec un vrai warmup MTP32 par processus, donne7,016/7,444/7,212tok/s. Memes tokens,15/15propositions acceptees dans les trois cas, processus/readers joints. Un seul point corrige, controles en derive et compression differente : signal court encourageant, gain stable non etabli. Aucun gain TTFT/total demontre. Ce protocole est distinct de la campagne128tokens/90slots avec draft froid.

[Plan court](hcroute-speed-plan.json), [resultats](hcroute-speed-summary.json), [validation](hcroute-validation.json), [patch complet](hcroute-delivery.patch), [statuts GitHub de la tete publiee](ci-head.json). Le patch local n'a pas de run GitHub ; les echecs existants de la tete publiee restent visibles.

## Verification de la revue jointe

Le contre-exemple LFU11slots est reproduit avec le vrai planner et les vraies reservations/publications :15lectures initiales dans les deux cas, puis0miss scalaire contre4miss cohortes sur la route suivante. Quatre decalages IDs et controles15/20/53/90slots executes dans1test passe/74autres filtres. Ce scenario ne se produit pas dans ces controles de plus grande capacite et ne doit pas expliquer artificiellement les campagnes53/90slots. Aucun nouveau code de politique de cache ni benchmark GPU. Kani symbolique0..255/unwind18 tente : timeout45s, aucun harness verifie.

[Probe CPU](attachment-cache-repro.rs), [resultat](attachment-cache-test.log), [commandes](attachment-cache-execution.json), [identites](attachment-cache-source-identity.json), [limite Kani](attachment-cache-kani.json). Le rapport principal ajoute aussi la tete vocabulaire par ligne et les conditions du volume QSA.
