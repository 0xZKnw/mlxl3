# K1PM — qualité et mesures après fusion, plans non exécutés

Préparation ROOT seulement, aucun M/opti/index modifié et aucun build/test/preuve/GPU lancé. Base main9c8474ca + patch K1 9782c74c ; candidat CLI cc9669ad/tests041b3b09, 100sources2b4eb648. H est **post-merge/runtime/baseline-cli d8757fa6**, sources natives fusionnées91SHA ; sa metadata embarquée e551-dirty n’est pas son identité de code. Aucun ancien binaire80M/K1 batterie ni driver W n’est repris.

Les cinq run.json sont des matrices stock indépendantes et manuelles, pas un workflow d’enchaînement. Les premières versions null, antérieures à réception du vrai gel, restent sous versions/planned-null-v1 ; elles n’ont pas été exécutées. L’identité remplie n’est pas un succès de qualité. Les champs allow_gpu_launch restent false, gates null jusqu’à inspection par le parent.

| Plan | Phases | Sélections / résultats attendus |
|---|---:|---|
| quality | 9 | pré20, 7tests physiques exacts, post30 |
| h128-micro | 3 | pré20, H128704samples, post30 |
| activation-micro | 3 | pré20, activation192samples, post30 |
| screen | 6 | pré20, MAIN H-C puis HOLD H-C, post30 |
| confirm | 6 | pré20, MAIN C-H puis HOLD C-H, post30 |

## Gel et préflight avant lancement

Le superviseur stock ne fait pas respecter les champs SHA additionnels. `identity-frozen-quality-pending.json` et `freeze-gate.txt` rendent ce prérequis explicite : hash des trois binaires, source candidate100fichiers réellement active M, provenance baseline91, pilotes M, fixture50bf33e3 et cinq plans ; un --list --ignored borne30s devra confirmer que chaque sélecteur choisi existe exactement une fois. Aucune invocation --list n’a été faite par cette préparation. Les selectors ont seulement été retrouvés une fois dans leurs sources et sont détaillés dans selector-evidence.json, avec les marqueurs réels/templatisés.

Le parent inspecte fmt/Clippy CPU+MLX/suites/Kani CPU et la tentative réelle MLX, puis renseigne les seules gates effectivement satisfaites. L’ICE historique MLX n’est pas une preuve ; reviewed signifie que la limite est explicitement examinée, pas qu’une preuve GPU a réussi. Tous builds/proveurs et autres modèles doivent être joints, AC observé. L’environnement est fixé avant Popen et tous overrides anciens ainsi que les quatre options EXL3 de main sont unset de façon commune H/C, pour retrouver les mêmes defaults main ; aucune option K1 ajoutée.

Exemple préparé, **non exécuté**, à lancer depuis la racine M en errexit une fois les gates inscrites :

```sh
set -euo pipefail
cd /Users/justin/.codex/worktrees/v2-merge/mix-stq1_0
/Users/justin/Documents/mix-stq1_0/.venv/bin/python3.12 /Users/justin/Documents/mix-stq1_0/build/flash-opt-82/K1/post-merge/plans/freeze-gate.txt /Users/justin/Documents/mix-stq1_0/build/flash-opt-82/K1/post-merge/plans/quality/run.json
# Seulement après K1PM_FROZEN_PREFLIGHT_OK, la commande observer exacte est
# dans launch-commands.json pour ce même plan. Ne pas lancer automatiquement le suivant.
```

Le gate est une copie textuelle ciblée du préflight existant, non un nouveau superviseur ; il est préparé/non exécuté. Il n’ajoute aucune preuve de concurrence, alimentation ou mémoire. Conserver sa sortie, la commande exacte et la liste des tests au gel réellement utilisé. L’assertion des sources doit précéder le lancement même si le binaire a le bon SHA.

## Qualité minimale touchée

H128 `linear::hot_transform::tests::hot_h128_matches_native_half_roundings_masks_and_layouts` :3144cœur/240bords/10replis, Hot bit exact/Cold+0, vrais arrondisHalf, slots/répétitions/strides. Activation :960cas,2nonfinis intentionnels,1IDlimite ; test distinct28replis/10erreurs. Vrais slots EXL3 :528cas masqués et176historiques, marqueurs exacts conformes à la source main fusionnée.36préfixes DFlash logits/états byteexacts ; tête MTP cache-only/référence indépendante ; session MTP budgets1/2/3/17 et états.

Il s’agit de sept sélections libtest, pas d’autant de milliers de tests. Exiger running1, nom exact,1passé/0failed, marqueurs et enfant joint ; la tête MTP ne possède pas de marqueur de succès spécifique et repose sur ses assertions internes/bilan libtest. Les assertions d’égalité et finitude restent dans les vrais tests. Ne pas confondre le JSONcompact du préfixe avec un println de chaîne échappée. Aucune suite complète redondante, HC/LFM ou profil général n’est ajoutée à ce lot minimal ; le parent possède déjà les contrôles CPU et main fusionnés. Les six mutantsMSL/restauration et bridge43 K1 restent requis avant promotion, en dehors de ces cinq plans.

## Deux micros distincts

Même fixture50bf33e3 figée :8masques B2 réels/synthétiques,20routes/cap53. `MLXL3_HOT_TRANSFORM_FIXTURES` pointe sur K1/component-fixtures.json. Chaque `MLXL3_HOT_TRANSFORM_OUTPUT` pointe sur le component.json neuf de son plan.

H128 : selector exact `linear::hot_transform::tests::hot_h128_component_aa_abba_on_pinned_b2_masks`,8fixtures×4géométries×22samples=704, paire A/A puis5ABBA ; huit warmups/bras,32appels chaînés/sample, setup horsclock/frontière finale unique. Parité32chaînes finies avanttemps. Rapport `complete_component_wall_only`,704samples/32setup/8plans, champ parity descriptif **chaîne**, pas booléen. Le rapport est écrit seulement à la fin : absence de fichier sur échec ne devient pas succès.

Activation : selector exact `ssd_moe::hot_activation::bench::masked_hot_activation_component_aa_abba`,8fixtures×24samples=192, quatre A/A puis5ABBA ; huit warmups d’un appel/bras et32appels/31récurrences communes parsample. Oracle réel Array::swiglu et contrôle de toute la chaîne avantclock/aprèschaquesample. Rapport `complete_component_wall_not_throughput`, parity=true seulement après192samples ; incomplete/failed restent des échecs. Les GU sont synthétiques QA/asymétriques finis avecCold0, pas des captures d’activations du décodeur.

Les durées incluent graphe hôte/dispatch/complétion/récurrences et ne sont ni GPU exclusif, ni tok/s, ni des gains additionnables. Donation du H128 natif et allocation custom peuvent modifier le résultat ; mesurer au lieu de supposer un gain. Aucun micro n’est un oracle de débit complet.

## DFlash2 B2 réel HC puis CH

Chaque bras lance un nouveau bridge et une génération warmup128 puis une mesure128sorties/127decode, --repeats1/--tokens128/--cache-gib3/--dflash-mode1, WIDTH1/ADAPTIVE0, Q4ring, cache3Gio/53slots/alloc64MiB/4lecteurs/HotWide+UNIQUE+63, prélecture/LES/shared/stable0. Seul le binaire diffère H/C. Mêmes prompts MAIN/HOLD et référence gold60 ; quatre rapports par plan, huit réponses warm/mesure. Deux plans préparent donc huit rapports/seize réponses, rien n’est encore mesuré.

Les chemins de sortie parbras n’existent pas lors de la préparation ; aucun écrasement n’est permis. Au retour : statut report validated,128/127 exacts, texte ET hashgold60, SHA CLI exact, compteur MTP inactif null, DFlash actif B2/mode1/proposition1, acceptation/histogrammes/octetsSSD cohérents entrebras. Comparer séparément warm/mesure ; inspecter dérives et ordres HC/CH. Les durées draft/vérification/commit sont inclusives, pas une somme GPU exclusive. OFF/MTP frais seulement si signal DFlash à promouvoir selon journal parent.

## Supervision et mémoire

CwdM nécessaire : observer77 lit les fichiers stock49/45/43 relativement à la racine courante. Aucun wrapperbatterie ni nouveau superviseur n’est copié. Pré20/post30, puissance AC stable, un seul modèle à la fois, délais/reaping/lecteurs/samplers inchangés. Inspecter run-result, logs, system-samples,vm-curve et les outputs avant de poursuivre. Les matrices ne s’enchaînent jamais automatiquement.

Garde stock256MiB de swap système/phase et RSS10GiB du superviseur conservés ; cela ne qualifie pas zéro sortieswap. Cible mémoire processus≤12Go, compression/ledger privé/VM système distincts ; ledger0 ne prouve pas zéro swap-out privé. Caps compression77B historiquement FAILED restent FAILED. Toute origine non attribuée reste inconnue. Aucun résultat, gain, app installée ou CI verte ne découle des plans préparés.
