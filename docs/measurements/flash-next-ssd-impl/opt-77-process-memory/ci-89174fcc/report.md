# CI77 exacte — arrêt du runner Kani global

Inspection distante en lecture seule du commit **89174fccc37af8fa6cc37f353ee95d14b584dee7**, runNative **39958506522**, tentative1. Événement`push` sur`codex/flash-next-ssd-205`, actor et triggering_actor`0xZKnw`. Ces champs identifient le déclenchement du run, **pas l'auteur éventuel de son interruption**. Aucun rerun, annulation, changementCI, preuve/build/test local ou GPU lancé ; W/opti et les sources restent intacts.

## Statut observé

| Job | Identifiant | État au snapshot archivé |
| --- | ---: | --- |
|native ubuntu-latest|120430259174|completed/success|
|native macos-26|120430259562|completed/success|
|kani-flash-contracts|120430259523|in_progress, conclusionnull|
|kani global|120430259721|completed/failure ; étape de preuve cancelled|

Le workflowNative est encore`in_progress` au snapshot. **La CI n'est pas entièrement verte** : deux jobsnatifs réussis, un jobFlash encore en cours et le global échoué/incomplet. Aucun résultat du jobFlash ultérieur n'est prédit ici.

## Log réel et comparaison75

Le log a été récupéré directement via l'endpoint`GET /repos/0xZKnw/mlxl3/actions/jobs/120430259721/logs`, qui fonctionne pour le job terminé alors que le run est encore actif. Le premier appelCLI a refusé d'émettre les séquencesANSI ; le second a conservé le texte brut en fichier avec`gh api --allow-escape-sequences`. Ce refus local de lecture n'est pas une erreur du jobCI. Le parsing retire seulementANSI en mémoire ; le fichier brut reste inchangé.

Sur77, les31premiers harnais ont chacun un verdict`SUCCESSFUL`. Le32e démarre à **13:10:19.7095853UTC**, sans verdict final :

```
flash_cache_profile::verification::cache_allocation_is_optimal_and_exact_on_two_layers_two_levels
```

À **13:12:17.3805330UTC**, le runner signale explicitement son arrêt. À13:12:17.6588334UTC, le log donne une annulation de l'opération. L'API décrit le job`failure` et l'étapeKani`cancelled` ; ces deux états sont conservés tels quels. La seule annotationfailure répète l'annulation et ne précise pas de cause.

| Observation |75, SHA250488c/job120410093061|77, SHA89174fcc/job120430259721|
| --- | --- | --- |
|Harnais démarrés|32|32|
|VerdictsSUCCESSFUL terminés|31|31|
|Séquence de noms|Même ordre que77|Identique à75|
|Dernier harnais sans verdict|cache_allocation…two_layers_two_levels|Même harnais|
|Temps dernier harnais→shutdown|82,568793s|117,670948s|
|Shutdown explicite|Oui|Oui|
|Exit143 explicitement écrit|**Oui**|**Non : aucune ligne de code de sortie**|
|VERIFICATIONFAILED observé avant arrêt|Aucun|Aucun|

Il serait donc incorrect de recopier le code143 de75 dans le bilan77. La signature commune est **un arrêt de runner au même harnais non terminé**, pas un contre-exemple formel. Aucun verdict n'est disponible pour ce32e harnais ni pour les suivants. Les31succès ne prouvent pas le package entier.

## Ce que les métadonnées permettent de dire

Sourceworkflow du **SHA exact** archivée dans`workflow-89174fcc.yml` : jobglobal ubuntu-22.04, Kani0.68.0, commande`cargo kani --lib --no-default-features --jobs 1 --output-format terse`, limite déclarée60minutes. Le job entier dure5min40s, ce qui n'atteint pas cette limite configurée. Le log indique runner2.338.0, imageubuntu22.04.5/20261012.322.1, actionKani résolue`69d357bade1eb7b32bc3fc3d5c3d173c5bfa5237`. Le warningNode20/checkout forcéNode24 est présent, sans preuve qu'il ait provoqué l'arrêt.

Les messages observés **ne prouvent niOOM, ni timeout d'un solveur, ni annulationhumaine, ni régression du nouveau code77**. La répétition au même endroit rend ce harnais pertinent pour un diagnosticCI ultérieur, mais ne suffit pas à attribuer le shutdown. Aucun accès aux journaux système du runner ou métriques de sa mémoire n'est fourni dans ces éléments. Ne pas remplacer cette lacune par une hypothèse affirmée.

## Preuves et limites

- `run.json`, `jobs.json`, `global-kani-job.json` : provenance du run et statuts exacts.
- `global-kani.log` : log distant brut, SHA256`b445fd33ca5ce4cc0c4d2a7232c4031d963d2ffa41f64f8a21901264c0385214`.
- `global-kani-check-run.json`, `global-kani-annotations.json` : failure/annulation, aucune cause plus précise.
- `workflow-content.json`, `workflow-89174fcc.yml` : blobGitHub`98031fb381183070edde3c96ba8648fad8a06d89`, workflow exact examiné.
- `analysis.json` : comptage des32démarrages/31verdicts, horodatages, comparaison des séquences et hashes des logs75/77. L'ancienlog75 est conservé dansROOT/build/flash-opt-75-ci ; aucune réexécution de cette CI.

Sources distantes : [run77](https://github.com/0xZKnw/mlxl3/actions/runs/39958506522), [jobKani77](https://github.com/0xZKnw/mlxl3/actions/runs/39958506522/job/120430259721), [ancienjob75](https://github.com/0xZKnw/mlxl3/actions/runs/39952482884/job/120410093061), [workflow exact](https://github.com/0xZKnw/mlxl3/blob/89174fccc37af8fa6cc37f353ee95d14b584dee7/.github/workflows/rust.yml).

**Note à reprendre par le parent dansopti/final :** CI77partielle : Linux/macOS réussis ; Kani global interrompu après31succès au même32e harnais que75, shutdown explicite mais cause inconnue et aucun exit143 dans ce log77. Flashencours au snapshot ; aucun run entièrement vert ni preuveglobale revendiqué. Aucune relance/annulation effectuée.

## Contrôles77 réellement constatés dans les logs natifs

Les logs des jobsLinux/macOS ont ensuite été lus, pour dépasser le seul statut du workflow. Les deux checkouts impriment le SHAcomplet89174fccc37af8fa6cc37f353ee95d14b584dee7. L'API du commit indique son parent250488c et les seuls fichierssource77 annoncés (5natifs, analyseur/testPython etworkflow) ; aucun diff `ssd_moe`/`ssd_prefetch`/`flash_support`/`flash_draft`72 n'est inclus. Ces buildsCI concernent donc le commit77isolé, contrairement au binaire de mesure local69b919 issu d'un worktree mixte.

| Contrôle réel |Linux120430259174|macOS120430259562|
| --- | --- | --- |
|Ruff format/Ruff/analyseurpy_compile|Logs sans erreur, step14success|Logs sans erreur, step14success|
|Analyseur77B pytest|**220passés0,55s**,13:12:10UTC|**220passés0,60s**,13:12:18UTC|
|FixtureSDK C++ASan/UBSan|Explicitementskipped|Step16success, compilation puis exécution `build/ci/process-memory` sous bash-e|
|Build/lint livréMLX0.32.2 mlx,chat|Explicitementskipped|Step17success ; release terminé13:14:11UTC|
|DécodeurRustCPU77|Non exécutable dans le moduleMLXLinux|Nomexact `memory_stats_decode_keeps_zero_valid_and_preserves_all_fields` :**1passé/0échoué/0ignoré**,272filtrés|
|CABI null/capacitéCPU77|Même limitation|Nomexact `memory_stats_ffi_rejects_short_and_null_outputs_before_gpu_work` :**1passé/0échoué/0ignoré**,272filtrés|

Le testSDK est volontairement silencieux : son succèsGitHub après les deux commandes et la continuation vers l'étape suivante établissent leur sortie0 ; il n'imprime pas «32cas». Ce nombre vient des32combinaisons assertées par la fixture du commit exact, compilée sansNDEBUG sous ASan/UBSan. Aucun compteur32 fictif n'a été extrait du stdout. Les deux testsRust ne sont ni ignorés ni une sélection vide : leurs noms et leurs résultats1/1 sont présents à13:14:54.699UTC et13:15:10.063UTC. Le testRawDraft antérieur est lui aussi1/1, mais reste distinct de ces deux contrôles77.

Preuvesajoutées : `native-linux.log`, `native-macos.log`, `commit-files.json`, `native-validation.json`. L'avertissement futur `block0.1.6` persiste sans échecClippy/build. Cette validation distante des nouveaux steps n'efface pas l'arrêtKani global et ne remplace pas le composantApple physique exécuté localement auparavant. Aucun nouveau job, rerun ou test local n'a été demandé.
