# Inspection CI exacte du commit 09113b6f — 23 octobre 2026

Verdict : **échec de CI par arrêt du runner, aucune obligation Kani échouée observée et aucune preuve complète du job global**. La cause de l'arrêt reste inconnue. Aucun rerun, build, Kani local ou travail GPU n'a été lancé par cette revue.

## Identité et statuts

- Commit : `09113b6f2accdd008e8c39d457b45c9a4d95002a`.
- [Workflow Native Rust checks, run39823981330](https://github.com/0xZKnw/mlxl3/actions/runs/39823981330), événement push, tentative1 : completed/failure.
- [Global Kani, job119982266545](https://github.com/0xZKnw/mlxl3/actions/runs/39823981330/job/119982266545) : completed/failure. L'étape Kani est cancelled, post-checkout skipped.
- Flash Kani119982266166 : success ; macOS119982266494 : success ; Linux119982266595 : success.
- Le job macOS a bien passé « Compile and lint the shipped MLX bridge ». Linux ignore naturellement les contrôles SSD/macOS et MLX propres à Apple ; ce n'est pas une validation GPU physique.
- Ce constat porte sur le SHA publié ci-dessus, pas sur les sources72/73 locales non publiées.

Les réponses exactes sont conservées dans `run.json` et `jobs.json`. Le log reçu via le connecteur, décodé UTF8 avec BOM/ANSI d'origine, est conservé dans `job-119982266545-decoded-raw.log` (92877 caractères) ; `...-clean.log` retire uniquement les séquences ANSI et le BOM. L'octet de nouvelle ligne ajouté par l'outil de patch a été retiré pour retrouver le payload décodé original. Ce n'est pas une archive ZIP HTTP originale.

## Ce qui a échoué précisément

Commande du log et du workflow :
```text
cargo kani --lib --no-default-features --jobs 1 --output-format terse
```
Kani0.68.0 ; runner Ubuntu22.04.5, image20261015.610, agent2.338.0.

| Événement UTC22octobre | Constat |
| --- | --- |
|17:35:05 | Début du job |
|17:35:20 | Début de l'étape Kani |
|17:36:27.7347221 | Premier harnais observé |
|17:38:39.2381732 |31e résultat SUCCESSFUL |
|17:38:40.7375301 | Début de cache_allocation_is_optimal_and_exact_on_two_layers_two_levels |
|17:41:12.4288869 | Le runner annonce avoir reçu un shutdown signal |
|17:41:13.0113944 | L'opération est annulée |
|17:41:14 | Job terminé en failure |

32harnais ont commencé,31 ont un résultat SUCCESSFUL. Aucun résumé « N of M failed » avecN>0, aucun résultat VERIFICATION FAILED et aucun contre-exemple dans le log. Le dernier harnais n'a **aucun verdict**. Il a tourné environ151.69s avant le signal. Le job complet dure369s ; l'action de preuve rapporte331342ms.

**Pas de ligne exit143 dans ce log091.** Les exit143 historiques des commits82/2945 sont des observations distinctes à conserver, sans les attribuer à ce run.

Le message explicite d'arrêt du runner ne permet pas de départager manque de mémoire, arrêt d'infrastructure ou annulation externe. Une cause OOM est plausible vu la répétition au même harnais, mais non démontrée : aucune mesure RSS/pression mémoire ni diagnostic OOM n'est exporté ici. Les autres jobs de ce même run continuent après cet arrêt, dont FlashKani jusqu'à17:50:47 ; aucun indice d'annulation du workflow entier par un nouveau push dans ces preuves.

## Limites et lignes exactes du workflow

Le fichier exact au commit est archivé : `workflow-at-09113b6f.yml`.

- Lignes3–5 : push et pull_request.
- Lignes7–9 : groupe de concurrence partagé par workflow/branche ; cancel-in-progress:true.
- Ligne17 : **FlashKani30minutes =1800s**.
- Lignes63–73 : jobglobal Kani.
- Ligne65 : **global60minutes =3600s**.
- Ligne73 : arguments de Kani ; **aucun --harness-timeout**.
- Ligne81 : natif20minutes.

Le shutdown à369s ne correspond donc ni à la limite globale3600s ni à la limite Flash1800s. Aucun timeout de harnais explicite n'est configuré. Les lignes du workflow local en cours diffèrent de+3 après l'ajout des trois nouveaux harnais72 : global commence66, timeout68, args76. Ne pas citer ces lignes locales comme celles du SHA091.

Le dernier harnais est dans `native/src/flash_cache_profile.rs:373`, inchangé depuis `b2985386b8dcc23eb3c54c4af013085b55576650` du20octobre. Il appelle l'implémentation réelle allocate_levels avec quatre coûtsu32 symboliques, deux couches/deux niveaux, budgetusize symbolique et unwind5. Les budgetsinvalides sont contrôlés, budgets0/1/2 et usize::MAX ont des couvertures. Il n'appartient pas au nouveau calcul/prélecture72/73.

## Besoin CI minimal, sans masquer l'échec

1. Conserver les jobs FlashKani et natifs et leurs résultats distincts. Les nouveaux harnais72 y sont déjà ajoutés localement par l'agent principal ; leur réussite locale ne valide pas encore un run distant.
2. Pour diagnostiquer le global récurrent, isoler éventuellement le harnais d'allocation coûteux dans un job propre, sans le supprimer de la couverture globale : même implémentation, mêmes quatre coûtsu32 et budgets complets. Ajouter une sortie JSON et une observation mémoire légère persistée, avec timeout explicite. Cela rendrait un prochain arrêt attribuable ; **un timeout1800s ne corrigerait pas à lui seul un shutdown survenu à152s de preuve**.
3. Si une restructuration de harnais est nécessaire pour réduire le coût symbolique, partitionner les cas budget0/1/2 et invalides en conservant leur union complète. Ne pas borner artificiellement les coûts, introduire un stub ou ajouter continue-on-error pour obtenir du vert.
4. Aucun changement CI n'est indispensable à la correction fonctionnelle des nouveaux flags, puisque le défaut global préexiste ; sa couverture demeure toutefois non vérifiée et la CI totale ne doit pas être annoncée verte. Un rerun inchangé ne prouverait rien avant inspection de son résultat.

Ces propositions sont un diagnostic, pas une modification implémentée ou un succès de CI futur.

## Accès et commandes de consultation

Connecteur GitHub, lecture seulement :
- fetch_workflow_job_logs(repo_full_name=0xZKnw/mlxl3,job_id=119982266545)
- fetch_workflow_job_steps(...) même job
- fetch_workflow_run_jobs(...,run_id=39823981330)
- GET /repos/0xZKnw/mlxl3/actions/runs/39823981330
- GET /repos/0xZKnw/mlxl3/actions/runs/39823981330/jobs?filter=latest&per_page=100
- GET blob09113b6f/.github/workflows/rust.yml

Le connecteur générique refuse les endpoints directs actions/jobs/119982266545 et check-runs/119982266545 comme non pris en charge. Les wrappers de job et la collection des jobs ont fourni les données nécessaires. Aucune annotation supplémentaire n'a donc été inspectée et aucune escalade réseau n'a été demandée.

Lectures locales : sed/nl du workflow et du harnais ; git log -1 sur flash_cache_profile.rs. Aucune édition du worktree.
