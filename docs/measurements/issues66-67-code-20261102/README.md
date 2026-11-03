# Issues 66/67 : candidats codés, tests différés — 2 novembre 2026

Demande : « code deja tout ça, les tests on les fera plus tard ». Implémentation
locale terminée ; candidats expérimentaux non activés. Correction numérique,
preuves, erreurs physiques et gain **non vérifiés / non mesurés**. La demande
suivante autorise le push en commits distincts, y compris les travaux du
1er novembre ; cela ne constitue pas une activation ou une release.

## Options indépendantes

| Candidat | Option | Défaut | Portée |
|---|---|---|---|
| A, issue66 | `MLXL3_SSD_COLD_PRIORITY=0|1` | 0 | Trie stablement les lectures par multiplicité des occurrences, uniquement B2 et ColdReady admissible ; seuil ready2 conservé |
| B, issue67 | `MLXL3_SSD_COLD_READY_ROWS=2|4` | 2 | Autorise B3/B4 dans le chemin ColdReady existant ; autres gardes/fallbacks et protections conservés |
| C, issue67 | `MLXL3_FLASH_REFERENCE_MB34=0|1` | 0 | Partage dense MB3/MB4 uniquement pour qkv GDN cible2560→10240, couches canoniques0..47 ; B1/B2/autres largeurs et rôles gardent leur géométrie |

Le flag C utilise toujours le shader canonique, avec SG4/NT1 et réductions
de référence. La mémoire threadgroup déclarée atteint18Kio àMB4 avant
optimisation du compilateur. Aucun kernel z/out porté avant mesure qkv.
`MLXL3_FLASH_MULTI_TOKEN=1` est nécessaire pour atteindre ce dispatch depuis
le vérificateur. Hot/Wide/Unique et les options réellement actives doivent
être enregistrées lors des mesures : les défauts produit diffèrent du profil
tuned. Les trois nouvelles options refusent leurs valeurs invalides au
chargement et ne sont pas activées par ce travail.

Le tri de A porte sur les jobs, pas sur l'ordre de réduction des occurrences.
La préparation Cold B3/B4 vérifie le domaine `rows*10` et conserve le drain des
lecteurs/consommateurs sur erreur. Les traces natives mémorisent les occurrences
avant partition, l'ordre réel des jobs et les multiplicités. L'analyseur expose
`disk_batches[].cold_schedule` : doublons, occurrences débloquées par les deux
premières lectures achevées, queue I/O jusqu'à la dernière. Il refuse les
nouvelles métadonnées partielles/incohérentes ; les anciennes traces retournent
`unavailable_legacy`. Ces durées ne mesurent pas le GPU idle.

## Tests et preuves préparés, non exécutés

- CPU : parsing strict du domaine2/4 ; ordre stable/identité des jobs ;
  81 motifs de présence B2 et16 masques de disponibilité ; portée et géométrie
  MB34, limites et chemins de repli.
- Quatre harnais Kani CPU enregistrés dans les obligations CI :
  `cold_ready::two_job_priority_keeps_records_and_stable_order` (deux jobs,
  quatre slots symboliquesu32),
  `cold_ready::cold_row_limit_has_only_qualified_and_opt_in_values`
  (sélection symbolique parmi huit formes de valeur, pas tous les UTF8),
  `contracts::flash_reference_mb34_preserves_coverage_and_fallbacks`
  (rowsi32 et enabledbool),
  `contracts::flash_reference_mb34_shape_scope_rejects_other_dimensions`
  (dimensionsi32, deux préfixes fixes).
- Cold physique : oracles indépendants mixtes/Hot/Cold, B3/B4, doublons,
  slots réutilisés/cohortes, erreurs et reprise, contrôle du vrai ordre des jobs.
- EXL3 physique :24cas qkv F16/F32, entrées contiguës/stridées,
  comparaisons MB1/MB2/candidat, formes invalides et sélection réelle MB3/4 ;
  contrôles des rôles exclus et refus du flag au loader.
- Parcours modèle : test ignoré
  `qwen4::draft::runtime::tests::wide_cold_runtime_preserves_serial_tokens_and_states`
  avec largeurs fixes1/2/3 et budgets1/2/3/4/8/17 ; réutilise l'oracle série et
  la comparaison de tous les états. Le nom qualifié exact sera confirmé lors
  de l'inventaire de tests, pas supposé comme preuve de sélection.
- Analyseur :6fonctions/25cas préparés dans
  `tests/test_cold_schedule_diagnostic.py`, compatibilité legacy et18corruptions.

La CI PR/pushmain conserve les suites CPU/macOS et l'inventaire Kani, avec
le nouveau test Python ajouté explicitement. Aucun pushmain ou PR créé pour
déclencher ces tests dans cette préparation ; aucun run CI vert annoncé.

## Contrôles réellement exécutés

Après gel des agents :

```
MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx cargo check --locked --features mlx,chat --all-targets --jobs 1
MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx cargo clippy --locked --features mlx,chat --all-targets --jobs 1 -- -D warnings
cargo fmt --check
.venv/bin/ruff check scripts/profile_flash.py tests/test_cold_schedule_diagnostic.py
```

Compilation tous targets11,84s et Clippy strict6,77s réussis ; format et Ruff
réussis. Avertissement de dépendance préexistant : `block0.1.6` contient du
code qui sera refusé par une future version Rust. Un whitespace terminal dans
`test_flash_profile.py` a été corrigé pendant revue. Aucun test, Kani, CBMC,
CrossHair, génération modèle ou benchmark exécuté, conformément à la demande.
La compilation des tests ne signifie pas leur réussite.

Revue statique : loaders/callers, domaine des indices y compris cohortes,
leases/joins sur erreur, ordre de restauration, portée cible/head,
formats et fallbacks. La prochaine campagne devra qualifier la correction
avant vitesse et comparer chaque changement à largeur identique, puis leur
composition, avec l'ancre et les gardes explicitement autorisées.
