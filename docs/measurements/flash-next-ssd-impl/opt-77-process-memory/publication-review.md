# Publication77 — instrumentation mémoire et analyseur

23 octobre2026. **Revue statique seulement**, aucun staging/commit/push, source native modifiée, test, compilation, prouveur ou GPU lancé. HEAD de revue `250488c11f1cde97aba92b823ebbe4b06339f48d` ; travaux72/73/77 présents ensemble dans W. Les sources77 conservées ont été comparées aux identités des preuves existantes : toutes identiques. Détails machine : `publication-identity.json`.

## Titre et corps prêts pour le commit

Titre : `native: report attributed process memory pressure`

Corps :

> Expose les compteurs de compression, décompression et consommation logique de swap-in du processus moteur, avec PID/identité de démarrage et disponibilité explicite. Les snapshots ready/complete distinguent désormais une mesure valide à zéro d'un compteur indisponible.
>
> Ajoute un analyseur des rapports et ses contrôles d'identité, monotonie, budgets et erreurs. Les compteurs ne permettent pas d'attribuer les écritures de swap système au modèle ; aucun gain de RAM ni de débit n'est annoncé.
>
> Vérification locale :159testsRust passés/159ignorés,32casSDK C++ sous ASan/UBSan,220testsPython ; Kani0.68 sur le vrai décodeur127obligations+3couvertures. Composant physique1/1 et snapshots du bridge contrôlés. CBMC reste non vérifié (parseurSDK), CrossHair absent ; CI GitHub du nouveau commit à inspecter après publication. Les plafonds de compression77B dépassés restent échoués.

## Sélection exacte des sources77

| Fichier | Hunk / portée77 | Situation dans le diff courant |
| --- | --- | --- |
| `native/ffi/mlx_bridge.cpp` | include process_memory.h ; signature privée `mlxl3_memory_stats(out,count,reset_peak)` et vérification null/capacité avant reset/write ; lecture task_info self et encodage9champs | Tout le diff du fichier est77 |
| `native/ffi/process_memory.h` | Nouveau helperSDK réellement utilisé,9valeurs/mask ; capacitésRev0/5/6, refus des compteurs signés négatifs | Nouveau fichier77 entier |
| `native/src/array.rs` | DéclarationABI ; MemoryStats/from_raw/memory_stats ; deux testsCPU, un composant physique ignoré ; harnaisKani du vrai décodeur | Tout le diff courant est77,233lignes environ |
| `native/src/main.rs` | Seule ligne `ready.memory = memory_stats(false)` | Tout le diff courant est77 |
| `native/tests/process_memory.cpp` | Nouvelle fixtureSDK de l'encodeur,32combinaisons+zéro/max/négatif | Nouveau fichier77 entier |
| `scripts/analyze_flash_process_memory.py` | Analyse des snapshots existants, sans lancement de modèle ; sorties failed persistées, identité/compteurs stricts | Nouveau fichier77B entier |
| `tests/test_flash_process_memory.py` |220tests paramétrés, CLI réelle/refus écrasement/SIGTERM/SIGKILL/jonction | Nouveau fichier77B entier |
| `.github/workflows/rust.yml` | **Trois hunks77 seulement** : analyseurPython, C++SDK/macOS, deux testsRustCPU/macOS | Le premier hunk à trois harnaisKani est72 : l'exclure |
| `opti.md` | Entrées77R/77T/77B et leurs résultats/limites, plus éventuel résumé de publication77 | Ne pas publier tout son diff : il contient72/73/74/78 et données partagées |
| `docs/measurements/flash-next-ssd-impl/opt-77-process-memory/` | Identités, revuesSDK, logs/preuves, contrôles physiques et budgets77B échoués | Dossier77 entier acceptable ; copie des preuvesPython à compléter ci-dessous |

Ne pas inclure pour ce commit les diffs `flash_draft.rs`, `flash_support.rs`, `ssd_moe.rs`, `ssd_moe/prefetch_tests.rs`, `ssd_moe/shared_tests.rs`, `ssd_prefetch.rs`, les dossiers72/73/74 ou le prototype78 isolé. Les75 sont déjà publiés. Les artefacts65P préexistants (`opt-65-plan-diagnostics/`, `tests/test_flash_grouped_diagnostics.py`) restent exclus.

## Hunks CI à séparer

Le patch préparé `ROOT/build/flash-opt-77-process-memory/workflow-77-only.patch` part du workflowHEAD et contient les trois hunks77 seulement. C'est un fichier de préparation, **non appliqué et non staged**. Il évite d'embarquer les trois ajouts72 :

```
prediction_protection_is_bounded_by_available_work
unused_prefetch_priority_excludes_pending_and_preserves_free_slots
prediction_protection_preserves_the_explicit_cache_policy
```

Ces harnais72 appellent du code de `flash_support.rs`/`ssd_prefetch.rs` non publié par77. Prendre leur hunk seul casserait l'objectif du commit indépendant. Ils ne sont pas des preuves77.

CI77 conservée : setupPython3.12/pytest8.4.2/Ruff0.16.5 déjà présent ; le nouveau step exécute format/Ruff/py_compile/220tests de l'analyseur sur les deux OS. Le C++ est limité àmacos-26, compile le vrai headerSDK avec `-std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined`, puis exécute la fixture (assertions actives). Les deux testsRust ont chacun un nomexact, `--test-threads=1`, pipefail et contrôle explicite `1 passed;0 failed`; aucune sélection vide n'est acceptée. Le bridge est construit avec MLX0.32.2 épinglé.

**Limite CI :** le harnais77 `memory_stats_decoder_preserves_validity_and_values` est bien présent/vérifié localement dans array.rs, mais le jobKani hébergé sansfeaturesMLX ne compile pas ce module. Ne pas présenter cette preuve comme automatiquement rejouée en CI. Aucun nouveau jobKani macOS/SDK n'est ajouté ici pour résoudre artificiellement cette différence. Le composant nécessitant le runtimeApple reste ignoré et exécuté séparément sur le Mac réel. Le workflow courant n'a pas encore de runGitHub du nouveauSHA.

WorkflowSHA actuel `7a9e1c665d6d80d9e30735920fdba4e1dd1e2631ba527139dcb11cc8e0cbbbd8` ; le SHA `f4f1ff34...` de summary-cpu décrit la version native77 avant ajout du stepPython77B. Les hunks natifs77 n'ont pas changé ; leurs sources/propriétés restent identiques. Le workflow77-seul préparé a SHA `14388c6e88e925593b167cfda8fd304a70ebf8353ac06b1873b2d6bf971dd3e2`.

## Preuves couvrant les sources conservées

Les cinq SHA de `identity-v1.json` correspondent aux fichiersW actuels, sans modification depuis leur validation : bridge `d2a01595`, helper `77642e25`, array `6773c073`, main `4cf39e78`, fixtureCPP `7f54860f`. Les deux sourcesPython correspondent exactement à `analyzer-cpu/source-identity.json` : analyseur `2ec7a7f9`, tests `d9870957`. `publication-identity.json` conserve les valeurs complètes et la comparaison.

Commandes déjà exécutées (aucune relancée pendant cette revue) :

- `cargo clippy --release --all-targets --features mlx,chat -- -D warnings` : succès8,12s ; `cargo build --release --features mlx,chat` : succès1,03s ; fmt passé, commande/log de l'époque conservés dans le dossier de travail. `cargo test --release --features mlx,chat -- --test-threads=1` :159passés/159ignorés,166,58s compilation incluse. Les deux testsCPU77 sont bien `ok` dans tests.log ; le composant physique y est explicitement ignoré.
- C++SDK : fixture réelle sous ASan/UBSan,32combinaisons count/API + bords zéro/négatifs/maxima. Mutation isolée retirant vm_ok : normal0, mutantSIGABRT−6, restauré0 ; SHA du header normal/restauré correspond au header conservé. La mutation n'a jamais touché la production.
- `cargo kani --lib --features mlx,chat --harness memory_stats_decoder_preserves_validity_and_values -Z unstable-options --harness-timeout 180s --export-json .../kani.json` : Kani0.68.0/CBMC6.11.0,127obligations passées+3covers satisfaites,0inatteignable/indéterminée ;12u64arbitraires+PIDu32, aucune assumption, unwind13. Cela vérifie le vrai `MemoryStats::from_raw`, pas le SDK/OS/FFI/Metal.
- CBMC6.11.0 sur l'encodeurC++ et les vrais headersSDK27.0 : exit6 avant obligations, parseurApplelibc++ ; aucun stub, propriétéC++ **non prouvée**. ESBMC absent.
- `python -m pytest -q tests/test_flash_process_memory.py` :220passés/0,31s ; Ruff/format/py_compile passent après quatre findings initiaux conservés. Mutation isolée ignorant PID/start :2tests normaux passent,2mutants échouent,2restaurés passent. SIGTERM/SIGKILL CLI réels et jonction exercés. `python -m crosshair check scripts/analyze_flash_process_memory.py --analysis_kind PEP316 --per_condition_timeout 5` : moduleCrossHair absent,exit1, aucune preuvePython. La suite1078 antérieure n'a pas été répétée pour ce module isolé.
- Parent : composant `array::tests::native_process_memory_stats_are_available_and_monotone`1/1, puis quatre témoinsfrais B2/MTP MAIN/HOLD et snapshotsready/complete réels,8réponses exactes128sorties/post30clos. Ces mesures ne sont pas un gaincandidat77 ; les budgetscompression77B ont échoué et le restent. Références dans fresh-controls-v1/summary.json/rapports bruts.

Les binairesv1 (`69b919ae...` CLI, `12a7472f...` tests) incluent l'état local72/73 avec les sources77 ci-dessus. Ils ne doivent pas être annoncés comme les binaires reconstruits d'un futur commit77seul. Les vérifications77 portent sur les sources exactes conservées ; la CI du commit isolé devra au minimum confirmer sa compilation. Pas de nouvelle inférence requise pour simplement publier les observations77 déjà établies.

## Taille et preuvesPython à rendre durables

Le dossier77W contient actuellement **63fichiers /880069octets**. Les deux plus gros JSON (VM268736octets et échantillons237776octets) documentent l'incident et les observations ; les conserver est raisonnable. Aucun moteur, objet natif, dylib, metallib, core dump ou modèle dans ce dossier. `research/sourceexcerpts.raw.txt.gz` est une petite archive de texte source3842octets, identifiée par son manifeste, pas un exécutable de build. Les binairesv1 d'environ20Mo restent horsGit sousROOT/build.

Révision autorisée par le parent : les preuvesCPU de l'analyseur ont été copiées vers `docs/measurements/flash-next-ssd-impl/opt-77-process-memory/analyzer-cpu/` : **20fichiers texte /12496octets**, plus `copy-manifest.json`. Les SHA avant/après de chaque source/destination sont conservés dans ce manifeste et dans `ROOT/build/flash-opt-77-process-memory/analyzer-publication-copy-manifest.json`. Échecs initiaux, CrossHairabsent et mutation restent présents ; les deux`.pyc`, `pytest-cache/`, `mutation-pytest-cache/`, `ruff-cache/` sont exclus. Aucun fichierexécutable binaire ni source native modifié.

Le patch préparé `ROOT/build/flash-opt-77-process-memory/opti-77-only.patch` part de `git show HEAD:opti.md`, ajoute uniquement les trois sections77R-A/77T/77B, leurs préenregistrements et paragraphes de résultats77, et les entréesprocess45qui pointent vers le dossier77. Les paragraphes de résultats72/73/74/75/76/78 non publiés sont exclus ; leurs références historiques dans les protocoles77 demeurent du contexte, pas des résultats promus. La copie complète résultante et le manifeste de sélection sont dans `publication-subset/`. **W/opti.md n'a pas été écrit ni tronqué**, donc toutes les entrées du parent et des agents y restent intactes.

`git diff --check` des quatre fichiers suivis natifs/CI77 passe (exit0). Inspection statique des quatre nouveaux fichiers sourceCPP/Python : aucune fin de ligne avec whitespace superflu. Les gros logs historiques ne sont pas réécrits pour leur mise en forme. Aucun résultatCIvert n'est déduit de ces contrôles locaux.

## Contrat ABI et limites à conserver

Le CABI mémoire est privé au couple Rust/C++ construit ensemble :5→12valeurs et ajout de count. Les cinq positions initiales sont conservées ; null/capacité<12 sont refusés **avant reset_peak et toute écriture**. Le binaireRust et son bridge doivent toujours être reconstruits ensemble ; ce n'est pas une extension d'ABI compatible avec un ancien appelant binaire à deux arguments. Aucun clientdesktopJSON n'appelle cette ABI directement.

Le masque de validité distingue `Some(0)` et `None` selon succèsAPI/révision retournée. PID+startabstime identifient le processus vivant, sans suivre un processus disparu. La compression est en octets logiques, decompressions en pages avec saturationINT32_MAX et ledger_swapins en octets logiques consommés de segments swap-in : **aucun compteur n'attribue les écritures de swap système au modèle, ni les octets SSD physiques**. La footprint inclut la compression, donc une footprint stable ne prouve pas la résidence. Les états/reset du modèle, les poids et les budgets du cache ne sont pas modifiés. Les champsnatifs sont observationnels ; les plafonds77B appartiennent à l'analyseur d'expérience, pas à une politique universelle de l'app.
