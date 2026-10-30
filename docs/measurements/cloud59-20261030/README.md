# Issue 59 — préparation cloud du 30 octobre 2026

État : **prototypes locaux opt-in, non qualifiés sur M5**. Aucun gain annoncé,
aucune app modifiée, aucun push/PR/merge/release, aucune commande sur le Mac.
Baseline propre : `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`.

Branche de session : `codex/cloud59-engine-options`. Commits de code :
`4d34e61c` (T17/T04, options séparées), `bb8bab02` (T18a/profil),
`40f45f80` (correction du dtype de production T04). Utiliser les trois ; le dernier
est nécessaire. Les sources finales sont identifiées dans `source-manifest.json`.

Les [issues 59](https://github.com/0xZKnw/mlxl3/issues/59) et
[60](https://github.com/0xZKnw/mlxl3/issues/60) ont été lues via GitHub, corps et
commentaires, notamment la révision du 29 octobre de l'issue 59. Pas de commentaires
au relevé. Les changements non publiés du checkout `7978fce…` et les mesures
locales du 29 ne sont pas présents ici. **Comparer leur diff avant cherry-pick** :
ce travail ne démontre pas l'absence de doublons dans un arbre inaccessible.

`AGENTS.md`, le journal et les antécédents pertinents ont été consultés. Le dossier
`.agents/skills` manque au checkout ; le skill utilisateur a été retrouvé via le
lien de `docs/qwen27-m5-round2.md` :
[formal-proof-skill](https://github.com/0xZKnw/formal-proof-skill/tree/5f00bb443c9d0d14f705a3098381850166a83e5c),
SKILL et références Rust/C++/Python lus à cette révision. Il n'a pas été ajouté
comme dépendance du projet.

## Options indépendantes

Toutes acceptent seulement `0` / `1`, et sont OFF si absentes.

| Option | Changement effectif | Référence et limites |
| --- | --- | --- |
| `MLXL3_FLASH_GDN_EPILOGUE=1` | T17 : graphe MLX compilé cast F32 → sigmoid F32 → produit F32 → cast F16. Norme native par ligne, puis un épilogue commun B2/B4/B8. | `0` garde la chaîne non compilée. Ni facteur z, ni fusion norme/récurrence/H128/out.suh. Garde F16 `[1,B,48,128]`, B=1..8, fallback de référence hors garde. |
| `MLXL3_FLASH_SWIGLU_DOWN=1` | T04 : étend le SwiGLU déjà compilé par le cast F16 puis le produit F16 avec down.suh, seulement pour les slots Cold non masqués/non groupés. GU/SwiGLU gardent leur dtype F32 ou F16 ; H128 natif ensuite. | `0`, les slots Hot masqués et les groupes gardent l'ancien chemin. Deux frontières F16 explicites et dtype de sortie down restauré ; mêmes gathers, splits, sommes et soumissions. Ce n'est pas POSTPROFILE05-C. |
| `MLXL3_SSD_DYNAMIC_READERS=1` | T18a : les lecteurs de `read_disk_records` réclament le prochain job dans une file atomique bornée, puis joignent tout le lot. | `0` garde les bandes statiques. Même nombre de lecteurs 1..4, mêmes requêtes/validations/destinations, erreurs et publication après lot complet. Option fixée au premier lot par thread ; A/B en processus distincts. Prélecteurs persistants inchangés. |

Les expressions et dtypes demandés sont conservés dans le graphe. **Cela ne prouve
pas que le JIT Metal préserve les bits** : T17/T04 doivent passer les tests
physiques avant usage ou vitesse. Une divergence exige rejet/correction, sans
relâcher la comparaison. Nombre de dispatchs effectivement supprimés, occupation,
spills, latences et performances : **non mesurés**.

Le profil natif rapporte maintenant la politique réellement utilisée par le lot.
`scripts/profile_flash.py` accepte les affectations dynamiques, vérifie chaque
job et les intervalles par worker, et conserve la validation stricte historique
pour les anciennes traces sans champ `schedule`. Les tests couvrent aussi le cas
où un seul worker draine le lot et les traces incomplètes/corrompues.

La revue finale a détecté un contre-exemple de dtype dans le premier prototype
T04 : GU peut être F32, et `expert_mapped_layout` restitue normalement ce dtype.
Le garde F16 initial aurait refusé ce parcours, puis le helper prétransformé aurait
retourné F16. Corrigé avant tout run physique : SwiGLU dans le dtype d'origine,
cast F16 à la frontière historique, produit/H128 F16, sortie down dans le dtype
GU. Un cinquième test physique appelle directement les deux entrées EXL3 down en
F16/F32 sur des banques synthétiques, sans checkpoint, et exige dtype/forme/bits.
Le check de types ne détectait pas ce défaut sémantique : seule la vérification
physique de ces oracles pourra qualifier le candidat corrigé.

## Parcours tracés et contrats

- T17 : `FlashNext.forward` / `verify_pair` / vérificateur draft →
  `Layer.prepare_rows` → `Gdn.forward{,_rows}` → `gate_output` → nouvelle opération
  binaire FFI 14 → projection out habituelle. Aucune mutation des tapes/états par
  l'épilogue. Norme par ligne conservée ; pas de norme vectorisée expérimentale.
- T04 : `SsdMoe.calculate_slots` → `expert_slots_swiglu_down` → nouvelle FFI
  `mlxl3_array_swiglu_scaled` → H128 MLX → `expert_mapped_layout` avec entrée déjà
  transformée. Projections K2/MUL1 640→2560, même disposition et sélection de
  partitions qu'avant ; les entrées non admissibles prennent le chemin existant.
- T18a : `fill_many` → `Array.read_records` → `mlxl3_arrays_preadv` →
  `read_disk_records` → `DiskWorkQueue.claim`. Attribution atomique relaxed
  uniquement ; la visibilité des données et des erreurs dépend toujours du join
  synchrone GCD. Aucun slot partiellement publié, aucune lease relâchée tôt.

## Résultats cloud

Linux x86_64, pas de Metal, ni de M5, ni de checkpoint. Outillage isolé sous
`/workspace/.tooling`, aucun fichier de configuration du Mac touché.

| Contrôle | Résultat et portée |
| --- | --- |
| Cargo tests, sans défaut + `chat` | **245 passés / 6 ignorés** : lib159, bin35, contracts16, autre harnais35. `runtime_gpu` : 0 tests sélectionnés sur Linux ; aucune validation GPU. Voir `rust-cpu-tests.log`. |
| Cargo fmt / Clippy strict CPU / build CPU | Passés ; `clippy-cpu.log`, `build-cpu.log`. |
| File de travail C++ | **4 084 cas** concurrents dans chaque exécution ASan+UBSan et TSan, comptes 1..512 × lecteurs admissibles 1..4 × deux politiques ; chaque job une fois, bornes, épuisement stable, réaffectation et erreurs d'options. `queue-{asan,tsan}.log`. |
| Mutation ordonnanceur | Variante temporaire renvoyant job+1 détectée (abort attendu, source de production intacte). `queue-mutation.{json,log}`. Ce n'est pas un benchmark. |
| Cache factories existant | Test ASan/UBSan : 65 536 paires ×4 comparaisons, capacité128 et allocation après hit inchangées. `kernel-cache-reference.log`. Aucun nouveau cache ni gain déduit. |
| Python pertinent | **312 passés** : profil/export, qualité, bridge benchmark, drafts, DFlash2, MTP bridge ; `python-tests.log`. Ruff et format passent. |
| YAML CI / diff | YAML parsé, PR/push main conservés ; nouveau harnais Kani ciblé et tests ordonnanceur ASan/TSan câblés. `git diff --check` passe. Aucun run distant de ces commits non poussés. |
| Frontend Rust MLX | Check puis Clippy strict de tous les targets `mlx,chat` pour `aarch64-apple-darwin` passent dans un **manifest temporaire** : build.rs désactivé, sources originales en chemins absolus, backend non lié ; `tokenizers` utilise fancy-regex au lieu des dépendances C natives, uniquement dans ce manifest. `clippy-mlx-isolated.log`. Cela vérifie les types Rust des nouveaux tests, **pas la configuration livrée, la FFI C++ ni Metal**. |
| Vrai build `mlx,chat` | **Bloqué par plateforme** : build.rs exige macOS, diagnostic dans `build-mlx-unavailable.log`. |
| Tentatives de cross-check initiales | Linux : dépendance Metal indisponible. Cible Apple avec chat réel : compilateur C sans `-arch`/SDK. Sans chat : tests préexistants importent tokenizer sans garde. Diagnostics conservés dans `rust-mlx-*typecheck.log`, non remplacés par le check isolé. |
| Kani / CBMC / CrossHair | Tentés, outils absents : `*-unavailable.log`. Harnais Rust sur le vrai garde de forme et C++ sur la vraie file conservés, **non vérifiés**. Pas de preuve formelle revendiquée. |
| Tests natifs I/O macOS / GPU / états modèle / bridge physique / débit | **Non exécutés**. La file portable ne teste pas libdispatch, preadv, leases ou Metal. |

Le harnais CBMC explore des interleavings séquentiels de claims dynamiques sur
1..8 jobs et 1..4 lecteurs, sans prouver la concurrence atomique. Le harnais Kani
emploie quatre i32 symboliques, sans hypothèse, pour le garde T17 ; B1/B2/rejet ont
des couvertures. Un test fini ou un check de types ne remplace pas ces preuves.

## Ce soir : qualité avant vitesse

1. Récupérer les commits/bundle de la session dans une branche/worktree séparé,
   inspecter le diff contre le moteur local du 29, conserver les modifications
   Desktop et les mesures négatives. Ne pas utiliser un ancien binaire `target/`.
2. Sur secteur, un seul modèle GPU. Réutiliser le même checkpoint EXL3 2,05 bpw,
   le même draft, les oracles indépendants déjà conservés et MLX0.32.2. Indiquer
   leurs chemins locaux, ne pas en télécharger d'autres pour ce protocole.

```bash
export MLXL3_FLASH_MODEL="/chemin/vers/le/checkpoint-Flash-EXL3"
export MLXL3_MLX_ROOT="/chemin/vers/le/package/mlx-0.32.2"
export MLXL3_FLASH_REFERENCE="/chemin/vers/les/oracles-Flash"
export MLXL3_FLASH_DRAFT_MODEL="/chemin/vers/le/draft-DFlash2-epingle"
export MLXL3_FLASH_DRAFT_ORACLE="/chemin/vers/oracle-DFlash2"
export MLXL3_FLASH_DRAFT_KIND=dflash2
export MLXL3_FLASH_DRAFT_WIDTH=1
export MLXL3_SSD_CACHE_GIB=3 MLXL3_SSD_READERS=4 MLXL3_SSD_OVERLAP=1
export MLXL3_SSD_GPU_HOT=1 MLXL3_SSD_GPU_HOT_WIDE=1 MLXL3_FLASH_MULTI_TOKEN=1
export MLXL3_SSD_SHARED_EXL3=0 MLXL3_SSD_LOOKAHEAD_DEPTH=0
export MLXL3_SSD_PREFETCH_PIPELINE=0 MLXL3_ALLOCATOR_CACHE_MIB=64
export MLXL3_FLASH_GDN_EPILOGUE=0 MLXL3_FLASH_SWIGLU_DOWN=0 MLXL3_SSD_DYNAMIC_READERS=0
mkdir -p build/cloud59-m5
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --features mlx,chat
cargo build --locked --release --features mlx,chat
cargo test --locked --release --lib --features mlx,chat --no-run --message-format=json > build/cloud59-m5/build-tests.json
shasum -a 256 target/release/mlxl3-rs > build/cloud59-m5/binary.sha256
git rev-parse HEAD > build/cloud59-m5/source.txt
git diff --binary > build/cloud59-m5/source.patch
```

3. Après les compilations, exécuter les tests physiques **seuls**. Le fragment
   suivant trouve le vrai binaire test, borne chaque test à 1 heure, joint le
   processus sur timeout et refuse zéro test sélectionné. Arrêt au premier échec ;
   logs persistés. Rien de ce fragment n'a été exécuté dans le cloud.

```bash
python - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path('build/cloud59-m5')
artifacts = [json.loads(line) for line in (root/'build-tests.json').read_text().splitlines()]
bins = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
        and a.get('profile', {}).get('test') and a.get('executable')
        and a['target']['kind'] == ['lib']]
assert len(bins) == 1, bins
micro = ['qwen4::boundary_tests::t17_epilogue_matches_uncompiled_fp32_sigmoid_product',
         'qwen4::boundary_tests::t04_swiglu_down_preserves_both_half_roundings_and_h128',
         'qwen4::boundary_tests::t04_down_projection_preserves_production_dtype_and_bits',
         'qwen4::boundary_tests::t17_real_gdn_outputs_states_and_every_tape_prefix',
         'qwen4::boundary_tests::cloud59_candidates_match_reference_logits_states_and_pair_commits']
integration = ['ssd_moe::tests::two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads',
               'qwen4::tests::flash_pair_verification_preserves_serial_logits_and_all_states',
               'qwen4::tests::flash_mtp_session_preserves_greedy_sequence_target_and_head_states',
               'qwen4::draft::flash_draft_captures_and_all_prefix_commits_match_serial',
               'qwen4::draft::flash_draft_greedy_generation_matches_tokens_and_states']
flags = ['MLXL3_FLASH_GDN_EPILOGUE', 'MLXL3_FLASH_SWIGLU_DOWN', 'MLXL3_SSD_DYNAMIC_READERS']
arms = {'A': (0,0,0), 'T17': (1,0,0), 'T04': (0,1,0), 'T18a': (0,0,1)}
def run(name, env, label):
    log = root / (label + '-' + name.split('::')[-1] + '.log')
    try:
        with log.open('w') as out:
            result = subprocess.run([bins[0], name, '--exact', '--ignored', '--nocapture',
                                     '--test-threads=1'], env=env, stdout=out,
                                    stderr=subprocess.STDOUT, timeout=3600)
        assert result.returncode == 0 and 'test result: ok. 1 passed; 0 failed;' in log.read_text(), log
    except BaseException:
        (root/'quality-status.txt').write_text('FAILED_OR_INTERRUPTED: '+str(log)+'\n')
        raise
for name in micro:
    run(name, {**os.environ, **dict.fromkeys(flags, '0')}, 'boundary')
for arm, values in arms.items():
    env = {**os.environ, **dict(zip(flags, map(str, values)))}
    for name in integration:
        run(name, env, arm)
(root/'quality-status.txt').write_text('PASSED_SELECTED_PHYSICAL_TESTS\n')
PY
```

Les cinq nouveaux tests couvrent les gates F16 finies et layouts stridés,
les deux arrondis SwiGLU/down et H128, les entrées réelles de out, toutes les
couches GDN présentes, B1/B2/B4/B8 et chaque préfixe de tape. Le modèle complet
compare chaque candidat seul puis T17+T04 aux logits/états référence, commits1/2,
erreurs et continuation. Les tests existants ajoutent les oracles indépendants,
le chemin SSD Cold/Hot, les erreurs de lecture et les véritables MTP/DFlash.
L'oracle de capture DFlash doit correspondre au draft fourni. **Une fixture
manquante est un blocage, pas un test sauté avec succès.**

4. Vérifier le bridge physique pour chaque bras : `scripts/check-mtp-depths.py`
   couvre un autre modèle/head ; ne pas lui inventer une compatibilité Flash.
   Utiliser `scripts/benchmark-flash-bridge.py` sur Flash après les tests ci-dessus,
   en AR, `--mtp` et `--dflash-draft "$MLXL3_FLASH_DRAFT_MODEL" --dflash-mode 1`.
   Ses gardes exigent sorties complètes, hash stable et enfants/lecteurs joints.
   Comparer aussi les hashes entre bras ; sa stabilité interne seule ne prouve
   pas la parité contre A. La parité des textes ne remplace pas les logits/états.

## A/B release MAIN/HOLD

Exemple AR, un candidat à la fois. Chaque appel crée un processus indépendant,
chauffe128 puis mesure3×128 ; ACCA apporte deux processus par bras et prompt.
Après validation qualité et relevés secteur/mémoire, exécuter :

```bash
set -euo pipefail
for candidate in T17 T04 T18a; do
  for prompt in MAIN HOLD; do
    pass=0
    for arm in A C C A; do
      pass=$((pass + 1))
      export MLXL3_FLASH_GDN_EPILOGUE=0 MLXL3_FLASH_SWIGLU_DOWN=0 MLXL3_SSD_DYNAMIC_READERS=0
      if [ "$arm" = C ]; then
        case "$candidate" in
          T17) export MLXL3_FLASH_GDN_EPILOGUE=1 ;;
          T04) export MLXL3_FLASH_SWIGLU_DOWN=1 ;;
          T18a) export MLXL3_SSD_DYNAMIC_READERS=1 ;;
        esac
      fi
      python scripts/benchmark-flash-bridge.py target/release/mlxl3-rs "$MLXL3_FLASH_MODEL" \
        "build/cloud59-m5/speed-AR-$candidate-$prompt-$pass-$arm" \
        --cache-gib 3 --prompt "$prompt" --tokens 128 --repeats 3
    done
  done
done
```

Répéter cette matrice **séparément** pour MTP (`--mtp`, dossiers `speed-MTP-*`),
puis DFlash2 (`--dflash-draft ... --dflash-mode 1`, dossiers `speed-DF-*`). Ne pas
utiliser `--compare` : cette option compare les modes, pas les optimisations.
Ne pas lancer une matrice de vitesse dont les tests physiques ont échoué.

Conserver les fichiers `report.json`, conditions `pmset -g batt`, snapshots
`vm_stat`/swap et températures/fréquences disponibles avant/après chaque bras.
Arrêter/reprendre dans une nouvelle campagne en cas de swap, changement
d'alimentation, dérive importante ou interruption ; ne pas masquer le premier
essai. Le harnais bridge borne les échanges mais n'est pas une garde secteur ni
un superviseur thermique : relever ces conditions avec le superviseur habituel.

Comparer, par prompt et mode, **somme(decode_tokens) / somme(decode_seconds)**
des seules mesures, avec les dérives des références et les deux processus par
bras explicités. Ne pas traiter les trois répétitions d'un processus comme trois
essais indépendants. T17/T04 : capturer les vrais dispatchs GDN/Cold et leurs
frontières avant promotion. T18a : comparer fin/p95 des lots, spread des workers,
CPU et débit release ; les octets logiques ne mesurent pas la NAND. Aucun plafond
des issues n'est un gain anticipé.

## Pistes différées après inspection

- **T01** : le cache de factories C++128 et ses clés view sont déjà là ; après
  hit restent les descriptions de sorties, `CustomKernelFunction`, allocation et
  encodage. Pas de handle préparé/replay ajouté sans attribution après hit ni
  contrôle du backend privé MLX0.32.2. Le test cache existant reste intact.
- **T09/T12** : aucune projection critique isolée par labels/counters dans les
  preuves disponibles ; pas de nouveau finish/compteur/fusion attention au hasard.
- **T07/T10** : `ExpertSlots.grouped` existe déjà et les shaders ont toujours
  `x_values[MB][4]`. Routes B2 naturelles, taux de répétition Hot/Cold et spills
  manquent ici. Aucun grouped65/66 réactivé, ni élargissement artificiel.
- **T18b** : publication progressive impose un autre protocole de leases,
  cohortes et conservation des sommes. Différé jusqu'au résultat T18a et au bilan
  des soumissions ajoutées. T18a ne publie jamais une lecture partielle.
- **T06** : shared unique et overlap exigent un plan Hot/Cold mesuré ; inchangé.
- **T19/T13–16** : chantier TTFT/prefill dépendant de vrais chunks ; pas de faux
  port vers `forward_tokens` scalaire, aucune métrique decode revendiquée.
- T05 reste mixte/non robuste (+MAIN/−HOLD), ARC et cache statique47/55 restent
  rejetés, seuils Metal et suppression de Hot submission inchangés. Aucun de ces
  essais négatifs n'a été relancé.

Le livrable contient du code et des tests récupérables. Il reste à compiler la
vraie FFI sur Apple, valider les bits et les états sur M5, puis mesurer. Aucune
expérimentation de cette session n'est activée par défaut.
