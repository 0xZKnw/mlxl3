# Suite des issues66/67 — 2 novembre 2026

Préparation demandée par « continue de coder sans faire les tests ». Base
main4f22ae7563bac2df726de7b00f1944b9d6431cc0, PR68. Branche dédiée
`codex/flash-mb34-roles-20261102`. Aucun test, proveur, modèle ou benchmark
exécuté ni relancé dans cette étape. Les nouvelles voies restent opt-in.

## D : rôles MB3/MB4 indépendants

Le maître `MLXL3_FLASH_REFERENCE_MB34=0|1` reste à0 par défaut.
`MLXL3_FLASH_REFERENCE_MB34_ROLES=qkv|z|out|all` sélectionne les projections :

| Rôle | Suffixe GDN cible | Entrée → sortie |
|---|---|---|
| qkv, défaut compatible CODE01 | linear_attn.in_proj_qkv | 2560→10240 |
| z | linear_attn.in_proj_z | 2560→6144 |
| out | linear_attn.out_proj | 6144→2560 |

Le préfixe doit désigner une couche canonique0..47 du modèle cible. Les rôles
QSA et têteMTP restent exclus, même à dimensions identiques. Le loader vérifie
les options avant les poids. Le shader canonique n'est pas modifié ; NT1/SG4,
ordre des réductions et conversions par ligne sont conservés. B3 utiliseMB3,
B4 utiliseMB4 ; B1/B2 et les autres largeurs conservent leur géométrie.
Cette extension prépare les z/out avant leur mesure, sans déduire de gain ni
valider qkv. La sélection par rôle permettra leur isolation et la composition.

Préparés, non exécutés :24cas physiques par rôle,72pourall, oracles MB1/MB2,
dtypes/strides/alias et compteurs du vrai dispatch ; refus des options et des
autres rôles. Deux nouveaux harnais Kani CPU portent sur les formesi32 et une
sélection de huit fixtures de parsing ; ce dernier domaine fini n'est pas une
preuve de toutUTF8. Ils figurent dans le manifeste CI, sans exécution ici.

## E : indices immuables de réplication des routes

`MLXL3_SSD_ROUTED_INDEX_CACHE=0|1`, défaut0, active un cache privé parSsdMoe.
Il conserve seulement les indicesUInt32 `[0×10,1×10,…]` pour huit formes1..8,
créés paresseusement. Payload maximal1440octets parinstance (69120octets pour
48instances), hors métadonnées, allocation et empreinte physique non mesurées.
Les inputs et le slot-map changeant avec les évictions ne sont pas mis en cache.
Le takeGPU reste nécessaire. Les formes hors du domaine gardent leur chemin
d'allocation ; la cache ne croît pas au-delà de huit entrées. Aucun global ni
RefCell ajouté ; les deux callers utilisent le même propriétaireSsdMoe.

Préparés, non exécutés :64cas composant (formes1..8, FP16/FP32, strides,
inputs différents), graphes encore vivants après destruction des inputs et
instances, fallback9 et isolation entre instances ;40sorties MoE Hot/Cold
contre oracle externe, cinq entrées refusées et option invalide. Une entrée
Kani sur le cheminMLX réel est conservée pour tentative ultérieure, non prouvée.

## Diagnostic Cold : corrections après audit statique

Deux contre-exemples de métadonnées ont été trouvés par lecture : W1pouvait
annoncer plus d'occurrences que les jobs prêts n'en fournissent ; un job pouvait
annoncer une fin au-delà du span du lot. Les corrections :

- Nouvelle trace `ready_job_indices`, snapshot initial de `wait_ready`, permet
  l'égalité exacte entre W1 et les multiplicités des jobs prêts.
- CODE01 sans masque reste lisible avec bornes min/max de multiplicité,
  explicitement marquées `bounded_multiplicities`, et non indices exacts.
- Les nouveaux timings sont bornés par la durée du lot passé par validate_events.
- Le nombreCold est recoupé avec `resident_gpu_rows` du parent si présent ;
  les données anciennes sans compteur sont explicitement signalées.
- Masque isolé/partiel, doublons et indices invalides sont refusés.

Le diagnostic décrit des métadonnées de scheduling, pas le GPU idle ni la
correction du modèle. La suite préparée comporte50cas dans16fonctions Python ;
elle n'a pas été exécutée. Les profils legacy restent distingués des nouveaux.

## Correctif du contrôle d'inventaire Kani

Le jobUbuntu de la CI déjà déclenchée surmain4f22ae75 a été lu, sans rerun :
[échec observé](https://github.com/0xZKnw/mlxl3/actions/runs/41421208073/job/125196915289),
43testsCIpassés/1échec, assertion327==322. `ci-observed-failure.log` en conserve
l'extrait ; le premier essai de lectureCLI refusait les logs tant que le run
global était en cours, puis le connecteur a livré le log du job terminé.

Le contrôle préparé garde les322obligations historiques du commitd5a776e5 dans
une fixture immuable. Il impose leur inclusion et la différence nommée des
nouvelles obligations, au lieu d'un compteur courant figé. Il vérifie aussi
que le retrait d'une nouvelle obligation est refusé. Le manifeste final contient
329noms uniques, dont les sept ajouts explicites depuis cette baseline. Ce
correctif de code n'a pas été validé par réexécution de la suite.

## Contrôles réellement exécutés

```
MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx cargo check --locked --features mlx,chat --all-targets --jobs 1
MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx cargo clippy --locked --features mlx,chat --all-targets --jobs 1 -- -D warnings
cargo fmt --check
ruff check scripts/profile_flash.py tests/test_cold_schedule_diagnostic.py tests/test_kani_ci.py
ruff format --check scripts/profile_flash.py tests/test_cold_schedule_diagnostic.py tests/test_kani_ci.py
```

Compilation tous targets9,01s et Clippy strict10,29s réussis. Rustfmt, Ruff et
revue du diff passent ; syntaxePython parsée sans importer/exécuter les tests.
Avertissement dépendance préexistant `block0.1.6` conservé. Les fixtures et
harnais préparés ne sont pas des résultats. Aucune correction numérique,
réutilisation physique, régression mémoire ou performance qualifiée ici.

Les prochains contrôles devront vérifier chaque rôle et option séparément,
puis leur composition à largeur identique avec AR/MTP/DFlash et prefill.
L'ancre mémoire/swap reste à contrôler au prévol, sans réinitialisation
silencieuse. Aucun merge ou remplacement de l'application pour cette suite.
Statut : code préparé, tests et mesures différés, aucun essai actif.

## Publication

Quatre commits de source poussés sur
[`codex/flash-mb34-roles-20261102`](https://github.com/0xZKnw/mlxl3/tree/codex/flash-mb34-roles-20261102),
du381bde3c au242ed1140a028707c5a05f52ab2c386bded89520, SHA distant relu et
identique. Le correctif CI, les rôles MB34, le cache d'indices et le diagnostic
ont chacun leur commit. `identity.json` conserve les empreintes du code source,
`commits.json` les quatre commits ; cette note fait l'objet d'un cinquième
commit documentaire. Aucun nouveau test ou workflow déclenché par nos outils,
aucune PR/merge de cette suite, options expérimentales toujours désactivées.
