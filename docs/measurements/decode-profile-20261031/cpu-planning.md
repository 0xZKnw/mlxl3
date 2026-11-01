# DECODE-PROFILE-08-PLAN — plan CPU, 31 octobre 2026

Lecture/inventaire uniquement. Aucun moteur, capture, benchmark, build ou proveur
lancé. La sous-entrée physique reste à préenregistrer par le coordinateur après
le témoin labels/counters et l'installation du toolchain Metal. TurboEXL3 sera
relu après le profilage demandé, pas anticipé par ce plan.

## Cible et inconnues

Ancienne fenêtre20pas : **12,142ms/token de CPU actif∩GPU idle dans les
soumissions**, dont11,897 dans le premier async et0,245 dans le second. L'ensemble
du decode exposait22,150ms/token. Les249échantillons à1kHz (243/6 par rôle) ne
donnaient pas des durées exclusives par fonction ;48feuilles étaient
`__kdebug_trace64`. Ce budget historique doit être **remesuré**, pas recopié sur
macOS27.2/Xcode27.2 ou réparti selon les anciens pourcentages.

Priorité : séparer, dans le premier async, la construction de tape, le backend
réel de chaque primitive, le suivi des ressources/encodeur, puis la rétention et
les callbacks. Les callbacks exécutés sur d'autres threads sont un poste séparé
du temps actif du thread d'inférence. Aucun des appels d'encodeur n'est à lui
seul une attente CPU de fin GPU.

## Parcours réel et points mesurables

`ssd_moe.rs:555–578` → `Array::async_eval` →
`mlx_bridge.cpp:382–383` (`mx::async_eval({arr(p)})`) → `eval_impl(...,true)`.
Le premier `slots.indices().async_eval()` soumet ses ancêtres : Cold précédent,
combinaison/HC, GDN ou QSA, routeur et cache-map. Le second soumet le Hot. Garder
les deux points de disponibilité : POSTPROFILE03 a déjà réfuté leur réunion par
le test «routes waited for resident compute», avant benchmark.

Sources MLX0.32.2 primaires relues et identité :

| Source | Blob / contenu utile |
| --- | --- |
| `mlx/transforms.cpp` | `694bca53d841ebe26fa86431e91b867542a6e95a`, copie locale `Documents/mix-stq1_0/build/issue46/mlx-sources/mlx__transforms.cpp`, SHA256 `2609416aa6a2bd26396ac747892bd98bfd43e986f891928e321b0e7195c0c2a5` ; degrés/DFS+BFS80–225, events/fences236–263, eval265–269, pression/scheduler271–284, statuts/détachement298–308, signal/finalize337–345. |
| `mlx/backend/metal/eval.cpp` | `43a6fb993288a5ad2e1422ccea6357e89a921d33`, [primaire relue](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/eval.cpp) ; pool autorelease, outputs, `eval_gpu`, set de rétention inputs/siblings/donation, callback par primitive ou commit29–70. |
| `device.cpp` | `2f25f894e494a4328886148bb6e3332d9f9b298a`, copie `Documents/mix-stq1_0/build/flash-strata57/device.cpp`, SHA256 `008b67e599f5508ab670a91f2ea80372162a8767b5226fbbeb9f3d23c85cdb7c`. |
| `custom_kernel.cpp` | `b73dd72def76fa3412ef9155749507fe5d5cb48e`, même dossier, SHA256 `92548fba86690996a30725ef96bb8e0e299b69a4559e71c60340e7e11c1cac5e` ; sorties/allocation/copies24–45, hash/library/PSO50–54, binding55–80, dispatch99 et temporaires101. |

Distinctions à garder dans les résultats :

- `maybeInsertBarrier`393–405 fait aussi insert/clear des ensembles **sans**
  barrière GPU. Compter ses appels n'est pas compter les barrières émises.
- `get_command_encoder`576–583 crée encodeur concurrent/fence seulement si
  absent ; séparer acquisition répétée et création réelle.
- `set_output_array` appelle `set_input_array` ; une somme de leurs temps
  inclusifs doublerait ce travail. Idem dispatch→barrier→get_encoder.
- `gpu::eval` construit **après** `eval_gpu` un
  `unordered_set<shared_ptr<array::Data>>`, enlève la sortie donnée par donation,
  puis capture l'ensemble dans un completed-handler (ou commit). Construction
  sur le thread encodeur ; callback/destruction potentiellement ailleurs.
- `end_encoding`428–497 : mutex et ensembles de fences, wait/updateFence encodés,
  création du callback retenant temporaires/fences, endEncoding. Le callback
  futur reprend le mutex et nettoie la carte. Les waits encodés ne sont pas des
  waits CPU ; une contention effective du mutex reste à mesurer.
- `commit`516–561 : résidence, callback erreurs/events, appel Metal, nouveau
  command-buffer ; `finalize` termine/commit. Un buffer pris après finalize est
  le nouveau buffer vide. Ne pas interpréter sa durée comme celle du travail.

## Inventaire local qualifié en lecture

`/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace version` :
**27.2(27B52)**. `xcode-select -p` pointe actuellement CLT et `/usr/bin/xcrun`
ne trouve pas xctrace : utiliser le chemin absolu, sans changement global.
`sample` et `spindump` sont présents ; leur aide a été lue sans cible.
`/usr/sbin/dtrace` est présent, **droits/provider pid non qualifiés** ; pas de
proposition de hooks opaques ni de modification SIP/permissions.

`nm -U` confirme les symboles locaux conservés dans le libmlx installé :
`gpu::eval`, `eval_impl`, `CustomKernel::eval_gpu`, allocator malloc/reuse/free,
CommandEncoder et opérateurs des callbacks de gpu::eval/end_encoding/commit.
Leur présence permet une première capture sans reconstruire MLX ; elle ne
garantit pas la résolution de chaque frame inline ou kernel système.

Identité de départ : CLI **1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436**,
libtest **de3616e3d2e0c3d9507e075cf9af4dfe8b3290f0b291ac704b3e1df537f8d910**,
UUID libtest `A11AD849-D166-37FD-A41C-C314053E2E1A`. libmlx SHA256
`d24c7a9b9d55a76bfd3bbcd1d042251a185cbadcb6340c3244a6ffa3dcb7c7e8`, UUID
`F949D357-CE50-3CDC-BD53-D49566E9F72C`. Source541w2 à préserver ; une édition
d'instrumentation exigera sa propre identité et son propre libtest.

## Acquisition proposée, après témoin matériel

Entrée existante :
`qwen4::profile::flash_detailed_profile_preserves_full_outputs_and_states`,
`--exact --ignored --nocapture --test-threads=1`, `MLXL3_PROFILE_KIND=natural`,
`MLXL3_FLASH_PROFILE_MTP=0`, promptMAIN. Même modèle/référence indépendante,
cache3Gio,4lecteurs,40ops/40MB,allocator64Mio,Hot/Wide/Unique1,lookahead0,
T04/T17/T05=0, **T18a explicite0**. Ne pas modifier les seuils ou l'overlap pour
faciliter le profilage. Nouveaux répertoires, gardes et ancre décidés avant PID.

La commande suivante est un **modèle non exécuté** ; options réellement
supportées vérifiées avec `--show-recording-options` (qui ne démarre aucune
capture). JSON proposé :

```json
{
  "Time Profiler": {
    "contextSwitchSampling": true,
    "highFrequencySampling": true,
    "recordKernelStacks": true,
    "recordWaitingThreads": false
  }
}
```

```sh
/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace record \
  --template 'Time Profiler' --instrument 'GPU' \
  --instrument 'Thread Activity' --instrument 'System Call Trace' \
  --recording-options <options.json> --attach <pid> \
  --time-limit 6s --output <nouveau.trace>
```

Première capture minimale : ni Allocations, ni CPU Counters, ni stack logging,
ni Disk I/O Latency (ancien témoin sans événements), ni tous les événements API
Metal ajoutés indistinctement. Si le témoin impose un autre instrument GPU pour
obtenir la timeline utile, le nom et le coût observateur doivent entrer dans
le préenregistrement physique. Ne pas lancer xctrace/sample/spindump ensemble.

Objectif : **40 transitions AR complètes au point chaud**. Règle de sélection
à figer : les40premiers `output_step` complets contenus dans la capture dont
le rang decode est>=16 ; publier aussi la couverture entière et ses marges.
L'ancienne fenêtre2s donnait20pas ;6s est une borne de collecte proposée, pas
la garantie de40pas. Si la capture n'en contient pas40, conserver l'insuffisance,
sans prolongation automatique ni sélection des tokens les plus rapides.

**Limite de déclenchement existante :** le test imprime
`FLASH_DETAILED_PHASE` et écrit ses JSON après la phase complète ; il n'émet
pas aujourd'hui un marqueur vivant «trace_1 decode begin». Son epoch de decode
est enregistré en mémoire puis sauvegardé. Pour garantir l'attache au decode,
le coordinateur doit soit réutiliser son déclencheur qualifié et publier les
rangs réellement couverts, soit préenregistrer un marqueur test-only unique
avant decode. Ne pas annoncer une attache au token exact à partir d'un délai
arbitraire après la fin de control_before, qui précède encore le prefill.

Chauffe puis contrôle sans xctrace avant/après **sur les mêmes rangs** ; ne pas
couper le processus avant les reçus nécessaires au contrôle après. Si arrêt
planifié anticipé, qualifier explicitement la campagne partielle et le contrôle
après manquant. Réutiliser les oracles full logits/états du test et le golden
externe128 ; une trace seule n'est pas un passage du test complet.

## Calage et analyse : amélioration par rapport aux249 points

1. Exporter TOC, `time-info`, time-profile, états/activité de tous les threads,
   GPU intervals, syscalls. Utiliser les vrais noms/schémas du témoin nouveau ;
   vérifier champs/lignes non vides, PID **et durée de vie**, TID et UUID.
2. Origine via `time-info`/timebase et `decode_epoch` natif. Comme l'analyse
   historique, contraindre indépendamment l'offset par l'emboîtement exact des
   `preadv` dans leurs jobs natifs (nombre et octets concordants). Publier plage
   compatible/incertitude ; pas de calage choisi pour réduire artificiellement
   le GPU idle, pas du seul start-date du TOC.
3. Remesurer les unions/intersections Running/GPUbusy/Idle, par rôle des deux
   async et par token/couche. Les appels sont actuellement569/578 dans ssd_moe,
   les anciens numéros560/569 sont historiques. Garder Running/Blocked/Runnable/
   Interrupted et le résidu ; ne pas additionner axes CPU/GPU concurrents.
4. Plus de transitions et sampling haute fréquence améliorent la résolution.
   **Aucun Hz précis promis** : le package Apple décrit le maximum supporté ;
   inspecter les deltas et `weight` réels. Les commutations ajoutent des points
   non périodiques : séparer les populations si l'export le permet, sinon
   signaler le mélange et éviter la comparaison naïve des comptes à249.
5. Garder le PC, user/kernel, état et poids de chaque point ; mêmes règles
   explicites de rattachement Interrupted→Running précédent, avec comptes des
   refus. Les poids restent statistiques, pas du temps CPU exact d'une frame
   (documentation locale `Sampling.instrdst/Contents/Documentation/fulldoc.html`).
   Ne pas convertir57,4% de samples en57,4% des12ms.
6. Classer les frames exclusives les plus proches par familles disjointes :
   tape/events/scheduler ; primitive eval_gpu ; allocation/copies ; hash/PSO ;
   suivi ressources/barrières/bindings/dispatch ; commit/end_encoding ;
   rétention/destruction/callbacks ; driver/kernel/tracing ; inconnu. Conserver
   les piles inclusives et les inconnus, sans double comptage d'allocations.
   Les callbacks des autres threads ont leur propre compte, jamais ajouté aux
   12ms du seul thread d'inférence. Rapporter dispersion par tokens/blocs.
7. Quantifier l'observateur avec temps/token des mêmes positions avant/capture/
   après, proportion de feuilles tracing, taille de trace/RSS et gardes. Une
   forte perturbation borne la portée des ms exposées ; ne pas la soustraire
   comme une correction certaine. Si les nouvelles tables restent vides ou si
   les40pas manquent, documenter le gap avant de proposer une autre acquisition.

## Ce que cette passe ne peut pas promettre

Le sampling existant peut préciser les **familles de coût**, pas compter
exactement chaque primitive, allocation, barrière ou retry. Le parent Rust de
l'async n'identifie pas forcément le producteur de chaque nœud : ses ancêtres
ont été construits plus tôt. Un événement de requête lazy n'est pas un dispatch.

Si un poste important reste confondu par l'inlining, étape distincte à
préenregistrer : instrumentation bornée MLX0.32.2 par primitive/stream et
command-buffer, compteurs appels vs créations réelles, phases disjointes de
`gpu::eval`/CustomKernel, rétention et callbacks. Ne logger que des IDs/tailles
numériques : retenir des Array/shared_ptr pour la trace modifierait donation
et allocations. Éviter strings/JSON/flush par primitive ; mémoire et overflows
bornés, compteurs perdus invalidant la collecte. Oracle et comparaison
instrumentation OFF/ON avant interprétation ; cette étape n'est ni préenregistrée
pour exécution ni implémentée ici.

Choix après cette première attribution : dominantes hash/PSO→étudier T01B ;
dominantes wrappers/rétention→frontière de projection multistage conservant
les kernels ; dominantes GPU→T09 sur projection réellement coûteuse. Ne pas
relancer T01A source-seule, POSTPROFILE03, seuils20/80 ou fusions T17/T04 déjà
documentés. Aucun plafond intégrable de12ms ni gain15tok/s promis.
