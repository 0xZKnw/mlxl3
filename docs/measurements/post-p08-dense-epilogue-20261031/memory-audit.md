# F02 — lecture mémoire, 31 octobre 2026

Statut : audit de sources et de reçus existants seulement. Aucune modification
native, politique mémoire, sonde, compilation ou mesure supplémentaire. L'identité
F02-S32 reste celle des 636 sources, CLI `6f35338c…` et libtest `43c55cd1…`.
MTP/DF128 sont des essais déjà préenregistrés, distincts de cette lecture.

## Conclusion bornée

La compression privée du processus natif varie réellement entre les quatre
processus AR128. Elle constitue une observation plus ciblée que les seuls
compteurs système de l'essai 50, mais elle ne démontre ni une compression des
banques EXL3 précises, ni un coût exclusif pendant le decode, ni une cause de la
dérive. Il n'y a donc pas de motif pour modifier F02, rejouer AR pour le
« stabiliser », corriger les débits par un facteur mémoire, ou déclarer son gain
causal établi.

Une résidence MLX sur Flash SSD 3 Gio serait un essai **distinct** du dense rejeté
HENK0O05 : modèle MoE, banques réécrites, environ 6,58 Go actifs MLX, cache experts
3 Gio et données privées au PID maintenant disponibles. Ce sont des différences
concrètes pour une hypothèse ultérieure. La piste Flash n'est toutefois pas
nouvelle : 52D avait déjà étudié sa pertinence à 6 Gio et n'avait lancé aucun
prototype. Cette lecture ne justifie pas à elle seule une reprise physique ; il
faudrait une question propre, une borne/restauration contrôlée et un
préenregistrement distinct, sans réutiliser les résultats F02 comme mesure de
résidence.

## Politique conservée et API

- Aucun appel `set_wired_limit` ni option/garde wired ne subsiste dans
  `native/src` ou `native/ffi`. La FFI actuelle expose `set_cache_limit`,
  `clear_cache` et les statistiques mémoire, pas le setter wired
  (`native/ffi/mlx_bridge.cpp:138–175`, `native/src/array.rs:315–397`).
- F02 fixe `MLXL3_ALLOCATOR_CACHE_MIB=64`. Cela borne les buffers **libres**
  conservés par l'allocateur ; ce n'est pas le cache d'experts. Le défaut Flash
  de 512 Mio est appliqué seulement en l'absence de cet override
  (`native/src/qwen4.rs:1412`, `native/src/array.rs:392–397`). Les snapshots AR
  mesurés montrent environ 48–57 Mo de cache allocateur, pas plusieurs Go de
  temporaires libres conservés.
- Le cache experts reste 53 slots/couche × 48 couches, budget nominal 3 Gio.
  Les banques slot-major sont des Arrays persistants alloués via
  `Array::uninitialized` (`native/src/ssd_moe.rs:198–210`) puis
  `mx::allocator::malloc` (`native/ffi/mlx_bridge.cpp:86–94`). Une politique de
  résidence de cet allocateur pourrait donc les concerner, mais un budget global
  n'identifie ni ne sélectionne les banques chaudes.
- Le header installé MLX 0.32.2
  `/Users/justin/Documents/localqwen3.8flash/.venv/lib/python3.12/site-packages/mlx/include/mlx/memory.h:66–78`
  expose `size_t set_wired_limit(size_t)`, retourne l'ancienne limite, documente
  le défaut 0 et l'erreur au-delà de la limite système. `set_cache_limit` est
  une autre API : il ne rend pas les poids résidents. Aucun setter n'est appelé
  pour cette lecture.
- La sonde 52D avait effectivement trouvé limite wired 0, actif/pic MLX 0,
  recommandé Metal 19 069 665 280 B et RAM 25 769 803 776 B sur M5/MLX0.32.2.
  C'est une observation historique sans modèle, pas une nouvelle vérification
  de la valeur dynamique du processus F02 ni une preuve de gain. Sources et
  SHA du header sont conservés dans
  `../flash-next-ssd-impl/opt-52-residency-diagnostic/sources.json` et résultat
  dans `probe-stdout.txt` du même dossier.

Une éventuelle garde future devrait restaurer la valeur antérieure sur succès
et erreur, interdire des propriétaires concurrents et tenir compte du draft et
des temporaires, pas seulement des 3 Gio de slots. La limite recommandée Metal
historique n'est pas une garantie de marge RAM présente ; rendre des allocations
résidentes peut déplacer la pression vers d'autres pages/processus. Les tests
anciens du prototype retiré ne valident pas d'avance une nouvelle FFI/garde.

## Ce que mesurent les reçus F02

`mlxl3_memory_stats` interroge le **PID courant** via `proc_pid_rusage(getpid())`
et `task_info(mach_task_self(), TASK_VM_INFO)` ; les champs ne proviennent pas
de `vm_stat` (`native/ffi/mlx_bridge.cpp:155–175`,
`native/ffi/process_memory.h:15–40`). Les masks de validité sont conservés par
Rust : indisponible devient `None`, pas zéro (`native/src/array.rs:343–374`).

- `process_compressed_logical_bytes` : stock logique actuellement compressé,
  pas la taille physique du compresseur ni des octets SSD ; peak et lifetime
  sont des quantités distinctes.
- `process_decompressions` : compteur de pages du processus, saturant à
  `i32::MAX`, pas un temps ni un nombre de lectures d'experts.
- `process_ledger_swapins_logical_bytes` : logique consommée depuis des
  segments swap-in, pas des lectures SSD physiques ni des Swapouts.
- `mlx_active/cache/peak_bytes` : comptabilité de l'allocateur MLX ; elle ne
  prouve pas que tous ces octets sont physiquement résidents.
- Footprint native, RSS et compression logique n'ont pas la même définition.
  Les valeurs « RSS 6,78→5,52 Go » viennent ici de
  `children_maximum_rss_bytes`, obtenu par le superviseur via
  `getrusage(RUSAGE_CHILDREN).ru_maxrss` : high-water des enfants, pas un
  échantillon de RSS instantané au même point que le snapshot natif
  (`../slotstream-20261026/supervisor.py:185–188`).

Le snapshot natif est pris à la fin de chaque génération
(`native/src/main.rs:2656`). Il couvre donc chargement/histoire du processus
pour les compteurs cumulatifs. Entre deux générations il couvre aussi la
frontière de requête, le prefill et le decode suivants ; il n'isole pas le
decode. Les compteurs système du superviseur portent sur toute la machine.

### Quatre processus AR128, mesures MAIN puis HOLD

Sources : `prepared-s32/AR128-root-summary.json`,
`prepared-s32/runs/f02-s32-pilot128-AR-{1-A,2-C,3-C,4-A}/report.json`,
et leurs `*-plan-result.json`. Valeurs en Go décimaux ; aucune donnée retirée.

| Bras | PID natif | Débit agrégé tok/s | Compressé logique après MAIN / HOLD (Go) | Décompressions : fin chauffe HOLD → fin mesure HOLD (pages) | High-water enfants (Go) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 40437 | 10,585675 | 0,749650 / 0,749584 | 21 603 → 24 213 (+2 610) | 6,784221 |
| C2 | 40706 | 10,784553 | 1,154384 / 1,154400 | 50 814 → 51 038 (+224) | 5,930254 |
| C3 | 40884 | 10,577762 | 1,260880 / 1,260650 | 54 376 → 56 144 (+1 768) | 5,775000 |
| A4 | 41140 | 10,148837 | 1,486848 / 1,510015 | 67 450 → 70 253 (+2 803) | 5,524685 |

Les snapshots mesurés ont un actif MLX quasi constant (~6,5794–6,5795 Go),
footprint natif ~7,133–7,142 Go et ledger swap-in logique 0. Le compteur système
Swapouts reste 786432 B avant/après chacun (512 Kio cumulés depuis l'ancre w1 de
262144 B). La compression peut donc exister sans nouveau Swapout. Les stocks
privés plus grands ne se traduisent pas mécaniquement en davantage de pages
décompressées dans les deux requêtes mesurées : C2 est un contre-exemple à cette
assimilation. Cela ne prouve pas non plus leur gratuité.

## Antécédents à conserver

`opti.md:4165–4178`, HENK0O-QWEN27-05 : dense 9,67 Go, budget effectivement
10 233 576 045 B, ABBA48 tokens/2 répétitions/repos25 s. Sorties exactes et tests
de garde succès/erreur/second propriétaire/restauration passés. Court complet
+6,61 %, document complet +13,28 %, decode document −17,09 % ; dérive des témoins
persistante, donc aucune régression universelle causale prouvée, mais aucun gain
robuste. Prototype entièrement rejeté/retiré. La synchronisation de sortie
proposée pendant la campagne n'a finalement pas été recompilée/testée ni livrée.
Reçus encore présents : `../qwen27-m5/wired-abba/results.json` et
`../qwen27-m5/wired-guard.log`. Le patch temporaire historique cité par le journal
n'a pas été utilisé pour cette lecture.

`opti.md:14020–14026`, FLASHSSD52D : revue de la même API et sonde sans poids,
motivée par la compression **système** de Flash à 6 Gio ; aucun prototype ou
benchmark wired Flash exécuté. F02 ajoute la spécificité 3 Gio et les compteurs
privés, pas une démonstration que la résidence récupérera la dérive ni un budget
de millisecondes récupérables.

Décision de cet audit : conserver F02 et sa campagne inchangés, qualifier AR
NON CONCLUANT sous dérive, terminer uniquement les modes déjà préenregistrés.
Aucune activation ou répétition pour stabiliser n'est déduite de ce mémo.
