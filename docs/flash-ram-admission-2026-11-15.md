# Flash Next : refus « mémoire insuffisante » au chargement (15 novembre 2026)

Essai `FLASH-RAM-ADMISSION-20261115` dans [opti.md](../opti.md). Base origin/main `c42e4c95` (engine 2.0.11-beta.1, Desktop 2.0.1), branche locale `claude/flash-ram-admission`. Non poussé, aucune release, app installée inchangée.

Révision du 15 novembre 2026 (soir) : la branche a été rebasée sans conflit sur origin/main `24a6e6e2` (engine 2.0.11-beta.2) et les contrôles CPU et Kani ont été refaits avec succès. Le test GPU de chargement réel n'a **pas été exécuté**. Le checkpoint EXL3 n'est plus présent sur ce Mac, et le retélécharger (58,36 Gio) demande l'accord de l'utilisateur.

## Modèle concerné

« 3.8 Flash » = `turboderp/Qwen3.8-Flash-Next-exl3`, branche `2.05bpw_h4_ng4`, révision `65c895314393431c09050b2e04e250836b3a6eb4` (`model_type` `qwen4_exp`). Payload 58,3587 Gio : 24,4380 Gio de n-grammes lus par ligne, 48 × 512 experts × 1 248 012 o = 28,5648 Gio d'experts streamés depuis le SSD, le reste résident. La taille du téléchargement n'est pas la RAM demandée.

## Cause

Sans `MLXL3_SSD_CACHE_GIB`, `flash_config::layer_slots` (appelé par les backends MLX `qwen4.rs` et natif `gpu/flash.rs` avant tout chargement) dimensionne le cache experts avec `automatic_cache_slots(available)` :

- `available` = (`free_count` − `speculative_count` + `external_page_count`) × taille de page, via `host_statistics64`. C'est la mémoire libre plus le cache de fichiers. Formule vérifiée contre XNU : `host.c:818` déclare `free_count = vm_page_free_count + speculative_count`, et `vm_resident.c:5939-5940` compte aussi les pages spéculatives dans `pageable_external`. Il n'y a ni double compte ni erreur d'unité. La RAM physique n'intervient pas.
- Budget = min(available − **8 Gio fixes**, 6 Gio) / (48 × 1 248 012 o). Le chargement est refusé sous 10 slots par couche, soit **9 188 980 352 o (8,558 Gio) disponibles** quelle que soit la machine. Le message ne donnait aucun montant : `not enough system memory for Flash's minimum SSD cache and memory reserve`.

La réserve de 8 Gio (essai 12-F/G) a été conçue et mesurée sur le M5 / 24 Gio. Le besoin réel hors cache y a été mesuré à 4,68 Gio : pic de vie du processus moins les banques d'experts, MTP actif, contexte 4096, allocateur 512 Mio (5 021 365 320 o et 5 027 746 912 o sur les deux bras G). Le reste, environ 3,3 Gio, est une marge pour les autres applications. Sur un Mac de 16 Gio, la réserve vaut la moitié de la RAM. Il faudrait 8,56 Gio de pages libres ou de fichiers alors que macOS et les applications ouvertes en occupent déjà une grande part. Pendant ce diagnostic, le M5 de l'utilisateur ne présentait d'ailleurs que (70 123 + 484 413) × 16 384 o = 8,46 Gio : l'ancien défaut aurait refusé Flash sur cette machine aussi.

Conclusion : refus de politique, pas d'erreur de mesure. Sur 16 Gio, c'est un faux négatif partiel, puisque le moteur tient avec un petit cache (4,68 + 0,56 Gio). Sur 8 Gio, le refus reste légitime. La RAM et la version de macOS de la machine signalée ne sont **pas connues**.

## Correctif

- Réserve = 5 Gio pour le moteur (mesure 4,68 Gio plus une marge) + marge système clamp(RAM physique / 8, 1 Gio, 3 Gio), avec la RAM physique lue par `sysctlbyname("hw.memsize")`.
- Plafond 6 Gio et minimum de 10 slots inchangés, comme la formule `available`, les kernels, le LFU et l'override explicite.
- `automatic_cache_plan` renvoie `CacheShortfall { available, physical, reserve, required }`. Son message, qui est le message le plus externe et donc celui que le bridge transmet par `to_string()`, donne le besoin, le disponible, la réserve, la RAM et le contournement.

| RAM physique | Réserve avant → après | Disponible minimal avant → après |
|---|---:|---:|
| 8 Gio | 8 → 6 Gio | 8,558 → 6,558 Gio (refus quasi certain, message chiffré) |
| 16 Gio | 8 → 7 Gio | 8,558 → 7,558 Gio |
| 18 Gio | 8 → 7,25 Gio | 8,558 → 7,808 Gio |
| ≥ 24 Gio | 8 → 8 Gio | inchangé (budgets 12-G conservés) |

Exemple de message (6 Gio disponibles sur 16 Gio) : `not enough memory to load Flash Next: it needs 7.56 GiB of free or file-cache memory, macOS reports 6.00 GiB available (7.00 GiB reserved for resident weights, MTP, buffers and other applications on this 16 GiB Mac, plus 0.56 GiB minimum SSD expert cache). Quit other applications and retry, or set MLXL3_SSD_CACHE_GIB (0.6 to 12) to size the expert cache manually`.

## Contournement sans nouvelle version

L'app transmet son environnement au moteur. Pour forcer un cache de 1 Gio (17 slots par couche) :

```sh
launchctl setenv MLXL3_SSD_CACHE_GIB 1
```

Quitter puis relancer MLXL3 Desktop. Revenir en arrière avec `launchctl unsetenv MLXL3_SSD_CACHE_GIB`. Ce chemin contourne toute réserve automatique. Il n'est **pas testé** sur la machine signalée et peut faire compresser ou swapper macOS si la mémoire manque vraiment.

## Vérification

Voir l'entrée d'opti.md pour les commandes, comptes et chemins de logs (`build/flash-ram-admission/`, local au worktree). Aucun chargement GPU ni modèle n'a été lancé. Le seuil 16 Gio reste une politique : elle n'est pas mesurée sur un Mac de 16 Gio et ne garantit pas l'absence de compression ou de swap. La croissance du KV avec un long contexte n'est pas incluse dans la réserve, ni avant ni après ce changement.
