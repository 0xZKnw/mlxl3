# 77R-A — attribution de la pression mémoire au moteur

Revue du 23 octobre 2026, lecture seule : aucun prototype, build, appel GPU ou chargement modèle. 75M inchangé. L’utilisateur autorise désormais du swap global lorsque le comportement du moteur est compris ; cette permission ne prouve pas que les anciens Swapouts étaient étrangers au modèle.

## Conclusion utilisable

Le meilleur ajout minimal est une mesure **self**, dans la fonction native `memory_stats` existante, de `TASK_VM_INFO` : `compressed`, `compressed_peak`, `compressed_lifetime`, `decompressions` et `ledger_swapins`. Ajouter un masque de validité, le vrai PID natif et le `ri_proc_start_abstime` déjà accessible via `proc_pid_rusage`. Garder les snapshots aux frontières existantes, sans nouvel `eval()` ni nouvel appel de synchronisation GPU.

Ces données permettent de dire « swap global observé, compression privée du moteur X, consommation par le moteur de pages provenant de segments swap-in Y ». Elles ne permettent pas d’attribuer exactement les écritures swap système, ni de conclure « aucun octet du modèle n’a été écrit en swap ».

## Sources et versions

- SDK local **27.0**, en-têtes dans `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include`. SHA publiés dans `sdk-sources.json` ; copies exactes locales dans ROOT/build/flash-opt-77-process-memory/sdk/. Les extraits publiés ont une copie brute gzip et un manifeste de formatage.
- Noyau exécuté : Darwin **27.2.0**, `xnu-13432.41.4~3`, ARM64 T8142.
- Source publique Apple XNU épinglée : [`f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`](https://github.com/apple-oss-distributions/xnu/commit/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea), publiée le 16 octobre 2025. **Ce n’est pas le source exact du noyau local**. Le SDK local confirme les champs ; les sources publiques expliquent leur implémentation publiée, à vérifier par contrôle physique sur le système réel.
- Tous les fichiers téléchargés, URL et SHA : `xnu-sources.json`. Extraits numérotés : `sourceexcerpts.txt`. Les chemins `pmap` initialement essayés en 404 ont été corrigés après consultation de l’arbre Git officiel ; aucune disponibilité n’a été inventée.
- [Documentation Apple de task_vm_info_data_t](https://developer.apple.com/documentation/kernel/task_vm_info_data_t) confirme les propriétés publiques.

## Unités et limites précises

| Champ | Source publiée | Interprétation correcte |
|---|---|---|
| `compressed` | [task.c:6050](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L6050), [pmap.c:3856](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/arm64/sptm/pmap/pmap.c#L3856) | Balance d’octets **logiques** anonymes représentés par des pages compressées. Crédit par taille de page logique, pas taille du payload comprimé. Peut représenter des données encore dans le compresseur ou paginées sur stockage. Ce n’est pas l’occupation physique du compresseur système. |
| `compressed_peak` | [task.c:6051](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L6051) | Maximum de vie de cette balance. Ne se remet pas à zéro avec `mx::reset_peak_memory`. Un ancien pic inchangé n’invalide pas automatiquement une phase actuelle. |
| `compressed_lifetime` | [task.c:6052](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L6052) | Crédit cumulé du ledger de compression privée. Sa différence détecte du churn même si la balance actuelle revient au point initial. Pas un compteur d’octets NVMe écrits. |
| `decompressions` | [task.c:6217](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L6217), [vm_fault.c:212](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_fault.c#L212) | Nombre d’événements/pages décompressés, somme de la tâche et de ses threads, **saturé à INT32_MAX**. Décompression RAM possible : pas nécessairement de lecture swap. Au plafond, delta nul n’est plus une preuve d’absence d’activité. |
| `ledger_swapins` | [task.c:6230](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L6230), [vm_compressor.c:5454](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_compressor.c#L5454) | Ledger en **octets logiques** du processus consommateur : crédit `PAGE_SIZE` quand une page demandée provient d’un segment marqué `c_swappedin`. Un segment peut déjà avoir été rapatrié ; ce n’est ni le nombre exact de lectures disque synchrones, ni les octets SSD physiques, ni les Swapouts du processus. |
| `ri_pageins` | [bsd_kern.c:1281](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/bsd_kern.c#L1281) | Compteur 64 bits `task->pageins`, incluant les page-ins du processus ; ne pas l’appeler « major faults swap » ni imposer zéro à un moteur SSD. Le chargement de fichiers est une autre cause. |
| `TASK_EVENTS_INFO.pageins/faults` | [task.c:5824](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L5824) | Version limitée/saturée INT32_MAX. Préférer `ri_pageins` 64 bits pour les lectures, et garder les défauts de pages comme diagnostic distinct. |
| `phys_footprint` | [task.c:1227](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c#L1227) | Inclut notamment `internal_compressed`, les ajustements IOKit et tables de pages. **Une empreinte stable peut masquer résident→compressé**. Les 26 points appariés précédents prouvent seulement l’absence d’une croissance majeure de l’empreinte, pas l’absence de compression du moteur. |

Le process accounting porte sur tout le moteur (heap, stack, buffers CPU/GPU attribués à cette tâche), pas exclusivement sur les banques d’experts. La mémoire GPU partagée/IOKit n’est pas une simple somme de `internal` et `compressed`.

### Champs V6 récents à ne pas utiliser tout de suite

Le SDK local nomme `ri_page_wait_time_mach`, `ri_page_cache_hits` et `ri_conclave_footprint`, là où le XNU public épinglé garde `ri_reserved[9]`. Le même flavor `RUSAGE_INFO_V6` ne suffit donc pas à distinguer « champ non implémenté sur ce noyau » d’un vrai zéro. Unité et portée du temps d’attente non confirmées par le source public correspondant au noyau local. Exclure ces champs du 77T minimal plutôt que deviner une conversion.

`os_proc_available_memory` n’est pas un substitut : le SDK le marque `API_UNAVAILABLE(macos)` et décrit une limite de mémoire sale de l’application, pas la RAM système libre.

## Disponibilité et sécurité FFI

1. Utiliser `task_info(mach_task_self(), TASK_VM_INFO, ...)`. Pas de `task_for_pid`, d’entitlement de débogage, d’API privée ou de lecture d’un processus utilisateur tiers.
2. Initialiser complètement le struct et passer **TASK_VM_INFO_COUNT**, en nombre d’éléments `natural_t`, jamais `sizeof` en octets. Vérifier `KERN_SUCCESS` et le count retourné.
3. Champs de compression de base : count au moins `TASK_VM_INFO_REV0_COUNT`. Décompressions : Rev5. Ledger swap-ins : Rev6. Ne pas interpréter un champ situé au-delà du count retourné, même si le struct est initialisé à zéro.
4. Masque de validité distinct des valeurs : **Some(0) est valide**, `None` signifie indisponible/échec. Ne pas recopier le motif existant `value != 0` sur ces compteurs.
5. `ledger_swapins` est signé dans le SDK ; une valeur négative doit rester invalide, pas être convertie en énorme `u64`. Décompressions négatives : idem ; INT32_MAX doit être signalé comme saturé pour toute conclusion de deltas.
6. Conserver le vrai PID natif (`getpid()`/`std::process::id`) et `ri_proc_start_abstime` pour détecter réutilisation de PID/redémarrage. L’UUID rusage est celui de l’exécutable, pas un identifiant unique de processus.
7. Attention au PID du moniteur : `Popen` peut désigner le checker Python ou `/usr/bin/time`, et non le moteur. Les stats self évitent cette confusion. Après la sortie, l’échec de lecture d’un PID ne vaut pas mémoire nulle ; dernier snapshot natif valide ou statut inconnu.
8. Élargir l’array CABI de 5 vers 11/12 champs exige une mise à jour **coordonnée** C++/déclaration Rust/capacité Rust/tests/artefact. Garder l’ordre des cinq anciens champs. Ne jamais livrer un appelant allouant cinq valeurs avec un producteur qui en écrit douze.

## Changement limité recommandé

L’implémentation actuelle dans `native/ffi/mlx_bridge.cpp:147` appelle déjà `proc_pid_rusage` V4. Ajouter un `task_vm_info_data_t` et cinq champs au même instant de collecte ; aucune opération MLX nouvelle. Ajouter un emplacement de validité et, si souhaité, le start-abstime via la même lecture rusage existante. PID peut venir de Rust sans élargir encore la CABI. `native/src/array.rs:329` décode les nouveaux champs vers `Option<u64>` en fonction du masque. Les sérialisations `complete.stats.memory`, `pong`, événements de cache et tests existants profitent alors des champs additifs.

Vérifier que `ready` contient réellement une baseline mémoire (actuellement la fonction de stats est appelée à `complete`, `pong` et plusieurs événements d’état ; ajouter le seul champ `memory` à `ready` si absent). Le point de fin doit être conservé avant toute sortie/kill ; aucune lecture différée d’un PID disparu.

Pour vérifier sans stub OS : tests Rust du vrai décodage des valeurs/masques/bornes ; tests C++ du vrai encodage avec struct SDK initialisé et count Rev0/5/6, compte négatif, zéro valide. Un petit helper commun n’est justifié que si c’est l’encodage réellement appelé en production. Kani sur le décodage réellement employé ; tentative CBMC sur la vraie FFI (limite Apple libc++ déjà connue, à conserver honnêtement). Une preuve de conversion ne prouve pas le noyau ni l’attribution des Swapouts.

## Politique future compatible avec la demande utilisateur

Deux protocoles explicites : 75M strict inchangé disponible ; 77 « swap système observé » autorisé seulement après validation des champs natifs.

- Ne pas effacer les +49 479 680 B ni réécrire l’ancien bridge échoué. Nouvelle campagne et nouveaux seuils documentés.
- Préenregistrer un plafond de Swapouts global **non nul** et observer la courbe système. Ce plafond limite les perturbations globales ; il n’attribue pas ces écritures au moteur.
- Préenregistrer footprint maximale moteur, cap de balance compressée et cap de **delta compressed_lifetime**. Contrôler le delta `ledger_swapins` du vrai processus (on peut commencer par zéro ou un petit budget justifié), plus décompressions non saturées. Pas de seuil universel choisi par cette revue.
- Calculer les différences depuis les snapshots du même processus avant/ready puis avant/après chaque requête/warmup/mesure. Refuser compteurs inversés, champs nécessaires absents ou identité changée. Égalité `compressed` avant/après seule est insuffisante : le lifetime détecte les allers-retours.
- Conserver pageins comme diagnostic ; **ne pas remplacer Swapins par pageins**. La lecture SSD intentionnelle et le pagecache produisent du travail légitime.
- Conserver exactitude des sorties/états, pics et métriques déjà présentes de vérification/draft/commit, comparateurs alternés et critères de dérive. Si des latences par bloc sont déjà disponibles, examiner leur queue sans nouvel `eval()` ; une moyenne TPS seule peut cacher du thrash.
- Les snapshots aux frontières ne peuvent pas couper immédiatement au milieu d’une requête ; ils doivent classer l’échec/inconnu si le modèle meurt avant le dernier snapshot. Le moniteur global borné reste une protection, même sous politique assouplie. Pas de nouvel échantillonneur natif nécessaire au prototype minimal demandé.

Qualification exacte recommandée : « swap global observé ; critères de pression du moteur respectés sur les frontières mesurées ». Éviter « le modèle n’a pas swappé » : un ledger de swap-ins ne mesure pas les écritures, et des pages déjà compressées avant la phase peuvent être écrites sans nouveau crédit de compression observable pendant cette phase.
