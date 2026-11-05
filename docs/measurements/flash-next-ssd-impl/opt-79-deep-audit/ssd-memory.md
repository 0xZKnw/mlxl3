# 79R-B — SSD, cache, GPU Hot et mémoire

Revue statique du 23 octobre 2026. Tête documentaire `88cbfd2232ab23a0a7ca1d8dc312792347ca9d31`, runtime inchangé depuis `89174fccc37af8fa6cc37f353ee95d14b584dee7`. Sources lues dans `/Users/justin/.codex/worktrees/flash-splash60/mix-stq1_0`. Aucun code, index ou journal du dépôt modifié ; aucun modèle, test, build, proveur ou benchmark lancé. Les sept prototypes 72/73 ont été retirés : ils ne sont ni réactivés ni recommandés pour une relance identique.

Les observations ci-dessous distinguent un travail effectivement présent dans le graphe ou le programme d'un gain de débit. **Aucun nouveau gain de tok/s, aucune fuite ou course numérique n'est démontré par cette revue.** Les pistes 1–5 touchent le meilleur chemin actuel sans prélecture ; 6 ne concerne que le mode expérimental de prélecture. Le point 7 concerne les diagnostics après une erreur.

## Parcours et contrats conservés

`main.rs::native_bridge` → `bridge_generate` → `Session::advance` / `FlashNext::verify_draft` → préparation des couches Qwen → `SsdMoe::forward_draft_rows` → `forward_rows_prepared` → `ExpertCache::reserve` → `fill_many` → `Array::read_records` → `mlxl3_arrays_preadv` → `read_disk_records` / `preadv`.

Avec prélecture69 active, le vérificateur passe par `forward_draft_lookahead`; la prédiction concerne au plus deux lignes, puis `next.prefetch` prépare les lectures du pool de quatre workers. `ReaderPool::wait_demand` réutilise une complétion demandée dans une échéance commune nominale de 1 ms ; un timeout conserve son lease d'écriture. Le repli demande un autre slot et la publication tardive du doublon est rejetée. Ce mécanisme existe déjà et n'est pas une découverte79.

Le snapshot Hot est construit **avant** réservation/remplissage Cold. Il exclut les slots pending. La réservation protège tous les IDs réellement routés ; le repli64L comptabilise aussi la demande des Hot réutilisés. `ExpertFill` invalide les clés sur erreur, sans libérer les écritures pending. Les owners MLX et le fichier d'un `PreparedRead` restent détenus sur le thread d'inférence jusqu'à complétion ou jonction. Le `Drop` du pool ferme les émetteurs puis joint les workers avant de libérer les owners. Aucun cycle de propriété ni abandon d'une écriture sur un buffer libéré n'a été trouvé dans ce parcours.

`FlashNext::reset` conserve volontairement les banques et leur cache d'experts, mais réinitialise les états causaux et les caches DFlash. La présence de banques après une requête n'est donc pas une fuite. La garde75 se situe dans le superviseur, pas dans l'allocateur du produit. Les compteurs77 distinguent la mémoire logique compressée, l'empreinte et le swap global ; un delta ledger_swapins nul ne prouve pas l'absence d'écritures privées en swap.

## 1. Convertir avant de dupliquer les activations routées

**Nature : redondance de graphe confirmée ; optimisation non mesurée, nouvelle petite piste.** `native/src/ssd_moe.rs:523–529` construit les indices `[0×10,1×10,…]`, fait `x.take`, puis convertit en FP16. Les entrées FP32 du chemin Qwen créent ainsi une sortie Gather de `10×B×2560` FP32 avant la conversion.

Pour B2 : intermédiaire Gather FP32 204 800 octets, puis sortie FP16 102 400 octets. Convertir les deux lignes d'abord produit seulement 10 240 octets avant le Gather FP16. Différence nominale des intermédiaires : **194 560 octets par couche**, soit 9 338 880 octets sur 48 couches par passage B2. Ce n'est ni une baisse mesurée du pic du modèle, ni une économie permanente de RAM. Pour FP16, l'ordre actuel peut déjà éliminer le cast identitaire.

Petit changement proposé : déplacer uniquement `.astype(Float16)` devant `take`, sans remplacer simultanément les indices par un broadcast. C'est une conversion élémentaire suivie d'une duplication, sans modification des sommes EXL3. La FFI appelle directement `mx::take` et `mx::astype`, sans fonction `mx::compile` commune. Dans les sources primaires MLX0.32.2 déjà conservées pour78, Take devient Gather+squeeze, AsType appelle `copy_gpu`; la donation du copy requiert la même taille d'élément et ne réemploie donc pas le buffer FP32 pour FP16.

Oracle discriminant futur : comparer les octets du vrai `routed_input` à une duplication CPU indépendante après conversion, B1/B2/B3/B4/B8, FP16/FP32, strides, zéros signés, valeurs proches des frontières FP16 et valeurs finies représentatives. Puis MoE froid/mixte/chaud et préfixes/logits/états réels. Vérifier les allocations/dispatchs avant tout tok/s. Aucun essai dédié à ce déplacement n'a été retrouvé dans les rapports56/64/65/69/70/72/73/74 examinés ; ce n'est pas shared batching68 ni early_shared73.

## 2. Réunir Hot et Cold sans concaténer toutes leurs lignes

**Nature : copies présentes confirmées ; backlog74 A06, non mesuré.** `ssd_moe.rs:976–1002` concatène Hot et Cold, reconstruit un vecteur d'ordre sur CPU puis fait `take`. Cela reste actif sur **B2** : il suffit d'un snapshot Hot non vide et d'au moins une contribution classée Cold. En Hot anticipé, le tableau Hot contient déjà toutes les positions masquées, y compris les emplacements Cold ; ceux-ci sont copiés une première fois dans la concaténation puis écartés par `take`.

Petit changement distinct de65/57 : un seul copieur/choisisseur exact lit, pour chaque position de sortie, soit la même position Hot, soit l'origine Cold compacte. Il ne partage aucun décodage EXL3 et ne change aucun dispatch d'expert. Garder ensuite **exactement** `astype(x.dtype)` → multiplication par les poids déjà normalisés → reshape `[B,10,H]` → somme dans l'ordre de rang existant (`:1010–1014`). Ne pas fusionner la réduction ou changer l'ordre FP32 dans ce premier palier.

Oracle futur : tableau de valeurs Hot/Cold distinctes, toutes les partitions d'une petite cohorte et partitions réelles B1–B8, sorties byte exactes, cas zéro Hot sélectionné malgré snapshot non vide, tout Hot, pending et repli64L. Compléter par les préfixes et bridge. Le point était explicitement proposé dans74 ; ni le regroupement65/66 avec concat/take, ni le compact57 rejeté ne qualifient ce copieur isolé. Ne pas présenter ce backlog comme une découverte nouvelle.

## 3. Deux lectures de métadonnées de fichier par expert froid

**Nature : redondance CPU confirmée ; backlog74 A22/A23/A24 précisé.** Pour **N records experts distincts** demandés par `fill_many` :

- `ssd_moe.rs:296–304` appelle `TensorInfo::source` une seule fois par expert, pas une fois par segment. `checkpoint.rs:96–108` fait alors N `File::metadata`.
- `Array::read_records`, `array.rs:785–807`, refait N `File::metadata` sur les mêmes handles pour contrôler le record complet.
- `mlxl3_arrays_preadv` et `read_disk_records` n'ajoutent pas de métadonnées fichier. Les 12 destinations sont des segments mémoire du même record contigu.

Total : **2N appels File::metadata par batch** (famille fstat côté OS), indépendamment des 12 segments. B2 entièrement froid : N≤20, donc au plus40 par couche,1920 sur48 couches ; OFF froid : N≤10, donc au plus960. Ce sont des bornes statiques, pas les comptes réellement observés. En prélecture, source+prepare_read donnent également deux contrôles de taille par expert, avec un `try_clone`/dup supplémentaire retenu jusqu'à la fin de lecture.

Fix à borner : séparer le calcul vérifié de l'offset de la vérification de taille, et vérifier une fois par **handle ouvert distinct dans le batch** tous les intervalles complets avant la première écriture. Ne pas conserver aveuglément la taille du fichier au chargement. `read_header` épingle un `Arc<File>` (`checkpoint.rs:58–60`) : un remplacement atomique du pathname doit continuer de lire l'ancien inode inspecté. Réouvrir le chemin ferait perdre cette protection. Une troncature entre deux batches doit toujours être détectée avant écriture; une troncature après validation reste une erreur de lecture, avec invalidation des clés et jonction de tous les lecteurs.

Tests discriminants futurs : remplacement atomique du pathname après inspection, troncature avant le batch, second record hors bornes avec premier destination inchangée, erreur/short read après validation et reprise exacte. Garder le contrôle existant `disk_batches_reject_aliases_and_validate_every_record_before_writing`. Mesurer séparément préparation CPU et I/O. Aucun cache de taille persistant ni gain n'est recommandé sans ce contrat.

## 4. Neuf collections Rust reconstruites avant chaque batch froid

**Nature : préparation répétée confirmée ; backlog74, optimisation conditionnelle.** `fill_many` construit `sources`, `destinations`, `views` (`:296–312`). `Array::read_records` construit `ptrs`, `fds`, `offsets`, `indices`, `starts`, `lengths` (`array.rs:809–826`) : **neuf Vec collectés par batch non vide**, avant les vecteurs C++ de pointeurs, capacités, records et iovecs (`mlx_bridge.cpp:290–312`). Les mêmes sept banques et le schéma des12segments sont stables pendant la vie du MoE. `buffer.byte_len` reste un calcul Rust, pas une synchronisation GPU.

Petit mécanisme : réutiliser un espace de travail borné par instance pour ces petites listes et leurs capacités, sans persister des payloads d'experts ni créer un plan pour chaque expert×slot. Une première version peut ne recycler que les listes Rust; garder la validation globale des alias et des erreurs avant écriture. L'API PreparedRead existe déjà pour la prélecture : l'ajouter partout ne constitue pas à lui seul un mécanisme nouveau et peut ajouter des owners/dup inutiles.

Oracle futur : comparer métadonnées de batches réels, n=0/1/20/80/512, workers1–4, banques remplacées en fixture, erreurs au dernier record et alias via deux handles MLX. Tester absence de croissance après appels répétés et invalidation sur changement de banques. Il faut d'abord distinguer le coût CPU exclusif de l'attente SSD; aucun pourcentage de débit ne découle du nombre de Vec.

## 5. Affectation statique des records aux lecteurs de demande

**Nature : limite d'ordonnancement confirmée ; backlog56, effet non mesuré.** `native/ffi/disk_reader.h:128–144` affecte les jobs `i = worker; i += workers`. Un worker ayant deux records lents ne partage pas son second record avec un worker qui a fini ses records rapides. Des records de même taille peuvent avoir des latences différentes selon leurs pages déjà en cache et les erreurs. Le dernier record d'une cohorte peut donc déterminer toute sa durée.

Mécanisme futur distinct du simple ajout de threads : même nombre1–4 de lecteurs, index atomique de prochain record, récupération par le premier lecteur disponible. Continuer à terminer tous les records et joindre tous les workers sur erreur; ne pas arrêter la validation après le premier succès. Le test C++ actuel impose `timings[i].worker == i%workers` : c'est une assertion de la stratégie historique à remplacer, pas une propriété de sûreté à conserver.

Test discriminant futur : source injectable de lecture avec premier job d'un worker volontairement lent et plusieurs suivants indépendants, compter chaque job exactement une fois, destinations disjointes, erreur partielle et tous lecteurs joints. Puis analyser les timings déjà possibles par job sans attribuer leur durée à du GPU exclusif.56 identifiait déjà l'affectation statique sans la remplacer. Aucun gain n'est établi; à1–4 misses, cette modification peut ne rien apporter.

## 6. Le nombre de lecteurs n'est pas un budget global demande+prélecture

**Nature : concurrence maximale structurelle, pas une course ; mode prélecture seulement, faible priorité actuelle.** `ReaderPool::new` crée quatre workers fixes (`ssd_prefetch.rs:98–117`), indépendamment de `MLXL3_SSD_READERS`, qui borne les lecteurs synchrones de demande (`ssd_moe.rs` load et `disk_reader.h:117`). Des lectures spéculatives d'une autre couche, ou non terminées après l'échéance nominale, peuvent rester en vol pendant quatre lectures de demande : **jusqu'à huit lecteurs logiques simultanés**. Cela ne signifie pas huit I/O physiques observées. Le meilleur B2 sans prélecture ne crée pas ce pool.

Un futur ordonnanceur à budget commun pourrait suspendre l'admission spéculative quand des demandes urgentes arrivent, sans annuler ni réaffecter les destinations déjà écrites. Ce serait un nouveau mécanisme architectural, pas réactiver weighted/late/bounded/unused/lfu/packed. Son intérêt est conditionné à des timings montrant une contention réelle;69 et72 ont déjà montré que davantage de lectures utiles/logiques ne suffit pas à gagner du débit. Aucun essai identique de leurs flags n'est proposé.

Oracle futur : lecteurs OS instrumentés sans GPU, compter les pointes d'activité demande/prélecture, priorité d'admission, absence de famine et jonction après erreur/cancel; puis protections de slots et oracles physiques. Ne pas transformer le nombre4 des plans historiques en garantie de quatre lectures totales.

## 7. Les octets synchrones d'un batch échoué disparaissent des compteurs moteur

**Nature : limite d'observabilité confirmée sur erreur, pas un défaut de logits ni un boost.** `fill_many` n'incrémente `read_bytes` qu'après `result?` (`ssd_moe.rs:319–325`). Si un `preadv` échoue après une partie de la lecture, tous les lecteurs terminent quand même, mais aucun octet de ce batch n'arrive dans `FlashNext::io_stats`. Le pool asynchrone ajoute, lui, `bytes_read` même en échec (`ssd_prefetch.rs:107–111`). Le test C++ existant `disk_reader.cpp:103–125` démontre que des siblings réussissent et que le record tronqué peut avoir lu un octet avant l'erreur.

Un profil comprenant erreur puis reprise ne doit donc pas interpréter `io_stats().2` comme tous les octets d'API effectivement lus. Les campagnes réussies restent évaluables selon leur contrat; leurs octets ne sont de toute façon pas les lectures NVMe physiques. Petit correctif éventuel : remonter séparément les octets réellement lus dans la FFI synchrone sur succès/échec, en conservant le statut erreur, ou renommer/documenter explicitement le compteur de records réussis. Une somme du profil test-only ne suffit pas au runtime produit.

Test futur : batch avec7siblings réussis et1lecture partielle échouée, vérifier compteur exact non nul malgré erreur, aucune clé Cold valide, aucun worker restant, puis retry exact. C'est une correction de diagnostic pour les échecs, à séparer des chantiers TPS. Les erreurs partielles sont déjà testées en C++; le transfert de leur compteur jusqu'à `io_stats` ne l'est pas dans le chemin lu.

## Priorité proposée et exclusions

Premier petit essai possible : **1**, car il change seulement l'emplacement d'une conversion élémentaire et touche OFF/MTP/B2 sans prélecture. Ensuite **2**, si le coût du copieur ressort, puis **3–4** après isolation du temps hôte. **5** seulement si les timings par worker montrent une traîne. **6** est une architecture conditionnelle pour un mode déjà moins bon : ne pas y consacrer une nouvelle campagne sans preuve de contention. **7** améliore les diagnostics, pas le débit nominal.

Pas de fuite démontrée, pas de cache expert réduit, pas de F_NOCACHE réintroduit, pas de promesse zéro swap. Le pic70M au chargement a déjà été corrigé et les mesures77 restent indispensables pour distinguer compression propre et swap système. Les scans de cache/carte inverse, fréquence par occurrence et gates résidentes sont déjà identifiés74/72R : ils ne sont pas présentés ici comme découvertes nouvelles. Le saut vers15tok/s reste non démontré.

Sources historiques lues :56,64L/64D,65,69R/69–70/70R,70M-E dans opti,72R et retrait72W-STOP,74review/first-review,75README,77README. Sources MLX primaires locales de78 : tag0.32.2 commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, `mlx/ops.cpp`, `mlx/backend/gpu/primitives.cpp`, `mlx/backend/metal/copy.cpp`, `mlx/backend/common/copy.h`. Une tentative web de récupérer indexing.cpp à ce commit a renvoyé cache-miss; aucune conclusion supplémentaire n'en est tirée. Pas de nouvelle affirmation sur les HEADs actuels Strata/Splash : l'autre sous-revue79R-D traite les amonts.
