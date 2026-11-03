# 77T — compteurs de mémoire du moteur

Instrumentation additive locale, aucun changement du calcul du modèle ni du cache53slots/3Gio. Les cinq anciens champs mémoire restent à leur place. Le CABI privé vérifie une capacité de12valeurs avant tout accès ou opération MLX. Les nouveaux champs sont optionnels selon le masque de disponibilité ; zéro est une mesure valide.

Champs JSON : `process_id`, `process_start_abstime`, `process_compressed_logical_bytes`, `process_compressed_peak_logical_bytes`, `process_compressed_lifetime_logical_bytes`, `process_decompressions`, `process_ledger_swapins_logical_bytes`. Ils sont exposés aux snapshots existants et à `ready.memory`, sans nouvel eval ni synchronize. Startabstime est une identité en ticks Mach, pas une durée en nanosecondes ; decompressions sature INT32_MAX ; ledger_swapins n’attribue pas les écritures de swap système.

## Vérification avant priorité débits

- C++ :32combinaisons de countSDK/échecsAPI, plus zéro valide, valeurs signées négatives et maxima ; vrai helper utilisé en production, compilé directement sous ASan/UBSan.
- Rust :159tests passés/159ignorés ; deux nouveaux tests CPU du décodeur/JSON et du vrai CABI null/capacité0/5/11. Clippy strict, build release mlx,chat et fmt passent. Processus joints, buildjobs1 ; logs et commandes dans les fichiers process.json.
- CI : C++SDK sur macOS et deux tests Rust exacts, chacun exige réellement1pass/0fail. Workflow parsé, runGitHub non exécuté.
- Kani réel sur `MemoryStats::from_raw` : **127obligations passent,3/3covers satisfaites,0inatteignable/indéterminée**,12u64arbitraires+PIDu32sansassumptions, unwind13 ;5,69s incluantcompilation, RSSgroupeéchantillonnée594526208B. Vérification bornée du décodeur seulement, pas du noyau/FFI/MLX. CBMC réel sur encodeurSDK : exit6/erreursparseApplelibc++avantobligations, aucunepreuveC++. Mutation isolée retire vm_ok du contrôle de compression : fixture normaleexit0, mutantSIGABRT attendu, restauréexit0 ; prodjamaismodifiée/scratchsupprimé. Voir verification-summary.json et logs.
- Composant physique ignoré : `array::tests::native_process_memory_stats_are_available_and_monotone`, proposé au parent pour exécution isolée après tous les builds. Cet agent ne l’a pas lancé ; le parent l’a validé1/1 dans fresh-controls-v1 avant ses contrôlesGPU, puis a clos le post30 avant les preuves ci-dessus. Aucun autre build/prover/GPU ne doit être lancé pendant sa campagne ou son postflight.

Binaires immuables : ROOT/build/flash-opt-77-process-memory/v1 ; cinq sources et deuxSHA dansidentity-v1.json. L’instrumentation ne constitue pas un gain de RAM ou de débit et ne prouve pas que le modèle ne swappe pas.

RevueSDK/XNU complète et sources primaires épinglées : [research/review.md](research/review.md), extraits/manifestes dans research/. XNUpublicf6217f89 est distinct du noyau localDarwin27.2 ; la disponibilité réelle doit être contrôlée sur le Mac.

## Référence de validation physique et témoins frais

Le parent a exécuté le composant physique77 **1/1 passé**, puis quatre témoins (DFlash2B2 etMTP1 surMAIN/HOLD), avec8réponses128sorties/127decode exactes et post30 clos. Résultats et compteurs propres au moteur : [fresh-controls-v1/summary.json](fresh-controls-v1/summary.json). Ces résultats ne sont pas une répétition par cet agent.

Les plafonds de compression privés préenregistrés77B (512Mio de balance,1Gio de delta lifetime) ont été dépassés par les témoins. Les débits restent descriptifs, sans gain nouveau qualifié et sans modification rétroactive des seuils. Un delta ledger_swapins de zéro ne prouve pas l’absence de pages du moteur écrites en swap. L’ancien incident72 conserve son statut échoué selon son propre protocole.
