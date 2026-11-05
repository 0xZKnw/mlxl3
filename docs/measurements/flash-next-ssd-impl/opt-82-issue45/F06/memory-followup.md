# Suivi mémoire F06 — lecture seule, 2026-10-25T00:00:48.339453+00:00

**Aucune rétention permanente de plusieurs gigaoctets ni double payload CPU/GPU n’est démontré dans la voie DFlash2 B2 mesurée.** La compression propre au processus est réelle et reste hors des anciens plafonds77B ; elle n’est ni une preuve de fuite, ni assimilable au swap disque. Aucun modèle, test, benchmark, build ou prouveur lancé par cette revue ; aucune édition du moteur ou du worktree.

L’historique69L a déjà contrôlé64 cohortes/256 lectures : pending≤4, propriétaires1→5→1, absence de croissance MLX et libération Weak. Cela ne prouve pas tout le moteur.70M a corrigé l’initialisation des banques libres (zeros/eval remplacés par buffers bruts), avec contrats de couverture, poison, erreurs/retry puis vrais chargements. Les hypothèses anciennes ne doivent pas être rejouées sans mécanisme nouveau.72/73 et77 constatent déjà de la compression privée sous un pic≈7,94Go, sans attribution exclusive ni résolution mémoire ; ces faits restent distincts des prototypes72/73 retirés.

## Ce qui reste effectivement en mémoire

- **Banques d’experts : 3174942528B**,48 couches×53 slots×1248012B. Sept tableaux par couche, allocation brute unique (`ssd_moe.rs:198–211`). Les lectures `fill_many` écrivent dans leurs12 segments finaux (`ssd_moe.rs:299–324`, `mlx_bridge.cpp:282–315`). `TensorInfo` garde métadonnées et `Arc<File>` (`checkpoint.rs:18–30`), pas le payload de tous les experts. C’est une réserve volontaire de3Gio, pas une seconde copie détectée. Une fraction peut être froide entre ses usages ; la seule proximité entre son volume et les3,09Go compressés ne permet pas d’attribuer ces pages aux banques.
- **Poids permanents : une représentation par projection.** `checkpoint_array` utilise `Array::from_file` pour les tenseurs usuels (`linear.rs:76–92`) ; la FFI alloue le tampon MLX final et `pread` y écrit directement (`mlx_bridge.cpp:94–111,261–267`). Le cas BOOL utilise un petit staging Rust. EXL3 matérialise trellis/scales/bias à la construction (`linear.rs:245–258`) et les poids denses F16 sont évalués (`lfm2.rs:65–87`) : pas de graphe de chargement complet manifestement conservé. Les clones FFI créent un handle `mx::array`, sans copie explicite de payload (`mlx_bridge.cpp:179–180`). Cela ne constitue pas une preuve de l’allocateur MLX.
- **Draft Q4 : pas de BF16 complet gardé en parallèle.** `flash_draft.rs:451–474` insère chaque matrice soit dans `quantized` après évaluation des trois sorties, soit dans `weights`. La matrice BF16 source reste un temporaire de chargement ; la carte ne contient pas les deux. Les deux codebooks BF16248320×256 restent volontairement résidents (254279680B au total), car ils sont utilisés par le sélecteur. Le cache circulaire5 couches est borné à5×2×8×2048×128×2=41943040B (`flash_draft.rs:856–893`), pas une chaîne KV sans limite dans ce mode.
- **Pas de gros propriétaire caché dans ce parcours.** Flash DFlash retire le `PromptCache` des modèles Qwen et repart par `prefill_dflash2` (`main.rs:1822–1827`) ; ce cache Qwen ne capture d’ailleurs que `NativeChatModel::Qwen` (`main.rs:1761–1764`). Lookahead0 implique aucun `ReaderPool` (`qwen4.rs:1417–1425`). `ResidentGate=0` n’ajoute pas sa copie dense FP32 (`ssd_moe.rs:214–233`). Les états de vérification sont détenus par le bloc et consommés/abandonnés au commit/drop (`flash_draft.rs:1020–1075`), pas accumulés dans une liste de tous les tokens. Le champ MoE`last` conserve une sortie de travail pour protéger les lectures bancaires, puis `take/eval` au prochain usage ou `drain` (`ssd_moe.rs:539–542,1026,1184–1188`) ; ce n’est pas une banque dupliquée par token.

## Les mesures réelles déjà disponibles

Chaque ligne ci-dessous compare **la même identité PID/start** entre warmup et mesure, pas des processus différents. Les compteurs incluent le préfill plus le décodage de la deuxième requête ; ils ne mesurent pas le seul decode.

| Bras | Compression après mesure (B) | ΔMLX actif (B) | Δempreinte (B) | Δcompression courante (B) | Δdécompressions (pages) | Δledger swap-ins (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOLD-0-H | 2085879808 | 2048 | 147432 | 174522368 | 30848 | 0 |
| HOLD-1-C | 2430713856 | -4096 | -16384 | 155648000 | 30455 | 0 |
| MAIN-0-H | 3088924672 | -8192 | 3031040 | 165986304 | 30960 | 0 |
| MAIN-1-C | 2285469696 | -4096 | -2916352 | 130596864 | 30730 | 0 |

Le poids MLX actif reste≈7,144Go et varie de−8192 à+2048B entre requêtes ; l’empreinte reste≈7,74–7,76Go et varie d’environ−2,92 à+3,03Mo. Cela ne ressemble pas à une croissance de payload de3Go entre warmup et mesure. En revanche, des pages déjà allouées passent au compresseur puis sont décompressées : activité réelle, dont le coût sur le chemin critique n’est pas isolé. Les différences de compression entre H et C ne constituent pas un effet causal F06 démontré.

Au ready, MLX actif vaut6456399040B, empreinte≈3,75Go ; après génération MLX actif≈7,144Go et empreinte≈7,75Go. La différence actif/empreinte au ready est cohérente avec les banques brutes encore partiellement non touchées depuis70M ; elle ne prouve pas que des octets actifs sont physiquement résidents. L’augmentation ready→génération comprend le chargement ultérieur du draft et les états/KV : ce n’est pas à elle seule une fuite. Le cache libre MLX reste≈64Mio ; `clear_cache` ne peut pas libérer les7,144Go actifs.

La compression logique décrit des octets avant compression, les décompressions sont des pages éventuellement revisitées, et le ledger swap-ins mesure une autre catégorie. Ne pas les additionner à l’empreinte, les assimiler aux octets NVMe, ou présenter ledger0 comme preuve de zéro swap-out privé. Les observations globales de campagne (0 nouveaux outs signalés par le superviseur) ne donnent pas la propriété de ces pages.

## Prochain pas minimal proposé, non exécuté

Il n’y a pas de correctif de fuite justifié par cette lecture. Avant de modifier les banques/cache, il faudrait attribuer la compression aux régions effectivement allouées (banques, poids permanents, draft, états) et distinguer la fenêtre ready→préfill→decode avec le même processus, sans `eval` ajouté pour le profilage. Les statistiques agrégées actuelles ne donnent pas cette attribution. Garder les comptes par phase déjà disponibles évite de confondre une recréation de KV après reset avec une rétention.

Une inefficacité connue reste réelle ailleurs : `configure_dflash2` recharge les47 matrices quand seules les options de politique changent (`flash_draft_runtime.rs:231–269`, piste76T différée). Elle ne s’applique pas aux deux requêtes B2 fixes aux Options identiques (retour précoce:244–249) et ne doit pas être utilisée pour expliquer cette compression stable. Réactiver76, diminuer le cache, fermer des applications ou modifier les seuils n’est pas autorisé ni proposé par ce suivi.

Sources et chiffres sont ceux du gel F06 ; aucune conclusion de performance ni attribution causale nouvelle. Les anciens plafonds77B restent échoués, distincts de l’empreinte sous12Go et des observations de swap de la campagne.

## Complément bridge réel terminé

Le bridge43 est aussi relu, sans relance : `complete`, `parity=true`, trois phases attendues et jointes, sampler joint. Le fichier contient70 occurrences mémoire en comptant les références dupliquées ; son flux`events` contient69 occurrences, soit36 événements munis de mémoire après déduplication des `round_stats[0].memory`/`stats.memory` identiques du même événement, sous une seule identité PID90459/start1395145734506. Pic vie7938120112B, compression courante maximale2985656320B et pic3008102400B ; ledger cumulatif0 à tous ces points. Les plafonds77B restent dépassés. Le compteur cumulatif de compression et les décompressions couvrent toute la succession de requêtes/modes/annulations, pas une requête ni des octets actifs nouvellement retenus.

Le superviseur relève **0 nouveau swap-out global**, mais **131072B de swap-in global pendant le bridge** ; pré/post0. Cela n’invalide pas la parité et ne prouve pas une faute privée du moteur. Ne pas résumer ce bridge par «zéro swapin/out» ; son ledger privé observé reste0 et mesure une autre portée. La dernière observation mémoire du flux est l’événement dflash_status42 : empreinte7557306632B, compression2852896768B, décompressions724345. Ce niveau n’est pas une preuve de fuite ni d’absence de fuite ; aucun inventaire par adresse ne permet d’identifier les pages compressées.
