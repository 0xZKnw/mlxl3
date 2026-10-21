# Flash Next SSD62H : GPU Hot pour1 à8 lignes

21 octobre 2026. Étape issue de la [revue61](flash-next-ssd-splash-61-audit-2026-10-21.md), base publiée15c4b85e (moteur précédent95d08366/dc33e3e4). **Correction numérique validée ; aucun gain global de débit qualifié.** Aucun remplacement de l'app installée.

## Changement et limites

`MLXL3_SSD_GPU_HOT_WIDE=1` étend le chemin hybride aux cohortes de3 à8 lignes ; le helper GPU accepte maintenant80 routes. L'option reste expérimentale, désactivée par défaut, et demande aussi `MLXL3_SSD_GPU_HOT=1`, overlap1 et lookahead0. Le chemin Hot attend toujours le dernier utilisateur des banques avant leur réemploi et conserve sa dépendance GPU en cas d'erreur.

Le découpage utilise les slots disponibles, en excluant les prélectures non publiées. Les prélectures terminées sont publiées avant ce calcul. Avec53 slots et4prélectures en vol, une demande B5 est donc découpée4+1. **Le découpage reste conservateur selon10 routes par ligne** : ce changement ne mutualise pas encore les kernels des experts répétés et n'implémente pas le plan par union réelle des routes.

Les sources runtime de cette étape concernent uniquement Flash SSD et son helper de slots ; aucun changement du calcul LFM2. Les [identitésv1](measurements/flash-next-ssd-impl/opt-62-wide-hot/identity-v1.json) et [v2](measurements/flash-next-ssd-impl/opt-62-wide-hot/identity-v2.json) fixent146 sources et les SHA256 des binaires/archives. V2 ajoute seulement le test physique des claims pending ; **le CLI release v2 est bit à bit identique àv1**, utilisé pour les tests complets.

## Vérification inspectée

| Contrôle | Résultat |
| --- | --- |
| `cargo test --locked --no-default-features` |112 passés :96 lib+16 contrats ;3 ignorés. |
| Mutation `available_slots=capacity` | Le test de régression échoue comme attendu ; source restaurée, test repassé dans la suite MLX. |
| `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings` | Succès v1 etv2. |
| `cargo test --locked --release --features mlx,chat` |141 passés :113 lib+12 CLI+16 contrats ;140 ignorés. Les tests physiques sélectionnés séparément sont décrits ci-dessous. |
| `cargo build --locked --release --features mlx,chat --bin mlxl3-rs` et fmt | Succès. Avertissement de compatibilité future déjà présent dans `block0.1.6`. |
| `python -m pytest -q` |915 passés/4 sautés ; dépendances `ponyexl3.convert` et modèle Ling absents. |
| Kernel slots physique |88 formes,264 masques froid/mixte/chaud ; FP16/FP32, strides et contrôle mauvais slot ; égalité bit à bit avec projections indépendantes. |
| Plan résident physique |28 plans couvrant514IDs, dont IDs invalides ; strides et mauvaise carte contrôlés. |
| Diagnostic asynchrone B3–B8 |12 cas OFF/ON ; le Hot ralenti reste en cours aux frontières routes et lecture urgente ; sorties bit à bit identiques. Ces délais artificiels ne sont pas des mesures tok/s. |
| MoE physique |1..8 lignes,9capacités10..512 slots×6 combinaisons overlap/lecteurs, caches froid/chaud, répétitions, formes invalides, lecture échouée et reprise ; oracle indépendant bit à bit. |
| Claims de prélecture |3 cas :53 slots B5/B8,80 slots B8 ;4 claims non publiés,2cohortes attendues, sorties et propriété des slots préservées. Simulation des claims réels ; pas une preuve des courses I/O concurrentes. |
| Vérificateur cible |36 préfixes, largeurs1..8 : logits248320, captures et états bit à bit égaux au séquentiel ; erreurs et abandon couverts. |
| Runtime natif |20 cas BF16/Q4/ring/adaptatif×budgets1/2/3/17/64 ;136 états comparés par cas, annulation et reprise. |
| Bridge/CLI réel |43 contrôles, parité vraie, processus/lecteur joints. |

La campagne qualitév1 a exécuté8 processus sélectionnés explicitement ;v2 ajoute1 processus ciblé. Les [matrices](measurements/flash-next-ssd-impl/opt-62-wide-hot/quality-v1/run.json), [résultats](measurements/flash-next-ssd-impl/opt-62-wide-hot/quality-v1/run-result.json), [complément](measurements/flash-next-ssd-impl/opt-62-wide-hot/quality-v2/run.json) et commandes archivées dans le même dossier permettent de reproduire les sélections. Un test sélectionné à zéro est refusé par le superviseur existant.

## Vérification bornée et chemins non prouvés

Kani0.68.0 passe le nouveau harnais `wide_cohorts_fit_every_pending_pattern_in_eighty_slots` : tous les80booléens pending sont symboliques,574 obligations sans échec,7 inatteignables,3/3 couvertures, unwind81 avec assertions d'unwinding conservées. Propriétés : décompte exact des places disponibles et `10*lignes+pending<=80`, y compris7 lignes/4pending,8 lignes libres et80pending. Cela ne prouve ni les autres capacités de ce harnais, ni Metal, MLX, les threads ou la génération entière.

Trois harnais liés passent aussi : bornes des cohortes pour `usize` symbolique, admission spéculative24 slots/au plus4pending et parseurs sur chaînes UTF-8 symboliques de0 à5 octets, dont le nouveau contrôle0/1. Les logs conservent les obligations et couvertures individuelles.

Tentative réelle MLX : la première commande omettait `--tests` et n'a trouvé aucun harnais. La commande corrigée `cargo kani --tests --lib --features mlx,chat --harness actual_wide_hybrid_hot_demand_entry` échoue par ICE `kani-compiler/src/intrinsics.rs:243`. **Parcours MLX/Metal non prouvé.** CrossHair est absent ; les commandes ad hoc de collecte sont archivées, pas présentées comme prouvées. Les C++/Metal n'ont pas changé dans62H.

Le workflow PR/push ajoute le nouveau harnais CPU à `kani-flash-contracts`. La CI du commit source `2eccb469` a été inspectée : régressions, Linux, macOS MLX et Kani Flash réussis. Le job Kani global échoue en fin de run sur l’ancien harnais `cache_allocation_is_optimal_and_exact_on_two_layers_two_levels`, comme les baselines précédentes ; le log s’interrompt sans contre-exemple ni verdict de ce harnais. Le workflow natif global reste donc en échec. Les snapshots et l’extrait du log exact sont archivés dans le dossier62.

## Mémoire et protocole de débit

Mac M5/24 Gio, secteur100%, applications laissées ouvertes, MLX0.32.2. Cible2,05bpw `65c89531`, draft5 couches `a22a1744`, cache experts3 Gio/53 slots, allocateur512 Mio,4lecteurs, overlap/Hot1, lookahead0. Origine VM propre à62 explicitement enregistrée : la campagne60 clôturée conserve son bilan précédent et n'est pas réécrite.

La qualitév1 atteint8 947 522 832 octets de pic physique pour le processus runtime, distinct de la mémoire MLX et du RSS. Zéro nouveau Swapout durant les8 processus et le postflight30s ; garde VM cumulative256 Mio/2s. Le complémentv2 utilise la même origine VM. Les fichiers `campaign-result.json` et `postflight.json` gardent le statut effectif ; un échec tardif invalide la qualification.

Préenregistrement du débit dans `opti.md` avant les changements : OFF puis MTP1 comme témoins, DFlash Q4/ring fixe3 puis adaptatif7. Pour chaque mode : MAIN HCCH puis HOLD HCCH,128 tokens/127décodés, un warmup et une mesure par processus neuf ; seul WIDE0/1 varie au sein d'un mode et tous les bras utilisent le même SHA de binaire. Qualification DFlash : au moins1% dans chaque paire, dérive témoin≤5%, étendue des gains≤3points, sorties identiques, pic≤12e9 octets et0nouveau Swapout jusqu'au postflight. Les résultats ci-dessous ne qualifient aucun gain global.


## Débit final MAIN et HOLD

Les32 bras v2 sont terminés et joints, avec64 réponses (warmup compris) identiques au témoin OFF, texte et empreinte des128 tokens. Même binaire CLI v2 dans tous les bras, SHA256 `9d45a5869a6f8de13b5b4cd1ab30bf78b8ec39e0735deb188b4fd717c5f23f26` ; WIDE0/1 est la seule différence par paire.

| Mode | MAIN : gains WIDE1 vs0, deux paires | HOLD : gains, deux paires | Pic physique maximal | Décision débit |
| --- | --- | --- | --- | --- |
| OFF | Témoin B1,10,51–10,70tok/s | Témoin B1,10,22–10,23tok/s |7 141 251 000 octets | Chemin WIDE inactif, aucun gain attribué |
| MTP1 | Témoin B2,10,04–10,39tok/s | Témoin B2,9,56–9,59tok/s |8 196 019 456 octets | Chemin WIDE inactif ; dispersion MAIN3,158points |
| DFlash Q4/ring fixe3 |+0,074% /+1,227% |+0,844% /+1,162% |8 423 905 808 octets | **Rejeté** : seuil1% non atteint dans chaque paire |
| DFlash Q4/ring adaptatif7 |+1,323% /+2,838% |+4,855% /+0,847% |8 936 970 768 octets | **Non concluant** : étendue HOLD4,008points>3 |

Les contrôles fixes dérivent de0,256% MAIN/0,076% HOLD ; les contrôles adaptatifs de0,240%/1,856%. Une paire favorable ne suffit pas à qualifier un gain. Les tok/s et données exactes, sans arrondi, sont conservés dans [speed-v2-result.json](measurements/flash-next-ssd-impl/opt-62-wide-hot/speed-v2-result.json) et les rapports individuels des32bras. Aucun15tok/s atteint ; WIDE reste0 par défaut. Pas de poursuite isolée de62 sans nouveau mécanisme.

## Interruptions de collecte et contrôle différé

La première tentative s’est interrompue au troisième bras sur `BlockingIOError: [Errno35] Resource temporarily unavailable` lors de la création d’un sous-processus de surveillance. Après la relance explicitement demandée par l’utilisateur, la v2 a terminé toutes les générations, puis subi le même refus pendant le postflight après11 échantillons/environ20s. Les deux résultats initiaux restent **`failed_postflight`, `parity:false`, `confirmed_gain:false`**. Les comparaisons de64 réponses réussies ne changent pas ces statuts de campagne incomplète.

La cause précise de l’EAGAIN macOS n’est pas établie. Le diagnostic au repos ne montre pas de modèle de cette campagne encore actif ; les processus enfants ont été joints. Aucun réglage système ni application utilisateur n’a été modifié. Le relevé de tous les noms de processus est conservé localement seulement ; le diagnostic public contient les compteurs agrégés.

Une observation tardive distincte de30s,16échantillons, confirme **0 nouveau Swapout** depuis l’origine62 et13 107 200octets de Swapin, secteur100%, sans refus de création. Le collecteur autorisait au plus3tentatives espacées100ms pour un EAGAIN d’observation, erreurs conservées, aucun zéro substitué. [Postflight tardif](measurements/flash-next-ssd-impl/opt-62-wide-hot/postflight-late-v2.json). Ce contrôle ne réécrit pas le postflight initial échoué et ne qualifie pas le gain.

La campagne62 est close. Le projet de sonde sans fork62S est différé avant implémentation pour donner la priorité au vérificateur63. Les résultats, y compris faibles et négatifs, sont conservés ; aucune installation de l’application n’a été faite.
