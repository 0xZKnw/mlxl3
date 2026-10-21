# Flash Next SSD62H : GPU Hot pour1 à8 lignes

21 octobre2026. Étape issue de la [revue61](flash-next-ssd-splash-61-audit-2026-10-21.md), base publiée15c4b85e (moteur précédent95d08366/dc33e3e4). **Correction numérique validée ; débit non encore mesuré.** Aucun remplacement de l'app installée.

## Changement et limites

`MLXL3_SSD_GPU_HOT_WIDE=1` étend le chemin hybride aux cohortes de3 à8 lignes ; le helper GPU accepte maintenant80 routes. L'option reste expérimentale, désactivée par défaut, et demande aussi `MLXL3_SSD_GPU_HOT=1`, overlap1 et lookahead0. Le chemin Hot attend toujours le dernier utilisateur des banques avant leur réemploi et conserve sa dépendance GPU en cas d'erreur.

Le découpage utilise les slots disponibles, en excluant les prélectures non publiées. Les prélectures terminées sont publiées avant ce calcul. Avec53slots et4prélectures en vol, une demande B5 est donc découpée4+1. **Le découpage reste conservateur selon10 routes par ligne** : ce changement ne mutualise pas encore les kernels des experts répétés et n'implémente pas le plan par union réelle des routes.

Les sources runtime de cette étape concernent uniquement Flash SSD et son helper de slots ; aucun changement du calcul LFM2. Les [identitésv1](measurements/flash-next-ssd-impl/opt-62-wide-hot/identity-v1.json) et [v2](measurements/flash-next-ssd-impl/opt-62-wide-hot/identity-v2.json) fixent146sources et les SHA256 des binaires/archives. V2 ajoute seulement le test physique des claims pending ; **le CLI release v2 est bit à bit identique àv1**, utilisé pour les tests complets.

## Vérification inspectée

| Contrôle | Résultat |
| --- | --- |
| `cargo test --locked --no-default-features` |112 passés :96lib+16contrats ;3ignorés. |
| Mutation `available_slots=capacity` | Le test de régression échoue comme attendu ; source restaurée, test repassé dans la suite MLX. |
| `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings` | Succès v1 etv2. |
| `cargo test --locked --release --features mlx,chat --lib` |113 passés ;130ignorés env2. Ne constitue pas l'exécution des tests GPU ignorés. |
| `cargo build --locked --release --features mlx,chat --bin mlxl3-rs` et fmt | Succès. Avertissement de compatibilité future déjà présent dans `block0.1.6`. |
| `python -m pytest -q` |915passés/4sautés ; dépendances `ponyexl3.convert` et modèle Ling absents. |
| Kernel slots physique |88formes,264masques froid/mixte/chaud ; FP16/FP32, strides et contrôle mauvais slot ; égalité bit à bit avec projections indépendantes. |
| Plan résident physique |28plans couvrant514IDs, dont IDs invalides ; strides et mauvaise carte contrôlés. |
| Diagnostic asynchrone B3–B8 |12cas OFF/ON ; le Hot ralenti reste en cours aux frontières routes et lecture urgente ; sorties bit à bit identiques. Ces délais artificiels ne sont pas des mesures tok/s. |
| MoE physique |1..8lignes,9capacités10..512slots×6combinaisons overlap/lecteurs, caches froid/chaud, répétitions, formes invalides, lecture échouée et reprise ; oracle indépendant bit à bit. |
| Claims de prélecture |3cas :53slots B5/B8,80slots B8 ;4claims non publiés,2cohortes attendues, sorties et propriété des slots préservées. Simulation des claims réels ; pas une preuve des courses I/O concurrentes. |
| Vérificateur cible |36préfixes, largeurs1..8 : logits248320, captures et états bit à bit égaux au séquentiel ; erreurs et abandon couverts. |
| Runtime natif |20cas BF16/Q4/ring/adaptatif×budgets1/2/3/17/64 ;136états comparés par cas, annulation et reprise. |
| Bridge/CLI réel |43contrôles, parité vraie, processus/lecteur joints. |

La campagne qualitév1 a exécuté8processus sélectionnés explicitement ;v2 ajoute1processus ciblé. Les [matrices](measurements/flash-next-ssd-impl/opt-62-wide-hot/quality-v1/run.json), [résultats](measurements/flash-next-ssd-impl/opt-62-wide-hot/quality-v1/run-result.json), [complément](measurements/flash-next-ssd-impl/opt-62-wide-hot/quality-v2/run.json) et commandes archivées dans le même dossier permettent de reproduire les sélections. Un test sélectionné à zéro est refusé par le superviseur existant.

## Vérification bornée et chemins non prouvés

Kani0.68.0 passe le nouveau harnais `wide_cohorts_fit_every_pending_pattern_in_eighty_slots` : tous les80booléens pending sont symboliques,574obligations sans échec,7inatteignables,3/3couvertures, unwind81 avec assertions d'unwinding conservées. Propriétés : décompte exact des places disponibles et `10*lignes+pending<=80`, y compris7lignes/4pending,8lignes libres et80pending. Cela ne prouve ni les autres capacités de ce harnais, ni Metal, MLX, les threads ou la génération entière.

Trois harnais liés passent aussi : bornes des cohortes pour `usize` symbolique, admission spéculative24slots/au plus4pending et parseurs sur chaînes UTF-8 symboliques de0 à5octets, dont le nouveau contrôle0/1. Les logs conservent les obligations et couvertures individuelles.

Tentative réelle MLX : la première commande omettait `--tests` et n'a trouvé aucun harnais. La commande corrigée `cargo kani --tests --lib --features mlx,chat --harness actual_wide_hybrid_hot_demand_entry` échoue par ICE `kani-compiler/src/intrinsics.rs:243`. **Parcours MLX/Metal non prouvé.** CrossHair est absent ; les commandes ad hoc de collecte sont archivées, pas présentées comme prouvées. Les C++/Metal n'ont pas changé dans62H.

Le workflow PR/push ajoute le nouveau harnais CPU à `kani-flash-contracts`. La CI du prochain commit source reste à inspecter. La CI15c4b85e a terminé avec succès la régression ; le workflow natif global n'est pas vert, comme la baseline60. Aucun succès local ne remplace l'inspection des jobs du commit livré.

## Mémoire et protocole de débit

Mac M5/24Gio, secteur100%, applications laissées ouvertes, MLX0.32.2. Cible2,05bpw `65c89531`, draft5couches `a22a1744`, cache experts3Gio/53slots, allocateur512Mio,4lecteurs, overlap/Hot1, lookahead0. Origine VM propre à62 explicitement enregistrée : la campagne60 clôturée conserve son bilan précédent et n'est pas réécrite.

La qualitév1 atteint8 947 522 832octets de pic physique pour le processus runtime, distinct de la mémoire MLX et du RSS. Zéro nouveau Swapout durant les8processus et le postflight30s ; garde VM cumulative256Mio/2s. Le complémentv2 utilise la même origine VM. Les fichiers `campaign-result.json` et `postflight.json` gardent le statut effectif ; un échec tardif invalide la qualification.

Préenregistrement du débit dans `opti.md` avant les changements : OFF puis MTP1 comme témoins, DFlash Q4/ring fixe3 puis adaptatif7. Pour chaque mode : MAIN HCCH puis HOLD HCCH,128tokens/127décodés, un warmup et une mesure par processus neuf ; seul WIDE0/1 varie au sein d'un mode et tous les bras utilisent le même SHA de binaire. Qualification DFlash : au moins1% dans chaque paire, dérive témoin≤5%, étendue des gains≤3points, sorties identiques, pic≤12e9octets et0nouveau Swapout jusqu'au postflight. Aucun gain n'est encore revendiqué.
