# K1PM — transformations Cold de GPU Hot : résultat après fusion

Le prototype K1 est **retiré du runtime** : les deux ordres réels DFlash2 B2 ne montrent aucun gain.
La correction numérique testée passe ; ces résultats ne démontrent pas une mauvaise implémentation.
Cette publication proposée est documentaire, sans kernel actif ajouté, release ni mise à jour de l’app installée.

## Hypothèse et changement réellement testé

Le diagnostic K1D antérieur à la fusion trouvait des positions Cold dans le travail soumis au chemin GPU Hot :
20 871 / 66 240 sur MAIN (**31,508 %**) et 21 739 / 67 200 sur HOLD (**32,350 %**).
Ces proportions décrivent des positions et une géométrie de calcul, **pas 32 % de latence supprimable**.
HOLD comporte aussi une queue B1 séparée : les dénominateurs ci-dessus portent sur les vérifications B2.
Sources et conditions de cet ancien diagnostic restent distinctes de la nouvelle campagne K1PM.
Preuves : [MAIN](../diagnostic/MAIN-snapshot-diagnostic.json), [HOLD](../diagnostic/HOLD-snapshot-diagnostic.json).

K1 remplaçait uniquement les deux sites H128 masqués et l’activation SwiGLU des experts Hot.
Il conservait le routage, le lancement Hot avant readback CPU, les slots protégés, les lectures SSD et les réductions.
Les branches Cold écrivaient +0 sans lire leurs données ; les chemins non masqués/autres formats gardaient MLX.
Les helpers CPU limitaient les formes, capacités, répétitions et offsets. Aucun nouveau flag produit.
Revue de transposition : [cross-review.md](cross-review.md), patch complet : [source.patch](source.patch).

## Identité et protocole

Base fusionnée : `9c8474ca1292e29bae3c7d4d6383bd1820549349` ; MLX 0.32.2, Apple M5 à 10 cœurs GPU, 24 Gio.
Modèle `Qwen3.8-Flash-Next-exl3-205-65c89531`, EXL3 2,05 bpw ; draft Q4 `a22a1744` avec cache circulaire.
Cache experts 3 Gio / 53 slots, allocateur 64 Mio, 4 lecteurs, contexte 4096, secteur stable aux endpoints.
GPU Hot Wide, UNIQUE et vérificateur multi-token actifs ; prélecture/lookahead, LES et shared désactivés.
DFlash2 fixe : une proposition (B2), `ADAPTIVE=0`, `WIDTH=1`. MTP inactif : ses compteurs sont null.
Chaque processus : 128 tokens warmup puis 128 mesurés, soit 127 tokens de décodage ; une mesure par bras.
Ordres séparés H-C puis C-H sur MAIN et HOLD, sans compilation, proveur ou autre modèle pendant les débits.
Les [plans et empreintes](plans/identity-frozen-quality-pending.json) conservent argv et environnement exacts.

| Artefact | SHA-256 |
|---|---|
| CLI baseline | `d8757fa6d9f62648d66f7a2d58617d6d54ff33905fefd408475519edd85105a6` |
| CLI candidat | `cc9669ad864dd73dadbed0bb503a440fefcec2c736e065d61f7ec0936639e9a4` |
| Manifeste candidat, 100 chemins | `2b4eb648ff2e0c4362bb88b1705c2022a6acb6d1896f1f2766923ed8d1ca98c4` |
| Patch sur la base fusionnée | `9782c74c49613acc520a38975376eefac5a52217941cfc254909702521dd358b` |

La metadata embarquée de la baseline indique e551-dirty, mais ses sources runtime avaient été vérifiées byte à byte contre main 9c.
Les SHA des binaires et manifestes fixent l’identité ; aucun ancien témoin 80M n’est réutilisé.
[Reconstruction des 100 sources](../provenance/reconstruction-preflight.json) : base9c + patch reproduisent100/100SHA.
Le tar source redondant reste local (SHA `bd158271ae5ea95b05c1dbd00e77f1ef2de72dff25ea21a7b486d5fecf111417`) ; il n’est pas publié dans Git.
Le patch permet de reconstruire le prototype retiré ; les exécutables et modèles ne sont pas publiés dans ces documents.

## Débits réels

| Prompt | H-C : baseline → K1, tok/s | Écart | C-H : baseline → K1, tok/s | Écart |
|---|---:|---:|---:|---:|
| MAIN | 11,399755 → 11,254234 | −1,2765 % | 11,576340 → 11,439044 | −1,1860 % |
| HOLD | 10,903516 → 10,896760 | −0,0620 % | 11,128820 → 10,946912 | −1,6346 % |

La géomoyenne descriptive des deux ratios appariés vaut −1,2313 % MAIN et −0,8514 % HOLD ; aucun gain additionné.
Les témoins H progressent entre ordres de 1,549 % MAIN et 2,066 % HOLD. Une mesure par bras ne donne pas un intervalle de confiance.
Les 16 réponses warmup/mesure sont non vides, égales en texte et hash au gold60 externe, avec budgets 128/127 exacts.
Compteurs identiques : MAIN 69 blocs/propositions, 58 acceptations, 138 lignes ; HOLD 70/56/140 plus un token séquentiel.
La demande logique d’experts mesurée reste 23 930 630 100 octets MAIN et 25 474 420 944 HOLD, identique entre bras et ordres.
Ces octets ne sont pas les lectures physiques SSD. Vérification, draft et commit/cache sont conservés séparément : les profils sont inclusifs.
[Bruts H-C](plans/screen/run-result.json), [bruts C-H](plans/confirm/run-result.json), [synthèse C-H](plans/confirm/summary.json), [contre-revue complète](real-tps-review.md).

Les 12 phases, enfants et deux samplers sont joints ; 120 + 118 observations système et 108 + 107 points VM conservés.
Zéro nouvelle sortie de swap système. Entrées système : 65 536 octets pendant post30 H-C seulement, zéro pendant les huit processus et tout C-H.
Pic physique lifetime maximal publié : **7 851 071 800 octets**, sous 12 Go ; ledger des entrées privées à zéro sur les 24 snapshots ready/complete.
Il n’existe pas de compteur de sorties privées dans ces rapports ; ces zéros ne prouvent pas l’absence de pression mémoire.
Compression logique courante jusqu’à 3,93 Go ; compression physique système jusqu’à 6,87 Go H-C et 6,77 Go C-H, sans attribution causale.
Les critères mémoire 77B antérieurs ne sont pas requalifiés. La première prose « zéro entrée système partout » est archivée et corrigée dans la contre-revue.

## Correction et vérification

Format, Clippy CPU et `mlx,chat`, builds release et suites locales passent : **228 tests CPU, 5 ignorés**, puis **164 natifs, 160 ignorés**.
Ces suites se recouvrent ; leurs effectifs ne s’additionnent pas comme une couverture unique.
Sept sélections physiques passent explicitement, une seule à la fois, avec sorties et marqueurs non vides :

- H128 : 3 144 cas, 240 frontières et 10 replis ; masques, strides, arrondis Half et poison Cold.
- Activation : 960 cas finis, 2 non finis, 1 frontière d’ID ; 28 replis et 10 refus de géométrie.
- 528 cas de slots réels ; 36 préfixes DFlash avec commits/continuations, logits et états contre cible séquentielle.
- Capture CPU externe indépendante de 64 000 valeurs ; MTP tête/cache et sessions budgets 1/2/3/17.

Les préfixes comparent le candidat groupé à sa cible séquentielle ; ce n’est pas un différentiel complet entre deux CLI baseline/candidat.
Six mutants MSL produisent chacun exit101 avec divergence numérique exigée, puis trois normaux restaurés repassent exit0.
Le premier lancement mutants v1 échoue sur collision de log **avant enfant**, sans détection numérique revendiquée ; v2 isolée conserve les onze phases.
[Qualité et portée](quality-review.md), [matrice normale](plans/quality/run-result.json), [échec v1](mutants/gpu-run-result.json), [mutants v2](mutants/gpu-v2/run-result.json).
Le bridge K1 de 43 requêtes et les débits OFF/MTP n’ont pas été exécutés : le débit DFlash négatif ne justifiait pas une promotion.

Kani 0.68.0 / CBMC 6.11.0 : **804 obligations SUCCESS, 6 chemins std inatteignables, 18 couvertures satisfaites**, zéro échec.
Unwinds H128 / SwiGLU / lookup : 17 / 5 / 513 ; rangs ≤4 et slots ≤2, dimensions/capacités i32 symboliques, lookup u32 et carte de 512 entrées.
Les 6 chemins std concernent alloc/ptr/fmt/num/raw_vec ; ce ne sont ni six obligations métier prouvées ni uniquement calloc.
La tentative sur les vrais appels MLX termine exit1 après 5,37 s avec ICE `intrinsics.rs:243`, sans stub, sans timeout ; enfant joint.
Les contrats CPU ne prouvent pas MLX, MSL, l’allocateur, la concurrence ou la génération entière.
[Log Kani CPU](build/kani-cpu.log), [commande/résultat API MLX](build/kani-mlx.json), [diagnostic ICE](build/kani-mlx.log).

## Ce que disent les 896 échantillons de microbenchmark

H128 : 704 échantillons / 32 cellules ; SwiGLU : 192 / 8 cellules, avec A/A puis cinq blocs A-B-B-A et oracles de chaîne hors horloge.
H128 a 17 médianes meilleures et 15 moins bonnes ; SwiGLU 6 et 2. Ces comptes ne sont pas un gain global.
L’hétérogénéité temporelle reste forte : A/A max/min 1,598 H128 et 2,903 SwiGLU ; aucun point filtré.
Les temps comprennent graphe, dispatch et attente finale ; H128 répété 32 fois et SwiGLU avec 31 récurrences communes ne reproduisent pas tout le MoE.
Les huit fixtures de masques/quantiles ne sont pas des poids de fréquence. Ni sommation des étapes ni extrapolation du +45 % local à des tok/s.
[Analyse complète](micro-review.md), [H128](plans/h128-micro/analysis.json), [SwiGLU](plans/activation-micro/analysis.json).

La revue de conception confirme un point souvent omis : `native/ffi/mlx_bridge.cpp:533–537` utilise déjà `mx::compile` pour SwiGLU.
Les trois nœuds sigmoid/multiply/multiply n’impliquent donc pas trois dispatchs distincts garantis à supprimer.
MLX Hadamard peut donner son buffer contigu (`hadamard.cpp:189–194`) ; CustomKernel alloue ses sorties (`custom_kernel.cpp:24–29`).
K1 garde la grille complète, écrit les zéros Cold et conserve les casts/scales sur toutes les positions ; il ne compacte pas le travail.
Ces différences sont établies statiquement, mais leur coût exclusif n’est pas mesuré : aucune cause clock/donation/dispatch du résultat TPS n’est affirmée.

Commandes principales conservées dans les plans/logs (préfixe `env`, gardes et chemins complets dans les JSON liés) :

- API MLX : `cargo kani --lib --tests --features mlx,chat --harness linear::hot_transform::tests::actual_hot_h128_preserves_native_half_bits --harness ssd_moe::hot_activation::tests::actual_masked_activation_preserves_mlx_nodes --exact`.
- Qualité H128 : `candidate-tests linear::hot_transform::tests::hot_h128_matches_native_half_roundings_masks_and_layouts --exact --ignored --nocapture --test-threads=1` ; même sélection isolée pour les six autres tests de [quality/run.json](plans/quality/run.json).
- Débit, payload enregistré : `python3.12 scripts/benchmark-flash-bridge.py candidate-cli Qwen3.8-Flash-Next-exl3-205-65c89531 MAIN-0-C --tokens 128 --repeats 1 --cache-gib 3 --prompt MAIN --dflash-draft Qwen3.8-Flash-Next-DFlash2-a22a1744 --dflash-mode 1` ; les chemins absolus réels et les quatre bras sont dans [confirm/run.json](plans/confirm/run.json).

La [revue de conception détaillée](performance-design-review.md) distingue les mécanismes vérifiés des causes de timing non mesurées.

## Clôture et publication

Le [retrait effectif](withdrawal-result.json) restaure 95 chemins à la base 9c ; les 5 nouveaux modules restent détachés, non stagés.
Le diff runtime est vide et le binaire de développement est restauré à d8757fa6 ; aucun build ou GPU n’a été répété pour ce retrait.
L’app installée n’a pas changé. Les sources du prototype, plans, échecs et résultats négatifs restent archivés ; aucun boost K1 livré.
La nouvelle PR proposée ne contient que ces preuves documentaires et le résumé du journal, sans nouvelle optimisation.

[Instantané CI du 25 octobre à 22:38 UTC](ci-snapshot-1.json) : main Linux/macOS/Flash-Kani réussis, Kani global encore en cours ; PR native encore en cours, régression/conversion réussies.
Cet instantané concerne les SHA fusionnés indiqués, pas une CI K1 complète ni la future PR documentaire ; aucune CI globalement verte proclamée.
Les anciens arrêts de runner Kani restent distincts d’un contre-exemple. Les vérifications physiques locales ne sont pas remplacées par la CI.

Mise à jour CI après cet instantané : [échec Kani PR42c et annotation du runner](ci-42c0958/README.md).
La PR42c est en échec sur le job Kani global ; main expose la même annotation avec un état encore incomplet.
Les logs sont indisponibles (404 / log not found) : aucune cause OOM, timeout ou contre-exemple n’est établie.
