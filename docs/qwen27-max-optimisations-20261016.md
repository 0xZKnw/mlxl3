# Qwen3.8-27B EXL3 : recherche consolidée vers 15 tok/s

> Révision du16octobre2026 : le [moteur1.4.8](release-engine-v1.4.8.md) active désormais le chemin qualifié Q27-19 par défaut dans l'app. Les chiffres ci-dessous restent ceux de la campagne historique ; le comparatif récent du paquet est non concluant en vitesse à cause de la dérive. Le défaut d'activation et l'[audit des autres options](qwen27-defaults-audit-20261016.md) sont consignés séparément.

**Résultat confirmé : 13,65 tok/s en français et 13,38 tok/s en code sur batterie**, avec l'option locale d'ordre des activations. Les témoins appariés donnent 10,97 et 10,77 tok/s : **+24,42 % et +24,27 %**, sur huit processus en ABBA puis BAAB. Les 15 tok/s ne sont pas atteints. L'option reste désactivée par défaut dans le code local ; l'app installée est inchangée.

**34 mécanismes distincts examinés**, après déduplication des 49 lignes des trois notes (19 entrées de la note arXiv, 18 entrées de la note projets, 12 pistes locales). Ce sont des mécanismes examinés, dont certains sont déjà rejetés ou exigent un autre contrat ; ce ne sont pas 34 nouvelles optimisations ni 34 gains additionnables. Le comptage des octets et le diagnostic des goulots sont une méthode de sélection, pas une 35e optimisation.

La configuration de référence qualifiée reste le M5 standard, dix cœurs GPU et 24 Go, MLX 0.32.2, checkpoint EXL3 Mul1 inchangé, MLP M2/M4 et MTP1 : **11,0765 tok/s en français et 10,9362 en code**, lors de la campagne sur secteur Q27-03. Viser 15 demande respectivement environ +35,4/+37,2 % de débit, soit −26,2/−27,1 % de temps par token. Les essais récents sur batterie, explicitement autorisés par l'utilisateur, ne permettent pas d'attribuer une différence entre sessions au code.

## Ce que la nouvelle campagne a réellement établi

| Essai | Résultat inspecté | Intégration et portée |
|---|---|---|
| Q27-15, extraction par funnels 32 bits | Parité physique. Le filtre initial utilisait un mauvais NT pour onze cellules M1 : ce défaut est conservé. Les onze cellules corrigées donnent **0/11 filtres** ; les 21 cellules initialement correctes ne sont pas répétées | **Rejeté** sur les formes courantes. Aucun shader de production changé. Les deux sessions restent distinctes |
| Q27-16, activations en ordre des lanes, lecture half4 | 48 cas synthétiques et 32 cellules réelles M1/M2 exacts en FP32 puis chaînes FP16 ; mutation détectée. **20/32 microfiltres** ; douze cellules non qualifiées restent visibles | Filtre exploratoire terminé. Port natif évalué globalement sous Q27-19, sans sélectionner seulement les vingt cellules gagnantes |
| Q27-17, fusion dense SwiGLU/suh/H128 | **268 contrôles exacts**, 64 échelles réelles et tous les 63 488 patterns FP16 finis de gate, mutation d'arrondi détectée. **0/6 filtres**, dérives d'environ 4 à 54 % | Qualité validée dans ce domaine ; vitesse **non concluante**, mesures très courtes sensibles à la soumission/synchronisation. Aucun port natif, aucune relance sans nouveau protocole justifié |
| Q27-18, fusion préparation/récurrence GDN Hv48 | CPU :176 passés/84 ignorés ; garde Kani91 obligations/cinq couvertures. GPU :32 comparaisons et40 préfixes exacts, mutation détectée puis restauration. Hv48T1 **−1,44 %**, T2 négatif avec dérive invalidante ; **0/2 filtres** | **Rejeté et retiré**. Trois sources et deux binaires restaurés aux empreintes antérieures ; aucun appel Qwen intégré. [Preuves](measurements/qwen27-max-20261016/gdn18/withdrawal.json) |

**Q27-19 est maintenant validé localement**, avec ses oracles et ses deux campagnes bridge. Les pourcentages de QMV isolés restent distincts du gain global mesuré ; aucun gain externe n'est additionné. Le réglage est opt-in et les **15 tok/s restent à atteindre**.

## Sources primaires et identité de lecture

Chaque référence ci-dessous fixe une révision ou une version. Les dates d'articles sont celles des versions lues ; une lecture d'abstract reste distinguée d'une lecture du corps dans la note arXiv. Dans les lignes portant O seul, la source primaire est le code et l'essai local datés, pas un résultat externe. Les chemins locaux cités plus bas sont relatifs au worktree `/Users/justin/.codex/worktrees/qwen27-smallm/mix-stq1_0` ; état de départ `7ce611c1c0db9a704bbe35abccc6293fc6ee7995` avec modifications locales enregistrées, moteur qualifié `4db44674…`. L'état local modifié n'est pas assimilé au commit propre.

| Référence | Source primaire fixée, date et portée |
|---|---|
| S | [Sushi : lecteurs et disposition EXL3](https://github.com/beamivalice/sushi/blob/70dc0e807e4113179389377253fe8270d5a3333b/src/exl3/expert_exl3_kernels.zig), SHA `70dc0e80`, 16 octobre 2026 ; [documentation](https://github.com/beamivalice/sushi/blob/70dc0e807e4113179389377253fe8270d5a3333b/docs/engine-exl3-experts.md). Code lu par sections, pas audit intégral |
| E | [ExLlamaV3 GEMV](https://github.com/turboderp-org/exllamav3/blob/058b8630fd081f3ff46fce6246955b157c3a2e3b/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh), SHA `058b8630`, 14 octobre 2026 ; [codebook](https://github.com/turboderp-org/exllamav3/blob/058b8630fd081f3ff46fce6246955b157c3a2e3b/exllamav3/exllamav3_ext/quant/codebook.cuh). CUDA/RDNA, pas Metal |
| L | [Lithos mixer](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/docs/design/mixers.md), [barrières](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/compiler/barriers.py), [runtime](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/runtime/src/metal_core.mm). SHA `5485ad6b`, résolu le 16 octobre ; date exacte de ce merge non établie par les en-têtes de patch |
| LG/LA | Études Lithos archivées au SHA `46b2bc4c` : [GDN, achevée le 2 octobre 2026](https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/m5max-gdn-mixer-optimization.md), [attention](https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/m5max-27b-attention-optimization.md). M5 Max 40 GPU/48 Go, formats et calculs différents |
| M | [MLX backend Metal 0.32.2](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/device.cpp), SHA `1f8e74e3`, version locale. [Hadamard](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/hadamard.h), [opérations](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/unary_ops.h). Headers installés comparés aux sources dans Q27-17 |
| M4 | [MLX 0.32.4](https://github.com/ml-explore/mlx/commit/36969c094354086e15d2a327a9c50feb20b9cf11), 14 octobre 2026, SHA de tag `36969c09` relevé précédemment. Aucun boost EXL3 promis ; ABI et chemin numérique à requalifier |
| C | [llama.cpp GDN Metal](https://github.com/ggml-org/llama.cpp/blob/7182ef71e6d5dd0c8ea76486fac134fcd25025d4/ggml/src/ggml-metal/kernels/gated_delta_net.metal) et [fusions](https://github.com/ggml-org/llama.cpp/blob/7182ef71e6d5dd0c8ea76486fac134fcd25025d4/ggml/src/ggml-metal/ggml-metal-fusion.cpp), SHA `7182ef71`, 16 octobre 2026 |
| I | [ik_llama : allocation des sorties MTP](https://github.com/ikawrakow/ik_llama.cpp/commit/cea2d61ffdac38ee2958ba88d120fa07cb67703a), SHA `cea2d61f`, 16 octobre 2026. README : maintenance surtout CPU/CUDA, pas validation du backend Metal |
| P | [PonyEXL3 : chemin fusion/cache](https://github.com/beamivalice/PonyExl3/blob/8e7fa6b1556f59fc669e25087903b279b9b0346f/ponyexl3/mlx/exl3_fused.py), SHA `8e7fa6b1`, 24 août 2026 ; HEAD résolu le 16 octobre et inchangé depuis les revues précédentes |
| Q | [QTIP](https://arxiv.org/abs/2406.11235v1), v1 du 17 juin 2024 ; abstract/métadonnées lus |
| F | [FLUTE](https://arxiv.org/abs/2407.10960v1), v1 du 15 juillet 2024 ; sections de layout/unpacking lues |
| MA | [MARLIN](https://arxiv.org/abs/2408.11743v1), v1 du 21 août 2024 ; kernels CUDA/affine, pas recipe M1 EXL3/M5 |
| B/B5 | [BaseRT initial](https://arxiv.org/abs/2607.00501v1), 1er juillet 2026, et [étude M5](https://arxiv.org/abs/2607.19438v1), 21 juillet 2026. Sections techniques et tables sélectionnées lues ; M5 Pro, quantification différente |
| MP | [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219v1), v1 du 22 décembre 2025 ; sections tâches/graphes CUDA lues |
| G | [Gated Delta Networks](https://arxiv.org/abs/2412.06464v1), v1 du 9 décembre 2024 ; section chunkwise/WY lue |
| PS | [État récurrent persistant sur FPGA](https://arxiv.org/abs/2603.05931v1), 6 mars 2026 ; abstract/métadonnées, pas une propriété du GPU Apple |
| LE/FI | [LeanAttention](https://arxiv.org/abs/2405.10480v1), 17 mai 2024, abstract ; [FlashInfer](https://arxiv.org/abs/2501.01005v1), 2 janvier 2025, sections de planification lues |
| DF | [DFlash](https://arxiv.org/abs/2602.06036v1), 5 février 2026 ; sections 4 et 5.4 lues |
| DT | [DDTree](https://arxiv.org/abs/2604.12989v1), 14 avril 2026 ; ancêtres, vérification et budget lus |
| DL/DA | [DFlare](https://arxiv.org/abs/2606.02091v1), 1er juin 2026 ; [Direct Alignment](https://arxiv.org/abs/2403.00858v4), v4 du 13 mai 2024 ; abstracts/métadonnées |
| QS/TM | [QuantSpec](https://arxiv.org/abs/2502.10424v1), 5 février 2025 ; [T-MAC](https://arxiv.org/abs/2407.00088v1), 25 juin 2024 et dépôt Microsoft ; abstracts/README |
| CT | [CurveTQ](https://arxiv.org/abs/2610.09212v1), 6 octobre 2026 ; abstract, nouveau checkpoint nécessaire |
| O | Source locale primaire : Rust/Metal/FFI et `opti.md` lus le 16 octobre 2026. Les numéros historiques ci-dessous identifient les essais et leurs limites ; leurs gains ne sont pas réattribués à un autre binaire |

Les références TensorFold `46a08c18` et Uzu `2c16e967`, du 16 octobre, ont déjà leur analyse Q27-09. TensorFold emploie notamment BF16/sommes compensées et Uzu son codec/attention BF16 : ce ne sont pas des remplacements implicites de nos projections FP16 EXL3. Elles n'ajoutent pas un mécanisme au comptage lorsqu'elles recouvrent ceux du tableau.

## 21 mécanismes de décodage examinés

P1 : à concrétiser ou premier filtre local raisonnable ; P2 : conditionné par un profil et un bénéfice plausible ; P3 : coût élevé/pouvoir faible ; R : rejet historique ou nouveau rejet, pas de répétition sans différence précise. « Non mesuré » signifie qu'aucun gain de génération n'est établi.

| ID / priorité | Mécanisme, source et point local | Nouveauté, contrat et premier oracle | État réel |
|---|---|---|---|
| **D01 R** | Extraction de quatre codewords par opérations 32 bits. Q/S (documentation) ; `native/shaders/_qmv_tile_kernel.metal`, `_qmv_mapped_tile_kernel.metal` | Distinct des huit états K2 HENK0O-02 et du K6/MCG PERF-106. Garder fenêtres, FMA et réductions. Oracle entier offsets/bitbasis/ring-wrap, puis FP32 stock | Q27-15 exact mais rejeté : 0/11 corrigés et 21 anciens contrôles conservés séparément |
| **D02 validé localement** | Activations préparées en ordre des lanes, half4. S ; préparation xhat dans linear.rs et les deux QMV | Même permutation des valeurs et même ordre FMA ; indices créés au chargement, take inclus dans chaque projection. Oracles stock, états, mutations et bridge | Q27-19 :13,65/13,38 tok/s sur batterie, environ+24 %, confirmé ABBA/BAAB. Option OFF par défaut ; app non installée |
| **D03 P3** | Fusion dense SwiGLU → échelle `suh` → H128. O/M ; `qwen35::Mlp::forward`, FFI opération6, `Exl3Linear` | Distinct du batch MLP ; le helper MoE F32 n'est pas oracle. Conserver Sigmoid/multiplications FP16 et arrondis radix16/radix8. Oracle réel MLX indépendant, échelles des 64 couches, puis down stock | Q27-17 exact ; 0/6 filtres et dérive invalidante. Aucun port ni gain établi |
| **D04 R/conditionnel** | Décoder plusieurs codewords par uint2/half2. E/S ; codebook_header | Le bytesum SWAR existe déjà. **Révision du16octobre** : prototype uint4/half4 retrouvé au commit8be9b7bc, déjà rejeté sur27B/M5. La largeur2 de Sushi reste différente mais exige un avantage concret d'instructions/registres avant nouvel essai | Aucun nouveau prototype. [Revue et sources historiques](measurements/qwen27-max-20261016/d04-documentary-review.md) ; ne pas répéter le rejet sous un autre nom |
| **D05 R** | Adresses QMV incrémentales au lieu de recalculs d'offset. O ; boucles `tile_k` | PERF-35 a déjà rejeté l'adressage incrémental. Il faut un changement concret de layout ou de code généré avant nouvel essai. Oracle d'adresses/overflow puis FP32, sans changer l'ordre de visite | Non retenu ; ne pas relancer sous un nouveau nom |
| **D06 P2** | Deux tuiles K chargées avant décodage, géométrie fixe. S (documentation)/MA ; boucles QMV | Différent du seul réglage NT/MB, mais unrolling antérieur existe. Même ordre FMA et gestion des queues obligatoires ; oracle brut FP32, queues impaires et chaîne | Non mesuré sur cette variante exacte. Risque de registres/spills, à justifier par profil |
| **D07 R** | Charger collectivement les mots puis les distribuer par shuffle. E/O ; lecteurs trellis QMV | Déjà rejeté dans `qwen38-decode-local.md`. La coalescence peut amortir les loads actuels. Oracle de tous les mots/lanes et mêmes FMA ; aucune baisse de trafic DRAM déduite du seul nombre d'instructions | Dépriorisé ; pas de répétition sans différence matérielle/source démontrée |
| **D08 P2** | Repacking sans perte pour simplifier l'unpacking. F ; loader `Exl3Linear`/`Exl3Group` → shader | Le repack des paires N du head PERF-111 n'est pas nouveau. Une disposition différente K/lanes dense doit être explicitée. Roundtrip intégral codes/scales, adresses et produits exacts ; budget de double résidence explicite | Non mesuré ; pas de changement de quantification autorisé |
| **D09 P3** | Paire des petites projections GDN a/b par vecteurs colonnes batchés. O, SMALLM-B ; `GatedDelta::{run,forward_verification_impl}` | SMALLM-B a déjà 162 cas exacts mais faible coût absolu. Ne pas remplacer par un GEMM ordinaire qui réassocie. Premier nouvel oracle : vraie intégration M2, états et bridge | Fonctionnel historiquement en proxy, non intégré ; aucun gain modèle |
| **D10 R** | Fusion préparation conv/SiLU/QK-norm et récurrence. G/L/O ; gated_delta.rs | Ancien Hv32 retiré après dérive ; essai Hv48/27B distinct. Oracle split réel T1..4 et tous préfixes, mutation réelle détectée | Q27-18 terminé : correction vérifiée, Hv48T1−1,44 % stable, T2 négatif/dérivant ;0/2 filtres. API retirée, aucun gain modèle |
| **D11 P2** | Fin GDN → préparation de l'out_proj : norme, gate, échelle et H128. LG/O ; fin de `GatedDelta::run` | HENK0O-09 norme+gate seul n'a pas établi de gain. Nouvelle fusion doit inclure la préparation EXL3 en gardant les frontières FP16 et gate F32. Oracle composite MLX, états et logits | Non mesuré ; plus ambitieux que D03, pas une activation de flag existant |
| **D12 P3** | Chevaucher une projection indépendante et le cœur GDN. L/M ; graphes `qwen35.rs`, backend Metal | MLX gère déjà des hazards. Démontrer une sérialisation coûteuse avant de modifier les queues. Oracle de dépendances/alias, répétitions, cancel et états ; liveness bornée | Non mesuré ; risque de concurrence, pas de preuve par Kani CPU seul |
| **D13 P2 long contexte** | Politique stream-K/split-K de l'attention. LE/LA ; `Array::sdpa`, attention D256/Hq24/Hkv4 | Pas une copie Uzu BF16. Nouvelle réduction online-softmax peut changer les bits. Oracle Q1/2/4 × KV23/129/1024/4096/long, sorties/états puis proportion du coût total | Non mesuré ; si ordre flottant changé, hors chemin bitexact tant que non démontré |
| **D14 P2** | Réutiliser plans et métadonnées d'attention malgré KV variable. FI ; SDPA/FFI, cache des formes | Différent de réserver du KV ou de D13 qui change l'algorithme. Les clés doivent inclure dimensions, suffixes et budgets. Oracle frontières/rollback/invalidation et profil CPU de setup | Non mesuré ; dépend d'un coût de plan réellement observé |
| **D15 P2** | Append KV cible une fois par bloc, vues causales par ligne. O ; `Attention::forward_verification_impl` | Le chemin MTP append actuellement ligne par ligne. Préserver Q/K/RoPE M1 et chaque préfixe. Oracle logits/128 états à 256/4K/8K, tous commits ; surveiller strides et dispatch SDPA | Non mesuré ; une simple économie de concat ne valide pas la parité |
| **D16 P2** | AR run-ahead borné d'un token pendant decode/flush du précédent. O ; `main::bridge_generate_round` | Différent du lazy/argmax PERF-07 rejeté. Un token device et état spéculé ne doivent pas devenir un snapshot committé. Oracle EOS/stop/cancel/budget/backpressure et bridge | Non mesuré ; ne coder que si CPU/I/O occupe une fraction utile |
| **D17 P2** | Rebuild isolé avec MLX 0.32.4 et ABI correspondante. M4 ; build/FFI/métallib | Version nouvelle, pas remplacement DYLD sous ancien binaire. Requalifier logits, exp/log, caches et dtypes entre versions avant vitesse | Non exécuté. Aucun gain EXL3 annoncé par le changelog |
| **D18 P3** | Conserver des descriptions Rust immuables des kernels. O/M ; `array::metal_kernel`, CString/Vec/header | Le cache C++ mesuré à 0,111 % ne couvre pas toute préparation Rust, mais ne justifie pas un autre cache. Profil des allocations/noms d'abord ; oracle clés/collisions/ownership/erreurs NUL | Non mesuré ; ne pas confondre avec le cache Metal déjà diagnostiqué |
| **D19 P3** | Paquet MTP draft+posterior, un transfert des IDs. O ; `Head::draft_chain`, `Session::advance` | Distinct du MTPDEVICE conservant `logits.eval`. Ancienne variante DFlash retirée après dérive. Garder cible, acceptation, lookup et rollback. Oracle IDs/NaN/erreurs/reset/commit | Non mesuré dans ce domaine 27B/MTP1 ; précédents faibles, profil nécessaire |
| **D20 P3** | Argmax brut du draft, tête complète inchangée. O ; `Head::draft_chain` | La cible garde son sélecteur canonique. Retirer la normalisation draft peut modifier les égalités et l'acceptation. Oracle ties/zéros/NaN et corpus indépendant, puis E2E | Non mesuré ; ce n'est pas une optimisation strictement neutre du draft |
| **D21 P3** | Commandes figées/rejouées, ICB et contrôle GPU. MP/L ; `Session::advance`, Array/FFI | Différent du pipeline par couche Q27-05 sous 3 %. Nouvelle gestion de buffers/synchronisation nécessaire. Profil de bulles CPU avant prototype ; oracle replay changeant, cancel, ring/timeout/états | Non mesuré ; coût de réécriture élevé. Ne pas présenter « moins de dispatchs » comme un gain |

## Trois mécanismes visant surtout le préfill et le TTFT

| ID / priorité | Mécanisme et source | Nouveau domaine, contrat et premier oracle | État réel |
|---|---|---|---|
| **T01 P3** | GDN par blocs, état réutilisé ou formulation chunkwise/WY. G/C ; `gated_delta.rs`, préfill Qwen | La récurrence exacte T≤8 existe déjà. Une boucle séquentielle plus large peut garder le contrat ; WY réassocie généralement les calculs. Oracle séquentiel indépendant T23/24/129/256/1024 et tous états avant TTFT | Non mesuré ; ne compte pas comme un gain de décodage court |
| **T02 P2 TTFT** | Couvrir une forme préfill encore exclue des TensorOps/GEMM. B5/O ; `linear.rs` dispatch QMM | TensorOps et BM64 sont déjà intégrés. Il faut identifier une forme manquante, pas réannoncer le mécanisme existant. Oracle des projections, logits/caches et TTFT du même prompt | Aucun nouveau domaine qualifié. Les résultats M5 Pro ne sont pas ceux du M5 standard |
| **T03 P3** | Cache de poids décodés borné pendant un chunk. P ; chemin QMM/loader | PonyEXL3 inchangé ; cache FP16 complet de 27B incompatible avec 24 Go. Seule une stratégie partielle explicite serait nouvelle. Oracle chunk-invariance, sorties exactes et pic mémoire borné | Non mesuré ; ce n'est pas une accélération gratuite du decode |

## Trois mécanismes de mémoire ou de résidence

| ID / priorité | Mécanisme et source | Contrat, différence historique et premier oracle | État réel |
|---|---|---|---|
| **R01 P2 diagnostic** | Buffers scratch, constantes et sorties dimensionnés/réutilisés. B/I/L ; GDN constants, allocations Array, sorties MTP | Fusion de trois déclinaisons proches en une seule piste. Le pooling MLX et PERF-36 peuvent déjà couvrir les petits invariants ; ik_llama corrige un autre runtime. Compter allocations/coût avant port ; oracle alias, snapshots, annulation et OOM | Non mesuré. Distinguer bytes MLX, cache et empreinte processus ; aucun gain RAM attribué au seul nombre d'allocations |
| **R02 P2 long contexte** | Workspaces attention dimensionnés aux partitions utiles et réutilisés par durée de vie. LA ; SDPA/FFI | Différent d'un nouveau softmax : changer adresses/capacités seules peut garder l'ordre. Oracle tous consommateurs, alias, suffixes/append/commit et formes, puis mémoire réelle | Non mesuré. Le KV réservé historique ne valide pas ce workspace |
| **R03 P3 matériel** | État GDN conservé sur puce entre dispatchs. PS/L ; états `GatedDelta` | Le kernel conserve déjà l'état entre tokens d'un dispatch. La nouveauté serait entre dispatchs : 48 couches × 48 heads × 128² × 4 octets = **144 Mio d'état logique**, sans autres buffers | Non implémenté ; capacité/résidence physique à établir avant code. Une propriété BRAM FPGA n'est pas une garantie Metal |

## Sept mécanismes de draft ou de contrat distinct

Les quatre premières pistes peuvent préserver le modèle cible en le vérifiant intégralement, mais exigent un autre draft, un entraînement ou une gestion d'état nouvelle. Les trois dernières ne sont pas des substitutions exactes du checkpoint courant.

| ID / priorité | Mécanisme et source | Condition de pertinence et premier oracle | État réel |
|---|---|---|---|
| **X01 P2 conditionnel** | Draft diffusion parallèle compatible. DF ; modules DFlash existants | Le moteur possède déjà DFlash : nouveauté éventuelle = checkpoint 27B/H5120/tokenizer/features compatible et moins coûteux. Vérifier métadonnées, RAM et coût marginal du verifier avant téléchargement ; cible/caches exacts puis acceptation | Aucun nouveau checkpoint compatible confirmé par cette revue ; aucun poids téléchargé |
| **X02 P3** | Arbre de propositions à budget fixé. DT ; Session/commit GDN et attention | Ancestors-mask seul insuffisant : conv et état GDN par branche. Sur traces indépendantes, borner refus récupérables/coût puis deux branches et oracle de tous préfixes | Non mesuré ; pas une simple profondeur MTP supplémentaire |
| **X03 P3 entraînement** | Features cible différentes selon la couche du draft. DL ; captures DFlash | Exige un draft entraîné et des features/cache supplémentaires ; aucun flag runtime ne crée ce checkpoint. Vérifier compatibilité et ratio tokens acceptés/coût total sur corpus tenu à l'écart | Recherche future, pas entraînement lancé ni gain local |
| **X04 P3 entraînement** | Aligner le draft sur les sorties du checkpoint EXL3 réel. DA ; traces draft/cible | Distinct du pruning de tête qui a réduit l'acceptation. Diagnostiquer d'abord un mismatch sur corpus indépendant ; adaptation du draft seulement si bénéfice plausible et protocole séparé | Non mesuré ; rien ne prouve actuellement que le mismatch d'entraînement domine |
| **X05 exclu ici** | Self-spec avec caches/poids hiérarchiques quantifiés. QS ; caches cible/draft | KV8 a déjà échoué sur des domaines longs. Maintenir le KV cible exact interdit de réintroduire cette approximation sous un autre nom. Une étude draft-only serait une autre campagne | Aucune expérience recommandée sur le chemin cible exact actuel |
| **X06 exclu ici** | Tables de combinaisons d'activations / bitserial. TM ; décodeur EXL3 | Mul1 non linéaire ne se transforme pas gratuitement en bitplanes affines. Exiger une décomposition et le même ordre FMA avant tout GPU ; différent du LUT codebook Q27-08 mais pas démontré compatible | Pas de prototype ; aucun gain CPU/NPU transféré au GPU EXL3 |
| **X07 autre checkpoint** | Treillis sans rotations. CT ; quantificateur, `suh/svh`/H128 | Enlever Hadamard des poids EXL3 stockés change leur sens. Nécessite nouvelle quantification/checkpoint et évaluation qualité séparée ; pas un flag de runtime | Hors contrat courant ; aucun poids ni quantification modifiés |

## Ce que le contrôle Lithos change dans la décision

L'étude GDN compare la fusion avec un contrôle ayant le **même packing et les mêmes géométries** : 596,02 µs pour le contrôle multi-dispatch, contre 598,86 µs pour la fusion et 599,72 µs pour le défaut optimisé. Elle ne démontre donc pas un avantage de fusion sur ce contrôle. Le bénéfice face à l'ancien moteur inclut surtout ses autres optimisations. C'est une raison de garder un contrôle non fusionné équivalent, pas de recopier son mégakernel. [Source primaire LG](https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/m5max-gdn-mixer-optimization.md).

Son calcul BF16/FP8/NVFP4 et sa normalisation commutée n'ont pas le contrat FP16 local ; déplacer un arrondi ne devient pas exact parce que les textes d'un exemple sont égaux. Les prototypes locaux sont évalués contre les vrais opérateurs exécutés et des états complets. Aucun pourcentage Lithos, CUDA, M5 Pro ou M5 Max n'est injecté dans une estimation à 15 tok/s.

## Déduplication contrôlable

Les identifiants d'origine sont ceux des trois notes jointes. « P » désigne ici la ligne du tableau projets, et non la référence PonyEXL3 du registre des sources.

- Articles A01→D01, A02→D08, A03→D06, A04→R01, A05→D21, A06→D10, A07→T01, A08→R03, A09→D13, A10→D14, A11→T02, A12→X01, A13→X02, A14→X03, A15→X04, A16→X05, A17→X06, A18→X07. A19, diagnostic format servi/roofline, n'est pas compté.
- Projets P01→D01, P02→D03, P03→D02, P04→D04, P05→D05, P06→D06, P07→D07, P08→D08, P09→D09, P10→R01, P11→D10, P12→D11, P13→D12, P14→R02, P15→T01, P16→R01, P17→D21, P18→T03.
- Local L01→D01, L02→D10, L03→D11, L04→D03, L05→D15, L06→D16, L07→D17, L08→D18, L09→D09, L10→D19, L11→D20, L12→T01/D13. Une ligne locale mentionnant deux familles n'est pas une nouvelle optimisation supplémentaire.

Le compte est ainsi **21 D + 3 T + 3 R + 7 X = 34**. Les sources d'idées supplémentaires (EAGLE3, FlashDecoding++, Snakes and Ladders, GDN2) figurent dans la note articles mais ne gonflent pas ce compte lorsqu'elles recouvrent une famille ou changent le modèle.

## Audit documentaire des références

Les 17 articles versionnés cités dans le registre ont été rouverts sur leurs pages primaires arXiv pendant cette vérification : titres, versions et dates correspondent aux références retenues. Cela vérifie les liens bibliographiques, pas l'intégralité des méthodes ni leurs performances. Aucun SHA de PDF/HTML n'est inventé : les articles ne disposent pas de copies locales de source dans cette campagne.

Lectures limitées à l'abstract/métadonnées : QTIP, FPGA persistant, LeanAttention, DFlare, Direct Alignment, QuantSpec, T-MAC et CurveTQ. Les autres articles ont une lecture ciblée de sections dans la note arXiv ; aucun n'est présenté comme un audit intégral. D01 et D13 combinent ainsi un article lu superficiellement avec du code/profil local ou une étude de projet plus précise.

Les liens GitHub vers des fichiers ont été rapprochés des URL brutes épinglées des manifestes et des SHA des copies. Le commit ik_llama et MLX 0.32.4 disposent de métadonnées historiques distinctes. Le chemin PonyEXL3 pointe désormais directement vers le fichier déjà empreinté, plutôt que vers la racine du dépôt. Les treize téléchargements projets échoués ne sont utilisés par aucun lien de fichier retenu ; les chemins d'architecture Lithos introuvables au HEAD ne servent pas de preuves. L'API GitHub limitée et les endpoints HF Markdown indisponibles restent des limites d'accès documentées.

Les trois notes et les deux manifestes liés sont vérifiés relativement à la future destination `docs/qwen27-max-optimisations-20261016.md`. Le rapport [d'audit des 34 mécanismes et de leurs références](measurements/qwen27-max-20261016/research-link-audit.json) distingue copies empreintées, métadonnées seules et lecture bibliographique.

## Sélection et limites

1. Q27-19 clos et conservé comme option native locale : les deux ordres de campagne passent. Aucune promotion implicite de l'app ni activation par défaut.
2. Q27-18 terminé et retiré : le filtre exact ne montre aucun gain utile. Ne pas relancer la fusion seule sans nouvelle raison mesurable.
3. Conserver les rejets Q27-15 et la vitesse non concluante Q27-17. Pour les pistes non mesurées, un profil de la vraie frontière de génération décide du prochain petit prototype, plutôt qu'une grande réécriture justifiée par un nom de projet.

Couverture : cette recherche est **large mais non systématique**. Certaines sources sont lues par sections, d'autres seulement par abstract ; ce niveau figure dans [la note articles](measurements/qwen27-max-20261016/research-arxiv.md). [La note projets](measurements/qwen27-max-20261016/research-projects.md), [l'audit local](measurements/qwen27-max-20261016/research-local.md), [le manifeste projets](measurements/qwen27-max-20261016/research-projects-source-manifest.json), [les révisions](measurements/qwen27-max-20261016/research-projects-revisions.json) et [la provenance numérique](measurements/qwen27-max-20261016/dense-prepare-notes.md) conservent provenance, versions, copies/SHA et erreurs d'accès. La recherche projets a obtenu 40 fichiers publics et conservé 13 échecs ; une source téléchargée n'est pas réputée intégralement auditée.

Non vérifiés : disponibilité et qualité d'un nouveau draft compatible, performance long contexte, gains des pistes sans mesure, preuve de MLX/Metal/concurrence et déploiement. Les tests physiques et preuves CPU ont des domaines bornés ; ils ne constituent pas une preuve du modèle complet. Cette synthèse documentaire ne lance aucun benchmark, build, proveur ou GPU et ne modifie ni le runtime ni l'app installée.

## Vérification du port natif Q27-19

Le candidat utilise la même quantification et la même tête MTP. Son option de démarrage est `MLXL3_EXL3_LANE_ORDER=1`, désactivée par défaut. La garde couvre les seize géométries de l'inventaire, Mul1 sur M5, et M1/M2 par appel de projection. Une vérification M3 peut toujours appeler des projections M1 via le MLP en série ; les tests exercent ce cas.

Les contrôles fonctionnels avant vitesse comprennent vingt projections réelles, avec une mutation détectée sur 246 227 valeurs FP16 finies puis restauration, 384 comparaisons MLP et les préfixes 24/257. Le modèle compare 248 320 logits, 5 120 valeurs cachées et 128 tableaux d'état, puis tous les commits retenus et les continuations contre le chemin stock. Les 32 cycles Session couvrent 29 blocs spéculatifs et trois cycles AR. Le bridge vérifie six budgets, deux préfixes réutilisés, deux annulations suivies de reprises exactes, trois requêtes invalides et le repli avec pénalité de répétition.

Commandes principales, exécutées dans le worktree avec les dépendances locales identifiées dans les preuves :

~~~sh
cargo fmt --all --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --release --features mlx,chat
cargo build --locked --release --features mlx,chat
cargo kani --lib --no-default-features --harness exl3_lane_layout_key_distinguishes_options
cargo kani --lib --no-default-features --harness exl3_lane_permutation_is_bijective_and_preserves_tile_bounds
cargo kani --lib --no-default-features --harness exl3_lane_order_preserves_shape_and_large_m_fallback
~~~

La suite Rust totalise **179 tests passés et 85 ignorés** ; les tests physiques sont sélectionnés séparément. Kani vérifie les vraies fonctions CPU : clé, 2 obligations/2 couvertures ; permutation u32 complète, 9/3 ; garde i32/usize/bools complète, 68/6. Aucun `assume`, aucune boucle ni stub ; zéro obligation inaccessible. Cela ne prouve ni les shaders Metal, ni MLX, ni la concurrence. La tentative sur la vraie API Array échoue par erreur interne du vérificateur (`intrinsics.rs:243`) et reste non vérifiée.

Le validateur du benchmark passe **34 tests**, dont des rapports incomplets, sorties divergentes, NaN/infini, mauvais compteurs, mauvaises clés, absence de gain et dérive excessive. CrossHair a été tenté mais n'est pas installé. L'échec initial de lint sur l'ordre des imports est conservé et corrigé. Les contrôleurs de processus ont leurs dix-neuf tests de délais, erreurs, arrêt et fermeture ; un code de sortie nul seul ne signifie jamais parité.

Les anciens échecs de parseur sont conservés : le contrôle de mutation GDN attendait un mot de diagnostic différent, et le contrôleur de suite Rust attendait une seule suite au lieu d'agréger les six. Les logs ont permis d'inspecter les résultats réels sans refaire les mesures ni effacer les erreurs.

La CI couvre PR/push et Kani CPU ; deux sélections explicites de tests macOS avec MLX vérifient l'override et les huit clés Tune. Le YAML et ces sélections sont contrôlés localement. Aucun run GitHub du code local n'est déclaré vert, puisqu'aucun push n'a été effectué. L'app installée n'est pas remplacée.

Les identités, commandes, options et logs sont dans [les preuves Q27-19](measurements/qwen27-max-20261016/lane19/candidate-v2-identity.json), [les tests physiques](measurements/qwen27-max-20261016/lane19/physical-status.json), [le bridge](measurements/qwen27-max-20261016/lane19/bridge-boundaries-status.json) et [les limites formelles](measurements/qwen27-max-20261016/lane19/formal-limitations.json). La répétition ciblée ajoute une assertion explicite de taille/finitude des continuations ; ses résultats sont consignés séparément dans `prefixes-values-status.json`.

## Débit réel confirmé et conditions

| Campagne | Français A → B, tok/s | Gain | Code A → B, tok/s | Gain |
|---|---:|---:|---:|---:|
| ABBA | 10,9664 → 13,4962 | +23,07 % | 10,7434 → 13,3071 | +23,86 % |
| BAAB | 10,9712 → 13,6785 | +24,68 % | 10,8212 → 13,4013 | +23,84 % |
| Médianes des quatre valeurs par bras | **10,9712 → 13,6508** | **+24,42 %** | **10,7673 → 13,3806** | **+24,27 %** |

A désactive le layout, B l'active, avec **le même binaire d6a4ea44**, les mêmes poids EXL3 et la tête MTP d'origine. MTP1, MLP M2/M4 par défaut, contexte4096, greedy, cache prompt désactivé, allocateur par défaut ; pipeline/adaptatif/lookup désactivés. Chaque processus chauffe64 tokens par prompt, puis produit130 tokens français jusqu'à EOS et256 tokens code. Ce n'est pas un test soutenu de256 tokens français.

Les huit processus finissent avec code0 et sont joints. Les seize sorties mesurées et seize warmups ont les mêmes identifiants, texte, historique, nombres de tokens et compteurs MTP, également par rapport à Q27-03. En français,71 propositions/60 acceptées ; en code142/113. Les clés ready séparent bien les layouts0/1 et conservent smallm-v2=1 et allocator-cache-v1=0.

La variation maximale max/min sur les quatre répétitions candidates est **2,83 % en français et2,49 % en code** ; les témoins varient de1,64 % et1,47 %. Chaque campagne passe séparément les seuils préenregistrés, et même chaque candidat des deux campagnes dépasse chaque témoin de plus de3 %. Cette « dérive » décrit le bruit de vitesse entre répétitions, pas une variation des réponses. Aucun résultat de qualité n'est déduit de la seule stabilité du texte : les oracles numériques complets et la mutation sont séparés.

Les conditions batterie sont conservées par passe dans les résultats. Aucun autre modèle GPU, build ni proveur n'a tourné pendant ces mesures. Températures et fréquences ne sont pas mesurées ; aucun gain RAM n'est revendiqué. La référence historique sur secteur reste distincte des témoins appariés de cette campagne sur batterie.

**Utilisation du code local :** ajouter MLXL3_EXL3_LANE_ORDER=1 à l'environnement au démarrage du nouveau moteur ;0, absence ou valeur invalide gardent le chemin stock. Les profils Tune portent une clé distincte. Le gain ci-dessus concerne cette option activée ; ni release, ni push, ni remplacement de l'app installée n'a eu lieu.

[Résumé et conditions complets](measurements/qwen27-max-20261016/lane19/confirmed-summary.json), [ABBA](measurements/qwen27-max-20261016/lane19/abba-status.json), [BAAB](measurements/qwen27-max-20261016/lane19/baab-status.json), [répétition de l'oracle renforcé](measurements/qwen27-max-20261016/lane19/prefixes-values-status.json), [diff natif exact](measurements/qwen27-max-20261016/lane19/final-native.patch.txt), [revue indépendante et suivi](measurements/qwen27-max-20261016/lane19/independent-review-final.md).
