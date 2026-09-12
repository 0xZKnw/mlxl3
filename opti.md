# MLXL3 — journal des optimisations

### OPT-2026-09-12-RUST-01 — Port natif Rust/Metal — socle validé, migration en cours

- Demande : réécriture Rust sur une nouvelle branche GitHub. Branche
  `codex/rust-rewrite`, base `de318a8`, checkout isolé `../mlxl3-rust`.
- Hypothèse : une orchestration native peut réduire le coût CPU ; aucun gain
  de decode, prefill, TTFT ou RAM n'est établi par le changement de langage.
- Antécédents lus : journal complet, audit-runtime-2026-09-04 et roadmap decode.
  Les shaders MSL existants restent la référence ; ne pas changer leur calcul
  et leur ordonnanceur simultanément pour revendiquer un gain.
- Premier contrôle : codec CPU Rust (K=1..8, trois codebooks), lecture des
  checkpoints sans allocation GPU, registre compatible, découpage thinking,
  puis packing/décodage/QMV Metal appelés directement par Rust sans Python.
- Protocole : tests CPU déterministes, parité différentielle Python/Rust,
  tests Metal sur M5 avec erreurs propagées, tests CLI en dossiers temporaires.
  Aucun benchmark modèle complet avant les architectures et caches natifs.
- Environnement : M5/macOS local, Rust stable installé sans changer le PATH
  du shell. Versions exactes et résultats à consigner après exécution.
- Résultats/performance : non mesurés. Intégration : chantier de branche ;
  moteur de production, modèles et app installée inchangés.
- Première validation : cinq tests unitaires CPU puis test GPU exhaustif des
  196 608 codewords, K=1..8 pack/unpack réussis. Release compilée Rust 1.98.1.
  Douze contrats checkpoint/registre/CLI supplémentaires réussis. Pas de mesure
  de débit. Le test différentiel Python+Metal suivant est interrompu au premier
  accès GPU par le nouveau sandbox (`no Metal device`) ; relance hors sandbox
  nécessaire pour cette validation matérielle, mêmes données et même code.
- Checkout déplacé dans `work/mlxl3-rust` pour respecter les droits d'écriture
  actuels, branche inchangée. Route native MLX étudiée : libmlx 0.32.2 déjà
  installée, aucune dépendance Python du dylib. FFI mince Rust/C++ en cours
  pour conserver les kernels/fusions de production lors du portage modèle.
- Contrôle différentiel direct Metal, relancé avec accès GPU : **124 cas
  réussis**, comprenant 196 608 codewords CPU, tous K pack/unpack, codecs GPU
  et 72 QMV non nuls en FP16. Comparaison bit-à-bit avec les références Python.
  Le risque de différence FMA MUL1 trouvé à la lecture n'est pas reproduit dans
  cette matrice exécutée avec le compilateur Metal actuel. Pas un test modèle.
- Port du registre : parent relatif/override vide corrigés, CRLF coupé entre
  fragments corrigé dans Rust et dans la référence Python sur cette branche.
  **14 contrats Rust** et **10 parités de streaming Python/Rust** réussis.

### OPT-2026-09-12-RUST-02 — Couches et LFM2 via MLX natif — parité LFM2 validée

- Objectif : supprimer Python de l'orchestration, réutiliser les sources QMV
  de production à l'identique via libmlx 0.32.2 et une FFI Rust/C++ limitée.
  Aucun nouveau kernel mathématique pour revendiquer artificiellement un gain.
- Baseline : branche Python de `de318a8`, mêmes poids/sources MSL/MLX local.
  Candidat : feature Rust `mlx`, couches EXL3 sérialisées et architecture LFM2
  dense. Qwen, Gemma, quantification et GUI ne sont pas encore portés.
- Protocole : comparaison bit-à-bit de projections synthétiques tous K/CB,
  puis tokens imposés LFM2.5-1.2B-Thinking EXL3 4bpw, logits et caches à chaque
  étape. Première vérification token-par-token des deux côtés ; ce n'est pas
  une comparaison du prefill groupé Python à un nouveau prefill Rust.
- Conditions : même M5/macOS/MLX ; les tests matériels exigent accès Metal hors
  sandbox. Contexte/longueur imposés dans les commandes de preuve. Aucun gain
  de performance et aucune parité modèle encore établis à cette étape.
- Smoke GPU FFI `array::tests::native_array_and_kernel_smoke` : réussi,
  incluant transpose non contiguë, matmul, conversion FP16, kernel et erreurs.
  Build complet ensuite bloqué par `metal::device_info` non exporté du dylib ;
  détection M5 déplacée vers Metal natif. Validation complète à reprendre.
- Build `mlx,chat` réussi avec cible macOS 26.2. `check_parity.py --mlx`
  réussit **92 cas exacts** : 20 codecs CPU et 72 projections complètes
  (128×128, 1024×512, 2048×128 ; K1..8 × CB0..2, split-K inclus).
  Cinq tests tokenizer réussis dont prompt, IDs et décodage strictement égaux
  à Transformers pour une conversation LFM2.5-1.2B multilingue de quatre tours.
  Prochaine validation : modèle complet, huit tokens imposés du protocole.
- Première parité modèle : **rejetée**, step 0, cache couche 10 (7 octets
  différents sur 1024, écarts d'un bit). L'ordre lexicographique visitait 10
  avant 2 : correction du diagnostic pour identifier la première couche
  divergente en ordre d'exécution. Ce n'est pas une tolérance assouplie.
- Tri corrigé : couches 0..9 exactes, première divergence confirmée couche 10.
  Hypothèse : QKV groupé Python choisit un split-K différent des projections
  Rust séparées (512 sorties K/V vs 3072 groupées). Diagnostic une fois avec
  groupement Python désactivé ; son succès ne vaudrait pas parité production.
- Diagnostic confirmé : tous les logits et caches exacts au premier token
  sans groupement Python. Port du groupement QKV et de son shader mapped
  inchangé ; même split-K et même profondeur SIMD que la baseline groupée.
  Revalidation prévue des huit tokens contre production (groupement actif).
- **Validé numériquement** : groupement QKV natif intégré, huit tokens
  `1,2,3,19,225,4096,17,7`, tous logits et états KV/ShortConv bit-à-bit égaux
  au moteur Python de production. `cargo clippy --features mlx,chat
  --all-targets -- -D warnings` réussi. Preuve reproductible :
  `PYTHONPATH=src .venv/bin/python native/check_model_parity.py MODEL
  --binary target/debug/mlxl3-rs` (Python du dépôt parent pour ce worktree).
- Prochain contrôle fonctionnel : build release, génération greedy CLI sur
  le même LFM2, conversation suivie et `/clear`. Timing affiché expérimental,
  pas de benchmark comparable ni de gain revendiqué (prefill séquentiel).
- Chat réel : première réponse achevée « Bonjour. », 180 tokens, streaming
  thinking/réponse correct. Deux tours puis `/clear` et `/exit` réussis ; le
  second tour mentionne le prénom du premier mais atteint la limite 256.
  26 tests Rust passés (12 unitaires + 14 contrats), 11 checks Python passés.
  Le garde de provenance avait d'abord échoué sur un seul saut de ligne final ;
  ignore désormais uniquement les espaces/sauts finaux, pas le contenu des shaders.
- Extension de validation : 42 projections groupées K1..6/8 × trois CB × deux
  formes (dont QKV 2048/512/512) contre production ; 72 projections simples
  répétées pour vérifier l'intégration. Contrôle des arrêts Unicode ajouté au
  tokenizer ; génération CLI vidant le suffixe UTF-8 à la limite de tokens.
- Matrice étendue **validée : 134 cas exacts** (20 CPU + 72 projections
  simples + 42 groupes, chacun comprenant deux/trois sorties). Arrêts Unicode
  testés à chaque position d'une chaîne accentuée avec emoji : réussite.
  Inspection I16 de trellis ajoutée comme dans Python, avec contrat dédié ;
  bornes de grilles Metal protégées contre l'overflow.
- Contrôle de généralisation prévu : LFM2.5-2.6B EXL3 4bpw, mêmes huit tokens
  imposés, mêmes checks logits/caches, pour ne pas valider une seule taille.
- Généralisation **validée** : LFM2.5-2.6B EXL3 4bpw, huit étapes, tous les
  logits et tous les états bit-à-bit exacts contre production. Aucun changement
  spécifique de modèle requis. Clippy complet revalidé après les derniers
  gardes de bornes et d'I16. État : moteur Rust expérimental local, pas installé
  dans l'app et aucun gain temporel comparatif revendiqué.
- Vérification finale : 26 tests Rust réussis, lint sans warnings de notre
  code, tests de tokenizer/Unicode et provenance MSL réussis. Dépendance
  `block 0.1.6` signale une incompatibilité future Rust (pas une erreur actuelle).
  Build release actualisé et arrêt borné CLI revalidés avant publication.
  Aucun benchmark GPU ou quantificateur ne reste en cours. Les prochains ports
  (Qwen/Gemma/MoE, prefill groupé, quantification, GUI) restent à réaliser.

### OPT-2026-09-12-RUST-03 — Primitives Qwen3.5 MoE — en cours

- Hypothèse / changement : porter d'abord les deux primitives qui structurent
  le checkpoint local `Qwen3.6-35B-A3B-EXL3-2.49bpw` : mise à jour récurrente
  Gated DeltaNet Dk/Dv=128 et routeur 256→top-8. Réutiliser les shaders de
  production et la FFI MLX existante, sans nouveau backend ni copie CPU.
- Antécédents : `src/mlxl3/recurrent.py`, `src/mlxl3/moe.py`, implémentations
  `mlx_lm.models.qwen3_5` et `gated_delta.py` relues. Le checkpoint est bien
  Qwen3.5-MoE texte : 40 couches (30 linéaires, 10 attention), 256 experts,
  top-8, état récurrent FP32. MTP exclu conformément aux choix précédents.
- Baseline / candidat : Python `load_exl3_model` commit `de318a8` contre
  branche Rust `59ecef4`, mêmes entrées déterministes et même libmlx 0.32.2.
- Protocole prévu : parité bit-à-bit sortie + état GDN sur état nul et non nul,
  puis indices/scores du routeur y compris égalités/NaN ; enfin intégration
  token imposé et comparaison couche par couche. Aucun benchmark avant parité.
- Environnement : M5, macOS 27.0/SDK cible 26.2, GPU local. Résultats : non
  mesurés côté performance. Première étape validée le 12 septembre : le shader
  Gated DeltaNet empaqueté compile via libmlx 0.32.2 et ses sorties FP16 ainsi
  que son état FP32 sont bit-à-bit identiques à `mlx_lm` sur état nul et non
  nul. Commande : `native/check_parity.py --binary target/debug/mlxl3-rs
  --mlx`, 142/142 cas réussis (134 projections existantes + 2 GDN + 6 routeur).
  Le routeur natif restitue exactement indices et scores FP16 avec/sans
  normalisation, y compris égalités, zéros signés et NaN. L'intégration d'une
  couche Qwen complète reste non réalisée. Essai suivant enregistré avant
  modification : porter le QMV expert mappé déjà utilisé par Python (K=2/3/4,
  matrices synthétiques, routes répétées gate/up, sortie brute FP32 puis sortie
  FP16), et exiger la parité bit-à-bit avant le SwiGLU fusionné. Résultat :
  validé bit-à-bit sur les six variantes ; la matrice globale passe désormais
  148/148 cas. Le prochain essai sera le prepare SwiGLU+Hadamard puis la
  réduction pondérée du down, toujours face aux helpers Python inchangés.
  Première compilation interrompue avant benchmark : le générateur Rust des
  sept étages Hadamard avait deux références `str` aux durées de vie distinctes
  lors de leur permutation ; aucune exécution GPU ni mesure. Correction prévue :
  une durée de vie commune, sans changement du shader.
  Deuxième exécution validée : prepare SwiGLU/down et réduction pondérée sont
  bit-à-bit identiques aux helpers MLXL3 Python ; matrice 150/150. Essai suivant
  enregistré : chaîner les deux QMV mappés et ces transforms dans une structure
  `Exl3SwitchGlu`, puis comparer sa sortie finale à `EXL3SwitchGLU` sur un bloc
  synthétique top-2. Résultat : validé bit-à-bit, matrice 151/151 ; le chemin
  expert complet reste entièrement sur MLX/Metal. Aucun benchmark de vitesse
  avant la parité d'une couche réelle. Essai réel enregistré avant modification :
  charger uniquement le MLP MoE de la couche 0 du checkpoint Qwen local, entrée
  FP16 déterministe `[1,2048]`, et comparer sa sortie au même assemblage Python
  (gate dense, top-8, experts EXL3, expert partagé) bit-à-bit ; aucune génération
  ni chargement simultané de deux modèles complets.
  Première compilation de ce jalon interrompue : référence vers un nom de
  scale legacy temporaire dans le loader Rust ; aucune donnée modèle chargée
  et aucune mesure. Correction limitée à posséder la chaîne avant l'appel.
  Deuxième exécution validée le 12 septembre : `native/check_qwen_moe.py`
  compare le MLP MoE réel de la couche 0 et obtient une égalité FP16 bit-à-bit.
  Le script ne matérialise côté Rust qu'une couche d'experts, et aucune mesure
  de débit n'est revendiquée. Prochaine étape : bloc Gated DeltaNet complet de
  couche 0 avec état nul/non nul, avant assemblage des 40 couches. Protocole
  enregistré : deux entrées FP16 déterministes `[1,1,2048]`, mêmes poids réels,
  comparaison bit-à-bit des sorties, du cache convolutionnel FP16 et de l'état
  récurrent FP32 après chaque token ; arrêt au premier désaccord.
  Résultat : validé sur les deux tokens avec égalité bit-à-bit de la sortie et
  des deux caches (`native/check_qwen_gdn.py`). Essai suivant enregistré :
  assembler la couche linéaire 0 complète avec les deux RMSNorm corrigées par
  le sanitizer Qwen (`weight + 1`), les résidus et le MLP MoE déjà validé ; une
  entrée réelle déterministe, comparaison FP16 exacte avant tout benchmark.
  Résultat : couche linéaire 0 validée bit-à-bit, sortie et deux caches inclus
  (`native/check_qwen_layer.py`). Essai suivant enregistré : couche attention
  complète 3 sur deux tokens, RoPE partiel 64/256 à offsets 0 puis 1, caches KV,
  résidus et son MLP MoE ; égalité bit-à-bit requise, débit non mesuré.
  Première exécution interrompue dans l'oracle Python avant calcul : l'API
  `mx.fast.rope` 0.32.2 exige `scale=1.0` explicite. Aucun résultat candidat ;
  protocole inchangé après correction de l'appel de référence.
  Deuxième exécution validée : couche attention 3 exacte sur deux tokens,
  sorties FP16 et caches K/V compris (`native/check_qwen_attention.py`). Les
  deux types de couche du modèle sont donc couverts isolément. Essai suivant
  enregistré : assemblage des 40 couches, embedding/norm/head puis comparaison
  couche par couche et logits pour un token imposé ; surveiller la RAM processus
  pendant le chargement et ne pas lancer de benchmark de vitesse avant parité.
  Premier essai complet rejeté : oracle Python produit en 7,57 s avec empreinte
  mémoire pic rapportée 12,95 GB ; candidat Rust contrôlé en 66,32 s via le
  processus Python de comparaison, mais 239045/248320 logits FP16 diffèrent.
  Les chiffres mémoire du wrapper ne couvrent pas correctement tous les enfants
  et ne sont pas comparables. Aucune conclusion de performance. Diagnostic
  enregistré : tracer les sorties après chaque couche dans deux processus
  successifs et identifier la première divergence, sans modifier les tolérances.
  Une relance du diagnostic a été interrompue avant chargement : `python`
  n'est pas présent dans le `PATH` de ce worktree. Aucun calcul ni résultat ;
  relance inchangée avec l'interpréteur `.venv` absolu du dépôt parent. Cette
  relance localise la première divergence dès `layer_0` : 1 923/2 048 valeurs
  FP16 diffèrent, première valeur 40979 contre 40981 en représentation brute.
  Le test isolé de cette même couche étant exact, la prochaine vérification
  compare son entrée et les chemins exacts des deux oracles avant tout patch.
  Essai enregistré avant modification : inclure l'embedding du token 1 dans
  les deux traces, exiger son égalité bit-à-bit, puis conserver le diagnostic
  couche par couche inchangé. Si l'embedding est exact, comparer les états de
  couche 0 et l'effet des frontières d'évaluation, sans toucher aux kernels.
  Résultat : embedding exact ; la première divergence reste `layer_0` avec
  les mêmes 1 923/2 048 valeurs. Le loader et la sélection de token sont donc
  écartés. Prochaine comparaison : appel réel de `Qwen3_5MoeDecoderLayer`
  contre l'oracle manuel isolé, en inspectant notamment le masque SSM et les
  conversions de dtype ; aucun changement de tolérance ni benchmark prévu.
  Inspection terminée : le masque initial est bien `None` et les dtypes sont
  FP16, mais `fuse_compatible_linear_groups` groupe en production Q/K/V,
  GDN QKV/Z et gate/up partagé. Les oracles isolés et le candidat Rust les
  exécutaient séparément, ce qui explique qu'ils soient exacts entre eux mais
  pas face au modèle chargé. Candidat enregistré : réutiliser `Exl3Group` via
  un seul helper de projections groupées/fallback, puis rerun de la trace dès
  la couche 0 ; exiger ensuite les 40 couches et logits exacts. Premier rerun
  après groupement toujours rejeté, avec exactement 1 923 divergences dès la
  couche 0. Vérification suivante enregistrée : rejouer l'ancien oracle manuel
  sur entrée aléatoire pour confirmer que le groupement Rust est réellement
  sélectionné, puis tracer les sorties internes QKV/Z/MLP face au modèle Python.
  L'oracle aléatoire reste exact après le patch, donc le helper groupé ne casse
  pas ce cas. Sa variante avec l'embedding réel s'est arrêtée avant calcul GPU :
  NumPy ne sait pas importer directement le buffer BF16 brut. Aucun résultat ;
  convertir explicitement par MLX en FP16 comme le loader de production.
  Relance corrigée : sortie et caches de couche 0 toujours exacts sur l'embedding
  réel face à l'oracle manuel. Essai diagnostic suivant enregistré : générer une
  trace Python production avec seulement le groupement GDN QKV/Z désactivé par
  son option existante, puis comparer au même Rust. Cela isole cette fusion sans
  charger deux modèles simultanément ni modifier le candidat. Résultat : même
  divergence couche 0 ; cette fusion est écartée. Un diagnostic ponctuel des
  poids de norme a ensuite trouvé la cause : 1 723/2 048 coefficients de la
  norme d'entrée et 1 576/2 048 de la post-norme diffèrent. Le sanitizer Python
  calcule `BF16 + 1` puis convertit en FP16 ; l'oracle manuel et Rust faisaient
  `BF16 -> FP16` puis `+1`. Candidat enregistré : helper Qwen unique reproduisant
  l'ordre du sanitizer sur toutes les normes concernées (entrée/post, Q/K et
  finale), puis trace complète et logits bit-à-bit ; garder les groupes EXL3
  puisqu'ils reproduisent le graphe de production. Résultat après correction :
  embedding et couches 0 à 4 exacts ; première divergence déplacée à la couche
  linéaire 5, 1 013/2 048 valeurs, première 8974 contre 8972. Le correctif de
  sanitizer est donc validé sur les deux types de couche. Essai suivant
  enregistré : charger seulement la couche 5 Rust, lui fournir exactement la
  sortie Python de la couche 4 et comparer à `layer_5`, puis inventorier K/CB
  de ses projections contre les couches exactes ; caches initiaux nuls.
  Résultat ciblé : même divergence 1 013/2 048, donc l'état des couches
  précédentes est écarté. L'inventaire couche 0/1/2/4/5/6 est identique sur
  les projections structurantes (GDN et partagé K4/MCG, experts K3/MCG).
  Essai suivant enregistré : exposer seulement dans l'opération codec de
  diagnostic les cinq frontières de la couche 5 (norme entrée, GDN, résidu,
  post-norme, MLP), produire les mêmes frontières Python et arrêter à la
  première divergence. Le chemin normal conserve une seule implémentation.
  Première compilation interrompue : la méthode de trace a été insérée sur
  l'autre type de couche portant le même `forward`, donc `LinearLayer::trace`
  est absent. Aucun modèle ni GPU exécuté. Déplacer ce refactor dans
  `LinearLayer` et restaurer l'autre couche, sans changer le protocole.
  Deuxième compilation et trace ciblée réussies : norme d'entrée couche 5
  exacte ; première divergence dans la sortie Gated DeltaNet, 977/2 048
  valeurs (première position 7). Résidu/MLP non interprétés après ce point.
  Essai suivant enregistré : comparer les références Python groupée/non
  groupée déjà produites et tracer QKV/Z, convolution puis update récurrente
  de la GDN couche 5. Cela départage projection, convolution et kernel d'état.
  Résultat : QKV, Z, A/B, convolution, Q/K/V normalisés, beta, softplus, decay,
  sortie récurrente et RMSNorm sont tous exacts. Première divergence à la
  sortie finale de la GDN. Essai suivant enregistré : tracer le produit gated
  FP16 juste avant `out_proj`; s'il est exact, isoler `out_proj`, sinon corriger
  l'ordre précis SwiGLU. Pas de modification du calcul avant ce résultat.
  Résultat : une seule valeur gated diffère sur 4 096 (index 3 242), puis son
  amplification par `out_proj` explique les 977 écarts. Essai suivant enregistré :
  rejouer sur Z/RMS exacts l'expression MLX inline, `nn.silu` compilé et le helper
  Qwen compilé, comparer leurs bits à la référence et au Rust. Corriger ensuite
  la frontière de compilation, pas le QMV déjà établi exact. Résultat diagnostic :
  expression inline 43383 contre référence 43382 à l'index 3 242 ; `nn.silu`
  compilé et helper Qwen tous deux exacts. Le bridge utilise désormais cette
  frontière compilée ; la couche 5 ciblée repasse entièrement bit-à-bit exacte.
  Essai suivant enregistré : trace complète des 40 couches et logits avec ces
  deux corrections, arrêt au premier écart ; aucune mesure de performance.
  Résultat : embedding, 40 couches, norme finale et 248 320 logits sont tous
  bit-à-bit exacts (`native/diagnose_qwen_trace.py`). Répétition finale prévue
  avec `native/check_qwen_model.py`, oracle logits indépendant déjà produit,
  pour vérifier le contrat public sans données de diagnostic additionnelles.
  Répétition réussie : 248 320/248 320 logits FP16 exacts pour le token 1,
  66,85 s de temps mur en build debug ; ce temps inclut le chargement et n'est
  pas un benchmark d'inférence. Essai suivant enregistré : séquence imposée
  `1,2,3` dans une seule instance Python puis Rust, logits exacts à chaque pas,
  afin de valider caches GDN/KV et offsets avant branchement au chat natif.
  Résultat : trois étapes, chacune 248 320 logits FP16, toutes exactes. Les
  caches GDN/KV et offsets natifs sont validés sur cette séquence courte.
  Étape fonctionnelle enregistrée : ajouter `reset` aux caches Qwen et choisir
  LFM2/Qwen par `model_type` dans l'unique boucle de chat existante, sans dupliquer
  tokenizer/streaming/sampling. Smoke release Qwen borné à quelques tokens,
  puis `/clear` seulement si le smoke non interactif réussit. Première validation :
  26 tests Rust réussis, mais Clippy interrompt la chaîne avant build release sur
  deux variantes d'enum trop grandes (`ProjectionBundle`, `Layer`). Aucun smoke
  modèle lancé. Correction prévue : boxer seulement les variantes lourdes comme
  indiqué par le lint, puis relancer lint/build/tests sans changer le graphe MLX.
  Deuxième lint encore interrompu avant build : après avoir boxé la variante
  linéaire, la variante attention est devenue la plus grande. Boxer les deux
  variantes de `Layer`; aucune exécution modèle ni mesure entre ces deux lints.
  Lint strict et build release réussis après les deux boxes. Smoke non interactif
  Qwen réussi : prompt « Salut », 4 tokens générés, streaming thinking actif,
  fin propre. Mesures indicatives seulement (prefill séquentiel) : 2,4 tok/s,
  decode 10,2 tok/s, TTFT 5 094 ms après chargement ; un seul run, aucune
  comparaison Python. Dernier contrôle fonctionnel prévu : `/clear` entre deux
  prompts courts dans le même processus, pour vérifier le reset des 40 caches.
  Contrôle réussi : deux prompts de 2 tokens générés séparés par `/clear`, puis
  `/exit`; aucun crash, état résiduel ni erreur. Les débits courts 8,3/8,5 tok/s
  ne constituent pas un benchmark. Avant publication du jalon : corriger les
  anciens oracles ciblés qui reproduisaient l'ancien ordre FP16/+1, relancer
  leurs parités, tests Rust, Clippy et provenance.
  Validation finale réussie : anciens oracles corrigés pour l'ordre BF16/+1 et
  le SwiGLU précis ; GDN couche 0 sur deux pas, couche linéaire 0, attention
  couche 3 sur deux pas et MoE couche 0 tous bit-à-bit exacts. Suite codec/MLX
  **151/151**, tests Python **11/11**, tests Rust **26/26** (plus 3 tests matériel/
  tokenizer ignorés explicitement), Clippy strict et build release réussis.
  Le chat Qwen natif, son streaming et `/clear` sont donc validés localement.
  Intégration : prototype de branche uniquement ; GUI/app installée et moteur
  Python de production inchangés. Prefill Rust encore séquentiel, aucun gain de
  vitesse revendiqué ni comparé au moteur existant.

### OPT-2026-09-12-RUST-04 — LFM2 MoE natif — en cours

- Hypothèse / changement : étendre l'unique implémentation LFM2 Rust au type
  `lfm2_moe`, en conservant opérateurs, caches, chat et couches denses existants.
  Réutiliser `Exl3SwitchGlu` pour les couches expertes ; ajouter seulement les
  noms LFM `w1/w3/w2` et la sélection top-k biaisée requise par l'architecture.
- Antécédents consultés : `mlx_lm.models.lfm2_moe`, `src/mlxl3/moe.py`, port
  LFM dense validé dans RUST-02 et primitives MoE exactes de RUST-03. Le modèle
  local est LFM2.5-8B-A1B, 24 couches, 32 experts/top-4, deux couches denses,
  biais expert et normalisation des scores.
- Baseline / candidat : moteur Python de `a654a64` contre branche Rust au même
  commit, checkpoint EXL3 3.10 bpw local. Aucun kernel expert nouveau : même
  chemin Metal déjà validé sur Qwen, avec routeur biaisé 32 voies.
- Protocole prévu : d'abord bloc MoE réel couche 2 (routes, scores et sortie
  FP16 exacts), puis token imposé couche par couche et enfin plusieurs tokens
  avec tous caches. Build/chat seulement après parité ; aucun benchmark ni gain
  revendiqué pendant ce port. Environnement : M5/macOS, MLX 0.32.2, alimentation
  et thermique non contrôlées puisque seules des comparaisons exactes sont prévues.
- Première primitive validée : routeur biaisé LFM 32 voies/top-4, indices et
  scores FP16 bruts bit-à-bit identiques au kernel Python. La matrice MLX passe
  **152/152** cas. Le bloc LFM utilise ensuite les primitives expertes déjà
  validées, avec normalisation et facteur effectués dans le même ordre MLX ;
  prochaine preuve : MoE réel couche 2 avant toute exécution du modèle complet.
- Révision du protocole avant exécution : ne pas ajouter un codec de diagnostic
  permanent uniquement pour ce bloc. Le checker LFM complet existant exerce le
  même chemin et compare tous logits/caches ; commencer par un seul token. En
  cas d'écart seulement, ajouter une trace éphémère couche 2 pour localiser la
  divergence, puis la retirer. Cela réduit le code de test sans relâcher le
  critère bit-à-bit.
- Premier modèle complet réussi : token imposé `1`, tous les logits FP16 et
  tous les caches conv/KV du LFM2.5-8B-A1B EXL3 3.10 bpw sont bit-à-bit égaux
  au moteur Python. Aucune trace supplémentaire n'est donc ajoutée. Répétition
  suivante enregistrée : huit tokens du protocole LFM dense, même instance et
  états conservés, afin de valider routing changeant et progression des caches.
- Séquence complète réussie : `1,2,3,19,225,4096,17,7`, huit sorties vocabulaire
  et tous les états des 24 couches bit-à-bit exacts. Le port LFM MoE est donc
  validé numériquement sur ce checkpoint. Étape suivante enregistrée : Clippy,
  build release, chat borné puis `/clear`; timings purement indicatifs puisque
  le prefill natif reste token-par-token.
- Validation fonctionnelle réussie : Clippy strict, 26 tests Rust et build
  release passent. Chat LFM MoE borné à 4 tokens puis deux prompts séparés par
  `/clear` terminent proprement. Le smoke isolé a affiché ~64,8 tok/s decode et
  548 ms TTFT ; séquences trop courtes, sans paire Python, donc aucune conclusion
  de performance. Intégration : moteur/CLI Rust de branche seulement ; GUI,
  quantification et app installée inchangées.

### OPT-2026-09-12-RUST-05 — Qwen3.5 dense natif — en cours

- Hypothèse / changement : accepter le `qwen3_5` dense du checkpoint local
  Qwen3.8-27B 2.75 bpw en réutilisant intégralement attention, Gated DeltaNet,
  caches, normes et head du port Qwen MoE exact. Seul le MLP devient une variante
  gate/up groupée + SwiGLU + down ; aucune logique vision ni MTP.
- Antécédents consultés : `mlx_lm.models.qwen3_5`, Qwen3NextMLP, RUST-03 et
  inventaire réel du checkpoint. Les 64 couches ont le même cycle trois GDN/
  une attention et les poids conv non sanitisés exigent le même `weight + 1`
  déjà corrigé. Le checkpoint Gemma n'est plus présent, donc aucun port Gemma
  non vérifiable n'est tenté maintenant.
- Baseline / candidat : production Python du commit `ecf73a4`, même checkpoint
  local de 12 GB, contre moteur Rust sur cette branche. Protocole prévu : oracle
  Python écrit sur disque puis processus libéré, Rust ensuite, afin de ne pas
  garder deux modèles de 12+ GB simultanément. Token 1 puis séquence 1/2/3,
  logits FP16 bit-à-bit ; chat seulement après succès. Performance non mesurée.
- Premier contrôle réussi : oracle Python écrit en processus séparé, puis
  248 320/248 320 logits Rust exacts pour le token 1. Aucun modèle concurrent
  ni comparaison de temps (le candidat était un build debug). Prochaine
  répétition enregistrée : séquence imposée 1/2/3 dans une instance de chaque
  moteur, toujours séquentiellement, pour exercer caches GDN/KV et offsets.
- Séquence stateful réussie : trois fois 248 320 logits FP16 exacts. Les caches
  GDN/KV et offsets du Qwen3.8 dense sont donc validés indirectement à chaque
  étape sans conserver deux modèles en RAM. Étape suivante enregistrée : lint,
  tests, build release, chat court et reset ; aucune mesure comparative.
- Première chaîne finale interrompue par Clippy avant tests/build : la variante
  MoE de l'enum MLP est ~984 octets contre ~248 pour la dense. Boxer uniquement
  la variante MoE comme recommandé, puis relancer la même chaîne ; aucun modèle
  ni benchmark exécuté pendant cet échec.
- Deuxième lint encore interrompu : après ce box, la variante dense de 248 octets
  dépasse à son tour la petite variante. Boxer aussi la dense, comme pour l'enum
  de couches Qwen déjà validé ; aucun changement du graphe MLX.
- Après les deux boxes : Clippy strict, 26 tests Rust et build release réussis.
  Chat Qwen3.8 dense borné à quatre tokens terminé avec streaming thinking.
  Mesures indicatives défavorables : 7,1 tok/s prefill séquentiel, 5,4 tok/s
  decode, TTFT 45,2 s lors de cette première compilation/instance. Ce n'est pas
  un gain ; le port est correct mais pas encore performant face au moteur Python.
  Dernier contrôle fonctionnel prévu : deux prompts d'un token avec `/clear`.
- `/clear` validé : deux générations d'un token terminées dans le même processus,
  sans crash ni état résiduel observable. TTFT 43,1 puis 47,0 s, confirmant que
  le reset invalide/reconstruit aujourd'hui des graphes coûteux ; aucun benchmark
  comparable. Avant publication du jalon, répéter la séquence Qwen MoE 1/2/3
  avec son oracle conservé pour vérifier que l'enum MLP partagé ne régresse pas.
- Régression Qwen MoE réussie : trois étapes et tous les logits bit-à-bit exacts
  avec le build release. État : Qwen dense intégré au moteur/CLI Rust de branche,
  GUI et app installée inchangées ; aucune optimisation de ses temps encore faite.

À lire **avant** toute optimisation ; à mettre à jour **avant et après chaque
essai**, y compris les essais ratés. Voir [AGENTS.md](AGENTS.md).

## Format des nouvelles entrées

```text
### OPT-AAAA-MM-JJ-NN — Nom — statut
- Hypothèse / changement :
- Antécédents consultés / raison de retester, le cas échéant :
- Baseline / candidat (commit + diff ou options) :
- Environnement : matériel, OS, dépendances, alimentation, conditions connues.
- Protocole : modèle/bpw, shapes ou contexte, tokens, warmup, répétitions.
- Commandes et preuves :
- Résultats avant → après, unités et dispersion :
- Qualité / exactitude vérifiée ; contrôles non effectués :
- Conclusion / limites / prochaine condition de réexamen :
- Intégration : prototype / code local / app / publication, selon vérification.
```

## Historique à consulter avant de proposer une piste

Journal initialisé le 10 septembre 2026 à partir des rapports existants.
**Ce résumé n'est pas une transcription exhaustive des anciens essais.**
Pour les domaines concernés, lire également ces archives et rechercher la
piste dedans ; elles restent la source des protocoles et résultats détaillés.

| Domaine | Rapports existants |
| --- | --- |
| Decode général, Qwen, LFM | [10 septembre](docs/decode-investigation-2026-09-10.md), [R&D du 7 septembre](docs/general-performance-2026-09-07.md), [roadmap decode](docs/decode-roadmap-2026-09-04.md), [audit runtime](docs/audit-runtime-2026-09-04.md) |
| Qwen3.8 dense | [Essais decode Qwen3.8](docs/qwen38-decode-local.md) |
| Gemma | [Decode](docs/gemma-decode-investigation.md), [SDPA512](docs/gemma-sdpa512-investigation.md) |
| Prefill après outils/MCP | [R&D prefill](docs/tool-prefill-rd-2026-09-05.md) |
| Quantification EXL3 Metal | [Optimisations quantification](docs/metal-quantization-optimization.md), [LFM2.6](docs/lfm26-local-quantization.md), [Ling](docs/ling-local-quantization.md) |
| UI, streaming, sessions | [R&D du 7 septembre](docs/general-performance-2026-09-07.md), [audit v1](docs/audit-v1-2026-09-06.md), [validation v1](docs/v1-validation.md) |

Les statuts ci-dessous décrivent les conclusions des rapports à leur date,
pas une vérification de la version actuellement installée ou publiée.
Les chemins `build/` sont des preuves locales temporaires, non garanties dans Git.

## 10 septembre 2026 — recherche decode générale

Source commune : [rapport complet et preuves](docs/decode-investigation-2026-09-10.md).
M5, 24 GiB, macOS 27.0 (26A428), MLX 0.32.2. Trois paires alternées,
96 tokens générés, warmup exclu. Batterie puis secteur, thermique non contrôlée.
Les différences ci-dessous sont les médianes des variations appariées.
Texte identique dans les paires ; pas de nouvelle validation bit-à-bit des
logits/caches. **Aucun changement de production retenu dans cette série.**

### OPT-2026-09-10-01 — Chargement TensorOps — bloqué

Le chargement normal de Qwen échoue au warmup sur
`get_destination_cooperative_tensor` (contrainte de template dans les headers
MetalPerformancePrimitives). Cause exacte non établie. Pour les essais 02–06,
les deux côtés utilisent `MLXL3_TENSOR_QMM=0 MLXL3_TENSOR_SEGMENTED_QMM=0`.
Ce contournement de benchmark n'est pas activé dans l'app ; les mesures ne
valident pas le chemin prefill TensorOps normal. Réexaminer après correction
de compatibilité macOS 27.

### OPT-2026-09-10-02 — Budgets command-buffer, Qwen — rejeté

- Candidat : `MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000` vs défauts.
- Qwen3.6-35B-A3B EXL3 2.49 bpw : decode non caché **−0,38 %**.
- Pic d'allocation MLX : **12,422 → 13,302 GB**. Un fort gain apparent sur une
  paire provient d'une baseline subitement lente, pas d'un gain fiable.
- Preuve : `build/decode-command-buffers-qwen-fallback-20260910.jsonl`.

### OPT-2026-09-10-03 — Mêmes budgets, LFM8 — rejeté

- LFM2.5-8B-A1B EXL3 3.10 bpw : decode **−2,41 %**, caché **−1,59 %**.
- Preuve : `build/decode-command-buffers-lfm8-20260910.jsonl`.

### OPT-2026-09-10-04 — Compiler 24 feed-forward LFM8 — non concluant

- Réutilisation du compilateur stateless existant : decode **+0,11 %**, bruit.
- Preuve : `build/decode-ff-compile-lfm8-all-20260910.jsonl`.
- Le pilote limité à deux blocs n'a pas établi de gain non plus ; ne pas
  additionner les résultats de ce pilote et de la série complète.

### OPT-2026-09-10-05 — Compiler 30 feed-forward LFM2.6 — non concluant

- LFM2.5-2.6B EXL3 4 bpw : decode **+0,30 %**, bruit.
- Preuve : `build/decode-ff-compile-lfm26-20260910.jsonl`.

### OPT-2026-09-10-06 — Compiler les entrées QMV partagées — rejeté

- `mx.compile` des fonctions denses/groupées/experts réellement appelées.
- LFM8, série séparée sur secteur : decode **−3,06 %** médian.
- Preuve : `build/decode-qmv-compile-lfm8-ac-20260910.jsonl` ; runner local :
  `build/bench_decode_ff_compile.py` (`--qmv` pour cette variante).
- Attention : l'ancien `work/compiled_qmv_bench.py` cible des alias obsolètes.
  Ne pas utiliser ses résultats comme validation du chemin actuel.

## 7 septembre 2026 — résultats antérieurs à ne pas redécouvrir

Source et protocoles : [rapport général](docs/general-performance-2026-09-07.md).
Les validations rapportées ici ne lèvent pas le blocage macOS 27 découvert
le 10 septembre. Les gains portent uniquement sur le périmètre indiqué.

| ID | Essai | Résultat documenté | Conclusion historique |
| --- | --- | --- | --- |
| OPT-2026-09-07-01 | Sortir les calculs d'adresse QMM de la boucle | Prefill LFM2.6 +6,84 %, Qwen +6,09 % ; decode dans le bruit ; logits/caches forcés bit-à-bit | Validé sur le chemin TensorOps de l'époque |
| OPT-2026-09-07-02 | Évincer les snapshots avant la session entière | TTFT second tour 5,1517 → 0,1673 s ; 3 500 tokens réutilisés dans le scénario contraint | Validé pour ce scénario, pas un gain decode universel |
| OPT-2026-09-07-03 | Préparation incrémentale des blocs de code UI | À 1 MiB : 52,482 → 1,640 ms par ajout | Validé en microbenchmark CPU, pas FPS/tok/s |
| OPT-2026-09-07-04 | Comptages de chaînes bornés | Grande ligne ASCII 3,561 → 2,946 ms ; Unicode 54,674 → 29,909 ms | Validé séparément, ne pas multiplier les gains |
| OPT-2026-09-07-05 | Validation UUID du bridge sans attendre la queue IO | Délai synthétique 53,877 → 0,034 ms quand IO occupée 50 ms | Validé pour la réactivité du bridge, pas le decode |
| OPT-2026-09-07-06 | Omettre les diagnostics quant non utilisés | Temps −1,51 % médian, quatre projections K=4 ; fingerprints exacts | Mesure limitée, remplacée par la comparaison combinée suivante |
| OPT-2026-09-07-07 | Quatre changements de préparation quant combinés | Temps −2,46 % médian ; fingerprints/scores exacts | Validé sur quatre projections, pas conversion complète ; ne pas ajouter −1,51 % |
| OPT-2026-09-07-08 | Supprimer copie gate/up MoE prefill | Prefill Qwen environ −3 à −6 % malgré contrôles numériques réussis | Rejeté |
| OPT-2026-09-07-09 | Fusion gather/Hadamard decode | LFM8 +0,46 % non caché, +0,02 % caché ; Qwen instable | Retiré, gain non établi |
| OPT-2026-09-07-10 | Extraction QMM 64 bits → funnel 32 bits | Plus lent, pourcentage non précisé dans ce rapport | Rejeté |
| OPT-2026-09-07-11 | Préchargement des mots QMV | Résultats synthétiques mixtes | Non retenu |
| OPT-2026-09-07-12 | Grouper les projections LFM w1/w3 | Decode +1,33 %, prefill −6,59 %, environ +171 MB pic requête cachée | Rejeté ; résoudre les copies compactantes avant réessai |
| OPT-2026-09-07-13 | Cacher les métadonnées récurrentes vides | Wrapper CPU ~580–600 → 403–434 ns ; pas de gain modèle mesuré | Retiré, impact estimé négligeable |

## Pistes ouvertes, non validées

- Corriger la compatibilité TensorOps macOS 27 avant les nouvelles validations
  bout en bout du chemin normal.
- Étudier des régions decode compilées plus larges : attention, normalisation,
  RoPE et cache ensemble ; les blocs Qwen récurrents et plusieurs MLP sont
  déjà compilés. Les essais isolés 04–06 n'ont pas montré de gain.
- Projections groupées mixtes (bits/codebooks) et suppression des copies
  compactantes : vérifier les formes réellement exclues avant tout prototype.
- Profiler les dispatchs GPU et les lectures réelles avant une nouvelle variante
  QMV ; taille du fichier / tok/s n'est pas une mesure de bande passante GPU.

## Nouvelle série du 10 septembre 2026

### OPT-2026-09-10-07 — Masquage top-k par les tokens conservés — non concluant

- Hypothèse : conserver exactement `argpartition(-logprobs, kth=k-1)`, mais
  remplacer le scatter des V-k tokens rejetés par un tableau rempli de -inf
  et un scatter des k valeurs conservées, économise des écritures irrégulières.
- Antécédents : D07 du roadmap/audit proposait le sampler ; aucun essai de ce
  changement précis trouvé. Pas de changement du greedy, du RNG ou de top-k.
- Baseline : `mlx_lm.sample_utils.apply_top_k` installé, MLX 0.32.2.
- Protocole : parité des sorties FP16/BF16/FP32, ties/NaN/Inf et seeds ; puis
  microbenchmark vocab 65 536/248 320/262 144, k=40. Si favorable, trois paires
  modèle à température 0,7, top-k 40, seeds identiques, 128 tokens.
- M5 24 GiB, macOS 27, secteur (82 %), thermique non contrôlée ; workaround
  TensorOps de l'essai 01 identique des deux côtés pour les essais modèle.
- Commande/prototype : `.venv/bin/python benchmarks/bench_topk_scatter.py` ;
  preuves prévues `build/topk-scatter-20260910.jsonl` et fichiers modèle séparés.
- Résultat : 252 cas de masques bit-à-bit, sampling avec seed contrôlé réussi.
  Microbenchmark favorable mais Qwen decode +0,70 % médian, paires
  [+0,70 ; −0,86 ; +16,27] %, trop instables pour promotion.
  Preuve modèle : `build/topk-scatter-qwen-20260910.jsonl`.
- Reproduction modèle avec le runner actuel : ajouter
  `--model models/Qwen3.6-35B-A3B-EXL3-2.49bpw --temperature 0.7 --top-k 40`.
  Les défauts ultérieurs du runner sont ceux de l'essai 08 (0,2 / 80).
- Intégration : diagnostic uniquement, pas l'app.

### OPT-2026-09-10-08 — Top-k hiérarchique exact — non concluant

- Nouvelle hypothèse : dans le backend Metal MLX installé, ArgPartition appelle
  un tri complet. Trier des blocs, retenir leurs k meilleurs puis trier ces
  candidats évite les grandes fusions du tri global. Aucun token du top-k
  global ne peut être exclu par un top-k local. Les ties doivent conserver
  l'ordre initial via les tris stables du backend actuel ; contrôler NaN/Inf.
- Différence avec 07 : réduit le travail du tri, pas seulement le masquage.
- Prototype : `benchmarks/bench_topk_scatter.py --hierarchical`. Bloc 1024,
  fallback référence si petit vocabulaire ou k >= bloc. Padding NaN en fin,
  comme les sentinelles du tri MLX. Non destiné aux backends non validés.
- Contrôles prévus : masques bit-à-bit FP16/BF16/FP32, ties, NaN/Inf, bords de
  blocs, k=1/40/80/V−1 ; sampling avec même seed ; puis modèle température
  0,2/top-k 80 (défauts GUI), trois paires de 128 tokens si parité réussie.
- Mêmes matériel/OS et workaround que 07. Prototypes seulement, non intégrés.
- Preuves : `build/topk-hierarchical-20260910.jsonl` (1 020 cas bit-à-bit,
  sampling seed identique), `build/topk-hierarchical-qwen-20260910.jsonl`.
  Qwen : decode +0,38 % médian, paires [−0,36 ; +1,54 ; +0,38] %, bruit.
  LFM1.2 Thinking : −0,96 % médian, paires [−0,96 ; −6,53 ; +6,39] %,
  textes identiques ; `build/topk-hierarchical-lfm12-20260910.jsonl`.
  Pas de gain modèle fiable, pas de promotion malgré le gain du filtre isolé.
- Microbenchmark synchronisé (temps CPU+GPU, pas temps GPU pur) : réduction
  médiane du temps top-k de **18,99 % / 33,71 % / 33,60 %** pour les vocabulaires
  65 536 / 248 320 / 262 144. Ne pas appliquer ces pourcentages au decode.

### OPT-2026-09-10-09 — Compiler la pénalité de répétition bornée — rejeté

- Hypothèse : compiler gather/arithmétique/scatter de la pénalité existante,
  avec la fenêtre de tokens bornée avant l'entrée compilée pour ne pas créer
  un graphe par longueur d'historique. Ne change pas la pénalité ni son ordre.
- Antécédent D01 : borne déjà l'historique stable du préfixe, pas cette fusion
  du processeur. Les tests QMV/FFN 04–06 concernaient d'autres opérations.
- Référence : factory MLX-LM installée. Candidat : même closure sous
  `mx.compile`, entrée `tokens[-20:]`, cache borné des factories dans le prototype.
- Protocole : comparaison bit-à-bit sur dtypes, tokens répétés, signes et
  longueurs de fenêtre ; microbenchmark puis trois paires greedy modèle si
  exact. Même environnement/workaround que 08 ; pas de top-k modifié simultanément.
- Prototype/proofs : `benchmarks/bench_topk_scatter.py --penalty`,
  `build/penalty-compile-20260910.jsonl` puis fichiers modèle séparés.
- Premier contrôle de parité échoue : FP16, vocab 257, historique 21 tokens,
  pénalité 0,8. Arrêt avant benchmark modèle. Diagnostic des différences en
  FP16 confirme aussi un écart avec la pénalité par défaut 1,05 :
  0,65185546875 → 0,65234375 (un logit fini). Preuve :
  `build/penalty-compile-diagnostic-20260910.jsonl`. Aucun gain revendiqué.
  Non intégré dans le moteur/l'app.

### OPT-2026-09-10-10 — Types explicites TensorOps macOS 27 — validé en prototype

- Suite de 01 justifiée par l'inspection des nouveaux headers MPP : les
  contraintes de types n'effacent pas l'espace d'adressage `thread` porté
  par `decltype(variable)`. Tester les mêmes types tensor/cooperative_tensor
  via leurs alias explicites sans ce qualificatif ; ne pas changer le calcul.
- Ce n'est pas un boost decode annoncé : rétablir un chemin normal de prefill
  est nécessaire aux mesures complètes. Pas de modification des headers Apple.
- Prototype : intercepteur local de la factory Metal dans un runner de test,
  puis suite QMM existante. Même MLX/macOS, aucun TensorOps désactivé.
- Commande : `.venv/bin/python build/check_tensor_types.py` ; sortie
  `build/tensor-types-check-20260910.log` (premiers 24 cas), puis
  `build/tensor-types-full-20260910.log` : **87 tests QMM réussis**.
- Chargement + warmup + 32 tokens Qwen avec TensorOps activé réussis :
  `.venv/bin/python build/check_tensor_types.py --model models/Qwen3.6-35B-A3B-EXL3-2.49bpw` ;
  `build/tensor-types-qwen-20260910.jsonl`, stderr vide. Une seule génération,
  pas un A/B de performance. Parité macOS 26/27 non mesurée ; pas de promesse
  bit-à-bit inter-OS ni validation exhaustive de tous les modèles.
- Correctif testé dans les factories `_qmm_tensor_kernel` et
  `_segmented_expert_qmm_tensor_kernel` : remplacer les arguments
  `decltype(first_left), decltype(right), float` de l'accumulateur par
  `tensor<device half, dextents<int, 2>, tensor_inline>`,
  `tensor_ops::matmul2d<descriptor, execution_simdgroup>::cooperative_tensor_right_input_t<half, half, float>`,
  `float`. Les calculs, tuiles et poids ne changent pas.
- Intégration : intercepteur du runner de diagnostic uniquement. Sources de
  production, app installée et dépendances non modifiées. Avant intégration :
  ajouter un contrôle de source durable, vérifier l'autre OS supporté et le
  chemin segmenté sur une matrice dédiée. Pas de boost decode revendiqué.

## 11 septembre 2026 — objectif : au moins 10 % sur une métrique complète

### OPT-2026-09-11-01 — Borner les chunks de prefill en vol — non concluant

- Hypothèse : `_prepare` soumet tous les chunks par `async_eval` avant
  l'attente terminale ; borner cette file peut réduire le pic des temporaires
  sans changer taille des chunks, poids, cache ou précision.
- Antécédents : les changements de taille de chunks du 5 septembre ont été
  rejetés pour dérive numérique. Ici seul l'ordonnancement change. Aucun
  essai de backpressure prefill trouvé dans les archives consultées.
- Protocole : chemin réel `_stream_response`, prompts fixes ~8k puis ~16k
  selon mémoire, 16 tokens générés ; A/B synchronisation de chaque chunk
  stable, puis pipeline de deux chunks si utile. Comparer pic MLX, temps,
  texte et empreintes des caches ; confirmer en processus séparés si favorable.
- M5 24 GiB, macOS 27.0/26A428, MLX 0.32.2, batterie 46 %, thermique non
  contrôlée. Appliquer le correctif de types de 2026-09-10-10 aux deux côtés
  du benchmark, uniquement dans le prototype, pour garder TensorOps normal.
- Préserver les autres changements locaux ; aucune refonte/changement de
  langage, aucun push ni remplacement d'app sans validation supplémentaire.
- Preuves prévues : `build/prefill-flight-20260911*.jsonl`, runner dans
  `benchmarks/bench_prefill_flight.py`.
- Pilote LFM1.2 Thinking, 6 154 tokens : même pic MLX **1,851846224 GB**,
  TTFT 3,461 → 3,497 s ; texte et états finaux identiques. Pas de gain RAM ;
  ne pas poursuivre cette synchronisation systématique sur ce résultat.
  `build/prefill-flight-20260911-lfm12.jsonl`.

### OPT-2026-09-11-02 — Partager les KV du préfixe après génération — non concluant

- Hypothèse : dans `GenerationSession.finish`, le préfixe stable KV dupliqué
  est identique au début des KV de la génération terminée. Remplacer seulement
  ces copies KV par des forks/vues du cache final peut économiser la RAM
  retenue entre les tours, en conservant les états récurrents de chaque date.
- Différence avec COW au début de génération : pas de copie immédiatement
  déclenchée par les tokens générés. Réutilise `SharedKVCache` existant ; pas
  de pagination ni changement de langage. Mesurer séparément RAM active au
  repos et pic pendant la génération, ne pas confondre les deux.
- Prototype : option `--share-finished-kv` du runner 01, LFM1.2 Thinking,
  ~16k tokens, 16 générés ; baseline inchangée contre partage post-finish.
  Si favorable : trois paires, qualité préfixe/final, conversation suivante,
  branche au préfixe et snapshots indépendants ; vérification deuxième modèle.
- Mêmes environnement et correctif de types de benchmark que 01. Logs
  `build/finished-kv-20260911*.jsonl`. Aucun résultat mesuré ni intégration.
- Pilote 16 394 tokens : allocation MLX retenue **1 483 087 368 → 1 278 647 816
  octets (−13,78 %)** ; texte + préfixe + cache final identiques. Pic pendant
  génération inchangé (2,301 GB). Une seule paire, pas encore une validation.
- Étape suivante : vider aussi les allocations libérées après partage pour
  restituer la RAM au système, et mesurer `proc_pid_rusage.ri_phys_footprint`
  comme le widget. Contrôler séparément pool allocateur/MLX actif/empreinte OS.
  Candidat combiné partage + restitution ; ne pas additionner des pourcentages.
- Trois paires LFM16k : −13,79 % d'allocation MLX active retenue, mêmes textes
  et caches. **Pas de baisse nette d'empreinte physique macOS** (−1,95 %, +0,08 %,
  +0,05 % de réduction), pic inchangé. Ne pas présenter cela comme un gain
  de RAM système ou de pic. `build/finished-kv-20260911-lfm12-final.jsonl`.
  Non intégré ; contrôle des tours suivants encore requis pour promotion.
- Recherche suspendue sur ce candidat : pas de gain RAM système établi,
  aucune modification de `GenerationSession.finish` conservée en production.

### OPT-2026-09-11-03 — Étendre BM64 dense aux autres matrices — non concluant

- Antécédent : M64 existe déjà pour grandes matrices MUL1 à entrée >=4096.
  Nouvelle couverture testée : entrées 2048 et autres codebooks, uniquement
  lorsque les lignes sont multiples de 64, >=128, hors head >=65536 sorties.
  Garder BK16/BN32, donc même ordre de réduction ; parité à vérifier.
- Hypothèse : amortir la déquantification EXL3 sur 64 plutôt que 32 lignes,
  en évitant de modifier chunks/padding/poids. Pas de nouveau langage/refonte.
- Runner 01 option `--dense-m64`, LFM2.6 4 bpw, ~4k tokens, 16 générés ;
  comparaisons suivantes seulement si caches et texte exacts. Les deux côtés
  utilisent la correction de types TensorOps, sans autre candidat RAM combiné.
- M5/macOS27, batterie, thermique non mesurée. Logs
  `build/dense-m64-20260911*.jsonl`. Aucun résultat ni intégration.
- Pilote 4 107 tokens : 789,57 → 811,95 tok/s prefill (+2,84 %), TTFT
  5,227 → 5,071 s ; pic identique, texte/préfixe/cache final exacts.
  Une seule paire : non concluant pour promotion, pas de gain généralisé.
  `build/dense-m64-20260911-lfm26.jsonl`. Prototype seulement.

### OPT-2026-09-11-04 — BM64 TensorOps segmenté MoE — validé sur M5, code local

- Hypothèse : déquantifier chaque tuile de poids une fois pour 64 lignes au
  lieu de deux fois pour 32, dans les blocs experts déjà paddés à 64 lignes.
  Antécédents : BM8/16/32 testés, BM64 pas dans la matrice historique.
- Prototype : `_SEGMENTED_TENSOR_ROWS=64` en mémoire contre 32 ; buckets et
  locality désactivés, BK16/BN32 inchangés. Pas de refonte ni de nouveau langage.
- Qwen3.6 35B A3B EXL3 2.49 bpw, prompt fixe, 16 tokens, pilote puis trois
  paires alternées si texte et empreintes de tous les caches restent exacts.
  Même correctif TensorOps 2026-09-10-10 des deux côtés ; M5/macOS27 sur batterie.
- Runner `bench_prefill_flight.py --segmented-m64 --repeats 512`, preuves
  `build/segmented-m64-20260911*.jsonl`. Aucun résultat ni intégration.
- Pilote interrompu avant baseline : compilation Metal du chemin segmenté
  refuse la référence au descripteur local dans l'alias imbriqué utilisé par
  le correctif 10-10. Le warmup court précédent ne couvrait pas ce chemin.
  `build/segmented-m64-20260911-qwen.stderr`. Pas de mesure de performance.
- Révision du protocole : nommer les alias d'opération/opérande avant leur
  usage comme arguments template, sans changer leurs types ni les calculs.
  Réessai de compilation puis A/B dans un nouveau log `*-qwen-alias.jsonl`.
- Alias seul : même erreur, avant mesure. Le segmenté est inutilement instancié
  avec `T=half` alors que sa source utilise exclusivement `half`. Retirer ce
  paramètre template inutilisé dans le runner pour éviter la contrainte sur
  le descripteur local ; calcul inchangé. Log `*-qwen-no-template.jsonl`.
- Compilation rétablie. Pilote 4 106 tokens : prefill 414,33 → 517,90 tok/s
  (+25,00 %), TTFT 9,967 → 7,951 s ; cache/préfixe/texte exacts. Les compilations
  propres à la forme ne sont pas exclues : pas encore de gain validé.
- Confirmation prévue : une paire complète exclue (`pair=-1`), puis trois
  paires AB/BA, mêmes 4 106 tokens, options `--warmup-pairs 1 --pairs 3`,
  log `build/segmented-m64-20260911-qwen-warm.jsonl`. Contrôle numérique
  dédié : tous les codebooks et bits supportés, segments courts/64/65/129 et
  experts vides ; extension à LFM8 si les résultats restent favorables.
- Trois paires après warmup : prefill **+7,89 / +10,46 / +9,43 %** (médiane
  +9,43 %), TTFT **−7,26 / −9,41 / −8,64 %** ; caches et texte exacts.
  Le pilote +25 % surestimait le gain chaud. Pic ~14,05–14,06 GB inchangé.
  Gain prometteur mais objectif 10 % pas encore confirmé ; prototype seulement.

### OPT-2026-09-11-05 — Adresses invariantes dans le QMM segmenté — validé en combinaison, code local

- Antécédent : le hoisting du 7 septembre ne couvre que le QMM dense.
  Le QMM segmenté recalcule encore permutation/positions/shift à chaque BK16.
  Même technique appliquée aux adresses du bloc expert, sans changer les
  lectures ni l'ordre des additions. Garde de capacité coopérative conservée.
- Prototype dans la factory du runner : préparer offsets/shifts avant la boucle
  K, puis incrémenter uniquement la base de profondeur ; aucun poids déquantifié
  matérialisé. Comparer candidat combiné BM64 + hoisting à baseline BM32 normale,
  sans additionner les gains. Mêmes Qwen4k, warmup exclu, trois paires si pilote
  exact, puis matrice numérique dédiée. Logs `build/segmented-hoist-20260911*`.
- Aucune mesure ni intégration à ce stade.
- Pilote combiné chaud Qwen4k : 483,68 → 532,94 tok/s (+10,18 %), TTFT
  8,511 → 7,727 s (−9,21 %) ; pic 14,0523 GB identique, texte et caches exacts.
  Confirmation nécessaire. Avant cela, exécuter les 24 cas de
  `tests/test_segmented_m64.py` avec l'intercepteur compatible macOS27 et
  le hoisting activé : `build/segmented-m64-quality-20260911.log`.
- Matrice 24 cas réussie : BM32/BM64 avec hoisting actif des deux côtés,
  sorties finies et bit-à-bit. Cette matrice ne compare pas encore hoisting
  actif/inactif ; la comparaison combinée modèle vérifie ce dernier.
- Validation externe de forme : LFM8 A1B 3.10 bpw, même prompt répété 512,
  paire de warmup exclue puis trois paires, candidat BM64 + hoisting,
  log `build/segmented-hoist-20260911-lfm8-warm.jsonl`.
- LFM8, 4 106 tokens : trois gains prefill **+31,43 / +32,37 / +29,40 %**,
  TTFT **−23,80 / −24,40 / −22,67 %** ; textes et caches exacts. Pic ~4,923 GB
  inchangé. Candidat combiné en prototype uniquement ; objectif de mesure
  atteint sur ce scénario, sans conclusion llama.cpp ni decode universel.
- Ablation avant intégration : BM64 seul sur LFM8, mêmes trois paires chaudes,
  pour éviter de conserver un hoisting inutile ; log
  `build/segmented-m64-20260911-lfm8-warm.jsonl`.
- Intégrer maintenant uniquement le correctif de compilation macOS27 commun
  (alias de types explicites et suppression du template T inutilisé segmenté),
  puis contrôler les suites existantes. Cela n'est pas compté comme un boost.
- Ablation BM64 seul : LFM8 prefill **+22,97 / +23,67 / +22,98 %**,
  TTFT **−18,59 / −19,03 / −18,68 %**, texte/caches exacts, pic inchangé.
  Les séries ne sont pas directement additionnables ; l'essai combiné reste
  la preuve du gain cumulé. Garder les deux candidats pour validation locale.
- Intégration prévue dans le kernel commun : BM64 automatique à partir de
  4 096 lignes routées, BM32 en dessous, surcharge manuelle préservée (0=auto).
  Hoisting segmenté distinct, désactivable pour A/B et garde de capacité.
  Aucun chemin decode modifié. Petits prompts et suites numériques à vérifier
  avant de qualifier le code local ; ni app reconstruite ni publication.
- Code local intégré pour vérification. Le runner utilise désormais les vraies
  options de production (baseline BM32/hoist off, candidat auto/hoist on),
  sans réécriture des sources Metal. Les anciennes commandes utilisant
  `tensor_types_factory()` décrivent le prototype retiré du runner.
- Suite lancée : `.venv/bin/python -m pytest -q -x tests/test_segmented_m64.py
  tests/test_qmm_address_hoist.py tests/test_dense_prefill.py tests/test_audit_candidates.py`.
  Log `build/segmented-production-quality-20260911.log`. Matrice segmentée compare
  maintenant BM32 sans hoist à BM64 avec hoist. Ajout de contrôles durables
  capacité insuffisante et dispatch auto ; à exécuter après la suite en cours.
- **175 tests réussis** sans intercepteur Metal. Tests complémentaires ajoutés
  ensuite encore à exécuter ; compatibilité autre version macOS non mesurée.
- Confirmation finale du code réellement intégré : Qwen4k et LFM8 4k,
  `--segmented-m64 --segmented-hoist --warmup-pairs 1 --pairs 3 --max-tokens 96`,
  puis prompts courts `--repeats 16 --pairs 3 --max-tokens 32`. Journaux
  `build/segmented-final-20260911-*.jsonl`. Garder la comparaison à 96 tokens
  distincte de celles à 16 ; vérifier aussi decode et pic sans revendiquer un
  gain decode sur ce kernel de prefill. Aucun benchmark llama.cpp effectué.
- Holdout prévu après validation initiale : document technique non répétitif
  `docs/audit-runtime-2026-09-04.md`, option `--prompt-file`, warmup puis trois
  paires à 32 tokens sur LFM8 et Qwen. Log `build/segmented-heldout-20260911-*`.
  Ce contrôle évite de limiter la conclusion à une phrase répétée qui peut
  concentrer le routing sur quelques experts ; conserver les résultats séparés.
- Production Qwen4k/96 générés : gains prefill **+14,42 / +10,70 / +12,35 %**
  (médiane +12,35 %), TTFT **−12,57 / −9,69 / −11,02 %**. Decode
  −1,73 / +2,74 / −0,48 % : pas de gain établi, fluctuation de batterie/thermique.
  Pic identique 14,0523 GB ; empreinte OS candidat +~6–8 MB, pas un gain RAM.
  Caches et texte exacts. `build/segmented-final-20260911-qwen.jsonl`.
- Production LFM8 4k/96 générés : prefill **+34,86 / +36,84 / +30,85 %**
  (médiane +34,86 %), TTFT **−25,74 / −26,88 / −23,54 %**. Decode +1,08 / +1,55 /
  +2,41 %, sans changement decode direct ni revendication indépendante. Pic
  4,92316 GB identique ; texte/caches exacts. Empreinte OS ~4,70 GB, sans baisse.
  `build/segmented-final-20260911-lfm8.jsonl`. Mesures sur batterie, pas de
  garantie de ces pourcentages sur tous les prompts ou tous les Mac.
- Prompts courts 138 tokens / 32 générés : LFM8 prefill +16,14 / +12,99 /
  +17,46 %, TTFT −12,94 / −10,96 / −15,13 % ; Qwen prefill +11,84 / +8,95 /
  +6,95 %, TTFT −10,58 / −7,98 / −6,35 %. Caches/textes exacts, pics inchangés.
  Decode court : LFM médiane −2,49 %, Qwen −1,31 % (31 tokens chronométrés,
  petites fluctuations opposées à certaines paires de 96 tokens) ; ne pas
  cacher cette limite ni présenter une accélération decode.
  `build/segmented-final-20260911-{lfm8,qwen}-short.jsonl`.
- Holdout LFM8, document varié 2 746 tokens / 32 générés : prefill **+24,77 /
  +25,82 / +25,59 %** ; TTFT **−19,84 / −20,45 / −20,40 %**. Texte/caches exacts,
  pic ~4,72155 GB stable. Decode −3,79 / −3,89 / +0,32 % : petite régression
  dans deux des trois courts échantillons, à conserver dans le bilan.
  `build/segmented-heldout-20260911-lfm8.jsonl`.
- Holdout Qwen, 2 842 tokens / 32 générés : prefill +6,63 / +14,12 / +9,37 %,
  TTFT −6,32 / −12,18 / −8,63 %, decode médiane −0,62 %. Caches et texte exacts,
  pic ~13,91–13,92 GB sans gain établi. `build/segmented-heldout-20260911-qwen.jsonl`.
- **177 tests réussis** après ajout des contrôles capacité/auto-tile, sans
  intercepteur. `build/segmented-production-quality-final-20260911.log`.
  Dernier smoke prévu du chemin non-TensorOps :
  `MLXL3_TENSOR_SEGMENTED_QMM=0 .venv/bin/python -m pytest -q tests/test_audit_candidates.py -k device_side_route_buckets`,
  log `build/segmented-fallback-quality-20260911.log`. Pas un test sur un vrai M1.
- Rapport durable : [prefill-investigation-2026-09-11.md](docs/prefill-investigation-2026-09-11.md).
  Mesures finales/caches/manifestes copiés par patch dans
  [prefill-segmented-20260911.json](benchmarks/results/prefill-segmented-20260911.json),
  donc conservables dans Git même si les logs `build/` sont ensuite nettoyés.
- Intégration : `src/mlxl3/kernels/qmv.py` local, chemin partagé du moteur/CLI ;
  pas d'app reconstruite/installée, de push ou de publication. Aucun changement
  de langage/refonte. Aucun gain RAM/decode ni supériorité sur llama.cpp annoncé.
  Les candidats RAM/tuilage dense étendu non concluants restent hors production.
- Contrôle final : **8 tests non-TensorOps réussis**, 52 désélectionnés ;
  `git diff --check` et compilation Python des fichiers modifiés réussis.
  Le lanceur `~/.local/bin/mlxl3` utilise bien la `.venv` de ce dépôt ; import
  vérifié vers ce `qmv.py`, auto-tile=0 et hoisting actif. Aucun benchmark ne
  reste en cours. L'objectif >=10 % est atteint en prefill et TTFT sur les
  scénarios précisés ; les limites decode/RAM et de généralisation restent
  celles du rapport, sans prétendre avoir optimisé tout modèle/tout contexte.

### Publication demandée le 11 septembre 2026

- L'utilisateur demande de terminer puis pousser sur GitHub. Préparer le lot
  validé sur `main` : kernel, tests, runner, preuves, rapports et consignes du
  journal. Les modifications Ling/quantification restent locales hors du commit.
- Aucune release/DMG ni reconstruction de l'app n'est demandée par ce push.
  Le résultat du push sera consigné après vérification du commit distant.
- Push du code **réussi** : `f7825f49986b31c44a44b225faef3b9586eb0d0d`
  sur `https://github.com/0xZKnw/mlxl3`, branche `main`. Commit distant confirmé
  avec `git ls-remote origin refs/heads/main`. Tests/préuves/journal inclus ;
  changements Ling préservés hors commit. App installée et DMG inchangés.

### REL-2026-09-11 — Desktop v1.0.1 build 12 — en cours

- Demande utilisateur : intégrer le nouveau moteur dans la GUI et publier v1.0.1.
  Construire depuis un checkout propre du commit de release, sans les travaux
  Ling non publiés. Versions Python/Swift synchronisées ; signature ad-hoc
  maintenue, aucun certificat Developer ID disponible.
- Validation du correctif 10-10/11-04 : moteur gelé dans l'app, warmup réel activé
  (le smoke existant le désactivait), chargement Qwen, deux tours de chat et
  prefill long. Puis signature, DMG, empreinte SHA256 et métadonnées du bundle.
  Il s'agit de validation de distribution, pas d'un nouveau benchmark de gains.
- Publication prévue : tag GitHub v1.0.1, DMG build 12, manifestes/validation ;
  conserver une copie de l'app locale précédente avant son remplacement.
- Checkout propre `98da8c5` : 177 tests réussis. Runtime PyInstaller : warmup
  Qwen et deux tours avec rappel CEDAR-42 réussis ; prefill de 4k tokens puis
  génération réussis. Logs `build/release-v101-{clean-tests,qwen-smoke,qwen-long}*`.
- Build GUI interrompu : SDK27 fourni par CommandLineTools sans plugin
  `SwiftUIMacros.StateMacro`. Aucun changement UI nécessaire. Réessayer avec
  le SDK26.5 déjà installé, sélection explicite `MLXL3_MACOS_SDK` et SDK inscrit
  au manifeste. Ne pas publier l'artefact incomplet ; log du premier échec
  `build/release-v101-build.log`. L'option `--version` du CLI n'existe pas :
  version vérifiée via manifeste/Info.plist, sans ajouter une commande hors scope.
- Révision après demande explicite de SDK27 : le build SDK26.5 a terminé
  (exit 0, `build/release-v101-sdk265-build.log`), mais reste un artefact local
  non installé et non publié. La publication est suspendue pour résoudre la
  chaîne SwiftUI27. Le SDK27 contient bien SwiftUI/SwiftUICore et déclare
  `State()` via `SwiftUIMacros.StateMacro`, mais aucun plugin SwiftUIMacros
  n'a été trouvé dans `/Library/Developer` ou `/Applications` ; seul
  CommandLineTools est sélectionné, sans Xcode.app. Aucun SDK ni code SwiftUI
  contourné/modifié. Installation des outils Xcode27 complets à confirmer.
- Installation Xcode27 autorisée par l'utilisateur. Espace disponible vérifié
  (~554 GiB), aucune archive Xcode dans Downloads. Le site officiel Apple
  redirige les téléchargements Applications vers la connexion Apple Account :
  page ouverte et laissée à l'utilisateur pour authentification. Aucun
  téléchargement/installation Xcode ni publication v1.0.1 effectué à ce stade.
- L'utilisateur choisit finalement SDK26.5. Reprise du paquet propre `4e1fd76`,
  sans téléchargement Xcode27. Vérification finale prévue : signature et
  intégrité DMG, self-checks GUI et deux tours Qwen via le runtime de l'app
  finale (répétition du smoke précédent pour valider la copie embarquée).
- Contrôles du paquet SDK26.5 réussis : signature stricte, intégrité DMG,
  self-checks timeline/MCP/mémoire Metal et deux tours Qwen avec warmup réel
  (`build/release-v101-final-app-smoke.*`). Le manifeste contient toutefois
  un ancien doublon editable `mlxl3.egg-info` 1.0.0, malgré le code 1.0.1 :
  correction du générateur pour donner priorité à la version source embarquée.
  Reconstruction du paquet prévue, moteur/GUI inchangés ; revérifier
  manifeste, signature, DMG et identité du runtime avec celui testé.
- **Validé et publié** : paquet propre `14061bf320fc0d41f498c2ffcdfe522e3bf49717`,
  version 1.0.1/build 12, SDK26.5. Manifeste cohérent, signature stricte et
  self-checks GUI réussis dans l'app puis le DMG monté. Runtime SHA256
  `2213fd25219b8319ed469fdde15e4f3a17a6a8736d7f95bba8ed9af80f2124bb`
  identique au runtime testé ; deux tours Qwen montés réussis, warmup activé,
  CEDAR-42 rappelé, 537 tokens cachés/20 évalués au second tour, stderr vide
  (`build/release-v101-mounted-smoke.*`). Pas de mesure de gain supplémentaire.
- Installation locale effectuée dans `dist/MLXL3 Desktop.app`. Ancienne app
  conservée dans `build/app-backups/MLXL3 Desktop-v1.0.0-before-v1.0.1.app`.
  Aucun historique/modèle modifié ; l'app n'a pas été relancée automatiquement.
- `main` et tag `v1.0.1` poussés. Release publique confirmée via `/releases/latest` :
  https://github.com/0xZKnw/mlxl3/releases/tag/v1.0.1 ; DMG, SHA256, manifeste et
  `validation-v1.0.1.txt` présents. Empreinte DMG locale/distance identique :
  `2a98101aec9a4d718e76821f64b7d83885cb660ec9aca7a3ab858be6c08b2de4`.
  CI du tag réussie : https://github.com/0xZKnw/mlxl3/actions/runs/34637258944.
  Signature toujours ad-hoc/non notarisée. Volume de test éjecté, aucun moteur
  de test encore actif. Travaux Ling non publiés préservés hors des commits.

### OPT-2026-09-12-RUST-PERF-01 — Synchronisations Qwen Rust par couche — validé, code local

- Hypothèse : le port Rust synchronise actuellement `hidden` puis l'état dans
  chaque couche Qwen, soit des dizaines de barrières CPU/GPU par token, alors
  que le moteur Python validé synchronise le graphe complet au point de
  sampling. Regrouper ces évaluations à la frontière du token doit restaurer
  une part importante du decode sans changer aucun calcul ni ordre numérique.
- Antécédents consultés : `src/mlxl3/kernels/qmv.py`, `src/mlxl3/linear.py`,
  `src/mlxl3/moe.py`, `src/mlxl3/recurrent.py`, ainsi que
  `docs/decode-investigation-2026-09-10.md`,
  `docs/general-performance-2026-09-07.md` et
  `docs/prefill-investigation-2026-09-11.md`. Aucun essai historique ne mesure
  cette barrière propre au nouveau port Rust.
- Baseline/candidat : runtime Rust release, checkpoint local
  `models/Qwen3.6-35B-A3B-EXL3-2.49bpw`, température 0, MCP désactivé, prompt
  fixe, 32 tokens maximum ; un tour de chauffe puis au moins trois tours
  mesurés si la stabilité thermique le permet. Comparer texte/token IDs et
  logits imposés avant/après ; parité requise. Commande orchestrée via le
  protocole `mlxl3-rs bridge`, preuves sous `build/rust-perf-01-*.jsonl`.
- Conditions initiales : Apple M5 10 cœurs GPU, Metal 4, macOS local ; batterie
  100 %, débranchée. Température non mesurée. Le moteur GUI est fermé et aucun
  autre modèle n'est chargé. État d'intégration : analyse seulement, aucune
  mesure ni modification de kernel/runtime pour cet essai.
- Baseline mesurée avec 24 tokens de prompt et 32 générés : tour de compilation
  decode 8,72 tok/s, prefill 3,83 tok/s, TTFT 6,278 s ; trois tours chauds
  decode 8,39 / 8,10 / 8,80 tok/s (médiane **8,39**), prefill 6,82 / 5,96 /
  5,79 tok/s (médiane **5,96**), TTFT 3,519 / 4,029 / 4,149 s (médiane
  **4,029 s**). Les quatre sorties sont identiques. Pic reporté 13,064 GB,
  uniquement taille résidente estimée du checkpoint dans ce runtime, pas une
  mesure du processus. Preuve `build/rust-perf-01-baseline.jsonl`.
- Première vérification interrompue avant compilation : `cargo` n'est pas dans
  le `PATH` non interactif de cette session (`command not found`). Aucun test
  candidat ni résultat de performance ; localiser la toolchain déjà utilisée
  par le build de release puis relancer exactement les mêmes contrôles.
- Deuxième lancement encore interrompu avant compilation : le binaire Cargo
  absolu a été trouvé mais son `rustc` frère n'était toujours pas dans `PATH`.
  Relance suivante avec le dossier complet de la toolchain stable préfixé ;
  toujours aucune donnée candidat à ce stade.
- Troisième lancement a atteint le build script puis s'est arrêté avant les
  tests : `MLXL3_MLX_ROOT` absent. Aucun binaire candidat produit. Réutiliser
  exactement le chemin MLX 0.32.2 enregistré par le build d'app, sans installer
  ni changer de dépendance.
- Build candidat réussi avec la toolchain stable et MLX 0.32.2. Le filtre de
  tests `qwen` ne sélectionne actuellement aucun test Rust (0 exécuté) : il ne
  constitue pas une validation. Contrôle différentiel réel effectué avec
  l'ancien runtime `build/rust-runtime/mlxl3` et le candidat
  `target/release/mlxl3-rs`, tokens imposés `1,2` : sorties JSON/logits de
  2 978 940 octets strictement identiques, SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`.
  Le warning `rust-objcopy` sans `libLLVM.dylib` n'empêche ni le build ni
  l'exécution ; il concerne uniquement le strip de debug.
- Candidat, même protocole : tour de compilation decode 28,70 tok/s, prefill
  13,86 tok/s, TTFT 1,734 s ; trois tours chauds decode 27,38 / 29,50 / 28,93
  tok/s (médiane **28,93**, **+244,9 %** contre 8,39), prefill 29,41 / 29,69 /
  29,43 tok/s (médiane **29,43**, **+393,8 %** contre 5,96), TTFT 0,817 /
  0,809 / 0,816 s (médiane **0,816 s**, **−79,7 %** contre 4,029 s).
  Sorties identiques entre tous les tours et au runtime baseline ; preuve
  `build/rust-perf-01-candidate.jsonl`. Batterie toujours débranchée, charge
  descendante non enregistrée, température non mesurée ; l'amplitude dépasse
  largement le bruit possible mais les pourcentages restent ceux de ce prompt.
- Décision : conserver la synchronisation unique sur les logits à la frontière
  de chaque token et supprimer les synchronisations par couche. Cela ne porte
  encore ni le QMM TensorOps du Python ni le prefill par séquence ; aucun gain
  n'est revendiqué pour les autres architectures. App installée inchangée,
  aucune publication.

### OPT-2026-09-12-RUST-PERF-02 — Sampling greedy entièrement Metal — validé fonctionnel, code local

- Hypothèse : avec température 0/top-k 1 et pénalité neutre, le runtime Rust
  matérialise aujourd'hui tout le `log_softmax` vocabulaire en FP32 sur CPU puis
  y cherche le maximum. Le moteur Python calcule le même `log_softmax` et son
  argmax sur Metal, puis ne lit qu'un index scalaire. Ajouter l'opération MLX
  native déjà disponible doit réduire le temps decode sans approximation.
- Baseline : candidat validé de RUST-PERF-01, même Qwen3.6 35B A3B, prompt fixe
  24 tokens, 32 générés, trois tours chauds : decode médian 28,93 tok/s,
  prefill 29,43 tok/s, TTFT 0,816 s. Batterie débranchée ; température non
  mesurée. Protocole identique, preuve candidate prévue
  `build/rust-perf-02-candidate.jsonl`.
- Contrôle qualité prévu : argmax Rust unitaire sur GPU, puis sortie complète
  identique au candidat précédent. Le chemin CPU existant reste utilisé pour
  sampling non greedy ou pénalité de répétition non neutre. État : aucun code
  ni résultat pour cet essai.
- Premier contrôle unitaire : sampling greedy réussi, mais le smoke Array a
  échoué sur une attente de test incorrecte (`[3]`) : l'argmax est bien réalisé
  sur le dernier axe d'une matrice 2×2 et retourne donc `[1,1]`, comme MLX-LM.
  Le code d'opération n'a pas échoué. Corriger uniquement l'oracle du test puis
  relancer ; aucune mesure modèle candidate avant ce contrôle vert.
- Après correction de l'oracle, smoke Array GPU et test sampling réussis.
  Candidat : tour de compilation 29,32 tok/s ; trois tours chauds decode 28,90 /
  30,00 / 29,53 tok/s (médiane **29,53**, +2,09 % contre RUST-PERF-01),
  prefill médian 29,69 tok/s et TTFT médian 0,8086 s. Sorties complètes
  identiques. Preuve `build/rust-perf-02-candidate.jsonl`.
- La série n'est pas alternée et la machine se réchauffe : le petit écart decode
  reste **non concluant comme pourcentage**. Décision fonctionnelle validée :
  conserver le chemin Metal, qui supprime objectivement la copie CPU du
  vocabulaire entier ; fallback CPU inchangé pour sampling/pénalité non neutres.
  App installée inchangée, aucune publication.

### OPT-2026-09-12-RUST-PERF-03 — Référence du moteur Python sur le même Mac — validé

- Objectif : mesurer le moteur Python actuel qui contient les kernels validés,
  au lieu de prendre les anciens chiffres ~50 decode/~500 prefill comme une
  baseline interchangeable. Cela permettra de porter seulement les chemins
  manquants du Rust et de comparer sous les mêmes conditions.
- Protocole Python existant : `mlxl3 benchmark qwen3.6-35b-a3b
  --prompt-tokens 128 --max-tokens 32 --warmup-runs 1 --repeats 3`, température
  0, sans MCP/réseau, sortie `build/rust-perf-03-python-reference.json`.
  Apple M5 sur batterie, niveau/thermique à relever avec le rapport. Aucune
  modification de moteur dans cet essai ; résultats non mesurés.
- Résultat à 134 tokens de prompt / 32 générés, après un warmup : prefill
  322,62 / 323,69 / 321,13 tok/s (médiane **322,62**), decode 53,66 / 53,40 /
  53,00 tok/s (médiane **53,40**), TTFT médian **416,25 ms**, pic MLX
  **12,433 GB**. Batterie 95 %, débranchée ; température non mesurée. Preuve
  `build/rust-perf-03-python-reference.json`.
- Conclusion : la cible decode ~50 tok/s est confirmée sur le moteur Python,
  mais le ~500 tok/s prefill n'est pas la valeur comparable de ce prompt court.
  Le Rust après RUST-PERF-02 reste à ~55 % du decode Python et son prefill
  token-par-token n'est pas comparable au QMM séquentiel Python.

### OPT-2026-09-12-RUST-PERF-04 — Hadamard/scales EXL3 compilés comme en Python — rejeté

- Hypothèse : le Rust exécute actuellement cast, scale, reshape, Hadamard puis
  scale de sortie comme opérations MLX séparées autour de chaque QMV. Le chemin
  Python validé utilise deux fonctions `mx.compile` (`_reference_scaled_hadamard_*`)
  qui gardent exactement le même ordre/arrondis mais fusionnent ces graphes.
  Réutiliser ces deux graphes via l'API C++ MLX doit réduire les dispatchs de
  tous les linéaires EXL3, particulièrement en decode.
- Baseline : runtime RUST-PERF-02 sauvegardé sous
  `build/rust-perf-02-runtime` (SHA256
  `9b23d145c849e01238a555187ec8d35806239f574259debee804d74f282a9e38`),
  Qwen decode chaud médian 29,53 tok/s. Candidat : mêmes 24/32 tokens, un
  warmup + trois tours, batterie débranchée, température non mesurée.
- Qualité prévue : smoke des deux opérations, logits imposés `1,2` strictement
  comparés au runtime sauvegardé, puis sortie chat identique. Aucun kernel
  Metal nouveau ni mode rapide ; calculs FP16/Hadamard identiques au Python.
- Premier build/smoke GPU réussi. `cargo fmt --check` a seulement signalé deux
  lignes à reformater et le compilateur une variable devenue inutilisée après
  fusion ; corrections mécaniques appliquées avant le contrôle de logits. Pas
  encore de résultat modèle candidat.
- Premier contrôle modèle **rejeté avant benchmark** : le candidat échoue au
  chargement avec `Cannot reshape array of size 2048 into shape (1,96,128)`.
  Le smoke de largeur 128 avait tracé le graphe C++ avec `shapeless=true` ; les
  dimensions calculées dans la lambda ont donc été réutilisées à tort pour les
  largeurs suivantes. Baseline intacte et aucune mesure de performance issue
  de ce candidat. Relance prévue avec le mode par défaut sensible aux shapes,
  identique au décorateur Python `@mx.compile` utilisé comme référence.
- Relance sensible aux shapes : build réussi, mais le contrôle différentiel a
  été **interrompu par une erreur de protocole** avant chargement du candidat :
  `cargo build --release` sans `--features mlx,chat` a remplacé le binaire par
  la variante minimale qui n'expose pas `forward`. Le filtre de smoke utilisé
  n'a sélectionné aucun test (0 exécuté), donc aucun succès ne lui est attribué.
  Relancer build, smoke et `forward` avec les deux features explicites ; aucune
  donnée de performance candidate à ce stade.
- Candidat corrigé construit avec `--features mlx,chat`. Le listing confirme
  l'existence du smoke GPU (il n'était pas exécuté dans la commande précédente).
  Contrôle modèle Qwen tokens imposés `1,2` désormais **strictement identique**
  au runtime RUST-PERF-02 : 2 978 940 octets et SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`.
  Benchmark encore non mesuré ; exécuter explicitement le smoke puis la série.
- Smoke GPU explicitement exécuté avec `--ignored` : réussi. Candidat, un tour
  de compilation puis trois tours chauds : decode 28,21 / 28,36 / 27,53 tok/s
  (médiane **28,21**, **−4,49 %** contre 29,53), prefill médian **28,43** tok/s
  (−4,26 %) et TTFT médian **0,8445 s** (+4,44 %). Sorties identiques ; preuve
  `build/rust-perf-04-candidate.jsonl`.
- Décision : **rejet et retrait**. La compilation locale fidèle au Python
  ralentit le graphe Rust déjà différé entre tokens ; conserver la chaîne MLX
  primitive et tester ensuite un écart structurel plus haut niveau.

### OPT-2026-09-12-RUST-PERF-05 — Cache de la détection GPU comme le Python — validé, code local

- Hypothèse : le Python protège `_is_m5_gpu()` avec `@cache`, tandis que le
  Rust recrée un `metal::Device::system_default()` et lit son nom dans chaque
  `expert_mapped`. Qwen appelle ce chemin deux fois par couche MoE et par token.
  Mettre en cache ce booléen immuable avec `OnceLock` supprime donc des appels
  Objective-C/Metal répétés sans toucher au calcul ni aux kernels.
- Baseline : runtime RUST-PERF-02 sauvegardé, même Qwen 24 tokens prompt / 32
  générés, decode chaud médian 29,53 tok/s ; un warmup + trois tours candidat,
  batterie débranchée et température non mesurée. Preuve prévue
  `build/rust-perf-05-candidate.jsonl`.
- Qualité prévue : tests Rust, logits `1,2` strictement identiques au binaire
  sauvegardé, sortie complète identique. État : aucun résultat candidat.
- Tests Rust réussis et logits imposés `1,2` strictement identiques au runtime
  sauvegardé (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate : decode chaud 31,46 / 31,56 / 33,22 tok/s
  (médiane **31,56**, +6,87 %), prefill médian **32,41** tok/s (+9,18 %) et
  TTFT médian **0,7407 s** (−8,40 %). Preuve
  `build/rust-perf-05-candidate.jsonl`; sorties identiques.
- Le candidat a été exécuté après d'autres séries et la machine est sur batterie :
  ces pourcentages sont **préliminaires**. Répétition explicite prévue avec le
  binaire baseline sauvegardé immédiatement dans les mêmes conditions, afin de
  séparer le gain du cache du bruit/thermique avant décision.
- Validation alternée baseline puis candidat : baseline chaude decode 29,00 /
  28,22 / 28,59 tok/s (médiane **28,59**), prefill médian **28,75** tok/s,
  TTFT médian **0,8351 s** ; candidat chaud 32,52 / 32,50 / 32,85 tok/s
  (médiane **32,52**, **+13,74 %**), prefill médian **33,30** tok/s
  (+15,83 %) et TTFT médian **0,7211 s** (−13,65 %). Preuves
  `build/rust-perf-05-baseline-recheck.jsonl` et
  `build/rust-perf-05-candidate-recheck.jsonl`.
- Décision : conserver. C'est le même cache immuable que le Python et il retire
  des appels Objective-C/Metal du chemin chaud sans changer les sorties. App
  installée inchangée, aucune publication.

### OPT-2026-09-12-RUST-PERF-06 — Cache de kernels par spécialisation comme le Python — rejeté

- Hypothèse : les factories Python `@cache` retrouvent leur
  `CustomKernelFunction` par quelques entiers. Le bridge Rust/C++ recrée à
  chaque dispatch une clé ordonnée qui copie et compare nom, listes d'arguments,
  header et source Metal entiers. Utiliser le nom de spécialisation déjà unique
  comme clé évite ces copies dans le chemin chaud ; rendre aussi le nom du seul
  kernel générique `grouped` dépendant de sa shape garantit l'absence de collision.
- Baseline : runtime RUST-PERF-05 sauvegardé (SHA256
  `e673f285514b957a6e9f3b9ec7a5a7379c0b29e4914553c7a80cc859e54c4484`),
  série alternée précédente decode médian 32,52 tok/s, prefill 33,30 tok/s,
  TTFT 0,7211 s. Même protocole Qwen 24/32 ; batterie débranchée, thermique non
  mesuré. Preuve candidate prévue `build/rust-perf-06-candidate.jsonl`.
- Qualité prévue : smoke GPU, logits `1,2` identiques, sortie complète identique.
  Aucun shader ni paramètre numérique ne change. État : aucun résultat candidat.
- Logits Qwen `1,2` strictement identiques au baseline (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate chaude : decode 32,80 / 33,05 / 33,49 tok/s
  (médiane **33,05**), prefill médian **33,64** tok/s et TTFT médian
  **0,7137 s** ; preuve `build/rust-perf-06-candidate.jsonl`. L'écart contre la
  dernière série PERF-05 n'est qu'environ +1–2 %, donc encore **non concluant**.
  Répéter immédiatement le binaire PERF-05 sous le même état thermique avant
  de conserver ou retirer cette simplification.
- Contrôle alterné avec le binaire PERF-05 exécuté juste après : decode chaud
  33,97 / 33,82 / 33,95 tok/s (médiane **33,95**), prefill médian
  **34,28** tok/s et TTFT médian **0,7004 s** ; preuve
  `build/rust-perf-06-baseline-recheck.jsonl`. Le candidat à 33,05 tok/s est
  donc **−2,65 %** plus lent, malgré l'ordre thermique qui aurait dû l'avantager.
- Décision : **rejet et retrait**. La copie de clé n'est pas le bottleneck ;
  conserver la clé complète qui protège aussi les collisions de métadonnées.

### OPT-2026-09-12-RUST-PERF-07 — Un graphe decode Qwen jusqu'à l'argmax — rejeté

- Hypothèse : le Rust synchronise les logits Qwen puis lance `log_softmax` et
  `argmax` dans une seconde évaluation. Le générateur Python conserve au
  contraire le prochain token dans le graphe Metal jusqu'à la frontière de
  streaming. Exposer un `forward_lazy` uniquement au decode doit fusionner
  modèle + normalisation + argmax en une seule évaluation, sans modifier le
  prefill (toujours eager par token pour borner le graphe et la RAM).
- Baseline : binaire RUST-PERF-05 sauvegardé ; sa dernière série chaude donne
  decode médian 33,95 tok/s, prefill 34,28 tok/s et TTFT 0,7004 s. Même Qwen,
  prompt 24 tokens, génération 32, un warmup + trois tours, batterie
  débranchée, thermique non mesuré. Preuve prévue
  `build/rust-perf-07-candidate.jsonl`.
- Qualité prévue : forward eager/différentiel inchangé, test/sortie complète
  identique. Le chemin lazy n'est utilisé qu'après le premier token sélectionné.
  État : aucun résultat candidat.
- Test du sampler réussi et forward eager `1,2` strictement identique au
  baseline (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate chaude : decode 47,52 / 48,08 / 47,15 tok/s
  (médiane **47,52**), prefill médian **48,29** tok/s et TTFT médian
  **0,4973 s** ; sortie complète identique, preuve
  `build/rust-perf-07-candidate.jsonl`.
- Le saut decode est important mais la baseline immédiate a dérivé pendant les
  séries sur batterie. État encore **préliminaire** : relancer RUST-PERF-05 puis
  le candidat afin de quantifier le gain alterné avant validation.
- Première alternance a révélé une dérive majeure indépendante du patch : le
  binaire PERF-05 est lui aussi monté à **47,99 tok/s** médian, puis le candidat
  relancé juste après est retombé à **34,25 tok/s**. Batterie 83 %, débranchée ;
  `pmset -g therm` ne rapporte aucun warning, mais ces deux fenêtres ne sont pas
  comparables. Preuves `build/rust-perf-07-baseline-recheck.jsonl` et
  `build/rust-perf-07-candidate-recheck.jsonl`.
- État **non concluant** : effectuer une seconde mesure PERF-05 sous le régime
  ralenti actuel. Si elle rejoint ~34 tok/s, ne revendiquer aucun gain PERF-07 ;
  si elle reste ~48, retirer le lazy decode comme régression.
- Seconde mesure du binaire PERF-05 sous le même régime : decode chaud 47,23 /
  47,90 / 46,10 tok/s (médiane **47,23**), prefill médian **48,12** tok/s et
  TTFT médian **0,4990 s** ; preuve
  `build/rust-perf-07-baseline-recheck-2.jsonl`. Le baseline reste donc proche
  de 48 tok/s alors que le candidat relancé était à 34,25 tok/s.
- Décision : **rejet et retrait** du lazy decode. Construire un graphe modèle
  jusqu'à l'argmax est instable/coûteux après retrace ; le premier résultat à
  47,52 tok/s était une coïncidence de la dérive observée aussi sur le baseline.
  Le meilleur code validé reste RUST-PERF-05.

### OPT-2026-09-12-RUST-PERF-08 — Broadcast MoE sans copies comme le Python — validé, code local

- Hypothèse : le chemin Python forme les activations gate/up routées avec
  `broadcast_to(...).reshape(...)`, donc une vue sans copie. Le Rust concatène
  actuellement `2 × top_k` clones du token dans chaque couche MoE, créant une
  opération et un buffer inutiles avant chaque expert QMV. Porter l'opération
  MLX native `broadcast_to` doit réduire decode, TTFT et scratch sans changer
  un calcul numérique.
- Baseline : binaire RUST-PERF-05 sauvegardé ; dernière série stable decode
  médian 47,23 tok/s, prefill 48,12 tok/s, TTFT 0,4990 s. Même Qwen 24/32,
  un warmup + trois tours, batterie 83 % ou moins et débranchée, thermique non
  mesuré. Preuve candidate prévue `build/rust-perf-08-candidate.jsonl`.
- Qualité prévue : smoke `broadcast_to`, logits `1,2` strictement identiques et
  sortie complète identique. État : aucun résultat candidat.
- Smoke GPU réussi ; logits `1,2` et sortie complète strictement identiques au
  baseline (SHA256 logits commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate chaude : decode 47,86 / 45,06 / 47,81 tok/s
  (médiane **47,81**), prefill médian **46,05** tok/s, TTFT médian
  **0,5215 s** ; preuve `build/rust-perf-08-candidate.jsonl`.
- Face à la dernière baseline à 47,23 tok/s le decode ne gagne que 1,24 % et
  prefill/TTFT baissent d'environ 4 %, donc résultat **non concluant**. Relancer
  le binaire PERF-05 immédiatement avant décision ; aucun gain mémoire n'est
  revendiqué sans mesure de scratch MLX.
- Baseline immédiate suivante a de nouveau changé de régime : decode médian
  **34,55** tok/s, prefill **34,54** tok/s, TTFT **0,6950 s**, preuve
  `build/rust-perf-08-baseline-recheck.jsonl`. La machine alterne donc des
  plateaux ~34 et ~48 tok/s sans warning thermique, rendant la comparaison 32
  tokens invalide. Protocole complémentaire décidé avant exécution : pour
  baseline puis candidat, un warmup 64 tokens et une génération mesurée jusqu'à
  128 tokens, afin de comparer après montée en fréquence sur une fenêtre plus
  longue.
- Série soutenue baseline : warmup 64 tokens à 34,31 tok/s, puis 128 tokens à
  **34,05 tok/s** decode, **33,78 tok/s** prefill et **0,7116 s** TTFT. Série
  soutenue candidate : warmup 64 tokens à 34,69 tok/s, puis 128 tokens à
  **34,32 tok/s** decode (**+0,81 %**), **35,35 tok/s** prefill (**+4,64 %**)
  et **0,6793 s** TTFT (**−4,54 %**). Preuve brute commune
  `build/rust-perf-08-sustained.jsonl` ; batterie, modèle, prompt et options
  identiques, baseline immédiatement avant candidat.
- Décision : **validé, code local**. Conserver la vue MLX employée par le moteur
  Python : elle supprime une concaténation et ne régresse pas la charge longue.
  Le gain decode est modeste et aucun gain RAM n'est revendiqué, car la métrique
  disponible ne mesure pas séparément les buffers scratch. Logits et texte sont
  strictement identiques. App installée et publication inchangées.

### OPT-2026-09-12-RUST-PERF-09 — Sortie/réduction MoE fusionnée comme le Python — rejeté

- Hypothèse : le Python applique `@mx.compile` à l'ensemble Hadamard de sortie,
  échelle par expert, pondération de routage et réduction top-k. Le Rust expose
  ces étapes comme une chaîne d'opérations génériques distinctes dans
  `finish_and_reduce`. Porter exactement cette frontière de compilation doit
  réduire les dispatchs et buffers intermédiaires de chaque couche MoE, surtout
  au decode `M=1`, sans changer les kernels QMV ni le résultat numérique.
- Baseline : runtime RUST-PERF-08 sauvegardé (SHA256
  `b9ffd7dcb5f73e9f52db8335e4c3c55f04f2f3d97a2fbb29b3add23c6fc44f3b`),
  série soutenue 128 tokens à 34,32 tok/s decode, 35,35 tok/s prefill et
  0,6793 s TTFT. Même Qwen, prompt, options et batterie débranchée ; comparer
  baseline puis candidat sur 64 tokens de warmup et 128 tokens mesurés.
- Qualité prévue : tests Rust, logits `1,2` et texte strictement identiques au
  binaire sauvegardé. État : aucun résultat candidat.
- Build et smoke GPU réussis ; logits `1,2` strictement identiques au baseline
  (2 978 940 octets, SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Série soutenue baseline : warmup 64 à 34,73 tok/s, puis 128 tokens à
  **34,37 tok/s** decode, **34,97 tok/s** prefill et **0,6872 s** TTFT.
  Candidat : warmup 64 à 34,37 tok/s, puis 128 tokens à **34,15 tok/s** decode
  (**−0,65 %**), **34,75 tok/s** prefill (−0,63 %) et **0,6908 s** TTFT
  (+0,53 %). Texte strictement identique ; preuve
  `build/rust-perf-09-sustained.jsonl`.
- Décision : **rejet et retrait**. La compilation explicite de cette chaîne ne
  réduit pas le coût du graphe Rust déjà différé et ajoute une petite régression.

### OPT-2026-09-12-RUST-PERF-10 — Isoler le gain des blocs récurrents Python compilés — validé, diagnostic

- Hypothèse : les QMV/mapped-QMV, split-K, regroupements et transformations MoE
  Rust correspondent désormais au chemin Python. La différence structurante
  restante au decode Qwen est `compile_recurrent_layers`, qui compile chaque
  bloc Gated DeltaNet avec son état explicite. Mesurer le Python avec puis sans
  `MLXL3_COMPILED_RECURRENT_LAYERS` quantifie la part réellement récupérable
  avant tout port complexe.
- Baseline : référence Python RUST-PERF-03 à **53,40 tok/s** decode, 322,62 tok/s
  prefill et 416,25 ms TTFT, même Qwen et protocole 134/32. Répéter dans le même
  processus de benchmark avec compilation activée puis désactivée, batterie
  débranchée ; aucune modification de poids ni de sampling.
- Qualité prévue : texte greedy identique ; ce test est diagnostic et ne modifie
  ni moteur Rust, ni app. État : aucun résultat.
- Mesure dans les mêmes conditions : Python compilé, decode médian
  **48,37 tok/s**, prefill **324,70 tok/s**, TTFT **413,72 ms** ; Python sans
  compilation récurrente, decode **41,51 tok/s**, prefill **285,48 tok/s**,
  TTFT **470,48 ms**. Pics MLX identiques à 12,43 GB et sorties greedy
  identiques. Preuves `build/rust-perf-10-python-compiled.json` et
  `build/rust-perf-10-python-uncompiled.json`.
- Conclusion diagnostic : la compilation explique **+16,50 %** de decode,
  +13,74 % de prefill et −12,07 % de TTFT sur cette fenêtre. Le runtime Rust
  PERF-08 atteint déjà 47–48 tok/s sur son palier rapide, donc son QMV est au
  niveau du Python compilé actuel ; le goulet massif restant est son prefill
  token-par-token. État : **validé, diagnostic seulement**, aucun code intégré.

### OPT-2026-09-12-RUST-PERF-11 — Synchronisation Qwen par blocs au prefill — rejeté

- Hypothèse : le Rust synchronise les logits après chaque token de prompt alors
  que les états GDN/KV créent déjà les dépendances correctes dans le graphe MLX.
  Ne synchroniser que le dernier logits de petits blocs doit amortir les barrières
  CPU/GPU sans changer le calcul, avant le port beaucoup plus large du QMM multi-row.
- Matrice prévue : blocs de 2, 4, 8 puis 16 tokens, même Qwen et prompt de 134
  tokens, 32 tokens générés, un warmup et trois mesures ; arrêter/retirer une
  variante si elle régresse, change les logits ou augmente excessivement la RAM.
  Baseline Rust PERF-08 sauvegardée ; référence soutenue récente 34,37 tok/s
  decode et 34,97 tok/s prefill, mais le critère principal de cette série est le
  prefill alterné sous le même état machine.
- Qualité prévue : logits `1,2`, texte greedy et état final strictement identiques.
  Prototype piloté par `MLXL3_RUST_PREFILL_CHUNK`, à retirer ou figer après choix.
  État : aucun résultat candidat.
- Logits imposés `1,2` strictement identiques au baseline (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Sur prompt de 212 tokens, bloc 1 chaud : **47,37 tok/s** prefill,
  **47,13 tok/s** decode et **4,4757 s** TTFT. Bloc 2 : **42,83 tok/s**
  prefill (−9,58 %), **33,54 tok/s** decode (−28,83 %) et **4,9500 s** TTFT
  (+10,60 %), sortie greedy identique. Preuve
  `build/rust-perf-11-matrix.jsonl`.
- Décision : **rejet et retrait**. Le graphe récurrent inter-token plus grand
  provoque un retrace coûteux puis fait retomber le GPU sur le palier lent.
  Les variantes 4/8/16 ont été intentionnellement interrompues avant mesure,
  puisque leur hypothèse est strictement la même et leur graphe encore plus
  grand. Le prochain gain prefill exige un vrai chemin QMM multi-row, pas une
  accumulation de QMV token-par-token.
