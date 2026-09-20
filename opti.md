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

### OPT-2026-09-12-RUST-PERF-12 — QMM TensorOps EXL3 multi-token en Rust — validé

- Hypothèse : le prefill Rust reste limité à une suite de QMV `M=1` (~47 tok/s)
  alors que le Python sélectionne son QMM TensorOps M5 à partir de 24 lignes
  (~325 tok/s end-to-end). Porter d'abord le kernel QMM dense existant, sans
  réinventer son algorithme, doit fournir la primitive multi-row requise avant
  la généralisation des couches Qwen et du MoE segmenté.
- Première portée : `Exl3Linear` dense, lignes multiples, M5/macOS compatible
  TensorOps ; garder QMV inchangé pour `M=1`. Microbenchmark de matrices réelles
  Qwen sur M=32/64/128, puis comparaison numérique QMM contre une référence
  Python/EXL3 et contre les QMV ligne par ligne. Aucun gain end-to-end ne sera
  revendiqué avant intégration du modèle complet.
- Baseline : prefill Rust Qwen **47,37 tok/s** sur 212 tokens ; Python actuel
  **324,70 tok/s** sur 134 tokens. Batterie débranchée, M5, MLX 0.32.2.
  Contrôles : mêmes formes/dtypes, sorties finies, tolérances FP16 documentées,
  commandes et mesures brutes conservées. État : aucun code ni résultat.
- Premier build réussi, mais la validation a été **interrompue avant le kernel
  candidat** : le nouveau cas du script appelait par erreur l'oracle mono-ligne
  `qmv_exl3` avec une matrice M32, qui l'a correctement rejetée. Corriger
  l'import/appel vers `qmm_exl3` puis relancer ; aucune mesure ni conclusion de
  qualité issue de ce passage.
- Validation corrigée : `python native/check_parity.py --mlx` passe **178 cas**,
  dont les nouvelles matrices TensorOps M=32, K=1..8 et codebooks 0..2. Les
  sorties Rust et Python ont les mêmes motifs binaires FP16 (`uint16`) sur tous
  ces cas. Le kernel compile et s'exécute donc correctement sur le M5 ; état :
  **prototype local validé numériquement**, pas encore intégré au prefill Qwen
  end-to-end et aucun gain modèle revendiqué à ce stade.
- Intégré ensuite au modèle complet par PERF-13. État final : **validé et
  intégré localement** ; les gains end-to-end sont consignés ci-dessous.

### OPT-2026-09-12-RUST-PERF-13 — Prefill Qwen multi-token QMM — validé

- Hypothèse : le principal écart restant vient du prefill Rust qui exécute le
  modèle token par token et ne peut donc jamais sélectionner le QMM TensorOps.
  Faire traverser un bloc de 32 tokens dans les projections, l'attention, le
  Gated DeltaNet et le MoE doit supprimer cette sérialisation sans modifier le
  chemin decode M=1.
- Changement prévu : réutiliser le QMM TensorOps validé par PERF-12 pour les
  projections EXL3 groupées et séparées ; généraliser uniquement les formes
  temporelles déjà supportées par le kernel GDN et les kernels MoE mappés ;
  conserver le chemin mono-token actuel pour le decode.
- Baseline end-to-end : Qwen Rust **47,37 tok/s prefill**, **47,13 tok/s
  decode** sur 212 tokens ; Python actuel **324,70 tok/s prefill**, **48,37
  tok/s decode** sur son protocole 134 tokens. M5, MLX 0.32.2, batterie.
- Protocole : build/tests Rust, parité primitive, puis benchmark CLI avec le
  même modèle et le même prompt 212 tokens. Contrôles : sortie finie, cache
  causal et états GDN valides, decode non régressé ; mesures alternées si le
  plateau thermique change. État : **en cours**, aucun résultat.
- Première commande de contrôle **interrompue avant compilation** : elle visait
  à tort `native/Cargo.toml`, alors que le manifeste est à la racine. Aucun code
  ni kernel n'a été exécuté par ce passage ; relancer avec `Cargo.toml`.
- Le build corrigé passe, avec un avertissement de paramètre devenu inutile
  après factorisation (retiré aussitôt). La parité n'a pas démarré car la
  commande remplaçait `PATH` au lieu de le préfixer et ne trouvait plus
  `python`; aucun résultat kernel supplémentaire issu de ce passage.
- La vue QMM des poids groupés est maintenant contrôlée elle aussi : build
  release réussi puis `native/check_parity.py --mlx` passe **187 cas**, dont 9
  nouveaux cas groupés M=32 (K=2/3/4, trois codebooks), avec égalité bit-à-bit
  FP16 face aux projections Python contiguës. L'avertissement `rust-objcopy`
  reste limité au strip optionnel (`libLLVM.dylib` absent) ; le binaire produit
  et tous les contrôles s'exécutent. État : primitive dense/groupée validée,
  intégration modèle end-to-end encore en cours.
- Premier lancement end-to-end après intégration (prompt CLI court, 8 tokens de
  sortie) : **5,6 tok/s prefill**, **15,5 tok/s decode**, **4297 ms TTFT**. Ce
  passage inclut la compilation à froid de toutes les nouvelles variantes QMM
  par forme et n'est donc pas comparable à la baseline chaude ; résultat
  **non concluant**, à répéter à chaud puis sur le prompt 212 tokens. La sortie
  est finie et le modèle ne crashe pas.
- Mesure comparable dans un bridge résident, après warmup, trois répétitions du
  prompt exact de 212 tokens et 32 tokens greedy : médiane **154,09 tok/s
  prefill** contre 47,37 (**+225,29 %**), **1,3761 s TTFT** contre 4,4757 s
  (**−69,25 %**) et **49,17 tok/s decode** contre 47,13 (**+4,33 %**). Les trois
  répétitions donnent le même SHA256 de texte que la baseline PERF-11 :
  `5aed1d0102507d0399f27be1efab3b223301bde26a23adcf8ecd537ffe38434c`.
  Pic MLX inchangé à **13,0644 GB**. Commande :
  `python3 benchmarks/benchmark_bridge.py <Qwen> --native-binary
  target/release/mlxl3-rs --max-tokens 32 --repeats 3` avec le filler PERF-11
  répété 14 fois. M5, MLX 0.32.2, batterie, thermique non contrôlée.
- Décision : **validé, code local**. Le prefill Qwen utilise des blocs QMM de
  32 tokens ; les résidus <24 et le decode gardent le chemin QMV mono-token.
  L'attention causale, les états GDN et le MoE multi-token sont validés par le
  texte greedy strictement identique. Publication de ce lot encore à faire.
- Contrôle différentiel étendu après le benchmark : build release réussi et
  `native/check_parity.py --mlx` passe **189 cas** bit-à-bit, incluant désormais
  le Gated DeltaNet T=32 avec état initial non nul et le routeur MoE 32×256.
  L'avertissement de strip `rust-objcopy` reste non bloquant et inchangé.
- Suite finale locale : **18 tests unitaires passés** (5 GPU explicitement
  ignorés), test sampler passé, **14 contrats passés**, doc-tests passés et
  `git diff --check` propre. État : prêt à pousser sur
  `codex/rust-performance` ; aucune app installée ni release produite.

### OPT-2026-09-12-RUST-PERF-14 — Taille de bloc Qwen QMM — validé

- Hypothèse : le bloc conservateur de 32 tokens laisse du coût de lancement et
  de routage MoE non amorti. Des blocs de 64 puis 128 peuvent augmenter le
  prefill sans changer les kernels ni le decode ; arrêter dès régression.
- Baseline PERF-13, même bridge/prompt 212/32 et trois répétitions : **154,09
  tok/s prefill**, **49,17 tok/s decode**, **1,3761 s TTFT**, texte SHA256
  `5aed1d...934c`. Pic **13,0644 GB**.
- Protocole : changer uniquement la constante de chunk, build release, warmup
  puis trois répétitions ; contrôler le hash greedy, le pic et le decode. M5,
  MLX 0.32.2, batterie, thermique non contrôlée. Premier candidat : 64.
  État : **en cours**, aucun résultat candidat.
- Bloc 64, trois répétitions : médiane **112,99 tok/s prefill**, **32,81 tok/s
  decode**, **1,8771 s TTFT**, hash et pic inchangés. Le decode simultanément
  tombé de 49 à 33 tok/s montre un changement de palier machine ; comparaison
  brute **non concluante**. Revenir immédiatement à 32 et mesurer sous le même
  palier avant toute décision ; 128 n'est pas lancé à ce stade.
- Retour immédiat au bloc 32 sous le même palier bas : médiane **102,29 tok/s
  prefill**, **32,95 tok/s decode**, **2,0728 s TTFT**, hash/pic inchangés.
  Comparaison alternée valide donc le bloc 64 à **+10,46 % prefill** et
  **−9,44 % TTFT**, avec −0,43 % decode (bruit). Tester maintenant 128 sous le
  même protocole ; 64 est le meilleur candidat conservé jusque-là.
- Bloc 128 : les trois répétitions passent de **111,92 à 191,25 puis 228,76
  tok/s prefill**, pendant que le decode remonte de 31,60 à 45,20 tok/s. Hash
  et pic inchangés. La transition de palier en plein processus empêche une
  comparaison propre ; résultat provisoirement **non concluant**. Revenir à 64
  immédiatement pour obtenir une référence au palier remonté.
- Bloc 64 remesuré au palier haut : médiane **169,26 tok/s prefill**, **48,63
  tok/s decode**, **1,2528 s TTFT**, hash/pic inchangés. Face au bloc 32 au même
  palier (154,09/49,17/1,3761), cela confirme **+9,85 % prefill** et **−8,96 %
  TTFT**, avec −1,10 % decode compatible avec le bruit. Remesurer 128 maintenant
  que le palier est stable.
- Bloc 128 au palier haut stable : médiane **235,68 tok/s prefill**, **48,63
  tok/s decode**, **0,8998 s TTFT**, hash/pic inchangés. Face à 64 : **+39,24 %
  prefill**, **−28,18 % TTFT**, decode inchangé à <0,01 %. Une cause simple est
  aussi éliminée : 128 découpe 212 en 128+84, tous deux QMM, tandis que 64 laisse
  un résidu de 20 sur le chemin token-par-token. Tester 256 (un bloc de 212)
  avant de figer la valeur.
- Bloc 256 sur 212 tokens : médiane **242,23 tok/s prefill**, **48,39 tok/s
  decode**, **0,8755 s TTFT**, hash/pic inchangés, soit encore +2,78 % prefill
  et −2,70 % TTFT face à 128 (decode −0,49 %, bruit). Sur un second prompt de
  **1018 tokens**, bloc 256 : médiane **231,93 tok/s prefill**, TTFT 4,3896 s,
  hash stable et pic inchangé ; le decode court traverse encore un changement
  de palier et n'est pas utilisé. Tester 512 sur ces 1018 tokens.
- Bloc 512 sur 1018 tokens : médiane brute **221,93 tok/s prefill**, TTFT
  4,5875 s, hash/pic inchangés, mais le decode chute simultanément de 43,38 à
  **27,59 tok/s** (premier run 6,19), signal d'un nouveau changement de palier.
  Résultat **non concluant** ; revenir à 256 et comparer immédiatement. Le bloc
  512 n'est pas conservé sans A/B au même palier.
- Retour bloc 256 sur 1018 tokens : médiane **236,81 tok/s prefill**, **41,31
  tok/s decode**, **4,2990 s TTFT**, hash/pic inchangés. Malgré les variations
  de fréquence, 256 dépasse 512 de **+6,71 % prefill** et garde un decode bien
  supérieur sur cette alternance ; 512 est rejeté.
- Décision : **bloc 256 validé et intégré**. Sur le prompt court comparable il
  améliore PERF-13/32 de 154,09 à 242,23 tok/s (**+57,20 %**) et le TTFT de
  1,3761 à 0,8755 s (**−36,38 %**), sans changement de hash, de pic mémoire ni
  de chemin decode. Build release validé ; publication encore à faire.

### OPT-2026-09-12-RUST-PERF-15 — QMM MoE segmenté par expert — en cours

- Hypothèse : le prefill Rust trie déjà implicitement les mêmes routes mais
  exécute encore un QMV expert par slot. Le moteur Python trie les routes par
  expert et réutilise chaque tuile de poids décodée pour toutes les lignes du
  segment. Porter ce chemin existant doit fermer une partie de l'écart entre
  **242,23 tok/s Rust** et **324,70 tok/s Python**, sans modifier le decode M=1.
- Changement prévu : réutiliser le kernel TensorOps segmenté Python, construire
  sur GPU l'ordre, son inverse et la table de segments, puis appliquer gate/up
  et down aux routes triées. Aucun aller-retour CPU et aucun nouvel algorithme.
- Baseline : bloc 256, prompt exact de 212 tokens, 32 tokens greedy, trois
  répétitions chaudes : **242,23 tok/s prefill**, **48,39 tok/s decode**,
  **0,8755 s TTFT**, pic **13,0644 GB**, hash texte
  `5aed1d0102507d0399f27be1efab3b223301bde26a23adcf8ecd537ffe38434c`.
- Protocole : parité primitive bit-à-bit avec le chemin Python segmenté sur
  routes répétées, puis build/tests et benchmark résident identique. Garder le
  chemin mappé actuel pour moins de 24 lignes et pour tout M=1 ; retirer le
  candidat s'il change le texte, le pic ou régresse le prefill alterné.
- Environnement : Apple M5, MLX 0.32.2, batterie, thermique non contrôlée.
  État : **en cours**, aucun résultat candidat.
- Premier contrôle interrompu avant compilation : `cargo fmt --check` a détecté
  uniquement la mise en forme de la constante de bloc 256 déjà intégrée dans
  `main.rs`. Aucun kernel candidat ni benchmark n'a été exécuté. Appliquer le
  formateur officiel puis reprendre le même build ; protocole inchangé.
- Build release après formatage : réussi. L'avertissement `rust-objcopy` reste
  le strip optionnel déjà documenté (`libLLVM.dylib` absent) ; le binaire est
  produit. Aucun résultat numérique ni débit n'est encore attribué au kernel.
- Parité primitive réussie : `native/check_parity.py --mlx` passe **190 cas**,
  dont le nouveau SwitchGLU segmenté sur 64 tokens, quatre experts et top-2.
  La sortie complète est identique bit-à-bit FP16 au moteur Python segmenté ;
  les 189 cas antérieurs restent exacts. Passer au benchmark modèle résident.
- Premier benchmark modèle 212/32, trois répétitions : la première inclut la
  compilation des nouvelles variantes (**124,00 tok/s**), puis les deux tours
  chauds atteignent **450,02** et **449,59 tok/s prefill**. Médiane **449,59
  tok/s**, soit **+85,61 %** face au bloc 256 à 242,23 tok/s ; TTFT médian
  **0,4718 s** contre 0,8755 s (**−46,11 %**). Texte strictement identique,
  hash `5aed1d...934c`, et pic MLX inchangé à **13,0644 GB**. Decode médian
  **45,14 tok/s** ; le chemin M=1 n'appelle aucun nouveau code, et la baisse
  brute face à 48,39 suit les paliers machine déjà documentés, sans causalité
  attribuable. Preuve `build/rust-perf-15-candidate.json`.
- Contrôle suivant enregistré avant exécution : prompt de **1018 tokens**
  (filler PERF-14 répété 76 fois), mêmes 32 tokens greedy et trois répétitions
  résidentes. Exiger hash constant et comparer au bloc 256 à **236,81 tok/s** ;
  cela vérifie quatre blocs successifs et l'absence de gain limité au petit cas.
- Contexte 1018/32 : après le premier tour de compilation à 260,80 tok/s, les
  deux tours chauds atteignent **530,74** et **531,29 tok/s prefill** ; médiane
  **530,74 tok/s**, soit **+124,12 %** face au bloc 256 à 236,81. TTFT médian
  **1,9183 s** contre 4,2990 s (**−55,38 %**), decode médian 46,58 tok/s et pic
  inchangé. Les trois hashes 32 tokens sont identiques entre eux
  (`af6d4bf5...e8770fb`), mais l'ancien contrôle avait seulement 8 tokens :
  comparaison directe impossible. Preuve `build/rust-perf-15-long.json`.
- Contrôle qualité complémentaire enregistré : relancer exactement le même
  prompt avec 8 tokens ; le hash doit rester `8e004879...d19cf` avant de valider.
- Contrôle 1018/8 réussi : trois tours à **528,17 / 529,45 / 529,38 tok/s
  prefill**, hash exact historique `8e0048794f2135a30e5dd2736936612464d9eadba035d5748aa535ce4f4d19cf`
  à chaque fois, pic inchangé et decode médian **46,69 tok/s**. Preuve
  `build/rust-perf-15-long-quality.json`.
- Décision : **validé, code local**. Le prefill chaud atteint **449,59 tok/s**
  sur 212 tokens et **529,38 tok/s** sur 1018 tokens, contre respectivement
  242,23 et 236,81 avant segmentation. Le decode M=1 reste inchangé dans le
  code, la parité primitive et les hashes end-to-end passent, et la RAM poids
  rapportée ne bouge pas. Lancer la suite complète avant publication.
- Suite Rust réussie : **18 tests unitaires passés**, 5 GPU ignorés comme
  prévu, sampler passé, **14 contrats passés**, doc-tests et clippy strict
  réussis. La commande composée s'est ensuite arrêtée sur un ancien nom de
  script `native/check_sampler.py` qui n'existe pas ; aucun test réel n'a
  échoué et les contrôles sampler/contrats venaient déjà de Cargo. Reprendre
  seulement format/diff puis publier le lot validé.
- Contrôle final après garde de compatibilité (>256 experts conserve le chemin
  mappé) : build release réussi, parité **190/190** bit-à-bit et `git diff
  --check` propre. Le warning de strip optionnel reste inchangé. État final :
  **validé, prêt à publier** sur `codex/rust-performance` ; app installée et
  release inchangées.

### OPT-2026-09-12-RUST-LOAD-01 — Chargement et premier token natifs — en cours

- Hypothèse : le chargement Rust Qwen paie `read_exact_at → Vec → memcpy` pour
  chaque tenseur, puis des matérialisations/concaténations séparées pour les
  experts ; le premier prompt paie en plus la compilation des spécialisations
  Metal. Ces coûts doivent être mesurés séparément avant toute modification.
- Antécédents consultés : journal complet, `audit-runtime-2026-09-04.md`,
  `general-performance-2026-09-07.md` et `qwen38-decode-local.md`. Aucun essai
  précédent n'isole le temps de chargement du nouveau runtime Rust ; les TTFT
  chaudes PERF-13/15 excluent explicitement le premier tour de compilation.
- Baseline / candidat : commit `26bf6e2`, moteur Rust release avec MLX 0.32.2,
  checkpoint Qwen3.6-35B-A3B EXL3 2.49 bpw. Aucun changement de poids, kernel,
  sampling ou contexte. Le binaire installé est fermé pendant les mesures.
- Environnement : Apple M5, macOS local, secteur, batterie 100 %, aucun warning
  thermique signalé par `pmset`; cache fichiers macOS non contrôlé.
- Protocole : mesurer le délai processus→`ready`, le `load_seconds` moteur et
  le TTFT du premier prompt puis d'un prompt chaud dans le même bridge. Ajouter
  seulement ces champs au benchmark résident existant ; conserver stdout brut,
  texte/hash, prefill/decode et pic rapporté sous `build/rust-load-01-*.json`.
- Résultats baseline : `ready.load_seconds` **24,756 s**, délai processus→ready
  **40,981 s**, soit **16,225 s avant le chrono interne**. Le premier prompt
  de 15 tokens prend **1,335 s TTFT** à 11,28 tok/s prefill ; le suivant,
  après compilation, **0,683 s** à 39,55 tok/s. Temps processus total 43,54 s,
  CPU 10,09 s utilisateur + 25,65 s système. Hashes enregistrés dans
  `build/rust-load-01-baseline.json`, temps dans `*.time`.
- Isolation de l'inspection via un registre temporaire : **15,96 s** mur,
  4,40 s utilisateur + 11,47 s système, 348 MB RSS max
  (`rust-load-01-inspect.*`). Le bridge inspecte actuellement le checkpoint
  une fois avant son chrono puis le chargeur de modèle l'inspecte une seconde
  fois : le double scan explique la quasi-totalité des 40,98 s.
- Qualité : sorties greedy finies ; hash du prompt chaud historique exact
  `8e004879...d19cf`. Aucune modification du moteur dans cette mesure.
- Intégration : instrumentation locale du benchmark uniquement ; app et GitHub
  inchangés. Conclusion : **validé, diagnostic**.

### OPT-2026-09-12-RUST-LOAD-02 — Index checkpoint en table de hachage — en cours

- Hypothèse : les 124 579 tenseurs et les 31 243 entrées de stockage sont
  désérialisés/consultés dans des `BTreeMap`, alors que l'ordre n'est utilisé
  ni par le loader ni par les kernels. Une `HashMap` standard doit supprimer
  les insertions et recherches logarithmiques sans relâcher les validations.
- Baseline : inspection isolée **15,96 s** et bridge→ready **40,98 s** sur
  Qwen3.6 2.49 bpw, commit `26bf6e2`, mêmes conditions secteur/M5.
- Protocole : changer uniquement les tables du header et du checkpoint ; tests
  contrats, inspection isolée deux fois puis bridge froid identique. Garder le
  tri explicite des plages et des modules, ainsi que toutes les erreurs de
  doublon/index/forme. La sortie greedy et le hash doivent rester identiques.
- Résultats : contrats 14/14 et build release réussis, mais inspections isolées
  **16,59 s** puis **16,16 s**, contre 15,96 s baseline. CPU pratiquement
  identique (~4,5 s utilisateur, ~11,6 s système) et RSS plus haute
  (~396 MB contre 348 MB). Aucun gain ; sortie `register` identique.
- Conclusion : **rejeté et retiré**. Le coût n'est pas la structure d'index.
  Étape diagnostic enregistrée avant exécution : échantillonner le processus
  d'inspection et chronométrer parsing header, validation stockage et index,
  sans modifier le résultat ni désactiver un contrôle.
- Échantillonnage 5 s : 3 643/3 643 échantillons du thread principal sont dans
  `serde_json::from_reader(File)` lors de la lecture du gros JSON de
  quantification, majoritairement bloqués dans des appels `read`. Preuve
  `build/rust-load-02-inspect.sample.txt`. Le lecteur `File` non bufferisé,
  pas la validation EXL3, est donc le goulet établi.

### OPT-2026-09-12-RUST-LOAD-03 — JSON checkpoint bufferisé — en cours

- Hypothèse : entourer les trois lectures JSON du checkpoint d'un `BufReader`
  standard évite les appels système minuscules observés, sans changer le parseur,
  les structures ni une seule validation.
- Baseline : inspection **15,96 s** ; bridge processus→ready **40,98 s** dont
  `load_seconds` 24,76 s. Profil LOAD-02 : 100 % des échantillons dans la
  désérialisation `File` non bufferisée du manifeste de quantification.
- Protocole : modification standard-library limitée à `checkpoint::inspect`,
  contrats 14/14, build release, deux inspections isolées puis bridge complet.
  Comparer premier TTFT et hash greedy ; aucune modification kernels/poids.
- Résultats : contrats **14/14** et build release réussis. L'inspection isolée
  tombe à **0,66 s** puis **0,32 s**, contre **15,96 s** (−95,9 à −98,0 %).
  Le bridge complet atteint `ready` en **10,865 s** contre **40,981 s**
  (−73,5 %), avec `load_seconds` **10,533 s** contre 24,756 s. Le premier
  prompt après un build release froid révèle toutefois la compilation Metal :
  **7,728 s TTFT** à 1,94 tok/s, puis **0,226 s** et 119,79 tok/s au second
  prompt. Hash chaud exact `8e004879...d19cf`; premier hash
  `620a...2877`, inchangé. Preuves `build/rust-load-03-inspect-{1,2}.*` et
  `build/rust-load-03-bridge.{json,time}`.
- Décision : **validé, code local** pour le chargement. Le buffering standard
  supprime le goulet sans modifier les poids, kernels ou sorties. Le TTFT froid
  est maintenant le coût dominant et doit être traité séparément.

### OPT-2026-09-12-RUST-LOAD-04 — Warmup Metal avant `ready` — en cours

- Hypothèse : le runtime Rust ne précompile aucun graphe, donc le premier prompt
  utilisateur paie les spécialisations Metal. Un passage synthétique remis à
  zéro avant `ready` doit déplacer ce coût dans le chargement et réduire le TTFT
  sans conserver de KV ni changer le texte.
- Baseline après LOAD-03 : processus→ready **10,865 s**, premier TTFT froid
  **7,728 s**, deuxième TTFT **0,226 s**. Qwen utilise QMM à partir de 24
  tokens et le chemin MoE segmenté à partir de 64 ; les autres architectures
  n'ont pas de prefill batch natif dans ce dispatcher.
- Changement prévu : warmup Qwen de 64 tokens puis un token M=1 ; un seul token
  M=1 pour Gemma/LFM/Ling. Évaluer logits/état, puis appeler le reset existant,
  y compris après erreur. Aucun cache, kernel ou poids supplémentaire.
- Protocole : build/tests, bridge Qwen froid identique, comparer
  processus→ready + premier TTFT et le hash greedy. Le premier TTFT doit se
  rapprocher du tour chaud ; le coût total lancement→premier token ne doit pas
  régresser. Intégration : prototype local, résultats non mesurés.
- Premier contrôle interrompu avant exécution : `cargo` n'est pas présent dans
  le `PATH` de cette session (`command not found`). Aucun test ni benchmark n'a
  démarré ; reprendre avec le toolchain local explicite, protocole inchangé.
- Deuxième contrôle interrompu avant compilation : le toolchain explicite est
  disponible, mais le dépôt utilise le manifeste racine et non
  `native/Cargo.toml`. Le formatage officiel a été appliqué ; aucun test n'a
  démarré. Reprendre depuis `Cargo.toml` sans changer le candidat.
- Troisième contrôle interrompu par le build script avant compilation C++ :
  `MLXL3_MLX_ROOT` n'était pas défini dans ce worktree isolé. Aucun test n'a
  démarré. Reprendre avec l'installation MLX 0.32.2 locale explicite ; code et
  protocole inchangés.
- Première compilation réelle rejetée par Rust avant link : le type d'erreur
  de la closure de reset n'était pas inférable (`E0282/E0283`). Aucun binaire
  ni benchmark candidat. Ajouter l'annotation `Result<()>` demandée par le
  compilateur, sans changer le comportement prévu.
- Build et contrats **14/14** réussis. Candidat Qwen 64+1 : processus→ready
  **28,376 s** (`load_seconds` 27,781 s), premier TTFT **0,470 s** à
  32,01 tok/s, puis **0,200 s** à 136,26 tok/s. Le warmup supprime 7,26 s du
  premier TTFT mais ajoute 17,51 s au chargement ; lancement→premier token
  passe d'environ **18,59 s à 28,85 s**. Pic poids inchangé 13,0644 GB.
  Preuve `build/rust-load-04-bridge.{json,time}`.
- Décision : **rejeté**. Précompiler QMM et MoE segmenté pour 64 tokens au
  démarrage dégrade nettement le temps utilisateur total. Remplacer par un
  essai M=1 séparé, qui ne compile que le chemin decode/petit prefill.

### OPT-2026-09-12-RUST-LOAD-05 — Warmup Metal M=1 — en cours

- Hypothèse : un token synthétique compile les kernels communs au decode et au
  petit prefill, qui dominaient le premier prompt de 15 tokens, sans payer les
  spécialisations QMM/segmented du warmup LOAD-04.
- Baseline LOAD-03 : ready **10,865 s**, premier TTFT **7,728 s**, total
  **18,59 s**. Candidat LOAD-04 rejeté : ready 28,376 s, TTFT 0,470 s.
- Changement prévu : réutiliser exactement le helper et son reset, mais faire
  un seul `forward(0)` pour toutes les architectures. Build/14 contrats, bridge
  froid, hash/output et pic obligatoires. Valider seulement si le total
  lancement→premier token baisse et si le TTFT se rapproche du chaud.
- Résultats : build release et contrats **14/14** réussis. Processus→ready
  **19,423 s**, premier TTFT **0,907 s** à 16,57 tok/s, puis **0,255 s** à
  105,85 tok/s. Le warmup M=1 ajoute 8,56 s au ready et retire 6,82 s du
  premier TTFT ; lancement→premier token monte de **18,59 s à 20,33 s**.
  Pic poids inchangé 13,0644 GB. Preuve
  `build/rust-load-05-bridge.{json,time}`.
- Décision : **rejeté** selon le critère annoncé. Le TTFT affiché est meilleur,
  mais le délai réel utilisateur régresse. Retirer entièrement le helper et
  valider une baseline contemporaine LOAD-03 : cette répétition vérifie que la
  comparaison ne dépend pas d'un palier thermique/cache entre builds.
- Arrêt demandé avant la répétition de baseline : le helper et son appel ont
  été **entièrement retirés**. Contrats **14/14**, build release, `py_compile`,
  formatage et `git diff --check` réussis. Aucun warmup n'est intégré ni publié.
  État final : **rejeté et retiré**.

### État de publication du lot chargement — 2026-09-12

- LOAD-03 bufferisé est **validé et prêt à publier** : inspection 15,96 s →
  0,32–0,66 s, processus→ready 40,98 s → 10,87 s, hashes conservés.
- L'instrumentation benchmark processus→ready/première génération accompagne
  le changement. LOAD-04/05 restent documentés comme essais négatifs, sans
  code runtime résiduel. App installée et release inchangées à cet instant.

### OPT-2026-09-13-RUST-LOAD-06 — Chargement commun sans copie intermédiaire — en cours

- Demande : poursuivre les optimisations chargement/TTFT pour toutes les
  architectures, pas uniquement Qwen. Le point commun réel est
  `checkpoint_array`: chaque tenseur fait actuellement fichier → `Vec<u8>`
  Rust → allocation MLX → `memcpy`, pour Gemma 4, LFM2/LFM2-MoE, Ling 3 et
  Qwen3.5 dense/MoE.
- Antécédents consultés : journal complet jusqu'à LOAD-05, rapports chargement
  cités, diagnostic/production du skill inference-engineering. Les warmups
  globaux LOAD-04/05 sont rejetés car ils déplacent plus de temps qu'ils n'en
  retirent. Aucun essai historique trouvé pour une lecture directe dans le
  buffer MLX natif.
- Baseline/candidat : commit publié `f544027` (JSON déjà bufferisé). Modèles
  locaux disponibles pour mesures : LFM2.5 1.2B Thinking 4 bpw, LFM2.5 2.6B
  4 bpw et Qwen3.6 35B-A3B 2.49 bpw. Aucun checkpoint Gemma/Ling local : leur
  chemin partagé sera couvert par tests/compilation mais aucun pourcentage ne
  leur sera inventé.
- Protocole baseline : bridge release, processus→ready, `load_seconds`, premier
  TTFT puis tour chaud, un processus par modèle ; AC/thermique à relever.
  Candidat prévu seulement après ces mesures : `pread` robuste directement
  dans une allocation MLX avec la même validation de taille/inode/offset,
  sans mmap, lazy loading ni changement de dtype. Parité octets/checkpoints,
  14 contrats, build, puis mêmes trois bridges. Preuves sous
  `build/rust-load-06-*`.
- Baselines locales, batterie 85 % débranchée, aucun warning thermique :
  LFM1.2 ready **0,719 s**, `load_seconds` **0,399 s**, premier TTFT 0,179 s,
  chaud 0,254 s ; LFM2.6 ready **0,658 s**, `load_seconds` **0,644 s**,
  premier TTFT 0,465 s, chaud 0,537 s. Pics poids rapportés 0,793/1,764 GB.
  Qwen réutilise la mesure LOAD-03 au même code : ready 10,865 s,
  `load_seconds` 10,533 s. Preuves `build/rust-load-06-lfm{12,26}-baseline.*`.
- Lecture : les petits LFM sont déjà sous une seconde ; tout candidat commun
  doit donc éviter une régression mesurable chez eux. Le surcoût Qwen n'est pas
  uniquement proportionnel aux octets et inclut l'assemblage de ses experts.
- Premier contrôle candidat : formatage, contrats **14/14** et build release
  réussis. La commande du smoke GPU a sélectionné **0 test** car `--exact`
  recevait le nom court et non le chemin de module ; aucun succès matériel ne
  lui est attribué. Relancer seulement `array::tests::native_array_and_kernel_smoke`
  avec `--lib`, puis benchmarker si la lecture offset/longueur est exacte.
- Smoke GPU corrigé : **1/1 réussi**, y compris lecture de deux UInt32 à un
  offset non nul depuis un fichier réel. Candidat LFM1.2 : ready **0,719 →
  0,562 s** (−21,9 %), load **0,399 → 0,257 s** (−35,6 %). LFM2.6 : ready
  **0,658 → 0,518 s** (−21,3 %), load **0,644 → 0,509 s** (−21,0 %).
  Événements/textes des deux générations identiques. Les TTFT courtes fluctuent
  dans les deux sens et aucun gain d'inférence n'est attribué à cette lecture.
- Qwen face à la mesure de la veille : ready **10,865 → 11,166 s** et load
  **10,533 → 10,816 s** (+2,7 %), tandis que TTFT/decode changent fortement de
  palier sans changement de hash. Cette comparaison non alternée ne permet pas
  d'attribuer la petite régression au candidat. Construire le commit `f544027`
  dans un worktree/target séparé et mesurer baseline puis candidat immédiatement,
  avant décision. Preuves `build/rust-load-06-*-candidate.*`.
- A/B Qwen contemporain, baseline puis candidat : ready **9,145 → 7,625 s**
  (**−16,61 %**) et load **8,469 → 7,251 s** (**−14,38 %**). Temps processus
  total **13,34 → 9,70 s**, CPU système 3,13 → 2,66 s, RSS max hôte
  3,742 → 2,549 GB. Hashes premier/chaud exacts respectivement
  `620a...2877` et `8e004879...d19cf`; le TTFT change de palier et n'est pas
  attribué à la lecture. Preuves `rust-load-06-qwen-{baseline,candidate}-recheck.*`.
- Contrôle final enregistré avant décision : refaire le même A/B contemporain
  sur les deux LFM rapides, où 0,1 s peut être sensible au cache de fichiers.
  Conserver uniquement si ready/load restent non régressifs et hashes identiques.
- A/B LFM contemporain réussi : LFM1.2 ready **0,304 → 0,144 s** (−52,7 %),
  load **0,289 → 0,137 s** (−52,5 %) ; LFM2.6 ready **0,597 → 0,291 s**
  (−51,3 %), load **0,588 → 0,284 s** (−51,8 %). Hashes premier prompt et
  tour chaud strictement identiques pour les deux modèles. Preuves
  `rust-load-06-lfm{12,26}-{baseline,candidate}-recheck.json`.
- Décision : **validé, code local**. La lecture directe supprime une allocation
  hôte et une copie de chaque tenseur, réduit le chargement sur trois tailles et
  deux familles mesurées, et conserve le fallback validant les tenseurs BOOL.
  Gemma/Ling compilent le même `checkpoint_array`, mais restent **non mesurés**
  faute de checkpoint local. App installée et GitHub inchangés.

### OPT-2026-09-13-RUST-LOAD-07 — Inspection unique du checkpoint — en cours

- Hypothèse : le bridge inspecte le checkpoint pour ses métriques, puis chaque
  loader d'architecture répète la même inspection avant les poids. LOAD-03 a
  réduit ce scan à 0,32–0,66 s sur Qwen, mais il reste entièrement évitable et
  concerne Gemma, LFM, Ling et Qwen.
- Antécédents : LOAD-01 a établi le double appel ; LOAD-03 l'a bufferisé sans
  le supprimer. Pas de cache global/stale : passer explicitement le
  `Checkpoint` déjà validé aux loaders et garder leurs `load(path)` publics
  comme wrappers pour les autres commandes/tests.
- Baseline : code LOAD-06, A/B récents : Qwen ready 7,625 s/load 7,251 s ;
  LFM1.2 0,144/0,137 s ; LFM2.6 0,291/0,284 s. Candidat : aucune modification
  de poids, dtype, ordre de chargement ou kernels.
- Protocole : 14 contrats, build, parités existantes si le chargement passe,
  bridge des trois checkpoints, hashes exacts. Mesurer surtout Qwen ; sur les
  LFM sub-seconde, marquer le résultat bruité plutôt que revendiquer des ms.
- Build et contrats **14/14** réussis, trois checkpoints chargés et hashes
  exacts. Mesure non alternée contre LOAD-06 : LFM1.2 ready 0,144 → 0,455 s,
  LFM2.6 0,291 → 0,344 s, Qwen 7,625 → 8,422 s. Ces régressions contredisent
  le travail supprimé et suivent un changement global de palier/cache ; elles
  sont **non concluantes**.
- Diagnostic enregistré avant exécution : ajouter temporairement au même
  binaire un drapeau privé qui force l'ancienne seconde inspection, puis lancer
  ancien/nouveau dans la même fenêtre sur les trois modèles. Retirer le drapeau
  après mesure. Cela isole le seul changement sans reconstruire deux moteurs ni
  conserver une option de production.
- A/B dans le même binaire réussi : LFM1.2 double→simple inspection ready
  **0,556 → 0,144 s**, load **0,243 → 0,137 s** ; LFM2.6 **0,492 → 0,298 s**,
  load **0,484 → 0,290 s** ; Qwen **7,964 → 7,160 s**, load **7,640 →
  6,841 s**. Les hashes premier/chaud restent exacts pour les trois modèles.
  Preuves `build/rust-load-07-*-{repeat,single}-inspect.json`.
- Décision : **validé, code local**. Le drapeau diagnostic a été entièrement
  retiré ; le bridge passe désormais l'unique checkpoint aux quatre loaders.
  Leurs API `load(path)` conservent inspection+validation pour tous les autres
  appels. Gemma/Ling compilés mais non mesurés faute de poids locaux. App et
  GitHub inchangés.

### OPT-2026-09-13-RUST-LOAD-08 — Assemblage des poids MoE — diagnostic en cours

- Hypothèse : après lecture directe et inspection unique, Qwen charge encore
  en 6,84 s alors que les LFM denses sont sous 0,3 s. Le loader commun
  `Exl3SwitchGlu` crée une Array MLX par tenseur de chaque expert, puis plusieurs
  concaténations et matérialisations par couche ; Qwen répète cela pour 256
  experts. Le même loader est utilisé par les MoE Qwen, LFM, Gemma et Ling.
- Antécédents : LOAD-06 prouve que les copies fichier→Vec dominaient les modèles
  denses mais pas tout Qwen. Les essais historiques MoE concernent les kernels
  d'inférence, pas l'assemblage des poids au chargement.
- Protocole diagnostic : échantillonner 5 s du bridge Qwen pendant le chargement
  final LOAD-07, puis ne prototyper un pack direct qu'en présence de temps
  significatif dans `concatenate`/allocations/copies. Aucune modification des
  layouts ou du calcul avant cette preuve. Modèles Gemma/Ling non disponibles.
- État intermédiaire : une première tentative de rebuild du binaire de profil
  a été interrompue avant compilation, car `MLXL3_MLX_ROOT` n'était pas fourni.
  Aucune mesure n'a été produite ; relance prévue avec MLX 0.32.2 explicite.
- Résultat du diagnostic : bridge Qwen terminé correctement en **7,521 s**
  (`build/rust-load-08-profile.out`). L'échantillonnage 5 s place `pread` en
  tête avec **2 615** échantillons au sommet de pile ; les trois lectures des
  treillis experts dans `Exl3SwitchGlu::from_checkpoint_names` représentent à
  elles seules 607 + 594 + 576 échantillons visibles. Les concaténations et
  attentes GPU restent très minoritaires devant les lectures fichier
  (`build/rust-load-08-profile.sample.txt`).
- Décision : **rejeté** pour le prototype de pack/concat demandé par
  l'hypothèse initiale : il déplacerait les mêmes octets et ajouterait un
  buffer hôte. Prochaine piste à documenter séparément : supprimer la copie
  fichier→allocation MLX par mapping/chargement natif, si l'API MLX permet de
  conserver correctement la durée de vie du stockage.
- État d'intégration : diagnostic seulement ; aucun changement MoE appliqué.

### OPT-2026-09-13-RUST-LOAD-09 — Poids mappés sans copie — en cours

- Hypothèse : le chemin commun alloue une Array MLX puis copie chaque plage
  safetensors avec `pread`. Sur les MoE à milliers de tenseurs, ces copies
  dominent le chargement. Une vue MLX adossée à un mapping fichier pourrait
  supprimer la copie et rendre le coût initial proportionnel aux pages
  réellement touchées, sans changer les poids ni les kernels d'inférence.
- Périmètre : loader EXL3 natif partagé par Qwen, LFM, Gemma et Ling ; aucun
  chemin spécifique à une architecture.
- Baseline : LOAD-07/08, Qwen ready **7,16–7,52 s**, LFM dense **0,14–0,30 s**.
- Protocole : vérifier d'abord les constructeurs et garanties de durée de vie
  de MLX 0.32.2. Si une API publique sûre existe, prototyper derrière le même
  `checkpoint_array`, puis comparer A/B mêmes poids avec hashes de sortie,
  temps ready/load, RAM processus et tests de contrat. Sinon marquer bloqué,
  sans créer une abstraction propriétaire.
- Résultat : **bloqué/rejeté sans prototype**. MLX expose bien un constructeur
  de `array` sur pointeur utilisateur, mais son allocator Metal exige un buffer
  externe réutilisable et les offsets safetensors ne donnent pas directement
  un buffer page-aligné par tenseur. La discussion officielle MLX #615 décrit
  le même conflit mmap/offset Metal et le déplacement imprévisible du coût vers
  les page faults d'inférence. Notre chemin direct `pread` écrit déjà dans
  l'allocation unifiée finale ; le mapping ajouterait ici durée de vie, vues et
  risque de régression TTFT sans preuve d'un gain global.
- Preuves : headers MLX 0.32.2 locaux `mlx/array.h`, `mlx/allocator.h` et source
  allocator officielle `ml-explore/mlx`; discussion officielle
  https://github.com/ml-explore/mlx/discussions/615.
- État d'intégration : aucune modification mmap appliquée.

### OPT-2026-09-13-RUST-LOAD-10 — Lecture MoE concurrente — validé

- Hypothèse : LOAD-08 mesure un loader MoE essentiellement bloqué dans des
  `pread` indépendants, exécutés aujourd'hui strictement en série. Quelques
  workers stdlib peuvent maintenir plusieurs lectures SSD en vol et mieux
  alimenter la mémoire unifiée, sans modifier les layouts ni les valeurs.
- Périmètre : `Exl3SwitchGlu`, donc MoE Qwen, LFM, Gemma et Ling. Les modèles
  denses conservent le loader direct LOAD-06.
- Baseline : Qwen ready **7,16–7,52 s**, load **6,84–7,52 s** ; hashes de sortie
  LOAD-07 exacts. Conditions thermiques non garanties, donc variantes alternées.
- Protocole : helper local utilisant `std::thread::scope`, ordre de sortie
  déterministe, calibration 1/2/4/8 workers via `MLXL3_LOAD_THREADS`. Comparer
  au moins deux répétitions Qwen par variante, vérifier hashes first/warm,
  contrats, smoke GPU et RAM. Garder uniquement un gain robuste ; 1 worker doit
  rester un oracle fonctionnel.
- Résultat matrice alternée (médiane de 2 lancements par variante) :
  - 1 worker : ready **7,954 s**, load **7,446 s** ;
  - 2 workers : ready **5,130 s**, load **4,738 s** ;
  - 4 workers : ready **4,100 s**, load **3,716 s** ;
  - 8 workers : ready **3,970 s**, load **3,568 s**.
  Les 8 workers donnent **−50,09 % ready** et **−52,09 % load** face à
  l'oracle 1 worker ; même 4→8 reste favorable (−3,16 % ready). Les hashes
  first et warm sont identiques sur les 8 exécutions et la RAM moteur annoncée
  reste identique à **13,064 GB**.
- Preuves : `build/rust-load-10-qwen-{1,2,4,8}-{a,b}.json`. Les 14 contrats et
  le smoke GPU batched/direct passent avant la matrice.
- État intermédiaire : gain Qwen validé ; contrôle RSS hôte 1/8 workers et
  non-régression des modèles locaux LFM encore à effectuer avant décision.
- Contrôle `/usr/bin/time -l` : 1 worker **7,65 s**, **2 407 940 096 B** RSS
  max ; 8 workers **3,90 s**, **2 439 561 216 B** RSS max. Le parallélisme
  ajoute **31,6 MB / 1,31 %** de RSS hôte mesuré, sans modifier la RAM moteur,
  pour −49,0 % de temps mur sur ce contrôle
  (`build/rust-load-10-qwen-time-{1,8}.{out,txt}`).
- Limite locale : les deux LFM disponibles sont denses ; ils vérifient le
  chemin commun LOAD-06/07 mais n'exercent pas `Exl3SwitchGlu`. Gemma/Ling MoE
  ne sont pas présents localement, donc leur bénéfice reste **non mesuré** et
  ne doit pas être chiffré malgré le partage exact du loader.
- Validation intermédiaire : suite Rust complète **33 tests passés**, 5 GPU/
  tokenizer explicitement ignorés. Le premier `clippy -D warnings` a échoué
  uniquement sur la préférence mécanique `chunks_exact(2)` →
  `as_chunks::<2>()`; aucune exécution ou mesure affectée, correction prévue
  avant relance.
- Validation finale : défaut fixé à `min(cœurs disponibles, 8)`, surcharge
  possible avec `MLXL3_LOAD_THREADS=1..32`. Qwen par défaut : ready **4,288 s**,
  load **3,611 s**, hashes first/warm identiques à la matrice. LFM 1.2B et 2.6B
  denses chargent et génèrent avec le même hash first que LOAD-07
  (`build/rust-load-10-{qwen,lfm12,lfm26}-default.json`).
- Contrôles finaux : `cargo fmt --check`, `clippy --all-targets -D warnings`,
  build release, suite Rust **33 passés / 5 ignorés / 0 échec**, smoke GPU
  direct+batché **1/1**, et parité MLX **190/190** bit-à-bit.
- Décision : **validé**. Le gain mesuré porte sur Qwen MoE ; la même primitive
  est intégrée aux loaders MoE LFM/Gemma/Ling mais reste non mesurée faute de
  checkpoints locaux. Aucun gain n'est revendiqué pour leurs variantes denses.
- État d'intégration : **code local uniquement** sur `codex/rust-performance` ;
  app installée et GitHub inchangés, aucun push demandé à ce stade.

### OPT-2026-09-13-RUST-PERF-16 — Prefill LFM2 multi-token — validé

- Hypothèse : les LFM2 dense et MoE passent encore chaque token du prompt dans
  le modèle séparément, alors que les projections QMM, l'attention causale,
  ShortConv et le SwitchGLU segmenté acceptent déjà plusieurs lignes. Faire
  traverser un bloc complet doit amortir les poids et les dispatchs sans toucher
  au decode `M=1`, aux poids, au sampling ni à la précision.
- Antécédents consultés : journal complet jusqu'à LOAD-10, rapports decode,
  runtime et prefill cités à sa racine, ainsi que l'implémentation LFM2/LMF2-MoE
  de MLX-LM 0.32.0. PERF-11 a rejeté l'accumulation paresseuse de QMV ; le présent
  essai utilise le QMM multi-row validé par PERF-12/13 et ne le répète pas.
- Baseline/candidat : branche locale `codex/rust-performance`, HEAD `f544027`
  plus LOAD-06/07/10 validés non commités. Modèles locaux LFM2.5 1.2B Thinking
  4 bpw, LFM2.5 2.6B 4 bpw et LFM2.5 8B-A1B MoE 3.10 bpw. Baseline = chunks de
  1 ; candidat = même chemin par blocs, avec QMV conservé pour le decode.
- Environnement : Apple M5 24 Gio, macOS local, MLX 0.32.2, batterie 96 %
  débranchée, aucun avertissement thermique/performance ; fréquences non
  instrumentées. Aucun autre moteur MLXL3 actif au départ.
- Protocole : bridge résident, prompt français fixe répété, 64 tokens greedy,
  un warmup exclu puis trois répétitions. Relever tokens réels, prefill, TTFT,
  decode, hash et pic pour les trois modèles. Avant le benchmark candidat,
  comparer une continuation forcée sérielle/batchée, logits et tous caches
  bit-à-bit ; tester notamment le décalage causal attention et la fenêtre
  ShortConv. Preuves sous `build/rust-perf-16-*`.
- Baselines, trois tours chauds : LFM1.2, 217 tokens de prompt, **91,87 tok/s
  prefill**, **2,3622 s TTFT**, **107,40 tok/s decode**, pic 0,793 GB ; LFM2.6,
  202 tokens, **46,62 tok/s**, **4,3336 s**, **54,89 tok/s**, pic 1,764 GB ;
  LFM8 MoE, 201 tokens, **73,29 tok/s**, **2,7427 s**, **102,96 tok/s**, pic
  3,942 GB. Les trois hashes sont stables entre répétitions pour chaque modèle.
  Preuves `build/rust-perf-16-lfm{12,26,8}-baseline.json`.
- État : **en cours**. Baseline validée ; aucun candidat encore compilé.
  Intégration : essai local ; app et GitHub inchangés.
- Premier contrôle interrompu avant compilation : `cargo fmt --check` demande
  uniquement la mise en forme standard de deux chaînes d'appels LFM. Aucun
  test, modèle ou benchmark candidat n'a été exécuté. Appliquer le formateur
  officiel puis reprendre le même build, sans changement du protocole.
- Build release candidat réussi. Premier contrôle batch LFM1.2 face au modèle
  Python batché : tous les caches exportés passent bit-à-bit, mais 118/65 536
  logits finaux diffèrent d'un bit FP16. Le Python projette les 32 positions du
  head puis sélectionne la dernière ; le Rust sélectionne d'abord la dernière
  activation et garde le head QMV, ce qui change la partition QMM sans changer
  le decode. Résultat **non concluant**, aucun benchmark candidat : comparer
  maintenant batch Rust et référence Rust sérielle, logits et caches complets.
- En projetant le head sur le bloc comme la production, LFM1.2 puis LFM2.6
  passent chacun logits et tous caches bit-à-bit face au Python batché. LFM8
  s'arrête avant calcul expert : le routeur biaisé Metal est artificiellement
  limité à une ligne, bien que chaque SIMD group soit indépendant. Aucun
  benchmark candidat. Étendre ce même kernel à une grille de lignes, puis
  ajouter un cas multi-row à la matrice de parité avant de relancer LFM8.
- Routeur biaisé multi-row validé bit-à-bit dans la matrice MLX, qui passe
  désormais **191/191** cas. LFM8 batch 32 franchit le routeur mais diverge au
  premier cache observé de la couche attention 10 (296/32 768 octets), après
  plusieurs couches MoE ; aucun benchmark lancé. Tester 64 tokens, seuil exact
  du SwitchGLU segmenté déjà validé, afin de distinguer le fallback mappé
  multi-row du chemin QMM segmenté avant toute modification supplémentaire.
- Première commande 64 tokens interrompue avant chargement : `seq -s,` de BSD
  a produit une liste avec séparateur final, refusée par le parseur du checker.
  Aucun modèle/GPU exécuté et aucun résultat ; reconstruire la liste sans virgule
  terminale puis reprendre strictement le même contrôle.
- LFM8 à 64 tokens franchit le chemin segmenté mais diverge plus tard, au
  cache attention de la couche 14 (9 309/65 536 octets). Le routage multi-row
  est exact isolément ; la chaîne experte complète ne satisfait donc pas le
  contrat batch strict de ce checkpoint. Décision : ne pas activer le batch
  sur `lfm2_moe`, retirer l'extension de routeur devenue sans consommateur, et
  mesurer seulement les LFM denses 1.2B/2.6B dont la parité complète passe.
- Nettoyage appliqué : extension multi-row du routeur biaisé et son cas de test
  retirés ; le batch est maintenant exposé uniquement si toutes les couches
  feed-forward du LFM sont denses. Le LFM8 MoE reste sur les chunks de 1.
- Contrôles après garde dense : LFM1.2 et LFM2.6, bloc de 32 tokens, logits et
  tous caches **bit-à-bit** face à la production Python batchée ; LFM8 MoE,
  huit étapes sérielles, logits et tous caches **bit-à-bit**. Build release
  réussi ; seul l'avertissement `rust-objcopy`/`libLLVM.dylib` déjà connu reste.
  Prochaine étape : benchmark candidat trois tours, puis contrôle long >256.
- Benchmark candidat, même prompt/tokens/hashes et trois tours : LFM1.2
  **91,87 → 2 059,59 tok/s prefill (+2 141,8 %)**, TTFT médian **2,3622 →
  0,1055 s (-95,5 %)** ; LFM2.6 **46,62 → 836,59 tok/s (+1 694,6 %)**,
  TTFT **4,3336 → 0,2417 s (-94,4 %)**. Les hashes des trois sorties de chaque
  modèle sont identiques à la baseline. Decode observé +15,1 %/+7,4 %, mais
  non revendiqué car le chemin decode n'a pas changé et les tours sont courts.
- Contrôle LFM8 MoE non batché : même hash sur les trois tours ; prefill
  **73,29 → 75,78 tok/s (+3,4 %)**, TTFT **2,7427 → 2,6526 s (-3,3 %)** et
  decode +2,8 %, tous traités comme bruit/conditions. Pic inchangé pour les
  trois modèles (0,793 / 1,764 / 3,942 GB). Preuves
  `build/rust-perf-16-lfm{12,26,8}-{baseline,candidate}.json`.
- Premier contrôle long interrompu avant tout chargement : la variable locale
  `path` a écrasé le tableau spécial `$path` de zsh, rendant `python3`
  introuvable. Aucun modèle, GPU ou résultat exécuté. Renommer cette variable
  en `model_dir`, conserver strictement le prompt et relancer.
- Contrôle long validé face au binaire baseline sauvegardé avant PERF-16 :
  LFM1.2, 633 tokens donc trois chunks (256/256/121), **103,37 → 2 246,77
  tok/s**, TTFT **6,1241 → 0,2820 s**, hash greedy identique ; LFM2.6, 586
  tokens (256/256/74), **50,36 → 974,53 tok/s**, TTFT **11,6366 →
  0,6016 s**, hash identique. Preuves
  `build/rust-perf-16-lfm{12,26}-long-{baseline,candidate}.json`.
- Contrôles finaux : `cargo fmt --check`, `clippy --all-targets -D warnings`,
  suite Rust **33 passés / 5 ignorés / 0 échec**, matrice MLX **190/190**
  bit-à-bit, `git diff --check`. Décision : **validé** pour LFM2 dense ; rejeté
  explicitement pour LFM2-MoE faute de parité batch stricte. Le prefill/TTFT
  dense bénéficie du QMM multi-token ; decode, sampling et poids sont inchangés.
- État d'intégration : **code local uniquement** sur `codex/rust-performance` ;
  CLI/app installée et GitHub inchangés, aucun push ni rebuild GUI demandé.

### OPT-2026-09-13-RUST-PERF-17 — Prefill multi-token des architectures encore sérielles — validé Gemma

- Hypothèse : Gemma4 et Ling passent encore les prompts token par token dans le
  bridge. Au moins un de leurs chemins peut probablement réutiliser le QMM
  multi-row, l'attention causale et les primitives récurrentes déjà présentes,
  comme Qwen et LFM dense, sans nouveau format ni perte numérique.
- Antécédents : journal complet relu ; PERF-12/13 ont validé QMM/Qwen et PERF-16
  LFM dense, tandis que le batch LFM-MoE a été rejeté faute de parité de chaîne.
  Aucun essai Gemma/Ling multi-token n'est documenté. Chercher d'abord les
  contrats shape/state dans les deux runtimes et leur référence Python.
- Baseline/candidat prévu : binaire release sauvegardé avant PERF-16 contre
  branche locale courante ; premier checkpoint local compatible trouvé, prompt
  fixe >256 tokens, greedy, un warmup et trois répétitions si le coût le permet.
  Contrôle obligatoire avant mesure : logits et tous états bit-à-bit face à la
  production Python ; abandon immédiat de toute architecture non exacte.
- Environnement : Apple M5 24 Gio, MLX 0.32.2, batterie débranchée ; température
  et fréquences non instrumentées. État : **en cours**, recherche de chemin
  seulement ; aucune modification PERF-17 ni mesure à ce stade.
- Checkpoints : Ling EXL3 complet absent (sources/plans seulement) ; Gemma4
  26B-A4B EXL3 3,54 bpw disponible dans le cache de validation, 15,1 GB
  résidents. Baseline Gemma, prompt fixe 82 tokens, 16 tokens greedy, un warmup
  exclu puis trois tours : **39,84 tok/s prefill**, **2,0587 s TTFT**,
  **32,85 tok/s decode**, hashes stables, chargement 4,748 s. Preuve
  `build/rust-perf-17-gemma-baseline.json`.
- Lecture du chemin : les projections, le routeur standard et le SwitchGLU
  acceptent déjà plusieurs lignes ; les seules limites artificielles sont les
  reshapes `time=1`, le masque SDPA, et le head final. Ling exige en revanche
  un scan KDA récurrent multi-token absent : ne pas le refactorer dans cet essai.
- Prototype Gemma limité à la fenêtre glissante (batch seulement tant que la
  fin du bloc reste ≤1 024) : bloc de 32 tokens accepté par la référence de
  production, même top-1, erreur absolue max ≤0,5 et KL ≤0,005 selon le contrat
  Gemma existant. Le modèle a déjà une tolérance non bit-à-bit à cause du
  groupement QKV ; aucun seuil n'a été relâché.
- Benchmark candidat Gemma, mêmes 82 tokens, trois tours : **39,84 → 142,36
  tok/s prefill (+257,4 %)**, TTFT **2,0587 → 0,5763 s (-72,0 %)**, decode
  **32,85 → 32,66 tok/s (-0,6 %, bruit)**, pic inchangé 15,108 GB et les trois
  hashes greedy sont identiques à la baseline. Preuve
  `build/rust-perf-17-gemma-{baseline,candidate}.json`.
- Contrôle long Gemma, 565 tokens (256/256/53), un tour : **35,59 → 126,51
  tok/s prefill (+255,5 %)**, TTFT **15,8748 → 4,4663 s (-71,9 %)**, hash
  greedy identique et pic inchangé. Preuves
  `build/rust-perf-17-gemma-long-{baseline,candidate}.json`. Le batch est
  volontairement plafonné à 1 024 tokens ; au-delà, le fallback sériel garde
  la sémantique exacte de sliding attention sans ajouter un masque dédié.
- Contrôle de frontière sliding, 1 117 tokens : les quatre premiers blocs sont
  batchés puis les 93 derniers tokens repassent en sériel ; **32,39 → 68,02
  tok/s prefill (+110,0 %)**, TTFT **34,4830 → 16,4225 s (-52,4 %)**, hash
  greedy identique, pic inchangé. Preuves
  `build/rust-perf-17-gemma-window-{baseline,candidate}.json`.
- Contrôles finaux : parité Gemma batch 32 puis decode sériel 8 étapes contre
  production (même top-1, max abs ≤0,5, KL ≤0,005), `cargo fmt --check`, clippy
  strict, suite Rust **33 passés / 5 ignorés / 0 échec**, matrice MLX **190/190**
  bit-à-bit et `git diff --check`. Décision : **validé pour Gemma4**. Ling reste
  inchangé et non mesuré : aucun checkpoint EXL3 complet et son KDA exige un
  scan causal dédié, donc aucun refactor spéculatif n'a été ajouté.
- État d'intégration : **code local uniquement** sur `codex/rust-performance` ;
  app installée et GitHub inchangés, aucun push/release demandé.

### REL-2026-09-13 — Desktop v1.0.2 build 13 — validé et publié

- Demande utilisateur : embarquer le moteur Rust optimisé courant dans la GUI,
  pousser la nouvelle version, publier la release GitHub v1.0.2 et remplacer
  l'application locale. Signature ad-hoc maintenue comme v1.0.1 ; aucun
  certificat Developer ID/notarisation disponible.
- Source prévue : branche `codex/rust-performance`, commits PERF-01 à LOAD-10,
  PERF-16 LFM dense et PERF-17 Gemma validés. Versions Python/Info.plist/README
  synchronisées sur 1.0.2, build 13. Le DMG doit embarquer le binaire Rust et
  MLX 0.32.2, sans poids de modèle.
- Validation prévue : contrôles Rust/MLX déjà verts, E2E protocole/Desktop,
  build SDK26.5, manifeste sans changements suivis, signature stricte, montage
  DMG, self-checks GUI et smoke réel via le runtime monté. Conserver l'app
  installée précédente avant remplacement ; vérifier commit/tag/release et
  empreintes locales/distantes. État : **en cours**, non publié/non installé.
- Premier E2E local : tests Python **521 passés / 6 ignorés**, puis build Swift
  interrompu par le SDK27 actif sans plugin `SwiftUIMacros.StateMacro`, défaut
  de toolchain déjà documenté lors de v1.0.1. Aucun test Desktop n'a été exécuté
  après cet échec et aucun artefact publié. Relancer le même E2E avec
  `MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`,
  présent sur ce Mac ; aucune modification UI de contournement.
- Reprise SDK26.5 réussie : build Swift, checks hardening/timeline/MCP/mémoire
  Metal, callback bridge, rendu Markdown/code Unicode et transport CLI tous
  passés. Les avertissements SwiftMath dépréciés et chemins framework CLT déjà
  connus restent non bloquants. E2E source complet : **521 Python passés / 6
  ignorés**, puis tous les contrôles Desktop passés. Publication toujours non
  démarrée ; prochaine étape commit propre puis build du paquet final.
- Le premier mini-check de cohérence de version a été interrompu avant lecture
  des fichiers : le `python3` système ne fournit pas `tomllib`. `plutil` est
  néanmoins passé. Aucun artefact/commit affecté ; relancer le même assert avec
  la venv Python 3.12 déjà utilisée par la suite de tests.
- Reprise Python 3.12 réussie : Info.plist, `pyproject.toml` et
  `src/mlxl3/__init__.py` annoncent tous **1.0.2**, build **13** ; plist valide
  et `git diff --check` propre.
- Source figée au commit `55eb7461a925dafda5401938efc3dedb30bcf128`.
  Le build SDK26.5 est propre, le manifeste confirme le moteur Rust avec MLX
  0.32.2 et l'absence de Python embarqué. `codesign --verify --deep --strict`
  passe ; signature ad-hoc, sans notarisation, conformément à la limite acceptée.
- Artefact validé : `dist/MLXL3-Desktop-v1.0.2-b13-Apple-Silicon.dmg`,
  71 181 400 octets, SHA-256
  `5f1b87bb5d9e7ef6af53161a226f83e2ac0c10077170967cac8d610373e4339d`.
  Le runtime monté correspond au runtime source, SHA-256
  `de4299204bc7e844e95c812ca65e294ebf57bc23bb6284835f3ab868fd406c7c`.
- Validation depuis le DMG monté : signature, timeline, Metal et MCP passent ;
  deux tours réels LFM2 passent avec continuité CEDAR-42. Le second tour chaud
  mesure 2 066,83 tok/s prefill, 125,14 tok/s decode et 148,8 ms TTFT. Preuves
  `build/release-v102-build.log` et `build/release-v102-mounted-smoke.log`.
- Publication réussie : `main`, `codex/rust-performance` et le tag annoté
  `v1.0.2` pointent sur la source publiée ; release GitHub
  `https://github.com/0xZKnw/mlxl3/releases/tag/v1.0.2`, déclarée latest, avec
  DMG, checksum, manifeste et rapport de validation. L'empreinte de l'asset
  GitHub est identique à l'empreinte locale.
- CI finale : les trois exécutions `Native Rust checks` et les trois
  `Regression checks` déclenchées par branche, `main` et tag sont **réussies**.
  Une première requête de statut a utilisé le champ `isLatest`, absent de cette
  version de `gh` ; la vérification a été reprise via l'API `releases/latest`.
- Installation locale validée dans `/Applications/MLXL3 Desktop.app`, version
  **1.0.2 (13)**, même runtime et mêmes self-checks. L'ancienne 1.0.1 reste
  récupérable sous
  `build/app-backups/MLXL3 Desktop-v1.0.1-before-v1.0.2.app`. Application laissée
  fermée. État final : **validé, publié et installé**.

### CONV-2026-09-15-LING3-TINY-4BPW — reprise locale — validé localement

- Demande : produire localement Ling 3.0 Tiny en EXL3 uniforme 4 bpw, sans QAT.
  Source déjà complète : `models/source/Ling-3.0-tiny-HF`, révision documentée
  `e3a47d5b986e7141b6efd62597d598ebb392060d`; sortie
  `models/Ling-3.0-tiny-EXL3-4bpw`, travail `build/ling-4bpw`.
- Reprise vérifiée avant lancement : 1 600/9 031 projections mesurées dans 50
  checkpoints, 7 431 restantes. Les activations de calibration avaient été
  nettoyées (18 fichiers seulement, aucun `complete.json`) et seront donc
  régénérées ; les mesures existantes restent reprises par groupe exact de 32.
- Protocole : recette documentée `scripts/quantize_ling.py`, K=4/MCG, 2 048
  lignes réelles, deux séquences de 1 024 tokens, deux workers, backend Metal.
  Après conversion : inventaire strict, chargement EXL3, perplexité tenue à
  part contre BF16 sur les mêmes tokens, puis génération CLI. Aucun push HF
  n'est demandé à ce stade.
- ETA avant reprise : calibration ~15 min observées ; mesure restante ~108 min
  d'après la médiane des 20 derniers groupes (0,871 s/projection), puis émission,
  finalisation et validation estimées 45–90 min selon la réutilisation du cache
  quantifié 6 Gio. Total annoncé **2 h 45 à 3 h 30**. Apple M5 24 Gio, environ
  580 Gio libres ; `pmset` annonce alimentation secteur, batterie 83 % avec le
  sous-état « discharging ». État : **en cours**, chiffres à remplacer par les
  temps réels ; modèle final absent au lancement.
- Premier lancement bloqué avant import MLX par le sandbox sans GPU ; aucun
  calcul. Reprise hors sandbox autorisée, puis arrêt manuel à la demande après
  6/24 couches de calibration pour chercher un chemin plus court. Aucun poids
  final émis ; les checkpoints existants sont conservés.
- Reprise du 16 septembre, **en cours avant lancement** : recette identique
  ci-dessus avec bypass uniforme OPT-01, capture OPT-11 et scale expert 0,908
  OPT-12, sur la révision source épinglée. Baseline de qualité BF16 déjà
  mesurée : WikiText-2 test, 2 048 tokens, fenêtres 256, PPL 21,05421
  (`build/ling-4bpw/source-perplexity.json`). Mesurer le temps de conversion
  complet, puis inventaire strict, PPL EXL3 sur les mêmes données/fenêtres,
  chargement et génération CLI. Estimation pilote ~55–60 min, **non mesurée**
  sur modèle entier. M5/24 Gio, macOS local, ~557 Gio libres et secteur annoncé
  (`pmset` indique toutefois « discharging » à 83 %) ; état thermique inconnu.
  Commande : `.venv/bin/python scripts/quantize_ling.py --in-dir
  models/source/Ling-3.0-tiny-HF --out-dir
  models/Ling-3.0-tiny-EXL3-4bpw --work-dir build/ling-4bpw --bits 4
  --head-bits 4 --calibration-rows 2048 --calibration-seq-len 1024
  --max-workers 2 --search-backend metal`. Aucune publication demandée.
- Résultat du 16 septembre : conversion complète **9 031/9 031 modules EXL3
  K4/MCG**, aucun module omis, un shard HF standard, inventaire strict et
  validation du checkpoint réussis. Modèle final :
  `models/Ling-3.0-tiny-EXL3-4bpw` (4,1 Gio sur disque, ~4,4 Go décimaux) ;
  travail/reprise : `build/ling-4bpw`. Durée murale observée **environ 92 min**
  de lancement à la fin, contre 55–60 min estimées du pilote. Le Mac était
  initialement en décharge malgré « AC Power » (83 → 35 %), puis en charge ;
  état thermique non mesuré. Ne pas présenter la durée comme représentative
  d'un Mac stable sur secteur ou additionner des gains pilotes.
- Qualité tenue à part, mêmes 2 048 tokens WikiText-2 test, SHA256 de corpus
  `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`,
  fenêtres 256 et chunks d'exécution 128 : PPL BF16 **21,05421** → EXL3
  **21,66936** (+2,92 %). Preuves : `build/ling-4bpw/source-perplexity.json`,
  `build/ling-4bpw/exl3-perplexity.json`, `pipeline_summary.json` et
  `pipeline_state.json`. KL et capacités larges non mesurés ; cette PPL ne
  permet pas d'isoler l'effet du scale fixe OPT-12 par rapport à une autre
  conversion K4.
- Chargement CLI strict réussi (9 031 modules, 4,38 GB résidents dans ce run)
  et génération de la réponse finale correcte « 7 + 5 égale 12. » ;
  87,8 tok/s decode sur **un seul** prompt court, pas un benchmark comparatif.
  Les limites 96/256 tokens tronquaient la réflexion avant la réponse ; le
  test à 1 024 tokens s'est terminé naturellement après 424 tokens. Une
  alerte `transformers` sur `bailing_hybrid` apparaît au chargement du
  tokenizer mais n'empêche ni PPL ni génération. Enregistré dans le registre
  local comme `ling3.0-tiny-4bpw` et visible dans `mlxl3 list` ; GUI non testé.
  GitHub/HF inchangés. Quatre tests ciblés adaptateur/PPL réussis après run.

### OPT-2026-09-15-QUANT-01 — plan uniforme sans mesure redondante — en cours

- Hypothèse : avec `candidate_bits=[4]`, tête K=4 et unique shrinkage 0, la
  phase `measure_ldlq_candidates` ne peut choisir aucun autre plan. Elle
  quantifie pourtant 9 031 projections avant `convert_module_set`. Construire
  directement le plan K=4 conserve exactement recette, activations, Hessienne,
  LDLQ, codebook et poids, tout en laissant la passe d'émission quantifier une
  seule fois et utiliser le groupement Metal gate/up existant.
- Antécédents : optimisations Metal K=4 et préparation déjà intégrées ; ne pas
  les retester. Aucun bypass uniforme trouvé dans le converter. La mesure Ling
  interrompue avait 1 600 projections, médiane récente 0,871 s/projection ; son
  score ne sert pas à une décision avec un seul candidat.
- Baseline/candidat prévus : pipeline actuel contre détection automatique du
  cas strictement uniforme entier, sur les 18 mêmes premières projections,
  nouvelles sorties/work dirs, M5 24 Gio, secteur annoncé, backend Metal, deux
  workers, mêmes 2 048 lignes. Contrôler K=4, tenseurs EXL3 valides et comparer
  les empreintes d'une projection entre chemin mesuré et chemin direct. Si ce
  garde qualité échoue, retirer. ETA complète révisée seulement après le pilote.
- Une optimisation indépendante de stockage sera évaluée séparément : les
  gate/up des 128 experts reçoivent exactement le même tableau d'activation ;
  des hardlinks peuvent éviter les copies disque sans changer un octet lu. Ne
  pas cumuler son gain avec le bypass avant mesure séparée.
- Premier pilote direct, 18 projections (`build/ling-uniform-direct`,
  `models/Ling-3.0-tiny-pilot-direct`) : plan `fixed_uniform`, **0 candidat
  mesuré**, K=4 sur 18/18, calibration 12 s et pipeline 55 s d'après les mtimes.
  Le pipeline a bien émis les 18 shards ; le processus a ensuite échoué au
  validateur parce que le pilote demandait volontairement `--no-finalize`
  (shards Pony incrémentaux, pas de shards HF standard). Ce n'est pas un crash
  de quantification, mais le protocole sera relancé avec finalisation pour le
  garde bout-en-bout. Tests du garde : **2 réussis** avec Metal ; compilation
  et `diff --check` réussis.
- Le triplet expert observé prend ~2,78 s (down 0,94 s, gate/up groupés 1,84 s),
  mesuré par mtimes des shards. Extrapoler naïvement 2 944 experts donnerait
  ~136 min : le bypass retire la mesure redondante mais ne suffit pas seul pour
  viser <1 h. L'ancienne estimation d'émission 45–90 min est donc invalidée par
  ce pilote et ne doit plus être citée.
- Hardlinks gate/up implémentés localement, pas encore mesurés : un seul inode
  par couche remplace jusqu'à 256 copies strictement identiques. Les 6 couches
  interrompues occupent actuellement 10 Gio/2 200 fichiers et prouvent que le
  coût disque est matériel. Validation prévue : test d'identité octet/inode et
  nouveau pilote ; aucun gain temps n'est encore annoncé.
- Contrôle hardlinks terminé : **2 tests réussis** (`test_ling_conversion_adapter`),
  valeurs gate/up strictement égales et inode unique vérifié. Statut : validé
  fonctionnellement, gain disque théorique/temps complet encore non mesuré.

### OPT-2026-09-15-QUANT-02 — batch multi-experts Ling — en cours

- Hypothèse : le coût dominant est désormais l'enchaînement de 2 944 minuscules
  triplets experts, pas le calcul utile. `ldlq_quantize_group` sait déjà préparer
  et chercher plusieurs matrices Metal ; évaluer des lots de plusieurs experts
  de même shape/activation doit amortir dispatchs, synchronisations et lectures
  de Hessienne sans changer K, codebook, ordre LDLQ ni sorties par module.
- Baseline : expert 0 couche 1, K4, 2 048 lignes : 2,78 s pour down+gate+up sur
  le pilote direct ci-dessus. Protocole prévu : lots 1/2/4 experts sur la même
  couche et mêmes activations, empreintes exactes des tenseurs EXL3 contre lot 1,
  mémoire processus surveillée ; conserver uniquement un lot strictement égal
  et plus rapide. Aucun cumul/ETA <1 h avant ce contrôle.
- Premier lot 4 experts (`build/ling-expert-batched`, limite 33 modules) : 10,20 s
  pour down(4)+gate/up(8), soit **2,55 s/expert contre 2,78 s**, gain ~8,3 %.
  Le reliquat de 2 experts donne 2,61 s/expert. Les tenseurs EXL3 de l'expert 0
  sont **bit-identiques** au chemin lot 1 pour down/gate/up. Calibration 12 s,
  émission+écriture 56 s, total 68 s ; fin attendue en erreur uniquement parce
  que `--no-finalize` n'est pas accepté par le validateur MLXL3 post-pilote.
- Hardlinks sur ce pilote : 199 Mo de tailles logiques mais **124 Mo occupés** ;
  calibration toujours 12 s ici car seulement six experts. Gain stockage validé,
  gain temps à mesurer à l'échelle complète. Le batching seul est conservable
  mais ne suffit pas à <1 h ; prochaine piste : paralléliser la préparation/GSS
  actuellement forcée à un seul worker pour `scale_mode=computed`, avec même
  comparaison bit-à-bit et retour arrière si instable/régressif.
- Variante préparation 4 workers : pilote identique limite 33 dans
  `build/ling-expert-parallel`. Protocole complémentaire prévu après résultat :
  profiler un expert isolé avec `PONYEXL3_CONVERT_TIMING=1` pour attribuer le
  temps entre basis/GSS, Hessienne, LDL et boucle LDLQ avant tout nouveau kernel.
- Résultat 4 workers : 14,84 s pour six experts contre 15,42 s, soit **+3,8 %**
  sur le segment expert et 67,18 s contre 67,97 s bout-en-bout (**+1,2 %**).
  Les 18 projections expert (six triplets) sont bit-identiques. Deux tests LDLQ
  groupés réussis. Gain réel mais trop faible ; le profilage isolé décidera si
  cette concurrence reste intégrée ou si elle doit être retirée.
- Profil isolé K4 MCG : down 0,90 s (basis/GSS 0,48, LDLQ 0,38), gate 0,97 s
  (basis/GSS 0,43, LDLQ 0,50) ; GPU actif 84–85 %, 45/109 appels. Les deux
  moitiés sont donc matérielles. Prochain microbenchmark strict : 256 tiles K4
  MCG, scratch 256 contre 512 MiB, 7 paires alternées, même entrée et parité
  exacte. Motif nouveau malgré l'essai historique K6 négatif : les lots Ling
  K4 font exactement 256 tiles, donc 512 MiB peut supprimer un dispatch sur
  deux ; K6/shape lm_head ne validait pas cette géométrie.
- Résultat 256 tiles, 7 paires : médiane 256 MiB 24,793 ms, 512 MiB
  22,875 ms, soit **+8,38 %** sur le search microbenchmark ; sorties décodées
  et états bit-identiques. Variante complémentaire prévue sur 384 tiles
  (géométrie du lot down) à 256/512/768 MiB avant choix du budget ; aucun gain
  projection/complet encore revendiqué.
- Résultat 384 tiles, 6 paires : 256/512/768 MiB = 41,084/35,801/34,905 ms ;
  **+15,0 %** à 768 MiB contre 256, parité exacte. Décision de test : conserver
  256 MiB pour les modules seuls, autoriser 768 MiB uniquement dans le chemin
  groupé multi-experts, puis relancer le pilote six experts. Le surcoût actif
  maximal attendu est +512 MiB pendant le search groupé ; mesurer projection
  complète avant intégration définitive.
- Projection complète scratch 768 MiB : 14,709 s pour six experts contre
  14,839 s, **+0,88 %** seulement, malgré parité bit-à-bit et deux tests Metal
  réussis. Rejeté : +512 MiB actif ne vaut pas ce gain ; revenir au budget
  256 MiB. Nouvelle piste exacte : vectoriser le GSS des membres d'un groupe.
  Les 13 évaluations restent identiques par module, mais chaque ronde concatène
  les échantillons en un appel Metal et re-sépare les MSE. Garde prévu : GSS
  multi vs scalaire sur fonctions déterministes, puis 18 projections expert
  bit-identiques contre `ling-expert-parallel` et timing du même pilote.
- GSS vectorisé : 4 tests unitaires/Metal réussis et les 18 projections expert
  sont bit-identiques, mais six experts prennent **15,459 s contre 14,839 s**
  (−4,18 %) et le pilote 69,87 s contre 67,18 s. Rejeté et à retirer : la
  concaténation agrandit les batches sans réduire assez le calcul par tile.
  Ne pas cumuler ce résultat avec le batching multi-experts positif.

### OPT-2026-09-15-QUANT-03 — récursion LDLQ Metal réellement batchée — en cours

- Hypothèse : le lot actuel concatène la recherche treillis, mais exécute encore
  compensation, mise à jour et `matmul` LDLQ dans une boucle Python par expert.
  Le chemin officiel ExLlamaV3 `ldlq_batched` empile au contraire poids et
  facteurs L de tenseurs de même forme et remplace ces produits par des BMM.
  Porter uniquement ce chemin homogène vers `mx.matmul` batched doit réduire les
  dispatchs sans changer l'algorithme, K, codebook, ordre de feedback ni qualité.
- Baseline : `build/ling-expert-parallel`, six experts complets, **14,839 s**
  entre premier et dernier shard expert ; pipeline pilote 67,18 s. Candidat :
  mêmes 33 modules, activations, K4/MCG, 2 048 lignes, M5 24 Gio, backend Metal.
  Contrôles requis : les 18 treillis/suh/svh experts bit-identiques, tests groupés,
  mémoire MLX et temps du segment. Rejet immédiat si divergence ou régression.
  Ce test ne promet pas encore <1 h ; l'ETA ne sera révisée qu'après mesure.
- Résultat batch LDLQ empilé, six experts : **12,582 s** contre 14,839 s,
  soit **−15,2 %** sur le segment (9,828 s entre premier et dernier shard).
  Les 72 tenseurs expert comparés sont bit-identiques ; trois tests ciblés
  Metal/driver réussissent. L'extrapolation des 2 944 experts reste ~103 min,
  donc le chemin est validé mais ne suffit pas à l'objectif <1 h.
- Variante suivante, avant essai : augmenter le lot homogène de 4 à 8 experts
  (8 down, 16 gate/up) sur les 16 premiers experts de la même couche. Baseline
  conservée ci-dessus ; contrôler la parité des six experts communs, le temps
  par expert et l'absence d'explosion mémoire. Revenir aux petits lots si la
  latence moyenne ne baisse pas.
- Résultat lot 8 sur 16 experts : **33,956 s**, soit 2,122 s/expert contre
  2,097 s/expert avec les petits lots (régression ~1,2 %). Les 72 tenseurs
  communs restent bit-identiques. Variante rejetée ; limites revenues à 4 down
  et 8 gate/up. Prochaine hypothèse : partager le calcul de Hessienne brute entre
  gate/up qui lisent exactement les mêmes activations, puis appliquer la
  transformation de signes propre à chaque `suh` sans refaire `X.T @ X`.
- Variante ordonnancement, avant essai : le lot gate/up contient huit matrices
  mais la préparation est plafonnée à quatre workers. Tester huit workers sur
  le même pilote six experts et le même batch LDLQ empilé ; conserver seulement
  si le segment bat 12,582 s avec 72 tenseurs identiques. Ce contrôle rapide
  précède la refonte Hessienne, plus invasive.
- Résultat huit workers : **14,696 s** contre 12,582 s (régression 16,8 %),
  malgré 72 tenseurs bit-identiques. Rejeté et plafond remis à quatre : les
  threads supplémentaires saturent la même file Metal au lieu de la remplir.

### OPT-2026-09-15-QUANT-04 — recherche g-scale groupée en deux passes — en cours

- Hypothèse : la préparation est dominée par la recherche dorée séquentielle,
  13 recherches treillis et synchronisations par projection. Le quantificateur
  ExLlamaV3 actuel remplace cela, pour les groupes homogènes, par une grille
  grossière sous-échantillonnée puis une grille fine complète : deux lots Metal
  pour tout le groupe. Porter ce schéma uniquement aux experts doit supprimer
  la majorité des barrières sans QAT ni réduction de calibration.
- Baseline : batch LDLQ empilé, six experts, **12,582 s**. Protocole candidat :
  mêmes 33 modules/activations/K4, quatre experts par lot, grille amont 10+5,
  comparer les métriques `inner_mse` module par module et les tenseurs/scales.
  Puis, seulement si l'erreur agrégée n'augmente pas matériellement, estimer le
  modèle complet. Ce chemin n'est pas censé rester bit-identique car il choisit
  le minimum sur une grille globale plutôt qu'un minimum local de la recherche
  dorée ; retour arrière si le gain qualité/temps n'est pas simultanément établi.
- Résultat grille groupée : **11,862 s**, soit −5,7 % contre le batch LDLQ seul
  et −20,1 % contre la baseline 14,839 s. `inner_rel_rms` moyen passe de
  0,0883233 à 0,0883052 ; 8/18 projections s'améliorent, pire variation
  individuelle +0,103 %. Statut provisoire : gain réel et métrique agrégée non
  dégradée, mais tenseurs différents ; perplexité complète requise avant validation.
- Variante suivante, avant essai : avec seulement deux barrières g-scale par
  groupe, retester 8 experts (8 down, 16 gate/up) sur 16 experts. Le précédent
  essai à gros lots utilisait encore la recherche dorée par module et ne répond
  donc pas à cette nouvelle hypothèse. Comparer temps/expert et métriques aux
  petits lots ; retour à 4/8 si la moyenne ne baisse pas.
- Résultat gros lots avec grille : 16 experts en **31,857 s**, soit 1,991 s/expert
  contre 1,977 s/expert en petits lots (régression 0,7 %). Rejeté ; limites
  revenues à 4/8. La métrique moyenne sur ces 16 experts reste du même ordre
  (0,0883553), mais les populations diffèrent donc elle ne sert pas de gain.

### OPT-2026-09-15-QUANT-05 — attribution du temps groupé — en cours

- Avant nouvel essai d'optimisation, mesurer séparément préparation (poids,
  régularisation, Hessienne/LDL), g-scale groupé, allocation et récursion LDLQ
  sur le pilote six experts désormais à 11,862 s. Ajouter uniquement quatre
  compteurs de durée aux stats existantes, sans changer le calcul, puis retirer
  ou conserver ces compteurs selon leur utilité. Cette mesure choisira le prochain
  kernel ; aucune estimation <1 h ne sera faite sans identifier le poste dominant.
- Résultat six experts, quatre groupes : préparation cumulée 0,312 s, g-scale
  **4,983 s**, allocation 0,012 s, récursion LDLQ **6,513 s**. Les deux postes
  Metal expliquent presque tout le segment 11,86 s ; la piste Hessienne/CPU est
  abandonnée faute de plafond utile. Les compteurs restent locaux pour vérifier
  les prochains essais et seront retirés si non nécessaires à la fin.

### OPT-2026-09-15-QUANT-06 — g-scale sans packing jeté — en cours

- Hypothèse : la grille g-scale appelle le chemin de conversion complet, qui
  transforme les tiles, exécute la recherche, compacte les états en treillis puis
  inverse la permutation ; seul le MSE est utilisé. Appeler directement le kernel
  de recherche sur les tiles déjà permutées supprime packing et aller-retour sans
  changer le score (la permutation préserve exactement la somme des carrés).
- Baseline : g-scale cumulé 4,983 s, segment 11,862 s. Protocole : même pilote,
  vérifier mêmes `regularize_g_scale`, mêmes 72 tenseurs expert et temps. Rejet
  si un seul tenseur diffère ou si le g-scale ne baisse pas au-delà du bruit.
- Résultat : g-scale **5,553 s** contre 4,983 s (+11,4 %), segment 12,513 s
  contre 11,982 s sur la répétition instrumentée, et 6/72 tenseurs diffèrent
  à cause de l'ordre numérique du scale/permutation. Rejeté ; chemin direct
  précédent restauré. Ce résultat ne doit pas être cumulé avec les gains validés.

### OPT-2026-09-16-QUANT-07 — supprimer les barrières LDLQ par feedback — validé en pilote

- Hypothèse : le chemin empilé appelle `mx.eval` après chaque tranche de 16 rangs,
  soit jusqu'à 96 barrières CPU/GPU pour un groupe gate/up de 1 536 rangs. Cette
  barrière protège les très grandes matrices d'un graphe différé géant, mais les
  experts empilés ne gardent que huit étapes dans le bloc borné de 128 rangs.
  Évaluer une fois par bloc, comme la récursion device-side amont, doit supprimer
  7/8 des synchronisations sans changer une opération ni son ordre de dépendance.
- Baseline instrumentée : LDLQ cumulé **6,513 s**, segment 11,982 s sur la
  répétition. Protocole : déplacer uniquement `mx.eval(packed,b_reconstructed)`
  hors de la boucle feedback du chemin homogène ; mêmes 72 tenseurs bit-identiques,
  mêmes scales et mémoire sous le seuil existant. Rejet à la moindre divergence.
- Résultat : segment **10,988 s** contre 11,982 s (−8,3 %), LDLQ cumulé
  5,940 s contre 6,513 s (−8,8 %), et **72/72 tenseurs bit-identiques**.
  Trois tests ciblés Metal/driver passent. Validé sur le pilote ; estimation
  experts complets ~90 min, donc l'objectif <1 h n'est pas encore atteint.

### OPT-2026-09-16-QUANT-08 — barrière LDLQ tous les quatre blocs — rejeté

- Hypothèse : après OPT-07, il reste une barrière par bloc de 128 rangs (4 pour
  down, 12 pour gate/up). Les matrices expert empilées sont petites ; différer
  quatre blocs conserve le même graphe/opérations et borne les temporaires à
  quelques centaines de Mo, tout en divisant encore les synchronisations par 4.
- Baseline : LDLQ 5,940 s, segment 10,988 s. Protocole identique, 72 tenseurs
  exacts requis ; surveiller cache Metal et revenir à un bloc si mémoire ou
  temps régressent. Aucun changement de `buf_size_rows` ni d'association matmul.
- Résultat : **11,521 s** contre 10,988 s (+4,9 %), avec tenseurs exacts.
  LDLQ 6,055 s contre 5,940 s et g-scale 4,923 s contre 4,575 s ; différer
  davantage agrandit le graphe MLX sans accélérer le kernel. Rejeté, retour à
  une barrière par bloc de 128 rangs.

### OPT-2026-09-16-QUANT-09 — arithmétique K4 alignée sur CUDA — rejeté

- Hypothèse : après les gains d'orchestration, g-scale (4,575 s) et LDLQ
  (5,940 s) passent presque tout leur temps dans le même treillis K4. Le kernel
  CUDA officiel charge la cible en FP16 et effectue différences/FMA/comparaisons
  en `half2`, tandis que le port Metal conserve ces calculs en `float2`. Une
  variante Metal K4/MCG strictement calée sur cette arithmétique amont peut
  exploiter le débit FP16 du M5 et viser les ~1,5x encore nécessaires.
- Antécédents : les variantes exactes de lookup, vecteurs, threadgroups,
  compactage et scratch du rapport Metal sont déjà épuisées et ne seront pas
  répétées. Le mode `fast math` avait divergé et reste exclu. Cette variante est
  nouvelle mais ne promet pas la bit-identité avec le chemin Metal FP32 ; elle
  doit d'abord égaler le comportement numérique du quantificateur CUDA officiel.
- Baseline : noyau local K4/MCG actuel, puis pilote Ling six experts à
  **10,988 s** (72 tenseurs exacts contre son propre chemin scalaire). Candidat :
  même lookup/codebook, tail-biting, tie-break et calibration, seuls target,
  erreur et coûts passent par l'arithmétique half2 de l'amont.
- Protocole : microbenchmark 256/384 tiles en paires alternées, puis le même
  pilote 33 modules. Comparer `inner_rel_rms` par projection et agrégé ; ne pas
  intégrer ni annoncer une absence de perte avant perplexité BF16/EXL3 sur les
  mêmes tokens. Rejet si le débit ne permet pas plausiblement <1 h ou si les
  métriques se dégradent matériellement. M5 24 Gio, MLX 0.32.2, conditions
  thermiques non contrôlées ; aucune publication.
- Résultat microbenchmark, sept paires alternées : 256 tiles
  **25,161 → 23,757 ms** (+5,9 %) ; 384 tiles **38,039 → 37,146 ms** (+2,4 %).
  Le MSE synthétique est quasi inchangé/légèrement meilleur, mais seulement
  93,8 % des états correspondent au chemin FP32. Le plafond de 2–6 % sur le
  search ne peut pas faire passer l'estimation complète de ~90 à <60 min et
  imposerait tout de même une validation perplexité complète. Variante rejetée
  avant le pilote modèle ; aucun poids produit, chemin FP32 restauré.

### OPT-2026-09-16-QUANT-10 — scale représentatif par lot expert — remplacé

- Hypothèse : les 18 experts du pilote choisissent des g-scales très proches
  (0,8981–0,9148, moyenne 0,9070, écart-type 0,0048). Chercher 15 candidats sur
  le premier membre de chaque lot homogène puis réutiliser son scale pour les
  3/7 autres membres réduirait de 75–87,5 % le poste g-scale, actuellement 42 %
  du segment. Le treillis LDLQ final, Hessienne et calibration restent complets.
- Antécédents : ce n'est ni `skip_g_scale=1` (scale fixe 1,0), ni la grille
  groupée OPT-04 qui cherche encore 15 candidats par projection. Aucun essai de
  scale représentatif trouvé. Les scales observés justifient un pilote, pas une
  conclusion de qualité.
- Baseline/candidat : mêmes six experts et 33 modules que OPT-07, baseline
  **10,988 s**, `inner_rel_rms` moyen expert à recalculer depuis le résumé.
  Candidat expérimental activé uniquement par environnement ; premier scale du
  lot appliqué aux autres bases, diagnostics de recherche marqués non mesurés.
- Protocole : mesurer segment/g-scale/LDLQ et comparer `inner_rel_rms` des 18
  projections à OPT-07. Si la moyenne/pire dérivent matériellement, rejeter. Si
  le temps rend <1 h plausible, conserver seulement après perplexité complète
  BF16/EXL3 identique en corpus/tokens. M5 24 Gio, aucune publication.
- Résultat pilote : sommes des quatre groupes préparation/g-scale/LDLQ
  **0,304/4,575/5,940 → 0,269/0,944/5,363 s**, soit **10,83 → 6,58 s
  (−39,2 %)** sur ces postes. La moyenne `inner_rel_rms` passe de 0,08830522 à
  0,08830839 (**+0,0036 % relatif**) ; pire projection +0,224 %, meilleure
  −0,127 %. Les écritures des 18 shards couvrent 8,58 → 5,26 s, cohérent mais
  ne couvrent pas le premier groupe. Estimation experts seuls : ~54 min.
- Statut : gain prometteur mais **qualité provisoire**, car les treillis changent
  et aucune perplexité complète n'existe encore. Le mode reste expérimental par
  variable d'environnement ; ne pas l'utiliser pour la conversion finale avant
  un garde sur davantage d'experts puis la perplexité du modèle complet.
- Intégration : prototype retiré au profit du scale pré-calibré OPT-12, plus
  rapide et meilleur sur la population étendue ; aucun chemin partagé actif.

### OPT-2026-09-16-QUANT-11 — calibration down experts batchée — validé localement

- Hypothèse : `capture_experts` lance séparément gate et up pour chacun des 128
  experts afin de fabriquer les 2 048 entrées de down, soit 256 petits matmuls
  par couche. Empiler huit experts et utiliser `mx.matmul` batched supprime la
  majorité des dispatchs, sans changer poids, lignes, SwiGLU ni fichiers.
- Antécédents : aucun essai de batching de la capture Ling trouvé. Les anciens
  hardlinks ne concernent que gate/up et n'accélèrent pas ces matmuls down.
- Baseline : six experts sélectionnés dans le pilote précédent et leurs `.npy`
  issus du chemin scalaire ; le run complet interrompu avait annoncé ~15 min de
  calibration pour les 128 experts/couche. Candidat : lots de huit, pic Metal
  surveillé, comparaison bit-à-bit après cast FP16 contre les fichiers baseline.
- Protocole : recapturer le même pilote 33 modules dans un dossier neuf, mesurer
  calibration, comparer chaque activation down, puis microbenchmark 128 experts
  d'une couche seulement si la parité tient. Rejeter si divergence matérielle ou
  mémoire excessive. Aucun poids/publication.
- Résultat : les six activations du pilote sont bit-identiques mais 12,85 s
  contre 11,07 s de capture totale, car six experts n'amortissent pas le batch.
  Sur le cas réel de **128 experts**, les 128 fichiers FP16 sont bit-identiques
  et leur plage d'écriture/calcul tombe de **1,125 à 0,205 s (5,48x)**. La
  capture candidate complète des 24 couches, avec une seule couche expert
  sélectionnée, termine en 13,9 s. Le batching par huit est retenu ; pic mémoire
  processus non mesuré séparément, aucune erreur d'allocation observée.

### OPT-2026-09-16-QUANT-12 — scale expert Ling pré-calibré — validé en pilote

- Hypothèse : les 1 552 recherches déjà terminées sur les couches 1–5 donnent
  un g-scale expert global très stable autour de 0,908 (écarts-types par
  couche/projection 0,010–0,013). Pour **ce checkpoint et cette recette**, fixer
  0,908 réutilise une calibration déjà payée au lieu de refaire 15 treillis par
  projection/lot. Cela retire les ~7,7 min de g-scale encore estimées après
  OPT-10 et place les experts vers 46 min.
- Antécédents : différent de `skip_g_scale` à 1,0 et du scale représentatif par
  lot. Les résultats historiques viennent du même source, K4/MCG, 2 048 lignes
  et sigma ; ils ne généralisent pas à un autre modèle/recipe et ne doivent pas
  devenir une valeur globale du moteur.
- Baseline/candidat : OPT-10 à 6,58 s de postes groupés et moyenne
  `inner_rel_rms` 0,08830839 ; candidat expérimental 0,908 sur les mêmes six
  experts, aucune recherche g-scale. Comparer les 18 erreurs et temps ; garder
  uniquement derrière l'adaptateur Ling exact (révision source/recette), puis
  exiger la perplexité complète avant publication.
- Protocole : nouveau pilote 33 modules, activation par environnement, mêmes
  poids/calibration/Metal. Rejet si l'erreur moyenne ou le pire module dérivent
  matériellement. Aucun modèle final ni publication.
- Résultat six experts : postes groupés **10,830 → 5,769 s (−46,7 %)** contre
  la baseline OPT-07 ; g-scale 4,575 → 0,003 s. `inner_rel_rms` moyen
  0,08830522 → 0,08831060 (**+0,0061 % relatif**), pire projection +0,158 %,
  plusieurs projections meilleures. Estimation experts seuls ~47,2 min.
- Extension déclarée avant essai : quantifier les 128 experts complets de la
  couche 1 (384 projections, mêmes 2 048 activations) et comparer leurs erreurs
  à la mesure historique per-projection disponible. Cette population couvre la
  plage de g-scales optimale 0,874–0,940 absente du petit pilote. Mesurer temps,
  pic et dispersion ; la perplexité modèle reste ensuite obligatoire.
- Résultat couche complète : **384/384 projections** produites en 127,25 s de
  plage d'écriture ; sommes des 64 groupes : préparation 5,71 s, g-scale
  0,06 s, états 0,26 s, LDLQ 117,47 s. Face aux 384 mesures historiques avec
  recherche individuelle, `inner_rel_rms` moyen **0,08835081 → 0,08833356
  (−0,0195 % relatif)** ; pire projection +0,332 %, p95 +0,169 %, meilleure
  −0,348 %. Le processus pilote complet (capture, 15 denses, 384 experts,
  copie/écriture) prend 177,1 s.
- Décision : retenir 0,908 uniquement dans l'adaptateur Ling K4/MCG avec cette
  recette ; estimation 23 couches experts ~48,8 min, modèle complet **~55–60
  min** sous conditions similaires. Qualité modèle encore provisoire jusqu'à
  perplexité BF16/EXL3 complète ; aucune publication ni modèle final à ce stade.
- Intégration : code local activé automatiquement par `quantize_ling.py` pour
  K4/MCG seulement ; autres modèles/bpw/codebooks inchangés. Batching calibration
  intégré au même adaptateur. **86 tests réussis, 1 ignoré** (fixture MiniCPM
  absente), compilation et `diff --check` réussis. App, GitHub et HF inchangés.
- Révision du 16 septembre : le checkpoint complet avec cette option s'est
  converti en ~92 min, PPL BF16 21,05421 → EXL3 21,66936 (mêmes tokens).
  Ce résultat valide l'utilisabilité de cette conversion mais ne démontre pas
  la non-infériorité du scale 0,908 contre une conversion complète avec recherche
  individuelle des scales. Le temps pilote 55–60 min était trop optimiste.
- Publication du 16 septembre : le code et le patch PonyExl3 sont sur `main`
  (`cc2049c`) ; les tests après rebase passent (112/112). Le checkpoint public
  est sur https://huggingface.co/0xzknw/Ling-3.0-tiny-EXL3-4bpw, commit
  `45a71f6ce370ca4cde4f1eff50583090ec8e31d8`, avec ses 12 fichiers
  et sa carte. Smoke natif Rust avec l'app installée : chargement Ling réussi,
  réponse finale `12` à « Combien font 7 + 5 ? Réponds directement. » ; 146
  tokens, TTFT 927 ms, decode 33,1 tok/s sur un seul essai non comparatif.
  Le GUI lui-même n'a pas été mis à jour ni testé pour ce modèle.

### OPT-2026-09-16-RUST-PERF-18 — Ling EXL3 : baseline decode/prefill — en cours

- Demande : optimiser Ling 3.0 Tiny EXL3 4 bpw localement, avec cibles 150 tok/s
  decode et 500 tok/s prefill sur chat court, sans spéculation, requantification
  ni changement de sampling. Aucun objectif n'est considéré acquis d'avance.
- Antécédents : PERF-01 a validé le retrait des barrières par couche sur Qwen ;
  PERF-11 a rejeté le QMV paresseux pour le prefill ; PERF-12/13 ont validé le
  QMM multi-token, PERF-16 a rejeté le batch LFM MoE faute de parité, PERF-17
  n'a pas testé Ling, encore absent. Le présent essai établit une baseline Ling
  reproductible avant le moindre changement de runtime/kernel.
- Environnement initial : `main` `a473c96`, Apple M5/macOS 27.0 (26A428),
  alimentation secteur annoncée par `pmset` (batterie 98 %, état thermique non
  mesuré), checkpoint local `models/Ling-3.0-tiny-EXL3-4bpw`, binaire release
  `target/release/mlxl3-rs`. Version MLX à relever avant comparaison.
- Protocole : bridge Rust résident, un warmup exclu, cinq tours identiques,
  prompt français court et 128 tokens maximum, greedy et contexte 4096 ; noter
  tokens effectivement évalués/générés, cache, TTFT, prefill, decode, hash et
  RAM/pic tels que définis par le bridge. Une seconde mesure à prompt court
  mais suffisamment long pour tester QMM sera distincte et étiquetée. Si le
  contrôle de processus GPU concurrents est indisponible, le signaler.
- Résultat : non mesuré. Intégration : aucun changement moteur, CLI, GUI ou
  publication. Preuve brute prévue sous `build/ling-perf-18-baseline.json`.
- Première tentative interrompue avant chargement : le sandbox ne voit aucun
  GPU Metal (`[metal::load_device] No Metal device available`). Le `tee` a
  retourné 0 malgré l'échec du bridge et laissé un fichier de preuve vide ;
  aucune mesure n'existe. Relance du même protocole avec accès GPU, et
  `pipefail` pour propager l'échec du benchmark.
- Relance GPU réussie : `PYTHONPATH=src .venv/bin/python
  benchmarks/benchmark_bridge.py models/Ling-3.0-tiny-EXL3-4bpw
  --native-binary target/release/mlxl3-rs --max-tokens 128 --repeats 5
  --prompt "Explique en français, avec des exemples précis, comment fonctionne
  un modèle MoE local. Compare le routage des experts, la mémoire utilisée,
  le temps de préremplissage et la vitesse de génération. Termine par deux
  limites concrètes et une conclusion courte."` ; bridge contexte 4096,
  84 tokens prompt évalués, 128 tokens générés, cache 0/84 chaque tour.
  Cinq tours chauds : decode **33,8867 tok/s** médian (33,66–33,99), prefill
  **36,3923 tok/s** médian (36,21–36,47), TTFT **2,30999 s** médian, hash
  des cinq sorties identique. Chargement 1,426 s. Le `peak_memory_gb`
  **4,42837 GB** est l'estimation des poids résidents rapportée par le bridge,
  pas une mesure du pic RAM processus. Preuve
  `build/ling-perf-18-baseline.json`. Aucun accès fiable à la liste des autres
  processus GPU dans ce sandbox ; contention éventuelle non exclue. Batterie
  sur secteur d'après `pmset`, thermique et fréquences non mesurées.
- Décision : **baseline validée** pour ce protocole court. Écart brut aux cibles :
  4,43× pour le decode et 13,74× pour le prefill ; ce ne sont pas des gains
  attendus ni une comparaison équitable au benchmark GGUF de l'utilisateur.
  Aucun code moteur/app changé. Le profilage suivant utilisera cette référence.

### OPT-2026-09-16-RUST-PERF-19 — Ling : une synchronisation par token — en cours

- Hypothèse : `Ling::run` évalue `hidden` et l'état attention après chacune des
  24 couches, comme l'ancien Qwen avant PERF-01. Une seule synchronisation des
  logits à la frontière du token doit réduire le coût CPU/Metal sans changer
  l'ordre des opérations, les poids, les caches ou le sampling.
- Antécédents : PERF-01 a validé cette stratégie sur Qwen avec parité exacte.
  PERF-07 (graphe decode entier jusqu'à argmax) a régressé ; on ne répète pas
  cette piste. Les états Ling KDA/MLA sont distincts, donc le succès Qwen ne
  préjuge ni de leur coût ni de la stabilité mémoire Ling.
- Baseline : PERF-18, 84/128 tokens, cinq tours, **33,8867 tok/s** decode,
  **36,3923 tok/s** prefill, **2,30999 s** TTFT, hash stable. Candidat : ne
  retirer que les deux `eval()` par couche de `Ling::run`, évaluer les logits
  une fois en fin de token. Binaire baseline sauvegardé avant reconstruction.
- Protocole : build release identique, sortie forcée sur au moins quatre
  tokens (logits complets FP16, comparaison bit-à-bit) et hash bridge 84/128
  identiques ; cinq tours chauds appariés avec le binaire baseline si possible.
  Mesurer decode, prefill, TTFT, stabilité sur une génération >128 tokens et
  mémoire processus si disponible. Rejeter si logits divergent, crash/OOM ou
  régression stable. Preuves sous `build/ling-perf-19-*`.
- État : avant modification ; résultat **non mesuré**, code local uniquement.
- Build release réussi avec MLX 0.32.2 ; trois méthodes `eval_state` Ling sont
  devenues inutilisées et seront retirées après validation. Avertissement
  `rust-objcopy`/`libLLVM.dylib` historique, non bloquant. Binaire baseline
  `build/ling-perf-19-baseline-bin` SHA256
  `d938d72f8ae4e600e628a27f105634558c153dd5b015e6eed36dd95d9630a26a`.
- Contrôle forcé `forward --tokens 1,2,3,4` : fichiers JSON/logits complets
  du baseline et du candidat **bit-à-bit identiques**, SHA256 commun
  `0309e48c0101ed2293061d87153eff7175e6893c5b5d816`; preuves
  `build/ling-perf-19-forced-{baseline,candidate}.jsonl`. Les états KDA/MLA
  ne sont pas exportés par le checker actuel : parité d'état non démontrée.
- Candidat, même bridge 84/128, cinq tours chauds : decode **47,8943 tok/s**
  médian (47,69–48,14), soit **+41,33 %** vs PERF-18 ; prefill
  **48,0186 tok/s** (+31,95 %) ; TTFT **1,7495 s** (−24,26 %). Hash greedy
  identique sur les cinq tours et au baseline, `peak_memory_gb` rapporté
  inchangé 4,42837 GB (poids résidents seulement). Preuve
  `build/ling-perf-19-candidate.json`.
- Décision provisoire : **gain validé pour la forme 84/128**, sans preuve de
  mémoire processus ni de stabilité longue. Avant intégration définitive,
  tester 256 tokens générés sur le même prompt et ajouter un contrôle des
  états KDA/MLA ; ne pas interpréter l'estimation résidente comme pic RAM.
- Extension 84/256, un warmup et un tour candidat : **256 tokens générés sans
  crash ni ralentissement manifeste**, decode 50,2238 tok/s, prefill 51,8312
  tok/s, TTFT 1,6209 s, hash
  `08b82405862f00df38e2cebb148860920cc1e3ec19a89a83262ed94a2fd1a3e2`.
  Preuve `build/ling-perf-19-long.json`. Un seul tour n'établit pas un gain
  comparatif ni la RAM processus ; contrôle baseline 256 à faire.
- Contrôle baseline 84/256 : 35,1428 tok/s decode, 37,6237 tok/s prefill,
  TTFT 2,2329 s, **même hash complet** que le candidat. Gain candidat sur
  ce tour comparatif : +42,91 % decode ; preuve
  `build/ling-perf-19-long-baseline.json`. Pas de conclusion sur RAM.
- Contrôle d'état prévu avant essai suivant : exposer `--states` Ling dans la
  commande de parité déjà existante, sans changer le runtime de production,
  et comparer tous les caches convolutionnels/récurrents KDA et KV/RoPE MLA
  au modèle MLX-LM Python pour deux tokens imposés. Supprimer les anciennes
  méthodes `eval_state` devenues inutilisées. Ce test n'est pas un nouvel
  essai de performance ni un assouplissement de tolérance.
- Première exécution du contrôle d'état **interrompue avant le Rust** : pour
  `ArraysCache`, `state` contient `(cache, left_padding, lengths)`, pas les
  quatre tableaux KDA directement ; `np.asarray` refuse ce tuple hétérogène.
  Aucun écart numérique observé. Corriger l'oracle Python pour lire
  `layer.cache` comme le modèle le fait, sans changer la tolérance ni les poids.
- Deuxième exécution de parité production **rejetée au premier état récurrent** :
  les trois caches convolutionnels de la couche KDA 0 passent, mais
  `model.layers.0.recurrent` diffère sur 516 653/1 048 576 octets dès le
  premier token. Ce test compare Rust optimisé à MLX-LM Python ; il ne prouve
  pas que PERF-19 a introduit l'écart, puisque l'ancien moteur Rust n'exportait
  pas ses états. Les logits forcés avant/après et le texte 256 tokens restent
  exacts. Nouvelle vérification nécessaire : construire la variante Rust
  baseline avec le même export `--states`, comparer les flux d'état bit-à-bit
  entre deux binaires et ne conserver PERF-19 que si l'écart préexiste.
- Contrôle différentiel Rust effectué : les deux variantes partagent le même
  export `--states`, la baseline restitue les évaluations `hidden` puis caches
  par couche et le candidat les diffère. Sur `--tokens 1,2 --states`, le flux
  JSON complet (tous logits et caches KDA/MLA) a le **même SHA256**
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`
  dans les deux binaires. L'écart FP32 avec MLX-LM Python préexistait donc
  à PERF-19 ; il reste à investiguer séparément et n'est pas une régression
  de l'optimisation. La variante source optimisée doit être restaurée après
  cette vérification, puis revalidée en build release.
- Source optimisée restaurée. Le checker Python garde les logits Ling exacts
  par défaut ; `--ling-states` active séparément le diagnostic d'état déjà
  connu divergent contre MLX-LM. Aucun seuil numérique assoupli.

### OPT-2026-09-16-RUST-PERF-20 — Ling KDA : regrouper cinq projections — en cours

- Hypothèse : les projections KDA q/k/v/f/g partagent la même entrée et le
  même format K4/MCG. Réutiliser `ProjectionBundle`/`Exl3Group` déjà validé
  pour Qwen/LFM peut réduire lancements QMV et préparation Hadamard sur les
  18 couches KDA, sans nouveau kernel ni nouveau format.
- Antécédents : PERF-19 (barrières Ling) est la baseline ; le rapport decode
  du 10 septembre indique que le groupement LFM a peu gagné et peut coûter en
  RAM/prefill. Ce test Ling n'est donc pas supposé positif. Les cinq sorties
  doivent conserver leur ordre, leurs dimensions et les scales indépendantes.
- Baseline/candidat : binaire `build/ling-perf-19-candidate-bin`, modèle Ling
  EXL3 4 bpw, 84/128 tokens, cinq tours chauds, médianes **47,8943 tok/s**
  decode, **48,0186 tok/s** prefill, **1,7495 s** TTFT ; candidat = groupement
  KDA uniquement, MoE/MLP/router et prefill sériel inchangés.
- Contrôles avant mesure : `forward --tokens 1,2 --states` du candidat
  bit-à-bit identique au binaire baseline (SHA256 ci-dessus), puis même hash
  greedy 84/128. Mesurer cinq tours et temps de chargement, éviter de revendiquer
  un pic RAM réel à partir de `resident_gb`. Rejeter en cas de divergence ou
  régression stable. Preuves `build/ling-perf-20-*`.
- État : avant modification ; **non mesuré**, aucun GUI/GitHub changé.
- Premier candidat construit avec MLX 0.32.2 : le flux complet `forward
  --tokens 1,2 --states` est bit-à-bit identique à PERF-19 (SHA256
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`).
  Bridge 84/128, cinq tours chauds : decode **52,9065 tok/s** médian
  (52,68–53,03), prefill **53,9263 tok/s** médian, TTFT **1,5579 s**
  médian ; les cinq hash greedy sont identiques à PERF-19. Par rapport aux
  médianes historiques PERF-19 : +10,47 % decode, +12,30 % prefill et
  −10,95 % TTFT. Preuve `build/ling-perf-20-candidate.json`. Chargement
  0,685 s sur cache disque chaud, non comparable aux 1,426 s initiaux ; RAM
  processus toujours non mesurée. **Provisoire** : effectuer A/B apparié avec
  les deux binaires maintenant avant de conclure, car le thermique et le cache
  système peuvent expliquer une partie du delta.
- Contrôle A/B/A immédiat, même prompt 84/128 et trois tours chauds par
  binaire : ancien A `build/ling-perf-20-control-a.json` **67,6628 tok/s**
  decode et **68,5514 tok/s** prefill ; candidat B
  `build/ling-perf-20-control-b.json` **69,1228 tok/s** decode et **69,6417
  tok/s** prefill ; ancien A2 `build/ling-perf-20-control-a2.json`
  **66,5412 tok/s** decode et **67,6777 tok/s** prefill. Le candidat dépasse
  les deux contrôles de ~2–4 % en decode et ~2–3 % en prefill, mais la hausse
  globale de ~48 à ~68 tok/s entre les séries historiques vient manifestement
  aussi des conditions machine. Le gain **+10,47 %** précédent est donc
  invalide comme attribution causale. Les trois séries ont le même hash de
  réponse et l'export d'état bit-à-bit reste identique. Décision : **gain
  faible/provisoire**, code local conservé pour évaluer d'autres formes,
  aucune publication ou installation GUI ; mesure RAM réelle non faite.

### OPT-2026-09-16-RUST-PERF-21 — Ling MLP partagé : groupement gate/up — en cours

- Hypothèse : le MLP partagé est invoqué dans chaque couche MoE Ling, et ses
  projections gate/up lisent le même vecteur. Le `ProjectionBundle` existant
  peut supprimer une transformation Hadamard et un lancement QMV par couche
  sans changer les poids ni le résultat. Le seul MLP dense initial suit la
  même voie. Nouveau périmètre par rapport à PERF-20 (KDA uniquement) ; les
  résultats LFM de groupement modestes sont une raison de mesurer, pas un gain
  présumé.
- Baseline : code local PERF-20, binaire `target/release/mlxl3-rs` avant
  modification, 84/128 tokens, trois tours appariés. Dernière série
  **69,1228 tok/s** decode, **69,6417 tok/s** prefill, TTFT **1,2064 s** ;
  conditions machine variables, à mesurer en A/B/A. Contrôle strict prévu :
  `forward --tokens 1,2 --states` SHA256 identique, hash greedy identique.
- Changement : `Mlp` Ling réutilise `ProjectionBundle` pour gate/up et conserve
  down inchangé. Aucun kernel inédit, quantification, sampler ou GUI modifié.
  Mesurer chargement et RAM processus si possible ; rejeter toute divergence,
  crash ou régression stable. Preuves sous `build/ling-perf-21-*`.
- État avant essai : **non mesuré**, code local uniquement.
- Build release réussi (MLX 0.32.2, avertissement non bloquant `rust-objcopy`
  identique aux builds précédents). Les logits et tous les états KDA/MLA de
  `forward --tokens 1,2 --states` conservent exactement le SHA256
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`.
  Le hash greedy 84/128 reste identique sur tous les tours.
- A/B/A rapproché, trois tours chauds chacun : A PERF-20 **58,834** decode,
  **59,327** prefill, **1,416 s** TTFT ; B candidat **59,770** decode,
  **60,014** prefill, **1,400 s** TTFT ; A2 PERF-20 **58,132** decode,
  **58,241** prefill, **1,442 s** TTFT. Unités tok/s hors TTFT. Preuves
  `build/ling-perf-21-control-a.json`, `build/ling-perf-21-candidate.json`,
  `build/ling-perf-21-control-a2.json`. Le candidat est ~1,6–2,8 % au-dessus
  des deux contrôles en decode, ~1,2–3,0 % en prefill. **Gain faible validé
  pour cette forme**, pas un progrès vers 150/500 à lui seul. Chargement
  ~0,69–0,70 s sur cache chaud ; pic RAM processus non mesuré. État : code
  local seulement ; GUI/app non reconstruits, aucun commit/publication.

### OPT-2026-09-16-RUST-PERF-22 — KDA vector : noyau Metal multi-token — en cours

- Hypothèse : le noyau `gated_delta::step_vector` n'accepte que `T=1`, ce qui
  interdit le prefill Ling par lots. Parcourir `T` dans chaque thread Metal
  tout en gardant la même accumulation FP32 et le même ordre par token évite
  de relancer ce noyau à chaque token. Le chemin decode `T=1` doit rester
  bit-à-bit inchangé ; ce changement seul n'accélère pas encore le prefill de
  l'app avant que les autres blocs Ling acceptent le batch.
- Antécédents : Qwen utilise déjà un noyau Gated DeltaNet multi-token, mais
  Ling a une décroissance vectorielle `[B,T,H,128]` et un état différent.
  PERF-18–21 montrent surtout que Ling est actuellement sériel en prefill.
- Baseline : `step_vector` actuel, test GPU one-hot `vector_step_matches_one_hot_update`
  et binaire PERF-21. Nouveau contrôle : comparer sur deux pas imposés le
  résultat et l'état d'un appel `T=2` aux deux appels `T=1` enchaînés, d'abord
  sur un cas simple puis sur tenseurs déterministes. Toute divergence décisive
  doit être expliquée avant intégration. Mesure de performance : non applicable
  au chat tant que le batch Ling complet n'est pas activé ; ne pas revendiquer
  de gain end-to-end prématurément.
- État avant essai : **non mesuré**, code local uniquement, pas de GUI/push.
- Implémentation locale : boucle temporelle dans le noyau Metal vectoriel,
  un seul chargement et une seule écriture de l'état FP32 par thread, mêmes
  opérations par token. `cargo test --release --locked --features mlx
  gated_delta::tests::vector_batch_matches_serial_steps -- --ignored --nocapture`
  sur M5 : **réussi**, sorties FP16 et état FP32 exactement égaux pour deux
  tokens déterministes. L'avertissement `rust-objcopy` reste non bloquant.
  Contrôle sur tenseurs variés/modèle entier encore à faire avant activation
  du prefill. Statut : **prototype validé pour ce cas**, aucun gain chat mesuré,
  aucune app installée ni publication.

### OPT-2026-09-16-RUST-PERF-23 — Ling prefill groupé 84 tokens — en cours

- Hypothèse : le bridge sérialise Ling à un token par appel, ce qui maintient
  toutes les projections EXL3 sur QMV et relance 24 couches par token. Avec
  PERF-22 (état KDA temporel), réutiliser QMM et MoE segmenté déjà présents
  devrait améliorer surtout le prefill/TTFT. Ce test est spécifique à Ling ;
  les poids, le sampling et le chemin decode `T=1` restent inchangés.
- Plan minimal : accepter `[1,T,H]` dans KDA, MLA et feed-forward Ling ; faire
  parcourir une ligne par thread au routeur groupé (même calcul par ligne) ;
  ajouter le masque causal au biais MLA et conserver uniquement les logits du
  dernier token ; autoriser des chunks de 128 tokens sur le bridge. Pas de
  nouveau QMM/quantification. Les caches doivent représenter tous les tokens
  du chunk, et le chemin `T=1` doit conserver les résultats bit-à-bit.
- Baseline : PERF-21, 84/128 tokens, trois tours, **59,770 tok/s** decode,
  **60,014 tok/s** prefill, **1,400 s** TTFT sur son dernier run ; les
  conditions machine varient. Pour une attribution causale : A/B/A rapproché
  contre le binaire PERF-21 préservé, avec cinq tours si stable. Vérifier
  d'abord un prefill batch de 2/25/84 tokens vs sérial : logits, tous caches
  KDA/MLA, séquence greedy et absence de crash. Si l'arithmétique QMM diffère
  bit-à-bit de QMV, appliquer un seuil numérique justifié et vérifier le
  routing/texte, sans prétendre à une égalité exacte.
- Rejeter/limiter le batch s'il corrompt l'état, diverge dans le texte ou
  fait exploser la RAM processus. Preuves `build/ling-perf-23-*`.
- État avant essai : **non mesuré**, code local seulement, aucune installation
  ou publication.
- Premier test GPU `cargo test --release --locked --features mlx
  ling::tests::batched_prefill_matches_serial_state -- --ignored --nocapture`
  **échoué** avant benchmark : 25 tokens imposés, logits finaux écart maximal
  absolu 0,31445313 ; pire cache `model.layers.13.conv_q` 0,48632813. Le
  chemin sériel `forward --tokens 1,2 --states` reste exactement identique au
  SHA256 précédent. Le candidat batch ne peut pas être activé en production en
  l'état. Causes possibles : QMM vs QMV, convolution batch ou routage ; isoler
  par couche avant toute mesure de vitesse. Aucun gain revendiqué.
- Répétition diagnostique du même test, avec sortie des premiers caches qui
  divergent : cette répétition ne cherche pas un gain, elle doit déterminer si
  l'écart démarre dès la convolution KDA 0 (projection/conv) ou plus tard
  (routage/MLA/récurrence). Aucun seuil n'est assoupli.
- Résultat diagnostic : premier écart à `model.layers.0.conv_q` de 0,001953125,
  compatible avec un changement d'arithmétique QMM/QMV ; les écarts croissent
  ensuite (MLA couche 3 : `kv_cache` 0,0480 ; cache conv q couche 13 : 0,4863).
  Ce test seul ne distingue pas une petite différence de projection amplifiée
  par MoE d'une erreur de masque/état. Test suivant : chat réel 84 tokens avec
  hash greedy et débits, **diagnostic seulement** ; ne pas intégrer même si
  rapide tant que parité et stabilité ne sont pas établies.
- Première commande bridge diagnostique **interrompue avant chargement** :
  `cargo test --features mlx` a remplacé `target/release/mlxl3-rs` par un
  binaire sans feature `chat` ; le bridge signale `requires a build with
  --features mlx,chat`. Aucune mesure. Reconstruire explicitement ces deux
  features puis relancer le même diagnostic.
- Diagnostic chat après rebuild correct : prompt réel 84 tokens, 128 générés,
  un tour chaud, **104,36 tok/s prefill**, **55,31 tok/s decode**, TTFT
  **0,805 s** ; hash greedy **identique** à PERF-21. Preuve
  `build/ling-perf-23-diagnostic.json`. Warmup 25 tokens : prefill seulement
  12,88 tok/s et TTFT 1,941 s, potentiellement compilation/shape ; ne pas
  masquer ce coût. Le test numérique forcé reste échoué, et ce seul hash réel
  ne suffit pas à valider la fidélité générale. Contrôle A/B/A et plusieurs
  prompts nécessaires après localisation de l'écart ; aucune activation
  durable, installation ou publication encore validée.
- Profil diagnostic prévu : exécuter un chunk de 84 tokens avec une barrière
  `eval()` **uniquement dans un test ignoré**, après chaque couche, relever
  chaque durée et comparer KDA/MLA/MoE. Le profil modifiera l'ordonnancement
  GPU et ne sera **pas** une mesure end-to-end ni une optimisation. Il doit
  seulement choisir la prochaine piste, sans ajouter de barrières au moteur.
- Profil 84 tokens réussi : couche 0 56,9 ms, couche 1 **458,5 ms**, toutes les
  couches 2–23 ensuite ~3,3–10,1 ms, head 3,0 ms. Test ignoré
  `ling::tests::profile_batched_prefill_layers`, sortie console de ce run
  (pas de log persistant). La couche 1 n'est pas intrinsèquement lente : elle
  semble payer la compilation du premier MoE QMM segmenté ; les couches
  suivantes partagent la shape compilée. C'est une **inférence**, pas encore
  prouvé par un deuxième run chaud. Répéter le benchmark bridge **dans le
  même processus et avec la même forme 84 tokens** doit distinguer coût de
  compilation/TTFT initial et débit stable. Cette répétition reprend le
  protocole PERF-23, sans changement de code moteur.
- Cinq requêtes consécutives 84/128 dans le même bridge, après warmup de
  **25 tokens** : prefill **569,95 tok/s** médian (566,5–573,6), TTFT
  **0,148 s** médian, decode **55,31 tok/s** médian (55,2–55,8) ; cinq hash
  greedy identiques à PERF-21. Preuve `build/ling-perf-23-batch-five.json`.
  La requête précédente à 104 tok/s était la première compilation de cette
  shape 84, pas le débit chaud stable. La cible **500 tok/s en prefill chaud
  sur cette forme** est atteinte par le prototype, mais ni le prefill froid,
  ni la fidélité numérique sur 25 tokens imposés, ni le 150 tok/s decode.
  Ne pas généraliser à d'autres longueurs/contextes : compilation par shape
  probable. Statut : **prototype non validé pour activation** tant que l'écart
  logits/caches et plusieurs prompts ne sont pas évalués.
- Vérification numérique suivante avant décision : tester le routeur groupé
  multi-lignes séparément contre quatre appels mono-ligne, avec logits/biais
  déterministes et comparaison **bit-à-bit** des indices et scores. Cela
  cherche une erreur logique de batch, pas un gain de vitesse ni un
  assouplissement du seuil modèle entier.
- Test GPU `router::tests::grouped_router_batch_matches_serial` **réussi** :
  quatre lignes, 128 experts, top-8, indices et scores exactement identiques
  aux quatre calculs mono-ligne. Le routeur multi-lignes seul n'explique pas
  l'écart numérique observé dans le modèle. Compilation/trace du test dans
  la sortie de `cargo test` locale, non conservée en fichier brut.
- Contrôle qualité supplémentaire prévu, sans changement du moteur : avec le
  binaire PERF-23 restauré, comparer le dernier logit de `forward` sériel et
  `forward --batch` sur 25 IDs imposés, puis sur une phrase française tokenisée.
  Lire les valeurs comme FP16 (pas comme entiers), relever top-1, écart absolu
  moyen/maximal et KL des softmax FP32/64. Vérifier ensuite deux chunks et le
  hash greedy réel. Une égalité du texte seule ne suffit pas ; si l'écart
  numérique reste grand, garder le batch hors du chemin normal. Preuves
  `build/ling-perf-23-quality-*`, alimentation/thermique non contrôlées mais
  non pertinentes pour ce contrôle de valeur.
- Résultat : 25 IDs imposés, même top-1 et top-10/10, écart absolu moyen
  **0,05437**, maximal **0,31445**, KL **0,0000689** ; prompt français de
  17 tokens passé par le chemin sériel (seuil 24), égalité exacte attendue.
  Prompt français répété jusqu'à ~85 tokens, batch effectivement activé :
  top-1 identique, top-10/10, moyenne **0,10165**, maximum **0,57422**,
  KL **0,0027523**. Fichiers `build/ling-perf-23-quality-{serial,batch}-{25,fr,fr85}.jsonl`.
  Le seuil provisoire max 0,5 cité pour d'autres familles est dépassé ;
  cela ne prouve ni corruption logique ni absence de dérive en multi-tours.
  **Statut non concluant pour activation générale** : maintenir le batch en
  prototype local et ne pas l'installer/publier avant davantage de prompts et
  contrôle des deux chunks. Le decode mono-token reste la référence fidèle.
- Validation chat supplémentaire prévue, **répétition de PERF-23** motivée par
  le nouveau routeur PERF-26 et par l'absence de test multi-chunks : comparer
  le binaire sériel PERF-21 (`build/ling-perf-21-candidate-bin`) au candidat
  batch+routeur PERF-26 sur un prompt >128 tokens (deux chunks), un prompt de
  code et un autre français. Un warmup puis une répétition par forme ; vérifier
  hash greedy, absence de crash, nombres de tokens, préfill/TTFT/decode, et ne
  pas extrapoler la RAM du bridge. Preuves `build/ling-perf-23-quality-chat-*`.
- Première tentative multi-chunks **interrompue sans mesure** : le binaire
  PERF-21 archivé a été compilé sans `chat`, malgré l'aide CLI montrant
  `bridge` ; erreur explicite `requires a build with --features mlx,chat`.
  `build/ling-perf-23-quality-chat-long-serial.json` est vide/non valide.
  Reprendre le même prompt avec le binaire sériel PERF-19 archivé avec
  `chat` après vérification ; son hash forcé était identique à PERF-21.
- Prompt français 146 tokens (deux chunks 128+18), 96 tokens générés, même
  hash greedy sur baseline sérielle PERF-19 et batch+routeur PERF-26.
  Sériel : prefill **58,42 tok/s**, TTFT **2,500 s**, decode **55,93 tok/s** ;
  batch : prefill **97,71 tok/s**, TTFT **1,495 s**, decode **78,28 tok/s**.
  Le second chunk de 18 reste sériel par seuil 24 et la première shape 128
  paye probablement la compilation ; ce test n'est pas un débit chaud
  stabilisé, mais établit que deux chunks passent sans crash ni divergence
  greedy sur cet exemple. Preuves `build/ling-perf-23-quality-chat-long-{serial19,batch26}.json`.
  Fidélité numérique stricte multi-prompt encore non démontrée : **prototype**.
- Dernier contrôle qualité prévu avant arrêt de cette série : comparer sur une
  demande de code distincte (prompt >24 tokens, 96 tokens greedy) la version
  sérielle PERF-19 et le build final batch+routeur PERF-26, un warmup + un
  tour par binaire. Noter hash, tokens, débit ; ce contrôle de texte ne remplace
  pas une évaluation de qualité large. Preuves
  `build/ling-perf-23-quality-chat-code-{serial,batch}.json`.
- Demande de code 87 tokens / 96 générés : baseline sérielle et batch ont
  **des hashes greedy différents**, bien que les premiers ~120 caractères
  enregistrés soient identiques. Sériel 56,17 tok/s decode, 57,33 tok/s
  prefill, TTFT 1,518 s ; batch+PERF-26 78,14 decode, 344,57 prefill,
  TTFT 0,253 s (première compilation shape 87). Preuves
  `build/ling-perf-23-quality-chat-code-{serial,batch}.json`. La vitesse
  ne compense pas cette divergence tant que sa qualité n'est pas examinée.
  **Décision provisoire : ne pas activer le préfill batch Ling par défaut** ;
  conserver `forward_tokens` comme prototype diagnostic/benchmark et revenir
  au préfill sériel pour le chat normal. Le gain decode PERF-26 est indépendant
  et conserve la parité exacte sur les tokens forcés.

### OPT-2026-09-16-RUST-PERF-24 — Ling decode : profil attention/MoE — en cours

- Hypothèse de diagnostic : avec le prefill chaud désormais >500 tok/s,
  l'objectif restant (150 tok/s decode) nécessite de réduire le coût par
  token de ~18 ms à <6,7 ms. Mesurer d'abord la part des 18 couches KDA,
  6 MLA et 23 MoE ; ne pas supposer que l'EXL3 QMV est seul responsable.
- Antécédents : PERF-18–21 ont retiré des barrières et groupé les projections
  Ling ; gains decode validés ou modestes. Le rapport decode du 10 septembre
  a rejeté les changements globaux de budgets command-buffer et quelques
  compilations FFN/QMV. Aucune répétition de ces pistes ici.
- Baseline : bridge local PERF-23, decode ~55 tok/s dans cinq tours, même
  texte que l'ancien moteur, machine/thermique variables. Protocole : test
  ignoré sur Metal, un warmup `T=1`, puis un token imposé avec barrières de
  diagnostic séparant attention et feed-forward par couche. Ces barrières
  changent le coût absolu : interpréter le profil **relatif seulement**. Pas
  de changement production, mesures RAM et TTFT non applicables à ce profil.
- État avant essai : **non mesuré**, code local, aucune publication.
- Profil diagnostique mono-token réussi après un warmup, avec `eval()` après
  chaque sous-bloc : hors deux pointes probables de JIT/scheduler (couche 1
  FF 2,893 ms, couche 7 attention 2,918 ms et couche 19 attention 2,325 ms),
  attention ~0,40–0,58 ms/couche, FF MoE ~0,54–0,63 ms/couche, dense couche
  0 ~0,30 ms. Trace console du test
  `ling::tests::profile_decode_components`, pas de log brut conservé. Les
  barrières rendent la somme non représentative du débit bridge, mais le
  MoE est probablement la première cible decode. Étape suivante : séparer
  dans un seul MoE routeur, experts EXL3 et branche MLP partagée avant de
  changer un kernel. Statut : **diagnostic**, aucune optimisation validée.
- Extension du même diagnostic prévue : sur une couche MoE représentative,
  après warmup, cinq répétitions séparant routeur, experts `Exl3SwitchGlu` et
  MLP partagé. Les `eval()` de profil empêchent la fusion/chevauchement et
  chaque durée est une borne indicative, pas un débit réel. Cette répétition
  vise à sélectionner la partie à optimiser, pas à déclarer un gain.
- Résultat couche MoE 8, répétitions 2–4 après premier JIT : routage
  **0,553–0,560 ms**, experts EXL3 **0,274–0,279 ms**, MLP partagé
  **0,194–0,205 ms**. Le routage inclut matmul FP32, sigmoid, sélection
  mono-thread et normalisation des scores, avec barrières `eval()` ; sa part
  est la plus importante de ce microprofil mais n'est pas additionnable aux
  temps end-to-end. Trace `ling::tests::profile_decode_moe` console seulement.
  Décision : essayer une **fusion du routeur Ling mono-token** dans un kernel
  Metal, avec fallback inchangé pour le batch et test strict des experts
  sélectionnés ; ne pas toucher au QMV expert pour l'instant.

### OPT-2026-09-16-RUST-PERF-25 — Ling : routeur MoE mono-token fusionné — en cours

- Hypothèse : fusionner multiplication dense x·W, sigmoid, top-groups/top-k
  et normalisation dans un seul kernel Metal pour 128 experts évite plusieurs
  dispatchs par couche MoE, sans changer poids, experts EXL3 ou sampling.
  Le chemin multi-token PERF-23 garde le routeur existant. Aucun routeur Qwen,
  Gemma ou LFM n'est modifié.
- Baseline : code/binaire PERF-23 préservé avant modification, bridge 84/128,
  cinq tours chauds, **~55,31 tok/s** decode dans la série la plus récente,
  prefill batch **569,95 tok/s** chaud, hash constant. Machine variable : A/B/A
  rapproché indispensable. Contrôles : parité indices de route sur entrées
  déterministes puis `forward --tokens 1,2 --states` contre binaire baseline,
  hash greedy 84/128 et 256 tokens, RAM et TTFT rapportés séparément.
- Rejeter si experts changent, régression stable ou erreur Metal ; si les
  scores ne sont pas bit-à-bit identiques, quantifier l'écart et vérifier les
  logits/caches, sans appeler cela parité exacte. Preuves `build/ling-perf-25-*`.
- État avant essai : **non mesuré**, code local uniquement, aucun push/GUI.
- Kernel Metal fusionné local ajouté pour `T=1`, avec matrice de gate FP32
  stockée aussi en vue rangées et routeur multi-token inchangé. Test GPU
  déterministe `router::tests::fused_ling_router_matches_reference` **réussi**
  pour 128 experts, 128 entrées : huit indices identiques et erreur maximale
  de score <1e-4. Ce microtest ne prouve pas encore la parité du modèle réel
  1536 entrées. Prochaine étape déjà prévue : build bridge puis comparer
  `forward --tokens 1,2 --states` au SHA256 du binaire PERF-23 avant tout
  benchmark de débit.
- Contrôle `forward --tokens 1,2 --states` **non identique** au binaire
  PERF-23 : SHA256 candidat
  `9e269218137925b1a27e230982573497f446c4beddc0d02c5a2f55323ab3578e`
  contre baseline
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`.
  Le test synthétique ne suffisait donc pas. Avant toute mesure de vitesse,
  comparer les logits forcés détaillés et identifier si la différence est
  une petite variation FP32 de scores ou un changement de route/résultat.
- Détail logits forcés `--tokens 1,2` : au token 1, **85 814/157 184**
  valeurs FP16 diffèrent ; au token 2 **156 236/157 184**, même argmax aux
  deux tokens mais divergence trop grande pour accepter la fusion. Preuves
  `build/ling-perf-25-forced-{baseline,candidate}.jsonl`. Le `maxbits`
  calculé sur encodages FP16 n'est pas un écart réel et n'est pas retenu.
  Diagnostic suivant : comparer directement les routes/scores fusionnés et
  non fusionnés avec les **vrais poids** des couches Ling 1 et 8, afin de
  déterminer si la disposition de la matrice FP32 est fautive. Pas de
  benchmark de vitesse tant que le résultat est faux.
- Vrais poids Ling couches 1 et 8, entrée embedding token 1 : **indices
  identiques** et scores à <1e-4 (test GPU
  `ling::tests::fused_router_matches_ling_weights` réussi). La disposition
  FP32 n'est donc pas grossièrement inversée. Ce test utilisait l'embedding
  brut, pas l'entrée MoE normalisée des couches en cours. Répétition
  diagnostique prévue : inspecter indices et écarts de score à chaque couche
  sur la vraie trajectoire du token imposé, sans nouvelle optimisation.
- Vraies entrées MoE des 23 couches sur deux tokens : les indices fusionnés
  et non fusionnés sont égaux **sur une même trajectoire fusionnée**, écarts
  des scores 7e-8 à 1,55e-6 (`fused_router_matches_decode_inputs`). Mais
  les deux trajectoires de modèle complètes divergent : après conversion
  correcte des bits FP16, token 1 logits max abs **0,015625**, moyenne abs
  **0,00206**, même argmax ; token 2 max abs **1,3046875**, moyenne abs
  **0,1957**, **argmax différent (220 vs 16)**. Le calcul initial `argmax`
  sur entiers bruts était erroné et est corrigé ici. Le routeur fusionné
  amplifie donc un changement numérique jusqu'au texte, ce qui viole la
  parité demandée. Décision : **rejeté**, ne pas benchmarker ni activer ;
  retirer ce kernel et ses champs/tests spécifiques du code de production.
- Retrait effectué de `router.rs` et `ling.rs` : plus de chemin routeur fusionné
  ni de champ supplémentaire. Le journal conserve la conclusion négative.
  `cargo fmt --check` à refaire après retrait, puis rebuild et contrôle hash
  forcé pour certifier le retour au moteur PERF-23.
- Rebuild release `mlx,chat` après retrait réussi ; `forward --tokens 1,2
  --states` revient exactement au SHA256 PERF-23
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`.
  Le routeur fusionné PERF-25 n'est donc plus présent dans le candidat local.

### OPT-2026-09-16-RUST-PERF-26 — Ling : sélection MoE parallèle sans fusion des logits — en cours

- Hypothèse : le routeur groupé de 128 experts exécute les scans top-groups et
  top-k dans un unique thread, avec une recherche `used` répétée. Répartir le
  calcul des scores par expert/groupe sur 128 threads, puis conserver la
  sélection finale ordonnée dans le thread 0, peut réduire le temps de route
  sans toucher à `x·W`, sigmoid, normalisation ni poids. Contrairement à
  PERF-25 rejeté, aucune modification de l'arithmétique des logits/scores.
- Baseline : binaire PERF-23 conservé `build/ling-perf-23-batch-bin`, hash
  `forward --tokens 1,2 --states` =
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c` ;
  84/128 tokens, decode chaud **55,31 tok/s** médian dans cinq tours, prefill
  chaud **569,95 tok/s** (thermique/alimentation non contrôlées). Test GPU
  existant des lignes du routeur, puis hash exact logits+états forcés,
  comparaison greedy et benchmark A/B/A rapproché seulement si ces contrôles
  réussissent. Preuves prévues `build/ling-perf-26-*`.
- Statut initial : **en cours**, code local seulement ; aucune publication.
- Kernel de sélection parallèle `experts=128` compilé ; quatre lignes donnent
  les mêmes indices/scores que quatre appels mono-ligne dans le test GPU
  `grouped_router_batch_matches_serial`. Cela ne compare pas encore le kernel
  antérieur, ni ne garantit le modèle entier. Rebuild `mlx,chat` et hash forcé
  requis avant benchmark. Avertissement `rust-objcopy/libLLVM` non bloquant.
- Contrôle modèle entier : SHA256 `forward --tokens 1,2 --states` **exactement
  identique** à la baseline (`9715bf...10b74a8c`). A/B/A rapproché, trois
  requêtes chaudes par binaire, prompt 84 tokens / 128 générés, même hash
  greedy sur les neuf requêtes : ancien A **54,918 tok/s decode**, **566,60
  tok/s prefill**, TTFT **148,47 ms** ; candidat B **75,128 tok/s decode**,
  **586,09 tok/s prefill**, TTFT **143,56 ms** ; ancien A2 **54,800 tok/s
  decode**, **568,99 tok/s prefill**, TTFT **147,82 ms**. Gain decode
  **+36,8–37,1 %** face aux deux contrôles. Le préfill +3–3,4 % est plus
  petit et peut contenir du bruit ; TTFT −3 % idem. Preuves
  `build/ling-perf-26-{control-a,candidate,control-a2}.json`.
- **Validé pour ce scénario**, code local uniquement, pas d'app installée ni
  publication. `peak_memory_gb` bridge identique 4,428 GB mais représente
  les poids, **pas** la RAM/pic processus. Tester plusieurs prompts/contexte
  plus long et vérifier la fidélité batch PERF-23 avant activation GUI.

### OPT-2026-09-17-RUST-PERF-27 — Ling : profil MoE après routeur parallèle — en cours

- Hypothèse de diagnostic : PERF-26 a retiré ~4,9 ms/token du routage sur le
  test de 84/128 tokens, mais le decode reste ~13,3 ms/token. Réexécuter le
  profil MoE de la couche 8 pour vérifier que le routage n'est plus le seul
  coût dominant avant tout nouveau changement. Même test GPU ignoré que
  PERF-24, cinq répétitions, temps après la première compilation ; barrières
  `eval()` artificielles, donc **microprofil non comparable au débit chat**.
- Baseline : PERF-24 route **0,553–0,560 ms**, experts **0,274–0,279 ms**,
  MLP partagé **0,194–0,205 ms** (avant PERF-26). Candidat : PERF-26,
  binaire `cargo test --release --locked --features mlx` avec le même
  checkpoint 4 bpw/M5. Aucune modification du code et pas de publication.
- Statut : **en cours**, mesure/log brut à conserver dans
  `build/ling-perf-27-moe-profile.log`.
- Profil effectué : routes des répétitions chaudes 0,365 / 0,365 / 0,659 /
  0,414 ms ; un pic scheduler/JIT à 0,659. Expert EXL3 ~0,285–0,335 ms
  hors pic, MLP partagé ~0,203–0,218 ms hors pic. La baisse de route contre
  ~0,55 ms de PERF-24 concorde avec le gain end-to-end PERF-26, sans en être
  une mesure équivalente. Le routeur reste coûteux et le scan top-k mono-thread
  pourrait encore être réduit ; les projections experts/partagées sont aussi
  une limite. Preuve brute `build/ling-perf-27-moe-profile.log`.
- **Diagnostic terminé**, aucune modification moteur, aucun push/app installée.

### OPT-2026-09-17-RUST-PERF-28 — Ling : top-k MoE par réductions SIMD — en cours

- Hypothèse : PERF-26 parallélise l'initialisation des 128 candidats mais
  laisse au thread 0 huit scans de 128 experts. Une réduction `simd_max` par
  groupe de 32 avec bris d'égalité par plus petit index, puis un scan de
  quatre groupes, doit préserver exactement l'ordre de sélection tout en
  réduisant le travail sériel. Garder sigmoid, matmul, normalisation et
  tous les buffers persistants inchangés. Applicable au seul routeur 128
  experts Ling ; fallback générique inchangé.
- Baseline : PERF-26, binaire actuel à archiver, 84/128 trois tours A/B/A,
  **75,128 tok/s decode**, prefill **586,09 tok/s**, TTFT **143,56 ms** ;
  hash forcé des logits+états `9715bf...10b74a8c`, neuf hashes greedy
  identiques. Vérifier test GPU routeur multi-lignes, hash forcé puis A/B/A.
  Si sortie non identique ou gain non reproductible, revenir à PERF-26.
  Conditions M5/MLX 0.32.2, alimentation/thermique non contrôlées ; preuves
  prévues `build/ling-perf-28-*`.
- Statut : **en cours**, code local, aucun push ni installation GUI.
- Test GPU routeur multi-lignes sur M5 **réussi** avec réduction SIMD ; sortie
  exacte face aux quatre appels mono-ligne du même candidat. Il faut encore
  comparer modèle entier à PERF-26 et mesurer A/B/A avant d'accepter.
- Modèle entier `forward --tokens 1,2 --states` SHA256 **identique** à
  PERF-26 (`9715bf...10b74a8c`). Premier bridge candidat 84/128, trois
  tours : **77,205 tok/s decode**, **591,51 tok/s prefill**, TTFT
  **142,57 ms**, même hash greedy. Face au PERF-26 historique 75,128 tok/s,
  le +2,8 % decode n'est **pas encore attribuable** sans A/B/A. Le fichier
  `build/ling-perf-26-bin` copié avant le test est invalide pour le bridge
  (sans feature `chat`, écrasé par `cargo test`) ; reconstruire explicitement
  ce binaire de contrôle, ne pas prétendre l'avoir mesuré.
- A/B/A après reconstruction correcte du binaire PERF-26, trois tours chauds
  chacun : A **76,629 tok/s decode**, **591,40 tok/s prefill**, TTFT
  **143,11 ms** ; candidat SIMD B **77,002 tok/s decode**, **590,33 tok/s
  prefill**, TTFT **142,56 ms** ; A2 **77,556 tok/s decode**, **586,82 tok/s
  prefill**, TTFT **143,36 ms**. Tous les hashes greedy identiques. Le
  candidat est entre A et A2 : **aucun gain reproductible**, le +2,8 %
  précédent venait des conditions. Preuves
  `build/ling-perf-28-control-{a,b,a2}.json`.
- Décision : **rejeté**. Revenir à la sélection PERF-26 plus simple, laisser
  le journal/preuves négatifs ; ne pas installer/publier PERF-28. RAM processus
  non mesurée ; mémoire `peak_memory_gb` du bridge n'est que poids résidents.

### OPT-2026-09-17-RUST-PERF-29 — Ling : normalisation des scores dans le routeur — en cours

- Hypothèse : après sélection des huit experts, `scores.sum` puis `div` et
  `scalar_mul` passent par MLX et créent des opérations GPU additionnelles
  à chaque couche MoE. Normaliser les scores FP32 dans le thread 0 du kernel
  de sélection pourrait économiser ces dispatchs, sans changer les experts,
  poids ou sampling. Risque connu : ordre de réduction FP32 différent de MLX
  donc perte de parité ; **rejet immédiat si hash forcé diverge**, même si la
  réponse greedy semble identique.
- Baseline : PERF-26, binaire `build/ling-perf-26-bin` reconstruit avec chat,
  **76,63–77,56 tok/s** decode dans A/B/A récent, hash forcé
  `9715bf...10b74a8c`. Vérifier d'abord test GPU routeur et modèle entier
  `forward --tokens 1,2 --states`; benchmark A/B/A seulement si exacte.
  M5/MLX 0.32.2 ; conditions thermiques non contrôlées. Preuves prévues
  `build/ling-perf-29-*`.
- Statut : **en cours**, code local, aucun push/app installée.
- Test GPU routeur multi-lignes réussi ; `forward --tokens 1,2 --states`
  SHA256 **exactement identique** à PERF-26 (`9715bf...10b74a8c`). L'ordre
  de réduction de huit scores dans ce cas retrouve donc la même sortie
  complète ; cela ne démontre pas tous les contextes/routes. Passer au
  benchmark A/B/A 84/128 avant décision. Preuves
  `build/ling-perf-29-router-test.log`, `build/ling-perf-29-forced.jsonl`.
- A/B/A, trois tours chauds chacun, même hash greedy : A **78,308 tok/s
  decode**, **596,93 tok/s prefill**, TTFT **140,97 ms** ; candidat B
  **78,280 tok/s decode**, **582,52 tok/s prefill**, TTFT **144,44 ms** ; A2
  **76,291 tok/s decode**, **594,26 tok/s prefill**, TTFT **141,66 ms**.
  Aucun gain decode reproductible, préfill ~2 % plus bas dans ce test.
  Preuves `build/ling-perf-29-{control-a,candidate,control-a2}.json`.
- Décision : **rejeté** malgré parité exacte sur deux tokens. Les opérations
  MLX séparées peuvent être masquées/fusionnées par le graphe ; la
  normalisation dans le kernel n'accélère pas le bridge. Revenir à PERF-26,
  ne pas publier/installer PERF-29.

### OPT-2026-09-17-RUST-PERF-30 — Ling : gate qualité du prefill de chat — en cours

- Hypothèse : les tests PERF-23 montrent un gain batch important mais une
  divergence greedy sur une demande de code. Le chat de production doit donc
  utiliser le chemin sériel de référence tant que cette fidélité n'est pas
  qualifiée, en conservant `forward --batch` explicite pour le diagnostic.
  Le routeur PERF-26, indépendant, reste actif et strictement paritaire.
- Changement minimal : `NativeChatModel::forward_many` dispatch Ling vers
  `forward_many_serial`; `Forward --batch` appelle directement
  `Ling::forward_tokens`. Aucun autre modèle, format, sampler ou poids
  modifié. Baseline qualité : binaire PERF-19 sériel et hash forcé
  `9715bf...10b74a8c`. Mesurer le chat normal 84/128 avec trois répétitions
  et comparer hash/température connue ; puis smoke test explicite `--batch`
  pour s'assurer que le prototype reste accessible. Preuves
  `build/ling-perf-30-*`. Préfill 500 tok/s **non revendiqué** pour le chat
  sûr ; TTFT/RAM mesurés séparément.
- Statut : **en cours**, code local, pas d'app installée ni publication.
- Revalidation physique M5 du 19 septembre : arrays MLX, codec Metal,
  Gated DeltaNet mono-token et batched, et routeur groupé mono/batch passent.
  Le test diagnostic Ling 25 tokens confirme en revanche la divergence connue
  du préfill batched (logits max abs **0,314**, état max abs **0,486**) ; ce
  chemin reste donc réservé à `forward --batch` et n'est pas utilisé par le
  chat/GUI. Le chemin de production sériel, comparé sur huit tokens à MLX-LM,
  conserve le même top-1 à chaque étape, avec max abs **0,3114** et KL max
  **0,002234**. Le vérificateur a été corrigé pour appliquer à Ling la borne
  numérique déjà utilisée pour les modèles non bit-exacts, au lieu d'exiger
  à tort une égalité FP16 bit-à-bit. Aucun débit n'a été remesuré dans cette
  revalidation.
- **Validé et intégré pour publication sur `main`** : gate sériel actif dans
  le chat/GUI, prototype batch explicitement hors chemin normal. L'app locale
  et une release ne sont pas modifiées par ce push.

### OPT-2026-09-19-RUST-PERF-31 — Campagne générale : baseline et profil Qwen complet — en cours

- Hypothèse de diagnostic : le moteur Rust actuel possède déjà les gains de
  synchronisation, QMV/QMM, routage et chargement documentés ci-dessus ; une
  nouvelle optimisation doit donc partir des coûts réellement dominants du
  modèle complet, et non répéter les réglages de command buffers, compilation
  FFN/QMV, gather/Hadamard ou top-k déjà rejetés.
- Cible initiale : Qwen3.6-35B-A3B EXL3 2,49 bpw, puis validation de toute
  piste générale sur Ling 3 Tiny et les autres architectures compatibles.
  Priorité : decode, prefill, TTFT, RAM, chargement, CPU/dispatch/allocations.
- Baseline prévue : binaire `main` `ce5ede9`, bridge natif, un warmup exclu,
  au moins cinq générations chaudes de 256 tokens avec prompt fixé, puis
  tailles de prefill séparées. Relever débit moteur/client, TTFT, tokens,
  hash du texte, mémoire MLX et RSS. Les comparaisons de candidats seront
  alternées A/B/A avec binaire de contrôle archivé et sorties forcées/parité.
- Profil prévu : trace Qwen synchronisée existante sur une passe mono-token et
  une passe multi-token, complétée par échantillonnage CPU du bridge. Les
  barrières de profil modifient le débit et ne seront utilisées que pour
  classer les sous-blocs. Instruments/xctrace et le compilateur Metal ne sont
  pas disponibles via le Command Line Tools actuellement sélectionné ; aucune
  durée GPU fine ne sera inventée.
- Conditions initiales : Apple M5, 24 Gio, Mac sur batterie (86 %, décharge),
  aucun autre moteur MLX/modèle détecté. Les variations thermiques/fréquence
  seront traitées par répétitions proches, sans additionner des gains issus de
  séries incompatibles. Résultats et classement des dix hotspots à ajouter
  avant toute modification du moteur.
- Statut : **en cours**, diagnostic local uniquement ; aucune optimisation,
  installation, publication ou revendication de gain à ce stade.
- Baseline exécutée : cinq runs chauds 27/256, decode médian **43,719 tok/s**
  (43,508–44,105), prefill chaud médian **167,21 tok/s**, TTFT chaud médian
  **161,97 ms**, hash identique sur les cinq sorties. Le premier run après
  warmup subit encore une compilation/JIT (TTFT 1,472 s). Chargement moteur
  5,421 s, modèle 13,064 GB annoncé ; `/usr/bin/time` ne comptabilise pas
  correctement le footprint unifié du processus enfant et n'est pas retenu
  comme mesure RSS. Preuves `build/perf-31/qwen-baseline.{json,time}`.
- Profil modèle complet : capture CPU `sample` de cinq secondes au milieu
  d'une génération 768 tokens. Le thread principal est dans `Qwen35Moe::
  run_tokens`/`array.eval` pour **2 673/3 082 échantillons (86,7 %)** ; au
  sein de cette phase, 961 échantillons attendent une condition GPU et 780
  construisent/soumettent le graphe. La sélection de token représente 61
  échantillons. Le profil est perturbant et ne donne pas les temps GPU purs.
  Preuves `build/perf-31/qwen-decode-cpu.sample` et
  `qwen-decode-sampled-run.json` (42,234 tok/s sous échantillonnage).
- Classement initial des dix coûts, fondé sur la capture et la structure réelle
  (40 couches = 30 GDN + 10 attention ; 256 experts, top-8), à confirmer pour
  chaque candidat : (1) exécution/attente du graphe GPU complet ; (2) QMV
  experts gate/up/down des 40 MoE ; (3) projections/état GDN des 30 couches ;
  (4) branche experte partagée des 40 MoE ; (5) `lm_head` 248 320 sorties ;
  (6) construction/soumission CPU des nombreux graphes/kernels ; (7) dix
  attentions et croissance KV ; (8) transformées Hadamard/scales/épilogues
  séparés ; (9) allocations/destructions/copies temporaires MLX observées par
  `sample` ; (10) sampler/synchronisation finale par token. Ce classement ne
  prétend pas répartir le temps GPU sans Instruments.
- Limites : batterie 86→83 %, aucune trace GPU fine (`xctrace`/`metal`
  absents du Command Line Tools sélectionné), aucun autre processus MLX.
  Statut : **diagnostic terminé**, aucun gain intégré par PERF-31.

### OPT-2026-09-19-RUST-PERF-32 — Greedy : argmax direct sans log-softmax — en cours

- Profil déclencheur : pendant cinq secondes de decode Qwen complet, le thread
  principal passe 2 673/3 082 échantillons dans l'évaluation MLX ; la sélection
  apparaît 61 fois et matérialise actuellement `log_probs` avant un `argmax`
  GPU. Le vocabulaire compte 248 320 entrées. Le chemin température/repetition
  penalty doit rester strictement inchangé.
- Hypothèse : pour `temperature=0` ou `top_k=1` sans pénalité, `argmax(logits)`
  évite le log-softmax complet. La transformation est monotone mais les égalités
  et non-finis imposent une validation sur le modèle réel ; rejet immédiat si
  un token/hash diffère.
- Baseline : PERF-31, cinq runs chauds 27/256 sur batterie : decode médian
  **43,719 tok/s** (43,508–44,105), prefill chaud **167,21 tok/s** médian hors
  premier JIT, TTFT chaud **161,97 ms** médian, hash de texte constant
  `55b6be28...d6a5bc6`. Chargement moteur 5,42 s, poids résidents annoncés
  13,064 GB. Preuve `build/perf-31/qwen-baseline.json`.
- Protocole candidat : archiver le binaire baseline, modifier uniquement le
  fast path greedy partagé du bridge, rebuild release, contrôler 256 tokens
  greedy/hash, puis A/B/A avec cinq runs de 256 tokens par binaire. Mesurer
  decode, prefill, TTFT et empreinte processus ; ne conserver que si le gain
  dépasse le bruit sans régression de fidélité.
- Statut : **en cours**, code non encore modifié.
- Candidat B1, contrôle A, candidat B2, cinq runs 27/256 chacun : decode
  médian **43,743 / 43,716 / 43,502 tok/s** ; prefill chaud médian
  **167,47 / 165,76 / 161,59 tok/s** ; TTFT chaud **161,41 / 163,22 /
  167,29 ms**. Les quinze générations ont le même hash que la baseline.
  L'empreinte poids annoncée reste 13,064 GB. Preuves
  `build/perf-31/qwen-argmax-{b1,a,b2}.json`.
- Le candidat ne dépasse pas le bruit et le second passage est plus lent avec
  la dérive batterie/thermique. Décision : **rejeté** ; fast path restauré au
  log-softmax de référence, aucun commit/push d'optimisation.

### OPT-2026-09-19-RUST-PERF-33 — Cache de factories Metal par identité de kernel — en cours

- Profil déclencheur : la capture PERF-31 observe, dans chaque token, des
  allocations/libérations de chaînes, comparaisons d'une clé C++ comprenant
  le source Metal complet, et créations répétées de descripteurs de custom
  kernel. La factory MLX est déjà cachée, mais sa recherche recopie et compare
  `name + input names + output names + header + source` à chaque dispatch.
- Hypothèse : les noms de kernels MLXL3 peuvent constituer l'identité de la
  factory si toutes les spécialisations dynamiques y figurent. Une recherche
  par nom court évite les copies/comparaisons du source sans changer le graphe,
  les poids, les dimensions de dispatch ou l'arithmétique GPU.
- Changement prévu : compléter les noms aujourd'hui ambigus (QMV groupé et
  QMM expert segmenté), puis indexer le cache C++ par nom. Aucun nouveau cache,
  thread ou dépendance. Vérifier par revue exhaustive des 11 sites de dispatch,
  tests Metal, parité forcée Qwen et hashes greedy ; benchmark A/B/A 27/256.
- Baseline de contrôle : binaire PERF-31 archivé SHA256 `3c25f59e...53d2cfac`,
  decode médian récent **43,716 tok/s**, prefill chaud **165,76 tok/s**, TTFT
  **163,22 ms** dans `qwen-argmax-a.json`. Conditions batterie, dérive connue.
- Statut : **en cours**, aucune modification appliquée à ce stade.
- Résultats A/B/A, cinq runs chauds 27/256 : candidat B1 **44,229 tok/s**
  decode, **167,84 tok/s** prefill, **161,09 ms** TTFT ; contrôle A
  **43,602 tok/s**, **161,97 tok/s**, **166,95 ms** ; candidat B2
  **43,619 tok/s**, **162,25 tok/s**, **166,66 ms**. Les quinze sorties ont
  le même hash. Le passage B2 ne reproduit pas le +1,4 % apparent de B1 et
  décroît avec la batterie/chauffe. Preuves
  `build/perf-31/qwen-factory-cache-{b1,a,b2}.json`.
- Contrôles : quatre tests Metal physiques réussis et parité Qwen forcée sur
  huit étapes bit-à-bit (`build/perf-31/qwen-factory-cache-parity-8.log`).
- Revue historique tardive : cette hypothèse est la même que
  `OPT-2026-09-14-RUST-PERF-06`, déjà rejetée à **-2,65 %**. La répétition
  n'aurait pas dû être lancée ; le rapprochement exact n'a été retrouvé
  qu'après le benchmark. Le nouveau résultat confirme l'absence de gain.
- Décision : **rejeté**, code restauré, aucun commit/push. Les mesures et cet
  écart au protocole restent consignés pour empêcher une nouvelle répétition.

### OPT-2026-09-19-RUST-PERF-34 — Qwen GDN : convolution causale mono-token — en cours

- Profil déclencheur : Qwen exécute 30 couches Gated DeltaNet par token. Chacune
  construit actuellement un `concatenate` état+QKV, lance une convolution
  depthwise de longueur 4, puis crée un `slice` pour le nouvel état avant le
  SiLU. Le roadmap `docs/decode-roadmap-2026-09-04.md` identifie précisément
  cette séquence comme prochaine fusion utile ; aucun essai historique du
  kernel convolution+état Rust n'a été trouvé.
- Hypothèse : pour `T=1`, un kernel Metal par canal calculant exactement la
  convolution FP16 et le décalage d'état remplace concat+conv+slice par un seul
  dispatch à deux sorties. Le SiLU MLX reste séparé au premier essai afin de
  réduire le risque numérique. Le chemin multi-token/prefill reste inchangé.
- Baseline : PERF-31, Qwen 2,49 bpw, prompt 27 / génération 256, decode médian
  **43,719 tok/s**, prefill chaud **167,21 tok/s**, TTFT chaud **161,97 ms**,
  hash `55b6be28...d6a5bc6`. Batterie et chauffe variables ; comparaison A/B/A
  de cinq runs avec binaires archivés et sortie identique requise.
- Protocole : test physique kernel contre `concatenate+conv1d` sur formes
  déterministes, `native/check_qwen_gdn.py` sur deux états réels, parité modèle
  forcée huit étapes incluant tous les caches, puis benchmark A/B/A. Rejet si
  un bit des sorties/états réels diverge ou si le gain ne se reproduit pas.
- Statut : **en cours**, aucun code candidat appliqué à ce stade.
- Contrôles du candidat : kernel synthétique exact contre MLX, GDN réelle
  couche 0 exacte sur deux états, puis modèle complet exact sur huit tokens
  imposés, logits et caches bit-à-bit. Preuves
  `build/perf-31/qwen-causal-conv-{gdn,parity-8}.log`.
- A/B/A, cinq runs 27/256 : candidat B1 **43,831 tok/s** decode,
  **168,02 tok/s** prefill, **160,98 ms** TTFT ; contrôle A
  **43,427 tok/s**, **163,49 tok/s**, **167,34 ms** ; candidat B2
  **43,164 tok/s**, **165,85 tok/s**, **163,66 ms**. Les quinze sorties ont
  le hash de référence. Batterie 78→76 %, dérive thermique visible.
- Le +0,9 % apparent face au contrôle n'est pas reproduit par B2, qui termine
  sous A et sous la baseline initiale. La réduction des opérations du graphe
  n'abaisse donc pas le temps modèle mesurable dans ces conditions.
- Décision : **rejeté**, code et test candidat retirés, aucun commit/push.
  Preuves `build/perf-31/qwen-causal-conv-{b1,a,b2}.json`.

### OPT-2026-09-19-RUST-PERF-35 — Splash/Inco et adressage incrémental QMV — rejeté

- Recherche déclenchée par l'utilisateur : analyse du moteur Apple Silicon
  Splash/Inco (dépôt Apache-2.0 et billet technique, état du 17 septembre
  2026). Son avantage annoncé repose notamment sur DFlash2/speculative decode,
  exclu de cette campagne car l'utilisateur exige un décodage exact sans
  spéculation. Les idées transférables sans perte sont : plans Metal fixes par
  forme, poids déjà disposés pour leur consommateur, arènes préallouées, et
  fusion complète du Gated DeltaNet. Leur kernel GDN fusionne convolution,
  SiLU, normalisations q/k, gates, récurrence et normalisation/gate de sortie ;
  cela explique pourquoi le seul sous-kernel convolution de PERF-34 n'a pas
  produit de gain modèle. Le format Q4 privé et le command graph Splash ne sont
  pas directement réutilisables par un moteur EXL3 construit sur les graphes
  MLX, et une réécriture de runtime n'est pas engagée silencieusement.
- Premier candidat général et réversible avant la fusion GDN : dans les QMV
  dense et expert mappé, calculer la base de la tuile K une seule fois par
  itération puis l'avancer d'un stride constant. Le kernel actuel répète
  `(tile_k * TILES_N + tile_n) * PACKED_U32` pour chaque tuile de sortie ; le
  changement conserve codewords, ordre des FMA, réduction, dispatch et poids.
  Aucun repacking, nouvelle copie ni allocation.
- Baseline : binaire archivé PERF-31, Qwen3.6-35B-A3B 2,49 bpw, prompt 27 et
  génération 256 : decode médian initial **43,719 tok/s**. La batterie et la
  température dérivent ; décision uniquement sur une alternance B/A/B récente
  d'au moins cinq runs, texte/hash identique, plus tests QMV et parité modèle.
- Protocole : vérifier d'abord les kernels Metal physiques et huit tokens Qwen
  forcés, puis B/A/B. Rejeter si l'écart ne se reproduit pas ou reste dans le
  bruit. Si rejeté, passer à la fusion GDN complète indiquée par Splash, sans
  conserver ce micro-changement.
- Statut : **en cours**, journalisé avant modification ; aucune publication ni
  revendication de gain.
- Le candidat passe la parité Qwen complète sur huit étapes bit-à-bit. Mesures
  longues 27/512, cinq runs par passage : candidat B1 **41,410 tok/s**,
  contrôle A1 **40,841 tok/s**, candidat B2 **41,175 tok/s**, puis contrôle A2
  **42,533 tok/s**. Les vingt sorties ont le même hash ; pic MLX identique
  13,064 GB. Le prefill chaud est respectivement **156,79 / 157,22 / 153,92 /
  162,57 tok/s** et ne montre aucun gain. Batterie 76→70 %, dérive thermique
  et énergétique importante. Preuves `build/perf-31/qwen-qmv-address-*.json`
  et `qwen-qmv-address-parity-8.log`.
- Le dernier contrôle dépasse le candidat de 3,3 % : les +1,39/+0,82 % vus
  autour d'A1 ne sont pas reproductibles et ne peuvent pas être attribués au
  code. Décision : **rejeté**, shaders restaurés, aucun commit/push.

### OPT-2026-09-19-RUST-PERF-36 — Qwen GDN : invariants persistants — rejeté

- Profil et inspiration Splash : la part CPU de PERF-31 montre beaucoup de
  construction/destruction de graphes et d'arrays ; Splash alloue et prépare
  ses ressources fixes au chargement. Dans chaque appel mono-token de chacune
  des 30 couches GDN Qwen, MLXL3 recrée actuellement trois scalaires MLX
  (`1/Dk`, `1/sqrt(Dk)`, zéro) et reconstruit `exp(A_log)` alors que ces quatre
  valeurs sont invariantes pour toute la vie du modèle.
- Hypothèse : matérialiser une seule fois au chargement les trois scalaires et
  `-exp(A_log)`, puis les réutiliser, supprime 90 petites allocations/objets de
  graphe par token et le calcul invariant des 30 vecteurs A. Le signe est
  déplacé avant la multiplication, opération exactement équivalente sur ces
  poids finis ; aucun ordre de réduction, poids, cache ou kernel EXL3 ne change.
- Protocole : parité GDN deux états puis modèle Qwen huit étapes bit-à-bit ;
  benchmark B/A/B 27/512 avec cinq runs et contrôle archivé. Mesurer decode,
  prefill, TTFT et pic. Rejeter si le gain ne se reproduit pas ; ce candidat
  reste distinct de la future fusion Metal GDN.
- Baseline récente pertinente : contrôle A2 PERF-35 **42,533 tok/s** decode,
  prefill chaud **162,57 tok/s**, TTFT **168,20 ms**, pic 13,064 GB, batterie
  70 % en décharge. Statut : **en cours**, aucune modification appliquée avant
  cette entrée.
- Parité : GDN couche 0 sur deux états, puis modèle complet huit étapes,
  sorties et caches **bit-à-bit**. Mesures B/A/B 27/512, cinq runs : decode
  **49,696 / 48,563 / 48,320 tok/s** ; prefill chaud **187,34 / 186,82 /
  186,47 tok/s** ; TTFT **144,68 / 146,42 / 146,40 ms** ; pic identique
  13,064 GB et hash identique. Batterie 70→66 %, forte remontée globale de
  fréquence par rapport à PERF-35 puis dérive dans la série. Preuves
  `build/perf-31/qwen-gdn-invariants-{b1,a,b2}.json`, `*-gdn.log` et
  `*-parity-8.log`.
- B1 paraît +2,33 %, mais B2 est **−0,50 %** face au contrôle central : le gain
  n'est pas reproduit. Décision : **rejeté**, code restauré ; les allocations
  minuscules/invariants sont vraisemblablement masqués ou déjà fusionnés par
  MLX. Aucun commit/push.

### OPT-2026-09-19-RUST-PERF-37 — Qwen GDN : gates Metal fusionnées — rejeté

- Étape minimale vers la fusion complète observée dans Splash : remplacer les
  deux graphes élémentaires séparés `sigmoid(b)` et
  `exp(-exp(A_log) * softplus(a + dt_bias))` par un kernel Metal mono-token à
  deux sorties (`beta` FP16 et decay FP32). Les projections, la convolution,
  les normalisations, la récurrence et les caches restent inchangés.
- Hypothèse : 30 couches GDN par token peuvent éviter au moins une frontière de
  dispatch/compilation chacune. Le calcul doit reproduire les arrondis FP16 de
  `a + dt_bias` puis `logaddexp`; aucune approximation `fast`, table ou
  spéculation. Le chemin multi-token garde le graphe MLX actuel.
- Protocole : test GDN réel deux états exigeant sortie et caches bit-à-bit,
  puis modèle Qwen huit étapes bit-à-bit. Si la formule MSL ne reproduit pas la
  référence, rejet immédiat ou correction avant tout benchmark. Si exacte,
  B/A/B 27/512 cinq runs ; pic, TTFT et prefill suivis.
- Baseline de proximité : contrôle PERF-36 **48,563 tok/s**, prefill chaud
  **186,82 tok/s**, TTFT **146,42 ms**, pic 13,064 GB ; batterie 66 % en
  décharge. Statut : **en cours**, journalisé avant code.
- La première formule MSL a été rejetée avant benchmark : le `sigmoid` calculé
  en FP32 différait d'un ULP FP16 sur plusieurs entrées et faisait diverger la
  GDN réelle. La correction reproduit l'opérateur MLX : exponentielle et
  branche stable en `half`, ainsi que son `logaddexp` et son `log1p` compensé.
  Le test GPU synthétique passe alors bit-à-bit, de même que la GDN réelle sur
  deux états et le modèle sur huit étapes.
- B/A/B 27/512, cinq runs : decode **44,673 / 44,187 / 44,122 tok/s** ; prefill
  chaud **168,75 / 166,32 / 166,74 tok/s** ; TTFT **164,56 / 162,92 /
  164,90 ms** ; pic identique 13,064 GB et tous les hashes identiques. Batterie
  66→61 %. Preuves `build/perf-31/qwen-gdn-gates-{b1,a,b2}.json`,
  `qwen-gdn-gates-gdn.log`, `qwen-gdn-gates-parity-8.log`.
- B1 est +1,10 %, mais B2 est **−0,15 %** et le TTFT n'est pas amélioré : un
  dispatch Metal supplémentaire remplace des expressions que MLX fusionne déjà
  efficacement. Décision : **rejeté**, kernel/test retirés, aucun commit/push.

### OPT-2026-09-19-RUST-PERF-38 — Qwen GDN : gates intégrées à la récurrence — validé

- Suite directe de PERF-37 : ne plus produire `g` et `beta` dans un kernel
  séparé. Pour le decode `T=1`, le kernel récurrent existant reçoit directement
  `a`, `b`, `A_log` et `dt_bias` et calcule les deux scalaires du head avant la
  mise à jour d'état. La formule half/FP32 exacte validée dans PERF-37 est
  réutilisée ; la géométrie et l'ordre des FMA récurrentes restent inchangés.
- Hypothèse : supprimer réellement le dispatch et les deux buffers
  intermédiaires sur chacune des 30 couches peut rendre visible le bénéfice que
  PERF-37 masquait. Le calcul des deux scalaires est répété par row-group, coût
  minuscule face aux 16 384 valeurs d'état du head et sans synchronisation
  inter-groupe.
- Protocole : test dédié des gates contre le graphe MLX, GDN deux états et
  modèle huit étapes bit-à-bit, puis B/A/B 27/512 cinq runs. Le prefill `T>1`
  garde le chemin de référence. Statut : **en cours**, aucune revendication.
- Contrôles numériques initiaux réussis : test GPU du kernel intégré contre le
  graphe de gates + kernel récurrent, GDN réelle couche 0 sur deux états, puis
  modèle Qwen complet huit étapes ; sorties et caches bit-à-bit. Le chemin
  trace et le prefill multi-token continuent explicitement d'utiliser les
  opérations MLX de référence.
- B/A/B 27/512, cinq runs : decode **44,698 / 43,972 / 44,576 tok/s**, soit
  **+1,65 % / +1,37 %** face au contrôle central ; prefill chaud **169,42 /
  163,18 / 168,05 tok/s** ; TTFT **161,04 / 166,76 / 162,76 ms** ; pic identique
  **13,064 GB**, tous les hashes identiques. Batterie 61→57 % en décharge ; les
  deux passages candidat encadrant le contrôle reproduisent le gain malgré la
  dérive. Preuves `build/perf-31/qwen-gdn-integrated-{b1,a,b2}.json`,
  `qwen-gdn-integrated-{gdn,parity-8}.log`.
- Décision performance : **validé** pour Qwen GDN decode. Vérifications
  complètes : test GPU dédié, GDN réelle sur deux états et modèle Qwen huit
  étapes bit-à-bit ; 190 cas de parité MLX exacts ; 35 tests Rust réussis
  (12 tests matériels explicitement ignorés par la suite standard) ; format,
  Clippy strict et `git diff --check` réussis. Kani 0.68 / CBMC 6.11 vérifie
  **14/14 harnesses**, zéro échec, sur le cœur Rust sans features MLX. Kani ne
  couvre pas Metal, MLX ni leur FFI : ces chemins sont validés par les tests
  différentiels physiques précédents, pas formellement prouvés. Preuve Kani
  `build/perf-31/qwen-gdn-integrated-kani.log`. Intégration : code inclus dans
  le commit d'optimisation dédié et destiné à `main` ; application installée et
  release non modifiées dans cet essai.

### OPT-2026-09-19-RUST-PERF-39 — Ling KDA : beta intégré à la récurrence — validé

- Suite générale de PERF-38 et de l'analyse Splash/Inco : Ling matérialise
  encore le vecteur de decay FP32 et le beta FP16 avant son kernel récurrent
  vectoriel. Le candidat transmet au même kernel les sorties brutes déjà
  calculées et les constantes `A_log`, `dt_bias` et `lower_bound`, puis y
  reproduit exactement les transformations élémentaires. Les projections,
  poids EXL3, caches, ordre des FMA et sampling restent inchangés.
- Hypothèse : sur les couches KDA, retirer les graphes/buffers intermédiaires
  de sigmoid, cast et decay réduit les dispatches de decode. Le matmul de beta
  reste nécessaire et n'est pas fusionné. Le prefill multi-token conserve le
  chemin de référence tant que l'égalité numérique n'est pas établie.
- Baseline : binaire `b7de3f8` archivé avant modification, modèle
  `Ling-3.0-tiny-EXL3-4bpw`. Protocole prévu : sortie et états Ling bit-à-bit
  sur tokens imposés, test GPU synthétique, puis A/B/A d'au moins cinq runs
  sur la même longueur ; decode, prefill, TTFT, pic et hashes suivis. Rejet à
  la moindre divergence inexpliquée ou si le gain ne se reproduit pas.
- Conditions : Apple M5 24 Go, sur batterie et température dérivante ; niveau
  de batterie et alternance consignés. Statut : **en cours**, journalisé avant
  baseline, modification et benchmark ; aucune publication.
- Revue avant modification : contrairement à Qwen, le decay Ling contient 128
  valeurs par head et est partagé par 128 lignes d'état. Le recalculer dans la
  géométrie actuelle répéterait ses exponentielles 32 à 128 fois. Pour éviter
  cette régression prévisible, ce premier candidat n'intègre que le sigmoid et
  cast FP16 de beta, scalaire par head ; le decay reste matérialisé une fois.
  Une fusion decay correcte nécessitera une géométrie/threadgroup distincte et
  fera l'objet d'un essai séparé. Baseline A mesurée : **97,672 tok/s** decode,
  prefill chaud médian **100,080 tok/s**, TTFT médian **839,5 ms**, pic déclaré
  4,428 GB, hashes identiques ; batterie 55 %. Preuve
  `build/perf-39/ling-gates-control-a.json`.
- Candidat : le kernel vectoriel reçoit les logits beta FP32, reproduit le
  sigmoid MLX puis son cast FP16 avant la récurrence. Test GPU synthétique :
  sortie FP16 et état FP32 bit-à-bit contre le graphe de référence. Sur huit
  tokens imposés du modèle réel, logits complets baseline/candidat identiques,
  SHA256 commun
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`.
- Alternance A/B/A/B, cinq runs 84/128 : decode **97,672 / 102,008 / 100,373 /
  101,697 tok/s**. Les deux candidats sont à **+4,44 % / +1,32 %** face au
  contrôle précédent et restent groupés autour de 102 tok/s. Prefill chaud
  **100,079 / 104,652 / 102,916 / 104,589 tok/s** ; TTFT **839,53 / 802,88 /
  816,43 / 803,41 ms**. Pic déclaré identique 4,428371 GB et hash de génération
  identique sur les vingt runs. Batterie 55→49 %, en décharge. Preuves
  `build/perf-39/ling-gates-{control-a,candidate-b1,control-a2,candidate-b2}.json`.
- Décision : **validé** pour le decode Ling. Suite : 20 tests lib, 1 test CLI
  et 14 tests contractuels réussis ; 13 tests matériels ignorés par la suite
  standard, dont le nouveau test exécuté séparément avec succès. Format,
  Clippy strict et `git diff --check` réussis. Kani 0.68 / CBMC 6.11 : **14/14
  harnesses**, zéro échec, cœur Rust sans features MLX. Kani ne vérifie pas le
  shader, MLX ou leur FFI ; leur contrôle est différentiel sur GPU physique.
  Preuve `build/perf-39/ling-gates-kani.log`. Intégration : commit dédié pour
  `main`, sans rebuild de l'app ni release dans cet essai.

### OPT-2026-09-19-RUST-PERF-40 — Ling KDA : bloc récurrent partagé/fusionné — rejeté

- Suite de PERF-39 et de l'architecture Splash/Inco : la géométrie actuelle
  relit q/k pour chacune des 128 lignes d'état. Le candidat decode `T=1`
  charge q/k une fois en mémoire threadgroup, calcule les 128 decay une fois
  par groupe à partir du gate brut, puis chaque thread traite plusieurs lignes
  avec le même ordre FP32 de réduction et de mise à jour. Beta brut reste
  intégré comme dans PERF-39. Huit groupes par head sont prévus afin de garder
  de l'occupation sans répéter 128 fois les transcendantes.
- `exp(A_log)` sera matérialisé une fois au chargement et réutilisé. Le chemin
  multi-token reste inchangé. Aucun poids, quantification, cache, sampler ou
  approximation mathématique n'est modifié ; les opérateurs Metal doivent
  reproduire exactement sigmoid/exp MLX.
- Baseline : commit `4c87e3d`, à archiver avant modification ; dernier A/B/A/B
  PERF-39 candidat **101,697–102,008 tok/s** decode, prefill chaud
  **104,589–104,652 tok/s**, TTFT **802,88–803,41 ms**, pic 4,428371 GB.
- Protocole : test GPU synthétique sortie/état bit-à-bit, huit tokens Ling
  imposés contre le binaire archivé, puis A/B/A/B 84/128 à cinq runs. Rejet
  immédiat pour divergence ou gain non reproduit. Apple M5 24 Go sur batterie,
  température non instrumentée. Statut : **en cours**, journalisé avant code ;
  aucune publication.
- Premier prototype full-decay : test nul puis état non nul synthétique
  réussis, mais le modèle réel diverge dans l'état récurrent dès le deuxième
  token (couche 0, 4 578 floats, erreur max 3,73e-9), puis dans les logits au
  troisième token. Cause : les élémentaires du decay fusionné ne reproduisent
  pas tous les arrondis des kernels MLX séparés. Décision : variante
  **rejetée avant benchmark** malgré son faible écart ; les transcendantes et
  le `A_log` pré-évalué sont retirés.
- Variante corrigée en cours : conserver le decay MLX exact, mais utiliser la
  nouvelle géométrie partagée pour charger q/k/decay huit fois par head au lieu
  de 128, tout en gardant beta fusionné. Ce changement isolera le gain mémoire
  de la réorganisation sans modifier le calcul des gates.
- La variante corrigée est bit-à-bit exacte sur le test GPU avec état non nul
  et sur huit tokens réels (SHA256 logits identique
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`).
  Mesures B/A/B, cinq runs : decode **98,970 / 99,972 / 99,908 tok/s**,
  prefill chaud **99,522 / 103,228 / 101,841 tok/s**, TTFT **844,31 / 813,92 /
  825,03 ms**. Un run candidat à 89,50 tok/s est un outlier, mais même sans
  lui le second candidat reste au niveau du contrôle, pas au-dessus. Batterie
  43 %, en décharge. Preuves `build/perf-40/ling-shared-*.json`.
- Décision finale : **rejeté**. La baisse des lectures q/k ne compense pas la
  perte d'occupation due aux boucles de lignes ; code partagé et test retirés.
  PERF-39 reste le chemin production. Aucun commit/push de ce prototype.

### OPT-2026-09-19-RUST-PERF-41 — Ling KDA : pré-évaluation de A — rejeté

- Variante minimale issue du plan fixe Splash, distincte du full-decay rejeté
  en PERF-40 : calculer `exp(A_log)` une fois au chargement, puis fournir
  exactement cet array FP32 au graphe MLX de decay inchangé. Aucun élémentaire
  sigmoid/exp n'est déplacé dans Metal et la récurrence reste celle de PERF-39.
- PERF-36 a déjà rejeté une idée voisine sur Qwen ; Ling diffère car son decay
  vectoriel est construit dans chaque couche KDA. Ce nouvel essai ne sera gardé
  que s'il est bit-à-bit et reproduit un gain A/B/A ou B/A/B.
- Baseline : binaire production `4c87e3d`, mesures récentes PERF-40 contrôle
  **99,972 tok/s** decode, **103,228 tok/s** prefill chaud, **813,92 ms** TTFT.
  Protocole 84/128 cinq runs, huit tokens imposés, pic et hashes suivis. Apple
  M5 24 Go sur batterie 43 %. Statut : **en cours**, journalisé avant code.
- Parité : huit tokens imposés bit-à-bit, hash commun
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`.
  B/A/B cinq runs : decode **98,814 / 102,873 / 97,558 tok/s**, prefill chaud
  **100,934 / 105,524 / 99,210 tok/s**, TTFT **832,43 / 796,23 / 847,01 ms**,
  pic identique 4,428371 GB. Batterie 41 %. Preuves
  `build/perf-41/ling-ascale-*.json`.
- Décision : **rejeté** et code retiré. L'évaluation anticipée brise la fusion
  paresseuse de la chaîne élémentaire MLX et ajoute une lecture intermédiaire ;
  la suppression d'un `exp` apparent régresse de 3,9 à 5,2 %. Aucun push.

### OPT-2026-09-19-RUST-PERF-42 — Qwen GDN : gates partagées par threadgroup — rejeté

- Suite de PERF-38 : dans le kernel récurrent Qwen `T=1`, chaque thread
  recalcule actuellement le même sigmoid beta et le même decay scalaire du
  head. Avec 64 threads par groupe et huit groupes par head, cela répète les
  transcendantes 512 fois. Le candidat les calcule une fois par threadgroup,
  les place dans 6 octets de mémoire partagée puis synchronise avant la boucle
  récurrente. Équations, arrondis et ordre des FMA restent identiques.
- Hypothèse : le coût d'une barrière est inférieur aux exponentielles répétées.
  Le chemin non fusionné et le prefill restent inchangés. Baseline production
  commit `4c87e3d`, binaire archivé
  `build/perf-40/ling-beta-fused-baseline-bin`; mesures fraîches à faire car
  batterie 41 % et température non instrumentée.
- Protocole : test kernel/GDN et huit tokens Qwen bit-à-bit, puis B/A/B 27/512
  cinq runs ; decode, prefill, TTFT, pic et hashes. Rejet si la barrière
  régresse ou si le gain ne se reproduit pas. Statut : **en cours**, journalisé
  avant modification ; aucune publication.
- Parité GPU et huit tokens Qwen bit-à-bit, hash commun
  `c06b13baecf5d4b0eb189418e974f63a5259a8732a16d29badb47b85dd69d373`.
  B/A/B cinq runs : decode **45,721 / 46,973 / 44,945 tok/s**, prefill chaud
  **173,488 / 181,680 / 172,943 tok/s**, TTFT **155,85 / 148,83 / 156,36 ms**,
  pic identique 13,064372 GB. Batterie 39→36 %. Preuves
  `build/perf-42/qwen-shared-gates-*.json`.
- Décision : **rejeté**, shader restauré. La barrière threadgroup coûte plus
  que les calculs redondants sur M5 et dégrade aussi TTFT/prefill. Aucun push.

### OPT-2026-09-19-RUST-PERF-43 — Qwen GDN : gates diffusées dans le SIMDgroup — rejeté

- Révision motivée par PERF-42 : supprimer la barrière coûteuse. Seule la lane
  0 de chacun des deux SIMDgroups calcule beta/decay, puis
  `simd_broadcast_first` diffuse leurs bits aux 31 autres lanes. Cela réduit
  les transcendantes 32×, sans mémoire threadgroup ni synchronisation entre
  SIMDgroups ; chaque groupe traite déjà des lignes indépendantes avec les
  mêmes gates.
- Baseline production `4c87e3d`, contrôle frais PERF-42 **46,973 tok/s**,
  prefill **181,680 tok/s**, TTFT **148,83 ms**, pic 13,064372 GB ; batterie
  36 %, dérive forte donc B/A/B obligatoire. Parité kernel/GDN et huit tokens
  Qwen bit-à-bit avant benchmark. Statut : **en cours**, journalisé avant code.
- Parité kernel et huit tokens bit-à-bit, hash commun
  `c06b13baecf5d4b0eb189418e974f63a5259a8732a16d29badb47b85dd69d373`.
  B/A/B cinq runs : decode **45,767 / 45,910 / 44,306 tok/s**, prefill chaud
  **178,343 / 178,054 / 174,429 tok/s**, TTFT **151,59 / 151,85 / 155,01 ms**,
  pic identique 13,064372 GB. Batterie 32 % puis secteur reconnecté en fin de
  série ; forte dérive sur B2. Preuves `build/perf-43/qwen-simd-gates-*.json`.
- Décision : **rejeté**, shader restauré. B1 est dans le bruit du contrôle et
  B2 régresse ; même sans barrière, masquer les transcendantes aux lanes
  inactives n'apporte pas de gain modèle. Aucun push.

### OPT-2026-09-19-RUST-PERF-44 — Ling KDA : boucle de lignes sans barrière — rejeté

- Révision de PERF-40 selon les résultats PERF-42/43 : garder le decay MLX
  exact et supprimer toute mémoire/barrière threadgroup. Pour `T=1`, chaque
  SIMDgroup charge q/k/decay une fois dans ses registres puis traite plusieurs
  lignes d'état. Beta est calculé par lane 0 et diffusé avec
  `simd_broadcast_first`. La grille conserve huit threadgroups par head, soit
  assez d'occupation, et réduit les lectures q/k/decay de 4×.
- Baseline production `4c87e3d`, dernier contrôle Ling **102,873 tok/s** decode,
  **105,524 tok/s** prefill chaud, **796,23 ms** TTFT ; secteur maintenant
  attaché, batterie 32 % non chargée. Test état non nul et huit tokens exacts,
  puis B/A/B cinq runs. Statut : **en cours**, journalisé avant code.
- Test GPU état non nul et huit tokens réels bit-à-bit, hash commun
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`.
  B/A/B cinq runs : decode **100,715 / 102,305 / 98,984 tok/s**, prefill chaud
  **102,347 / 106,027 / 100,027 tok/s**, TTFT **820,96 / 792,45 / 839,98 ms**,
  pic identique 4,428371 GB. Secteur attaché puis batterie en charge 32→36 %.
  Preuves `build/perf-44/ling-looped-*.json`.
- Décision : **rejeté**, code retiré. Réduire les threadgroups et boucler les
  lignes perd davantage en occupation qu'il ne gagne en lectures répétées.
  PERF-39 reste la meilleure géométrie Ling. Aucun push du prototype.

### Clôture de la passe Splash/Inco — 2026-09-19

- Conservé et publié : PERF-38, gates Qwen intégrées au kernel récurrent,
  commit `b7de3f8`, **+1,37 à +1,65 % decode** ; PERF-39, beta Ling intégré,
  commit `4c87e3d`, **+1,32 à +4,44 % decode**. Sorties forcées exactes et pic
  déclaré inchangé pour les deux.
- Rejeté et absent du code final : adressage QMV, invariants pré-évalués,
  kernel gates séparé, decay Ling fusionné, géométries partagées/bouclées et
  diffusion Qwen. Les mesures PERF-35–44 empêchent de les retester sans
  nouvelle géométrie ou nouveau matériel.
- Le test PERF-39 est renforcé avec un état récurrent non nul. Vérification
  finale : 20 tests lib, 1 test CLI et 14 contrats réussis ; test GPU ciblé
  réussi ; format, Clippy strict et `git diff --check` réussis. Kani 0.68 /
  CBMC 6.11 : **14/14 harnesses**, zéro échec, cœur Rust sans features MLX.
  Kani ne couvre toujours ni MLX, ni Metal, ni leur FFI ; les kernels sont
  contrôlés par différentiel physique, pas formellement prouvés. Preuve
  `build/perf-44/final-kani.log`.

### OPT-2026-09-19-RUST-PERF-45 — DFlash 2 Qwen3.6-35B-A3B — en cours

- Demande : intégrer le drafter DFlash 2 officiel à la cible locale
  `Qwen3.6-35B-A3B-EXL3-2.49bpw`, sans modifier la distribution du modèle
  cible, puis viser **100 tok/s** en decode. Le package Apache-2.0
  `incoai/Qwen3.6-35B-A3B-Splash` contient un drafter Q4 de six couches
  (capture des couches cible 1/6/11/16/22/27/32/37), sept propositions et une
  vérification cible de huit lignes.
- Baseline production pertinente : PERF-42/43 mesure le moteur Qwen actuel à
  **44,3–47,0 tok/s** decode, **172,9–181,7 tok/s** prefill chaud et
  **148,8–156,4 ms** TTFT sur Apple M5 24 Go, avec forte dérive batterie et
  thermique. Une nouvelle baseline secteur/thermique stabilisée sera mesurée
  avant toute revendication DFlash.
- Blocages confirmés avant code : le chemin EXL3 natif accepte aujourd'hui
  seulement `M=1` ou `M>=24`, tandis que DFlash vérifie `M=8`; Qwen ne calcule
  le `lm_head` que sur la dernière ligne et fait avancer immédiatement ses 30
  états GDN/convolution et ses 10 caches KV. Le drafter officiel est fourni en
  format fixe `splash-packed-q4-moe`, pas en EXL3/safetensors.
- Plan d'essais indépendants : (1) contrôleur d'acceptation exact et borné,
  vérifié par tests/Kani ; (2) QMM EXL3 `M=8` et logits cible par ligne, avec
  parité autoregressive ; (3) états Qwen transactionnels commit/rollback ;
  (4) chargeur et exécution du drafter Q4 officiel ; (5) boucle greedy exacte,
  puis rejection sampling exact ; (6) A/B/A long avec taux d'acceptation,
  coût draft/verify/commit, decode, prefill, TTFT, pic et hash de sortie.
- Garde-fous : chaque jalon doit rester inactif ou revenir au decode normal si
  le drafter manque ; aucune mesure Splash M5 Pro n'est reprise comme résultat
  MLXL3. Le jalon est rejeté à la moindre divergence greedy, distribution
  sampling invalide, corruption d'état ou régression stable. Statut initial :
  **en cours**, aucun code DFlash publié et aucun gain revendiqué.
- Jalon 1 : contrôleur greedy exact ajouté, sans branchement au moteur. Il
  accepte uniquement le plus long préfixe identique aux choix de la cible et
  termine toujours le cycle par le token cible suivant ; les longueurs
  incohérentes sont rejetées. Test Rust ciblé réussi, Clippy strict réussi.
  Kani 0.68 / CBMC 6.11 vérifie exhaustivement les longueurs 0 à 7 et des
  tokens `u32` symboliques : préfixe maximal, borne d'acceptation et token
  final cible, **216 propriétés réussies**, 2/2 couvertures atteintes, aucun
  échec (une branche standard inaccessible). Ce contrôle ne prouve ni Metal,
  ni les logits, ni les caches ; ceux-ci restent à implémenter et valider.

### OPT-2026-09-19-RUST-PERF-46 — Qwen : transaction d'état DFlash — en cours

- Hypothèse : les arrays MLX sont fonctionnels et leur clonage duplique le
  handle, pas les données. Une transaction légère peut donc capturer l'offset,
  les 30 couples convolution/récurrence GDN et les 10 couples KV, puis les
  restaurer sans copie GPU. C'est requis pour rejeter un suffixe spéculatif
  sans laisser le modèle cible dans un état futur invalide.
- Changement prévu : un snapshot opaque et typé, avec validation stricte du
  nombre/type de couches à la restauration. Aucun branchement au decode normal
  et aucune allocation de tenseur supplémentaire hors clonage de handles.
- Baseline fonctionnelle : après un préfixe fixé, `snapshot → tokens d'essai →
  restore → mêmes tokens` doit produire exactement les mêmes logits et états
  que le premier passage ; un snapshot d'un autre modèle/état doit être rejeté.
  Baseline performance : decode Qwen PERF-42/43 **44,3–47,0 tok/s** ; ce jalon
  ne doit pas modifier ce chemin et aucun gain de decode n'est revendiqué.
- Protocole : test réel Qwen sur quelques tokens avec comparaison bit-à-bit des
  logits et de chaque état restauré ; tests négatifs de structure ; build,
  Clippy strict et Kani sur les invariants Rust qui n'appellent pas MLX. Le coût
  snapshot/restore sera mesuré séparément avant intégration. Statut : **en
  cours**, journalisé avant code ; aucune publication.
- Résultat : **validé et intégré comme primitive inactive**. Sur le vrai
  Qwen3.6-35B-A3B EXL3 2.49 bpw, après le préfixe `[1,2,3]`, une branche
  `[4,5]`, un rollback puis le rejeu produisent exactement les mêmes logits
  FP16 aux deux pas et les mêmes octets pour les **80 tenseurs** conv/GDN/KV.
  Le test GPU ciblé réussit en 11,97 s, chargement du modèle compris.
- Vérification : format et Clippy strict réussis ; tests Rust complets et E2E
  Desktop réussis lors du jalon de nettoyage adjacent. Kani 0.68 / CBMC 6.11
  vérifie 15/15 harnesses, zéro échec (`build/dflash-cleanup-kani.log`), mais
  ne compile pas le feature MLX : le clonage de handles, Metal et ce rollback
  sont donc validés par différentiel physique, pas formellement prouvés.
- Le chemin autoregressif existant n'appelle jamais `snapshot`/`restore` : coût
  normal nul. La mesure microsecondes et le coût sous spéculation seront faits
  avec le contrôleur complet, afin de ne pas présenter un timing isolé comme
  un gain de decode. Publication prévue dans le jalon DFlash suivant.

### OPT-2026-09-19-RUST-PERF-47 — EXL3 : vérification cible `M=8` — en cours

- Hypothèse : le QMM TensorOps EXL3 existant, aujourd'hui réservé à `M>=24`,
  peut traiter les huit lignes de vérification DFlash en un graphe. Le padding
  à 32 lignes gaspille 75 % du calcul, mais évite d'abord un nouveau décodeur
  de poids et fournit une référence mesurable avant d'écrire un QMV batch.
- Baseline fonctionnelle : sur une projection réelle du Qwen3.6-35B-A3B
  EXL3 2.49 bpw, comparer `forward(M=8)` aux huit appels `M=1` : dimensions,
  valeurs FP16, argmax et timing chaud. Le contrôle final exigera aussi les
  mêmes tokens et le même état en exécution autoregressive complète.
- Protocole : autoriser temporairement `M=8` dans le QMM, ajouter un test GPU
  ignoré et mesurer plusieurs répétitions après warmup. **Rejet immédiat** si
  le QMM n'est pas plus rapide ou si l'écart numérique change les tokens ; si
  l'écart FP16 existe sans changer les tokens, il restera uniquement une
  référence de performance et ne sera pas intégré au chemin lossless.
- Baseline modèle : PERF-42/43, **44,3–47,0 tok/s** decode, **172,9–181,7
  tok/s** prefill et **148,8–156,4 ms** TTFT, avec dérive batterie/thermique.
  Statut : **en cours**, journalisé avant code ; aucune publication.
- Sous-essai TensorOps sur `layers.0.linear_attn.in_proj_qkv`, entrée
  déterministe `[8,2048]`, deux warmups puis cinq répétitions : huit QMV
  série **2,055 ms**, QMM paddé à 32 **0,680 ms**, soit **3,02×** sur cette
  projection. Mais **555/65 536** sorties FP16 diffèrent, écart absolu maximal
  **0,0014648438**. Le QMM est donc **rejeté comme vérificateur lossless** ; il
  reste une référence de plafond et n'est pas branché au chemin production.
- Prochain sous-essai : QMV `M<=8` à lignes parallèles, même shader, même
  partition de K et même réduction par ligne afin de viser la parité bit-à-bit.
  Mesurer projection puis tokens/états complets avant intégration. Statut global
  PERF-47 : **en cours**.
- Sous-essai QMV exact sur la même projection et la même entrée, deux warmups
  puis cinq répétitions : batch **0,751 ms**, référence huit branches QMV
  **1,478 ms**, soit **1,97×** dans ce microbenchmark, avec **0/65 536** valeur
  FP16 différente et écart maximal nul. Le résultat peut inclure du cache et
  l'ordonnancement MLX ; ce n'est pas encore un gain modèle.
- Étape suivante journalisée avant code : exposer les huit lignes de logits du
  passage Qwen déjà vectorisé, puis comparer batch contre huit pas séquentiels,
  logits FP16 et 80 tenseurs d'état compris. Un écart du GDN/SDPA invalidera le
  chemin comme vérification exacte ou imposera un kernel séquentiel équivalent.
- Résultat modèle du batch vectorisé : **rejeté**. Après le même préfixe de
  trois tokens, huit pas séquentiels prennent **162,503 ms**, contre **425,245
  ms** en batch (`0,38×`). **1 913 011/1 986 560** logits FP16 diffèrent
  (écart absolu max 0,34179688) et **72/80** tenseurs d'état diffèrent. Les
  opérations GDN/SDPA multi-token n'ont pas l'arrondi exact du chemin `M=1`,
  et le lm_head/MoE batch est ici plus coûteux. Le chemin n'est pas exposé.
- Sous-essai suivant, journalisé avant code : construire les huit pas dans
  l'ordre autoregressif exact avec les kernels `M=1`, sans `eval` entre les
  tokens, concaténer les logits puis synchroniser une seule fois. Attendu :
  parité bit-à-bit par construction et gain limité aux synchronisations et à
  l'ordonnancement du graphe. Si le graphe grossit ou régresse, le rejeter.
- Résultat du graphe autoregressif différé sur le modèle réel : **validé comme
  primitive lossless**. Huit pas séquentiels synchronisés prennent **179,207
  ms**, contre **153,829 ms** avec une seule synchronisation, soit **1,16×**.
  Les **1 986 560** logits FP16 et les **80/80** tenseurs d'état sont identiques
  bit-à-bit. Ce résultat ponctuel correspond à ~52 vérifications/s et ne
  revendique pas encore un débit DFlash complet : draft, sélection, acceptation
  et commit ne sont pas branchés.
- Le fallback QMV `2<=M<24` est également exact sur la projection test
  (**0/65 536** différence) ; il sert aux futurs kernels batch, mais le chemin
  target retenu ici reste token-par-token différé pour préserver l'arithmétique
  GDN/SDPA.
- Vérification du jalon : format et Clippy strict réussis ; **21** tests lib,
  **1** test CLI et **14** contrats réussis. Les deux tests GPU ciblés réussissent
  sur M5. Kani 0.68 / CBMC 6.11 vérifie **15/15 harnesses**, zéro échec et 2/2
  couvertures (`build/perf-47-kani.log`). Kani est exécuté sans le feature MLX :
  il ne prouve ni les graphes MLX, ni Metal, ni les états Qwen ; leur parité est
  couverte par les différentiels physiques bornés décrits ci-dessus. Statut :
  **validé et intégré comme primitive inactive**.

### OPT-2026-09-19-RUST-PERF-48 — DFlash 2 : package Splash et Q4 — en cours

- Hypothèse : réutiliser le format public Apache-2.0 et les géométries exactes
  de `incoai/Qwen3.6-35B-A3B-Splash` évite de reconvertir ou réentraîner le
  drafter. Seul le sous-répertoire `draft/` est requis ; les poids cible Q4 de
  Splash ne seront pas téléchargés ni utilisés par la cible EXL3 MLXL3.
- Géométrie issue du manifeste officiel : 6 couches, hidden 2048, dynamique
  512, QKV 6144, attention 4096, MLP 6144, vocabulaire 248320, projection de
  contexte 16384→2048 et sélecteur rank 256. Q4 affine, groupes de 64,
  StorageN=256. Taille déclarée : six fichiers de 34 324 480 octets et un
  `model.bin` de 273 498 112 octets, soit ~457 MiB de poids draft.
- Changement prévu : téléchargement ciblé, chargeur Rust strict (magic
  `MDFD0004`, layer/type, alignement 16 KiB, arithmétique vérifiée, consommation
  exacte), puis port minimal des kernels Q4 officiels avant le graphe draft.
- Protocole : refuser header, taille, offset ou section incorrect ; comparer les
  offsets calculés à `layout.json` ; test réel du package, tests négatifs
  synthétiques et Kani pour l'arithmétique pure. Aucun débit n'est revendiqué
  avant que le drafter produise et que la cible accepte des tokens. Statut :
  **en cours**, journalisé avant code.
- Jalon chargeur : **validé**. Le lecteur Rust refuse les mauvais magic/type/id,
  les fichiers non réguliers, toute taille inattendue, tout dépassement et tout
  offset non aligné. Les offsets calculés correspondent au manifeste publié
  (`layer qkv=638 976`, `layer down=27 246 592`, codebooks modèle
  `19 218 432/146 358 272`) et les sept fichiers réels du package local sont
  acceptés ; le runtime ne dépend d'aucun poids cible Splash.
- Contrôles : tests synthétiques et package réel réussis, Clippy strict avec
  tous les features réussi. Kani 0.68 / CBMC 6.11 vérifie les 11 propriétés
  d'alignement sur tout `u64` et les 49 propriétés de géométrie/taille Q4 sur
  tout couple `u16`, avec 2/2 couvertures atteintes. Un ICE Kani causé par
  `is_multiple_of` a été éliminé en conservant l'équivalent `%`, puis les deux
  harnesses ont réussi. La passe complète vérifie **17/17 harnesses**, zéro
  échec (un chemin standard inaccessible), dont 162 propriétés pour le plus
  gros contrat et 2/2 couvertures. Cela ne vérifie ni les octets de poids, ni
  Metal.
- Étape suivante, journalisée avant essai : mapper une projection Q4 réelle et
  comparer sur M5 plusieurs kernels compatibles (Splash MPP servant de
  référence, kernel MLXL3 retenu s'il gagne), d'abord en exactitude puis en
  temps chaud. Aucun gain d'inférence n'est encore revendiqué. Statut global :
  **en cours**, chargeur prêt à intégrer.

### OPT-2026-09-19-RUST-PERF-49 — DFlash Q4 M=8 : sélection kernel M5 — en cours

- Hypothèse : le MPP TensorOps Q4 affine, groupe 64 et StorageN=256 est une
  bonne référence pour les huit lignes du draft, mais sa tuile et son nombre de
  groupes persistants ne sont pas supposés optimaux pour le M5 testé. Comparer
  au minimum N128 séquentiel, N128 pipeliné et N256, avec plusieurs grilles.
- Projection réelle : `draft/layer-0.bin:qkv`, forme 2048→6144, huit entrées
  BF16 déterministes. Baseline externe : kernel Splash ; baseline MLXL3 : aucun
  kernel DFlash Q4 avant cet essai. Deux warmups de compilation puis au moins
  cinq mesures chaudes synchronisées par candidat, ordre alterné si la durée le
  permet. Conditions secteur/thermique consignées au résultat.
- Validité : toutes les variantes retenues doivent donner les mêmes octets BF16
  que la référence MPP séquentielle sur le tenseur complet ; un contrôle CPU
  indépendant sur un sous-ensemble vérifiera aussi le décodage affine Q4. Une
  variante divergente ou plus lente est rejetée et ne reste pas dans le chemin
  production. Mesurer séparément compilation/chargement et exécution chaude.
- Ce microbenchmark ne prédit pas encore le débit DFlash complet : attention,
  convolutions, sélecteur, vocabulaire et acceptation restent absents. Statut :
  **en cours**, journalisé avant code.
- Essai de compilation 1 : **rejeté/corrigé avant mesure**. Les constantes de
  forme étaient émises après le corps helper dans le header MLX, donc Metal ne
  pouvait pas résoudre `DFLASH_INPUT`. Aucun kernel n'a été exécuté et aucune
  mesure n'est issue de cet essai. Les `#define` sont désormais placés avant le
  helper ; le nouveau build doit encore être validé.
- Premier passage GPU réel, secteur, M5 10 cœurs/Metal 4 : tous les candidats
  N128/N128-pipeliné/N256 rendent exactement les mêmes **98 304 octets BF16**.
  Le contrôle CPU indépendant sur 512 sorties distingue bien l'ordre des
  nibbles : erreur basse `(max=0, moyenne=0)` contre ordre inversé
  `(max=6,9648438, moyenne=1,636845)`.
- Les chronos ne sont pas encore stabilisés : le premier processus a mesuré
  N128 g48 **0,807 ms**, pipeliné g48 **0,776 ms**, N256 g24 **1,136 ms** ; le
  rejeu chaud immédiatement après donne respectivement **0,367/0,337/0,326
  ms**. Cette dérive dépasse les écarts entre variantes : résultat
  **non concluant** pour le choix final. La prochaine passe alternera l'ordre,
  augmentera le warmup et mesurera plusieurs cycles A/B/A avant intégration.
- Passe alternée, 100 warmups puis 30 échantillons/candidat, répétée deux fois
  sur secteur : N128-pipeliné g40/g48 tient **0,290–0,297 ms**, N128 g40
  **0,294–0,296 ms**, N256 g24 **0,309–0,310 ms**. Le pipelinage apporte donc
  seulement ~1,7–2,0 % sur cette projection ; g40 et g48 sont dans le bruit.
  Les sorties restent identiques. Prochain essai journalisé : enlever les tests
  `is_valid_element` internes lorsque la capacité coopérative couvre exactement
  les 8×N éléments, tout en gardant une variante gardée comme oracle. Comparer
  N128 pipeliné et N256, mêmes 30 échantillons ; rejet si un octet change.
- Essai traversal direct : **rejeté**. Sur le vrai QKV, la suppression des
  gardes provoque un `kIOGPUCommandBufferCallbackErrorPageFault` avant toute
  mesure : la capacité coopérative contient bien des emplacements invalides
  avec ce compilateur/descriptor. Aucune sortie ni performance n'est attribuée
  à cette variante. Les variantes `Fast` sont retirées ; le kernel conservé
  continue d'appeler `is_valid_element`, comme le fallback sûr publié.
- Revalidation après retrait, même protocole : N128-pipeliné g48 **0,290 ms**
  médiane (p10 0,265, p90 0,319), N128 g48 **0,298 ms**, N128 g40 **0,298
  ms**, N256 g24 **0,306 ms**. Sur trois passes chaudes, le choix pipeliné g48
  reste entre **0,289 et 0,291 ms**, soit ~2–3 % devant le N128 gardé non
  pipeliné et ~5–7 % devant N256 g24 ; 98 304/98 304 octets restent identiques.
  **Validé pour cette forme 2048→6144**, sans extrapoler aux autres formes.
- Décision : conserver les trois implémentations sûres pour l'autotuning, avec
  N128-pipeliné/full-grid comme choix provisoire de la forme QKV M5. La mesure
  est un microbenchmark de projection, pas encore un gain de decode du modèle.
  Statut PERF-49 : **validé et prêt à intégrer**, publication après contrôles
  Rust/Metal complets.
- Contrôles finaux du jalon : Clippy strict tous targets/features ; 23 tests
  lib, 1 CLI et 14 contrats réussis. Le test GPU ignoré exécute le QKV réel,
  le contrôle CPU et 15 géométries Metal ; toutes les variantes conservées
  sont bit-à-bit identiques. Kani vérifie séparément 11 propriétés d'alignement
  et 49 propriétés Q4 avec 2/2 couvertures ; il ne couvre pas MLX/Metal, validés
  ici uniquement par différentiel physique. Statut d'intégration : code local
  validé, push du jalon suivant.

### OPT-2026-09-19-RUST-PERF-50 — DFlash 2 : graphe draft MLXL3 — en cours

- Hypothèse : conserver le format et l'arithmétique du drafter Splash, mais
  sélectionner séparément chaque primitive sur M5, permet d'obtenir un graphe
  plus rapide sans lier MLXL3 à l'ordonnanceur Splash. Une primitive MLX native
  ou MLXL3 sera retenue seulement si elle bat le kernel Splash à sortie BF16
  identique ; Splash reste l'oracle de compatibilité, pas une contrainte
  d'implémentation.
- Premier sous-essai : charger strictement les treize sections de chacune des
  six couches et les sections globales, puis implémenter les deux phases de la
  convolution dynamique 8×2048. Comparer le kernel Metal au calcul CPU BF16
  indépendant sur toutes les 16 384 sorties, balayer les groupes persistants et
  mesurer après warmup sur le M5. Rejet de toute variante qui diverge ou plante.
- Baseline utile : Q4 QKV 2048→6144 M=8 validé dans PERF-49 à **0,289–0,291
  ms** médiane pour N128 pipeliné g48. Le débit DFlash complet et le taux
  d'acceptation restent **non mesurés** ; il est interdit d'extrapoler ce timing
  isolé à des tok/s.
- Étapes suivantes déjà bornées : Q/K RMS+RoPE, attention glissante 2048,
  gate/up+SwiGLU fusionné, six couches, sélecteur, puis boucle strictement
  lossless verify/rollback. Chaque alternative sera comparée sur le même graphe
  et le même état. Statut : **en cours**, journalisé avant code ; aucune
  publication de ce jalon.
- Chargeur complet : **validé** sur les 457 MiB officiels. Les six couches et
  les 13 sections par couche, les projections globales, les deux normes et les
  codebooks 248320×256 sont chargés avec les formes attendues en **0,14 s** de
  test (processus complet **0,23 s**). `/usr/bin/time -l` rapporte un RSS max de
  **502 349 824 octets** ; la mesure mélange mappings et runtime de test et ne
  constitue pas encore la RAM incrémentale de l'app.
- Première exécution convolution : **corrigée avant timing**. Le grid MLX avait
  été exprimé en groupes au lieu de threads (`groups×256`) : seules quelques
  sorties étaient écrites. Le différentiel exhaustif l'a détecté ; aucune
  performance n'est attribuée à ce lancement.
- Convolution corrigée : les phases prepare et residual, pour g8/12/16/24/32/
  48/64, donnent les mêmes **32 768 octets BF16** que la référence CPU
  indépendante (16 384 sorties chacune). Après 100 warmups, 50 échantillons :
  médianes **183,375–185,834 µs** ; g24 est nominalement premier à **183,375
  µs**, mais tout l'intervalle est du bruit. Décision provisoire : g24, sans
  supprimer les autres possibilités avant le benchmark du graphe enchaîné.
- Les valeurs ci-dessus incluent `eval()` et la synchronisation par primitive ;
  le vrai graphe draft différé doit amortir ce coût. Étape suivante : Q/K
  RMS+RoPE et attention avec état, puis comparaison primitive MLX versus kernel
  spécialisé. Statut PERF-50 : **en cours**, chargeur et convolution validés
  localement, pas encore publiés.
- Premier graphe des six couches, contexte vide : **validé fonctionnellement**.
  Il enchaîne normes, projections Q4, convolutions, Q/K RMS+RoPE, SDPA GQA,
  gate/up+SwiGLU, down, norme finale et sélecteur. Deux exécutions donnent des
  sorties hidden 8×2048 et selector 8×256 identiques bit-à-bit, sans BF16 non
  fini. Après 20 warmups et 30 mesures : médiane **4,621 ms**, p10 **4,559
  ms**, p90 **4,898 ms** pour les six couches et le sélecteur, hors lm_head,
  sélection, vérification cible, acceptation et commit.
- Diagnostic : ce draft n'est déjà plus le facteur limitant. La vérification
  cible lossless PERF-47 prend **153,829 ms / 8 positions** ; même sept drafts
  tous acceptés donnent un plafond d'environ **50,5 tok/s** avant les autres
  frais. Atteindre 100 tok/s exige donc d'abord une vérification cible M=8 sous
  ~75 ms, sans l'écart numérique du QMM paddé rejeté. Prochain essai : kernel
  Qwen M=8 séquentiel-fusé/streamé qui conserve l'ordre M=1 et supprime les
  relectures et dispatchs inutiles ; le taux d'acceptation sera mesuré seulement
  après la boucle complète. Statut : **en cours**.
- Contrôles du jalon : format et Clippy strict tous targets/features réussis ;
  23 tests lib, 1 test CLI et 14 contrats réussis. Les trois tests GPU réels
  ignorés par défaut valident le chargement officiel, la convolution exhaustive
  et le graphe six couches. Kani 0.68 vérifie **17/17 harnesses**, 0 échec ; il
  couvre le lecteur/planificateur pur mais pas MLX ni Metal, contrôlés ici par
  les différentiels et exécutions physiques. Décision : publier ce jalon draft
  mesuré, tout en gardant PERF-50 **en cours** jusqu'à la boucle lossless et au
  débit DFlash end-to-end.
- Révision du 20 septembre, avant nouvel essai : l'audit externe fourni et la
  lecture directe du kernel Splash `draft_attention_split_phase` montrent que
  les huit K/V courants sont ajoutés sans masque causal entre lignes. MLXL3
  utilise encore SDPA causal à contexte vide et masque les lignes courantes
  futures avec cache. Sous-essai E01 : rendre les huit lignes courantes visibles
  tout en conservant la fenêtre causale uniquement sur l'historique, puis laisser
  un test où la valeur de la ligne 7 doit modifier la sortie de la ligne 0. Le
  débit ne sera interprété qu'après ce contrôle de fidélité. État : **en cours**.
- E01 reproduit puis corrigé : le test qui ne modifie que les V de la ligne 7
  échouait avec le SDPA causal (la sortie de la ligne 0 restait identique), puis
  réussit après passage du bloc courant en non-causal. Avec historique, seules
  les colonnes antérieures respectent la borne glissante ; les huit colonnes
  courantes sont visibles pour chaque requête, comme dans le kernel Splash lu.
  Le graphe six couches reste déterministe et fini ; timing ponctuel médian
  **5,390 ms**, p10 **4,567**, p90 **5,525**. La dispersion interdit d'attribuer
  ici une régression par rapport aux 4,621 ms précédents. Fidélité Splash des
  tenseurs complets et cas anneau restent à qualifier avant de clore E01.
- Extension du même test avec un cache historique d'une position : échec avant
  l'oracle, car le masque FP32 ne peut pas promouvoir la sortie SDPA BF16. Le
  cache draft n'avait donc pas de chemin exécutable validé. Correction minimale
  à la frontière commune : caster le masque additif en BF16 avant SDPA ; relance
  du test causal/non-causal requise avant toute mesure avec contexte.
- Après correction du dtype, le test physique réussit à contexte vide et avec
  une position historique : modifier uniquement V à la ligne courante 7 modifie
  bien la sortie de la ligne 0 dans les deux cas. Cela valide le défaut initial,
  sa correction et l'exécution du cache court ; la parité numérique Splash et
  le wrap à 2048 restent non mesurés.

### OPT-2026-09-19-RUST-PERF-51 — Vérification cible M=8 exacte — en cours

- Hypothèse : la vérification lossless actuelle calcule les huit `lm_head`
  EXL3 comme huit QMV indépendants. Un QMV Metal M=2..8 qui ajoute seulement
  l'indice de ligne à la géométrie existante peut partager l'ordonnancement et
  améliorer la localité des poids sans changer l'ordre arithmétique interne de
  chaque ligne. Le reste du modèle demeure strictement autoregressif M=1.
- Baseline : Qwen3.6-35B-A3B EXL3 2,49 bpw, tokens cibles 4..11 après le préfixe
  1,2,3, Apple M5 ; vérification différée exacte PERF-47 **153,829 ms / 8**.
  Batterie et thermique du nouveau passage seront relevées ; aucun gain n'est
  revendiqué avant une série alternée.
- Protocole : comparer chaque logit FP16 et chaque état cible au chemin huit
  forwards M=1, puis mesurer au moins cinq passages chauds. Rejeter au premier
  écart. La distribution cible, les tokens acceptés et le rollback ne doivent
  pas changer. État : **en cours**, journalisé avant code.
- Première compilation du prototype interrompue correctement : le shader QMV
  partagé référençait `INPUT_DIMS`, absent du header du chemin M=1. Aucun timing
  n'a été retenu ; le define a été ajouté aux deux géométries avant relance.
- Projection réelle Qwen 2048→8192, K=4, M=8 : QMV ligne par ligne **1,872 ms**,
  QMV à grille 2D **0,573 ms**, soit **3,27×** sur ce microbenchmark ; 0/65 536
  sorties FP16 différentes, écart max 0. Le kernel conserve une accumulation
  indépendante et le même ordre par ligne ; seul l'indice de batch est ajouté.
- Vérificateur complet, série appariée ABBA sur batterie : ancien graphe différé
  médian **168,334 ms** (166,859–173,020), tête M=8 partagée **159,950 ms**
  (156,112–161,647), soit **1,052×**. Les 12 passages conservent exactement les
  1 986 560 logits FP16 et les 80 états ; les QMV M=2/4/8/16/23 sont aussi
  identiques ligne par ligne. Décision : conserver ce premier partage exact ;
  il améliore V d'environ 5 % ici, loin du budget 45–75 ms visé. E06/E10 restent
  nécessaires. K=7 reste volontairement sur le fallback série faute de fixture
  locale permettant de qualifier le kernel multi-lignes.
- Contrôles finaux du jalon : `cargo fmt --all -- --check`, Clippy strict
  tous targets/features, 23 tests lib, 1 test CLI, 14 tests de contrats et le
  build release MLX/chat réussissent. Kani 0.68/CBMC 6.11 vérifie **17/17
  harnesses**, 0 échec et 2/2 couvertures ; il ne couvre pas MLX/Metal. Les
  chemins GPU sont donc validés séparément sur le M5 par les différentiels QMV
  M=2/4/8/16/23, le replay exact logits+état Qwen, le test de visibilité
  DFlash vide+cache et le graphe draft complet déterministe/fini. Statut :
  **validé et prêt à intégrer** pour le partage exact du `lm_head` et les
  corrections d'attention ; la vérification cible layer-major et le débit
  DFlash end-to-end restent non implémentés.

### OPT-2026-09-20-RUST-PERF-52 — Vérification exacte couche-major — en cours

- Hypothèse issue de l'audit fourni, recoupée avec le code : le vérificateur
  exact avance encore chaque token dans les 40 couches avant le suivant. Sans
  modifier un kernel, avancer les huit tenseurs mono-token dans une couche
  avant de passer à la suivante conserve les récurrences GDN/KV propres à
  chaque couche et rapproche les lectures d'un même jeu de poids.
- Baseline appariée la plus récente, Qwen3.6-35B-A3B EXL3 2,49 bpw, préfixe
  `[1,2,3]`, vérification `[4..11]`, batterie 100 % : chemin token-major avec
  tête M=8 **159,950 ms** médiane (156,112–161,647), logits FP16 et 80 états
  identiques à l'ancien chemin différé.
- Prototype minimal prévu : garder chaque opération sensible en M=1 et le même
  ordre temporel *dans chaque couche* ; seule la boucle tokens/couches est
  transposée. Comparaison ABBA d'au moins six passages par variante, de chaque
  logit FP16 et des 80 états. Rejet au premier octet différent ou si le gain
  n'est pas stable. Aucun débit DFlash end-to-end ne sera extrapolé. Statut :
  **en cours**, journalisé avant code.
- Première commande interrompue avant compilation : lancée depuis `native/`,
  elle a résolu `MLXL3_MLX_ROOT` sous `native/.venv` au lieu de la racine.
  Aucun kernel ni timing n'a été exécuté. La relance utilise le chemin absolu
  du runtime MLX du dépôt.
- Première série ABBA, six mesures par variante, batterie 100 % : token-major
  médian **161,909 ms** (143,916–162,827), couche-major **149,137 ms**
  (141,490–171,693), soit **1,086×**. Les 12 passages produisent exactement les
  mêmes **1 986 560 logits FP16** et les mêmes **80 états**. Le signal est
  positif mais les plages se recouvrent et un outlier couche-major existe ; une
  seconde série indépendante est requise avant intégration.
- Seconde série indépendante : token-major médian **144,678 ms**
  (137,309–146,584), couche-major **143,745 ms** (140,090–158,833), soit
  seulement **1,006×**. L'identité complète reste vérifiée, mais le gain du
  simple réordonnancement n'est pas stable ; il n'est pas suffisant seul.
- Sous-essai suivant, journalisé avant code : `ProjectionBundle::Grouped`,
  utilisé par QKV et gate/up, sérialise encore chaque ligne malgré le QMV exact
  multi-lignes de PERF-51. Étendre le shader mappé avec un axe de lignes doit
  grouper ces dispatchs sans changer l'accumulation de chaque sortie. Comparer
  toutes les sorties FP16 du bundle à huit appels M=1, puis le modèle complet
  couche-major au token-major. Rejet au premier écart ; le réordonnancement
  couche-major sera retiré si le bundle groupé n'apporte pas un gain stable.
- Microbenchmark réel couche 0, bundle QKV+Z 2048→(6144+4096), M=8 : huit
  appels groupés M=1 **1,584 ms**, nouvel axe de lignes **0,804 ms**, soit
  **1,97×**. Les deux sorties, **81 920 valeurs FP16**, sont identiques bit à
  bit. Ce résultat justifie le test modèle complet mais ne constitue pas encore
  un gain du vérificateur.
- Intégration expérimentale suivante : uniquement dans les 30 couches GDN,
  batcher QKV/Z/A/B et la projection de sortie avec les QMV exacts, tout en
  exécutant convolution et récurrence token par token dans leur ordre canonique.
  Les normes, résiduels et MoE restent mono-token. Cette frontière minimale
  isole le partage des poids ; elle sera comparée au token-major sur logits et
  80 états avant toute extension aux dix couches d'attention.
- Première exécution arrêtée par la validation de forme avant mesure : les
  sorties SwiGLU concaténées gardaient `[1,8,Hv,Dv]` au lieu d'être repliées en
  `[1,8,Hv×Dv]` pour `out_proj`. Aucun timing ni résultat numérique n'est
  attribué à ce passage ; le reshape identique au chemin canonique est ajouté.
- Première série modèle après correction, six passages ABBA par variante :
  token-major médian **145,190 ms** (140,824–165,353), couche-major avec
  projections GDN groupées **113,266 ms** (110,779–116,259), soit **1,282×**.
  Les 12 passages conservent exactement les 1 986 560 logits FP16 et les 80
  états. Le signal est net ; une seconde série indépendante doit confirmer la
  stabilité avant extension ou intégration.
- Seconde série indépendante : token-major médian **137,759 ms**
  (135,110–138,697), candidat **108,866 ms** (107,657–111,032), soit
  **1,265×**. Les intervalles ne se recouvrent pas et l'identité logits+états
  est de nouveau complète. Le gain est **validé pour M=8** ; les largeurs
  1/2/4 restent à contrôler avant intégration.
- Contrôle des largeurs **M=1/2/4/8** réussi sur le modèle réel : pour chaque
  largeur, tous les logits FP16 et les 80 tenseurs d'état correspondent octet
  pour octet au chemin token-major. Le candidat peut passer aux contrôles
  complets ; cela reste une vérification physique bornée, pas une preuve de
  tous les prompts ni de Metal.
- Contrôles du jalon : format, Clippy strict tous targets/features, 23 tests
  lib, 1 CLI, 14 contrats et build release MLX/chat réussis. Kani 0.68/CBMC
  6.11 vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures ; il ne
  compile pas MLX/Metal. Les nouveaux axes Metal et la transposition des états
  sont donc couverts par les différentiels physiques M=1/2/4/8 et les deux
  séries ABBA, pas par Kani. Statut : **validé et prêt à intégrer** ; coût cible
  M=8 ramené de 137,759 à 108,866 ms dans la série indépendante, encore au-dessus
  du budget ~75 ms nécessaire à 100 tok/s même avec acceptation parfaite.

### OPT-2026-09-20-RUST-PERF-53 — Attention cible : projections exactes M=8 — en cours

- Suite de PERF-52 : les dix couches d'attention exécutent encore Q/Gate/K/V et
  `o_proj` huit fois. Hypothèse : utiliser les QMV groupés multi-lignes pour ces
  seules projections, puis avancer RoPE, append KV et SDPA causal en M=1 dans
  l'ordre canonique, partage les poids sans modifier le masque ni la réduction
  d'attention.
- Baseline indépendante : vérificateur PERF-52 **108,866 ms** médian
  (107,657–111,032), contre token-major 137,759 ms, batterie. Protocole : série
  ABBA, logits FP16 et 80 états octet par octet, M=1/2/4/8. Rejet à tout écart
  ou si le gain n'est pas reproductible. Les MoE restent mono-token afin
  d'isoler la projection d'attention. Statut : **en cours**, journalisé avant
  code.
- Exactitude physique M=1/2/4/8 : tous les logits FP16 et les 80 états restent
  identiques au token-major. Première série ABBA M=8 : token-major médian
  **141,809 ms** (139,389–145,197), candidat GDN+attention **104,353 ms**
  (103,519–105,943), soit **1,359×** face à la référence du même processus.
  Par rapport aux 108,866 ms indépendants de PERF-52, l'attention apporte
  environ 4,5 ms supplémentaires ; répétition indépendante requise.
- Répétition indépendante : token-major médian **143,504 ms**
  (139,091–145,417), candidat **104,722 ms** (103,746–107,606), soit
  **1,370×**, toujours strictement identique. Les projections d'attention sont
  donc **validées** ; le coût cible reste toutefois ~30 ms au-dessus du budget
  optimiste de 75 ms.
- Contrôles finaux : le différentiel physique M=1/2/4/8 réussit après retrait
  du prototype MoE divergent. Format, Clippy strict, 23 tests lib, 1 test CLI,
  14 contrats et build release MLX/chat réussissent. Kani 0.68 / CBMC 6.11
  vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures ; il ne couvre pas
  MLX/Metal, validés séparément par le différentiel bit à bit. Statut :
  **validé et intégré**, prêt à publier.

### OPT-2026-09-20-RUST-PERF-54 — MoE exact multi-lignes du vérificateur — en cours

- Observation : après PERF-53, les normes et résiduels sont déjà disponibles
  couche par couche, mais chaque MoE traite encore huit tokens séparément. Pour
  M=8, le chemin existant `Exl3SwitchGlu` reste sous le seuil TensorOps 64 et
  utilise le même QMV mappé par route ; le batch peut donc mutualiser dispatchs,
  routeur et projections partagées sans changer l'accumulation des experts.
- Baseline : candidat PERF-53 **104,722 ms** médian, référence token-major
  **143,504 ms**. Essai : calculer chaque RMSNorm post-attention séparément,
  concaténer seulement ses huit lignes pour `Mlp::forward`, puis restaurer les
  résiduels dans l'ordre. Contrôler M=1/2/4/8, logits et 80 états bit à bit,
  puis deux séries ABBA. Rejet immédiat à tout écart. Statut : **en cours**,
  journalisé avant code.
- Résultat : **rejeté**. Les largeurs M=1/2/4 restent exactes, mais M=8 produit
  de nombreuses divergences dans les logits FP16 par rapport au chemin
  token-major. Le seuil M=8 change donc le chemin d'exécution MoE et ne conserve
  pas l'arithmétique canonique ; aucun timing n'a été retenu. Le batch MoE
  complet est retiré, tandis que PERF-53 reste inchangé et exact.

### OPT-2026-09-20-RUST-PERF-55 — Expert partagé MoE multi-lignes — en cours

- Hypothèse : la divergence PERF-54 vient du routage/expert sparse à huit
  lignes, pas des projections d'expert partagé qui sont indépendantes par
  ligne. Conserver gate, top-k et experts routés en huit appels M=1, mais
  calculer `shared_expert` et son multiplicateur en M=1/2/4/8 doit mutualiser
  leurs lectures de poids tout en gardant l'arithmétique lossless.
- Baseline : PERF-53 **104,722 ms** médian pour huit positions, référence
  token-major **143,504 ms**. Protocole : séparer sans duplication les chemins
  routé/partagé du MoE, comparer logits FP16 et 80 états octet par octet pour
  M=1/2/4/8, puis deux séries ABBA. Rejet immédiat au premier écart. Statut :
  **en cours**, journalisé avant code.
- Première compilation interrompue par une accolade fermante manquante dans le
  helper de vérification ; aucun modèle, kernel ni timing exécuté. Correction
  syntaxique uniquement avant reprise du protocole inchangé.
- Exactitude physique M=1/2/4/8 : tous les logits FP16 et les 80 états sont
  identiques octet par octet au token-major. Deux séries ABBA M=8 donnent
  respectivement **94,658 ms** (93,190–95,924) contre 136,622 ms, puis
  **98,081 ms** (93,340–105,459) contre 144,954 ms, soit **1,44–1,48×** face à
  la référence du même processus. Comparé à PERF-53 (104,722 ms), l'expert
  partagé économise environ **6,6–10,1 ms** selon la passe. Statut :
  **validé**.
- Contrôles finaux : format, Clippy strict, 23 tests lib, 1 test CLI,
  14 contrats et build release MLX/chat réussissent. Kani 0.68 / CBMC 6.11
  vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures. Comme auparavant,
  Kani n'exécute pas MLX/Metal ; le graphe GPU est couvert par les différentiels
  physiques exacts et les séries ABBA. Statut : **validé et intégré**, prêt à
  publier.

### OPT-2026-09-20-RUST-PERF-56 — Gate et top-k MoE multi-lignes — en cours

- Hypothèse : PERF-54 a seulement démontré que le calcul groupé des experts
  sparse diverge à M=8. La projection gate, le softmax et le kernel top-k sont
  indépendants par ligne et peuvent produire les routes des huit positions en
  un seul graphe, puis alimenter huit appels experts M=1 inchangés. Cette
  frontière complète PERF-55 sans toucher l'accumulation sparse.
- Baseline : PERF-55 **94,658–98,081 ms** médian pour huit positions. Protocole :
  batcher gate/softmax/top-k, découper indices et scores par ligne, exécuter les
  experts routés en M=1, puis comparer logits FP16 et 80 états pour M=1/2/4/8
  et deux séries ABBA. Rejet immédiat au premier écart. Statut : **en cours**,
  journalisé avant code.
- Résultat : **rejeté**. M=1/2/4 reste exact, mais M=8 diverge massivement
  dans les logits FP16 avant toute mesure. La projection gate/softmax/top-k
  multi-lignes change donc l'arithmétique ou les routes à cette largeur. Aucun
  timing n'est retenu ; gate, top-k et experts sparse reviennent entièrement
  en M=1, tandis que l'expert partagé PERF-55 reste intégré.
