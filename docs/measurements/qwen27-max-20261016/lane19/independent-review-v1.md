# Q27-19 — revue indépendante finale, lecture seule

Date : 2026-10-16. Revue du portage lane-order Q27-19 et de ses preuves existantes, après lecture de opti.md Q27-19 et des antécédents Q27-15/16/18. Aucune compilation, preuve, inférence GPU, benchmark ou nouveau test lancé ; aucune écriture au dépôt ou au journal. Les seules opérations exécutées sont lecture/recherche/diff/hash et analyse hors GPU des journaux.

## Décision

Aucun défaut fonctionnel bloquant trouvé dans les sept sources gelées. La qualité numérique existante permet de passer à l'évaluation de vitesse déjà préenregistrée. La décision de conservation pour accélération reste subordonnée au vrai ABBA/BAAB ; les 20/32 filtres micro Q27-16 ne prouvent aucun tok/s global et les 12 cellules non qualifiées n'ont pas été effacées.

Deux précisions de compte rendu sont nécessaires : 32 cycles Session = 29 blocs spéculatifs + 3 cycles AR, et M>=3 est un repli par projection, pas une garantie que tout le parcours D2/M3 évite le code M1. Ces précisions n'invalident pas l'oracle.

## Identité et préservation de l'état local

Racine examinée : /Users/justin/.codex/worktrees/qwen27-smallm/mix-stq1_0.
Les sept fichiers actuels sont octet pour octet identiques à build/qwen27-lane19-20261016/candidate/ :

- native/src/contracts.rs — 1383bf08bfa339e71bc3e051febb1fee7dd6a20ae4ade01c1e07c14d3c5377f9
- native/src/lib.rs — 0f14cad9235c92feeb45e814cdafa90c8aa25e8350354d86dd909f7bcc82136c
- native/src/linear.rs — dfbf9a0f5e0aec2cb109dbbb380a351ff1a7c4d6f64645957741976d00002fe7
- native/src/mtp_tune.rs — 0f95daaacbf47f30a127f5de1250ee7c02085a90961ae537df2e780ede725d6e
- native/src/qwen35.rs — 029afa648e54883203afb901bf440e91af8c9c4de36514e1b6c3e3eebe87a379
- native/shaders/_qmv_tile_kernel.metal — fb7c934cb24d7c84e6a0cb87c4bd85a59ce5c3e329ef7cf4c767f27f277db959
- native/shaders/_qmv_mapped_tile_kernel.metal — b8c3fedc2fc1b3eda85ce901e1f5be4e7dbb44700605030cac545dbd5e79e3ef

Comparaison effectuée avec build/qwen27-lane19-20261016/before/ et les identités avant/candidate du dossier de mesures. Le contracts avant est 0b21c96a, après retrait effectif Q27-18 : le portage19 ne réintroduit pas son guard GDN. Les modifications de qwen35 sont exclusivement dans le module cfg(test) : le préfixe de production avant ce module est exactement identique à before. Les modifications locales antérieures, notamment MLP M2/M4, restent dans before et n'ont pas été attribuées à19.

Moteur examiné par les preuves : d6a4ea4425ee525add3f3102e9d9fb91cc88cd084334ef42de2e0aa43db24536 ; binaire test d4bcf00d036fb8709ab851cce9409226e12d4b678367bbfe8c711ab3c097c951.

## Routage, math et caches

- Flag de démarrage MLXL3_EXL3_LANE_ORDER : seul le littéral 1 active ; absent/0/invalide OFF. OnceLock en production, override TLS cfg(test) avec restauration imbriquée/panic et isolement par thread. Les modèles des nouveaux tests sont chargés sous true pour préparer les indices, puis le même jeu de poids peut exécuter stock0 ou candidat1.
- Guard réelle : M5, Codebook::Mul1, 16 géométries physiques inventoriées, rows 1|2, K1..4 exactement selon whitelist. Les deux groupements sont également limités aux widths [10240,6144] et [17408,17408]. Les autres formes/codebooks/M>=3 ne reçoivent pas la permutation par cet appel.
- Permutation réelle u32 par bits ; table indépendante [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] testée sur chaque indice des entrées5120/6144/17408. Les indices positifs i32 sont créés et évalués au chargement. Array.take(axis2) est appliqué après le suh/Hadamard FP16 existant dans les quatre chemins simple/groupé M1/batch.
- Les deux nouvelles branches half4 lisent exactement les valeurs stock half2+shuffle [2q,2q+1,2q+8,2q+9]. Les lignes/tiles sont alignés et les strides groupes/entrée sont multiples de128 ; aucun changement NT/MB/SG/splits/FMA/ordre de réduction n'est introduit. La branche stock reste présente sous macro0/undefined.
- Le header inclut MLXL3_QMV_LANE_ORDER 0|1. Le cache C++ OwnedKernelKey compare header ET source (native/ffi/kernel_cache_key.h:49 et mlx_bridge.cpp:449..457), donc les deux layouts d'un même nom de kernel ne collisionnent pas.
- runtime_key utilise la même interprétation explicite du flag et ajoute lanelayout-v1=0|1 avant smallm-v2. Le test exerce8 clés distinctes (2layouts x 4choix smallM) et vérifie la clé de production ; les profils Tune ne traversent pas silencieusement les layouts.

Limite de portée : le flag est une configuration de démarrage. Changer l'environnement après chargement dans le même processus n'est pas un réglage runtime pris en charge. Pour M>=3, le repli porte sur la projection : le MLP série peut appeler plusieurs projections M1 au sein d'une vérification M3, et celles-ci peuvent utiliser le layout. Les tests modèle couvrent explicitement ces combinaisons.

## Preuves de correction existantes inspectées

Evidence : docs/measurements/qwen27-max-20261016/lane19/. Les quatre logs physiques contiennent chacun running1test puis 1passed/0failed/0ignored. Les quatre enfants ont code0 et reaped=true, délais300s chacun. Cache256 et batterie90→87% consignés ; aucune vitesse tirée de ces durées de tests.

1. Projections : 20 comparaisons imposées par ensure(cases==20), simples et groupées M1/2/3/4. Les sorties attendues ont taille non nulle connue et finitude ; égalité bit à bit vérifie aussi la longueur du candidat. Les indices sont effectivement présents dans les fixtures. Mutation de deux indices lm_head détectée sur246227 valeurs FP16 finies, puis restauration et nouvelle égalité. Les chemins M3/M4 stock sont donc réellement exercés, et pas seulement supposés par guard.
2. MLP : 64couches x M2/3/4 x MB3OFF/ON =384 comparaisons avec ensure(cases==384). Oracle stock lane0 + MLPsérie ; candidat lane1 + batch. Tailles5120/par ligne, nombres de lignes, finitude de l'oracle et égalité exacte. Entrées signées/subnormales, limites vide/mauvaiseforme/largeur/M9 refusées.
3. Préfixes : longueurs24 et257, M1/2/3/4, cinq configurations batch/MB3/pipeline, chaque préfixe retenu. 40 lignes de diagnostic représentent100 commits et100 continuations de2tokens (2x5x(1+2+3+4)). Logits248320/par token et hidden5120/par token non vides, finis et bit-exacts contre série lane0. Chaque état est présent, son nombre de valeurs correspond à sa forme et toutes les valeurs sont finies ;64couches x2 =128 tableaux comparés.
   La faiblesse d'oracle signalée avant gel est corrigée : replay retenu et deux forward de référence [113,127] sont sous with_lane_order_test(false), même lorsque le candidat englobant est lane1. Les états des continuations sont comparés bit à bit après vérification de finitude par state_bytes.
4. Session réelle : quatre modes x8cycles =32 cycles ; D1/2/3 ont chacun8 blocs spéculatifs, le quatrième mode force [0,3,0,1,2,0,3,1] et rapporte5blocs +3AR. Le log totalise donc29blocs spéculatifs, et non32. Chaque cycle compare tokens, états, hidden, stockage KV brut et cache de tête avec la cible série. Les budgets0 et profondeurs invalides sont refusés. Ceci valide l'intégration candidat1 ; l'indépendance math lane0 est fournie par les trois tests précédents, pas par Session seul.

Nuance de finitude : dans les deux continuations du test préfixes, les logits sont bit à bit comparés au replay lane0 sans assertion séparée de compte/finitude à cet endroit. Les assertions explicites248320/fini portent sur logits série/verify ; les états de continuation sont explicitement finis. Il ne faut donc pas présenter chaque logit de continuation comme un contrôle séparé de finitude. Aucune divergence ni sortie vide observée ; cette lacune locale ne contredit pas les oracles principaux et n'impose pas à elle seule de reconstruire le binaire gelé.

## CPU, Kani et CI

Rust suite : six résultats lus, 90+42+14+33=179passés, 75+8+0+0+2=85ignorés. Le parseur externe initial exigeant une seule suite>170 avait échoué ; cpu-inspected.json conserve cette erreur et l'agrégat corrigé, sans réécrire l'échec ni inventer une nouvelle exécution. fmt/Clippy/build mlx,chat présents dans les logs et terminés code0.

Kani sur les vraies fonctions CPU, sans assume, boucle ou stub :

- clé : bool x2 complets,2obligations SUCCESS/2covers SATISFIED ;
- permutation : u32 x2 complets,9obligations SUCCESS/3covers SATISFIED, bijection et conservation de tile16, y compris u32::MAX ;
- guard : i32 rows/input/output,usize K et bools complets,68obligations SUCCESS/6covers SATISFIED ; couvre M1/M2/groupes/deuxentréesdown/M3repli/OFF.

Zéro FAILURE et zéro UNREACHABLE dans ces trois logs. La tentative réelle Array/MLX a échoué par ICE kani-compiler intrinsics.rs:243 avant preuve : non vérifiée. Aucun résultat Kani n'est présenté comme preuve des shaders, de MLX, de la concurrence ou de la génération complète. Les fragments MSL ne sont pas prouvés par CBMC C++ ; limites explicites dans formal-limitations.json.

.github/workflows/rust.yml inspecté : push et PR, permissions lecture, Kani0.68, fmt/Clippy/tests/build, macos26 avec MLX0.32.2 pinné. Le nouveau step mac CPU exécute les deux tests de TLS et Tune avec --exact et exige textuellement 1passed/0failed/0ignored pour éviter le zéro-sélection. Il n'appelle pas de test numérique GPU. Pas de run GitHub du diff local consulté ni déclaré vert ; il faut distinguer workflow adéquat et CI effectivement passée.

## Frontière bridge et état de performance

bridge-boundaries-status.json lu après achèvement : statuscomplete/paritytrue, engine d6a4ea44, code0/reapedtrue,241.27s. Six budgets [1,2,3,4,17,64],2préfixes,2annulations avec récupération,3refus attendus ; journalready lanelayout-v1=1,MLX0.32.2/pipelineoff/adaptive0/lookup0. Le journal propre du checker confirme chaque catégorie. Les mesures de débit incidentes du smoke qualité ne constituent pas une campagne de vitesse.

Au moment de cette revue, aucun résultat ABBA Q27-19 n'a été examiné. Il reste à juger la vitesse avec le protocole préenregistré même binaire0/1 et les critères de dérive/gain, puis confirmer si requis. L'absence de preuve MLX/MSL, les85tests ignorés de suite CPU, la CI distante non exécutée et l'app installée inchangée restent visibles. Aucun chiffre15tok/s ni gain natif19 n'est établi par cette note.

## Commandes de revue

Lectures ciblées via rg/sed/cat ; comparaison Python hors GPU des7fichiers réels/before/candidate avec hashlib.sha256 ; vérification du préfixe qwen35 de production avant cfg(test) ; comptage regex des statuts Kani et des résumés Rust ; inspection des quatre logs physiques, physical-status.json/cpu-inspected.json/formal-limitations.json/bridge-boundaries-status.json et du workflow. Aucun nouveau contrôle exécutable de l'implémentation n'a été lancé.


## Révision après décision du parent — correction en cours, à revalider

Après cette revue, le parent renforce les deux chemins logits de continuation (candidat et replay lane0) par les assertions explicites longueur248320 et finitude, avant toute vitesse. Le calcul runtime reste inchangé ; le binaire de test et le SHA qwen35 ci-dessus désignent la version précédente, dont la preuve demeure conservée. Le point de couverture est donc « corrigé dans le patch prévu, revalidation CPU et test préfixes seuls encore à constater », et non « corrigé/validé » à ce stade. Il faudra consigner les nouvelles empreintes et le résultat réel de cette revalidation ciblée. La présente revue n'a déclenché aucun contrôle.
