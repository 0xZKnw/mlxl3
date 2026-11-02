# POSTPROFILE05 — fusion des échelles EXL3, audit avant décision

28–29 octobre 2026. **Essai clos : variantes B/C rejetées après audit et correction ;
chemin précédent restauré et vérifié.** Aucun gain livré, aucune app
remplacée, aucun push. Les sources, binaires et résultats restent conservés.

Hypothèse : fusionner gather et produit FP16, Cold et Hot, en préservant les
Hadamard MLX, les réductions, les deux soumissions précoces et le cache de 3 Gio.
M5/24 Gio, Flash Next EXL3, MLX 0.32.2, 53 slots/couche, 4 lecteurs,
Hot/Wide/Unique activés, lookahead 0, seuils MLX 40 opérations/40 MB.

## Comparatif release du premier port B

Deux processus par bras et prompt, ordre ABBA, 128 tokens, une chauffe puis
trois générations mesurées par processus : 24 mesures et 8 chauffes.
Tous les processus sont joints ; zéro nouvelle écriture swap depuis l'ancre w1.
Chaque réponse vérifie le texte/hash golden complets, les comptes 128/127 et
le footprint physique inférieur à 12 Go. Pics observés : 7,12–7,16 Go.

| Prompt | Référence A | Port B | Decode | Préfill | TTFT |
|---|---:|---:|---:|---:|---:|
| MAIN | 10,2849 tok/s | 10,2311 tok/s | −0,52 % | −0,19 % | +0,17 % |
| HOLD | 9,8749 tok/s | 10,0470 tok/s | +1,74 % | −0,24 % | +0,25 % |

Les moyennes par processus de référence dérivent de 10,4654 à 10,1045 tok/s
sur MAIN, de 9,9917 à 9,7582 sur HOLD. **Gain non établi ; audit avant
conservation ou retrait.** [Mesures et bornes](speed-inspection-v2.json),
[plan](speed-plan-v2.json), [bruts](speed-plan-v2-result.json).
La première extraction a refusé le statut `validated`, attendu à tort comme
`complete` ; seule l'analyse a été corrigée, aucun GPU relancé pour cela.

## Correction numérique et chemins réels

14 tests physiques passent : produits (50 cas, poison et 5 refus), projections
GU/Down complètes (96 cas, 16 refus), MoE Cold/mixte/Hot (9 cas), overlap SSD/Hot,
erreurs/récupération, Wide/Unique/prefetch non publié, oracle indépendant sur
5 positions avec 248320 logits et états GDN, composants, paires et MTP.
Les tests comparent des sorties non vides et finies, bit à bit selon le contrat.
Le test MoE exige quatre produits bankés par calcul de ≤10 routes, et le
fallback historique au-delà ; Cold, mixte, Hot et wide sont effectivement vus.

Le vrai shader a ensuite été muté pour lire le mauvais slot : le test produit
échoue immédiatement comme attendu (code 101). Après restauration/rebuild,
le libtest est byte-identique au V2 antérieur ; les deux tests affectés repassent.
[Qualité modèle](model-quality-plan-v2-result.json),
[micro](micro-plan-v2-result.json), [mutation](mutant-plan-result.json),
[restauration](restored-micro-plan-result.json).

## Audit de l'adaptation et nouvelle variante C

À la demande de l'utilisateur, la règle d'audit avant rejet est ajoutée au
journal obligatoire d'AGENTS.md. Les flags prétransformé/raw, formes, dtypes,
activation Cold/Hot et fallback ont été relus. Aucune rotation doublée observée.

Un coût évitable existe dans B : chaque produit aplatit x et bank via deux
appels MLX, soit huit `reshape` par calcul d'experts. Enlever quatre gathers
ne garantissait donc pas de réduire le travail de construction du graphe.
L'API Metal générique peut recevoir les entrées contiguës de rangs 2 et 3
directement. Le [backend MLX 0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp)
vérifie la contiguïté puis lie les buffers et les métadonnées demandées.
[Copie de la source examinée](proof/mlx-v0.32.2-custom-kernel.cpp.txt).

Le diagnostic A/B sur 17 tokens vérifie les sorties et états finaux complets
entre quatre phases ; tokens/hash/texte/prompt IDs sont égaux entre bras,
préfixe golden externe vérifié. Sur la trace non fenced : 1460 calculs pour
16 tokens décodés ; construction des calculs 2,279 → 4,083 ms/token,
GU 0,966 → 2,576, Down 0,880 → 1,184 ; les 5840 spans de produits ajoutés par B
totalisent 1,434 ms/token. **Temps host imbriqués avec instrumentation différente :
ne pas les additionner, les qualifier de CPU actif/GPU idle, ou les convertir
en gain release.** [Diagnostic et limites](audit-inspection.json).

La variante C a enlevé ces vues et ajusté seulement la garde MSL à
`x_shape[0] * WIDTH`. Le validateur CPU et le nombre d'éléments restent identiques.
Les sources antérieures sont conservées dans [before-c.json](before-c.json).
Son protocole est préenregistré dans opti.md : 279 tests Rust passent,
170 physiques sont ignorés par la suite automatique ; les 14 tests GPU
sélectionnés passent ensuite explicitement. 165 tests Python passent.

**Rectification des sous-spans GU :** A construit les gathers d'entrée avant
`expert_gate_up_build`, B/C à l'intérieur. Les chiffres GU ci-dessus n'ont pas
la même frontière et leurs deltas ne mesurent pas un travail égal. Comparer
l'enveloppe complète `expert_calculate_build` ; garder les limites de profilage.

Le second comparatif release reprend une référence fraîche, ordre ACCA,
24 mesures + 8 chauffes, mêmes prompts/128 tokens/3 Gio/goldens/gardes.

| Prompt | Référence A | Port corrigé C | Decode | Préfill | TTFT |
|---|---:|---:|---:|---:|---:|
| MAIN | 10,5593 tok/s | 10,1602 tok/s | −3,78 % | −2,89 % | +3,02 % |
| HOLD | 10,4442 tok/s | 10,3891 tok/s | −0,53 % | −0,33 % | +0,33 % |

Sur MAIN, les deux moyennes A sont 10,5685 et 10,5501, les deux C 10,1231 et
10,1973 tok/s. Sur HOLD, la référence dérive de 10,3136 à 10,5748 ; C passe de
10,2039 à 10,5742. Aucune amélioration reproductible qualifiée.
Les débits agrégés (tokens/somme des durées) donnent aussi −3,81 % et −0,54 %.
[Résultats et bornes](speed-inspection-c.json), [bruts](speed-plan-c-result.json).

L'audit C utilise la même instrumentation supplémentaire que B. Tous les bras
ont les mêmes compteurs de cache avant préfill/après et les mêmes 17 tokens,
texte/hash/prompt IDs ; chaque bras vérifie ses logits/états complets entre
quatre phases. 22 clés de kernels EXL3 sont observées par bras ; le cache global
de factories n'est pas intégralement instrumenté. La construction complète C
totalise 2,132 ms/token (Hot 1,051 + Cold 1,080), contre 2,279 pour A et 4,083
pour B. Les soumissions host C totalisent 14,843 ms/token contre 16,940 pour A,
21,884 pour B. Conditions historiques et instrumentation interdisent d'en
déduire un gain release ou une réduction exacte de GPU idle.
[Ventilation Hot/Cold et limites](audit-abc-inspection.json).

L'audit a donc trouvé et corrigé des appels évitables, sans qualifier de
boost global. La forme, les dtypes, les masques, l'activation Cold/Hot, le
fallback, les rotations et la contiguïté des banques ont été relus. Aucun autre
défaut concret évident n'a été identifié. La cause exclusive CPU/GPU du delta
release reste non mesurée ; cela ne rejette pas Sushi dans son ensemble.
Les vues retirées ne suffisent pas à expliquer causalement le ralentissement.

Le port incrémental B/C est retiré du chemin actif. Les trois sources restaurées
sont identiques aux sauvegardes d'avant POSTPROFILE05, qui contiennent les
modifications locales antérieures. Les modules/shader/harness inutilisés et
le test MoE propre au prototype sont archivés avec eux, puis retirés. Les
tests Python du superviseur et la règle AGENTS.md restent actifs.
[Décision et empreintes](retirement.json), [101 fichiers source C](retired-c-sources.tar.gz).

## Vérification du code final restauré

Le moteur recompilé est byte-identique à A (`a60eb4ad…`), et le libtest à son
contrôle historique (`d4bb7309…`). Les changements locaux antérieurs sont
préservés ; l'ancien fichier hybrid_tests retrouve aussi exactement son hash.
278 tests Rust passent, 0 échoue, 167 restent ignorés par la suite automatique.
Les 165 tests Python concernés passent. Fmt, Clippy strict mlx/chat all targets,
Ruff, build et validation YAML/liaisons PR/main passent.

Commandes exécutées, avec MLXL3_MLX_ROOT pointant vers le paquet MLX 0.32.2
identifié dans les plans, CARGO_BUILD_JOBS=1 pour Cargo :

```sh
cargo fmt --check
cargo clippy --locked --features mlx,chat --all-targets -- -D warnings
cargo build --locked --release --features mlx,chat --bin mlxl3-rs
cargo test --locked --release --features mlx,chat -- --test-threads=1
python -m pytest -q tests/test_flash_slotstream_supervisor.py tests/test_flash_quality_run.py tests/test_flash_profile.py
python -m ruff check docs/measurements/post-sushi-20261028/gpu-runner.py tests/test_flash_slotstream_supervisor.py
python -m ruff format --check docs/measurements/post-sushi-20261028/gpu-runner.py tests/test_flash_slotstream_supervisor.py
```

[Résultat final](final-state.json), [plan Cargo exact](restored-final-cpu-plan.json),
[sorties](restored-final-cpu-plan-result.json), [Python](python-restored-final.log).
Le bilan physique de l'essai est de 33 contrôles positifs et un mutant échoué
comme attendu. Les deux comparatifs distincts totalisent 48 générations mesurées
et 16 chauffes ; leurs gains ne sont pas additionnés. Zéro nouvelle écriture
swap pendant les mesures, tous les processus de l'essai sont joints.

## Identités et limites

A : moteur `a60eb4ad…`, libtest `d4bb7309…`. B : moteur `67434011…`, libtest
`0e24b725…`. C : moteur `1cf6523b…`, libtest `126a72c2…`.
HEAD `7978fcea…` avec modifications locales préexistantes.
[Sources/binaires B](identity-v2.json), [campagne release](identity-speed.json),
[ancre w1](anchor.json). Les nouveaux tests V2 ne sont pas dans le libtest V1.

Kani CPU : 146 obligations réussies et 6 couvertures satisfaites, sans assume,
sur le validateur de géométrie/indices. Cela ne prouve ni Metal ni la concurrence.
Les tentatives sur la vraie API MLX échouent avec l'ICE Kani intrinsics.rs:243 ;
CrossHair est absent. Le compilateur Metal offline manque de MetalToolchain ;
les contrôles GPU utilisent la compilation runtime MLX.

La qualification de performance MTP/DFlash2 n'a pas été lancée : aucune variante
AR n'a justifié une promotion. La parité de 128 états à chaque position n'est pas établie par
les seuls textes CLI. Les tests et vérifications bornées ne sont pas une preuve
universelle. CI locale adaptée ; aucun run sur cet arbre non publié, et aucune
CI verte déduite du résultat vide du connecteur limité aux runs de PR.

La [revue de l'issue 58](issue58-triage.md) reste séparée ; aucune de ses
optimisations moteur n'est ajoutée avant la fin de cet essai.
