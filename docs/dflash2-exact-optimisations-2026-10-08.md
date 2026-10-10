# DFlash2 : optimisations exactes du 35B, 8 octobre 2026

Une optimisation du cache des kernels Metal est conservée dans le code et le
bundle **local** : elle évite de recopier les clés lors d'un accès déjà en cache.
Le microtest CPU passe d'environ **3,50 à 2,63 µs par recherche**, soit une baisse
de latence de **24,90 % / 24,95 %** dans les deux ordres prévus. La dérive maximale
entre passes d'une même variante est **0,137 %**. Le gain en tokens/s de génération
reste **non mesuré** ; ce cache est partagé avec les autres modes d'inférence.

Les contrôles réalisés n'observent aucune divergence numérique ou de tokens.
Une alerte d'alignement MPP préexistante reste cependant ouverte : la qualité
ciblée de cette modification ne certifie pas la stabilité de tout le moteur.
Le 27B n'a pas été chargé sur le GPU. L'application personnelle installée et les
releases n'ont pas été modifiées.

## Sources et périmètre

Cette campagne poursuit [WEB-02](dflash2-web-optimisations-2026-10-08.md), après
lecture de `opti.md`, des résultats négatifs et des chemins déjà optimisés.
Elle utilise le worktree `codex/dflash2-stable`, base
`db52b7304a057ad8f0d8fcf464cd66b22b44fe08`, avec les modifications locales
préexistantes conservées. Chaque prototype et chaque mesure ont été enregistrés
avant leur lancement dans le journal.

La recherche amont a notamment examiné
[mlx-node, commit 10efec703d8f](https://github.com/mlx-node/mlx-node/blob/10efec703d8f5f4bf1e532ac8e3197dbc1a4ebaf/crates/mlx-core/src/models/quantized_linear.rs).
Son module de projections quantifiées concatène des rangées gate/up sous
conditions. Son [module DFlash2](https://github.com/mlx-node/mlx-node/blob/10efec703d8f5f4bf1e532ac8e3197dbc1a4ebaf/crates/mlx-core/src/models/qwen3_5/dflash2.rs)
utilise encore deux projections séparées pour ce MLP. La transposition à notre
Q4/MPP a donc été traitée comme une hypothèse à mesurer. Les fichiers consultés
et leurs empreintes figurent dans le
[manifeste amont](measurements/dflash2-web-03/mlx-node-sources.json).
Les autres moteurs et les pistes précédemment rejetées restent documentés dans
WEB-02 ; aucun ancien essai n'a été relancé pour rechercher un résultat favorable.

Matériel et logiciels : MacBook Air Apple M5, 24 Gio, macOS 27.2
`26B5091g`, MLX 0.32.2, Rust 1.98.1, SDK macOS 26.5, cible de déploiement 26.2.
La batterie était autorisée : environ 61 à 49 % durant cette campagne.
`pmset` ne rapportait aucune alerte thermique ; températures et fréquences
exactes n'ont pas été mesurées.

Modèles : Qwen3.6-35B-A3B EXL3 **2,49 bpw** et son draft DFlash2 packed Q4
de **479 444 992 octets**. Chemins et empreintes des sept fichiers du draft :
[identité Q4](measurements/dflash2-web-03/gate-up-identity.json).
Aucun nouveau checkpoint n'a été téléchargé. La campagne n'établit pas
l'existence d'une meilleure tête entraînée pour ce 35B.

## Optimisation conservée : recherche de clé sans copie

`array::metal_kernel` appelle le C ABI `mlxl3_metal_kernel`, partagé par DFlash2,
les projections EXL3 et d'autres kernels. Le bridge construisait un tuple
contenant le nom, les noms d'entrées et sorties, le header et le shader avant
chaque recherche, y compris sur un accès déjà en cache.

Le bridge utilise maintenant des vues empruntées pendant la recherche et crée
une clé propriétaire uniquement en cas d'absence. Le comparateur conserve
l'ordre lexicographique complet des **cinq champs**, ainsi que l'éviction
lexicographique à **128 entrées** et le stockage local au thread. Les vues ne
sont pas conservées dans le cache. Les métadonnées nulles sont refusées avant
leur conversion en vues ; les chaînes valides restent terminées par zéro,
comme le garantit l'appelant Rust. Les calculs MLX, les shaders, les poids,
la quantification et l'acceptation des propositions restent ceux de la baseline.

Fichiers : `native/ffi/mlx_bridge.cpp`, `native/ffi/kernel_cache_key.h`, deux tests
Rust dans `native/src/array.rs`, tests C++ portables et étape CI. Aucun réglage
utilisateur supplémentaire. Le chemin de copie du contexte de WEB-02 reste
**désactivé par défaut**, avec son bilan de vitesse antérieur non concluant.

Le test indépendant compare le nouveau comparateur à l'ancien tuple C++ :
65 536 paires × quatre combinaisons propriétaire/vue, soit **262 144 contrôles**.
Il couvre les champs vides, préfixes, octets élevés, listes de noms, 256 clés,
l'éviction à 128 et la destruction/modification des clés appelantes. Sur la
fixture de 128 accès, les allocations de clés passent de **768 à zéro**.
Les autres allocations d'un lancement Metal et la RAM résidente n'ont pas été
mesurées. ASan/UBSan passent. La mutation qui ignore le header est détectée ;
la source de production n'a pas été modifiée par cette mutation.
[Contrôles](measurements/dflash2-web-03/key-view-checks.json),
[mutation](measurements/dflash2-web-03/key-view-mutation.json).

La mesure CPU utilise 128 clés résidentes, 100 000 recherches de warmup par
variante, puis une campagne unique **ABBA et BAAB**, sept échantillons de
200 000 recherches par passe. A est l'ancien tuple, B la recherche par vues.
Les valeurs ci-dessous sont les moyennes des deux médianes de passe de chaque
variante ; réduction de latence = `100 × (1 − B/A)`. Les seuils fixés avant
mesure sont 5 % de réduction et au plus 1 % de dérive par variante et par ordre.

| Ordre | A, µs/recherche | B, µs/recherche | Réduction de latence | Dérive A / B |
| --- | ---: | ---: | ---: | ---: |
| ABBA | 3,50262 | 2,63043 | 24,901 % | −0,119 % / +0,041 % |
| BAAB | 3,50433 | 2,62986 | 24,954 % | −0,137 % / −0,045 % |

Les deux ordres qualifient le microgain CPU. Aucun autre build, proveur ou
processus d'inférence de la campagne n'était actif. Aucun gain de génération,
gain de RAM processus ou avantage sur MTP n'en est déduit.
[Plan](measurements/dflash2-web-03/key-view-plan.json),
[mesures et conditions](measurements/dflash2-web-03/key-view-micro.json),
[bruts](measurements/dflash2-web-03/raw-evidence.tar.gz).

## Pistes GPU testées et retirées

Les variantes Q4 ont d'abord passé la comparaison des projections et du SwiGLU
sur six couches et trois entrées : **1 769 472 valeurs de projection** et
**884 736 valeurs SwiGLU**, finies et identiques bit à bit. Des vues volontairement
décalées sont détectées. Les variantes réagencées contrôlent aussi les octets
packed par transformation inverse et passent leur test instrumenté Apple.

La mesure est un microtest des six paires gate/up résidentes, avec 200 parcours
de warmup par variante, puis sept échantillons de 1 000 parcours par passe,
ABBA et BAAB une seule fois. Le gain de débit minimum est 3 %, la dérive maximum
1 %. Cette mesure ne suit pas les dépendances des six couches d'une génération
réelle. Chaque variante dispose de son plan, de ses sources et de son binaire
identifiés ; les gains de ces prototypes ne se cumulent pas avec KEY-VIEW.

| Variante | Débit micro ABBA / BAAB | Dérive et décision |
| --- | --- | --- |
| Concaténation exacte gate/up, kernel initial | +0,877 % / +0,916 % | ≤0,056 %, **rejetée** sous le seuil de 3 % |
| Sommes d'entrée partagées | Non mesuré | Validation Apple échouée, **bloquée puis retirée** |
| Sommes partagées et rangées K2048 alignées | −6,916 % / −6,883 % | ≤0,253 %, **rejetée** |
| Sommes partagées et blocs K256 alignés | ABBA non qualifié ; BAAB −6,676 % | ABBA dérive jusqu'à 3,717 % ; BAAB ≤0,056 %, **rejetée** |

L'ABBA de la dernière variante n'est pas une mesure stable. Il n'a pas été
relancé. Aucun de ces prototypes n'a été intégré ou chronométré en génération.
`dflash.rs` et `dflash_q4_mpp.h` ont été restaurés à l'identique avant le build
final ; le shader expérimental a été supprimé.

[Gate/up](measurements/dflash2-web-03/gate-up-decision.json),
[sommes partagées](measurements/dflash2-web-03/gate-up-sums-quality.json),
[rangées K2048](measurements/dflash2-web-03/gate-up-aligned-decision.json),
[blocs K256](measurements/dflash2-web-03/gate-up-block256-decision.json),
[restauration](measurements/dflash2-web-03/q4-retirement.json).
Sources et binaires locaux : `build/dflash2-web-03/`. Un
[patch du dernier prototype retiré](measurements/dflash2-web-03/patch-evidence.tar.gz)
conserve son reproducer dans les preuves du dépôt.

## Qualité physique et parcours livré localement

Sur le code conservé, cinq tests GPU ont été sélectionnés explicitement et
exécutés seuls, avec un seul modèle cible chargé à la fois :

- Cache Metal : **320 clés, 330 appels et 1 320 valeurs** finies exactement égales
  à l'oracle CPU, avec validation API/shader Apple activée.
- Cible 35B : **24 vérifications capturées, 108 commits et 80 tableaux d'états**
  finis identiques bit à bit au chemin séquentiel.
- Copie du contexte : neuf blocs adversariaux, toutes les longueurs d'acceptation
  de 0 à N pour N de 1 à 3, états/caches et continuation neuronale exacts.
- Bridge de production : **35 réponses non vides**, budgets, cache et modes
  Auto/fixes exacts, **96 blocs copiés vérifiés**.
- Tune : quatre échantillons non vides exacts, politique commune au chat.

Les contrôles du 35B et la comparaison à l'ancien moteur utilisent
`MLXL3_DFLASH_CONTEXT_COPY=1`, `MLXL3_QWEN_PIPELINE=0` et `MLXL3_MTP_LOOKUP=0`,
comme la référence WEB-02. Ces conditions couvrent le draft neuronal et la
copie optionnelle ; le défaut utilisateur de COPY reste OFF.

[Sélection et commandes](measurements/dflash2-web-03/key-view-gpu-plan.json),
[résultats](measurements/dflash2-web-03/key-view-gpu-quality.json).
Le premier lancement du nouveau test de cache a échoué à la compilation du
shader de test : un retour à la ligne manquait après une directive. Cette erreur
a été corrigée avant toute mesure ; son
[rapport](measurements/dflash2-web-03/key-view-gpu-quality-fixture-failure.json)
et son [log](measurements/dflash2-web-03/raw-evidence.tar.gz)
restent conservés, avec `parity: null`.

Le bundle local **1.4.3/build 25** a été reconstruit et sa signature vérifiée.
Le vrai moteur signé passe les **45 contrôles** de `scripts/check_dflash2.py` :
activation, trois politiques, budgets, erreurs, annulation/reprise, réutilisation
du préfixe, désactivation et Tune. En plus de l'oracle cible/draft, les
**33 réponses complètes** ont les mêmes empreintes de tokens, historiques de
cache et nombres de tokens que le rapport de l'ancien moteur signé `ad0bef…`.
La vérification du draft étranger lit ses métadonnées ; elle ne charge pas le 27B
sur le GPU. Les temps enregistrés par ces contrôles servent au diagnostic,
sans constituer un benchmark de débit.
[Plan et oracle antérieur](measurements/dflash2-web-03/key-view-bridge-plan.json),
[contrôle signé](measurements/dflash2-web-03/key-view-bridge-run.json),
[événements](measurements/dflash2-web-03/key-view-bridge.json).

## Vérification, commandes et limites

Sur le code final : `cargo fmt --all --check` et
`cargo clippy --locked --all-targets --features mlx,chat -- -D warnings` passent.
`cargo test --locked --no-default-features --features chat` : **63 passés,
deux ignorés**. `cargo test --locked --release --features mlx,chat` :
**78 passés, 66 ignorés** dans la suite automatique, dont cinq exécutés ensuite
explicitement sur GPU. Les autres tests ignorés restent non exécutés dans cette
campagne. `.venv/bin/python -m pytest -q` : **360 passés, quatre ignorés**
(trois dépendances PonyExl3 absentes et un modèle Ling absent).
[Contrôles finaux Rust](measurements/dflash2-web-03/key-view-final-checks.json),
[Python](measurements/dflash2-web-03/key-view-python.json).

Les tests C++ utilisent `c++ -std=c++17 -O3 -Wall -Wextra -Werror`, puis une
compilation séparée `-O2 -fsanitize=address,undefined -fno-omit-frame-pointer`.
Le test portable et la vérification syntaxique du harnais de preuve sont ajoutés
à la CI Linux/macOS, sur PR et push. Les quatre jobs inspectés sur GitHub sont
verts au **commit de base db52b7304a05** ; aucun run GitHub ne valide le diff
local non commité. [CI inspectée](measurements/dflash2-web-03/key-view-ci.json).

`/bin/zsh scripts/build-macos-app.sh` et `codesign --verify --deep --strict`
passent. La suite Desktop complète avait passé dans WEB-02 ; les sources Swift
n'ont pas changé pendant cette campagne et cette suite n'a pas été répétée.
Le moteur signé actuel est exercé par le vrai bridge ci-dessus.
[Packaging actuel](measurements/dflash2-web-03/key-view-package.json),
[suite Desktop antérieure](measurements/dflash2-web-02/copy-package-checks.json).

La tentative de preuve du comparateur utilise le vrai header C++ et l'ancien
tuple comme oracle : textes symboliques de 0 à 3 octets, listes de 0 à deux noms,
octets sur tout le domaine u8, unwind 33 prévu et assertions d'unwinding
conservées dans la commande tentée.
`goto-cc -std=c++17` échoue sur le contexte libc++ réel ; aucun programme goto
valide n'est transmis à CBMC. ESBMC est absent. La tentative Kani `--tests` du
test Rust de métadonnées FFI ne sélectionne aucun harnais, même après correction
des features et du filtre. **Aucune propriété formelle de KEY-VIEW n'est déclarée
prouvée.** Les commandes initiales incorrectes, corrections et diagnostics
restent conservés. Le harnais C++ passe sa compilation syntaxique et reste
disponible pour un vérificateur compatible.
[Bilan formel](measurements/dflash2-web-03/key-view-formal-summary.json).

Les preuves Kani réussies des prototypes Q4 portent sur leurs gardes CPU de
formes, stockage et lancement, sur les dimensions i32 annoncées, sans hypothèse
supprimant les cas valides : 91 obligations pour la concaténation ciblée,
38 pour la garde de lancement des sommes partagées ; les suites de cinq puis
six harnais totalisent 215 puis 253 obligations réussies et 10 puis 12 couvertures.
Ces prototypes sont retirés. Ces résultats ne prouvent ni le shader MSL ni le
changement C++ final. La tentative CBMC du vrai header MSL échoue sur
`metal_stdlib` ; aucun modèle réécrit dans un autre langage n'est présenté comme
une preuve. Les limites de preuve du chemin COPY conservé restent celles de
WEB-02. [Gardes Q4](measurements/dflash2-web-03/gate-up-verification-summary.json),
[sommes](measurements/dflash2-web-03/gate-up-sums-verification-summary.json),
[MSL non vérifié](measurements/dflash2-web-03/gate-up-sums-cbmc.json).

L'instrumentation documentée par
[Apple](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage)
signale `Tensor strides[1]=64 is not aligned to 128 bytes row boundary` sur le
Q4/MPP initial. L'ancien binaire WEB-02 reproduit cette erreur sur l'entrée de
production `draft_prefix_cache_clone_keeps_all_kv_bytes`, avec les deux couches
de validation Apple actives et arrêt sur faute. Il s'agit d'un échec hérité,
également observé avant le partage des sommes, et **encore ouvert** dans le
kernel conservé. L'égalité numérique normale ne l'efface pas. Les variables
d'instrumentation restent limitées aux processus de validation et sont absentes
du benchmark et du bundle d'exécution.
[Qualification sur l'ancien moteur](measurements/dflash2-web-03/stock-alignment-result.json),
[log](measurements/dflash2-web-03/raw-evidence.tar.gz).

## État d'intégration et preuves

KEY-VIEW est conservé en code local et dans le bundle généré
`dist/MLXL3 Desktop.app`. Le moteur signé fait **9 764 752 octets**, SHA-256
`23e0f3c16ec08a54be6b54ee83009f16e4a1e3be43253a609088181661c55e69`.
Sources, binaires de test, microtest et artefacts signés sont identifiés dans
[l'identité finale](measurements/dflash2-web-03/key-view-identity.json) et archivés
localement sous `build/dflash2-web-03/key-view/`.

Les quatre prototypes GPU sont retirés. Aucun essai de cette campagne ne reste
actif. Aucun push, publication, release, remplacement de l'application installée
ou nouveau chargement GPU du 27B. Gain de génération et comparaison avec MTP :
**non mesurés pour la modification conservée**. Les résultats ne s'étendent pas
aux autres matériels, quantifications ou prompts sans validation adaptée.

## Révision du 10 octobre 2026 : première mesure de génération

Les conclusions CPU du 8 octobre restent inchangées. Le premier essai de
génération du 9 octobre est **non concluant** : ordre ABBA interrompu par timeout,
swap croissant et exécutions concurrentes. L'utilisateur confirme le lancement
accidentel de MLXL3 en parallèle. Aucun gain ni régression de génération ne peut
être attribué au changement de clé à partir de ce test. Les résultats partiels,
les causes observées et les limites restent dans le
[rapport du contrôle réel](dflash2-key-view-tps-2026-10-09.md).

Une [reprise isolée sur secteur](dflash2-key-view-tps-isolated-2026-10-10.md)
termine ABBA le 10 octobre : code 65,21 → 67,18 tok/s et français
55,36 → 58,74 tok/s, écarts **indicatifs** uniquement. Les 32 réponses sont
exactes, mais les variations de vitesse atteignent 4,206 %, au-dessus du seuil
de 1 %. Aucun gain de génération qualifié ni confirmation BAAB. Les mesures
CPU et leurs conclusions historiques restent conservées.

Preuves brutes archivées sans perte pour publication : chaque chemin original
dans les anciens rapports JSON est répertorié dans `raw-evidence-index.json`
et récupérable dans `raw-evidence.tar.gz` de son dossier de mesures. Les sources
historiques et les rapports originaux restent également dans ces archives.
