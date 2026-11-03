# Registre de migration Metal natif

L’objectif final est un produit distribuable sans MLX. Le registre commence sur `a7eb377c` (PR 76). Main est à `b9d5cbdc`, la PR 72 à `a6397210` et la PR 74 à `ede7f3b8` au relevé du 3 novembre 2026. La 72 porte le Desktop. La 74 modifie également le bridge et MTP. Leurs têtes ne sont pas incluses dans la base de ce registre. Refaire la comparaison de couverture à leur intégration. [Objectif et gates N0–N8](https://github.com/0xZKnw/mlxl3/issues/75#issuecomment-6601668261) · [Fondations](https://github.com/0xZKnw/mlxl3/pull/76).

## Statuts et référence

Les statuts de port sont **non porté**, **porté non qualifié**, **qualifié**, **activé** et **ancien chemin retiré**. Un résultat sur une ligne ne se transfère pas à une autre. Tous les parcours de génération ci-dessous sont **non portés** à un runtime indépendant. On distingue la qualification physique de l’ancien moteur de l’avancement du port.

La référence Flash standard M5/24 Go est le CLI de SHA-256 `065915db5218de92973d01692e19e623d44a4746e91c52be8de266308732df39`, MLX 0.32.2, checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`. Politiques effectives et budgets se trouvent dans les plans par bras, fixtures dans le manifeste de 741 empreintes. MAIN/HOLD et une conversation de 2 050 entrées ont tourné sur ce moteur MLX. Pour comparer le futur candidat natif, conserver des exécutables et processus séparés. [Référence, contrôles, traces et limites](../measurements/issue75-migration-20261103/README.md) · [Origines AR-QUICK](../measurements/issue75-lot0-20261103/README.md).

## Entrées effectivement offertes

`run`, `forward` et `bridge` passent par la distribution de [NativeChatModel](../../native/src/main.rs). Le serveur OpenAI utilise le bridge et le Desktop lance le moteur. Conserver les réponses JSONL, les IDs, le découpage du streaming, les erreurs, l’arrêt et la réutilisation.

| Modèle déclaré | Mode et données de calcul | État à conserver | Oracle ou parcours attribué |
|---|---|---|---|
| `qwen4_exp` | Flash AR B1, prefill sériel et multi-token, matrice EXL3 K2/MUL1 d’experts 2560↔640, entrée F16, routes 10/512, SSD Hot/Cold | 36 GDN, 12 QSA, PLE, dernière position, banques compressées et baux | Cinq pas indépendants, fixtures MAIN/HOLD, erreurs/annulation/reprise, [dossier M5](../measurements/issue75-migration-20261103/README.md) |
| `qwen4_exp` | QSA dense puis sparse lorsque le nombre de blocs complets dépasse 512, cache vide, hit, reset, contexte long | K/V, quatre positions par bloc et queue 0–3, tri des indices, offset | [Oracle QSA et conversation longue](../measurements/issue75-migration-20261103/decisions.md), compléments N5 sur toutes les queues et les branches |
| `qwen4_exp` | Tête MTP ou DFlash, AR du draft, vérification B2/B3/B4 et autres largeurs acceptées, lookup/copie, rejet total et préfixes | États target/draft, RNG, sauvegarde GDN ou tape, budget et commit | [MTP](../../native/src/qwen4/mtp_prefill_tests.rs), [vérification des préfixes](../../native/src/flash_draft.rs), [bridge](../../tests/test_mtp_bridge.py) |
| `qwen3_5` | Dense résident, AR un token, prefill sériel/batch et petits M, projection EXL3 ou dense selon le checkpoint | K/V et états GDN, tête de sortie, compteur de position | [Contrôles Qwen](../../native/src/qwen35.rs), [mesures Qwen 27B historiques](../measurements/qwen27-20261016/) |
| `qwen3_5_moe` | MoE résident, sélection et combinaison pondérée, AR/prefill. MTP et DFlash selon compatibilité du modèle | Cache target et du draft, sélection/commit des préfixes | [Contrôles de commit et de reprise Qwen](../../native/src/qwen35.rs), [changement de modèle](../../tests/test_mtp_model_switch.py) |
| `lfm2` | AR résident, convolutions courtes et attention, chemins sériels et batch autorisés par la configuration | Histoire convolution, cache attention, limite de contexte | [Modèle et contrats](../../native/src/lfm2.rs), [contrôle contexte Desktop](../../tests/context-memory-check.swift) |
| `lfm2_moe` | Experts résidents, prefill exact en microblocs lorsque la géométrie le permet, fallback sériel, AR B1 | Routes, ordre de combinaison, historiques et K/V | [Comparaison microbloc/sériel](../../native/src/lfm2_prefill_tests.rs), [qualification historique LFM](../measurements/audit-api-lfm-20261016/) |
| `bailing_hybrid` | Ling, AR et prefill sériel sur l’entrée du moteur, attention et récurrence selon couche | États récurrents et attention, nombre d’avancées et resets | [Tests d’états et prefill](../../native/src/ling.rs), [intégration de référence](../../tests/test_ling_integration.py) |
| `gemma4` | AR, prefill en batch si admis par `can_batch`, repli sériel. Attention à fenêtre ou globale, softcap et GeGLU | Cache par couche, fenêtre et position correspondante | [Moteur](../../native/src/gemma4.rs), [investigation physique historique](../gemma-sdpa512-investigation.md) |

## Opérations, formats et propriétaires

| Groupe | Layout, contrat et durée de vie à porter | Implémentation de référence et test |
|---|---|---|
| Stockage et vues | Dimensions i32, dtype et taille d’élément, buffers contigus/stridés, offset, strides signés, broadcasts, vues vides, copy/compact et alias. L’objet MLX retient le backing aujourd’hui. | [Array et FFI](../../native/src/array.rs), [admission et bilans](../../native/src/qwen4/memory.rs). N1 exige des bornes et des tests de destruction pendant un vol GPU. |
| Index et réductions | Slice/update, concat, gather, reshape, transpose, somme/axes, argsort et argmax. Préserver sélection et ordre en présence d’ex aequo. | [Bridge et sélecteurs d’opération](../../native/ffi/mlx_bridge.cpp), [QSA](../../native/src/qwen4.rs). |
| Activations et normalisation | Add, mul, divide, pow, logaddexp, exp, rsqrt, sigmoid, tanh, RMS, variantes SwiGLU/GeGLU. Dtypes réels, casts et ordre d’accumulation propres à chaque chaîne. | [Bridge](../../native/ffi/mlx_bridge.cpp), [contrôles de projections](../../native/src/linear/prepared_tests.rs). |
| EXL3 et projections | Codebooks Default, Mcg, Mul1, K=1..8 du codec. Pour chaque géométrie acceptée, GEMV B1, petit M, prefill, biais, H128, suh/svh. Les contrats de layout du codec et du matmul sont distincts. | [Codec](../../native/src/codec.rs), [linear](../../native/src/linear.rs), [tests GPU](../../native/tests/runtime_gpu.rs). Les commandes de codec disposent déjà de kernels directs dans [gpu.rs](../../native/src/gpu.rs). |
| Fichiers et experts | Safetensors, dtype, taille et bornes du record, troncation, lecture vectorisée, slot/génération, invalidation, publication tardive et réduction de banques. | [Store](../../native/src/ssd_moe.rs), [lecteur asynchrone](../../native/tests/disk_reader_async.cpp), [test CLI/bridge](../../tests/test_flash_request_memory.py). |
| Attention et récurrence | RoPE/fréquences, masques, GQA/SDPA, QSA, GDN, convolution et histories, PLE/HC. Conserver précision par étape, état de chaque couche et position de reprise. | [GDN](../../native/src/gated_delta.rs), [Flash](../../native/src/qwen4.rs), [LFM](../../native/src/lfm2.rs), [Ling](../../native/src/ling.rs). |
| Tête et sorties | Embeddings compactés et usuels, dernier hidden, tête complète ou restreinte, logits/probabilités, RNG/stop, masquage structuré. Le sampler et le contrôle des outils peuvent rester côté CPU. | [Sampling et génération](../../native/src/main.rs), [contrats de sortie structurée](../../native/src/structured.rs), [tests du bridge](../../tests/test_bridge_benchmark.py). |

## Distribution et conversion

| Entrée | Dépendance actuelle et propriété à vérifier | Contrôle |
|---|---|---|
| Serveur OpenAI, streaming, outils/MCP et Desktop | Le moteur fournit la génération. Le serveur et le Desktop doivent être qualifiés en cas d’interruption, de réponse incomplète, de changement de modèle et de redémarrage. | [Protocole/régressions](../../.github/workflows/check.yml), [bridge Swift](../../tests/bridge-dispatch-check.swift) |
| Installation et mise à jour moteur | Le build dépend du wheel et de `MLXL3_MLX_ROOT`. L’archive et le validateur Desktop exigent des bibliothèques MLX et un champ de capacité. Migrer le schéma et tester ancienne → nouvelle version ensemble. | [build.rs](../../build.rs), [paquet](../../scripts/package_engine.py), [validateur Desktop](../../apps/MLXL3Studio/Sources/MLXL3Studio/EngineRuntimeStore.swift), [tests d’archive](../../tests/test_package_engine.py) |
| Quantization Studio et convertisseur | Python/PyInstaller embarque MLX, mlx-lm et PonyEXL3. Porter recherche/calibration, rotations, transformations des poids et packing en conservant tie-break, passes, format, arrêt et reprise. | [Paquet du convertisseur](../../scripts/build-quantizer.sh), [kernels](../../src/mlxl3_quantizer/metal.py), [tests studio](../../tests/test_quantization_studio.py), [packing](../../tests/test_quantize_packing.py) |

## Inventaire de sources et CI

`python scripts/check_native_coverage.py` compare les sources de l’arbre Git et la [référence revue](mlx-sites.json). Les appels des wrappers Array, les 54 symboles FFI déclarés et définis dans la base, les appels C++ qualifiés, les références de build/distribution et les appels Python importés (alias compris) sont ventilés par fichier et nom. Les nombres par méthode Rust peuvent inclure des homonymes sur d’autres types. La référence garde 199 fichiers, dont 75 dans le sous-bundle PonyEXL3. Le catalogue Rust distingue 148 clés de configuration et 79 références littérales de shader ou de kernel. Le répertoire benchmarks est inclus car il est sur le chemin d’import du convertisseur. Les sources de PonyEXL3 embarquées sont inspectées, y compris les sous-modules collectés par PyInstaller. Les résultats et anciennes références sous `docs` restent hors de cette barrière.

Pour modifier la référence, inspecter la différence produite par `--inventory` et faire revoir les ajouts et suppressions. Les tests injectent un nouvel appel avec alias, un nouveau fichier Rust et la disparition d’un ancien site. Un inventaire lexical ne mesure pas les appels dynamiques, les allocations du pilote ou une bibliothèque réellement chargée. Le contrôle final N8 exige aussi les dépendances Mach-O, le contenu de chaque bundle et un parcours réel en environnement sans MLX.

**N0 reste ouvert.** La campagne #76 a produit les fixtures et parcours Flash cités. Les lignes des autres familles affectent des tests existants, dont certains couvrent seulement un composant. Il reste à figer des fixtures indépendantes et des traces par checkpoint/mode pour les lignes sans preuve de bout en bout. Pour une frontière exacte, vérifier forme, dtype, valeurs finies, octets et première divergence. Pour une frontière à tolérance, fixer la tolérance et la métrique spécifiques au checkpoint avant de mesurer. Le lancement des autres modèles devra respecter les gardes M5 et le budget des poids/drafts.

Contrôles locaux de ce premier inventaire. Les quatre tests de régression passent, dont 25 paires d’inventaires finis et le parcours d’un dépôt temporaire complet. Avec les tests voisins d’archives moteur et de Quantization Studio, 86 tests passent. Lors de la première invocation de cette suite élargie, le Python partagé résolvait un autre checkout du paquet. Le contrôle a été repris avec le `src` et le PonyEXL3 de ce worktree dans `PYTHONPATH`. Ruff et la compilation Python passent. CrossHair sur `differences` avec un budget de 20 secondes ne confirme pas le contrat sur les dictionnaires arbitraires. Cette phase n’a lancé aucun modèle.

### Baselines disponibles sur ce Mac

- **Flash-Next** réutilise les quatre captures MAIN/HOLD et la qualification #76. Aucune génération supplémentaire dans N0.
- **Qwen dense 27B**. Deux shards et dix fichiers d’identité correspondent au manifeste historique. Les 1 548 fichiers des douze captures sont intacts. Chaque capture contient 128 états et 248 320 logits FP16 finis. Sur le binaire R5, les quatre étapes du prefill de 23 tokens ont redonné les octets du golden. Le superviseur a arrêté le processus sur le plafond cumulatif de swap. Les huit étapes suivantes n’ont pas été confirmées sur R5. [Empreintes récupérées](baselines/qwen-dense-fixtures.json) · [Quatre comparaisons physiques](baselines/qwen-dense-partial.log) · [Arrêt et maxima](baselines/qwen-dense-frozen.json).
- **Qwen MoE 35B**. Les deux shards correspondent à la référence du test de commutation avec MTP, qui couvre douze cas sur ce modèle. Le registre conserve séparément ces résultats historiques et l’état de la campagne actuelle. [Vérification du modèle](baselines/local-checkpoints.json) · [Essai MTP historique](../measurements/engine-v1.4.1/model-switch-validation.json).
- **LFM MoE 24B V3**. Les quatre shards locaux et les 29 816 entrées de l’index correspondent au modèle final après compactage de la tête. Le test historique trouvait 16 384 000 logits FP16 identiques avant/après ce compactage. L’évaluation de qualité de la quantification V3 a échoué à son propre critère. Ces deux contrôles portent sur des propriétés différentes. [Rapprochement des preuves](baselines/model-evidence.json) · [Compactage](../measurements/quant-lfm25-v3-20261015/head-compact.json) · [Évaluation V3](../measurements/quant-lfm25-v3-20261015/quality.json).

Les cinq processus prévus concernaient le contrôle Qwen dense puis les captures CLI sérielles et batchées de Qwen MoE et LFM MoE. Le premier a été rejoint avec SIGTERM après 11,585 s, à 523 632 640 octets de swap système écrit depuis l’ancre fixe, au-dessus du plafond de 33 554 432 octets. Les quatre autres n’ont pas été lancés. Les bruts de fixtures et les poids restent locaux. Les [commandes de contrôle](baselines/guarded-capture.py.txt) et [vérifications de fichiers](baselines/verify-files.py.txt) précisent les données utilisées.

#### Capture Qwen MoE à deux tokens

Sur le M5 24 Go, le CLI R5 a traité successivement les IDs 1 et 2 du Qwen MoE 35B. Chaque pas fournit un vecteur de 248 320 logits FP16 finis en ordre de vocabulaire. Les deux fichiers gzip conservent les mots FP16 little-endian, à comparer après décompression aux SHA-256 du reçu. Ce golden provient du moteur MLX de référence. Le CLI ne fournit pas les états internes Qwen dans ce mode. [Premier token](baselines/qwen-moe-two-tokens/swap-observed/token-1.f16le.gz) · [Second token](baselines/qwen-moe-two-tokens/swap-observed/token-2.f16le.gz) · [Reçu](baselines/qwen-moe-two-tokens/swap-observed/qwen-moe-serial.json).

Après 60,317 s de compteurs stables, le premier essai a dépassé 12 Go d’empreinte, puis le second a dépassé 32 Mio de swap écrit. Les deux processus ont été rejoints avant le premier logit. À la demande de l’utilisateur, la troisième exécution mesure le swap sans seuil d’arrêt, depuis la même ancre. Elle termine en 5,867 s, avec 209 Mio de swap système écrit cumulé et 12,405 Go d’empreinte maximale échantillonnée. Durée de capture, démarrage et chargement compris. [Ancre](baselines/qwen-moe-two-tokens/anchor.json) · [Essai 12 Go](baselines/qwen-moe-two-tokens/qwen-moe-serial.json) · [Essai 14 Go](baselines/qwen-moe-two-tokens/retry14/qwen-moe-serial.json) · [Superviseur utilisé](baselines/qwen-moe-two-tokens/observe-swap.py.txt).

#### Layouts disque et tuiles EXL3

Les en-têtes, formes, dtypes et octets couvrent tous les shards des quatre cibles et les sidecars Flash. L’index conserve aussi les drafts Qwen MTP dense/MoE et Flash DFlash/DSpark présents. Chaque payload a été parcouru par intervalles contigus, avec vérification du nombre d’octets pour sa forme et son dtype. Le DFlash packé Qwen MoE est distribué séparément. L’installation draft contient huit des 56 fichiers du manifeste, tous vérifiés par taille et SHA-256. Les 48 autres concernent le target, le tokenizer et la vision. Le CLI a reconnu cette installation. [Layouts](baselines/tensor-layouts.json) · [Vérificateur](baselines/index-layouts.py.txt) · [Manifeste DFlash vérifié](baselines/qwen-moe-dflash.json) · [Inspection CLI](baselines/qwen-moe-dflash-inspection.json).

Les fixtures de composant conservent les mots compactés d’une projection par cible. Trois tuiles par projection couvrent la première, la centrale et la dernière. Le décodeur de référence NumPy produit 256 états et 256 valeurs FP16 finies par tuile, en ordre kernel avant permutation, Hadamard et échelles. Le CLI Rust reproduit les 3 072 états et les 3 072 valeurs bit à bit. Une modification d’un bit dans chacune des douze tuiles déclenche un écart. Le test Rust relit ces fixtures sur Linux et macOS, sans les poids. [Fixtures et provenance](baselines/exl3) · [Test de régression](../../native/tests/contracts.rs).

| Cible | Projection | Bits | Codebook |
|---|---|---:|---|
| Flash-Next | couche 0, expert 0, gate | 2 | Mul1 |
| Qwen dense | couche 0, gate | 1 | Mul1 |
| Qwen MoE | couche 0, expert 0, gate | 3 | Mcg |
| LFM MoE | couche 10, expert 0, w1 | 4 | Mcg |

Sur le M5 physique, le CLI compilé avec Metal direct a reproduit les mêmes 3 072 états et 3 072 poids FP16. Les 192 projections sur vecteurs de base 16×16 ont rendu les 3 072 coefficients attendus après permutation, bit à bit. Une requête invalide en début de flux a été rejetée et les 204 requêtes suivantes ont abouti dans le même processus. Le binaire charge Metal et aucune des 706 images signalées par dyld ne correspond à MLX. Cette comparaison couvre le décodeur et la projection interne EXL3 sur ces tuiles. Le Hadamard, les échelles externes et l’inférence du modèle complet restent à qualifier séparément. [Résultat physique](baselines/exl3/physical-metal.json) · [Rejeu CLI](baselines/exl3/check-metal-tiles.py.txt).

### Préparation H128 en Metal direct

L’opération JSONL `metal-prepare128` du CLI `codec` applique les échelles FP16 puis le Hadamard normalisé par blocs de 128 sur des lignes FP16 contiguës. Le champ `cols` indique la largeur de ligne, `x` contient les mots FP16 et `data` contient un facteur par colonne ou un facteur par élément. La sortie est un tableau de mots FP16. Une largeur positive multiple de 128 est nécessaire, même pour un lot vide. Ce composant est disponible dans le CLI Metal direct. L’intégration à l’ordonnanceur d’inférence et la prise en charge des vues stridées / entrées FP32 de Flash restent ouvertes. [Kernel](../../native/shaders/exl3.metal) · [Test physique](../../tests/physical_prepare128.py).

Le binaire de release a reproduit 173 184 sorties FP16 finies sur M5 en 17 cas de comparaison avec MLX 0.32.2, dont les 128 vecteurs de base, les largeurs 128/640/2 560, les lots 1/3/20, des sous-normaux et des zéros signés. Trois requêtes mal formées sont refusées, puis le même processus accepte un lot vide et les entrées valides. Supprimer l’arrondi FP16 au quatrième étage provoque 53 écarts sur 128 dès le premier cas mixte. Le harnais Kani vérifie les bornes du nombre de blocs et de l’étendue des échelles côté hôte sur tailles symboliques, avec quatre cas atteints. [Rejeu de release](h128/release.json) · [Mutation détectée](h128/removed-rounding.json).

### Projection résidente Flash en Metal direct

`metal-checkpoint-linear` dans le CLI `codec` prend `path`, `prefix` et les activations FP16 `x`. Le loader lit le treillis dans un buffer Metal depuis le fichier inspecté et charge les échelles et le biais éventuel. La préparation, le QMV avec l’ordre de réduction Flash et l’épilogue utilisent une commande Metal et deux buffers temporaires privés. Le résultat asynchrone garde sa sortie accessible après la destruction de la couche. Ce chemin couvre les projections Flash FP16 contiguës, K1 à K8, avec un ou deux splits. Le scheduler du modèle complet reste à connecter.

Sur M5, l’oracle de projection Flash a produit 17 792 valeurs FP16 bit à bit identiques pour les 24 cas de format et deux matrices réelles de la couche 0. Les trois codebooks, les échelles F16/BF16/F32, les noms modernes et historiques, le biais, les lots de trois et quatre erreurs suivies de succès sont couverts. Le processus candidat charge Metal sans image MLX détectée. Les comparaisons H128 et codec de #77/#78 restent identiques après extraction des corps de shader partagés. [Résultats](projection/physical.json) · [Mutation de biais](projection/missing-bias.json) · [Sources et commandes](projection/provenance.json).

### Expert Flash résident en Metal direct

Le CLI `codec` accepte `metal-checkpoint-expert` avec le répertoire du checkpoint, le préfixe de l’expert et `x` en mots FP16. Les projections gate/up partagent le même upload d’entrée. Le SwiGLU FP16 et le down consomment directement les résultats GPU, avec dix encodeurs tracés dans le code et une seule soumission. Une erreur de forme est renvoyée avant le premier upload. Une erreur post-encodage abandonne la commande sans publication. Les experts restent résidents pendant l’exécution. Le routage, le mélange d’experts et la lecture SSD à la demande sont des étapes distinctes.

Les 63 488 gates FP16 finis avec six multiplicateurs fixes donnent 380 928 valeurs identiques au SwiGLU compilé. La comparaison de release couvre 32 cas et 38 784 valeurs, dont 20 480 sorties des experts routé et partagé du premier bloc réel en lots de un et trois. Une commande non soumise, deux sorties lues en ordre inverse après destruction des handles et l’abandon d’une autre sortie sont exercés sur le M5. [Comparaisons physiques](expert/physical.json) · [SwiGLU](expert/swiglu.json) · [Mutation détectée](expert/missing-rounding.json).

### Vues natives des activations Flash

Les entrées de préparation, de projection et d’expert acceptent un backing immuable F16/F32 et une vue matricielle en éléments (forme, offset, strides signés). Le kernel convertit les sources F32 en FP16 avant le produit par les échelles. Les buffers privés du gate/up/down restent sur le GPU. En cas de dimension vide, la préparation accepte un offset juste après le dernier élément ; une projection exige au moins une ligne. Les layouts transposés, inversés et diffusés sont vérifiés sur le M5. [Préparation](views/prepare.json) · [Chaîne sur checkpoint](views/chain.json) · [Mutation du stride](views/missing-row-stride.json).

### Classement des experts sur Metal direct

Le CLI `metal-router` consomme une vue de scores FP16 ou FP32 et rend les indices des experts et les bits de leurs scores. L’API Rust permet au lecteur SSD de transférer seulement les IDs ; les scores publiés après succès peuvent alimenter une autre file Metal. Le contrôle indépendant compare les 80 cas au tri scalaire, au routeur MLX courant et, pour FP16, au binaire d’avant extraction du kernel. Les 32 940 indices sont exacts, ainsi que les 28 136 scores finis et le signe des infinis. Pour les NaN, le test compare la classe. [Comparaison](router/physical.json) · [Mutation des égalités](router/reversed-tie.json) · [Périmètre et références](router/provenance.json).

### Routeur Flash avec les poids du checkpoint

`metal-checkpoint-router` prend le préfixe d’une gate Flash dense [512,2560] sans biais et une vue d’activation FP16/F32. La projection, le softmax précis et les dix scores ordonnés utilisent la même commande Metal. Le diagnostic `return_raw` permet de lire les logits et les probabilités. Trois gates du checkpoint Flash (couches 0 et 23, MTP) et une gate synthétique ont été qualifiées sur entrées imposées. [Comparaison physique](gate/physical.json) · [Divergence de normalisation](gate/widened-denominator.json) · [Provenance et domaine](gate/provenance.json).

### Bloc MoE Flash depuis les poids réels

Le CLI direct exécute les dix experts routés, l’expert partagé et sa gate pour une ligne de 2 560 activations. Pour essayer la couche 0 sur un Mac Apple Silicon, compiler `cargo build --locked --release --features chat`, puis utiliser le chemin du checkpoint Flash et celui du binaire produit. Les deux lignes identiques partagent leur cache d’experts.

```sh
python3 -c 'import json,sys; print(json.dumps(dict(op="metal-checkpoint-moe",path=sys.argv[1],prefix="model.language_model.layers.0.mlp",slots=10,x_f32=[(i%73-36)/53 for i in range(2560)],x_layout=dict(shape=[2,2560],offset=0,strides=[0,1]))))' "$MODEL" |
  "$CLI" codec |
  python3 -c 'import json,sys; print([(len(r["values"]),r["hits"],r["read_bytes"]) for r in json.load(sys.stdin)["data"]])'
```

Résultat attendu en (coefficients, experts en cache, octets lus) : `[(2560, 0, 12480120), (2560, 10, 0)]`. Cette commande accepte des activations numériques. Le parcours prompt → tokens exige aussi l’embedding, l’attention, les états récurrents, les 48 couches et la tête de sortie. [Comparaison physique](moe/physical.json) · [Erreurs détectées et reproduction](moe/provenance.json).

### Sous-couche MLP Flash, du flux d’entrée au résidu

Dans le CLI `codec`, `metal-checkpoint-mlp` prend `path`, un préfixe de couche tel que `model.language_model.layers.0`, et une vue `x_layout` de forme [1,10240] sur `x_f32` ou des mots FP16 `x`. Le résultat contient le résidu des quatre streams, les IDs des dix experts, le vecteur mélangé et les quatre gates. `return_raw` permet aussi de comparer les normalisations, les deux projections HC et l’activation. `metal-checkpoint-hc` permet de lire un mélangeur isolé, dont le mélangeur final du modèle.

Sur M5, huit lignes réelles ou imposées en F16/F32 reproduisent les 230 464 mots et indices de référence, dont les 40 960 valeurs du résidu après le MoE de la couche standard et MTP. Six captures indépendantes d’écriture couvrent également 61 440 valeurs avec branches F16 et F32. Le test de sous-couche reçoit des activations numériques ; la génération à partir d’un prompt attend encore l’attention et l’assemblage du modèle. [Rejeu physique](hc/physical.json) · [Protocole et contrôles rejetés](hc/provenance.json).

### Couche Flash à attention récurrente

Le CLI direct accepte `metal-checkpoint-gdn-layer` avec le chemin du modèle, le préfixe d’une couche GDN sans PLE et une vue F16/F32 de largeur 10 240. Il renvoie les quatre flux FP32 après l’attention récurrente et le MoE pour chaque ligne. Une requête conserve les états entre ses une à huit lignes. `metal-checkpoint-gdn` accepte le préfixe `.linear_attn` et une largeur 2 560. Le mode `return_raw` de cette opération permet d’inspecter les états et les activations.

Sur le M5, cinq entrées successives de la couche 2 reproduisent 4 375 520 valeurs de calcul et de cache. Les six résidus complets de la couche 0 reproduisent 61 440 valeurs FP32. Le test de reprise force une panne dans le MLP après l’attention puis compare le prochain résidu sur une branche restaurée. [Résultats numériques](gdn/physical.json) · [Preuves ciblées, témoins négatifs et commandes](gdn/provenance.json).

### Couche Flash avec embeddings de n-grammes

`metal-checkpoint-ple-layer` accepte le préfixe `model.language_model.layers.1`, des streams de largeur 10 240 et un ID de vocabulaire par ligne dans `token_ids`. Chaque pas exécute PLE, GDN puis le MLP. `metal-checkpoint-ple` isole la branche PLE avec le préfixe `model.language_model.layers.1.ple` et expose ses stades en mode `return_raw`. Les mots des n-grammes sont lus dans les 128 shards du checkpoint et conservés dans le cache partagé avec le lecteur historique.

Cinq captures indépendantes vérifient 832 040 valeurs, avec tous les états de l’historique. Deux séquences de douze pas en FP16 et FP32 comparent 2 703 360 valeurs supplémentaires de l’historique, de la sortie PLE et du résidu complet de la couche. La reprise après lecture invalide est exercée au dernier head de PLE puis à l’intérieur du MoE. [Rapport numérique](ple/physical.json) · [Provenance et mutation de la convolution](ple/provenance.json).

### Attention QSA Flash en contexte court

Les opérations de diagnostic `metal-checkpoint-qsa` et `metal-checkpoint-qsa-layer` évaluent des lignes successives du même contexte (1 à 512 tokens). La première accepte les 2 560 activations d’attention et la deuxième les 10 240 valeurs du résidu. La sélection de blocs sparse commence au-delà de cette plage. Un snapshot reste rejouable après l’écriture d’un suffixe divergent.

Les couches 15 et 35 reproduisent les captures indépendantes. Les 512 positions de deux séquences FP16 et FP32 et quatre sorties de la couche 3 par dtype correspondent à la référence de production. [Comparaisons et empreintes](qsa/physical.json) · [Mutation, sources et vérification de l’append](qsa/provenance.json).

### Tester Flash avec un prompt sans MLX

Sur un Mac Apple Silicon, compiler la CLI en release avec `cargo build --locked --release --features chat`. Utiliser ensuite le chemin du checkpoint local ou son nom dans le registre.

```sh
MLXL3_SSD_CACHE_GIB=0.6 target/release/mlxl3-rs run /chemin/vers/Flash-Next-exl3 --prompt 'Combien font 2 + 2 ?' --max-tokens 256
```

Sans `--prompt`, le terminal ouvre une conversation. `/clear` efface l’historique et `/quit` quitte. Le moteur affiche la réflexion et la réponse au fur et à mesure. Cette première version qualifie la génération gloutonne sur les 48 couches de Flash, à concurrence de 512 positions au total, template et sortie compris. Le budget de 0,6 Gio choisit dix slots SSD par couche. Les modes spéculatifs, les contextes longs et les parcours Studio suivent leurs qualifications séparées. La commande `forward` accepte aussi une séquence de tokens imposés, ou `--batch` pour les logits du dernier.

Le prompt français du rapport se termine par « 4 » après 77 tokens sur les deux binaires. Sur quatre tokens imposés, tous les 993 280 logits sont identiques, ainsi que les 248 320 logits du batch. [Rapport physique](flash-ar/physical.json) · [Mutation, reprise et preuves](flash-ar/provenance.json).

### Flash AR au premier bloc écarté

La sélection s’active au 513e bloc complet, à la position 2 052. Elle conserve les 512 blocs les mieux classés, les remet dans l’ordre chronologique, puis ajoute les tokens du bloc incomplet. À égalité de score, les blocs les plus récents passent en premier à la coupure. Le cache double de capacité à la demande. Un préfixe restauré conserve ses clés, valeurs, blocs compressés et la queue de l’index.

Sur le M5, une couche a parcouru 4 096 positions, avec comparaison des projections d’index, des scores, des positions, des sorties et du cache complet à la dernière position. La même suite a aussi exercé les variantes Max et Ultra imposées sur ce M5. La CLI a traversé les 48 couches sur une entrée de 2 053 tokens avec 248 320 logits finaux identiques. La limite configurée est de 262 144 tokens. [Détail par tenseur](qsa-long/layer.json), [parcours release complet](qsa-long/whole-model-release.json), [conditions, preuves CPU et tests](qsa-long/provenance.json).

Pour reproduire le parcours complet avec deux exécutables release, lancer la commande suivante sur un Mac Apple Silicon avec le checkpoint Flash.

```sh
python tests/physical_flash_run.py DIRECT STOCK CHECKPOINT NOUVEAU_RAPPORT --long-only
```

### Export des états du modèle Flash

La CLI native accepte `forward --states`. Pour chaque token imposé, elle émet les logits FP16, les 48 sorties de couche et les 36 états récurrents GDN au format de la CLI MLX. Les noms, l’ordre, les formes et les octets sont comparés sur la même entrée. En mode `--batch --states`, Flash renvoie les derniers logits et une liste d’états vide, conformément au protocole existant. Un instantané interne conserve aussi la position, la dernière ligne cachée, les historiques PLE et les caches QSA pour reprendre la génération. [Empreintes par tenseur](flash-state/initial.json), [validation sur la branche de contexte long](flash-state/release.json).

```sh
python tests/physical_flash_trace.py --cli DIRECT --stock STOCK --model CHECKPOINT --report NOUVEAU_RAPPORT
```

### Tête MTP Flash et vérification des propositions

Le moteur Metal peut charger la tête MTP intégrée au checkpoint Flash et vérifier une proposition à la fois. Sur dix pas conservés par l’oracle indépendant, les quatre flux mixtes, la sortie QSA, le mélange final, les 248 320 logits par pas et les caches capturés concordent bit à bit. La mise à jour du cache seule reproduit aussi les blocs comprimés de la tête complète. Les quatre limites de sortie testées sont 1, 2, 3 et 17. Pour 17 tokens, 5 des 12 propositions ont été acceptées. La continuation après refus ou après une lecture interrompue retrouve l’état AR. [Empreintes et transcriptions](flash-mtp/replay.json).

Pour rejouer, définir `MLXL3_FLASH_MODEL` et `MLXL3_FLASH_MTP_REFERENCE` (répertoire produit par `scripts/check-flash-reference.py --mtp-only`). Extraire l’objet `session.oracle` du rapport vers un fichier JSON et en fournir le chemin dans `MLXL3_FLASH_MTP_SESSION_ORACLE`. Utiliser `MLXL3_SSD_CACHE_GIB=0.6`. Exécuter chaque test isolément.

```sh
cargo test --locked --lib --features chat gpu::flash::mtp::draft_head_replays_retained_steps_and_vocabulary -- --exact --ignored
cargo test --locked --lib --features mlx,chat qwen4::tests::flash_mtp_session_preserves_greedy_sequence_target_and_head_states -- --exact --ignored --nocapture
cargo test --locked --lib --features chat gpu::flash::mtp::speculative_commits_match_stock_sequence_and_ordinary_state -- --exact --ignored

### Bridge Studio et API HTTP Flash sans MLX

La commande `bridge` utilise le même rendu des messages, streaming de raisonnement et réponse, filtre d’outils, orchestration MCP et validation de schéma que le chemin de référence. Le modèle Flash, l’AR et la tête MTP sont exécutés par le moteur natif. Les capacités annoncées suivent les modes réellement disponibles. `runtime-info` identifie explicitement les backends `metal`, `mlx` et `cpu`.

Les dialogues AR/MTP/MTP/AR sur deux limites de sortie produisent les mêmes hashes de tokens que le bridge MLX indépendant. Les changements de tête, requêtes invalides, pénalités et annulations à deux phases retrouvent une génération identique après reprise. Le serveur HTTP réel vérifie aussi l’authentification, les erreurs, la parité JSON/SSE, les compteurs et l’arrêt des processus. Le relevé de mémoire sépare les allocations Metal du footprint processus. [Résultats et limites](flash-bridge/checks.json), [événements natifs](flash-bridge/native.jsonl), [événements de référence](flash-bridge/stock.jsonl).

```sh
mlxl3-rs bridge CHECKPOINT --context-length 4096
mlxl3-rs serve CHECKPOINT --host 127.0.0.1 --port 8000
python scripts/check-mtp-bridge.py DIRECT CHECKPOINT CHECKPOINT --tokens 2,17 --context 256
MLXL3_FLASH_MODEL=CHECKPOINT cargo test --locked --features chat --test openai_transport native_flash_serve_streams_the_same_completion_as_json -- --exact --ignored --nocapture
```
