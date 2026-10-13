# LFM2.5-24B-A2B → EXL3 sur Apple Silicon

Le convertisseur `scripts/quantize_lfm25_moe.py` prend en charge le checkpoint
natif [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B),
épinglé à `1769516ac588c4c558089c94f27d96b0dc67b183`. Son index contient
7 610 tenseurs, dont **7 402 projections à quantifier** : 100 projections
convolution/attention, six projections denses et 7 296 projections d'experts.
Les deux premières couches sont denses ; les 38 suivantes comportent 64 experts.
Le source occupe 47 943 180 288 octets. Les métadonnées, leur empreinte et la
configuration sont conservées dans [source.json](measurements/quant-lfm25-20261013/source.json).

**La conversion complète et la qualité du 24B n'ont pas été mesurées.** La PR
valide le parcours sur un checkpoint miniature jetable et le packing sur GPU
physique. Aucun poids du 24B n'a été téléchargé ou publié pendant cette tâche.

## Ce qui accélère la conversion

- Le search Metal MLXL3 existant conserve les deux passes tail-biting, les
  codebooks, les comparaisons FP32 et le tie-break original.
- Un kernel Metal propre assemble directement les mots du treillis EXL3,
  K=1..8, sans tableaux MLX intermédiaires de masques, shifts et réductions.
  Le format, les hauts bits ignorés et le swap des mots restent identiques.
- À précision entière, sans tête de sortie indépendante, le convertisseur
  saute la mesure de candidats qui ne peut changer aucun choix. Le défaut de
  `head_bits` ne force plus une mesure de toutes les projections du LFM lié.
  Les précisions fractionnaires conservent leur attribution mesurée.
- Les experts `w1/w2/w3` utilisent la récursion LDLQ groupée existante : au
  plus quatre down ou huit gate/up par groupe. Chaque projection conserve sa
  propre recherche de scale ; le scale Ling 0,908 et sa grille groupée restent
  exclus de LFM.
- La calibration charge une couche BF16 à la fois. Les gate/up partagent les
  mêmes lignes observées par hardlinks ; chaque down est calculé avec les
  propres poids gate/up de cet expert. Le cache de lecture garde huit fichiers.
  Les routeurs et biais FP32 sont conservés, les embeddings restent liés.

Il faut télécharger le source entier sur disque pour calibrer la trajectoire
réelle. La calibration n'évalue pas tous les poids simultanément. Les captures
down de 2 048 lignes occupent environ 14,25 Gio logiques pour les 2 432 experts ;
prévoir aussi le source, les shards intermédiaires et leur finalisation. Ce
calcul de stockage n'est pas une mesure de RAM ni une garantie de pic mémoire.

## Préparer et lancer

Utiliser la venv normale du dépôt avec ses dépendances épinglées. Le convertisseur
PonyExl3 est une dépendance de développement facultative ; l'inférence native
ne l'importe pas. Sur un **nouveau** checkout du convertisseur :

```sh
git clone https://github.com/beamivalice/PonyExl3.git references/PonyExl3
git -C references/PonyExl3 checkout 8e7fa6b1556f59fc669e25087903b279b9b0346f
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-mlxl3-conversion.patch
```

Ne pas empiler ce patch sur une ancienne version déjà patchée : utiliser un
checkout neuf ou comparer explicitement son diff. Le patch consolidé contient
aussi les optimisations historiques ; conserver les travaux locaux existants.

```sh
.venv/bin/hf download LiquidAI/LFM2.5-24B-A2B \
  --revision 1769516ac588c4c558089c94f27d96b0dc67b183 \
  --local-dir models/source/LFM2.5-24B-A2B

PYTHONPATH=src .venv/bin/python scripts/quantize_lfm25_moe.py \
  --in-dir models/source/LFM2.5-24B-A2B \
  --out-dir models/LFM2.5-24B-A2B-EXL3-4bpw \
  --work-dir build/lfm25-24b-4bpw \
  --bits 4 --codebook mcg --search-backend metal \
  --calibration-rows 2048 --calibration-seq-len 1024 --max-workers 1
```

Cette commande choisit 4 bits pour les projections quantifiées, pas pour tous
les paramètres du checkpoint. Les embeddings, normes et convolutions non
linéaires restent hors EXL3. Une précision entière Metal entre 2 et 8 évite la
passe d'attribution ; la qualité respective de ces recettes sur le 24B reste
à évaluer. Aucun enregistrement de modèle ni upload n'est automatique.

Relancer la même commande reprend les shards terminés. La calibration sur
disque exige un marqueur complet correspondant au source, corpus et dimensions ;
une interruption ne crée pas ce marqueur. `--module-limit` est réservé aux
pilotes : un checkpoint partiel ne constitue pas un modèle utilisable.
Après conversion complète, vérifier la couverture, la qualité et le chat :

```sh
PYTHONPATH=src .venv/bin/python -c \
  'from mlxl3_quantizer import validate_checkpoint_files; validate_checkpoint_files("models/LFM2.5-24B-A2B-EXL3-4bpw")'
target/release/mlxl3-rs run models/LFM2.5-24B-A2B-EXL3-4bpw \
  --prompt 'Explique brièvement ce que fait une mixture of experts.' --max-tokens 128
```

## Mesures du 13 octobre 2026

Apple M5, 24 Gio, MLX 0.32.2, NumPy 2.5.2, Python 3.12, secteur/batterie en
charge, pmset sans avertissement. Température physique non mesurée. Aucun
autre travail GPU, compilation ou proveur lancé par cette tâche pendant les
mesures. Les groupes ci-dessous ne doivent pas être additionnés.

| Périmètre | Baseline | Candidat | Résultat |
|---|---:|---:|---|
| Packing K4, 768 tiles | 257,99 µs | 224,10 µs | 1,151×, packing seul |
| Packing K5, 768 tiles | 433,62 µs | 225,51 µs | 1,923×, packing seul |
| Packing K4, 96 tiles | 226,13 µs | 222,44 µs | pratiquement inchangé |
| LDLQ synthétique 1536×2048, K4 | 1,136759 s | 1,130932 s | −0,51 %, vitesse non concluante |
| CLI miniature, 23 projections K4 | 2,258758 s | 2,089908 s | −7,48 % sur ce pilote uniquement |
| Modèle 24B complet | non mesuré | non mesuré | aucun gain global établi |

Packing : trois warmups exclus, sept paires alternées, vingt appels
synchronisés par observation, oracle CPU et hashes identiques. Les mesures
incluent dispatch hôte et attente GPU, pas uniquement le temps GPU matériel.
Les douze cas sont dans [packing-speed.json](measurements/quant-lfm25-20261013/packing-speed.json).
Reproduction :

```sh
PYTHONPATH=src:references/PonyExl3 .venv/bin/python benchmarks/bench_quantize_packing.py \
  --output build/packing-speed.json --repeats 7 --iterations 20
```

LDLQ : shape d'un down LFM, poids/Hessienne synthétiques seed93, Hessienne
192 lignes +0,1I, warmup exclu puis trois paires. Les poids et reconstructions
sont exacts. Les pics actifs MLX des trois répétitions archivées culminent à
**333 399 064 / 333 400 072 octets**, référence/candidat ; aucun gain mémoire
établi, RAM processus non mesurée. Le journal corrige le relevé initial,
non conservé dans ces bruts hors warmup.
[Données](measurements/quant-lfm25-20261013/packing-ldlq.json).

CLI : checkpoint BF16 synthétique seed57 de trois couches/128 dimensions/deux
experts, mêmes tenseurs, corpus et paramètres sur les deux chemins. Warmup exclu
puis trois paires alternées ; temps de processus complet, incluant calibration,
compilation Metal propre au processus, recherche, LDLQ et export. Le chemin
mesuré traite 23 candidats ; le chemin uniforme zéro. Tous les tenseurs des
huit passages sont identiques. [Données et configuration](measurements/quant-lfm25-20261013/uniform-cli.json).

## Vérification et limites

Le test CLI convertit les 23 projections, valide les markers/scales/biais FP32,
compare tous les tenseurs au chemin mesuré, reprend sans changer les shards,
puis exerce le bridge natif dans un registre jetable : quatre tokens « c c c c ».
La capture couche par couche est comparée au modèle MLX BF16 complet sur
16 et 129 tokens par séquence, RMSE relative <0,5 %, avec formes/finitude et
les propres entrées down de chaque expert. Ce contrôle limité ne mesure pas
la perplexité du nouveau checkpoint.

Le packing est comparé à l'oracle CPU indépendant pour chaque position et bit
utile, K1..8, hauts bits, lots multidimensionnels/non contigus, tableaux vides
et erreurs. Le vrai parcours LDLQ compare aussi poids et reconstructions.
Une mutation du swap est détectée ; la régression uniforme détecte le garde
précédent. Les erreurs/annulations du benchmark conservent un rapport
`failed`, avec `parity=false`.

CBMC 6.11.0 : **53 obligations réussies**, K1..8 symbolique, 512 valeurs uint16
symboliques/deux tiles et tous les mots valides ; loops bornées à17/513 avec
assertions d'unwinding et contrôles pointeurs/bornes/overflow/shifts. Le harnais
inclut **la source Metal originale**, sous une vue C++ qui retire l'espace
`device` et définit les types entiers. Il ne modélise pas les adresses GPU ni
son ordonnancement. Les six goals explicites de couverture sont atteints dans
une commande distincte. ASan/UBSan passent la vérification hôte de tous les
bits/positions, y compris les hauts bits.

CrossHair 0.0.101 a tenté les deux contrats du garde uniforme réel : aucun
contre-exemple, conditions non confirmées en15 s chacune. **Python non prouvé.**
Deux contrats supplémentaires exercent le rejet d'une matrice de benchmark vide
(répétitions/itérations symboliques1..8) et de K hors1..8 ; l'outil les signale
« Confirmed over all paths ». Cela reste une recherche symbolique, pas une
preuve déductive. L'ancien harnais sans garde reproduit une fausse parité avec
zéro cas ; les nouveaux tests refusent cette entrée avant toute utilisation GPU.
Les logs conservent les premiers échecs de compilation/harnais et leurs
corrections. CI Linux vérifie CBMC ; macOS hébergé vérifie les contrats CPU et
saute explicitement les contrôles Metal physiques.

Les sources finales, le patch réel PonyExl3, la version du harnais lors des
mesures, le moteur natif et le wheel sont identifiés par SHA-256 dans
[source-manifest.json](measurements/quant-lfm25-20261013/source-manifest.json).
Base Git : `b4dfd3915e262b2d33581102ae3320042acee5c9`, worktree isolé ; les
modifications préexistantes du checkout personnel ne sont pas reprises.
Le kernel numérique n'a pas changé après les mesures. Le harnais de packing
a reçu uniquement les gardes de paramètres. Les pilotes ponctuels LDLQ/CLI
ne sont pas conservés comme scripts exécutables : leurs seeds, options,
configurations et hashes de résultats restent dans les rapports bruts.

Commandes de contrôle utilisées, avec le Python de la venv du dépôt principal
et le CBMC6.11.0 extrait dans `build/proofs/cbmc/6.11.0/bin/cbmc` :

```sh
ruff format --check src/mlxl3_quantizer/packing.py src/mlxl3_quantizer/checkpoint.py \
  scripts/quantize_lfm25_moe.py benchmarks/bench_quantize_packing.py \
  tests/test_lfm25_moe_conversion.py tests/test_quantize_packing.py \
  tests/test_quantize_packing_benchmark.py tests/quantize_conversion_contracts.py
ruff check src/mlxl3_quantizer/packing.py src/mlxl3_quantizer/checkpoint.py \
  scripts/quantize_lfm25_moe.py benchmarks/bench_quantize_packing.py \
  tests/test_lfm25_moe_conversion.py tests/test_quantize_packing.py \
  tests/test_quantize_packing_benchmark.py tests/quantize_conversion_contracts.py
python -m compileall -q src/mlxl3_quantizer scripts/quantize_lfm25_moe.py \
  benchmarks/bench_quantize_packing.py tests/quantize_conversion_contracts.py

MLXL3_MLX_ROOT="$PWD/.venv/lib/python3.12/site-packages/mlx" \
  cargo build --locked --release --features mlx,chat
PYTHONPATH=src:references/PonyExl3:scripts \
  MLXL3_QUANT_TEST_ENGINE="$PWD/target/release/mlxl3-rs" python -m pytest -q
MLXL3_SKIP_METAL_CHECK=1 PYTHONPATH=src:references/PonyExl3:scripts python -m pytest -q \
  tests/test_lfm25_moe_conversion.py tests/test_quantize_packing.py \
  tests/test_quantize_packing_benchmark.py tests/test_lfm_conversion_adapter.py

c++ -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined \
  -fno-omit-frame-pointer tests/quantize_packing_proof.cpp -o build/proofs/quantize-packing
build/proofs/quantize-packing
cbmc -D__CPROVER tests/quantize_packing_proof.cpp --function proof --unwind 17 \
  --unwindset 'proof().0:513' --unwinding-assertions --bounds-check --pointer-check \
  --signed-overflow-check --unsigned-overflow-check
cbmc -D__CPROVER -DMLXL3_COVER tests/quantize_packing_proof.cpp --function proof \
  --unwind 17 --unwindset 'proof().0:513' --cover assertion \
  --property 'proof().coverage.2' --property 'proof().coverage.3' \
  --property 'proof().coverage.4' --property 'proof().coverage.5' \
  --property 'proof().coverage.6' --property 'proof().coverage.7'
PYTHONPATH=src:references/PonyExl3:scripts:benchmarks crosshair check \
  tests/quantize_conversion_contracts.py --analysis_kind=PEP316 \
  --per_condition_timeout=15 --report_all

python -m pip wheel --no-deps --no-build-isolation . -w build/wheel
git -C references/PonyExl3 apply --reverse --check \
  ../../scripts/patches/ponyexl3-mlxl3-conversion.patch
```

Le build natif est réussi avec `mlx,chat`, sans modification Rust ; son warning
`block v0.1.6` de compatibilité future est préexistant. Le wheel contient la
source `.metal`. Aucun outil de typage Python supplémentaire n'est configuré
dans ce dépôt ; les imports et compilations sont contrôlés par les suites.

Résultats locaux finaux :

| Contrôle | Résultat | Preuve |
|---|---|---|
| Suite Python complète, bridge natif activé | 451 réussis, 1 sauté : modèle Ling local absent | [Log](measurements/quant-lfm25-20261013/python-full-reviewed.log) |
| Conversion/treillis amont, hooks MLXL3 actifs | 113 réussis, 7 sautés : checkpoints Qwen/MiniCPM absents | [Log](measurements/quant-lfm25-20261013/upstream-mlxl3-hooks.log) |
| Contrats CPU, configuration de la nouvelle CI | 31 réussis, 34 tests GPU explicitement sautés | [Log](measurements/quant-lfm25-20261013/ci-host-contracts.log) |
| Même sélection, imports PonyExl3 indisponibles | 22 réussis, 43 sauts : GPU/dépendance facultative | [Log](measurements/quant-lfm25-20261013/optional-converter-absent.log) |
| CBMC sur le corps partagé | 53 obligations réussies, 6/6 couvertures distinctes | [Preuve](measurements/quant-lfm25-20261013/cbmc-final-second.log), [couverture](measurements/quant-lfm25-20261013/cbmc-cover-targeted.log) |
| ASan/UBSan et mutations | Réussis ; swap incorrect et garde uniforme précédent détectés | [Hôte](measurements/quant-lfm25-20261013/packing-host-check-second.log), [mutation swap](measurements/quant-lfm25-20261013/packing-mutation.log), [mutation garde](measurements/quant-lfm25-20261013/uniform-mutation.log) |

Les premiers échecs sont conservés : accès scalaire Metal, incompatibilité
libc++ du premier harnais CBMC, intrinsic de couverture C++ incorrect,
itération `safe_open`, fixture sans EOS et reshape du premier oracle de
benchmark. Ils ont été corrigés avant les résultats finaux ; les logs initiaux
ne deviennent pas des contrôles réussis. La conversion complète24B, ses
logits/PPL/KL et son pic mémoire restent non vérifiés.

Sources : [PonyExl3 épinglé](https://github.com/beamivalice/PonyExl3/tree/8e7fa6b1556f59fc669e25087903b279b9b0346f),
[format EXL3 upstream](https://github.com/turboderp-org/exllamav3),
[kernels Metal MLX](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
Le search et les optimisations historiques sont détaillés dans
[metal-quantization-optimization.md](metal-quantization-optimization.md).
