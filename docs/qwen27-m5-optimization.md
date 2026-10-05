# Qwen3.8-27B EXL3 : validation et optimisations natives sur Mac M5

Travail sur HENK0O/mlxl3, branche `optimize/qwen27-m5`, base `1ac910d`.
Le commit `d628e0b` fournit une référence qui corrige uniquement la lecture;
le commit suivant contient les optimisations et leurs preuves.
Le modèle local est chargé et génère correctement avec le moteur corrigé.
Les poids, la quantification et les paramètres de génération sont conservés.

## Changements retenus

1. **Chargement Darwin** : l'embedding BF16 `[248320,5120]` contient
   2 542 796 800 octets. Le fork échouait avant `ready` sur une lecture dépassant
   `INT_MAX`. Les lectures sont maintenant bornées à 64 Mio, directement dans
   le buffer MLX. Lectures partielles, EINTR, débordements et troncature gardent
   leurs contrôles. L'erreur de chargement indique aussi le nom du tenseur.
2. **Préfill M5** : la QMM native utilise des tuiles de 64 lignes pour MUL1,
   entrée ≥4096, sortie <65536, M≥128 divisible par64. Les autres formes gardent
   BM32. BN32/BK16, poids, FMA et ordre de réduction sont conservés; le nom
   du kernel distingue les spécialisations. La tête de vocabulaire reste exclue.
3. **Mémoire** : après chargement d'un gros embedding converti vers F16,
   le bridge libère une fois les buffers inutilisés du cache allocateur MLX.
   Les tableaux des poids et les états du modèle restent détenus. Le nettoyage
   ne s'exécute pas dans la boucle de génération.

Options de repli : `MLXL3_DENSE_PREFILL_M64=0` restaure BM32;
`MLXL3_RETAIN_LOAD_CACHE=1` conserve les allocations libres de chargement.

## Modèle et environnement

MacBook Air M5, GPU 10 cœurs, CPU 10 cœurs, 24 Gio unifiés, macOS 27.2/build 26B5091g,
SDK 27.0, déploiement 26.2, Rust 1.99.0, MLX 0.32.2. Python 3.12.14 et NumPy 2.5.3
servent aux harnais; le moteur reste entièrement Rust/Metal.

Checkpoint Qwen3.8-27B-exl3 local : architecture qwen3_5 dense, 64 couches,
hidden5120, FFN17408, vocabulaire248320. Métadonnées nominales2bits/tête3bits,
stockage réellement mixteK1/2/3/4, 3080 tenseurs et9 672 422 388octets de
payload. Les composants vision et MTP présents dans les fichiers ne sont pas
évalués par cette campagne d'inférence texte.

Inventaire et empreintes des configurations/tokenizer/headers :
[checkpoint-summary.json](measurements/qwen27-m5/checkpoint-summary.json),
[checkpoint-manifest.json](measurements/qwen27-m5/checkpoint-manifest.json).
Les payloads ne sont pas hachés intégralement. L'inspection complète est
conservée temporairement dans le workspace; aucun poids n'est ajouté au Git.

## Validation numérique

- Régression de chargement reproduite avant correctif, puis lecture simple
  et groupée d'un fichier sparse >INT_MAX avec marqueurs aux frontières64Mio,
  offset non aligné, début/fin, troncature et overflow.
- 64 cas physiques QMM couvrant K1..8, trois codebooks, matrices groupées
  stridées et chemins de repli : toutes sorties F16 finies et identiques en bits.
- Modèle complet : préfills23/128/256, suivis chacun de trois tokens imposés.
  À chacune des12étapes, tous les248320logits du dernier token et128tableaux
  d'états GDN/conv/KV sont identiques en bits à la référence avec le seul
  correctif de lecture. Les logits de toutes les positions intermédiaires
  du préfill ne sont pas comparés.
- Le test de cache vérifie qu'un vrai buffer temporaire32Mio est libéré
  et que les tableaux vivants restent accessibles/identiques.
- Chaque campagne de génération vérifie les empreintes des tokens et du
  texte après warmup et sur toutes les répétitions; cache de prompts nul.

Les références d'états (~1,5Go) et les binaires de comparaison sont temporaires
et exclus de Git. Les logs de contrôle sont dans
[le dossier de preuves](measurements/qwen27-m5/).
Ces contrôles vérifient l'équivalence des optimisations au moteur de référence;
la qualité de la quantification par rapport aux poids originaux n'est pas mesurée.

## Performance : protocole et limites

Un seul modèle GPU à la fois, aucun build/prover pendant la mesure, température
et fréquences GPU non instrumentées. Mac sur batterie en décharge; power,
thermique et swap sont relevés par passage. Aucune alerte thermique déclarée
ne garantit des fréquences stables. Les contrôles inchangés dérivent fortement.

Deux prompts fixes : question française69tokens et document508tokens
([texte exact](measurements/qwen27-m5/prompt-document.txt)). Contexte4096,
greedy, sans réutilisation de préfixe, MCP/MTP/DFlash désactivés. Un warmup par
prompt puis deux répétitions par passage. Les chiffres agrègent les médianes
par passage; la variation des contrôles limite leur portée. Les campagnes48
et8tokens ne sont pas comparées entre elles ni additionnées.

La première ABBA48tokens observe +6,52% de préfill sur le document, −7,26% de
TTFT, mais un temps complet comparable. La confirmation BAAB8tokens avec25s
de repos avant chaque chargement compare le même binaire avec BM32/BM64,
nettoyage identique des deux côtés :

| Document508tokens | BM32 | BM64 | Variation |
| --- | ---: | ---: | ---: |
| Préfill, tokens/s | 80,69 | 93,87 | +16,34% |
| Premier token, s | 6,332 | 5,466 | −13,68% |
| Complet8tokens, s | 7,462 | 6,535 | −12,42% |

Les rapports préfill-document/préfill-court inchangé s'améliorent dans les deux
passages BM64 face à leurs contrôles adjacents. C'est un contrôle descriptif,
pas une mesure de fréquence. Sur le court, le complet varie de+15,32% alors que
la forme69tokens utilise BM32 des deux côtés : ne pas attribuer ce mouvement
au nouveau kernel. Aucun boost général de décodage ni gain universel sur tous
les Mac/prompts n'est établi.

[ABBA48](measurements/qwen27-m5/bm64-abba/results.json),
[confirmation BAAB8](measurements/qwen27-m5/bm64-baab-confirm/results.json).

La campagne mémoire ABBA8 compare le même binaire, nettoyage activé/désactivé,
avec25s de repos. Les médianes d'empreinte physique du processus passent de
12,496→9,844Go sur le court (−2,652Go) et13,953→11,285Go sur le document
(−2,668Go). Le cache MLX diminue de≈2,645Go; la mémoire active et le pic
alloué MLX sont pratiquement inchangés. Le pic physique du processus **depuis
son lancement**, après le document, passe de13,953→11,984Go (−1,968Go).
Ce pic comprend le chargement et diffère d'un pic par requête.
Temps complet court−0,64%, document+0,60% : économie de mémoire, sans gain
vitesse revendiqué.

[Campagne mémoire](measurements/qwen27-m5/cache-abba/results.json),
[détail des mémoires](measurements/qwen27-m5/cache-memory-summary.json).

## Pistes retirées

- Fenêtre K2 de huit états : deux prototypes et162cas FP32 exacts par
  variante; les microbenchmarks révèlent plusieurs régressions. Shaders QMV
  restaurés, aucun gain de décodage revendiqué.
- Résidence MLX bornée : garde/restauration et12étapes modèle exactes,
  mais aucune accélération robuste; essais ABBA défavorables. Code et option
  retirés. L'[API MLX de résidence](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
  et [l'allocateur0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp)
  ont servi à borner ce prototype, sans modifier de limite système.
- Nettoyage anticipé juste après conversion d'embedding : parité réussie,
  mais pic de chargement pratiquement identique à la version retenue
  (≈11,99Go). Cette extension est retirée; les résultats ne sont pas ajoutés
  aux économies de la campagne mémoire principale.
  [Essai anticipé](measurements/qwen27-m5/cache-early-abba/results.json).

Les tentatives et erreurs de harnais/compilation sont conservées dans
[opti.md](../opti.md) et les logs bruts. Aucune mesure échouée ne compte comme
validation ni comme gain.

## Reproduire

Construire un moteur de référence avec le correctif de lecture et un candidat
sur cette branche, en utilisant la même MLX0.32.2 et le profil release.
Pour les variantes finales, le même binaire peut servir de référence avec
l'option de repli appropriée. Utiliser de nouveaux dossiers de résultats.

```sh
cargo build --release --locked --features mlx,chat
python native/check_qmm_tiles.py REF CAND --output work/parity-new.json
python benchmarks/compare_native.py MODEL --baseline CAND --candidate CAND \
  --baseline-env MLXL3_DENSE_PREFILL_M64=0 --order BAAB --tokens 8 \
  --repeats 2 --settle-seconds 25 \
  --prompt-file docs/measurements/qwen27-m5/prompt-document.txt \
  --output work/prefill-new
python benchmarks/compare_native.py MODEL --baseline CAND --candidate CAND \
  --baseline-env MLXL3_RETAIN_LOAD_CACHE=1 --tokens 8 --repeats 2 \
  --settle-seconds 25 \
  --prompt-file docs/measurements/qwen27-m5/prompt-document.txt \
  --output work/memory-new
```

Pour le contrôle de tous les états, définir `MLXL3_QWEN_TEST_MODEL` et
`MLXL3_QWEN_REFERENCE_DIR`; écrire la référence avec
`MLXL3_QWEN_WRITE_REFERENCE=1` uniquement dans le moteur de référence, puis
relancer sans cette variable dans le candidat. Filtre de test :
`checkpoint_optimization_matches_saved_logits_and_states -- --ignored --nocapture`.
Les tests ignorés requièrent un vrai GPU Apple; la CI sans MLX ne les remplace pas.

Vérifications finales locales réussies : format Rust, Clippy strict tous targets
MLX/chat, 59 tests Rust release (53 ignorés, contrôles GPU ciblés exécutés à part),
build release, Ruff check/format, 16 tests Python bridge/packaging et contrôle du diff.
La suite Python/Desktop complète et les autres tests GPU ignorés ne sont pas
relancés localement. Kani n’est pas installé localement; sa nouvelle propriété
porte uniquement sur le sélecteur de tuiles. Au contrôle GitHub, aucun
workflow ni run CI n’est enregistré sur le fork, malgré Actions activé.
La cause n’est pas déterminée et les réglages du dépôt ne sont pas modifiés.
La propriété Kani et la suite CI distante restent donc **non exécutées**.
Les preuves de ces contrôles sont les logs `*-final.log` du dossier de mesures.
L'application installée n'est pas remplacée, aucune release ni PR n'est créée.

Branche poussée et empreinte distante vérifiée : `408905a` (optimisations),
`d628e0b` (chargement). [Preuve de livraison](measurements/qwen27-m5/delivery-proof.json).
