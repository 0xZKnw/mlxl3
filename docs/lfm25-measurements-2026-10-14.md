# LFM2.5-24B-A2B : conversion et mesures du 14 octobre 2026

La conversion EXL3 K4/MCG complète prend **3 h 58 min 39 s** sur Apple M5 / 24 Gio,
calibration et export compris. Les 7 402 projections sont présentes, aucune
sautée ; trois shards totalisent **12 443 937 520 octets**, soit 12,44 Go / 11,59 Gio.
Le téléchargement prend séparément 567,18 s. La cible 2–4 h est atteinte pour
cet exemplaire ; le temps d'une conversion amont complète n'a pas été mesuré.

Source : `LiquidAI/LFM2.5-24B-A2B`, révision
`1769516ac588c4c558089c94f27d96b0dc67b183`, 19 fichiers vérifiés contre leurs
checksums. Recette : 2 048 lignes réelles / deux séquences de 1 024, sigma 0,025,
feedback 16, buffer 128, un worker ; scales propres à chaque projection,
calibration BF16 par couche et LDLQ groupé. Aucun prototype rejeté n'est utilisé.
Les optimisations de conversion conservent la recette et le calcul validés
précédemment ; cela ne signifie pas identité de la quantification avec le BF16.

Les sources exécutées sont le commit `d97c9aeb` avec le correctif local du
routeur 64 experts. Le [manifeste source/binaire](measurements/quant-lfm25-eval-20261014/source-identity.json)
épingle chaque fichier, le moteur SHA-256 `99830cc5…`, les versions et conditions.
Ce moteur est un build local `mlx,chat` ; l'app installée n'a pas été remplacée.
Les trois hashes de shards et le contrôle de finitude sont dans
[checkpoint-integrity.json](measurements/quant-lfm25-eval-20261014/checkpoint-integrity.json).
15 012 tenseurs flottants / 294 212 224 valeurs stockées sont finis.
Le compte de paramètres réel est 23 971 587 712 : environ **4,15289556 bpw**
effectifs, poids non quantifiés et overhead compris. Le calcul initial à partir
des seuls éléments BF16 omettait 2 432 biais F32 ; sa
[correction](measurements/quant-lfm25-eval-20261014/parameter-count-correction.json)
reste distincte du contrôle initial, dont les hashes sont inchangés.

## Qualité et sensibilité au protocole

Corpus WikiText-2 **test**, différent de la capture de calibration **train**.
Toutes les positions ont 128 000 logits finis. Le teacher utilise le source BF16
couche par couche, puis libère ses allocations avant les processus EXL3.
La cible EXL3 native calcule en FP16 ; la différence inclut quantification et
calcul du moteur. Elle n'isole pas la perte d'un kernel de packing.

| Protocole | PPL BF16 | PPL EXL3 natif | Écart PPL | KL BF16 → natif |
|---|---:|---:|---:|---:|
| 2 048 tokens, fenêtres 256, toutes positions, sans BOS | 47,9208 | 54,4585 | +13,6429 % | 0,42794 |
| 8 192 tokens, fenêtres 1 024, toutes positions, sans BOS | 28,1536 | 29,4795 | +4,7094 % | 0,32305 |
| 8 fenêtres 1 024, BOS par fenêtre, seconde moitié : 4 088 labels | 17,5560 | 18,3656 | +4,6112 % | 0,15741 |

Les trois résultats portent sur **les mêmes poids**, avec des contextes et
positions évaluées différents. Ils ne constituent pas une amélioration avant/après.
Le premier +13,64 % reste conservé ; une fenêtre y concentre une forte perte.
PonyExl3 exact, sans fold16 ni quantification supplémentaire, retrouve cette
perte : PPL 126,1398 contre BF16 53,1153 sur cette fenêtre ; natif et Pony
diffèrent de 0,01877 de NLL. Sur la première fenêtre, KL Pony → natif 0,00807,
accord argmax 246/256. Ces deux diagnostics ne valident pas une parité complète
des moteurs. Le chargement Pony atteint un pic MLX de 23,97 Go, proche de la RAM
physique ; ce chemin de référence ne garantit pas un faible pic mémoire.

Preuves : [premier test](measurements/quant-lfm25-eval-20261014/quality.json),
[test élargi](measurements/quant-lfm25-eval-20261014/quality-8192.json),
[protocole commun](measurements/quant-lfm25-eval-20261014/paired-bf16-native.json),
[diagnostic Pony](measurements/quant-lfm25-eval-20261014/pony-reference-window-05.json).
Les matrices de logits brutes sont conservées dans le work-dir local, pas en Git.

## Comparaison avec le GGUF présent dans LM Studio

Le fichier fourni est **UD-Q4_K_XL**, malgré la désignation Q4_K_M dans la
conversation : `LFM2.5-24B-A2B-Preview-UD-Q4_K_XL.gguf`, SHA-256 `a235d236…`.
Il pèse 14 876 901 632 octets / **4,96484482 bpw** effectifs : 62,569 % des
paramètres Q4_K, 33,594 % Q5_K, 1,680 % Q6_K et 2,135 % Q8_0.
Les 38 matrices de routage correspondent exactement au source, le tokenizer
correspond après alignement BOS, les dimensions et RoPE sont identiques.
La provenance de l'ensemble des poids source n'est toutefois pas prouvée.
Le nom Preview seul ne suffit pas à démontrer une version différente : le
dépôt Hugging Face ancien nom redirige actuellement vers le modèle Final.

Pour la comparaison, suivre le [code officiel épinglé de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/21a19cd3b/tools/perplexity/perplexity.cpp) :
8 chunks de 1 024, BOS au début, états remis à zéro, logits des positions
512..1022 inclus, soit **511 labels par chunk / 4 088 au total**. Le corpus du
teacher/natif reproduit exactement ces IDs ; le dump GGUF confirme header,
8 chunks, 8 192 IDs, positions et taille finale. La première tentative d'alignement
avait détecté le BOS supplémentaire et reste enregistrée.

| Checkpoint évalué | PPL | Écart au teacher BF16 | Taille des poids |
|---|---:|---:|---:|
| Source BF16, teacher MLX | 17,556020 | référence | 47,94 Go de tenseurs source |
| EXL3 K4, natif MLXL3 | 18,365561 | +4,6112 % | 12,44 Go |
| GGUF UD-Q4_K_XL, llama.cpp | 18,3467 | +4,5038 % | 14,88 Go |

L'écart entre les deux cibles est environ **0,10 % de PPL**. Ce test ne départage
pas leur qualité générale : GGUF rapporte une erreur standard de PPL 0,97099,
et une différence appariée statistiquement significative n'a pas été établie.
L'observation utile ici est une PPL très proche avec un fichier EXL3 **16,35 %
plus petit**. Cela ne prouve pas une supériorité universelle du format EXL3 sur
GGUF, ni de notre quantizer sur les recettes Unsloth. Les moteurs et précisions
de calcul diffèrent aussi. La qualité en agent/code ne se déduit pas de WikiText.

Backend GGUF : archive officielle b11620, commit `21a19cd3b`, SHA-256 vérifié
contre le digest GitHub, installé uniquement dans `build/lfm25-eval`.
`llama-perplexity -c 1024 -b 128 -ub 128 -ngl 99 --chunks 8 --save-all-logits … --offline`.
Le PPL scalaire est calculé avant le clamp du dump ; KL comparative non mesurée,
car le dump utilise un plancher max-logit−16 et une quantification uint16.
Ni la vitesse de génération GGUF ni le backend personnel LM Studio ne sont
modifiés/benchmarqués. [Résultat et commande](measurements/quant-lfm25-eval-20261014/gguf-paired.json),
[en-tête/recette](measurements/quant-lfm25-eval-20261014/gguf-inspection.json),
[backend](measurements/quant-lfm25-eval-20261014/gguf-backend-identity.json).

## Génération native et mémoire

Quatre réponses finales vérifiées via le vrai bridge : définition FR des MoE,
17×23=391, JSON exact et fonction Python `add(a,b)` avec retour `a+b`.
Le premier budget Python 256 s'arrête pendant Thinking : campagne rejetée avant
toute mesure de vitesse, rapport conservé. À budget 1 024, réponse correcte en
370 tokens ; les trois contrôles précédents sont explicitement réutilisés.

Un bridge, contexte 4 096, greedy, cache désactivé, sans MTP/DFlash. Deux prompts
FR/code raw 256 tokens, **266 après template**, un warmup de 256 outputs chacun,
puis cinq répétitions de 128 outputs chacun, ordre FR/code alterné. Les 10 sorties
ont exactement 128 tokens et les mêmes hashes par prompt ; le débit inclut la
phase de raisonnement du template par défaut.

| Médiane de cinq répétitions | FR | Code |
|---|---:|---:|
| Génération | 58,3848 tok/s | 58,3421 tok/s |
| Prefill | 42,8409 tok/s | 42,9514 tok/s |
| Premier token | 6,2095 s | 6,1935 s |
| Durée totale par réponse | 8,3900 s | 8,3459 s |

Plage decode 56,64..59,65 tok/s. Ces valeurs sont **absolues**, aucun gain A/B
n'est établi : l'ancien moteur refuse 64 experts. Le MoE natif utilise encore
le prefill par token. Le long prefill optionnel n'a pas été chronométré séparément.

MLX actif médian 12,4405 Go, cache 0,2324 Go, pic MLX 12,4545 Go ; footprint
médian 12,9138 Go, pic footprint lifetime 13,2265 Go. RSS observée maximale
93,90 Mo après réponse, distincte des allocations GPU et du footprint, pas un
pic RSS. Conversion : RSS maximum 8,375 Go / footprint maximum 17,127 Go.
Ces deux campagnes et métriques ne sont pas interchangeables.
M5 / 24 Gio, batterie **84→82 % en décharge**, pmset sans avertissement ;
température physique non mesurée, swap 3544,25→3512,25 Mio. Aucun compilateur,
prouveur ou autre modèle GPU de cette tâche pendant les campagnes.
[Résultats bruts](measurements/quant-lfm25-eval-20261014/native-measures-v2.json).

## Correctif, vérification et limites

Le smoke initial du pipeline échoue/exit2 sur la limite du routeur biaisé à
32 experts. Cette erreur historique reste un échec. `topk_biased` traite désormais
64 experts avec deux candidats par lane ; tie par petit ID et scores non biaisés
conservés, toujours une seule ligne. Les poids de quantification sont inchangés.

- `cargo fmt --check`, Clippy strict `--all-targets --features mlx,chat`,
  build release `--features mlx,chat` passent ; warning block0.1.6 préexistant.
- `cargo test --locked --release --features mlx,chat` : **95 passés /79 ignorés**.
  Deux tests GPU de routeur sont sélectionnés seuls puis passent : 105 cas finis
  N=1/2/31/32/33/63/64, K=1/top4/full, égalités/dernière lane/±0 et dix rejets.
- Suite Python, bridge natif activé : **530 passés /1 sauté** (modèle Ling absent).
  Le nouvel évaluateur a 25 tests : oracle indépendant logaddexp/distribution
  binaire, invariance d'offset, formes/labels/NaN et vrai subprocess malformé,
  incomplet/excédentaire/silencieux/nonzero, nettoyage et rapport failed.
- Mutations NLL sur mauvais label et suppression des candidats 32..63 détectées ;
  sources restaurées, tests ciblés repassés et binary final reconstruit avant GPU.
- Kani sur le vrai garde Rust : **59 obligations /6 couvertures** satisfaites,
  i32/usize symboliques complets, aucun assume. Cela vérifie le domaine de dispatch
  CPU, pas le shader ni le modèle. Suite complète locale : 54 succès /55 démarrés,
  interruption >13 min sur un ancien harnais DFlash ; pas une suite complète verte.
- CBMC sur le shader MSL réel : type de fichier refusé, **Metal non prouvé**.
  CrossHair `window_count` signale Confirmed ; recherche symbolique, pas preuve
  déductive. Vérification de poids/GPU/concurrence/génération complète non prouvée.

Commandes et logs détaillés sont dans `opti.md` essais14/15/16 et
[les preuves](measurements/quant-lfm25-eval-20261014/). La CI PR/push vérifie aussi
le nouveau harnais CPU ; les tests Apple GPU restent explicitement sélectionnés
localement. Pour la tête précédente d97c9aeb, cinq jobs réussissent et Kani échoue
sur un shutdown du runner/exit143, sans contre-exemple identifié. La CI de la
nouvelle tête est à inspecter après push ; aucun vert global n'est annoncé ici.
Aucune release, fusion, publication de poids ou remplacement de l'app installée.
