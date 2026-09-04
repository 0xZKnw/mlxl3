# Decode EXL3 : résultats de la seconde investigation

Branche de recherche `codex/exl3-decode-bandwidth`, basée sur Desktop v0.3.1.
**Aucune nouvelle optimisation activée, aucun changement du GUI/release.**
Les prototypes restent opt-in et ne sont pas des gains prêts à déployer.
Arrêt des essais demandé par l'utilisateur.

## Corriger la lecture des chiffres de bande passante

115–124 Go/s désignait le volume logique de poids divisé par le temps d'un
kernel isolé sur des matrices synthétiques de forme similaire aux heads.
Ce n'est ni une mesure du trafic DRAM ni le débit du modèle complet. Le
microbenchmark incluait une synchronisation CPU ; ce ne sont pas des timestamps
GPU. Il utilisait le codebook numérique 2 (MUL1), pas nécessairement le codebook
du checkpoint. Ne pas en déduire une saturation de la bande passante du Mac.

Qwen : environ 1,389 Go de poids actifs par token selon les métadonnées et une
approximation de routage uniforme. À 36 tok/s : environ **50 Go/s utiles**,
hors KV, scales, activations et relectures. Les 12 Go du modèle ne sont pas
tous lus à chaque token d'un MoE ; les poids partagés et le head restent inclus.

## Conditions et dérive observée

Apple M5 24 GiB, MLX 0.32.2, Qwen3.6 35B A3B EXL3 2.49 bpw.
L'utilisateur a déchargé le GUI avant les mesures. AC connecté, mode économie
d'énergie désactivé lors du contrôle. LFM n'est plus disponible localement.
Les benchmarks streaming alternent référence/candidat et utilisent deux
prompts (explication et code), température zéro, 128 tokens, quatre paires.

La référence a varié d'environ 34–36 tok/s à 45–57 tok/s plus tard dans la
session. **La cause n'a pas été isolée** : aucun relevé de fréquence/température
GPU ne permet de l'attribuer formellement au throttling. Ne pas comparer les
scores absolus entre expériences comme un gain logiciel.

## Expériences

| Piste | Résultat | Décision |
|---|---|---|
| Repacking tuiles dense `[N-groupe,K,N-local,mots]` | Petits écarts, pas de configuration gagnante sur toutes les formes | Prototype microbench uniquement |
| Même repacking sur experts routés | Environ 0,26–0,29 ms synchronisées, écarts faibles | Pas d'intégration au chargement |
| Ancien SIMD scaled-Hadamard | +1,71 % médian apparié Qwen, sorties identiques | Déjà existant ; ne pas compter comme nouveau gain |
| Fenêtres 8 codewords K=2/4, version ulong | +0,33 % médian apparié | Non concluant |
| Même idée avec extraction uint32 | −4,80 % médian apparié | Désactivée |
| Gather scales + Hadamard fusionnés | −0,33 % médian apparié | Désactivée |
| Command buffers plus grands | Pas de gain clair, pic RAM jusqu'à +1,14 Go | Conserver MLX par défaut |
| Table exacte de codebook | Environ 12,33 tok/s contre 52,13 puis 44,65 en référence | Régression majeure, arrêt après deux paires |

La LUT conserve des valeurs float32 pour ne pas introduire de nouvel arrondi ;
elle coûte 256 Kio par codebook. Son coût en accès dispersés semble défavorable,
mais aucun compteur matériel n'a confirmé la cause. Les tokens des deux paires
restaient identiques : une optimisation exacte peut être beaucoup plus lente.

### Repacking : microbenchmarks (ms, référence / meilleure variante)

- Dense 2048×248320 K6 : 3,300 / 3,150 (groupe stockage 16).
- Dense 2816×262144 K6 : 4,517 / 4,468 (groupe stockage 2).
- Dense 2048×6144 K5 : 0,268 / 0,256 (groupe stockage 2).
- Experts 2048→512 K2 : 0,273 / 0,263.
- Experts 2048→512 K3 : 0,270 / 0,262.
- Experts 512→2048 K2 : 0,285 / 0,289 (aucun gain).

Sorties identiques bit à bit dans ces microbenchmarks. Le benchmark garde
plusieurs copies pour comparer les layouts ; ce n'est pas un memory manager
destiné au runtime. Aucun poids utilisateur n'a été réécrit.

### Command buffers : médianes des deux générations par processus

| Ops / Mo | Aller tok/s | Retour tok/s | Pic MLX Go |
|---|---:|---:|---:|
| 40 / 40 (référence) | 36,30 | 34,39 | 12,58 |
| 128 / 256 | 36,16 | 35,76 | 13,46–13,66 |
| 512 / 1024 | 35,94 | 35,45 | 13,72 |
| 10 / 40 | 35,29 | 35,50 | 12,58 |

96 tokens identiques pour les seize générations. Le protocole aller/retour
ne supprime pas la dérive au cours du temps ; aucun nouveau défaut retenu.

## Profils et limites

Le profil synchronisé attribue beaucoup de temps aux Hadamard, mais désactive
la compilation et synchronise entre étapes : ses 184,6 ms/token ne représentent
pas le débit réel. Il est impropre à mesurer une occupation ou une bande passante.

Un profil cProfile du chemin normal, après warmup, a mesuré 57,28 tok/s sur
96 tokens. Sur 1,950 s total : 1,618 s attribuées au corps de `generate_step`,
0,115 s aux wrappers récurrents compilés, 0,085 s à deux appels `get_vocab`.
Les appels natifs et attentes GPU sont inclus dans ces attributions Python ;
ce profil ne sépare pas le temps GPU du temps bloqué côté CPU.

## Reproduction des prototypes (non recommandés en production)

```sh
.venv/bin/python benchmarks/benchmark_decode_ab.py MODELE --feature window
.venv/bin/python benchmarks/benchmark_decode_ab.py MODELE --feature gather
.venv/bin/python benchmarks/benchmark_decode_ab.py MODELE --feature lut
.venv/bin/python benchmarks/benchmark_command_buffers.py MODELE
.venv/bin/python benchmarks/benchmark_exl3_layout.py
.venv/bin/python benchmarks/benchmark_expert_layout.py
.venv/bin/pytest tests/test_qmv_windows.py tests/test_gather_hadamard.py -q
```

Flags de recherche, tous désactivés par défaut : `MLXL3_K24_WINDOW_DECODE`,
`MLXL3_GATHER_HADAMARD`, `MLXL3_CODEBOOK_LUT`. Ne pas activer la LUT pour
chercher de meilleures performances. Le CLI et le moteur embarqué existants
conservent leurs réglages de production.

Tests de parité ajoutés : 48 cas fenêtres/LUT, 16 cas gather-Hadamard.
Ces tests vérifient les cas indiqués, pas une preuve exhaustive de tous les états.

## Sources primaires consultées

- [MLX v0.32.2 — limites des command buffers](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp)
- [Apple — occupancy et mémoire M5](https://developer.apple.com/videos/play/tech-talks/111431/)
- [MLX — kernels quantifiés](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h)

La prochaine investigation utile nécessite un relevé GPU/CPU plus discriminant,
pas de déclarer le moteur proche du plafond à partir du microbenchmark du head.
