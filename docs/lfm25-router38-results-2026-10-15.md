# LFM2.5-24B-A2B — recalibration des routeurs, essai 38

**Résultat : les trois candidats sont rejetés en validation ; aucun n’est promu.** Le pipeline termine avec `status="complete"`, `stage="rejected_validation"`, `selected=null` et `final_quality_passed=false`. La V2 publiée, Hugging Face, le moteur et l’application installée restent inchangés. [Rapport cloud complet](measurements/quant-lfm25-router38-20261015/lfm25-router38-resume-cloud.json), [manifeste GPU](measurements/quant-lfm25-router38-20261015/gpu-verification.json), [préenregistrement 38](../opti.md).

## Protocole et artefacts

L’essai reprend les treillis EXL3 de la V2 et ajuste seulement les **38 gates F16 de forme 64 × 2048**. Il ne réalise pas une nouvelle quantification complète des 24B paramètres. Les biais, scales et autres tenseurs restent inchangés ; les trois checkpoints expérimentaux sont exportés puis réellement rechargés dans ExLlamaV3 avant mesure. Le forward du candidat utilise ses propres décisions, sans teacher ni routes imposées.

La calibration contient 16 séquences textuelles de 2 048 tokens, soit 32 768 positions, choisies avant mesure et séparées de la validation et des tests réservés. Les logits réels des routeurs source BF16 sont appariés aux entrées normalisées et logits F16 du modèle V2. Un ajustement ridge F64 sans intercept, masqué par le top 20 source, utilise `λ = 0,1 × trace(H) / 2048`, puis `W′ = F16(W0 + αΔW)` pour les seuls α préenregistrés 0,25 / 0,5 / 1. Le contrôle d’export α = 0 est byteexact. Cette recette expérimentale ne garantit pas une amélioration des sorties du modèle complet.

La référence est **Transformers 5.16.1 BF16 officiel en décodage causal**, experts `batched_mm`, avec les 38 biais source restaurés en F32. Ses logits BF16 sont stockés en F32 ; ce stockage ne constitue pas une exécution FP32. La capture source complète du premier lancement est réutilisée après vérification des 53 artefacts, sans nouvelle inférence teacher lors de la reprise. La capture V2 produit 39 artefacts ; les 38 gates recalibrées et trois fichiers de gates sont conservés avec leurs SHA.

La validation comporte 12 fenêtres de 1 023 inputs/labels, soit **12 276 labels par bras**, deux fenêtres et 2 046 labels par domaine. Le vocabulaire complet de 128 000 entrées est conservé. Le scorer calcule `KL(teacher || candidat)` en nats et les NLL en F64, puis `PPL=exp(NLL moyenne pondérée par positions)`. Il contrôle les formes, dtypes, finitude, comptes, labels, rapports complets et SHA des NPY, par mmap avec des blocs d’au plus 32 lignes. Les chiffres ci-dessous portent uniquement sur cette nouvelle référence et cette validation ; ils ne sont pas comparés aux anciens scores utilisant d’autres teachers ou corpus.

## Résultats de validation

| Bras rechargé | KL moyen | Variation KL / V2 | PPL | Domaines refusés |
|---|---:|---:|---:|---|
| V2, baseline inchangée | 0,108002630 | — | 10,999832 | — |
| α = 0,25 | 0,107533788 | −0,434 % | 11,020998 | code, multilingual, tiny |
| α = 0,5 | 0,114167554 | +5,708 % | 11,086917 | c4, code, multilingual, wiki, tiny |
| α = 1 | 0,135970209 | +25,895 % | 10,958061 | les six domaines |

PPL du teacher sur ces 12 276 labels : **11,152831**. Chaque cellule du tableau suivant indique **KL / PPL** ; chaque ligne porte sur 2 046 labels.

| Domaine | V2 | α = 0,25 | α = 0,5 | α = 1 |
|---|---:|---:|---:|---:|
| c4 | 0,081414046 / 19,360533 | 0,081081995 / 19,485932 | 0,085616490 / 19,526257 | 0,101583274 / 19,798223 |
| code | 0,084555450 / 3,467713 | 0,098793841 / 3,516004 | 0,106308791 / 3,461882 | 0,107787590 / 3,446184 |
| multilingual | 0,150791053 / 17,218780 | 0,156443388 / 17,507963 | 0,171578038 / 17,763667 | 0,201944598 / 17,040101 |
| technical | 0,155683570 / 26,365052 | 0,137279039 / 25,476002 | 0,136871660 / 25,724198 | 0,165136123 / 25,430657 |
| wiki | 0,114652916 / 7,469896 | 0,106176166 / 7,517811 | 0,119114951 / 7,637843 | 0,127182611 / 7,379597 |
| tiny | 0,060918748 / 7,780551 | 0,065428295 / 7,800038 | 0,065515395 / 7,872059 | 0,112187057 / 7,935516 |

Les règles restent celles fixées avant les mesures : baisse du KL moyen d’au moins 1 % par rapport à V2 ; dans chacun des six domaines, `KL≤1,02×KL_V2` et `PPL≤1,02×max(PPL_V2,PPL_teacher)`. Le candidat admissible aurait été celui de KL minimal, avec départage stable α = 0,25 / 0,5 / 1.

α = 0,25 n’atteint pas 1 % de baisse et dépasse le plafond KL dans trois domaines. α = 0,5 et α = 1 régressent aussi en KL moyen ; α = 1 dépasse en outre le plafond PPL du domaine c4. La PPL globale légèrement inférieure d’α = 1 ne suffit donc pas à le sélectionner. Ces résultats n’établissent pas une cause unique aux régressions de validation.

## Exécution et provenance

Reprise sur **H100 80 Go**, un seul modèle GPU à la fois, ExLlamaV3 `1.5.4+cu128.torch2.10.0`, PyTorch `2.10.0+cu128`, NumPy `2.5.2`, Safetensors `0.8.0`. Source LFM révision `1769516ac588c4c558089c94f27d96b0dc67b183`, quantifier officiel `16a49792a3c93d8432d72e6c4bce800841566577`. Le teacher provient du code pipeline `f46c06b3b32ebefe6532dc66b532186edca81592` ; la reprise utilise `3da63cb55b7ead34b5ee3f57d0bd0fb775166108`.

La fonction de reprise dure **897,182 s**, chargements, contrôles, fit, exports, captures et scores inclus. La capture teacher réutilisée avait duré 1 447,454 s lors du premier lancement ; ces durées ne sont pas un benchmark de vitesse. Le premier lancement s’était arrêté avant le premier token quantifié sur une précondition de capacité FlashAttention : échec conservé et reprise distincte. Seul le paramètre `batch_shape` a été aligné sur la capacité physique au multiple de 256 exigé par le cache, sans ajout d’inputs ou labels. Les contrôles passifs sans/avec hook sont byteexacts sur 128 tokens pour le teacher et pour V2.

Le cache d’autotuning reste strictement identique avant/après, **176 octets**, SHA256 `1cd9c092bd749dc10fdc017a936477ed9ef0d6072d4af401711d5a7ce5bb0f65`. Le rapport local et le rapport distant sont identiques ; fichier de 193 667 octets, SHA256 `e834316a2936a1b7b16ea516b2cede329c3391a70aa334bdf36a5a0e73e487f4`. Le plan est figé à `76806e9bc30a0f196c4ac48052030d4d4acfdf726b37462e2d4706794cfdbfbb`. Le rapport teacher conservé porte le SHA256 `bd18d5894b02af3b91efb7f2e61253635b5c06785ffc154d5680c7b14abf5478`.

Les trois fichiers de gates font chacun 9 965 648 octets ; ils restent dans `build/` et sur le Volume privé. Le [manifeste GPU](measurements/quant-lfm25-router38-20261015/gpu-verification.json) conserve les empreintes détaillées et l’archive du rapport de fit, avec les résultats négatifs. L’application Modal est arrêtée, zéro tâche GPU active.

## Vérification et limites

Avant la reprise : **322 tests ciblés passent, 0 ignoré** ; contexte Desktop sans Torch ni ajout de `scripts` à `PYTHONPATH` : **824 passent, 30 ignorés**, correspondant aux chemins optionnels Torch et au modèle Ling absent. La suite antérieure **1 258 passés / 1 ignoré** reste une preuve distincte pour les chemins inchangés ; ces suites se recouvrent. Le vérificateur du teacher compte 34 tests passés et détecte la suppression temporaire du garde SHA. Les mutations de RHS, lecture de hashes, garde PPL et réserve de nettoyage ont également été détectées puis restaurées. Ruff format/lint, compilation Python et CLI Modal réelle passent. [Prévol initial](measurements/quant-lfm25-router38-20261015/cpu-verification.json), [prévol de reprise](measurements/quant-lfm25-router38-20261015/resume-verification.json).

CrossHair confirme localement 12 postconditions scalaires avec 20 s par condition. La CI du commit 3da confirme 11 / 12 conditions ; `evaluate_lfm25_router.domain_gate` reste **non confirmé** dans ce délai, sans contre-exemple observé. Ce sont des recherches symboliques de contre-exemples, sans preuve complète des tenseurs, calculs NumPy, CUDA, MLX ou I/O. Les premières sorties du scorer sont archivées comme transcriptions explicites des retours d’outils, les flux originaux n’ayant pas été redirigés vers des fichiers.

Au dernier instantané CI archivé du 15 octobre à 18:43:59 UTC, [conversion/packing](https://github.com/0xZKnw/mlxl3/actions/runs/38785738444) et [Regression](https://github.com/0xZKnw/mlxl3/actions/runs/38785738428) réussissent ; les jobs natifs Ubuntu/macOS réussissent, mais [Kani reste en cours](https://github.com/0xZKnw/mlxl3/actions/runs/38785738409). Le précédent run f46 avait un Kani annulé, d’origine non déterminée ; aucune limite de 25 minutes n’a été démontrée. Aucune CI globale verte ni preuve Kani complète n’est annoncée.

Faute de candidat admissible, **les tests finaux chat/CJK et les cinq jeux historiques, le compactage, le rechargement natif et les smokes bridge du candidat n’ont pas été lancés**. Les références teacher chat/CJK existent déjà, mais aucune qualité finale d’un candidat n’est déduite de leur présence. Aucun modèle de cet essai n’est publié, installé ou promu ; la V2 demeure la version disponible.
