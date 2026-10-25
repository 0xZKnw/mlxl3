# LFM2.5-24B-A2B EXL3 v1 : audit de la qualité

Les accélérations Metal contrôlées ne changent pas les poids par rapport au chemin PonyExl3 de référence. Sur **12 projections réelles**, le treillis, les scales et le flag MCG sont identiques bit à bit entre la référence, nos kernels et le checkpoint v1 exporté. Le principal résultat nouveau concerne la calibration : une variante conditionnelle réduit l'erreur de trois projections sur des tokens de test réellement routés vers ces experts. Ce résultat local ne démontre pas une meilleure perplexité du modèle complet.

## Modèle figé et livraison

- Source `LiquidAI/LFM2.5-24B-A2B`, révision `1769516ac588c4c558089c94f27d96b0dc67b183`.
- [Modèle public v1](https://huggingface.co/0xzknw/LFM2.5-24B-A2B-EXL3-4bpw/tree/v1), commit `07c024ce62c2f60c6b697569bfd08981838b6589`. Les 13 fichiers distribués ont été contrôlés ; les trois shards restent identiques au checkpoint évalué, pour 12 443 937 520 octets.
- [PR35](https://github.com/0xZKnw/mlxl3/pull/35) fusionnée en `408663fbf2b77ecdbc26b6a2f92f8f4e9ba618e2`.
- [Moteur 1.4.4](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.4), source réellement compilée `8386fbfc557f1e740e46a5a6ab250102b047a665`. Version Cargo/lock incrémentée, sans changement numérique après PR35. Archive, signatures, architecture, bibliothèques chargées et hashes vérifiés ; source du tag exact, trois assets distants contrôlés. Desktop 1.4.3 reste latest.
- Modèle copié avec clones APFS indépendants dans `~/Library/Application Support/io.mlxl3.desktop/Models/LFM2.5-24B-A2B-EXL3-4bpw`, enregistré sans remplacer les trois entrées existantes. Moteur géré sélectionné : 1.4.4, précédent : 1.4.3. Le vrai bridge empaqueté a répondu `391` et au JSON exact demandé ; ses processus ont été rejoints. L'interface graphique n'a pas été exercée.

Le comparatif antérieur sur WikiText-2 test donne PPL **17,5560 BF16**, **18,3656 EXL3**, **18,3467 UD-Q4_K_XL**, sur 8 192 tokens / 4 088 labels appariés. L'EXL3 occupe 16,35 % de moins. Le fichier GGUF de l'utilisateur est UD-Q4_K_XL, avec environ 4,96 bpw effectifs ; EXL3 utilise 4,15 bpw. Cette mesure ne constitue pas une égalité générale de qualité. [Protocole et résultats](lfm25-measurements-2026-10-14.md).

## 17 : comparaison des kernels sur les vraies projections

Panel préenregistré : `conv.in_proj` et `feed_forward.w1` de la couche 0, `self_attn.q_proj` de la couche 2, puis w1/w2/w3 des experts couche2/id0, couche20/id31 et couche39/id63. Il couvre les formes convolution, attention, MLP dense jusqu'à 11 776 sorties et experts au début, au milieu et à la fin.

Pour chaque projection, appel de production `ldlq_quantize_layer` sur les mêmes poids BF16 et les mêmes captures de conversion. Paramètres : K4/MCG, `scale_mode="computed"`, sigma 0,025, shrinkage 0, buffer 128, feedback 16, seed 0, `fast_metrics=True`, sans comparaison circulaire à l'oracle pendant le calcul. Premier passage avec `metal_search.quantize_tiles_mlx` et `mlx_trellis.pack_trellis_mlx` originaux, second passage avec les deux hooks MLXL3. Comparaison indépendante des tableaux complets avec les tenseurs réellement stockés dans v1 : formes, dtype, nombre de valeurs, finitude et SHA-256 des octets.

**Résultat : 12/12 projections, 48 tenseurs et 17 359 884 valeurs identiques bit à bit.** Même le passage individuel reproduit le fichier issu du batching de conversion. Pic MLX observé 2 959 818 752 octets ; aucune mesure de vitesse comparative, RSS ou RAM physique à en déduire.

[Plan, captures et paramètres identifiés](measurements/quant-lfm25-quality-audit-20261014/real-projections-plan.json), [résultats et empreintes des six sources numériques](measurements/quant-lfm25-quality-audit-20261014/real-projections.json). Cela exclut une divergence observée sur ce panel ; les 7 402 projections n'ont pas toutes été re-quantifiées.

## Pistes dans la recette et le runtime

La calibration de v1 utilise **2 048 tokens**, deux séquences de 1 024 tokens du WikiText-2 train embarqué. La [documentation EXL3 épinglée](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/doc/convert.md) décrit un défaut de 250 séquences de 2 048 tokens, soit 512 000 tokens. Les options et les pipelines diffèrent ; ce facteur 250 ne prédit pas à lui seul une variation de perplexité.

Notre `capture_experts` présente tous les tokens à chaque expert. [PonyExl3 capture les affectations réellement routées](https://github.com/beamivalice/PonyExl3/blob/8e7fa6b1556f59fc669e25087903b279b9b0346f/ponyexl3/convert/capture.py). EXL3 CUDA prévoit aussi une [calibration avec activation de tous les experts](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/conversion/convert_model.py) : le choix tous-tokens n'est donc pas un bug en soi. Ce pipeline collecte également la sensibilité des sorties, pondérée par le routage réel ; notre chemin `computed` ne fournit ni `out_sensitivity` ni `hessian_diag` à la régularisation. L'auto-scaling des gate/up et les traces adaptées à un modèle de raisonnement sont des pistes supplémentaires, sans validation locale de gain pour l'instant. La [recette optimisée amont](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md) reste expérimentale et signale qu'elle n'est pas validée pour les modèles sparse.

Sur les captures train, les trois experts sélectionnés reçoivent réellement **181, 66 et 26** des 2 048 tokens. Les diagonales de covariance conditionnelles diffèrent sensiblement des diagonales tous-tokens, particulièrement pour down. [Diagnostic](measurements/quant-lfm25-quality-audit-20261014/activation-diagnostic.json).

Les biais F32 sont conservés bit à bit dans le checkpoint. Les loaders Pony et natif les convertissent en F16 à l'inférence. Sur des entrées fixées, ce seul arrondi du biais n'a changé aucun choix top4 des trois couches testées ; le passage des calculs de gate BF16 à FP16 a changé 24/33/19 choix sur 2 048. Ce contrôle ne reproduit pas les états du modèle quantifié complet et n'isole pas une perte de PPL. Il confirme que le score BF16→natif mélange quantification et précision du runtime.

## 18 : contre-épreuve sur des données séparées

Captures de test obtenues avec le vrai `capture_lfm_moe`, source BF16 couche par couche, six projections w1/w2, deux fenêtres de 1 024 tokens provenant du **WikiText-2 test**, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. Les 40 couches ont été parcourues, six captures complètes, sans fallback ni données synthétiques.

Pour chaque expert, re-quantification en mémoire de w2 avec ses seuls tokens train routés ; paramètres de v1 inchangés. Hadamard MLX conservé pour éviter un changement vers le backend CPU sous 192 rows. Reconstruction des poids par le décodeur NumPy indépendant ; comparaison à la matrice BF16 source représentée en FP32, multiplication FP32 et sommes d'erreurs en FP64. Aucun poids du modèle publié ou installé n'est modifié.

| Projection down | Rows train / test routées | Erreur relative v1 | Variante conditionnelle | Variation relative |
|---|---:|---:|---:|---:|
| Couche 2, expert 0 | 181 / 143 | 7,0531 % | 6,9108 % | −2,017 % |
| Couche 20, expert 31 | 66 / 106 | 8,4828 % | 7,3868 % | −12,920 % |
| Couche 39, expert 63 | 26 / 72 | 2,2512 % | 2,1510 % | −4,450 % |

L'erreur diminue sur les tokens test **routés** des trois cas. Le contrôle tous-tokens se dégrade de **4,122 % / 6,575 %** pour les couches 2/39 et s'améliore de 12,262 % pour la couche20. Les progrès sur les tokens train routés sont bien plus grands que sur le test : risque de surapprentissage avec si peu de points. Les tokens actuellement non routés pourraient aussi changer d'expert après quantification des couches précédentes.

[Résultats complets, contrôles négatifs et empreintes des captures](measurements/quant-lfm25-quality-audit-20261014/conditional-ablation.json). Première capture refusée avant calcul pour un chemin relatif absent ; [rapport conservé](measurements/quant-lfm25-quality-audit-20261014/conditional-ablation.first-capture-failed.json). Première commande d'ablation refusée avant exécution pour une fermeture de dictionnaire ; corrigée sans changement du protocole, puis les trois cas ont terminé. Deadlines de 600 s, aucune interruption masquée.

## Décision et portée de la vérification

Conserver **v1** et les kernels actuels. Aucun bug de nos accélérations n'est détecté dans le panel réel. Une calibration plus riche, adaptée au modèle et à la distribution des experts, est la piste étayée à poursuivre pour v2. La variante conditionnelle obtient trois améliorations de proxy ; il faut une conversion et une évaluation globales avant de changer les valeurs par défaut ou de publier d'autres poids. Aucune PPL v2 n'a été mesurée.

Moteur 1.4.4 : `cargo fmt --check`, Clippy strict `--locked --all-targets --features mlx,chat`, build release et tests `--locked --release --features mlx,chat` réussis, **95 tests Rust / 79 ignorés**, plus **2 tests physiques** du routeur exécutés seuls. **45 tests Python** de packaging/évaluation réussis. `cargo kani --lib --no-default-features --harness biased_router_domain_bounds_dispatch --output-format terse` : **59 obligations / 6 couvertures** sur les types symboliques complets, sans hypothèses. Contrat CPU uniquement. Les anciens contrôles de packing et les limites Metal sont conservés dans les rapports15 et de conversion ; ces sondes numériques ne constituent pas une preuve formelle.

Cinq jobs logiciels/packing de la CI du commit exact8386 réussissent. La suite Kani complète reste en cours à la dernière inspection ; le précédent run PR35 avait été interrompu après54harnais réussis par l'arrêt du runner. Aucun succès global de preuve ni validation UI n'est annoncé. Les processus locaux de conversion, test et preuve lancés pour cet audit sont terminés.
