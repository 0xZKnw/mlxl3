# MLXL3 moteur 1.4.4 — LFM2.5-24B-A2B en EXL3

Cette version corrige le chargement de LFM2.5-24B-A2B : son routeur biaisé utilise 64 experts, alors que le moteur précédent acceptait au maximum 32 experts. Le checkpoint [EXL3 4 bpw v1](https://huggingface.co/0xzknw/LFM2.5-24B-A2B-EXL3-4bpw/tree/v1) peut maintenant générer dans le bridge natif. [PR35](https://github.com/0xZKnw/mlxl3/pull/35), fusionnée en `408663fbf2b77ecdbc26b6a2f92f8f4e9ba618e2`.

## Nouveautés et correctifs

- Routeur biaisé à 1–64 experts, avec deux candidats par voie SIMD. La sélection en FP32, le biais, le départage des égalités et les scores originaux sont conservés. Les cas déjà acceptés à 32 experts restent couverts par les contrôles physiques.
- Le moteur conserve le bridge protocole1 et ses options actuelles. Le parcours de préfill MoE reste monotoken ; cette release n'ajoute pas de préfill MoE groupé.
- Le dépôt contient aussi le convertisseur LFM MoE et le packing treillis Metal exact utilisés pour produire v1. Ces outils Python de conversion ne sont pas inclus dans l'archive du moteur natif.

## Performances et qualité

Mesures du code dePR35 sur Apple M5/24Gio, avant cet incrément de version : conversion complète K4/MCG en **3h58min39s**, calibration et export compris, téléchargement séparé9min27s ; poids **12,44Go**. Génération native médiane **58,38/58,34tokens/s** sur les prompts français/code, greedy,266tokens d'entrée et128de sortie, cinq répétitions après warmup, sansMTP/DFlash/cache. Aucun nouveau gain comparatif de vitesse n'est annoncé.

Sur le protocole WikiText-2 test apparié de8 192tokens/4 088labels : perplexité BF16 **17,5560**, EXL3 **18,3656**, UD-Q4_K_XL **18,3467**. Scores proches sur cet échantillon ; EXL3 occupe **16,35%** de moins. Le GGUF utilise environ4,96bpw effectifs contre4,15pourEXL3. Cette comparaison ne démontre pas une équivalence générale de qualité et ne permet pas d'attribuer toute la perte aux kernels de quantification. Les poidsv1 publiés sont inchangés.

[Protocoles, conditions, anciens résultats et limites](lfm25-measurements-2026-10-14.md), [guide de conversion](lfm25-moe-quantization.md).

## Installation et compatibilité

Archive **MLXL3-Engine-v1.4.4-arm64.tar.gz**, manifeste JSON et rapport de validation joints à la release. Apple Silicon, macOS26.2+, Desktop1.4.0+, MLX0.32.2. Le moteur s'installe indépendamment du bundle Desktop ; une nouvelle activation du modèle utilise le moteur sélectionné. Les poids se téléchargent séparément surHuggingFace.

Le canal moteur est publié avec `latest=false` ; Desktop1.4.3/build25 reste proposé par l'updater de l'application. Aucun DMG Desktop n'est publié dans cette release. Le SHA source réellement compilé et les tailles/SHA-256 des artefacts figurent dans le rapport de validation.

## Vérification et limites

PR35 : format/lint/build `mlx,chat`,530tests Python réussis/1sauté,95tests Rust réussis/79ignorés, plus2tests physiques du routeur sélectionnés séparément. Les fixtures physiques couvrent105cas finis et10rejets ; les mutations du routeur et de l'oracle de perte sont détectées puis restaurées. Les mesures incluent quatre réponses réelles via bridge.

Kani du contrat CPU de routage :59obligations et6couvertures sur les types symboliques complets, sans hypothèses. Packing entier :53obligations/6couvertures avec abstraction hôte de l'espace d'adressage. Ces contrôles ne prouvent pas Metal, MLX, l'allocation, la concurrence ni le modèle entier. La suite Kani GitHub du commitfb1a a été interrompue par l'arrêt du runner après54harnais réussis, pendant un ancien harnais DFlash ; aucun succès global de cette suite n'est annoncé. Les limites et échecs historiques restent conservés.

Les contrôles propres à l'archive1.4.4 et leur état CI sont consignés dans le rapport joint ; seuls les résultats inspectés sont annoncés comme réussis.

## Remerciements

Merci à [LiquidAI](https://huggingface.co/LiquidAI) pour LFM2.5-24B-A2B, à [PonyExl3](https://github.com/beamivalice/PonyExl3) pour la conversion et l'inférence MLX, à [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pour EXL3, et à [MLX](https://github.com/ml-explore/mlx) pour le runtime Apple Silicon. [@0xZKnw](https://github.com/0xZKnw) a intégré le convertisseur, le packing et le support du routeur64 dans [PR35](https://github.com/0xZKnw/mlxl3/pull/35).
