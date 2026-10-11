# Flash Next EXL3 2,05 bpw avec streaming SSD

État : parcours texte sans MTP implémenté et contrôlé localement sur M5 ; MTP et campagne comparative de débit encore en cours. Aucun remplacement de l'app installée.

## Première mesure sans MTP, avant correction de la divergence

Binaire baseline `011dc68`, SHA256 `986a6a307a914e0be2cb8e429149a2b7c3a9f8f5d58771653a6315b7f0f175bc`, M5/24Gio, MLX0.32.2, contexte4096, cache8Gio et overlap1. Deux prompts fixes,64tokens chacun, warmup puis3répétitions alternées, greedy/MTPoff/MCPoff/préfixeoff. [Rapport complet](measurements/flash-next-ssd-impl/baseline-speed/report.json).

| Prompt | Decode médian | Étendue | TTFT médian | Prefill médian |
| --- | ---: | ---: | ---: | ---: |
| MAIN, cache LFU | 7,08tok/s | 6,77–7,59tok/s | 12,78s | 6,81tok/s |
| HOLD, merge_intervals | 7,03tok/s | 6,61–7,51tok/s | 12,50s | 6,56tok/s |

Six générations de64tokens,63decode chacune ; mêmes hashes et textes dans chaque prompt, y compris warmup. RAM processus11,70Gio ; actif MLX11,15Gio ; picMLX11,21Gio. Batterie45→40% en décharge, swap utilisé5736,94Mio en fin de mesure, aucune alerte pmset. Ce sont les performances du premier binaire sans MTP, pas un gain établi. La divergence numérique ci-dessous doit être corrigée avant de poursuivre MTP, selon la dernière demande de l'utilisateur.

## Modèle et architecture

Checkpoint `turboderp/Qwen3.8-Flash-Next-exl3`, branche `2.05bpw_h4_ng4`, révision immuable `65c895314393431c09050b2e04e250836b3a6eb4`. Les sept fichiers LFS ont leurs tailles et SHA-256 vérifiés : [empreintes](measurements/flash-next-ssd-impl/checkpoint-hashes.json). Payload poids 58,3587 Gio, dont 24,4380 Gio de n-grammes. Le modèle reste sur le disque.

Le moteur couvre les 48 couches texte, les quatre flux HyperConnections, 36 blocs Gated DeltaNet, 12 blocs QSA et le PLE/n-grammes de la couche 1. Les embeddings BF16 sont lus par ligne. Les poids communs et la tête EXL3 restent résidents ; les experts K2/MUL1 utilisent des banques persistantes, un cache LFU borné et des lectures `preadv` directes. La cohorte entière est protégée avant toute éviction. Tous les dix experts sélectionnés sont calculés ; pas de suppression approximative d'expert. Le travail GPU sur les hits peut recouvrir la lecture des misses.

`MLXL3_SSD_CACHE_GIB` vaut 8 par défaut (0,6 à 12 autorisés) ; `MLXL3_SSD_OVERLAP=0` désactive le recouvrement pour les contrôles. Le cache n-grammes est borné à 16 384 lignes. Le nombre 2,05 bpw décrit le checkpoint publié, pas une nouvelle quantification faite ici.

## Correction sans MTP

Référence indépendante : MLX-VLM 0.7.3 pour l'architecture, PonyExl3 `8e7fa6b1556f59fc669e25087903b279b9b0346f` pour les projections, même checkpoint. Le script utilise les activations FP16 amont et un cache d'experts limité à la cohorte active. Les cinq tokens `[248044,1,42,248044,7]` exercent le padding EOS et l'état récurrent ; chaque sortie doit être non vide et finie.

[Contrôle final](measurements/flash-next-ssd-impl/gpu-reference-15.log) : 2 tests GPU réussis, 1 440 sorties de composants sur entrées identiques sous 0,5 % RMSE relative, 240 ensembles de dix routes égaux, états GDN des couches 0/1 sous 0,3 %, cinq argmax égaux et reset contrôlé. [Mesures structurées](measurements/flash-next-ssd-impl/baseline-quality.json).

**La parité numérique globale à 1 % a échoué.** Le maximum observé est 2,6487 % RMSE relative des logits au cinquième token, après une rupture de sélection MoE près du seuil. La qualification finale emploie une borne explicite de 5 % pour les logits, plus les contrôles stricts des composants, routes et argmax. Ce n'est pas une parité bit à bit entre moteurs ni une évaluation générale de qualité. Les essais négatifs et les hypothèses réfutées (post-Hadamard, portes GDN, partition SIMD) restent dans [opti.md](../opti.md).

Les contrôles GPU incluent aussi l'adressage des slots versus EXL3 résident et les lectures directes limitées aux plages propriétaires, avec sources/tranches invalides. Le bridge release a généré 16 tokens sans MTP sur un prompt réel de 60 tokens : [événements](measurements/flash-next-ssd-impl/baseline-bridge.jsonl). Son empreinte processus finale était 12 547 024 640 octets ; cette exécution de contrôle n'est pas une campagne comparative de performance.

## Vérification source et limites

Kani 0.68 : contrat d'offsets u64, 12 obligations et deux couvertures satisfaites ; hash n-grammes u64, boucle au plus trois éléments, 87 obligations (une inatteignable) et deux couvertures satisfaites. Le harnais de réservation sur deux routes symboliques et dix slots a été tenté sur la fonction réelle ; après correction d'une ICE liée aux erreurs anyhow, l'exploration a été interrompue après environ cinq minutes. **Cache non prouvé** ; 2 000 cohortes déterministes couvrent vieillissement, évictions, doublons, erreurs et invalidation après lecture échouée.

CBMC a été tenté sur la FFI C++ réelle préprocessée : parsing libc++/MLX incompatible. CrossHair a exploré le parseur de tokens pendant huit secondes : deux propriétés non confirmées. Aucun de ces résultats ne prouve la concurrence GPU, MLX, Metal ou l'inférence complète. Le contexte QSA au-delà de 2 048 tokens n'est pas encore contrôlé physiquement ; le parcours multimodal n'est pas implémenté dans ce port texte.

Au worktree isolé : format/Ruff, Clippy strict et compilation release `mlx,chat` réussis ; 38 tests CPU et 15 contrats passent, trois tests sont ignorés ; 20 tests Python passent. Le mutant de l’adresse du slot est détecté puis restauré. Cinq contrôles GPU passent. Le bridge du release isolé reproduit les 16 IDs de la baseline ([log](measurements/flash-next-ssd-impl/pr-baseline-bridge.jsonl)). Ces contrôles sont consignés dans le répertoire des mesures. Les GPU sont sélectionnés explicitement et exécutés seuls. La CI hébergée couvre les contrats CPU et le script de référence ; elle n'a ni le modèle ni un GPU Apple physique. Son statut doit être vérifié au SHA de la PR.

## Sources et intégration

Sushi `beamivalice/sushi` à `eaaca12c5508e6a24a01671384a9a41089370058` a été inspecté : kernels HC/QSA/EXL3 et format n-grammes. Son format de banques compilées diffère du checkpoint brut ; aucun corps de kernel Sushi n'est copié et sa sélection tolérante d'experts n'est pas utilisée. Le moteur réutilise les kernels EXL3 de MLXL3/Pony et les idées de caches directs du projet SSD local. [Recherche et antécédents](recherche-flash-next-ssd-205-2026-10-11.md), [attributions](../THIRD_PARTY_NOTICES.md).

Les changements préexistants concernant la projection draft Q4, `mtp.rs`, `qwen35.rs`, le README et le journal sont conservés dans le checkout de travail ; ils ne font pas partie du commit Flash Next. La PR est construite dans un worktree séparé pour vérifier cette séparation.

## Utilisation locale

```sh
MLXL3_MLX_ROOT=/chemin/vers/site-packages/mlx cargo build --locked --release --features mlx,chat
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 65c895314393431c09050b2e04e250836b3a6eb4 --local-dir models/flash-next-205
target/release/mlxl3-rs run models/flash-next-205 --prompt "Bonjour" --max-tokens 64
```

La fixture indépendante peut être recréée avec `scripts/check-flash-reference.py` dans un environnement contenant MLX-VLM 0.7.3 et le checkout PonyExl3 épinglé. Les fichiers binaires des oracles restent locaux ; le script, les mesures et les logs sont fournis dans la PR. MTP sera ajouté après sa vérification contre le chemin natif ordinaire.
