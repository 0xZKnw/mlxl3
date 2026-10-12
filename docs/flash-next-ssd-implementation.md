# Flash Next EXL3 2,05 bpw avec streaming SSD

État : parcours texte et MTP1 implémentés, contrôlés et mesurés sur M5 ; aucun gain de MTP établi sur les deux prompts testés, défaut OFF conservé. Aucun remplacement de l'app installée.

## Débit du binaire corrigé, sans MTP puis avec MTP1

Source publiée `d506cc78243e680668aaf73961c56824852b4798`, release SHA256 `e89da70ad32906743e332702575053ead81922a42d85d3f33a65e277bf434dfb`, M5 avec 24 Gio et MLX 0.32.2. Même processus et binaire, contexte de 4096 tokens, cache experts de 8 Gio, overlap1 et greedy ; MCP, DFlash et cache préfixe OFF. Quatre warmups puis trois passages de 64 tokens (63 en decode) par couple prompt/mode, ordre alterné. [Résultats bruts](measurements/flash-next-ssd-impl/mtp-speed-resume-20261012/report.json), [médianes, étendues, mémoire et conditions](measurements/flash-next-ssd-impl/mtp-speed-resume-20261012/summary.json), [identités](measurements/flash-next-ssd-impl/mtp-speed-resume-20261012/identity.json).

| Prompt | Sans MTP | MTP1 | Acceptation MTP |
| --- | ---: | ---: | ---: |
| MAIN, cache LFU | **7,21 tok/s** (6,62–7,54) | **7,20 tok/s** (7,15–7,48) | 28/34 (82,35%) |
| HOLD, merge_intervals | **7,15 tok/s** (6,92–7,50) | **6,89 tok/s** (6,80–7,15) | 27/35 (77,14%) |

Les 16 générations, warmups inclus, donnent les mêmes hashes et textes dans les deux modes pour chaque prompt. L’écart médian MAIN de −0,17% est dans le bruit ; HOLD atteint −3,70%, avec MTP inférieur au contrôle dans les trois paires. Aucun gain MTP établi ; le mode reste disponible pour activation explicite. TTFT médian MAIN : 12,74 s/OFF contre 13,12 s/MTP ; HOLD : 12,07 s contre 15,88 s. Temps complet MAIN : 21,48/21,80 s et HOLD : 20,88/25,08 s ; les passages qui réactivent MTP incluent le rechargement de sa tête.

Actif MLX : environ 11,147 Gio/OFF contre 11,812 Gio/MTP (+0,665 Gio), pic maximal 11,207/12,042 Gio. RAM processus médiane : environ 12,977 Gio dans les deux modes, mesurée séparément ; ce processus conserve des allocations et caches. Sur secteur, batterie de 22 à 27% en charge, swap de 1931,38 à 4531,31 Mio, aucune alerte pmset. Températures et octets SSD physiques non mesurés ; deux prompts et trois répétitions ne démontrent pas un gain général. La précédente campagne sur batterie a été interrompue à 11/12 passages : [rapport incomplet conservé](measurements/flash-next-ssd-impl/mtp-speed/report.json), [état d’interruption](measurements/flash-next-ssd-impl/mtp-speed/interruption.json). Ses chiffres ne sont pas combinés avec cette reprise.

## Première mesure sans MTP, avant correction de la divergence

Binaire baseline `011dc68`, SHA256 `986a6a307a914e0be2cb8e429149a2b7c3a9f8f5d58771653a6315b7f0f175bc`, M5/24Gio, MLX0.32.2, contexte4096, cache8Gio et overlap1. Deux prompts fixes,64tokens chacun, warmup puis3répétitions alternées, greedy/MTPoff/MCPoff/préfixeoff. [Rapport complet](measurements/flash-next-ssd-impl/baseline-speed/report.json).

| Prompt | Decode médian | Étendue | TTFT médian | Prefill médian |
| --- | ---: | ---: | ---: | ---: |
| MAIN, cache LFU | 7,08tok/s | 6,77–7,59tok/s | 12,78s | 6,81tok/s |
| HOLD, merge_intervals | 7,03tok/s | 6,61–7,51tok/s | 12,50s | 6,56tok/s |

Six générations de64tokens,63decode chacune ; mêmes hashes et textes dans chaque prompt, y compris warmup. RAM processus11,70Gio ; actif MLX11,15Gio ; picMLX11,21Gio. Batterie45→40% en décharge, swap utilisé5736,94Mio en fin de mesure, aucune alerte pmset. Ce sont les performances du premier binaire sans MTP, pas un gain établi. La correction numérique ci-dessous a ensuite été validée ; cette première mesure reste attachée au binaire antérieur.

## Modèle et architecture

Checkpoint `turboderp/Qwen3.8-Flash-Next-exl3`, branche `2.05bpw_h4_ng4`, révision immuable `65c895314393431c09050b2e04e250836b3a6eb4`. Les sept fichiers LFS ont leurs tailles et SHA-256 vérifiés : [empreintes](measurements/flash-next-ssd-impl/checkpoint-hashes.json). Payload poids 58,3587 Gio, dont 24,4380 Gio de n-grammes. Le modèle reste sur le disque.

Le moteur couvre les 48 couches texte, les quatre flux HyperConnections, 36 blocs Gated DeltaNet, 12 blocs QSA et le PLE/n-grammes de la couche 1. Les embeddings BF16 sont lus par ligne. Les poids communs et la tête EXL3 restent résidents ; les experts K2/MUL1 utilisent des banques persistantes, un cache LFU borné et des lectures `preadv` directes. La cohorte entière est protégée avant toute éviction. Tous les dix experts sélectionnés sont calculés ; pas de suppression approximative d'expert. Le travail GPU sur les hits peut recouvrir la lecture des misses.

`MLXL3_SSD_CACHE_GIB` vaut 8 par défaut (0,6 à 12 autorisés) ; `MLXL3_SSD_OVERLAP=0` désactive le recouvrement pour les contrôles. Le cache n-grammes est borné à 16 384 lignes. Le nombre 2,05 bpw décrit le checkpoint publié, pas une nouvelle quantification faite ici.

## Correction de la divergence avant MTP

Les normes et opérations HC compilées, l’ordre d’accumulation EXL3 (ligne puis SIMD), la partition des experts et le cast avant les matrices denses GDN suivent désormais la référence. PLE utilise la sigmoid stable, la convolution native et la SiLU compilée. QSA calcule les fréquences comme MLX-VLM et utilise le même MRoPE FP32 avec cast final FP16 ; les positions restent des données, sans compilation par token.

Un défaut du premier oracle a aussi été corrigé : MLX démarre les modules en mode entraînement, qui choisissait une récurrence GDN différente. Le nouvel oracle appelle `.eval()` sur les véritables couches avant les wrappers d’observation et refuse un mauvais mode. Les anciennes fixtures et leurs échecs sont conservés.

Référence indépendante : MLX-VLM0.7.3, PonyExl3 `8e7fa6b1556f59fc669e25087903b279b9b0346f`, MLX0.32.2, même checkpoint. Les cinq tokens originaux et onze tokens tenus à part (français/chinois/code/EOS/dernier ID vocabulaire) couvrent16positions. **Les3 973 120logits, les32états GDN0/1 de786432valeurs chacun, les768cohortes et les16argmax sont identiques**, avec égalité bit à bit pour logits/états. Le seuil initial1% reste dans le test, renforcé par cette égalité. Les4 608sorties de composants passent leurs contrôles, dont les mélanges HC et sorties attention en bits ;40normes,20projections,6injections et200stades isolés sont exacts.

[Contrôle final des octets, original](measurements/flash-next-ssd-impl/divergence/byte-parity-original.log), [séquence tenue à part](measurements/flash-next-ssd-impl/divergence/byte-parity-held-02.log), [slots](measurements/flash-next-ssd-impl/divergence/slots-final-02.log), [sources et qualification](measurements/flash-next-ssd-impl/divergence/quality.json). Les probes détectaient effectivement les anciens arrondis ; leurs échecs sont conservés avec tous les essais retirés dans [opti.md](../opti.md). Cette validation finie ne prouve pas tous les prompts/contextes ni la qualité générale du checkpoint.

La première qualification historique [baseline](measurements/flash-next-ssd-impl/baseline-quality.json) avait échoué à1%, jusqu’à2,6487%RMSE, puis employé une borne5%. Elle est **remplacée par la qualification d’inférence corrigée**, sans effacer les chiffres ni leur protocole initial. Les premières mesures de débit précèdent cette correction et ne mesurent pas le code corrigé.

## MTP natif intégré

La tête `mtp.*` et son patch HyperConnections proviennent du même checkpoint. Elle utilise une pré-norme cachée globale, une projection distincte de chacun des quatre flux et une seule couche QSA sans PLE. Les512experts de cette couche restent résidents (0,595Gio de payload d'experts). La vérification cible calcule deux lignes dans une cohorte SSD protégée, conserve le premier état après rejet et les deux après acceptation. Le cache historique de la tête est prolongé par les seuls calculs K/V/index nécessaires.

Profondeur1, mode greedy et pénalité1 ; les autres réglages passent en décodage ordinaire. Le Desktop utilise le dossier du modèle comme tête intégrée, garde MTP OFF pour un nouveau modèle et sauvegarde le choix ON/OFF par modèle et moteur. Aucun téléchargement de tête supplémentaire ni tuner de profondeur n'est nécessaire pour Flash Next.

Dix positions tenues à part concordent en octets avec la vraie classe MTP de MLX-VLM : fusion, six composants, couche, mixer,248320logits par position, argmax, K/V et tail. Le cache-only est identique au cache du passage complet natif. Les blocs cible T2/commit1/2 reproduisent les logits, tous les états GDN/QSA/PLE/HC et le token suivant du T1. Les sessions budgets1/2/3/17 couvrent acceptation et rejet avec mêmes tokens et états cible/tête. Trente paires SSD contre l'oracle indépendant couvrent caches10/20/143, froid/chaud, overlap ON/OFF, doublons, lecture corrompue et reprise. Les16positions du tronc corrigé restent exactes après intégration. [Preuves numériques et échecs conservés](measurements/flash-next-ssd-impl/mtp-impl/).

Le bridge release passe16générations A/B/B/A aux budgets1/2/3/17, mêmes hashes/texte/historique/counts. Il refuse les cinq requêtes invalides, reprend exactement après annulation au début du prefill et au premier delta, et conserve le fallback avec pénalité1,1. Le bridge Qwen3.6 existant passe aussi12générations, les erreurs et annulations. [Flash](measurements/flash-next-ssd-impl/mtp-impl/bridge-flash-01.jsonl), [Qwen3.6](measurements/flash-next-ssd-impl/mtp-impl/bridge-qwen-legacy-01.jsonl). Ces temps courts sont fonctionnels, pas une comparaison de débit.

## Vérification source et limites

Kani 0.68 : contrat d'offsets u64, 12 obligations et deux couvertures satisfaites ; hash n-grammes u64, boucle au plus trois éléments, 87 obligations (une inatteignable) et deux couvertures satisfaites. Le harnais de réservation sur deux routes symboliques et dix slots a été tenté sur la fonction réelle ; après correction d'une ICE liée aux erreurs anyhow, l'exploration a été interrompue après environ cinq minutes. **Cache non prouvé** ; 2 000 cohortes déterministes couvrent vieillissement, évictions, doublons, erreurs et invalidation après lecture échouée.

Kani du calcul de partitions réel : domaines u32 complets, unwind26,37obligations réussies et3couvertures satisfaites ; vérification bornée CPU, sans preuve des réductions flottantes Metal. [Preuve](measurements/flash-next-ssd-impl/divergence/kani-partition.log).

CBMC a été tenté sur la FFI C++ réelle préprocessée : parsing libc++/MLX incompatible. CrossHair a exploré le parseur de tokens pendant huit secondes : deux propriétés non confirmées. Aucun de ces résultats ne prouve la concurrence GPU, MLX, Metal ou l'inférence complète. Le contexte QSA au-delà de 2 048 tokens n'est pas encore contrôlé physiquement ; le parcours multimodal n'est pas implémenté dans ce port texte.

Pour MTP : six harnais Kani réussis,837obligations/21couvertures,3obligations inatteignables dans les contrats d'acceptation existants. Intervalle tensoriel u64 complet et commit i32 complet sans hypothèses ; partition u32 complète/unwind26 ; préfixe greedy au plus7propositions/unwind9 ; profondeur1/unwind4 et profondeurs0..3/unwind6 avec budgets symboliques. Ce sont des vérifications bornées de helpers CPU de production. La suite `mlx,chat` passe43tests bibliothèque+7main+16contrats,65ignorés ; les14contrôles physiques ciblés de cette étape sont sélectionnés séparément. Python68pass/4skips ; Desktop E2E et sources app complètes Swift6/warnings-as-errors réussis. Le build strict du package entier échoue sur les dépréciations CoreText préexistantes de SwiftMath ; le vendor n'est pas modifié. CBMC/Swift/Metal/concurrence et cache symbolique restent non prouvés. [Commandes, bornes, sources et limites](measurements/flash-next-ssd-impl/mtp-impl/quality.json).

CI du commit MTP d506cc7 inspectée : [Desktop/Python/E2E réussis](https://github.com/0xZKnw/mlxl3/actions/runs/38166474586) ; [Rust Linux/macOS et Kani ciblé réussis, Kani complet annulé](https://github.com/0xZKnw/mlxl3/actions/runs/38166474556). Le harnais de cache ajouté avec le streaming Flash dépasse30minutes d’exploration dans la suite complète ; il reste non prouvé. Les logs des jobs sont conservés, sans déclarer cette CI intégralement verte.

Pour la correction :11contrôles GPU ciblés,39tests CPU+15contrats et39tests Python réussis,3tests CPU ignorés ; fmt/Ruff et Clippy strict avec/sans MLX, build release `mlx,chat` réussis. Le bridge release vérifie16tokens par prompt MAIN/HOLD avec warmup+1répétition et hash/texte stables ; la CLI génère4tokens. Le rebuild final conserve exactement l’empreinte du binaire testé. Ces timings courts sont des contrôles fonctionnels, sans gain revendiqué. CBMC reste incompatible avec libc++ et CrossHair ne confirme pas les deux contrats du parseur en8s. CI précédente894c796 : Desktop/Rust Linux/macOS réussis, Kani annulé ; statut du nouveau SHA à inspecter après push.

Qualification du premier commit, conservée : au worktree isolé, format/Ruff, Clippy strict et compilation release `mlx,chat` réussis ; 38 tests CPU et 15 contrats passent, trois tests sont ignorés ; 20 tests Python passent. Le mutant de l’adresse du slot est détecté puis restauré. Cinq contrôles GPU passent. Le bridge du release isolé reproduit les 16 IDs de la baseline ([log](measurements/flash-next-ssd-impl/pr-baseline-bridge.jsonl)). Ces contrôles sont consignés dans le répertoire des mesures. Les GPU sont sélectionnés explicitement et exécutés seuls. La CI hébergée couvre les contrats CPU et le script de référence ; elle n'a ni le modèle ni un GPU Apple physique. Son statut doit être vérifié au SHA de la PR.

## Sources et intégration

Sushi `beamivalice/sushi` à `eaaca12c5508e6a24a01671384a9a41089370058` a été inspecté : kernels HC/QSA/EXL3 et format n-grammes. Son format de banques compilées diffère du checkpoint brut ; aucun corps de kernel Sushi n'est copié et sa sélection tolérante d'experts n'est pas utilisée. Le moteur réutilise les kernels EXL3 de MLXL3/Pony et les idées de caches directs du projet SSD local. [Recherche et antécédents](recherche-flash-next-ssd-205-2026-10-11.md), [attributions](../THIRD_PARTY_NOTICES.md).

Les changements préexistants concernant la projection draft Q4, `mtp.rs`, `qwen35.rs`, le README et le journal sont conservés dans le checkout de travail ; ils ne font pas partie du commit Flash Next. La PR est construite dans un worktree séparé pour vérifier cette séparation.

## Utilisation locale

```sh
MLXL3_MLX_ROOT=/chemin/vers/site-packages/mlx cargo build --locked --release --features mlx,chat
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 65c895314393431c09050b2e04e250836b3a6eb4 --local-dir models/flash-next-205
target/release/mlxl3-rs run models/flash-next-205 --prompt "Bonjour" --max-tokens 64
```

Pour MTP, activer **Génération → MTP** dans le Desktop construit avec ce moteur, ou utiliser le bridge JSON-lines :

```sh
cat <<'JSON' | target/release/mlxl3-rs bridge models/flash-next-205 --context-length 4096
{"type":"generate","request_id":"demo","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":64,"temperature":0,"top_k":1,"repetition_penalty":1,"mtp":true,"mtp_depth":1,"mtp_head_path":"models/flash-next-205","reuse_prompt_cache":false}
{"type":"shutdown"}
JSON
```

La fixture indépendante peut être recréée avec `scripts/check-flash-reference.py` dans un environnement contenant MLX-VLM 0.7.3 et le checkout PonyExl3 épinglé ; `--mtp-only` prend une fixture du tronc et exporte la tête seule. Les fichiers binaires des oracles restent locaux ; le script, leurs empreintes, les mesures et les logs sont fournis dans la PR. La commande `run` garde le décodage ordinaire ; MTP utilise le bridge/Desktop.
