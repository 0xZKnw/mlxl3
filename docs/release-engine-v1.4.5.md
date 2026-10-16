# MLXL3 moteur 1.4.5 — fiabilité de l’API et profil mémoire LFM

**Release publique en préparation, non publiée.** Ce moteur accompagne Desktop **1.4.4 build 26**. Il corrige le protocole de fin de génération et le cycle de vie HTTP, et fournit le profil de contexte LFM. Les variantes de performance attendent leur qualification ; toutes les propositions de l’audit ne sont pas implémentées.

## Nouveautés

- **`GET /v1/capabilities`** décrit l’adaptateur HTTP et les capacités publiques du bridge. Streaming et sampling pris en charge sont distingués des fonctions absentes : stop sequences, tool calls HTTP, JSON Schema contraint, images, MTP et DFlash via cet adaptateur. Les métadonnées privées du runtime ne sont pas exposées.
- **`GET /ready`** répond 503 si le bridge est occupé ou indisponible ; `/health` conserve son rôle de liveness.
- Le bridge transmet un **profil de cache logique LFM dense/MoE**. Sur LFM2.5-24B-A2B : **20 480 octets/token + 245 760 octets fixes**.

## Correctifs

- La cause explicite `complete.stop_reason` distingue `eos`, `max_tokens` et `context_full`. L’API rend respectivement `stop`, `length` et `length`, en streaming comme en réponse complète.
- Une seule génération est admise, **avant lecture du corps**. Une requête concurrente reçoit **429 et `Retry-After: 1`**, sans file d’attente non bornée.
- La fermeture du client annule aussi avant le premier delta. Le bridge est drainé avant réutilisation ; une écriture de commande ou un drainage non confirmé le rend indisponible.
- Corps, événements, réponses et attentes sont bornés. Le délai d’écriture s’applique aussi au **transport TCP**, sous le buffer Hyper ; un progrès partiel ne le remet pas à zéro avant flush complet.
- Cause de fin absente/invalide, événements malformés et compteurs de tokens invalides produisent une erreur explicite. Les messages d’erreur sont bornés à une frontière UTF-8 valide. Les pénalités devenant non finies ou nulles en f32 sont refusées avant le prefill.
- Origin, Host, CORS et bearer suivent le même périmètre local. Les CLI sans Origin restent acceptés. Sans clé, un Host présent doit désigner loopback ; avec clé, un Host réseau reste utilisable avec authentification. L’écoute IPv6 accepte `::1` et `[::1]`.
- SIGINT/SIGTERM sont installés avant le chargement. Les sorties normales, erreurs et timeouts nettoient le child et joignent le lecteur stderr ; un nettoyage non confirmé reste un échec explicite.
- Le packaging utilise la version Cargo du moteur. `--version` reste prioritaire sans exiger la lecture de Cargo.

## Performances et mémoire

Qualification numérique locale du préfill MoE : **32 cas V2** passent, longueurs **1..8/23/24/25/63/64/65/128/266**, préfixes **0 et17**, avec **128000logits F16 terminaux,50états, routes/affinités des38MoE et17continuations forcées bitidentiques**. La capture de routes est passive sur128tokens. Ce résultat porte sur le binaire figé `dac890b…`, avant le raccordement KV à capacité ; il ne valide pas automatiquement le code modifié ensuite. Le chemin reste **opt-in, désactivé par défaut** à cette étape. [Preuves et limites LFM-01](measurements/audit-api-lfm-20261016/lfm-01.md).

**À compléter avant publication :** décision d’intégration et résultats physiques des variantes LFM. Aucun gain global de tokens/s, de prefill ou de mémoire physique n’est annoncé à ce stade. Il faut d’abord comparer logits, états et caches du modèle réel, puis mesurer après warmup et en ordre alterné.

Le profil mémoire est logique, batch un, états persistants F16. Il exclut poids, backing supplémentaire des vues, activations temporaires, cache allocateur et RAM processus. Cette métadonnée n’est pas une amélioration KL/PPL. Aucun nouveau checkpoint de quantification n’est publié par cette release moteur.

## Installation et compatibilité

Archive prévue : **MLXL3-Engine-v1.4.5-arm64.tar.gz**, avec manifeste et rapport de validation. Apple Silicon, macOS **26.2+**, Desktop **1.4.0+**, MLX **0.32.2**. Desktop **1.4.4 build 26** embarquera ce moteur et les correctifs de son inspecteur. Poids et outils Python de quantification ne font pas partie de l’archive native.

Le canal moteur doit être publié avec **`--latest=false`**, indépendamment du DMG Desktop. Distribution prévue **ad-hoc, sans notarisation Apple** ; signature et sélection du runtime final restent à consigner. Cette préparation ne remplace aucune app dans `/Applications` : l’app actuelle est conservée à la demande de l’utilisateur.

## Vérification et limites

**API : 33 tests CPU réussis, aucun ignoré**, format et Clippy strict passés. Ils exercent le vrai adaptateur avec bridge jetable et frontières HTTP/processus : causes terminales, stats invalides, admission, corps lent/trop grand, silence, EOF, JSON partiel, drainage, requête suivante propre, auth, startup interrompu et vrai TCP non lecteur. Les trois mutations de détection de fermeture, délai producteur et délai socket sont détectées puis restaurées. [Protocoles et preuves API](measurements/audit-api-lfm-20261016/api.md).

**Kani API : 65 obligations réussies / 3 couvertures satisfaites**, sur le vrai constructeur de compteurs et le total saturé. Domaine : deux u64 et un budget `Option<u64>` symboliques complets, sans hypothèse ni boucle. Cela ne prouve pas le parsing JSON, Tokio/Hyper, les signaux ou la concurrence. Le harnais Origin à deux octets de suffixe symboliques, unwind 64, a dépassé **30 secondes** : propriété **non prouvée**, diagnostic conservé.

**Profil mémoire/UI, campagne précédente distincte :** **80 tests Rust CPU réussis / 3 ignorés**, **220 obligations Kani / 11 couvertures** sur les domaines de dimensions documentés. Un vrai chargement LFM V2 a vérifié profil, capacités et contexte 4096/max131072, sans génération. Les **821 assertions / 180 états Swift** concernent le Desktop et ne valident pas le nouveau transport API. [Preuves UI/mémoire](measurements/lfm-mtp-ui-20261016/README.md).

Le packaging antérieur compte **45 tests réussis** et deux contrats CrossHair non confirmés. Ces contrôles ne prouvent ni I/O, ni MLX/Metal, ni la qualité universelle du modèle. Tests physiques ignorés et timeouts de preuve restent hors validation.

**À compléter avant publication :** suite intégrée finale, build livré `mlx,chat`, qualification numérique et mesures LFM retenues, CI du SHA exact, signature et cohérence archive/manifeste/binaire.

## Identité de la release préparée

Tag prévu : `engine-v1.4.5`. **SHA source, nom du manifeste et rapport final à compléter après le dernier build.** Les empreintes de la précédente construction locale ne décrivent pas encore ce lot API/LFM.

| Artefact prévu | Taille en octets | SHA-256 |
|---|---:|---|
| `MLXL3-Engine-v1.4.5-arm64.tar.gz` | À compléter | À compléter |
| Manifeste runtime publié | À compléter | À compléter |
| Rapport de validation final | À compléter | À compléter |

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw) pour l’intégration et les validations LFM/API ; [@HENK0O](https://github.com/HENK0O) pour les fondations mémoire/contexte de [PR31](https://github.com/0xZKnw/mlxl3/pull/31) et [PR32](https://github.com/0xZKnw/mlxl3/pull/32).
- [LiquidAI](https://huggingface.co/LiquidAI) pour LFM et [MLX](https://github.com/ml-explore/mlx) pour l’exécution Apple Silicon.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pour EXL3 et [PonyExl3](https://github.com/beamivalice/PonyExl3) pour le travail de conversion/inférence MLX utilisé dans cet écosystème.
- [Hyper](https://github.com/hyperium/hyper) et [Tokio](https://github.com/tokio-rs/tokio) pour HTTP, I/O et processus ; [Kani](https://github.com/model-checking/kani) pour les contrôles symboliques CPU.
