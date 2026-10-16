# MLXL3 moteur 1.4.5 — fiabilité API et LFM

Ce moteur accompagne **Desktop 1.4.5 build 27**. Desktop1.4.4 a été retiré du canal
stable après un défaut de préflight de l’app ; le moteur publié reste inchangé.
Il corrige la fin de génération,
le cycle de vie du serveur HTTP et le profil mémoire LFM.

## Nouveautés

- **`GET /v1/capabilities`** décrit l'adaptateur HTTP et les capacités publiques
  du bridge. Stop sequences, tools HTTP, JSON Schema contraint, images, MTP et
  DFlash restent explicitement non pris en charge par cet adaptateur.
- **`GET /ready`** répond 503 si le bridge est occupé, mort ou invalidé ;
  `/health` conserve son rôle de liveness.
- Profil de contexte LFM dense/MoE. Sur LFM2.5-24B-A2B :
  **20 480 octets/token + 245 760 octets fixes**.

## Correctifs

- `complete.stop_reason` distingue `eos`, `max_tokens` et `context_full`.
  L'API rend `stop`, `length` et `length`, en JSON comme en streaming.
- Une seule génération est admise avant lecture du corps. Une requête concurrente
  reçoit **429 et `Retry-After: 1`**.
- Une déconnexion est détectée même avant le premier delta. Le bridge est annulé
  et drainé avant réutilisation ; un protocole irrécupérable le rend indisponible.
- Corps, événements, sorties et attentes sont bornés. Les écritures sont bornées
  jusqu'au transport TCP, sous le buffer Hyper ; un progrès partiel ne réarme pas
  le délai avant un flush complet.
- Causes terminales ou compteurs invalides produisent une erreur explicite.
  Les erreurs sont bornées à une frontière UTF-8 ; les pénalités non représentables
  en f32 sont refusées avant toute inférence.
- Origin, Host, CORS et bearer suivent des règles locales explicites. CLI sans
  Origin accepté ; clé obligatoire hors loopback. L'écoute IPv6 accepte `::1`
  et `[::1]`.
- SIGINT/SIGTERM sont installés avant le chargement. Toutes les sorties terminent
  et attendent le child puis le lecteur stderr.
- La version du paquet moteur est dérivée de Cargo ; `--version` reste prioritaire.

## Outils de calibration et diagnostics

Les [PR #37](https://github.com/0xZKnw/mlxl3/pull/37) et
[PR #38](https://github.com/0xZKnw/mlxl3/pull/38) ajoutent au dépôt les scripts de
conversion officielle EXL3 sur H100, la collecte facultative des entrées réellement
routées et les diagnostics appariés BF16/EXL3. Les reprises vérifient les artefacts
et les seuils de fidélité refusent les candidats régressifs. Ces outils Python
restent séparés du runtime natif embarqué.

Les trois candidats de recalibration du routeur n'ont pas franchi les gardes de
qualité : **V2 reste le modèle publié**, avec ses poids inchangés. L'intégration
n'effectue aucune nouvelle quantification et ne revendique aucune baisse KL/PPL.
[Résultats et limites](lfm25-router38-results-2026-10-15.md).

## Performances et mémoire

**Option expérimentale** : `MLXL3_LFM_MOE_PREFILL=1` active les microblocs MoE≤8
sur la géométrie V2 et les projections experts K4/MCG qualifiées. Attention,
convolution, normes, résiduels, gates et head gardent les formes M1 canoniques.
Défaut, `0` ou valeur invalide : chemin sériel. Les autres modèles gardent leur
sélection existante.

Les comparaisons numériques du chemin actif passent sur la matrice historique
32 cas jusqu'à 266 tokens, complétée par des contrôles affectés jusqu'à 600 :
128 000 logits F16 terminaux,50 états et routes/affinités des38 MoE, puis17
continuations forcées bitidentiques, non vides et finies. Cela ne prouve ni tous
les logits intermédiaires ni le contexte maximal131072.

Les deux séries finales de vitesse sont complètes, mais leurs dérives dépassent
les seuils de stabilité ; les études antérieures interrompues sont conservées. **Aucun gain de vitesse n'est livré comme établi ; l'option reste
OFF par défaut.** Le prototype KV à capacité a été retiré : sur son crible,
préfill−0,479%, décodage+0,391%, aucun bénéfice utile≥3%.
[Protocoles, sources et décisions](measurements/audit-api-lfm-20261016/lfm-final.md).

Le profil mémoire est logique, batch 1/F16 ; il exclut poids, backings retenus,
activations, cache allocateur et RAM processus. Aucun nouveau checkpoint ou
changement de quantification, ni amélioration KL/PPL, ne fait partie de ce moteur.

## Installation et compatibilité

**MLXL3-Engine-v1.4.5-arm64.tar.gz** et son manifeste
**MLXL3-Engine-v1.4.5-arm64.json** : Apple Silicon, macOS **26.2+**, Desktop
**1.4.0+**, MLX **0.32.2**. Desktop **1.4.5 build 27** embarque ce moteur et corrige
son inspecteur. Les poids et outils Python de quantification sont séparés.

Publication moteur avec **`latest=false`**, indépendamment du DMG Desktop.
Distribution **ad-hoc, sans notarisation Apple**. L'app installée est conservée.

## Vérification et limites

- **118 tests Rust CPU /3 ignorés**, dont **33 fixtures API incluses** : vrai
  adaptateur, faux bridge jetable, TCP/processus, stats invalides, déconnexion,
  drainage, startup, UTF-8 et client non lecteur. Trois mutations de fermeture
  et délais sont détectées. Format et Clippy strict `chat`/`mlx,chat` passent.
- Suite intégrée : **1267 Python réussis /64 sautés** avec Pony/Torch CPU, puis Desktop complet ;
  les changements Rust finaux ont été retestés séparément. Les ajouts Python des
  PR37/38 ont une vérification combinée distincte, consignée dans le manifeste
  final ; les correctifs LaTeX sont détaillés dans le changelog Desktop.
- Kani : cause terminale **15/4**, usage API **65/3**, gardes routeur/préfill/
  géométrie **121/13**, profil mémoire **220/11** dans leurs domaines documentés.
  Sélecteur opt-in final **33 obligations/5 couvertures**, chaînes0..4 octets,
  unwind6, booléens arbitraires, aucune hypothèse. Model checking borné du CPU,
  sans preuve universelle MLX/Metal ou concurrence.
- Limites conservées : Origin symbolique timeout 30 s, offsets généraux timeout 120 s.
  Le prototype KV retiré garde ses diagnostics de budget timeout 120 s et CBMC
  C++ incompatible avec libc++ du SDK. Aucun timeout ou contrôle ignoré n'est
  transformé en succès.
- Les vrais parcours bridge/API ont été exercés avec le préfill actif ; la cause
  de fin, la récupération de la requête suivante et l'arrêt des processus passent.
  Les résultats du binaire livré sont consignés séparément des prototypes.

L’intégration37/38/39 conserve les poidsV2 et les sources natives/UI qualifiées
byte-identiques. CrossHair confirme29postconditions scalaires dans son modèle ;
7conditions de chaînes restent non confirmées après20s chacune. Les64tests
sautés sont63tests Metal explicitement exclus de la suite CPU et1modèleLing
absent. [Commandes, couverture et échecs conservés](measurements/audit-api-lfm-20261016/integration-37-38-39/verification.json).

Les identités source/binaires, jobs CI du commit exact et leurs éventuelles
interruptions, contrôles du paquet signé, noms/tailles/SHA-256 sont publiés dans
**MLXL3-v1.4.4-validation.json**, joint à cette release et à Desktop 1.4.4.
Une CI en cours ou interrompue n'est pas présentée comme verte.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), intégration et validation API/LFM dans
  [PR #39](https://github.com/0xZKnw/mlxl3/pull/39), ainsi que les outils de
  calibration/diagnostic des [PR #37](https://github.com/0xZKnw/mlxl3/pull/37) et
  [PR #38](https://github.com/0xZKnw/mlxl3/pull/38) ;
  [@HENK0O](https://github.com/HENK0O), bases mémoire/contexte des
  [PR #31](https://github.com/0xZKnw/mlxl3/pull/31) et
  [PR #32](https://github.com/0xZKnw/mlxl3/pull/32).
- [LiquidAI](https://huggingface.co/LiquidAI), modèles LFM ;
  [MLX / mlx-lm](https://github.com/ml-explore/mlx-lm), exécution Apple Silicon
  et architecture de référence.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3), format EXL3 ;
  [PonyExl3](https://github.com/beamivalice/PonyExl3), conversion/inférence MLX.
- [Hyper](https://github.com/hyperium/hyper) et [Tokio](https://github.com/tokio-rs/tokio),
  transport HTTP/processus ; [Kani](https://github.com/model-checking/kani),
  vérification symbolique des contrats CPU.
