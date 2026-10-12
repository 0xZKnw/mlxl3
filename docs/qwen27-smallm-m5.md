# Qwen3.8-27B EXL3 : vérification small-M sur M5

Campagne des 6–7 octobre 2026, branche `optimize/qwen27-smallm-verify`, base parent main `5de518e0fbe7d3367416b9bc5999c7bd1f92c74d`.

**Aucune nouvelle accélération reproductible du débit livré n’est établie.** Le vrai MB3 grouped est prometteur au microkernel et exact dans les contrôles modèle effectués, mais la première campagne longue dérive fortement et expire ; la tentative sur secteur complète reste instable et change d’alimentation. Le prototype natif a donc été retiré. Le moteur conservé est identique à la base ; la branche ajoute les benchmarks, leurs tests, la CI et les preuves. Aucun push, PR, merge, changement de version ou remplacement de l’app.

## Résultats et décisions

| Piste | Correction observée | Performance observée | Décision |
|---|---|---|---|
| A : grouped M3, vrai MB3/NT1 | Partielles FP32, feedback FP16, logits complets et états/caches contrôlés exacts ; rollback et sessions MTP passent | Deux fenêtres micro positives ; verify court apparié +4,70 % ; débit long non concluant, y compris après la tentative sur secteur | Prototype retiré du runtime, patch et binaire archivés |
| MB4 grouped | Non testé | Non mesuré | Condition de gain livré MB3 non remplie |
| B : GEMV GDN a/b à vecteurs colonnes batchés | 162 cas, 96 poids et 48 paires, sorties FP16 exactes | Paire M3 : +27,31 % sur chaîne proxy, petit coût absolu | Filtre exact, pas d’intégration ni de gain modèle revendiqué |
| C : TensorOps BM16/BN32/BK16 | 598/34 816 mots FP16 finaux différents dès M2 | Aucun timing après divergence | Rejet immédiat |
| Retune / MTP D4…D7 | Non exécuté | Non mesuré | Coût small-M livré pas démontré amélioré |

## A — MB3 réel, sans padding

Le shader mapped existant conserve son ordre FMA/réduction par sortie. Le prototype sélectionne MB3/NT1 seulement sur M5, M3, MUL1, input5120, output16384/K2 ou output34816/K1…3, derrière `MLXL3_EXPERIMENTAL_GROUPED_MB3=1`, OFF par défaut. Le M3 separate wide était déjà MB3 sur main.

Trois warmups et 40 paires AB/BA par forme et fenêtre. Les fixtures couvrent 24 combinaisons K1…4/CB0…2 et deux géométries de split. Les quatre formes réelles sont également exactes, y compris leurs scales/transforms originaux et huit forwards avec feedback borné. Le gain indiqué est la médiane des ratios appariés, pas le ratio des médianes.

| Forme grouped, CB2/MUL1 | Gain isolé fenêtre1 / 2 | Gain chaîne fenêtre1 / 2 |
|---|---:|---:|
| 5120 → 10240 + 6144, K2 | 22,95 / 20,98 % | 30,55 / 25,47 % |
| 5120 → 17408 + 17408, K1 | 26,68 / 25,90 % | 25,96 / 26,26 % |
| 5120 → 17408 + 17408, K2 | 24,77 / 25,88 % | 24,98 / 24,72 % |
| 5120 → 17408 + 17408, K3 | 28,19 / 29,07 % | 26,96 / 28,26 % |

Contrôles physiques du prototype : prefill1/23/24/129/256, verify M2/3/4 et tous les préfixes retenus contre le chemin cible canonique ; logits, 128 états GDN/KV finis, hidden et erreurs/rollback. Sessions récursives MTP1…3 : huit blocs par profondeur, IDs/états et cache draft exacts. Ces contrôles testent des trajectoires et ne constituent pas une preuve de toutes les entrées.

Une première fenêtre native a été invalidée pour le débit parce que des tests auxiliaires ont été lancés pendant la mesure. Son log reste disponible pour la parité. La répétition seule fait huit paires AB/BA, après quatre warmups, avec huit blocs `verify_mtp` M3 dépendants et IDs imposés par passage. Après chaque passage, les 744 960 logits du dernier bloc et les 128 états sont finis et comparés en bits. Médianes A/B : **4,146245 / 3,886540 s** ; gain apparié **4,70277 %**, sept paires gagnantes sur huit. Le contrôle A dérive de 3,225 à 7,301 s : ce résultat court ne valide pas le débit soutenu.

Le bridge utilise le même binaire, flag0/1, MTP2, température0, top_k1, répétition1, cache OFF et contexte4096, avec 64 tokens de warmup puis 256 tokens demandés. Ordre prévu ABBA/BAAB, un enfant/modèle à la fois, attente de génération bornée à180 s et nettoyage/reaping via `JsonProcess`.

| Passage | Variante | Débit decode (tok/s) | Résultat |
|---|---|---:|---|
| 0 | A | 2,6673 | 256 tokens terminés |
| 1 | B | 2,6951 | 256 tokens terminés |
| 2 | B | 2,6172 | 256 tokens terminés |
| 3 | A | 1,5179 | 256 tokens terminés |
| 4 | B | Non mesuré | Deadline180 s pendant la génération mesurée |

Les quatre sorties complètes sont sur batterie ; le cinquième passage est sur secteur. Les quatre sorties complètes ont les mêmes hashes IDs/texte/cache_context, 141/225 propositions acceptées et 113 blocs. La dérive de contrôle d’environ −43 % interdit une attribution au kernel. Le rapport global reste **failed/parity=false**, puisqu’il est incomplet ; les passages validés restent conservés. Pas de ratio global présenté comme gain. Ces réglages MTP2 et ce prompt diffèrent du chat Desktop à7,7 tok/s : ces valeurs ne mesurent pas une régression de ce chat.

## Tentative A sur secteur — résultat séparé

L’audit des premières conditions révèle le passage sur secteur au cinquième passage précédent. Le Mac est observé branché76% avant une nouvelle campagne préenregistrée `SMALLM-2026-10-07-A-AC`, sans changement de source/protocole/binaire. Critères fixés avant mesure : alimentation constante, contrôle A max/min≤1,10 et gain apparié médian≥3% dans chacune des fenêtres ABBA/BAAB.

Les huit passages terminent :2048tokens mesurés+512warmup, mêmes hashes IDs/texte/cache_context/count,141/225propositions acceptées et113blocs. `mb3-bridge-ac-256.json` reste **complete/parity=true** pour ces sorties ; les contrôles logits/états du prototype sont les tests physiques distincts décrits plus haut.

| Passage | Variante | Alimentation relevée au départ | Decode (tok/s) |
|---|---|---|---:|
| 0 | A | Secteur, charge78% | 4,8082 |
| 1 | B | Secteur, charge79% | 2,4149 |
| 2 | B | Secteur, batterie80%, charge arrêtée | 2,5643 |
| 3 | A | Secteur, batterie80%, charge arrêtée | 2,4809 |
| 4 | B | Batterie80% | 2,6452 |
| 5 | A | Batterie80% | 2,4253 |
| 6 | A | Batterie80% | 2,6687 |
| 7 | B | Batterie80% | 2,7505 |

Le relevé final est sur batterie78%. Les critères d’alimentation et stabilité échouent : contrôle max/min**1,982516**, médianes appariées des fenêtres**−23,2064%/+6,0657%**. Le second chiffre reste diagnostique et ne suffit pas à promouvoir le candidat. La référence chute aussi : pas de ralentissement imputé au seul MB3, pas de cause thermique démontrée. **Décision non concluante, prototype retiré conservé uniquement en archive.** Pas de nouvelle répétition après cette clôture. Résumé et conditions finales : `mb3-bridge-ac-summary.json`.

## B — préserver le GEMV canonique de chaque ligne

Le helper dense existant et un GEMM M×K ne sont pas présumés exacts : l’échec historique FIX-09 reste pertinent. Le filtre utilise `weight[None,:,:] @ x[:,:,None]`, avec batch de matrice broadcasté et vecteurs colonnes distincts, puis une paire a/b concaténée. La lecture du [dispatch officiel MLX0.32.2](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/matmul.cpp) explique le choix GEMV M1 avec K≥16×N ; la correction est contrôlée numériquement.

Les 96 poids a/b et 48 paires passent en M3 ; couche0 passe M2…8, chaînes de huit projections incluses avant timing. Paire M3 : isolé **0,237438 → 0,230021 ms**, chaîne **0,067401 → 0,049810 ms/projection**, gain apparié chaîne **27,31 %**. Le coût proxy de48 paires est de l’ordre de3,24 ms, et l’économie proxy de0,84 ms, face à un bloc verify natif autour de518 ms dans une autre fenêtre. **Frontières différentes : cette estimation n’est ni une borne formelle ni une prédiction de débit.**

Le filtre identifie une géométrie exacte intéressante, mais ne justifie pas sa promotion au modèle dans cette campagne. Aucun nouveau kernel dense natif, aucune préparation de poids, pas de parité modèle complet/bridge pour B. Le script et les résultats permettent de réexaminer cette piste après un profil exploitable.

## C — divergence TensorOps

Après A/B, une seule forme originale est sélectionnée : gate couche10, 5120→17408/K2/MUL1, BM16/BN32/BK16, activations paddées localement à16 et sortie limitée aux lignes réelles, flag standalone OFF par défaut. Le premier M2 compile sur M5 mais change **598 mots FP16 finaux sur34 816**, avec100 mots internes différents ; les sorties restent finies. Arrêt immédiat : timings vides, pas de M3/4/6/8, pas d’essai long, pas de dispatch runtime ajouté.

## Inventaire, capture et conditions

[Tableau complet des 80 géométries EXL3 et 21 lignes de mesure GDN](measurements/qwen27-smallm/inventory-table.md), avec projections, input/output, grouped/separate, K/CB, M/MB/NT/SG/splits et médianes isolées/dépendantes. Les chaînes micro utilisent les vrais poids/scales/transforms, mais un feedback synthétique : l’essai natif A fournit séparément une vraie chaîne verify.

MacBook Air M5, 10 GPU cores, 24 GiB, macOS27.2, MLX0.32.2, Python3.12.14, NumPy2.5.3. Batterie en décharge :56 % au début, 46 % en charge sur secteur au cinquième passage du bridge ; swap autour de2,9–3,1 GiB. Aucun warning `pmset` enregistré. Fréquences, températures et cause de la dérive **non mesurées** ; aucune attribution thermique démontrée.

Une capture Metal courte qkv/z M3 de production a réussi après trois warmups et parité M1 exacte (`MTL_CAPTURE_ENABLED=1`, `mx.metal.start_capture`, évaluation/synchronisation, `stop_capture`). Le `.gputrace` de204 309 985 octets reste local dans `../work/smallm-qkv-m3.gputrace`, hors Git. `xctrace` absent : occupancy, registres/spills, bandwidth, stalls, temps GPU pur et nombre de dispatchs **non mesurés**, capture non analysée. Métadonnées dans `metal-capture.json`.

## Validation et provenance

Source finale : aucune différence native/Cargo/shaders/version par rapport à la base. Fmt, Clippy strict all-targets release avec MLX/chat, build release MLX/chat réussis ; **62 tests Rust passés, 55 ignorés**, aucun ignoré assimilé à une exécution physique. Ruff format/check et py_compile réussis ; suite Python pertinente finale **258/258**, comprenant erreurs CLI, rapports partiels, zéro cas, divergence, valeurs non finies, interruption, timeout/transport, démarrage lent et protection des résultats existants. Les quatre timeouts du transport sous charge du premier essai auxiliaire restent archivés ; la relance seule231/231 puis la suite finale passent sans modifier leurs délais.

Le contrôle final du comparateur reproduit trois fausses validations avant correction : métrique booléenne, hash nontexte, contexte vide (`bridge-validator-red.log`). La garde renforcée refuse ces entrées ; les douze sorties bridge complètes enregistrées passent aussi cette version, avec leur texte brut retrouvé dans `cache_context` et confirmé par le SHA256 original (`bridge-validator-raw-audit.json`). La source de mesure avant durcissement est conservée dans `bridge-campaign-source.py.txt` ; ses hashes et ceux du code final sont distingués dans `campaign-identity.json`. Timers, GPU et paramètres de génération inchangés.

Trois suites avant correction du harnais QMM passent255cas mais échouent sur deux fixtures préexistantes au délai300ms (erreurs spécifiques remplacées par un timeout d’écriture). Les sept cas passent seuls ; éjecter le modèle ne suffit pas. Un démarrage retardé400ms reproduit le défaut, puis le budget de ces fixtures passe à1s avec watchdog10s. L’oracle reste le message d’erreur spécifique, le rapport vide/paritynull et le reaping de tousPID : huitcas verts, puis **258/258** en suite complète. Aucun délai du moteur ou benchmark de performance modifié. Tous les logs rouges restent archivés.

Avant retrait du prototype, Kani0.68.0/CBMC6.11.0 vérifie le **vrai sélecteur Rust** avec sept entrées scalaires symboliques, aucune `assume`, aucune boucle :164 obligations et6/6 covers, zéro échec. Suite CPU du prototype35/35 harnais. Ces preuves concernent le patch archivé, pas Metal/MLX/FFI, la concurrence ou la génération complète. CrossHair absent dans le venv : tentatives et diagnostics conservés, Python et Metal **non prouvés formellement**. CI étendue localement ; aucun run GitHub de cette branche locale, aucun statut distant vert revendiqué.

Les seize fichiers du checkpoint/tête sont empreintés dans [campaign-identity.json](measurements/qwen27-smallm/campaign-identity.json) ; dix fichiers de cible sont identiques au manifeste historique1.4.1. Les headers safetensors complets et leurs hashes figurent aussi dans les rapports. Le moteur installé1.4.1 garde SHA256 `79f027af26c18cb4fecead8d55b0dfbf3500671d8a2b42360bc1bc4d7dc28721`. Qwen a été rechargé et l’application affiche **Prêt sur Metal**, conversation et réglages conservés.

Le binaire prototype MB3 est archivé localement dans `../work/smallm-mb3-prototype-rs`, SHA256 `232f8c3026b2eccddbb7ea721b6e4d27f27533a574e2b935c6b62f33d9f2317f`. Son [patch complet](measurements/qwen27-smallm/mb3-prototype.patch) inclut garde, harnais Kani, dispatch et contrôle modèle apparié ; les sources et la décision sont dans [prototype-provenance.json](measurements/qwen27-smallm/prototype-provenance.json). Le bridge small-M doit utiliser ce prototype ou une reconstruction explicite de ce patch : le flag MB3 n’existe plus dans le runtime final. Les mesures de ce prototype ne valident pas une optimisation conservée.

Commandes, options et résultats bruts : [protocol.md](measurements/qwen27-smallm/protocol.md). Les deux campagnes bridge sont closes ; leurs conditions variables et la forte dérive du contrôle empêchent de valider une accélération. Le candidat reste retiré du runtime. Une éventuelle campagne future devra garantir une alimentation et un contrôle stables, en conservant ces résultats négatifs.
