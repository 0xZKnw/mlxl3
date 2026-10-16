# État des 40 propositions de l’audit API / LFM

Révision documentaire du 16 octobre 2026, campagne
`AUDIT-2026-10-16-API-LFM` dans [opti.md](../opti.md). Source du backlog :
« MLXL3 : audit technique et feuille de route d’amélioration », arrêté au
14 octobre, §19. Le document de 1 629 lignes a été lu intégralement ; sa
[revue et son empreinte](measurements/lfm-mtp-ui-20261016/audit-review.json)
sont conservées. Ses recommandations sont distinguées des résultats observés.

Baseline de l’audit : `c9aa43a312c63246eeb4b5a680ada26092c65a1c`.
Point de départ de cette campagne : `2eae717fb0bccc34d6ab0c7e6ca2da8ce1fbb230`
(correctifs UI/mémoire et paquet local), branche `codex/audit-api-lfm`.
Les chemins ci-dessous désignent le code et les preuves inspectés pendant cette
campagne. **Corrigé local / qualifié numériquement ne signifie pas publié,
installé ou plus rapide.** La rédaction de cette matrice n’ajoute aucune mesure.
Les résultats API CPU et LFM physiques restent deux validations distinctes.

Légende : **corrigé local** = code et contrôles liés disponibles, sans présumer
publication/installation ; **présent** = brique existante, pas preuve universelle ;
**planifié** = correction en préparation ; **à mesurer** = mécanisme plausible
sans gain établi ; **distinct** = chantier hors de la campagne immédiate.
Le passage en « validé » exige les contrôles de sortie indiqués et l’identité
du code testé, pas une simple case cochée.

## Fiabilité, API et documentation — F01 à F10

| ID | Objet / nature | État actuel | Code / preuve existante | Condition de clôture et limite |
|---|---|---|---|---|
| F01 | Cause d’arrêt explicite ; bug confirmé | Corrigé local, transport CPU validé | `GenerationStopReason`, `main.rs::bridge_generate_round` et `openai.rs` ; [rapport API](measurements/audit-api-lfm-20261016/api.md) | EOS→`stop`, budget/contexte→`length`, stream et non-stream ; ancien comportement rouge. Contrat CPU : 15 obligations Kani/4 couvertures. Le vrai parcours API→modèle passe aussi les13contrôles du [rapport final](measurements/audit-api-lfm-20261016/lfm-final.md). |
| F02 | Admission/file bornée ; risque structurel | Corrigé local, fixtures CPU validées | `openai.rs::handle_inner` : sémaphore de capacité1 avant lecture du corps ; [rapport API](measurements/audit-api-lfm-20261016/api.md) | Saturation→429 + `Retry-After: 1`, corps borné16Mio/10s et place libérée après échec. Ne constitue pas un quota global de connexions HTTP. |
| F03 | Annulation avant premier delta ; cycle de vie | Corrigé local, régression/mutation détectées | `response_closed`, `cancel_and_drain`, propriété du child ; [rapport API](measurements/audit-api-lfm-20261016/api.md) | Fermeture détectée indépendamment des deltas, JSON partiel préservé, drainage requestID avant réutilisation ; protocole irrécupérable→503. SIGTERM pendant chargement nettoie aussi child/stderr. HTTP/1 pipeliné peut retarder EOF ; délais globaux restent bornés. |
| F04 | Backpressure/client silencieux ; risque | Corrigé local, vrai TCP non lecteur validé | `openai.rs::send_bytes`, [openai_io.rs](../native/src/openai_io.rs) ; [rapport API](measurements/audit-api-lfm-20261016/api.md) | Frames16Kio, sortie16Mio, délai producteur et socket séparés ; progrès partiel ne réarme pas le timer avant flush complet. Mutation timer détectée. Ce délai de phase bloquée ne limite pas toute la durée d’une connexion. |
| F05 | Liveness/readiness/occupé ; limite produit | Corrigé local, fixtures CPU validées | `openai.rs::bridge_alive`, `/health`, `/ready` ; [rapport API](measurements/audit-api-lfm-20261016/api.md) | `/health` reste200 pour le serveur vivant ; `/ready` renvoie503 si occupé, mort ou invalidé. Champs `ready`, `busy`, `bridge_healthy` explicites ; vérification et signal PID sous le même verrou. |
| F06 | Origin/Host/auth locale ; durcissement | Corrigé local, tests IPv4/IPv6/auth validés | `listen_addr`, `loopback_authority`, `origin_allowed`, CORS ; [rapport API](measurements/audit-api-lfm-20261016/api.md) | Origin HTTP(S) loopback seulement, rejet null/étranger/malformé/dupliqué ; sans clé, Host présent loopback ; clé requise hors loopback. CLI sans Origin accepté. Tentative Kani Origin timeout30s : non prouvée, aucun exploit navigateur revendiqué. |
| F07 | Documentation native obsolète ; dette confirmée | Corrigé documentaire local | [README natif](../native/README.md), [archive exacte datée](native-runtime-readme-before-audit-2026-10-16.md) | Port déjà natif, QMM/prefixes/MTP présents et limites LFM/API décrits ; liens locaux et archive contrôlés. Pas de validation d’exécution induite par la documentation. |
| F08 | Matrice de capacités commune ; partiellement présent | UI effective corrigée et endpoint API ajouté ; contrat transversal partiel | `main.rs` ready/generation_mode, `openai.rs::api_capabilities` ; `Domain.swift`, `StudioModel.swift`, `GenerationInspector.swift` ; [preuve UI](measurements/lfm-mtp-ui-20261016/README.md) | `/v1/capabilities` décrit séparément adaptateur HTTP et capacités publiques du bridge ; champs privés filtrés. Aucun outil/JSON/vision/MTP HTTP ajouté par une étiquette. Le registre commun général reste une extension. |
| F09 | Suite conformité API ; couverture insuffisante | Étendue et validée localement,33 fixtures CPU | [openai_tests.rs](../native/src/openai_tests.rs), [cible réelle](../tests/openai_transport.rs), [manifestes et limites](measurements/audit-api-lfm-20261016/api.md) | Stream/JSON, EOS/budgets, UTF-8, admission, déconnexion, EOF/JSON/silence/stats invalides, startup/signaux/TCP et requête suivante propre. Trois mutations détectées ; Kani usage65 obligations/3 couvertures. Première tête : CI fonctionnelle Linux/macOS/Desktop réussie ; runner Kani interrompu sur preuve DFlash historique. CI de la tête finale encore à inspecter. |
| F10 | Long contexte Flash ; limite matérielle historique | Distinct, non qualifié dans la campagne citée | PR34 `fb782d985972ab9f4eed2926389365c713811095`, `docs/flash-profile-after-hc-2026-10-12.md` sur cette branche | Avant promotion Flash : tests physiques 2047/2048/2049 et proche limite, OFF/ON, reset/reprise/chunks, logits/états/RAM. Limite configurée 4096 ≠ qualification. Ne pas modifier ni fusionner cette branche par effet de bord. |

## Moteur, mémoire et kernels — K01 à K18

| ID | Objet / nature | État actuel | Code / preuve existante | Condition de clôture et limite |
|---|---|---|---|---|
| K01 | Trace par phase ; instrumentation partielle | À compléter selon besoin de diagnostic | `main.rs` statistiques de génération ; `ChatComponents.swift::StatsRow`; rapports Flash inclusifs | Timestamps monotones/requestID et définitions TTFT/prefill/decode/transport/UI ; ne pas additionner phases imbriquées ni confondre temps moteur et visible. |
| K02 | Routeur LFM biaisé multi-ligne ; extension | Qualification numérique locale validée sur binaire figé | `router.rs::topk_biased`, `contracts.rs` ; [rapport LFM-01](measurements/audit-api-lfm-20261016/lfm-01.md) |1365 fixtures contre tri CPU indépendant, IDs/ties/scores F16 exacts, entrées invalides refusées ; routes complètes38MoE identiques sur matrice V2. Pas de preuve universelle GPU ni de gain vitesse induit. |
| K03 | Prefill LFM MoE ; limitation confirmée | Chemin opt-in qualifié numériquement ; vitesse non concluante | `lfm2.rs::prefill_moe_exact`, sélecteur commun chunk/dispatch, config+7296projectionsK4/MCG ; [rapport final](measurements/audit-api-lfm-20261016/lfm-final.md) | Matrice historique32cas et nouveau binaire5f91/f672 : T8,266 et600/préfixes0/17+capturepassive128,128000logitsF16/50états/routes38MoE et17continuations bitidentiques. DéfautOFF, seule valeur exacte1 active le chemin admissible ; défautON retiré avant livraison. Deux ordres finaux sur runtime signé complets/parité vraie mais dérives>3% : vitesse non qualifiée, défautOFF maintenu. Anciennes interruptions conservées. |
| K04 | KV LFM à capacité ; copies visibles | Rejeté pour ce lot, prototype retiré | Patch et preuves dans [archive physique](measurements/audit-api-lfm-20261016/physical-evidence.tar.gz), [décision](measurements/audit-api-lfm-20261016/lfm-final.md) | Primitives5/5 et modèle20/20 bitexacts sur prototype ; stockage après matérialisation ne prouve pas donation bridge. Crible FR266 : préfill−0,479%, décodage+0,391%, durée+0,483%, pas de bénéfice≥3%.1314lignes retirées, concat conservé ; autres formes non mesurées, limites formelles archivées. |
| K05 | Derniers logits utiles ; coût hypothétique | Diagnostic fait, aucun nouveau gain établi | `lfm2.rs::forward`, historique AUDIT-03/PERF-16 ; [plan physique](measurements/audit-api-lfm-20261016/physical-plan.md) | MLX peut jeter les heads intermédiaires lazy ; slicer avant head a déjà changé des logitsF16. Aucun changement de tête/représentation sans essai distinct ; l’évaluateur conserve toutes positions requises. |
| K06 | Plans Rust préparés ; coût hypothétique | Cache factory natif déjà présent ; extension Rust non mesurée | `linear.rs`, `moe.rs`, cache C++ complet5champs/128entrées ; [plan physique](measurements/audit-api-lfm-20261016/physical-plan.md) | Ne pas doubler ce cache. Profiler formats/CString côté Rust avant préconstruction ; préserver device/dtypes/sortie puis démontrer gain modèle. |
| K07 | Vues et backing ; risque de rétention | Diagnostic fait, gain MoE non établi | `lfm2.rs::ShortConv::forward`, `memory_policy.rs` ; [plan physique](measurements/audit-api-lfm-20261016/physical-plan.md) | MoE M1 : environ120Kio de backing nominal supplémentaire, pas mesure de fuite. Dense non qualifié ici ; `mx::copy` peut partager le backing. Aucun compactage systématique livré sans coût/bénéfice. |
| K08 | Mémoire explicable ; partiellement présent | Profil logique LFM corrigé local ; tableau détaillé restant | `lfm2.rs::context_memory_profile`, `ContextMemoryProfile.swift`, `StatsRow`; [ready physique](measurements/lfm-mtp-ui-20261016/physical-ready.json) | Profil LFM V2 : 20480 octets/token +245760 fixes vérifié ; distinguer poids fichier, MLX actif/cache/pic, footprint, aliases/backing. Le profil logique n’est pas un registre d’allocations physiques. |
| K09 | Préparation commune des projections ; partiellement présente | Groupes existants ; extensions à mesurer | `linear.rs::ProjectionGroup::forward`, projections groupées LFM/Qwen | Partage seulement si input, échelles, rotation/codebook et arrondis compatibles ; oracle indépendant, bénéfice après coût préparation. |
| K10 | Histogramme stable au lieu de tri ; proposition | À mesurer ; antécédent issue36 | `moe.rs::route_plan` utilise argsort | Reproduire permutation, inverses, ties et ordre canonique ; routes adverses puis vrai MoE ; éviter re-prototypage sans hypothèse nouvelle inscrite. |
| K11 | Contributions experts directes ; proposition | À mesurer ; antécédent issue36 | `moe.rs` route/expert output et reconstruction | Écrire contributions dans slots déterministes puis réduction canonique ; atomiques float non assimilées à la référence ; vérifier gain mémoire et débit. |
| K12 | Activations MoE par blocs ; proposition | Microblocs≤8 qualifiés numériquement pour V2 ; contrôleur général distinct | `lfm2.rs::prefill_moe_exact`, chunking bridge ; [rapport final](measurements/audit-api-lfm-20261016/lfm-final.md) | Voie segmentée≥64routes historiquement divergente laissée inactive. Parcours réel et annulation par défaut passent ; campagne vitesse interrompue, aucun contrôleur adaptatif général revendiqué. |
| K13 | Décodage fragment QMM mutualisé ; R&D | À mesurer ; issue36 existante | [shader QMM](../native/shaders/_qmm_tensor_kernel.metal), dispatch `linear.rs` | Démontrer extractions redondantes et mêmes codewords/arrondis ; microkernel puis modèle, occupation/registres inclus. Aucun multiplicateur annoncé. |
| K14 | Réutilisation entre blocs tokens ; R&D | À mesurer ; issue36 existante | Shader QMM/dispatch existants, sans gain de cette variante établi | Amortir stockage/synchronisations et conserver réduction ; tester tailles/routes réelles, puis A/B modèle après parité. |
| K15 | Vérification cible multi-ligne ; partiellement présente | Qwen verify_mtp existe ; optimisation à mesurer | `qwen35.rs::verify_mtp`, snapshots/restore ; rapports MTP/DFlash | Comparer logits et tous états après chaque préfixe accepté (0/partiel/tout), EOS/budget ; coût par token réellement livré, pas acceptation seule. LFM MTP absent. |
| K16 | Contrôleur commun RAM/draft/SSD ; proposition | Distinct R&D | `memory_policy.rs::AdaptiveMtp`, budgets existants ; SSD PR34 distinct | D’abord petite matrice fixe OFF/MTP/budgets, corpus tenu à part ; montrer gain coût utile sans oscillation/évictions coûteuses. Pas de gain additionné. |
| K17 | Préparation bornée des kernels fréquents ; proposition | À mesurer | Cache natif/kernel dispatch existants ; warmups spécifiques historiques | Ensemble borné par modèle/mode, clés versionnées, mesurer démarrage froid + premier TTFT + RAM ; ne pas précompiler tout par défaut. |
| K18 | Autotuner par capacités/contrats ; proposition | Dispatch par device/forme présent ; tuner général non établi | `contracts.rs`, `linear.rs`, cache noyaux natif | Variantes d’abord prévalidées dans même classe numérique ; identité device/MLX/binaire/checkpoint, repli déterministe ; recherche autorisée ≠ livraison validée. |

## Produit et développement — A01 à A12

| ID | Objet / nature | État actuel | Code / preuve existante | Condition de clôture et limite |
|---|---|---|---|---|
| A01 | Préférence/mode effectif/repli ; bug confirmé | Corrigé et vérifié local, non installé/publié par ce lot initial | `StudioModel.swift` mtpApplicable/dflash2Applicable/samplingLocked, `GenerationInspector.swift`; [preuve complète](measurements/lfm-mtp-ui-20261016/README.md) | 821 assertions/180 états, échec ancien et mutation détectés, OFF et curseurs vérifiés visuellement ; vrai ready LFM. Maintenir hors génération/tuning/erreur et lors retour Qwen. Anciennes valeurs déjà écrasées non récupérables. |
| A02 | Diagnostic exportable expurgé ; proposition | Informations dispersées présentes, export global non établi | `runtime-info`, `Domain.swift`, `GenerationInspector.swift`, rapports de mesures | Export reproductible versions/capacités/métriques/erreurs ; zéro prompt, secret, réponse ou chemin personnel par défaut ; redaction testée. |
| A03 | Stop sequences ; fonctionnalité absente de l’API | Distinct, non implémenté au départ | `openai.rs::bridge_request` refuse stop non vide | Automate/suffixe borné, chevauchements token/UTF-8, plusieurs stops et thinking ; mêmes sémantique, comptage et cause terminale CLI/app/API. |
| A04 | Tool calls HTTP ; nouvelle fonctionnalité | MCP/app présent, HTTP non supporté | `openai.rs` refuse tools/tool_choice ; [mcp.rs](../native/src/mcp.rs), parseurs natifs existants | Cycle client complet IDs/arguments/résultat/reprise/stream/erreurs ; description d’outil ne vaut pas autorisation serveur d’exécution. Dépend F01/F08. |
| A05 | JSON Schema contraint ; nouvelle fonctionnalité | Format texte seul, refus explicite | `openai.rs` validation response_format | Sous-ensemble annoncé, contraintes/Unicode/JSON incomplet/schémas impossibles et budgets de compilation ; pas simple consigne au modèle, distribution modifiée explicitement. |
| A06 | Presets par modèle/usage ; extension produit | Sampling persistant présent ; presets complets à inventorier | `StudioModel.swift` réglages libres/sampling effectif, `ConversationStore.swift` | Paramètres utilisateur séparés des contraintes MTP ; profils transparents, réversibles et testés, pas recette universelle présentée comme optimale. |
| A07 | Persistance incrémentale ; coût hypothétique | Protection existante, évolution à mesurer | `ConversationStore.swift::save/write` sérialise workspace, valide ancien et backup atomique ; `StudioModel.swift::schedulePersistence` détache I/O | Profil gros historiques avant migration ; conserver refus corruption/version future, révisions monotones, backup/reprise et export ; snapshot MainActor à mesurer séparément. |
| A08 | Longs blocs code ouverts ; partiellement présent | Chunking/virtualisation déjà implémentés ; coût parse à mesurer | `MarkdownResponseView.swift` CodeTextChunker(~2048), cache lexical, LazyVStack grands blocs, copie source intégrale | Mesurer fence ouverte/fermée, parsing/colour/layout/hitches ; préserver retours ligne et copie intégrale ; ne pas réimplémenter la présentation déjà présente. |
| A09 | Navigation/recherche/branches ; extension | Navigation/import/export existants ; branches exactes à inventorier | `StudioModel.swift::exportConversations/importConversations`, `Domain.swift`, vues de conversation | Édition invalide suffixe ou crée branche ; cache limité au préfixe exact, réglages/provenance cohérents et migration testée. Aucune absence générale déduite de l’audit ciblé. |
| A10 | Sessions inactives SSD ; capacité/reprise | Distinct, proposition issue36 | Snapshots hybrides Qwen en mémoire ; stockage session durable non établi dans ce lot | Snapshot complet versionné/atomique/checksum/identité, disque borné, corruption/reprise/branches sans contamination ; I/O experts critiques prioritaires. |
| A11 | Developer ID/notarisation ; distribution | Ad-hoc actuel ; chantier externe distinct | `scripts/build-macos-app.sh` utilise codesign --sign - ; [release1.4.3](release-v1.4.3.md) | Nécessite identité Developer ID valide et accès notarisation Apple ; disponibilité non vérifiée, aucun secret demandé/lu. Release ad-hoc honnêtement décrite reste possible ; ne pas annoncer notarisation. |
| A12 | Catalogue structuré essais négatifs ; productivité | Journal/rapports existants ; index dérivé à concevoir | [opti.md](../opti.md), `docs/measurements`, issue36 et rapports SSD | Index avec source/shape/device/SHA/contrat/décision/condition de reprise ; liens traces et échecs conservés ; ne pas remplacer historique ni inventer métriques. |

## Vérification, périmètre et livraison

Le correctif UI/mémoire local est documenté dans
[son dossier de preuves](measurements/lfm-mtp-ui-20261016/README.md).
Ses tests ne valident pas les changements API/LFM suivants. Une mesure du modèle
BF16 contre EXL3 évalue qualité+runtime ; comparer deux moteurs sur les mêmes
poids et tokens imposés évalue une autre propriété. L’évaluateur qualité doit
conserver toutes les positions/logits nécessaires, même si le chemin utilisateur
optimise la dernière ligne.

Chaque prototype d’optimisation attend son préenregistrement détaillé dans
`opti.md`, puis parité non vide/finie et états complets avant mesure de vitesse.
Les [rapports API et LFM-01 de cette campagne](measurements/audit-api-lfm-20261016/README.md)
contiennent les premiers résultats et leurs domaines ; Le prototype K04 est retiré ; les A/B et la livraison finale restent séparés
de la qualification numérique. Les pistes rejetées, interrompues ou
non concluantes restent visibles ; aucune de ces 40 lignes n’est une promesse
de gain, ni une preuve qu’un mécanisme n’existe nulle part dans le dépôt.

Processus de release observé, **non exécuté par cette revue documentaire** :

1. Fixer source/version et inspecter les résultats de CI du SHA réellement livré.
   Workflows PR/push : `.github/workflows/rust.yml`, `check.yml` et
   `quantization.yml` selon les fichiers affectés. Compiler le vrai `mlx,chat` ;
   les GPU physiques et preuves CPU sont des campagnes séparées.
2. Versions indépendantes : moteur dans `Cargo.toml` (point de départ 1.4.5),
   Desktop dans `apps/MLXL3Studio/Resources/Info.plist` (1.4.4, build26).
   Ce sont les versions des artefacts locaux, pas de nouvelles releases publiques.
   Déterminer les numéros finaux avec le responsable de campagne, sans écraser
   de tag/asset existant.
3. `scripts/build-macos-app.sh` construit/bundle Rust+MLX+Swift et signe ad-hoc.
   `scripts/build-macos-dmg.sh` crée
   `MLXL3-Desktop-vVERSION-bBUILD-Apple-Silicon.dmg` ; aucune installation dans
   `/Applications` n’est faite par ces scripts. L’app actuelle reste conservée
   selon la demande utilisateur antérieure.
4. `scripts/package_engine.py` dérive la version Cargo (ou `--version`), contrôle
   quatre fichiers runtime, hashes et stabilité pendant packaging, crée manifeste
   et `MLXL3-Engine-vVERSION-arm64.tar.gz`. Le manifeste est produit dans le
   répertoire runtime, pas automatiquement un asset externe au bon nom : vérifier
   et copier le JSON voulu avant publication.
5. Conventions existantes : tag Desktop `vVERSION`, moteur `engine-vVERSION`.
   `UpdateManager.swift` filtre ces canaux et les noms d’assets. Publier le moteur
   avec `--latest=false`, et vérifier que `/releases/latest` reste le bon DMG.
   Version du moteur géré et sélection du runtime embarqué doivent aussi être
   vérifiées ; le simple nom du fichier ne prouve pas l’identité du binaire.
6. Changelogs distincts `docs/release-vVERSION.md` et
   `docs/release-engine-vVERSION.md` : nouveautés, performances contextualisées,
   correctifs, installation/compatibilité, vérification/limites, remerciements
   réels liés. Joindre SHA source, noms/tailles/SHA256, signatures inspectées,
   rapport de validation et état de publication réel.

Prérequis externes : droits GitHub/CI/publication à vérifier au moment opportun ;
aucun échec d’accès n’a été observé par cette revue. La notarisation requiert des
identifiants/une identité Apple non établis ici ; cela ne bloque pas la correction,
les tests ni une distribution ad-hoc explicitement décrite. Le build et la
publication ne valent jamais installation, qualification universelle des modèles
ou validation de toutes les propositions de cet audit.

## Capture supplémentaire : formules LaTeX

La capture du16octobre a révélé deux commandes non reconnues par SwiftMath,
`\dots` et `\boxed`. Elles sont corrigées avec des régressions de parsing,
copie, styles et pixels, plus le parcours de réponse ; voir [rapport LaTeX](measurements/audit-api-lfm-20261016/latex.md).
Ce correctif d’affichage ne change ni les poids ni l’évaluation du raisonnement.
