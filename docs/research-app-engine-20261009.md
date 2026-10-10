# MLXL3 — pistes interface et moteur, 9 octobre 2026

Audit documentaire demandé par l'utilisateur, qui a confirmé parler de l'interface Desktop. Base locale : `dee440230bd1e3662eb0dc8c1a619ab8406b5a69`, branche `codex/ram-qwen27-optis`. Comparaison du code, du journal et de sources primaires publiques. Aucun prototype, installation, test ou benchmark lancé pour cet audit. Aucun nouveau gain mesuré ; les propositions ci-dessous ne sont pas des fonctionnalités livrées ni une garantie de compatibilité.

## Ce qui existe déjà

Le moteur possède des kernels EXL3/Metal, des chemins TensorOps M5, un cache de préfixe Qwen, des états GDN réparés après vérification, MTP/Tune D0..3 et une API HTTP locale. Le Desktop possède recherche dans les conversations, import/export, PDF/texte, réglage de contexte par modèle avec estimation mémoire, compteurs TTFT/cache/RAM, téléchargement reprenable, MCP et déchargement automatique. Les listes de messages/conversations utilisent déjà LazyVStack et le rendu incrémental est testé. L'option mémoire et les deux chemins small-M de la PR #31 restent OFF par défaut ; leur vitesse n'est pas établie.

Ajouter « un cache », « une API », « une jauge RAM » ou « du lazy rendering » n'est donc pas une découverte. Les évolutions pertinentes portent sur leur politique, leur exposition et les cas encore absents.

## Interface : ajouts prioritaires

| Idée | Apport concret | Effort et limite |
|---|---|---|
| **Profils enregistrés par modèle** | Profils Code, Discussion, Économie, avec prompt système et paramètres cohérents ; conserver le choix explicite de l'utilisateur et expliquer quand MTP est inéligible. | Moyen. Le contexte est déjà enregistré par modèle, mais les paramètres du workspace sont globaux. Les profils ne créent pas une accélération intrinsèque. Inspiration : [presets LM Studio](https://lmstudio.ai/docs/app/presets). |
| **Modifier, régénérer et créer une branche** | Corriger un ancien prompt, comparer une autre réponse, conserver les alternatives. Dossiers et favoris complètent la recherche existante. | Moyen. Migration du stockage, identification du préfixe actif et invalidation exacte du cache nécessaires. [Branches Open WebUI](https://docs.openwebui.com/features/chat-conversations/data-controls/import-export/), [organisation LM Studio](https://lmstudio.ai/docs/app/basics/chat). |
| **Documents longs avec recherche et citations** | Indexer les documents, sélectionner les passages pertinents et afficher leurs pages/fichiers, plutôt qu'injecter tout le texte. Commencer par une recherche lexicale locale ; embeddings ensuite si utiles. | Grand. L'import actuel accepte PDF/texte, rejette les PDF scannés sans texte et borne la taille extraite ; il ne réalise pas ce parcours RAG. Mesurer pertinence et erreurs de citation. Moins de contexte peut réduire le prefill, sans gain chiffré établi. [RAG LM Studio](https://lmstudio.ai/docs/app/basics/rag). |
| **Diagnostic de performance exportable** | Réutiliser les métriques existantes, ajouter conditions d'alimentation/mode économie/état thermique et identifier exactement modèle, moteur et options. Séparer chargement, prefill, premier affichage et decode. Un comparatif borné doit signaler la dérive. | Moyen. Diagnostic volontaire, pas un benchmark automatique au lancement. Apple fournit [thermalState](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.property) et le [mode économie d'énergie](https://developer.apple.com/documentation/foundation/processinfo/powerstatedidchangemessage) ; cela ne donne pas une température ni une fréquence GPU. |
| **Budget mémoire et contexte conseillé** | Compléter l'estimation statique déjà présente par poids + tête draft + cache retenu + buffers observés, et proposer une limite de contexte adaptée. Afficher pourquoi un modèle risque de manquer de marge avant téléchargement/chargement. | Moyen. L'estimation actuelle exclut buffers, autres caches et macOS ; une taille de fichier n'est pas une promesse de RAM. Ne pas confondre allocation MLX et footprint processus. |
| **Panneau serveur local** | Démarrer/arrêter le serveur déjà existant, afficher son URL, sa clé d'accès, son modèle et son état ; simplifier l'usage depuis un éditeur ou un client local. | Moyen. Éviter de charger un deuxième exemplaire du modèle pour GUI et API. [Gestion du serveur LM Studio](https://lmstudio.ai/docs/cli). |
| **Sorties JSON contraintes** | Mode JSON/schema utile aux scripts et outils ; validation visible et erreur claire si le schéma est impossible. | Grand, moteur et interface. L'API actuelle rejette response_format et les contraintes. Un prompt demandant du JSON n'est pas un décodage contraint. [LM Studio](https://lmstudio.ai/docs/developer/openai-compat/structured-output). |
| **Résumé optionnel des anciens tours** | Proposer une compaction quand le contexte sature, conserver l'historique original et afficher le résumé réellement envoyé. | Moyen/grand. Modifie le prompt et peut perdre des informations : impossible à présenter comme optimisation exacte. Ne pas couper un appel outil de son résultat. [Compaction TensorFold](https://github.com/ashhart/TensorFold#context-compaction). |
| **OCR et dictée locale** | Lire les PDF scannés ; dicter et éventuellement écouter les réponses. | Moyen/grand, priorité secondaire. OCR ≠ compréhension d'images ; la vision et l'audio du modèle demandent poids, format et moteur supplémentaires. Préférer un traitement à la demande pour limiter la mémoire résidente. |

## Moteur : pistes restantes

### 1. Profondeur MTP adaptative — priorité élevée, piste connue non prototypée

Tune choisit une profondeur pour un modèle/runtime, pas pour chaque phase d'une réponse. Un contrôleur pourrait choisir D0..3 selon coût mesuré et nombre de tokens réellement acceptés, avec hystérésis et retour conservateur. Optimiser le temps par token engagé, pas seulement le taux d'acceptation.

Déjà identifié dans RESEARCH-2026-10-07-RAPIDMLX-SUITE et la revue TensorFold ; ne pas le présenter comme nouvel essai réussi. Le retour D0→draft doit réparer la tête MTP, contrairement au fallback terminal actuel. Exiger budget, annulation, états/logits et reprise exacts avant mesure. La [recherche sur l'adaptation](https://arxiv.org/abs/2405.19715) ne valide pas notre ordonnanceur ; d'autres méthodes changent même le critère d'acceptation et ne respectent pas le contrat exact.

### 2. Budget de cache plus précis — priorité élevée

PromptCache::capture borne à 256 Mio la taille logique de l'état cible, avec un seul checkpoint. Cette borne ne couvre pas automatiquement stockage partagé, tête draft, logits et footprint physique. Étudier une politique explicite de rétention, une mesure des allocations détenues et un éventuel checkpoint compact. Ne pas remplacer une mesure d'allocation par un delta de footprint bruité.

Le budget de résidence wired déjà essayé sur Qwen27 a été rejeté ; ce n'est pas la même politique et il ne doit pas être réintroduit silencieusement. [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py) fournit des caches avec limites d'octets ; leur présence ne prouve pas que tous leurs chemins sont exempts de pics mémoire.

### 3. Cache KV quantifié, d'abord 8 bits — priorité moyenne, contextes longs

Réduire K/V dans les couches d'attention complète peut diminuer la mémoire croissante avec le contexte. Garder les états récurrents GDN et le contrat de rollback séparés. Intégrer un seuil de longueur et une comparaison précision/performance, pas une activation systématique dès le premier token.

[Ollama](https://docs.ollama.com/faq#how-can-i-set-the-quantization-type-for-the-kv-cache) expose ce compromis. La [documentation MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md#kv-cache-quantization) précise que son attention quantifiée n'est pas fusionnée et peut créer une grosse matrice temporaire. Les ratios de taille du KV ne sont pas des ratios de RAM totale et ce n'est pas nécessairement plus rapide. Passage numérique non bitexact : évaluer qualité et contextes longs en plus des états/caches.

### 4. Reprofilage des vrais kernels — priorité élevée pour décider, gain inconnu

La PR #31 modifie la construction des petits batches MLP ; les anciens profils ne suffisent pas à déterminer le nouveau coût dominant. Exploiter les compteurs [Apple](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth) : bande passante, occupancy, temps/dispatch, stalls. Le journal signale une capture non analysée faute de xctrace.

Seules les projections réellement coûteuses justifient de nouveaux essais de tuiles, partage de lectures ou fusion exacte. Les GEMV a/b multi-lignes du filtre SMALLM-B étaient exacts mais leur économie proxy très petite face au bloc complet ; pas de répétition sans profil montrant une nouvelle raison. Même règle pour les fusions GDN déjà non concluantes et la compilation des graphs rejetée.

### 5. Cache de préfixe entre conversations ou sur disque — priorité moyenne/basse

Le cache actuel accélère le tour suivant, mais n'est pas un système général multi-conversations persistant. Un LRU avec budget total explicite ou un checkpoint disque pourrait aider quand on revient à un projet ou après redémarrage. Commencer avec une option désactivée et éviter de multiplier les copies.

[MLX-LM sait sauvegarder un cache](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/cache_prompt.py). Pour Qwen hybride, sauvegarder les vrais états récurrents, tokens rendus, tête draft et identité complète du runtime ; un fichier KV seul est insuffisant. Compter temps et pic RAM d'écriture/lecture. Ce n'est pas prioritaire pour une seule conversation active.

### 6. MTP avec sampling — évolution importante, effort élevé

MTP Desktop est limité aux paramètres compatibles avec son parcours greedy. Étudier une vérification conservant le sampler cible et sa gestion du hasard, avec reprise exacte des états ; mesurer l'acceptation avant d'espérer un gain. [TensorFold python-0.6](https://github.com/ashhart/TensorFold/tree/python-0.6) décrit une exactitude contre son propre moteur/settings, pas contre MLXL3. Un port changeant le RNG ou la distribution ne satisfait pas automatiquement le contrat existant.

### 7. Mémoire des embeddings — exploration secondaire, difficulté élevée

Le checkpoint local contient une table BF16 [248320,5120] de 2 542 796 800 octets, convertie au chargement en F16 par half_weight ; lm_head est distinct. C'est une piste concrète pour un artefact d'embedding quantifié optionnel ou un chargement de lignes à la demande. **2,54 Go est la taille de la table, pas un gain promis.** La quantification change les entrées du modèle ; elle exige une validation de qualité indépendante. Le chargement à la demande doit reproduire les conversions et éviter que le budget réel ne se déplace dans le cache fichier.

ExLlamaV3 [sépare parfois les embeddings vers la RAM système](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md), mais CPU et GPU partagent la RAM sur Mac : déplacer la table au CPU ne réduit pas sa taille totale. MTP utilise des IDs restant sur GPU pour éviter une synchronisation par profondeur ; une lecture CPU pourrait annuler les gains. Pas de port direct ni mesure nouvelle.

### 8. Stockage des conversations et énergie de l'interface — selon usage

ConversationStore charge un snapshot JSON global et réécrit l'ensemble, avec backup. La sauvegarde est déjà temporisée, 10 s pendant génération/1 s sinon. Un stockage par conversation avec index et chargement à la demande peut réduire les copies et le travail sur les très gros historiques. Vérifier migration, crash/recovery et recherche globale ; benchmark UI avec des données jetables représentatives avant d'annoncer un gain. Les LazyVStack et le rendu incrémental existants restent une baseline, pas une nouveauté.

## Ce que je ne prioriserais pas

- PagedAttention/continuous batching : intéressants pour un serveur multi-utilisateur, peu justifiés pour un chat résident solo ; complexité supplémentaire avec les états GDN.
- Offload disque des poids denses : le [prototype discuté dans MLX](https://github.com/ml-explore/mlx/discussions/615) illustre le coût des défauts de pages ; ce n'est pas une accélération gratuite.
- Copier les kernels CUDA/AVX/NVFP4 ou les débits « ×4 » : EXL3/F16, formats, hardware et frontières de mesure différents. Le [TensorFold natif actuel](https://github.com/ashhart/TensorFold) rapporte aussi un 27B plus lent dans sa comparaison servie que sa ligne Python antérieure.
- BM16 small-M et changements d'ordre arithmétique : divergence déjà documentée. MB3/batch MLP, pipeline et lookup existent déjà ou restent expérimentaux ; leurs nouveaux flags ne doivent pas devenir des boosts affirmés.

## Ordre proposé

1. Interface : profils par modèle et diagnostic exportable ; branches de conversation ensuite.
2. Moteur : audit du budget cache puis contrôleur MTP adaptatif, avec des essais distincts préenregistrés.
3. Documents longs : recherche locale avec citations, puis KV 8 bits si les longues conversations dominent l'usage.
4. JSON contraint, sampling MTP et embeddings : travaux séparés, de portée plus grande.

Avant tout essai moteur : stabilité/alimentation observées, oracle indépendant non vide et fini, garde/Kani sur le vrai Rust, parcours bridge/CLI réel, puis mesures répétées alternées. Aucune somme de gains historiques incompatibles. Les difficultés numériques de MLX/Metal doivent rester visibles.

Sources GitHub et empreintes locales : [manifest](measurements/research-app-engine-20261009/sources.json). Les pages web dynamiques/indexées ne constituent pas un audit complet de chaque dernière révision. L'ancien URL turboderp/exllamav3 et une page Apple/Open WebUI ont échoué ; les conclusions utilisent leurs adresses primaires accessibles corrigées, pas ces erreurs.
