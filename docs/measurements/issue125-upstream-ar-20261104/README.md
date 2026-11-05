# Flash Next AR : recherche après publication de la PR129

Recherche de sources du4novembre2026, **sans changement exécutable, build, preuve, inférence ou benchmark**. La [PR129](https://github.com/0xZKnw/mlxl3/pull/129) a été publiée avant cette recherche, head `63787f1cd137248fe6d1bb482e354ed458395ba9`. L1 reste actif sur Apple M5 exact ; L2/L3/L4/L5 restent opt-in/OFF. L’utilisateur a explicitement exclu de nouveaux tests L3/L5. Aucun merge, remplacement d’app ou release effectué.

La priorité proposée est **de préparer le consommateur GPU avant d’attendre le MoE final**. Le petit changement le plus simple à isoler ensuite serait de supprimer une conversion GDN devenue évitable pour les streamsF32. Ce sont des hypothèses issues du code et du profil corrigé, sans gain annoncé.

## Classement des écarts concrets

| Priorité | Piste | Écart avec le moteur publié | Limite principale |
|---|---|---|---|
|1|Décaler l’attente CPU du MoE final jusqu’à son consommateur|48appels `experts.completed()`/row précèdent encore l’encodage HC suivant ; L1 ne supprime pas cette attente|Conserver chaque erreur producteur, owner/cache et rollback ; ne pas déplacer la frontière routeur→SSD|
|2|Consommer directement la sortie GDN half dans HC sur streamsF32|La projection produit déjà half ; on matérialise un bufferF32 avant un HC qui sait déjà convertir half en lecture|Chemin interne seulement ; préserver promotion historique pour streamsF16, API publique et trace|
|3|Fusionner norm+sigmoid gate du GDN natif|Deux kernels et un intermédiaireF32 restent séparés, alors qu’oMLX possède une fusion AR spécialisée|Réduction, sous-normales, casts et exp précis ; l’ancien T17 corrigé existe déjà|
|4|File commune de lectures SSD dans le service Rust natif|Quatre files round-robin ; le travail restant d’un worker ne peut pas être repris par un autre|Ancien T18a déjà essayé sur l’autre backend, résultats mixtes ; déséquilibre actuel non attribué|
|5|Residency set Metal limité aux poids fixes|TensorFold maintient un ensemble de ressources sur sa queue ; nous n’utilisons pas cette API|Pas de défaut de résidence attribué ; intérêt possiblement surtout après idle, budget mémoire à préserver|

**1 — Piste de scheduling la mieux reliée au profil.** llama.cpp soumet ses segments Metal sans attendre chaque retour ; c’est un exemple concret d’une frontière asynchrone explicite. Chez nous, une adaptation bornée aux34consommateurs L1 pourrait soumettre le consommateur avant de joindre les producteurs. Garder des CB distincts pour commencer distingue cette hypothèse d’une fusion de CB qui pourrait retarder le lancement de la dernière cohorte froide. Les mathématiques, l’ordre des routes et le lancement des lectures demandées restent identiques. [Soumission llama.cpp](https://github.com/ggml-org/llama.cpp/blob/6f1e6543704bc5b080c7a98e7b6f707b31f6ed51/ggml/src/ggml-metal/ggml-metal-context.m#L474-L608), [audit local et contraintes](scheduling.md).

**2 — Petite suppression de travail, sans nouveau shader.** Le couple `gdn.rs:290–299` / `hc.rs:573–614` laisse envisager36dispatchs de cast et36buffersF32 de2560valeurs évités par row (368640octets demandés). Ce décompte n’est ni une mesure de trafic ni un boost. Le contrat F16/public doit rester intact ; le profil composite ne donne pas le coût propre de ces casts. [Audit des producteurs/consommateurs](mlxlm-llamacpp.md).

**3 — Fusion différente de l’ancien T17.** L’étape minimale ajoute le producteur de normalisation native au consommateur gate déjà qualifié. Elle peut supprimer36intermédiaires6144×F32 et36dispatchs par row, pas les CB déjà regroupés par L1/P13. Le chemin oMLX utilise une précision intermédiaire différente : copier son kernel ne préserverait pas automatiquement notre oracle. Le whole-GDN oMLX est une perspective plus large, après qualification de cette frontière étroite. [Kernel AR oMLX](https://github.com/jundot/omlx/blob/4213c3284abc7b3746a91464a7af7ccc3732b759/omlx/patches/qwen35_gdn_prework.py#L355-L395), [historique numérique et adaptation](strata-omlx.md).

**4 — Port manquant, idée historique.** Strata distribue les travaux entre workers depuis une file partagée. Notre nouveau `ReadService` Rust a conservé la distribution statique, alors que `DiskWorkQueue` sur l’ancien backend contient déjà T18a. Les observations actuelles n’identifient pas les workers : aucune durée de queue critique récupérable n’est établie. [Workers Strata](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/metal/src/engine/stream.cpp#L309-L347), [différence native et résultats historiques](strata-omlx.md).

**5 — Mécanisme absent, priorité secondaire.** TensorFold maintient réellement un residency set sur son chemin Flash Next. Notre queue et notre allocateur n’en ont pas. Toutefois, le bénéfice CPU décrit par Apple vise des déclarations de ressources explicites que notre chemin n’émet déjà pas ; aucun gain de decode chauffé n’en découle. Ses kernels experts groupés sont une référence plus ambitieuse, dans un autre format de quantification : un port EXL3 ne serait pas un petit patch. [Chemin natif TensorFold](https://github.com/ashhart/TensorFold/blob/a26fdae91e0b120fc1e2f3cb8fa3e4095b779f1c/zig/src/native/flashnext_residency.zig#L6-L26), [audit et limites](tensorfold.md).

## Profil utilisé et exclusions

Réanalyse du profil native8 déjà publié, pas une nouvelle capture :7rows,238triplesL1+14doubles,336attentesMoE. Les attentes CPU inclusives centrales couvrent67.245ms/row et MoE27.441ms/row. Les intervalles entre le dernier CB MoE et le prochain CB central enregistrés sur la même horlogeGPU ont médiane0.328ms/p950.740ms,336cas couvrant les48frontières/row ; le sous-ensemble L1 n’est pas attribué séparément. **Ces nombres ne sont ni du temps récupérable, ni de l’inactivité globale GPU ; ils ne s’additionnent pas.** [Résumé avec SHA du profil](profile-summary.json), [méthode/portée](scheduling.md).

Déjà présents : P01 lecture SSD directe vers Metal Shared, records experts contigus, NT2/static-K, scratch routé interne, GDN packed, réduction MoE canonique, slabs QSA compatibles snapshots, L1/P13. MLX-LM confirme plusieurs de ces mécanismes plutôt qu’une optimisation oubliée. Son lookahead de token dépend du graphe d’échantillonnageGPU et ne se transpose pas directement à notre routeur/SSD CPU ; aucun retour de P04. [Code AR MLX-LM](https://github.com/ml-explore/mlx-lm/blob/736ab0d8c1d0edc79c3d798044fe6904e2fa23be/mlx_lm/generate.py#L414-L492), [comparaison détaillée](mlxlm-llamacpp.md).

Les gains annoncés par les cinq projets ne sont pas des prédictions sur notre M5/EXL3/3GiB. MTP/DFlash, expert-encoder annulé, SSD5/3 non compilé et nouveaux tests L3/L5 sont exclus. Avant tout futur essai choisi, enregistrer son hypothèse/protocole séparément, qualifier ses oracles/erreurs réels, puis seulement un écran court32tokens au-dessus de L1 accepté. Aucun essai n’a été commencé ici.

## Sources et état de vérification

- [Strata/oMLX : identités, HEADs, kernels, files et historique](strata-omlx.md).
- [MLX-LM/llama.cpp : HEADs, scheduling, conversions, caches et mécanismes existants](mlxlm-llamacpp.md).
- [TensorFold : moteur natif courant et différences de format/lifetime](tensorfold.md).
- [Publication et qualifications antérieures](../issue125-current-engine-20261104/README.md). CI du head63787 : Flash profile réussi ; autres jobs/workflows encore actifs/queued lors du contrôle. Aucune CI globale verte revendiquée. Le Kani d’ascendance a une ICE locale conservée ; cette recherche documentaire n’ajoute aucun code à prouver.

État : recherche de sources uniquement. Les pistes n’ont pas de validation numérique ni de vitesse nouvelle ; elles ne modifient aucun défaut du moteur.
