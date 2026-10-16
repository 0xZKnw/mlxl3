# MLXL3 Desktop 1.4.4 — build 26

**Release publique en préparation, non publiée.** Cette version embarquera le moteur **1.4.5**. Elle réunit les correctifs de l’inspecteur LFM et de l’API locale ; les variantes de performance LFM attendent encore leur qualification finale.

## Nouveautés

- L’inspecteur distingue la préférence MTP/DFlash enregistrée du mode réellement disponible. Sur LFM sans tête compatible, température, Top K, répétition et bouton OFF restent accessibles.
- L’estimation de contexte reçoit le profil de cache logique des modèles LFM denses/MoE. Pour LFM2.5-24B-A2B : **20 480 octets/token + 245 760 octets fixes**.
- Le moteur expose `/v1/capabilities` pour décrire les fonctions réellement acceptées par son API. Voir les [notes du moteur 1.4.5](release-engine-v1.4.5.md).

## Correctifs

Une préférence spéculative conservée après un changement Qwen→LFM ne verrouille plus le sampling d’un modèle incompatible. Les réglages libres sont conservés lorsqu’on active puis désactive un mode compatible ; MTP/DFlash continuent d’imposer greedy lorsqu’ils sont effectivement utilisés.

L’API distingue désormais EOS, budget de sortie et contexte plein dans `finish_reason`. Elle refuse immédiatement une génération supplémentaire lorsque le moteur est occupé, annule une requête déconnectée même avant le premier delta et borne les attentes du bridge et des clients HTTP. Les réponses invalides ne sont plus présentées comme une réussite. L’arrêt pendant le chargement nettoie également le sous-processus moteur.

Le packaging prend la version du moteur dans `Cargo.toml`, indépendamment de la version Desktop ; l’option explicite reste prioritaire.

## Performances et mémoire

Qualification numérique locale du préfill MoE : **32 cas V2** passent, longueurs **1..8/23/24/25/63/64/65/128/266**, préfixes **0 et17**, avec **128000logits F16 terminaux,50états, routes/affinités des38MoE et17continuations forcées bitidentiques**. La capture de routes est passive sur128tokens. Ce résultat porte sur le binaire figé `dac890b…`, avant le raccordement KV à capacité ; il ne valide pas automatiquement le code modifié ensuite. Le chemin reste **opt-in, désactivé par défaut** à cette étape. [Preuves et limites LFM-01](measurements/audit-api-lfm-20261016/lfm-01.md).

**À compléter avant publication :** variantes LFM retenues ou rejetées, parité numérique, matériel, modèle, protocole et résultats comparatifs. Aucun gain de vitesse ni amélioration KL/PPL n’est établi par les contrôles CPU. Un prototype ne constitue pas une accélération livrée.

Le profil mémoire décrit les états persistants logiques. Il exclut poids, grands backings retenus par des vues, temporaires, cache allocateur et RAM processus. La taille initiale du checkpoint affichée n’est pas une mesure de toute la mémoire résidente.

## Installation et compatibilité

Artefact prévu : **MLXL3-Desktop-v1.4.4-b26-Apple-Silicon.dmg**. Apple Silicon, macOS **26.2+**, moteur **1.4.5** et MLX **0.32.2** embarqués ; les poids se téléchargent séparément. Distribution prévue **signée ad-hoc, sans notarisation Apple** ; la signature du build final reste à vérifier.

Le canal moteur indépendant doit conserver `--latest=false` pour que l’updater Desktop propose le DMG. Cette préparation n’a pas remplacé l’application installée ; aucun remplacement dans `/Applications` n’est prévu au titre de la décision utilisateur de conserver l’app actuelle.

## Vérification et limites

**Campagne UI/mémoire antérieure au lot API :** suite Desktop exécutée, puis **821 assertions / 180 états** après les dernières retouches ; ancien comportement et mutation détectés. Le contrôle visuel vivant a vérifié les curseurs, OFF et l’estimation mémoire. Un chargement réel LFM V2 a confirmé capacités et profil, sans génération. Rust : **80 tests CPU réussis / 3 ignorés**, **220 obligations Kani / 11 couvertures** sur les domaines arithmétiques documentés. [Preuves UI/mémoire](measurements/lfm-mtp-ui-20261016/README.md).

**Nouveau lot API : 33 tests CPU réussis, aucun ignoré**, format et Clippy strict passés. Les fixtures exercent HTTP, faux bridge, déconnexion, saturation, erreurs, SIGTERM pendant le chargement et vrai TCP non lecteur. Trois mutations retirant des protections de fermeture/délai sont détectées, puis restaurées. Kani valide **65 obligations / 3 couvertures** sur les compteurs et budgets u64. Le harnais du parseur Origin a atteint **30 secondes sans conclusion** : il reste non prouvé. [Rapport API](measurements/audit-api-lfm-20261016/api.md).

Ces preuves restent séparées : elles ne prouvent ni SwiftUI, ni les échanges concurrents, ni MLX/Metal, ni la fidélité de toute génération. Les valeurs de sampling déjà écrasées par une ancienne version ne peuvent pas être reconstituées. Le packaging antérieur dispose de **45 tests réussis** ; deux contrats CrossHair restent non confirmés, sans preuve I/O.

**À compléter avant publication :** suites intégrées finales, build livré `mlx,chat`, qualification physique des sources finales LFM et mesures retenues, CI du SHA exact, signature et cohérence des artefacts.

## Identité de la release préparée

Tag prévu : `v1.4.4`. **SHA source et rapport final à compléter après le dernier build.** Les anciennes empreintes du paquet local ne décrivent pas encore ce nouveau lot.

| Artefact prévu | Taille en octets | SHA-256 |
|---|---:|---|
| `MLXL3-Desktop-v1.4.4-b26-Apple-Silicon.dmg` | À compléter | À compléter |
| Rapport/manifeste de validation final | À compléter | À compléter |

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw) pour les signalements, l’intégration Desktop/moteur et les validations LFM/API.
- [@HENK0O](https://github.com/HENK0O) pour les bases de gestion mémoire et de recommandation de contexte de [PR31](https://github.com/0xZKnw/mlxl3/pull/31) et [PR32](https://github.com/0xZKnw/mlxl3/pull/32).
- [LiquidAI](https://huggingface.co/LiquidAI) pour LFM et [MLX](https://github.com/ml-explore/mlx) pour le runtime Apple Silicon ; les autres projets du moteur sont crédités dans [son changelog](release-engine-v1.4.5.md).
