# MLXL3 Desktop 1.4.4 build 26 / moteur 1.4.5 — correctif local

Ce build local corrige l’inspecteur de génération des modèles LFM. Aucune release GitHub n’est publiée par cette tâche.

## Correctifs

- Une préférence MTP ou DFlash conservée depuis Qwen ne verrouille plus l’échantillonnage d’un modèle LFM incompatible.
- Le panneau distingue la préférence enregistrée du mode applicable et laisse sa désactivation accessible. Les contrôles de profondeur et de tête restent indisponibles sur un modèle incompatible.
- Les réglages libres de température, Top K et répétition sont conservés. Pour un modèle compatible utilisant MTP/DFlash, l’affichage et la requête restent greedy 0/1/1 ; désactiver le mode retrouve les réglages libres.
- Le bridge fournit désormais le profil de cache logique des LFM denses/MoE ; l’estimation et la recommandation de contexte peuvent apparaître.

## Performances et mémoire

Aucun gain de vitesse ni changement de quantification ou de kernels. Pour LFM2.5-24B-A2B, le profil logique est de 20 480 octets/token + 245 760 octets fixes. Il exclut le backing retenu par les vues, les temporaires, les caches allocateur et la RAM de macOS. Les poids initiaux affichés restent une taille de checkpoint ; ce n’est pas une mesure complète de RAM processus.

## Installation et compatibilité

macOS 26.2 minimum, Apple Silicon, MLX 0.32.2. Desktop 1.4.4 est associé au moteur 1.4.5 pour éviter de sélectionner l’ancien moteur géré 1.4.4. Le remplacement local doit conserver une sauvegarde de l’ancienne app, les conversations, les préférences et les modèles. Version/tailles/SHA-256 et état réel d’installation sont consignés dans le manifeste de livraison ; ne pas assimiler build et installation.

## Vérification et limites

Suite Desktop complète passée, puis 821 assertions / 180 états sur les dernières retouches UI ; vrais événements/requests via faux moteur, relance, OFF, retours Qwen/LFM, erreurs et réglages persistés. Reproduction rouge initiale et mutation de l’ancien verrouillage détectées. Contrôle Cua vivant : trois curseurs modifiables, température 0,73→0,85, OFF accessible et effectif, estimation mémoire visible.

Rust : 5 nouveaux tests, 80 tests CPU réussis / 3 ignorés historiques, format/Clippy strict / build mlx,chat / compilation des tests MLX passés. Kani : 3 harnais réels, 220 vérifications SUCCESS et 11 couvertures SATISFIED ; dimensions générales i8×i8 et totaux i64 non négatifs, plus dimensions LFM réelles avec compte i32. Ni ces domaines bornés ni les tests ne prouvent MLX/Metal, SwiftUI, la concurrence ou les allocations.

Le vrai bridge LFM V2 s’est chargé et fermé proprement sans génération : MTP/DFlash indisponibles, contexte 4096 / max 131072 et profil exact par oracle config. Aucun nouveau KL/PPL ou benchmark ; poids V2 immuables. Vérification UI isolée, pas de lecture de l’historique personnel. Les anciennes valeurs de sampling déjà écrasées par une version précédente ne peuvent pas être reconstituées.

[Preuves et revue de l’audit joint](measurements/lfm-mtp-ui-20261016/). Le constat distinct sur finish_reason API n’est pas corrigé par cette tâche.

## Remerciements

[0xZKnw](https://github.com/0xZKnw) pour le signalement et la capture du défaut ; [LiquidAI](https://huggingface.co/LiquidAI) pour LFM ; [MLX](https://github.com/ml-explore/mlx) pour le runtime Apple Silicon.
