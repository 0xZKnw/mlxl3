# MLXL3 Desktop v0.4.2

## Contexte et mémoire

- Compteur de tokens utilisé / limite dans la barre de chat, actualisé pendant
  la génération. Il inclut le prompt, les résultats MCP et la génération ;
  le brouillon non envoyé n'est pas inclus.
- Limite de contexte réglable par modèle, avec préréglages et bouton
  **Enregistrer et recharger le modèle**. Le réglage est conservé au redémarrage.
- La limite est appliquée par le moteur : les prompts trop longs sont refusés
  avant le calcul GPU, sans supprimer silencieusement l'historique.
- Estimation **modèle + contexte** en Go, recalculée immédiatement pendant
  la saisie. Elle utilise les caches réellement créés au chargement : dimensions,
  précision, fenêtres glissantes et états récurrents.

L'estimation correspond à une conversation à contexte plein, pas au pic RAM
total. Les buffers de calcul, les autres conversations en cache et macOS
nécessitent une marge supplémentaire. Pour un cache inconnu ou sans warmup,
l'app indique que l'estimation est indisponible.

## Français / English

Le choix de langue est disponible dans **Réglages → Langue de l'app**, à côté
des mises à jour. Il est conservé au redémarrage et ne traduit ni ne modifie
le contenu des conversations.

## Installation

Télécharger le DMG ci-dessous et glisser **MLXL3 Desktop** dans Applications,
ou utiliser les réglages de mise à jour de l'app. Le moteur Python/MLX est inclus ;
les modèles sont téléchargés séparément.

- Mac Apple Silicon, macOS **26.2 ou plus récent**.
- Application signée ad hoc, **non notarisée** : macOS peut demander une
  autorisation à la première ouverture.
- Exa reste préconfiguré, désactivé par défaut ; les choix MCP existants sont conservés.

Validation : **349 tests Python réussis**, 2 tests ignorés faute de modèles
locaux ; contrôles Swift de chronologie, persistance, langue, contexte et mémoire ;
vérification visuelle FR/EN et du recalcul mémoire ; génération réelle sur Qwen.
Un fichier SHA-256 accompagne le DMG.
