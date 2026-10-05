# MLXL3 Desktop v1.2.2

Le compteur de contexte inclut maintenant les tokens de la réponse une fois la génération terminée. Il utilise le total exact fourni par le moteur, y compris après un appel MCP, et le conserve avec la conversation. Les anciennes conversations disposant de statistiques corrigent aussi leur compteur à la réouverture.

**Exa est ajouté automatiquement à la configuration MCP** dès le lancement de l’app, sur une installation neuve comme après une mise à jour. Les autres serveurs et un Exa déjà personnalisé ou désactivé sont conservés. La préférence d’activation globale MCP reste mémorisée. Configuration hébergée officielle : [Exa MCP](https://exa.ai/docs/get-started/exa-mcp).

Cette mise à jour Desktop1.2.2/build21 garde le moteur1.2.1. Le packaging et la validation de l’updater respectent désormais leurs versions indépendantes. Apple Silicon, macOS26.2 ou supérieur, signature ad hoc ; aucun poids inclus.

[Validation](desktop-v1.2.2-validation.md). Le défaut d’activation à chaud d’un moteur mis à jour dans une app déjà ouverte reste prévu pour plus tard.
