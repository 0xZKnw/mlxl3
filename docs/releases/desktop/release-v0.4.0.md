# MLXL3 Desktop v0.4.0

## Nouvelle interface

- Interface native repensée : palette noir et ivoire, navigation compacte et espace de lecture épuré.
- Messages utilisateur à droite, réponses du modèle à gauche.
- Historique recherchable et repliable, retour au dernier message et contrôles adaptés aux petites fenêtres.
- Réglages et gestionnaire de modèles harmonisés ; Markdown, code coloré, copie et LaTeX conservés.

## Exa et interrupteur MCP

- Exa est préconfiguré dans le DMG : recherche web et lecture de pages, sans Node ni clé API pour l'offre gratuite d'Exa (limites applicables).
- Interrupteur MCP directement dans la barre de saisie, **désactivé au premier lancement**.
- Le choix est conservé entre les lancements et les conversations, jusqu'au prochain changement.
- Activation/désactivation sans recharger le modèle ; aucune connexion MCP tant que le mode reste désactivé.
- L'interrupteur est temporairement verrouillé pendant une génération ou une mise à jour de connexion.
- Connexion HTTPS native aux serveurs MCP Streamable HTTP (JSON/SSE), en plus des serveurs locaux stdio.
- Les recherches et URL envoyées à Exa quittent le Mac. L'inférence reste locale.

## Installation

Mac Apple Silicon, macOS 26.2 ou ultérieur. Ouvrir le DMG et déplacer MLXL3 Desktop dans Applications, ou utiliser les mises à jour intégrées. Le moteur et l'interface sont inclus ; les poids des modèles se téléchargent séparément.

L'application est signée ad hoc, non notariée. Les instructions Gatekeeper habituelles du README restent applicables.

## Vérifications

Tests ciblés du protocole MCP, du pont de génération et du registre ; autocontrôle natif de la préférence persistante ; essais visuels isolés et connexion Exa réelle. Pas de nouveau benchmark de débit ni de modification des kernels d'inférence dans cette version.
