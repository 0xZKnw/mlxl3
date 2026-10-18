# MLXL3 Desktop 1.5.0 — build 30

## Nouveautés

- **Conversations :** modifier un message, régénérer une réponse ou poursuivre depuis un point de l'historique crée une branche indépendante et conserve l'original.
- **Projets et profils :** conversations, instructions et documents regroupés par projet ; profils enregistrés pour le modèle, le contexte, l'instruction, le sampling et le schéma JSON.
- **Documents locaux :** import de fichiers texte/PDF et de dossiers, recherche par mots sans service distant, extraits et citations conservés avec les réponses. Les sources s'ouvrent dans l'app ; le PDF original reste accessible à la page citée lorsqu'il est disponible. Limites : 128 documents et 16 Mio de texte extrait par espace de travail. **Aucun OCR.**
- **MCP :** gestion graphique des serveurs locaux et HTTPS, test explicite des connexions et autorisations exactes par projet. Un nouveau projet n'autorise aucun outil. Les conversations sans projet conservent le réglage MCP global.
- **JSON validé :** schéma optionnel dans l'inspecteur et prise en charge de `response_format` par l'API compatible OpenAI. La réponse est retenue jusqu'à validation ; au plus trois tentatives partagent le budget de tokens explicite. Les schémas non pris en charge et les réponses non conformes échouent explicitement. Ce mode désactive MCP pour la requête.
- **Voix et export :** dictée avec reconnaissance sur le Mac uniquement, lecture des réponses par la voix système et export Markdown des conversations et de leurs sources. La dictée dépend de la disponibilité locale de la langue et des autorisations macOS.
- **Catalogue :** indication de la famille du modèle et estimation prudente du budget mémoire avant téléchargement. Le cache théorique n'est estimé que pour les configurations reconnues ; ce conseil ne garantit pas le chargement d'un checkpoint.

## Correctifs

Les nouveaux panneaux ont un fond sombre opaque et des onglets lisibles. Leur bouton Fermer reste disponible pendant une génération. Suppression, annulation et restauration d'une conversation suivent correctement son projet. La récupération de sauvegarde restaure aussi les documents et profils. Un import dépassant les limites est rejeté avant de modifier l'espace de travail.

Les configurations MCP sont remplacées atomiquement à partir de fichiers privés. La validation accepte les URL HTTP IPv6 de loopback et rejette les identifiants dans les URL. Les correctifs de streaming, LaTeX, mise à jour, sampling, DFlash et gestion mémoire de la branche 1.4.8 sont conservés.

## Performances

Aucune nouvelle accélération, économie de RAM ou amélioration de qualité n'est revendiquée pour cette version. La recherche documentaire est une recherche lexicale locale. Le JSON utilise validation et tentatives bornées, sans décodage contraint token par token.

## Installation et compatibilité

**MLXL3-Desktop-v1.5.0-b30-Apple-Silicon.dmg**, Apple Silicon arm64, macOS **26.2+**, MLX **0.32.2**. Signature ad-hoc, sans notarisation Apple. Le paquet embarque le moteur **1.5.0**, nécessaire au JSON validé et aux permissions par projet. Les canaux Desktop et moteur restent indépendants.

L'historique version 1 est lisible et migre vers le format 2. Les anciens Desktop ne savent pas relire ce nouveau format : conserver une copie séparée de l'ancien historique pour un éventuel retour arrière. La sauvegarde tournante peut déjà être en version 2 après les sauvegardes suivantes. Les modèles ne sont pas déplacés. Cette livraison ne remplace pas automatiquement l'application installée par les tests.

## Vérification et limites

Les résultats de cette intégration et du paquet final sont consignés dans [le rapport de vérification](desktop-v1.5.0-verification.md). Les premiers contrôles de fonctionnalités sur la base 1.3.0 sont conservés séparément dans [le rapport initial](desktop-project-features-2026-10-18.md) ; ils ne valident pas à eux seuls le paquet 1.5.0.

Le schéma JSON implémente un sous-ensemble explicite : types, objets/tableaux, propriétés requises, bornes, alternatives, `enum`, `const` et unicité. Les références, expressions régulières et formats non implémentés sont refusés. La recherche et les citations restent à vérifier dans les réponses du modèle. Microphone, qualité de dictée/lecture et services MCP distants ne sont pas qualifiés par les tests automatisés. Les vérifications Kani sont bornées et ne prouvent ni SwiftUI, ni MLX/Metal, ni toute la génération ; les échecs ou interruptions restent indiqués dans le rapport.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), demandes produit, intégration et validation sur Mac.
- [MLX](https://github.com/ml-explore/mlx), calcul local sur Apple Silicon ; [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique conservé.
- [Qwen](https://github.com/QwenLM), modèles des contrôles locaux ; [Model Context Protocol](https://github.com/modelcontextprotocol), protocole des outils.
- [Kani](https://github.com/model-checking/kani), vérification bornée des contrats CPU. Les licences et autres attributions du moteur sont conservées dans le dépôt.
