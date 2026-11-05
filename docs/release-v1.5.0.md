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

Contrôles locaux : **167 tests Rust CPU et 192 MLX**, **840 tests Python**, **219 contrôles** des nouvelles fonctions et les parcours Desktop concernés réussis. Restent ignorés : 3 tests CPU, 85 MLX et 115 Python. Le vrai updater accepte le DMG ; signature, intégrité, moteur relocalisé et repli sont vérifiés. Génération JSON réelle Qwen3.6 EXL3 conforme à `jsonschema` avec budget 1024 (372 tokens), rejet du budget 1 et annulation/reprise vérifiés. Le budget 256 n'a pas suffi et l'échec explicite est conservé.

Kani : nouveau contrat de budget (57 obligations/3 covers) réussi ; suite bibliothèque (70 harnais réussis et **23 timeouts non vérifiés**) ; contrat d'usage HTTP (65 obligations/3 covers par cible), deux cibles réussies. Ces vérifications bornées ne prouvent pas toute l'application. CI du SHA exact : [conversion EXL3](https://github.com/0xZKnw/mlxl3/actions/runs/39151944365), [jobs Rust Linux/macOS](https://github.com/0xZKnw/mlxl3/actions/runs/39151944315) et [suite Desktop complète](https://github.com/0xZKnw/mlxl3/actions/runs/39151944336) réussis. Le job Kani de la bibliothèque est encore en cours au relevé de publication ; aucun succès global de ce workflow n'est revendiqué.

Source du paquet : `eaf296468c7ea4d92edab9e656bf749ac71f3a36`, checkout propre au build. Le moteur 1.5.0 est embarqué ; aucune release moteur indépendante n'est publiée.

Le schéma JSON implémente un sous-ensemble explicite : types, objets/tableaux, propriétés requises, bornes, alternatives, `enum`, `const` et unicité. Les références, expressions régulières et formats non implémentés sont refusés. La recherche et les citations restent à vérifier dans les réponses du modèle. Microphone, qualité de dictée/lecture et services MCP distants ne sont pas qualifiés par les tests automatisés. Les vérifications Kani sont bornées et ne prouvent ni SwiftUI, ni MLX/Metal, ni toute la génération ; les échecs ou interruptions restent indiqués dans le rapport.

## Fichiers publiés

| Fichier | Octets | SHA-256 |
| --- | ---: | --- |
| `MLXL3-Desktop-v1.5.0-b30-Apple-Silicon.dmg` | 73805588 | `590003037dbd8daa7ee6bc32f7cb530846cc72c33768ebd458e0cb628113208d` |
| `MLXL3-Desktop-v1.5.0-b30-Apple-Silicon.dmg.sha256` | 109 | `8f7e094b96ad9faf427a5a1ec5d632eb1ee8b0d2e677437abe6115ae31efc8d0` |
| `MLXL3-Desktop-v1.5.0-b30-validation.json` | 6523 | `dc9e3c5b70d5f94998b8e0f79b216e19a18f0431db636a022adb4f8419e8467a` |

Les trois téléchargements publics ont été contrôlés. L'updater réel sélectionne Desktop1.5.0/build30 ; le canal moteur indépendant reste à1.4.8. L'app installée n'a pas été remplacée.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), demandes produit, intégration et validation sur Mac.
- [MLX](https://github.com/ml-explore/mlx), calcul local sur Apple Silicon ; [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique conservé.
- [MTPLX](https://github.com/youssofal/mtplx), base des fonctions MTP conservées avec leurs attributions d'origine.
- [Qwen](https://github.com/QwenLM), modèles des contrôles locaux ; [Model Context Protocol](https://github.com/modelcontextprotocol), protocole des outils.
- [Kani](https://github.com/model-checking/kani), vérification bornée des contrats CPU. Les licences et autres attributions du moteur sont conservées dans le dépôt.
