# Desktop1.2.2 — validation du 5 octobre 2026

## Contexte

Le moteur1.2.1 émet context_usage avant génération et fournit ensuite stats.context_used final, incluant les tokens générés et le dernier round MCP. StudioModel n’utilisait que le premier événement. Régression par le vrai CLI/moteur factice : context_usage12, prompt_tokens12, generated_tokens30, context_used42 ; le code original échoue avec « Context only shows input tokens after completion » (`context-before.log`).

Appliquer les statistiques finales à la conversation de la requête active avant sauvegarde, lire ces stats pour les historiques dont contextUsage est ancien. Préférer le total exact ; fallback prompt+generated uniquement pour anciennes stats d’un seul round, sans overflow, valeurs négatives ou limites invalides. Fraction/total bornés à la fenêtre ; les générations/caches du moteur restent inchangés. Pendant la génération, le compteur conserve le dernier relevé exact reçu ; aucune estimation des tokens à partir du texte affiché. L’annulation sans stats finales conserve ce relevé, pas de promesse de compteur token par token.

Tests : total42, valeur persistée, ancien historique12 corrigé en42, réouverture et modèle différent, final MCP77 préféré aux compteurs agrégés, multi-round sans total refusé, sept bornes dont négatif/0/Int.max, overflow et compteurs négatifs anciens. La première attente de relance600ms précédait la sauvegarde existante1s : correction du harnais pour attendre le fichier (maximum12s), sans changer la sauvegarde. Premier build refusé par signature verify nécessitant expectedVersion, corrigé avec version du manifeste. Logs négatifs conservés.

## Exa

Endpoint hébergé `https://mcp.exa.ai/mcp`, [documentation officielle](https://exa.ai/docs/get-started/exa-mcp) consultée. Exa existait en mémoire côté moteur, mais une config déjà présente pouvait ne pas le contenir. Migration atomique à l’initialisation normale de StudioModel et à l’ouverture du fichier, prévisualisation isolée. Aucun appel de recherche, texte utilisateur, secret ou modèle nécessaire.

`swiftc -swift-version 6 -warnings-as-errors -parse-as-library -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk <MCPConfiguration.swift Localization.swift tests/mcp-configuration-check.swift>`, puis exécution : **réussis**. Création de fichier/parents, ajout à ancienne configuration, conservation serveur local/clé inconnue, idempotence exacte en octets, Exa existant personnalisé ou désactivé inchangé, six fichiers invalides refusés et préservés. Le hardening-check vérifie aussi l’ajout par le constructeur réel StudioModel dans un MLXL3_HOME jetable.

## Vérification et livraison

Sources Desktop complètes hors main compilées avec Swift6/warnings-as-errors, transport réel/fake-engine et préférences/dossiers jetables. `python3 -m py_compile tests/fake-desktop-engine.py`, `zsh -n scripts/build-macos-app.sh scripts/check-desktop.sh`, `plutil -lint apps/MLXL3Studio/Resources/Info.plist`, `git diff --check` ; **tous réussis**, dont le hardening-check après correction de l’attente de sauvegarde ; CI complète à consigner. Pas de vérificateur formel Swift/Foundation disponible ; contrats de concurrence du compilateur, tests déterministes et cas finis, pas preuve complète de Swift/JSON/IO. Aucun code Rust/Metal/FFI modifié, Kani distant porte sur les contrats préexistants. Aucun modèle ni GPU/benchmark local.

Moteur embarqué version1.2.1 choisie depuis runtime-info par build-macos-app.sh, manifeste validé par l’updater selon sa propre version et sa compatibilité Desktop1.2.2. Ajout d’une régression EngineRuntimeStore avec app1.2.2/runtime1.2.0 en fixture ; validation de l’app1.2.2/runtime1.2.1 réelle dans le DMG prévue. Paquet final, signature/empreintes et publication en attente à cette étape.


Révision de livraison avant publication : l’updater de Desktop1.2.1 exige un manifeste moteur de version égale à l’app entrante. Corriger seulement l’updater entrant ne permettrait donc pas à l’ancienne app d’installer1.2.2 avec moteur1.2.1. **Abandon de ce découplage dans cette livraison**, restauration du packager/validateur/tests updater précédents. Cargo.toml/lock passent à1.2.2 (identité uniquement), algorithmes Rust/Metal inchangés ; moteur embarqué identifié1.2.2, asset indépendant engine-v1.2.1 conservé. Première compilation de paquet interrompue/supersédée avant publication pour appliquer cette contrainte. Nouvelle compilation justifiée par la compatibilité réelle du chemin de mise à jour.
