# Correctif local LFM : MTP, sampling et mémoire

La préférence globale MTP/DFlash pouvait survivre à un changement Qwen→LFM. La génération désactivait déjà ces modes sur LFM, mais l’inspecteur verrouillait les curseurs sur la préférence enregistrée et masquait son bouton OFF. Le correctif utilise les capacités du modèle pour le verrouillage et les paramètres envoyés, conserve les réglages libres et laisse OFF accessible. Le bridge fournit aussi le profil de cache logique LFM.

## Vérification

- [Swift](swift-verification.json) : suite Desktop complète, puis 821 assertions / 180 états sur le code final ; ancien comportement et mutation du verrouillage détectés. Compilation Swift 6 stricte des harnais. Aucun vérificateur source Swift applicable disponible : ces tests ne constituent pas une preuve formelle.
- [Rust](rust-verification.json) : 80 tests CPU réussis, 3 ignorés historiques ; format, Clippy strict, build mlx,chat et compilation des tests MLX réussis. Trois harnais Kani sur le vrai accumulateur CPU : 220 obligations SUCCESS et 11 couvertures SATISFIED. Domaines : total i64 non négatif, dimensions i32 avec l’autre dimension LFM fixe, et paires de dimensions i8 promues. Pas de preuve des allocations, de Metal, de SwiftUI ou de la concurrence. Le premier résumé humain indiquait 209 obligations à tort ; le recompte est conservé.
- [Bridge physique](physical-ready.json) : unique chargement du LFM2.5-24B-A2B EXL3 V2, contexte 4096 / maximum 131072 ; MTP/DFlash false, profil 20 480 octets/token + 245 760 octets fixes, oracle indépendant de la config. Aucune génération ni nouvelle mesure KL/PPL. Processus fermé proprement.
- [Contrôle visuel vivant](live-ui-observations.json) : curseurs modifiables, température 0,73→0,85, OFF cliqué et effectif, mémoire et recommandation visibles. Préférences et conversations jetables, processus nettoyé. L’essai AX headless antérieur échoué reste dans les preuves ; le contrôle Cua l’a complété.
- Les commandes exactes, empreintes et diagnostics figurent dans les manifestes et archives de preuve. Les workflows existants exécutent les suites Desktop et packaging sur PR/push ; aucun run GitHub n’est revendiqué pour cette branche locale non poussée.

## Audit joint

[Revue statique des 1629 lignes et empreinte du document](audit-review.json). Le document est traité comme une liste de propositions tierces. Plusieurs protections et optimisations existent déjà. Le défaut distinct de finish_reason API lorsque max_tokens est épuisé est confirmé mais reste à traiter ; les pistes de performance non mesurées ne sont pas présentées comme des gains.

## Livraison

Desktop 1.4.4 build 26 et moteur 1.4.5 sont versionnés séparément. Le manifeste final de livraison consigne l’état d’installation, les SHA-256 et la sauvegarde locale. Aucune publication, fusion ou release GitHub dans cette tâche.

L’estimation LFM décrit le cache logique persistant ; elle exclut backing retenu par les vues, temporaires, allocateur et RAM processus. Les anciennes valeurs de sampling déjà écrasées par une app antérieure ne sont pas récupérables.
