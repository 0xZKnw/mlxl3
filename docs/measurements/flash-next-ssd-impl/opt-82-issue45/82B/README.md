# 82B — fermeture du faux succès à l’arrêt du benchmark

Le contrôleur pouvait publier `validated` après la dernière réponse correcte malgré une erreur ou un JSON invalide à la fermeture. Huit régressions nouvelles échouent sur le code initial (`DID NOT RAISE`). Le correctif distingue l’EOF normal, attend la fin du lecteur et inspecte sa file avant de construire les médianes et le statut final. Une erreur tardive, un événement invalide, une sortie non nulle ou l’absence d’EOF normal produit un rapport `failed`. Les événements de statut sains restent acceptés.

Seuls `scripts/benchmark-flash-bridge.py` et `tests/test_flash_benchmark.py` ont été modifiés. Leurs avant/après, empreintes et patch sont archivés ici. Aucun moteur, modèle, GPU, journal, index, workflow ou push n’a été modifié par cet agent. Ce correctif commun aux deux bras ne constitue pas une accélération moteur.

## Vérification inspectée

- Régression sur l’ancien code : **8 échecs attendus**, 55 cas désélectionnés, 2,49 s ; log conservé.
- Fichier complet pertinent : **66 tests réussis**, 22,15 s, comprenant les processus sains, erreurs avant/après réponse, EOF, JSON invalide, exit 7, silence, DFlash et comparaison OFF/MTP, nettoyage et rapport final. Dix-huit cas ont été ajoutés.
- Ruff format et lint strict : succès ; deux fichiers déjà formatés.
- Tentative CrossHair réelle : échec de disponibilité (`No module named crosshair`). Aucune propriété n’est présentée comme prouvée. Aucun outil installé pour contourner cette limite.
- CI existante : `.github/workflows/rust.yml` exécute le fichier pytest et le format/lint des deux sources. Le commit local n’a pas encore de run CI inspectable.

Les commandes, sorties, SHA et limites exactes figurent dans `evidence.json`. Les tests utilisent de vrais sous-processus Python jetables à l’entrée de production `run`; aucun modèle réel n’a été chargé. La suite Python complète du dépôt n’a pas été répétée par cet agent. Cette correction ne certifie pas rétrospectivement tous les anciens rapports et n’affirme pas qu’ils contenaient une erreur tardive.

Tous les tests et la tentative symbolique étaient terminés et joints avant le lancement GPU du parent. Aucune commande lourde n’a été relancée pendant la campagne GPU. Les plans suivants doivent figer la nouvelle empreinte du contrôleur commune aux deux bras.
