# MLXL3 Desktop v1.2.1

La recherche de modèles EXL3 dans **Models → Discover** fonctionne de nouveau, notamment pour `qwen3.6`. L'app accepte les formes du champ d'accès renvoyées par le catalogue (`null`, booléen, approbation automatique ou manuelle), au lieu d'échouer sur une réponse contenant des modèles valides. Une recherche en erreur n'affiche plus simultanément « Aucun résultat ».

Les paramètres **Sampling** restent verrouillés quand **MTP est activé**. Un message en français ou en anglais explique le verrouillage et indique de désactiver MTP pour modifier les paramètres.

Les téléchargements disposent d’une nouvelle carte intégrée au style de l’app : barre de progression, pourcentage, volume reçu et débit en **Mo/s** (MB/s en anglais). La pause conserve l’avancement et propose la reprise ; un état distinct confirme la fin du téléchargement. Le calcul du débit ignore les octets conservés d’un téléchargement précédent.

Cette version embarque le **moteur 1.2.1**, avec l'optimisation MTP publiée séparément. Apple Silicon, macOS26.2 ou supérieur ; aucun poids de modèle inclus. Signature ad hoc, comme les versions précédentes.

La recherche, l'ouverture des fiches, les erreurs, les réponses vides, l'annulation et les caches sont couverts par des tests de régression utilisant le vrai transport CLI et un moteur factice. Le contrôle réseau initial retourne60 modèles Qwen3.6, désormais décodés correctement par l'app ; aucun benchmark ni inférence supplémentaire n'a été lancé.

Le problème signalé d'activation d'une mise à jour du moteur dans une app déjà ouverte reste prévu pour une correction ultérieure. Après installation d'un moteur seul, fermer et rouvrir l'app si nécessaire. La mise à jour complète de Desktop redémarre l'app.

[Validation et limites](desktop-v1.2.1-validation.md).
