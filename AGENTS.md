# MLXL3 — consignes pour les agents

## Journal obligatoire des optimisations

Ces consignes s'appliquent à tout le dépôt : moteur, kernels Metal, inférence,
quantification, CLI, GUI, mémoire, cache et téléchargements.

1. **Avant toute recherche, modification ou benchmark d'optimisation, lire
   `opti.md` à la racine**, puis les rapports historiques pertinents qu'il
   référence. Chercher aussi dans le code si la piste existe déjà.
2. Ne pas refaire un essai déjà documenté sans raison nouvelle et explicite
   (nouveau kernel, forme, modèle, matériel, version ou protocole corrigé).
   Une répétition de validation reste permise : noter l'essai précédent et
   ce que cette répétition doit vérifier avant de la lancer.
3. **Avant chaque nouvel essai, ajouter une entrée dans `opti.md`**, avec un
   identifiant, l'hypothèse, le changement, la baseline et le protocole prévu.
   Pour une série paramétrée, une entrée peut référencer une matrice détaillée
   conservant chaque variante et son résultat.
4. **Après chaque essai, mettre cette entrée à jour immédiatement**, même
   pour un échec, crash, interruption, régression ou résultat non concluant.
   Indiquer les résultats, les contrôles de qualité, les limites et la décision.
   Avant de s'arrêter ou de rendre la main, consigner l'état des essais en cours.
5. Enregistrer les commandes/options, modèle et quantification, versions,
   contexte/tokens, répétitions, alimentation/conditions connues, métriques
   avant/après avec unités, et chemins des preuves. Marquer « non mesuré »
   plutôt qu'inventer une valeur. Distinguer microbenchmark et gain réel,
   mémoire allouée et RAM processus, égalité du texte et validation numérique.
6. Conserver les résultats négatifs et les anciennes conclusions. Ajouter une
   révision datée si elles changent ; ne pas les effacer. Ne pas additionner
   des gains issus de tests incompatibles ni présenter du bruit comme un gain.
7. Le journal doit distinguer **validé**, **rejeté**, **non concluant**,
   **bloqué/interrompu** et **en cours**, ainsi que l'état d'intégration réel
   (prototype, code local, app installée, publication). Un benchmark réussi
   n'implique pas que l'app a été mise à jour.

Les rapports détaillés peuvent rester dans `docs/` et les mesures brutes dans
leurs fichiers habituels, mais `opti.md` doit toujours contenir un résumé et
leurs liens. Si les logs temporaires ont disparu, le signaler : leur absence
n'autorise pas à refaire silencieusement un essai. Ne pas lancer d'optimisation
uniquement pour remplir le journal, ni publier sans demande de l'utilisateur.
