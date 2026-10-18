# Desktop 1.5.0 — vérification de livraison

État : intégration à la base publiée 1.4.8 terminée ; contrôles de livraison en cours. Base main : `0f5609effa88086fabb9e594d68a1d82bd016d41`. Les modifications locales préexistantes du checkout personnel sont exclues.

Les contrôles portent sur Desktop 1.5.0/build30 et son moteur embarqué 1.5.0, Apple Silicon, MLX 0.32.2, SDK macOS 26.5. Ils sont exécutés dans le worktree isolé attaché à cette tâche, avec préférences, registres et historiques jetables. Aucun benchmark de performances n'est prévu et aucune installation utilisateur n'est effectuée.

Les commandes, journaux bruts et états d'échec sont conservés localement dans `build/desktop-v1.5.0/`. Le relevé public de validation donnera les SHA source, tailles/empreintes des artefacts, comptes de tests et liens de CI effectivement inspectés. Les preuves initiales de FEAT-WORKSPACE sur la base 1.3.0 restent historiques et distinctes.
