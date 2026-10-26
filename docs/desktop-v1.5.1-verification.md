# Desktop 1.5.1 — vérification ciblée avant packaging

Base : tag annoté v1.5.0 `33c76331d2d91e6929a89da452f6250067e48004`, commit `eaf296468c7ea4d92edab9e656bf749ac71f3a36`. Desktop 1.5.1/build 33 ; moteur et Cargo/lock 1.5.0 conservés. Les 61 fichiers natifs suivis restent ceux de cette stable ; aucun port Flash Next, catalogue ou kernel.

Le patch canal commun `5ca22f981b3a8bb9ead1a6e644d2248cd0484acf71270456e715c0989636bb84` est porté à fuzz zéro sur UpdateManager, AppSettingsView, EngineRuntimeStore et les deux fixtures updater. La phrase UI finale rappelle seulement que les versions Desktop et moteur sont indépendantes.

- Deux compilations ciblées : Swift 6, `-warnings-as-errors`, SDK macOS26.5, sources de production UpdateManager/EngineRuntimeStore/Localization et vraies fixtures. Réussies.
- Suite updater : Stable/Bêta, suffixes/assets officiels, persistance, pas de downgrade, beta1→beta2 par build, annulation lookup/download/ready ; contrôles antérieurs de versions/URLs/hashes, archives hostiles, activation/fallback et bornes conservés. Réussie.
- Suite application : 15 cas signés de versions indépendantes, compatibilité/intégrité, activation moteur et remplacement/backup d’apps jetables. Premier arrêt sandbox SIGTRAP/code-5 conservé ; même binaire réussi hors sandbox, sans correction source. La cause basse-niveau du premier arrêt n’est pas démontrée.

Les tests emploient URLSession/URLProtocol, UserDefaults et fichiers jetables ; aucune vraie requête GitHub/HF, app utilisateur ou modèle. Tous processus lancés sont joints. Journaux et commandes : ROOT/build/stable151-channels/validation/run-result-v1-failed.json et application-v2-escalated.json. Le premier rapport reste un échec, pas un succès réécrit.

Limites au gel : AppSettingsView sera compilé dans le build officiel ; aucun paquet réel, DMG monté, nouvelle génération, débit ou preuve formelle Swift n’a été validé par ces seuls tests. Source committée, résultats du paquet final, artefacts et CI restent à compléter par le parent avant publication. La validation historique1.5.0 ne remplace pas ces contrôles.
