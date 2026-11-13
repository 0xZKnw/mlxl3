# Desktop : notifications, modèle de démarrage et chargements

Suite de `88415ac` sur `codex/desktop-theme-startup`. Thèmes et chargement au
démarrage déjà présents sont conservés. Aucun changement Rust, Metal ou numérique.

## Fonctions

Les réglages proposent désormais un modèle de démarrage explicite, identifié par
son chemin, ou « Dernier modèle sélectionné ». Si un modèle explicite a disparu,
aucun autre modèle n’est chargé silencieusement : l’utilisateur peut choisir un
autre modèle ou charger manuellement. OFF conserve le choix sans charger.

Notification de fin de génération, OFF par défaut : autorisation macOS demandée
uniquement à l’activation. Notification générique avec son après une réponse
réussie en arrière-plan. Aucun prompt, réponse ou titre de conversation inclus.
Pas de notification au premier plan, sur erreur, arrêt, limite de contexte ou
événement complet dupliqué/non corrélé. Refus et erreurs de livraison restent
visibles sans arrêter le moteur. Une autorisation tardive ne réactive pas une
préférence déjà désactivée. Le client OS est injecté dans les tests pour ne pas
modifier les permissions réelles de l’utilisateur.

## Chargement Flash Next

L’incident signalé (« Indisponible » rouge au premier chargement) est intermittent
et a cessé de se produire pendant le diagnostic, selon l’utilisateur. **Sa cause
spécifique n’a pas été reproduite et n’est pas affirmée corrigée dans le moteur.**

Deux défauts du chemin Desktop sont corrigés : sélection identique ignorée après
un échec tant que le processus restait actif ; anciennes tâches de chargement
pouvant encore agir après une nouvelle sélection. Les tentatives sont maintenant
annulées et identifiées, le contexte est capturé avec la sélection, les anciennes
erreurs sont ignorées. Le bridge contrôle l’annulation avant d’arrêter/créer un
processus et après l’attente de l’ancien. L’éjection annule les tâches en attente.
L’erreur apparaît en texte avec un bouton Réessayer, sans passage par un autre
modèle. La relance conserve le contexte et les réglages, aucune réduction de qualité.

Preuve de régression : le moteur factice échoue une première fois sans quitter,
puis réussit à la tentative suivante. La version avec l’ancien garde échoue au
contrôle « same-model retry » ; le nouveau code passe. Cette preuve ne permet pas
d’attribuer rétrospectivement l’incident intermittent à ce même mécanisme.

Contrôles physiques successifs et isolés, moteur installé2.0.2/commit9df28b6d2a10 :
Flash Next direct contexte4096 puis contexte0, chacun ready +8tokens nonvides/hash,
processus fermés/rejoints. Puis vrai StudioModel.start→liste du registre jetable→
choix explicite Flash Next→ready, sans autre modèle préalable, moteur du bundle
final, arrêté et rejoint. Bornes180s par chargement/génération et contrôleur240s
pour le dernier contrôle. Aucun compilateur/proveur concurrent, aucun benchmark
vitesse. Le checkpoint de63Go utilise le chemin natif du moteur récent inchangé.

## Vérification et limites

Suite Desktop complète passée, dont **55 assertions préférences/démarrage**,
**14 assertions notifications/relance**, 114 assertions idle, import150cas/256
encodages, bridge, annulation, rendu, CLI, MCP et Tune64grilles.
Python **358 passés/4 ignorés historiques**, dont20tests de copie du runtime.
Ruff, format Python, py_compile, syntaxe Zsh et git diff --check passent.
Build debug/release, signature stricte du bundle et hashes du moteur embarqué passent.

Skill formal-proof-skill relu avec référence Python ; aucun vérificateur Swift
configuré, CrossHair tenté mais module absent. Swift/IO/notifications macOS ne
sont pas prouvés formellement ; tests avec client OS injecté, pas de vraie demande
d’autorisation ni de notification système déclenchée automatiquement. Le clic
natif des nouveaux contrôles n’est pas automatisé. Premiers échecs de lint/imports
et de packaging conservés, relances finales réussies. Les anciennes notes de
nettoyage locales d’opti.md sont préservées hors du commit de fonctionnalité.

## Essayer la version complète

Fermer MLXL3 Desktop puis lancer depuis la racine du dépôt :

```sh
./scripts/build-desktop-test.sh --launch
```

Ce script construit le Desktop release et crée `dist/MLXL3 Desktop Test.app`.
Il copie le moteur installé actif seulement après validation du manifeste, de la
compatibilité et des hashes. Il ne recompile pas l’ancien moteur du checkout,
qui ne prend pas en charge qwen4_exp. Il ne remplace pas /Applications et ne
change ni le registre des modèles ni les checkpoints. L’app de test utilise les
conversations et préférences habituelles ; le lanceur refuse une UI déjà ouverte
pour éviter deux interfaces écrivant l’historique simultanément.

Preuves : [résumé](../measurements/desktop-followup-20261102/verification-summary.json).
