# Desktop v2 beta — revue du 13 novembre 2026

Cette livraison reste une **bêta de la v2**. Base : `3a5019d29eee3135b172eb14ef9c802963cfee49`, branche locale `codex/v2-readiness`. Le moteur de calcul n'a pas été modifié. Cette revue cible les parcours sensibles et les surfaces demandées ; elle ne constitue pas un audit exhaustif ni une garantie d'absence de bugs.

## Problèmes reproduits et corrigés

| Priorité | Déclencheur et conséquence | Correctif et régression |
| --- | --- | --- |
| P2 | Passer de Chat à Agent, ou d'Agent à Chat, chargeait un modèle pourtant déchargé, même avec le chargement au démarrage désactivé. | Les trois entrées de navigation conservent l'état déchargé. Tests des deux sens, démarrage désactivé/activé, déchargement manuel/automatique, erreur et rechargement explicite. Les modèles déjà chargés continuent à fonctionner dans les deux modes. |
| P2 | « Nouvelle conversation » depuis le Lab créait deux chats lorsqu'il n'existait que des tâches Agent. | Réutilisation du chat déjà créé par le changement de mode ; scénario reproduit avant le correctif et vérifié après. |
| P2 | Le widget conservait une hauteur fixe après chargement ; son contenu pouvait dépasser le popover. | La taille réelle du contenu pilote celle du popover. Contrôle avec modèle absent et nom long, puis avec un véritable `NSPopover` : 388 pt au repos, 409 pt chargé. |
| P2 | La CI native macOS échouait avant le test des routes MoE : `MLXL3_MLX_ROOT` manquait dans cette étape. | Même résolution du paquet MLX que dans les étapes voisines. Corps exact de l'étape exécuté localement : **1 test réussi, 0 ignoré** ; `actionlint` réussi. |
| P2 | Pendant la refonte, la composition des styles lisait l'état activé hors de la vue installée ; certains boutons désactivés gardaient leur luminosité. | Lecture de l'environnement dans la surface du contrôle, sans survol actif lorsqu'elle est désactivée. Le test de pixels échoue avant le correctif et passe après. |

Les régressions initiales sont conservées localement dans `build/v2-readiness/autoload-before.log`, `new-chat-before.log` et `preview-sized.log`. La fixture Desktop a aussi été corrigée pour interpréter l'option globale `--registry` : elle annonçait auparavant le chemin du registre à la place du nom du modèle.

## Interface

- Widget compact avec le vrai logo, état moteur exact, modèle actif, métriques, mémoire et actions ; le cas d'erreur n'est plus affiché comme un moteur inactif.
- Menu d'affichage cohérent en clair et sombre : choix de largeur sobres, aperçu lisible, taille du texte et réinitialisation conservés.
- Boutons de comparaison MTP/DFlash compacts, avec les mêmes états et actions qu'avant.
- Verre dépoli natif teinté à la place des bordures spéculaires blanches ; application aux boutons partagés, retours, sélecteur de modèle, barre et inspecteur. Réduction de transparence et réduction des animations respectées.
- Aurora statique dans les popovers ; aucune nouvelle animation continue ni nouvelle fréquence de rafraîchissement du widget. Le rafraîchissement mémoire existant reste à 1 Hz. Aucun gain de consommation chiffré n'est annoncé pour ce changement.

Rendus natifs inspectés sur écran virtuel Retina : espace principal et inspecteur à 1230 × 780 pt, menu d'affichage à 320 pt, widget à 340 pt, thèmes clair/sombre et accent ambre. Les tests de disposition MTP couvrent aussi français/anglais, 200–320 pt et présence/absence de résultats. Captures et harnais locaux : `build/v2-readiness/screenshots/` ; les valeurs des captures sont des données de test.

## Vérification locale

Machine : Apple M5, 24 Gio de RAM, macOS 27.2 (26B123), Swift 6.4.2, SDK macOS 26.5, Rust 1.98.1, MLX 0.32.2. Les contrôles GPU ont été exécutés séparément, sans compilation ni autre modèle GPU concurrent. Le SDK 26.5 évite l'incompatibilité locale de chargement des macros SwiftUI avec le SDK par défaut.

| Contrôle | Résultat inspecté |
| --- | --- |
| Python complet, via `scripts/check-e2e.sh` | **2 828 réussis, 106 ignorés, 155 sous-tests réussis**, 10 avertissements. |
| Suite Desktop Swift 6, avertissements traités comme erreurs | Contrôles exécutés en plusieurs passes : interruption initiale du compilateur sur un fichier modifié, puis contrôles visuels finaux et suite restante réussis sur sources figées. |
| Préférences / déchargement / modes | **404 / 120 / 2 324 assertions réussies** ; vrais sous-processus jetables, registres isolés et serveur HTTP authentifié. |
| Expérience Desktop / taille du texte / disposition MTP | **51 / 51 / 24 cas réussis**, plus dimensionnement du widget et rendu des surfaces. |
| Rust CPU, `cargo test --locked --no-default-features --features chat` | Bibliothèque : **434 réussis, 10 ignorés** ; autres exécutables et tests d'intégration terminés sans échec. |
| Variante MLX, `cargo test --locked --no-default-features --features mlx,chat --lib -- --test-threads=1` | **461 réussis, 241 ignorés**. Les tests physiques ignorés ne sont pas couverts par cette commande. |
| Format et lint | `cargo fmt --check`, Clippy strict CPU et MLX, Ruff des fixtures, `git diff --check`, `actionlint` : réussis. |
| C++ ASan + UBSan | Capacités Metal : **1 190 contrôles** ; clés de cache : 256 clés, 65 536 paires × 4 combinaisons ; lecteur disque : **308 cas** ; sans erreur détectée. |
| Build Desktop optimisé | Réussi ; avertissements de dépréciation préexistants dans SwiftMath et SwiftPM. |

Commande principale : `MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk MLXL3_TEST_PYTHON=<venv>/bin/python zsh scripts/check-e2e.sh`. Les sources Swift de production sont compilées avec `-swift-version 6 -warnings-as-errors` par les harnais existants. Le parcours couvre notamment démarrage, annulation, crash, registres Agent, sauvegarde/récupération, téléchargements, MTP/DFlash, imports, streaming, rendu et mises à jour. La première commande complète a terminé Python et les contrôles Desktop jusqu'au streaming, puis le compilateur a refusé un fichier de style modifié pendant sa lecture. Le contrôle visuel final a été recompilé et réussi ; seuls les contrôles suivants sont repris depuis les fonctions du script de production, sur sources figées. Logs locaux : `build/v2-readiness/e2e.log`, `experience-controls-final.log`, `desktop-rest.log`, `rust-cpu.log`, `rust-mlx.log`, `clippy-cpu.log`, `clippy-mlx.log`, `ci-codec-step.log`, `release-final.log`.

## Moteur physique et propriétés vérifiées

Le moteur livré avec la bêta 20 est utilisé sans recompilation : version 2.0.9, source `e7923197cbafc4f63688a8d5cb4a2ba455c4e6e9`, SHA-256 du binaire `1bef4f88d92f8264cb5b1785c6ebedc1e21cb91d5df6698a7d6492f91bfb65e7`. Les sources `native/`, `Cargo.toml` et `Cargo.lock` de cette source et de la base Desktop sont identiques.

- Primitives vision LFM3 contre oracles scalaires : **1 test physique réussi**.
- Q4/F16/BF16 et récupération après entrée invalide : **1 test physique réussi**. Une première invocation sélectionnait zéro test ; le lanceur l'a refusée et le contrôle a été repris avec le filtre exact. L'échec initial reste consigné dans `opti.md`.
- Bridge publié : LFM3-1.7B original 8 bpw et Qwen3.6-35B-A3B 2,49 bpw, contexte 4096, greedy/top-k 1, budgets **1/2/8 tokens**. Six générations non vides aux comptes attendus, refus d'un budget nul et arrêts propres avec processus joints.
- Kani 0.68 / CBMC 6.11 : `cargo kani --bin mlxl3-rs --no-default-features --features chat --tests --harness openai::proofs::usage_total_never_wraps --output-format terse` : **5 obligations réussies**. Sur tous les couples symboliques de `u64`, le total est la somme exacte lorsqu'elle existe et sature à `u64::MAX` sinon. Ce harnais ne comporte pas de boucle à borner.
- Le harnais GQA `model_gqa_widths_reject_invalid_groups_and_overflow` a été interrompu après plus de cinq minutes dans CBMC : **non vérifié**, sans contre-exemple obtenu. Une première tentative du harnais de total sans `--tests` ne sélectionnait aucun harnais ; seul le passage corrigé ci-dessus est une validation.

Le champ historique `parity` des rapports du bridge MLX signifie ici budgets/texte/hash/historique cohérents, **pas une comparaison des logits avec une référence indépendante**. Ces tests et la preuve CPU ne prouvent ni l'ensemble du moteur, ni Metal, ni la qualité générale des modèles. Swift ne dispose pas ici d'un vérificateur fonctionnel adapté : compilation stricte et tests de production, sans revendication de preuve formelle Swift. Aucun nouveau benchmark de vitesse n'a été effectué. Protocole physique et clôture : entrée `V2-READINESS-20261113` dans `opti.md`.

Une tentative de test par injection de `accessibilityReduceTransparency` a été refusée à la compilation : cette clé d'environnement est en lecture seule dans ce SDK. Le harnais temporaire a été retiré ; aucune préférence système de l'utilisateur n'a été modifiée. Le repli opaque est implémenté, mais ce chemin d'accessibilité n'est pas qualifié par un rendu automatisé. Diagnostic conservé dans `build/v2-readiness/experience-accessibility.log`.

## Consommation du rendu final

Qualification `DESKTOP-MATTE-20261113`, deux passes de la fenêtre native optimisée 1024 × 640 pt Retina, 3 s de stabilisation puis 7 s par état, sans moteur chargé, après les compilations et tests. M5 sur batterie, thermique nominal. CPU du processus, avec **100 % = un cœur** :

| État | CPU des deux passes | Empreinte physique |
| --- | --- | --- |
| Repos visible | 1,315 / 1,323 % | 53,74 / 55,92 Mio |
| Génération simulée | 2,572 / 1,865 % | 67,88 / 68,11 Mio |
| Scène inactive visible | 0,274 / 0,016 % | 65,45 / 65,61 Mio |
| Fenêtre masquée | 0,009 / 0,018 % | 61,08 / 61,14 Mio |

Le mouvement et le rendu sont conservés. La dispersion des échantillons en génération et après inactivation reste visible ; aucune comparaison causale avec les anciens chiffres ni mesure de watts n'est annoncée. Les popovers statiques ne sont pas profilés séparément. [Sources, empreintes, protocole et chaque passe](measurements/desktop-matte-20261113.json). Le compteur GPU du processus renvoie zéro et WindowServer est partagé : aucune conclusion sur le coût GPU global ou l'autonomie.

## CI distante et limites restantes

Sur la base exacte `3a5019d2`, [Regression checks](https://github.com/0xZKnw/mlxl3/actions/runs/43504260266) échoue avec `Mode fixture timed out`. Ce timeout n'a pas été reproduit localement ; le test passe avec 2 324 assertions. Le diagnostic inclut désormais la ligne d'appel et l'état moteur/MCP/serveur, sans augmenter le délai ni masquer l'échec.

[Native Rust checks](https://github.com/0xZKnw/mlxl3/actions/runs/43504260446) comporte l'erreur de chemin MLX corrigée localement et un timeout Kani de 600 secondes sur `grouped_expert_routes_eighty_symbolic_inputs_are_bounded` : aucune preuve pour ce dernier domaine. Les jobs CPU Ubuntu, paquet MLX, Kani spéculatif et allocateur ont réussi. Les autres jobs encore actifs ou annulés ne sont pas déclarés validés. Les workflows existants couvrent PR et pushes sur `main` ; cette branche locale n'a pas de nouveau run distant.

Les 106 tests Python ignorés et les 241 tests MLX ignorés restent non exécutés par les suites générales : dépendances/modèles optionnels ou contrôle physique explicite. Cette campagne ne qualifie pas Gemma/Ling sur modèle physique, une génération vision complète avec images, toutes les longueurs de contexte ni la parité numérique générale. Les mesures CPU portent sur la fenêtre décrite ci-dessus, sans consommation GPU attribuable. Ces limites restent compatibles avec une livraison bêta, mais ne permettent pas de qualifier une v2 stable sans vérifications complémentaires.

## Livraison

Publication autorisée par l'utilisateur : **Desktop v2.0.0-beta.21 / build 52**, sur `main`, avec le moteur 2.0.9 inchangé. Le canal reste bêta. L'application installée n'est pas remplacée pendant la qualification. Source, DMG, taille, checksum et vérification du paquet figurent dans les [notes de version](release-v2.0.0-beta.21.md).
