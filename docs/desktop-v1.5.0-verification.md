# Desktop 1.5.0 — vérification de livraison

État au 18 octobre 2026 : **Desktop1.5.0/build30 publié**, [release stable](https://github.com/0xZKnw/mlxl3/releases/tag/v1.5.0) du18octobre à18:15:24UTC. Les contrôles fonctionnels locaux et les jobs CI Desktop/Rust Linux/Rust macOS/EXL3 du SHA exact ont réussi. Téléchargement public et sélection par l'updater vérifiés. L'accord humain « vas y » a levé le refus automatique initial du push ; ce refus et l'état local antérieur restent conservés dans le journal. L'application utilisateur n'a pas été remplacée. Les 23 timeouts Kani locaux restent non vérifiés ; le job Kani bibliothèque distant est encore en cours au relevé de publication.

## Sources et paquet

- Source du DMG : `eaf296468c7ea4d92edab9e656bf749ac71f3a36`, checkout propre au moment du build (`tracked_changes: false`). Base distante intégrée : `0f5609effa88086fabb9e594d68a1d82bd016d41`, après Desktop 1.4.8/build29. Les modifications personnelles préexistantes sont exclues.
- Desktop **1.5.0/build30**, moteur embarqué **1.5.0**, protocole bridge 1, MLX **0.32.2**, arm64, macOS **26.2+**. Signature ad-hoc, sans notarisation Apple. Aucun nouveau canal moteur publié.
- Machine de contrôle : Apple M5/Mac17,2, 24 Gio, macOS 27.2 (26B5112a) ; Swift 6.4, SDK 26.5, Rust 1.98.1, Kani 0.68.0, Python 3.12.14. Les temps imprimés par les régressions existantes ne sont pas des gains de performance qualifiés.

| Artefact local | Octets | SHA-256 |
| --- | ---: | --- |
| `MLXL3-Desktop-v1.5.0-b30-Apple-Silicon.dmg` | 73805588 | `590003037dbd8daa7ee6bc32f7cb530846cc72c33768ebd458e0cb628113208d` |
| `MLXL3-Engine-v1.5.0-arm64.tar.gz` — produit de build, non publié séparément | 66984710 | `be58705754b90e5b5321349a57950d16fa6e5b6f313d8cbc72461985e7a674dd` |

Le vrai validateur de l'updater accepte le DMG : montage en lecture seule, signature, versions, compatibilité, empreintes du runtime et exécution de `list --json`, puis démontage. Les **15 cas signés** de versions indépendantes, incompatibilité et corruption passent. L'archive moteur est installée dans un registre jetable : signature, déplacement, exécution, rejet et repli passent. `codesign --verify --deep --strict` et `hdiutil verify` réussissent.

Les quatre fichiers du runtime ont des tailles et SHA-256 conformes au manifeste, et sont identiques entre l'archive et l'app signée. Les quatre Mach-O contrôlés sont arm64. Le `runtime-info` réel annonce `eaf296468c7e`, version 1.5.0, profil release, chat/MLX activés. Le [reçu du paquet](measurements/desktop-v1.5.0/package-final.json) conserve les empreintes individuelles et celles des **38 sources Swift**.

`vtool -show-build` relève minOS 26.0/SDK 26.0 pour l'exécutable Swift, et minOS 26.2/SDK 26.5 pour le moteur et ses deux dylibs. L'Info.plist fixe la compatibilité effective à 26.2. Le SDK 26.5 est l'option transmise aux commandes de build et au manifeste ; ces versions de liaison observées sont consignées séparément.

## Tests fonctionnels et compilation

| Contrôle | Résultat inspecté |
| --- | --- |
| Rust CPU, `cargo test --locked --features chat` | **167 réussis, 3 ignorés**, zéro échec à la seconde passe |
| Rust MLX/chat, `cargo test --locked --features mlx,chat -- --test-threads=1` | **192 réussis, 85 ignorés**, zéro échec |
| Format Rust et Clippy strict CPU + MLX, tous les targets | Réussis |
| Python, compilation puis suite complète | **840 réussis, 115 sautés**, 10 avertissements |
| Nouvelles fonctions Desktop | **219 contrôles** : documents/recherche, persistance/migration, branches, projets/profils, MCP, schémas et parcours bridge |
| Autres contrôles Desktop | Mémoire/contexte (240 cas), économie mémoire, composer (49 états), DFlash/MTP, Hub/transferts/erreurs, récupération, déchargement (114 assertions), sampling (821 assertions/180 états) réussis |
| Imports et rendu | Texte/PDF/erreurs, 150 cas de frontière et 256 encodages, bridge/annulation, 1 194 fragments de code et 4 845 de Markdown, **786 cas de fade/63 images**, scroll réel, **9 formules LaTeX**, 63 parités parser/worker, copie et historique réussis |
| Fin de suite Desktop | CLI : gros flux/NDJSON/annulation/erreurs ; téléchargement (300 cas), Tune (64 grilles) et configuration/migration/corruption MCP réussis |
| Harnais `streaming-performance` modifié | Compilation stricte et chemin d'argument invalide vérifiés : `failed`, `complete=false`, zéro échantillon. **Aucun benchmark exécuté.** |

La commande complète `scripts/check-e2e.sh` a initialement terminé avec **code 133**, après les succès Python et des premiers contrôles Desktop : un hôte de test du transcript omettait le `StudioModel` désormais utilisé par les nouveaux boutons. Les deux hôtes actifs concernés ont été corrigés avec historique et préférences jetables. Le test de réponse et tous les contrôles qui suivaient cet arrêt ont été compilés et exécutés séparément avec succès. La commande complète n'est donc **pas présentée comme un unique run réussi**. Aucune source Swift de production n'a changé après les 219 contrôles ; les tests déjà réussis n'ont pas été répétés.

Quatre vues AppKit/SwiftUI ont été rendues hors écran et inspectées : [chat](measurements/desktop-v1.5.0/chat.png), [projets](measurements/desktop-v1.5.0/projects.png), [MCP](measurements/desktop-v1.5.0/mcp.png), [source](measurements/desktop-v1.5.0/source.png). Fonds sombres opaques et textes lisibles. Sources UI identiques entre ces rendus et le DMG final ; le dernier commit ne change que deux fixtures et le journal. Les auto-contrôles du bundle signé de la deuxième construction passent aussi. Ils comprennent le contrôle physique existant d'allocation Metal de64Mio ; ce n'est pas une validation numérique complète du GPU.

## Défauts trouvés pendant l'intégration

1. La première suite Rust reproduit un faux marqueur SSE de réussite `[DONE]` après une erreur JSON. Le chemin d'erreur envoie maintenant l'erreur puis termine le flux ; les régressions HTTP complètes passent à nouveau.
2. Le premier Clippy MLX échoue sur cinq anciens tests GPU, qui n'avaient pas reçu le nouvel argument de mise en mémoire tampon. Les appels sont synchronisés, Clippy et les tests MLX passent. Ces tests GPU spécifiques restent ignorés et ne sont pas annoncés exécutés.
3. Le test de scroll détecte l'environnement SwiftUI manquant ci-dessus. L'app réelle fournissait déjà cet environnement ; correction limitée aux fixtures. Le test initial en échec est conservé.

Les candidats `ffe423b7` et `20869113` sont remplacés par `eaf29646` ; leurs reçus et logs restent conservés. Leurs contrôles ne sont pas attribués silencieusement à un autre binaire.

## Vérification source

Kani cible d'abord `structured_retry_budget_never_restarts_or_wraps` : **1 harnais réussi, 57 obligations sans échec, 3/3 covers**. Il utilise les entrées symboliques `i64` et `usize` 64 bits sans `assume` et un oracle `u128` indépendant : budget positif restant, épuisement/rejet des limites invalides, absence de débordement et valeur spéciale `-1` illimitée. Aucun déroulement de boucle n'est nécessaire. Le budget consommé est partagé entre les tentatives JSON dans l'appelant de production.

La suite bibliothèque est terminée avec code 1 : **70 harnais réussis sur 93 ; 23 timeouts**, chacun limité à 60 s. Dans les harnais conclus, Kani comptabilise **9 555 obligations dont 160 inatteignables**, avec **202/202 covers** satisfaits. Les obligations inatteignables ne sont pas des chemins exercés. La suite entière n'est donc pas validée. Les 23 propriétés non vérifiées sont les transitions `dflash_copy_len_0` à `dflash_copy_len_20`, `lookup_proposes_only_latest_in_bounds_history` et `tuning_never_selects_missing_or_slower_candidate` ; aucun contre-exemple n'est fourni par ces timeouts. Ces fonctions héritées ne sont pas modifiées par cette release. Les résultats de chaque harnais sont conservés dans le résumé JSON et le log brut.

Le contrat HTTP `validated_usage_respects_budget_without_total_overflow` passe dans ses **deux cibles de test**, soit 65 obligations et 3 covers par cible. Entrées symboliques `u64` pour prompt/completion et `Option<u64>` pour le budget, sans hypothèse : rejet d'un prompt nul ou dépassement du budget, somme exacte quand représentable et saturation à `u64::MAX` autrement. Il s'agit de la même propriété compilée deux fois, pas de deux preuves indépendantes. Cela ne prouve pas le transport HTTP asynchrone.

Les bornes et `#[kani::unwind]` des harnais existants restent inchangés, avec assertions de déroulement activées. Les sources de bibliothèque vérifiées sont identiques au DMG final ; après leur compilation, seules des fixtures GPU sous `cfg(test,mlx)` et des fixtures Swift ont changé. Un succès de ces contrôles CPU ne prouve ni SwiftUI, ni la FFI MLX/Metal, ni le comportement concurrent ou une génération entière. Le harnais URL MCP qui provoquait une erreur interne Kani0.68 dans la campagne initiale reste **non vérifié** ; son source et son diagnostic sont conservés dans le rapport historique. Il n'a pas été relancé sans changement d'outil ou de code pertinent.

Pas de preuve source Swift/Python revendiquée : compilation Swift6 stricte et tests natifs ; CrossHair absent dans l'environnement. La preuve bornée CPU et les tests sont distincts d'une preuve de l'application entière.

Les autres contrats structurés ont un domaine précis : longueurs et bornes symboliques `usize`64bits ; schéma booléen appliqué à la valeur JSON `null` ; permissions sur une liste vide ou un nom parmi `read`/`write` et ces mêmes noms en requête (deux booléens symboliques, déroulement16). Ils ne constituent pas une preuve de tous les schémas JSON ou de toutes les listes d'outils. Le schéma booléen fixé exclut volontairement les autres branches du validateur, ce qui explique une partie des obligations inatteignables.

## Moteur final sur modèle réel

Le binaire signé du paquet final est exécuté avec **Qwen3.6-35B-A3B EXL3 2,49 bpw**, contexte 2048, greedy/topK1, pénalité 1, MCP/MTP/DFlash désactivés, aucun cache de prompt réutilisé, un seul modèle GPU et registre/home jetables. Aucun compilateur ou proveur de cette campagne encore actif pendant ces contrôles ; **aucune vitesse mesurée**.

- Première passe conservée en échec, budget 256 : `JSON token budget exhausted`, aucun succès global, deux contrôles préalables réussis, processus joint/code 0. Ce budget ne suffit pas au modèle. Le moteur ne réinitialise plus le budget à chaque tentative.
- Avant la seconde passe, changement explicite du seul budget de la requête valide à 1024 ; binaire/checkpoint/prompt/contexte inchangés. **6 contrôles passent** : capacité annoncée, schéma non pris en charge rejeté avant contenu, réponse `{"ok":true}` validée avec l'oracle indépendant `jsonschema`, schéma faux/budget 1 rejeté sans réponse publiée, annulation sans réponse invalide et ping après annulation. **372 tokens générés au total, ≤1024**. Arrêt et lecteur joints proprement, code 0.

Les deux rapports restent distincts dans `physical/` et `physical-1024/`, avec sources des contrôleurs et stderr. Il s'agit d'un contrôle fonctionnel sur ce checkpoint et ce prompt, pas d'une preuve de conformité ou de qualité pour tous les modèles.

## Commandes et preuves

Dans le worktree isolé, avec le SDK26.5, les caches Swift dans `/private/tmp`, le Python du virtualenv existant et `MLXL3_MLX_ROOT` pointant vers MLX0.32.2 :

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo test --locked --features chat
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --features mlx,chat -- --test-threads=1
cargo kani --lib --no-default-features --harness structured_retry_budget_never_restarts_or_wraps
cargo kani --lib --no-default-features --jobs 1 --output-format terse -Z unstable-options --harness-timeout 60s
cargo kani --no-default-features --tests --harness validated_usage_respects_budget_without_total_overflow --jobs 1 --output-format terse -Z unstable-options --harness-timeout 60s
MLXL3_SKIP_METAL_CHECK=1 scripts/check-e2e.sh
scripts/build-macos-dmg.sh
```

Les reruns concernés reprennent les commandes de `scripts/check-desktop.sh` à partir de `response-ui-check`, avec `swiftc -O -swift-version 6 -warnings-as-errors`, puis CLI/téléchargement/Tune/MCP. Le contrôle DMG est compilé à partir des sources réelles `UpdateManager.swift`, `EngineRuntimeStore.swift`, `Localization.swift` et du test `updater-application-check.swift` concaténé. L'installation jetable utilise `tests/updater-check.swift`. Les journaux, y compris les premiers échecs, sont conservés dans [measurements/desktop-v1.5.0](measurements/desktop-v1.5.0/) ; les binaires de test et intermédiaires restent dans `build/desktop-v1.5.0/` et ne sont pas publiés.

Un contrôleur supplémentaire compile les trois sources réelles de l'updater ci-dessus avec `swiftc -warnings-as-errors -swift-version 6 -parse-as-library` et [ce point d'entrée conservé](measurements/desktop-v1.5.0/PublishedReleaseCheck.swift.txt). Il exige Desktop1.5.0/build30, le nom, la taille et le digest exacts du DMG, et le canal moteur toujours à1.4.8. Avant publication, le vrai flux GitHub est correctement refusé. Trois entrées vides/incomplètes/invalides sont refusées sans faux succès. Huit cas d'intégration passent : brouillon ignoré, publication synthétique conforme acceptée, mauvais digest/taille/build, URL non approuvée et canal moteur absent refusés. La fixture synthétique ne vaut pas vérification de la publication réelle.

Incidents de préparation conservés : compilation du contrôleur d'abord interrompue par une modification de son source puis par une sortie de linker concurrente ; compilation séquentielle finale réussie. La première fixture synthétique conservait les URL `untagged` du brouillon et a été correctement refusée ; seules les URL de cette fixture ont été corrigées. Une commande de checksum a aussi échoué avec le Python système sans `hashlib.file_digest`, puis réussi avec le virtualenv Python3.12. Aucune de ces étapes n'a modifié le DMG ou le code de production.

## Limites et publication

Les workflows existants `check.yml`, `rust.yml` et `quantization.yml` ont été inspectés : PR et push sur main, compilation/test Desktop, variantes CPU/MLX et Kani, dépendances identifiées, sans credential de production. Les nouvelles régressions font partie des scripts exécutés par ces workflows. Le push fast-forward du SHA exact `eaf29646` sur main a déclenché [les contrôles Rust](https://github.com/0xZKnw/mlxl3/actions/runs/39151944315), [les régressions Desktop](https://github.com/0xZKnw/mlxl3/actions/runs/39151944336) et [les contrats EXL3](https://github.com/0xZKnw/mlxl3/actions/runs/39151944365). Les jobs Rust Linux et macOS, dont compilation/lint MLX, ont réussi ; leurs résultats et comptes non nuls ont été inspectés. EXL3 et la suite Desktop **complète en CI** ont réussi. Ce run CI est distinct du parcours local initialement interrompu puis réparé en deux passes. Le job Kani bibliothèque est encore en cours au [relevé de publication](measurements/desktop-v1.5.0/ci-release-snapshot.json), sans succès global revendiqué ; le contrat HTTP Kani est déjà réussi. Les anciens relevés sans run et pendant les tests sont conservés. Les annotations CI signalent aussi des actions Node20 exécutées avec Node24 et la capacité des runners macOS ; les jobs concernés se terminent néanmoins avec succès.

Non qualifiés : 85 tests Rust MLX ignorés (notamment checkpoints et parcours GPU spécifiques), 115 tests Python sautés, reconnaissance/qualité audio et microphone, services MCP distants, comportement utilisateur complet sur tous les modèles. Les preuves initiales sur la base1.3.0 restent [historiques](desktop-project-features-2026-10-18.md). Aucun OCR, gain de vitesse, gain mémoire ou gain de qualité n'est revendiqué. L'historique version2 demande une copie séparée de l'historique version1 pour revenir à un ancien Desktop ; l'app installée n'a pas été remplacée par ces tests.

Le tag annoté `v1.5.0` (objet `33c76331d2d91e6929a89da452f6250067e48004`) vise exactement la source du DMG `eaf29646`. Les commits ultérieurs de cette livraison ne contiennent que documentation, métadonnées et preuves ; aucune source exécutable du paquet n'a changé. Le canal Desktop `releases/latest` renvoie maintenant `v1.5.0`. Le vrai sélecteur Swift, appliqué au flux GitHub public obtenu sans authentification, sélectionne Desktop1.5.0/build30 avec taille/digest exacts et conserve le canal moteur `engine-v1.4.8`.

Les trois fichiers publics ont été retéléchargés sans authentification et comparés aux octets locaux et aux digests GitHub : DMG, checksum et validation JSON réduite. Leurs noms, tailles, SHA-256, URL et l'heure de publication sont conservés dans [le reçu public](measurements/desktop-v1.5.0/publication.json). Le JSON de release exclut chemins locaux, logs bruts et métadonnées du checkpoint. Aucune release moteur indépendante, aucun poids et aucun remplacement de l'app installée. Les logs initiaux restent inchangés ; inventaires SHA-256 initial et de publication contrôlés.
