# MLXL3 — journal des optimisations

### ENGINE-2026-10-05-1.3.1 — livraison de la PR23 — en cours

- Demande utilisateur : « fais une nouvelle version du moteur pour ça », après fusion de PR23. Publier le canal moteur `engine-v1.3.1`, dernière version publique `engine-v1.3.0`. Baseline source fusionnée `f43118519d6af6cb15496e23520d4f61bd47e913`, tête revue/corrigée `a2b2d31a5d8995ff8d4c0102123004de539943be`, même arbre validé. Travail isolé ; modifications MTP locales du checkout principal préservées.
- Périmètre : Cargo/lock 1.3.1 et notes de release ; lecture Darwin bornée, cache de chargement, dispatch BM64 M5 et corrections des comparateurs déjà fusionnés. Pas de nouveau code d'optimisation ni benchmark. Protocole bridge1, MLX0.32.2, arm64, macOS≥26.2/Desktop≥1.2.0 ; canal Desktop conserve sa version1.3.0.
- Antécédents : revue PR23 et FIX01..03, 31 harnais Kani CPU, 64 cas QMM debug exacts ; livraisons engine1.2.1 et engine/Desktop1.3.0, installation/relocation/repli en store jetable et défaut d'activation à chaud différé. Nouvelle répétition justifiée par la variante **release 1.3.1 réellement packagée**, ses versions/empreintes/signatures et la compatibilité d'installation, sans réestimer un gain.
- Protocole préenregistré : commit source propre ; fmt/Clippy strict, build/tests release `--locked --features mlx,chat`, packaging/tests existants ; signatures ad hoc/architectures/minOS/manifestes ; QMM synthétique BM32/BM64 sur le moteur packagé après compilations, 64 cas attendus/1 887 232 mots FP16. Updater Swift de production : archive installée/activée/déplacée/exécutée/rejetée dans un dossier jetable, runtime-info et list sans poids ; sélection des deux canaux sur métadonnées GitHub, taille/SHA-256 de l'asset téléversé puis retéléchargé. CI existante PR/push lue et vérification du commit source exact, dont Kani31 ; publication finale après inspections.
- Conditions : Apple M5/macOS27.2 arm64, SDK26.5, Rust1.98.1, Python3.12.14, MLX0.32.2 ; aucun modèle chargé pour cette livraison, température/débit/gain/RAM **non mesurés**. Kani borné CPU et tests finis ne prouvent pas Metal/MLX/la génération ; parité checkpoint Qwen complet non rejouée, CrossHair antérieur inconclusif. Preuves prévues sous `docs/measurements/engine-v1.3.1/`, rapport `docs/engine-v1.3.1-validation.md`. État : préparation, publication non effectuée.

### FIX-2026-10-05-PR23-03 — deadlines et nettoyage des processus de vérification — validé par tests, poussé

- Livraison précédente : correctif 02 poussé (`c1a3bf3d657d1b9d723ba80b345ba1a13e07340c`), tête GitHub vérifiée. Antécédent REVIEW-PR23 : un bridge silencieux reste vivant et bloque `readline`; l'attente finale de 15 s ne borne pas la génération et `cancelled` est ignoré.
- Changement prévu : helper JSON partagé fondé sur les pipes non bloquants POSIX (Linux/macOS), deadline globale par chargement/requête incluant l'écriture et chaque événement, durées positives finies configurables. Refuser erreur/annulation/EOF/JSON invalide ; fermer, terminer, tuer si nécessaire et rejoindre chaque enfant, y compris si le second codec ne démarre pas. Les rapports de QMM conserveront aussi un statut d'échec explicite.
- Protocole avant implémentation : régression rouge d'annulation du vrai CLI sur la baseline (processus synthétique, timeout du test) ; tests déterministes silence au chargement/requête, réponse partielle et événement continu, stdin fermé/bloqué, EOF/erreur/JSON invalide, SIGTERM ignoré et échec de démarrage du second enfant. Vérifier disparition des processus et succès des parcours précédents. Ruff/compilation/suite Python et E2E Desktop, tentative CrossHair de la validation réelle des durées, CI PR/push Linux/macOS et Kani existant à inspecter/approuver sur la tête finale. Aucun modèle/benchmark GPU ; preuves `docs/measurements/pr23-fixes/fix03-*`. Un commit/push séparé puis bilan de CI.
- Régression rouge : annulation du vrai CLI ne termine pas ; le garde-fou externe du test l'interrompt après 3 s, test en échec `TimeoutExpired` (`fix03-red.log`). Le bridge factice sort à la fermeture de stdin par son parent arrêté. Aucun processus GPU utilisé. Implémentation des délais et de l'annulation suivante.
- Premier lint : 3 diagnostics (directives noqa inutiles et validation du type d'événement), corrigés. Première version compilée ; tests de réussite précédents et nouvelles frontières processus en vérification. Aucun résultat final annoncé.
- Première suite des comparateurs : 41 tests réussis (10,98 s), dont annulation auparavant bloquante (`fix03-initial-tests.log`). Nouveaux tests processus : 3 diagnostics de contextes `with` imbriqués ; `ruff --fix` ne les applique pas automatiquement, donc correction manuelle sans changement de comportement. Ruff/compilation/diff désormais réussis.
- Contrôles des frontières : 85 tests ciblés réussis (24,55 s, `fix03-tests.log`), transports réels et CLI incluant stdout silencieux/partiel, flux d'événements sans fin, stdin bloqué/fermé, EOF/JSON invalide, annulation, SIGTERM ignoré puis SIGKILL/reaping, erreur de démarrage du second enfant. Les fixtures confirment la disparition des PID. Suite complète/E2E et diagnostic CrossHair à inspecter avant livraison.
- CrossHair 0.0.101, validation de durée réelle `positive_seconds`, 40 s/condition/5 s par chemin : aucun contre-exemple, propriété « Not confirmed » (`fix03-crosshair.log`). Aucun modèle symbolique des pipes/signaux ni preuve d'absence de course système. YAML des deux workflows parsé avec PyYAML 6.0.3/BaseLoader : déclencheurs PR/push et permissions contents:read contrôlés ; CI distante reste à lancer sur la tête finale.
- Répétition physique préenregistrée, après fin de l'E2E/compilations : refaire les 64 cas QMM de REVIEW-PR23 avec le **nouveau transport et les contrôles de dimensions**, même binaire debug inchangé SHA-256 `aa8ac02d21605687880629110884a8a50b7ade5358867f8ff3eea661cd9ba22b`, M64=0 puis M64=1 par processus. Nouvelle raison : vérifier le protocole réellement utilisé par les corrections, pas une piste ni mesure de vitesse. Aucun modèle chargé, appels GPU séquentiels, résultats/commandes `fix03-qmm-physical*` ; contrôler indépendamment 1 887 232 valeurs attendues et tous les statuts finaux. État : pas encore lancé.
- Suite complète du source final : 103 tests Python réussis/4 skips inchangés (24,19 s), E2E Desktop encore en cours ; log `fix03-e2e.log`. Précision de portée des recherches CrossHair précédentes : les annotations bornent les types symboliques (QMM : listes de mots/groupes, rows et widths entiers ; échec : dicts/exception ; durée : chaîne argparse). Aucun `assume` supplémentaire, mais les autres types JSON invalides sont couverts par tests et ne sont pas tous dans le domaine symbolique. Les résultats restent « Not confirmed », aucune preuve globale revendiquée.
- E2E entièrement terminé avec code 0 : compilation Python/Swift, Desktop/bridge/CLI transport/streaming/UI/imports/updater et contrôles existants réussis, SDK 26.5/Metal compilation explicitement sautée comme prévu (`MLXL3_SKIP_METAL_CHECK=1`). Les modèles PonyExl3/Ling absents restent non exécutés. Répétition physique QMM suivante, aucune compilation ni recherche symbolique locale concurrente.
- Répétition physique terminée : nouveau script sur le binaire moteur initial inchangé, **64 cas/1 887 232 valeurs FP16 finies, tailles exactes et bits identiques**, statut complete/parity true contrôlés indépendamment. Commandes, environnements, empreinte et plateforme dans `fix03-qmm-physical-metadata.json`, sorties `fix03-qmm-physical.json`/`.log`. Wrappers temporaires supprimés ; leur commande `env MLXL3_DENSE_PREFILL_M64=0|1 <binary> codec` est archivée. Il s'agit d'un contrôle fonctionnel, aucun nouveau débit/RAM modèle mesuré.
- CI CPU étendue : pytest 8.4.2/NumPy 2.5.2/Ruff 0.16.5 épinglés, format/lint/compilation des scripts et 3 suites de régression sous Linux/macOS ; Kani et contrôles Desktop existants conservés. Source final testé et prêt pour son commit/push séparé. Vérification GitHub du SHA livré suivante ; aucun essai de code/benchmark local encore en cours.
- Livraison confirmée : commit `f01511f23253c68b13d82ecca65c0583e84839b0` poussé, tête distante et base vérifiées, checkout propre. À cet instant les runs GitHub 37358798342/37358798470 sont `action_required` (PR externe), pas verts. Les workflows ont été inspectés ; exécution de la CI sur la tête finale après ce bilan documentaire. Résultats publics sur [les checks de la PR 23](https://github.com/0xZKnw/mlxl3/pull/23/checks), sans confondre attente d'approbation et réussite.

### FIX-2026-10-05-PR23-02 — rapport de campagne incomplet ou divergent — validé par tests, poussé

- Livraison précédente : correctif 01 poussé (`7e64de62e19ca258ead8cac77e375a102738407e`), tête GitHub vérifiée. Baseline du constat 02 : le CLI quitte en erreur après divergence B, mais laisse `results.json` avec A seulement et `parity: true`.
- Changement prévu : états explicites running/failed/complete, parité inconnue avant campagne complète et fausse après divergence, conservation immédiate de la passe/prompt/runs partiels et de l'erreur. Préserver les hashes, budgets, ordre ABBA/BAAB et médianes des campagnes réussies.
- Protocole préenregistré : régression CLI rouge de divergence B sur la baseline ; vrais processus synthétiques pour succès ABBA/BAAB, panne au chargement et erreur après un run. Contrôles Python complets/Ruff/compilation ; tentative CrossHair sur la transition d'échec réelle, limites fichiers/processus explicites. CI Linux/macOS étendue. Aucun modèle GPU ni benchmark de vitesse ; preuves `docs/measurements/pr23-fixes/fix02-*`. Un commit/push séparé après inspection.
- Régression rouge avant modification : `python -m pytest -q tests/test_compare_native.py::test_divergent_candidate_is_saved_as_a_failed_pass` échoue (1 échec), car le résultat persistant est `parity: true` malgré la divergence B. Log `fix02-red.log`. Implémentation suivante.
- Premier contrôle : 5 tests CLI réussis (2,29 s), succès ABBA/BAAB et leurs médianes, divergence sauvegardée, panne au chargement, run partiel conservé. Ruff/compilation réussis. Première tentative CrossHair interrompue par le contrat d'immutabilité implicite de l'objet exception lors de son formatage, pas par un contre-exemple sur les états/parité. Le contrat autorise maintenant les effets de l'exception (aucune exigence d'immutabilité dans le protocole), sans filtrer les entrées ; log négatif `fix02-crosshair.log` conservé. Relance et suite complète suivantes.
- Suite Python complète : 58 réussis, 4 skips PonyExl3/Ling inchangés, 10,12 s (`fix02-full-python.log`). Format/Ruff/compilation et diff final réussis. Les snapshots JSON sont remplacés atomiquement après chaque génération ; cette persistance se fait hors des durées moteur, mais de futurs benchmarks doivent identifier ce nouveau script, sans réattribuer les anciennes campagnes. Recherche symbolique finale en cours ; I/O système/fichiers ne sont pas prouvés par CrossHair.
- Clôture : CrossHair final sur `record_failure`, 40 s/condition et 5 s/chemin, sans précondition : aucun contre-exemple, 2 « Not confirmed » (`fix02-crosshair-final.log`). Aucune preuve déductive ni certification I/O revendiquée. Transition de parité/d'état couverte par tests CLI et contrat ; correctif prêt pour son commit/push séparé, CI du SHA exact à suivre.
- Livraison confirmée : `c1a3bf3d657d1b9d723ba80b345ba1a13e07340c`, push séparé puis tête de PR vérifiée avant le correctif 03.

### FIX-2026-10-05-PR23-01 — formes et valeurs des sorties du comparateur QMM — validé par tests, poussé

- Autorisation : correction des constats de revue, un commit/push par correctif. Livraison documentaire `b925641` poussée et SHA de PR vérifié. Baseline Python de cette tête : deux codecs répondant vide font accepter64cas sans valeur (REVIEW-PR23).
- Changement prévu : valider chaque réponse avant comparaison/conversion, un groupe par projection et exactement `rows*width` mots uint16 finis ; refus formes vides/tronquées/extra, bool/float/hors bornes et NaN/Inf. Garder le protocole numérique et les64cas GPU existants, aucun benchmark ni code Rust/Metal changé.
- Protocole avant code : régression CLI rouge sur baseline avec codecs factices vides ; tests déterministes tailles/groupes/finitude/types et exploration exhaustive des65536motsF16 contre `struct.unpack` ; tests d'intégration du vrai script, NumPy/pytest reproductibles en CI. Compilation/Ruff/fullpytest puis CrossHair existant sur la fonction de validation réelle (recherche symbolique, pas preuve). Corriger les échecs et pousser ce seul correctif avec tests/journal. Preuves `docs/measurements/pr23-fixes/fix01-*`.
- Régression rouge exécutée avant correction : `python -m pytest -q tests/test_qmm_tiles.py::test_cli_rejects_empty_codec_outputs` échoue comme attendu (1 échec, 6,29 s). Le vrai CLI annonce 64 cas valides et sort avec le code 0 pour deux sorties vides. Log `docs/measurements/pr23-fixes/fix01-red.log`. Correctif et contrôles suivants ; vitesse non mesurée.
- Premier lint du correctif : 4 diagnostics (contrat d'erreur de JSON invalide et `check=False` explicite dans les tests), corrigés. Régressions et recherche symbolique en cours ; aucun succès de vérification encore annoncé.
- Contrôles ciblés : 35 tests réussis (7,29 s), dont CLI vide désormais refusé, candidat Inf refusé et CLI complet 64 cas/1 887 232 valeurs synthétiques ; exploration exhaustive 65 536 mots FP16 contre le décodeur indépendant `struct`. Ce sont des fixtures CPU, pas une nouvelle mesure GPU. Ruff strict et compilation Python réussis après correction des diagnostics. Suite Python complète et CrossHair suivants.
- Suite Python complète : 53 réussis, 4 skips documentés (PonyExl3/modèle Ling absents), 8,54 s, log `fix01-full-python.log`. CrossHair 0.0.101 sur les deux postconditions de la fonction réelle, sans précondition excluant les entrées invalides : aucun contre-exemple trouvé mais deux résultats « Not confirmed » dans la limite de 40 s/condition et 5 s/chemin. Propriétés non prouvées ; relance identique sur le source final après correction du lint, pas une nouvelle hypothèse ni benchmark.
- Clôture locale : source final Ruff 0.16.5 check/format, `py_compile` et `git diff --check` réussis ; recherche CrossHair finale identique, 2 « Not confirmed », `fix01-crosshair-final.log`. Python 3.12.14/NumPy 2.5.2 ; CI Linux/macOS installe NumPy 2.5.2/pytest 8.4.2 et sélectionne les nouveaux tests. Aucun Rust/Metal changé : Kani/GPU de la revue initiale restent les preuves moteur, sans prétendre prouver ces scripts. Correctif prêt pour son commit/push séparé ; statut GitHub à vérifier après livraison.
- Livraison confirmée : `7e64de62e19ca258ead8cac77e375a102738407e`, push séparé puis tête de PR vérifiée avant le correctif 02.

### DELIVERY-2026-10-05-PR23 — publication de la revue et corrections séparées — livrée, bilan CI externe

- Autorisation explicite : « push déjà ce que t'as fait dans la PR23 et fixe ces trucs ; à chaque truc fixé, un push ». Branche distante observée `HENK0O/mlxl3:optimize/qwen27-m5`, tête initiale `0e6dc2f9bafcb2586f19f8e480555fddea41fc53`. Réutiliser le checkout de revue, préserver les changements locaux MTP du checkout principal, aucun force-push ni merge/release/app installée.
- Première livraison : consignes AGENTS renforcées, rapport et preuves indépendantes de revue déjà exécutées, journal de revue clos. Contrôle documentaire/diff et inspection du contenu avant commit/push. Trois correctifs Python ensuite, un commit testé et un push par défaut corrigé ; les tentatives/échecs et limites seront consignés séparément.
- Aucun nouveau benchmark prévu ni gain moteur revendiqué ; les contrôles numériques/performances de la revue restent datés et limités à la tête initiale. Les scripts et frontières processus feront l'objet de fixtures déterministes, régressions rouge/verte, lint/compilation Python et vérification source pratique ; CI PR/push à compléter et runs du SHA exact à inspecter.
- Livraison documentaire effectuée : commit `b9256411efc450e18599cda1efee2bfaf8ad5f84` poussé sur la branche de PR, tête distante vérifiée. Preuves de la revue `docs/measurements/pr23-review/`, corrections suivantes `docs/measurements/pr23-fixes/`.
- Trois corrections livrées chacune par son commit/push, dans l'ordre `7e64de6`, `c1a3bf3`, `f01511f`. Toutes les régressions rouges/vertes, limites source et preuves conservées. 103 tests Python/4 skips, E2E complet et nouveau comparateur GPU 64 cas/1 887 232 valeurs passent ; aucun Rust/Metal touché par ces corrections. Aucun essai local en cours, modifications MTP du checkout principal préservées. Ce bilan documentaire est poussé ensuite ; les résultats GitHub du SHA final seront inspectés dans les checks publics de la PR. Pas de merge/release/app installée.

### REVIEW-2026-10-05-PR23 — revue indépendante des changements HENK0O-QWEN27 — terminée, trois constats d'outillage

- Demande : revue complète de la PR23, tête `0e6dc2f9bafcb2586f19f8e480555fddea41fc53`, base `1ac910d0a40efcad049c6e5a8cf081d770723465`. Checkout isolé ; aucun code moteur modifié, aucun push/commentaire GitHub/installation/publication.
- Antécédents lus : essais HENK0O-QWEN27-01..06 et FIX-04, Qwen dense RUST-05, LOAD-06, OPT-2026-09-11-03, rapports `docs/qwen27-m5-optimization.md`, `docs/qwen38-decode-local.md` et historique prefill/allocateur. Répétition de validation seulement, aucune nouvelle piste ni benchmark de débit.
- Hypothèse de contrôle : vérifier indépendamment compilation/tests et nouvelle garde BM64 ; examiner erreurs/bornes/I/O des scripts et cohérence des preuves numériques/mémoire/performance. Rust1.98.1/Kani0.68.0/CBMC6.11.0 et MLX0.32.2 locaux, Python3.12.14. Conditions de débit/RAM modèle : non mesurées.
- Protocole : fmt, Clippy strict MLX/chat, build/tests Rust, Python/Ruff et E2E Desktop si toolchain compatible ; Kani nouvelle propriété puis suite CPU existante avec timeout par harnais ; tests GPU ciblés de lecture/cache après fin des compilations/provers, uniquement si aucun modèle concurrent. Les données de modèle d'origine et références ~1,5Go sont absentes de ce checkout ; ne pas refaire silencieusement la campagne. Reproductions script déterministes avec processus factices et données jetables, pas de benchmarks. Preuves sous `docs/measurements/pr23-review/`, rapport `docs/review-pr23-2026-10-05.md`.
- CI distante : deux workflows de la PR ont `conclusion=action_required`, zéro job ; GitHub attend une approbation de PR externe. Pas un succès CI, aucun réglage modifié.
- Contrôles intermédiaires : fmt/Clippy strict MLX+chat/build et suite Rust réussis (38lib+7bin+14contrats=59,53ignorés explicites). Nouvelle propriété Kani :61obligations,0échec,2couvertures satisfaites ; tous `i32` symboliques/bool, aucun `assume`, uniquement sélecteurCPU, pas calculMetal. Python18pass/4skip (PonyExl3 et modèleLingabsents), Ruffcheck/format réussis. Desktop et suiteKani existante encore en cours. Premier script d'audit des mesures arrêté sur un champ`ready` absent dans un ancien schéma ; audit corrigé réussi sur5campagnes (passes/hashes/budgets/médianes/synthèses cohérents), ce n'est pas un nouveau benchmark.
- Reproductions des échecs des scripts terminées, fixtures jetables sansMLX : divergence A/B détectée avecexit1, mais `results.json` conserve `parity:true` et seulementA, sans traceB ; deuxcodecs factices répondant des sortiesvides font accepter64cas/0valeur par `check_qmm_tiles.py` ; bridgesilencieux bloque la lecture jusqu'au watchdog externe3s (processusfactices tués/rejoints). Preuves `script-fault-reproductions.json`, `divergence-stale-results.json`. Ces défauts concernent la robustesse/validité de l'outillage ; aucune divergence moteur ni invalidité des64mesures fournies établie. Corrections à proposer dans la revue, aucun codeexécutable dePRmodifié.
- SuiteKani complète terminée :31/31harnais réussis,aucuntimeout/contre-exemple ; limite120s/harnais, bornes individuelles existantes conservées. DesktopE2E réussi avecSDK26.5 et`MLXL3_SKIP_METAL_CHECK=1` (testsMetal physiques séparés),208,252s. Tentative de deuxanciens testsPython mentionnés dansnative/README arrêtée car fichiersabsents deHEAD, zérotestattribué ; aucun changementPRne les supprime. Touscompilateurs/provers arrêtés avant contrôlesGPU ciblés de lecture/cache et répétition des64casBM32/BM64 (mêmebinairededebug, options0/1explicites, aucunmodèlechargé, contrôle supplémentaire du nombre/formes de sorties pour éviter la validationvide reproduite).
- ContrôlesGPU terminés/réussis : lecture>INT_MAX simple/batch, frontières64Mio/offset/troncature/overflow ; clear_cache32Mio/tableauxvivants ; smokelecture/kernel. Chaque filtreexact a exécuté1test. Répétition64casQMMBM32/BM64 réussie26,707s, toutesvaleursFP16 finies/exactes aveccontrôle indépendant `M*sum(widths)>0` ; aucune divergence numérique primitive. Ce sont des tests, aucun nouveau débit/RAM modèle mesuré. AucunprocessusGPU/prover conservé. Preuves `gpu-checks.json`, `qmm-physical-parity.json`, `verification-summary.json`.
- Clôture : 31 harnais Kani, 4 536 obligations sans échec (70 checks inatteignables déclarés dans les harnais existants), 71/71 couvertures satisfaites. 64 cas QMM : 1 887 232 valeurs F16 exactes/finies et de taille correcte. Trois constats P2 sur sorties vides, rapport partiel trompeur après divergence et lectures sans délai ; aucune régression moteur identifiée dans le domaine testé. Parité modèle de l'auteur non rejouée (références temporaires/checkpoint exact non fournis), autres tests GPU et quatre skips Python explicitement non exécutés. Gain mémoire des preuves cohérent ; gain BM64 conditionnel, forte dérive, aucune nouvelle promesse chiffrée. Rapport [review-pr23-2026-10-05.md](docs/review-pr23-2026-10-05.md). CI toujours `action_required` au dernier contrôle, SHA inchangés. Consignes `AGENTS.md` renforcées à la demande de l'utilisateur dans le checkout principal ; changement documentaire seulement. Revue et preuves locales, aucun commentaire/push/merge/installation/publication. Aucun essai de cette revue en cours.

### OPT-2026-10-05-HENK0O-QWEN27-01 — audit du checkpoint dense K2 et baseline M5 — modèle local validé

- Demande : vérifier et optimiser Qwen3.8-27B-exl3 sur le fork HENK0O/mlxl3, puis laisser l'utilisateur créer sa PR. Baseline fork `1ac910d`; travail sur une branche dédiée, sans publication de release ni modification de l'app installée.
- Antécédents consultés : index et synthèses du journal, Qwen dense RUST-05, `docs/qwen38-decode-local.md`, expériences MTP-05 et PERF-135/136. Lire les rapports decode/EXL3 pertinents avant chaque piste; ne pas répéter leurs rejets sans nouvelle forme/mécanisme explicite.
- Nouveau checkpoint local : `~/Library/Application Support/io.mlxl3.desktop/Models/Qwen3.8-27B-exl3-6a9ca9d0`, architecture qwen3_5_text dense, 64 couches, hidden5120/intermediate17408, 48 value heads GDN, MUL1 K2 et head K3 (différent du checkpoint historique 2.75bpw). Vérifier les métadonnées des shards et les types/tailles réelles avant mesure.
- Environnement : MacBook Air M5 GPU10, mémoire24Go, macOS27.2/26B5091g. MLX0.32.2 disponible dans le cache local; Rust absent du PATH. Alimentation et thermique à relever; aucune température supposée.
- Protocole : établir un environnement de build isolé dans le workspace, inspecter les dispatchs et formes réelles; tests de parité physique avant promotion, puis baseline et candidat alternés avec warmup, sorties exactes et mémoire distincte de la RAM processus. Un seul modèle GPU actif, aucun build pendant une mesure. Préenregistrer chaque variante séparément. Preuves sous `docs/measurements/qwen27-m5/` et rapport dédié.
- État : inspection et préparation seulement; nouvelles performances non mesurées. Échec réseau sandbox du premier ls-remote, clone hors sandbox réussi. Aucun kernel modifié, modèle non chargé par cette tâche, app non installée, aucun push/PR/release.
- Première baseline **échouée au chargement**, `pread failed: Invalid argument`, aucune génération/timing. Inventaire :3080tenseurs,9 672 422 388octets sérialisés, bits réels1/2/3/4; l'embedding F16 contient2 542 796 800octets et dépasse la limite Darwin d'une lecture. Traces `baseline-initial.stderr`, `checkpoint-summary.json` (inventaire complet archivé temporairement dans `../work/checkpoint-inspect.json`). Correction chargement04 requise avant toute comparaison.

### FIX-2026-10-05-HENK0O-QWEN27-04 — lectures checkpoint supérieures à2Gio sur Darwin — validé

- Reproduction : baseline1ac910d/MLX0.32.2 refuse le checkpoint local avant ready. `pread_exact` demande `count-done` sans borne à Darwin; embedding `[248320,5120]` F16 de2 542 796 800octets. Les tenseurs/shards passent l'inspection, pas une corruption de header.
- Changement prévu : borner chaque syscall pread à64Mio, tout en écrivant directement dans le buffer MLX, conserver retryEINTR/lectures partielles/troncature/overflow. Ajouter contexte du nom de tenseur à l'erreur de chargement.
- Protocole : régression GPU avec fichier sparse>INT_MAX et marqueurs aux limites/début/fin, contrôles existants de lecture décalée et groupée, puis chargement et génération du modèle réel. Pas de changement aux poids ni à l'arithmétique. La première baseline de performance sera ce chargement corrigé, distincte du fork incapable de charger.
- Statut : avant correctif; vitesse non mesurée. Preuves sous `docs/measurements/qwen27-m5/`.
- Régression rouge reproduite : `checkpoint_reads_larger_than_darwin_syscall_limit` échoue en0,03s avec EINVAL avant correctif (`load-red.log`). `man2read` local confirme nbyte>INT_MAX/EINVAL. Correctif64Mio appliqué; régression verte puis modèle réel suivants.
- Régression verte réussie : lecture simple et batch d'un fichier sparse2 147 483 652octets à offset3, marqueurs de part et d'autre de64Mio, début/fin, overflow/troncature rejetés (`load-green.log`,0,60s). SmokeGPU lecture décalée/groupée réussi (`array-smoke.log`,0,51s). Build référence corrigée réussi et archivé `mlxl3-load-fixed`; référence modèle23/128/256tokens+3decode avec128états et tous logits en cours. Durées de tests≠mesures de performance.

### OPT-2026-10-05-HENK0O-QWEN27-02 — fenêtre unique de huit états QMV K2 — rejeté, retiré

- Hypothèse : K2 nécessite seulement30bits pour huit états16bits séparés de2bits. Remplacer les deux fenêtres de quatre états par une seule fenêtre, dans les QMV dense et groupé; conserver FMA, codebook, SG, tiles, split-K et ordre de réduction.
- Antécédents : `docs/qwen38-decode-local.md` rejette la fenêtre K3 groupée sur2.75bpw; K3 expert existe déjà. Cette piste vise K2/30bits, absent du dispatch actuel, sur un nouveau checkpoint principalement K2 (quelques K1/3/4 conservés).
- Baseline : fork1ac910d construit Rust1.99.0/MLX0.32.2, binaire `../work/binaries/mlxl3-baseline`. M5 GPU10/24Go/macOS27.2. Alimentation indique AC mais batterie67% en décharge; `pmset therm` indisponible, température non mesurée.
- Protocole : matrices déterministes sur les trois codebooks, K2, SG4/8, plusieurs NT, split-K, queues batch et routage; comparer toutes sorties FP32/FP16 en bits contre shader historique, puis tous logits/états modèle après prefill et decode imposé. Microbench alterné uniquement après exactitude; A/B/B/A réel sur deux prompts, warmup exclu, 32 ou64tokens, cacheOFF, contexte4096. Conserver seulement si baisse mesurée≥3% reproductible; sinon retirer. Aucun prover/build simultané aux mesures.
- Statut : avant prototype, gain non mesuré, modèle référence en préparation. Pas d'installation/publication.
- Premier harnaisGPU interrompu à la compilation : newline terminal absent du header Python, déclaration kernel absorbée par le dernier define; aucune sortie/timing validés (`k2-kernels-initial-error.log`). Header corrigé et fixture expert uniformisée pour garder les routes dans les bornes. TestCPU indépendant16 384positions bit-basis/ringwrap réussi. Relance du même filtre physique prévue, pas gain mesuré.
- Filtre initial réussi :162cas FP32 exacts. Microprofilage20paires synchronisées donne dense-down0,6765→0,9381ms (régression), group-gate/up1,2539→1,2182ms (faible), group-qkv/z0,9381→0,6802ms (signal). Variante64bits non promue. Révision préenregistrée : ne garder que la fenêtre uint32 de30bits vivante plutôt que ulong+shift, même extraction/FMA. Nouveau filtre et microprofilage identique avant sélection des formes; gains E2E toujours non mesurés.
- Varianteuint32 :162cas encore exacts, mais microprofilage alterné dense0,8737→0,9520ms, gate/up0,9745→1,0284ms, qkv/z0,6505→0,6735ms, soit régressions de3,4..8,2% (`k2-kernels-v2.json`). Décision **rejeté**, aucun test modèle ni gainE2E revendiqué, shaders restaurés exactement au fork. Prototype/driver archivés dans `../work/k2-experiment/` (temporaires); aucune optimisation K2 de cet essai conservée dans le moteur. Les mesures initiales contrastées ne justifient pas une activation.

### OPT-2026-10-05-HENK0O-QWEN27-03 — BM64 TensorOps dense natif — validé sur formes M5 ciblées

- Hypothèse : le Rust utilise toujours BM32, bien que BM64 dense ait été validé dans l'ancien moteur Python. Porter la sélection conservatrice MUL1/entrée≥4096/sortie<65536/M≥128 et M multiple64, BK16/BN32 inchangés; garder BM32 pour les autres formes et ragged.
- Antécédents : `docs/qwen38-decode-local.md` (BM64), OPT-2026-09-11-03 (extension non concluante), PERF-12/13 (port QMM Rust). Nouvelle raison : chemin natif BM32 et checkpoint K2 dense, pas réexécution de l'extension rejetée aux petites entrées/autres codebooks.
- Protocole : sorties exactes BM32/BM64 sur K1..8, shapes/poids groupés stridés et queues gardées; logits et caches complets modèle avec prefill128/256 puis decode imposé, puis mesures préfill sur prompt court et document non répétitif. Même hardware/dépendances que02, preuves `docs/measurements/qwen27-m5/`. Mesurer séparément de02 avant résultat combiné; aucun gain additionné.
- Statut : avant prototype, gain non mesuré; garder uniquement les variantes validées, pas de publication.
- Candidat BM64 **parité modèle réussie** : préfills23/128/256 puis3tokens imposés chacun,248320logits finis et128états récurrents/KV égaux en bits à la référence sur les12étapes (`bm64-states.log`). Construction candidate réussie; validation primitive K1..8/3codebooks/strides/ragged suivante. Mesures ABBA prévues ensuite, prompt français69tokens et extrait1800caractères du rapport runtime (fichier exact archivé),48tokens générés, un warmup par prompt puis2répétitions par passage. Comparaison contre chargement corrigé04 uniquement, K2 restauré.
- Contrôle primitif interrompu **dans le harnais**, après48cas denses exacts : conversion NumPy d’un bundle ragged `[128,256]` refusée (ValueError), pas de divergence kernel. Log négatif archivé `qmm-parity-initial-error.log`. Aplatir séparément les sorties groupées pour le contrôle de finitude puis relancer les64cas complets, suivi du protocoleABBA inchangé; aucune mesure ABBA encore exécutée.

- Contrôle primitif corrigé **réussi** :64cas natifs, K1..8/3codebooks, groupés stridés et fallbacks, toutes sorties F16 finies/égales en bits (`qmm-parity.json`, `qmm-parity.log`). ABBA modèle démarré, première passe de référence court/document terminée; décision à la fin des4passes.

- ABBA03 **terminé**, textes/tokens identiques, cache0 : document508tokens préfill médian des passages75,14→80,04tok/s (+6,52%), TTFT6,958→6,452s (−7,26%); temps complet14,989→15,026s (+0,24%, non concluant). Court69tokens complet9,223→9,233s (+0,10%). Forte dérive du contrôle decode inchangé7,97→4,83tok/s et batterie61→56%, thermique sans niveau déclaré/température non mesurée : ne pas attribuer les variations de decode au kernel. Signal préfill à confirmer, pas de boost global revendiqué. Preuves `bm64-abba/results.json`, binaire conservé provisoirement pour05, pas de publication.

- Confirmation03 BAAB **terminée/validée préfill local** : document508tokens 80.69→93.87tok/s (+16.34%), TTFT6.332→5.466s (-13.68%), complet-12.42%. Court69tokens complet+15.32%; forte dérive du contrôledecode demeurant. Comparaison des deuxB aux Aadjacents après normalisation descriptive par le tempspréfillcourt inchangé donne un signal préfill dans chacunB; cela ne remplace pas une mesure de fréquences. Tokens/textes exacts. RetenirBM64 sur les formes conservatrices, aucun boostdecodegénéral revendiqué. Source/changements numériquement validés, preuves `bm64-baab-confirm/results.json`. Gain conditionnelM5/document, pas universel ni somme avec03initial.


### OPT-2026-10-05-HENK0O-QWEN27-05 — résidence MLX bornée pour le checkpoint dense — rejeté, retiré

- Hypothèse : le modèle dense relit plusieurs Go à chaque token; le moteur laisse le budget de résidence MLX à0. Reprendre la piste PERF-136, jamais implémentée/mesurée, avec le nouveau checkpoint local9,67Go. Source primaire MLX0.32.2 `mlx/backend/metal/allocator.cpp` et `memory.h` consultée : setter disponible, limite recommandée Metal à respecter. Ne pas modifier les paramètres système.
- Changement prévu : garde limitée à la durée du bridge natif, acquise avant lecture des poids et restaurée sur retour/erreur. Budget=poids+512Mio borné par la limite Metal et RAM totale moins max(25%,4Gio); désactiver si les poids ne tiennent pas. Mutex de garde pour interdire des propriétaires concurrents, voie de repli sans résidence si erreur, option `MLXL3_WIRED_MEMORY=0`. Aucun changement arithmétique. Tests CPU des bornes et garde/restauration/erreurs sur GPU avant mesure.
- Baseline : binaire BM64 validé numériquement, sa campagne03 en cours; compiler seulement après fin du benchmark. Vérifier les logits/128états contre la même référence puis ABBA binaireBM64 / candidatrésidence, prompts et48tokens identiques, warmup par prompt,2répétitions par passage, cacheOFF/contexte4096. Relever alimentation/swap/thermique/mémoires; un seul modèle actif et aucun build/prover pendant la mesure. Retenir un gain uniquement≥3% reproduit, sinon retirer le prototype. Ne pas additionner deux campagnes.
- Statut : prototype local écrit, benchmark03 terminé; compiler et vérifier les bornes/garde/parité avant toute mesure05. Budget effectif et gain05 encore non mesurés, pas de publication ni installation.
- Révision du protocole05 **avant mesures** : ajouter25secondes de repos GPU avant chaque chargement pour limiter la dérive constatée03; garder48tokens/2répétitions/prompts inchangés. Un temps d’attente n’est pas une mesure thermique. Ajouter un contrôle final contre le chargement corrigé si05 est retenu; confirmation03 avec génération courte et ordreBAAB préenregistrée après05.
- Première compilation05 arrêtée sur un nom de module non qualifié dans le nouveau test (`array::` au lieu de `crate::array::`), avant toute exécutionGPU/mesure. Trace `wired-build-initial-error.log`; nom corrigé et même séquence de validation relancée.

- Validation05 **réussie** : testCPU bornes/overflow, gardeGPU succès/retourerreur/secondpropriétaire/limiteinvalide/restauration (0,06s), modèle12étapes avec248320logits et128états exacts (`wired-cpu.log`, `wired-guard.log`, `wired-states.log`). Build release archivé `mlxl3-wired`; démarrageABBA avec repos25s. Durées des contrôles≠mesures de vitesse.

- Revue05 pendant la mesure : ajouter une synchronisationGPU avant restitution de résidence, y compris sur échec de chargement; tenter la restitution même si la synchronisation échoue. Cette révision ne modifie pas la génération ni les métriques chronométrées, seulement la sortie du scope. Recompiler et répéter la gardeGPU après finABBA; les empreintes de cette campagne restent celles du binaire archivé précédent.

- ABBA05 **terminé**, sorties exactes, budget10233576045octets dans les deuxB : court complet+6,61% et document+13,28%, pas de bénéfice mémoire. TTFTdocument+10,14%, decode−17,09%; dérive des contrôles persiste, donc pas de causalité universelle de régression, mais aucun gain robuste justifiant activation. Décision **rejeté**, retirer toute la garde/FFI/options05 du moteur; snapshot expérimental temporaire `../work/wired-experiment/ending-prototype.patch`, binaire de mesure archivé, logs bruts `wired-abba/results.json`. La révision de synchronisation de sortie n'a pas été recompilée/testée car le prototype entier est retiré, elle n'est pas livrée.


### OPT-2026-10-05-HENK0O-QWEN27-06 — libération unique des temporaires de chargement BF16 — validé mémoire, extension anticipée rejetée

- Correction factuelle de01/04 : l'embedding sérialisé est **BF16**, pasF16; même taille2 542 796 800octets et même dépassementINT_MAX. Le moteur convertit enF16 dans `half_weight`, déjà évalué au chargement. Inventaire compact `checkpoint-summary.json`; données d'inspection complètes conservées temporairement dans le workspace. Les anciennes mentionsF16 désignaient à tort le dtype du fichier.
- Observation nouvelle : après warmup, actifMLX9 334 954 876octets contre cache3 065 620 213octets et empreinte12,53Go (`wired-abba/0-A.json`). `half_weight` évalue bien la conversion; le gros buffer source peut rester dans le cache allocateur. Ce cache est distinct des étatsKV/GDN et n'améliore pas la réutilisation des prompts. Historique `docs/tool-prefill-rd-2026-09-05.md` essais11/12 lu : conservation allocateur en boucle non concluante surMoE; ici nettoyage **une fois après chargement**, pas à chaque requête, nouveau checkpoint dense BF162,54Go.
- Hypothèse/changement : exposer MLXclear_cache et vider les allocations libres une seule fois avant ready, uniquement si un embedding nonF16≥64Mio a été converti; tableaux des poids/caches de modèle restent détenus. Option `MLXL3_RETAIN_LOAD_CACHE=1` pour repli. Rien au chemin chaud ni à l'arithmétique.
- Baseline : binaireBM64 sans résidence (05 encore en mesure, décision séparée). Tester clear_cache garde les tableaux vivants, puis12étapeslogits/128états contre la référence; comparer mémoire avantready/aprèsprompt court/document. ABBA génération8tokens,2répétitions etpause25s pour limiter le tempsGPU continu; chiffres de48tokens antérieurs non comparables. ConfirmationBM64 BAAB avec8tokens/pause25s également préenregistrée, en isolantBM64 du nettoyage/résidence retenus. Rejeter si sorties diffèrent ou régression stable; retenir une baisse mémoire reproduite, sans la qualifier de boost vitesse.
- Statut : avant prototype, économieRAM réelle et performance non mesurées. Pas de publication ni installation.

- Premier test06 échoué avantappelclear_cache : fixturezeros32Mio restée un broadcastscalaire MLX, cache<32Mio. Ce résultat ne teste pas la libération et ne valide aucune économie. Trace `cache-fixture-initial-error.log`; remplacer la fixture par un vrai bufferF32 chargé32Mio et relancer le même contrôle, puis parité modèle/build.

- Validation06 **réussie** : testGPU buffer32Mio réellement alloué/libéré, cache0 et tableaux vivants conservés; modèle12étapeslogits/128états exacts (9,75s, pas un débit), buildrelease archivé `mlxl3-cache`. Le comparateur reçoit maintenant des overridesMLXL3 explicites : pour06, mêmebinairedeuxcôtés, A=`MLXL3_RETAIN_LOAD_CACHE=1`, B=normal; pour confirmation03, A=`MLXL3_DENSE_PREFILL_M64=0`, B=normal, nettoyage activé danslesdeux. Aucun mélange de changements :8tokens/pause25s/2répétitions etprompts identiques, ABBA06 puisBAAB03.

- ABBA06 **terminé/validé mémoire**, textes/tokens exacts : short empreintephysique12.496→9.844Go (−2.652Go); document empreintephysique13.953→11.285Go (−2.668Go). Actif/picMLX inchangés; baisse dans le cache libre. Complet document+0.60%, court-0.64%; dérive de contrôle, aucun boost vitesse attribué. Nettoyage unique retenu, preuves `cache-abba/results.json`. ConfirmationBAAB03 suivante avec le mêmebinaireet8tokens, nettoyage identiqueA/B.

- Révision06B **avant code/mesure** : le nettoyage àready laisse encore un picprocessus de chargement≈11,98Go (`cache-memory-summary.json`), car la sourceBF16 libérée par `half_weight` attend la fin des64couches. Vider aussi les buffers libres immédiatement après retour de cette conversion, dans le loaderQwen et seulement si la sourceembedding nonF16≥64Mio. Même option de repli, aucun nettoyage en génération. Le nettoyage final existant reste pour les autres temporaires. Hypothèse : réduire le pic de chargement, pas uniquement l’empreinte aprèswarmup. Baseline : candidat06A archivé `mlxl3-cache`; vérifier à nouveau logits/états et comparer ABBA8tokens/pause25s mêmebinaireavec/sans les deux nettoyages, picprocessusdepuislancement distinct des autres mémoires. Pas de gain06B présumé ni addition des économies06A/06B. Compiler uniquement après fin de confirmation03 en cours.

- Validation06B **parité réussie** :12étapes de référence, touslogits/128états exacts avec libération anticipée; buildrelease final réussi et archivé `mlxl3-final`. ABBA mémoire final suivant, aucune compilation pendant ses passages.

- ABBA06B **terminé, extension anticipée rejetée** : sorties exactes, piccourtB11.987Go, pratiquement celui06A11,984Go; aucune réduction supplémentaire du pic de chargement établie. Nettoyage anticipé supprimé, seul le nettoyage final06A validé est retenu. Completdocument-5.31% avec contrôles dérivants, pas de gain causal revendiqué. Preuves `cache-early-abba/results.json` et `cache-early-states.log`; source06A restaurée avant contrôles finaux. Tous essaisGPU terminés, aucune poursuite de calcul spéculatif.

- Contrôles finaux du code06A retenu **réussis** : cargo fmt, Clippy strict alltargets MLX/chat, Rust release59tests réussis/53testsGPUoufixtures ignorés, buildreleaseMLX/chat, Ruffcheck/format des deuxharnais,16testsPython bridge/packaging, gitdiffcheck. Preuves `*-final.log` et `build-retained-final.log`. TestsGPU ciblés exécutés séparément; la suiteignored entière et Python/Desktop complète ne sont pas relancées localement (sourcesDesktop/Pythonproduction inchangées). Kani n'est pas installé localement; nouvelle propriété sera vérifiée par la CI du fork. Aucun processusmodèleGPU restant de cette campagne. Préparation de deuxcommits reviewables, puis push de la branche; pas de PR/release.

- PréparationGit : retirer uniquement les lignes vides terminales de13logs stdout/stderr pour le contrôle whitespace, contenu numérique/erreurs conservé; originaux bruts archivés temporairement `../work/raw-proof-logs/`. Aucun code ni mesure modifiés.

- Livraison : commits `d628e0b` (lectureDarwin) et `408905a` (optimisations/preuves) poussés sur [HENK0O/optimize/qwen27-m5](https://github.com/HENK0O/mlxl3/tree/optimize/qwen27-m5), empreinte distante408905ab3284704b0be94eb2117ab45102526bc8 confirmée. [Rapport](docs/qwen27-m5-optimization.md). API GitHub au contrôle : Actionsenabled=true mais0workflows/0runs pour ce commit; cause non déterminée, réglages du fork conservés. Kani/suiteCI distante **non exécutés**, ne pas les déclarer verts. Contrôles locaux etGPU ci-dessus réussis. Aucun PR/release/appinstallé; l'utilisateur créera sa PR. Étatfinal : lecture/BM64/nettoyage06A intégrés sur branche;02/05/06B rejetés et retirés; aucun benchmark actif.


### OPT-2026-10-05-MTP-05 — petits batches EXL3 et chaîne GPU sans synchronisations — validé fonctionnellement, publié

- Nouvelle demande : améliorer MTP2/MTP3 après MTP-04. Historique PERF-101/107/108/109 et routeur PERF-59/61 relu ; les gains anciens ne sont pas additionnés. MTPLX `9882703f3105363ddc37eca9f97aa09a1d387112`, `verify_kernels.py`, `qwen_row_owned_router.py`, `moe_packed_projections.py` et génération relus. Ses kernels affine/BF16 ne peuvent remplacer directement notre cible EXL3/FP16.
- Cause source supplémentaire : `AffineLinear::gather` télécharge les indices trois fois dans chaque pas draft (gate/up/down), interrompant la chaîne GPU malgré le feedback paresseux MTP-04. Variante A prévue : routes opaques produites exclusivement par notre top-k, nombre d'experts vérifié contre les poids ; garder le gather public contrôlé pour les entrées externes, supprimer seulement ces synchronisations internes. Parité indépendante de la tête et états/tokens D1/2/3 obligatoires.
- Variante B prévue, indépendante : partage MB=2 des poids EXL3 sur M=2/3/4, actuellement limité à M>=5/6. Pour le bundle groupé M=3, ajouter chargement nul/écriture interdite sur la quatrième ligne inexistante et grille arrondie ; conserver FMA/réductions par sortie. Cible MTP2=M3 et MTP3=M4 ; kernels/sources déjà présents, nouvelles formes et nouvelle queue motivent cet essai.
- Piste C : routeur SIMD hiérarchique MTPLX (8 gagnants locaux par SIMDgroup puis fusion64→8) contre le classement quadratique256×256 actuel ; adaptation obligatoire de l'ordre/ties/NaN/FP16, pas encore prototype. Ne l'essayer qu'avec une entrée préalable distincte et une raison de profilage.
- Baseline : binaire release MTP-04 archivé avant modifications, source main d0160f4 + travail local1.3.0, Qwen3.6 EXL3 2.49bpw/tête affine4/group64, M5/MLX0.32.2/contexte4096. Débits indicatifs précédents baseline51,037/D1 63,443/D2 63,522/D3 60,042tok/s ; nouvelles mesures **non mesurées**. Protocole : contrôles exacts avant vitesse, comparaison courte A/B/B/A par le tuner de production (32 warmup par mode,96tokens×2prompts, ordre inversé), un seul modèle GPU et aucune compilation concurrente. Conditions relevées, hashes/acceptations/temps bruts conservés ; stopper si divergence/régression. Preuves `docs/measurements/mtp-05/`. Aucun gain universel présumé ; publication1.3.0 suspendue pendant cette vérification.
- Premier contrôle A terminé : tuner réel complet et résultat brut conservé sous `mtp-05/a-initial/0-baseline-result.json`. Le harnais Python a ensuite échoué dans son résumé (`accepted` au lieu du champ réel `accepted_tokens`) ; aucune mesure perdue, aucune relance silencieuse. Résumé corrigé avant variantes. Variante A maintenant locale (routes opaques, gather public conservé), compilation/parité suivantes. Le fmt final MTP-04 avait refusé une mise en forme de l'assertion budget0 ; formatage mécanique inclus dans la prochaine commande.
- Baseline initiale : 50,937/62,029/62,615/59,693tok/s pour baseline/D1/D2/D3, acceptations89/99,117/142,129/176, mêmes deux hashes MTP-04. Révision du relevé recopié : le fichier brut indique batterie97% en décharge et swap3316,38Mio ; les valeurs100%/3632,39Mio inscrites initialement dans ce résumé étaient incorrectes. Aucune température mesurée. Première invocation du nouveau test gather avec `--exact` et nom non qualifié a sélectionné0test ; elle ne valide rien. Relance sans ce filtre : **1test exécuté/réussi**, sorties opaque/publique exactes, expertcount/type/indices externes invalides refusés. SessionsD1..3,24blocs/80états/KV/résidus/tokens exacts **réussis** en7,93s. Référence indépendante trois résidus **réussie**, max_abs0 ; buildA réussi. Après première baselineA, suite B/B/A avec binaires archivés en cours ; pas de gain annoncé avant résultat.
- Variante A : A/B/B/A terminé, sorties et acceptations identiques. Contrôles D2 62,615/61,703 et D3 59,693/56,713 ; candidats D2 61,374/62,422 et D3 57,479/59,175tok/s. Baseline normale dérive50,937→48,776→50,335→47,998tok/s, plages recouvrantes : **non concluant pour la vitesse**, aucun gain causal revendiqué. Synchronisations internes supprimées et contraintes de routes vérifiées ; changement local retenu pour poursuivre B, qui sera comparé à ce binaireA. PisteB maintenant implémentée : MB2 petits batches, queue impaire gardée dans le shader groupé, contrat de grille CPU partagé/propriété Kani et différentiels GPU élargis M2..8/K1..8. Validation avant mesure, aucune compilation concurrente avec le tuner.
- Première commande B : test portable refusé immédiatement car `MLXL3_DISABLE_TENSOR_OPS=1` omis ; zéro kernel/timing validé. Relance avec le fallback explicitement activé, puis tests cible complets avec configuration GPU normale. Log négatif `b-portable.log` conservé.
- VarianteB **parité validée** : test portable K1..8, lignes1..8/23/24/25/46/47/256/513 et bundles petits/queues réussit7,90s ; rollback total2..4 et tous préfixes retenus sur prompts1/23/24/129/256 réussit15,59s ; sessions24blocs avec tokens/80états/résidu/KV exacts réussit7,01s. Ces durées sont des contrôles, pas un débit. BuildB réussi ; matrice A/B/B/A contre binaireA (sans synchronisations) suivante, aucune compilation/prover pendant ses mesures.
- VarianteB terminée : A/B/B/A D2 **63,781/60,796/58,603/62,253tok/s**, D3 **60,443/62,814/62,110/58,698tok/s**, tous hashes/acceptations identiques. M3 apparié avec une ligne rembourrée **rejeté** : D2 perd4,7..5,9% aux voisins ; M2/M4 signal positif mais confirmation finale requise. Baselines normales50,491/50,349/49,595/47,481, batterie94% première passe et swap3678,94Mio, températures non mesurées.
- VarianteB2 avant essai : retirer MB2 à M3 pour les petits/bundles groupés, conserver M2/M4 ; uniquement le grand `lm_head` M3 passe MB3/NT1 (une lecture pour trois lignes,24 accumulateurs, pas de padding). PERF-104/105 relus : MB3/NT1 rejeté à M6, MB3/NT2 trop de registres ; nouvelle raison explicite = formeM3 absente et grandeN seulement, une seule rangée de threadgroups plutôt que deux. Contrat CPU inclut grandeN/groupe et bornepadding ; oracle états/tokens et grandhead exacts avant paire courte A/B. Deux autres passes seulement si signal crédible ; ne pas répéter la campagne entière pour du bruit.
- B2 exact :24sessions/80états/KV/tokens réussis7,59s ; comparaison de **tous les logits** et rollback sur chaque préfixe/M2..4 réussie13,99s. Propriété CPU renforcée avec cover MB3/dernièreligne, pas simple condition d'entrée. Mesure courte A/B suivante, puis décision ; pas de gain inventé avant résultat.
- B2 paire terminée : référence D1/D2/D3 **64,331/63,444/60,530**, candidat **65,670/63,790/63,450tok/s** ; baseline normale50,451/50,187. MTP3 signal+4,82%, MTP2 seulement+0,54% (pas établi). Tous hashes/acceptations exacts. Deux passes B/A restantes prévues après le prochain filtre, pas de conclusion finale sur une seule paire.
- PisteC maintenant préenregistrée pour microprofilage : après faible signalM3 deB2, réduire le classement quadratique top8 sur256experts/1..4lignes selon MTPLX. Comparer d'abord le kernel hérité au SIMD hiérarchique, mêmes clés monotones/tie-index et accumulation FP16 ascendante. Cas extrêmes/allties/NaN/inf/±0, toutes sorties/indices et scores en bits ; microbenchmark alterné40appels après warmup, mêmeprocess/aucunmodèle. N'activer le kernel produit que si filtre numérique et réduction de temps passent, puis sessions/rollback/tuner exacts avec la cible. DébitE2E non mesuré ;32-bit SIMD est une adaptation de sélection, pas reprise incompatible de BF16/affine target.
- C premier filtre **échoué**, avant activation/mesure : IDs/ties égaux, mais signe NaN des scores différent pour fixture±NaN (`c-router.log`). Lecture directe half→float→half optimisée conserve le signe ; l'intermédiaire float du kernel hérité canonicalise NaN positif. Variante corrigée avec NaN canonique explicite, sans modifier clés/valeurs finies ; même filtre relancé. Aucun débit/gain tiré de l'échec.
- C filtre corrigé **réussi** :96comparaisons normalisées/non normalisées, lignes1..4/allties/±0/±NaN/±inf/denorms/valeursFP16 diversifiées, IDs et scores en bits identiques. Microprofilage40appels alternés après4warmups, host+GPU : M1 318,375→311,750µs ; M2 250,792→246,542 ; M3 251,958→244,000 ; M4 223,708→221,333. Cette fenêtre inclut appels/évaluations synchrones, pas temps kernel pur ; gainE2E **non mesuré**. Activation locale bornée256experts/top8/1..4lignes, autres formes gardent le kernel hérité. Premier patch d'activation refusé car rustfmt avait réordonné les imports, aucun changement partiel ; patch corrigé. Validation cible/bridge suivante, puis **matrice finale A/B/B/A face au binaire original MTP-04** pour le changement combiné A+B2+C ; elle remplace la clôtureB2 isolée, documentée sans attribuer chaque delta séparément.
- C activé : sessions24blocs/80états/KV/résidus/tokens **réussies**7,21s et touslogits/rollback **réussis**14,49s. BuildC réussi. Tests de dispatch de production/types/k invalides/fallback5lignes ajoutés pour la clôture ; seuls les tests changent après ce binaire. Attribution Apache mise à jour. Matrice finale commence, un seul processus modèle, aucun build/prover simultané.
- Matrice finale terminée : médianes référence→candidat ordinary49,156→50,434 ; D1 61,917→65,603 ; D2 62,623→63,715 ; D3 58,734→62,605tok/s. Changements observés+2,60/+5,95/+1,74/+6,59%, mêmes hashes/acceptations ; D1 choisi sur les deux candidats. D2 gainfaible **non concluant au-delà du régime machine**, D3 signalpositif local ; aucun gain universel, normalisation par ordinary également modifié n'isole pas la causalité. Résumé et relevés sous `mtp-05/summary.json`/`final-matrix/`. Pas d'addition aux anciens résultats.
- Vérification finale **réussie** : fmt/Clippy strict, Rust37+7+14 tests, Python26pass/1skipLing absent, bridge39requêtes budgets/profondeurs/prefixes/cancel-reprise, référenceMLX troisrésidus zéroécart, routeur96fixtures+dispatch/fallback/rejets. Swift6 strict/hardeningfinal3 passé, menu réel rendu avec valeurs explicitementfictives, image inspectée sans coupure. **30/30 harnaisKani** réussis, nouvellegrille85obligations/0échec/3covers, domaines i32/grouped/wide symboliques et toute lignevalide, auplus1padding. CPURust borné, pas preuveGPU/FFI/Swift ; mutant de grille tronquée échoue comme prévu. Preuves `mtp-05/*`, rapport `docs/desktop-v1.3.0-validation.md`.
- Intégration : A+B2+C **validés fonctionnellement et locaux**, formeM3 rembourrée rejetée et négatifs conservés. À la demande « pars sur ça déjà, push main et publie1.3.0 », recherche/mesures arrêtées. Push directmain et paquets app1.3.0/build22 + moteur1.3.0/protocole1 suivants ; aucune app installée ni release publiée à ce stade.
- Clôture livraison 2026-10-05 : source `3733e9df7cbc0298e81e778865e3abd8ac1ebd4b` poussée directement sur main, sans PR ; paquets compilés sur cette source propre (`tracked_changes=false`). `scripts/build-macos-dmg.sh`, validation DMG par les fonctions réelles de l'updater (signature deep/strict, arm64, version/build/OS, manifestes et SHA-256), installation/relocation/exécution/repli de l'archive moteur dans un dossier jetable **réussis**, aucun modèle chargé. [CI Rust Linux/macOS et Kani](https://github.com/0xZKnw/mlxl3/actions/runs/37304447742) et [CI Desktop/Python/E2E](https://github.com/0xZKnw/mlxl3/actions/runs/37304447782) **réussies** sur cette source ; Kani30/30 harnais,0échec, mêmes limites CPU bornées décrites ci-dessus. Logs et preuves dans [le rapport de livraison](docs/desktop-v1.3.0-validation.md#delivery).
- [Desktop v1.3.0/build22](https://github.com/0xZKnw/mlxl3/releases/tag/v1.3.0) publiée à11:52:39UTC, canal latest ; [moteur v1.3.0](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.3.0) publié à11:52:34UTC, canal moteur indépendant, non-latest. Tags pointent sur3733e9d. DMG73252487octets/SHA-256 `bf5ec38cdeb1d022a79ccd70d029e2f54b6837bc74699c43ca6a6857dc850d72` ; archive moteur66851015octets/SHA-256 `ab2cbf3f756468e1606ed29b9e900c5ee7ee8aa7ac1843c6664b3f82a2709530`. Téléversements terminés, tailles/digests GitHub puis fichiers publics retéléchargés **identiques** ; updater de production choisit bien app1.3.0/build22 et moteur1.3.0. Lecture API par tag avant publication renvoyait404 pour les brouillons ; vérification par inventaire authentifié réussie avant de publier. Signatures ad hoc, pas de notarisation. Intégration finale : **main et deux releases publiées**, app installée non remplacée ; activation moteur à chaud demeure le correctif différé. Signal D2 reste non concluant, aucun gain universel revendiqué. Aucun essai encore en cours.

### OPT-2026-10-05-MTP-04 — profondeurs 1/2/3 et Tune MTP, v1.3.0 — validé, publié

- Demande : moteur et Desktop1.3.0, MTP2/MTP3 et bouton comparant baseline/MTP1/MTP2/MTP3 puis conservant le meilleur. Baseline main `d0160f43a3c0379ab3c2bfa0d289b62b4ab885f8`, Desktop1.2.2/build21 ; MTP-01/02/03, validation Desktop1.2.0 et mesures MTP-03 relus. Nouvelle expérience : profondeur récursive auparavant absente, pas répétition du benchmark MTP-03.
- Source primaire MTPLX Apache2.0 : `9882703f3105363ddc37eca9f97aa09a1d387112`, `qwen3_5_mtp_patch.py`, `generation.py`, `commands/public.py`. Hypothèse : réutiliser la même tête avec résidu **avant** norm final, vérification cible groupée, restauration du premier KV exact et réparation groupée cache-only des seules propositions acceptées. Garder le cache préfixe aligné et MTP-03 déjà présents. Pas de kernels copiés sans correspondance EXL3/MLX démontrée ; débits upstream ne sont pas nos mesures.
- Tuner prévu : modèle déjà résident, greedy identique, outils désactivés pour l'essai, deux prompts déterministes courts, échauffement de chaque mode avant chronométrage, ordre aller/retour pour limiter la dérive, aucune réutilisation de cache entre modes. Tokens vérifiés contre la baseline, vitesse decode hors chargement/prefill ; zéro acceptation/divergence/non-fini excluent le candidat. Baseline si gain inférieur à3% ; sauvegarde par modèle/tête/contexte/moteur/matériel, annulation/erreur sans écraser le réglage. Résultats indicatifs propres aux prompts, pas garantie universelle.
- Protocole de validation : contrats CPU/propriétés Kani pour profondeurs/budgets/choix ; tests déterministes états cible et tête après acceptation/rejet pour profondeurs1..3, puis bridge budgets1/2/3/4/17/64, cache/reprise/annulation et tuner réel court. Source Swift6 stricte, régressions subprocess factice menu/progression/résultats/sauvegarde/isolation/annulation/ancien moteur ; suites applicables, CI push main, paquet et deux canaux de mise à jour vérifiés. Un seul modèle GPU à la fois, pas de campagne thermique prolongée ; M5 24Gio/MLX0.32.2, conditions relevées avant test, températures non mesurées. Preuves sous `docs/measurements/mtp-04/`, rapport `docs/desktop-v1.3.0-validation.md`.
- Statut : recherche terminée, avant code/essais ; performances nouvelles **non mesurées**, aucune installation/publication1.3.0.

- Première implémentation locale : chaîne paresseuse GPU avec feedback pré-norm, vérification1..4 tokens et réparation KV groupée ; tuner résident32tokens échauffement par mode puis96tokens ×2prompts en ordre inversé, progression/annulation et sauvegarde clé modèle/tête/version/commit/MLX/matériel/contexte. Premier apply_patch du tuner refusé car fonction nommée `native_bridge` et non `run_bridge`, aucun code partiel appliqué ; module dédié créé ensuite.
- Contrôles CPU **réussis** : cargo check/clippy mlx,chat strict, suite portable chat33+14tests ; sources Desktop Swift6/warnings-as-errors et hardening subprocess réussis (sélectionD2, envoi profondeur, sauvegarde/réouverture, choixmanuelD3, baseline, acceptationeffondrée, doublons, erreur, annulation, moteurancien). Kani profondeur :274 obligations,0échec/1inatteignable,4/4covers ; classement :395 obligations,0échec/6inatteignables,2/2covers,93,58s. Inatteignables liés aux gardes d'overflow/paniques exclues par bornes déjà établies ; boucles bornées à3propositions/4candidats, taillesusize/u64 et tokensu32 symboliques, pas preuveGPU/FFI/Swift.
- Avant premiers contrôles GPU nouveaux : batterie100% en décharge, aucun avertissement thermique/performance `pmset`, swaputilisé2788Mio ; température non mesurée, autres apps ouvertes. Tests de parité/états/cache seulement, pas benchmark simultané avec compilation/prover. Build release en cours ; ensuite un tuner court comme essai de performance indicatif. Publication non effectuée.

- Premier test GPU **échec utile** : D1 passe, puis KV tête différent en D2/bloc2 lors de réparation groupée ; tokens et états cible restent exacts jusqu'à cette assertion.7,36s, `gpu-sessions.log` (dump volumineux conservé). Hypothèse révisée : les projections/RoPE sur plusieurs lignes peuvent changer l'arrondi FP16 du cache ; conserver le calcul cache-only mais construire les lignes sériellement, puis évaluer seulement les deux caches finaux. Cette variante évite les sorties inutiles et respecte la forme d'exécution du KV de référence. Relance du même contrôle prévue, motivée par cet échec, pas mesure de vitesse.

- Réparation sérielle paresseuse **validée** :24blocs, D1/D2/D3, tokens/80états cible/résidu/KV tête exacts en bits,7,22s (`gpu-sessions-fixed.log`) ; propositionsacceptées5/8,7/16,8/24 sur cette fixture, pas benchmark. Vérification/rollback cible totale2..4 et **chaque préfixe retenu**, prompts1/23/24/129/256, erreurs/bornes, **réussie** en15,58s (`gpu-rollback.log`). Copie mutant ignorant acceptationzéro rejetée par la régressionCPU (`mutant-rejected.log`), source de production non mutée. Swift final2 passé, clés moteur/tête remplacée et baseline sauvegardée/ejection pendanttest ajoutées. Aucun gain de vitesse encore mesuré, GPU arrêté entre contrôles.

- Bridge réel **réussi** : budgets1/2/3/4/17/64 pour baseline+D1+D2+D3, hashes/texte/historique/comptes exacts ; invaliddepth0/4 refusées, annulations prefill/delta D2/D3 et reprise, cache préfixe partagéD2→D3/requête différente, tuner annulé/reprise, tuner complet et génération gagnante. `bridge.jsonl`, stderr vide. Essai tuner32warmup/96×2 : baseline51,037 ; D1 63,443 ; D2 63,522 ; D3 60,042tok/s ; D2 sélectionné, acceptations89/99,117/142,129/176. Batterie100% en décharge/swap3060,69Mio/autres apps ouvertes, pas avertissement `pmset`, température non mesurée. **Mesure indicative**, D1/D2 à0,124% d'écart, pas preuve d'avantage reproductibleD2 ni campagne isolée ; aucune accélérationuniverselle revendiquée. [Résultat brut](docs/measurements/mtp-04/tuning-result.json).
- Suite Rust mlx,chat/Clippy strict et **29 harnais Kani** réussis ; nouvelle preuve de filtre à débit fixe64tok/s, parity/accepted/proposed/depth entièrementsymboliques et2/2covers, propriétésfloat générales non prouvées. Test Swift de grille a été refusé à compiler : littéraux20. Rust au lieu de20.0Swift, puis `try` dans autoclosureprecondition ; harnais corrigé avant exécution, logs conservés. Source/test guard output_remaining0 renforcé même lorsque tokenspending ; relance ciblée prévue. Aucune nouvelle mesure de vitesse répétée.
- Clôture code : guardpending/budget0 et tests Swiftfinal3 réussis ; référence indépendante récursive troisrésidus max_abs0. Fonctions/profondeurs/tuner **validés localement**, complétés par MTP-05 kernels/30preuves. Livraison directe main/app+moteur1.3.0 demandée ; publication encore en préparation, performances/limites finales dans MTP-05.
- Clôture livraison : MTP1/2/3 et Tune MTP intégrés à main3733e9d, Desktop1.3.0/build22 et moteur1.3.0 **publiés**, contrôles CI/paquets/téléchargements/canaux réussis. Preuves et limites dans MTP-05 et [le rapport](docs/desktop-v1.3.0-validation.md#delivery). Aucun essai en cours, aucune app installée par cette livraison.

### FIX-2026-10-05-DESKTOP-1.2.2 — contexte et Exa MCP — publié

- Demande : publier Desktop1.2.2 avec compteur de contexte comprenant la réponse et ajout automatique d’Exa MCP à l’installation/mise à jour. Baseline0b183b9, Desktop1.2.1/build20, moteur1.2.1. Journal/historiques Desktop/MCP relus ; aucun benchmark, inférence ni poids nécessaires.
- Cause contexte : événement `context_usage` émis avant génération, `NativeStats.context_used` contient déjà le total final exact (dernier round MCP) mais StudioModel ne l’applique pas à complete. Correction prévue : appliquer les stats finales à la conversation active avant sauvegarde, corriger l’affichage d’anciens historiques via leurs stats ; utiliser les compteurs explicites, sans estimer les tokens à partir du texte. Préserver isolation modèle/requête et bornes ; tester12 entrées+30 sorties=42, historique, limites, stats anciennes, MCP multi-round où les totaux agrégés ne doivent pas être additionnés au contexte final.
- Exa : builtin existant `https://mcp.exa.ai/mcp` côté moteur, ajouté en mémoire aux configurations existantes mais absent du fichier persisté si celui-ci était vide. L’app crée même mcpServers={} lors de l’ouverture. Ajouter Exa manquant automatiquement au lancement et à l’ouverture du fichier, écriture atomique, configurations/serveurs personnalisés et désactivation explicite conservés, prévisualisation isolée. Endpoint officiel confirmé via https://exa.ai/docs/get-started/exa-mcp ; pas de clé requise pour le mode hébergé de base. Tests de création/migration/idempotence/fichiers invalides/serveurs existants, aucun appel outil réel.
- Dépendance de livraison trouvée : le packager adopte par défaut la version de l’app et l’updater exige un moteur de version identique. Cette version Desktop garde le moteur1.2.1 ; passer explicitement sa vraie version lors du packaging et valider sa compatibilité par manifeste au lieu de comparer à la version Desktop. Contrôle du DMG monté et runtime-info après build, moteur indépendant publié inchangé. Skill formal-proof et références relus, Swift6 strict/tests déterministes/CI complète ; aucune preuve Swift disponible, aucun Rust exécutable modifié.
- Protocole : fixture subprocess montrant contexte initial12/stats finales42, assertion qui échoue avant correction, puis tests limites/erreurs/historiques/migration MCP ; compilation ciblée sans modèle, CI distante Rust/Kani/Desktop/Python et un build release final. Publication app1.2.2/build21, moteur1.2.1, après paquet vérifié. Activation à chaud du moteur reste différée.

- Reproduction **confirmée** par le vrai subprocess factice et sources originales : assertion `Context only shows input tokens after completion` échoue après context_usage12/stats.context_used42. Log `build/desktop-v1.2.2/context-before.log`. Correction frontend et migration Exa locales, tests suivants ; moteur natif inchangé.

- Migration Exa **réussie** : création/migration/idempotence, configurations personnalisées ou désactivées préservées, six fichiers invalides refusés sans écrasement. Premier build refusé par signature de verify exigeant expectedVersion ; appel corrigé avec version réelle du manifeste. Test contexte après correctif : compteur final42 et historique corrigés ; test de relance échouait après600ms, avant le délai existant de sauvegarde1s. Protocole de test corrigé pour attendre la création réelle du fichier (maximum12s), fonction de sauvegarde inchangée ; nouvelle compilation/régression suivantes. Logs négatifs conservés.

- Vérification locale finale **réussie** : compilation des sources complètes Swift6/warnings-as-errors et hardening-check corrigé, compteur42/persistance/historique/réouverture/isolation modèle/MCP/bornes/overflow, plus régressions précédentes catalogue/téléchargement/lifecycle. Migration Exa réussie et appelée par l’initialisation réelle, home/préférences/données jetables. Syntaxe zsh, Python py_compile, plutil et diff-check réussis. Logs dans docs/measurements/desktop-v1.2.2 ; CI complète distante et paquet final suivants, aucune inférence/benchmark.

- Livraison réorientée à la demande « fais pas de pr cette fois ci, push direct sur main et fais direct la1.2.2 » : PR #22 venait d’être créée avant réception de l’instruction, fermée **sans fusion**. Intégration par fast-forward/push direct sur main, puis paquet/publication. Espaces de fin dans un log de diagnostic importé normalisés après diff-check ; sources et résultat négatif conservés.

- Révision technique avant publication : l’updater de **l’ancienne** app1.2.1 exige app entrante/moteur de même version ; modifier seulement l’updater nouveau ne résout pas sa validation. Abandon du découplage proposé, packager/validateur/tests updater restaurés. Version Cargo/lock1.2.2 pour identifier le moteur embarqué sans changement d’algorithme ; asset public engine-v1.2.1 inchangé. Paquet sur002b02c **interrompu volontairement** durant le build Swift, descendants du seul processus packaging arrêtés ; aucun artefact publié. Nouvelle compilation prévue sur source finale, justifiée par la compatibilité avec le client installé.

- Paquet compatible **réussi** sur sourcefd1bfca28368d8bb2e0c9e630452cbc8f6d99f5b/main : Desktop1.2.2/build21 et moteur1.2.2/protocole1/MLX, signatures deep/strict et architectures vérifiées par le code de l’updater précédent, manifestes/empreintes corrects, build-info tracked_changes=false. DMG73192215octets, SHA-25658d7ee93b617b9a0a7c57daedd7e5a58700179ecce582074c890c0a544c40c5a. Vérification MCP de l’app packagée sans test GPU réussie. Vérification réseau métadonnées Exa via CLI en registre jetable :1serveur,2outils (recherche/lecture), aucune erreur ; aucun appel d’outil/requête utilisateur ni poids. Téléversement de la release brouillon en cours ; contrôle de l’asset anticipé avant fin d’upload a échoué, pas de publication par cette tentative. Attendre la fin effective avant comparaison/publication.

- CI finale sur main/sourcefd1bfca **entièrement réussie**, tous les jobs et logs inspectés : [Desktop/Python/E2E](https://github.com/0xZKnw/mlxl3/actions/runs/37286589009), [Rust Linux/macOS/Kani](https://github.com/0xZKnw/mlxl3/actions/runs/37286588892). Régressions contexte/persistance et migration MCP exécutées en CI ;26 harnais Kani Rust préexistants réussis, zéro échec. Tests finis pour Swift/JSON/IO, pas preuve complète ni validation GPU nouvelle. Publication encore en téléversement, transfert réseau confirmé actif ; pas de CPU/GPU de modèle en arrière-plan.

- Livraison **terminée** : sourcefd1bfca sur main sans fusion de PR, [Desktop v1.2.2 publié](https://github.com/0xZKnw/mlxl3/releases/tag/v1.2.2), latest=true. Téléversement lent finalement confirmé complet par GitHub ; taille/digest identiques au paquet vérifié, DMG public téléchargé/empreinte identique, sélection réelle de l’updater Desktop1.2.2/build21 réussie. Tag sur source exacte du paquet, moteur embarqué identifié1.2.2, asset indépendant engine-v1.2.1 conservé. CI source finale entièrement verte. App **publiée/proposée par l’updater**, non installée par cette tâche dans l’app utilisateur ; activation à chaud du moteur toujours différée. [Preuve de livraison](docs/measurements/desktop-v1.2.2/release-proof.json), [validation/bornes/limites](docs/desktop-v1.2.2-validation.md).

### FIX-2026-10-05-DESKTOP-1.2.1 — catalogue, explication MTP et téléchargements — publié

- Demande : réparer l'écran Découvrir montrant « The data couldn't be read because it isn't in the correct format » pour `qwen3.6`, puis publier Desktop1.2.1. Baseline main9642c77, Desktop1.2.0/build19, moteur1.2.1 publié. Aucun benchmark/inférence ni téléchargement de poids autorisé ou nécessaire ; défaut d'activation à chaud antérieur toujours différé.
- Reproduction initiale sans modèle : `build/engine-v1.2.1/runtime/mlxl3 hub search qwen3.6 --limit 60` retourne60 dépôts valides, plusieurs `gated:null`. Le moteur sérialise `ModelSummary.gated` comme valeur JSON brute, tandis que `HubModel` Swift exige un Bool : incompatibilité de frontière CLI/GUI. Résultat réseau conservé sous `build/desktop-v1.2.1/qwen36-search-before.{json,stderr}` ; aucune valeur de performance mesurée.
- Hypothèse/correctif prévu : normaliser au décodage Swift les formes légitimes de `gated` (bool/null/absent/auto/manual), conserver un rejet explicite des types/valeurs invalides. Régression deterministe du payload qui échoue avant correction, recherche/détails via le vrai transport CLI avec fixture, annulation/cache/erreurs et contrôle catalogue réseau sans poids. Vérification Swift6 et CI Desktop/Python complète ; skill formal-proof chargé, aucune preuve formelle de Swift disponible dans l'outillage existant. Publication DMG Desktop1.2.1/build20 après vérification du bundle et sélection de l'updater, moteur1.2.1 embarqué ; ne pas remplacer l'asset engine déjà publié.
- Périmètre ajouté par l'utilisateur : réparer aussi les curseurs Sampling. Cause trouvée : `.disabled(studio.mtpEnabled)` dans l'inspecteur ; le bridge accepte déjà les valeurs choisies et bascule en décodage normal pour les réglages incompatibles avec MTP greedy. Correction prévue : rendre les contrôles modifiables, afficher ce mode de repli, conserver les réglages sauvegardés au redémarrage au lieu de les remettre à0/1/1 et effacer une ancienne raison de repli lorsque le mode MTP redevient actif. Tester paramètres envoyés au subprocess, limites, modes et persistance avec moteur factice ; aucun changement du moteur natif ni mesure de débit nouvelle.
- Reproduction Swift réelle avant correction : type `HubModel` d'origine compilé avec Swift6 et décodage de la réponse CLI60 modèles ; échec `DecodingError.typeMismatch`, Bool attendu au champ `[0].gated`, valeur null (`build/desktop-v1.2.1/decoder-before.log`). Les modes auto/manual sont également documentés par [Hugging Face ModelInfo](https://huggingface.co/docs/huggingface_hub/v0.30.2/en/package_reference/hf_api), en plus du format observé. Prototype de correction local, tests suivants ; publication encore non effectuée.
- Révision de périmètre avant livraison : « ahh ça bloque pour les parametres du mtp, bah bloque et explique que c'est pcq le mtp est on ». **Conserver le verrouillage et le greedy MTP existants**, ajouter seulement une explication dans Sampling ; abandonner/restituer les changements de compatibilité/repli/persistance et leurs tests expérimentaux avant publication. Aucun de ces changements n'a été publié. Premier compile du décodeur prototype refusé : expression `||` avec branche lançant une erreur sans `try` sur l'expression entière ; syntaxe corrigée, relance prévue. Échec conservé dans `build/desktop-v1.2.1/decoder-build.log`.
- Vérification locale **réussie** : décodeur de production corrigé relit exactement le payload initial60 modèles (`decoder-after.log`), contre l'échec `[0].gated` du décodeur d'origine. Build des sources Desktop complètes avec `swiftc -swift-version 6 -warnings-as-errors`/SDK26.5 et exécution du hardening-check réussis :8 formes de gated légitimes,5 types/valeurs invalides refusés, recherche/détails via subprocess réel factice, réponses vides, erreurs réseau/décodage, annulation d'une recherche lente, reprise/cache/plus/rafraîchissement, plus contrôles lifecycle/persistance/MTP existants. Les valeurs Sampling restent verrouillées lorsque MTP est activé comme demandé ; seul le libellé explicatif change. Moteur natif et ses algorithmes inchangés. Preuve formelle Swift/JSONDecoder non disponible : contrôles de compilation/tests finis, pas preuve complète. [Notes](docs/release-v1.2.1.md), [validation](docs/desktop-v1.2.1-validation.md) ; CI/DMG/publication suivants.
- Nouvelle demande avant publication : ajouter une progression de téléchargement soignée, débit Mo/s et barre cohérents avec le style de l'app. Antécédent : le footer actuel expose déjà les octets/progression/pause, mais pas le débit. Réutiliser les événements natifs réels, calculer un débit sur intervalle monotone en excluant les octets déjà conservés à la reprise ; tests déterministes de temps/octets/limites, transport NDJSON/annulation/reprise et rendu sans poids. Aucune prétention d'accélération. PR #21 créée sur490df07 ; première compilation du paquet supersédée par ce périmètre supplémentaire, aucun artefact de cette première tentative publié. État : UI/débit en cours, prochaine compilation finale après intégration de ces changements.

- Téléchargements : modèle de progression et carte SwiftUI intégrés ; **tests réussis** (`download-progress-check.log`, `hardening-check-final.log`) : débit sur horloge monotone, exclusion des octets conservés à la reprise, stagnation à0, reset des compteurs/temps, taille inconnue, NaN/infini/négatifs et overflow ; matrice finie300 cas. Transport CLI réel avec flux NDJSON factice, pause, reprise, succès et échec vérifiés en dossiers/préférences jetables, aucun poids reçu. Sources Desktop complètes compilées Swift6/warnings-as-errors. Pas de vérificateur formel Swift disponible : tests finis, pas preuve complète. Prévisualisation : premier harnais incomplet sans type EngineState, puis utilisation d'une clé d'environnement SwiftUI en lecture seule ; erreurs limitées au harnais de rendu, pas aux sources de l'app, correction du harnais en cours. La première CI du commit490df07 est entièrement verte ; nouvelle CI nécessaire sur les changements de téléchargement.
- Contrôle négatif prévu : compiler une copie jetable du calcul de débit qui compte à tort les octets déjà conservés dès le premier événement ; vérifier que le test de reprise l'intercepte. Sources de production conservées, pas de benchmark. Puis rendu des vues de production pour contrôler lisibilité/progression/pause/fin, dernière compilation DMG et publication.

- Contrôle négatif du débit **réussi** : la copie jetable comptant les octets conservés échoue sur la régression de reprise ; test de production relancé et réussi. Harnais de rendu corrigé, build complet Swift6 et image des vues de production réussis, inspection visuelle sans contenu tronqué ; débits et volumes de l’image sont des fixtures. Carte cohérente avec la palette/fontes de l’app, progression, pourcentage, volume, Mo/s, pause/reprise et état final. [Preuves](docs/measurements/desktop-v1.2.1/), CI finale et paquet en préparation.

- Paquet final sur source66c5a9aa7bf322fa5341ce75687fd3db1ab44c03 **réussi** : `scripts/build-macos-dmg.sh` SDK26.5, moteur release Rust/MLX et Desktop1.2.1/build20 compilés ; avertissements préexistants SwiftMath/fontes/chemins CLT et dépréciation hdiutil sans échec. Validation par fonctions réelles de l’updater : digest, montage DMG, signature deep/strict, arm64, version/build/minOS et empreintes du runtime réussis ; build-info source exacte et tracked_changes=false, runtime-info1.2.1/protocole1/MLX actif. DMG73161734octets, SHA-256399a9505ee33333c7d0e8e0da24a763b03f9b684815391fbd083047a4d178337. Moteur embarqué recherche Qwen3.6 :60 modèles décodés sans erreur. Asset téléversé dans une release **brouillon**, taille/digest GitHub identiques ; CI finale encore en cours, aucune publication Desktop à ce stade. La requête API par tag renvoyait404 avant création publique du tag ; lecture par inventaire des releases authentifié réussie.

- Livraison demandée « vas y merge la pr et publish la1.2.1 » : [PR #21](https://github.com/0xZKnw/mlxl3/pull/21) fusionnée à08:28:55UTC, mergeb9fcb1077b1cd8860d7422ea9ed32383d40ed78e. [Desktop v1.2.1 publié](https://github.com/0xZKnw/mlxl3/releases/tag/v1.2.1) à08:28:58UTC, tag sur source66c5a9aa7bf322fa5341ce75687fd3db1ab44c03, latest=true. Digest/taille publics identiques au DMG téléchargé et vérifié, sélection réelle de l’updater : Desktop1.2.1/build20 et moteur1.2.1. Asset du moteur indépendant inchangé. Rust Linux/macOS et Kani réussis ; dernier résultat Desktop encore attendu à cet instant, aucun échec observé. App **publiée/proposée par l’updater**, app installée de l’utilisateur non remplacée par cette tâche ; activation à chaud du moteur toujours différée. [Preuve de livraison](docs/measurements/desktop-v1.2.1/release-proof.json).

- Clôture de vérification : CI du commit source66c5a9a entièrement **réussie**, résultats et logs inspectés : [Desktop/Python/E2E](https://github.com/0xZKnw/mlxl3/actions/runs/37283271706), [Rust Linux/macOS/Kani](https://github.com/0xZKnw/mlxl3/actions/runs/37283271797). Régression de transfert NDJSON et300 cas finis exécutés par CI ; contrats Rust existants26 harnais Kani réussis, zéro échec, sans preuve de Swift/FFI/GPU. Aucune inférence/benchmark ni installation dans l’app utilisateur supplémentaire. Publication Desktop1.2.1 terminée.

### RELEASE-2026-10-05-ENGINE-1.2.1 — publication du moteur MTP-03 — publié

- Autorisation nouvelle : « bah publie comme nouvelle version du moteur pour faire une maj ». Les mentions antérieures « non publié/non autorisé » restent l'état historique des essais. Publier le canal `engine-v1.2.1`, compatible avec Desktop 1.2.0 ; version Cargo 1.2.1, protocole bridge 1, arm64, macOS ≥26.2. Aucun modèle inclus.
- Périmètre : uniquement l'entretien KV MTP-03 et ses contrats/tests déjà vérifiés ci-dessous ; aucune nouvelle optimisation ni campagne de performance. Desktop et bibliothèques MLX 0.32.2 conservés. [Notes de version](docs/release-engine-v1.2.1.md).
- Protocole de livraison : commit/PR dédié, vérifications CI Rust/Kani/Python/Desktop sur GitHub, une compilation release locale, signatures ad hoc et architecture, manifeste/empreintes/archive, installation de l'archive par le code réel de l'updater dans un dossier jetable sans charger de modèle. Publier avec `latest=false` pour garder le canal Desktop indépendant ; vérifier l'asset distant et sa sélection par l'updater. Aucun benchmark ni inférence relancé. Publication et installation dans l'app de l'utilisateur **non effectuées à ce stade**.
- Préparation réussie : [PR #20](https://github.com/0xZKnw/mlxl3/pull/20), source `954cfe37167f9f877b04dc840394d303106be687`. `MACOSX_DEPLOYMENT_TARGET=26.2 cargo build --release --locked --features mlx,chat` terminé en32,18 s ; runtime-info version1.2.1/protocole1/MLX activé/revision954cfe37167f. Signatures ad hoc `codesign --verify --strict` et `lipo -verify_arch arm64` réussis sur binaire/deux dylibs, `vtool` confirme minOS26.2. `python scripts/package_engine.py build/engine-v1.2.1/runtime dist --version 1.2.1` ; archive66827147 octets, SHA-256 `ffa0300d89c03562c56b3a05b4eca8c1fc29d245b20bf3a2169b7f611f147902`, identique au digest de l'asset GitHub téléversé dans une release **brouillon**. 15 tests de packaging réussis ; test Swift existant adapté temporairement à l'archive1.2.1, compilant les sources réelles `UpdateManager`/`EngineRuntimeStore`/`Localization`, installation/relocation/exécution/repli réussis en dossier jetable avec appVersion1.2.0. Aucun poids chargé. Logs de livraison dans `build/engine-v1.2.1/` ; CI distante en cours, publication encore non effectuée.
- Livraison du 2026-10-05 : demande explicite « merge meme avant que le ci finisse ». PR #20 fusionnée à06:59:57 UTC, merge `1f9ba848d6ea58478d9aa490e38404d01034706f`. [Moteur engine-v1.2.1 publié](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.2.1) à07:00:09 UTC, tag sur le commit source exact954cfe37167f, `latest=false` ; Desktop v1.2.0 reste la release latest. Asset distant téléchargé et SHA-256/taille identiques au paquet installé pendant le contrôle jetable ; sélection réelle `UpdateManager.selectRelease` sur la réponse GitHub publique : moteur1.2.1/Desktop1.2.0. Le moteur est **publié et proposé par le canal de mise à jour**, mais **pas installé dans l'app de l'utilisateur par cette tâche**. [Preuve de livraison](docs/measurements/engine-v1.2.1-proof.json), [validation et limites](docs/engine-v1.2.1-validation.md).
- CI après publication : contrôles PR Rust Linux/macOS et Kani **réussis** ; [suite Desktop/Python](https://github.com/0xZKnw/mlxl3/actions/runs/37274850631) encore **en cours** au relevé, nouvelle CI du merge également en cours. Ne pas transformer cet état en validation complète. Aucun nouveau benchmark/inférence, aucune recherche PERF-135/136 reprise ; processus locaux de compilation, installation et téléchargement terminés.
- Retour utilisateur après mise à jour : l'app ouverte conserve apparemment l'ancien moteur jusqu'à fermeture/réouverture ; vitesse rapportée56tok/s avant relance,62tok/s après. **À corriger plus tard**, selon sa demande : activation du nouveau runtime et rechargement du modèle sans redémarrage manuel, vérification de la version réellement utilisée par le subprocess. Observation déclarative, ni protocole contrôlé ni nouvelle preuve de gain ; cause non investiguée, aucune correction exécutable dans cette livraison. Le test d'installation jetable ci-dessus ne couvre pas cette transition dans l'app déjà ouverte. [Suivi du défaut](docs/engine-v1.2.1-validation.md#activation-dans-lapp-ouverte--à-corriger-plus-tard).
- Révision finale : les deux runs PR sur le commit source954cfe37167f sont désormais **terminés et réussis**, tous leurs jobs verts : [Rust Linux/macOS + Kani](https://github.com/0xZKnw/mlxl3/actions/runs/37274850532), [régression Desktop/Python/protocole](https://github.com/0xZKnw/mlxl3/actions/runs/37274850631). Résultats individuels inspectés, aucun échec à corriger ; preuve de livraison complétée. Les contrôles du merge/documentation constituent des runs distincts. La release est publiée, l'archive distante vérifiée et le défaut d'activation à chaud consigné pour plus tard ; aucune inférence locale supplémentaire.

### OPT-2026-10-05-MTP-03 — entretien du cache sans sortie inutilisée — validé localement sur le calcul

- Demande : « cherche a booster le mtp deja » ; priorité MTP, suite des essais DFlash mise de côté. Antécédents lus : MTP-01/02 et `docs/desktop-v1.2.0-validation.md`, scripts/parcours natifs et preuves `mtp-final.*`. Le cache de préfixe aligné est déjà présent ; il ne sera pas réinventé. Les +16,2 % MTP/normal historiques sont un pilote non isolé, pas le gain de ce nouvel essai.
- Recherche nouvelle : après une proposition acceptée, `Session::advance` exécute la tête complète **et le lm_head cible** pour un résultat immédiatement jeté ; la préfill exécute aussi attention/MoE/norme de sortie dont seul le KV est conservé. Hypothèse : ajouter un parcours limité à embedding/normes/fc puis K/V de la même attention réduit ce travail tout en gardant les mêmes matrices, RoPE, dtype et formes. Vérifier les sources primaires MTP actuelles avant choix définitif. Pas de profondeur augmentée dans ce premier candidat.
- Baseline : HEAD `0d89e013899be6b5e4c6a7e9bb2db35869aea087`, moteur v1.2.0 courant, Qwen3.6-35B-A3B EXL3 2,49 bpw + tête MTP affine4/group64 (`0295b81421bf4d0fccca9a7c0fcfb1418dda3516`), M5 24 Gio, macOS27.2/MLX0.32.2. Nouveau relevé : batterie89 % en décharge, aucun avertissement thermique/performances rapporté par `pmset`, swap système utilisé5582,31 Mio ; température non mesurée, autres apps ouvertes. Archiver les sources et le release avant code. Aucune comparaison aux timings secteur du 4 octobre.
- Protocole prévu : test différentiel cache K/V bit à bit contre tête complète, cache vide/existant, chunks1/2/3/17/24/255 et erreurs (vide, dimensions, token hors vocabulaire), puis prédiction suivante exacte après reprise. Tests intégrés cible/caches et bridge budgets1/2/3/17/128, préfixe/suffixe/conversation, annulation/récupération et pénalité. Kani des contrats/bornes Rust pertinents ; aucun prover ni compilation durant les mesures. Bridge comparant anciens/nouveaux binaires en A/B/B/A, trois prompts court/matrice/long, warmup128 et deux répétitions mesurées MTP + contrôles normaux par passe, greedy/cache OFF/contexte4096. Mesurer decode/complet/TTFT, tokens/hashes/acceptés/proposés/blocs, allocations et empreinte processus, conditions par passe. Retenir une baisse de durée ≥3 % reproduite au-delà de la dérive des contrôles ou un gain préfill isolé sans régression ; sinon rejeter/restaurer.
- Vérification : `$formal-proof-skill` et référence Rust déjà relus ; fmt/Clippy strict/build/suite ciblée et applicable, Kani, CI inspectée. GPU/FFI non prouvés ; comparaisons finies seulement. Preuves `docs/measurements/mtp-03-*`. Recherche/prototype avant essai ; métriques nouvelles **non mesurées**, aucune installation/publication autorisée par cette tâche.
- Prototype local : `Head::extend_cache` réutilise la préparation numérique de la tête, n'exécute que K/V, et remplace les sorties jetées en préfill et après acceptation. Release A archivé `/tmp/mlxl3-mtp03-a`, SHA-256 `c2a9e0eb01dd30236a923dee44eab8f37fa930c00a3ace0772f8aa1159b38d9d`, sources/empreintes dans `mtp-03-baseline.json`. Clippy strict réussi. Premier lancement GPU **n'a exécuté aucun test** : filtre court combiné à `--exact`, 74 filtrés (`mtp-03-cache-exact.log`) ; corriger par le nom complet, aucune parité revendiquée avant cette relance.
- Relance GPU **réussie**, un test physique exécuté en 3,81 s (`mtp-03-cache-exact-gpu.log`) : K/V égaux en bits à la tête complète, caches vides puis cumulatifs, lignes1/2/3/17/23/24/255, prédiction normalisée suivante exacte ; entrées vides, mauvais nombre de tokens, largeur et token hors vocabulaire refusés sans modifier le cache. Ce temps est un temps de test, pas un gain mesuré. Tests cible/rollback et référence MLX indépendante, build B et Kani suivants ; aucun benchmark encore lancé.
- Contrôles suivants **réussis** : préfill/verifier/rollback cible, préfixes1/23/24/129/256 et commits1/2, logits/états exacts et erreurs/capacité (`mtp-03-target-exact.log`, 9,24 s) ; référence indépendante MLX-LM47 positions en chunks1/2/3/17/24, hidden exact/max_abs=0 (`mtp-03-reference-exact.log`, 3,64 s). Kani du contrat réel d'append avec offset/rows i32 entièrement symboliques : aucune propriété échouée, 4/4 couvertures (vide, overflow, origine, limite maximale), `mtp-03-kani.log` ; ne prouve pas le GPU. B construit/archivé, preuve SHA séparée avant mesures. Début prévu de l'A/B/B/A, un seul moteur et aucun build/prover actif.
- Salut A1 terminé : hash `b38c8e6fb96d7548`, 128 tokens, MTP **59/67 acceptés en67blocs**, normal adjacent exact ; médianes decode MTP **2,240439 s /56,685 tok/s**, normal **2,541226 s /49,976 tok/s**, complet MTP **2,556977 s**. Reference seulement, aucune comparaison B à ce stade (`mtp-03-salut-a1.{jsonl,stderr}`, résumé `mtp-03-summary.json`). Conditions A1 relevées après lancement et non avant la passe, batterie83 % en décharge au relevé ; ne pas les qualifier de préalables. B SHA-256 `c77c56a9b0c936728999fa7f582c8816d7b01fac83114ecfdc989dc2ca929287`.
- Salut B1 terminé (`mtp-03-salut-b1.{jsonl,stderr,conditions.json}`, stderr vide) : hash/budget/59 sur67/67blocs inchangés ; decode **2,110404 s /60,277 tok/s**, complet **2,436738 s**, normal **2,583465 s /49,219 tok/s**. Face à A1, environ −5,8 % de durée decode alors que le contrôle normal ralentit de1,7 %. Signal positif individuel ; B2 et A2 requis avant décision, aucun gain global annoncé.
- Salut B2 terminé (`mtp-03-salut-b2.*`) : hash/budget/acceptation inchangés ; decode **2,197801 s /57,785 tok/s**, complet **2,520910 s**, normal **2,563688 s /49,551 tok/s**. Gain plus faible face à A1 (~1,9 % de durée decode) ; résultat **non concluant individuellement** au seuil3 %, la référence A2 doit distinguer variabilité machine et effet du code. Aucun boost uniforme revendiqué.
- Salut A2 terminé (`mtp-03-salut-a2.*`) : hash/budget/acceptation inchangés ; decode **2,191446 s /57,953 tok/s**, complet **2,506666 s**, normal **2,470242 s /51,412 tok/s**. Face à la médiane des passages A, B1 réduit le decode de4,76 %, B2 de0,82 % seulement ; contrôles normaux B ralentis de3,10 % et2,31 %, ratios de vitesse MTP corrigés de ce contrôle +8,26 %/+3,16 %. La réduction brute ≥3 % n'est pas reproduite sur le salut : **vitesse non concluante sur ce cas**, poursuivre matrice/long pour leur régime propre et contrôler les régressions, sans changer le code B ni répéter silencieusement le salut.

<!-- mtp03-progress -->
- Clôture à la demande du 2026-10-05 : « arrete de faire chauffer le mac la t'as fais tes tests deja ». **Plus aucun moteur de test, benchmark ni prover actif** lors du dernier contrôle ; les deux dernières commandes étaient déjà terminées. Aucun essai relancé. Source et release B conservés localement, aucune app installée ni publication par cette tâche.
- Vérification finale inspectée : `cargo fmt --all -- --check`, `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`, même Clippy avec `--features chat`, `cargo test --release --locked --features mlx,chat` : **54 tests réussis, zéro échec**, 47 tests GPU/autres explicitement ignorés dans cette suite et contrôles physiques ciblés exécutés séparément ci-dessus. `cargo kani --lib --no-default-features --output-format terse` : **26 harnais réussis, zéro échec** (`mtp-03-final-kani.log`) ; nouveau contrat de position sur offset/rows i32 symboliques, 12 propriétés/4 couvertures au test ciblé, ne prouve ni FFI ni GPU. Avertissement futur de la dépendance préexistante `block0.1.6`, sans refus de Clippy. Logs `mtp-03-final-{fmt,clippy,clippy-cpu,tests,kani}.log`.
- Bridge final réel **réussi** : `python3 scripts/check-mtp-bridge.py /tmp/mlxl3-mtp03-b models/Qwen3.6-35B-A3B-EXL3-2.49bpw models/Qwen3.6-35B-A3B-MTP-4bit --tokens 1,2,3,17,128 --prompt-file docs/audit-runtime-2026-09-04.md --check-prefix` (`mtp-03-bridge-complete.{jsonl,stderr}`, stderr vide) : IDs/texte/historique exacts sur budgets courts/128, préfixe/suffixe et conversation différente, refus tête vide/manquante/MTP+DFlash, annulation au contexte/au delta et récupération, fallback pénalité1.1. Les timings de ce contrôle fonctionnel, effectué avec vérification CPU en parallèle, **ne sont pas utilisés comme benchmarks**.
- Résultat retenu : matrice **57,22→63,89 tok/s** (rapport des médianes de passage environ+11,65 %), B1/B2 **+14,28 %/+9,02 %** de vitesse face aux A encadrants, **+9,81 %/+12,47 %** après correction descriptive par le contrôle normal. Court et long restent non concluants selon le seuil préenregistré ; pas de promesse générale. CI existante inspectée et réussie sur HEAD avant ces changements ; nouvelle CI distante, suite Python/Desktop complète et chemins GPU non ciblés **non exécutés pour ce candidat**, arrêt explicite respecté. Les contrôles complémentaires PERF-135 non encore exécutés et PERF-136 sans code/benchmark sont laissés dans cet état, aucune poursuite de calcul en arrière-plan.
- Série MTP-03 **terminée** : 12 passages, 24 répétitions MTP et24 contrôles ordinaires, toutes les sorties/compteurs égaux à la baseline sur chaque prompt. [Décision complète](docs/measurements/mtp-03-decision.json), [médianes et empreintes](docs/measurements/mtp-03-summary.json). **Gain decode reproduit sur la matrice**, réduction brute ≥3 % dans les deux passages B et gain ≥3 % après prise en compte de la dérive du contrôle normal ; court et long **non concluants** au même critère, donc aucune accélération générale annoncée. Pas de régression stable >3 % du temps complet dans les deux passages B d'un cas. Conserver le candidat local pour sa réduction mesurée sur la matrice, finir les contrôles full suite/bridge complet/contrats ; aucun nouveau benchmark prévu pour gonfler le résultat. Aucun boost de pic MLX revendiqué.
- mtp-03-long-a2 terminé, parité hash/budget/compteurs réussie : MTP decode 2.638822 s /48.128 tok/s, complet 8.760394 s, TTFT 6.121561 s ; normal decode 2.733981 s. MTP 52/74 acceptés en74blocs, hash `cea6d766333b2e34`. Preuves `mtp-03-long-a2.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- mtp-03-long-b2 terminé, parité hash/budget/compteurs réussie : MTP decode 2.677376 s /47.443 tok/s, complet 9.087373 s, TTFT 6.409986 s ; normal decode 2.768638 s. MTP 52/74 acceptés en74blocs, hash `cea6d766333b2e34`. Preuves `mtp-03-long-b2.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- mtp-03-long-b1 terminé, parité hash/budget/compteurs réussie : MTP decode 2.502911 s /50.753 tok/s, complet 8.451313 s, TTFT 5.948390 s ; normal decode 2.837969 s. MTP 52/74 acceptés en74blocs, hash `cea6d766333b2e34`. Preuves `mtp-03-long-b1.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- mtp-03-long-a1 terminé, parité hash/budget/compteurs réussie : MTP decode 2.797264 s /45.436 tok/s, complet 9.120219 s, TTFT 6.322943 s ; normal decode 2.776754 s. MTP 52/74 acceptés en74blocs, hash `cea6d766333b2e34`. Preuves `mtp-03-long-a1.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- mtp-03-matrice-a2 terminé, parité hash/budget/compteurs réussie : MTP decode 2.241150 s /56.672 tok/s, complet 2.405747 s, TTFT 0.164587 s ; normal decode 2.627537 s. MTP 61/65 acceptés en65blocs, hash `70fb796b1d06bb21`. Preuves `mtp-03-matrice-a2.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- mtp-03-matrice-b2 terminé, parité hash/budget/compteurs réussie : MTP decode 2.036064 s /62.395 tok/s, complet 2.198972 s, TTFT 0.162896 s ; normal decode 2.674245 s. MTP 61/65 acceptés en65blocs, hash `70fb796b1d06bb21`. Preuves `mtp-03-matrice-b2.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- mtp-03-matrice-b1 terminé, parité hash/budget/compteurs réussie : MTP decode 1.942427 s /65.383 tok/s, complet 2.097411 s, TTFT 0.154974 s ; normal decode 2.490907 s. MTP 61/65 acceptés en65blocs, hash `70fb796b1d06bb21`. Preuves `mtp-03-matrice-b1.{jsonl,stderr,conditions.json}`, médianes `mtp-03-summary.json` ; résultat individuel, décision après la série.
- Matrice A1 terminée : hash `70fb796b1d06bb21`, 128 tokens, MTP61/65 acceptés/65blocs ; decode **2,198280 s /57,772 tok/s**, complet **2,361580 s**, normal **2,556908 s /49,669 tok/s**. Baseline individuelle, contrôle de parité réussi (`mtp-03-matrice-a1.*`) ; B1/B2/A2 puis long prévus, mêmes binaires.

### OPT-2026-10-04-MTP-02 — cache de préfixe MTP aligné — en cours

- Antécédent : MTP-01 a confirmé la parité de sortie, mais reconstruit cible et tête à chaque requête. Le cache cible existant AUDIT-23 conserve les frontières de chunks pour garder l'arrondi exact ; pas de gain hérité revendiqué.
- Relance finale **validée** : `mtp-final.{jsonl,stderr}`, stderr vide, tests préfixe/suffixe/conversation différents, rejets tête vide/manquante/MTP+DFlash2, annulations/reprises et pénalité1.1 réussis. Les tokens/textes/history restent égaux. Débit64tokens observé normal48.85/49.32 contre MTP57.48/56.64tok/s, ratio de médianes+16.2% sur ce seul prompt402tokens ; autres apps ouvertes, aucune isolation thermique, un groupe ABBA. Ne pas généraliser ni additionner aux gains préfill. Cache256tokens réutilisé. Intégration : code/bundle local validé, publication en attente de CI.
- Premier essai **validé pour la parité de préfixe** : prompts402/408 tokens, 256 réutilisés, froid/rejoué et conversation différente donnent les mêmes IDs ; timings de diagnostic conservés sans gain final (compilations Desktop contemporaines). Essai arrêté sur un test négatif mal câblé : champ `dflash` ignoré par le protocole, test corrigé en `dflash2` ; tête vide/manquante déjà refusées. Logs `mtp-prefix-bridge.{jsonl,stderr}`, résultat complet relancé et validé dans `mtp-final.*`.
- Hypothèse/changement : conserver aussi le KV de la tête à une frontière de 256 tokens. À la position K, cache tête contient K−1 paires connues ; la dernière paire n'est complétée qu'à lecture du token K. Cela évite de mémoriser un token situé hors du préfixe réutilisable.
- Baseline : MTP-01 sans cache, mêmes poids Qwen3.6 EXL3 2.49 bpw/tête4bit, M5/macOS27.2/MLX0.32.2. Protocole prévu : froid/rejoué, suffixe modifié, conversation différente, annulation puis récupération ; IDs/texte exacts, états/caches après acceptation/refus. Débit/TTFT/RAM avant/après : non mesurés ; aucun benchmark lancé avant la parité.
- Preuves prévues : tests bridge MTP de préfixe et logs `docs/measurements/desktop-v1.2.0/mtp-prefix-*`. Intégration : code local en cours, aucune app installée/publication.

### OPT-2026-10-04-MTP-01 — tête native Qwen et vérification exacte — en cours

- Parité numérique indépendante **validée** : tête native contre `mlx_lm.models.qwen3_5.DecoderLayer`, poids norm/affine identiques en Float16, 47 positions en chunks 1/2/3/17/24, sorties normalisées bit-à-bit égales et max_abs=0. Logs `mtp-reference-{python,rust}.log`, script reproductible `scripts/check-mtp-reference.py`, fixture locale `build/verification/mtp-reference.json`. Cela exerce affine QMM, gather experts, routeur, RoPE et KV récurrent de la tête ; pas preuve formelle GPU.
- Premier pilote bridge **validé fonctionnellement** : budgets 1/2/3/17/64, A/B/B/A dans un processus résident, IDs/hashes/texte/historique identiques ; annulation dans prefill et au premier delta puis reprise exacte, pénalité 1.1 confirmée en moteur normal. À 64 tokens : 31 propositions acceptées ; débit observé normal 48.51/47.21 contre MTP 58.32/56.31 tok/s. **Pilote non isolé**, compilations Swift/Kani éventuellement contemporaines et pas de relevé thermique ; aucun gain final annoncé. Logs `docs/measurements/desktop-v1.2.0/mtp-bridge-first.{jsonl,stderr}` (stderr vide). Le prefill MTP reconstruit encore son cache à chaque round ; optimisation de réutilisation séparée à examiner après vérification.
- Tentatives formelles complémentaires : CrossHair0.0.101 sur `package_engine.valid_version`, 20 s/condition : aucune contre-exemple trouvé mais **conditions non confirmées**, pas une preuve. CBMC6.11.0 sur la vraie FFI affine : **bloqué** par parsing du libc++ Apple moderne ; FFI/MLX/Metal non prouvés. Logs `crosshair.log`/`cbmc-ffi.log`.
- Vérification finale en cours : premier Clippy refusé pour taille de variante MLP et Default dérivable ; structures internes mises en boîtes et derive corrigés, relance prévue. Clippy corrigé réussi ; suite Python : 25 puis 26 réussis, 1 modèle Ling absent ignoré. E2E a détecté deux anciennes attentes de restauration DFlash contraires à la mise de côté explicitement demandée ; tests corrigés pour désactivation DFlash au relancement et restauration MTP. Le test MTP préfill a d’abord échoué à compiler (nom de champ), corrigé avant exécution ; aucune mesure GPU tirée de ces échecs.
- Incidents conservés : premier build affine refusé (nom de constructeur Array), puis durée de vie du paramètre de closure Rust ; corrigés. Premiers builds Swift refusés par isolation de `repository` puis lecture d’une propriété avant initialisation ; corrigés. Kani de tous les contrats lancé, résultats individuels à inspecter. Poids téléchargés : SHA-256 exact et inspection du header réussie.

- Nouvelle demande explicite : remplacer le parcours DFlash Desktop par MTP. Antécédent : MTP exclu dans la roadmap du 4 septembre et RUST-03 ; ce choix est révisé uniquement par la demande du 4 octobre. Référence de câblage : MTPLX `qwen3_5_mtp_patch.py` (licence Apache-2.0), pré-norme cible, concaténation embedding/hidden, attention complète et MoE ; ne pas importer ses annonces de débit.
- Hypothèse/changement : poids MTP natifs séparés, projections affine MLX et expert gather-QMM de la bibliothèque, cache KV transactionnel et propositions acceptées seulement lorsqu'elles correspondent à la cible greedy. Sampling non greedy conserve le moteur ordinaire dans cette première intégration et affiche son mode réel. Toute tête manquante/incompatible est signalée, sans faux MTP actif.
- Source poids : `mlx-community/Qwen3.6-35B-A3B-MTP-4bit`, commit `0295b81421bf4d0fccca9a7c0fcfb1418dda3516`, `model.safetensors` 475130833 octets, SHA-256 `77fbc6594cdd830cae89e0b693f18278dd0a7a7c5749fd33d4bc5817dbb91bad`. Modèle cible local EXL3 2.49 bpw ; pas de nouvelle quantification cible.
- Baseline : code d'ouverture V120, moteur normal greedy ; protocole : tests des dimensions/checkpoint, affine QMM/gather comparés à MLX Python, tête comparée à la formule Python, cible vérifiée contre forward mono-token, tokens/textes/états/cache après refus ou acceptation, budgets 1/2/3 et annulation/reprise. Microtests synthétiques puis bridge réel court/long ; un seul processus GPU modèle à la fois. A/B temporel seulement après parité, prompts et conditions enregistrés ; performances **non mesurées**.
- Environnement : Apple M5 24 Gio, macOS 27.2, Swift 6.4/SDK26.5, Rust1.98.1, MLX0.32.2 ; secteur100%, autres apps de bureau ouvertes, thermique non contrôlée. Fallback M1–M4 exercé sur M5 par désactivation du chemin TensorOps, pas matériel réel.
- Preuves prévues : `docs/measurements/desktop-v1.2.0/mtp-*`, tests Rust/Swift/Python, Kani des contrats CPU ; GPU/FFI/sampling non greedy ne seront pas qualifiés de preuves formelles. Intégration : avant prototype, aucun MTP installé/publié.

### OPT-2026-10-04-V120 — solidification Desktop/moteur et MTP Qwen — en cours

- Suite Desktop complète **réussie**, installation de l’archive signée réelle dans un dossier jetable, exécution déplacée, rejet/fallback inclus ; DMG construit. Clippy CPU a signalé le contrat GDN utilisé uniquement par MLX/Kani : test de bornes ajouté, relance réussie. Suite CPU43 tests réussis ; suite MLX52tests réussis (tests GPU explicites séparés). Preuves `e2e-final.log`, `dmg-build.log`.
- Vérification CPU/GPU initiale inspectée : **25 harnais Kani réussis**, 0 échec (source CPU et bornes spécifiques, pas preuve du moteur complet). Chemin portable EXL3 exercé sur M5 avec `MLXL3_DISABLE_TENSOR_OPS=1` : matrices simples K1…8 et groupes K1…6/8, seuils 23/24 et queues jusqu'à 513 lignes, sorties bit-à-bit égales aux lignes indépendantes. Log `docs/measurements/desktop-v1.2.0/portable-gpu.log`, 1 test physique réussi. Simulation du chemin M1–M4, sans matériel M1–M4.

- Demande : v1.2.0, robustesse Apple Silicon, mises à jour app/moteur indépendantes, MTP Qwen3.5/3.6, DFlash mis de côté, nouvelle identité M3 et copie sous les messages.
- Antécédents consultés : index du journal, UI-113, AUDIT-01 à AUDIT-05, PERF-134/135/136, `docs/desktop-v1.1.3-validation.md`, `docs/release-v1.1.3.md`. Les gains UI-113 et les essais DFlash existants ne seront pas refaits ni additionnés. Les modifications préexistantes sont conservées ; diff/statut à l'ouverture : `docs/measurements/desktop-v1.2.0/opening-{tracked.patch,status.txt}`.
- Hypothèse : séparer les canaux de mise à jour avec validation du protocole et installation transactionnelle supprime le besoin de remplacer toute l'app pour chaque moteur ; rejeter un runtime incompatible et garder le runtime embarqué évite une panne au démarrage. Vérifier les fallbacks GPU au lieu de supposer la génération de puce. MTP doit utiliser les poids natifs du modèle et vérifier les propositions avec la cible, jamais inventer un gain ou activer une tête absente.
- Baseline : HEAD `d54540a`, Desktop 1.1.3 build 18 + checkout modifié (preuves d'ouverture). Nouvelle branche `codex/v1.2.0`. Débit, latence et RAM nouveaux : **non mesurés**.
- Protocole prévu : inspection des historiques pertinents et des chemins de production ; nouveaux essais MTP enregistrés séparément avant GPU. Tests déterministes des versions/URLs/digests, archives malformées, ABI, annulation, rollback, sélection app/moteur/les deux ; bridge réel, UI/copie et persistance. Fmt/Clippy/build/tests Rust, Kani sur contrats CPU, suite Swift/Desktop, tests Python et CI. Parité logits/états et IDs pour MTP avant tout benchmark.
- Environnement : matériel/OS/outils/alimentation à relever. Mac M1–M4 non disponibles à ce stade ; une compilation et un fallback simulé ne constituent pas un test matériel. Source-level verifier SwiftUI/Metal non disponible ; ne pas appeler ces chemins prouvés. Aucune promesse de 100 % ou zéro bug.
- Intégration : audit/code local en cours ; aucune app v1.2.0 installée ou publication à ce stade.

### OPT-2026-10-04-UI-113 — réactivité pendant la réponse Desktop — en cours

- Demande : v1.1.3, copier le message entier et corriger les blocages pendant la génération en conservant animations, Markdown, coloration et mise en page.
- Antécédents lus : index du journal, OPT-2026-09-07-03/04/05, AUDIT-05, `docs/general-performance-2026-09-07.md`, `docs/audit-bridge-prefill-2026-09-22.md`. Le bridge regroupe déjà les deltas à 50 ms et les blocs de code ont un cache lexical. Le cache Markdown AUDIT-05 est validé localement mais absent de v1.1.2 ; les grands fences et le rendu réel restent ouverts (I02).
- Nouvelle raison : régression de réactivité signalée sur la v1.1.2 publiée ; mesurer le chemin AppKit/SwiftUI complet et les gros fences, et non refaire le seul microbenchmark AUDIT-05.
- Hypothèse : le redécoupage/reparsing de tout le fence actif et le parcours des glyphes à chaque image monopolise le thread principal ; confirmer avec une baseline avant de choisir le correctif. Aucune suppression d'animation prévue.
- Baseline : commit propre `a7ba6ab4664830668b8b413981cc3fb7bc164cbc`, v1.1.2 build 17, worktree isolé ; changements moteur locaux préexistants exclus. Mesures avant/après : non mesurées.
- Protocole prévu : Swift 6 optimisé avec SDK macOS 26.5, rendu des sources de production via NSHostingView, 64/256/1024 KiB de prose et de code, appends identiques après warmup ; coût par mise à jour et retard du thread principal, essais sans modèle GPU simultané. Conserver logs et cas négatifs dans `docs/measurements/desktop-v1.1.3/`. Comparaisons de rendu/parsing/Unicode, remplacement, outils, fin/annulation et copie exacte ; suite Desktop complète, CI, bundle/DMG vérifiés.
- Environnement : Apple M5 / 24 GiB ; alimentation, versions et charge concurrente relevées au lancement ; modèle/quantification/tokens : sans objet pour la mesure UI synthétique. FPS et débit d'inférence : non mesurés tant qu'un protocole distinct ne les mesure pas.
- Intégration : diagnostic en cours ; aucun code candidat, aucune app installée, aucune publication v1.1.3.
- Première baseline AppKit terminée : code 64/256/1024 KiB, médiane **5,140 / 11,076 / 33,521 ms** par append (p95 **5,828 / 11,696 / 33,972 ms**), 20 mesures après 2 warmups. La fenêtre de mesure couvre `append` et `layoutSubtreeIfNeeded`, pas tout le cycle d'affichage ; prose virtualisée **0,156 / 0,400 / 0,342 ms**, insuffisant pour mesurer tout son rendu. Preuve `docs/measurements/desktop-v1.1.3/baseline-appkit.log`. macOS 27.2, Swift 6.4, SDK 26.5, secteur 96 %, Deezer/WindowServer actifs ; aucun moteur GPU lancé par l'essai. Ne pas appeler ces valeurs FPS.
- Incidents du protocole : premier build du harnais refusé (chemin SwiftMath supposé au lieu du bin-path réel), puis lancement anticipé refusé avec exit 127 ; corrigés avant la mesure. Tentative `sample` après la fin du processus : aucune trace capturée, aucune conclusion de profilage. Candidat choisi : préparer le Markdown sur un acteur de fond, conserver les mêmes vues/animations, publier seulement le résultat encore courant ; reprendre le cache AUDIT-05 avec ses régressions, plus tests de changements rapides et de fin du streaming.
- Pilote candidat 1 terminé, **non concluant/rejeté pour intégration** : code 1 MiB ~2,785 ms, mais prose ~110,338 ms dans ce harnais. Compilation de test concurrente et cycle offscreen incomplet : ne pas revendiquer un gain. Log `candidate-appkit-pilot.log`. Correction prévue avant A/B final : chunks préparés `Equatable` pour garder l'identité des parties figées ; fenêtre native attachée et cadence 50 ms identique au bridge, timer de réveil relevé. Première régression copie échouée : NSHostingView sans NSWindow n'expose aucun bouton AX ; fenêtre de test privée ajoutée, investigation en cours.
- Candidat 2 avant essai ABBA : acteur de préparation, données de chunks `Equatable`, une tâche active et un seul snapshot en attente ; publier un préfixe encore valide permet de progresser même si le calcul prend plus d'un intervalle de delta. Remplacement/troncature et disparition invalident les résultats anciens. Tests ciblés **réussis** : copie byte-exacte, 56 parités de préparation, changements rapides, remplacement, disparition/réapparition, bridge de 20 000 lignes et historique. AX reste vide même avec NSWindow ; pas de clic AX revendiqué. Le contrôle natif est rendu, son action de production est testée sur presse-papiers privé. Mesure finale ABBA avec fenêtres natives attachées et cadence 50 ms, aucune compilation concurrente, **en cours**.
- Baseline native attachée encore active après 64 s à ~105 % CPU : capture `sample` de 5 s prévue sur cette première passe pour identifier le coût dominant. Cette passe profilée ne servira pas à comparer les chiffres définitifs ; la seconde baseline ABBA reste sans profiler. Conditions : secteur 100 %, aucun moteur modèle de l'essai.
- La seconde tentative de capture baseline arrive également après la fin du processus : échec `sample`, pas de trace. Baseline attachée 1 terminée ; gros pics prose (jusqu'à 3 416 ms) reproduits dans le cycle natif. Capture prévue maintenant sur le candidat 1 encore actif (25 s, ~100 % CPU) ; candidat 2 et baseline 2 fourniront la comparaison sans profiler.
- Capture candidat réussie : `candidate-sample.txt`, 5 s à 1 ms. Le thread principal passe surtout dans RenderBox/QuartzCore, dont allocation de surfaces ; déplacer le parsing seul laisse donc un coût de rendu des chunks hors écran. Candidat 3 prévu : rendre les chunks de réponse et de réflexion avec `LazyVStack` dans leurs scroll views existantes, mêmes textes, espacement, couleurs et renderer de fondu. Ce changement limite les surfaces aux contenus visibles, sans supprimer animation ni contenu. Refaire le protocole attaché pour cette nouvelle variante ; les mesures candidat 2 sont conservées comme essai distinct.
- Candidat 2, passe non profilée terminée : code 64/256/1024 KiB **0,496 / 0,981 / 2,612 ms** médianes, mais prose **3,438 / 64,994 / 250,203 ms**, retard timer p95 à 1 MiB **323,668 ms**. Preuve `candidate-attached-2.log`. **Rejeté comme correctif complet** : l'acteur seul ne règle pas le rendu hors écran. Une compilation du candidat 3 a chevauché cette campagne : ces valeurs restent un diagnostic, pas une comparaison finale isolée.
- Candidat 3 : deux passes attachées, sans compilation/profilage/modèle concurrents, **validé pour les charges UI mesurées**. Baseline attachée 2 → candidat final 2, code 64/256/1024 KiB médianes **2,885/8,968/34,093 → 0,521/1,034/3,117 ms** ; prose **17,643/82,288/67,778 → 0,737/5,233/3,827 ms**. Prose 1 MiB : p95 update **282,217 → 8,284 ms**, max **3 565,689 → 13,931 ms**, retard timer p95 **189,343 → 26,667 ms** (première passe finale : 76,946 ms de retard ; variabilité conservée). Logs `baseline-attached-2.log`, `final-attached-1.log`, `final-attached-2.log`, harnais `benchmark.swift`. Ces fenêtres de coût ne mesurent ni FPS ni gain modèle ; quelques pauses subsistent, aucune garantie sur tous contenus/Macs. La trace du candidat rejeté indiquait 4,6 GB d'empreinte physique ; RAM finale comparative **non mesurée**. Intégration : code local validé en benchmark, vérification complète/DMG/publication encore à faire.
- Correction de protocole : les deux passes du candidat 3 étaient isolées, mais le début de la baseline 2 peut avoir chevauché la compilation du candidat 3. Nouvelle raison explicite de répétition : une paire baseline 3 → candidat 3 après la fin de toutes compilations/tests/profiler, mêmes sources et harnais, secteur 100 %, autres apps laissées ouvertes. Les mesures précédentes sont conservées ; cette paire corrigée sera celle du rapport de release.
- Vérification complète locale **réussie** : `scripts/check-e2e.sh` (Python 2 réussis/4 optionnels ignorés, Desktop complet), lint Swift des nouveaux fichiers, plutil, Rust fmt/clippy/tests avec et sans default-features (35 réussis par configuration), Kani 17 harnais/0 échec. La mutation isolée qui ne copie que 5 caractères échoue bien sur la vérification indépendante texte/Unicode ; source de production jamais remplacée. La première invocation Cargo échouait faute de PATH ; relance par chemin explicite réussie. Swift 6 valide l'isolation de l'acteur et les données Sendable ; aucune preuve formelle de SwiftUI/clipboard/animation revendiquée.
- Paire corrigée isolée **terminée et validée** : code 64/256/1024 KiB médianes **2,754/8,983/34,190 → 0,606/1,241/3,145 ms** ; prose **17,608/78,104/66,233 → 0,862/3,684/0,965 ms**. Prose 1 MiB p95 **284,232 → 8,137 ms**, max **3 460,541 → 74,232 ms**, retard timer p95 **186,709 → 75,199 ms**. Les pauses restantes ne sont pas cachées. Preuves `baseline-isolated.log` et `final-isolated.log`, protocole et limites dans `docs/desktop-v1.1.3-validation.md`. Statut performance : **validé sur ces charges**, source final testé ; app installée non remplacée, commit/DMG/GitHub encore à faire.

### OPT-2026-09-12-RUST-01 — Port natif Rust/Metal — socle validé, migration en cours

- Demande : réécriture Rust sur une nouvelle branche GitHub. Branche
  `codex/rust-rewrite`, base `de318a8`, checkout isolé `../mlxl3-rust`.
- Hypothèse : une orchestration native peut réduire le coût CPU ; aucun gain
  de decode, prefill, TTFT ou RAM n'est établi par le changement de langage.
- Antécédents lus : journal complet, audit-runtime-2026-09-04 et roadmap decode.
  Les shaders MSL existants restent la référence ; ne pas changer leur calcul
  et leur ordonnanceur simultanément pour revendiquer un gain.
- Premier contrôle : codec CPU Rust (K=1..8, trois codebooks), lecture des
  checkpoints sans allocation GPU, registre compatible, découpage thinking,
  puis packing/décodage/QMV Metal appelés directement par Rust sans Python.
- Protocole : tests CPU déterministes, parité différentielle Python/Rust,
  tests Metal sur M5 avec erreurs propagées, tests CLI en dossiers temporaires.
  Aucun benchmark modèle complet avant les architectures et caches natifs.
- Environnement : M5/macOS local, Rust stable installé sans changer le PATH
  du shell. Versions exactes et résultats à consigner après exécution.
- Résultats/performance : non mesurés. Intégration : chantier de branche ;
  moteur de production, modèles et app installée inchangés.
- Première validation : cinq tests unitaires CPU puis test GPU exhaustif des
  196 608 codewords, K=1..8 pack/unpack réussis. Release compilée Rust 1.98.1.
  Douze contrats checkpoint/registre/CLI supplémentaires réussis. Pas de mesure
  de débit. Le test différentiel Python+Metal suivant est interrompu au premier
  accès GPU par le nouveau sandbox (`no Metal device`) ; relance hors sandbox
  nécessaire pour cette validation matérielle, mêmes données et même code.
- Checkout déplacé dans `work/mlxl3-rust` pour respecter les droits d'écriture
  actuels, branche inchangée. Route native MLX étudiée : libmlx 0.32.2 déjà
  installée, aucune dépendance Python du dylib. FFI mince Rust/C++ en cours
  pour conserver les kernels/fusions de production lors du portage modèle.
- Contrôle différentiel direct Metal, relancé avec accès GPU : **124 cas
  réussis**, comprenant 196 608 codewords CPU, tous K pack/unpack, codecs GPU
  et 72 QMV non nuls en FP16. Comparaison bit-à-bit avec les références Python.
  Le risque de différence FMA MUL1 trouvé à la lecture n'est pas reproduit dans
  cette matrice exécutée avec le compilateur Metal actuel. Pas un test modèle.
- Port du registre : parent relatif/override vide corrigés, CRLF coupé entre
  fragments corrigé dans Rust et dans la référence Python sur cette branche.
  **14 contrats Rust** et **10 parités de streaming Python/Rust** réussis.

### OPT-2026-09-12-RUST-02 — Couches et LFM2 via MLX natif — parité LFM2 validée

- Objectif : supprimer Python de l'orchestration, réutiliser les sources QMV
  de production à l'identique via libmlx 0.32.2 et une FFI Rust/C++ limitée.
  Aucun nouveau kernel mathématique pour revendiquer artificiellement un gain.
- Baseline : branche Python de `de318a8`, mêmes poids/sources MSL/MLX local.
  Candidat : feature Rust `mlx`, couches EXL3 sérialisées et architecture LFM2
  dense. Qwen, Gemma, quantification et GUI ne sont pas encore portés.
- Protocole : comparaison bit-à-bit de projections synthétiques tous K/CB,
  puis tokens imposés LFM2.5-1.2B-Thinking EXL3 4bpw, logits et caches à chaque
  étape. Première vérification token-par-token des deux côtés ; ce n'est pas
  une comparaison du prefill groupé Python à un nouveau prefill Rust.
- Conditions : même M5/macOS/MLX ; les tests matériels exigent accès Metal hors
  sandbox. Contexte/longueur imposés dans les commandes de preuve. Aucun gain
  de performance et aucune parité modèle encore établis à cette étape.
- Smoke GPU FFI `array::tests::native_array_and_kernel_smoke` : réussi,
  incluant transpose non contiguë, matmul, conversion FP16, kernel et erreurs.
  Build complet ensuite bloqué par `metal::device_info` non exporté du dylib ;
  détection M5 déplacée vers Metal natif. Validation complète à reprendre.
- Build `mlx,chat` réussi avec cible macOS 26.2. `check_parity.py --mlx`
  réussit **92 cas exacts** : 20 codecs CPU et 72 projections complètes
  (128×128, 1024×512, 2048×128 ; K1..8 × CB0..2, split-K inclus).
  Cinq tests tokenizer réussis dont prompt, IDs et décodage strictement égaux
  à Transformers pour une conversation LFM2.5-1.2B multilingue de quatre tours.
  Prochaine validation : modèle complet, huit tokens imposés du protocole.
- Première parité modèle : **rejetée**, step 0, cache couche 10 (7 octets
  différents sur 1024, écarts d'un bit). L'ordre lexicographique visitait 10
  avant 2 : correction du diagnostic pour identifier la première couche
  divergente en ordre d'exécution. Ce n'est pas une tolérance assouplie.
- Tri corrigé : couches 0..9 exactes, première divergence confirmée couche 10.
  Hypothèse : QKV groupé Python choisit un split-K différent des projections
  Rust séparées (512 sorties K/V vs 3072 groupées). Diagnostic une fois avec
  groupement Python désactivé ; son succès ne vaudrait pas parité production.
- Diagnostic confirmé : tous les logits et caches exacts au premier token
  sans groupement Python. Port du groupement QKV et de son shader mapped
  inchangé ; même split-K et même profondeur SIMD que la baseline groupée.
  Revalidation prévue des huit tokens contre production (groupement actif).
- **Validé numériquement** : groupement QKV natif intégré, huit tokens
  `1,2,3,19,225,4096,17,7`, tous logits et états KV/ShortConv bit-à-bit égaux
  au moteur Python de production. `cargo clippy --features mlx,chat
  --all-targets -- -D warnings` réussi. Preuve reproductible :
  `PYTHONPATH=src .venv/bin/python native/check_model_parity.py MODEL
  --binary target/debug/mlxl3-rs` (Python du dépôt parent pour ce worktree).
- Prochain contrôle fonctionnel : build release, génération greedy CLI sur
  le même LFM2, conversation suivie et `/clear`. Timing affiché expérimental,
  pas de benchmark comparable ni de gain revendiqué (prefill séquentiel).
- Chat réel : première réponse achevée « Bonjour. », 180 tokens, streaming
  thinking/réponse correct. Deux tours puis `/clear` et `/exit` réussis ; le
  second tour mentionne le prénom du premier mais atteint la limite 256.
  26 tests Rust passés (12 unitaires + 14 contrats), 11 checks Python passés.
  Le garde de provenance avait d'abord échoué sur un seul saut de ligne final ;
  ignore désormais uniquement les espaces/sauts finaux, pas le contenu des shaders.
- Extension de validation : 42 projections groupées K1..6/8 × trois CB × deux
  formes (dont QKV 2048/512/512) contre production ; 72 projections simples
  répétées pour vérifier l'intégration. Contrôle des arrêts Unicode ajouté au
  tokenizer ; génération CLI vidant le suffixe UTF-8 à la limite de tokens.
- Matrice étendue **validée : 134 cas exacts** (20 CPU + 72 projections
  simples + 42 groupes, chacun comprenant deux/trois sorties). Arrêts Unicode
  testés à chaque position d'une chaîne accentuée avec emoji : réussite.
  Inspection I16 de trellis ajoutée comme dans Python, avec contrat dédié ;
  bornes de grilles Metal protégées contre l'overflow.
- Contrôle de généralisation prévu : LFM2.5-2.6B EXL3 4bpw, mêmes huit tokens
  imposés, mêmes checks logits/caches, pour ne pas valider une seule taille.
- Généralisation **validée** : LFM2.5-2.6B EXL3 4bpw, huit étapes, tous les
  logits et tous les états bit-à-bit exacts contre production. Aucun changement
  spécifique de modèle requis. Clippy complet revalidé après les derniers
  gardes de bornes et d'I16. État : moteur Rust expérimental local, pas installé
  dans l'app et aucun gain temporel comparatif revendiqué.
- Vérification finale : 26 tests Rust réussis, lint sans warnings de notre
  code, tests de tokenizer/Unicode et provenance MSL réussis. Dépendance
  `block 0.1.6` signale une incompatibilité future Rust (pas une erreur actuelle).
  Build release actualisé et arrêt borné CLI revalidés avant publication.
  Aucun benchmark GPU ou quantificateur ne reste en cours. Les prochains ports
  (Qwen/Gemma/MoE, prefill groupé, quantification, GUI) restent à réaliser.

### OPT-2026-09-12-RUST-03 — Primitives Qwen3.5 MoE — en cours

- Hypothèse / changement : porter d'abord les deux primitives qui structurent
  le checkpoint local `Qwen3.6-35B-A3B-EXL3-2.49bpw` : mise à jour récurrente
  Gated DeltaNet Dk/Dv=128 et routeur 256→top-8. Réutiliser les shaders de
  production et la FFI MLX existante, sans nouveau backend ni copie CPU.
- Antécédents : `src/mlxl3/recurrent.py`, `src/mlxl3/moe.py`, implémentations
  `mlx_lm.models.qwen3_5` et `gated_delta.py` relues. Le checkpoint est bien
  Qwen3.5-MoE texte : 40 couches (30 linéaires, 10 attention), 256 experts,
  top-8, état récurrent FP32. MTP exclu conformément aux choix précédents.
- Baseline / candidat : Python `load_exl3_model` commit `de318a8` contre
  branche Rust `59ecef4`, mêmes entrées déterministes et même libmlx 0.32.2.
- Protocole prévu : parité bit-à-bit sortie + état GDN sur état nul et non nul,
  puis indices/scores du routeur y compris égalités/NaN ; enfin intégration
  token imposé et comparaison couche par couche. Aucun benchmark avant parité.
- Environnement : M5, macOS 27.0/SDK cible 26.2, GPU local. Résultats : non
  mesurés côté performance. Première étape validée le 12 septembre : le shader
  Gated DeltaNet empaqueté compile via libmlx 0.32.2 et ses sorties FP16 ainsi
  que son état FP32 sont bit-à-bit identiques à `mlx_lm` sur état nul et non
  nul. Commande : `native/check_parity.py --binary target/debug/mlxl3-rs
  --mlx`, 142/142 cas réussis (134 projections existantes + 2 GDN + 6 routeur).
  Le routeur natif restitue exactement indices et scores FP16 avec/sans
  normalisation, y compris égalités, zéros signés et NaN. L'intégration d'une
  couche Qwen complète reste non réalisée. Essai suivant enregistré avant
  modification : porter le QMV expert mappé déjà utilisé par Python (K=2/3/4,
  matrices synthétiques, routes répétées gate/up, sortie brute FP32 puis sortie
  FP16), et exiger la parité bit-à-bit avant le SwiGLU fusionné. Résultat :
  validé bit-à-bit sur les six variantes ; la matrice globale passe désormais
  148/148 cas. Le prochain essai sera le prepare SwiGLU+Hadamard puis la
  réduction pondérée du down, toujours face aux helpers Python inchangés.
  Première compilation interrompue avant benchmark : le générateur Rust des
  sept étages Hadamard avait deux références `str` aux durées de vie distinctes
  lors de leur permutation ; aucune exécution GPU ni mesure. Correction prévue :
  une durée de vie commune, sans changement du shader.
  Deuxième exécution validée : prepare SwiGLU/down et réduction pondérée sont
  bit-à-bit identiques aux helpers MLXL3 Python ; matrice 150/150. Essai suivant
  enregistré : chaîner les deux QMV mappés et ces transforms dans une structure
  `Exl3SwitchGlu`, puis comparer sa sortie finale à `EXL3SwitchGLU` sur un bloc
  synthétique top-2. Résultat : validé bit-à-bit, matrice 151/151 ; le chemin
  expert complet reste entièrement sur MLX/Metal. Aucun benchmark de vitesse
  avant la parité d'une couche réelle. Essai réel enregistré avant modification :
  charger uniquement le MLP MoE de la couche 0 du checkpoint Qwen local, entrée
  FP16 déterministe `[1,2048]`, et comparer sa sortie au même assemblage Python
  (gate dense, top-8, experts EXL3, expert partagé) bit-à-bit ; aucune génération
  ni chargement simultané de deux modèles complets.
  Première compilation de ce jalon interrompue : référence vers un nom de
  scale legacy temporaire dans le loader Rust ; aucune donnée modèle chargée
  et aucune mesure. Correction limitée à posséder la chaîne avant l'appel.
  Deuxième exécution validée le 12 septembre : `native/check_qwen_moe.py`
  compare le MLP MoE réel de la couche 0 et obtient une égalité FP16 bit-à-bit.
  Le script ne matérialise côté Rust qu'une couche d'experts, et aucune mesure
  de débit n'est revendiquée. Prochaine étape : bloc Gated DeltaNet complet de
  couche 0 avec état nul/non nul, avant assemblage des 40 couches. Protocole
  enregistré : deux entrées FP16 déterministes `[1,1,2048]`, mêmes poids réels,
  comparaison bit-à-bit des sorties, du cache convolutionnel FP16 et de l'état
  récurrent FP32 après chaque token ; arrêt au premier désaccord.
  Résultat : validé sur les deux tokens avec égalité bit-à-bit de la sortie et
  des deux caches (`native/check_qwen_gdn.py`). Essai suivant enregistré :
  assembler la couche linéaire 0 complète avec les deux RMSNorm corrigées par
  le sanitizer Qwen (`weight + 1`), les résidus et le MLP MoE déjà validé ; une
  entrée réelle déterministe, comparaison FP16 exacte avant tout benchmark.
  Résultat : couche linéaire 0 validée bit-à-bit, sortie et deux caches inclus
  (`native/check_qwen_layer.py`). Essai suivant enregistré : couche attention
  complète 3 sur deux tokens, RoPE partiel 64/256 à offsets 0 puis 1, caches KV,
  résidus et son MLP MoE ; égalité bit-à-bit requise, débit non mesuré.
  Première exécution interrompue dans l'oracle Python avant calcul : l'API
  `mx.fast.rope` 0.32.2 exige `scale=1.0` explicite. Aucun résultat candidat ;
  protocole inchangé après correction de l'appel de référence.
  Deuxième exécution validée : couche attention 3 exacte sur deux tokens,
  sorties FP16 et caches K/V compris (`native/check_qwen_attention.py`). Les
  deux types de couche du modèle sont donc couverts isolément. Essai suivant
  enregistré : assemblage des 40 couches, embedding/norm/head puis comparaison
  couche par couche et logits pour un token imposé ; surveiller la RAM processus
  pendant le chargement et ne pas lancer de benchmark de vitesse avant parité.
  Premier essai complet rejeté : oracle Python produit en 7,57 s avec empreinte
  mémoire pic rapportée 12,95 GB ; candidat Rust contrôlé en 66,32 s via le
  processus Python de comparaison, mais 239045/248320 logits FP16 diffèrent.
  Les chiffres mémoire du wrapper ne couvrent pas correctement tous les enfants
  et ne sont pas comparables. Aucune conclusion de performance. Diagnostic
  enregistré : tracer les sorties après chaque couche dans deux processus
  successifs et identifier la première divergence, sans modifier les tolérances.
  Une relance du diagnostic a été interrompue avant chargement : `python`
  n'est pas présent dans le `PATH` de ce worktree. Aucun calcul ni résultat ;
  relance inchangée avec l'interpréteur `.venv` absolu du dépôt parent. Cette
  relance localise la première divergence dès `layer_0` : 1 923/2 048 valeurs
  FP16 diffèrent, première valeur 40979 contre 40981 en représentation brute.
  Le test isolé de cette même couche étant exact, la prochaine vérification
  compare son entrée et les chemins exacts des deux oracles avant tout patch.
  Essai enregistré avant modification : inclure l'embedding du token 1 dans
  les deux traces, exiger son égalité bit-à-bit, puis conserver le diagnostic
  couche par couche inchangé. Si l'embedding est exact, comparer les états de
  couche 0 et l'effet des frontières d'évaluation, sans toucher aux kernels.
  Résultat : embedding exact ; la première divergence reste `layer_0` avec
  les mêmes 1 923/2 048 valeurs. Le loader et la sélection de token sont donc
  écartés. Prochaine comparaison : appel réel de `Qwen3_5MoeDecoderLayer`
  contre l'oracle manuel isolé, en inspectant notamment le masque SSM et les
  conversions de dtype ; aucun changement de tolérance ni benchmark prévu.
  Inspection terminée : le masque initial est bien `None` et les dtypes sont
  FP16, mais `fuse_compatible_linear_groups` groupe en production Q/K/V,
  GDN QKV/Z et gate/up partagé. Les oracles isolés et le candidat Rust les
  exécutaient séparément, ce qui explique qu'ils soient exacts entre eux mais
  pas face au modèle chargé. Candidat enregistré : réutiliser `Exl3Group` via
  un seul helper de projections groupées/fallback, puis rerun de la trace dès
  la couche 0 ; exiger ensuite les 40 couches et logits exacts. Premier rerun
  après groupement toujours rejeté, avec exactement 1 923 divergences dès la
  couche 0. Vérification suivante enregistrée : rejouer l'ancien oracle manuel
  sur entrée aléatoire pour confirmer que le groupement Rust est réellement
  sélectionné, puis tracer les sorties internes QKV/Z/MLP face au modèle Python.
  L'oracle aléatoire reste exact après le patch, donc le helper groupé ne casse
  pas ce cas. Sa variante avec l'embedding réel s'est arrêtée avant calcul GPU :
  NumPy ne sait pas importer directement le buffer BF16 brut. Aucun résultat ;
  convertir explicitement par MLX en FP16 comme le loader de production.
  Relance corrigée : sortie et caches de couche 0 toujours exacts sur l'embedding
  réel face à l'oracle manuel. Essai diagnostic suivant enregistré : générer une
  trace Python production avec seulement le groupement GDN QKV/Z désactivé par
  son option existante, puis comparer au même Rust. Cela isole cette fusion sans
  charger deux modèles simultanément ni modifier le candidat. Résultat : même
  divergence couche 0 ; cette fusion est écartée. Un diagnostic ponctuel des
  poids de norme a ensuite trouvé la cause : 1 723/2 048 coefficients de la
  norme d'entrée et 1 576/2 048 de la post-norme diffèrent. Le sanitizer Python
  calcule `BF16 + 1` puis convertit en FP16 ; l'oracle manuel et Rust faisaient
  `BF16 -> FP16` puis `+1`. Candidat enregistré : helper Qwen unique reproduisant
  l'ordre du sanitizer sur toutes les normes concernées (entrée/post, Q/K et
  finale), puis trace complète et logits bit-à-bit ; garder les groupes EXL3
  puisqu'ils reproduisent le graphe de production. Résultat après correction :
  embedding et couches 0 à 4 exacts ; première divergence déplacée à la couche
  linéaire 5, 1 013/2 048 valeurs, première 8974 contre 8972. Le correctif de
  sanitizer est donc validé sur les deux types de couche. Essai suivant
  enregistré : charger seulement la couche 5 Rust, lui fournir exactement la
  sortie Python de la couche 4 et comparer à `layer_5`, puis inventorier K/CB
  de ses projections contre les couches exactes ; caches initiaux nuls.
  Résultat ciblé : même divergence 1 013/2 048, donc l'état des couches
  précédentes est écarté. L'inventaire couche 0/1/2/4/5/6 est identique sur
  les projections structurantes (GDN et partagé K4/MCG, experts K3/MCG).
  Essai suivant enregistré : exposer seulement dans l'opération codec de
  diagnostic les cinq frontières de la couche 5 (norme entrée, GDN, résidu,
  post-norme, MLP), produire les mêmes frontières Python et arrêter à la
  première divergence. Le chemin normal conserve une seule implémentation.
  Première compilation interrompue : la méthode de trace a été insérée sur
  l'autre type de couche portant le même `forward`, donc `LinearLayer::trace`
  est absent. Aucun modèle ni GPU exécuté. Déplacer ce refactor dans
  `LinearLayer` et restaurer l'autre couche, sans changer le protocole.
  Deuxième compilation et trace ciblée réussies : norme d'entrée couche 5
  exacte ; première divergence dans la sortie Gated DeltaNet, 977/2 048
  valeurs (première position 7). Résidu/MLP non interprétés après ce point.
  Essai suivant enregistré : comparer les références Python groupée/non
  groupée déjà produites et tracer QKV/Z, convolution puis update récurrente
  de la GDN couche 5. Cela départage projection, convolution et kernel d'état.
  Résultat : QKV, Z, A/B, convolution, Q/K/V normalisés, beta, softplus, decay,
  sortie récurrente et RMSNorm sont tous exacts. Première divergence à la
  sortie finale de la GDN. Essai suivant enregistré : tracer le produit gated
  FP16 juste avant `out_proj`; s'il est exact, isoler `out_proj`, sinon corriger
  l'ordre précis SwiGLU. Pas de modification du calcul avant ce résultat.
  Résultat : une seule valeur gated diffère sur 4 096 (index 3 242), puis son
  amplification par `out_proj` explique les 977 écarts. Essai suivant enregistré :
  rejouer sur Z/RMS exacts l'expression MLX inline, `nn.silu` compilé et le helper
  Qwen compilé, comparer leurs bits à la référence et au Rust. Corriger ensuite
  la frontière de compilation, pas le QMV déjà établi exact. Résultat diagnostic :
  expression inline 43383 contre référence 43382 à l'index 3 242 ; `nn.silu`
  compilé et helper Qwen tous deux exacts. Le bridge utilise désormais cette
  frontière compilée ; la couche 5 ciblée repasse entièrement bit-à-bit exacte.
  Essai suivant enregistré : trace complète des 40 couches et logits avec ces
  deux corrections, arrêt au premier écart ; aucune mesure de performance.
  Résultat : embedding, 40 couches, norme finale et 248 320 logits sont tous
  bit-à-bit exacts (`native/diagnose_qwen_trace.py`). Répétition finale prévue
  avec `native/check_qwen_model.py`, oracle logits indépendant déjà produit,
  pour vérifier le contrat public sans données de diagnostic additionnelles.
  Répétition réussie : 248 320/248 320 logits FP16 exacts pour le token 1,
  66,85 s de temps mur en build debug ; ce temps inclut le chargement et n'est
  pas un benchmark d'inférence. Essai suivant enregistré : séquence imposée
  `1,2,3` dans une seule instance Python puis Rust, logits exacts à chaque pas,
  afin de valider caches GDN/KV et offsets avant branchement au chat natif.
  Résultat : trois étapes, chacune 248 320 logits FP16, toutes exactes. Les
  caches GDN/KV et offsets natifs sont validés sur cette séquence courte.
  Étape fonctionnelle enregistrée : ajouter `reset` aux caches Qwen et choisir
  LFM2/Qwen par `model_type` dans l'unique boucle de chat existante, sans dupliquer
  tokenizer/streaming/sampling. Smoke release Qwen borné à quelques tokens,
  puis `/clear` seulement si le smoke non interactif réussit. Première validation :
  26 tests Rust réussis, mais Clippy interrompt la chaîne avant build release sur
  deux variantes d'enum trop grandes (`ProjectionBundle`, `Layer`). Aucun smoke
  modèle lancé. Correction prévue : boxer seulement les variantes lourdes comme
  indiqué par le lint, puis relancer lint/build/tests sans changer le graphe MLX.
  Deuxième lint encore interrompu avant build : après avoir boxé la variante
  linéaire, la variante attention est devenue la plus grande. Boxer les deux
  variantes de `Layer`; aucune exécution modèle ni mesure entre ces deux lints.
  Lint strict et build release réussis après les deux boxes. Smoke non interactif
  Qwen réussi : prompt « Salut », 4 tokens générés, streaming thinking actif,
  fin propre. Mesures indicatives seulement (prefill séquentiel) : 2,4 tok/s,
  decode 10,2 tok/s, TTFT 5 094 ms après chargement ; un seul run, aucune
  comparaison Python. Dernier contrôle fonctionnel prévu : `/clear` entre deux
  prompts courts dans le même processus, pour vérifier le reset des 40 caches.
  Contrôle réussi : deux prompts de 2 tokens générés séparés par `/clear`, puis
  `/exit`; aucun crash, état résiduel ni erreur. Les débits courts 8,3/8,5 tok/s
  ne constituent pas un benchmark. Avant publication du jalon : corriger les
  anciens oracles ciblés qui reproduisaient l'ancien ordre FP16/+1, relancer
  leurs parités, tests Rust, Clippy et provenance.
  Validation finale réussie : anciens oracles corrigés pour l'ordre BF16/+1 et
  le SwiGLU précis ; GDN couche 0 sur deux pas, couche linéaire 0, attention
  couche 3 sur deux pas et MoE couche 0 tous bit-à-bit exacts. Suite codec/MLX
  **151/151**, tests Python **11/11**, tests Rust **26/26** (plus 3 tests matériel/
  tokenizer ignorés explicitement), Clippy strict et build release réussis.
  Le chat Qwen natif, son streaming et `/clear` sont donc validés localement.
  Intégration : prototype de branche uniquement ; GUI/app installée et moteur
  Python de production inchangés. Prefill Rust encore séquentiel, aucun gain de
  vitesse revendiqué ni comparé au moteur existant.

### OPT-2026-09-12-RUST-04 — LFM2 MoE natif — en cours

- Hypothèse / changement : étendre l'unique implémentation LFM2 Rust au type
  `lfm2_moe`, en conservant opérateurs, caches, chat et couches denses existants.
  Réutiliser `Exl3SwitchGlu` pour les couches expertes ; ajouter seulement les
  noms LFM `w1/w3/w2` et la sélection top-k biaisée requise par l'architecture.
- Antécédents consultés : `mlx_lm.models.lfm2_moe`, `src/mlxl3/moe.py`, port
  LFM dense validé dans RUST-02 et primitives MoE exactes de RUST-03. Le modèle
  local est LFM2.5-8B-A1B, 24 couches, 32 experts/top-4, deux couches denses,
  biais expert et normalisation des scores.
- Baseline / candidat : moteur Python de `a654a64` contre branche Rust au même
  commit, checkpoint EXL3 3.10 bpw local. Aucun kernel expert nouveau : même
  chemin Metal déjà validé sur Qwen, avec routeur biaisé 32 voies.
- Protocole prévu : d'abord bloc MoE réel couche 2 (routes, scores et sortie
  FP16 exacts), puis token imposé couche par couche et enfin plusieurs tokens
  avec tous caches. Build/chat seulement après parité ; aucun benchmark ni gain
  revendiqué pendant ce port. Environnement : M5/macOS, MLX 0.32.2, alimentation
  et thermique non contrôlées puisque seules des comparaisons exactes sont prévues.
- Première primitive validée : routeur biaisé LFM 32 voies/top-4, indices et
  scores FP16 bruts bit-à-bit identiques au kernel Python. La matrice MLX passe
  **152/152** cas. Le bloc LFM utilise ensuite les primitives expertes déjà
  validées, avec normalisation et facteur effectués dans le même ordre MLX ;
  prochaine preuve : MoE réel couche 2 avant toute exécution du modèle complet.
- Révision du protocole avant exécution : ne pas ajouter un codec de diagnostic
  permanent uniquement pour ce bloc. Le checker LFM complet existant exerce le
  même chemin et compare tous logits/caches ; commencer par un seul token. En
  cas d'écart seulement, ajouter une trace éphémère couche 2 pour localiser la
  divergence, puis la retirer. Cela réduit le code de test sans relâcher le
  critère bit-à-bit.
- Premier modèle complet réussi : token imposé `1`, tous les logits FP16 et
  tous les caches conv/KV du LFM2.5-8B-A1B EXL3 3.10 bpw sont bit-à-bit égaux
  au moteur Python. Aucune trace supplémentaire n'est donc ajoutée. Répétition
  suivante enregistrée : huit tokens du protocole LFM dense, même instance et
  états conservés, afin de valider routing changeant et progression des caches.
- Séquence complète réussie : `1,2,3,19,225,4096,17,7`, huit sorties vocabulaire
  et tous les états des 24 couches bit-à-bit exacts. Le port LFM MoE est donc
  validé numériquement sur ce checkpoint. Étape suivante enregistrée : Clippy,
  build release, chat borné puis `/clear`; timings purement indicatifs puisque
  le prefill natif reste token-par-token.
- Validation fonctionnelle réussie : Clippy strict, 26 tests Rust et build
  release passent. Chat LFM MoE borné à 4 tokens puis deux prompts séparés par
  `/clear` terminent proprement. Le smoke isolé a affiché ~64,8 tok/s decode et
  548 ms TTFT ; séquences trop courtes, sans paire Python, donc aucune conclusion
  de performance. Intégration : moteur/CLI Rust de branche seulement ; GUI,
  quantification et app installée inchangées.

### OPT-2026-09-12-RUST-05 — Qwen3.5 dense natif — en cours

- Hypothèse / changement : accepter le `qwen3_5` dense du checkpoint local
  Qwen3.8-27B 2.75 bpw en réutilisant intégralement attention, Gated DeltaNet,
  caches, normes et head du port Qwen MoE exact. Seul le MLP devient une variante
  gate/up groupée + SwiGLU + down ; aucune logique vision ni MTP.
- Antécédents consultés : `mlx_lm.models.qwen3_5`, Qwen3NextMLP, RUST-03 et
  inventaire réel du checkpoint. Les 64 couches ont le même cycle trois GDN/
  une attention et les poids conv non sanitisés exigent le même `weight + 1`
  déjà corrigé. Le checkpoint Gemma n'est plus présent, donc aucun port Gemma
  non vérifiable n'est tenté maintenant.
- Baseline / candidat : production Python du commit `ecf73a4`, même checkpoint
  local de 12 GB, contre moteur Rust sur cette branche. Protocole prévu : oracle
  Python écrit sur disque puis processus libéré, Rust ensuite, afin de ne pas
  garder deux modèles de 12+ GB simultanément. Token 1 puis séquence 1/2/3,
  logits FP16 bit-à-bit ; chat seulement après succès. Performance non mesurée.
- Premier contrôle réussi : oracle Python écrit en processus séparé, puis
  248 320/248 320 logits Rust exacts pour le token 1. Aucun modèle concurrent
  ni comparaison de temps (le candidat était un build debug). Prochaine
  répétition enregistrée : séquence imposée 1/2/3 dans une instance de chaque
  moteur, toujours séquentiellement, pour exercer caches GDN/KV et offsets.
- Séquence stateful réussie : trois fois 248 320 logits FP16 exacts. Les caches
  GDN/KV et offsets du Qwen3.8 dense sont donc validés indirectement à chaque
  étape sans conserver deux modèles en RAM. Étape suivante enregistrée : lint,
  tests, build release, chat court et reset ; aucune mesure comparative.
- Première chaîne finale interrompue par Clippy avant tests/build : la variante
  MoE de l'enum MLP est ~984 octets contre ~248 pour la dense. Boxer uniquement
  la variante MoE comme recommandé, puis relancer la même chaîne ; aucun modèle
  ni benchmark exécuté pendant cet échec.
- Deuxième lint encore interrompu : après ce box, la variante dense de 248 octets
  dépasse à son tour la petite variante. Boxer aussi la dense, comme pour l'enum
  de couches Qwen déjà validé ; aucun changement du graphe MLX.
- Après les deux boxes : Clippy strict, 26 tests Rust et build release réussis.
  Chat Qwen3.8 dense borné à quatre tokens terminé avec streaming thinking.
  Mesures indicatives défavorables : 7,1 tok/s prefill séquentiel, 5,4 tok/s
  decode, TTFT 45,2 s lors de cette première compilation/instance. Ce n'est pas
  un gain ; le port est correct mais pas encore performant face au moteur Python.
  Dernier contrôle fonctionnel prévu : deux prompts d'un token avec `/clear`.
- `/clear` validé : deux générations d'un token terminées dans le même processus,
  sans crash ni état résiduel observable. TTFT 43,1 puis 47,0 s, confirmant que
  le reset invalide/reconstruit aujourd'hui des graphes coûteux ; aucun benchmark
  comparable. Avant publication du jalon, répéter la séquence Qwen MoE 1/2/3
  avec son oracle conservé pour vérifier que l'enum MLP partagé ne régresse pas.
- Régression Qwen MoE réussie : trois étapes et tous les logits bit-à-bit exacts
  avec le build release. État : Qwen dense intégré au moteur/CLI Rust de branche,
  GUI et app installée inchangées ; aucune optimisation de ses temps encore faite.

À lire **avant** toute optimisation ; à mettre à jour **avant et après chaque
essai**, y compris les essais ratés. Voir [AGENTS.md](AGENTS.md).

## Format des nouvelles entrées

```text
### OPT-AAAA-MM-JJ-NN — Nom — statut
- Hypothèse / changement :
- Antécédents consultés / raison de retester, le cas échéant :
- Baseline / candidat (commit + diff ou options) :
- Environnement : matériel, OS, dépendances, alimentation, conditions connues.
- Protocole : modèle/bpw, shapes ou contexte, tokens, warmup, répétitions.
- Commandes et preuves :
- Résultats avant → après, unités et dispersion :
- Qualité / exactitude vérifiée ; contrôles non effectués :
- Conclusion / limites / prochaine condition de réexamen :
- Intégration : prototype / code local / app / publication, selon vérification.
```

## Historique à consulter avant de proposer une piste

Journal initialisé le 10 septembre 2026 à partir des rapports existants.
**Ce résumé n'est pas une transcription exhaustive des anciens essais.**
Pour les domaines concernés, lire également ces archives et rechercher la
piste dedans ; elles restent la source des protocoles et résultats détaillés.

| Domaine | Rapports existants |
| --- | --- |
| Decode général, Qwen, LFM | [10 septembre](docs/decode-investigation-2026-09-10.md), [R&D du 7 septembre](docs/general-performance-2026-09-07.md), [roadmap decode](docs/decode-roadmap-2026-09-04.md), [audit runtime](docs/audit-runtime-2026-09-04.md) |
| Qwen3.8 dense | [Essais decode Qwen3.8](docs/qwen38-decode-local.md) |
| Gemma | [Decode](docs/gemma-decode-investigation.md), [SDPA512](docs/gemma-sdpa512-investigation.md) |
| Prefill après outils/MCP | [R&D prefill](docs/tool-prefill-rd-2026-09-05.md) |
| Quantification EXL3 Metal | [Optimisations quantification](docs/metal-quantization-optimization.md), [LFM2.6](docs/lfm26-local-quantization.md), [Ling](docs/ling-local-quantization.md) |
| UI, streaming, sessions | [R&D du 7 septembre](docs/general-performance-2026-09-07.md), [audit v1](docs/audit-v1-2026-09-06.md), [validation v1](docs/v1-validation.md) |

Les statuts ci-dessous décrivent les conclusions des rapports à leur date,
pas une vérification de la version actuellement installée ou publiée.
Les chemins `build/` sont des preuves locales temporaires, non garanties dans Git.

## 10 septembre 2026 — recherche decode générale

Source commune : [rapport complet et preuves](docs/decode-investigation-2026-09-10.md).
M5, 24 GiB, macOS 27.0 (26A428), MLX 0.32.2. Trois paires alternées,
96 tokens générés, warmup exclu. Batterie puis secteur, thermique non contrôlée.
Les différences ci-dessous sont les médianes des variations appariées.
Texte identique dans les paires ; pas de nouvelle validation bit-à-bit des
logits/caches. **Aucun changement de production retenu dans cette série.**

### OPT-2026-09-10-01 — Chargement TensorOps — bloqué

Le chargement normal de Qwen échoue au warmup sur
`get_destination_cooperative_tensor` (contrainte de template dans les headers
MetalPerformancePrimitives). Cause exacte non établie. Pour les essais 02–06,
les deux côtés utilisent `MLXL3_TENSOR_QMM=0 MLXL3_TENSOR_SEGMENTED_QMM=0`.
Ce contournement de benchmark n'est pas activé dans l'app ; les mesures ne
valident pas le chemin prefill TensorOps normal. Réexaminer après correction
de compatibilité macOS 27.

### OPT-2026-09-10-02 — Budgets command-buffer, Qwen — rejeté

- Candidat : `MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000` vs défauts.
- Qwen3.6-35B-A3B EXL3 2.49 bpw : decode non caché **−0,38 %**.
- Pic d'allocation MLX : **12,422 → 13,302 GB**. Un fort gain apparent sur une
  paire provient d'une baseline subitement lente, pas d'un gain fiable.
- Preuve : `build/decode-command-buffers-qwen-fallback-20260910.jsonl`.

### OPT-2026-09-10-03 — Mêmes budgets, LFM8 — rejeté

- LFM2.5-8B-A1B EXL3 3.10 bpw : decode **−2,41 %**, caché **−1,59 %**.
- Preuve : `build/decode-command-buffers-lfm8-20260910.jsonl`.

### OPT-2026-09-10-04 — Compiler 24 feed-forward LFM8 — non concluant

- Réutilisation du compilateur stateless existant : decode **+0,11 %**, bruit.
- Preuve : `build/decode-ff-compile-lfm8-all-20260910.jsonl`.
- Le pilote limité à deux blocs n'a pas établi de gain non plus ; ne pas
  additionner les résultats de ce pilote et de la série complète.

### OPT-2026-09-10-05 — Compiler 30 feed-forward LFM2.6 — non concluant

- LFM2.5-2.6B EXL3 4 bpw : decode **+0,30 %**, bruit.
- Preuve : `build/decode-ff-compile-lfm26-20260910.jsonl`.

### OPT-2026-09-10-06 — Compiler les entrées QMV partagées — rejeté

- `mx.compile` des fonctions denses/groupées/experts réellement appelées.
- LFM8, série séparée sur secteur : decode **−3,06 %** médian.
- Preuve : `build/decode-qmv-compile-lfm8-ac-20260910.jsonl` ; runner local :
  `build/bench_decode_ff_compile.py` (`--qmv` pour cette variante).
- Attention : l'ancien `work/compiled_qmv_bench.py` cible des alias obsolètes.
  Ne pas utiliser ses résultats comme validation du chemin actuel.

## 7 septembre 2026 — résultats antérieurs à ne pas redécouvrir

Source et protocoles : [rapport général](docs/general-performance-2026-09-07.md).
Les validations rapportées ici ne lèvent pas le blocage macOS 27 découvert
le 10 septembre. Les gains portent uniquement sur le périmètre indiqué.

| ID | Essai | Résultat documenté | Conclusion historique |
| --- | --- | --- | --- |
| OPT-2026-09-07-01 | Sortir les calculs d'adresse QMM de la boucle | Prefill LFM2.6 +6,84 %, Qwen +6,09 % ; decode dans le bruit ; logits/caches forcés bit-à-bit | Validé sur le chemin TensorOps de l'époque |
| OPT-2026-09-07-02 | Évincer les snapshots avant la session entière | TTFT second tour 5,1517 → 0,1673 s ; 3 500 tokens réutilisés dans le scénario contraint | Validé pour ce scénario, pas un gain decode universel |
| OPT-2026-09-07-03 | Préparation incrémentale des blocs de code UI | À 1 MiB : 52,482 → 1,640 ms par ajout | Validé en microbenchmark CPU, pas FPS/tok/s |
| OPT-2026-09-07-04 | Comptages de chaînes bornés | Grande ligne ASCII 3,561 → 2,946 ms ; Unicode 54,674 → 29,909 ms | Validé séparément, ne pas multiplier les gains |
| OPT-2026-09-07-05 | Validation UUID du bridge sans attendre la queue IO | Délai synthétique 53,877 → 0,034 ms quand IO occupée 50 ms | Validé pour la réactivité du bridge, pas le decode |
| OPT-2026-09-07-06 | Omettre les diagnostics quant non utilisés | Temps −1,51 % médian, quatre projections K=4 ; fingerprints exacts | Mesure limitée, remplacée par la comparaison combinée suivante |
| OPT-2026-09-07-07 | Quatre changements de préparation quant combinés | Temps −2,46 % médian ; fingerprints/scores exacts | Validé sur quatre projections, pas conversion complète ; ne pas ajouter −1,51 % |
| OPT-2026-09-07-08 | Supprimer copie gate/up MoE prefill | Prefill Qwen environ −3 à −6 % malgré contrôles numériques réussis | Rejeté |
| OPT-2026-09-07-09 | Fusion gather/Hadamard decode | LFM8 +0,46 % non caché, +0,02 % caché ; Qwen instable | Retiré, gain non établi |
| OPT-2026-09-07-10 | Extraction QMM 64 bits → funnel 32 bits | Plus lent, pourcentage non précisé dans ce rapport | Rejeté |
| OPT-2026-09-07-11 | Préchargement des mots QMV | Résultats synthétiques mixtes | Non retenu |
| OPT-2026-09-07-12 | Grouper les projections LFM w1/w3 | Decode +1,33 %, prefill −6,59 %, environ +171 MB pic requête cachée | Rejeté ; résoudre les copies compactantes avant réessai |
| OPT-2026-09-07-13 | Cacher les métadonnées récurrentes vides | Wrapper CPU ~580–600 → 403–434 ns ; pas de gain modèle mesuré | Retiré, impact estimé négligeable |

## Pistes ouvertes, non validées

- Corriger la compatibilité TensorOps macOS 27 avant les nouvelles validations
  bout en bout du chemin normal.
- Étudier des régions decode compilées plus larges : attention, normalisation,
  RoPE et cache ensemble ; les blocs Qwen récurrents et plusieurs MLP sont
  déjà compilés. Les essais isolés 04–06 n'ont pas montré de gain.
- Projections groupées mixtes (bits/codebooks) et suppression des copies
  compactantes : vérifier les formes réellement exclues avant tout prototype.
- Profiler les dispatchs GPU et les lectures réelles avant une nouvelle variante
  QMV ; taille du fichier / tok/s n'est pas une mesure de bande passante GPU.

## Nouvelle série du 10 septembre 2026

### OPT-2026-09-10-07 — Masquage top-k par les tokens conservés — non concluant

- Hypothèse : conserver exactement `argpartition(-logprobs, kth=k-1)`, mais
  remplacer le scatter des V-k tokens rejetés par un tableau rempli de -inf
  et un scatter des k valeurs conservées, économise des écritures irrégulières.
- Antécédents : D07 du roadmap/audit proposait le sampler ; aucun essai de ce
  changement précis trouvé. Pas de changement du greedy, du RNG ou de top-k.
- Baseline : `mlx_lm.sample_utils.apply_top_k` installé, MLX 0.32.2.
- Protocole : parité des sorties FP16/BF16/FP32, ties/NaN/Inf et seeds ; puis
  microbenchmark vocab 65 536/248 320/262 144, k=40. Si favorable, trois paires
  modèle à température 0,7, top-k 40, seeds identiques, 128 tokens.
- M5 24 GiB, macOS 27, secteur (82 %), thermique non contrôlée ; workaround
  TensorOps de l'essai 01 identique des deux côtés pour les essais modèle.
- Commande/prototype : `.venv/bin/python benchmarks/bench_topk_scatter.py` ;
  preuves prévues `build/topk-scatter-20260910.jsonl` et fichiers modèle séparés.
- Résultat : 252 cas de masques bit-à-bit, sampling avec seed contrôlé réussi.
  Microbenchmark favorable mais Qwen decode +0,70 % médian, paires
  [+0,70 ; −0,86 ; +16,27] %, trop instables pour promotion.
  Preuve modèle : `build/topk-scatter-qwen-20260910.jsonl`.
- Reproduction modèle avec le runner actuel : ajouter
  `--model models/Qwen3.6-35B-A3B-EXL3-2.49bpw --temperature 0.7 --top-k 40`.
  Les défauts ultérieurs du runner sont ceux de l'essai 08 (0,2 / 80).
- Intégration : diagnostic uniquement, pas l'app.

### OPT-2026-09-10-08 — Top-k hiérarchique exact — non concluant

- Nouvelle hypothèse : dans le backend Metal MLX installé, ArgPartition appelle
  un tri complet. Trier des blocs, retenir leurs k meilleurs puis trier ces
  candidats évite les grandes fusions du tri global. Aucun token du top-k
  global ne peut être exclu par un top-k local. Les ties doivent conserver
  l'ordre initial via les tris stables du backend actuel ; contrôler NaN/Inf.
- Différence avec 07 : réduit le travail du tri, pas seulement le masquage.
- Prototype : `benchmarks/bench_topk_scatter.py --hierarchical`. Bloc 1024,
  fallback référence si petit vocabulaire ou k >= bloc. Padding NaN en fin,
  comme les sentinelles du tri MLX. Non destiné aux backends non validés.
- Contrôles prévus : masques bit-à-bit FP16/BF16/FP32, ties, NaN/Inf, bords de
  blocs, k=1/40/80/V−1 ; sampling avec même seed ; puis modèle température
  0,2/top-k 80 (défauts GUI), trois paires de 128 tokens si parité réussie.
- Mêmes matériel/OS et workaround que 07. Prototypes seulement, non intégrés.
- Preuves : `build/topk-hierarchical-20260910.jsonl` (1 020 cas bit-à-bit,
  sampling seed identique), `build/topk-hierarchical-qwen-20260910.jsonl`.
  Qwen : decode +0,38 % médian, paires [−0,36 ; +1,54 ; +0,38] %, bruit.
  LFM1.2 Thinking : −0,96 % médian, paires [−0,96 ; −6,53 ; +6,39] %,
  textes identiques ; `build/topk-hierarchical-lfm12-20260910.jsonl`.
  Pas de gain modèle fiable, pas de promotion malgré le gain du filtre isolé.
- Microbenchmark synchronisé (temps CPU+GPU, pas temps GPU pur) : réduction
  médiane du temps top-k de **18,99 % / 33,71 % / 33,60 %** pour les vocabulaires
  65 536 / 248 320 / 262 144. Ne pas appliquer ces pourcentages au decode.

### OPT-2026-09-10-09 — Compiler la pénalité de répétition bornée — rejeté

- Hypothèse : compiler gather/arithmétique/scatter de la pénalité existante,
  avec la fenêtre de tokens bornée avant l'entrée compilée pour ne pas créer
  un graphe par longueur d'historique. Ne change pas la pénalité ni son ordre.
- Antécédent D01 : borne déjà l'historique stable du préfixe, pas cette fusion
  du processeur. Les tests QMV/FFN 04–06 concernaient d'autres opérations.
- Référence : factory MLX-LM installée. Candidat : même closure sous
  `mx.compile`, entrée `tokens[-20:]`, cache borné des factories dans le prototype.
- Protocole : comparaison bit-à-bit sur dtypes, tokens répétés, signes et
  longueurs de fenêtre ; microbenchmark puis trois paires greedy modèle si
  exact. Même environnement/workaround que 08 ; pas de top-k modifié simultanément.
- Prototype/proofs : `benchmarks/bench_topk_scatter.py --penalty`,
  `build/penalty-compile-20260910.jsonl` puis fichiers modèle séparés.
- Premier contrôle de parité échoue : FP16, vocab 257, historique 21 tokens,
  pénalité 0,8. Arrêt avant benchmark modèle. Diagnostic des différences en
  FP16 confirme aussi un écart avec la pénalité par défaut 1,05 :
  0,65185546875 → 0,65234375 (un logit fini). Preuve :
  `build/penalty-compile-diagnostic-20260910.jsonl`. Aucun gain revendiqué.
  Non intégré dans le moteur/l'app.

### OPT-2026-09-10-10 — Types explicites TensorOps macOS 27 — validé en prototype

- Suite de 01 justifiée par l'inspection des nouveaux headers MPP : les
  contraintes de types n'effacent pas l'espace d'adressage `thread` porté
  par `decltype(variable)`. Tester les mêmes types tensor/cooperative_tensor
  via leurs alias explicites sans ce qualificatif ; ne pas changer le calcul.
- Ce n'est pas un boost decode annoncé : rétablir un chemin normal de prefill
  est nécessaire aux mesures complètes. Pas de modification des headers Apple.
- Prototype : intercepteur local de la factory Metal dans un runner de test,
  puis suite QMM existante. Même MLX/macOS, aucun TensorOps désactivé.
- Commande : `.venv/bin/python build/check_tensor_types.py` ; sortie
  `build/tensor-types-check-20260910.log` (premiers 24 cas), puis
  `build/tensor-types-full-20260910.log` : **87 tests QMM réussis**.
- Chargement + warmup + 32 tokens Qwen avec TensorOps activé réussis :
  `.venv/bin/python build/check_tensor_types.py --model models/Qwen3.6-35B-A3B-EXL3-2.49bpw` ;
  `build/tensor-types-qwen-20260910.jsonl`, stderr vide. Une seule génération,
  pas un A/B de performance. Parité macOS 26/27 non mesurée ; pas de promesse
  bit-à-bit inter-OS ni validation exhaustive de tous les modèles.
- Correctif testé dans les factories `_qmm_tensor_kernel` et
  `_segmented_expert_qmm_tensor_kernel` : remplacer les arguments
  `decltype(first_left), decltype(right), float` de l'accumulateur par
  `tensor<device half, dextents<int, 2>, tensor_inline>`,
  `tensor_ops::matmul2d<descriptor, execution_simdgroup>::cooperative_tensor_right_input_t<half, half, float>`,
  `float`. Les calculs, tuiles et poids ne changent pas.
- Intégration : intercepteur du runner de diagnostic uniquement. Sources de
  production, app installée et dépendances non modifiées. Avant intégration :
  ajouter un contrôle de source durable, vérifier l'autre OS supporté et le
  chemin segmenté sur une matrice dédiée. Pas de boost decode revendiqué.

## 11 septembre 2026 — objectif : au moins 10 % sur une métrique complète

### OPT-2026-09-11-01 — Borner les chunks de prefill en vol — non concluant

- Hypothèse : `_prepare` soumet tous les chunks par `async_eval` avant
  l'attente terminale ; borner cette file peut réduire le pic des temporaires
  sans changer taille des chunks, poids, cache ou précision.
- Antécédents : les changements de taille de chunks du 5 septembre ont été
  rejetés pour dérive numérique. Ici seul l'ordonnancement change. Aucun
  essai de backpressure prefill trouvé dans les archives consultées.
- Protocole : chemin réel `_stream_response`, prompts fixes ~8k puis ~16k
  selon mémoire, 16 tokens générés ; A/B synchronisation de chaque chunk
  stable, puis pipeline de deux chunks si utile. Comparer pic MLX, temps,
  texte et empreintes des caches ; confirmer en processus séparés si favorable.
- M5 24 GiB, macOS 27.0/26A428, MLX 0.32.2, batterie 46 %, thermique non
  contrôlée. Appliquer le correctif de types de 2026-09-10-10 aux deux côtés
  du benchmark, uniquement dans le prototype, pour garder TensorOps normal.
- Préserver les autres changements locaux ; aucune refonte/changement de
  langage, aucun push ni remplacement d'app sans validation supplémentaire.
- Preuves prévues : `build/prefill-flight-20260911*.jsonl`, runner dans
  `benchmarks/bench_prefill_flight.py`.
- Pilote LFM1.2 Thinking, 6 154 tokens : même pic MLX **1,851846224 GB**,
  TTFT 3,461 → 3,497 s ; texte et états finaux identiques. Pas de gain RAM ;
  ne pas poursuivre cette synchronisation systématique sur ce résultat.
  `build/prefill-flight-20260911-lfm12.jsonl`.

### OPT-2026-09-11-02 — Partager les KV du préfixe après génération — non concluant

- Hypothèse : dans `GenerationSession.finish`, le préfixe stable KV dupliqué
  est identique au début des KV de la génération terminée. Remplacer seulement
  ces copies KV par des forks/vues du cache final peut économiser la RAM
  retenue entre les tours, en conservant les états récurrents de chaque date.
- Différence avec COW au début de génération : pas de copie immédiatement
  déclenchée par les tokens générés. Réutilise `SharedKVCache` existant ; pas
  de pagination ni changement de langage. Mesurer séparément RAM active au
  repos et pic pendant la génération, ne pas confondre les deux.
- Prototype : option `--share-finished-kv` du runner 01, LFM1.2 Thinking,
  ~16k tokens, 16 générés ; baseline inchangée contre partage post-finish.
  Si favorable : trois paires, qualité préfixe/final, conversation suivante,
  branche au préfixe et snapshots indépendants ; vérification deuxième modèle.
- Mêmes environnement et correctif de types de benchmark que 01. Logs
  `build/finished-kv-20260911*.jsonl`. Aucun résultat mesuré ni intégration.
- Pilote 16 394 tokens : allocation MLX retenue **1 483 087 368 → 1 278 647 816
  octets (−13,78 %)** ; texte + préfixe + cache final identiques. Pic pendant
  génération inchangé (2,301 GB). Une seule paire, pas encore une validation.
- Étape suivante : vider aussi les allocations libérées après partage pour
  restituer la RAM au système, et mesurer `proc_pid_rusage.ri_phys_footprint`
  comme le widget. Contrôler séparément pool allocateur/MLX actif/empreinte OS.
  Candidat combiné partage + restitution ; ne pas additionner des pourcentages.
- Trois paires LFM16k : −13,79 % d'allocation MLX active retenue, mêmes textes
  et caches. **Pas de baisse nette d'empreinte physique macOS** (−1,95 %, +0,08 %,
  +0,05 % de réduction), pic inchangé. Ne pas présenter cela comme un gain
  de RAM système ou de pic. `build/finished-kv-20260911-lfm12-final.jsonl`.
  Non intégré ; contrôle des tours suivants encore requis pour promotion.
- Recherche suspendue sur ce candidat : pas de gain RAM système établi,
  aucune modification de `GenerationSession.finish` conservée en production.

### OPT-2026-09-11-03 — Étendre BM64 dense aux autres matrices — non concluant

- Antécédent : M64 existe déjà pour grandes matrices MUL1 à entrée >=4096.
  Nouvelle couverture testée : entrées 2048 et autres codebooks, uniquement
  lorsque les lignes sont multiples de 64, >=128, hors head >=65536 sorties.
  Garder BK16/BN32, donc même ordre de réduction ; parité à vérifier.
- Hypothèse : amortir la déquantification EXL3 sur 64 plutôt que 32 lignes,
  en évitant de modifier chunks/padding/poids. Pas de nouveau langage/refonte.
- Runner 01 option `--dense-m64`, LFM2.6 4 bpw, ~4k tokens, 16 générés ;
  comparaisons suivantes seulement si caches et texte exacts. Les deux côtés
  utilisent la correction de types TensorOps, sans autre candidat RAM combiné.
- M5/macOS27, batterie, thermique non mesurée. Logs
  `build/dense-m64-20260911*.jsonl`. Aucun résultat ni intégration.
- Pilote 4 107 tokens : 789,57 → 811,95 tok/s prefill (+2,84 %), TTFT
  5,227 → 5,071 s ; pic identique, texte/préfixe/cache final exacts.
  Une seule paire : non concluant pour promotion, pas de gain généralisé.
  `build/dense-m64-20260911-lfm26.jsonl`. Prototype seulement.

### OPT-2026-09-11-04 — BM64 TensorOps segmenté MoE — validé sur M5, code local

- Hypothèse : déquantifier chaque tuile de poids une fois pour 64 lignes au
  lieu de deux fois pour 32, dans les blocs experts déjà paddés à 64 lignes.
  Antécédents : BM8/16/32 testés, BM64 pas dans la matrice historique.
- Prototype : `_SEGMENTED_TENSOR_ROWS=64` en mémoire contre 32 ; buckets et
  locality désactivés, BK16/BN32 inchangés. Pas de refonte ni de nouveau langage.
- Qwen3.6 35B A3B EXL3 2.49 bpw, prompt fixe, 16 tokens, pilote puis trois
  paires alternées si texte et empreintes de tous les caches restent exacts.
  Même correctif TensorOps 2026-09-10-10 des deux côtés ; M5/macOS27 sur batterie.
- Runner `bench_prefill_flight.py --segmented-m64 --repeats 512`, preuves
  `build/segmented-m64-20260911*.jsonl`. Aucun résultat ni intégration.
- Pilote interrompu avant baseline : compilation Metal du chemin segmenté
  refuse la référence au descripteur local dans l'alias imbriqué utilisé par
  le correctif 10-10. Le warmup court précédent ne couvrait pas ce chemin.
  `build/segmented-m64-20260911-qwen.stderr`. Pas de mesure de performance.
- Révision du protocole : nommer les alias d'opération/opérande avant leur
  usage comme arguments template, sans changer leurs types ni les calculs.
  Réessai de compilation puis A/B dans un nouveau log `*-qwen-alias.jsonl`.
- Alias seul : même erreur, avant mesure. Le segmenté est inutilement instancié
  avec `T=half` alors que sa source utilise exclusivement `half`. Retirer ce
  paramètre template inutilisé dans le runner pour éviter la contrainte sur
  le descripteur local ; calcul inchangé. Log `*-qwen-no-template.jsonl`.
- Compilation rétablie. Pilote 4 106 tokens : prefill 414,33 → 517,90 tok/s
  (+25,00 %), TTFT 9,967 → 7,951 s ; cache/préfixe/texte exacts. Les compilations
  propres à la forme ne sont pas exclues : pas encore de gain validé.
- Confirmation prévue : une paire complète exclue (`pair=-1`), puis trois
  paires AB/BA, mêmes 4 106 tokens, options `--warmup-pairs 1 --pairs 3`,
  log `build/segmented-m64-20260911-qwen-warm.jsonl`. Contrôle numérique
  dédié : tous les codebooks et bits supportés, segments courts/64/65/129 et
  experts vides ; extension à LFM8 si les résultats restent favorables.
- Trois paires après warmup : prefill **+7,89 / +10,46 / +9,43 %** (médiane
  +9,43 %), TTFT **−7,26 / −9,41 / −8,64 %** ; caches et texte exacts.
  Le pilote +25 % surestimait le gain chaud. Pic ~14,05–14,06 GB inchangé.
  Gain prometteur mais objectif 10 % pas encore confirmé ; prototype seulement.

### OPT-2026-09-11-05 — Adresses invariantes dans le QMM segmenté — validé en combinaison, code local

- Antécédent : le hoisting du 7 septembre ne couvre que le QMM dense.
  Le QMM segmenté recalcule encore permutation/positions/shift à chaque BK16.
  Même technique appliquée aux adresses du bloc expert, sans changer les
  lectures ni l'ordre des additions. Garde de capacité coopérative conservée.
- Prototype dans la factory du runner : préparer offsets/shifts avant la boucle
  K, puis incrémenter uniquement la base de profondeur ; aucun poids déquantifié
  matérialisé. Comparer candidat combiné BM64 + hoisting à baseline BM32 normale,
  sans additionner les gains. Mêmes Qwen4k, warmup exclu, trois paires si pilote
  exact, puis matrice numérique dédiée. Logs `build/segmented-hoist-20260911*`.
- Aucune mesure ni intégration à ce stade.
- Pilote combiné chaud Qwen4k : 483,68 → 532,94 tok/s (+10,18 %), TTFT
  8,511 → 7,727 s (−9,21 %) ; pic 14,0523 GB identique, texte et caches exacts.
  Confirmation nécessaire. Avant cela, exécuter les 24 cas de
  `tests/test_segmented_m64.py` avec l'intercepteur compatible macOS27 et
  le hoisting activé : `build/segmented-m64-quality-20260911.log`.
- Matrice 24 cas réussie : BM32/BM64 avec hoisting actif des deux côtés,
  sorties finies et bit-à-bit. Cette matrice ne compare pas encore hoisting
  actif/inactif ; la comparaison combinée modèle vérifie ce dernier.
- Validation externe de forme : LFM8 A1B 3.10 bpw, même prompt répété 512,
  paire de warmup exclue puis trois paires, candidat BM64 + hoisting,
  log `build/segmented-hoist-20260911-lfm8-warm.jsonl`.
- LFM8, 4 106 tokens : trois gains prefill **+31,43 / +32,37 / +29,40 %**,
  TTFT **−23,80 / −24,40 / −22,67 %** ; textes et caches exacts. Pic ~4,923 GB
  inchangé. Candidat combiné en prototype uniquement ; objectif de mesure
  atteint sur ce scénario, sans conclusion llama.cpp ni decode universel.
- Ablation avant intégration : BM64 seul sur LFM8, mêmes trois paires chaudes,
  pour éviter de conserver un hoisting inutile ; log
  `build/segmented-m64-20260911-lfm8-warm.jsonl`.
- Intégrer maintenant uniquement le correctif de compilation macOS27 commun
  (alias de types explicites et suppression du template T inutilisé segmenté),
  puis contrôler les suites existantes. Cela n'est pas compté comme un boost.
- Ablation BM64 seul : LFM8 prefill **+22,97 / +23,67 / +22,98 %**,
  TTFT **−18,59 / −19,03 / −18,68 %**, texte/caches exacts, pic inchangé.
  Les séries ne sont pas directement additionnables ; l'essai combiné reste
  la preuve du gain cumulé. Garder les deux candidats pour validation locale.
- Intégration prévue dans le kernel commun : BM64 automatique à partir de
  4 096 lignes routées, BM32 en dessous, surcharge manuelle préservée (0=auto).
  Hoisting segmenté distinct, désactivable pour A/B et garde de capacité.
  Aucun chemin decode modifié. Petits prompts et suites numériques à vérifier
  avant de qualifier le code local ; ni app reconstruite ni publication.
- Code local intégré pour vérification. Le runner utilise désormais les vraies
  options de production (baseline BM32/hoist off, candidat auto/hoist on),
  sans réécriture des sources Metal. Les anciennes commandes utilisant
  `tensor_types_factory()` décrivent le prototype retiré du runner.
- Suite lancée : `.venv/bin/python -m pytest -q -x tests/test_segmented_m64.py
  tests/test_qmm_address_hoist.py tests/test_dense_prefill.py tests/test_audit_candidates.py`.
  Log `build/segmented-production-quality-20260911.log`. Matrice segmentée compare
  maintenant BM32 sans hoist à BM64 avec hoist. Ajout de contrôles durables
  capacité insuffisante et dispatch auto ; à exécuter après la suite en cours.
- **175 tests réussis** sans intercepteur Metal. Tests complémentaires ajoutés
  ensuite encore à exécuter ; compatibilité autre version macOS non mesurée.
- Confirmation finale du code réellement intégré : Qwen4k et LFM8 4k,
  `--segmented-m64 --segmented-hoist --warmup-pairs 1 --pairs 3 --max-tokens 96`,
  puis prompts courts `--repeats 16 --pairs 3 --max-tokens 32`. Journaux
  `build/segmented-final-20260911-*.jsonl`. Garder la comparaison à 96 tokens
  distincte de celles à 16 ; vérifier aussi decode et pic sans revendiquer un
  gain decode sur ce kernel de prefill. Aucun benchmark llama.cpp effectué.
- Holdout prévu après validation initiale : document technique non répétitif
  `docs/audit-runtime-2026-09-04.md`, option `--prompt-file`, warmup puis trois
  paires à 32 tokens sur LFM8 et Qwen. Log `build/segmented-heldout-20260911-*`.
  Ce contrôle évite de limiter la conclusion à une phrase répétée qui peut
  concentrer le routing sur quelques experts ; conserver les résultats séparés.
- Production Qwen4k/96 générés : gains prefill **+14,42 / +10,70 / +12,35 %**
  (médiane +12,35 %), TTFT **−12,57 / −9,69 / −11,02 %**. Decode
  −1,73 / +2,74 / −0,48 % : pas de gain établi, fluctuation de batterie/thermique.
  Pic identique 14,0523 GB ; empreinte OS candidat +~6–8 MB, pas un gain RAM.
  Caches et texte exacts. `build/segmented-final-20260911-qwen.jsonl`.
- Production LFM8 4k/96 générés : prefill **+34,86 / +36,84 / +30,85 %**
  (médiane +34,86 %), TTFT **−25,74 / −26,88 / −23,54 %**. Decode +1,08 / +1,55 /
  +2,41 %, sans changement decode direct ni revendication indépendante. Pic
  4,92316 GB identique ; texte/caches exacts. Empreinte OS ~4,70 GB, sans baisse.
  `build/segmented-final-20260911-lfm8.jsonl`. Mesures sur batterie, pas de
  garantie de ces pourcentages sur tous les prompts ou tous les Mac.
- Prompts courts 138 tokens / 32 générés : LFM8 prefill +16,14 / +12,99 /
  +17,46 %, TTFT −12,94 / −10,96 / −15,13 % ; Qwen prefill +11,84 / +8,95 /
  +6,95 %, TTFT −10,58 / −7,98 / −6,35 %. Caches/textes exacts, pics inchangés.
  Decode court : LFM médiane −2,49 %, Qwen −1,31 % (31 tokens chronométrés,
  petites fluctuations opposées à certaines paires de 96 tokens) ; ne pas
  cacher cette limite ni présenter une accélération decode.
  `build/segmented-final-20260911-{lfm8,qwen}-short.jsonl`.
- Holdout LFM8, document varié 2 746 tokens / 32 générés : prefill **+24,77 /
  +25,82 / +25,59 %** ; TTFT **−19,84 / −20,45 / −20,40 %**. Texte/caches exacts,
  pic ~4,72155 GB stable. Decode −3,79 / −3,89 / +0,32 % : petite régression
  dans deux des trois courts échantillons, à conserver dans le bilan.
  `build/segmented-heldout-20260911-lfm8.jsonl`.
- Holdout Qwen, 2 842 tokens / 32 générés : prefill +6,63 / +14,12 / +9,37 %,
  TTFT −6,32 / −12,18 / −8,63 %, decode médiane −0,62 %. Caches et texte exacts,
  pic ~13,91–13,92 GB sans gain établi. `build/segmented-heldout-20260911-qwen.jsonl`.
- **177 tests réussis** après ajout des contrôles capacité/auto-tile, sans
  intercepteur. `build/segmented-production-quality-final-20260911.log`.
  Dernier smoke prévu du chemin non-TensorOps :
  `MLXL3_TENSOR_SEGMENTED_QMM=0 .venv/bin/python -m pytest -q tests/test_audit_candidates.py -k device_side_route_buckets`,
  log `build/segmented-fallback-quality-20260911.log`. Pas un test sur un vrai M1.
- Rapport durable : [prefill-investigation-2026-09-11.md](docs/prefill-investigation-2026-09-11.md).
  Mesures finales/caches/manifestes copiés par patch dans
  [prefill-segmented-20260911.json](benchmarks/results/prefill-segmented-20260911.json),
  donc conservables dans Git même si les logs `build/` sont ensuite nettoyés.
- Intégration : `src/mlxl3/kernels/qmv.py` local, chemin partagé du moteur/CLI ;
  pas d'app reconstruite/installée, de push ou de publication. Aucun changement
  de langage/refonte. Aucun gain RAM/decode ni supériorité sur llama.cpp annoncé.
  Les candidats RAM/tuilage dense étendu non concluants restent hors production.
- Contrôle final : **8 tests non-TensorOps réussis**, 52 désélectionnés ;
  `git diff --check` et compilation Python des fichiers modifiés réussis.
  Le lanceur `~/.local/bin/mlxl3` utilise bien la `.venv` de ce dépôt ; import
  vérifié vers ce `qmv.py`, auto-tile=0 et hoisting actif. Aucun benchmark ne
  reste en cours. L'objectif >=10 % est atteint en prefill et TTFT sur les
  scénarios précisés ; les limites decode/RAM et de généralisation restent
  celles du rapport, sans prétendre avoir optimisé tout modèle/tout contexte.

### Publication demandée le 11 septembre 2026

- L'utilisateur demande de terminer puis pousser sur GitHub. Préparer le lot
  validé sur `main` : kernel, tests, runner, preuves, rapports et consignes du
  journal. Les modifications Ling/quantification restent locales hors du commit.
- Aucune release/DMG ni reconstruction de l'app n'est demandée par ce push.
  Le résultat du push sera consigné après vérification du commit distant.
- Push du code **réussi** : `f7825f49986b31c44a44b225faef3b9586eb0d0d`
  sur `https://github.com/0xZKnw/mlxl3`, branche `main`. Commit distant confirmé
  avec `git ls-remote origin refs/heads/main`. Tests/préuves/journal inclus ;
  changements Ling préservés hors commit. App installée et DMG inchangés.

### REL-2026-09-11 — Desktop v1.0.1 build 12 — en cours

- Demande utilisateur : intégrer le nouveau moteur dans la GUI et publier v1.0.1.
  Construire depuis un checkout propre du commit de release, sans les travaux
  Ling non publiés. Versions Python/Swift synchronisées ; signature ad-hoc
  maintenue, aucun certificat Developer ID disponible.
- Validation du correctif 10-10/11-04 : moteur gelé dans l'app, warmup réel activé
  (le smoke existant le désactivait), chargement Qwen, deux tours de chat et
  prefill long. Puis signature, DMG, empreinte SHA256 et métadonnées du bundle.
  Il s'agit de validation de distribution, pas d'un nouveau benchmark de gains.
- Publication prévue : tag GitHub v1.0.1, DMG build 12, manifestes/validation ;
  conserver une copie de l'app locale précédente avant son remplacement.
- Checkout propre `98da8c5` : 177 tests réussis. Runtime PyInstaller : warmup
  Qwen et deux tours avec rappel CEDAR-42 réussis ; prefill de 4k tokens puis
  génération réussis. Logs `build/release-v101-{clean-tests,qwen-smoke,qwen-long}*`.
- Build GUI interrompu : SDK27 fourni par CommandLineTools sans plugin
  `SwiftUIMacros.StateMacro`. Aucun changement UI nécessaire. Réessayer avec
  le SDK26.5 déjà installé, sélection explicite `MLXL3_MACOS_SDK` et SDK inscrit
  au manifeste. Ne pas publier l'artefact incomplet ; log du premier échec
  `build/release-v101-build.log`. L'option `--version` du CLI n'existe pas :
  version vérifiée via manifeste/Info.plist, sans ajouter une commande hors scope.
- Révision après demande explicite de SDK27 : le build SDK26.5 a terminé
  (exit 0, `build/release-v101-sdk265-build.log`), mais reste un artefact local
  non installé et non publié. La publication est suspendue pour résoudre la
  chaîne SwiftUI27. Le SDK27 contient bien SwiftUI/SwiftUICore et déclare
  `State()` via `SwiftUIMacros.StateMacro`, mais aucun plugin SwiftUIMacros
  n'a été trouvé dans `/Library/Developer` ou `/Applications` ; seul
  CommandLineTools est sélectionné, sans Xcode.app. Aucun SDK ni code SwiftUI
  contourné/modifié. Installation des outils Xcode27 complets à confirmer.
- Installation Xcode27 autorisée par l'utilisateur. Espace disponible vérifié
  (~554 GiB), aucune archive Xcode dans Downloads. Le site officiel Apple
  redirige les téléchargements Applications vers la connexion Apple Account :
  page ouverte et laissée à l'utilisateur pour authentification. Aucun
  téléchargement/installation Xcode ni publication v1.0.1 effectué à ce stade.
- L'utilisateur choisit finalement SDK26.5. Reprise du paquet propre `4e1fd76`,
  sans téléchargement Xcode27. Vérification finale prévue : signature et
  intégrité DMG, self-checks GUI et deux tours Qwen via le runtime de l'app
  finale (répétition du smoke précédent pour valider la copie embarquée).
- Contrôles du paquet SDK26.5 réussis : signature stricte, intégrité DMG,
  self-checks timeline/MCP/mémoire Metal et deux tours Qwen avec warmup réel
  (`build/release-v101-final-app-smoke.*`). Le manifeste contient toutefois
  un ancien doublon editable `mlxl3.egg-info` 1.0.0, malgré le code 1.0.1 :
  correction du générateur pour donner priorité à la version source embarquée.
  Reconstruction du paquet prévue, moteur/GUI inchangés ; revérifier
  manifeste, signature, DMG et identité du runtime avec celui testé.
- **Validé et publié** : paquet propre `14061bf320fc0d41f498c2ffcdfe522e3bf49717`,
  version 1.0.1/build 12, SDK26.5. Manifeste cohérent, signature stricte et
  self-checks GUI réussis dans l'app puis le DMG monté. Runtime SHA256
  `2213fd25219b8319ed469fdde15e4f3a17a6a8736d7f95bba8ed9af80f2124bb`
  identique au runtime testé ; deux tours Qwen montés réussis, warmup activé,
  CEDAR-42 rappelé, 537 tokens cachés/20 évalués au second tour, stderr vide
  (`build/release-v101-mounted-smoke.*`). Pas de mesure de gain supplémentaire.
- Installation locale effectuée dans `dist/MLXL3 Desktop.app`. Ancienne app
  conservée dans `build/app-backups/MLXL3 Desktop-v1.0.0-before-v1.0.1.app`.
  Aucun historique/modèle modifié ; l'app n'a pas été relancée automatiquement.
- `main` et tag `v1.0.1` poussés. Release publique confirmée via `/releases/latest` :
  https://github.com/0xZKnw/mlxl3/releases/tag/v1.0.1 ; DMG, SHA256, manifeste et
  `validation-v1.0.1.txt` présents. Empreinte DMG locale/distance identique :
  `2a98101aec9a4d718e76821f64b7d83885cb660ec9aca7a3ab858be6c08b2de4`.
  CI du tag réussie : https://github.com/0xZKnw/mlxl3/actions/runs/34637258944.
  Signature toujours ad-hoc/non notarisée. Volume de test éjecté, aucun moteur
  de test encore actif. Travaux Ling non publiés préservés hors des commits.

### OPT-2026-09-12-RUST-PERF-01 — Synchronisations Qwen Rust par couche — validé, code local

- Hypothèse : le port Rust synchronise actuellement `hidden` puis l'état dans
  chaque couche Qwen, soit des dizaines de barrières CPU/GPU par token, alors
  que le moteur Python validé synchronise le graphe complet au point de
  sampling. Regrouper ces évaluations à la frontière du token doit restaurer
  une part importante du decode sans changer aucun calcul ni ordre numérique.
- Antécédents consultés : `src/mlxl3/kernels/qmv.py`, `src/mlxl3/linear.py`,
  `src/mlxl3/moe.py`, `src/mlxl3/recurrent.py`, ainsi que
  `docs/decode-investigation-2026-09-10.md`,
  `docs/general-performance-2026-09-07.md` et
  `docs/prefill-investigation-2026-09-11.md`. Aucun essai historique ne mesure
  cette barrière propre au nouveau port Rust.
- Baseline/candidat : runtime Rust release, checkpoint local
  `models/Qwen3.6-35B-A3B-EXL3-2.49bpw`, température 0, MCP désactivé, prompt
  fixe, 32 tokens maximum ; un tour de chauffe puis au moins trois tours
  mesurés si la stabilité thermique le permet. Comparer texte/token IDs et
  logits imposés avant/après ; parité requise. Commande orchestrée via le
  protocole `mlxl3-rs bridge`, preuves sous `build/rust-perf-01-*.jsonl`.
- Conditions initiales : Apple M5 10 cœurs GPU, Metal 4, macOS local ; batterie
  100 %, débranchée. Température non mesurée. Le moteur GUI est fermé et aucun
  autre modèle n'est chargé. État d'intégration : analyse seulement, aucune
  mesure ni modification de kernel/runtime pour cet essai.
- Baseline mesurée avec 24 tokens de prompt et 32 générés : tour de compilation
  decode 8,72 tok/s, prefill 3,83 tok/s, TTFT 6,278 s ; trois tours chauds
  decode 8,39 / 8,10 / 8,80 tok/s (médiane **8,39**), prefill 6,82 / 5,96 /
  5,79 tok/s (médiane **5,96**), TTFT 3,519 / 4,029 / 4,149 s (médiane
  **4,029 s**). Les quatre sorties sont identiques. Pic reporté 13,064 GB,
  uniquement taille résidente estimée du checkpoint dans ce runtime, pas une
  mesure du processus. Preuve `build/rust-perf-01-baseline.jsonl`.
- Première vérification interrompue avant compilation : `cargo` n'est pas dans
  le `PATH` non interactif de cette session (`command not found`). Aucun test
  candidat ni résultat de performance ; localiser la toolchain déjà utilisée
  par le build de release puis relancer exactement les mêmes contrôles.
- Deuxième lancement encore interrompu avant compilation : le binaire Cargo
  absolu a été trouvé mais son `rustc` frère n'était toujours pas dans `PATH`.
  Relance suivante avec le dossier complet de la toolchain stable préfixé ;
  toujours aucune donnée candidat à ce stade.
- Troisième lancement a atteint le build script puis s'est arrêté avant les
  tests : `MLXL3_MLX_ROOT` absent. Aucun binaire candidat produit. Réutiliser
  exactement le chemin MLX 0.32.2 enregistré par le build d'app, sans installer
  ni changer de dépendance.
- Build candidat réussi avec la toolchain stable et MLX 0.32.2. Le filtre de
  tests `qwen` ne sélectionne actuellement aucun test Rust (0 exécuté) : il ne
  constitue pas une validation. Contrôle différentiel réel effectué avec
  l'ancien runtime `build/rust-runtime/mlxl3` et le candidat
  `target/release/mlxl3-rs`, tokens imposés `1,2` : sorties JSON/logits de
  2 978 940 octets strictement identiques, SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`.
  Le warning `rust-objcopy` sans `libLLVM.dylib` n'empêche ni le build ni
  l'exécution ; il concerne uniquement le strip de debug.
- Candidat, même protocole : tour de compilation decode 28,70 tok/s, prefill
  13,86 tok/s, TTFT 1,734 s ; trois tours chauds decode 27,38 / 29,50 / 28,93
  tok/s (médiane **28,93**, **+244,9 %** contre 8,39), prefill 29,41 / 29,69 /
  29,43 tok/s (médiane **29,43**, **+393,8 %** contre 5,96), TTFT 0,817 /
  0,809 / 0,816 s (médiane **0,816 s**, **−79,7 %** contre 4,029 s).
  Sorties identiques entre tous les tours et au runtime baseline ; preuve
  `build/rust-perf-01-candidate.jsonl`. Batterie toujours débranchée, charge
  descendante non enregistrée, température non mesurée ; l'amplitude dépasse
  largement le bruit possible mais les pourcentages restent ceux de ce prompt.
- Décision : conserver la synchronisation unique sur les logits à la frontière
  de chaque token et supprimer les synchronisations par couche. Cela ne porte
  encore ni le QMM TensorOps du Python ni le prefill par séquence ; aucun gain
  n'est revendiqué pour les autres architectures. App installée inchangée,
  aucune publication.

### OPT-2026-09-12-RUST-PERF-02 — Sampling greedy entièrement Metal — validé fonctionnel, code local

- Hypothèse : avec température 0/top-k 1 et pénalité neutre, le runtime Rust
  matérialise aujourd'hui tout le `log_softmax` vocabulaire en FP32 sur CPU puis
  y cherche le maximum. Le moteur Python calcule le même `log_softmax` et son
  argmax sur Metal, puis ne lit qu'un index scalaire. Ajouter l'opération MLX
  native déjà disponible doit réduire le temps decode sans approximation.
- Baseline : candidat validé de RUST-PERF-01, même Qwen3.6 35B A3B, prompt fixe
  24 tokens, 32 générés, trois tours chauds : decode médian 28,93 tok/s,
  prefill 29,43 tok/s, TTFT 0,816 s. Batterie débranchée ; température non
  mesurée. Protocole identique, preuve candidate prévue
  `build/rust-perf-02-candidate.jsonl`.
- Contrôle qualité prévu : argmax Rust unitaire sur GPU, puis sortie complète
  identique au candidat précédent. Le chemin CPU existant reste utilisé pour
  sampling non greedy ou pénalité de répétition non neutre. État : aucun code
  ni résultat pour cet essai.
- Premier contrôle unitaire : sampling greedy réussi, mais le smoke Array a
  échoué sur une attente de test incorrecte (`[3]`) : l'argmax est bien réalisé
  sur le dernier axe d'une matrice 2×2 et retourne donc `[1,1]`, comme MLX-LM.
  Le code d'opération n'a pas échoué. Corriger uniquement l'oracle du test puis
  relancer ; aucune mesure modèle candidate avant ce contrôle vert.
- Après correction de l'oracle, smoke Array GPU et test sampling réussis.
  Candidat : tour de compilation 29,32 tok/s ; trois tours chauds decode 28,90 /
  30,00 / 29,53 tok/s (médiane **29,53**, +2,09 % contre RUST-PERF-01),
  prefill médian 29,69 tok/s et TTFT médian 0,8086 s. Sorties complètes
  identiques. Preuve `build/rust-perf-02-candidate.jsonl`.
- La série n'est pas alternée et la machine se réchauffe : le petit écart decode
  reste **non concluant comme pourcentage**. Décision fonctionnelle validée :
  conserver le chemin Metal, qui supprime objectivement la copie CPU du
  vocabulaire entier ; fallback CPU inchangé pour sampling/pénalité non neutres.
  App installée inchangée, aucune publication.

### OPT-2026-09-12-RUST-PERF-03 — Référence du moteur Python sur le même Mac — validé

- Objectif : mesurer le moteur Python actuel qui contient les kernels validés,
  au lieu de prendre les anciens chiffres ~50 decode/~500 prefill comme une
  baseline interchangeable. Cela permettra de porter seulement les chemins
  manquants du Rust et de comparer sous les mêmes conditions.
- Protocole Python existant : `mlxl3 benchmark qwen3.6-35b-a3b
  --prompt-tokens 128 --max-tokens 32 --warmup-runs 1 --repeats 3`, température
  0, sans MCP/réseau, sortie `build/rust-perf-03-python-reference.json`.
  Apple M5 sur batterie, niveau/thermique à relever avec le rapport. Aucune
  modification de moteur dans cet essai ; résultats non mesurés.
- Résultat à 134 tokens de prompt / 32 générés, après un warmup : prefill
  322,62 / 323,69 / 321,13 tok/s (médiane **322,62**), decode 53,66 / 53,40 /
  53,00 tok/s (médiane **53,40**), TTFT médian **416,25 ms**, pic MLX
  **12,433 GB**. Batterie 95 %, débranchée ; température non mesurée. Preuve
  `build/rust-perf-03-python-reference.json`.
- Conclusion : la cible decode ~50 tok/s est confirmée sur le moteur Python,
  mais le ~500 tok/s prefill n'est pas la valeur comparable de ce prompt court.
  Le Rust après RUST-PERF-02 reste à ~55 % du decode Python et son prefill
  token-par-token n'est pas comparable au QMM séquentiel Python.

### OPT-2026-09-12-RUST-PERF-04 — Hadamard/scales EXL3 compilés comme en Python — rejeté

- Hypothèse : le Rust exécute actuellement cast, scale, reshape, Hadamard puis
  scale de sortie comme opérations MLX séparées autour de chaque QMV. Le chemin
  Python validé utilise deux fonctions `mx.compile` (`_reference_scaled_hadamard_*`)
  qui gardent exactement le même ordre/arrondis mais fusionnent ces graphes.
  Réutiliser ces deux graphes via l'API C++ MLX doit réduire les dispatchs de
  tous les linéaires EXL3, particulièrement en decode.
- Baseline : runtime RUST-PERF-02 sauvegardé sous
  `build/rust-perf-02-runtime` (SHA256
  `9b23d145c849e01238a555187ec8d35806239f574259debee804d74f282a9e38`),
  Qwen decode chaud médian 29,53 tok/s. Candidat : mêmes 24/32 tokens, un
  warmup + trois tours, batterie débranchée, température non mesurée.
- Qualité prévue : smoke des deux opérations, logits imposés `1,2` strictement
  comparés au runtime sauvegardé, puis sortie chat identique. Aucun kernel
  Metal nouveau ni mode rapide ; calculs FP16/Hadamard identiques au Python.
- Premier build/smoke GPU réussi. `cargo fmt --check` a seulement signalé deux
  lignes à reformater et le compilateur une variable devenue inutilisée après
  fusion ; corrections mécaniques appliquées avant le contrôle de logits. Pas
  encore de résultat modèle candidat.
- Premier contrôle modèle **rejeté avant benchmark** : le candidat échoue au
  chargement avec `Cannot reshape array of size 2048 into shape (1,96,128)`.
  Le smoke de largeur 128 avait tracé le graphe C++ avec `shapeless=true` ; les
  dimensions calculées dans la lambda ont donc été réutilisées à tort pour les
  largeurs suivantes. Baseline intacte et aucune mesure de performance issue
  de ce candidat. Relance prévue avec le mode par défaut sensible aux shapes,
  identique au décorateur Python `@mx.compile` utilisé comme référence.
- Relance sensible aux shapes : build réussi, mais le contrôle différentiel a
  été **interrompu par une erreur de protocole** avant chargement du candidat :
  `cargo build --release` sans `--features mlx,chat` a remplacé le binaire par
  la variante minimale qui n'expose pas `forward`. Le filtre de smoke utilisé
  n'a sélectionné aucun test (0 exécuté), donc aucun succès ne lui est attribué.
  Relancer build, smoke et `forward` avec les deux features explicites ; aucune
  donnée de performance candidate à ce stade.
- Candidat corrigé construit avec `--features mlx,chat`. Le listing confirme
  l'existence du smoke GPU (il n'était pas exécuté dans la commande précédente).
  Contrôle modèle Qwen tokens imposés `1,2` désormais **strictement identique**
  au runtime RUST-PERF-02 : 2 978 940 octets et SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`.
  Benchmark encore non mesuré ; exécuter explicitement le smoke puis la série.
- Smoke GPU explicitement exécuté avec `--ignored` : réussi. Candidat, un tour
  de compilation puis trois tours chauds : decode 28,21 / 28,36 / 27,53 tok/s
  (médiane **28,21**, **−4,49 %** contre 29,53), prefill médian **28,43** tok/s
  (−4,26 %) et TTFT médian **0,8445 s** (+4,44 %). Sorties identiques ; preuve
  `build/rust-perf-04-candidate.jsonl`.
- Décision : **rejet et retrait**. La compilation locale fidèle au Python
  ralentit le graphe Rust déjà différé entre tokens ; conserver la chaîne MLX
  primitive et tester ensuite un écart structurel plus haut niveau.

### OPT-2026-09-12-RUST-PERF-05 — Cache de la détection GPU comme le Python — validé, code local

- Hypothèse : le Python protège `_is_m5_gpu()` avec `@cache`, tandis que le
  Rust recrée un `metal::Device::system_default()` et lit son nom dans chaque
  `expert_mapped`. Qwen appelle ce chemin deux fois par couche MoE et par token.
  Mettre en cache ce booléen immuable avec `OnceLock` supprime donc des appels
  Objective-C/Metal répétés sans toucher au calcul ni aux kernels.
- Baseline : runtime RUST-PERF-02 sauvegardé, même Qwen 24 tokens prompt / 32
  générés, decode chaud médian 29,53 tok/s ; un warmup + trois tours candidat,
  batterie débranchée et température non mesurée. Preuve prévue
  `build/rust-perf-05-candidate.jsonl`.
- Qualité prévue : tests Rust, logits `1,2` strictement identiques au binaire
  sauvegardé, sortie complète identique. État : aucun résultat candidat.
- Tests Rust réussis et logits imposés `1,2` strictement identiques au runtime
  sauvegardé (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate : decode chaud 31,46 / 31,56 / 33,22 tok/s
  (médiane **31,56**, +6,87 %), prefill médian **32,41** tok/s (+9,18 %) et
  TTFT médian **0,7407 s** (−8,40 %). Preuve
  `build/rust-perf-05-candidate.jsonl`; sorties identiques.
- Le candidat a été exécuté après d'autres séries et la machine est sur batterie :
  ces pourcentages sont **préliminaires**. Répétition explicite prévue avec le
  binaire baseline sauvegardé immédiatement dans les mêmes conditions, afin de
  séparer le gain du cache du bruit/thermique avant décision.
- Validation alternée baseline puis candidat : baseline chaude decode 29,00 /
  28,22 / 28,59 tok/s (médiane **28,59**), prefill médian **28,75** tok/s,
  TTFT médian **0,8351 s** ; candidat chaud 32,52 / 32,50 / 32,85 tok/s
  (médiane **32,52**, **+13,74 %**), prefill médian **33,30** tok/s
  (+15,83 %) et TTFT médian **0,7211 s** (−13,65 %). Preuves
  `build/rust-perf-05-baseline-recheck.jsonl` et
  `build/rust-perf-05-candidate-recheck.jsonl`.
- Décision : conserver. C'est le même cache immuable que le Python et il retire
  des appels Objective-C/Metal du chemin chaud sans changer les sorties. App
  installée inchangée, aucune publication.

### OPT-2026-09-12-RUST-PERF-06 — Cache de kernels par spécialisation comme le Python — rejeté

- Hypothèse : les factories Python `@cache` retrouvent leur
  `CustomKernelFunction` par quelques entiers. Le bridge Rust/C++ recrée à
  chaque dispatch une clé ordonnée qui copie et compare nom, listes d'arguments,
  header et source Metal entiers. Utiliser le nom de spécialisation déjà unique
  comme clé évite ces copies dans le chemin chaud ; rendre aussi le nom du seul
  kernel générique `grouped` dépendant de sa shape garantit l'absence de collision.
- Baseline : runtime RUST-PERF-05 sauvegardé (SHA256
  `e673f285514b957a6e9f3b9ec7a5a7379c0b29e4914553c7a80cc859e54c4484`),
  série alternée précédente decode médian 32,52 tok/s, prefill 33,30 tok/s,
  TTFT 0,7211 s. Même protocole Qwen 24/32 ; batterie débranchée, thermique non
  mesuré. Preuve candidate prévue `build/rust-perf-06-candidate.jsonl`.
- Qualité prévue : smoke GPU, logits `1,2` identiques, sortie complète identique.
  Aucun shader ni paramètre numérique ne change. État : aucun résultat candidat.
- Logits Qwen `1,2` strictement identiques au baseline (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate chaude : decode 32,80 / 33,05 / 33,49 tok/s
  (médiane **33,05**), prefill médian **33,64** tok/s et TTFT médian
  **0,7137 s** ; preuve `build/rust-perf-06-candidate.jsonl`. L'écart contre la
  dernière série PERF-05 n'est qu'environ +1–2 %, donc encore **non concluant**.
  Répéter immédiatement le binaire PERF-05 sous le même état thermique avant
  de conserver ou retirer cette simplification.
- Contrôle alterné avec le binaire PERF-05 exécuté juste après : decode chaud
  33,97 / 33,82 / 33,95 tok/s (médiane **33,95**), prefill médian
  **34,28** tok/s et TTFT médian **0,7004 s** ; preuve
  `build/rust-perf-06-baseline-recheck.jsonl`. Le candidat à 33,05 tok/s est
  donc **−2,65 %** plus lent, malgré l'ordre thermique qui aurait dû l'avantager.
- Décision : **rejet et retrait**. La copie de clé n'est pas le bottleneck ;
  conserver la clé complète qui protège aussi les collisions de métadonnées.

### OPT-2026-09-12-RUST-PERF-07 — Un graphe decode Qwen jusqu'à l'argmax — rejeté

- Hypothèse : le Rust synchronise les logits Qwen puis lance `log_softmax` et
  `argmax` dans une seconde évaluation. Le générateur Python conserve au
  contraire le prochain token dans le graphe Metal jusqu'à la frontière de
  streaming. Exposer un `forward_lazy` uniquement au decode doit fusionner
  modèle + normalisation + argmax en une seule évaluation, sans modifier le
  prefill (toujours eager par token pour borner le graphe et la RAM).
- Baseline : binaire RUST-PERF-05 sauvegardé ; sa dernière série chaude donne
  decode médian 33,95 tok/s, prefill 34,28 tok/s et TTFT 0,7004 s. Même Qwen,
  prompt 24 tokens, génération 32, un warmup + trois tours, batterie
  débranchée, thermique non mesuré. Preuve prévue
  `build/rust-perf-07-candidate.jsonl`.
- Qualité prévue : forward eager/différentiel inchangé, test/sortie complète
  identique. Le chemin lazy n'est utilisé qu'après le premier token sélectionné.
  État : aucun résultat candidat.
- Test du sampler réussi et forward eager `1,2` strictement identique au
  baseline (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate chaude : decode 47,52 / 48,08 / 47,15 tok/s
  (médiane **47,52**), prefill médian **48,29** tok/s et TTFT médian
  **0,4973 s** ; sortie complète identique, preuve
  `build/rust-perf-07-candidate.jsonl`.
- Le saut decode est important mais la baseline immédiate a dérivé pendant les
  séries sur batterie. État encore **préliminaire** : relancer RUST-PERF-05 puis
  le candidat afin de quantifier le gain alterné avant validation.
- Première alternance a révélé une dérive majeure indépendante du patch : le
  binaire PERF-05 est lui aussi monté à **47,99 tok/s** médian, puis le candidat
  relancé juste après est retombé à **34,25 tok/s**. Batterie 83 %, débranchée ;
  `pmset -g therm` ne rapporte aucun warning, mais ces deux fenêtres ne sont pas
  comparables. Preuves `build/rust-perf-07-baseline-recheck.jsonl` et
  `build/rust-perf-07-candidate-recheck.jsonl`.
- État **non concluant** : effectuer une seconde mesure PERF-05 sous le régime
  ralenti actuel. Si elle rejoint ~34 tok/s, ne revendiquer aucun gain PERF-07 ;
  si elle reste ~48, retirer le lazy decode comme régression.
- Seconde mesure du binaire PERF-05 sous le même régime : decode chaud 47,23 /
  47,90 / 46,10 tok/s (médiane **47,23**), prefill médian **48,12** tok/s et
  TTFT médian **0,4990 s** ; preuve
  `build/rust-perf-07-baseline-recheck-2.jsonl`. Le baseline reste donc proche
  de 48 tok/s alors que le candidat relancé était à 34,25 tok/s.
- Décision : **rejet et retrait** du lazy decode. Construire un graphe modèle
  jusqu'à l'argmax est instable/coûteux après retrace ; le premier résultat à
  47,52 tok/s était une coïncidence de la dérive observée aussi sur le baseline.
  Le meilleur code validé reste RUST-PERF-05.

### OPT-2026-09-12-RUST-PERF-08 — Broadcast MoE sans copies comme le Python — validé, code local

- Hypothèse : le chemin Python forme les activations gate/up routées avec
  `broadcast_to(...).reshape(...)`, donc une vue sans copie. Le Rust concatène
  actuellement `2 × top_k` clones du token dans chaque couche MoE, créant une
  opération et un buffer inutiles avant chaque expert QMV. Porter l'opération
  MLX native `broadcast_to` doit réduire decode, TTFT et scratch sans changer
  un calcul numérique.
- Baseline : binaire RUST-PERF-05 sauvegardé ; dernière série stable decode
  médian 47,23 tok/s, prefill 48,12 tok/s, TTFT 0,4990 s. Même Qwen 24/32,
  un warmup + trois tours, batterie 83 % ou moins et débranchée, thermique non
  mesuré. Preuve candidate prévue `build/rust-perf-08-candidate.jsonl`.
- Qualité prévue : smoke `broadcast_to`, logits `1,2` strictement identiques et
  sortie complète identique. État : aucun résultat candidat.
- Smoke GPU réussi ; logits `1,2` et sortie complète strictement identiques au
  baseline (SHA256 logits commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Première série candidate chaude : decode 47,86 / 45,06 / 47,81 tok/s
  (médiane **47,81**), prefill médian **46,05** tok/s, TTFT médian
  **0,5215 s** ; preuve `build/rust-perf-08-candidate.jsonl`.
- Face à la dernière baseline à 47,23 tok/s le decode ne gagne que 1,24 % et
  prefill/TTFT baissent d'environ 4 %, donc résultat **non concluant**. Relancer
  le binaire PERF-05 immédiatement avant décision ; aucun gain mémoire n'est
  revendiqué sans mesure de scratch MLX.
- Baseline immédiate suivante a de nouveau changé de régime : decode médian
  **34,55** tok/s, prefill **34,54** tok/s, TTFT **0,6950 s**, preuve
  `build/rust-perf-08-baseline-recheck.jsonl`. La machine alterne donc des
  plateaux ~34 et ~48 tok/s sans warning thermique, rendant la comparaison 32
  tokens invalide. Protocole complémentaire décidé avant exécution : pour
  baseline puis candidat, un warmup 64 tokens et une génération mesurée jusqu'à
  128 tokens, afin de comparer après montée en fréquence sur une fenêtre plus
  longue.
- Série soutenue baseline : warmup 64 tokens à 34,31 tok/s, puis 128 tokens à
  **34,05 tok/s** decode, **33,78 tok/s** prefill et **0,7116 s** TTFT. Série
  soutenue candidate : warmup 64 tokens à 34,69 tok/s, puis 128 tokens à
  **34,32 tok/s** decode (**+0,81 %**), **35,35 tok/s** prefill (**+4,64 %**)
  et **0,6793 s** TTFT (**−4,54 %**). Preuve brute commune
  `build/rust-perf-08-sustained.jsonl` ; batterie, modèle, prompt et options
  identiques, baseline immédiatement avant candidat.
- Décision : **validé, code local**. Conserver la vue MLX employée par le moteur
  Python : elle supprime une concaténation et ne régresse pas la charge longue.
  Le gain decode est modeste et aucun gain RAM n'est revendiqué, car la métrique
  disponible ne mesure pas séparément les buffers scratch. Logits et texte sont
  strictement identiques. App installée et publication inchangées.

### OPT-2026-09-12-RUST-PERF-09 — Sortie/réduction MoE fusionnée comme le Python — rejeté

- Hypothèse : le Python applique `@mx.compile` à l'ensemble Hadamard de sortie,
  échelle par expert, pondération de routage et réduction top-k. Le Rust expose
  ces étapes comme une chaîne d'opérations génériques distinctes dans
  `finish_and_reduce`. Porter exactement cette frontière de compilation doit
  réduire les dispatchs et buffers intermédiaires de chaque couche MoE, surtout
  au decode `M=1`, sans changer les kernels QMV ni le résultat numérique.
- Baseline : runtime RUST-PERF-08 sauvegardé (SHA256
  `b9ffd7dcb5f73e9f52db8335e4c3c55f04f2f3d97a2fbb29b3add23c6fc44f3b`),
  série soutenue 128 tokens à 34,32 tok/s decode, 35,35 tok/s prefill et
  0,6793 s TTFT. Même Qwen, prompt, options et batterie débranchée ; comparer
  baseline puis candidat sur 64 tokens de warmup et 128 tokens mesurés.
- Qualité prévue : tests Rust, logits `1,2` et texte strictement identiques au
  binaire sauvegardé. État : aucun résultat candidat.
- Build et smoke GPU réussis ; logits `1,2` strictement identiques au baseline
  (2 978 940 octets, SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Série soutenue baseline : warmup 64 à 34,73 tok/s, puis 128 tokens à
  **34,37 tok/s** decode, **34,97 tok/s** prefill et **0,6872 s** TTFT.
  Candidat : warmup 64 à 34,37 tok/s, puis 128 tokens à **34,15 tok/s** decode
  (**−0,65 %**), **34,75 tok/s** prefill (−0,63 %) et **0,6908 s** TTFT
  (+0,53 %). Texte strictement identique ; preuve
  `build/rust-perf-09-sustained.jsonl`.
- Décision : **rejet et retrait**. La compilation explicite de cette chaîne ne
  réduit pas le coût du graphe Rust déjà différé et ajoute une petite régression.

### OPT-2026-09-12-RUST-PERF-10 — Isoler le gain des blocs récurrents Python compilés — validé, diagnostic

- Hypothèse : les QMV/mapped-QMV, split-K, regroupements et transformations MoE
  Rust correspondent désormais au chemin Python. La différence structurante
  restante au decode Qwen est `compile_recurrent_layers`, qui compile chaque
  bloc Gated DeltaNet avec son état explicite. Mesurer le Python avec puis sans
  `MLXL3_COMPILED_RECURRENT_LAYERS` quantifie la part réellement récupérable
  avant tout port complexe.
- Baseline : référence Python RUST-PERF-03 à **53,40 tok/s** decode, 322,62 tok/s
  prefill et 416,25 ms TTFT, même Qwen et protocole 134/32. Répéter dans le même
  processus de benchmark avec compilation activée puis désactivée, batterie
  débranchée ; aucune modification de poids ni de sampling.
- Qualité prévue : texte greedy identique ; ce test est diagnostic et ne modifie
  ni moteur Rust, ni app. État : aucun résultat.
- Mesure dans les mêmes conditions : Python compilé, decode médian
  **48,37 tok/s**, prefill **324,70 tok/s**, TTFT **413,72 ms** ; Python sans
  compilation récurrente, decode **41,51 tok/s**, prefill **285,48 tok/s**,
  TTFT **470,48 ms**. Pics MLX identiques à 12,43 GB et sorties greedy
  identiques. Preuves `build/rust-perf-10-python-compiled.json` et
  `build/rust-perf-10-python-uncompiled.json`.
- Conclusion diagnostic : la compilation explique **+16,50 %** de decode,
  +13,74 % de prefill et −12,07 % de TTFT sur cette fenêtre. Le runtime Rust
  PERF-08 atteint déjà 47–48 tok/s sur son palier rapide, donc son QMV est au
  niveau du Python compilé actuel ; le goulet massif restant est son prefill
  token-par-token. État : **validé, diagnostic seulement**, aucun code intégré.

### OPT-2026-09-12-RUST-PERF-11 — Synchronisation Qwen par blocs au prefill — rejeté

- Hypothèse : le Rust synchronise les logits après chaque token de prompt alors
  que les états GDN/KV créent déjà les dépendances correctes dans le graphe MLX.
  Ne synchroniser que le dernier logits de petits blocs doit amortir les barrières
  CPU/GPU sans changer le calcul, avant le port beaucoup plus large du QMM multi-row.
- Matrice prévue : blocs de 2, 4, 8 puis 16 tokens, même Qwen et prompt de 134
  tokens, 32 tokens générés, un warmup et trois mesures ; arrêter/retirer une
  variante si elle régresse, change les logits ou augmente excessivement la RAM.
  Baseline Rust PERF-08 sauvegardée ; référence soutenue récente 34,37 tok/s
  decode et 34,97 tok/s prefill, mais le critère principal de cette série est le
  prefill alterné sous le même état machine.
- Qualité prévue : logits `1,2`, texte greedy et état final strictement identiques.
  Prototype piloté par `MLXL3_RUST_PREFILL_CHUNK`, à retirer ou figer après choix.
  État : aucun résultat candidat.
- Logits imposés `1,2` strictement identiques au baseline (SHA256 commun
  `ae5f577e91d448fbb78b8cb88f05a6e6a1eee6bdfc2e88e70958e0990bd48243`).
  Sur prompt de 212 tokens, bloc 1 chaud : **47,37 tok/s** prefill,
  **47,13 tok/s** decode et **4,4757 s** TTFT. Bloc 2 : **42,83 tok/s**
  prefill (−9,58 %), **33,54 tok/s** decode (−28,83 %) et **4,9500 s** TTFT
  (+10,60 %), sortie greedy identique. Preuve
  `build/rust-perf-11-matrix.jsonl`.
- Décision : **rejet et retrait**. Le graphe récurrent inter-token plus grand
  provoque un retrace coûteux puis fait retomber le GPU sur le palier lent.
  Les variantes 4/8/16 ont été intentionnellement interrompues avant mesure,
  puisque leur hypothèse est strictement la même et leur graphe encore plus
  grand. Le prochain gain prefill exige un vrai chemin QMM multi-row, pas une
  accumulation de QMV token-par-token.

### OPT-2026-09-12-RUST-PERF-12 — QMM TensorOps EXL3 multi-token en Rust — validé

- Hypothèse : le prefill Rust reste limité à une suite de QMV `M=1` (~47 tok/s)
  alors que le Python sélectionne son QMM TensorOps M5 à partir de 24 lignes
  (~325 tok/s end-to-end). Porter d'abord le kernel QMM dense existant, sans
  réinventer son algorithme, doit fournir la primitive multi-row requise avant
  la généralisation des couches Qwen et du MoE segmenté.
- Première portée : `Exl3Linear` dense, lignes multiples, M5/macOS compatible
  TensorOps ; garder QMV inchangé pour `M=1`. Microbenchmark de matrices réelles
  Qwen sur M=32/64/128, puis comparaison numérique QMM contre une référence
  Python/EXL3 et contre les QMV ligne par ligne. Aucun gain end-to-end ne sera
  revendiqué avant intégration du modèle complet.
- Baseline : prefill Rust Qwen **47,37 tok/s** sur 212 tokens ; Python actuel
  **324,70 tok/s** sur 134 tokens. Batterie débranchée, M5, MLX 0.32.2.
  Contrôles : mêmes formes/dtypes, sorties finies, tolérances FP16 documentées,
  commandes et mesures brutes conservées. État : aucun code ni résultat.
- Premier build réussi, mais la validation a été **interrompue avant le kernel
  candidat** : le nouveau cas du script appelait par erreur l'oracle mono-ligne
  `qmv_exl3` avec une matrice M32, qui l'a correctement rejetée. Corriger
  l'import/appel vers `qmm_exl3` puis relancer ; aucune mesure ni conclusion de
  qualité issue de ce passage.
- Validation corrigée : `python native/check_parity.py --mlx` passe **178 cas**,
  dont les nouvelles matrices TensorOps M=32, K=1..8 et codebooks 0..2. Les
  sorties Rust et Python ont les mêmes motifs binaires FP16 (`uint16`) sur tous
  ces cas. Le kernel compile et s'exécute donc correctement sur le M5 ; état :
  **prototype local validé numériquement**, pas encore intégré au prefill Qwen
  end-to-end et aucun gain modèle revendiqué à ce stade.
- Intégré ensuite au modèle complet par PERF-13. État final : **validé et
  intégré localement** ; les gains end-to-end sont consignés ci-dessous.

### OPT-2026-09-12-RUST-PERF-13 — Prefill Qwen multi-token QMM — validé

- Hypothèse : le principal écart restant vient du prefill Rust qui exécute le
  modèle token par token et ne peut donc jamais sélectionner le QMM TensorOps.
  Faire traverser un bloc de 32 tokens dans les projections, l'attention, le
  Gated DeltaNet et le MoE doit supprimer cette sérialisation sans modifier le
  chemin decode M=1.
- Changement prévu : réutiliser le QMM TensorOps validé par PERF-12 pour les
  projections EXL3 groupées et séparées ; généraliser uniquement les formes
  temporelles déjà supportées par le kernel GDN et les kernels MoE mappés ;
  conserver le chemin mono-token actuel pour le decode.
- Baseline end-to-end : Qwen Rust **47,37 tok/s prefill**, **47,13 tok/s
  decode** sur 212 tokens ; Python actuel **324,70 tok/s prefill**, **48,37
  tok/s decode** sur son protocole 134 tokens. M5, MLX 0.32.2, batterie.
- Protocole : build/tests Rust, parité primitive, puis benchmark CLI avec le
  même modèle et le même prompt 212 tokens. Contrôles : sortie finie, cache
  causal et états GDN valides, decode non régressé ; mesures alternées si le
  plateau thermique change. État : **en cours**, aucun résultat.
- Première commande de contrôle **interrompue avant compilation** : elle visait
  à tort `native/Cargo.toml`, alors que le manifeste est à la racine. Aucun code
  ni kernel n'a été exécuté par ce passage ; relancer avec `Cargo.toml`.
- Le build corrigé passe, avec un avertissement de paramètre devenu inutile
  après factorisation (retiré aussitôt). La parité n'a pas démarré car la
  commande remplaçait `PATH` au lieu de le préfixer et ne trouvait plus
  `python`; aucun résultat kernel supplémentaire issu de ce passage.
- La vue QMM des poids groupés est maintenant contrôlée elle aussi : build
  release réussi puis `native/check_parity.py --mlx` passe **187 cas**, dont 9
  nouveaux cas groupés M=32 (K=2/3/4, trois codebooks), avec égalité bit-à-bit
  FP16 face aux projections Python contiguës. L'avertissement `rust-objcopy`
  reste limité au strip optionnel (`libLLVM.dylib` absent) ; le binaire produit
  et tous les contrôles s'exécutent. État : primitive dense/groupée validée,
  intégration modèle end-to-end encore en cours.
- Premier lancement end-to-end après intégration (prompt CLI court, 8 tokens de
  sortie) : **5,6 tok/s prefill**, **15,5 tok/s decode**, **4297 ms TTFT**. Ce
  passage inclut la compilation à froid de toutes les nouvelles variantes QMM
  par forme et n'est donc pas comparable à la baseline chaude ; résultat
  **non concluant**, à répéter à chaud puis sur le prompt 212 tokens. La sortie
  est finie et le modèle ne crashe pas.
- Mesure comparable dans un bridge résident, après warmup, trois répétitions du
  prompt exact de 212 tokens et 32 tokens greedy : médiane **154,09 tok/s
  prefill** contre 47,37 (**+225,29 %**), **1,3761 s TTFT** contre 4,4757 s
  (**−69,25 %**) et **49,17 tok/s decode** contre 47,13 (**+4,33 %**). Les trois
  répétitions donnent le même SHA256 de texte que la baseline PERF-11 :
  `5aed1d0102507d0399f27be1efab3b223301bde26a23adcf8ecd537ffe38434c`.
  Pic MLX inchangé à **13,0644 GB**. Commande :
  `python3 benchmarks/benchmark_bridge.py <Qwen> --native-binary
  target/release/mlxl3-rs --max-tokens 32 --repeats 3` avec le filler PERF-11
  répété 14 fois. M5, MLX 0.32.2, batterie, thermique non contrôlée.
- Décision : **validé, code local**. Le prefill Qwen utilise des blocs QMM de
  32 tokens ; les résidus <24 et le decode gardent le chemin QMV mono-token.
  L'attention causale, les états GDN et le MoE multi-token sont validés par le
  texte greedy strictement identique. Publication de ce lot encore à faire.
- Contrôle différentiel étendu après le benchmark : build release réussi et
  `native/check_parity.py --mlx` passe **189 cas** bit-à-bit, incluant désormais
  le Gated DeltaNet T=32 avec état initial non nul et le routeur MoE 32×256.
  L'avertissement de strip `rust-objcopy` reste non bloquant et inchangé.
- Suite finale locale : **18 tests unitaires passés** (5 GPU explicitement
  ignorés), test sampler passé, **14 contrats passés**, doc-tests passés et
  `git diff --check` propre. État : prêt à pousser sur
  `codex/rust-performance` ; aucune app installée ni release produite.

### OPT-2026-09-12-RUST-PERF-14 — Taille de bloc Qwen QMM — validé

- Hypothèse : le bloc conservateur de 32 tokens laisse du coût de lancement et
  de routage MoE non amorti. Des blocs de 64 puis 128 peuvent augmenter le
  prefill sans changer les kernels ni le decode ; arrêter dès régression.
- Baseline PERF-13, même bridge/prompt 212/32 et trois répétitions : **154,09
  tok/s prefill**, **49,17 tok/s decode**, **1,3761 s TTFT**, texte SHA256
  `5aed1d...934c`. Pic **13,0644 GB**.
- Protocole : changer uniquement la constante de chunk, build release, warmup
  puis trois répétitions ; contrôler le hash greedy, le pic et le decode. M5,
  MLX 0.32.2, batterie, thermique non contrôlée. Premier candidat : 64.
  État : **en cours**, aucun résultat candidat.
- Bloc 64, trois répétitions : médiane **112,99 tok/s prefill**, **32,81 tok/s
  decode**, **1,8771 s TTFT**, hash et pic inchangés. Le decode simultanément
  tombé de 49 à 33 tok/s montre un changement de palier machine ; comparaison
  brute **non concluante**. Revenir immédiatement à 32 et mesurer sous le même
  palier avant toute décision ; 128 n'est pas lancé à ce stade.
- Retour immédiat au bloc 32 sous le même palier bas : médiane **102,29 tok/s
  prefill**, **32,95 tok/s decode**, **2,0728 s TTFT**, hash/pic inchangés.
  Comparaison alternée valide donc le bloc 64 à **+10,46 % prefill** et
  **−9,44 % TTFT**, avec −0,43 % decode (bruit). Tester maintenant 128 sous le
  même protocole ; 64 est le meilleur candidat conservé jusque-là.
- Bloc 128 : les trois répétitions passent de **111,92 à 191,25 puis 228,76
  tok/s prefill**, pendant que le decode remonte de 31,60 à 45,20 tok/s. Hash
  et pic inchangés. La transition de palier en plein processus empêche une
  comparaison propre ; résultat provisoirement **non concluant**. Revenir à 64
  immédiatement pour obtenir une référence au palier remonté.
- Bloc 64 remesuré au palier haut : médiane **169,26 tok/s prefill**, **48,63
  tok/s decode**, **1,2528 s TTFT**, hash/pic inchangés. Face au bloc 32 au même
  palier (154,09/49,17/1,3761), cela confirme **+9,85 % prefill** et **−8,96 %
  TTFT**, avec −1,10 % decode compatible avec le bruit. Remesurer 128 maintenant
  que le palier est stable.
- Bloc 128 au palier haut stable : médiane **235,68 tok/s prefill**, **48,63
  tok/s decode**, **0,8998 s TTFT**, hash/pic inchangés. Face à 64 : **+39,24 %
  prefill**, **−28,18 % TTFT**, decode inchangé à <0,01 %. Une cause simple est
  aussi éliminée : 128 découpe 212 en 128+84, tous deux QMM, tandis que 64 laisse
  un résidu de 20 sur le chemin token-par-token. Tester 256 (un bloc de 212)
  avant de figer la valeur.
- Bloc 256 sur 212 tokens : médiane **242,23 tok/s prefill**, **48,39 tok/s
  decode**, **0,8755 s TTFT**, hash/pic inchangés, soit encore +2,78 % prefill
  et −2,70 % TTFT face à 128 (decode −0,49 %, bruit). Sur un second prompt de
  **1018 tokens**, bloc 256 : médiane **231,93 tok/s prefill**, TTFT 4,3896 s,
  hash stable et pic inchangé ; le decode court traverse encore un changement
  de palier et n'est pas utilisé. Tester 512 sur ces 1018 tokens.
- Bloc 512 sur 1018 tokens : médiane brute **221,93 tok/s prefill**, TTFT
  4,5875 s, hash/pic inchangés, mais le decode chute simultanément de 43,38 à
  **27,59 tok/s** (premier run 6,19), signal d'un nouveau changement de palier.
  Résultat **non concluant** ; revenir à 256 et comparer immédiatement. Le bloc
  512 n'est pas conservé sans A/B au même palier.
- Retour bloc 256 sur 1018 tokens : médiane **236,81 tok/s prefill**, **41,31
  tok/s decode**, **4,2990 s TTFT**, hash/pic inchangés. Malgré les variations
  de fréquence, 256 dépasse 512 de **+6,71 % prefill** et garde un decode bien
  supérieur sur cette alternance ; 512 est rejeté.
- Décision : **bloc 256 validé et intégré**. Sur le prompt court comparable il
  améliore PERF-13/32 de 154,09 à 242,23 tok/s (**+57,20 %**) et le TTFT de
  1,3761 à 0,8755 s (**−36,38 %**), sans changement de hash, de pic mémoire ni
  de chemin decode. Build release validé ; publication encore à faire.

### OPT-2026-09-12-RUST-PERF-15 — QMM MoE segmenté par expert — en cours

- Hypothèse : le prefill Rust trie déjà implicitement les mêmes routes mais
  exécute encore un QMV expert par slot. Le moteur Python trie les routes par
  expert et réutilise chaque tuile de poids décodée pour toutes les lignes du
  segment. Porter ce chemin existant doit fermer une partie de l'écart entre
  **242,23 tok/s Rust** et **324,70 tok/s Python**, sans modifier le decode M=1.
- Changement prévu : réutiliser le kernel TensorOps segmenté Python, construire
  sur GPU l'ordre, son inverse et la table de segments, puis appliquer gate/up
  et down aux routes triées. Aucun aller-retour CPU et aucun nouvel algorithme.
- Baseline : bloc 256, prompt exact de 212 tokens, 32 tokens greedy, trois
  répétitions chaudes : **242,23 tok/s prefill**, **48,39 tok/s decode**,
  **0,8755 s TTFT**, pic **13,0644 GB**, hash texte
  `5aed1d0102507d0399f27be1efab3b223301bde26a23adcf8ecd537ffe38434c`.
- Protocole : parité primitive bit-à-bit avec le chemin Python segmenté sur
  routes répétées, puis build/tests et benchmark résident identique. Garder le
  chemin mappé actuel pour moins de 24 lignes et pour tout M=1 ; retirer le
  candidat s'il change le texte, le pic ou régresse le prefill alterné.
- Environnement : Apple M5, MLX 0.32.2, batterie, thermique non contrôlée.
  État : **en cours**, aucun résultat candidat.
- Premier contrôle interrompu avant compilation : `cargo fmt --check` a détecté
  uniquement la mise en forme de la constante de bloc 256 déjà intégrée dans
  `main.rs`. Aucun kernel candidat ni benchmark n'a été exécuté. Appliquer le
  formateur officiel puis reprendre le même build ; protocole inchangé.
- Build release après formatage : réussi. L'avertissement `rust-objcopy` reste
  le strip optionnel déjà documenté (`libLLVM.dylib` absent) ; le binaire est
  produit. Aucun résultat numérique ni débit n'est encore attribué au kernel.
- Parité primitive réussie : `native/check_parity.py --mlx` passe **190 cas**,
  dont le nouveau SwitchGLU segmenté sur 64 tokens, quatre experts et top-2.
  La sortie complète est identique bit-à-bit FP16 au moteur Python segmenté ;
  les 189 cas antérieurs restent exacts. Passer au benchmark modèle résident.
- Premier benchmark modèle 212/32, trois répétitions : la première inclut la
  compilation des nouvelles variantes (**124,00 tok/s**), puis les deux tours
  chauds atteignent **450,02** et **449,59 tok/s prefill**. Médiane **449,59
  tok/s**, soit **+85,61 %** face au bloc 256 à 242,23 tok/s ; TTFT médian
  **0,4718 s** contre 0,8755 s (**−46,11 %**). Texte strictement identique,
  hash `5aed1d...934c`, et pic MLX inchangé à **13,0644 GB**. Decode médian
  **45,14 tok/s** ; le chemin M=1 n'appelle aucun nouveau code, et la baisse
  brute face à 48,39 suit les paliers machine déjà documentés, sans causalité
  attribuable. Preuve `build/rust-perf-15-candidate.json`.
- Contrôle suivant enregistré avant exécution : prompt de **1018 tokens**
  (filler PERF-14 répété 76 fois), mêmes 32 tokens greedy et trois répétitions
  résidentes. Exiger hash constant et comparer au bloc 256 à **236,81 tok/s** ;
  cela vérifie quatre blocs successifs et l'absence de gain limité au petit cas.
- Contexte 1018/32 : après le premier tour de compilation à 260,80 tok/s, les
  deux tours chauds atteignent **530,74** et **531,29 tok/s prefill** ; médiane
  **530,74 tok/s**, soit **+124,12 %** face au bloc 256 à 236,81. TTFT médian
  **1,9183 s** contre 4,2990 s (**−55,38 %**), decode médian 46,58 tok/s et pic
  inchangé. Les trois hashes 32 tokens sont identiques entre eux
  (`af6d4bf5...e8770fb`), mais l'ancien contrôle avait seulement 8 tokens :
  comparaison directe impossible. Preuve `build/rust-perf-15-long.json`.
- Contrôle qualité complémentaire enregistré : relancer exactement le même
  prompt avec 8 tokens ; le hash doit rester `8e004879...d19cf` avant de valider.
- Contrôle 1018/8 réussi : trois tours à **528,17 / 529,45 / 529,38 tok/s
  prefill**, hash exact historique `8e0048794f2135a30e5dd2736936612464d9eadba035d5748aa535ce4f4d19cf`
  à chaque fois, pic inchangé et decode médian **46,69 tok/s**. Preuve
  `build/rust-perf-15-long-quality.json`.
- Décision : **validé, code local**. Le prefill chaud atteint **449,59 tok/s**
  sur 212 tokens et **529,38 tok/s** sur 1018 tokens, contre respectivement
  242,23 et 236,81 avant segmentation. Le decode M=1 reste inchangé dans le
  code, la parité primitive et les hashes end-to-end passent, et la RAM poids
  rapportée ne bouge pas. Lancer la suite complète avant publication.
- Suite Rust réussie : **18 tests unitaires passés**, 5 GPU ignorés comme
  prévu, sampler passé, **14 contrats passés**, doc-tests et clippy strict
  réussis. La commande composée s'est ensuite arrêtée sur un ancien nom de
  script `native/check_sampler.py` qui n'existe pas ; aucun test réel n'a
  échoué et les contrôles sampler/contrats venaient déjà de Cargo. Reprendre
  seulement format/diff puis publier le lot validé.
- Contrôle final après garde de compatibilité (>256 experts conserve le chemin
  mappé) : build release réussi, parité **190/190** bit-à-bit et `git diff
  --check` propre. Le warning de strip optionnel reste inchangé. État final :
  **validé, prêt à publier** sur `codex/rust-performance` ; app installée et
  release inchangées.

### OPT-2026-09-12-RUST-LOAD-01 — Chargement et premier token natifs — en cours

- Hypothèse : le chargement Rust Qwen paie `read_exact_at → Vec → memcpy` pour
  chaque tenseur, puis des matérialisations/concaténations séparées pour les
  experts ; le premier prompt paie en plus la compilation des spécialisations
  Metal. Ces coûts doivent être mesurés séparément avant toute modification.
- Antécédents consultés : journal complet, `audit-runtime-2026-09-04.md`,
  `general-performance-2026-09-07.md` et `qwen38-decode-local.md`. Aucun essai
  précédent n'isole le temps de chargement du nouveau runtime Rust ; les TTFT
  chaudes PERF-13/15 excluent explicitement le premier tour de compilation.
- Baseline / candidat : commit `26bf6e2`, moteur Rust release avec MLX 0.32.2,
  checkpoint Qwen3.6-35B-A3B EXL3 2.49 bpw. Aucun changement de poids, kernel,
  sampling ou contexte. Le binaire installé est fermé pendant les mesures.
- Environnement : Apple M5, macOS local, secteur, batterie 100 %, aucun warning
  thermique signalé par `pmset`; cache fichiers macOS non contrôlé.
- Protocole : mesurer le délai processus→`ready`, le `load_seconds` moteur et
  le TTFT du premier prompt puis d'un prompt chaud dans le même bridge. Ajouter
  seulement ces champs au benchmark résident existant ; conserver stdout brut,
  texte/hash, prefill/decode et pic rapporté sous `build/rust-load-01-*.json`.
- Résultats baseline : `ready.load_seconds` **24,756 s**, délai processus→ready
  **40,981 s**, soit **16,225 s avant le chrono interne**. Le premier prompt
  de 15 tokens prend **1,335 s TTFT** à 11,28 tok/s prefill ; le suivant,
  après compilation, **0,683 s** à 39,55 tok/s. Temps processus total 43,54 s,
  CPU 10,09 s utilisateur + 25,65 s système. Hashes enregistrés dans
  `build/rust-load-01-baseline.json`, temps dans `*.time`.
- Isolation de l'inspection via un registre temporaire : **15,96 s** mur,
  4,40 s utilisateur + 11,47 s système, 348 MB RSS max
  (`rust-load-01-inspect.*`). Le bridge inspecte actuellement le checkpoint
  une fois avant son chrono puis le chargeur de modèle l'inspecte une seconde
  fois : le double scan explique la quasi-totalité des 40,98 s.
- Qualité : sorties greedy finies ; hash du prompt chaud historique exact
  `8e004879...d19cf`. Aucune modification du moteur dans cette mesure.
- Intégration : instrumentation locale du benchmark uniquement ; app et GitHub
  inchangés. Conclusion : **validé, diagnostic**.

### OPT-2026-09-12-RUST-LOAD-02 — Index checkpoint en table de hachage — en cours

- Hypothèse : les 124 579 tenseurs et les 31 243 entrées de stockage sont
  désérialisés/consultés dans des `BTreeMap`, alors que l'ordre n'est utilisé
  ni par le loader ni par les kernels. Une `HashMap` standard doit supprimer
  les insertions et recherches logarithmiques sans relâcher les validations.
- Baseline : inspection isolée **15,96 s** et bridge→ready **40,98 s** sur
  Qwen3.6 2.49 bpw, commit `26bf6e2`, mêmes conditions secteur/M5.
- Protocole : changer uniquement les tables du header et du checkpoint ; tests
  contrats, inspection isolée deux fois puis bridge froid identique. Garder le
  tri explicite des plages et des modules, ainsi que toutes les erreurs de
  doublon/index/forme. La sortie greedy et le hash doivent rester identiques.
- Résultats : contrats 14/14 et build release réussis, mais inspections isolées
  **16,59 s** puis **16,16 s**, contre 15,96 s baseline. CPU pratiquement
  identique (~4,5 s utilisateur, ~11,6 s système) et RSS plus haute
  (~396 MB contre 348 MB). Aucun gain ; sortie `register` identique.
- Conclusion : **rejeté et retiré**. Le coût n'est pas la structure d'index.
  Étape diagnostic enregistrée avant exécution : échantillonner le processus
  d'inspection et chronométrer parsing header, validation stockage et index,
  sans modifier le résultat ni désactiver un contrôle.
- Échantillonnage 5 s : 3 643/3 643 échantillons du thread principal sont dans
  `serde_json::from_reader(File)` lors de la lecture du gros JSON de
  quantification, majoritairement bloqués dans des appels `read`. Preuve
  `build/rust-load-02-inspect.sample.txt`. Le lecteur `File` non bufferisé,
  pas la validation EXL3, est donc le goulet établi.

### OPT-2026-09-12-RUST-LOAD-03 — JSON checkpoint bufferisé — en cours

- Hypothèse : entourer les trois lectures JSON du checkpoint d'un `BufReader`
  standard évite les appels système minuscules observés, sans changer le parseur,
  les structures ni une seule validation.
- Baseline : inspection **15,96 s** ; bridge processus→ready **40,98 s** dont
  `load_seconds` 24,76 s. Profil LOAD-02 : 100 % des échantillons dans la
  désérialisation `File` non bufferisée du manifeste de quantification.
- Protocole : modification standard-library limitée à `checkpoint::inspect`,
  contrats 14/14, build release, deux inspections isolées puis bridge complet.
  Comparer premier TTFT et hash greedy ; aucune modification kernels/poids.
- Résultats : contrats **14/14** et build release réussis. L'inspection isolée
  tombe à **0,66 s** puis **0,32 s**, contre **15,96 s** (−95,9 à −98,0 %).
  Le bridge complet atteint `ready` en **10,865 s** contre **40,981 s**
  (−73,5 %), avec `load_seconds` **10,533 s** contre 24,756 s. Le premier
  prompt après un build release froid révèle toutefois la compilation Metal :
  **7,728 s TTFT** à 1,94 tok/s, puis **0,226 s** et 119,79 tok/s au second
  prompt. Hash chaud exact `8e004879...d19cf`; premier hash
  `620a...2877`, inchangé. Preuves `build/rust-load-03-inspect-{1,2}.*` et
  `build/rust-load-03-bridge.{json,time}`.
- Décision : **validé, code local** pour le chargement. Le buffering standard
  supprime le goulet sans modifier les poids, kernels ou sorties. Le TTFT froid
  est maintenant le coût dominant et doit être traité séparément.

### OPT-2026-09-12-RUST-LOAD-04 — Warmup Metal avant `ready` — en cours

- Hypothèse : le runtime Rust ne précompile aucun graphe, donc le premier prompt
  utilisateur paie les spécialisations Metal. Un passage synthétique remis à
  zéro avant `ready` doit déplacer ce coût dans le chargement et réduire le TTFT
  sans conserver de KV ni changer le texte.
- Baseline après LOAD-03 : processus→ready **10,865 s**, premier TTFT froid
  **7,728 s**, deuxième TTFT **0,226 s**. Qwen utilise QMM à partir de 24
  tokens et le chemin MoE segmenté à partir de 64 ; les autres architectures
  n'ont pas de prefill batch natif dans ce dispatcher.
- Changement prévu : warmup Qwen de 64 tokens puis un token M=1 ; un seul token
  M=1 pour Gemma/LFM/Ling. Évaluer logits/état, puis appeler le reset existant,
  y compris après erreur. Aucun cache, kernel ou poids supplémentaire.
- Protocole : build/tests, bridge Qwen froid identique, comparer
  processus→ready + premier TTFT et le hash greedy. Le premier TTFT doit se
  rapprocher du tour chaud ; le coût total lancement→premier token ne doit pas
  régresser. Intégration : prototype local, résultats non mesurés.
- Premier contrôle interrompu avant exécution : `cargo` n'est pas présent dans
  le `PATH` de cette session (`command not found`). Aucun test ni benchmark n'a
  démarré ; reprendre avec le toolchain local explicite, protocole inchangé.
- Deuxième contrôle interrompu avant compilation : le toolchain explicite est
  disponible, mais le dépôt utilise le manifeste racine et non
  `native/Cargo.toml`. Le formatage officiel a été appliqué ; aucun test n'a
  démarré. Reprendre depuis `Cargo.toml` sans changer le candidat.
- Troisième contrôle interrompu par le build script avant compilation C++ :
  `MLXL3_MLX_ROOT` n'était pas défini dans ce worktree isolé. Aucun test n'a
  démarré. Reprendre avec l'installation MLX 0.32.2 locale explicite ; code et
  protocole inchangés.
- Première compilation réelle rejetée par Rust avant link : le type d'erreur
  de la closure de reset n'était pas inférable (`E0282/E0283`). Aucun binaire
  ni benchmark candidat. Ajouter l'annotation `Result<()>` demandée par le
  compilateur, sans changer le comportement prévu.
- Build et contrats **14/14** réussis. Candidat Qwen 64+1 : processus→ready
  **28,376 s** (`load_seconds` 27,781 s), premier TTFT **0,470 s** à
  32,01 tok/s, puis **0,200 s** à 136,26 tok/s. Le warmup supprime 7,26 s du
  premier TTFT mais ajoute 17,51 s au chargement ; lancement→premier token
  passe d'environ **18,59 s à 28,85 s**. Pic poids inchangé 13,0644 GB.
  Preuve `build/rust-load-04-bridge.{json,time}`.
- Décision : **rejeté**. Précompiler QMM et MoE segmenté pour 64 tokens au
  démarrage dégrade nettement le temps utilisateur total. Remplacer par un
  essai M=1 séparé, qui ne compile que le chemin decode/petit prefill.

### OPT-2026-09-12-RUST-LOAD-05 — Warmup Metal M=1 — en cours

- Hypothèse : un token synthétique compile les kernels communs au decode et au
  petit prefill, qui dominaient le premier prompt de 15 tokens, sans payer les
  spécialisations QMM/segmented du warmup LOAD-04.
- Baseline LOAD-03 : ready **10,865 s**, premier TTFT **7,728 s**, total
  **18,59 s**. Candidat LOAD-04 rejeté : ready 28,376 s, TTFT 0,470 s.
- Changement prévu : réutiliser exactement le helper et son reset, mais faire
  un seul `forward(0)` pour toutes les architectures. Build/14 contrats, bridge
  froid, hash/output et pic obligatoires. Valider seulement si le total
  lancement→premier token baisse et si le TTFT se rapproche du chaud.
- Résultats : build release et contrats **14/14** réussis. Processus→ready
  **19,423 s**, premier TTFT **0,907 s** à 16,57 tok/s, puis **0,255 s** à
  105,85 tok/s. Le warmup M=1 ajoute 8,56 s au ready et retire 6,82 s du
  premier TTFT ; lancement→premier token monte de **18,59 s à 20,33 s**.
  Pic poids inchangé 13,0644 GB. Preuve
  `build/rust-load-05-bridge.{json,time}`.
- Décision : **rejeté** selon le critère annoncé. Le TTFT affiché est meilleur,
  mais le délai réel utilisateur régresse. Retirer entièrement le helper et
  valider une baseline contemporaine LOAD-03 : cette répétition vérifie que la
  comparaison ne dépend pas d'un palier thermique/cache entre builds.
- Arrêt demandé avant la répétition de baseline : le helper et son appel ont
  été **entièrement retirés**. Contrats **14/14**, build release, `py_compile`,
  formatage et `git diff --check` réussis. Aucun warmup n'est intégré ni publié.
  État final : **rejeté et retiré**.

### État de publication du lot chargement — 2026-09-12

- LOAD-03 bufferisé est **validé et prêt à publier** : inspection 15,96 s →
  0,32–0,66 s, processus→ready 40,98 s → 10,87 s, hashes conservés.
- L'instrumentation benchmark processus→ready/première génération accompagne
  le changement. LOAD-04/05 restent documentés comme essais négatifs, sans
  code runtime résiduel. App installée et release inchangées à cet instant.

### OPT-2026-09-13-RUST-LOAD-06 — Chargement commun sans copie intermédiaire — en cours

- Demande : poursuivre les optimisations chargement/TTFT pour toutes les
  architectures, pas uniquement Qwen. Le point commun réel est
  `checkpoint_array`: chaque tenseur fait actuellement fichier → `Vec<u8>`
  Rust → allocation MLX → `memcpy`, pour Gemma 4, LFM2/LFM2-MoE, Ling 3 et
  Qwen3.5 dense/MoE.
- Antécédents consultés : journal complet jusqu'à LOAD-05, rapports chargement
  cités, diagnostic/production du skill inference-engineering. Les warmups
  globaux LOAD-04/05 sont rejetés car ils déplacent plus de temps qu'ils n'en
  retirent. Aucun essai historique trouvé pour une lecture directe dans le
  buffer MLX natif.
- Baseline/candidat : commit publié `f544027` (JSON déjà bufferisé). Modèles
  locaux disponibles pour mesures : LFM2.5 1.2B Thinking 4 bpw, LFM2.5 2.6B
  4 bpw et Qwen3.6 35B-A3B 2.49 bpw. Aucun checkpoint Gemma/Ling local : leur
  chemin partagé sera couvert par tests/compilation mais aucun pourcentage ne
  leur sera inventé.
- Protocole baseline : bridge release, processus→ready, `load_seconds`, premier
  TTFT puis tour chaud, un processus par modèle ; AC/thermique à relever.
  Candidat prévu seulement après ces mesures : `pread` robuste directement
  dans une allocation MLX avec la même validation de taille/inode/offset,
  sans mmap, lazy loading ni changement de dtype. Parité octets/checkpoints,
  14 contrats, build, puis mêmes trois bridges. Preuves sous
  `build/rust-load-06-*`.
- Baselines locales, batterie 85 % débranchée, aucun warning thermique :
  LFM1.2 ready **0,719 s**, `load_seconds` **0,399 s**, premier TTFT 0,179 s,
  chaud 0,254 s ; LFM2.6 ready **0,658 s**, `load_seconds` **0,644 s**,
  premier TTFT 0,465 s, chaud 0,537 s. Pics poids rapportés 0,793/1,764 GB.
  Qwen réutilise la mesure LOAD-03 au même code : ready 10,865 s,
  `load_seconds` 10,533 s. Preuves `build/rust-load-06-lfm{12,26}-baseline.*`.
- Lecture : les petits LFM sont déjà sous une seconde ; tout candidat commun
  doit donc éviter une régression mesurable chez eux. Le surcoût Qwen n'est pas
  uniquement proportionnel aux octets et inclut l'assemblage de ses experts.
- Premier contrôle candidat : formatage, contrats **14/14** et build release
  réussis. La commande du smoke GPU a sélectionné **0 test** car `--exact`
  recevait le nom court et non le chemin de module ; aucun succès matériel ne
  lui est attribué. Relancer seulement `array::tests::native_array_and_kernel_smoke`
  avec `--lib`, puis benchmarker si la lecture offset/longueur est exacte.
- Smoke GPU corrigé : **1/1 réussi**, y compris lecture de deux UInt32 à un
  offset non nul depuis un fichier réel. Candidat LFM1.2 : ready **0,719 →
  0,562 s** (−21,9 %), load **0,399 → 0,257 s** (−35,6 %). LFM2.6 : ready
  **0,658 → 0,518 s** (−21,3 %), load **0,644 → 0,509 s** (−21,0 %).
  Événements/textes des deux générations identiques. Les TTFT courtes fluctuent
  dans les deux sens et aucun gain d'inférence n'est attribué à cette lecture.
- Qwen face à la mesure de la veille : ready **10,865 → 11,166 s** et load
  **10,533 → 10,816 s** (+2,7 %), tandis que TTFT/decode changent fortement de
  palier sans changement de hash. Cette comparaison non alternée ne permet pas
  d'attribuer la petite régression au candidat. Construire le commit `f544027`
  dans un worktree/target séparé et mesurer baseline puis candidat immédiatement,
  avant décision. Preuves `build/rust-load-06-*-candidate.*`.
- A/B Qwen contemporain, baseline puis candidat : ready **9,145 → 7,625 s**
  (**−16,61 %**) et load **8,469 → 7,251 s** (**−14,38 %**). Temps processus
  total **13,34 → 9,70 s**, CPU système 3,13 → 2,66 s, RSS max hôte
  3,742 → 2,549 GB. Hashes premier/chaud exacts respectivement
  `620a...2877` et `8e004879...d19cf`; le TTFT change de palier et n'est pas
  attribué à la lecture. Preuves `rust-load-06-qwen-{baseline,candidate}-recheck.*`.
- Contrôle final enregistré avant décision : refaire le même A/B contemporain
  sur les deux LFM rapides, où 0,1 s peut être sensible au cache de fichiers.
  Conserver uniquement si ready/load restent non régressifs et hashes identiques.
- A/B LFM contemporain réussi : LFM1.2 ready **0,304 → 0,144 s** (−52,7 %),
  load **0,289 → 0,137 s** (−52,5 %) ; LFM2.6 ready **0,597 → 0,291 s**
  (−51,3 %), load **0,588 → 0,284 s** (−51,8 %). Hashes premier prompt et
  tour chaud strictement identiques pour les deux modèles. Preuves
  `rust-load-06-lfm{12,26}-{baseline,candidate}-recheck.json`.
- Décision : **validé, code local**. La lecture directe supprime une allocation
  hôte et une copie de chaque tenseur, réduit le chargement sur trois tailles et
  deux familles mesurées, et conserve le fallback validant les tenseurs BOOL.
  Gemma/Ling compilent le même `checkpoint_array`, mais restent **non mesurés**
  faute de checkpoint local. App installée et GitHub inchangés.

### OPT-2026-09-13-RUST-LOAD-07 — Inspection unique du checkpoint — en cours

- Hypothèse : le bridge inspecte le checkpoint pour ses métriques, puis chaque
  loader d'architecture répète la même inspection avant les poids. LOAD-03 a
  réduit ce scan à 0,32–0,66 s sur Qwen, mais il reste entièrement évitable et
  concerne Gemma, LFM, Ling et Qwen.
- Antécédents : LOAD-01 a établi le double appel ; LOAD-03 l'a bufferisé sans
  le supprimer. Pas de cache global/stale : passer explicitement le
  `Checkpoint` déjà validé aux loaders et garder leurs `load(path)` publics
  comme wrappers pour les autres commandes/tests.
- Baseline : code LOAD-06, A/B récents : Qwen ready 7,625 s/load 7,251 s ;
  LFM1.2 0,144/0,137 s ; LFM2.6 0,291/0,284 s. Candidat : aucune modification
  de poids, dtype, ordre de chargement ou kernels.
- Protocole : 14 contrats, build, parités existantes si le chargement passe,
  bridge des trois checkpoints, hashes exacts. Mesurer surtout Qwen ; sur les
  LFM sub-seconde, marquer le résultat bruité plutôt que revendiquer des ms.
- Build et contrats **14/14** réussis, trois checkpoints chargés et hashes
  exacts. Mesure non alternée contre LOAD-06 : LFM1.2 ready 0,144 → 0,455 s,
  LFM2.6 0,291 → 0,344 s, Qwen 7,625 → 8,422 s. Ces régressions contredisent
  le travail supprimé et suivent un changement global de palier/cache ; elles
  sont **non concluantes**.
- Diagnostic enregistré avant exécution : ajouter temporairement au même
  binaire un drapeau privé qui force l'ancienne seconde inspection, puis lancer
  ancien/nouveau dans la même fenêtre sur les trois modèles. Retirer le drapeau
  après mesure. Cela isole le seul changement sans reconstruire deux moteurs ni
  conserver une option de production.
- A/B dans le même binaire réussi : LFM1.2 double→simple inspection ready
  **0,556 → 0,144 s**, load **0,243 → 0,137 s** ; LFM2.6 **0,492 → 0,298 s**,
  load **0,484 → 0,290 s** ; Qwen **7,964 → 7,160 s**, load **7,640 →
  6,841 s**. Les hashes premier/chaud restent exacts pour les trois modèles.
  Preuves `build/rust-load-07-*-{repeat,single}-inspect.json`.
- Décision : **validé, code local**. Le drapeau diagnostic a été entièrement
  retiré ; le bridge passe désormais l'unique checkpoint aux quatre loaders.
  Leurs API `load(path)` conservent inspection+validation pour tous les autres
  appels. Gemma/Ling compilés mais non mesurés faute de poids locaux. App et
  GitHub inchangés.

### OPT-2026-09-13-RUST-LOAD-08 — Assemblage des poids MoE — diagnostic en cours

- Hypothèse : après lecture directe et inspection unique, Qwen charge encore
  en 6,84 s alors que les LFM denses sont sous 0,3 s. Le loader commun
  `Exl3SwitchGlu` crée une Array MLX par tenseur de chaque expert, puis plusieurs
  concaténations et matérialisations par couche ; Qwen répète cela pour 256
  experts. Le même loader est utilisé par les MoE Qwen, LFM, Gemma et Ling.
- Antécédents : LOAD-06 prouve que les copies fichier→Vec dominaient les modèles
  denses mais pas tout Qwen. Les essais historiques MoE concernent les kernels
  d'inférence, pas l'assemblage des poids au chargement.
- Protocole diagnostic : échantillonner 5 s du bridge Qwen pendant le chargement
  final LOAD-07, puis ne prototyper un pack direct qu'en présence de temps
  significatif dans `concatenate`/allocations/copies. Aucune modification des
  layouts ou du calcul avant cette preuve. Modèles Gemma/Ling non disponibles.
- État intermédiaire : une première tentative de rebuild du binaire de profil
  a été interrompue avant compilation, car `MLXL3_MLX_ROOT` n'était pas fourni.
  Aucune mesure n'a été produite ; relance prévue avec MLX 0.32.2 explicite.
- Résultat du diagnostic : bridge Qwen terminé correctement en **7,521 s**
  (`build/rust-load-08-profile.out`). L'échantillonnage 5 s place `pread` en
  tête avec **2 615** échantillons au sommet de pile ; les trois lectures des
  treillis experts dans `Exl3SwitchGlu::from_checkpoint_names` représentent à
  elles seules 607 + 594 + 576 échantillons visibles. Les concaténations et
  attentes GPU restent très minoritaires devant les lectures fichier
  (`build/rust-load-08-profile.sample.txt`).
- Décision : **rejeté** pour le prototype de pack/concat demandé par
  l'hypothèse initiale : il déplacerait les mêmes octets et ajouterait un
  buffer hôte. Prochaine piste à documenter séparément : supprimer la copie
  fichier→allocation MLX par mapping/chargement natif, si l'API MLX permet de
  conserver correctement la durée de vie du stockage.
- État d'intégration : diagnostic seulement ; aucun changement MoE appliqué.

### OPT-2026-09-13-RUST-LOAD-09 — Poids mappés sans copie — en cours

- Hypothèse : le chemin commun alloue une Array MLX puis copie chaque plage
  safetensors avec `pread`. Sur les MoE à milliers de tenseurs, ces copies
  dominent le chargement. Une vue MLX adossée à un mapping fichier pourrait
  supprimer la copie et rendre le coût initial proportionnel aux pages
  réellement touchées, sans changer les poids ni les kernels d'inférence.
- Périmètre : loader EXL3 natif partagé par Qwen, LFM, Gemma et Ling ; aucun
  chemin spécifique à une architecture.
- Baseline : LOAD-07/08, Qwen ready **7,16–7,52 s**, LFM dense **0,14–0,30 s**.
- Protocole : vérifier d'abord les constructeurs et garanties de durée de vie
  de MLX 0.32.2. Si une API publique sûre existe, prototyper derrière le même
  `checkpoint_array`, puis comparer A/B mêmes poids avec hashes de sortie,
  temps ready/load, RAM processus et tests de contrat. Sinon marquer bloqué,
  sans créer une abstraction propriétaire.
- Résultat : **bloqué/rejeté sans prototype**. MLX expose bien un constructeur
  de `array` sur pointeur utilisateur, mais son allocator Metal exige un buffer
  externe réutilisable et les offsets safetensors ne donnent pas directement
  un buffer page-aligné par tenseur. La discussion officielle MLX #615 décrit
  le même conflit mmap/offset Metal et le déplacement imprévisible du coût vers
  les page faults d'inférence. Notre chemin direct `pread` écrit déjà dans
  l'allocation unifiée finale ; le mapping ajouterait ici durée de vie, vues et
  risque de régression TTFT sans preuve d'un gain global.
- Preuves : headers MLX 0.32.2 locaux `mlx/array.h`, `mlx/allocator.h` et source
  allocator officielle `ml-explore/mlx`; discussion officielle
  https://github.com/ml-explore/mlx/discussions/615.
- État d'intégration : aucune modification mmap appliquée.

### OPT-2026-09-13-RUST-LOAD-10 — Lecture MoE concurrente — validé

- Hypothèse : LOAD-08 mesure un loader MoE essentiellement bloqué dans des
  `pread` indépendants, exécutés aujourd'hui strictement en série. Quelques
  workers stdlib peuvent maintenir plusieurs lectures SSD en vol et mieux
  alimenter la mémoire unifiée, sans modifier les layouts ni les valeurs.
- Périmètre : `Exl3SwitchGlu`, donc MoE Qwen, LFM, Gemma et Ling. Les modèles
  denses conservent le loader direct LOAD-06.
- Baseline : Qwen ready **7,16–7,52 s**, load **6,84–7,52 s** ; hashes de sortie
  LOAD-07 exacts. Conditions thermiques non garanties, donc variantes alternées.
- Protocole : helper local utilisant `std::thread::scope`, ordre de sortie
  déterministe, calibration 1/2/4/8 workers via `MLXL3_LOAD_THREADS`. Comparer
  au moins deux répétitions Qwen par variante, vérifier hashes first/warm,
  contrats, smoke GPU et RAM. Garder uniquement un gain robuste ; 1 worker doit
  rester un oracle fonctionnel.
- Résultat matrice alternée (médiane de 2 lancements par variante) :
  - 1 worker : ready **7,954 s**, load **7,446 s** ;
  - 2 workers : ready **5,130 s**, load **4,738 s** ;
  - 4 workers : ready **4,100 s**, load **3,716 s** ;
  - 8 workers : ready **3,970 s**, load **3,568 s**.
  Les 8 workers donnent **−50,09 % ready** et **−52,09 % load** face à
  l'oracle 1 worker ; même 4→8 reste favorable (−3,16 % ready). Les hashes
  first et warm sont identiques sur les 8 exécutions et la RAM moteur annoncée
  reste identique à **13,064 GB**.
- Preuves : `build/rust-load-10-qwen-{1,2,4,8}-{a,b}.json`. Les 14 contrats et
  le smoke GPU batched/direct passent avant la matrice.
- État intermédiaire : gain Qwen validé ; contrôle RSS hôte 1/8 workers et
  non-régression des modèles locaux LFM encore à effectuer avant décision.
- Contrôle `/usr/bin/time -l` : 1 worker **7,65 s**, **2 407 940 096 B** RSS
  max ; 8 workers **3,90 s**, **2 439 561 216 B** RSS max. Le parallélisme
  ajoute **31,6 MB / 1,31 %** de RSS hôte mesuré, sans modifier la RAM moteur,
  pour −49,0 % de temps mur sur ce contrôle
  (`build/rust-load-10-qwen-time-{1,8}.{out,txt}`).
- Limite locale : les deux LFM disponibles sont denses ; ils vérifient le
  chemin commun LOAD-06/07 mais n'exercent pas `Exl3SwitchGlu`. Gemma/Ling MoE
  ne sont pas présents localement, donc leur bénéfice reste **non mesuré** et
  ne doit pas être chiffré malgré le partage exact du loader.
- Validation intermédiaire : suite Rust complète **33 tests passés**, 5 GPU/
  tokenizer explicitement ignorés. Le premier `clippy -D warnings` a échoué
  uniquement sur la préférence mécanique `chunks_exact(2)` →
  `as_chunks::<2>()`; aucune exécution ou mesure affectée, correction prévue
  avant relance.
- Validation finale : défaut fixé à `min(cœurs disponibles, 8)`, surcharge
  possible avec `MLXL3_LOAD_THREADS=1..32`. Qwen par défaut : ready **4,288 s**,
  load **3,611 s**, hashes first/warm identiques à la matrice. LFM 1.2B et 2.6B
  denses chargent et génèrent avec le même hash first que LOAD-07
  (`build/rust-load-10-{qwen,lfm12,lfm26}-default.json`).
- Contrôles finaux : `cargo fmt --check`, `clippy --all-targets -D warnings`,
  build release, suite Rust **33 passés / 5 ignorés / 0 échec**, smoke GPU
  direct+batché **1/1**, et parité MLX **190/190** bit-à-bit.
- Décision : **validé**. Le gain mesuré porte sur Qwen MoE ; la même primitive
  est intégrée aux loaders MoE LFM/Gemma/Ling mais reste non mesurée faute de
  checkpoints locaux. Aucun gain n'est revendiqué pour leurs variantes denses.
- État d'intégration : **code local uniquement** sur `codex/rust-performance` ;
  app installée et GitHub inchangés, aucun push demandé à ce stade.

### OPT-2026-09-13-RUST-PERF-16 — Prefill LFM2 multi-token — validé

- Hypothèse : les LFM2 dense et MoE passent encore chaque token du prompt dans
  le modèle séparément, alors que les projections QMM, l'attention causale,
  ShortConv et le SwitchGLU segmenté acceptent déjà plusieurs lignes. Faire
  traverser un bloc complet doit amortir les poids et les dispatchs sans toucher
  au decode `M=1`, aux poids, au sampling ni à la précision.
- Antécédents consultés : journal complet jusqu'à LOAD-10, rapports decode,
  runtime et prefill cités à sa racine, ainsi que l'implémentation LFM2/LMF2-MoE
  de MLX-LM 0.32.0. PERF-11 a rejeté l'accumulation paresseuse de QMV ; le présent
  essai utilise le QMM multi-row validé par PERF-12/13 et ne le répète pas.
- Baseline/candidat : branche locale `codex/rust-performance`, HEAD `f544027`
  plus LOAD-06/07/10 validés non commités. Modèles locaux LFM2.5 1.2B Thinking
  4 bpw, LFM2.5 2.6B 4 bpw et LFM2.5 8B-A1B MoE 3.10 bpw. Baseline = chunks de
  1 ; candidat = même chemin par blocs, avec QMV conservé pour le decode.
- Environnement : Apple M5 24 Gio, macOS local, MLX 0.32.2, batterie 96 %
  débranchée, aucun avertissement thermique/performance ; fréquences non
  instrumentées. Aucun autre moteur MLXL3 actif au départ.
- Protocole : bridge résident, prompt français fixe répété, 64 tokens greedy,
  un warmup exclu puis trois répétitions. Relever tokens réels, prefill, TTFT,
  decode, hash et pic pour les trois modèles. Avant le benchmark candidat,
  comparer une continuation forcée sérielle/batchée, logits et tous caches
  bit-à-bit ; tester notamment le décalage causal attention et la fenêtre
  ShortConv. Preuves sous `build/rust-perf-16-*`.
- Baselines, trois tours chauds : LFM1.2, 217 tokens de prompt, **91,87 tok/s
  prefill**, **2,3622 s TTFT**, **107,40 tok/s decode**, pic 0,793 GB ; LFM2.6,
  202 tokens, **46,62 tok/s**, **4,3336 s**, **54,89 tok/s**, pic 1,764 GB ;
  LFM8 MoE, 201 tokens, **73,29 tok/s**, **2,7427 s**, **102,96 tok/s**, pic
  3,942 GB. Les trois hashes sont stables entre répétitions pour chaque modèle.
  Preuves `build/rust-perf-16-lfm{12,26,8}-baseline.json`.
- État : **en cours**. Baseline validée ; aucun candidat encore compilé.
  Intégration : essai local ; app et GitHub inchangés.
- Premier contrôle interrompu avant compilation : `cargo fmt --check` demande
  uniquement la mise en forme standard de deux chaînes d'appels LFM. Aucun
  test, modèle ou benchmark candidat n'a été exécuté. Appliquer le formateur
  officiel puis reprendre le même build, sans changement du protocole.
- Build release candidat réussi. Premier contrôle batch LFM1.2 face au modèle
  Python batché : tous les caches exportés passent bit-à-bit, mais 118/65 536
  logits finaux diffèrent d'un bit FP16. Le Python projette les 32 positions du
  head puis sélectionne la dernière ; le Rust sélectionne d'abord la dernière
  activation et garde le head QMV, ce qui change la partition QMM sans changer
  le decode. Résultat **non concluant**, aucun benchmark candidat : comparer
  maintenant batch Rust et référence Rust sérielle, logits et caches complets.
- En projetant le head sur le bloc comme la production, LFM1.2 puis LFM2.6
  passent chacun logits et tous caches bit-à-bit face au Python batché. LFM8
  s'arrête avant calcul expert : le routeur biaisé Metal est artificiellement
  limité à une ligne, bien que chaque SIMD group soit indépendant. Aucun
  benchmark candidat. Étendre ce même kernel à une grille de lignes, puis
  ajouter un cas multi-row à la matrice de parité avant de relancer LFM8.
- Routeur biaisé multi-row validé bit-à-bit dans la matrice MLX, qui passe
  désormais **191/191** cas. LFM8 batch 32 franchit le routeur mais diverge au
  premier cache observé de la couche attention 10 (296/32 768 octets), après
  plusieurs couches MoE ; aucun benchmark lancé. Tester 64 tokens, seuil exact
  du SwitchGLU segmenté déjà validé, afin de distinguer le fallback mappé
  multi-row du chemin QMM segmenté avant toute modification supplémentaire.
- Première commande 64 tokens interrompue avant chargement : `seq -s,` de BSD
  a produit une liste avec séparateur final, refusée par le parseur du checker.
  Aucun modèle/GPU exécuté et aucun résultat ; reconstruire la liste sans virgule
  terminale puis reprendre strictement le même contrôle.
- LFM8 à 64 tokens franchit le chemin segmenté mais diverge plus tard, au
  cache attention de la couche 14 (9 309/65 536 octets). Le routage multi-row
  est exact isolément ; la chaîne experte complète ne satisfait donc pas le
  contrat batch strict de ce checkpoint. Décision : ne pas activer le batch
  sur `lfm2_moe`, retirer l'extension de routeur devenue sans consommateur, et
  mesurer seulement les LFM denses 1.2B/2.6B dont la parité complète passe.
- Nettoyage appliqué : extension multi-row du routeur biaisé et son cas de test
  retirés ; le batch est maintenant exposé uniquement si toutes les couches
  feed-forward du LFM sont denses. Le LFM8 MoE reste sur les chunks de 1.
- Contrôles après garde dense : LFM1.2 et LFM2.6, bloc de 32 tokens, logits et
  tous caches **bit-à-bit** face à la production Python batchée ; LFM8 MoE,
  huit étapes sérielles, logits et tous caches **bit-à-bit**. Build release
  réussi ; seul l'avertissement `rust-objcopy`/`libLLVM.dylib` déjà connu reste.
  Prochaine étape : benchmark candidat trois tours, puis contrôle long >256.
- Benchmark candidat, même prompt/tokens/hashes et trois tours : LFM1.2
  **91,87 → 2 059,59 tok/s prefill (+2 141,8 %)**, TTFT médian **2,3622 →
  0,1055 s (-95,5 %)** ; LFM2.6 **46,62 → 836,59 tok/s (+1 694,6 %)**,
  TTFT **4,3336 → 0,2417 s (-94,4 %)**. Les hashes des trois sorties de chaque
  modèle sont identiques à la baseline. Decode observé +15,1 %/+7,4 %, mais
  non revendiqué car le chemin decode n'a pas changé et les tours sont courts.
- Contrôle LFM8 MoE non batché : même hash sur les trois tours ; prefill
  **73,29 → 75,78 tok/s (+3,4 %)**, TTFT **2,7427 → 2,6526 s (-3,3 %)** et
  decode +2,8 %, tous traités comme bruit/conditions. Pic inchangé pour les
  trois modèles (0,793 / 1,764 / 3,942 GB). Preuves
  `build/rust-perf-16-lfm{12,26,8}-{baseline,candidate}.json`.
- Premier contrôle long interrompu avant tout chargement : la variable locale
  `path` a écrasé le tableau spécial `$path` de zsh, rendant `python3`
  introuvable. Aucun modèle, GPU ou résultat exécuté. Renommer cette variable
  en `model_dir`, conserver strictement le prompt et relancer.
- Contrôle long validé face au binaire baseline sauvegardé avant PERF-16 :
  LFM1.2, 633 tokens donc trois chunks (256/256/121), **103,37 → 2 246,77
  tok/s**, TTFT **6,1241 → 0,2820 s**, hash greedy identique ; LFM2.6, 586
  tokens (256/256/74), **50,36 → 974,53 tok/s**, TTFT **11,6366 →
  0,6016 s**, hash identique. Preuves
  `build/rust-perf-16-lfm{12,26}-long-{baseline,candidate}.json`.
- Contrôles finaux : `cargo fmt --check`, `clippy --all-targets -D warnings`,
  suite Rust **33 passés / 5 ignorés / 0 échec**, matrice MLX **190/190**
  bit-à-bit, `git diff --check`. Décision : **validé** pour LFM2 dense ; rejeté
  explicitement pour LFM2-MoE faute de parité batch stricte. Le prefill/TTFT
  dense bénéficie du QMM multi-token ; decode, sampling et poids sont inchangés.
- État d'intégration : **code local uniquement** sur `codex/rust-performance` ;
  CLI/app installée et GitHub inchangés, aucun push ni rebuild GUI demandé.

### OPT-2026-09-13-RUST-PERF-17 — Prefill multi-token des architectures encore sérielles — validé Gemma

- Hypothèse : Gemma4 et Ling passent encore les prompts token par token dans le
  bridge. Au moins un de leurs chemins peut probablement réutiliser le QMM
  multi-row, l'attention causale et les primitives récurrentes déjà présentes,
  comme Qwen et LFM dense, sans nouveau format ni perte numérique.
- Antécédents : journal complet relu ; PERF-12/13 ont validé QMM/Qwen et PERF-16
  LFM dense, tandis que le batch LFM-MoE a été rejeté faute de parité de chaîne.
  Aucun essai Gemma/Ling multi-token n'est documenté. Chercher d'abord les
  contrats shape/state dans les deux runtimes et leur référence Python.
- Baseline/candidat prévu : binaire release sauvegardé avant PERF-16 contre
  branche locale courante ; premier checkpoint local compatible trouvé, prompt
  fixe >256 tokens, greedy, un warmup et trois répétitions si le coût le permet.
  Contrôle obligatoire avant mesure : logits et tous états bit-à-bit face à la
  production Python ; abandon immédiat de toute architecture non exacte.
- Environnement : Apple M5 24 Gio, MLX 0.32.2, batterie débranchée ; température
  et fréquences non instrumentées. État : **en cours**, recherche de chemin
  seulement ; aucune modification PERF-17 ni mesure à ce stade.
- Checkpoints : Ling EXL3 complet absent (sources/plans seulement) ; Gemma4
  26B-A4B EXL3 3,54 bpw disponible dans le cache de validation, 15,1 GB
  résidents. Baseline Gemma, prompt fixe 82 tokens, 16 tokens greedy, un warmup
  exclu puis trois tours : **39,84 tok/s prefill**, **2,0587 s TTFT**,
  **32,85 tok/s decode**, hashes stables, chargement 4,748 s. Preuve
  `build/rust-perf-17-gemma-baseline.json`.
- Lecture du chemin : les projections, le routeur standard et le SwitchGLU
  acceptent déjà plusieurs lignes ; les seules limites artificielles sont les
  reshapes `time=1`, le masque SDPA, et le head final. Ling exige en revanche
  un scan KDA récurrent multi-token absent : ne pas le refactorer dans cet essai.
- Prototype Gemma limité à la fenêtre glissante (batch seulement tant que la
  fin du bloc reste ≤1 024) : bloc de 32 tokens accepté par la référence de
  production, même top-1, erreur absolue max ≤0,5 et KL ≤0,005 selon le contrat
  Gemma existant. Le modèle a déjà une tolérance non bit-à-bit à cause du
  groupement QKV ; aucun seuil n'a été relâché.
- Benchmark candidat Gemma, mêmes 82 tokens, trois tours : **39,84 → 142,36
  tok/s prefill (+257,4 %)**, TTFT **2,0587 → 0,5763 s (-72,0 %)**, decode
  **32,85 → 32,66 tok/s (-0,6 %, bruit)**, pic inchangé 15,108 GB et les trois
  hashes greedy sont identiques à la baseline. Preuve
  `build/rust-perf-17-gemma-{baseline,candidate}.json`.
- Contrôle long Gemma, 565 tokens (256/256/53), un tour : **35,59 → 126,51
  tok/s prefill (+255,5 %)**, TTFT **15,8748 → 4,4663 s (-71,9 %)**, hash
  greedy identique et pic inchangé. Preuves
  `build/rust-perf-17-gemma-long-{baseline,candidate}.json`. Le batch est
  volontairement plafonné à 1 024 tokens ; au-delà, le fallback sériel garde
  la sémantique exacte de sliding attention sans ajouter un masque dédié.
- Contrôle de frontière sliding, 1 117 tokens : les quatre premiers blocs sont
  batchés puis les 93 derniers tokens repassent en sériel ; **32,39 → 68,02
  tok/s prefill (+110,0 %)**, TTFT **34,4830 → 16,4225 s (-52,4 %)**, hash
  greedy identique, pic inchangé. Preuves
  `build/rust-perf-17-gemma-window-{baseline,candidate}.json`.
- Contrôles finaux : parité Gemma batch 32 puis decode sériel 8 étapes contre
  production (même top-1, max abs ≤0,5, KL ≤0,005), `cargo fmt --check`, clippy
  strict, suite Rust **33 passés / 5 ignorés / 0 échec**, matrice MLX **190/190**
  bit-à-bit et `git diff --check`. Décision : **validé pour Gemma4**. Ling reste
  inchangé et non mesuré : aucun checkpoint EXL3 complet et son KDA exige un
  scan causal dédié, donc aucun refactor spéculatif n'a été ajouté.
- État d'intégration : **code local uniquement** sur `codex/rust-performance` ;
  app installée et GitHub inchangés, aucun push/release demandé.

### REL-2026-09-13 — Desktop v1.0.2 build 13 — validé et publié

- Demande utilisateur : embarquer le moteur Rust optimisé courant dans la GUI,
  pousser la nouvelle version, publier la release GitHub v1.0.2 et remplacer
  l'application locale. Signature ad-hoc maintenue comme v1.0.1 ; aucun
  certificat Developer ID/notarisation disponible.
- Source prévue : branche `codex/rust-performance`, commits PERF-01 à LOAD-10,
  PERF-16 LFM dense et PERF-17 Gemma validés. Versions Python/Info.plist/README
  synchronisées sur 1.0.2, build 13. Le DMG doit embarquer le binaire Rust et
  MLX 0.32.2, sans poids de modèle.
- Validation prévue : contrôles Rust/MLX déjà verts, E2E protocole/Desktop,
  build SDK26.5, manifeste sans changements suivis, signature stricte, montage
  DMG, self-checks GUI et smoke réel via le runtime monté. Conserver l'app
  installée précédente avant remplacement ; vérifier commit/tag/release et
  empreintes locales/distantes. État : **en cours**, non publié/non installé.
- Premier E2E local : tests Python **521 passés / 6 ignorés**, puis build Swift
  interrompu par le SDK27 actif sans plugin `SwiftUIMacros.StateMacro`, défaut
  de toolchain déjà documenté lors de v1.0.1. Aucun test Desktop n'a été exécuté
  après cet échec et aucun artefact publié. Relancer le même E2E avec
  `MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`,
  présent sur ce Mac ; aucune modification UI de contournement.
- Reprise SDK26.5 réussie : build Swift, checks hardening/timeline/MCP/mémoire
  Metal, callback bridge, rendu Markdown/code Unicode et transport CLI tous
  passés. Les avertissements SwiftMath dépréciés et chemins framework CLT déjà
  connus restent non bloquants. E2E source complet : **521 Python passés / 6
  ignorés**, puis tous les contrôles Desktop passés. Publication toujours non
  démarrée ; prochaine étape commit propre puis build du paquet final.
- Le premier mini-check de cohérence de version a été interrompu avant lecture
  des fichiers : le `python3` système ne fournit pas `tomllib`. `plutil` est
  néanmoins passé. Aucun artefact/commit affecté ; relancer le même assert avec
  la venv Python 3.12 déjà utilisée par la suite de tests.
- Reprise Python 3.12 réussie : Info.plist, `pyproject.toml` et
  `src/mlxl3/__init__.py` annoncent tous **1.0.2**, build **13** ; plist valide
  et `git diff --check` propre.
- Source figée au commit `55eb7461a925dafda5401938efc3dedb30bcf128`.
  Le build SDK26.5 est propre, le manifeste confirme le moteur Rust avec MLX
  0.32.2 et l'absence de Python embarqué. `codesign --verify --deep --strict`
  passe ; signature ad-hoc, sans notarisation, conformément à la limite acceptée.
- Artefact validé : `dist/MLXL3-Desktop-v1.0.2-b13-Apple-Silicon.dmg`,
  71 181 400 octets, SHA-256
  `5f1b87bb5d9e7ef6af53161a226f83e2ac0c10077170967cac8d610373e4339d`.
  Le runtime monté correspond au runtime source, SHA-256
  `de4299204bc7e844e95c812ca65e294ebf57bc23bb6284835f3ab868fd406c7c`.
- Validation depuis le DMG monté : signature, timeline, Metal et MCP passent ;
  deux tours réels LFM2 passent avec continuité CEDAR-42. Le second tour chaud
  mesure 2 066,83 tok/s prefill, 125,14 tok/s decode et 148,8 ms TTFT. Preuves
  `build/release-v102-build.log` et `build/release-v102-mounted-smoke.log`.
- Publication réussie : `main`, `codex/rust-performance` et le tag annoté
  `v1.0.2` pointent sur la source publiée ; release GitHub
  `https://github.com/0xZKnw/mlxl3/releases/tag/v1.0.2`, déclarée latest, avec
  DMG, checksum, manifeste et rapport de validation. L'empreinte de l'asset
  GitHub est identique à l'empreinte locale.
- CI finale : les trois exécutions `Native Rust checks` et les trois
  `Regression checks` déclenchées par branche, `main` et tag sont **réussies**.
  Une première requête de statut a utilisé le champ `isLatest`, absent de cette
  version de `gh` ; la vérification a été reprise via l'API `releases/latest`.
- Installation locale validée dans `/Applications/MLXL3 Desktop.app`, version
  **1.0.2 (13)**, même runtime et mêmes self-checks. L'ancienne 1.0.1 reste
  récupérable sous
  `build/app-backups/MLXL3 Desktop-v1.0.1-before-v1.0.2.app`. Application laissée
  fermée. État final : **validé, publié et installé**.

### CONV-2026-09-15-LING3-TINY-4BPW — reprise locale — validé localement

- Demande : produire localement Ling 3.0 Tiny en EXL3 uniforme 4 bpw, sans QAT.
  Source déjà complète : `models/source/Ling-3.0-tiny-HF`, révision documentée
  `e3a47d5b986e7141b6efd62597d598ebb392060d`; sortie
  `models/Ling-3.0-tiny-EXL3-4bpw`, travail `build/ling-4bpw`.
- Reprise vérifiée avant lancement : 1 600/9 031 projections mesurées dans 50
  checkpoints, 7 431 restantes. Les activations de calibration avaient été
  nettoyées (18 fichiers seulement, aucun `complete.json`) et seront donc
  régénérées ; les mesures existantes restent reprises par groupe exact de 32.
- Protocole : recette documentée `scripts/quantize_ling.py`, K=4/MCG, 2 048
  lignes réelles, deux séquences de 1 024 tokens, deux workers, backend Metal.
  Après conversion : inventaire strict, chargement EXL3, perplexité tenue à
  part contre BF16 sur les mêmes tokens, puis génération CLI. Aucun push HF
  n'est demandé à ce stade.
- ETA avant reprise : calibration ~15 min observées ; mesure restante ~108 min
  d'après la médiane des 20 derniers groupes (0,871 s/projection), puis émission,
  finalisation et validation estimées 45–90 min selon la réutilisation du cache
  quantifié 6 Gio. Total annoncé **2 h 45 à 3 h 30**. Apple M5 24 Gio, environ
  580 Gio libres ; `pmset` annonce alimentation secteur, batterie 83 % avec le
  sous-état « discharging ». État : **en cours**, chiffres à remplacer par les
  temps réels ; modèle final absent au lancement.
- Premier lancement bloqué avant import MLX par le sandbox sans GPU ; aucun
  calcul. Reprise hors sandbox autorisée, puis arrêt manuel à la demande après
  6/24 couches de calibration pour chercher un chemin plus court. Aucun poids
  final émis ; les checkpoints existants sont conservés.
- Reprise du 16 septembre, **en cours avant lancement** : recette identique
  ci-dessus avec bypass uniforme OPT-01, capture OPT-11 et scale expert 0,908
  OPT-12, sur la révision source épinglée. Baseline de qualité BF16 déjà
  mesurée : WikiText-2 test, 2 048 tokens, fenêtres 256, PPL 21,05421
  (`build/ling-4bpw/source-perplexity.json`). Mesurer le temps de conversion
  complet, puis inventaire strict, PPL EXL3 sur les mêmes données/fenêtres,
  chargement et génération CLI. Estimation pilote ~55–60 min, **non mesurée**
  sur modèle entier. M5/24 Gio, macOS local, ~557 Gio libres et secteur annoncé
  (`pmset` indique toutefois « discharging » à 83 %) ; état thermique inconnu.
  Commande : `.venv/bin/python scripts/quantize_ling.py --in-dir
  models/source/Ling-3.0-tiny-HF --out-dir
  models/Ling-3.0-tiny-EXL3-4bpw --work-dir build/ling-4bpw --bits 4
  --head-bits 4 --calibration-rows 2048 --calibration-seq-len 1024
  --max-workers 2 --search-backend metal`. Aucune publication demandée.
- Résultat du 16 septembre : conversion complète **9 031/9 031 modules EXL3
  K4/MCG**, aucun module omis, un shard HF standard, inventaire strict et
  validation du checkpoint réussis. Modèle final :
  `models/Ling-3.0-tiny-EXL3-4bpw` (4,1 Gio sur disque, ~4,4 Go décimaux) ;
  travail/reprise : `build/ling-4bpw`. Durée murale observée **environ 92 min**
  de lancement à la fin, contre 55–60 min estimées du pilote. Le Mac était
  initialement en décharge malgré « AC Power » (83 → 35 %), puis en charge ;
  état thermique non mesuré. Ne pas présenter la durée comme représentative
  d'un Mac stable sur secteur ou additionner des gains pilotes.
- Qualité tenue à part, mêmes 2 048 tokens WikiText-2 test, SHA256 de corpus
  `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`,
  fenêtres 256 et chunks d'exécution 128 : PPL BF16 **21,05421** → EXL3
  **21,66936** (+2,92 %). Preuves : `build/ling-4bpw/source-perplexity.json`,
  `build/ling-4bpw/exl3-perplexity.json`, `pipeline_summary.json` et
  `pipeline_state.json`. KL et capacités larges non mesurés ; cette PPL ne
  permet pas d'isoler l'effet du scale fixe OPT-12 par rapport à une autre
  conversion K4.
- Chargement CLI strict réussi (9 031 modules, 4,38 GB résidents dans ce run)
  et génération de la réponse finale correcte « 7 + 5 égale 12. » ;
  87,8 tok/s decode sur **un seul** prompt court, pas un benchmark comparatif.
  Les limites 96/256 tokens tronquaient la réflexion avant la réponse ; le
  test à 1 024 tokens s'est terminé naturellement après 424 tokens. Une
  alerte `transformers` sur `bailing_hybrid` apparaît au chargement du
  tokenizer mais n'empêche ni PPL ni génération. Enregistré dans le registre
  local comme `ling3.0-tiny-4bpw` et visible dans `mlxl3 list` ; GUI non testé.
  GitHub/HF inchangés. Quatre tests ciblés adaptateur/PPL réussis après run.

### OPT-2026-09-15-QUANT-01 — plan uniforme sans mesure redondante — en cours

- Hypothèse : avec `candidate_bits=[4]`, tête K=4 et unique shrinkage 0, la
  phase `measure_ldlq_candidates` ne peut choisir aucun autre plan. Elle
  quantifie pourtant 9 031 projections avant `convert_module_set`. Construire
  directement le plan K=4 conserve exactement recette, activations, Hessienne,
  LDLQ, codebook et poids, tout en laissant la passe d'émission quantifier une
  seule fois et utiliser le groupement Metal gate/up existant.
- Antécédents : optimisations Metal K=4 et préparation déjà intégrées ; ne pas
  les retester. Aucun bypass uniforme trouvé dans le converter. La mesure Ling
  interrompue avait 1 600 projections, médiane récente 0,871 s/projection ; son
  score ne sert pas à une décision avec un seul candidat.
- Baseline/candidat prévus : pipeline actuel contre détection automatique du
  cas strictement uniforme entier, sur les 18 mêmes premières projections,
  nouvelles sorties/work dirs, M5 24 Gio, secteur annoncé, backend Metal, deux
  workers, mêmes 2 048 lignes. Contrôler K=4, tenseurs EXL3 valides et comparer
  les empreintes d'une projection entre chemin mesuré et chemin direct. Si ce
  garde qualité échoue, retirer. ETA complète révisée seulement après le pilote.
- Une optimisation indépendante de stockage sera évaluée séparément : les
  gate/up des 128 experts reçoivent exactement le même tableau d'activation ;
  des hardlinks peuvent éviter les copies disque sans changer un octet lu. Ne
  pas cumuler son gain avec le bypass avant mesure séparée.
- Premier pilote direct, 18 projections (`build/ling-uniform-direct`,
  `models/Ling-3.0-tiny-pilot-direct`) : plan `fixed_uniform`, **0 candidat
  mesuré**, K=4 sur 18/18, calibration 12 s et pipeline 55 s d'après les mtimes.
  Le pipeline a bien émis les 18 shards ; le processus a ensuite échoué au
  validateur parce que le pilote demandait volontairement `--no-finalize`
  (shards Pony incrémentaux, pas de shards HF standard). Ce n'est pas un crash
  de quantification, mais le protocole sera relancé avec finalisation pour le
  garde bout-en-bout. Tests du garde : **2 réussis** avec Metal ; compilation
  et `diff --check` réussis.
- Le triplet expert observé prend ~2,78 s (down 0,94 s, gate/up groupés 1,84 s),
  mesuré par mtimes des shards. Extrapoler naïvement 2 944 experts donnerait
  ~136 min : le bypass retire la mesure redondante mais ne suffit pas seul pour
  viser <1 h. L'ancienne estimation d'émission 45–90 min est donc invalidée par
  ce pilote et ne doit plus être citée.
- Hardlinks gate/up implémentés localement, pas encore mesurés : un seul inode
  par couche remplace jusqu'à 256 copies strictement identiques. Les 6 couches
  interrompues occupent actuellement 10 Gio/2 200 fichiers et prouvent que le
  coût disque est matériel. Validation prévue : test d'identité octet/inode et
  nouveau pilote ; aucun gain temps n'est encore annoncé.
- Contrôle hardlinks terminé : **2 tests réussis** (`test_ling_conversion_adapter`),
  valeurs gate/up strictement égales et inode unique vérifié. Statut : validé
  fonctionnellement, gain disque théorique/temps complet encore non mesuré.

### OPT-2026-09-15-QUANT-02 — batch multi-experts Ling — en cours

- Hypothèse : le coût dominant est désormais l'enchaînement de 2 944 minuscules
  triplets experts, pas le calcul utile. `ldlq_quantize_group` sait déjà préparer
  et chercher plusieurs matrices Metal ; évaluer des lots de plusieurs experts
  de même shape/activation doit amortir dispatchs, synchronisations et lectures
  de Hessienne sans changer K, codebook, ordre LDLQ ni sorties par module.
- Baseline : expert 0 couche 1, K4, 2 048 lignes : 2,78 s pour down+gate+up sur
  le pilote direct ci-dessus. Protocole prévu : lots 1/2/4 experts sur la même
  couche et mêmes activations, empreintes exactes des tenseurs EXL3 contre lot 1,
  mémoire processus surveillée ; conserver uniquement un lot strictement égal
  et plus rapide. Aucun cumul/ETA <1 h avant ce contrôle.
- Premier lot 4 experts (`build/ling-expert-batched`, limite 33 modules) : 10,20 s
  pour down(4)+gate/up(8), soit **2,55 s/expert contre 2,78 s**, gain ~8,3 %.
  Le reliquat de 2 experts donne 2,61 s/expert. Les tenseurs EXL3 de l'expert 0
  sont **bit-identiques** au chemin lot 1 pour down/gate/up. Calibration 12 s,
  émission+écriture 56 s, total 68 s ; fin attendue en erreur uniquement parce
  que `--no-finalize` n'est pas accepté par le validateur MLXL3 post-pilote.
- Hardlinks sur ce pilote : 199 Mo de tailles logiques mais **124 Mo occupés** ;
  calibration toujours 12 s ici car seulement six experts. Gain stockage validé,
  gain temps à mesurer à l'échelle complète. Le batching seul est conservable
  mais ne suffit pas à <1 h ; prochaine piste : paralléliser la préparation/GSS
  actuellement forcée à un seul worker pour `scale_mode=computed`, avec même
  comparaison bit-à-bit et retour arrière si instable/régressif.
- Variante préparation 4 workers : pilote identique limite 33 dans
  `build/ling-expert-parallel`. Protocole complémentaire prévu après résultat :
  profiler un expert isolé avec `PONYEXL3_CONVERT_TIMING=1` pour attribuer le
  temps entre basis/GSS, Hessienne, LDL et boucle LDLQ avant tout nouveau kernel.
- Résultat 4 workers : 14,84 s pour six experts contre 15,42 s, soit **+3,8 %**
  sur le segment expert et 67,18 s contre 67,97 s bout-en-bout (**+1,2 %**).
  Les 18 projections expert (six triplets) sont bit-identiques. Deux tests LDLQ
  groupés réussis. Gain réel mais trop faible ; le profilage isolé décidera si
  cette concurrence reste intégrée ou si elle doit être retirée.
- Profil isolé K4 MCG : down 0,90 s (basis/GSS 0,48, LDLQ 0,38), gate 0,97 s
  (basis/GSS 0,43, LDLQ 0,50) ; GPU actif 84–85 %, 45/109 appels. Les deux
  moitiés sont donc matérielles. Prochain microbenchmark strict : 256 tiles K4
  MCG, scratch 256 contre 512 MiB, 7 paires alternées, même entrée et parité
  exacte. Motif nouveau malgré l'essai historique K6 négatif : les lots Ling
  K4 font exactement 256 tiles, donc 512 MiB peut supprimer un dispatch sur
  deux ; K6/shape lm_head ne validait pas cette géométrie.
- Résultat 256 tiles, 7 paires : médiane 256 MiB 24,793 ms, 512 MiB
  22,875 ms, soit **+8,38 %** sur le search microbenchmark ; sorties décodées
  et états bit-identiques. Variante complémentaire prévue sur 384 tiles
  (géométrie du lot down) à 256/512/768 MiB avant choix du budget ; aucun gain
  projection/complet encore revendiqué.
- Résultat 384 tiles, 6 paires : 256/512/768 MiB = 41,084/35,801/34,905 ms ;
  **+15,0 %** à 768 MiB contre 256, parité exacte. Décision de test : conserver
  256 MiB pour les modules seuls, autoriser 768 MiB uniquement dans le chemin
  groupé multi-experts, puis relancer le pilote six experts. Le surcoût actif
  maximal attendu est +512 MiB pendant le search groupé ; mesurer projection
  complète avant intégration définitive.
- Projection complète scratch 768 MiB : 14,709 s pour six experts contre
  14,839 s, **+0,88 %** seulement, malgré parité bit-à-bit et deux tests Metal
  réussis. Rejeté : +512 MiB actif ne vaut pas ce gain ; revenir au budget
  256 MiB. Nouvelle piste exacte : vectoriser le GSS des membres d'un groupe.
  Les 13 évaluations restent identiques par module, mais chaque ronde concatène
  les échantillons en un appel Metal et re-sépare les MSE. Garde prévu : GSS
  multi vs scalaire sur fonctions déterministes, puis 18 projections expert
  bit-identiques contre `ling-expert-parallel` et timing du même pilote.
- GSS vectorisé : 4 tests unitaires/Metal réussis et les 18 projections expert
  sont bit-identiques, mais six experts prennent **15,459 s contre 14,839 s**
  (−4,18 %) et le pilote 69,87 s contre 67,18 s. Rejeté et à retirer : la
  concaténation agrandit les batches sans réduire assez le calcul par tile.
  Ne pas cumuler ce résultat avec le batching multi-experts positif.

### OPT-2026-09-15-QUANT-03 — récursion LDLQ Metal réellement batchée — en cours

- Hypothèse : le lot actuel concatène la recherche treillis, mais exécute encore
  compensation, mise à jour et `matmul` LDLQ dans une boucle Python par expert.
  Le chemin officiel ExLlamaV3 `ldlq_batched` empile au contraire poids et
  facteurs L de tenseurs de même forme et remplace ces produits par des BMM.
  Porter uniquement ce chemin homogène vers `mx.matmul` batched doit réduire les
  dispatchs sans changer l'algorithme, K, codebook, ordre de feedback ni qualité.
- Baseline : `build/ling-expert-parallel`, six experts complets, **14,839 s**
  entre premier et dernier shard expert ; pipeline pilote 67,18 s. Candidat :
  mêmes 33 modules, activations, K4/MCG, 2 048 lignes, M5 24 Gio, backend Metal.
  Contrôles requis : les 18 treillis/suh/svh experts bit-identiques, tests groupés,
  mémoire MLX et temps du segment. Rejet immédiat si divergence ou régression.
  Ce test ne promet pas encore <1 h ; l'ETA ne sera révisée qu'après mesure.
- Résultat batch LDLQ empilé, six experts : **12,582 s** contre 14,839 s,
  soit **−15,2 %** sur le segment (9,828 s entre premier et dernier shard).
  Les 72 tenseurs expert comparés sont bit-identiques ; trois tests ciblés
  Metal/driver réussissent. L'extrapolation des 2 944 experts reste ~103 min,
  donc le chemin est validé mais ne suffit pas à l'objectif <1 h.
- Variante suivante, avant essai : augmenter le lot homogène de 4 à 8 experts
  (8 down, 16 gate/up) sur les 16 premiers experts de la même couche. Baseline
  conservée ci-dessus ; contrôler la parité des six experts communs, le temps
  par expert et l'absence d'explosion mémoire. Revenir aux petits lots si la
  latence moyenne ne baisse pas.
- Résultat lot 8 sur 16 experts : **33,956 s**, soit 2,122 s/expert contre
  2,097 s/expert avec les petits lots (régression ~1,2 %). Les 72 tenseurs
  communs restent bit-identiques. Variante rejetée ; limites revenues à 4 down
  et 8 gate/up. Prochaine hypothèse : partager le calcul de Hessienne brute entre
  gate/up qui lisent exactement les mêmes activations, puis appliquer la
  transformation de signes propre à chaque `suh` sans refaire `X.T @ X`.
- Variante ordonnancement, avant essai : le lot gate/up contient huit matrices
  mais la préparation est plafonnée à quatre workers. Tester huit workers sur
  le même pilote six experts et le même batch LDLQ empilé ; conserver seulement
  si le segment bat 12,582 s avec 72 tenseurs identiques. Ce contrôle rapide
  précède la refonte Hessienne, plus invasive.
- Résultat huit workers : **14,696 s** contre 12,582 s (régression 16,8 %),
  malgré 72 tenseurs bit-identiques. Rejeté et plafond remis à quatre : les
  threads supplémentaires saturent la même file Metal au lieu de la remplir.

### OPT-2026-09-15-QUANT-04 — recherche g-scale groupée en deux passes — en cours

- Hypothèse : la préparation est dominée par la recherche dorée séquentielle,
  13 recherches treillis et synchronisations par projection. Le quantificateur
  ExLlamaV3 actuel remplace cela, pour les groupes homogènes, par une grille
  grossière sous-échantillonnée puis une grille fine complète : deux lots Metal
  pour tout le groupe. Porter ce schéma uniquement aux experts doit supprimer
  la majorité des barrières sans QAT ni réduction de calibration.
- Baseline : batch LDLQ empilé, six experts, **12,582 s**. Protocole candidat :
  mêmes 33 modules/activations/K4, quatre experts par lot, grille amont 10+5,
  comparer les métriques `inner_mse` module par module et les tenseurs/scales.
  Puis, seulement si l'erreur agrégée n'augmente pas matériellement, estimer le
  modèle complet. Ce chemin n'est pas censé rester bit-identique car il choisit
  le minimum sur une grille globale plutôt qu'un minimum local de la recherche
  dorée ; retour arrière si le gain qualité/temps n'est pas simultanément établi.
- Résultat grille groupée : **11,862 s**, soit −5,7 % contre le batch LDLQ seul
  et −20,1 % contre la baseline 14,839 s. `inner_rel_rms` moyen passe de
  0,0883233 à 0,0883052 ; 8/18 projections s'améliorent, pire variation
  individuelle +0,103 %. Statut provisoire : gain réel et métrique agrégée non
  dégradée, mais tenseurs différents ; perplexité complète requise avant validation.
- Variante suivante, avant essai : avec seulement deux barrières g-scale par
  groupe, retester 8 experts (8 down, 16 gate/up) sur 16 experts. Le précédent
  essai à gros lots utilisait encore la recherche dorée par module et ne répond
  donc pas à cette nouvelle hypothèse. Comparer temps/expert et métriques aux
  petits lots ; retour à 4/8 si la moyenne ne baisse pas.
- Résultat gros lots avec grille : 16 experts en **31,857 s**, soit 1,991 s/expert
  contre 1,977 s/expert en petits lots (régression 0,7 %). Rejeté ; limites
  revenues à 4/8. La métrique moyenne sur ces 16 experts reste du même ordre
  (0,0883553), mais les populations diffèrent donc elle ne sert pas de gain.

### OPT-2026-09-15-QUANT-05 — attribution du temps groupé — en cours

- Avant nouvel essai d'optimisation, mesurer séparément préparation (poids,
  régularisation, Hessienne/LDL), g-scale groupé, allocation et récursion LDLQ
  sur le pilote six experts désormais à 11,862 s. Ajouter uniquement quatre
  compteurs de durée aux stats existantes, sans changer le calcul, puis retirer
  ou conserver ces compteurs selon leur utilité. Cette mesure choisira le prochain
  kernel ; aucune estimation <1 h ne sera faite sans identifier le poste dominant.
- Résultat six experts, quatre groupes : préparation cumulée 0,312 s, g-scale
  **4,983 s**, allocation 0,012 s, récursion LDLQ **6,513 s**. Les deux postes
  Metal expliquent presque tout le segment 11,86 s ; la piste Hessienne/CPU est
  abandonnée faute de plafond utile. Les compteurs restent locaux pour vérifier
  les prochains essais et seront retirés si non nécessaires à la fin.

### OPT-2026-09-15-QUANT-06 — g-scale sans packing jeté — en cours

- Hypothèse : la grille g-scale appelle le chemin de conversion complet, qui
  transforme les tiles, exécute la recherche, compacte les états en treillis puis
  inverse la permutation ; seul le MSE est utilisé. Appeler directement le kernel
  de recherche sur les tiles déjà permutées supprime packing et aller-retour sans
  changer le score (la permutation préserve exactement la somme des carrés).
- Baseline : g-scale cumulé 4,983 s, segment 11,862 s. Protocole : même pilote,
  vérifier mêmes `regularize_g_scale`, mêmes 72 tenseurs expert et temps. Rejet
  si un seul tenseur diffère ou si le g-scale ne baisse pas au-delà du bruit.
- Résultat : g-scale **5,553 s** contre 4,983 s (+11,4 %), segment 12,513 s
  contre 11,982 s sur la répétition instrumentée, et 6/72 tenseurs diffèrent
  à cause de l'ordre numérique du scale/permutation. Rejeté ; chemin direct
  précédent restauré. Ce résultat ne doit pas être cumulé avec les gains validés.

### OPT-2026-09-16-QUANT-07 — supprimer les barrières LDLQ par feedback — validé en pilote

- Hypothèse : le chemin empilé appelle `mx.eval` après chaque tranche de 16 rangs,
  soit jusqu'à 96 barrières CPU/GPU pour un groupe gate/up de 1 536 rangs. Cette
  barrière protège les très grandes matrices d'un graphe différé géant, mais les
  experts empilés ne gardent que huit étapes dans le bloc borné de 128 rangs.
  Évaluer une fois par bloc, comme la récursion device-side amont, doit supprimer
  7/8 des synchronisations sans changer une opération ni son ordre de dépendance.
- Baseline instrumentée : LDLQ cumulé **6,513 s**, segment 11,982 s sur la
  répétition. Protocole : déplacer uniquement `mx.eval(packed,b_reconstructed)`
  hors de la boucle feedback du chemin homogène ; mêmes 72 tenseurs bit-identiques,
  mêmes scales et mémoire sous le seuil existant. Rejet à la moindre divergence.
- Résultat : segment **10,988 s** contre 11,982 s (−8,3 %), LDLQ cumulé
  5,940 s contre 6,513 s (−8,8 %), et **72/72 tenseurs bit-identiques**.
  Trois tests ciblés Metal/driver passent. Validé sur le pilote ; estimation
  experts complets ~90 min, donc l'objectif <1 h n'est pas encore atteint.

### OPT-2026-09-16-QUANT-08 — barrière LDLQ tous les quatre blocs — rejeté

- Hypothèse : après OPT-07, il reste une barrière par bloc de 128 rangs (4 pour
  down, 12 pour gate/up). Les matrices expert empilées sont petites ; différer
  quatre blocs conserve le même graphe/opérations et borne les temporaires à
  quelques centaines de Mo, tout en divisant encore les synchronisations par 4.
- Baseline : LDLQ 5,940 s, segment 10,988 s. Protocole identique, 72 tenseurs
  exacts requis ; surveiller cache Metal et revenir à un bloc si mémoire ou
  temps régressent. Aucun changement de `buf_size_rows` ni d'association matmul.
- Résultat : **11,521 s** contre 10,988 s (+4,9 %), avec tenseurs exacts.
  LDLQ 6,055 s contre 5,940 s et g-scale 4,923 s contre 4,575 s ; différer
  davantage agrandit le graphe MLX sans accélérer le kernel. Rejeté, retour à
  une barrière par bloc de 128 rangs.

### OPT-2026-09-16-QUANT-09 — arithmétique K4 alignée sur CUDA — rejeté

- Hypothèse : après les gains d'orchestration, g-scale (4,575 s) et LDLQ
  (5,940 s) passent presque tout leur temps dans le même treillis K4. Le kernel
  CUDA officiel charge la cible en FP16 et effectue différences/FMA/comparaisons
  en `half2`, tandis que le port Metal conserve ces calculs en `float2`. Une
  variante Metal K4/MCG strictement calée sur cette arithmétique amont peut
  exploiter le débit FP16 du M5 et viser les ~1,5x encore nécessaires.
- Antécédents : les variantes exactes de lookup, vecteurs, threadgroups,
  compactage et scratch du rapport Metal sont déjà épuisées et ne seront pas
  répétées. Le mode `fast math` avait divergé et reste exclu. Cette variante est
  nouvelle mais ne promet pas la bit-identité avec le chemin Metal FP32 ; elle
  doit d'abord égaler le comportement numérique du quantificateur CUDA officiel.
- Baseline : noyau local K4/MCG actuel, puis pilote Ling six experts à
  **10,988 s** (72 tenseurs exacts contre son propre chemin scalaire). Candidat :
  même lookup/codebook, tail-biting, tie-break et calibration, seuls target,
  erreur et coûts passent par l'arithmétique half2 de l'amont.
- Protocole : microbenchmark 256/384 tiles en paires alternées, puis le même
  pilote 33 modules. Comparer `inner_rel_rms` par projection et agrégé ; ne pas
  intégrer ni annoncer une absence de perte avant perplexité BF16/EXL3 sur les
  mêmes tokens. Rejet si le débit ne permet pas plausiblement <1 h ou si les
  métriques se dégradent matériellement. M5 24 Gio, MLX 0.32.2, conditions
  thermiques non contrôlées ; aucune publication.
- Résultat microbenchmark, sept paires alternées : 256 tiles
  **25,161 → 23,757 ms** (+5,9 %) ; 384 tiles **38,039 → 37,146 ms** (+2,4 %).
  Le MSE synthétique est quasi inchangé/légèrement meilleur, mais seulement
  93,8 % des états correspondent au chemin FP32. Le plafond de 2–6 % sur le
  search ne peut pas faire passer l'estimation complète de ~90 à <60 min et
  imposerait tout de même une validation perplexité complète. Variante rejetée
  avant le pilote modèle ; aucun poids produit, chemin FP32 restauré.

### OPT-2026-09-16-QUANT-10 — scale représentatif par lot expert — remplacé

- Hypothèse : les 18 experts du pilote choisissent des g-scales très proches
  (0,8981–0,9148, moyenne 0,9070, écart-type 0,0048). Chercher 15 candidats sur
  le premier membre de chaque lot homogène puis réutiliser son scale pour les
  3/7 autres membres réduirait de 75–87,5 % le poste g-scale, actuellement 42 %
  du segment. Le treillis LDLQ final, Hessienne et calibration restent complets.
- Antécédents : ce n'est ni `skip_g_scale=1` (scale fixe 1,0), ni la grille
  groupée OPT-04 qui cherche encore 15 candidats par projection. Aucun essai de
  scale représentatif trouvé. Les scales observés justifient un pilote, pas une
  conclusion de qualité.
- Baseline/candidat : mêmes six experts et 33 modules que OPT-07, baseline
  **10,988 s**, `inner_rel_rms` moyen expert à recalculer depuis le résumé.
  Candidat expérimental activé uniquement par environnement ; premier scale du
  lot appliqué aux autres bases, diagnostics de recherche marqués non mesurés.
- Protocole : mesurer segment/g-scale/LDLQ et comparer `inner_rel_rms` des 18
  projections à OPT-07. Si la moyenne/pire dérivent matériellement, rejeter. Si
  le temps rend <1 h plausible, conserver seulement après perplexité complète
  BF16/EXL3 identique en corpus/tokens. M5 24 Gio, aucune publication.
- Résultat pilote : sommes des quatre groupes préparation/g-scale/LDLQ
  **0,304/4,575/5,940 → 0,269/0,944/5,363 s**, soit **10,83 → 6,58 s
  (−39,2 %)** sur ces postes. La moyenne `inner_rel_rms` passe de 0,08830522 à
  0,08830839 (**+0,0036 % relatif**) ; pire projection +0,224 %, meilleure
  −0,127 %. Les écritures des 18 shards couvrent 8,58 → 5,26 s, cohérent mais
  ne couvrent pas le premier groupe. Estimation experts seuls : ~54 min.
- Statut : gain prometteur mais **qualité provisoire**, car les treillis changent
  et aucune perplexité complète n'existe encore. Le mode reste expérimental par
  variable d'environnement ; ne pas l'utiliser pour la conversion finale avant
  un garde sur davantage d'experts puis la perplexité du modèle complet.
- Intégration : prototype retiré au profit du scale pré-calibré OPT-12, plus
  rapide et meilleur sur la population étendue ; aucun chemin partagé actif.

### OPT-2026-09-16-QUANT-11 — calibration down experts batchée — validé localement

- Hypothèse : `capture_experts` lance séparément gate et up pour chacun des 128
  experts afin de fabriquer les 2 048 entrées de down, soit 256 petits matmuls
  par couche. Empiler huit experts et utiliser `mx.matmul` batched supprime la
  majorité des dispatchs, sans changer poids, lignes, SwiGLU ni fichiers.
- Antécédents : aucun essai de batching de la capture Ling trouvé. Les anciens
  hardlinks ne concernent que gate/up et n'accélèrent pas ces matmuls down.
- Baseline : six experts sélectionnés dans le pilote précédent et leurs `.npy`
  issus du chemin scalaire ; le run complet interrompu avait annoncé ~15 min de
  calibration pour les 128 experts/couche. Candidat : lots de huit, pic Metal
  surveillé, comparaison bit-à-bit après cast FP16 contre les fichiers baseline.
- Protocole : recapturer le même pilote 33 modules dans un dossier neuf, mesurer
  calibration, comparer chaque activation down, puis microbenchmark 128 experts
  d'une couche seulement si la parité tient. Rejeter si divergence matérielle ou
  mémoire excessive. Aucun poids/publication.
- Résultat : les six activations du pilote sont bit-identiques mais 12,85 s
  contre 11,07 s de capture totale, car six experts n'amortissent pas le batch.
  Sur le cas réel de **128 experts**, les 128 fichiers FP16 sont bit-identiques
  et leur plage d'écriture/calcul tombe de **1,125 à 0,205 s (5,48x)**. La
  capture candidate complète des 24 couches, avec une seule couche expert
  sélectionnée, termine en 13,9 s. Le batching par huit est retenu ; pic mémoire
  processus non mesuré séparément, aucune erreur d'allocation observée.

### OPT-2026-09-16-QUANT-12 — scale expert Ling pré-calibré — validé en pilote

- Hypothèse : les 1 552 recherches déjà terminées sur les couches 1–5 donnent
  un g-scale expert global très stable autour de 0,908 (écarts-types par
  couche/projection 0,010–0,013). Pour **ce checkpoint et cette recette**, fixer
  0,908 réutilise une calibration déjà payée au lieu de refaire 15 treillis par
  projection/lot. Cela retire les ~7,7 min de g-scale encore estimées après
  OPT-10 et place les experts vers 46 min.
- Antécédents : différent de `skip_g_scale` à 1,0 et du scale représentatif par
  lot. Les résultats historiques viennent du même source, K4/MCG, 2 048 lignes
  et sigma ; ils ne généralisent pas à un autre modèle/recipe et ne doivent pas
  devenir une valeur globale du moteur.
- Baseline/candidat : OPT-10 à 6,58 s de postes groupés et moyenne
  `inner_rel_rms` 0,08830839 ; candidat expérimental 0,908 sur les mêmes six
  experts, aucune recherche g-scale. Comparer les 18 erreurs et temps ; garder
  uniquement derrière l'adaptateur Ling exact (révision source/recette), puis
  exiger la perplexité complète avant publication.
- Protocole : nouveau pilote 33 modules, activation par environnement, mêmes
  poids/calibration/Metal. Rejet si l'erreur moyenne ou le pire module dérivent
  matériellement. Aucun modèle final ni publication.
- Résultat six experts : postes groupés **10,830 → 5,769 s (−46,7 %)** contre
  la baseline OPT-07 ; g-scale 4,575 → 0,003 s. `inner_rel_rms` moyen
  0,08830522 → 0,08831060 (**+0,0061 % relatif**), pire projection +0,158 %,
  plusieurs projections meilleures. Estimation experts seuls ~47,2 min.
- Extension déclarée avant essai : quantifier les 128 experts complets de la
  couche 1 (384 projections, mêmes 2 048 activations) et comparer leurs erreurs
  à la mesure historique per-projection disponible. Cette population couvre la
  plage de g-scales optimale 0,874–0,940 absente du petit pilote. Mesurer temps,
  pic et dispersion ; la perplexité modèle reste ensuite obligatoire.
- Résultat couche complète : **384/384 projections** produites en 127,25 s de
  plage d'écriture ; sommes des 64 groupes : préparation 5,71 s, g-scale
  0,06 s, états 0,26 s, LDLQ 117,47 s. Face aux 384 mesures historiques avec
  recherche individuelle, `inner_rel_rms` moyen **0,08835081 → 0,08833356
  (−0,0195 % relatif)** ; pire projection +0,332 %, p95 +0,169 %, meilleure
  −0,348 %. Le processus pilote complet (capture, 15 denses, 384 experts,
  copie/écriture) prend 177,1 s.
- Décision : retenir 0,908 uniquement dans l'adaptateur Ling K4/MCG avec cette
  recette ; estimation 23 couches experts ~48,8 min, modèle complet **~55–60
  min** sous conditions similaires. Qualité modèle encore provisoire jusqu'à
  perplexité BF16/EXL3 complète ; aucune publication ni modèle final à ce stade.
- Intégration : code local activé automatiquement par `quantize_ling.py` pour
  K4/MCG seulement ; autres modèles/bpw/codebooks inchangés. Batching calibration
  intégré au même adaptateur. **86 tests réussis, 1 ignoré** (fixture MiniCPM
  absente), compilation et `diff --check` réussis. App, GitHub et HF inchangés.
- Révision du 16 septembre : le checkpoint complet avec cette option s'est
  converti en ~92 min, PPL BF16 21,05421 → EXL3 21,66936 (mêmes tokens).
  Ce résultat valide l'utilisabilité de cette conversion mais ne démontre pas
  la non-infériorité du scale 0,908 contre une conversion complète avec recherche
  individuelle des scales. Le temps pilote 55–60 min était trop optimiste.
- Publication du 16 septembre : le code et le patch PonyExl3 sont sur `main`
  (`cc2049c`) ; les tests après rebase passent (112/112). Le checkpoint public
  est sur https://huggingface.co/0xzknw/Ling-3.0-tiny-EXL3-4bpw, commit
  `45a71f6ce370ca4cde4f1eff50583090ec8e31d8`, avec ses 12 fichiers
  et sa carte. Smoke natif Rust avec l'app installée : chargement Ling réussi,
  réponse finale `12` à « Combien font 7 + 5 ? Réponds directement. » ; 146
  tokens, TTFT 927 ms, decode 33,1 tok/s sur un seul essai non comparatif.
  Le GUI lui-même n'a pas été mis à jour ni testé pour ce modèle.

### OPT-2026-09-16-RUST-PERF-18 — Ling EXL3 : baseline decode/prefill — en cours

- Demande : optimiser Ling 3.0 Tiny EXL3 4 bpw localement, avec cibles 150 tok/s
  decode et 500 tok/s prefill sur chat court, sans spéculation, requantification
  ni changement de sampling. Aucun objectif n'est considéré acquis d'avance.
- Antécédents : PERF-01 a validé le retrait des barrières par couche sur Qwen ;
  PERF-11 a rejeté le QMV paresseux pour le prefill ; PERF-12/13 ont validé le
  QMM multi-token, PERF-16 a rejeté le batch LFM MoE faute de parité, PERF-17
  n'a pas testé Ling, encore absent. Le présent essai établit une baseline Ling
  reproductible avant le moindre changement de runtime/kernel.
- Environnement initial : `main` `a473c96`, Apple M5/macOS 27.0 (26A428),
  alimentation secteur annoncée par `pmset` (batterie 98 %, état thermique non
  mesuré), checkpoint local `models/Ling-3.0-tiny-EXL3-4bpw`, binaire release
  `target/release/mlxl3-rs`. Version MLX à relever avant comparaison.
- Protocole : bridge Rust résident, un warmup exclu, cinq tours identiques,
  prompt français court et 128 tokens maximum, greedy et contexte 4096 ; noter
  tokens effectivement évalués/générés, cache, TTFT, prefill, decode, hash et
  RAM/pic tels que définis par le bridge. Une seconde mesure à prompt court
  mais suffisamment long pour tester QMM sera distincte et étiquetée. Si le
  contrôle de processus GPU concurrents est indisponible, le signaler.
- Résultat : non mesuré. Intégration : aucun changement moteur, CLI, GUI ou
  publication. Preuve brute prévue sous `build/ling-perf-18-baseline.json`.
- Première tentative interrompue avant chargement : le sandbox ne voit aucun
  GPU Metal (`[metal::load_device] No Metal device available`). Le `tee` a
  retourné 0 malgré l'échec du bridge et laissé un fichier de preuve vide ;
  aucune mesure n'existe. Relance du même protocole avec accès GPU, et
  `pipefail` pour propager l'échec du benchmark.
- Relance GPU réussie : `PYTHONPATH=src .venv/bin/python
  benchmarks/benchmark_bridge.py models/Ling-3.0-tiny-EXL3-4bpw
  --native-binary target/release/mlxl3-rs --max-tokens 128 --repeats 5
  --prompt "Explique en français, avec des exemples précis, comment fonctionne
  un modèle MoE local. Compare le routage des experts, la mémoire utilisée,
  le temps de préremplissage et la vitesse de génération. Termine par deux
  limites concrètes et une conclusion courte."` ; bridge contexte 4096,
  84 tokens prompt évalués, 128 tokens générés, cache 0/84 chaque tour.
  Cinq tours chauds : decode **33,8867 tok/s** médian (33,66–33,99), prefill
  **36,3923 tok/s** médian (36,21–36,47), TTFT **2,30999 s** médian, hash
  des cinq sorties identique. Chargement 1,426 s. Le `peak_memory_gb`
  **4,42837 GB** est l'estimation des poids résidents rapportée par le bridge,
  pas une mesure du pic RAM processus. Preuve
  `build/ling-perf-18-baseline.json`. Aucun accès fiable à la liste des autres
  processus GPU dans ce sandbox ; contention éventuelle non exclue. Batterie
  sur secteur d'après `pmset`, thermique et fréquences non mesurées.
- Décision : **baseline validée** pour ce protocole court. Écart brut aux cibles :
  4,43× pour le decode et 13,74× pour le prefill ; ce ne sont pas des gains
  attendus ni une comparaison équitable au benchmark GGUF de l'utilisateur.
  Aucun code moteur/app changé. Le profilage suivant utilisera cette référence.

### OPT-2026-09-16-RUST-PERF-19 — Ling : une synchronisation par token — en cours

- Hypothèse : `Ling::run` évalue `hidden` et l'état attention après chacune des
  24 couches, comme l'ancien Qwen avant PERF-01. Une seule synchronisation des
  logits à la frontière du token doit réduire le coût CPU/Metal sans changer
  l'ordre des opérations, les poids, les caches ou le sampling.
- Antécédents : PERF-01 a validé cette stratégie sur Qwen avec parité exacte.
  PERF-07 (graphe decode entier jusqu'à argmax) a régressé ; on ne répète pas
  cette piste. Les états Ling KDA/MLA sont distincts, donc le succès Qwen ne
  préjuge ni de leur coût ni de la stabilité mémoire Ling.
- Baseline : PERF-18, 84/128 tokens, cinq tours, **33,8867 tok/s** decode,
  **36,3923 tok/s** prefill, **2,30999 s** TTFT, hash stable. Candidat : ne
  retirer que les deux `eval()` par couche de `Ling::run`, évaluer les logits
  une fois en fin de token. Binaire baseline sauvegardé avant reconstruction.
- Protocole : build release identique, sortie forcée sur au moins quatre
  tokens (logits complets FP16, comparaison bit-à-bit) et hash bridge 84/128
  identiques ; cinq tours chauds appariés avec le binaire baseline si possible.
  Mesurer decode, prefill, TTFT, stabilité sur une génération >128 tokens et
  mémoire processus si disponible. Rejeter si logits divergent, crash/OOM ou
  régression stable. Preuves sous `build/ling-perf-19-*`.
- État : avant modification ; résultat **non mesuré**, code local uniquement.
- Build release réussi avec MLX 0.32.2 ; trois méthodes `eval_state` Ling sont
  devenues inutilisées et seront retirées après validation. Avertissement
  `rust-objcopy`/`libLLVM.dylib` historique, non bloquant. Binaire baseline
  `build/ling-perf-19-baseline-bin` SHA256
  `d938d72f8ae4e600e628a27f105634558c153dd5b015e6eed36dd95d9630a26a`.
- Contrôle forcé `forward --tokens 1,2,3,4` : fichiers JSON/logits complets
  du baseline et du candidat **bit-à-bit identiques**, SHA256 commun
  `0309e48c0101ed2293061d87153eff7175e6893c5b5d816`; preuves
  `build/ling-perf-19-forced-{baseline,candidate}.jsonl`. Les états KDA/MLA
  ne sont pas exportés par le checker actuel : parité d'état non démontrée.
- Candidat, même bridge 84/128, cinq tours chauds : decode **47,8943 tok/s**
  médian (47,69–48,14), soit **+41,33 %** vs PERF-18 ; prefill
  **48,0186 tok/s** (+31,95 %) ; TTFT **1,7495 s** (−24,26 %). Hash greedy
  identique sur les cinq tours et au baseline, `peak_memory_gb` rapporté
  inchangé 4,42837 GB (poids résidents seulement). Preuve
  `build/ling-perf-19-candidate.json`.
- Décision provisoire : **gain validé pour la forme 84/128**, sans preuve de
  mémoire processus ni de stabilité longue. Avant intégration définitive,
  tester 256 tokens générés sur le même prompt et ajouter un contrôle des
  états KDA/MLA ; ne pas interpréter l'estimation résidente comme pic RAM.
- Extension 84/256, un warmup et un tour candidat : **256 tokens générés sans
  crash ni ralentissement manifeste**, decode 50,2238 tok/s, prefill 51,8312
  tok/s, TTFT 1,6209 s, hash
  `08b82405862f00df38e2cebb148860920cc1e3ec19a89a83262ed94a2fd1a3e2`.
  Preuve `build/ling-perf-19-long.json`. Un seul tour n'établit pas un gain
  comparatif ni la RAM processus ; contrôle baseline 256 à faire.
- Contrôle baseline 84/256 : 35,1428 tok/s decode, 37,6237 tok/s prefill,
  TTFT 2,2329 s, **même hash complet** que le candidat. Gain candidat sur
  ce tour comparatif : +42,91 % decode ; preuve
  `build/ling-perf-19-long-baseline.json`. Pas de conclusion sur RAM.
- Contrôle d'état prévu avant essai suivant : exposer `--states` Ling dans la
  commande de parité déjà existante, sans changer le runtime de production,
  et comparer tous les caches convolutionnels/récurrents KDA et KV/RoPE MLA
  au modèle MLX-LM Python pour deux tokens imposés. Supprimer les anciennes
  méthodes `eval_state` devenues inutilisées. Ce test n'est pas un nouvel
  essai de performance ni un assouplissement de tolérance.
- Première exécution du contrôle d'état **interrompue avant le Rust** : pour
  `ArraysCache`, `state` contient `(cache, left_padding, lengths)`, pas les
  quatre tableaux KDA directement ; `np.asarray` refuse ce tuple hétérogène.
  Aucun écart numérique observé. Corriger l'oracle Python pour lire
  `layer.cache` comme le modèle le fait, sans changer la tolérance ni les poids.
- Deuxième exécution de parité production **rejetée au premier état récurrent** :
  les trois caches convolutionnels de la couche KDA 0 passent, mais
  `model.layers.0.recurrent` diffère sur 516 653/1 048 576 octets dès le
  premier token. Ce test compare Rust optimisé à MLX-LM Python ; il ne prouve
  pas que PERF-19 a introduit l'écart, puisque l'ancien moteur Rust n'exportait
  pas ses états. Les logits forcés avant/après et le texte 256 tokens restent
  exacts. Nouvelle vérification nécessaire : construire la variante Rust
  baseline avec le même export `--states`, comparer les flux d'état bit-à-bit
  entre deux binaires et ne conserver PERF-19 que si l'écart préexiste.
- Contrôle différentiel Rust effectué : les deux variantes partagent le même
  export `--states`, la baseline restitue les évaluations `hidden` puis caches
  par couche et le candidat les diffère. Sur `--tokens 1,2 --states`, le flux
  JSON complet (tous logits et caches KDA/MLA) a le **même SHA256**
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`
  dans les deux binaires. L'écart FP32 avec MLX-LM Python préexistait donc
  à PERF-19 ; il reste à investiguer séparément et n'est pas une régression
  de l'optimisation. La variante source optimisée doit être restaurée après
  cette vérification, puis revalidée en build release.
- Source optimisée restaurée. Le checker Python garde les logits Ling exacts
  par défaut ; `--ling-states` active séparément le diagnostic d'état déjà
  connu divergent contre MLX-LM. Aucun seuil numérique assoupli.

### OPT-2026-09-16-RUST-PERF-20 — Ling KDA : regrouper cinq projections — en cours

- Hypothèse : les projections KDA q/k/v/f/g partagent la même entrée et le
  même format K4/MCG. Réutiliser `ProjectionBundle`/`Exl3Group` déjà validé
  pour Qwen/LFM peut réduire lancements QMV et préparation Hadamard sur les
  18 couches KDA, sans nouveau kernel ni nouveau format.
- Antécédents : PERF-19 (barrières Ling) est la baseline ; le rapport decode
  du 10 septembre indique que le groupement LFM a peu gagné et peut coûter en
  RAM/prefill. Ce test Ling n'est donc pas supposé positif. Les cinq sorties
  doivent conserver leur ordre, leurs dimensions et les scales indépendantes.
- Baseline/candidat : binaire `build/ling-perf-19-candidate-bin`, modèle Ling
  EXL3 4 bpw, 84/128 tokens, cinq tours chauds, médianes **47,8943 tok/s**
  decode, **48,0186 tok/s** prefill, **1,7495 s** TTFT ; candidat = groupement
  KDA uniquement, MoE/MLP/router et prefill sériel inchangés.
- Contrôles avant mesure : `forward --tokens 1,2 --states` du candidat
  bit-à-bit identique au binaire baseline (SHA256 ci-dessus), puis même hash
  greedy 84/128. Mesurer cinq tours et temps de chargement, éviter de revendiquer
  un pic RAM réel à partir de `resident_gb`. Rejeter en cas de divergence ou
  régression stable. Preuves `build/ling-perf-20-*`.
- État : avant modification ; **non mesuré**, aucun GUI/GitHub changé.
- Premier candidat construit avec MLX 0.32.2 : le flux complet `forward
  --tokens 1,2 --states` est bit-à-bit identique à PERF-19 (SHA256
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`).
  Bridge 84/128, cinq tours chauds : decode **52,9065 tok/s** médian
  (52,68–53,03), prefill **53,9263 tok/s** médian, TTFT **1,5579 s**
  médian ; les cinq hash greedy sont identiques à PERF-19. Par rapport aux
  médianes historiques PERF-19 : +10,47 % decode, +12,30 % prefill et
  −10,95 % TTFT. Preuve `build/ling-perf-20-candidate.json`. Chargement
  0,685 s sur cache disque chaud, non comparable aux 1,426 s initiaux ; RAM
  processus toujours non mesurée. **Provisoire** : effectuer A/B apparié avec
  les deux binaires maintenant avant de conclure, car le thermique et le cache
  système peuvent expliquer une partie du delta.
- Contrôle A/B/A immédiat, même prompt 84/128 et trois tours chauds par
  binaire : ancien A `build/ling-perf-20-control-a.json` **67,6628 tok/s**
  decode et **68,5514 tok/s** prefill ; candidat B
  `build/ling-perf-20-control-b.json` **69,1228 tok/s** decode et **69,6417
  tok/s** prefill ; ancien A2 `build/ling-perf-20-control-a2.json`
  **66,5412 tok/s** decode et **67,6777 tok/s** prefill. Le candidat dépasse
  les deux contrôles de ~2–4 % en decode et ~2–3 % en prefill, mais la hausse
  globale de ~48 à ~68 tok/s entre les séries historiques vient manifestement
  aussi des conditions machine. Le gain **+10,47 %** précédent est donc
  invalide comme attribution causale. Les trois séries ont le même hash de
  réponse et l'export d'état bit-à-bit reste identique. Décision : **gain
  faible/provisoire**, code local conservé pour évaluer d'autres formes,
  aucune publication ou installation GUI ; mesure RAM réelle non faite.

### OPT-2026-09-16-RUST-PERF-21 — Ling MLP partagé : groupement gate/up — en cours

- Hypothèse : le MLP partagé est invoqué dans chaque couche MoE Ling, et ses
  projections gate/up lisent le même vecteur. Le `ProjectionBundle` existant
  peut supprimer une transformation Hadamard et un lancement QMV par couche
  sans changer les poids ni le résultat. Le seul MLP dense initial suit la
  même voie. Nouveau périmètre par rapport à PERF-20 (KDA uniquement) ; les
  résultats LFM de groupement modestes sont une raison de mesurer, pas un gain
  présumé.
- Baseline : code local PERF-20, binaire `target/release/mlxl3-rs` avant
  modification, 84/128 tokens, trois tours appariés. Dernière série
  **69,1228 tok/s** decode, **69,6417 tok/s** prefill, TTFT **1,2064 s** ;
  conditions machine variables, à mesurer en A/B/A. Contrôle strict prévu :
  `forward --tokens 1,2 --states` SHA256 identique, hash greedy identique.
- Changement : `Mlp` Ling réutilise `ProjectionBundle` pour gate/up et conserve
  down inchangé. Aucun kernel inédit, quantification, sampler ou GUI modifié.
  Mesurer chargement et RAM processus si possible ; rejeter toute divergence,
  crash ou régression stable. Preuves sous `build/ling-perf-21-*`.
- État avant essai : **non mesuré**, code local uniquement.
- Build release réussi (MLX 0.32.2, avertissement non bloquant `rust-objcopy`
  identique aux builds précédents). Les logits et tous les états KDA/MLA de
  `forward --tokens 1,2 --states` conservent exactement le SHA256
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`.
  Le hash greedy 84/128 reste identique sur tous les tours.
- A/B/A rapproché, trois tours chauds chacun : A PERF-20 **58,834** decode,
  **59,327** prefill, **1,416 s** TTFT ; B candidat **59,770** decode,
  **60,014** prefill, **1,400 s** TTFT ; A2 PERF-20 **58,132** decode,
  **58,241** prefill, **1,442 s** TTFT. Unités tok/s hors TTFT. Preuves
  `build/ling-perf-21-control-a.json`, `build/ling-perf-21-candidate.json`,
  `build/ling-perf-21-control-a2.json`. Le candidat est ~1,6–2,8 % au-dessus
  des deux contrôles en decode, ~1,2–3,0 % en prefill. **Gain faible validé
  pour cette forme**, pas un progrès vers 150/500 à lui seul. Chargement
  ~0,69–0,70 s sur cache chaud ; pic RAM processus non mesuré. État : code
  local seulement ; GUI/app non reconstruits, aucun commit/publication.

### OPT-2026-09-16-RUST-PERF-22 — KDA vector : noyau Metal multi-token — en cours

- Hypothèse : le noyau `gated_delta::step_vector` n'accepte que `T=1`, ce qui
  interdit le prefill Ling par lots. Parcourir `T` dans chaque thread Metal
  tout en gardant la même accumulation FP32 et le même ordre par token évite
  de relancer ce noyau à chaque token. Le chemin decode `T=1` doit rester
  bit-à-bit inchangé ; ce changement seul n'accélère pas encore le prefill de
  l'app avant que les autres blocs Ling acceptent le batch.
- Antécédents : Qwen utilise déjà un noyau Gated DeltaNet multi-token, mais
  Ling a une décroissance vectorielle `[B,T,H,128]` et un état différent.
  PERF-18–21 montrent surtout que Ling est actuellement sériel en prefill.
- Baseline : `step_vector` actuel, test GPU one-hot `vector_step_matches_one_hot_update`
  et binaire PERF-21. Nouveau contrôle : comparer sur deux pas imposés le
  résultat et l'état d'un appel `T=2` aux deux appels `T=1` enchaînés, d'abord
  sur un cas simple puis sur tenseurs déterministes. Toute divergence décisive
  doit être expliquée avant intégration. Mesure de performance : non applicable
  au chat tant que le batch Ling complet n'est pas activé ; ne pas revendiquer
  de gain end-to-end prématurément.
- État avant essai : **non mesuré**, code local uniquement, pas de GUI/push.
- Implémentation locale : boucle temporelle dans le noyau Metal vectoriel,
  un seul chargement et une seule écriture de l'état FP32 par thread, mêmes
  opérations par token. `cargo test --release --locked --features mlx
  gated_delta::tests::vector_batch_matches_serial_steps -- --ignored --nocapture`
  sur M5 : **réussi**, sorties FP16 et état FP32 exactement égaux pour deux
  tokens déterministes. L'avertissement `rust-objcopy` reste non bloquant.
  Contrôle sur tenseurs variés/modèle entier encore à faire avant activation
  du prefill. Statut : **prototype validé pour ce cas**, aucun gain chat mesuré,
  aucune app installée ni publication.

### OPT-2026-09-16-RUST-PERF-23 — Ling prefill groupé 84 tokens — en cours

- Hypothèse : le bridge sérialise Ling à un token par appel, ce qui maintient
  toutes les projections EXL3 sur QMV et relance 24 couches par token. Avec
  PERF-22 (état KDA temporel), réutiliser QMM et MoE segmenté déjà présents
  devrait améliorer surtout le prefill/TTFT. Ce test est spécifique à Ling ;
  les poids, le sampling et le chemin decode `T=1` restent inchangés.
- Plan minimal : accepter `[1,T,H]` dans KDA, MLA et feed-forward Ling ; faire
  parcourir une ligne par thread au routeur groupé (même calcul par ligne) ;
  ajouter le masque causal au biais MLA et conserver uniquement les logits du
  dernier token ; autoriser des chunks de 128 tokens sur le bridge. Pas de
  nouveau QMM/quantification. Les caches doivent représenter tous les tokens
  du chunk, et le chemin `T=1` doit conserver les résultats bit-à-bit.
- Baseline : PERF-21, 84/128 tokens, trois tours, **59,770 tok/s** decode,
  **60,014 tok/s** prefill, **1,400 s** TTFT sur son dernier run ; les
  conditions machine varient. Pour une attribution causale : A/B/A rapproché
  contre le binaire PERF-21 préservé, avec cinq tours si stable. Vérifier
  d'abord un prefill batch de 2/25/84 tokens vs sérial : logits, tous caches
  KDA/MLA, séquence greedy et absence de crash. Si l'arithmétique QMM diffère
  bit-à-bit de QMV, appliquer un seuil numérique justifié et vérifier le
  routing/texte, sans prétendre à une égalité exacte.
- Rejeter/limiter le batch s'il corrompt l'état, diverge dans le texte ou
  fait exploser la RAM processus. Preuves `build/ling-perf-23-*`.
- État avant essai : **non mesuré**, code local seulement, aucune installation
  ou publication.
- Premier test GPU `cargo test --release --locked --features mlx
  ling::tests::batched_prefill_matches_serial_state -- --ignored --nocapture`
  **échoué** avant benchmark : 25 tokens imposés, logits finaux écart maximal
  absolu 0,31445313 ; pire cache `model.layers.13.conv_q` 0,48632813. Le
  chemin sériel `forward --tokens 1,2 --states` reste exactement identique au
  SHA256 précédent. Le candidat batch ne peut pas être activé en production en
  l'état. Causes possibles : QMM vs QMV, convolution batch ou routage ; isoler
  par couche avant toute mesure de vitesse. Aucun gain revendiqué.
- Répétition diagnostique du même test, avec sortie des premiers caches qui
  divergent : cette répétition ne cherche pas un gain, elle doit déterminer si
  l'écart démarre dès la convolution KDA 0 (projection/conv) ou plus tard
  (routage/MLA/récurrence). Aucun seuil n'est assoupli.
- Résultat diagnostic : premier écart à `model.layers.0.conv_q` de 0,001953125,
  compatible avec un changement d'arithmétique QMM/QMV ; les écarts croissent
  ensuite (MLA couche 3 : `kv_cache` 0,0480 ; cache conv q couche 13 : 0,4863).
  Ce test seul ne distingue pas une petite différence de projection amplifiée
  par MoE d'une erreur de masque/état. Test suivant : chat réel 84 tokens avec
  hash greedy et débits, **diagnostic seulement** ; ne pas intégrer même si
  rapide tant que parité et stabilité ne sont pas établies.
- Première commande bridge diagnostique **interrompue avant chargement** :
  `cargo test --features mlx` a remplacé `target/release/mlxl3-rs` par un
  binaire sans feature `chat` ; le bridge signale `requires a build with
  --features mlx,chat`. Aucune mesure. Reconstruire explicitement ces deux
  features puis relancer le même diagnostic.
- Diagnostic chat après rebuild correct : prompt réel 84 tokens, 128 générés,
  un tour chaud, **104,36 tok/s prefill**, **55,31 tok/s decode**, TTFT
  **0,805 s** ; hash greedy **identique** à PERF-21. Preuve
  `build/ling-perf-23-diagnostic.json`. Warmup 25 tokens : prefill seulement
  12,88 tok/s et TTFT 1,941 s, potentiellement compilation/shape ; ne pas
  masquer ce coût. Le test numérique forcé reste échoué, et ce seul hash réel
  ne suffit pas à valider la fidélité générale. Contrôle A/B/A et plusieurs
  prompts nécessaires après localisation de l'écart ; aucune activation
  durable, installation ou publication encore validée.
- Profil diagnostic prévu : exécuter un chunk de 84 tokens avec une barrière
  `eval()` **uniquement dans un test ignoré**, après chaque couche, relever
  chaque durée et comparer KDA/MLA/MoE. Le profil modifiera l'ordonnancement
  GPU et ne sera **pas** une mesure end-to-end ni une optimisation. Il doit
  seulement choisir la prochaine piste, sans ajouter de barrières au moteur.
- Profil 84 tokens réussi : couche 0 56,9 ms, couche 1 **458,5 ms**, toutes les
  couches 2–23 ensuite ~3,3–10,1 ms, head 3,0 ms. Test ignoré
  `ling::tests::profile_batched_prefill_layers`, sortie console de ce run
  (pas de log persistant). La couche 1 n'est pas intrinsèquement lente : elle
  semble payer la compilation du premier MoE QMM segmenté ; les couches
  suivantes partagent la shape compilée. C'est une **inférence**, pas encore
  prouvé par un deuxième run chaud. Répéter le benchmark bridge **dans le
  même processus et avec la même forme 84 tokens** doit distinguer coût de
  compilation/TTFT initial et débit stable. Cette répétition reprend le
  protocole PERF-23, sans changement de code moteur.
- Cinq requêtes consécutives 84/128 dans le même bridge, après warmup de
  **25 tokens** : prefill **569,95 tok/s** médian (566,5–573,6), TTFT
  **0,148 s** médian, decode **55,31 tok/s** médian (55,2–55,8) ; cinq hash
  greedy identiques à PERF-21. Preuve `build/ling-perf-23-batch-five.json`.
  La requête précédente à 104 tok/s était la première compilation de cette
  shape 84, pas le débit chaud stable. La cible **500 tok/s en prefill chaud
  sur cette forme** est atteinte par le prototype, mais ni le prefill froid,
  ni la fidélité numérique sur 25 tokens imposés, ni le 150 tok/s decode.
  Ne pas généraliser à d'autres longueurs/contextes : compilation par shape
  probable. Statut : **prototype non validé pour activation** tant que l'écart
  logits/caches et plusieurs prompts ne sont pas évalués.
- Vérification numérique suivante avant décision : tester le routeur groupé
  multi-lignes séparément contre quatre appels mono-ligne, avec logits/biais
  déterministes et comparaison **bit-à-bit** des indices et scores. Cela
  cherche une erreur logique de batch, pas un gain de vitesse ni un
  assouplissement du seuil modèle entier.
- Test GPU `router::tests::grouped_router_batch_matches_serial` **réussi** :
  quatre lignes, 128 experts, top-8, indices et scores exactement identiques
  aux quatre calculs mono-ligne. Le routeur multi-lignes seul n'explique pas
  l'écart numérique observé dans le modèle. Compilation/trace du test dans
  la sortie de `cargo test` locale, non conservée en fichier brut.
- Contrôle qualité supplémentaire prévu, sans changement du moteur : avec le
  binaire PERF-23 restauré, comparer le dernier logit de `forward` sériel et
  `forward --batch` sur 25 IDs imposés, puis sur une phrase française tokenisée.
  Lire les valeurs comme FP16 (pas comme entiers), relever top-1, écart absolu
  moyen/maximal et KL des softmax FP32/64. Vérifier ensuite deux chunks et le
  hash greedy réel. Une égalité du texte seule ne suffit pas ; si l'écart
  numérique reste grand, garder le batch hors du chemin normal. Preuves
  `build/ling-perf-23-quality-*`, alimentation/thermique non contrôlées mais
  non pertinentes pour ce contrôle de valeur.
- Résultat : 25 IDs imposés, même top-1 et top-10/10, écart absolu moyen
  **0,05437**, maximal **0,31445**, KL **0,0000689** ; prompt français de
  17 tokens passé par le chemin sériel (seuil 24), égalité exacte attendue.
  Prompt français répété jusqu'à ~85 tokens, batch effectivement activé :
  top-1 identique, top-10/10, moyenne **0,10165**, maximum **0,57422**,
  KL **0,0027523**. Fichiers `build/ling-perf-23-quality-{serial,batch}-{25,fr,fr85}.jsonl`.
  Le seuil provisoire max 0,5 cité pour d'autres familles est dépassé ;
  cela ne prouve ni corruption logique ni absence de dérive en multi-tours.
  **Statut non concluant pour activation générale** : maintenir le batch en
  prototype local et ne pas l'installer/publier avant davantage de prompts et
  contrôle des deux chunks. Le decode mono-token reste la référence fidèle.
- Validation chat supplémentaire prévue, **répétition de PERF-23** motivée par
  le nouveau routeur PERF-26 et par l'absence de test multi-chunks : comparer
  le binaire sériel PERF-21 (`build/ling-perf-21-candidate-bin`) au candidat
  batch+routeur PERF-26 sur un prompt >128 tokens (deux chunks), un prompt de
  code et un autre français. Un warmup puis une répétition par forme ; vérifier
  hash greedy, absence de crash, nombres de tokens, préfill/TTFT/decode, et ne
  pas extrapoler la RAM du bridge. Preuves `build/ling-perf-23-quality-chat-*`.
- Première tentative multi-chunks **interrompue sans mesure** : le binaire
  PERF-21 archivé a été compilé sans `chat`, malgré l'aide CLI montrant
  `bridge` ; erreur explicite `requires a build with --features mlx,chat`.
  `build/ling-perf-23-quality-chat-long-serial.json` est vide/non valide.
  Reprendre le même prompt avec le binaire sériel PERF-19 archivé avec
  `chat` après vérification ; son hash forcé était identique à PERF-21.
- Prompt français 146 tokens (deux chunks 128+18), 96 tokens générés, même
  hash greedy sur baseline sérielle PERF-19 et batch+routeur PERF-26.
  Sériel : prefill **58,42 tok/s**, TTFT **2,500 s**, decode **55,93 tok/s** ;
  batch : prefill **97,71 tok/s**, TTFT **1,495 s**, decode **78,28 tok/s**.
  Le second chunk de 18 reste sériel par seuil 24 et la première shape 128
  paye probablement la compilation ; ce test n'est pas un débit chaud
  stabilisé, mais établit que deux chunks passent sans crash ni divergence
  greedy sur cet exemple. Preuves `build/ling-perf-23-quality-chat-long-{serial19,batch26}.json`.
  Fidélité numérique stricte multi-prompt encore non démontrée : **prototype**.
- Dernier contrôle qualité prévu avant arrêt de cette série : comparer sur une
  demande de code distincte (prompt >24 tokens, 96 tokens greedy) la version
  sérielle PERF-19 et le build final batch+routeur PERF-26, un warmup + un
  tour par binaire. Noter hash, tokens, débit ; ce contrôle de texte ne remplace
  pas une évaluation de qualité large. Preuves
  `build/ling-perf-23-quality-chat-code-{serial,batch}.json`.
- Demande de code 87 tokens / 96 générés : baseline sérielle et batch ont
  **des hashes greedy différents**, bien que les premiers ~120 caractères
  enregistrés soient identiques. Sériel 56,17 tok/s decode, 57,33 tok/s
  prefill, TTFT 1,518 s ; batch+PERF-26 78,14 decode, 344,57 prefill,
  TTFT 0,253 s (première compilation shape 87). Preuves
  `build/ling-perf-23-quality-chat-code-{serial,batch}.json`. La vitesse
  ne compense pas cette divergence tant que sa qualité n'est pas examinée.
  **Décision provisoire : ne pas activer le préfill batch Ling par défaut** ;
  conserver `forward_tokens` comme prototype diagnostic/benchmark et revenir
  au préfill sériel pour le chat normal. Le gain decode PERF-26 est indépendant
  et conserve la parité exacte sur les tokens forcés.

### OPT-2026-09-16-RUST-PERF-24 — Ling decode : profil attention/MoE — en cours

- Hypothèse de diagnostic : avec le prefill chaud désormais >500 tok/s,
  l'objectif restant (150 tok/s decode) nécessite de réduire le coût par
  token de ~18 ms à <6,7 ms. Mesurer d'abord la part des 18 couches KDA,
  6 MLA et 23 MoE ; ne pas supposer que l'EXL3 QMV est seul responsable.
- Antécédents : PERF-18–21 ont retiré des barrières et groupé les projections
  Ling ; gains decode validés ou modestes. Le rapport decode du 10 septembre
  a rejeté les changements globaux de budgets command-buffer et quelques
  compilations FFN/QMV. Aucune répétition de ces pistes ici.
- Baseline : bridge local PERF-23, decode ~55 tok/s dans cinq tours, même
  texte que l'ancien moteur, machine/thermique variables. Protocole : test
  ignoré sur Metal, un warmup `T=1`, puis un token imposé avec barrières de
  diagnostic séparant attention et feed-forward par couche. Ces barrières
  changent le coût absolu : interpréter le profil **relatif seulement**. Pas
  de changement production, mesures RAM et TTFT non applicables à ce profil.
- État avant essai : **non mesuré**, code local, aucune publication.
- Profil diagnostique mono-token réussi après un warmup, avec `eval()` après
  chaque sous-bloc : hors deux pointes probables de JIT/scheduler (couche 1
  FF 2,893 ms, couche 7 attention 2,918 ms et couche 19 attention 2,325 ms),
  attention ~0,40–0,58 ms/couche, FF MoE ~0,54–0,63 ms/couche, dense couche
  0 ~0,30 ms. Trace console du test
  `ling::tests::profile_decode_components`, pas de log brut conservé. Les
  barrières rendent la somme non représentative du débit bridge, mais le
  MoE est probablement la première cible decode. Étape suivante : séparer
  dans un seul MoE routeur, experts EXL3 et branche MLP partagée avant de
  changer un kernel. Statut : **diagnostic**, aucune optimisation validée.
- Extension du même diagnostic prévue : sur une couche MoE représentative,
  après warmup, cinq répétitions séparant routeur, experts `Exl3SwitchGlu` et
  MLP partagé. Les `eval()` de profil empêchent la fusion/chevauchement et
  chaque durée est une borne indicative, pas un débit réel. Cette répétition
  vise à sélectionner la partie à optimiser, pas à déclarer un gain.
- Résultat couche MoE 8, répétitions 2–4 après premier JIT : routage
  **0,553–0,560 ms**, experts EXL3 **0,274–0,279 ms**, MLP partagé
  **0,194–0,205 ms**. Le routage inclut matmul FP32, sigmoid, sélection
  mono-thread et normalisation des scores, avec barrières `eval()` ; sa part
  est la plus importante de ce microprofil mais n'est pas additionnable aux
  temps end-to-end. Trace `ling::tests::profile_decode_moe` console seulement.
  Décision : essayer une **fusion du routeur Ling mono-token** dans un kernel
  Metal, avec fallback inchangé pour le batch et test strict des experts
  sélectionnés ; ne pas toucher au QMV expert pour l'instant.

### OPT-2026-09-16-RUST-PERF-25 — Ling : routeur MoE mono-token fusionné — en cours

- Hypothèse : fusionner multiplication dense x·W, sigmoid, top-groups/top-k
  et normalisation dans un seul kernel Metal pour 128 experts évite plusieurs
  dispatchs par couche MoE, sans changer poids, experts EXL3 ou sampling.
  Le chemin multi-token PERF-23 garde le routeur existant. Aucun routeur Qwen,
  Gemma ou LFM n'est modifié.
- Baseline : code/binaire PERF-23 préservé avant modification, bridge 84/128,
  cinq tours chauds, **~55,31 tok/s** decode dans la série la plus récente,
  prefill batch **569,95 tok/s** chaud, hash constant. Machine variable : A/B/A
  rapproché indispensable. Contrôles : parité indices de route sur entrées
  déterministes puis `forward --tokens 1,2 --states` contre binaire baseline,
  hash greedy 84/128 et 256 tokens, RAM et TTFT rapportés séparément.
- Rejeter si experts changent, régression stable ou erreur Metal ; si les
  scores ne sont pas bit-à-bit identiques, quantifier l'écart et vérifier les
  logits/caches, sans appeler cela parité exacte. Preuves `build/ling-perf-25-*`.
- État avant essai : **non mesuré**, code local uniquement, aucun push/GUI.
- Kernel Metal fusionné local ajouté pour `T=1`, avec matrice de gate FP32
  stockée aussi en vue rangées et routeur multi-token inchangé. Test GPU
  déterministe `router::tests::fused_ling_router_matches_reference` **réussi**
  pour 128 experts, 128 entrées : huit indices identiques et erreur maximale
  de score <1e-4. Ce microtest ne prouve pas encore la parité du modèle réel
  1536 entrées. Prochaine étape déjà prévue : build bridge puis comparer
  `forward --tokens 1,2 --states` au SHA256 du binaire PERF-23 avant tout
  benchmark de débit.
- Contrôle `forward --tokens 1,2 --states` **non identique** au binaire
  PERF-23 : SHA256 candidat
  `9e269218137925b1a27e230982573497f446c4beddc0d02c5a2f55323ab3578e`
  contre baseline
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`.
  Le test synthétique ne suffisait donc pas. Avant toute mesure de vitesse,
  comparer les logits forcés détaillés et identifier si la différence est
  une petite variation FP32 de scores ou un changement de route/résultat.
- Détail logits forcés `--tokens 1,2` : au token 1, **85 814/157 184**
  valeurs FP16 diffèrent ; au token 2 **156 236/157 184**, même argmax aux
  deux tokens mais divergence trop grande pour accepter la fusion. Preuves
  `build/ling-perf-25-forced-{baseline,candidate}.jsonl`. Le `maxbits`
  calculé sur encodages FP16 n'est pas un écart réel et n'est pas retenu.
  Diagnostic suivant : comparer directement les routes/scores fusionnés et
  non fusionnés avec les **vrais poids** des couches Ling 1 et 8, afin de
  déterminer si la disposition de la matrice FP32 est fautive. Pas de
  benchmark de vitesse tant que le résultat est faux.
- Vrais poids Ling couches 1 et 8, entrée embedding token 1 : **indices
  identiques** et scores à <1e-4 (test GPU
  `ling::tests::fused_router_matches_ling_weights` réussi). La disposition
  FP32 n'est donc pas grossièrement inversée. Ce test utilisait l'embedding
  brut, pas l'entrée MoE normalisée des couches en cours. Répétition
  diagnostique prévue : inspecter indices et écarts de score à chaque couche
  sur la vraie trajectoire du token imposé, sans nouvelle optimisation.
- Vraies entrées MoE des 23 couches sur deux tokens : les indices fusionnés
  et non fusionnés sont égaux **sur une même trajectoire fusionnée**, écarts
  des scores 7e-8 à 1,55e-6 (`fused_router_matches_decode_inputs`). Mais
  les deux trajectoires de modèle complètes divergent : après conversion
  correcte des bits FP16, token 1 logits max abs **0,015625**, moyenne abs
  **0,00206**, même argmax ; token 2 max abs **1,3046875**, moyenne abs
  **0,1957**, **argmax différent (220 vs 16)**. Le calcul initial `argmax`
  sur entiers bruts était erroné et est corrigé ici. Le routeur fusionné
  amplifie donc un changement numérique jusqu'au texte, ce qui viole la
  parité demandée. Décision : **rejeté**, ne pas benchmarker ni activer ;
  retirer ce kernel et ses champs/tests spécifiques du code de production.
- Retrait effectué de `router.rs` et `ling.rs` : plus de chemin routeur fusionné
  ni de champ supplémentaire. Le journal conserve la conclusion négative.
  `cargo fmt --check` à refaire après retrait, puis rebuild et contrôle hash
  forcé pour certifier le retour au moteur PERF-23.
- Rebuild release `mlx,chat` après retrait réussi ; `forward --tokens 1,2
  --states` revient exactement au SHA256 PERF-23
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c`.
  Le routeur fusionné PERF-25 n'est donc plus présent dans le candidat local.

### OPT-2026-09-16-RUST-PERF-26 — Ling : sélection MoE parallèle sans fusion des logits — en cours

- Hypothèse : le routeur groupé de 128 experts exécute les scans top-groups et
  top-k dans un unique thread, avec une recherche `used` répétée. Répartir le
  calcul des scores par expert/groupe sur 128 threads, puis conserver la
  sélection finale ordonnée dans le thread 0, peut réduire le temps de route
  sans toucher à `x·W`, sigmoid, normalisation ni poids. Contrairement à
  PERF-25 rejeté, aucune modification de l'arithmétique des logits/scores.
- Baseline : binaire PERF-23 conservé `build/ling-perf-23-batch-bin`, hash
  `forward --tokens 1,2 --states` =
  `9715bf2438ebccf5b695be4fdca1d77d8da8e3e7b7f95c115ee13a5d10b74a8c` ;
  84/128 tokens, decode chaud **55,31 tok/s** médian dans cinq tours, prefill
  chaud **569,95 tok/s** (thermique/alimentation non contrôlées). Test GPU
  existant des lignes du routeur, puis hash exact logits+états forcés,
  comparaison greedy et benchmark A/B/A rapproché seulement si ces contrôles
  réussissent. Preuves prévues `build/ling-perf-26-*`.
- Statut initial : **en cours**, code local seulement ; aucune publication.
- Kernel de sélection parallèle `experts=128` compilé ; quatre lignes donnent
  les mêmes indices/scores que quatre appels mono-ligne dans le test GPU
  `grouped_router_batch_matches_serial`. Cela ne compare pas encore le kernel
  antérieur, ni ne garantit le modèle entier. Rebuild `mlx,chat` et hash forcé
  requis avant benchmark. Avertissement `rust-objcopy/libLLVM` non bloquant.
- Contrôle modèle entier : SHA256 `forward --tokens 1,2 --states` **exactement
  identique** à la baseline (`9715bf...10b74a8c`). A/B/A rapproché, trois
  requêtes chaudes par binaire, prompt 84 tokens / 128 générés, même hash
  greedy sur les neuf requêtes : ancien A **54,918 tok/s decode**, **566,60
  tok/s prefill**, TTFT **148,47 ms** ; candidat B **75,128 tok/s decode**,
  **586,09 tok/s prefill**, TTFT **143,56 ms** ; ancien A2 **54,800 tok/s
  decode**, **568,99 tok/s prefill**, TTFT **147,82 ms**. Gain decode
  **+36,8–37,1 %** face aux deux contrôles. Le préfill +3–3,4 % est plus
  petit et peut contenir du bruit ; TTFT −3 % idem. Preuves
  `build/ling-perf-26-{control-a,candidate,control-a2}.json`.
- **Validé pour ce scénario**, code local uniquement, pas d'app installée ni
  publication. `peak_memory_gb` bridge identique 4,428 GB mais représente
  les poids, **pas** la RAM/pic processus. Tester plusieurs prompts/contexte
  plus long et vérifier la fidélité batch PERF-23 avant activation GUI.

### OPT-2026-09-17-RUST-PERF-27 — Ling : profil MoE après routeur parallèle — en cours

- Hypothèse de diagnostic : PERF-26 a retiré ~4,9 ms/token du routage sur le
  test de 84/128 tokens, mais le decode reste ~13,3 ms/token. Réexécuter le
  profil MoE de la couche 8 pour vérifier que le routage n'est plus le seul
  coût dominant avant tout nouveau changement. Même test GPU ignoré que
  PERF-24, cinq répétitions, temps après la première compilation ; barrières
  `eval()` artificielles, donc **microprofil non comparable au débit chat**.
- Baseline : PERF-24 route **0,553–0,560 ms**, experts **0,274–0,279 ms**,
  MLP partagé **0,194–0,205 ms** (avant PERF-26). Candidat : PERF-26,
  binaire `cargo test --release --locked --features mlx` avec le même
  checkpoint 4 bpw/M5. Aucune modification du code et pas de publication.
- Statut : **en cours**, mesure/log brut à conserver dans
  `build/ling-perf-27-moe-profile.log`.
- Profil effectué : routes des répétitions chaudes 0,365 / 0,365 / 0,659 /
  0,414 ms ; un pic scheduler/JIT à 0,659. Expert EXL3 ~0,285–0,335 ms
  hors pic, MLP partagé ~0,203–0,218 ms hors pic. La baisse de route contre
  ~0,55 ms de PERF-24 concorde avec le gain end-to-end PERF-26, sans en être
  une mesure équivalente. Le routeur reste coûteux et le scan top-k mono-thread
  pourrait encore être réduit ; les projections experts/partagées sont aussi
  une limite. Preuve brute `build/ling-perf-27-moe-profile.log`.
- **Diagnostic terminé**, aucune modification moteur, aucun push/app installée.

### OPT-2026-09-17-RUST-PERF-28 — Ling : top-k MoE par réductions SIMD — en cours

- Hypothèse : PERF-26 parallélise l'initialisation des 128 candidats mais
  laisse au thread 0 huit scans de 128 experts. Une réduction `simd_max` par
  groupe de 32 avec bris d'égalité par plus petit index, puis un scan de
  quatre groupes, doit préserver exactement l'ordre de sélection tout en
  réduisant le travail sériel. Garder sigmoid, matmul, normalisation et
  tous les buffers persistants inchangés. Applicable au seul routeur 128
  experts Ling ; fallback générique inchangé.
- Baseline : PERF-26, binaire actuel à archiver, 84/128 trois tours A/B/A,
  **75,128 tok/s decode**, prefill **586,09 tok/s**, TTFT **143,56 ms** ;
  hash forcé des logits+états `9715bf...10b74a8c`, neuf hashes greedy
  identiques. Vérifier test GPU routeur multi-lignes, hash forcé puis A/B/A.
  Si sortie non identique ou gain non reproductible, revenir à PERF-26.
  Conditions M5/MLX 0.32.2, alimentation/thermique non contrôlées ; preuves
  prévues `build/ling-perf-28-*`.
- Statut : **en cours**, code local, aucun push ni installation GUI.
- Test GPU routeur multi-lignes sur M5 **réussi** avec réduction SIMD ; sortie
  exacte face aux quatre appels mono-ligne du même candidat. Il faut encore
  comparer modèle entier à PERF-26 et mesurer A/B/A avant d'accepter.
- Modèle entier `forward --tokens 1,2 --states` SHA256 **identique** à
  PERF-26 (`9715bf...10b74a8c`). Premier bridge candidat 84/128, trois
  tours : **77,205 tok/s decode**, **591,51 tok/s prefill**, TTFT
  **142,57 ms**, même hash greedy. Face au PERF-26 historique 75,128 tok/s,
  le +2,8 % decode n'est **pas encore attribuable** sans A/B/A. Le fichier
  `build/ling-perf-26-bin` copié avant le test est invalide pour le bridge
  (sans feature `chat`, écrasé par `cargo test`) ; reconstruire explicitement
  ce binaire de contrôle, ne pas prétendre l'avoir mesuré.
- A/B/A après reconstruction correcte du binaire PERF-26, trois tours chauds
  chacun : A **76,629 tok/s decode**, **591,40 tok/s prefill**, TTFT
  **143,11 ms** ; candidat SIMD B **77,002 tok/s decode**, **590,33 tok/s
  prefill**, TTFT **142,56 ms** ; A2 **77,556 tok/s decode**, **586,82 tok/s
  prefill**, TTFT **143,36 ms**. Tous les hashes greedy identiques. Le
  candidat est entre A et A2 : **aucun gain reproductible**, le +2,8 %
  précédent venait des conditions. Preuves
  `build/ling-perf-28-control-{a,b,a2}.json`.
- Décision : **rejeté**. Revenir à la sélection PERF-26 plus simple, laisser
  le journal/preuves négatifs ; ne pas installer/publier PERF-28. RAM processus
  non mesurée ; mémoire `peak_memory_gb` du bridge n'est que poids résidents.

### OPT-2026-09-17-RUST-PERF-29 — Ling : normalisation des scores dans le routeur — en cours

- Hypothèse : après sélection des huit experts, `scores.sum` puis `div` et
  `scalar_mul` passent par MLX et créent des opérations GPU additionnelles
  à chaque couche MoE. Normaliser les scores FP32 dans le thread 0 du kernel
  de sélection pourrait économiser ces dispatchs, sans changer les experts,
  poids ou sampling. Risque connu : ordre de réduction FP32 différent de MLX
  donc perte de parité ; **rejet immédiat si hash forcé diverge**, même si la
  réponse greedy semble identique.
- Baseline : PERF-26, binaire `build/ling-perf-26-bin` reconstruit avec chat,
  **76,63–77,56 tok/s** decode dans A/B/A récent, hash forcé
  `9715bf...10b74a8c`. Vérifier d'abord test GPU routeur et modèle entier
  `forward --tokens 1,2 --states`; benchmark A/B/A seulement si exacte.
  M5/MLX 0.32.2 ; conditions thermiques non contrôlées. Preuves prévues
  `build/ling-perf-29-*`.
- Statut : **en cours**, code local, aucun push/app installée.
- Test GPU routeur multi-lignes réussi ; `forward --tokens 1,2 --states`
  SHA256 **exactement identique** à PERF-26 (`9715bf...10b74a8c`). L'ordre
  de réduction de huit scores dans ce cas retrouve donc la même sortie
  complète ; cela ne démontre pas tous les contextes/routes. Passer au
  benchmark A/B/A 84/128 avant décision. Preuves
  `build/ling-perf-29-router-test.log`, `build/ling-perf-29-forced.jsonl`.
- A/B/A, trois tours chauds chacun, même hash greedy : A **78,308 tok/s
  decode**, **596,93 tok/s prefill**, TTFT **140,97 ms** ; candidat B
  **78,280 tok/s decode**, **582,52 tok/s prefill**, TTFT **144,44 ms** ; A2
  **76,291 tok/s decode**, **594,26 tok/s prefill**, TTFT **141,66 ms**.
  Aucun gain decode reproductible, préfill ~2 % plus bas dans ce test.
  Preuves `build/ling-perf-29-{control-a,candidate,control-a2}.json`.
- Décision : **rejeté** malgré parité exacte sur deux tokens. Les opérations
  MLX séparées peuvent être masquées/fusionnées par le graphe ; la
  normalisation dans le kernel n'accélère pas le bridge. Revenir à PERF-26,
  ne pas publier/installer PERF-29.

### OPT-2026-09-17-RUST-PERF-30 — Ling : gate qualité du prefill de chat — en cours

- Hypothèse : les tests PERF-23 montrent un gain batch important mais une
  divergence greedy sur une demande de code. Le chat de production doit donc
  utiliser le chemin sériel de référence tant que cette fidélité n'est pas
  qualifiée, en conservant `forward --batch` explicite pour le diagnostic.
  Le routeur PERF-26, indépendant, reste actif et strictement paritaire.
- Changement minimal : `NativeChatModel::forward_many` dispatch Ling vers
  `forward_many_serial`; `Forward --batch` appelle directement
  `Ling::forward_tokens`. Aucun autre modèle, format, sampler ou poids
  modifié. Baseline qualité : binaire PERF-19 sériel et hash forcé
  `9715bf...10b74a8c`. Mesurer le chat normal 84/128 avec trois répétitions
  et comparer hash/température connue ; puis smoke test explicite `--batch`
  pour s'assurer que le prototype reste accessible. Preuves
  `build/ling-perf-30-*`. Préfill 500 tok/s **non revendiqué** pour le chat
  sûr ; TTFT/RAM mesurés séparément.
- Statut : **en cours**, code local, pas d'app installée ni publication.
- Revalidation physique M5 du 19 septembre : arrays MLX, codec Metal,
  Gated DeltaNet mono-token et batched, et routeur groupé mono/batch passent.
  Le test diagnostic Ling 25 tokens confirme en revanche la divergence connue
  du préfill batched (logits max abs **0,314**, état max abs **0,486**) ; ce
  chemin reste donc réservé à `forward --batch` et n'est pas utilisé par le
  chat/GUI. Le chemin de production sériel, comparé sur huit tokens à MLX-LM,
  conserve le même top-1 à chaque étape, avec max abs **0,3114** et KL max
  **0,002234**. Le vérificateur a été corrigé pour appliquer à Ling la borne
  numérique déjà utilisée pour les modèles non bit-exacts, au lieu d'exiger
  à tort une égalité FP16 bit-à-bit. Aucun débit n'a été remesuré dans cette
  revalidation.
- **Validé et intégré pour publication sur `main`** : gate sériel actif dans
  le chat/GUI, prototype batch explicitement hors chemin normal. L'app locale
  et une release ne sont pas modifiées par ce push.

### OPT-2026-09-19-RUST-PERF-31 — Campagne générale : baseline et profil Qwen complet — en cours

- Hypothèse de diagnostic : le moteur Rust actuel possède déjà les gains de
  synchronisation, QMV/QMM, routage et chargement documentés ci-dessus ; une
  nouvelle optimisation doit donc partir des coûts réellement dominants du
  modèle complet, et non répéter les réglages de command buffers, compilation
  FFN/QMV, gather/Hadamard ou top-k déjà rejetés.
- Cible initiale : Qwen3.6-35B-A3B EXL3 2,49 bpw, puis validation de toute
  piste générale sur Ling 3 Tiny et les autres architectures compatibles.
  Priorité : decode, prefill, TTFT, RAM, chargement, CPU/dispatch/allocations.
- Baseline prévue : binaire `main` `ce5ede9`, bridge natif, un warmup exclu,
  au moins cinq générations chaudes de 256 tokens avec prompt fixé, puis
  tailles de prefill séparées. Relever débit moteur/client, TTFT, tokens,
  hash du texte, mémoire MLX et RSS. Les comparaisons de candidats seront
  alternées A/B/A avec binaire de contrôle archivé et sorties forcées/parité.
- Profil prévu : trace Qwen synchronisée existante sur une passe mono-token et
  une passe multi-token, complétée par échantillonnage CPU du bridge. Les
  barrières de profil modifient le débit et ne seront utilisées que pour
  classer les sous-blocs. Instruments/xctrace et le compilateur Metal ne sont
  pas disponibles via le Command Line Tools actuellement sélectionné ; aucune
  durée GPU fine ne sera inventée.
- Conditions initiales : Apple M5, 24 Gio, Mac sur batterie (86 %, décharge),
  aucun autre moteur MLX/modèle détecté. Les variations thermiques/fréquence
  seront traitées par répétitions proches, sans additionner des gains issus de
  séries incompatibles. Résultats et classement des dix hotspots à ajouter
  avant toute modification du moteur.
- Statut : **en cours**, diagnostic local uniquement ; aucune optimisation,
  installation, publication ou revendication de gain à ce stade.
- Baseline exécutée : cinq runs chauds 27/256, decode médian **43,719 tok/s**
  (43,508–44,105), prefill chaud médian **167,21 tok/s**, TTFT chaud médian
  **161,97 ms**, hash identique sur les cinq sorties. Le premier run après
  warmup subit encore une compilation/JIT (TTFT 1,472 s). Chargement moteur
  5,421 s, modèle 13,064 GB annoncé ; `/usr/bin/time` ne comptabilise pas
  correctement le footprint unifié du processus enfant et n'est pas retenu
  comme mesure RSS. Preuves `build/perf-31/qwen-baseline.{json,time}`.
- Profil modèle complet : capture CPU `sample` de cinq secondes au milieu
  d'une génération 768 tokens. Le thread principal est dans `Qwen35Moe::
  run_tokens`/`array.eval` pour **2 673/3 082 échantillons (86,7 %)** ; au
  sein de cette phase, 961 échantillons attendent une condition GPU et 780
  construisent/soumettent le graphe. La sélection de token représente 61
  échantillons. Le profil est perturbant et ne donne pas les temps GPU purs.
  Preuves `build/perf-31/qwen-decode-cpu.sample` et
  `qwen-decode-sampled-run.json` (42,234 tok/s sous échantillonnage).
- Classement initial des dix coûts, fondé sur la capture et la structure réelle
  (40 couches = 30 GDN + 10 attention ; 256 experts, top-8), à confirmer pour
  chaque candidat : (1) exécution/attente du graphe GPU complet ; (2) QMV
  experts gate/up/down des 40 MoE ; (3) projections/état GDN des 30 couches ;
  (4) branche experte partagée des 40 MoE ; (5) `lm_head` 248 320 sorties ;
  (6) construction/soumission CPU des nombreux graphes/kernels ; (7) dix
  attentions et croissance KV ; (8) transformées Hadamard/scales/épilogues
  séparés ; (9) allocations/destructions/copies temporaires MLX observées par
  `sample` ; (10) sampler/synchronisation finale par token. Ce classement ne
  prétend pas répartir le temps GPU sans Instruments.
- Limites : batterie 86→83 %, aucune trace GPU fine (`xctrace`/`metal`
  absents du Command Line Tools sélectionné), aucun autre processus MLX.
  Statut : **diagnostic terminé**, aucun gain intégré par PERF-31.

### OPT-2026-09-19-RUST-PERF-32 — Greedy : argmax direct sans log-softmax — en cours

- Profil déclencheur : pendant cinq secondes de decode Qwen complet, le thread
  principal passe 2 673/3 082 échantillons dans l'évaluation MLX ; la sélection
  apparaît 61 fois et matérialise actuellement `log_probs` avant un `argmax`
  GPU. Le vocabulaire compte 248 320 entrées. Le chemin température/repetition
  penalty doit rester strictement inchangé.
- Hypothèse : pour `temperature=0` ou `top_k=1` sans pénalité, `argmax(logits)`
  évite le log-softmax complet. La transformation est monotone mais les égalités
  et non-finis imposent une validation sur le modèle réel ; rejet immédiat si
  un token/hash diffère.
- Baseline : PERF-31, cinq runs chauds 27/256 sur batterie : decode médian
  **43,719 tok/s** (43,508–44,105), prefill chaud **167,21 tok/s** médian hors
  premier JIT, TTFT chaud **161,97 ms** médian, hash de texte constant
  `55b6be28...d6a5bc6`. Chargement moteur 5,42 s, poids résidents annoncés
  13,064 GB. Preuve `build/perf-31/qwen-baseline.json`.
- Protocole candidat : archiver le binaire baseline, modifier uniquement le
  fast path greedy partagé du bridge, rebuild release, contrôler 256 tokens
  greedy/hash, puis A/B/A avec cinq runs de 256 tokens par binaire. Mesurer
  decode, prefill, TTFT et empreinte processus ; ne conserver que si le gain
  dépasse le bruit sans régression de fidélité.
- Statut : **en cours**, code non encore modifié.
- Candidat B1, contrôle A, candidat B2, cinq runs 27/256 chacun : decode
  médian **43,743 / 43,716 / 43,502 tok/s** ; prefill chaud médian
  **167,47 / 165,76 / 161,59 tok/s** ; TTFT chaud **161,41 / 163,22 /
  167,29 ms**. Les quinze générations ont le même hash que la baseline.
  L'empreinte poids annoncée reste 13,064 GB. Preuves
  `build/perf-31/qwen-argmax-{b1,a,b2}.json`.
- Le candidat ne dépasse pas le bruit et le second passage est plus lent avec
  la dérive batterie/thermique. Décision : **rejeté** ; fast path restauré au
  log-softmax de référence, aucun commit/push d'optimisation.

### OPT-2026-09-19-RUST-PERF-33 — Cache de factories Metal par identité de kernel — en cours

- Profil déclencheur : la capture PERF-31 observe, dans chaque token, des
  allocations/libérations de chaînes, comparaisons d'une clé C++ comprenant
  le source Metal complet, et créations répétées de descripteurs de custom
  kernel. La factory MLX est déjà cachée, mais sa recherche recopie et compare
  `name + input names + output names + header + source` à chaque dispatch.
- Hypothèse : les noms de kernels MLXL3 peuvent constituer l'identité de la
  factory si toutes les spécialisations dynamiques y figurent. Une recherche
  par nom court évite les copies/comparaisons du source sans changer le graphe,
  les poids, les dimensions de dispatch ou l'arithmétique GPU.
- Changement prévu : compléter les noms aujourd'hui ambigus (QMV groupé et
  QMM expert segmenté), puis indexer le cache C++ par nom. Aucun nouveau cache,
  thread ou dépendance. Vérifier par revue exhaustive des 11 sites de dispatch,
  tests Metal, parité forcée Qwen et hashes greedy ; benchmark A/B/A 27/256.
- Baseline de contrôle : binaire PERF-31 archivé SHA256 `3c25f59e...53d2cfac`,
  decode médian récent **43,716 tok/s**, prefill chaud **165,76 tok/s**, TTFT
  **163,22 ms** dans `qwen-argmax-a.json`. Conditions batterie, dérive connue.
- Statut : **en cours**, aucune modification appliquée à ce stade.
- Résultats A/B/A, cinq runs chauds 27/256 : candidat B1 **44,229 tok/s**
  decode, **167,84 tok/s** prefill, **161,09 ms** TTFT ; contrôle A
  **43,602 tok/s**, **161,97 tok/s**, **166,95 ms** ; candidat B2
  **43,619 tok/s**, **162,25 tok/s**, **166,66 ms**. Les quinze sorties ont
  le même hash. Le passage B2 ne reproduit pas le +1,4 % apparent de B1 et
  décroît avec la batterie/chauffe. Preuves
  `build/perf-31/qwen-factory-cache-{b1,a,b2}.json`.
- Contrôles : quatre tests Metal physiques réussis et parité Qwen forcée sur
  huit étapes bit-à-bit (`build/perf-31/qwen-factory-cache-parity-8.log`).
- Revue historique tardive : cette hypothèse est la même que
  `OPT-2026-09-14-RUST-PERF-06`, déjà rejetée à **-2,65 %**. La répétition
  n'aurait pas dû être lancée ; le rapprochement exact n'a été retrouvé
  qu'après le benchmark. Le nouveau résultat confirme l'absence de gain.
- Décision : **rejeté**, code restauré, aucun commit/push. Les mesures et cet
  écart au protocole restent consignés pour empêcher une nouvelle répétition.

### OPT-2026-09-19-RUST-PERF-34 — Qwen GDN : convolution causale mono-token — en cours

- Profil déclencheur : Qwen exécute 30 couches Gated DeltaNet par token. Chacune
  construit actuellement un `concatenate` état+QKV, lance une convolution
  depthwise de longueur 4, puis crée un `slice` pour le nouvel état avant le
  SiLU. Le roadmap `docs/decode-roadmap-2026-09-04.md` identifie précisément
  cette séquence comme prochaine fusion utile ; aucun essai historique du
  kernel convolution+état Rust n'a été trouvé.
- Hypothèse : pour `T=1`, un kernel Metal par canal calculant exactement la
  convolution FP16 et le décalage d'état remplace concat+conv+slice par un seul
  dispatch à deux sorties. Le SiLU MLX reste séparé au premier essai afin de
  réduire le risque numérique. Le chemin multi-token/prefill reste inchangé.
- Baseline : PERF-31, Qwen 2,49 bpw, prompt 27 / génération 256, decode médian
  **43,719 tok/s**, prefill chaud **167,21 tok/s**, TTFT chaud **161,97 ms**,
  hash `55b6be28...d6a5bc6`. Batterie et chauffe variables ; comparaison A/B/A
  de cinq runs avec binaires archivés et sortie identique requise.
- Protocole : test physique kernel contre `concatenate+conv1d` sur formes
  déterministes, `native/check_qwen_gdn.py` sur deux états réels, parité modèle
  forcée huit étapes incluant tous les caches, puis benchmark A/B/A. Rejet si
  un bit des sorties/états réels diverge ou si le gain ne se reproduit pas.
- Statut : **en cours**, aucun code candidat appliqué à ce stade.
- Contrôles du candidat : kernel synthétique exact contre MLX, GDN réelle
  couche 0 exacte sur deux états, puis modèle complet exact sur huit tokens
  imposés, logits et caches bit-à-bit. Preuves
  `build/perf-31/qwen-causal-conv-{gdn,parity-8}.log`.
- A/B/A, cinq runs 27/256 : candidat B1 **43,831 tok/s** decode,
  **168,02 tok/s** prefill, **160,98 ms** TTFT ; contrôle A
  **43,427 tok/s**, **163,49 tok/s**, **167,34 ms** ; candidat B2
  **43,164 tok/s**, **165,85 tok/s**, **163,66 ms**. Les quinze sorties ont
  le hash de référence. Batterie 78→76 %, dérive thermique visible.
- Le +0,9 % apparent face au contrôle n'est pas reproduit par B2, qui termine
  sous A et sous la baseline initiale. La réduction des opérations du graphe
  n'abaisse donc pas le temps modèle mesurable dans ces conditions.
- Décision : **rejeté**, code et test candidat retirés, aucun commit/push.
  Preuves `build/perf-31/qwen-causal-conv-{b1,a,b2}.json`.

### OPT-2026-09-19-RUST-PERF-35 — Splash/Inco et adressage incrémental QMV — rejeté

- Recherche déclenchée par l'utilisateur : analyse du moteur Apple Silicon
  Splash/Inco (dépôt Apache-2.0 et billet technique, état du 17 septembre
  2026). Son avantage annoncé repose notamment sur DFlash2/speculative decode,
  exclu de cette campagne car l'utilisateur exige un décodage exact sans
  spéculation. Les idées transférables sans perte sont : plans Metal fixes par
  forme, poids déjà disposés pour leur consommateur, arènes préallouées, et
  fusion complète du Gated DeltaNet. Leur kernel GDN fusionne convolution,
  SiLU, normalisations q/k, gates, récurrence et normalisation/gate de sortie ;
  cela explique pourquoi le seul sous-kernel convolution de PERF-34 n'a pas
  produit de gain modèle. Le format Q4 privé et le command graph Splash ne sont
  pas directement réutilisables par un moteur EXL3 construit sur les graphes
  MLX, et une réécriture de runtime n'est pas engagée silencieusement.
- Premier candidat général et réversible avant la fusion GDN : dans les QMV
  dense et expert mappé, calculer la base de la tuile K une seule fois par
  itération puis l'avancer d'un stride constant. Le kernel actuel répète
  `(tile_k * TILES_N + tile_n) * PACKED_U32` pour chaque tuile de sortie ; le
  changement conserve codewords, ordre des FMA, réduction, dispatch et poids.
  Aucun repacking, nouvelle copie ni allocation.
- Baseline : binaire archivé PERF-31, Qwen3.6-35B-A3B 2,49 bpw, prompt 27 et
  génération 256 : decode médian initial **43,719 tok/s**. La batterie et la
  température dérivent ; décision uniquement sur une alternance B/A/B récente
  d'au moins cinq runs, texte/hash identique, plus tests QMV et parité modèle.
- Protocole : vérifier d'abord les kernels Metal physiques et huit tokens Qwen
  forcés, puis B/A/B. Rejeter si l'écart ne se reproduit pas ou reste dans le
  bruit. Si rejeté, passer à la fusion GDN complète indiquée par Splash, sans
  conserver ce micro-changement.
- Statut : **en cours**, journalisé avant modification ; aucune publication ni
  revendication de gain.
- Le candidat passe la parité Qwen complète sur huit étapes bit-à-bit. Mesures
  longues 27/512, cinq runs par passage : candidat B1 **41,410 tok/s**,
  contrôle A1 **40,841 tok/s**, candidat B2 **41,175 tok/s**, puis contrôle A2
  **42,533 tok/s**. Les vingt sorties ont le même hash ; pic MLX identique
  13,064 GB. Le prefill chaud est respectivement **156,79 / 157,22 / 153,92 /
  162,57 tok/s** et ne montre aucun gain. Batterie 76→70 %, dérive thermique
  et énergétique importante. Preuves `build/perf-31/qwen-qmv-address-*.json`
  et `qwen-qmv-address-parity-8.log`.
- Le dernier contrôle dépasse le candidat de 3,3 % : les +1,39/+0,82 % vus
  autour d'A1 ne sont pas reproductibles et ne peuvent pas être attribués au
  code. Décision : **rejeté**, shaders restaurés, aucun commit/push.

### OPT-2026-09-19-RUST-PERF-36 — Qwen GDN : invariants persistants — rejeté

- Profil et inspiration Splash : la part CPU de PERF-31 montre beaucoup de
  construction/destruction de graphes et d'arrays ; Splash alloue et prépare
  ses ressources fixes au chargement. Dans chaque appel mono-token de chacune
  des 30 couches GDN Qwen, MLXL3 recrée actuellement trois scalaires MLX
  (`1/Dk`, `1/sqrt(Dk)`, zéro) et reconstruit `exp(A_log)` alors que ces quatre
  valeurs sont invariantes pour toute la vie du modèle.
- Hypothèse : matérialiser une seule fois au chargement les trois scalaires et
  `-exp(A_log)`, puis les réutiliser, supprime 90 petites allocations/objets de
  graphe par token et le calcul invariant des 30 vecteurs A. Le signe est
  déplacé avant la multiplication, opération exactement équivalente sur ces
  poids finis ; aucun ordre de réduction, poids, cache ou kernel EXL3 ne change.
- Protocole : parité GDN deux états puis modèle Qwen huit étapes bit-à-bit ;
  benchmark B/A/B 27/512 avec cinq runs et contrôle archivé. Mesurer decode,
  prefill, TTFT et pic. Rejeter si le gain ne se reproduit pas ; ce candidat
  reste distinct de la future fusion Metal GDN.
- Baseline récente pertinente : contrôle A2 PERF-35 **42,533 tok/s** decode,
  prefill chaud **162,57 tok/s**, TTFT **168,20 ms**, pic 13,064 GB, batterie
  70 % en décharge. Statut : **en cours**, aucune modification appliquée avant
  cette entrée.
- Parité : GDN couche 0 sur deux états, puis modèle complet huit étapes,
  sorties et caches **bit-à-bit**. Mesures B/A/B 27/512, cinq runs : decode
  **49,696 / 48,563 / 48,320 tok/s** ; prefill chaud **187,34 / 186,82 /
  186,47 tok/s** ; TTFT **144,68 / 146,42 / 146,40 ms** ; pic identique
  13,064 GB et hash identique. Batterie 70→66 %, forte remontée globale de
  fréquence par rapport à PERF-35 puis dérive dans la série. Preuves
  `build/perf-31/qwen-gdn-invariants-{b1,a,b2}.json`, `*-gdn.log` et
  `*-parity-8.log`.
- B1 paraît +2,33 %, mais B2 est **−0,50 %** face au contrôle central : le gain
  n'est pas reproduit. Décision : **rejeté**, code restauré ; les allocations
  minuscules/invariants sont vraisemblablement masqués ou déjà fusionnés par
  MLX. Aucun commit/push.

### OPT-2026-09-19-RUST-PERF-37 — Qwen GDN : gates Metal fusionnées — rejeté

- Étape minimale vers la fusion complète observée dans Splash : remplacer les
  deux graphes élémentaires séparés `sigmoid(b)` et
  `exp(-exp(A_log) * softplus(a + dt_bias))` par un kernel Metal mono-token à
  deux sorties (`beta` FP16 et decay FP32). Les projections, la convolution,
  les normalisations, la récurrence et les caches restent inchangés.
- Hypothèse : 30 couches GDN par token peuvent éviter au moins une frontière de
  dispatch/compilation chacune. Le calcul doit reproduire les arrondis FP16 de
  `a + dt_bias` puis `logaddexp`; aucune approximation `fast`, table ou
  spéculation. Le chemin multi-token garde le graphe MLX actuel.
- Protocole : test GDN réel deux états exigeant sortie et caches bit-à-bit,
  puis modèle Qwen huit étapes bit-à-bit. Si la formule MSL ne reproduit pas la
  référence, rejet immédiat ou correction avant tout benchmark. Si exacte,
  B/A/B 27/512 cinq runs ; pic, TTFT et prefill suivis.
- Baseline de proximité : contrôle PERF-36 **48,563 tok/s**, prefill chaud
  **186,82 tok/s**, TTFT **146,42 ms**, pic 13,064 GB ; batterie 66 % en
  décharge. Statut : **en cours**, journalisé avant code.
- La première formule MSL a été rejetée avant benchmark : le `sigmoid` calculé
  en FP32 différait d'un ULP FP16 sur plusieurs entrées et faisait diverger la
  GDN réelle. La correction reproduit l'opérateur MLX : exponentielle et
  branche stable en `half`, ainsi que son `logaddexp` et son `log1p` compensé.
  Le test GPU synthétique passe alors bit-à-bit, de même que la GDN réelle sur
  deux états et le modèle sur huit étapes.
- B/A/B 27/512, cinq runs : decode **44,673 / 44,187 / 44,122 tok/s** ; prefill
  chaud **168,75 / 166,32 / 166,74 tok/s** ; TTFT **164,56 / 162,92 /
  164,90 ms** ; pic identique 13,064 GB et tous les hashes identiques. Batterie
  66→61 %. Preuves `build/perf-31/qwen-gdn-gates-{b1,a,b2}.json`,
  `qwen-gdn-gates-gdn.log`, `qwen-gdn-gates-parity-8.log`.
- B1 est +1,10 %, mais B2 est **−0,15 %** et le TTFT n'est pas amélioré : un
  dispatch Metal supplémentaire remplace des expressions que MLX fusionne déjà
  efficacement. Décision : **rejeté**, kernel/test retirés, aucun commit/push.

### OPT-2026-09-19-RUST-PERF-38 — Qwen GDN : gates intégrées à la récurrence — validé

- Suite directe de PERF-37 : ne plus produire `g` et `beta` dans un kernel
  séparé. Pour le decode `T=1`, le kernel récurrent existant reçoit directement
  `a`, `b`, `A_log` et `dt_bias` et calcule les deux scalaires du head avant la
  mise à jour d'état. La formule half/FP32 exacte validée dans PERF-37 est
  réutilisée ; la géométrie et l'ordre des FMA récurrentes restent inchangés.
- Hypothèse : supprimer réellement le dispatch et les deux buffers
  intermédiaires sur chacune des 30 couches peut rendre visible le bénéfice que
  PERF-37 masquait. Le calcul des deux scalaires est répété par row-group, coût
  minuscule face aux 16 384 valeurs d'état du head et sans synchronisation
  inter-groupe.
- Protocole : test dédié des gates contre le graphe MLX, GDN deux états et
  modèle huit étapes bit-à-bit, puis B/A/B 27/512 cinq runs. Le prefill `T>1`
  garde le chemin de référence. Statut : **en cours**, aucune revendication.
- Contrôles numériques initiaux réussis : test GPU du kernel intégré contre le
  graphe de gates + kernel récurrent, GDN réelle couche 0 sur deux états, puis
  modèle Qwen complet huit étapes ; sorties et caches bit-à-bit. Le chemin
  trace et le prefill multi-token continuent explicitement d'utiliser les
  opérations MLX de référence.
- B/A/B 27/512, cinq runs : decode **44,698 / 43,972 / 44,576 tok/s**, soit
  **+1,65 % / +1,37 %** face au contrôle central ; prefill chaud **169,42 /
  163,18 / 168,05 tok/s** ; TTFT **161,04 / 166,76 / 162,76 ms** ; pic identique
  **13,064 GB**, tous les hashes identiques. Batterie 61→57 % en décharge ; les
  deux passages candidat encadrant le contrôle reproduisent le gain malgré la
  dérive. Preuves `build/perf-31/qwen-gdn-integrated-{b1,a,b2}.json`,
  `qwen-gdn-integrated-{gdn,parity-8}.log`.
- Décision performance : **validé** pour Qwen GDN decode. Vérifications
  complètes : test GPU dédié, GDN réelle sur deux états et modèle Qwen huit
  étapes bit-à-bit ; 190 cas de parité MLX exacts ; 35 tests Rust réussis
  (12 tests matériels explicitement ignorés par la suite standard) ; format,
  Clippy strict et `git diff --check` réussis. Kani 0.68 / CBMC 6.11 vérifie
  **14/14 harnesses**, zéro échec, sur le cœur Rust sans features MLX. Kani ne
  couvre pas Metal, MLX ni leur FFI : ces chemins sont validés par les tests
  différentiels physiques précédents, pas formellement prouvés. Preuve Kani
  `build/perf-31/qwen-gdn-integrated-kani.log`. Intégration : code inclus dans
  le commit d'optimisation dédié et destiné à `main` ; application installée et
  release non modifiées dans cet essai.

### OPT-2026-09-19-RUST-PERF-39 — Ling KDA : beta intégré à la récurrence — validé

- Suite générale de PERF-38 et de l'analyse Splash/Inco : Ling matérialise
  encore le vecteur de decay FP32 et le beta FP16 avant son kernel récurrent
  vectoriel. Le candidat transmet au même kernel les sorties brutes déjà
  calculées et les constantes `A_log`, `dt_bias` et `lower_bound`, puis y
  reproduit exactement les transformations élémentaires. Les projections,
  poids EXL3, caches, ordre des FMA et sampling restent inchangés.
- Hypothèse : sur les couches KDA, retirer les graphes/buffers intermédiaires
  de sigmoid, cast et decay réduit les dispatches de decode. Le matmul de beta
  reste nécessaire et n'est pas fusionné. Le prefill multi-token conserve le
  chemin de référence tant que l'égalité numérique n'est pas établie.
- Baseline : binaire `b7de3f8` archivé avant modification, modèle
  `Ling-3.0-tiny-EXL3-4bpw`. Protocole prévu : sortie et états Ling bit-à-bit
  sur tokens imposés, test GPU synthétique, puis A/B/A d'au moins cinq runs
  sur la même longueur ; decode, prefill, TTFT, pic et hashes suivis. Rejet à
  la moindre divergence inexpliquée ou si le gain ne se reproduit pas.
- Conditions : Apple M5 24 Go, sur batterie et température dérivante ; niveau
  de batterie et alternance consignés. Statut : **en cours**, journalisé avant
  baseline, modification et benchmark ; aucune publication.
- Revue avant modification : contrairement à Qwen, le decay Ling contient 128
  valeurs par head et est partagé par 128 lignes d'état. Le recalculer dans la
  géométrie actuelle répéterait ses exponentielles 32 à 128 fois. Pour éviter
  cette régression prévisible, ce premier candidat n'intègre que le sigmoid et
  cast FP16 de beta, scalaire par head ; le decay reste matérialisé une fois.
  Une fusion decay correcte nécessitera une géométrie/threadgroup distincte et
  fera l'objet d'un essai séparé. Baseline A mesurée : **97,672 tok/s** decode,
  prefill chaud médian **100,080 tok/s**, TTFT médian **839,5 ms**, pic déclaré
  4,428 GB, hashes identiques ; batterie 55 %. Preuve
  `build/perf-39/ling-gates-control-a.json`.
- Candidat : le kernel vectoriel reçoit les logits beta FP32, reproduit le
  sigmoid MLX puis son cast FP16 avant la récurrence. Test GPU synthétique :
  sortie FP16 et état FP32 bit-à-bit contre le graphe de référence. Sur huit
  tokens imposés du modèle réel, logits complets baseline/candidat identiques,
  SHA256 commun
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`.
- Alternance A/B/A/B, cinq runs 84/128 : decode **97,672 / 102,008 / 100,373 /
  101,697 tok/s**. Les deux candidats sont à **+4,44 % / +1,32 %** face au
  contrôle précédent et restent groupés autour de 102 tok/s. Prefill chaud
  **100,079 / 104,652 / 102,916 / 104,589 tok/s** ; TTFT **839,53 / 802,88 /
  816,43 / 803,41 ms**. Pic déclaré identique 4,428371 GB et hash de génération
  identique sur les vingt runs. Batterie 55→49 %, en décharge. Preuves
  `build/perf-39/ling-gates-{control-a,candidate-b1,control-a2,candidate-b2}.json`.
- Décision : **validé** pour le decode Ling. Suite : 20 tests lib, 1 test CLI
  et 14 tests contractuels réussis ; 13 tests matériels ignorés par la suite
  standard, dont le nouveau test exécuté séparément avec succès. Format,
  Clippy strict et `git diff --check` réussis. Kani 0.68 / CBMC 6.11 : **14/14
  harnesses**, zéro échec, cœur Rust sans features MLX. Kani ne vérifie pas le
  shader, MLX ou leur FFI ; leur contrôle est différentiel sur GPU physique.
  Preuve `build/perf-39/ling-gates-kani.log`. Intégration : commit dédié pour
  `main`, sans rebuild de l'app ni release dans cet essai.

### OPT-2026-09-19-RUST-PERF-40 — Ling KDA : bloc récurrent partagé/fusionné — rejeté

- Suite de PERF-39 et de l'architecture Splash/Inco : la géométrie actuelle
  relit q/k pour chacune des 128 lignes d'état. Le candidat decode `T=1`
  charge q/k une fois en mémoire threadgroup, calcule les 128 decay une fois
  par groupe à partir du gate brut, puis chaque thread traite plusieurs lignes
  avec le même ordre FP32 de réduction et de mise à jour. Beta brut reste
  intégré comme dans PERF-39. Huit groupes par head sont prévus afin de garder
  de l'occupation sans répéter 128 fois les transcendantes.
- `exp(A_log)` sera matérialisé une fois au chargement et réutilisé. Le chemin
  multi-token reste inchangé. Aucun poids, quantification, cache, sampler ou
  approximation mathématique n'est modifié ; les opérateurs Metal doivent
  reproduire exactement sigmoid/exp MLX.
- Baseline : commit `4c87e3d`, à archiver avant modification ; dernier A/B/A/B
  PERF-39 candidat **101,697–102,008 tok/s** decode, prefill chaud
  **104,589–104,652 tok/s**, TTFT **802,88–803,41 ms**, pic 4,428371 GB.
- Protocole : test GPU synthétique sortie/état bit-à-bit, huit tokens Ling
  imposés contre le binaire archivé, puis A/B/A/B 84/128 à cinq runs. Rejet
  immédiat pour divergence ou gain non reproduit. Apple M5 24 Go sur batterie,
  température non instrumentée. Statut : **en cours**, journalisé avant code ;
  aucune publication.
- Premier prototype full-decay : test nul puis état non nul synthétique
  réussis, mais le modèle réel diverge dans l'état récurrent dès le deuxième
  token (couche 0, 4 578 floats, erreur max 3,73e-9), puis dans les logits au
  troisième token. Cause : les élémentaires du decay fusionné ne reproduisent
  pas tous les arrondis des kernels MLX séparés. Décision : variante
  **rejetée avant benchmark** malgré son faible écart ; les transcendantes et
  le `A_log` pré-évalué sont retirés.
- Variante corrigée en cours : conserver le decay MLX exact, mais utiliser la
  nouvelle géométrie partagée pour charger q/k/decay huit fois par head au lieu
  de 128, tout en gardant beta fusionné. Ce changement isolera le gain mémoire
  de la réorganisation sans modifier le calcul des gates.
- La variante corrigée est bit-à-bit exacte sur le test GPU avec état non nul
  et sur huit tokens réels (SHA256 logits identique
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`).
  Mesures B/A/B, cinq runs : decode **98,970 / 99,972 / 99,908 tok/s**,
  prefill chaud **99,522 / 103,228 / 101,841 tok/s**, TTFT **844,31 / 813,92 /
  825,03 ms**. Un run candidat à 89,50 tok/s est un outlier, mais même sans
  lui le second candidat reste au niveau du contrôle, pas au-dessus. Batterie
  43 %, en décharge. Preuves `build/perf-40/ling-shared-*.json`.
- Décision finale : **rejeté**. La baisse des lectures q/k ne compense pas la
  perte d'occupation due aux boucles de lignes ; code partagé et test retirés.
  PERF-39 reste le chemin production. Aucun commit/push de ce prototype.

### OPT-2026-09-19-RUST-PERF-41 — Ling KDA : pré-évaluation de A — rejeté

- Variante minimale issue du plan fixe Splash, distincte du full-decay rejeté
  en PERF-40 : calculer `exp(A_log)` une fois au chargement, puis fournir
  exactement cet array FP32 au graphe MLX de decay inchangé. Aucun élémentaire
  sigmoid/exp n'est déplacé dans Metal et la récurrence reste celle de PERF-39.
- PERF-36 a déjà rejeté une idée voisine sur Qwen ; Ling diffère car son decay
  vectoriel est construit dans chaque couche KDA. Ce nouvel essai ne sera gardé
  que s'il est bit-à-bit et reproduit un gain A/B/A ou B/A/B.
- Baseline : binaire production `4c87e3d`, mesures récentes PERF-40 contrôle
  **99,972 tok/s** decode, **103,228 tok/s** prefill chaud, **813,92 ms** TTFT.
  Protocole 84/128 cinq runs, huit tokens imposés, pic et hashes suivis. Apple
  M5 24 Go sur batterie 43 %. Statut : **en cours**, journalisé avant code.
- Parité : huit tokens imposés bit-à-bit, hash commun
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`.
  B/A/B cinq runs : decode **98,814 / 102,873 / 97,558 tok/s**, prefill chaud
  **100,934 / 105,524 / 99,210 tok/s**, TTFT **832,43 / 796,23 / 847,01 ms**,
  pic identique 4,428371 GB. Batterie 41 %. Preuves
  `build/perf-41/ling-ascale-*.json`.
- Décision : **rejeté** et code retiré. L'évaluation anticipée brise la fusion
  paresseuse de la chaîne élémentaire MLX et ajoute une lecture intermédiaire ;
  la suppression d'un `exp` apparent régresse de 3,9 à 5,2 %. Aucun push.

### OPT-2026-09-19-RUST-PERF-42 — Qwen GDN : gates partagées par threadgroup — rejeté

- Suite de PERF-38 : dans le kernel récurrent Qwen `T=1`, chaque thread
  recalcule actuellement le même sigmoid beta et le même decay scalaire du
  head. Avec 64 threads par groupe et huit groupes par head, cela répète les
  transcendantes 512 fois. Le candidat les calcule une fois par threadgroup,
  les place dans 6 octets de mémoire partagée puis synchronise avant la boucle
  récurrente. Équations, arrondis et ordre des FMA restent identiques.
- Hypothèse : le coût d'une barrière est inférieur aux exponentielles répétées.
  Le chemin non fusionné et le prefill restent inchangés. Baseline production
  commit `4c87e3d`, binaire archivé
  `build/perf-40/ling-beta-fused-baseline-bin`; mesures fraîches à faire car
  batterie 41 % et température non instrumentée.
- Protocole : test kernel/GDN et huit tokens Qwen bit-à-bit, puis B/A/B 27/512
  cinq runs ; decode, prefill, TTFT, pic et hashes. Rejet si la barrière
  régresse ou si le gain ne se reproduit pas. Statut : **en cours**, journalisé
  avant modification ; aucune publication.
- Parité GPU et huit tokens Qwen bit-à-bit, hash commun
  `c06b13baecf5d4b0eb189418e974f63a5259a8732a16d29badb47b85dd69d373`.
  B/A/B cinq runs : decode **45,721 / 46,973 / 44,945 tok/s**, prefill chaud
  **173,488 / 181,680 / 172,943 tok/s**, TTFT **155,85 / 148,83 / 156,36 ms**,
  pic identique 13,064372 GB. Batterie 39→36 %. Preuves
  `build/perf-42/qwen-shared-gates-*.json`.
- Décision : **rejeté**, shader restauré. La barrière threadgroup coûte plus
  que les calculs redondants sur M5 et dégrade aussi TTFT/prefill. Aucun push.

### OPT-2026-09-19-RUST-PERF-43 — Qwen GDN : gates diffusées dans le SIMDgroup — rejeté

- Révision motivée par PERF-42 : supprimer la barrière coûteuse. Seule la lane
  0 de chacun des deux SIMDgroups calcule beta/decay, puis
  `simd_broadcast_first` diffuse leurs bits aux 31 autres lanes. Cela réduit
  les transcendantes 32×, sans mémoire threadgroup ni synchronisation entre
  SIMDgroups ; chaque groupe traite déjà des lignes indépendantes avec les
  mêmes gates.
- Baseline production `4c87e3d`, contrôle frais PERF-42 **46,973 tok/s**,
  prefill **181,680 tok/s**, TTFT **148,83 ms**, pic 13,064372 GB ; batterie
  36 %, dérive forte donc B/A/B obligatoire. Parité kernel/GDN et huit tokens
  Qwen bit-à-bit avant benchmark. Statut : **en cours**, journalisé avant code.
- Parité kernel et huit tokens bit-à-bit, hash commun
  `c06b13baecf5d4b0eb189418e974f63a5259a8732a16d29badb47b85dd69d373`.
  B/A/B cinq runs : decode **45,767 / 45,910 / 44,306 tok/s**, prefill chaud
  **178,343 / 178,054 / 174,429 tok/s**, TTFT **151,59 / 151,85 / 155,01 ms**,
  pic identique 13,064372 GB. Batterie 32 % puis secteur reconnecté en fin de
  série ; forte dérive sur B2. Preuves `build/perf-43/qwen-simd-gates-*.json`.
- Décision : **rejeté**, shader restauré. B1 est dans le bruit du contrôle et
  B2 régresse ; même sans barrière, masquer les transcendantes aux lanes
  inactives n'apporte pas de gain modèle. Aucun push.

### OPT-2026-09-19-RUST-PERF-44 — Ling KDA : boucle de lignes sans barrière — rejeté

- Révision de PERF-40 selon les résultats PERF-42/43 : garder le decay MLX
  exact et supprimer toute mémoire/barrière threadgroup. Pour `T=1`, chaque
  SIMDgroup charge q/k/decay une fois dans ses registres puis traite plusieurs
  lignes d'état. Beta est calculé par lane 0 et diffusé avec
  `simd_broadcast_first`. La grille conserve huit threadgroups par head, soit
  assez d'occupation, et réduit les lectures q/k/decay de 4×.
- Baseline production `4c87e3d`, dernier contrôle Ling **102,873 tok/s** decode,
  **105,524 tok/s** prefill chaud, **796,23 ms** TTFT ; secteur maintenant
  attaché, batterie 32 % non chargée. Test état non nul et huit tokens exacts,
  puis B/A/B cinq runs. Statut : **en cours**, journalisé avant code.
- Test GPU état non nul et huit tokens réels bit-à-bit, hash commun
  `7bfec320150e5c307ecf2fec425fd4a22e06035c91cbaa68de8a193a400d7446`.
  B/A/B cinq runs : decode **100,715 / 102,305 / 98,984 tok/s**, prefill chaud
  **102,347 / 106,027 / 100,027 tok/s**, TTFT **820,96 / 792,45 / 839,98 ms**,
  pic identique 4,428371 GB. Secteur attaché puis batterie en charge 32→36 %.
  Preuves `build/perf-44/ling-looped-*.json`.
- Décision : **rejeté**, code retiré. Réduire les threadgroups et boucler les
  lignes perd davantage en occupation qu'il ne gagne en lectures répétées.
  PERF-39 reste la meilleure géométrie Ling. Aucun push du prototype.

### Clôture de la passe Splash/Inco — 2026-09-19

- Conservé et publié : PERF-38, gates Qwen intégrées au kernel récurrent,
  commit `b7de3f8`, **+1,37 à +1,65 % decode** ; PERF-39, beta Ling intégré,
  commit `4c87e3d`, **+1,32 à +4,44 % decode**. Sorties forcées exactes et pic
  déclaré inchangé pour les deux.
- Rejeté et absent du code final : adressage QMV, invariants pré-évalués,
  kernel gates séparé, decay Ling fusionné, géométries partagées/bouclées et
  diffusion Qwen. Les mesures PERF-35–44 empêchent de les retester sans
  nouvelle géométrie ou nouveau matériel.
- Le test PERF-39 est renforcé avec un état récurrent non nul. Vérification
  finale : 20 tests lib, 1 test CLI et 14 contrats réussis ; test GPU ciblé
  réussi ; format, Clippy strict et `git diff --check` réussis. Kani 0.68 /
  CBMC 6.11 : **14/14 harnesses**, zéro échec, cœur Rust sans features MLX.
  Kani ne couvre toujours ni MLX, ni Metal, ni leur FFI ; les kernels sont
  contrôlés par différentiel physique, pas formellement prouvés. Preuve
  `build/perf-44/final-kani.log`.

### OPT-2026-09-19-RUST-PERF-45 — DFlash 2 Qwen3.6-35B-A3B — en cours

- Demande : intégrer le drafter DFlash 2 officiel à la cible locale
  `Qwen3.6-35B-A3B-EXL3-2.49bpw`, sans modifier la distribution du modèle
  cible, puis viser **100 tok/s** en decode. Le package Apache-2.0
  `incoai/Qwen3.6-35B-A3B-Splash` contient un drafter Q4 de six couches
  (capture des couches cible 1/6/11/16/22/27/32/37), sept propositions et une
  vérification cible de huit lignes.
- Baseline production pertinente : PERF-42/43 mesure le moteur Qwen actuel à
  **44,3–47,0 tok/s** decode, **172,9–181,7 tok/s** prefill chaud et
  **148,8–156,4 ms** TTFT sur Apple M5 24 Go, avec forte dérive batterie et
  thermique. Une nouvelle baseline secteur/thermique stabilisée sera mesurée
  avant toute revendication DFlash.
- Blocages confirmés avant code : le chemin EXL3 natif accepte aujourd'hui
  seulement `M=1` ou `M>=24`, tandis que DFlash vérifie `M=8`; Qwen ne calcule
  le `lm_head` que sur la dernière ligne et fait avancer immédiatement ses 30
  états GDN/convolution et ses 10 caches KV. Le drafter officiel est fourni en
  format fixe `splash-packed-q4-moe`, pas en EXL3/safetensors.
- Plan d'essais indépendants : (1) contrôleur d'acceptation exact et borné,
  vérifié par tests/Kani ; (2) QMM EXL3 `M=8` et logits cible par ligne, avec
  parité autoregressive ; (3) états Qwen transactionnels commit/rollback ;
  (4) chargeur et exécution du drafter Q4 officiel ; (5) boucle greedy exacte,
  puis rejection sampling exact ; (6) A/B/A long avec taux d'acceptation,
  coût draft/verify/commit, decode, prefill, TTFT, pic et hash de sortie.
- Garde-fous : chaque jalon doit rester inactif ou revenir au decode normal si
  le drafter manque ; aucune mesure Splash M5 Pro n'est reprise comme résultat
  MLXL3. Le jalon est rejeté à la moindre divergence greedy, distribution
  sampling invalide, corruption d'état ou régression stable. Statut initial :
  **en cours**, aucun code DFlash publié et aucun gain revendiqué.
- Jalon 1 : contrôleur greedy exact ajouté, sans branchement au moteur. Il
  accepte uniquement le plus long préfixe identique aux choix de la cible et
  termine toujours le cycle par le token cible suivant ; les longueurs
  incohérentes sont rejetées. Test Rust ciblé réussi, Clippy strict réussi.
  Kani 0.68 / CBMC 6.11 vérifie exhaustivement les longueurs 0 à 7 et des
  tokens `u32` symboliques : préfixe maximal, borne d'acceptation et token
  final cible, **216 propriétés réussies**, 2/2 couvertures atteintes, aucun
  échec (une branche standard inaccessible). Ce contrôle ne prouve ni Metal,
  ni les logits, ni les caches ; ceux-ci restent à implémenter et valider.

### OPT-2026-09-19-RUST-PERF-46 — Qwen : transaction d'état DFlash — en cours

- Hypothèse : les arrays MLX sont fonctionnels et leur clonage duplique le
  handle, pas les données. Une transaction légère peut donc capturer l'offset,
  les 30 couples convolution/récurrence GDN et les 10 couples KV, puis les
  restaurer sans copie GPU. C'est requis pour rejeter un suffixe spéculatif
  sans laisser le modèle cible dans un état futur invalide.
- Changement prévu : un snapshot opaque et typé, avec validation stricte du
  nombre/type de couches à la restauration. Aucun branchement au decode normal
  et aucune allocation de tenseur supplémentaire hors clonage de handles.
- Baseline fonctionnelle : après un préfixe fixé, `snapshot → tokens d'essai →
  restore → mêmes tokens` doit produire exactement les mêmes logits et états
  que le premier passage ; un snapshot d'un autre modèle/état doit être rejeté.
  Baseline performance : decode Qwen PERF-42/43 **44,3–47,0 tok/s** ; ce jalon
  ne doit pas modifier ce chemin et aucun gain de decode n'est revendiqué.
- Protocole : test réel Qwen sur quelques tokens avec comparaison bit-à-bit des
  logits et de chaque état restauré ; tests négatifs de structure ; build,
  Clippy strict et Kani sur les invariants Rust qui n'appellent pas MLX. Le coût
  snapshot/restore sera mesuré séparément avant intégration. Statut : **en
  cours**, journalisé avant code ; aucune publication.
- Résultat : **validé et intégré comme primitive inactive**. Sur le vrai
  Qwen3.6-35B-A3B EXL3 2.49 bpw, après le préfixe `[1,2,3]`, une branche
  `[4,5]`, un rollback puis le rejeu produisent exactement les mêmes logits
  FP16 aux deux pas et les mêmes octets pour les **80 tenseurs** conv/GDN/KV.
  Le test GPU ciblé réussit en 11,97 s, chargement du modèle compris.
- Vérification : format et Clippy strict réussis ; tests Rust complets et E2E
  Desktop réussis lors du jalon de nettoyage adjacent. Kani 0.68 / CBMC 6.11
  vérifie 15/15 harnesses, zéro échec (`build/dflash-cleanup-kani.log`), mais
  ne compile pas le feature MLX : le clonage de handles, Metal et ce rollback
  sont donc validés par différentiel physique, pas formellement prouvés.
- Le chemin autoregressif existant n'appelle jamais `snapshot`/`restore` : coût
  normal nul. La mesure microsecondes et le coût sous spéculation seront faits
  avec le contrôleur complet, afin de ne pas présenter un timing isolé comme
  un gain de decode. Publication prévue dans le jalon DFlash suivant.

### OPT-2026-09-19-RUST-PERF-47 — EXL3 : vérification cible `M=8` — en cours

- Hypothèse : le QMM TensorOps EXL3 existant, aujourd'hui réservé à `M>=24`,
  peut traiter les huit lignes de vérification DFlash en un graphe. Le padding
  à 32 lignes gaspille 75 % du calcul, mais évite d'abord un nouveau décodeur
  de poids et fournit une référence mesurable avant d'écrire un QMV batch.
- Baseline fonctionnelle : sur une projection réelle du Qwen3.6-35B-A3B
  EXL3 2.49 bpw, comparer `forward(M=8)` aux huit appels `M=1` : dimensions,
  valeurs FP16, argmax et timing chaud. Le contrôle final exigera aussi les
  mêmes tokens et le même état en exécution autoregressive complète.
- Protocole : autoriser temporairement `M=8` dans le QMM, ajouter un test GPU
  ignoré et mesurer plusieurs répétitions après warmup. **Rejet immédiat** si
  le QMM n'est pas plus rapide ou si l'écart numérique change les tokens ; si
  l'écart FP16 existe sans changer les tokens, il restera uniquement une
  référence de performance et ne sera pas intégré au chemin lossless.
- Baseline modèle : PERF-42/43, **44,3–47,0 tok/s** decode, **172,9–181,7
  tok/s** prefill et **148,8–156,4 ms** TTFT, avec dérive batterie/thermique.
  Statut : **en cours**, journalisé avant code ; aucune publication.
- Sous-essai TensorOps sur `layers.0.linear_attn.in_proj_qkv`, entrée
  déterministe `[8,2048]`, deux warmups puis cinq répétitions : huit QMV
  série **2,055 ms**, QMM paddé à 32 **0,680 ms**, soit **3,02×** sur cette
  projection. Mais **555/65 536** sorties FP16 diffèrent, écart absolu maximal
  **0,0014648438**. Le QMM est donc **rejeté comme vérificateur lossless** ; il
  reste une référence de plafond et n'est pas branché au chemin production.
- Prochain sous-essai : QMV `M<=8` à lignes parallèles, même shader, même
  partition de K et même réduction par ligne afin de viser la parité bit-à-bit.
  Mesurer projection puis tokens/états complets avant intégration. Statut global
  PERF-47 : **en cours**.
- Sous-essai QMV exact sur la même projection et la même entrée, deux warmups
  puis cinq répétitions : batch **0,751 ms**, référence huit branches QMV
  **1,478 ms**, soit **1,97×** dans ce microbenchmark, avec **0/65 536** valeur
  FP16 différente et écart maximal nul. Le résultat peut inclure du cache et
  l'ordonnancement MLX ; ce n'est pas encore un gain modèle.
- Étape suivante journalisée avant code : exposer les huit lignes de logits du
  passage Qwen déjà vectorisé, puis comparer batch contre huit pas séquentiels,
  logits FP16 et 80 tenseurs d'état compris. Un écart du GDN/SDPA invalidera le
  chemin comme vérification exacte ou imposera un kernel séquentiel équivalent.
- Résultat modèle du batch vectorisé : **rejeté**. Après le même préfixe de
  trois tokens, huit pas séquentiels prennent **162,503 ms**, contre **425,245
  ms** en batch (`0,38×`). **1 913 011/1 986 560** logits FP16 diffèrent
  (écart absolu max 0,34179688) et **72/80** tenseurs d'état diffèrent. Les
  opérations GDN/SDPA multi-token n'ont pas l'arrondi exact du chemin `M=1`,
  et le lm_head/MoE batch est ici plus coûteux. Le chemin n'est pas exposé.
- Sous-essai suivant, journalisé avant code : construire les huit pas dans
  l'ordre autoregressif exact avec les kernels `M=1`, sans `eval` entre les
  tokens, concaténer les logits puis synchroniser une seule fois. Attendu :
  parité bit-à-bit par construction et gain limité aux synchronisations et à
  l'ordonnancement du graphe. Si le graphe grossit ou régresse, le rejeter.
- Résultat du graphe autoregressif différé sur le modèle réel : **validé comme
  primitive lossless**. Huit pas séquentiels synchronisés prennent **179,207
  ms**, contre **153,829 ms** avec une seule synchronisation, soit **1,16×**.
  Les **1 986 560** logits FP16 et les **80/80** tenseurs d'état sont identiques
  bit-à-bit. Ce résultat ponctuel correspond à ~52 vérifications/s et ne
  revendique pas encore un débit DFlash complet : draft, sélection, acceptation
  et commit ne sont pas branchés.
- Le fallback QMV `2<=M<24` est également exact sur la projection test
  (**0/65 536** différence) ; il sert aux futurs kernels batch, mais le chemin
  target retenu ici reste token-par-token différé pour préserver l'arithmétique
  GDN/SDPA.
- Vérification du jalon : format et Clippy strict réussis ; **21** tests lib,
  **1** test CLI et **14** contrats réussis. Les deux tests GPU ciblés réussissent
  sur M5. Kani 0.68 / CBMC 6.11 vérifie **15/15 harnesses**, zéro échec et 2/2
  couvertures (`build/perf-47-kani.log`). Kani est exécuté sans le feature MLX :
  il ne prouve ni les graphes MLX, ni Metal, ni les états Qwen ; leur parité est
  couverte par les différentiels physiques bornés décrits ci-dessus. Statut :
  **validé et intégré comme primitive inactive**.

### OPT-2026-09-19-RUST-PERF-48 — DFlash 2 : package Splash et Q4 — en cours

- Hypothèse : réutiliser le format public Apache-2.0 et les géométries exactes
  de `incoai/Qwen3.6-35B-A3B-Splash` évite de reconvertir ou réentraîner le
  drafter. Seul le sous-répertoire `draft/` est requis ; les poids cible Q4 de
  Splash ne seront pas téléchargés ni utilisés par la cible EXL3 MLXL3.
- Géométrie issue du manifeste officiel : 6 couches, hidden 2048, dynamique
  512, QKV 6144, attention 4096, MLP 6144, vocabulaire 248320, projection de
  contexte 16384→2048 et sélecteur rank 256. Q4 affine, groupes de 64,
  StorageN=256. Taille déclarée : six fichiers de 34 324 480 octets et un
  `model.bin` de 273 498 112 octets, soit ~457 MiB de poids draft.
- Changement prévu : téléchargement ciblé, chargeur Rust strict (magic
  `MDFD0004`, layer/type, alignement 16 KiB, arithmétique vérifiée, consommation
  exacte), puis port minimal des kernels Q4 officiels avant le graphe draft.
- Protocole : refuser header, taille, offset ou section incorrect ; comparer les
  offsets calculés à `layout.json` ; test réel du package, tests négatifs
  synthétiques et Kani pour l'arithmétique pure. Aucun débit n'est revendiqué
  avant que le drafter produise et que la cible accepte des tokens. Statut :
  **en cours**, journalisé avant code.
- Jalon chargeur : **validé**. Le lecteur Rust refuse les mauvais magic/type/id,
  les fichiers non réguliers, toute taille inattendue, tout dépassement et tout
  offset non aligné. Les offsets calculés correspondent au manifeste publié
  (`layer qkv=638 976`, `layer down=27 246 592`, codebooks modèle
  `19 218 432/146 358 272`) et les sept fichiers réels du package local sont
  acceptés ; le runtime ne dépend d'aucun poids cible Splash.
- Contrôles : tests synthétiques et package réel réussis, Clippy strict avec
  tous les features réussi. Kani 0.68 / CBMC 6.11 vérifie les 11 propriétés
  d'alignement sur tout `u64` et les 49 propriétés de géométrie/taille Q4 sur
  tout couple `u16`, avec 2/2 couvertures atteintes. Un ICE Kani causé par
  `is_multiple_of` a été éliminé en conservant l'équivalent `%`, puis les deux
  harnesses ont réussi. La passe complète vérifie **17/17 harnesses**, zéro
  échec (un chemin standard inaccessible), dont 162 propriétés pour le plus
  gros contrat et 2/2 couvertures. Cela ne vérifie ni les octets de poids, ni
  Metal.
- Étape suivante, journalisée avant essai : mapper une projection Q4 réelle et
  comparer sur M5 plusieurs kernels compatibles (Splash MPP servant de
  référence, kernel MLXL3 retenu s'il gagne), d'abord en exactitude puis en
  temps chaud. Aucun gain d'inférence n'est encore revendiqué. Statut global :
  **en cours**, chargeur prêt à intégrer.

### OPT-2026-09-19-RUST-PERF-49 — DFlash Q4 M=8 : sélection kernel M5 — en cours

- Hypothèse : le MPP TensorOps Q4 affine, groupe 64 et StorageN=256 est une
  bonne référence pour les huit lignes du draft, mais sa tuile et son nombre de
  groupes persistants ne sont pas supposés optimaux pour le M5 testé. Comparer
  au minimum N128 séquentiel, N128 pipeliné et N256, avec plusieurs grilles.
- Projection réelle : `draft/layer-0.bin:qkv`, forme 2048→6144, huit entrées
  BF16 déterministes. Baseline externe : kernel Splash ; baseline MLXL3 : aucun
  kernel DFlash Q4 avant cet essai. Deux warmups de compilation puis au moins
  cinq mesures chaudes synchronisées par candidat, ordre alterné si la durée le
  permet. Conditions secteur/thermique consignées au résultat.
- Validité : toutes les variantes retenues doivent donner les mêmes octets BF16
  que la référence MPP séquentielle sur le tenseur complet ; un contrôle CPU
  indépendant sur un sous-ensemble vérifiera aussi le décodage affine Q4. Une
  variante divergente ou plus lente est rejetée et ne reste pas dans le chemin
  production. Mesurer séparément compilation/chargement et exécution chaude.
- Ce microbenchmark ne prédit pas encore le débit DFlash complet : attention,
  convolutions, sélecteur, vocabulaire et acceptation restent absents. Statut :
  **en cours**, journalisé avant code.
- Essai de compilation 1 : **rejeté/corrigé avant mesure**. Les constantes de
  forme étaient émises après le corps helper dans le header MLX, donc Metal ne
  pouvait pas résoudre `DFLASH_INPUT`. Aucun kernel n'a été exécuté et aucune
  mesure n'est issue de cet essai. Les `#define` sont désormais placés avant le
  helper ; le nouveau build doit encore être validé.
- Premier passage GPU réel, secteur, M5 10 cœurs/Metal 4 : tous les candidats
  N128/N128-pipeliné/N256 rendent exactement les mêmes **98 304 octets BF16**.
  Le contrôle CPU indépendant sur 512 sorties distingue bien l'ordre des
  nibbles : erreur basse `(max=0, moyenne=0)` contre ordre inversé
  `(max=6,9648438, moyenne=1,636845)`.
- Les chronos ne sont pas encore stabilisés : le premier processus a mesuré
  N128 g48 **0,807 ms**, pipeliné g48 **0,776 ms**, N256 g24 **1,136 ms** ; le
  rejeu chaud immédiatement après donne respectivement **0,367/0,337/0,326
  ms**. Cette dérive dépasse les écarts entre variantes : résultat
  **non concluant** pour le choix final. La prochaine passe alternera l'ordre,
  augmentera le warmup et mesurera plusieurs cycles A/B/A avant intégration.
- Passe alternée, 100 warmups puis 30 échantillons/candidat, répétée deux fois
  sur secteur : N128-pipeliné g40/g48 tient **0,290–0,297 ms**, N128 g40
  **0,294–0,296 ms**, N256 g24 **0,309–0,310 ms**. Le pipelinage apporte donc
  seulement ~1,7–2,0 % sur cette projection ; g40 et g48 sont dans le bruit.
  Les sorties restent identiques. Prochain essai journalisé : enlever les tests
  `is_valid_element` internes lorsque la capacité coopérative couvre exactement
  les 8×N éléments, tout en gardant une variante gardée comme oracle. Comparer
  N128 pipeliné et N256, mêmes 30 échantillons ; rejet si un octet change.
- Essai traversal direct : **rejeté**. Sur le vrai QKV, la suppression des
  gardes provoque un `kIOGPUCommandBufferCallbackErrorPageFault` avant toute
  mesure : la capacité coopérative contient bien des emplacements invalides
  avec ce compilateur/descriptor. Aucune sortie ni performance n'est attribuée
  à cette variante. Les variantes `Fast` sont retirées ; le kernel conservé
  continue d'appeler `is_valid_element`, comme le fallback sûr publié.
- Revalidation après retrait, même protocole : N128-pipeliné g48 **0,290 ms**
  médiane (p10 0,265, p90 0,319), N128 g48 **0,298 ms**, N128 g40 **0,298
  ms**, N256 g24 **0,306 ms**. Sur trois passes chaudes, le choix pipeliné g48
  reste entre **0,289 et 0,291 ms**, soit ~2–3 % devant le N128 gardé non
  pipeliné et ~5–7 % devant N256 g24 ; 98 304/98 304 octets restent identiques.
  **Validé pour cette forme 2048→6144**, sans extrapoler aux autres formes.
- Décision : conserver les trois implémentations sûres pour l'autotuning, avec
  N128-pipeliné/full-grid comme choix provisoire de la forme QKV M5. La mesure
  est un microbenchmark de projection, pas encore un gain de decode du modèle.
  Statut PERF-49 : **validé et prêt à intégrer**, publication après contrôles
  Rust/Metal complets.
- Contrôles finaux du jalon : Clippy strict tous targets/features ; 23 tests
  lib, 1 CLI et 14 contrats réussis. Le test GPU ignoré exécute le QKV réel,
  le contrôle CPU et 15 géométries Metal ; toutes les variantes conservées
  sont bit-à-bit identiques. Kani vérifie séparément 11 propriétés d'alignement
  et 49 propriétés Q4 avec 2/2 couvertures ; il ne couvre pas MLX/Metal, validés
  ici uniquement par différentiel physique. Statut d'intégration : code local
  validé, push du jalon suivant.

### OPT-2026-09-19-RUST-PERF-50 — DFlash 2 : graphe draft MLXL3 — en cours

- Hypothèse : conserver le format et l'arithmétique du drafter Splash, mais
  sélectionner séparément chaque primitive sur M5, permet d'obtenir un graphe
  plus rapide sans lier MLXL3 à l'ordonnanceur Splash. Une primitive MLX native
  ou MLXL3 sera retenue seulement si elle bat le kernel Splash à sortie BF16
  identique ; Splash reste l'oracle de compatibilité, pas une contrainte
  d'implémentation.
- Premier sous-essai : charger strictement les treize sections de chacune des
  six couches et les sections globales, puis implémenter les deux phases de la
  convolution dynamique 8×2048. Comparer le kernel Metal au calcul CPU BF16
  indépendant sur toutes les 16 384 sorties, balayer les groupes persistants et
  mesurer après warmup sur le M5. Rejet de toute variante qui diverge ou plante.
- Baseline utile : Q4 QKV 2048→6144 M=8 validé dans PERF-49 à **0,289–0,291
  ms** médiane pour N128 pipeliné g48. Le débit DFlash complet et le taux
  d'acceptation restent **non mesurés** ; il est interdit d'extrapoler ce timing
  isolé à des tok/s.
- Étapes suivantes déjà bornées : Q/K RMS+RoPE, attention glissante 2048,
  gate/up+SwiGLU fusionné, six couches, sélecteur, puis boucle strictement
  lossless verify/rollback. Chaque alternative sera comparée sur le même graphe
  et le même état. Statut : **en cours**, journalisé avant code ; aucune
  publication de ce jalon.
- Chargeur complet : **validé** sur les 457 MiB officiels. Les six couches et
  les 13 sections par couche, les projections globales, les deux normes et les
  codebooks 248320×256 sont chargés avec les formes attendues en **0,14 s** de
  test (processus complet **0,23 s**). `/usr/bin/time -l` rapporte un RSS max de
  **502 349 824 octets** ; la mesure mélange mappings et runtime de test et ne
  constitue pas encore la RAM incrémentale de l'app.
- Première exécution convolution : **corrigée avant timing**. Le grid MLX avait
  été exprimé en groupes au lieu de threads (`groups×256`) : seules quelques
  sorties étaient écrites. Le différentiel exhaustif l'a détecté ; aucune
  performance n'est attribuée à ce lancement.
- Convolution corrigée : les phases prepare et residual, pour g8/12/16/24/32/
  48/64, donnent les mêmes **32 768 octets BF16** que la référence CPU
  indépendante (16 384 sorties chacune). Après 100 warmups, 50 échantillons :
  médianes **183,375–185,834 µs** ; g24 est nominalement premier à **183,375
  µs**, mais tout l'intervalle est du bruit. Décision provisoire : g24, sans
  supprimer les autres possibilités avant le benchmark du graphe enchaîné.
- Les valeurs ci-dessus incluent `eval()` et la synchronisation par primitive ;
  le vrai graphe draft différé doit amortir ce coût. Étape suivante : Q/K
  RMS+RoPE et attention avec état, puis comparaison primitive MLX versus kernel
  spécialisé. Statut PERF-50 : **en cours**, chargeur et convolution validés
  localement, pas encore publiés.
- Premier graphe des six couches, contexte vide : **validé fonctionnellement**.
  Il enchaîne normes, projections Q4, convolutions, Q/K RMS+RoPE, SDPA GQA,
  gate/up+SwiGLU, down, norme finale et sélecteur. Deux exécutions donnent des
  sorties hidden 8×2048 et selector 8×256 identiques bit-à-bit, sans BF16 non
  fini. Après 20 warmups et 30 mesures : médiane **4,621 ms**, p10 **4,559
  ms**, p90 **4,898 ms** pour les six couches et le sélecteur, hors lm_head,
  sélection, vérification cible, acceptation et commit.
- Diagnostic : ce draft n'est déjà plus le facteur limitant. La vérification
  cible lossless PERF-47 prend **153,829 ms / 8 positions** ; même sept drafts
  tous acceptés donnent un plafond d'environ **50,5 tok/s** avant les autres
  frais. Atteindre 100 tok/s exige donc d'abord une vérification cible M=8 sous
  ~75 ms, sans l'écart numérique du QMM paddé rejeté. Prochain essai : kernel
  Qwen M=8 séquentiel-fusé/streamé qui conserve l'ordre M=1 et supprime les
  relectures et dispatchs inutiles ; le taux d'acceptation sera mesuré seulement
  après la boucle complète. Statut : **en cours**.
- Contrôles du jalon : format et Clippy strict tous targets/features réussis ;
  23 tests lib, 1 test CLI et 14 contrats réussis. Les trois tests GPU réels
  ignorés par défaut valident le chargement officiel, la convolution exhaustive
  et le graphe six couches. Kani 0.68 vérifie **17/17 harnesses**, 0 échec ; il
  couvre le lecteur/planificateur pur mais pas MLX ni Metal, contrôlés ici par
  les différentiels et exécutions physiques. Décision : publier ce jalon draft
  mesuré, tout en gardant PERF-50 **en cours** jusqu'à la boucle lossless et au
  débit DFlash end-to-end.
- Révision du 20 septembre, avant nouvel essai : l'audit externe fourni et la
  lecture directe du kernel Splash `draft_attention_split_phase` montrent que
  les huit K/V courants sont ajoutés sans masque causal entre lignes. MLXL3
  utilise encore SDPA causal à contexte vide et masque les lignes courantes
  futures avec cache. Sous-essai E01 : rendre les huit lignes courantes visibles
  tout en conservant la fenêtre causale uniquement sur l'historique, puis laisser
  un test où la valeur de la ligne 7 doit modifier la sortie de la ligne 0. Le
  débit ne sera interprété qu'après ce contrôle de fidélité. État : **en cours**.
- E01 reproduit puis corrigé : le test qui ne modifie que les V de la ligne 7
  échouait avec le SDPA causal (la sortie de la ligne 0 restait identique), puis
  réussit après passage du bloc courant en non-causal. Avec historique, seules
  les colonnes antérieures respectent la borne glissante ; les huit colonnes
  courantes sont visibles pour chaque requête, comme dans le kernel Splash lu.
  Le graphe six couches reste déterministe et fini ; timing ponctuel médian
  **5,390 ms**, p10 **4,567**, p90 **5,525**. La dispersion interdit d'attribuer
  ici une régression par rapport aux 4,621 ms précédents. Fidélité Splash des
  tenseurs complets et cas anneau restent à qualifier avant de clore E01.
- Extension du même test avec un cache historique d'une position : échec avant
  l'oracle, car le masque FP32 ne peut pas promouvoir la sortie SDPA BF16. Le
  cache draft n'avait donc pas de chemin exécutable validé. Correction minimale
  à la frontière commune : caster le masque additif en BF16 avant SDPA ; relance
  du test causal/non-causal requise avant toute mesure avec contexte.
- Après correction du dtype, le test physique réussit à contexte vide et avec
  une position historique : modifier uniquement V à la ligne courante 7 modifie
  bien la sortie de la ligne 0 dans les deux cas. Cela valide le défaut initial,
  sa correction et l'exécution du cache court ; la parité numérique Splash et
  le wrap à 2048 restent non mesurés.

### OPT-2026-09-19-RUST-PERF-51 — Vérification cible M=8 exacte — en cours

- Hypothèse : la vérification lossless actuelle calcule les huit `lm_head`
  EXL3 comme huit QMV indépendants. Un QMV Metal M=2..8 qui ajoute seulement
  l'indice de ligne à la géométrie existante peut partager l'ordonnancement et
  améliorer la localité des poids sans changer l'ordre arithmétique interne de
  chaque ligne. Le reste du modèle demeure strictement autoregressif M=1.
- Baseline : Qwen3.6-35B-A3B EXL3 2,49 bpw, tokens cibles 4..11 après le préfixe
  1,2,3, Apple M5 ; vérification différée exacte PERF-47 **153,829 ms / 8**.
  Batterie et thermique du nouveau passage seront relevées ; aucun gain n'est
  revendiqué avant une série alternée.
- Protocole : comparer chaque logit FP16 et chaque état cible au chemin huit
  forwards M=1, puis mesurer au moins cinq passages chauds. Rejeter au premier
  écart. La distribution cible, les tokens acceptés et le rollback ne doivent
  pas changer. État : **en cours**, journalisé avant code.
- Première compilation du prototype interrompue correctement : le shader QMV
  partagé référençait `INPUT_DIMS`, absent du header du chemin M=1. Aucun timing
  n'a été retenu ; le define a été ajouté aux deux géométries avant relance.
- Projection réelle Qwen 2048→8192, K=4, M=8 : QMV ligne par ligne **1,872 ms**,
  QMV à grille 2D **0,573 ms**, soit **3,27×** sur ce microbenchmark ; 0/65 536
  sorties FP16 différentes, écart max 0. Le kernel conserve une accumulation
  indépendante et le même ordre par ligne ; seul l'indice de batch est ajouté.
- Vérificateur complet, série appariée ABBA sur batterie : ancien graphe différé
  médian **168,334 ms** (166,859–173,020), tête M=8 partagée **159,950 ms**
  (156,112–161,647), soit **1,052×**. Les 12 passages conservent exactement les
  1 986 560 logits FP16 et les 80 états ; les QMV M=2/4/8/16/23 sont aussi
  identiques ligne par ligne. Décision : conserver ce premier partage exact ;
  il améliore V d'environ 5 % ici, loin du budget 45–75 ms visé. E06/E10 restent
  nécessaires. K=7 reste volontairement sur le fallback série faute de fixture
  locale permettant de qualifier le kernel multi-lignes.
- Contrôles finaux du jalon : `cargo fmt --all -- --check`, Clippy strict
  tous targets/features, 23 tests lib, 1 test CLI, 14 tests de contrats et le
  build release MLX/chat réussissent. Kani 0.68/CBMC 6.11 vérifie **17/17
  harnesses**, 0 échec et 2/2 couvertures ; il ne couvre pas MLX/Metal. Les
  chemins GPU sont donc validés séparément sur le M5 par les différentiels QMV
  M=2/4/8/16/23, le replay exact logits+état Qwen, le test de visibilité
  DFlash vide+cache et le graphe draft complet déterministe/fini. Statut :
  **validé et prêt à intégrer** pour le partage exact du `lm_head` et les
  corrections d'attention ; la vérification cible layer-major et le débit
  DFlash end-to-end restent non implémentés.

### OPT-2026-09-20-RUST-PERF-52 — Vérification exacte couche-major — en cours

- Hypothèse issue de l'audit fourni, recoupée avec le code : le vérificateur
  exact avance encore chaque token dans les 40 couches avant le suivant. Sans
  modifier un kernel, avancer les huit tenseurs mono-token dans une couche
  avant de passer à la suivante conserve les récurrences GDN/KV propres à
  chaque couche et rapproche les lectures d'un même jeu de poids.
- Baseline appariée la plus récente, Qwen3.6-35B-A3B EXL3 2,49 bpw, préfixe
  `[1,2,3]`, vérification `[4..11]`, batterie 100 % : chemin token-major avec
  tête M=8 **159,950 ms** médiane (156,112–161,647), logits FP16 et 80 états
  identiques à l'ancien chemin différé.
- Prototype minimal prévu : garder chaque opération sensible en M=1 et le même
  ordre temporel *dans chaque couche* ; seule la boucle tokens/couches est
  transposée. Comparaison ABBA d'au moins six passages par variante, de chaque
  logit FP16 et des 80 états. Rejet au premier octet différent ou si le gain
  n'est pas stable. Aucun débit DFlash end-to-end ne sera extrapolé. Statut :
  **en cours**, journalisé avant code.
- Première commande interrompue avant compilation : lancée depuis `native/`,
  elle a résolu `MLXL3_MLX_ROOT` sous `native/.venv` au lieu de la racine.
  Aucun kernel ni timing n'a été exécuté. La relance utilise le chemin absolu
  du runtime MLX du dépôt.
- Première série ABBA, six mesures par variante, batterie 100 % : token-major
  médian **161,909 ms** (143,916–162,827), couche-major **149,137 ms**
  (141,490–171,693), soit **1,086×**. Les 12 passages produisent exactement les
  mêmes **1 986 560 logits FP16** et les mêmes **80 états**. Le signal est
  positif mais les plages se recouvrent et un outlier couche-major existe ; une
  seconde série indépendante est requise avant intégration.
- Seconde série indépendante : token-major médian **144,678 ms**
  (137,309–146,584), couche-major **143,745 ms** (140,090–158,833), soit
  seulement **1,006×**. L'identité complète reste vérifiée, mais le gain du
  simple réordonnancement n'est pas stable ; il n'est pas suffisant seul.
- Sous-essai suivant, journalisé avant code : `ProjectionBundle::Grouped`,
  utilisé par QKV et gate/up, sérialise encore chaque ligne malgré le QMV exact
  multi-lignes de PERF-51. Étendre le shader mappé avec un axe de lignes doit
  grouper ces dispatchs sans changer l'accumulation de chaque sortie. Comparer
  toutes les sorties FP16 du bundle à huit appels M=1, puis le modèle complet
  couche-major au token-major. Rejet au premier écart ; le réordonnancement
  couche-major sera retiré si le bundle groupé n'apporte pas un gain stable.
- Microbenchmark réel couche 0, bundle QKV+Z 2048→(6144+4096), M=8 : huit
  appels groupés M=1 **1,584 ms**, nouvel axe de lignes **0,804 ms**, soit
  **1,97×**. Les deux sorties, **81 920 valeurs FP16**, sont identiques bit à
  bit. Ce résultat justifie le test modèle complet mais ne constitue pas encore
  un gain du vérificateur.
- Intégration expérimentale suivante : uniquement dans les 30 couches GDN,
  batcher QKV/Z/A/B et la projection de sortie avec les QMV exacts, tout en
  exécutant convolution et récurrence token par token dans leur ordre canonique.
  Les normes, résiduels et MoE restent mono-token. Cette frontière minimale
  isole le partage des poids ; elle sera comparée au token-major sur logits et
  80 états avant toute extension aux dix couches d'attention.
- Première exécution arrêtée par la validation de forme avant mesure : les
  sorties SwiGLU concaténées gardaient `[1,8,Hv,Dv]` au lieu d'être repliées en
  `[1,8,Hv×Dv]` pour `out_proj`. Aucun timing ni résultat numérique n'est
  attribué à ce passage ; le reshape identique au chemin canonique est ajouté.
- Première série modèle après correction, six passages ABBA par variante :
  token-major médian **145,190 ms** (140,824–165,353), couche-major avec
  projections GDN groupées **113,266 ms** (110,779–116,259), soit **1,282×**.
  Les 12 passages conservent exactement les 1 986 560 logits FP16 et les 80
  états. Le signal est net ; une seconde série indépendante doit confirmer la
  stabilité avant extension ou intégration.
- Seconde série indépendante : token-major médian **137,759 ms**
  (135,110–138,697), candidat **108,866 ms** (107,657–111,032), soit
  **1,265×**. Les intervalles ne se recouvrent pas et l'identité logits+états
  est de nouveau complète. Le gain est **validé pour M=8** ; les largeurs
  1/2/4 restent à contrôler avant intégration.
- Contrôle des largeurs **M=1/2/4/8** réussi sur le modèle réel : pour chaque
  largeur, tous les logits FP16 et les 80 tenseurs d'état correspondent octet
  pour octet au chemin token-major. Le candidat peut passer aux contrôles
  complets ; cela reste une vérification physique bornée, pas une preuve de
  tous les prompts ni de Metal.
- Contrôles du jalon : format, Clippy strict tous targets/features, 23 tests
  lib, 1 CLI, 14 contrats et build release MLX/chat réussis. Kani 0.68/CBMC
  6.11 vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures ; il ne
  compile pas MLX/Metal. Les nouveaux axes Metal et la transposition des états
  sont donc couverts par les différentiels physiques M=1/2/4/8 et les deux
  séries ABBA, pas par Kani. Statut : **validé et prêt à intégrer** ; coût cible
  M=8 ramené de 137,759 à 108,866 ms dans la série indépendante, encore au-dessus
  du budget ~75 ms nécessaire à 100 tok/s même avec acceptation parfaite.

### OPT-2026-09-20-RUST-PERF-53 — Attention cible : projections exactes M=8 — en cours

- Suite de PERF-52 : les dix couches d'attention exécutent encore Q/Gate/K/V et
  `o_proj` huit fois. Hypothèse : utiliser les QMV groupés multi-lignes pour ces
  seules projections, puis avancer RoPE, append KV et SDPA causal en M=1 dans
  l'ordre canonique, partage les poids sans modifier le masque ni la réduction
  d'attention.
- Baseline indépendante : vérificateur PERF-52 **108,866 ms** médian
  (107,657–111,032), contre token-major 137,759 ms, batterie. Protocole : série
  ABBA, logits FP16 et 80 états octet par octet, M=1/2/4/8. Rejet à tout écart
  ou si le gain n'est pas reproductible. Les MoE restent mono-token afin
  d'isoler la projection d'attention. Statut : **en cours**, journalisé avant
  code.
- Exactitude physique M=1/2/4/8 : tous les logits FP16 et les 80 états restent
  identiques au token-major. Première série ABBA M=8 : token-major médian
  **141,809 ms** (139,389–145,197), candidat GDN+attention **104,353 ms**
  (103,519–105,943), soit **1,359×** face à la référence du même processus.
  Par rapport aux 108,866 ms indépendants de PERF-52, l'attention apporte
  environ 4,5 ms supplémentaires ; répétition indépendante requise.
- Répétition indépendante : token-major médian **143,504 ms**
  (139,091–145,417), candidat **104,722 ms** (103,746–107,606), soit
  **1,370×**, toujours strictement identique. Les projections d'attention sont
  donc **validées** ; le coût cible reste toutefois ~30 ms au-dessus du budget
  optimiste de 75 ms.
- Contrôles finaux : le différentiel physique M=1/2/4/8 réussit après retrait
  du prototype MoE divergent. Format, Clippy strict, 23 tests lib, 1 test CLI,
  14 contrats et build release MLX/chat réussissent. Kani 0.68 / CBMC 6.11
  vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures ; il ne couvre pas
  MLX/Metal, validés séparément par le différentiel bit à bit. Statut :
  **validé et intégré**, prêt à publier.

### OPT-2026-09-20-RUST-PERF-54 — MoE exact multi-lignes du vérificateur — en cours

- Observation : après PERF-53, les normes et résiduels sont déjà disponibles
  couche par couche, mais chaque MoE traite encore huit tokens séparément. Pour
  M=8, le chemin existant `Exl3SwitchGlu` reste sous le seuil TensorOps 64 et
  utilise le même QMV mappé par route ; le batch peut donc mutualiser dispatchs,
  routeur et projections partagées sans changer l'accumulation des experts.
- Baseline : candidat PERF-53 **104,722 ms** médian, référence token-major
  **143,504 ms**. Essai : calculer chaque RMSNorm post-attention séparément,
  concaténer seulement ses huit lignes pour `Mlp::forward`, puis restaurer les
  résiduels dans l'ordre. Contrôler M=1/2/4/8, logits et 80 états bit à bit,
  puis deux séries ABBA. Rejet immédiat à tout écart. Statut : **en cours**,
  journalisé avant code.
- Résultat : **rejeté**. Les largeurs M=1/2/4 restent exactes, mais M=8 produit
  de nombreuses divergences dans les logits FP16 par rapport au chemin
  token-major. Le seuil M=8 change donc le chemin d'exécution MoE et ne conserve
  pas l'arithmétique canonique ; aucun timing n'a été retenu. Le batch MoE
  complet est retiré, tandis que PERF-53 reste inchangé et exact.

### OPT-2026-09-20-RUST-PERF-55 — Expert partagé MoE multi-lignes — en cours

- Hypothèse : la divergence PERF-54 vient du routage/expert sparse à huit
  lignes, pas des projections d'expert partagé qui sont indépendantes par
  ligne. Conserver gate, top-k et experts routés en huit appels M=1, mais
  calculer `shared_expert` et son multiplicateur en M=1/2/4/8 doit mutualiser
  leurs lectures de poids tout en gardant l'arithmétique lossless.
- Baseline : PERF-53 **104,722 ms** médian pour huit positions, référence
  token-major **143,504 ms**. Protocole : séparer sans duplication les chemins
  routé/partagé du MoE, comparer logits FP16 et 80 états octet par octet pour
  M=1/2/4/8, puis deux séries ABBA. Rejet immédiat au premier écart. Statut :
  **en cours**, journalisé avant code.
- Première compilation interrompue par une accolade fermante manquante dans le
  helper de vérification ; aucun modèle, kernel ni timing exécuté. Correction
  syntaxique uniquement avant reprise du protocole inchangé.
- Exactitude physique M=1/2/4/8 : tous les logits FP16 et les 80 états sont
  identiques octet par octet au token-major. Deux séries ABBA M=8 donnent
  respectivement **94,658 ms** (93,190–95,924) contre 136,622 ms, puis
  **98,081 ms** (93,340–105,459) contre 144,954 ms, soit **1,44–1,48×** face à
  la référence du même processus. Comparé à PERF-53 (104,722 ms), l'expert
  partagé économise environ **6,6–10,1 ms** selon la passe. Statut :
  **validé**.
- Contrôles finaux : format, Clippy strict, 23 tests lib, 1 test CLI,
  14 contrats et build release MLX/chat réussissent. Kani 0.68 / CBMC 6.11
  vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures. Comme auparavant,
  Kani n'exécute pas MLX/Metal ; le graphe GPU est couvert par les différentiels
  physiques exacts et les séries ABBA. Statut : **validé et intégré**, prêt à
  publier.

### OPT-2026-09-20-RUST-PERF-56 — Gate et top-k MoE multi-lignes — en cours

- Hypothèse : PERF-54 a seulement démontré que le calcul groupé des experts
  sparse diverge à M=8. La projection gate, le softmax et le kernel top-k sont
  indépendants par ligne et peuvent produire les routes des huit positions en
  un seul graphe, puis alimenter huit appels experts M=1 inchangés. Cette
  frontière complète PERF-55 sans toucher l'accumulation sparse.
- Baseline : PERF-55 **94,658–98,081 ms** médian pour huit positions. Protocole :
  batcher gate/softmax/top-k, découper indices et scores par ligne, exécuter les
  experts routés en M=1, puis comparer logits FP16 et 80 états pour M=1/2/4/8
  et deux séries ABBA. Rejet immédiat au premier écart. Statut : **en cours**,
  journalisé avant code.
- Résultat : **rejeté**. M=1/2/4 reste exact, mais M=8 diverge massivement
  dans les logits FP16 avant toute mesure. La projection gate/softmax/top-k
  multi-lignes change donc l'arithmétique ou les routes à cette largeur. Aucun
  timing n'est retenu ; gate, top-k et experts sparse reviennent entièrement
  en M=1, tandis que l'expert partagé PERF-55 reste intégré.

### OPT-2026-09-20-RUST-PERF-57 — Experts sparse batchés, routes M=1 — en cours

- Diagnostic PERF-56 : la gate Qwen est une matrice dense FP16 ; son matmul
  M=8 peut choisir un GEMM dont l'arrondi diffère du GEMV M=1. Pour isoler le
  calcul coûteux, produire gate/softmax/top-k séparément pour chaque ligne,
  concaténer ces routes exactes, puis exécuter uniquement `Exl3SwitchGlu` sur
  les huit lignes. L'expert partagé reste celui de PERF-55.
- Baseline : PERF-55 **94,658–98,081 ms** médian. Protocole : logits et 80
  états octet par octet pour M=1/2/4/8, puis deux séries ABBA. Cet essai est
  rejeté si le kernel expert multi-lignes change un seul résultat. Statut :
  **en cours**, journalisé avant code.
- Exactitude physique M=1/2/4/8 : tous les logits FP16 et les 80 états restent
  identiques octet par octet. Deux séries ABBA M=8 donnent **86,274 ms**
  (85,245–99,513) contre 138,620 ms, puis **86,132 ms** (85,483–86,212) contre
  139,159 ms, soit **1,61–1,62×** face au token-major. Par rapport à PERF-55,
  le batch sparse exact apporte encore environ **1,10–1,14×**. Statut :
  **validé**.
- Contrôles finaux : format, Clippy strict, 23 tests lib, 1 test CLI,
  14 contrats et build release MLX/chat réussissent. Kani 0.68 / CBMC 6.11
  vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures ; MLX/Metal reste
  hors de sa portée et est vérifié ici par les différentiels physiques exacts.
  Statut : **validé et intégré**, prêt à publier.

### OPT-2026-09-20-RUST-PERF-58 — Normes et résiduels cible multi-lignes — en cours

- Observation : les 40 couches calculent encore input RMSNorm, résiduel,
  post RMSNorm et résiduel MLP dans huit graphes mono-ligne, bien que chaque
  opération ne réduise que le dernier axe et n'échange aucune donnée entre
  positions. Concaténer les huit hidden, appliquer ces opérations une fois,
  puis redécouper uniquement à la frontière des routes M=1 doit supprimer des
  centaines de dispatchs sans changer l'ordre arithmétique interne d'une ligne.
- Baseline : PERF-57 **86,132–86,274 ms** médian pour huit positions. Protocole :
  M=1/2/4/8 avec logits FP16 et 80 états strictement identiques, puis deux
  séries ABBA. Rejet à tout écart. Statut : **en cours**, journalisé avant code.
- Résultat : exact sur M=1/2/4/8, mais **rejeté pour absence de gain stable**.
  Deux séries donnent **87,399 ms** (85,138–88,621), puis **85,393 ms**
  (85,292–86,435), contre 86,132–86,274 ms pour PERF-57. Les plages se
  recouvrent et le premier passage régresse ; le batch normes/résiduels est
  retiré. Aucun gain n'est revendiqué.

### OPT-2026-09-20-RUST-PERF-59 — Routeur MoE groupé après gates exactes — validé

- Diagnostic PERF-56 : la projection dense de gate en M=8 change les arrondis,
  mais cela ne démontre pas que le softmax précis et le top-k changent une fois
  les logits mono-token préservés. Hypothèse : calculer les huit gates en M=1,
  concaténer leurs logits puis lancer un seul softmax/top-k par couche réduit les
  dispatchs sans changer aucune route ni aucun score.
- Baseline : PERF-57 **86,132–86,274 ms** médian pour huit positions, modèle
  Qwen3.6-35B-A3B EXL3 2.49 bpw sur M5, batterie. Protocole : comparaison exacte
  des logits FP16 et des 80 états pour M=1/2/4/8, puis deux séries ABBA avec le
  test physique ignoré. Rejet au premier écart ou si le gain n'est pas stable.
  Statut : **en cours**, journalisé avant code.
- Première commande interrompue avant compilation : `cargo` n'était pas dans le
  `PATH` de ce shell (`command not found`). Aucun test ni timing n'a été produit ;
  reprise avec le binaire Rust installé explicitement, protocole inchangé.
- Deuxième commande interrompue par le build script : `MLXL3_MLX_ROOT` n'était
  pas défini. Aucun test physique ni timing n'a démarré ; reprise avec le paquet
  MLX 0.32.2 déjà installé dans `.venv`, protocole inchangé.
- Le benchmark AB temporaire n'a pas compilé à sa première tentative, car
  `QwenSnapshot` n'est volontairement pas clonable. Aucun test/timing produit ;
  le test conserve une nouvelle vue du snapshot après chaque restauration sans
  modifier l'API de production, puis reprend le même protocole.
- Sa correction initiale a modifié par erreur le snapshot du test voisin et
  laissé celui du benchmark inchangé ; compilation encore interrompue, sans
  exécution. Les deux déclarations sont corrigées explicitement avant reprise.
- Exactitude : le différentiel physique M=1/2/4/8 conserve tous les logits FP16
  et les 80 états octet par octet. Deux séries ABBA isolant seulement le routeur
  donnent **84,588 ms** groupé contre 85,906 ms série, puis **88,248 ms** contre
  89,050 ms : gain reproductible de **0,8–1,3 ms** (**1,009–1,016×**) pour huit
  positions malgré la chauffe. Le commutateur AB temporaire est retiré ; seules
  les huit gates M=1 suivies d'un softmax/top-k M=8 restent en production.
- Statut : **validé et intégré**. Contrôles de jalon en cours avant publication ;
  le gain est un microbenchmark cible réel et ne prédit pas encore un débit
  DFlash2 bout en bout tant que sa boucle d'acceptation n'est pas intégrée.
- Premier contrôle de jalon arrêté par `cargo fmt --check` sur une ligne vide
  laissée après retrait du benchmark temporaire. Aucun autre contrôle n'a été
  lancé par cette commande ; formatage mécanique puis reprise complète.
- Contrôles finaux réussis : format, Clippy strict tous targets/features,
  23 tests lib, 1 test CLI, 14 contrats, build release et différentiel physique
  M=1/2/4/8. Kani 0.68 / CBMC 6.11 vérifie **17/17 harnesses**, zéro échec et
  2/2 couvertures ; ses bornes portent sur le Rust pur et n'incluent pas MLX ou
  Metal, couverts ici par le différentiel exact sur le modèle réel. Statut :
  **validé, intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-60 — TensorOps MoE dès 64 routes — rejeté

- Observation : le bundle cible M=8 produit 64 routes (`8 × top_k 8`), mais
  `Exl3SwitchGlu` ne sélectionne le QMM segmenté qu'à partir de 64 lignes, soit
  512 routes. Le kernel, son plan trié et ses buffers sont dimensionnés en
  slots/routes ; le seuil de lignes peut donc priver la vérification du chemin
  TensorOps existant.
- Hypothèse : choisir le chemin segmenté lorsque `rows × top_k >= 64` partage
  mieux les poids experts et réduit les QMV, sans nouveau kernel. Baseline :
  PERF-59 **84,588 ms** groupé contre 85,906 ms série dans sa première ABBA,
  puis 88,248 contre 89,050 ms sous chauffe, modèle Qwen3.6-35B-A3B EXL3
  2.49 bpw sur M5, batterie.
- Protocole : modifier uniquement le seuil, exiger l'identité octet par octet
  des logits FP16 et des 80 états M=1/2/4/8 avant toute mesure, puis deux séries
  ABBA. Rejet immédiat au premier écart, crash ou gain non stable. Statut :
  **en cours**, journalisé avant code.
- Résultat : **rejeté avant timing**. M=1/2/4 reste sur le chemin canonique,
  mais M=8 diverge massivement dans les logits FP16 dès que les 64 routes passent
  par le QMM TensorOps segmenté. Ce chemin ne peut donc pas remplacer le QMV
  mappé dans la vérification lossless, même si sa géométrie accepte les buffers.
  Le seuil `rows >= 64` est restauré ; aucune modification exécutable conservée.

### OPT-2026-09-20-RUST-PERF-61 — routeur dense M=8 en un dispatch — validé

- Observation : après PERF-59, chaque couche MoE conserve huit appels séparés
  à la projection dense FP16 `[2048, 256]` afin de reproduire exactement le
  résultat M=1 ; seuls softmax et top-k sont regroupés. Sur les 40 couches, cela
  laisse 320 petits dispatchs de gate dans une vérification de huit tokens.
- Hypothèse : un kernel Metal à grille 2D peut calculer les huit lignes dans un
  seul dispatch tout en gardant, pour chaque ligne, le même ordre de réduction
  que le GEMV M=1. Le gain visé est au moins 1 ms par bundle sans aucun écart de
  logits, routes ou états. Baseline : PERF-59, **84,588 ms** dans la première
  ABBA et **88,248 ms** sous chauffe, modèle Qwen3.6-35B-A3B EXL3 2.49 bpw sur
  M5, batterie.
- Protocole : tester d'abord les sorties brutes du gate contre huit appels M=1,
  puis exiger l'identité octet par octet des logits et 80 états pour M=1/2/4/8.
  Deux séries ABBA seulement après identité complète ; supprimer le prototype
  au premier écart non corrigeable sans chemin spécial fragile. Statut :
  **en cours**, journalisé avant code.
- Incident de validation : la première commande s'est arrêtée avant compilation,
  car `cargo` n'était pas dans le `PATH` non interactif. Aucun test n'a été
  exécuté ; reprise avec `/Users/justin/.cargo/bin/cargo` explicite.
- Deuxième incident de commande : le manifest est à la racine, pas dans
  `native/`; `cargo fmt --manifest-path native/Cargo.toml` a donc échoué avant
  compilation. Reprise depuis la racine avec `Cargo.toml`.
- Troisième incident de commande : le filtre qualifié
  `qwen35::tests::verification_widths_match_token_major` a compilé le crate mais
  sélectionné **0 test** (17 tests de lib filtrés). Il ne constitue donc aucune
  validation ; reprise avec le nom court découvert dans la liste Cargo.
- La reprise avec `--features mlx` a correctement sélectionné le code GPU mais
  le build script s'est arrêté avant compilation faute de `MLXL3_MLX_ROOT`.
  Reprise avec le paquet local `.venv/lib/python3.12/site-packages/mlx` déjà
  utilisé par les builds de l'app.
- Premier build du prototype avec l'environnement MLX correct : échec de
  compilation Rust avant exécution (`array::metal_kernel` non importé dans
  `lfm2.rs`). Correction limitée à l'import du module, puis même test relancé.
- Premier test GPU exécuté : échec à M=1 avant comparaison, car le chemin de
  vérification appelle aussi le helper réservé aux batches 2–8. Aucun résultat
  numérique obtenu. Le chemin M=1 reste sur `Projection::forward`; le kernel
  candidat n'est appelé que pour M≥2.
- Résultat du premier différentiel complet : M=1/2/4 identiques, mais **M=8
  diverge fortement** dans les logits finaux. Aucun timing conservé. Le kernel
  ne peut pas être intégré tel quel ; analyse réduite aux sorties brutes du gate
  avant toute autre modification de production.
- Test isolé sur un tenseur déterministe `[8,2048]` : sorties du gate strictement
  identiques pour M=2/4/8. Sur les activations réelles du différentiel M=8, le
  premier gate fautif ne diffère que sur **1 valeur FP16 / 2 048** ; cette unique
  différence suffit à changer une route proche de la frontière. Le défaut vient
  donc de l'ordre de réduction du GEMV MLX sur certaines valeurs, pas de
  l'adressage 2D du kernel.
- Révision : le kernel reproduit maintenant la géométrie exacte du GEMV M=1 de
  MLX 0.32.2 (`BM=4, BN=1, SM=1, SN=32, TM=4, TN=4`) tout en portant les
  lignes proposées sur la seconde dimension de grille. Le test gate isolé et le
  différentiel physique complet passent désormais pour M=1/2/4/8 : logits FP16
  et 80 états strictement identiques. L'instrumentation comparative temporaire
  est retirée avant benchmark.
- Contrôle avant ABBA : `cargo fmt --check` a demandé deux reformattages Rust
  purement mécaniques ; aucun benchmark n'a été lancé avec ce diff non formaté.
- Première ABBA physique, six paires alternées, sortie et états comparés à
  chaque passe : série **90,240 ms**, gate groupé **87,123 ms**, soit **1,036×**
  et **−3,117 ms** pour huit tokens. Échantillons série `[84,713; 87,950;
  89,542; 90,240; 90,733; 90,805]` ms, groupé `[84,268; 84,398; 86,908;
  87,123; 87,561; 89,121]` ms. Une seconde ABBA est requise sous la chauffe
  courante avant décision.
- Deuxième ABBA sous chauffe : série **87,123 ms**, gate groupé **86,425 ms**,
  soit **1,008×** et **−0,698 ms**. Échantillons série `[85,635; 86,768;
  86,786; 87,123; 87,626; 89,252]` ms, groupé `[85,098; 85,908; 86,191;
  86,425; 86,555; 87,384]` ms. Les deux séries sont positives et toutes leurs
  sorties/états sont identiques. Le benchmark temporaire est retiré ; le test
  gate M=2/4/8 et le différentiel modèle M=1/2/4/8 restent. Statut : **validé,
  intégré localement**.
- Contrôles finaux réussis : format, Clippy strict tous targets/features,
  23 tests lib, 1 test CLI, 14 contrats, build release MLX/chat, différentiel
  gate M=2/4/8 et différentiel modèle M=1/2/4/8. Kani 0.68 / CBMC 6.11
  vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures. Ces propriétés
  bornées concernent le Rust pur ; Kani ne couvre ni MLX ni Metal, vérifiés ici
  uniquement par les différentiels physiques exacts. Statut : **validé,
  intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-62 — profil cible après gate batchée — diagnostic

- Objectif : localiser le coût résiduel du bundle exact M=8 avant toute nouvelle
  optimisation. PERF-61 mesure **86,425 ms** médian sous chauffe pour la cible,
  auxquels s'ajoutent environ **5,39 ms** pour le draft ; le plafond à 100 %
  d'acceptation est donc environ **87,1 tok/s**, pas 92 tok/s.
- Protocole prévu : sur le Qwen3.6-35B-A3B EXL3 2.49 bpw local, séparer par
  synchronisation physique le corps 40 couches de la projection `lm_head`, puis
  mesurer le `lm_head` seul sur le même hidden évalué. Une instrumentation de
  test temporaire sera supprimée après diagnostic. Conditions : M5 sur batterie,
  applications utilisateur actives et chauffe non contrôlée ; les temps servent
  à classer les goulots, pas à annoncer un gain. Statut : **en cours**.
- Mesure, sept répétitions : corps des 40 couches **71,992 ms** médian
  `[71,657; 71,824; 71,988; 71,992; 72,044; 72,161; 75,689]`, `lm_head`
  EXL3 M=8 seul **13,568 ms** médian `[13,452; 13,495; 13,529; 13,568;
  13,648; 13,660; 14,481]`. Le corps représente donc environ 84 % du temps
  séparé et reste la priorité, mais le head pèse encore environ 16 %.
- Aucun gain revendiqué : les synchronisations ajoutées changent la frontière
  de graphe. L'instrumentation temporaire est retirée. Statut : **diagnostic
  terminé**, données conservées pour sélectionner PERF-63.

### OPT-2026-09-20-RUST-PERF-63 — profil par famille de couche M=8 — diagnostic

- Hypothèse de diagnostic : parmi les **71,992 ms** du corps PERF-62, les 30
  couches Gated DeltaNet et les 10 couches attention n'ont pas le même coût ;
  choisir sans mesure entre récurrence, attention et MoE risquerait de refaire
  une piste déjà rejetée. Aucun changement de production prévu à ce stade.
- Protocole : une passe temporaire avec synchronisation après chaque couche,
  classement séparé des couches linéaires et attention, même contexte de trois
  tokens et bundle exact M=8. Les barrières ajoutent du coût et interdisent de
  sommer ces durées avec PERF-62 ; elles servent seulement à comparer les deux
  familles. Instrumentation retirée immédiatement après mesure. Statut :
  **en cours**.
- Résultat : médiane par couche avec barrière, **3,838 ms** pour les 30 couches
  Gated DeltaNet contre **3,324 ms** pour les 10 couches attention. Plages
  observées respectives **3,365–4,498 ms** et **3,064–4,186 ms**. Les couches
  linéaires sont un peu plus chères, mais les plages se recouvrent fortement,
  ce qui indique que leur MoE commun reste probablement majoritaire.
- Aucun gain revendiqué et aucune somme avec les 71,992 ms de PERF-62 : les 40
  barrières doublent presque le temps. Instrumentation retirée. Statut :
  **diagnostic terminé**.

### OPT-2026-09-20-RUST-PERF-64 — profil attention/GDN contre MoE — diagnostic

- Hypothèse de diagnostic : le coût commun aux deux familles vient surtout du
  MoE exact M=8 ; mesurer séparément sous-couche attention/GDN et MLP sur chaque
  couche permet de décider entre un nouveau kernel récurrent et une réduction
  de dispatch experts. Même modèle, contexte et bundle que PERF-63, avec
  synchronisations temporaires aux frontières. Les valeurs absolues ne seront
  pas additionnées au débit sans barrières. Statut : **en cours**.
- Résultat médian par couche : Gated DeltaNet **1,208 ms**, attention complète
  **0,878 ms**, MLP des couches linéaires **2,359 ms**, MLP des couches attention
  **2,354 ms**. Le MoE commun représente environ deux tiers du temps d'une
  couche synchronisée et ne dépend pratiquement pas de sa famille ; c'est le
  prochain goulot à décomposer. Aucun gain revendiqué, instrumentation retirée.
  Statut : **diagnostic terminé**.

### OPT-2026-09-20-RUST-PERF-65 — profil interne du MoE exact M=8 — diagnostic

- Hypothèse de diagnostic : après les gates batchées de PERF-61, les 64 routes
  d'experts sparse ou l'expert partagé doivent dominer le MoE à **~2,36 ms** par
  couche. Mesurer séparément routeur, experts sparse et expert partagé sur les
  40 couches, avec synchronisation temporaire, orientera le prochain kernel.
  Même modèle/contexte/bundle ; aucune somme avec le chemin sans barrières et
  aucune modification de production avant résultat. Statut : **en cours**.
- Résultat médian par couche avec barrières : routeur gate/softmax/top-k
  **0,210 ms**, experts sparse gate/up/down **0,857 ms**, expert partagé
  **0,289 ms**. Les routes sparse représentent environ 63 % de ces trois
  sous-blocs mesurés et sont le seul prochain candidat assez lourd.
- Aucun gain revendiqué ; instrumentation supprimée. Statut : **diagnostic
  terminé**.

### OPT-2026-09-20-RUST-PERF-66 — quatre tiles par threadgroup expert M=8 — validé

- Observation : le chemin mapped des 64 routes utilise actuellement `NT=2`
  dès que le nombre global de tiles dépasse 1 024, donc 2 048 threadgroups pour
  gate/up et 4 096 pour down à chaque couche. Les sorties tiles sont
  indépendantes ; `NT=4` divise ces lancements par deux mais double les
  accumulateurs par thread et peut réduire l'occupation.
- Hypothèse : réserver `NT=4` aux mapped-QMV ayant au moins 64 routes sur M5
  réduit le sous-bloc sparse sans changer une opération, un poids ou l'ordre de
  réduction d'une sortie. Baseline cible complète PERF-61 : **86,425 ms** sous
  chauffe ; sous-bloc sparse PERF-65 : **0,857 ms** médian par couche avec
  barrières. Protocole : exactitude M=1/2/4/8, puis deux ABBA cible complète ;
  rejet au premier écart ou si le gain n'est pas reproductible. Statut :
  **en cours**.
- Exactitude : différentiel modèle M=1/2/4/8 réussi, avec logits FP16 et 80
  états octet par octet identiques au chemin token-major. Les deux ABBA
  comparent aussi chaque sortie et état entre NT2 et NT4.
- Première ABBA : NT2 **86,462 ms** contre NT4 **81,168 ms**, soit **1,065×**
  et **−5,294 ms**. Échantillons NT2 `[84,808; 85,233; 85,308; 86,462;
  89,155; 89,592]`, NT4 `[80,445; 80,807; 80,865; 81,168; 83,006;
  84,597]` ms.
- Deuxième ABBA sous chauffe : NT2 **84,944 ms** contre NT4 **82,508 ms**,
  soit **1,030×** et **−2,436 ms**. Échantillons NT2 `[83,909; 83,915;
  84,129; 84,944; 87,386; 87,838]`, NT4 `[81,444; 81,621; 81,747;
  82,508; 85,086; 131,711]` ms ; le dernier outlier candidat est conservé et
  n'inverse pas la médiane.
- Décision : **validé, intégré localement**. Le commutateur AB temporaire est
  retiré ; la règle production reste minimale (`M5`, au moins 64 lignes mapped,
  largeur divisible par quatre tiles).
- Contrôles finaux réussis : format, Clippy strict tous targets/features,
  23 tests lib, 1 test CLI, 14 contrats, build release MLX/chat et différentiel
  physique M=1/2/4/8. Kani 0.68 / CBMC 6.11 vérifie **17/17 harnesses**,
  zéro échec et 2/2 couvertures. Ses propriétés bornées ne couvrent pas MLX ou
  Metal ; le chemin GPU est couvert par le différentiel modèle exact, pas par
  une preuve formelle. Statut : **validé, intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-67 — huit tiles par threadgroup expert M=8 — rejeté

- Hypothèse : après le gain NT4 de PERF-66, `NT=8` divise encore par deux les
  threadgroups sparse. Il porte toutefois 64 accumulateurs FP32 par thread et
  peut faire chuter l'occupation ou provoquer du spill ; ce risque impose un
  essai isolé plutôt qu'une généralisation.
- Baseline appariée : production NT4 de PERF-66, **81,168–82,508 ms** médian
  suivant la série pour la cible complète M=8. Protocole : NT4/NT8 alternés sur
  le même modèle et état, égalité stricte des logits/80 états à chaque passe,
  deux séries seulement si la première gagne. Rejet au premier écart, erreur
  Metal ou médiane non meilleure. Statut : **en cours**.
- Résultat : exact mais nettement plus lent. NT4 **82,843 ms** médian
  `[82,194; 82,554; 82,827; 82,843; 83,781; 83,909]` contre NT8
  **111,210 ms** `[109,696; 109,884; 110,480; 111,210; 112,235; 601,130]`,
  soit une régression médiane de **25,5 %** et un outlier extrême. Le surcoût
  d'accumulateurs/pression registres domine la réduction de threadgroups.
- Décision : **rejeté après une série**, comme prévu par le protocole ; NT8,
  son commutateur et son benchmark temporaire sont retirés. Production reste
  sur NT4 de PERF-66.

### OPT-2026-09-20-RUST-PERF-68 — épilogue sparse Hadamard/réduction fusionné — rejeté

- Observation : après chaque down sparse, `finish_and_reduce` matérialise un
  Hadamard FP16 de 64×2048, deux multiplications puis une somme top-8. PERF-65
  inclut cette chaîne dans les **0,857 ms** sparse, mais PERF-09 n'avait testé
  qu'une compilation MLX explicite, pas un kernel Metal fusionné.
- Hypothèse : un threadgroup par bloc de 128 sorties peut reproduire le radix
  16 puis radix 8 du Hadamard MLX, ses arrondis FP16, les scales/scores et la
  somme top-8 en un dispatch, sans buffer Hadamard intermédiaire. Périmètre
  initial : mapped sparse avec au moins 64 slots ; chemins decode M=1 et prefill
  segmenté inchangés.
- Baseline : production NT4 PERF-66, cible M=8 **~81–83 ms**. Protocole : rejet
  immédiat si logits/80 états M=1/2/4/8 diffèrent ; sinon deux ABBA complètes
  fusion/référence. Le kernel est retiré si l'ordre FP16 exact n'est pas
  reproductible ou si les deux médianes ne gagnent pas. Statut : **en cours**.
- Premier différentiel modèle : M=1/2/4 passent par la référence, M=8 diverge
  fortement dans les logits. Aucun timing lancé. Le kernel n'est pas éligible
  en l'état ; un différentiel isolé de l'épilogue doit déterminer si l'écart
  vient du Hadamard ou de la somme FP16 avant décision finale.
- Le premier différentiel isolé confirme une erreur dans le Hadamard : même avec
  `top_k=1`, **435/8 192** valeurs diffèrent ; avec scales/scores unitaires,
  **500/8 192** diffèrent. La suite de butterflies radix-2 ne reproduit donc
  pas les arrondis du kernel MLX, malgré des résultats proches.
- Révision minimale : le prototype reproduit ensuite exactement le radix-16,
  cast FP16 intermédiaire puis radix-8 de MLX avec huit threads. Le Hadamard
  seul devient exact (**0/8 192** écarts), tout comme l'épilogue production
  top-8 (**0/1 024** écart) ; le différentiel complet logits + 80 états
  M=1/2/4/8 réussit aussi octet par octet.
- ABBA cible complète : référence **83,459 ms** médiane
  `[81,752; 82,383; 83,277; 83,459; 83,673; 88,411]` contre fusion
  **84,892 ms** `[83,243; 84,102; 84,408; 84,892; 85,400; 86,358]`, soit
  **−1,7 %**. Le faible nombre de threads et les registres nécessaires aux
  16 sorties annulent les dispatchs/buffers économisés.
- Décision : **rejeté après une série**, conformément au protocole. Kernel,
  commutateur et tests temporaires supprimés ; production reste strictement
  identique à PERF-66. Le résultat négatif établit aussi qu'une fusion exacte
  doit préserver le découpage radix-16/radix-8 de MLX, pas sept butterflies
  radix-2 génériques.

### OPT-2026-09-20-RUST-PERF-69 — NT4 QMV batch pour grandes sorties — en cours

- Observation : PERF-62 mesure encore **13,568 ms** pour le `lm_head` M=8,
  soit près de 16 % de la cible. Son QMV batch 2048→248 320 utilise `NT=2`
  dès 1 024 tiles, alors que PERF-66 démontre que `NT=4` gagne sur les QMV
  mapped M=8 en divisant les threadgroups, sans changer l'accumulation interne.
- Hypothèse : appliquer `NT=4` uniquement au QMV batch M=8 et aux sorties dont
  le nombre de tiles est divisible par quatre réduit surtout le coût du
  `lm_head`; M=1 et les petits batches restent inchangés. Aucun nouveau kernel,
  seulement la géométrie déjà validée `NT=4`.
- Baseline : production PERF-66, cible complète M=8 **~81–83 ms** ; `lm_head`
  diagnostique **13,568 ms**. Protocole : identité stricte logits + 80 états
  M=1/2/4/8, puis deux ABBA complètes NT2/NT4 si la première gagne. Rejet au
  premier écart Metal/numérique ou si la médiane n'est pas meilleure. Statut :
  **en cours**, journalisé avant code.
- Exactitude : différentiel physique M=1/2/4/8 réussi, logits FP16 et 80 états
  octet par octet identiques au chemin token-major. Les 24 passages ABBA
  comparent également sortie et état entre NT2 et NT4.
- Première ABBA : NT2 **82,504 ms** `[81,448; 82,202; 82,417; 82,504;
  82,823; 88,334]` contre NT4 **80,339 ms** `[79,315; 80,034; 80,127;
  80,339; 80,895; 81,314]`, soit **1,027×** et **−2,165 ms**.
- Deuxième ABBA : NT2 **83,625 ms** `[82,150; 83,268; 83,488; 83,625;
  83,779; 83,877]` contre NT4 **80,526 ms** `[79,753; 80,149; 80,379;
  80,526; 80,622; 81,686]`, soit **1,038×** et **−3,098 ms**.
- Décision : **validé et intégré localement** sur M5 pour `matrix_rows >= 8`
  et largeur divisible par quatre tiles. M=1 et les autres GPU gardent leur
  géométrie précédente. Commutateur et benchmark temporaires supprimés ;
  contrôles complets/Kani requis avant publication.
- Contrôles finaux réussis : format, `git diff --check`, Clippy strict tous
  targets/features, 23 tests lib, 1 test CLI, 14 contrats, build release
  MLX/chat et différentiel physique M=1/2/4/8. Kani 0.68 / CBMC 6.11 vérifie
  **17/17 harnesses**, zéro échec et 2/2 couvertures. Les propriétés Kani
  portent sur le Rust pur ; la géométrie MLX/Metal est couverte par le
  différentiel exact du modèle réel, pas par une preuve formelle. Statut :
  **validé, intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-70 — séquence GDN exacte en un kernel — validé

- Observation : `GatedDelta::forward_verification` batch déjà ses projections,
  mais exécute encore huit convolutions mono-token et huit appels récurrents
  `step_with_gates` par couche. Le shader de production contient pourtant une
  boucle temporelle `T` et indexe déjà q/k/v/a/b à chaque pas ; seule la garde
  Rust le limite artificiellement à `T=1`.
- Hypothèse : produire la convolution causale des huit lignes en une opération,
  puis appeler le même shader gates fusionnées avec `T<=8`, conserve exactement
  l'ordre récurrent interne et supprime 14 dispatchs par couche GDN. Les calculs
  de gate et d'état restent identiques ; aucune spéculation ni approximation.
- Baseline : PERF-69 **80,339–80,526 ms** M=8 ; diagnostic PERF-64 : bloc GDN
  **1,208 ms** par couche avec barrières. Protocole : différentiel strict
  logits + 80 états M=1/2/4/8 avant tout timing, puis deux ABBA si exact et plus
  rapide. Rejet immédiat au premier écart ; chemins decode M=1 et prefill >8
  inchangés. Statut : **en cours**, journalisé avant code.
- Correctif durant l'essai : le premier prototype compilait encore le shader
  avec `T=1`, alors que les buffers contenaient plusieurs lignes ; M=2 a donc
  échoué nettement. Le header et la clé de cache Metal incluent désormais le
  vrai `T`. Après ce correctif de géométrie, le différentiel physique complet
  M=1/2/4/8 réussit : logits FP16 et 80 états identiques octet par octet.
- Première ABBA M=8 : série mono-token **81,455 ms** médiane
  `[79,857; 80,279; 80,711; 81,455; 88,073; 89,369]` contre séquence fusionnée
  **75,632 ms** `[71,448; 73,194; 75,118; 75,632; 77,829; 81,426]`, soit
  **1,077×**.
- Deuxième ABBA M=8 : série mono-token **84,435 ms** médiane
  `[78,691; 78,705; 81,864; 84,435; 86,234; 90,731]` contre séquence fusionnée
  **72,540 ms** `[71,769; 71,902; 72,067; 72,540; 77,873; 80,673]`, soit
  **1,164×**. Chaque passage compare les logits et les 80 états exactement.
- Décision : **validé et intégré localement** pour la vérification exacte
  `T<=8`. Le benchmark et le commutateur série temporaires sont retirés. À
  titre de borne, 72,540 ms de target plus 5,39 ms de draft donneraient
  **~102,7 tok/s** si les huit propositions étaient toujours acceptées ; ce
  n'est pas une mesure bout en bout et ne préjuge ni de l'acceptation réelle,
  ni de l'orchestration/rollback/streaming. Contrôles complets et Kani requis
  avant publication.
- Contrôles finaux réussis : format, `git diff --check`, Clippy strict tous
  targets/features, 23 tests lib, 1 test CLI, 14 contrats, build release
  MLX/chat et différentiel physique M=1/2/4/8. Kani 0.68 / CBMC 6.11 vérifie
  **17/17 harnesses**, zéro échec et 2/2 couvertures. Kani couvre le Rust pur ;
  le shader Metal et la géométrie temporelle sont couverts par le différentiel
  physique exact, pas par une preuve formelle. Statut : **validé, intégré et
  prêt à publier**.

### OPT-2026-09-20-RUST-PERF-71 — débit DFlash2 bout en bout — validé, activation rejetée

- Observation : le dépôt possède le lecteur du package Splash, le draft Q4,
  le sélecteur et la vérification cible exacte, mais `main.rs` génère encore
  exclusivement token par token. Le microbenchmark draft à contexte vide et
  la borne calculée de PERF-70 ne mesurent donc pas un DFlash2 fonctionnel.
- Hypothèse : raccorder le conditionnement par captures cible, la proposition
  DFlash2 officielle, la vérification exacte, l'acceptation greedy, le commit /
  rollback et le streaming permet de mesurer le débit réellement délivré sans
  modifier la sortie greedy. La version minimale réutilisera les structures et
  kernels existants ainsi que la sémantique de référence amont ; aucun nouveau
  framework ni chemin approximatif.
- Baselines connues, non comparables comme résultat final : decode ordinaire
  Qwen **~43,719 tok/s** (ancienne session PERF-31), draft partiel
  **~4,621 ms** (contexte vide), target M=8 PERF-70 **72,540–75,632 ms**.
  Protocole : prompts fixes greedy, warmup, alternance DFlash désactivé/activé,
  au moins trois générations utiles ; mesurer tokens émis / temps decode,
  acceptation par position, coût draft/target/rollback, TTFT et mémoire. La
  séquence DFlash doit être exactement identique au greedy ordinaire. Statut :
  **en cours**, journalisé avant raccord et benchmark.
- Premier smoke physique (12 tokens, 1 répétition) : compilation Rust/Metal
  réussie et exécution arrivée jusqu'au verify cible, puis arrêt contrôlé avant
  mesure avec `invalid DFlash target logits`. La tête renvoie légitimement
  `[1,T,V]` alors que le nouveau row-argmax validait seulement `[T,V]`.
  Statut : **échec diagnostique, corrigé localement** en normalisant toute forme
  dont la dernière dimension vaut `V` et qui contient 1 à 8 lignes ; aucune
  métrique de ce passage n'est retenue.
- Le smoke corrigé est exact sur 12 tokens : greedy **46,487 tok/s**, DFlash2
  **57,283 tok/s**, 14/14 propositions acceptées, soit **1,232×**. Cette série
  reste trop courte pour conclure et n'est pas retenue comme chiffre principal.
- Mesure utile, même prompt anglais, greedy déterministe, 48 tokens, warmup puis
  trois alternances greedy/DFlash2, Mac M5 :
  - passage 0 : **46,947 / 37,201 tok/s**, 39/63 acceptées (**61,9 %**),
    draft 0,160 s, target 0,690 s, rollback 0,412 s ;
  - passage 1 : **46,995 / 44,482 tok/s**, 39/63 (**61,9 %**), draft
    0,177 s, target 0,752 s, rollback 0,127 s ;
  - passage 2 : **41,085 / 41,570 tok/s**, 39/63 (**61,9 %**), draft
    0,189 s, target 0,808 s, rollback 0,132 s.
- Médianes : greedy **46,947 tok/s**, DFlash2 **41,570 tok/s**, soit
  **0,885×** (**−11,5 %**). Chaque passage compare toute la séquence de 48
  tokens ; aucun écart n'a été observé. Le prefill capturé vaut 0,152–0,177 s
  et la matérialisation initiale du contexte draft 0,007–0,010 s sur ce prompt.
- Interprétation : la vérification M=8 est bien accélérée, mais à 61,9 %
  d'acceptation le snapshot/restore puis recalcul exact du préfixe retenu coûte
  0,127–0,132 s après échauffement. Il faut un commit sélectif des états
  GDN/KV, sans second passage cible, avant d'espérer un gain bout en bout.
  La borne parfaite **~102,7 tok/s** de PERF-70 n'est donc pas représentative.
- Décision : **validé comme benchmark lossless, rejeté pour activation
  production**. Le raccord minimal et le test physique restent disponibles
  pour la prochaine optimisation du commit sélectif ; CLI/GUI continuent le
  greedy ordinaire. Statut d'intégration : code local, contrôles finaux requis
  avant publication.

### OPT-2026-09-20-RUST-PERF-72 — commit sélectif DFlash2 sans recalcul — validé

- Observation : PERF-71 mesure DFlash2 à **41,570 tok/s** contre greedy
  **46,947 tok/s**, avec 61,9 % d'acceptation. Après échauffement, le second
  passage cible nécessaire au rollback/recommit coûte encore 0,127–0,132 s
  sur neuf blocs et transforme le gain de vérification batchée en régression.
- Hypothèse : conserver, uniquement pendant le verify DFlash, l'état GDN exact
  après chacune des huit positions puis tronquer les KV/conv à la longueur
  acceptée permet de committer le préfixe sans restaurer ni recalculer le
  target. Le chemin greedy et le verify ordinaire ne doivent payer aucun buffer
  d'historique supplémentaire.
- Baseline/protocole : même prompt et mêmes 48 tokens que PERF-71, warmup puis
  trois alternances greedy/DFlash2. Exiger l'égalité de toute la séquence et de
  l'état committé avec un recalcul de référence pour chaque largeur 1..8 avant
  timing. Mesurer tok/s, acceptation, draft, target et coût de commit ; rejeter
  au premier écart. Statut initial : **en cours**, journalisé avant code.
- Implémentation : le verify DFlash conserve l'état GDN FP32 exact après chaque
  position et la base des caches attention. Le commit garde directement l'état
  retenu, tronque KV et convolution puis corrige l'offset. Le chemin greedy et
  le verify non DFlash ne créent aucun historique supplémentaire.
- Contrôle d'état physique : pour chaque largeur retenue de 1 à 8, les **80
  tableaux d'état** du vrai Qwen cible sont identiques au recalcul exact du
  préfixe (`selective_dflash_commit_matches_exact_prefix`). Aucun écart.
- Mesure bout en bout, trois alternances, 48 tokens exacts :
  - passage 0 : greedy **48,405 tok/s**, DFlash2 **54,456 tok/s**, 39/63
    propositions acceptées (**61,9 %**), draft 0,160 s, target 0,702 s,
    commit 0,001 s ;
  - passage 1 : **48,325 / 54,170 tok/s**, 39/63, draft 0,159 s, target
    0,708 s, commit 0,001 s ;
  - passage 2 : **48,293 / 54,619 tok/s**, 39/63, draft 0,158 s, target
    0,702 s, commit 0,001 s.
- Médianes : greedy **48,325 tok/s**, DFlash2 **54,456 tok/s**, soit
  **1,127× (+12,7 %)**. Les trois séquences complètes sont strictement
  identiques. Le prefill capturé vaut 0,152–0,155 s et la construction initiale
  du contexte draft 0,008 s. Le coût de commit remplace le rollback/recalcul de
  0,127–0,132 s de PERF-71 par environ **0,001 s**.
- Décision : **validé et intégré localement** dans le chemin de benchmark exact.
  Le gain est réel mais n'atteint pas 100 tok/s avec cette acceptation ; le
  CLI/GUI n'activent pas encore automatiquement DFlash2. Contrôles finaux et
  Kani requis avant publication.
- Contrôles finaux réussis : `git diff --check`, format Rust, Clippy strict
  tous targets/features, **23 tests lib + 1 CLI + 14 contrats**, build release
  MLX/chat, différentiel physique des 80 états pour les largeurs 1..8 et trois
  passages physiques bout en bout. Kani 0.68 / CBMC 6.11 vérifie **17/17
  harnesses**, zéro échec et 2/2 couvertures. Kani couvre le Rust pur ; les
  kernels Metal et l'état MLX sont couverts par les différentiels physiques,
  pas par une preuve formelle complète. Statut : **validé et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-73 — largeur de vérification DFlash — validé

- Observation : PERF-72 atteint **54,456 tok/s** contre **48,325 tok/s**
  (`+12,7 %`), mais le target verify M=8 domine encore à environ **0,702 s**
  sur neuf blocs, contre 0,160 s pour le draft, avec seulement **61,9 %** des
  propositions acceptées. Le sélecteur greedy a été comparé au code MLX
  officiel DFlash2 : top-k, score unary + transition et marche depuis l'ancre
  sont identiques ; aucune correction évidente n'y est disponible.
- Hypothèse : limiter la vérification exacte aux N premières propositions peut
  réduire le coût target lorsque les chaînes longues cassent tôt. La doc MLX
  DFlash recommande aussi des blocs `<=5` avec les matmuls quantifiés, mais ce
  conseil n'est qu'un indice : MLXL3 emploie ses propres kernels EXL3 et doit
  être mesuré physiquement.
- Protocole prévu : ajouter uniquement un réglage de benchmark, balayer N=1..7
  sur le même prompt, 48 tokens et greedy déterministe, puis confirmer les
  meilleurs N par trois alternances contre la baseline. Mesurer tok/s bout en
  bout, acceptation, blocs et temps draft/target/commit. Exiger l'égalité exacte
  de toute la séquence avec le greedy ordinaire ; rejeter toute largeur qui
  diverge. Aucun gain ne sera extrapolé au CLI/GUI. Statut : **en cours**,
  journalisé avant code.
- Balayage d'orientation, un passage de 32 tokens exacts par largeur : N=1
  **42,081 tok/s**, N=2 **52,827**, N=3 **56,103**, N=4 **51,624**, N=5
  **64,269**, N=6 **56,293**, N=7 **65,398**. Toutes les séquences égalent le
  greedy. Les baselines greedy ont varié de 42,892 à 49,526 tok/s pendant ce
  balayage ; ces chiffres courts ne suffisent donc pas à départager N=5 et
  N=7. Confirmation prévue sur 48 tokens et trois alternances pour ces deux
  largeurs. Statut intermédiaire : **non concluant**.
- Confirmation 48 tokens, trois alternances exactes : N=5 donne
  **59,116 tok/s** contre greedy **48,678 tok/s**, soit **1,214× (+21,4 %)**,
  avec 38/50 propositions acceptées (76,0 %), 10 blocs, draft 0,171–0,172 s,
  target 0,621–0,627 s et commit 0,001 s. N=7 donne une médiane de
  **52,182 tok/s** contre greedy **45,032 tok/s** sur une machine qui ralentit
  pendant la seconde série, avec 39/63 acceptées, 9 blocs et target
  0,704–0,766 s. Toutes les sorties sont identiques au greedy.
- Décision : **validé**. N=5 remplace N=7 comme largeur par défaut du benchmark
  exact ; le gain médian contre sa baseline passe de +12,7 % à **+21,4 %**.
  Il reste un chemin de benchmark et n'est pas annoncé comme actif dans le
  CLI/GUI.
- Revalidation finale après retrait de PERF-74, trois nouvelles alternances :
  greedy médian **47,545 tok/s**, DFlash **59,345 tok/s**, soit **1,248×
  (+24,8 %)**, toujours 38/50 acceptées et trois séquences exactes. Draft
  0,170–0,177 s, target 0,618–0,637 s, commit 0,001 s. Les deux campagnes
  indépendantes placent donc le gain validé entre **+21,4 % et +24,8 %**, avec
  un débit DFlash médian stable de 59,116–59,345 tok/s.
- Contrôles finaux : `git diff --check`, format Rust, Clippy strict tous
  targets/features, **23 tests lib + 1 CLI + 14 contrats**, build release,
  différentiel physique des 80 états puis trois passages physiques exacts.
  Kani 0.68 / CBMC 6.11 vérifie **17/17 harnesses**, zéro échec et 2/2
  couvertures. Kani ne couvre pas les kernels Metal/MLX ; ceux-ci restent
  vérifiés par les différentiels physiques bornés. Statut : **validé et prêt à
  publier**.

### OPT-2026-09-20-RUST-PERF-74 — sélecteur DFlash limité au préfixe — rejeté

- Observation : avec N=5, le draft calcule encore top-k, arêtes et marche du
  sélecteur pour sept positions alors que les deux dernières sont jetées. Les
  six couches draft restent M=8 car leur TensorOps est spécialisé, mais le
  sélecteur Metal peut borner ses grilles et sorties sans toucher aux poids ni
  à la qualité.
- Hypothèse : compiler le sélecteur avec le nombre de propositions réellement
  vérifié réduit un peu le temps draft, sans modifier les cinq premiers choix
  car la marche greedy est causale.
- Protocole : même prompt, N=5, 48 tokens, trois alternances ; égalité stricte
  avec greedy, comparaison aux **59,116 tok/s** et 0,171 s draft de PERF-73.
  Rejeter si le débit ne progresse pas au-delà du bruit ou si un token diverge.
  Statut initial : **en cours**, journalisé avant code.
- Résultat : sorties exactes, mais draft **0,173–0,181 s** et débit médian
  **57,570 tok/s** sur une machine ralentie, contre 0,171–0,172 s et
  59,116 tok/s avant changement. La réduction des grilles de sélection est
  noyée par la compilation/ordonnancement et n'apporte aucun gain mesurable.
- Décision : **rejeté** ; le sélecteur limité a été retiré. Seule la largeur
  de vérification N=5 de PERF-73 est conservée.

### OPT-2026-09-20-RUST-PERF-75 — NT4 grandes projections au verify M=6 — rejeté

- Observation : PERF-73 retient cinq propositions, donc six lignes target.
  `Exl3Linear::forward_qmv_batch` n'active pourtant le parcours NT4 M5 qu'à
  partir de huit lignes ; le `lm_head` 2048→248320 repasse ainsi sur NT2.
  PERF-69 avait validé NT4 à M=8 avec −2,2 à −3,1 ms par verify exact.
- Hypothèse : étendre uniquement ce seuil M5 de 8 à 6 conserve l'arithmétique
  de chaque sortie et réduit le `lm_head` M=6. Aucun changement de draft,
  d'acceptation ou de sampling.
- Baseline : PERF-73 revalidé à greedy **47,545 tok/s**, DFlash N=5
  **59,345 tok/s**, draft 0,170–0,177 s, target 0,618–0,637 s sur dix blocs.
  Protocole : ajouter M=6 au différentiel exact target, puis trois alternances
  E2E de 48 tokens. Rejeter au premier écart de logits/80 états ou si le temps
  target ne baisse pas au-delà du bruit. Statut : **en cours**, journalisé
  avant code.
- Exactitude : le différentiel physique M=1/2/4/6/8 conserve les logits FP16
  et les 80 états octet par octet.
- Mesure E2E, trois alternances de 48 tokens : greedy **46,385 tok/s** médian,
  DFlash **58,676 tok/s** médian, acceptation **38/50 (76 %)**. Le temps target
  reste à **0,622–0,629 s** sur dix blocs, contre **0,618–0,637 s** avant le
  changement ; aucun gain ne dépasse le bruit, et le débit reste sous les
  **59,345 tok/s** de PERF-73.
- Décision : **rejeté**. Le seuil M=8 et la matrice de test initiale sont
  restaurés ; aucune modification exécutable de cet essai n'est conservée.

### OPT-2026-09-20-RUST-PERF-76 — NT4 du down MoE à 48 routes — rejeté

- Observation : avec cinq propositions, chaque couche MoE vérifie six tokens.
  Les `gate_proj`/`up_proj` traitent 96 lignes et utilisent déjà le chemin NT4,
  mais le `down_proj` traite exactement 48 routes et retombe sur NT2 à cause du
  seuil `rows >= 64` de `expert_mapped`.
- Hypothèse : autoriser le même kernel NT4 M5 dès 48 routes divise les groupes
  du down sparse sans changer l'accumulation d'une sortie. Les chemins M=1 et
  hors M5 restent inchangés.
- Baseline : PERF-75 revalide DFlash N=5 à **58,676 tok/s**, target
  **0,622–0,629 s**, draft **0,174–0,176 s**, acceptation **76 %**. Protocole :
  différentiel physique exact M=1/2/4/6/8, puis trois alternances E2E de
  48 tokens. Rejet au premier écart ou si le débit/temps target ne gagne pas
  au-delà du bruit. Statut : **en cours**, journalisé avant code.
- Exactitude : le différentiel physique M=1/2/4/6/8 conserve les logits FP16
  et les 80 états octet par octet.
- Mesure E2E, trois alternances de 48 tokens : greedy **43,783 tok/s** médian,
  DFlash **54,097 tok/s** médian, acceptation **38/50 (76 %)**. Le temps target
  régresse à **0,631–0,707 s** et le draft à 0,175–0,192 s sous chauffe ; même
  le meilleur passage reste inférieur à PERF-73.
- Décision : **rejeté**. Le seuil `rows >= 64` et la matrice de test initiale
  sont restaurés ; aucune modification exécutable de cet essai n'est conservée.

### OPT-2026-09-20-RUST-PERF-77 — NT4 des projections groupées au verify M=6 — rejeté

- Observation : `Exl3Group::forward_qmv_batch` conserve NT2 lorsque sa largeur
  concaténée dépasse 1 024 tiles, même sur M5 et six lignes, alors que les
  projections linéaires non groupées M=8 bénéficient déjà de NT4 (PERF-69).
- Hypothèse : sélectionner la géométrie NT4 existante uniquement sur M5,
  `matrix_rows >= 6` et largeur divisible par quatre réduit les projections
  groupées du verify sans toucher au decode M=1 ni à l'arithmétique d'une
  sortie.
- Baseline : production PERF-73 **59,345 tok/s** ; mesures chaudes récentes
  PERF-75 target **0,622–0,629 s** et PERF-76 meilleur target **0,631 s**.
  Protocole : différentiel physique exact M=1/2/4/6/8, puis trois alternances
  E2E de 48 tokens. Rejet au premier écart ou sans amélioration reproductible.
  Statut : **en cours**, journalisé avant code.
- Exactitude : le différentiel physique M=1/2/4/6/8 conserve les logits FP16
  et les 80 états octet par octet.
- Mesure E2E, trois alternances de 48 tokens : greedy **47,812 tok/s** médian,
  DFlash **57,427 tok/s** médian, acceptation **38/50 (76 %)**, target
  **0,636–0,640 s**. La cible est plus lente que PERF-75 et le débit reste sous
  PERF-73 malgré une baseline greedy revenue à son niveau normal.
- Décision : **rejeté**. La géométrie NT2 production et la matrice de test
  initiale sont restaurées ; aucune modification exécutable n'est conservée.

### OPT-2026-09-20-RUST-PERF-78 — sélection globale Viterbi DFlash — rejeté

- Observation : le sélecteur calcule déjà les 16 candidats, leurs scores
  unaires et toutes les transitions 16×16 pour sept positions, mais choisit
  ensuite gloutonnement le meilleur token local. Une erreur précoce réduit
  directement le nombre de tokens acceptés et force une nouvelle vérification
  target complète.
- Hypothèse : le chemin Viterbi maximisant la somme globale unary+transition
  sur les sept positions exploite les scores déjà calculés, sans nouveau
  dispatch ni modèle, et peut augmenter l'acceptation. L'inférence reste
  lossless : chaque proposition est toujours vérifiée par le modèle cible et
  le premier token divergent est rejeté.
- Baseline : PERF-73 **59,345 tok/s**, **38/50 (76 %)** acceptés, dix blocs,
  target **0,618–0,637 s** pour 48 tokens. Protocole : remplacer uniquement le
  choix final mono-thread, exiger l'égalité complète des tokens target E2E,
  puis trois alternances de 48 tokens. Rejet si l'acceptation ou le débit ne
  progresse pas de manière reproductible. Statut : **en cours**, journalisé
  avant code.
- Résultat E2E exact, trois alternances de 48 tokens : greedy cible
  **45,941 tok/s** médian, DFlash Viterbi **51,534 tok/s**, acceptation
  **37/55 (67,3 %)** et onze blocs. Le texte reste strictement identique au
  greedy target, mais le chemin global choisit moins bien les premiers tokens
  utiles que le sélecteur glouton et ajoute une vérification complète.
- Décision : **rejeté**. Le shader glouton et sa clé Metal d'origine sont
  restaurés ; aucune modification exécutable de cet essai n'est conservée.

### OPT-2026-09-20-RUST-PERF-79 — calibration du poids de transition — validé

- Observation : le sélecteur glouton additionne les logits unaires et le score
  de transition avec un poids implicite de 1. La quantification BF16/Q4 et le
  port Metal peuvent changer leur échelle relative ; le calcul des deux scores
  est déjà payé.
- Hypothèse : calibrer un unique coefficient global de transition augmente la
  probabilité que les premiers tokens proposés coïncident avec le modèle cible,
  sans coût significatif et sans modifier la validation lossless.
- Baseline : poids 1, PERF-73 **38/50 (76 %)**, dix blocs, **59,345 tok/s**.
  Protocole : matrice courte `{0; 0,25; 0,5; 0,75; 1; 1,25; 1,5; 2}` sur le
  même prompt et 48 tokens ; retenir seulement un candidat qui réduit les blocs
  ou améliore l'acceptation, puis le confirmer sur trois alternances exactes.
  Le commutateur de benchmark sera retiré après décision. Statut : **en cours**,
  journalisé avant code.
- Exploration, un passage exact par coefficient : poids 0 → **67,3 %** et
  51,111 tok/s ; 0,25 → **86,7 %** et 66,024 tok/s ; 0,5 → **86,7 %** et
  66,048 tok/s ; 0,75/1/1,25/1,5/2 → **76 %** et dix blocs. Seuls 0,25 et 0,5
  suppriment une vérification target entière sur ce corpus.
- Confirmation sur trois alternances exactes : poids 0,25, greedy cible
  **47,467 tok/s** médian et DFlash **63,716 tok/s**, soit **1,342×** ; neuf
  blocs, **39/45 (86,7 %)** acceptés, draft **0,157–0,162 s**, target
  **0,571–0,597 s**. Poids 0,5 confirme la même séquence et acceptation mais
  sa série plus chaude est moins stable (**60,760 tok/s** médian).
- Décision : **validé** avec le poids 0,25, codé comme constante Metal. Le
  commutateur d'exploration est retiré. Par rapport à PERF-73, le débit médian
  monte de 59,345 à 63,716 tok/s (**+7,4 %**) et le speedup sur le greedy du
  même passage monte de 1,248× à **1,342×**. La sortie reste strictement égale
  au greedy target ; contrôles complets et Kani requis avant publication.
- Revalidation après retrait du commutateur, Mac plus chaud : greedy
  **39,211 tok/s**, DFlash **52,451 tok/s**, soit **1,338×**, toujours neuf
  blocs et 86,7 % acceptés. La baisse absolue affecte les deux chemins ; le gain
  relatif est stable à 0,4 point de la série froide.
- Contrôles finaux réussis : `git diff --check`, format Rust, Clippy strict tous
  targets/features, 23 tests lib, 1 test CLI, 14 contrats, build release
  MLX/chat, E2E physique exact et rollback sélectif exact pour les largeurs
  retenues 1 à 8. Kani 0.68 / CBMC 6.11 vérifie **17/17 harnesses**, zéro
  échec et 2/2 couvertures. Kani couvre ici le Rust pur, pas le shader Metal ;
  le caractère lossless du chemin GPU est contrôlé par le différentiel physique
  et ne constitue pas une preuve formelle non bornée. Statut : **validé, intégré
  et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-80 — largeur verify après calibration — rejeté

- Observation : PERF-73 avait retenu cinq propositions avec le poids de
  transition 1. PERF-79 augmente l'acceptation à 86,7 % et peut déplacer
  l'optimum entre coût target par bloc et nombre de blocs.
- Hypothèse : une largeur de six ou sept propositions amortit mieux les neuf
  vérifications restantes et rapproche le débit de 1,5×, sans aucun changement
  de correction puisque le target vérifie toujours tout le bloc.
- Baseline : N=5, poids 0,25, **63,716 tok/s** froid / **52,451 tok/s** chaud,
  neuf blocs, 86,7 % acceptés et speedup **1,342× / 1,338×**. Protocole : balayage
  N=1..7 sur 48 tokens, puis trois alternances exactes du meilleur candidat.
  Rejeter toute largeur qui augmente les blocs ou n'améliore pas le ratio face
  au greedy du même passage. Statut : **en cours**, journalisé avant mesure.
- Balayage exact, un passage par largeur : N=1 **0,861×**, N=2 **1,060×**,
  N=3 **1,128×**, N=4 **1,255×**, N=5 **1,347×**, N=6 **1,108×** et N=7
  **1,137×**. N=5 conserve neuf blocs et 86,7 % ; N=6/7 tombent à
  63,3/61,9 % et augmentent fortement le temps target.
- Décision : **rejeté**, l'optimum N=5 de PERF-73 reste inchangé. Aucun code
  exécutable ni défaut de correction ; seul ce résultat négatif est conservé.

### OPT-2026-09-20-RUST-PERF-81 — calibration de transition par position — rejeté

- Observation : le poids global 0,25 enlève une vérification mais laisse six
  propositions rejetées sur 45. Le sélecteur produit sept positions avec des
  distributions différentes ; un poids unique peut corriger un rang tout en
  dégradant un autre.
- Hypothèse : identifier les rangs de première divergence puis calibrer le
  coefficient seulement à ces positions augmente l'acceptation jusqu'à huit
  blocs sans changer le coût du sélecteur ni la correction lossless.
- Baseline : PERF-79, N=5, **39/45 (86,7 %)**, neuf blocs, **1,342×** froid.
  Protocole : instrumenter temporairement le test E2E pour relever le rang de
  chaque première divergence, balayer uniquement les positions concernées,
  puis confirmer toute combinaison gagnante sur trois alternances exactes.
  Retirer toute instrumentation et rejeter si neuf blocs restent nécessaires.
  Statut : **en cours**, journalisé avant code.
- Diagnostic : la trace d'acceptation est stable à
  `[5,5,5,5,0,5,4,5,5]`. Les coefficients des positions 0 puis 4, balayés
  séparément sur `{0; 0,1; 0,25; 0,5; 0,75; 1; 1,5; 2}`, ne changent aucun
  token, aucun rang de divergence et aucun nombre de blocs.
- Décision : **rejeté**. Le tableau de poids et la trace temporaire sont retirés ;
  la constante globale 0,25 de PERF-79 reste en production. Le résultat suggère
  que les cibles divergentes ne figurent pas dans les 16 candidats, ou que leur
  score reste dominé sur toute la plage mesurée.

### OPT-2026-09-20-RUST-PERF-82 — top-32 du sélecteur DFlash — rejeté

- Observation : PERF-81 montre qu'aucun poids raisonnable ne corrige les deux
  divergences du top-16. Le sélecteur lit déjà tout le vocabulaire ; conserver
  32 candidats augmente seulement les petits buffers/edges du sélecteur, pas le
  draft ni la vérification target.
- Hypothèse : inclure les candidats classés 17–32 permet au score de transition
  de récupérer au moins la divergence de rang 0 et de passer de neuf à huit
  blocs. Une vérification économisée (~60–70 ms) doit largement couvrir le
  surcoût du sélecteur.
- Baseline : top-16, N=5, **39/45 (86,7 %)**, neuf blocs, **1,342×** froid.
  Protocole : élargir uniquement K/Candidates 16→32, vérifier la sortie target
  exacte, mesurer trois alternances de 48 tokens et rejeter sans réduction des
  blocs ou gain de débit reproductible. Statut : **en cours**, journalisé avant
  code.
- Résultat E2E exact : **39/45 (86,7 %)**, neuf blocs et 63,483 tok/s sur le
  passage exploratoire, contre 65,078 tok/s pour top-16 dans la même campagne.
  Aucun token supplémentaire n'est accepté et le petit surcoût ne produit
  aucun gain réel.
- Décision : **rejeté**. Buffers, boucles, helpers et clés Metal reviennent à
  top-16 ; aucune modification exécutable de cet essai n'est conservée.

### OPT-2026-09-20-RUST-PERF-83 — profil target DFlash exact M=6 — diagnostic

- Objectif : localiser le coût des neuf vérifications restantes après PERF-79.
  Les profils M=8 de PERF-62 datent d'avant plusieurs kernels et ne séparent pas
  la matérialisation des huit captures DFlash.
- Protocole : instrumentation temporaire sur le modèle physique, six tokens et
  contexte identique, avec synchronisation après corps 40 couches+caches,
  captures et `lm_head`. Sept répétitions ; les barrières servent uniquement à
  classer les goulots et ne seront pas sommées au débit E2E. Retirer le test
  après mesure avant toute modification de production. Statut : **diagnostic
  en cours**, journalisé avant code.
- Résultat, sept répétitions après warmup : corps 40 couches **50,715 ms** médian
  `[49,989; 50,486; 50,501; 50,715; 50,784; 50,980; 52,082]`, captures
  **0,200 ms** et `lm_head` **10,153 ms**. Les captures sont négligeables ; le
  corps représente ~83 % du temps séparé et reste le seul levier assez gros.
- Aucun gain revendiqué et aucune somme avec l'E2E, car les barrières changent
  le graphe. Instrumentation retirée. Statut : **diagnostic terminé**.

### OPT-2026-09-20-RUST-PERF-84 — quatre SIMD-groups expert K=3 — rejeté

- Observation : les experts Qwen de ce checkpoint sont K=3 (trellis
  `[128,32,48]`) et `expert_mapped` leur attribue huit SIMD-groups par
  threadgroup. Quatre groupes divisent les threads et la mémoire threadgroup,
  au prix de deux fois plus d'itérations input par groupe.
- Hypothèse : sur M5 et M=6, l'occupation gagnée dépasse les itérations
  supplémentaires et réduit le corps de 50,7 ms ; le même changement peut aussi
  aider le decode M=1. Aucune arithmétique d'une sortie n'est modifiée.
- Baseline : PERF-79 **1,342×**, PERF-83 corps **50,715 ms**. Protocole : K=3
  M5 seulement, différentiel physique exact M=1/2/4/6/8 puis trois alternances
  E2E. Rejet au premier écart ou sans gain absolu et relatif reproductible.
  Statut : **en cours**, journalisé avant code.
- Résultat : le différentiel physique échoue à M=8 sur l'état de vérification
  exact. Le regroupement différent des réductions change l'arrondi flottant ;
  l'hypothèse « aucune arithmétique modifiée » était donc fausse au sens bit à
  bit requis par le chemin lossless. Aucun benchmark de débit n'a été lancé.
- Décision : **rejeté** au premier écart comme prévu. Le dispatch K=3 revient à
  huit SIMD-groups et la largeur M=6 temporaire est retirée du test.

### OPT-2026-09-20-RUST-PERF-85 — largeur DFlash adaptative — diagnostic

- Observation : PERF-79 vérifie neuf blocs de six lignes ; huit blocs acceptent
  cinq propositions et un bloc diverge immédiatement. Cette vérification de six
  lignes entièrement rejetée est du travail target perdu, mais une heuristique
  non corrélée au rejet déplacerait seulement ce coût ailleurs.
- Hypothèse : la marge ou le rang du token target dans les logits draft permet
  d'identifier le bloc fragile avant vérification et d'y réduire seulement la
  largeur. Le target vérifie toujours chaque token retenu, donc une éventuelle
  politique reste lossless ; seul le débit change.
- Protocole : instrumenter temporairement une génération exacte pour relever,
  par position et par bloc, rang du token target, rang du choix DFlash et marge
  top-1/top-2. Aucune métrique de débit avec cette copie CPU diagnostique. Ne
  modifier la production que si un signal observable *avant* le verify sépare
  les rejets ; sinon rejeter et retirer l'instrumentation. Statut : **diagnostic
  en cours**, journalisé avant code.
- Résultat : le bloc divergent au premier token présente des marges draft
  `[1,156; 1,172; 3,000; 4,250; 2,359]` et le token target est au rang 3 de
  la première ligne. Mais un bloc entièrement accepté a des marges encore plus
  faibles `[0,797; 0,281; 1,211; 0,672; 1,094]`, et un autre bloc accepté finit
  avec une marge 0,055. Aucun seuil de marge observable avant le verify ne
  sépare donc le rejet sans raccourcir aussi de bons blocs.
- Décision : **rejeté**. Aucune largeur adaptative heuristique n'est ajoutée ;
  la copie CPU diagnostique est retirée et aucun débit de cette passe instrumentée
  n'est retenu.

### OPT-2026-09-20-RUST-PERF-86 — `lm_head` M=6 avec sortie fusionnée — en cours

- Observation : PERF-83 mesure **10,153 ms** par `lm_head` M=6, soit ~17 % de
  chaque vérification. Le kernel QMV possède déjà un épilogue Hadamard+échelle
  exact prévu pour huit tiles, mais le chemin batch écrit actuellement les
  accumulations FP32 puis relit toute la matrice via plusieurs opérations MLX.
- Hypothèse : pour le gros head M5, `splits=1`, utiliser NT=8 et l'épilogue
  existant supprime ces buffers/passes sans changer l'ordre des accumulations ni
  les arrondis FP16 de l'épilogue. Le changement reste limité au batch M=6 et
  aux sorties très larges.
- Baseline : PERF-79 **1,342×**, head M=6 **10,153 ms**. Protocole : activer la
  fusion uniquement pour M=6, sortie divisible par 128 et >=65 536, exiger les
  logits FP16 et 80 états strictement identiques M=1/2/4/6/8, puis mesurer le
  head isolé et trois alternances E2E. Rejet au premier écart ou sans gain
  reproductible. Statut : **en cours**, journalisé avant code.
- Exactitude : le différentiel physique M=1/2/4/6/8 conserve tous les logits
  FP16 et les 80 états octet par octet.
- Résultat E2E, trois alternances : greedy **49,273 tok/s**, DFlash
  **62,444 tok/s** (**1,267×**), target **0,598–0,600 s**. La baseline PERF-79
  donnait 0,571–0,597 s et 1,342× à froid ; la fusion exacte M=6 n'améliore donc
  pas le coût target et régresse légèrement dans cette série.
- Décision : **rejeté**. La condition M=6 et sa largeur de test temporaire sont
  retirées ; aucun gain n'est revendiqué.

### OPT-2026-09-20-RUST-PERF-87 — `lm_head` draft M=8 avec sortie fusionnée — en cours

- Observation : les ~0,152 s de draft sur neuf blocs incluent à chaque bloc la
  projection du hidden DFlash par le `lm_head` target M=8. PERF-62 mesurait ce
  head à **13,568 ms** avant NT4 ; même quelques millisecondes économisées neuf
  fois ont plus d'effet que la fusion M=6 rejetée.
- Hypothèse : NT=8 + épilogue Hadamard/échelle fusionné, uniquement pour M=8 et
  les sorties >=65 536, évite les passes globales du head draft et réduit le
  temps draft tout en gardant exactement les mêmes logits.
- Protocole : différentiel strict M=1/2/4/8, puis trois alternances E2E ; comparer
  le temps draft à **0,152–0,162 s** et le speedup apparié à **1,342×**. Rejet
  au premier écart ou sans baisse reproductible du draft. Statut : **en cours**,
  journalisé avant code.
- Exactitude : le différentiel strict M=1/2/4/8 réussit, y compris tous les
  logits et états M=8.
- Résultat E2E : draft **0,233–0,244 s**, DFlash **56,582 tok/s** médian et
  **1,243×** face au greedy apparié, contre 0,152–0,162 s et 1,342× avant.
  NT=8 augmente fortement la pression registres et annule le bénéfice des
  passes mémoire supprimées.
- Décision : **rejeté** ; le QMV M=8 revient à NT4 et l'épilogue embarqué est
  désactivé. Aucun gain revendiqué.

### OPT-2026-09-20-RUST-PERF-88 — épilogue QMV Metal séparé mais fusionné — en cours

- Observation : PERF-86/87 prouvent que l'épilogue Metal existant reproduit
  exactement Hadamard+échelle, mais NT=8 ralentit le QMV. Le QMV NT4 rapide peut
  rester intact et fournir son FP32 à un unique kernel 128 threads au lieu du
  cast, Hadamard et multiply MLX séparés.
- Hypothèse : fusionner uniquement ces trois passes réduit les deux `lm_head`
  M=6/M=8 sans pression registres QMV ni changement arithmétique.
- Protocole : helper Metal limité aux sorties >=65 536, batch 6/8, `splits=1` ;
  différentiel strict M=1/2/4/6/8 puis trois alternances E2E. Rejet au premier
  écart ou sans réduction reproductible de target/draft. Statut : **en cours**,
  journalisé avant code.
- Exactitude : différentiel strict M=1/2/4/6/8 réussi.
- Résultat : première série fusionnée **64,586 tok/s**, target 0,568–0,571 s,
  puis contrôle apparié B/A/A/B sous dérive thermique : fusion **62,514 / 65,063
  tok/s**, baseline **63,292 / 62,914 tok/s**. Les temps target/draft se
  recouvrent également ; l'ordre des variantes explique davantage la mesure
  que l'épilogue.
- Décision : **rejeté, non concluant côté débit**. Le helper et le commutateur
  temporaire sont retirés ; aucun gain n'est poussé.

### OPT-2026-09-20-RUST-PERF-89 — LUT exacte du codebook K=3 sparse — en cours

- Observation : les experts Qwen K=3 dominent le corps target. Pour seulement
  huit codewords possibles, chaque produit recalcule encore le hash entier,
  construit deux FP16 et les additionne. Une LUT constante de huit valeurs
  tient dans 16 octets et élimine ce calcul dans gate/up/down.
- Hypothèse : indexer les huit bits FP16 pré-calculés par le codec Rust réduit
  le coût sparse sans changer aucun produit ni ordre de FMA. Le changement
  profite aussi au decode Qwen ordinaire et à tout expert EXL3 K=3.
- Protocole : LUT uniquement dans `expert_mapped` K=3 ; différentiel strict du
  modèle M=1/2/4/8 puis trois alternances E2E. Rejet au premier bit différent
  ou sans gain reproductible sur target et débit. Statut : **en cours**,
  journalisé avant code.
- Résultat : la génération de warmup termine prématurément sur EOS avant tout
  timing, donc les logits ne sont pas ceux de la baseline. Cause : K=3 encode
  trois bits de transition du treillis, mais l'état/codeword décodé reste sur
  16 bits ; le réduire à huit entrées était une hypothèse invalide.
- Décision : **rejeté immédiatement**, LUT retirée. Le différentiel batch contre
  token-major n'était pas une preuve suffisante ici car les deux chemins
  partageaient la même LUT modifiée ; l'E2E a correctement capté la dérive.

### OPT-2026-09-20-RUST-PERF-90 — tête draft limitée aux N positions utiles — validé

- Source : ticket 21 de `MLXL3_Audit_Decode_28_Optimisations.md`, relu et croisé
  avec PERF-74. PERF-74 limitait seulement les grilles du sélecteur ; il ne
  supprimait aucune ligne du `lm_head` M=8.
- Hypothèse : après les six couches bidirectionnelles M=8, projeter uniquement
  les lignes 1..N+1 par le head target et limiter le selector aux mêmes lignes
  supprime 3/8 des lignes vocabulaire à N=5, sans changer les cinq propositions.
- Baseline : PERF-79 draft **0,157–0,162 s**, target **0,571–0,597 s**,
  DFlash **63,716 tok/s** et **1,342×**. Protocole : comparer les cinq tokens
  sélectionnés à l'ancien M=8, vérifier l'E2E exact, puis trois alternances et
  un A/B/B/A si positif. Rejet au premier token différent ou sans baisse du
  draft. Statut : **en cours**, journalisé avant code.
- Exactitude : les campagnes gardent **39/45 (86,7 %)** propositions acceptées,
  neuf blocs et la séquence complète strictement égale au greedy target. Le
  slice est appliqué après les six couches draft ; leur contexte bidirectionnel
  M=8 reste donc inchangé.
- Première série de trois alternances : draft **0,138–0,154 s**, DFlash médian
  **59,945 tok/s** contre greedy **42,236 tok/s**, soit **1,419×** ; le target
  varie de 0,595 à 0,682 s sous chauffe, d'où le contrôle dédié.
- Deuxième A/B/B/A sans recompilation : head N=5 **67,195 / 66,796 tok/s**,
  draft **0,130 / 0,129 s** ; head M=8 puis slice **63,563 / 64,666 tok/s**,
  draft **0,157 / 0,155 s**. Le temps draft baisse de **16,8–17,2 %** et le
  débit livré gagne **3,3–5,7 %** à température entrelacée. Speedup apparié
  N=5 **1,410× / 1,401×**, contrôle M=8 **1,357× / 1,366×**.
- Décision : **validé et intégré localement**. Le commutateur A/B est retiré ;
  le nombre de positions est dérivé de N et inclus dans les clés/shapes Metal.
  Confirmation finale de production sur trois alternances : greedy médian
  **47,218 tok/s**, DFlash médian **66,652 tok/s**, soit **1,412× (+41,2 %)**,
  draft **0,129–0,132 s**, toujours 39/45 acceptés et sortie exacte.
- Contrôles finaux réussis : `git diff --check`, format Rust, Clippy strict tous
  targets/features, 23 tests lib, 1 test CLI, 14 contrats, build release
  MLX/chat, différentiel physique du head limité contre M=8 puis slice, E2E
  physique exact et rollback sélectif exact pour les largeurs 1 à 8. Kani 0.68
  / CBMC 6.11 vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures.
  Kani couvre ici le Rust pur, pas le shader Metal ; ce dernier est contrôlé
  par les différentiels physiques bornés et n'est pas formellement prouvé.
  Statut : **validé, intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-91 — projection KV-only du cache draft — validé

- Source : ticket 19 de `MLXL3_Audit_Decode_28_Optimisations.md`. À chaque
  `append_captured`, les six couches calculent actuellement les 6 144 sorties
  QKV, puis jettent les 4 096 sorties Q et ne conservent que K/V [4096,6144).
- Hypothèse : permettre au kernel Q4 existant de projeter une plage de sorties
  alignée à 256 réduit de deux tiers ce QKV de commit/préfill sans changer les
  poids, l'ordre des accumulations ni les BF16 K/V.
- Baseline : PERF-90, greedy médian **47,218 tok/s**, DFlash médian
  **66,652 tok/s**, **1,412×**, draft **0,129–0,132 s**, 39/45 propositions
  acceptées. Protocole : comparer bit à bit la plage KV au QKV complet puis
  slice, vérifier la séquence E2E exacte, et mesurer trois alternances. Rejeter
  au premier écart ou sans baisse reproductible du coût cache/débit. Statut :
  **en cours**, journalisé avant code.
- Exactitude : la projection [4096,6144) reproduit octet par octet le slice K/V
  de l'ancien QKV complet sur le vrai paquet DFlash ; l'E2E conserve **39/45
  (86,7 %)** propositions et une séquence strictement égale au greedy target.
- Contrôle A/B/B/A, deux répétitions par passage : KV-only **67,420 / 67,542
  tok/s**, QKV complet **67,176 / 66,746 tok/s**. Le draft passe de
  **0,131–0,133 s** à **0,126–0,127 s** (-3,8 à -5,3 %) ; le contexte mesuré
  passe de 0,007–0,008 s à 0,006–0,008 s. Le débit livré gagne environ
  **0,8 %** sur les médianes de campagne, malgré la dérive du target seul.
- Décision : **validé et intégré localement**. Le commutateur A/B est retiré ;
  le même kernel Q4 reçoit seulement une origine de sortie alignée, sans
  nouvelle abstraction ni nouvel ordre d'accumulation. Contrôles complets et
  publication requis.
- Contrôles finaux réussis : `git diff --check`, format Rust, Clippy strict tous
  targets/features, 23 tests lib, 1 test CLI, 14 contrats et build release
  MLX/chat. Kani 0.68 / CBMC 6.11 vérifie **17/17 harnesses**, zéro échec et
  2/2 couvertures. Kani ne couvre pas Metal ; la plage GPU est validée par le
  différentiel physique borné et l'E2E exact, sans constituer une preuve
  formelle non bornée. Statut : **validé, intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-92 — captures Q4 à la largeur retenue — bloqué/rejeté

- Source : ticket 20 de `MLXL3_Audit_Decode_28_Optimisations.md`.
  `append_captured` complète aujourd'hui chaque commit à huit lignes avant la
  projection de contexte puis les six projections KV, alors que le chemin
  courant en retient le plus souvent six et parfois une seule.
- Hypothèse : paramétrer le kernel Q4 TensorOps existant par M=1..8 et supprimer
  ce padding réduit proportionnellement le travail row-wise des captures sans
  toucher aux huit positions bidirectionnelles du réseau draft lui-même.
- Baseline : PERF-91, KV-only **67,420 / 67,542 tok/s**, draft
  **0,126–0,127 s**, contexte 0,006–0,008 s, 39/45 acceptés. Protocole :
  différentiel BF16 M=1..8 contre l'ancien calcul pad-to-8 puis slice, E2E exact,
  puis A/B entrelacé. Rejet au premier écart ou si le débit/cache ne baisse pas.
  Statut : **en cours**, journalisé avant code.
- Résultat : le premier différentiel M=1 échoue avant exécution à la compilation
  Metal. `MPPTensorOpsMatMul2dImpl.h` impose statiquement que M soit multiple de
  8 ou 16 ; le TensorOps Q4 actuel ne peut donc pas matérialiser M=1..7. Garder
  M=8 avec des lignes masquées ne supprimerait pas le matmul et n'est pas le
  gain visé.
- Décision : **bloqué/rejeté pour le kernel TensorOps actuel**. Le prototype est
  retiré avant tout benchmark E2E. Il faudrait un second micro-kernel Q4 non-MPP
  pour M<8, chantier nettement plus lourd à comparer au très faible coût cache
  observé (0,006–0,008 s) ; aucun code exécutable n'est conservé.

### OPT-2026-09-20-RUST-PERF-93 — EXL3 petit-M, deux lignes par décodage — validé

- Source : ticket 14 de `MLXL3_Audit_Decode_28_Optimisations.md`. Le QMV batch
  place actuellement chaque ligne sur `grid.y` et redécode donc les mêmes
  codewords pour chacune des six lignes de vérification.
- Hypothèse : pour M=6/8, un micro-tile MB=2 partage chaque décodage de poids
  entre deux lignes tout en conservant les accumulateurs et l'ordre des FMA par
  ligne. Le surcoût registres reste borné à deux lignes, contrairement à MB=4.
- Baseline : PERF-91/92, DFlash **67,420 / 67,542 tok/s**, target par génération
  **0,568–0,576 s**, sortie exacte. Protocole : différentiel physique EXL3
  batch M=8 et largeurs de vérification 1..8, puis profil target M=6 et A/B
  E2E entrelacé. Rejet au premier écart ou si spills/occupation annulent le gain.
  Statut : **en cours**, journalisé avant code.
- Exactitude : le différentiel physique M=8 donne zéro mismatch sur 65 536
  sorties et le test des largeurs de vérification 1..8 reste strictement égal au
  chemin token-major. L'E2E conserve 39/45 propositions et la séquence greedy
  target exacte.
- A/B/B/A, deux répétitions par passage : MB=2 **73,038 / 72,368 tok/s**, MB=1
  **69,019 / 68,313 tok/s**. Le débit médian des quatre exécutions passe
  d'environ 68,7 à **72,1 tok/s (+5,0 %)** ; le corps target passe de
  **0,556–0,565 s** à **0,517–0,526 s (-6,5 à -7,1 %)**. Le ratio apparié
  DFlash/greedy du candidat est **~1,450×**. Le draft et l'acceptation restent
  inchangés, ce qui localise le gain dans le vérificateur EXL3.
- Décision : **validé et intégré localement** pour M=6/8 avec MB=2 et NT<=2.
  Le commutateur A/B est retiré. Contrôles complets, confirmation production et
  publication immédiate requis.
- Confirmation production après retrait du commutateur, trois alternances :
  greedy médian **49,967 tok/s**, DFlash médian **73,055 tok/s**, soit
  **1,462× (+46,2 %)** ; target **0,518–0,521 s**, draft 0,124 s, 39/45
  acceptés et sortie exacte.
- Contrôles finaux réussis : `git diff --check`, format Rust, Clippy strict tous
  targets/features, 23 tests lib, 1 test CLI, 14 contrats, build release
  MLX/chat, différentiel M=8 et largeurs physiques 1..8. Kani 0.68 / CBMC
  6.11 vérifie **17/17 harnesses**, zéro échec et 2/2 couvertures. Kani ne
  couvre pas le shader Metal ; l'égalité GPU est un contrôle différentiel borné,
  pas une preuve formelle non bornée. Statut : **validé, intégré et prêt à
  publier**.

### OPT-2026-09-20-RUST-PERF-94 — mutualisation des routes par expert — diagnostic terminé

- Source : ticket 15 de `MLXL3_Audit_Decode_28_Optimisations.md`. Pour six
  lignes et top-8, le chemin vérifie 48 routes et `expert_mapped` redécode les
  poids séparément pour chaque route, même quand plusieurs lignes partagent le
  même expert.
- Hypothèse : si le recouvrement des routes adjacentes est élevé, regrouper les
  lignes par expert permet un micro-tile analogue à PERF-93 pour gate/up/down.
- Baseline : PERF-93 **73,055 tok/s**, target 0,518–0,521 s. Protocole : relever
  sur GPU, sans timing, nombre d'experts distincts, multiplicité moyenne/max et
  histogramme sur chaque couche M=6 ; n'implémenter un tri/gather que si le
  travail partagé dépasse clairement son coût. Instrumentation retirée avant
  toute mesure de débit. Statut : **diagnostic en cours**, journalisé avant code.
- Résultat sur 160 appels/couches M=6 : **29,29 experts distincts** en moyenne
  pour 48 routes, **29,75 routes** appartiennent à un expert répété, soit
  **13,29 paires partageables** par couche. Multiplicité maximale moyenne 4,33,
  maximum observé 6.
- Décision : le recouvrement est suffisant pour tester le regroupement #15.
  L'instrumentation et sa synchronisation CPU sont retirées ; aucun débit de
  cette passe diagnostique n'est retenu. Statut : **diagnostic terminé**.

### OPT-2026-09-20-RUST-PERF-95 — chemin expert segmenté à M=6 — rejeté

- Hypothèse : le plan GPU `route_plan` et le QMM segmenté déjà utilisés au
  préfill peuvent mutualiser les poids des ~13 paires par couche sans ajouter
  immédiatement un second kernel. C'est le premier palier minimal du ticket 15.
- Baseline : PERF-93 **73,055 tok/s**, target 0,518–0,521 s. Protocole : activer
  le chemin segmenté existant uniquement à M=6, vérifier logits/états et sortie
  E2E exacts, puis A/B. Le rejeter si le padding BM32 des ~29 experts coûte plus
  que la réutilisation ; dans ce cas seulement, passer à un micro-kernel MB=2.
  Statut : **en cours**, journalisé avant code.
- Exactitude bornée : les logits des largeurs 1..8 restent identiques au chemin
  token-major et la séquence E2E reste lossless. Cependant, les captures changent
  assez pour ramener l'acceptation de 39/45 à 38/50 dans cette campagne.
- Performance : **42,946 tok/s** contre 73,055 tok/s en baseline, target
  **0,958 s** contre 0,518–0,521 s. Le padding BM32 de ~29 segments distincts
  domine très largement la mutualisation.
- Décision : **rejeté** ; seuil et chemin préfill reviennent inchangés. Le #15
  exige bien un micro-kernel MB=2 sans padding BM32. Aucun code exécutable de
  cet essai n'est conservé.

### OPT-2026-09-20-RUST-PERF-96 — experts groupés par paires MB=2 — rejeté

- Hypothèse : trier les 48 routes sur GPU avec le `route_plan` existant, puis
  traiter chaque segment expert par paires dans un QMV dédié partage le décodage
  des poids sans le padding BM32 rejeté par PERF-95. Les accumulateurs et FMA
  restent indépendants et dans l'ordre original pour chaque route.
- Baseline : PERF-93 **73,055 tok/s**, target 0,518–0,521 s ; PERF-94 mesure
  13,29 paires partageables/couche. Protocole : différentiel complet logits et
  états M=6/8 contre le chemin route-major, E2E exact, puis A/B/B/A. Rejeter au
  premier bit différent ou si tri/gather et pression registres annulent le gain.
  Statut : **en cours**, journalisé avant code.
- Exactitude : le différentiel des largeurs 1..8 et l'E2E restent lossless,
  avec 39/45 propositions acceptées pour le micro-kernel MB=2.
- Performance : NT=2 atteint seulement **61,286 tok/s**, target 0,638–0,641 s.
  NT=4 augmente la pression registres et tombe à **49,887 tok/s**, target
  0,816–0,824 s, contre 73,055 tok/s et 0,518–0,521 s pour PERF-93.
- Décision : **rejeté**. Malgré 13,29 paires théoriques, tri/gather, dispatchs
  clairsemés et occupation réduite coûtent plus que le décodage de poids évité.
  Le kernel, la route expérimentale et le changement de plan sont intégralement
  retirés. Le ticket 15 est donc testé mais non intégré ; aucun gain n'est
  revendiqué.

### OPT-2026-09-20-RUST-PERF-97 — préparation attention de vérification batchée — rejeté

- Source : premier palier du ticket 16 de
  `MLXL3_Audit_Decode_28_Optimisations.md`. L'attention M=6 normalise et applique
  RoPE à Q/K, concatène K/V et met à jour le cache séparément pour chaque
  ligne avant six SDPA exacts.
- Hypothèse : calculer les normes et RoPE sur les six lignes, concaténer le
  bloc K/V une seule fois, puis conserver un SDPA par ligne sur le préfixe exact
  supprime des dispatchs et copies sans modifier l'ordre de réduction de
  l'attention. Ce jalon n'essaie pas encore un SDPA causal batché, susceptible
  de changer les arrondis.
- Baseline publiée : PERF-93, greedy **49,967 tok/s**, DFlash
  **73,055 tok/s**, **1,462× (+46,2 %)**, target **0,518–0,521 s**, draft
  0,124 s, 39/45 acceptés. Protocole : relever d'abord une baseline au commit
  courant, comparer logits et 80 états pour les largeurs 1..8, puis A/B/B/A
  et trois alternances E2E exactes de 48 tokens. Rejet au premier écart ou si
  le gain relatif ne dépasse pas le bruit thermique. Statut : **en cours**,
  journalisé avant code.
- Exactitude : le différentiel physique des largeurs 1..8 reste strictement
  identique au chemin token-major et les trois séquences E2E sont exactes.
- Mesure : baseline fraîche **72,974 tok/s** médians, target 0,517–0,523 s ;
  candidat **72,556 tok/s**, target 0,518–0,536 s, toujours 39/45 acceptés.
  Les préparations batchées ne diminuent pas le temps target au-delà du bruit.
- Décision : **rejeté**. Le calcul par ligne est restauré ; aucun code
  exécutable n'est conservé. Un vrai kernel d'attention multi-requêtes reste
  distinct de ce palier, mais n'est justifié qu'après un profil attention isolé.

### OPT-2026-09-20-RUST-PERF-98 — résiduel et RMSNorm compacts du vérificateur — rejeté

- Source : palier minimal du ticket 17 du rapport. Chaque couche applique
  actuellement deux RMSNorm et les additions résiduelles sur six petits
  tenseurs séparés, avec slices et concaténations autour des blocs déjà
  batchés.
- Hypothèse : concaténer les six lignes une fois par couche, appliquer
  RMSNorm/résidu sur `[1,M,H]`, puis ne redécouper qu'à la frontière MoE
  conserve exactement la réduction sur la dernière dimension et supprime des
  dispatchs MLX. Ce premier palier évite un refactor complet des interfaces.
- Baseline fraîche : greedy **50,630 tok/s**, DFlash **72,974 tok/s**,
  **1,441×**, target 0,517–0,523 s, draft 0,124–0,125 s, 39/45 acceptés.
  Protocole : différentiel physique largeurs 1..8, E2E exact, puis trois
  alternances de 48 tokens et A/B/B/A si le candidat dépasse le bruit. Rejet
  au premier écart de logits/états. Statut : **en cours**, journalisé avant
  code.
- Exactitude : les largeurs 1..8 et les trois sorties E2E restent strictement
  identiques au chemin token-major.
- Mesure : **73,067 tok/s** médians, target 0,516–0,524 s, contre
  **72,974 tok/s** et 0,517–0,523 s juste avant. L'écart de 0,13 % est du
  bruit et ne réduit pas le goulot target.
- Décision : **rejeté** ; les petits tenseurs sont restaurés et aucun code
  exécutable n'est conservé. Le refactor compact complet n'est pas justifié
  par ce premier palier mesuré.

### OPT-2026-09-20-RUST-PERF-99 — préparation Metal directe des routes MoE — validé

- Source : ticket 8 du rapport. Le petit-M construit aujourd'hui `x_gu` par
  broadcast/reshape, rassemble `gu_suh`, multiplie en FP16 puis lance le
  Hadamard avant chaque gate/up expert.
- Hypothèse : un kernel Metal produit directement les deux lignes transformées
  de chaque route depuis `x`, `selected` et `gu_suh`, avec exactement les
  arrondis Hadamard radix-16/radix-8 déjà utilisés par le chemin segmenté.
  Cela remplace plusieurs opérations MLX dans chaque couche MoE M=1..8 sans
  modifier le kernel EXL3 ni l'ordre de réduction des experts.
- Baseline : greedy **50,630 tok/s**, DFlash **72,974 tok/s**, **1,441×**,
  target 0,517–0,523 s. Protocole : différentiel direct de la préparation puis
  logits/80 états pour M=1..8, trois alternances E2E et A/B/B/A si positif.
  Rejet au premier bit différent ou sans gain reproductible. Statut : **en
  cours**, journalisé avant code.
- Contrôle préliminaire : le kernel direct reproduit bit à bit la chaîne MLX
  indépendante sur 3 lignes, 2 routes, 5 experts et 256 dimensions ; les
  largeurs de vérification 1..8 restent exactes. Une première exécution E2E
  isolée donne **73,578 tok/s**, target 0,513–0,522 s, contre 72,974 tok/s,
  mais ce petit écart n'est pas encore validé.
- Incident de protocole : la tentative A/B/B/A a lancé quatre commandes cargo
  en sessions longues simultanées au lieu de les attendre séquentiellement,
  chargeant quatre fois le Qwen 35B et saturant la RAM unifiée. Les huit PID
  cargo/test ont été arrêtés immédiatement ; aucune mesure de cette campagne
  parallèle n'est valide ni conservée.
- État : **interrompu à la demande de l'utilisateur**, prototype local non
  publié. La reprise devra exécuter strictement un seul processus de benchmark
  à la fois et vérifier le PID précédent avant l'alternance suivante.
- Reprise séquentielle A/B/B/A, deux répétitions par processus et contrôle
  d'absence du PID précédent avant chaque lancement : ancien chemin
  **73,264 / 74,234 tok/s**, kernel direct **74,684 / 74,760 tok/s**. Les
  médianes des quatre mesures passent de **73,749** à **74,722 tok/s
  (+1,32 %)** ; target passe globalement de 0,511–0,517 s à 0,506–0,508 s.
  Le greedy profite aussi de la préparation générale, donc ce gain absolu ne
  porte pas à lui seul le ratio DFlash à 1,5×.
- Décision : **validé**. Le commutateur A/B et l'ancien assemblage sont
  retirés ; le chemin petit-M utilise le kernel direct. Confirmation de
  production, contrôles complets et publication requis.
- Confirmation de production après retrait du commutateur, un seul processus
  et trois alternances : greedy médian **51,896 tok/s**, DFlash médian
  **74,645 tok/s**, soit **1,438× (+43,8 %)** ; target 0,507–0,508 s, draft
  0,122–0,123 s, 39/45 acceptés et trois sorties exactes. Par rapport au
  73,055 tok/s de PERF-93, le débit DFlash absolu gagne **2,18 %** ; le ratio
  baisse parce que le même kernel accélère davantage le greedy M=1.
- Contrôles finaux réussis : `git diff --check`, format Rust, Clippy strict
  tous targets/features, **23 tests lib + 14 contrats**, build release MLX/chat,
  différentiel Metal indépendant M=1/M=6 top-8, largeurs physiques 1..8 et
  trois passages E2E exacts. Kani 0.68 / CBMC 6.11 termine sans échec sur les
  harnesses Rust existants ; il ne couvre pas le shader Metal, dont l'égalité
  est un contrôle différentiel fini et non une preuve formelle non bornée.
  Statut : **validé, intégré et prêt à publier**.

### OPT-2026-09-20-RUST-PERF-100 — masque d'attention draft partagé — rejeté

- Source : jalon minimal du ticket 23. Les six couches du draft reconstruisent
  actuellement le même `Vec<f32>` de masque 8×contexte, le transfèrent et le
  convertissent en BF16 à chaque bloc.
- Hypothèse : valider que les six caches ont la même longueur, construire le
  masque une seule fois dans `forward_hidden` et partager son handle MLX entre
  les six SDPA supprime allocations, copies et dispatchs sans changer un seul
  élément du masque. Aucun cache circulaire n'est ajouté à ce stade.
- Baseline après PERF-99 : greedy **51,896 tok/s**, DFlash **74,645 tok/s**,
  **1,438× (+43,8 %)**, draft 0,122–0,123 s, target 0,507–0,508 s. Protocole :
  test unitaire des bornes de masque 0/2047/2048/2049, E2E exact puis A/B/B/A
  strictement séquentiel. Rejet au premier écart ou sans gain reproductible.
  Statut : **en cours**, journalisé avant code.
- Exactitude : le test des bornes 0/2047/2048/2049, la visibilité entre les
  huit lignes draft et la séquence E2E restent strictement identiques.
- Mesure A/B/B/A, un seul processus à la fois et deux répétitions par
  processus : ancien chemin **73,339 / 73,566 tok/s**, masque partagé
  **73,707 / 73,477 tok/s**. Les médianes des quatre mesures sont
  **73,453** contre **73,592 tok/s (+0,19 %)** ; draft reste 0,124–0,128 s.
- Décision : **rejeté**. Le faible écart est du bruit thermique et ne réduit
  pas le temps draft de manière stable. Le partage, le commutateur A/B et ses
  tests sont retirés ; aucun code exécutable de cet essai n'est conservé.

### OPT-2026-09-20-RUST-PERF-101 — gate/up Q4 draft groupés — rejeté

- Source : premier palier du ticket 22 du rapport. Chacune des six couches
  lance aujourd'hui deux dispatchs Q4 M=8 successifs sur la même entrée pour
  gate et up, puis un SwiGLU MLX séparé.
- Hypothèse : un seul kernel Metal couvrant les deux matrices conserve les
  sorties BF16 exactes tout en supprimant six dispatchs et une partie du coût
  de préparation. Ce palier ne fusionne pas encore l'épilogue SwiGLU afin
  d'attribuer le gain et de limiter la pression registres.
- Baseline après PERF-99 : greedy **51,896 tok/s**, DFlash **74,645 tok/s**,
  **1,438× (+43,8 %)**, draft 0,122–0,123 s, target 0,507–0,508 s. Protocole :
  différentiel physique bit à bit des sorties gate/up groupées contre deux
  projections indépendantes, séquence E2E exacte, puis A/B/B/A strictement
  séquentiel avec vérification d'absence du PID précédent. Rejet au premier
  écart ou si le gain est inférieur au bruit thermique. Statut : **en cours**,
  journalisé avant code.
- Exactitude : le kernel groupé reproduit bit à bit les deux sorties BF16
  calculées par les projections indépendantes ; la séquence E2E, 39/45
  propositions acceptées, reste identique.
- Mesure A/B/B/A strictement séquentielle, deux répétitions par processus :
  groupé **73,450 / 71,561 tok/s**, indépendant **72,985 / 73,558 tok/s**.
  Les médianes par variante sont **72,506** contre **73,272 tok/s (-1,05 %)**.
  Le temps draft ne baisse pas (0,127–0,128 s contre 0,124–0,127 s). Un run
  groupé à 63,788 tok/s, accompagné d'une chute greedy similaire, est traité
  comme chauffe et non comme signal du kernel.
- Décision : **rejeté**. Réunir les dispatchs sans partager le calcul interne
  réduit l'occupation et ne compense pas le lancement économisé. Le kernel,
  la méthode, le commutateur A/B et le test sont tous retirés ; aucun code
  exécutable de cet essai n'est conservé. La fusion SwiGLU complète reste une
  expérience distincte, mais doit partager les sommes d'entrée ou éviter les
  sorties intermédiaires pour avoir une chance de gagner.

### OPT-2026-09-20-RUST-PERF-102 — épilogue `lm_head` + argmax exact — rejeté

- Source : ticket 18 du rapport, après les épilogues NT8 et Metal séparé de
  PERF-86/87/88. Le vérificateur M=6 ne consomme les logits target que via un
  argmax par ligne, mais matérialise actuellement 1 489 920 logits puis les
  rescane dans un second kernel.
- Hypothèse : conserver le QMV NT2/MB2 déjà validé, puis fusionner dans un seul
  épilogue le Hadamard 128, le scale FP16, les frontières BF16→FP16 et la
  réduction argmax par blocs supprime les logits finaux et leur scan sans la
  pression registres qui avait fait régresser NT8. Un second petit étage réduit
  uniquement 1 940 maxima par ligne.
- Baseline après PERF-99 : greedy **51,896 tok/s**, DFlash **74,645 tok/s**,
  **1,438× (+43,8 %)**, target 0,507–0,508 s. Protocole : comparer chaque ID
  aux logits matérialisés pour M=1..8 et des égalités synthétiques, exiger la
  séquence E2E exacte, puis A/B/B/A avec un seul processus de modèle à la fois.
  Rejet au premier ID différent ou sans baisse reproductible du temps target.
  Statut : **en cours**, journalisé avant code.
- Exactitude : l'épilogue reproduit tous les argmax du chemin matérialisé pour
  M=2/6/8 et la séquence E2E reste identique avec 39/45 acceptés.
- Mesure A/B/B/A strictement séquentielle, deux répétitions par processus :
  fusion **70,387 / 73,958 tok/s**, contrôle **72,314 / 72,875 tok/s**. Les
  médianes par variante sont **72,173** contre **72,595 tok/s (-0,58 %)** ; le
  target fusionné varie 0,510–0,538 s contre 0,519–0,548 s sans baisse stable.
- Décision : **rejeté**. Le Hadamard MLX natif est déjà plus efficace que le
  kernel 128-thread proposé et le petit scan supprimé ne compense pas les
  barrières de l'épilogue. Helper Rust, shader, API, commutateur et tests sont
  entièrement retirés ; aucun code exécutable n'est conservé.

### OPT-2026-09-20-RUST-PERF-103 — profil du sélecteur DFlash — diagnostic terminé

- Source : prérequis minimal aux tickets 24 et 25. Le draft complet coûte
  0,122–0,128 s sur neuf blocs, mais aucune mesure récente ne sépare les six
  couches, le `lm_head` limité et les trois étages du sélecteur.
- Hypothèse : une synchronisation temporaire après réseau, tête et sélection
  permet de déterminer si fusionner les transitions/greedy ou réécrire le
  top-16 peut économiser les ~26 ms nécessaires pour atteindre 1,5×.
- Protocole : une seule génération de 48 tokens, trois répétitions dans un seul
  processus, mêmes 39/45 propositions ; relever chaque composant. Les barrières
  rendent le total non comparable au débit normal. Retirer l'instrumentation
  immédiatement ; n'implémenter #24/#25 que si leur coût mesuré est matériel.
  Statut : **diagnostic en cours**, journalisé avant code.
- Trois répétitions, 48 tokens et neuf blocs : réseau draft **0,057–0,058 s**,
  `lm_head` limité **0,089–0,091 s**, sélecteur complet **0,005–0,006 s**.
  Les barrières portent le draft instrumenté à 0,151–0,156 s et interdisent de
  comparer son débit au chemin asynchrone normal, mais classent clairement les
  composants.
- Décision : les tickets 24/25 ne peuvent récupérer au maximum que ~6 ms dans
  ce scénario, très loin des ~26 ms requises pour atteindre 1,5×. Le sélecteur
  n'est donc pas réécrit maintenant. La tête vocabulaire domine le draft ;
  l'instrumentation et toutes ses synchronisations sont retirées. Aucun gain
  revendiqué. Statut : **diagnostic terminé**.

### OPT-2026-09-20-RUST-PERF-104 — micro-tile EXL3 MB=3 pour M=6 — rejeté

- Source : prolongement mesuré du ticket 14 après MB=2 validé en PERF-93.
  PERF-103 attribue 0,089–0,091 s sur neuf blocs au `lm_head` draft ; la tête
  target M=6 est également répétée neuf fois. Le kernel actuel redécode chaque
  tile de poids trois fois, une fois par paire de lignes.
- Hypothèse : `MB=3, NT=1` garde 24 accumulateurs de sortie et 12 valeurs
  d'entrée par thread, contre 32+8 pour `MB=2, NT=2`, tout en ne redécodant les
  poids que deux fois par tile. La grille a davantage de threadgroups mais une
  pression registres comparable ; chaque ligne conserve son ordre de FMA.
- Baseline production : greedy **51,896 tok/s**, DFlash **74,645 tok/s**,
  **1,438× (+43,8 %)**, target 0,507–0,508 s. Protocole : différentiel physique
  M=6 contre MB=2, largeurs 1..8 et E2E exact, puis A/B/B/A strictement
  séquentiel. Rejet au premier bit différent ou si target/débit ne gagnent pas
  de façon reproductible. Statut : **en cours**, journalisé avant code.
- Exactitude : logits FP16 et 80 états M=1/2/4/6/8 restent identiques au chemin
  token-major ; les quatre générations gardent 39/45 propositions et la même
  séquence target.
- Mesure A/B/B/A, deux répétitions et un seul processus à la fois : MB=3
  **71,739 / 71,707 tok/s**, MB=2 **72,899 / 73,491 tok/s**. Les médianes par
  variante sont **71,723** contre **73,195 tok/s (-2,01 %)** ; target MB=3
  reste 0,530–0,533 s contre 0,515–0,526 s.
- Décision : **rejeté**. Malgré une redécompression de poids en moins, NT=1
  augmente le nombre de threadgroups et perd le parallélisme sur N. MB=2/NT=2
  reste la meilleure géométrie M=6 mesurée. Commutateur et largeur de test
  temporaire retirés ; aucun code exécutable de l'essai n'est conservé.

### OPT-2026-09-20-RUST-PERF-105 — micro-tile EXL3 MB=3/NT=2 — rejeté

- Révision motivée de PERF-104 : MB=3/NT=1 perd 2,01 % parce qu'il augmente la
  grille malgré le partage des poids. Conserver NT=2 réduit les threadgroups à
  un tile-N équivalent, soit un tiers de moins que MB=2/NT=2, et garde deux
  décodages de poids par tile au lieu de trois.
- Risque : 48 accumulateurs FP32 et 12 entrées par thread contre 32+8 pour la
  production peuvent réduire l'occupation ou provoquer du spill. Aucun autre
  chemin ni paramètre n'est changé.
- Baseline appariée PERF-104 : MB=2 **73,195 tok/s**, target 0,515–0,526 s.
  Protocole : même différentiel M=1/2/4/6/8 puis A/B/B/A, toujours avec un seul
  processus de modèle. Rejet au premier écart ou si le gain ne résiste pas aux
  quatre passages. Statut : **en cours**, journalisé avant code.
- Exactitude : le différentiel logits/80 états M=1/2/4/6/8 passe sans écart et
  la génération conserve 39/45 propositions ainsi que la sortie target.
- Arrêt anticipé motivé après un candidat et son contrôle : MB=3/NT=2 tombe à
  **62,150 tok/s** au meilleur des deux runs, target **0,622–0,688 s**, contre
  MB=2/NT=2 **72,901 tok/s**, target **0,518–0,534 s**. La régression est de
  **14,7 %** sur le meilleur débit et affecte directement le target ; elle est
  beaucoup trop large pour justifier les quatre passages prévus.
- Décision : **rejeté**. Les 60 registres environ par thread provoquent une
  perte d'occupation ou du spill qui domine la grille réduite. Commutateur et
  largeur M=6 temporaire retirés ; aucun code exécutable n'est conservé.

### OPT-2026-09-20-RUST-PERF-106 — extraction QMV K=6 en mots 32 bits — rejeté

- Observation : PERF-103 attribue **0,089–0,091 s** sur neuf blocs au
  `lm_head` limité ; le checkpoint encode cette tête 2048→248320 en K=6/MCG.
  Le kernel QMV batch reconstruit actuellement deux fenêtres par tuile avec des
  concaténations et décalages `ulong`, y compris un second décalage pour le
  quatrième codeword de chaque groupe.
- Hypothèse : reconstruire exactement les mêmes 32 bits depuis les deux mots
  avec des décalages/or 32 bits, uniquement pour les gros QMV batch K=6,
  supprime les opérations entières 64 bits sans changer codewords, poids, FMA,
  réduction ni géométrie. L'ancien essai `OPT-2026-09-07-10` concernait le QMM
  générique et avait régressé ; ce nouvel essai est motivé par le profil QMV
  K=6 répété et reste donc explicitement distinct.
- Baseline production : greedy **51,896 tok/s**, DFlash **74,645 tok/s**,
  **1,438× (+43,8 %)**, target 0,507–0,508 s. Protocole : différentiel physique
  M=1/2/4/6/8, puis A/B/B/A strictement séquentiel avec un seul processus de
  modèle. Rejet au premier bit différent ou si lm_head/target/débit ne gagnent
  pas de manière reproductible. Statut initial : **en cours**, journalisé avant
  code.
- Exactitude : le différentiel physique M=1/2/4/6/8 passe ; logits FP16 et les
  80 états restent identiques au chemin token-major. Les quatre générations
  conservent 39/45 propositions et la séquence target exacte.
- A/B/B/A séquentiel, deux répétitions par processus : extraction 32 bits
  **61,812 / 61,831 tok/s**, ancien `ulong` **63,797 / 64,138 tok/s**. Les
  médianes par variante sont **61,822** contre **63,968 tok/s (−3,35 %)**.
  Le draft candidat prend 0,160–0,165 s puis 0,161–0,163 s, contre
  0,142–0,143 s et 0,142 s pour le contrôle. Le target régresse également de
  0,590–0,595 s à 0,598–0,612 s selon le passage.
- Décision : **rejeté**. Sur ce M5, les expressions `ulong` d'origine sont
  mieux abaissées que la reconstruction manuelle 32 bits malgré leur apparence
  plus coûteuse. Le shader, le commutateur et la largeur M=6 temporaire du test
  sont intégralement retirés ; aucun code exécutable de l'essai n'est conservé.

### OPT-2026-09-20-RUST-PERF-107 — partage MB=2 du `lm_head` draft M=5 — validé

- Source : ticket D02 de
  `MLXL3_Decode_50_Upgrades_Priorises_2026-09-20.md`. PERF-90 limite le head
  draft aux cinq lignes utiles, mais `forward_qmv_batch` réserve encore MB=2
  aux seuls M=6/8 ; M=5 redécode donc les mêmes poids cinq fois.
- Hypothèse : une grille MB=2 à trois groupes traite M=5 en `2+2+1`, partage le
  décodage sur les quatre premières lignes et garde explicitement la queue
  impaire. La ligne inexistante charge des zéros et n'écrit rien ; les cinq
  lignes valides conservent exactement leurs FMA, réduction et épilogue.
- Baseline fraîche après PERF-106 rejeté : ancien M=5 **63,797 / 64,138 tok/s**
  dans les deux contrôles séquentiels, draft 0,142–0,143 s, 39/45 propositions
  acceptées. Protocole : oracle token-major indépendant pour M=1/2/3/4/5/6/7/8,
  contrôle direct du head limité, puis A/B/B/A E2E, un seul processus modèle à
  la fois. Rejet au premier écart ou sans gain reproductible. Statut initial :
  **en cours**, journalisé avant code.
- Exactitude : l'oracle physique indépendant passe pour chaque largeur M=1 à
  M=8 ; logits FP16 et 80 états sont identiques au chemin token-major. Le head
  limité M=5 égale aussi les cinq lignes correspondantes du head M=8. Toutes
  les campagnes E2E gardent 39/45 propositions et la séquence target exacte.
- Microbenchmark head isolé, 3 warmups puis 27 appels évalués par processus,
  A/B/B/A séquentiel : MB=2 **8,676 / 8,729 ms/appel**, MB=1
  **9,611 / 9,608 ms/appel**. Les médianes passent de **9,610** à **8,703 ms
  (−9,44 %, 1,104×)**.
- E2E : la première A/B/B/A donne MB=2 **63,677 / 62,542 tok/s** contre MB=1
  **62,518 / 62,971 tok/s**, soit **+0,58 %** sur les médianes de variante. Une
  paire adjacente supplémentaire au même régime donne **64,936** contre
  **64,862 tok/s (+0,11 %)**. Le draft candidat baisse de 0,140–0,147 s à
  0,130–0,137 s sur ces passages. Un contrôle isolé à 72,728 tok/s, après un
  changement brutal de régime machine, est explicitement exclu de la
  comparaison appariée.
- Décision : **validé et intégré localement**. MB=2 couvre maintenant M=5..8 ;
  la dernière paire impaire charge des zéros et n'écrit aucune sixième ligne.
  Le gain E2E court est modeste et ne doit pas être présenté comme +9,44 % :
  ce chiffre appartient au head isolé. Le commutateur et le microbenchmark
  temporaire sont retirés ; le différentiel permanent couvre désormais toutes
  les largeurs M=1..8.
- Confirmation production après retrait du commutateur, trois répétitions :
  greedy médian **45,158 tok/s**, DFlash médian **65,463 tok/s (1,450×)**,
  draft 0,129–0,133 s, target 0,585–0,588 s, 39/45 propositions et trois
  séquences exactes. Cette confirmation fixe l'état intégré ; elle ne remplace
  pas les comparaisons A/B ci-dessus pour attribuer le petit gain causal.
- Contrôles finaux réussis : `cargo fmt --all -- --check`, `git diff --check`,
  `cargo clippy --all-targets --all-features -- -D warnings`, 23 tests lib + 1
  test CLI + 14 tests de contrats, build release, et Kani 0.68.0 / CBMC 6.11.0
  (**17/17 harnesses vérifiés, 0 échec**, deux propriétés de couverture par
  harness concerné). Kani couvre les propriétés Rust bornées existantes ; la
  garde Metal de la ligne impaire est couverte par les différentiels physiques
  finis M=1..8, pas par une preuve exhaustive du shader.

### OPT-2026-09-20-RUST-PERF-108 — partage MB=2 des projections EXL3 groupées — validé

- Source : ticket D01 de
  `MLXL3_Decode_50_Upgrades_Priorises_2026-09-20.md`, placé avant D23/D09/D04
  dans l'ordre d'attaque confirmé. Le QMV standard partage déjà le décodage
  des poids entre deux lignes, mais `Exl3Group::forward_qmv_batch` soumet encore
  une ligne par groupe Y avec `MLXL3_BATCH_ROWS=1`.
- Hypothèse : sur le chemin groupé `IDENTITY_MAP=1`, MB=2 peut réutiliser chaque
  tuile de treillis pour deux activations indépendantes sans toucher aux maps
  d'experts. Les `suh` restent appliqués avant le kernel ; le shader ne partage
  que les poids et garde deux ensembles d'accumulateurs.
- Baseline production après PERF-107 : greedy médian **45,158 tok/s**, DFlash
  médian **65,463 tok/s**, draft 0,129–0,133 s, target 0,585–0,588 s, 39/45
  propositions exactes. Protocole prévu : bundle QKV/Z réel M=6 puis largeur
  M=8, comparaison bit-à-bit contre les projections série, microbenchmark
  apparié MB=1/MB=2 et A/B E2E strict, un seul processus modèle à la fois.
  Rejet immédiat au premier écart numérique ou sans gain reproductible.
- Exactitude : le bundle réel `linear_attn.in_proj_qkv` + `in_proj_z` de la
  couche 0 passe bit-à-bit contre les deux projections série pour M=6 et M=8.
  Les quatre générations complètes gardent 39/45 propositions et exactement
  la même séquence greedy.
- Microbenchmark bundle M=8, A/B/B/A séquentiel, cinq appels évalués par
  processus après deux warmups : MB=2 **0,646 / 0,716 ms**, MB=1
  **0,909 / 0,946 ms**. Les médianes de variante passent de **0,928** à
  **0,681 ms (−26,5 %, 1,36×)**. Le temps des projections série, donné seulement
  comme repère, variait de 1,337 à 1,473 ms.
- E2E DFlash, 48 tokens, cinq propositions, deux répétitions par processus,
  A/B/B/A strict : MB=2 **66,352 / 67,221 / 67,593 / 67,049 tok/s** ; MB=1
  **64,647 / 64,655 / 64,344 / 65,616 tok/s**. Les médianes supérieures des
  quatre passages sont **67,221 contre 64,655 tok/s, soit +3,97 %**. Le temps
  target baisse de 0,588–0,598 s à 0,565–0,577 s hors un premier passage à
  0,570 s ; draft et commit restent comparables.
- Décision : **validé et intégré localement** pour M=6/8 sur le seul chemin
  groupé `IDENTITY_MAP=1`. Le mapping d'experts n'est pas modifié. Le
  commutateur A/B temporaire a été retiré.
- Confirmation production après retrait du commutateur, trois répétitions :
  greedy médian **44,856 tok/s**, DFlash médian **67,792 tok/s (1,511×)**,
  target 0,562–0,566 s, draft 0,129 s, 39/45 propositions et trois séquences
  exactes. Face à la confirmation PERF-107 à 65,463 tok/s, le débit observé
  monte de **3,56 %** ; l'attribution causale reste fondée sur l'A/B/B/A à
  +3,97 %. Cette série chaude ne remplace pas les anciens records absolus de
  73–75 tok/s mesurés sous un autre régime machine ; elle valide le delta
  adjacent, pas un nouveau plafond absolu.
- Contrôles finaux réussis : `cargo fmt --all -- --check`, `git diff --check`,
  Clippy strict tous targets/features, 23 tests lib + 1 test CLI + 14 contrats,
  build release, oracle GPU M=6/8, et Kani 0.68.0 / CBMC 6.11.0 (**17/17
  harnesses, 0 échec**). Kani borne les propriétés Rust existantes ; le shader
  Metal est validé par les différentiels physiques finis, pas prouvé
  exhaustivement.

### OPT-2026-09-20-RUST-PERF-109 — préparation GDN conv/SiLU/QK-norm fusionnée — validé

- Source : ticket D23 de
  `MLXL3_Decode_50_Upgrades_Priorises_2026-09-20.md`, troisième chantier de
  l'ordre confirmé après D02 et D01. PERF-70 fusionne déjà la boucle récurrente
  T≤8, mais chaque couche GDN matérialise encore concaténation, conv1d, SiLU,
  deux slices/reshapes et deux RMSNorm avant cette récurrence.
- Antécédents : PERF-34 a rejeté convolution + état seule faute de gain E2E,
  bien que son calcul soit exact ; PERF-37 a rejeté des gates dans un dispatch
  séparé ; PERF-38/70 montrent qu'une fusion gagne lorsqu'elle retire vraiment
  plusieurs frontières. Le nouvel essai ne refait donc pas PERF-34 : il fusionne
  convolution, SiLU et les deux normalisations/scales Q/K en un dispatch, sans
  inclure les projections EXL3 ni la récurrence.
- Hypothèse : un SIMDgroup par head et par position peut reproduire le kernel
  depthwise MLX (accumulation FP32 ordonnée), puis le SiLU FP16 et la réduction
  RMS canonique à 32 lanes pour Dk=128, tout en produisant directement Q/K/V et
  le nouvel état convolutif. Cela retire les intermédiaires répétés sur les 30
  couches GDN.
- Baseline production courante chaude après PERF-108 : greedy médian **44,856
  tok/s**, DFlash médian **67,792 tok/s**, target 0,562–0,566 s, draft 0,129 s,
  39/45 propositions exactes. Protocole : différentiel séparé conv, Q, K, V et
  état pour T=1/6/8, puis modèle complet logits + 80 états, microprofil et
  A/B/B/A E2E, un seul processus modèle à la fois. Rejet au premier écart ou
  si le gain complet n'est pas reproductible.
- Premier smoke : compilation Metal refusée avant exécution car les scalaires
  MLX de rang zéro `q_scale`/`k_scale` sont exposés comme valeurs, pas comme
  pointeurs (`q_scale[0]`). Aucun calcul ni benchmark n'a eu lieu. Correction
  locale limitée à leur accès scalaire, puis reprise du différentiel.
- Différentiel isolé corrigé : sur poids convolutionnels réels et activations/
  état FP16 déterministes, Q, K, V et le nouvel état convolutif sont identiques
  bit-à-bit au graphe MLX `concatenate → conv1d → SiLU → RMSNorm → scales` pour
  **T=1, T=6 et T=8**. Le prototype peut donc être raccordé au modèle complet ;
  aucun chiffre de performance n'est encore attribué à ce seul passage.
- Raccord modèle : largeurs de vérification M=1..8, commit partiel DFlash et
  snapshot/rejeu passent bit-à-bit. Premier A/B/B/A E2E chaud, 48 tokens et
  deux répétitions par processus : fusion **66,472 / 66,891 / 67,959 / 67,699
  tok/s**, référence **65,939 / 65,981 / 67,099 / 67,465 tok/s**. La médiane
  conventionnelle des quatre mesures est **67,295 contre 66,540 tok/s
  (+1,13 %)** ; le greedy est quasi neutre (**44,909 contre 44,835 tok/s,
  +0,17 %**). Le signal positif reste petit face à la dérive thermique.
- Essai discriminant suivant, enregistré avant modification : mesurer dans un
  unique test GPU la préparation seule pour T=1/6/8, après warmup, en alternant
  le graphe MLX de référence et le kernel fusionné, avec évaluation forcée de
  Q/K/V/état. Conserver D23 seulement si cette frontière baisse nettement et
  si un second E2E apparié confirme l'absence de régression.
- Microbenchmark apparié, 8 warmups puis 40 mesures par variante : à T=1,
  fusion **0,171 ms** contre référence **0,400 ms (−57,1 %)** ; T=6,
  **0,152 contre 0,409 ms (−62,7 %)** ; T=8, **0,163 contre 0,401 ms
  (−59,5 %)**. Les quatre sorties restent contrôlées bit-à-bit avant mesure.
- Décision : **validé et intégré localement** pour T≤8 hors trace. La trace et
  les grands prefills gardent le graphe MLX existant. Le commutateur A/B
  temporaire est retiré ; la confirmation production et les contrôles finaux
  sont consignés ci-dessous.
- Confirmation production après retrait du commutateur, trois répétitions :
  greedy **43,305 / 44,713 / 44,636 tok/s** (médiane **44,636**), DFlash
  **67,489 / 67,325 / 67,206 tok/s** (médiane **67,325**, 1,508×), target
  0,567–0,569 s, draft 0,128–0,132 s et 39/45 propositions exactes. Cette
  série chaude confirme le palier courant sans constituer un nouveau record ;
  l'attribution causale reste l'A/B à +1,13 % et le microbenchmark à −57/−63 %.
- Contrôles finaux réussis : `cargo fmt --all -- --check`, `git diff --check`,
  Clippy strict tous targets/features, 23 tests lib + 1 test CLI + 14 contrats,
  build release, différentiels GPU T=1/6/8 et modèle/commit/snapshot, ainsi que
  Kani 0.68.0 / CBMC 6.11.0 (**17/17 harnesses vérifiés, 0 échec**). Kani
  couvre les contrats Rust bornés existants ; le shader Metal est couvert par
  les différentiels physiques finis et n'est pas formellement prouvé.

### OPT-2026-09-20-RUST-PERF-110 — Hadamard SIMD de `glu_down_input` — rejeté

- Source : ticket D09 de
  `MLXL3_Decode_50_Upgrades_Priorises_2026-09-20.md`, quatrième chantier de
  l'ordre confirmé. Le kernel courant alloue six tableaux threadgroup de 128
  floats et exécute 23 barrières pour les trois Hadamard gate/up/down.
- Antécédents : PERF-68 a montré qu'un Hadamard radix-2 naïf ne reproduit pas
  le découpage radix-16/radix-8 de l'épilogue MLX ; D09 vise toutefois le
  kernel `glu_down_input`, dont la référence actuelle est précisément sept
  butterflies radix-2 FP32, avec cast après le quatrième étage pour GELU.
  Les fusions d'épilogue et de `lm_head` rejetées ne sont pas répétées.
- Hypothèse : les cinq premiers étages restent à l'intérieur d'un SIMDgroup et
  peuvent utiliser `simd_shuffle_xor`. Les deux étages inter-SIMD utilisent
  quatre tableaux partagés double-bufferisés pour gate/up, ensuite réutilisés
  par down, avec quatre barrières au total. L'ordre add/sub et les casts restent
  inchangés ; le volume partagé baisse de 3 072 à 2 048 octets.
- Baseline chaude après PERF-109 : greedy médian **44,636 tok/s**, DFlash
  médian **67,325 tok/s**, 39/45 propositions. Protocole prévu : différentiel
  synthétique bit-à-bit SiLU/GELU, valeurs extrêmes et slots 8/48 ;
  microbenchmark apparié ancien/nouveau kernel ; puis logits + 80 états M=1..8,
  commit DFlash et A/B/B/A E2E. Un seul processus modèle à la fois. Rejet au
  premier écart ou si le gain de frontière ne produit aucun signal complet.
- 2026-09-21, différentiel GPU candidat/référence : sorties FP16 identiques
  bit-à-bit pour SiLU et GeLU, slots 8/48, largeur 512 et largeur logique GeLU
  384, y compris activations de grande amplitude. Microbenchmark apparié dans
  le même processus (8 warmups, 40 mesures par variante, temps d'évaluation
  MLX inclus) : SiLU 8 slots **0,302 → 0,292 ms**, SiLU 48 slots **0,308 →
  0,295 ms** ; GeLU 8 slots **0,298 → 0,296 ms**, GeLU 48 slots **0,264 →
  0,247 ms**. Signal de kernel modeste ; cela ne démontre pas encore un gain
  de modèle. Commande : `MLXL3_GLU_BENCH=1 cargo test --release --all-features
  glu_down_simd_matches_reference -- --ignored --nocapture` avec MLX 0.32.2
  local et SDK macOS 26.2. Logs en sortie terminal, non archivés séparément.
  Statut toujours **en cours** : essais modèle et benchmark apparié requis.
- Modèle Qwen complet : `verification_widths_match_token_major` passe pour
  M=1..8 (logits FP16 et 80 états), avec un seul processus modèle. Premier
  A/B/B/A DFlash (48 tokens, N=5, deux répétitions par processus, sur batterie
  à 86 %, 39/45 propositions et texte identique au greedy) : SIMD **72,114 /
  71,649 / 75,806 / 75,441 tok/s** ; référence **75,277 / 70,875 / 71,548 /
  73,536 tok/s**. Médianes conventionnelles des quatre passages **73,778**
  contre **72,542 tok/s (+1,70 %)**, mais dérive de régime importante
  (70,875–75,806 tok/s). Le ratio DFlash/greedy par processus varie aussi
  avec le greedy ; ce n'est pas une attribution causale. **Non concluant pour
  l'intégration** à ce stade ; une seconde série appariée doit confirmer.
- Seconde série A/B/B/A séquentielle avec trois répétitions par processus :
  SIMD **73,648 / 71,914 tok/s** (médianes par passage), référence **72,546 /
  75,760 tok/s**. Médianes combinées **72,781** contre **74,153 tok/s
  (−1,85 %)**, soit le signe inverse de la première série. Les runs individuels
  couvrent 68,847–75,921 tok/s selon le régime machine, toujours 39/45 et
  sorties target exactes. Décision : **rejeté, non intégré** ; le gain isolé du
  kernel n'est pas reproductible bout en bout et ne justifie pas le shader,
  l'occupation de registres ni le commutateur. Le prototype est retiré ; aucun
  gain +50 % n'est attribué à D09. Machine sur batterie pendant les deux
  séries, ce qui limite la stabilité des chiffres absolus.

### OPT-2026-09-21-RUST-PERF-111 — repacking interne du head EXL3 par paires N — rejeté

- Source : ticket D04 du rapport du Bureau, après D01/D02/D23 intégrés et D09
  rejeté. L'adresse QMV du head reste `(tile_k * TILES_N + tile_n) * PACKED_U32` ;
  pour une même sortie elle saute donc une grande plage en parcourant K. Ce
  constat d'indexation ne prouve pas à lui seul une perte de bande passante.
- Antécédents : PERF-104/105 (MB=3) et PERF-106 (extraction 32 bits) rejetés ;
  aucun essai de repacking de tuiles du head n'est documenté. Xcode complet,
  `xctrace` et l'outil `metal` sont absents du Mac (CommandLineTools seuls),
  donc aucun compteur L1/DRAM matériel n'est disponible ici.
- Hypothèse : conserver les codewords mais disposer les tuiles du seul
  `lm_head` en `[groupe_N/2, tuile_K, N_local(2), mots]` peut réduire les sauts
  du QMV M=1/5/6. Prototype de **ce seul head** : deux copies transitoires en
  RAM sont permises pour l'A/B, mais un résultat positif ne sera pas intégré
  tant que le QMM/prefill n'utilise pas la même disposition sans copie
  permanente supplémentaire.
- Baseline de référence : PERF-109 greedy **44,636 tok/s**, DFlash **67,325
  tok/s** chaud ; la série D09 a dérivé de 68,847 à 75,921 tok/s sur batterie.
  Protocole : inversion du repacking octet par octet, sorties du head M=1/5/6/8
  identiques, microbenchmark alterné ancien/nouveau, puis A/B/B/A E2E 48
  tokens N=5 avec un seul modèle à la fois. Mesurer aussi taille du buffer,
  charge et pic mémoire. Rejeter si aucun gain E2E reproductible. Statut :
  **en cours**, prototype avant code.
- Prototype du seul `lm_head` (trellis 128×15 520×96 en I16, **381 419 520
  octets ≈ 0,355 GiB**) : inversion du repacking octet par octet et sorties
  FP16 M=1/5/6/8 identiques à l'original. Microbenchmark alterné, quatre
  warmups puis 20 mesures/variante : M=1 **3,314 → 3,254 ms** ; M=5 **7,748
  → 7,689 ms** ; M=6 **7,839 → 7,738 ms** ; M=8 **10,374 → 10,230 ms**.
  Baisse isolée de **0,76 à 1,81 %** suivant la largeur, alors que ce seul
  prototype garde une seconde copie de 0,355 GiB et devrait aussi adapter le
  QMM/prefill pour remplacer l'original. À neuf blocs, même l'économie M=5
  mesurée de 0,059 ms/appel ne représenterait qu'environ 0,53 ms sur ~650 ms
  de decode spéculatif, sous les variations observées du banc E2E. Décision :
  **rejeté pour le head seul, non intégré** ; un test E2E de ce sous-cas n'aurait
  pas de pouvoir discriminant. Le repacking des experts (autre mécanisme, accès
  sparse) reste non testé, pas rejeté par extrapolation. Source de mesure :
  `MLXL3_HEAD_REPACK_BENCH=1 cargo test --release --all-features
  head_pair_repack_matches_original -- --ignored --nocapture` avec MLX 0.32.2,
  SDK 26.2, un processus GPU, sur batterie. Pic RAM processus et compteurs
  GPU **non mesurés**. Le prototype head est retiré.

### OPT-2026-09-21-RUST-PERF-112 — validation du seuil DFlash +50 % à 128 tokens — validé (diagnostic)

- Répétition de validation, pas une optimisation nouvelle : PERF-109 mesurait
  1,508× (48 tokens, N=5, un prompt) ; la campagne D09 a trouvé 1,428–1,599×
  selon passage et régime, sans gain causal D09. Avant de citer « +50 % » dans
  une communication, vérifier la tenue sur une sortie plus longue avec le code
  `main` inchangé (`ffb81a2`).
- Protocole : `MLXL3_DFLASH_TOKENS=128 MLXL3_DFLASH_REPEATS=3
  MLXL3_DFLASH_PROPOSALS=5 cargo test --release --all-features
  benchmarks_dflash_end_to_end_greedy -- --ignored --nocapture --test-threads=1`,
  Qwen3.6-35B-A3B EXL3 2.49 bpw + DFlash2 local, MLX 0.32.2, M5 sur batterie ;
  séquence greedy exacte obligatoire, un seul processus modèle. Mesurer débit,
  préfixes acceptés, target/draft et variation entre répétitions. Cette matrice
  reste limitée à un prompt ; si le seuil échoue, ne pas l'annoncer comme
  propriété du projet. Statut initial : **en cours**.
- Résultat : greedy **46,970 / 47,956 / 47,630 tok/s**, DFlash **63,682 /
  63,935 / 61,669 tok/s** ; médianes **47,630 contre 63,682 tok/s**, soit
  **1,337× (+33,7 %)**, et séquences strictement identiques. Acceptation
  **101/140 (72,1 %)**, 28 blocs ; draft 0,372–0,384 s, target 1,612–1,673 s.
  Batterie 84→82 % pendant la campagne, un seul modèle, aucun OOM. Conclusion
  **validée pour ce prompt et ce budget seulement** : le seuil +50 % observé
  à 48 tokens **ne tient pas** à 128 tokens. Ne pas le présenter comme gain
  général de MLXL3 ou de l'app. Aucun code de production modifié.

### OPT-2026-09-21-RUST-PERF-113 — largeur DFlash sur sortie longue — rejeté

- Nouvelle condition par rapport au balayage PERF-73/79 : leurs N=2..7
  portaient sur la sortie courte de 48 tokens et avaient retenu N=5. PERF-112
  montre qu'à 128 tokens l'acceptation tombe de 86,7 à 72,1 % et le ratio de
  +50,8 % environ à +33,7 %. L'hypothèse testée est qu'une largeur plus courte
  peut éviter les suffixes rejetés en seconde partie de génération.
- Baseline N=5, 128 tokens, trois répétitions : greedy médian **47,630 tok/s**,
  DFlash médian **63,682 tok/s**, 101/140 propositions. Protocole paramétré
  N=3, N=4, puis contrôle N=5 adjacent, deux ou trois répétitions par processus,
  même prompt/poids/code et texte exact ; éventuellement N=7 seulement si le
  signal justifie. Mesurer tok/s réel, préfixes acceptés, nombres de blocs et
  temps draft/target. Un seul modèle en mémoire à chaque moment. Rejet si la
  différence se noie dans la dérive, même si le ratio DFlash/greedy fluctue.
  Aucun code modifié avant ce balayage. Statut : **en cours**.
- Première sous-série longue : N=3 **58,163 / 58,328 tok/s**, 89/117 acceptés
  et 39 blocs ; N=4 **62,592 / 62,404 tok/s**, 98/124 et 31 blocs ; contrôle
  N=5 **59,077 / 64,261 tok/s**, 101/140 et 28 blocs. Un passage N=5 a subi
  une baisse simultanée du greedy (42,076 contre 48,530 tok/s) ; le meilleur
  contrôle proche reste 64,261, au-dessus des N=3/4. Le raccourcissement de
  largeur ne récupère donc pas les +50 %. Essai N=6/7 ajouté avant mesure :
  vérifier si moins de blocs compensent leur vérification plus large sur
  128 tokens, malgré le résultat court défavorable de PERF-73/79. Deux
  répétitions par variante, puis contrôle N=5 adjacent.
- Seconde sous-série : N=6 **56,352 / 56,407 tok/s**, 102/162 acceptés,
  27 blocs ; N=7 **55,020 / 55,064 tok/s**, 103/182, 26 blocs ; contrôle N=5
  **63,491 / 62,312 tok/s**, 101/140, 28 blocs. Les séquences sont égales au
  greedy pour chaque variante ; pas de changement de qualité. La baisse de
  deux blocs au plus ne compense ni le travail M=7/8 ni les propositions
  rejetées. Toutes les commandes utilisent `MLXL3_DFLASH_TOKENS=128`,
  `MLXL3_DFLASH_REPEATS=2`, `MLXL3_DFLASH_PROPOSALS=N`, `--test-threads=1`,
  MLX 0.32.2 et SDK 26.2. Batterie environ 82 % au départ, valeur finale
  **non mesurée** ; logs en sortie terminal, pas de fichier brut conservé.
  Décision : **rejeté** pour N≠5 dans ce workload ; conserver N=5. Aucun code
  exécutable modifié ou intégré, et le gain global +50 % n'est pas atteint à
  128 tokens par simple changement de largeur.

### OPT-2026-09-21-RUST-PERF-114 — D04 repacking des experts sparse — rejeté

- Suite distincte de PERF-111 : le `lm_head` repacké par paires ne gagnait que
  0,76–1,81 % dans son microbenchmark et a été rejeté. Les poids d'experts sont
  lus avec des routes sparse ; leur localité et leurs `tile_ns` diffèrent du
  head dense. Le rejet du head n'est donc pas une mesure des experts.
- Hypothèse : sur une couche MoE réelle Qwen 2.49 bpw, réordonner les seules
  tuiles trellis en `[groupe_N/2, tuile_K, paire, mots]` et adapter l'adresse
  du kernel mapped peut accélérer gate/up et down sans modifier codewords ni
  FMA. Prototype limité à **une couche et un seul processus**, pas tout le
  modèle, pour contrôler la RAM ; copies transitoires permises pour l'A/B.
- Baseline : PERF-112, 128 tokens N=5, **63,682 tok/s** DFlash et **47,630
  tok/s** greedy, +33,7 %, 101/140 acceptés ; ce benchmark n'est pas un
  résultat du repacking. Protocole : inversion octet par octet, sorties exactes
  pour 8/48 routes gate/up et down, puis microbenchmark alterné. Si le gain
  isolé est net, étudier QMM/prefill et mémoire avant une intégration complète
  et A/B E2E ; sinon retirer. Versions et températures mesurées dans les
  résultats, ne pas extrapoler le cas d'une couche aux 40 couches. Statut :
  **en cours**, journalisé avant code.
- Premier différentiel de la couche 0 : repacking et inversion octet par octet
  exacts pour gate/up et down ; sorties FP32 du QMV identiques bit-à-bit aux
  deux tailles (8 et 48 routes). Microbenchmark alterné dans un seul processus
  (4 warmups, 24 mesures/variante) : gate/up slots 8 **0,326 → 0,316 ms**,
  down slots 8 **0,457 → 0,268 ms**, gate/up slots 48 **0,483 → 0,492 ms**,
  down slots 48 **0,371 → 0,367 ms**. Le fort signal down à 8 routes ne se
  retrouve pas à 48 et peut être bruit/cache. Ce résultat ne démontre aucun
  gain DFlash M=6, ni un gain de modèle. Commande :
  `MLXL3_EXPERT_REPACK_BENCH=1 cargo test --release --all-features
  sparse_expert_pair_repack_matches_original -- --ignored --nocapture`; batterie
  ~80 %, MLX 0.32.2, SDK 26.2. Répéter le microbenchmark avec contrôle
  thermique avant de décider ; un seul processus GPU. Statut **en cours**.
- Deux répétitions supplémentaires avec le même binaire et les mêmes entrées :
  slots 8 gate/up **0,253 → 0,256** puis **0,254 → 0,250 ms** ; slots 8 down
  **0,219 → 0,218** puis **0,220 → 0,220 ms** ; slots 48 gate/up **0,486 →
  0,484** puis **0,488 → 0,486 ms** ; slots 48 down **0,365 → 0,364** puis
  **0,369 → 0,365 ms**. Le premier 0,457 → 0,268 ms à 8 routes était donc un
  transitoire/cache, pas un gain reproductible. L'économie stable est nulle à
  ~1 % et la forme DFlash slots 48 reste neutre. Décision : **rejeté**, aucun
  E2E n'est justifié ; prototype et commutateur d'adresse retirés, aucune
  disposition supplémentaire conservée. Les chemins autres qu'un simple
  groupement par paires N ne sont pas invalidés par cet essai. Aucun pic RAM
  physique ni compteur GPU mesuré, absence de Xcode Instruments.

### OPT-2026-09-21-RUST-PERF-115 — D34 choix Q4 par forme du draft — non concluant

- Après relecture du rapport du Bureau : D01/D02/D23 sont intégrés ; D03,
  D09 et les deux essais D04 sont rejetés sous leurs formes testées. D34 est
  distinct du balayage QKV historique : les petites projections 2048→512,
  les projections 4096→2048 et 6144→2048 utilisent encore le réglage QKV
  `N128Pipelined` sans mesure spécifique à leurs formes.
- Hypothèse : `N128` ou `N256` avec nombre de groupes adapté réduit le temps
  réel de certaines projections du draft, en conservant exactement les mêmes
  sorties BF16. Baseline E2E : PERF-112, 128 tokens, N=5, greedy médian
  **47,630 tok/s**, DFlash **63,682 tok/s**, +33,7 %, batterie M5, MLX
  0.32.2 et SDK 26.2. Ce chiffre n'est pas un résultat de D34.
- Protocole prévu : un seul processus GPU, poids réels de la couche 0,
  entrées BF16 fixes de 8 lignes ; comparer `N128`, `N128Pipelined` et `N256`
  avec géométries valides sur projections petites/moyennes/grandes. Vérifier
  les sorties octet par octet, réchauffer, alterner les ordres et relever des
  médianes/dispersion par forme. N'intégrer une variante que si le signal
  dépasse le bruit puis vérifier le draft complet et l'E2E exact A/B/B/A.
  Si l'écart isolé est faible ou inverse, rejeter et retirer le prototype.
  Aucune revendication de +50 % à 128 tokens sans nouveau test apparié.
- Premier microbenchmark avec les quatre formes, MLX 0.32.2/SDK 26.2,
  `cargo test --release --all-features benchmarks_real_q4_shape_kernels --
  --ignored --nocapture --test-threads=1` : toutes les variantes reproduisent
  exactement les sorties BF16. Médianes en ms (`N128Pipelined` actuel →
  `N128` → `N256`) : 2048→512 **0,230 → 0,230 → 0,255** ;
  4096→2048 **0,281 → 0,278 → 0,363** ; 6144→2048 **0,312 →
  0,312 → 0,383** ; 2048→6144 **0,294 → 0,288 → 0,304**.
  Dispersion p90 importante sur les petites formes : **non concluant** pour
  les écarts de 0–0,006 ms ; `N256` régresse de manière nette pour les formes
  moyennes. Le QKV N128 gagne peut-être ~2 % isolé mais avait déjà été étudié
  historiquement. Répéter avec le même binaire pour voir si l'écart tient avant
  tout changement de production. Pic RAM processus non mesuré.
- Répétition immédiate, même binaire/poids/entrées : médianes en ms dans le
  même ordre : 2048→512 **0,442 → 0,466 → 0,581** ; 4096→2048 **0,554 →
  0,527 → 0,672** ; 6144→2048 **0,425 → 0,476 → 0,577** ; 2048→6144
  **0,305 → 0,308 → 0,321**. Les p90 montent jusqu'à **1,856 ms** ; au
  moment du contrôle, Mac sur batterie à **78 %**, Deezer Renderer ~75 % CPU,
  WindowServer ~45 %, swap utilisé ~2,83 Go. Cela ne prouve pas la cause de
  l'instabilité GPU, mais rend ces écarts de quelques microsecondes non
  attribuables à D34. Les sorties BF16 restent identiques octet par octet sur
  les quatre formes et trois kernels ; aucun gain E2E ni gain +50 % n'est
  mesuré. Décision : **non concluant pour N128**, **N256 défavorable** sur les
  formes moyennes testées ; aucun changement de production. Le test
  temporaire a été retiré, le code de production est inchangé. Reprendre
  seulement avec une machine moins chargée et un A/B E2E si un micro-signal
  net réapparaît ; logs uniquement dans la sortie terminal, non archivés.

### OPT-2026-09-21-RUST-PERF-116 — D36 épilogue sparse à 32 lanes — interrompu

- Source : D36 du rapport du Bureau, distinct de PERF-68. Ce dernier
  reproduisait l'épilogue MLX exactement avec seulement 8 threads et 16
  valeurs par thread, mais régressait de **1,7 %** E2E. Le code courant
  `finish_and_reduce` matérialise Hadamard FP16, deux multiplications FP16
  puis une réduction top-k. Le header MLX 0.32.2 confirme le radix-16,
  cast FP16, puis radix-8 et multiplication de scale pour les blocs N=128.
- Hypothèse : répartir chaque bloc de 128 sur un SIMDgroup de 32 lanes
  (4 valeurs/lane), huit SIMDgroups par threadgroup pour top-k=8, puis
  réduire dans l'ordre canonique diminue registres et dispatchs sans changer
  les sorties. Première étape : scales déjà rassemblés, pour isoler la
  transformation ; lecture directe par route seulement si ce test gagne.
- Baseline E2E : PERF-112, Qwen3.6-35B-A3B EXL3 2.49 bpw, 128 tokens,
  N=5, greedy **47,630 tok/s**, DFlash **63,682 tok/s** (+33,7 %), mais
  ce n'est pas un résultat de D36. Protocole prévu : différentiel isolé
  Hadamard puis épilogue complet sur valeurs FP16 structurées/aléatoires,
  slots 8/48 et top-k 8, y compris routes/échelles/scores non triviaux ;
  rejet au premier écart bit-à-bit. Ensuite microbenchmark alterné, puis
  logits et 80 états M=1..8, puis A/B/B/A 48 et 128 tokens si le gain isolé
  dépasse le bruit. Un seul processus chargeur de modèle à la fois ; Mac
  actuellement sur batterie et activité tierce observée, donc ratio exact
  à confirmer. Aucun changement de poids ni de sampling.
- Premier prototype limité au Hadamard, 32 lanes et quatre valeurs/lane :
  sorties FP16 **identiques bit-à-bit** à MLX pour slots 8/48, largeur 2048,
  entrées structurées signées. Commande :
  `cargo test --release --all-features d36_hadamard_32_matches_mlx --
  --ignored --nocapture --test-threads=1` (MLX local, SDK 26.2).
  Micro-médianes alternées, 40 mesures/variante après 8 warmups : slots 8
  MLX **0,216 ms** contre 32-lane **0,224 ms** ; slots 48 MLX **0,208 ms**
  contre 32-lane **0,203 ms**. Ce test isolé est essentiellement neutre ;
  la raison d'essayer la fusion reste la suppression des multiplications,
  de la réduction et des buffers intermédiaires. Aucun gain E2E revendiqué.
- Deuxième étape du même essai : Hadamard 32-lanes **plus les deux produits
  FP16**, réduction MLX conservée. Pour slots 8 et 48, largeur 2048,
  scales/scores signés déterministes non triviaux : sorties pondérées
  **identiques bit-à-bit** à `hadamard_transform().mul(scales).mul(scores)`.
  Micro-médianes alternées (40 mesures/variante après 8 warmups) : slots 8
  MLX **0,324 ms** contre fusion **0,301 ms** (−7,1 %), slots 48 MLX
  **0,346 ms** contre fusion **0,304 ms** (−12,1 %). Commande :
  `cargo test --release --all-features d36_weighted_hadamard_32_matches_mlx
  -- --ignored --nocapture --test-threads=1`, MLX 0.32.2 et SDK 26.2,
  batterie. Ces temps sont des microbenchmarks, pas des gains de couche ou
  de génération. Étape suivante : conserver le test, brancher le candidat
  uniquement pour A/B ciblé, comparer logits/états et E2E avant décision.
- Contrôle supplémentaire de la réduction MLX inchangée : après la fusion,
  `reshape([rows,8,width]).sum(1)` produit aussi des sorties FP16 bit-à-bit
  identiques pour slots 8/48. Le second passage micro donne slots 8 **0,309
  → 0,297 ms** et slots 48 **0,348 → 0,302 ms**, même entrée et même méthode.
  Prototype branché derrière `MLXL3_D36_EPILOGUE` seulement, désactivé par
  défaut ; aucun gain de modèle ni d'application n'est encore établi.
- À la demande de l'utilisateur, campagne arrêtée avant le différentiel
  modèle référence/candidat et l'A/B E2E. Le test de cohérence interne Qwen
  M=1..8 avec le kernel candidat a réussi, mais il ne compare pas le modèle
  complet à la référence. La compilation du test différentiel modèle a été
  interrompue ; résultat **non mesuré**. Le candidat Metal, son interrupteur
  d'environnement et les tests temporaires ont été retirés avant la release.
  Le micro-gain ne doit pas être présenté comme gain d'inférence validé.

### OPT-2026-09-22-RUST-AUDIT-01 — contrat du bridge DFlash et mesures — validé localement

- Source : audit `MLXL3_Audit_Prefill_Decode_DFlash2_UI_2026-09-22.md`, tickets A01–A04/I03. Hypothèse : les écarts du benchmark et des compteurs Desktop empêchent d'attribuer correctement les variations de performance ; ce travail est d'abord diagnostique, **aucun boost n'est présumé**.
- Baseline historique, non reproduite ici : PERF-112, Qwen3.6-35B-A3B EXL3 2.49 bpw + DFlash2, M5 sur batterie, 128 tokens, greedy 47,630 tok/s, DFlash 63,682 tok/s (+33,7 %), 101/140 propositions acceptées. Le benchmark utilise `argmax` brut tandis que le bridge utilise `log_probs().argmax()` ; le bridge indique `peak_memory_gb` depuis `checkpoint.size_bytes`, et `first_text_seconds` depuis la fin du round.
- Changement prévu : exposer build/mode effectifs, partager la sélection greedy entre bridge et benchmark, corriger l'horodatage du premier texte et la sémantique mémoire, ajouter des compteurs DFlash et une mesure bridge reproductible. D'abord contrôles de contrat, puis au plus **un** processus modèle à la fois pour un A/B/A réel ; aucune publication demandée.
- Protocole : `cargo fmt --check`, `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`, `cargo test --locked --features mlx,chat`, checks Swift/Desktop, tentative Kani bornée ; comparaison des IDs et des événements JSON sur cas normaux, DFlash, erreur et rounds d'outils. Benchmark réel : mêmes poids, tokenizer, prompt et budget, relevés TTFT/prefill/decode/acceptation/RAM physique, conditions d'alimentation et activité concurrente notées. Statut **en cours**, code local non modifié au moment de cette entrée ; mesures nouvelles **non mesurées**.
- Avancement local : sélection greedy commune, identité build (commit/dirty, release/debug, MLX), capacité DFlash annoncée par le moteur indépendamment du nom du dossier, événement du mode effectif, compteurs proposés/acceptés/blocs et temps mural des blocs sans barrières GPU supplémentaires. Correction du premier texte, agrégation pondérée des temps/tokens de tous les rounds, durée totale incluant chargement draft et outils. La taille disque n'est plus appelée pic mémoire ; MLX active/cache/peak et empreinte physique processus sont séparées (pic processus explicitement **depuis le lancement**, non par requête). Swift conserve la lecture de l'ancien historique sans réétiqueter arbitrairement ses anciennes mesures.
- Vérifications intermédiaires : `cargo test --locked --features mlx,chat` **41 tests réussis**, 30 GPU/modèles ignorés ; test GPU `native_array_and_kernel_smoke -- --ignored --nocapture --test-threads=1` réussi, dont greedy avec égalités FP16/BF16/FP32 et FFI mémoire réelle. Clippy a d'abord refusé un `if` imbriqué du build-script ; corrigé, puis **réussi**. MLX 0.32.2, SDK 26.5. Le build Swift SDK 27 échouait sur les macros SwiftUI ; SDK 26.5 compile. Première suite Desktop : lifecycle/hardening, Markdown et MCP réussis, puis **interrompue par une modification concurrente de Domain.swift** pendant la compilation du test suivant ; suite complète à relancer sans modifier les sources.
- Kani 0.68.0 / CBMC 6.11.0, `cargo kani --no-default-features --lib --harness greedy_acceptance_is_the_maximal_matching_prefix` : **216 obligations, zéro échec, une inaccessible**, deux couvertures satisfaites (aucun/tous acceptés). Longueurs 0..7, IDs u32 symboliques, unwind 9 ; uniquement contrat CPU d'acceptation, pas GPU/MLX/FFI/horloges. Log `/tmp/mlxl3-audit-kani.log`. Tentative CBMC sur `native/ffi/mlx_bridge.cpp --function mlxl3_memory_stats --unwind 2 --unwinding-assertions` **bloquée au parsing libc++**, log `/tmp/mlxl3-audit-cbmc.log` ; aucune preuve du code C++ revendiquée. Contrôles ABI/MLX réels ci-dessus compensatoires. Logs temporaires à archiver dans le rapport final ; aucune mesure de gain nouvelle ni app installée/publication.

- Baseline réelle du bridge corrigé, `scripts/smoke-dflash-bridge.py /tmp/mlxl3-audit-baseline models/Qwen3.6-35B-A3B-EXL3-2.49bpw models/Qwen3.6-35B-A3B-DFlash2 --tokens 48,128,256 --repeats 2` : prompt « Explain lossless speculative decoding in one paragraph. », 20 tokens rendus, contexte 4096, greedy temp=0/top_k=1/pénalité=1, ABBA après deux warmups, un seul processus. M5 batterie 91 %, MLX 0.32.2, SDK 26.5, `fbfb357edffe-dirty` release. Identité/mode confirmés, IDs (hash), texte et budget **identiques** dans chaque paire. Médianes decode normal/DFlash : **45,539 → 68,739 tok/s** à 48 (**+50,94 %**, 38/45 acceptés), **44,922 → 47,423** à 128 (**+5,57 %**, 92/175), **44,758 → 43,142** à 256 (**−3,61 %**, 181/380). Résultat important : le gain long du sampler applicatif n'est pas celui de PERF-112 (prompt/protocole différents) ; aucune addition/comparaison causale à ces anciens chiffres. Le diagnostic ne crée aucun boost nouveau.
- Cold warmups : première génération normale TTFT 5,863 s ; DFlash 0,830 s et 18,83 tok/s pendant la première compilation de ses kernels. Compteurs steady-state distincts : à 48 tokens, TTFT normal 0,434–0,438 s, DFlash 0,161–0,206 s ; RAM physique moteur ~15,06 GB contre pic MLX ~13,07 GB, poids disque 13,064 GB. Début de warmup contemporain d'une compilation Clippy (~3 s), donc pas de comparaison de latence cold prétendument isolée. JSON brut `/tmp/mlxl3-audit-bridge-baseline.jsonl`, stderr vide. Suite Desktop complète relancée sans toucher Swift : **réussie** (`/tmp/mlxl3-audit-desktop-final.log`), y compris moteur renommé, contrat JSON mémoire/temps, historique legacy, 1194 fragments code Unicode, callbacks et transport.

### OPT-2026-09-22-RUST-AUDIT-02 — préfixe exact du bridge Qwen — validé localement

- Source : P01/P02 du même audit. Le Python historique avait déjà un cache, mais le bridge Rust réinitialise encore à chaque round. Réutilisation de la primitive Qwen snapshot/restore validée dans PERF-46, sans nouveau format ni copie des poids. Une seule conversation chaude et un seul checkpoint de préfill ; pas de LRU multi-modèles.
- Hypothèse : conserver l'état **avant** le decode et avant toute file DFlash pending évite de rejouer l'historique stable après un message ou MCP. Checkpoint uniquement sur une frontière de chunk complète (256 normal / 128 DFlash) pour conserver les mêmes formes/arrondis que le recalcul complet ; aucun trim arbitraire des états récurrents. IDs rendus comparés exactement, même conversation, modèle/tokenizer/processus/contexte et mode draft. Si le préfixe change, recalcul intégral.
- Protocole prévu : base locale AUDIT-01 sans cache, comparaison cache OFF/ON de la génération et du cache logique, tour répété/deuxième tour, modification d'historique/système/outils, changement de conversation et de mode, annulation dans prefill/decode et frontière du contexte. Différentiel GPU logits/états Qwen et cache draft ; E2E bridge séquentiel, mêmes tokens et même budget. Mesurer TTFT/tokens évalués et mémoire retenue, **pas un boost decode présumé**. M5 sur batterie, MLX 0.32.2 / SDK 26.5. Statut prototype à implémenter ; aucun gain mesuré.
- Implémentation locale : un checkpoint target + cache draft, limité à 256 MiB de payload logique target (hors backing MLX et draft borné à 48 MiB). La taille doit être distinguée de la RAM réellement retenue. `reuse_prompt_cache=false` libère le checkpoint ; changement de draft le libère également. Pas de réutilisation entre identifiants de conversation différents. Cache snapshot avant decode, donc jamais de tokens spéculatifs non livrés réutilisés.
- Test GPU de la fonction de production `bridge_prefix_cache_replays_cold_logits_exactly` : **réussi**, 20,53 s modèle chargé compris. Mode normal et DFlash ; replay après mutation du modèle par decode, suffixes 0/1/23/24/25 sur frontières 256/128 ; logits finaux FP16 bit-à-bit égaux au chemin froid indépendant cache OFF. Édition du premier token et changement de conversation : misses ; annulation reconnue et OFF libère l'état. Contrôle des 80 tenseurs/snapshots et des propositions draft à compléter ; ce test seul ne prouve pas tous les états.
- Première passe bridge OFF/ON ABBA (575 tokens rendus, 48 générés) : texte/hashes identiques, 512 tokens réutilisés, seulement 63 évalués. Dernier contrôle chaud : TTFT normal 1,344 s vs 0,271–0,276 s ; DFlash 1,784 s vs 0,284–0,288 s. Le premier contrôle OFF incluait de nouvelles compilations de grandes formes (2,413/2,672 s) : **pas de moyenne mélangeant froid et chaud**. Résultats `/tmp/mlxl3-audit-prefix-bridge.jsonl`. Répétition explicitement motivée : préchauffer maintenant le grand prompt et son suffixe avant ABBA, log `/tmp/mlxl3-audit-prefix-bridge-warm.jsonl`, afin d'isoler le bénéfice réel du cache. Aucune app installée/publication.
- Confirmation chaude ABBA : **TTFT 1,331 → 0,275 s en normal (−79,3 %, ×4,84)**, **1,783 → 0,285 s avec DFlash (−84,0 %, ×6,25)**. 575 tokens de contexte/48 générés, mêmes hashes/texte, 512 tokens cachés. Decode normal ~44,2 tok/s inchangé ; le petit écart DFlash 53,0 → 53,9 n'est pas revendiqué. Surcoût actif MLX de ce checkpoint : **~201 MB normal / ~151 MB DFlash** ; empreinte physique déjà avec cache allocateur ~15,70/~16,13 GB, sans pic nouveau comparable à cette différence de payload. Changement de système et de conversation invalidés dans le vrai bridge. Résultats `/tmp/mlxl3-audit-prefix-bridge-warm.jsonl`, stderr vide.
- Contrat d'éligibilité extrait dans `contracts::reusable_prefix_len` utilisé par la production. Première tentative Kani longueurs 0..8/IDs u32/chunk u8, unwind 10 : **échec d'unwinding dans le memcmp de starts_with** (jusqu'à 32 octets), pas un contre-exemple fonctionnel. Relance prévue avec unwind 34, assertions d'unwinding conservées ; log `/tmp/mlxl3-audit-prefix-kani.log`. Ne pas présenter l'essai initial comme réussi.
- Kani avec unwind 34 : **97 obligations, zéro échec, une inaccessible ; trois couvertures satisfaites** (prefixe partiel réutilisé, égalité complète, chunk nul refusé). Log `/tmp/mlxl3-audit-prefix-kani-34.log`. Domaine : longueurs 0..8, IDs u32 symboliques, chunk u8 y compris 0. Ne couvre pas les buffers GPU ni l'identité complète du runtime. Clone du cache draft : **12 tenseurs KV identiques octet par octet**, extension d'une branche ne modifie pas le snapshot, puis extensions indépendantes identiques ; test `draft_prefix_cache_clone_keeps_all_kv_bytes` réussi en 0,16 s, `/tmp/mlxl3-audit-draft-clone.log`.
- **P02 exercé dans le vrai bridge** : `--check-mcp` utilise une configuration temporaire et un serveur stdio `fixture.echo` avec délai contrôlé de 200 ms ; Exa explicitement désactivé, aucune requête réseau. Normal et DFlash, OFF/ON : un appel MCP et deux rounds, **202 tokens agrégés identiques**, réponse finale/hash identiques. Le second round passe de **895 tokens évalués à 127** (768 cachés). Les compteurs incluent les deux rounds et le délai outil observé 0,200–0,210 s. Log `/tmp/mlxl3-audit-mcp-bridge.jsonl`, stderr vide. Cette passe est un contrôle fonctionnel avec premières compilations de formes et non un nouveau benchmark de pourcentage.
- Test physique snapshot/replay étendu aux checkpoints batchés 128/256 + suffixe de 24 : **logits et 80 états exactement identiques**, 5,34 s, `/tmp/mlxl3-audit-snapshot.log`. Contrôle négatif formel : remplacer temporairement la comparaison des IDs par une comparaison de longueurs fait **échouer l'assertion fonctionnelle Kani** ; mutation immédiatement retirée. Nouvelle passe originale réussie (97 obligations / trois couvertures), `/tmp/mlxl3-audit-prefix-kani-mutant.log` et `/tmp/mlxl3-audit-prefix-kani-final.log`. Aucune mutation conservée. Clippy a signalé un cast i32 redondant dans le test snapshot, corrigé puis clippy strict réussi.
- Dernière validation prévue avant clôture du lot : suite Rust finale et replay bridge après arrêt pendant prefill puis au premier delta (y compris DFlash pending), avec comparaison au recalcul froid. Le test n'exécute toujours qu'un seul modèle ; résultats à consigner. Le test CPU négatif du banc (`tests/test_bridge_benchmark.py`) réussit : absence d'accusé de mode = erreur, jamais ratio de débit trompeur.
- Première passe annulation : **bloquée par le banc**, qui attendait un événement `error` alors que la production émet correctement `cancelled`. Le modèle était déjà au repos après l'arrêt ; aucune erreur moteur observée. Lecteur du banc corrigé pour exiger `cancelled` avec le bon request_id (les autres erreurs restent fatales). Passe interrompue puis relancée ; résultats partiels `/tmp/mlxl3-audit-prefix-cancellation.jsonl` à conserver, pas un test réussi.
- Relance corrigée **réussie** : interruption à `context_usage` puis au premier `delta`, reprise sur le même modèle/conversation ; texte et hash des tokens identiques au recalcul indépendant cache OFF, en mode normal **et** DFlash. Log `/tmp/mlxl3-audit-prefix-cancellation-fixed.jsonl`, stderr vide. La suite Desktop complète répétée après modification du serveur MCP fixture passe aussi (`/tmp/mlxl3-audit-desktop-final2.log`). Kani couvre le contrat CPU borné, ces tests couvrent les chemins GPU/processus concrets ; ils ne prouvent pas tous les entrelacements possibles.

### OPT-2026-09-22-RUST-AUDIT-03 — prefill sériel sans heads jetés — validé sur le chemin sériel

- Source : P03 du même audit ; constat source révisé : Qwen et Ling appellent explicitement `logits.eval()` pour chaque token sériel, donc exécutent réellement la projection vocabulaire ensuite jetée. LFM2/Gemma construisent le graphe head mais ne l'évaluent pas systématiquement : pas de gain GPU présumé pour eux. Le batching Ling/LFM-MoE reste désactivé, ce chantier ne change ni les formes des couches ni les arrondis de leurs calculs.
- Changement prévu : réutiliser le corps hidden/états existant en M=1 pour les tokens non finaux et n'appliquer le head qu'au dernier. Une évaluation hidden par token conserve une borne sur le graphe ; aucune suppression générale des synchronisations. Raccourci limité aux chemins qui font réellement les heads surnuméraires, normal decode inchangé.
- Baseline : AUDIT-01 petit prompt Qwen 20 tokens, ~0,434 s TTFT / 46 tok/s prefill. Protocole : références séparées avec `forward` intégral pour chaque token vs candidat, longueurs 1/2/23/24/25, logits finaux et tous les états bit-à-bit ; erreurs/bornes inchangées. Un modèle à la fois, mesures alternées série M=1 puis bridge au même prompt. MLX 0.32.2 / SDK 26.5, M5 batterie. Aucun gain encore mesuré.
- Première compilation : import `anyhow::Context` manquant dans Ling, corrigé avant test. Qwen : test `qwen_serial_prefill_skips_only_unused_heads` **réussi**, logits finaux et **80 états** bit-à-bit identiques, invalid token rejette et remet l'offset à zéro. Quatre mesures par variante, ordre ABBAABBA après validation/warmup : 1 token 19,590 → 19,761 ms (neutre/bruit) ; 2 tokens 38,880 → 36,037 ms ; 23 tokens **447,030 → 380,994 ms (−14,8 %)** ; 24 tokens **469,023 → 398,222 ms (−15,1 %)** ; 25 tokens **487,591 → 416,008 ms (−14,7 %)**. Micro/chemin prefill sériel, pas +15 % de decode ni de tous les prefills (le batching >=24 demeure inchangé dans le bridge Qwen). `/tmp/mlxl3-audit-head-qwen.log`, commande release `cargo test --features mlx,chat ... -- --ignored --nocapture --test-threads=1`. Test Ling en cours, un autre processus lancé uniquement après sortie de Qwen.
- Ling : même test différentiel **réussi** (logits + tous les états octet par octet, erreurs/reset vérifiés). Mêmes quatre mesures par variante : 1 token 9,226 → 9,205 ms ; 2 tokens 18,243 → 17,229 ms ; 23 tokens **210,170 → 187,407 ms (−10,8 %)** ; 24 tokens **220,147 → 195,289 ms (−11,3 %)** ; 25 tokens **228,337 → 203,233 ms (−11,0 %)**. Poids Ling-3.0-tiny EXL3 4 bpw, `/tmp/mlxl3-audit-head-ling.log`. Le bridge/CLI appelle maintenant ce chemin pour le prefill sériel Ling et les petites queues Qwen. Aucun batching non validé activé, aucun gain decode revendiqué. Build d'app installée/publication **non faits**.

### OPT-2026-09-22-RUST-AUDIT-04 — DFlash borné par la sortie restante — validé localement

- Source : **D04 de l'audit du 22 septembre**, distinct du ticket D04 repacking du rapport précédent (PERF-111/114). Le bridge ne transmet pas son budget restant à `DFlashChat::advance` : son dernier bloc peut vérifier cinq propositions alors qu'il ne manque qu'un token affichable. Ne modifie pas les huit positions bidirectionnelles du réseau draft, seulement le head/verify et le passage normal pour un seul token restant.
- Hypothèse : supprimer ce travail terminal inutile réduit la latence des réponses courtes sans changer les tokens. Baseline : code local AUDIT-03, sélection greedy canonique, Qwen EXL3 2.49 bpw + DFlash2, N=5 ; aucune amélioration présumée du débit des longs segments ni de l'acceptation.
- Protocole avant code : contrat CPU de budget avec Kani (incluant zéro et valeurs limites), test réel production `advance` sur budgets 1..16, frontières contexte, arrêt avec pending et replay depuis prefill. Référence indépendante en decode target normal ; comparaison avec `advance` non borné par la sortie pour ABBA chaud dans le même processus. Vérifier tokens, état logique borné, erreurs sans mutation. Un seul modèle GPU à la fois ; MLX 0.32.2/SDK 26.5, M5 batterie 82 %, swap ~2,70 GiB au départ. Journaliser les régressions de forme/compilation plutôt que promettre un gain global. État : **en cours**, non installé/non publié.
- Contrôle réel **réussi**, 46,21 s modèle chargé compris : deux prompts (explication anglaise / fonction Rust), budgets 1..16, contexte large et frontière serrée, référence target normale indépendante ; tous les IDs identiques. Aucune proposition en attente au-delà du budget borné, offset final exact, budget zéro rejeté sans mutation même avec pending. Quatre mesures par variante après warmups, ABBAABBA, temps du decode seul hors prefill : budget 2 **69,355 → 20,361 ms** / **69,488 → 20,526** selon prompt ; budget 4 **69,271 → 44,878** / **69,455 → 44,937** ; budget 6 **69,422 → 64,650** / **69,723 → 64,576** ; budget 16 **279,385 → 205,078** / **280,588 → 221,400**. Ce gain terminal n'est pas un +50 % général ni un meilleur taux d'acceptation. Log `/tmp/mlxl3-audit-tail-gpu.log` ; test `dflash_output_budget_matches_target_and_skips_unused_work -- --ignored --nocapture --test-threads=1`.
- Kani `speculative_work_reserves_target_within_both_budgets` : **28 obligations, zéro échec, trois couvertures satisfaites**, sans boucle, entrées `usize` symboliques couvrant zéro et MAX. Vérifie zéro/refus, borne N≤5, place pour token target et maximalité sous les deux budgets ; ne prouve pas les kernels ni EOS/ordonnancement. Log `/tmp/mlxl3-audit-tail-kani.log`. Validation bridge finale à budgets courts à venir avant clôture.
- Mutation de contrôle supprimant seulement la borne de sortie : Kani échoue bien sur le refus à zéro et sur `proposals < output`. Mutation immédiatement retirée, original revérifié **réussi** (`/tmp/mlxl3-audit-tail-kani-mutant.log`, `...-final.log`). Clippy strict et tests Rust généraux réussis après restauration ; aucun binaire mutant installé ni lancé sur modèle.
- Validation intégrée `scripts/smoke-dflash-bridge.py ... --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --repeats 1` **réussie** sur le moteur release reconstruit : les 16 budgets ont même texte/hash/nombre de tokens normal et DFlash, mode actif confirmé. C'est une passe fonctionnelle incluant premières compilations, **pas** une comparaison de gain steady-state. `/tmp/mlxl3-audit-tail-bridge.jsonl`, stderr vide. Suite Rust générale finale **42 réussis / 35 GPU-modèles ignorés**, les tests GPU nommés sont exécutés séparément. Décision D04 : **validé localement**, moteur CLI reconstruit, application installée et publication inchangées.

### OPT-2026-09-22-RUST-AUDIT-05 — découpage Markdown incrémental — validé en préparation UI

- Source I01 de l'audit ; antécédents `docs/general-performance-2026-09-07.md` et checks streaming existants : les chunks `Equatable` et le cache incrémental de coloration existent déjà. La liste Markdown rescane néanmoins toutes les lignes à chaque append. Ne remplace pas le renderer, ne retire pas le contrôle des tableaux ni les protections Unicode.
- Hypothèse : reprendre au début des deux derniers chunks, en conservant l'offset **source brut** et les en-têtes de continuation de tableaux (les IDs actuels comptent les en-têtes répétés), réduit la préparation UI des longs messages. Réutiliser le pattern `CodeTextChunker.Cache`, validation exacte du préfixe et fallback complet en cas de remplacement. Pas de nouvelle dépendance ; le contrôle de préfixe reste O(n) en octets, les très grands fences sans frontière restent le ticket I02 séparé.
- Protocole avant code : comparer chaque état incrémental au découpage complet existant, texte/IDs/offsets et tables/fences/math/listes/Unicode avec fragments 1..257, remplacements/troncatures/normalisation Unicode. Microbenchmark chaud alterné 64/256/1024 KiB, 30 appends par cas, temps de préparation (pas tok/s moteur ni FPS). Test AppKit/SwiftUI et suite Desktop SDK 26.5. Aucun modèle GPU pendant la mesure CPU. Statut **en cours**, aucun gain annoncé.
- Première passe : suite Desktop et fragments incrémentaux réussis ; micro préparation Markdown ~2,03 → 0,48 ms à 64 KiB et ~32,40 → 3,57 ms à 1 MiB (prose). Logs `/tmp/mlxl3-audit-markdown.log`. Relecture supplémentaire avant intégration : les pipes sans véritable séparateur et les en-têtes plus larges que 16k exigent de conserver **l'état des lignes de tableau**, pas seulement un préfixe de rendu artificiel. Scanner existant étendu avec ce seul état de continuation, nouveaux cas adverses ajoutés ; relance des contrôles et mesures après cette correction. Compilation Rust contemporaine de la validation fonctionnelle initiale ; les chiffres finaux seront repris sans compilation concurrente.
- Passe finale isolée **réussie** : SDK 26.5, `MLXL3_RENDER_BENCHMARK=1 scripts/check-desktop.sh`, 4 845 fragments Markdown comparés au découpage complet, en plus de 1 194 fragments de coloration et des cas adverses en-tête >16k/pipes sans tableau, normalisations, remplacements et troncatures. Toute la suite Desktop passe (lifecycle, MCP, sauvegarde, callbacks, transport). 30 appends contrebalancés : prose 64/256/1024 KiB **2,074 → 0,383 / 8,418 → 0,756 / 34,137 → 2,008 ms par mise à jour** ; tableaux **1,488 → 0,501 / 5,928 → 0,892 / 23,079 → 2,106 ms**. Ce sont des moyennes de coût CPU de préparation, pas FPS, gain de decode ni garantie d'absence d'écran noir. Logs `/tmp/mlxl3-audit-markdown-final.log`, archivés sous `docs/measurements/audit-2026-09-22-desktop.log`.
- Décision : **validé pour le découpage**. La vérification exacte du préfixe reste O(n) en octets ; les fences ouverts géants non coupés, leur parsing et le temps de rendu réel restent I02, distincts de ce gain. Pas de preuve formelle Swift ; comparaison finie avec le scanner complet + build Swift 6 et tests AppKit/SwiftUI.

### État du lot AUDIT-01 à AUDIT-05 — 22 septembre 2026

- Compte rendu et mesures conservées : [audit bridge/prefill](docs/audit-bridge-prefill-2026-09-22.md), `docs/measurements/audit-2026-09-22-*`. Les logs négatifs temporaires cités restent disponibles à la clôture ; leur disparition future ne les transforme pas en résultats réussis.
- Rust : fmt/diff check, Clippy strict, 42 tests généraux, GPU explicitement nommés, vrais bridges budgets/MCP/annulation ; trois contrats Kani (97/216/28 obligations), contrôles mutants prefix/budget rejetés puis source originale revérifiée. C++ bloqué au parsing CBMC ; Python/Swift et GPU vérifiés dynamiquement, pas de preuve universelle. Pas de test modèle en cours ni de processus GPU conservé.
- Intégration : **code local**, CLI lié au release reconstruit, Swift compile ; **application installée non remplacée, aucun push ni release de ce lot**. I03 n'a que les compteurs finaux, pas encore une progression/mesure du retard UI. L'audit de 32 pistes n'est **pas terminé** : P04, P06/P07, D01/D02/D08 et I02–I04 notamment restent à traiter/évaluer. Aucune publication n'est implicite.

### OPT-2026-09-22-RUST-AUDIT-06 — suffixe utile du prefill draft — validé fonctionnellement, gain complet non établi

- Source P04, audit du 22 septembre. Le cache draft conserve 2048 positions mais le bridge projette toutes les captures depuis le début, en blocs de huit, avant d'en jeter la majorité. Ne change ni le contexte target intégral ni ses chunks (128 avec DFlash), ni le padding ou la bidirectionnalité du draft. Le rejet historique du Q4 sans padding (PERF-92) n'est pas répété.
- Hypothèse : avant le suffixe utile, appeler le forward target batched existant **sans capture**, puis réutiliser le chemin de capture actuel. Première capture arrondie au chunk de 128 précédent `len−2048`. Puisque 2048 est multiple de 128, cela conserve aussi 2048 positions exactes au checkpoint de conversation précédant le dernier tail. Lors d'une reprise, le cache draft restauré n'est jamais utilisé pour proposer pendant le prefill ; sa queue finale sera reconstruite avant toute proposition.
- Baseline : AUDIT-01..05 locaux, N=5 / Qwen EXL3 2.49 bpw, aucun gain long-prompt encore mesuré. Protocole : Kani du calcul de frontière (toutes longueurs usize), comparaison exacte des 12 KV draft et des sorties draft sur longueurs 2047/2048/2049, 4k et 8k ; contrôle target avec/sans capture (logits/80 états), puis vraie fonction bridge, cache froid/replay et IDs de continuation. Micro-coût draft distinct du prefill E2E ; A/B chaud contrebalancé, tailles/contexte/pics conservés, un seul modèle GPU. M5 sur batterie 76 %, MLX 0.32.2/SDK 26.5, aucun processus modèle ouvert au départ. Ne pas revendiquer de gain decode. Statut **en cours**, aucune installation/publication.
- Contrat `prefill_capture_start` : Kani **34 obligations, zéro échec, trois couvertures satisfaites**, toutes longueurs usize, y compris MAX ; frontières alignées et fenêtre complète à la fin du prompt **et** à son checkpoint. Test GPU draft isolé `draft_prefill_suffix_matches_full_cache_exactly` **réussi** (19,74 s) : 12 tenseurs KV et sorties hidden/selector identiques octet par octet sur 2047/2048/2049/4096/4097/8192 positions. ABBA après warmup, cache draft seul : 2047 **383,447 → 383,452 ms** ; 2048 **382,919 → 385,270** ; 2049 **393,628 → 386,888** (mêmes calculs, bruit) ; 4096 **730,191 → 384,905** ; 4097 **743,491 → 388,057** ; 8192 **1433,544 → 389,484**. À 8k, seules 2048 captures sont projetées ; le target n'est pas dans ce microbenchmark. Logs `/tmp/mlxl3-audit-suffix-kani.log`, `/tmp/mlxl3-audit-suffix-kv.log`. Contrôle modèle/bridge en cours avant toute décision de gain E2E.
- Contrôle target réel `captured_prefill_keeps_uncaptured_target_state` **réussi**, 8,54 s : après un préfixe non nul de 128 tokens, blocs 23/24/127/128/129 ; capture présente/absente = mêmes logits FP16 et **80 tenseurs d'état octet par octet**. Les formes et l'ordre des couches ne changent pas. Log `/tmp/mlxl3-audit-suffix-target.log`. Lancement de la comparaison de la fonction bridge entière après libération de ce modèle (jamais deux instances).
- Test de la vraie fonction bridge **réussi** (346,52 s, `cargo test --release --locked --features mlx,chat bridge_draft_suffix_keeps_full_prefill_and_replay -- --ignored --nocapture --test-threads=1`) : aux longueurs 2047/2048/2049/4096/4097/8192, 32 IDs, logits de prefill, features draft et compteurs d'acceptation identiques à la référence full-capture ; reprise d'un checkpoint partiel contrôlée. Après warmup et ordre OFF/ON/ON/OFF, prefill complet à 4096 **11,703 → 11,659 s** et à 8192 **25,666 → 25,959 s**. Le micro-gain draft de 1,434 → 0,389 s à 8k ne se traduit donc pas en gain de prefill complet dans ces conditions ; l'écart E2E est du bruit ou défavorable. Empreinte physique moteur à 8k ~18,51 GB, pic depuis lancement ~18,57 GB, tandis que l'affichage système atteint ~22/24 GB avec les autres processus. Preuve `docs/measurements/audit-2026-09-22-suffix-bridge.log` ; MLX 0.32.2/SDK 26.5, M5 batterie 100 % au départ. **Décision : validé fonctionnellement, gain complet non établi ; ne pas promouvoir P04 comme optimisation de TTFT.** Le chemin candidat reste dans le checkout local existant, aucune app installée ni publication, aucun processus GPU conservé.

### OPT-2026-09-22-RUST-AUDIT-07 — cartographie complète decode puis prefill — terminé après correction

- Objet : audit et classement des prochaines optimisations, sans modification du moteur ni activation de prototype. Cette campagne de diagnostic est distincte des essais de kernels déjà consignés ; elle mesure le bridge et les modèles du checkout actuel, en donnant la priorité au decode, puis au prefill. Elle ne réexécute pas les variantes rejetées comme candidats.
- Référence figée avant journalisation : commit `fbfb357edffee4494a1be152666cf1ef33e47c9d`, `git diff --binary` SHA-256 `30e9dd5dee9153705e896db1ffd7d9bcaa821c0462e1a0680af3e7953bdba1a5`, avec 20 fichiers suivis modifiés et des fichiers non suivis ; les détails figurent dans `docs/measurements/audit-2026-09-22-inventory.txt`. Les changements AUDIT-01..06 restent locaux. Baseline de performance antérieure pertinente : AUDIT-01, même Qwen/MLX mais avant AUDIT-06 ; elle n'est pas additionnée aux nouveaux chiffres.
- Matrice prévue : Qwen3.6-35B-A3B EXL3 2.49 bpw normal et DFlash2 (draft Q4 local), Ling-3.0-tiny 4 bpw, LFM2.5-1.2B 4 bpw et LFM2.5-8B-A1B 3.10 bpw local. Gemma 4 sans checkpoint local : inspection statique, performance « non mesuré ». Decode : prompts courts/longs, 48/128/256 tokens si la sortie atteint le budget, warmup exclu, répétitions alternées normal/DFlash pour Qwen ; tokens/hash, acceptation, TTFT, débit moteur/client et mémoire. Prefill : 1/23/24/25/127/128/129/2048/4096/8192 selon le chemin et le contexte réellement disponible, cache OFF/ON, séparation du micro-coût draft et du bridge complet. Les entrées exactes, répétitions et limites sont enregistrées avec les résultats bruts.
- Contrôle qualité : terminaison de l'essai AUDIT-06 déjà enregistré avant toute nouvelle référence DFlash ; vérifications bit-à-bit préexistantes des logits/états et hashes/texte du bridge, sans déduire une parité numérique des seuls hashes. Un seul processus modèle GPU à la fois ; aucune comparaison de métriques de formes ou de protocoles différents. Captures synchronisées et échantillonnage CPU servent à attribuer les coûts, pas à revendiquer des temps GPU purs.
- Environnement initial : Apple M5, 24 Gio, macOS 27.0 `26A428`, batterie 100 % en décharge, MLX 0.32.2, SDK 26.5, Xcode Instruments et `metal` indisponibles avec les Command Line Tools actifs. Mesures et rapport à conserver sous `docs/measurements/` et `docs/`. Résultats : non mesurés au départ ; état d'intégration : diagnostic local seulement.
- Première matrice bridge courant (`.venv/bin/python scripts/smoke-dflash-bridge.py ./target/release/mlxl3-rs models/Qwen3.6-35B-A3B-EXL3-2.49bpw models/Qwen3.6-35B-A3B-DFlash2 --tokens 48,128,256 --repeats 3 --context-length 4096`, prompt par défaut, warmup exclu, ordre alterné) : à 48 tokens **46,129 → 72,809 tok/s** normal→DFlash (+57,8 %, 38/45 acceptés) ; à 128 **43,471 → 50,470** (+16,1 %, 92/166) ; à 256 **45,699 → 45,298** (−0,9 %, 179/377). Les trois budgets sont atteints et chaque paire normal/DFlash donne le même hash/texte ; le mode effectif est confirmé. Ratios de médianes de trois runs par mode, pas gains causaux d'AUDIT-06 ni preuve numérique complète. Premiers textes médians : 48 tokens **0,363 → 0,194 s**, mais le prefill normal et DFlash utilisent des chemins/chunks différents ; ce n'est pas un gain de decode. Pic MLX médian ~12,16 GB normal / 13,07 GB DFlash ; empreinte physique d'un processus résident commun ~15,05–15,52 GB selon la série, pas une attribution par mode. Batterie 100→97 %, stderr vide, aucun processus modèle laissé actif. Preuve `docs/measurements/audit-2026-09-22-current-qwen-dflash.jsonl`. La chute d'acceptation et du ratio avec la longueur confirme le problème de rendement variable, sans identifier à elle seule un kernel gagnant.
- Bridge Ling court (`.venv/bin/python benchmarks/benchmark_bridge.py models/Ling-3.0-tiny-EXL3-4bpw --native-binary ./target/release/mlxl3-rs --max-tokens 128 --repeats 3`) : modèle 4 bpw, 37 tokens d'entrée, 128 générés aux trois passages chauds, même SHA256 de texte ; decode moteur **81,99 tok/s** médian (81,83–83,29), decode client **81,99 tok/s**, TTFT **0,385 s** médian, prefill sériel **96,17 tok/s** médian. Empreinte physique processus ~4,775 GB, chargement 1,287 s. Ces débits ne sont pas directement comparables aux 84/128 tokens du PERF-30 ni au meilleur contrôle README, car le prompt et l'état thermique diffèrent. Preuve `docs/measurements/audit-2026-09-22-current-ling-short.json`. Aucun candidat testé.
- Premier bridge LFM2.5-1.2B 4 bpw court (`benchmarks/benchmark_bridge.py ... --max-tokens 128 --repeats 3`) : 26 tokens d'entrée, 128 générés, même hash ; decode **114,34 / 114,66 / 114,61 tok/s**. Le warmup court ne préchauffait pas la forme TensorOps M≥24 : première requête mesurée **TTFT 0,907 s / prefill 28,67 tok/s**, puis **0,035/0,034 s / 748,91/766,91 tok/s**. Ne pas mélanger la première compilation et les passages chauds. Empreinte physique ~0,955 GB, chargement 0,219 s. Preuve `docs/measurements/audit-2026-09-22-current-lfm12-short.json`. Répétition de validation enregistrée : cinq passages avec le même prompt, exclure explicitement le premier passage de forme froide, conserver les quatre suivants pour les médianes chaudes ; aucune comparaison de candidat.
- Répétition LFM1.2 terminée : nouvelle invocation identique avec `--repeats 5`, tous les passages désormais chauds (la compilation de forme de la passe précédente n'est pas refaite). Sur les quatre derniers : **126,97 tok/s decode**, **794,56 tok/s prefill**, **0,0330 s TTFT** médians, 26/128 tokens et hash identiques. Ce résultat n'est pas un +11 % attribuable à un changement de code face à la série précédente : charge, chauffe et caches compilés ont changé entre processus. Preuve `docs/measurements/audit-2026-09-22-current-lfm12-short-repeat.json`.
- Bridge LFM2.5-8B-A1B EXL3 3.10 bpw court (`benchmarks/benchmark_bridge.py ... --max-tokens 128 --repeats 3`) : 26 tokens d'entrée, 128 générés, SHA256 de texte identique ; decode **100,55 tok/s** médian (99,90–101,16), client **102,24 tok/s** médian, prefill sériel **75,77 tok/s** médian, TTFT **0,343 s** médian. Empreinte physique ~4,784 GB ; chargement 1,015 s. Le modèle MoE reste sur `forward_many_serial` par contrat source, alors que LFM1.2 dense peut grouper dès 24 tokens ; cette comparaison intermodèles ne quantifie pas à elle seule le gain possible d'un batch MoE exact. Preuve `docs/measurements/audit-2026-09-22-current-lfm8-short.json`.
- Ling, prompt moyen `README.md` lignes 1–45, 624 tokens tokenizer, sortie 48, trois passages après warmup du même processus : decode **91,11 / 91,03 / 89,97 tok/s**, prefill sériel **105,38 / 104,73 / 103,97 tok/s**, hash du texte identique. Le débit decode diffère du prompt court ; longueurs de sortie et contenu diffèrent aussi, donc aucun gain causal n'est imputé au contexte. Commande `benchmark_bridge.py models/Ling-3.0-tiny-EXL3-4bpw --native-binary ./target/release/mlxl3-rs --max-tokens 48 --repeats 3 --prompt "$(sed -n '1,45p' README.md)"`. Preuve `docs/measurements/audit-2026-09-22-current-ling-medium.json`.
- LFM1.2 dense avec le même prompt moyen (638 tokens tokenizer), 48 générés, trois passages : prefill batché **2191,49 / 2230,86 / 2240,30 tok/s**, decode **111,36 / 113,35 / 114,23 tok/s**, TTFT **0,292 / 0,287 / 0,285 s**, hash identique ; aucun token caché. Même invocation que Ling en remplaçant le modèle, sortie `docs/measurements/audit-2026-09-22-current-lfm12-medium.json`. Le premier passage peut encore subir une compilation de forme ; ne comparer que les passages chauds pour toute conclusion de seuil.
- LFM8 MoE avec le même prompt moyen (611 tokens tokenizer), 48 générés, trois passages : prefill sériel **70,40 / 72,99 / 63,99 tok/s**, TTFT **8,680 / 8,372 / 9,550 s**, decode **105,09 / 99,54 / 89,40 tok/s**, hash identique ; aucun token caché. L'écart du troisième passage suggère une variation de charge/chauffe, non un effet de code. Même invocation en remplaçant le modèle, sortie `docs/measurements/audit-2026-09-22-current-lfm8-medium.json`. Le coût de prefill sériel à ~600 tokens est mesuré ; gain de batching exact **non mesuré**.
- Qwen prompt long `docs/audit-runtime-2026-09-04.md`, 2825 tokens tokenizer, contexte 8192, budgets 48/128, deux passages alternés après warmup, commande `scripts/smoke-dflash-bridge.py ... --tokens 48,128 --repeats 2 --context-length 8192 --prompt-file docs/audit-runtime-2026-09-04.md` : le script a **échoué au contrôle de parité sur 128 tokens** (sortie 1). À 48, même hash et 48 générés ; decode normal **39,91/39,92 tok/s**, DFlash **37,52/38,99 tok/s**, 34 propositions acceptées. À 128, chaque mode donne 128 tokens mais **deux hashes distincts** normal/DFlash ; normal **39,23/38,33**, DFlash **30,78/31,38 tok/s**, 83 acceptés. Les hashes sont stables entre répétitions du même mode, mais la comparaison intermode est invalide. Preuves `docs/measurements/audit-2026-09-22-current-qwen-long.jsonl` et `.stderr.log`. Statut de cette série **échec de parité, investigation requise** ; ne pas recommander DFlash à cette forme ni rapporter un gain E2E de 128 tokens. Un seul processus bridge était actif, fermé après l'échec.
- Révision du 23 septembre : l'audit a repris après FIX-09 ; synthèse et classement decode/prefill dans [le rapport final](docs/audit-decode-prefill-2026-09-23.md). Les mesures après correction et leurs limites figurent ci-dessous dans FIX-09. Gemma et les temps GPU purs restent **non mesurés**. L'essai du 22 septembre demeure un échec historique, pas une référence de débit comparable au code corrigé. Aucun processus modèle conservé, aucune app installée ni publication.

### OPT-2026-09-22-RUST-AUDIT-08 — diagnostic divergence Qwen/DFlash au contexte 2825 — résolu localement par FIX-09

- Déclencheur : AUDIT-07 ci-dessus ; répétition justifiée par une différence de **prompt et contexte** face aux contrôles antérieurs et par un échec de parité inédit dans cette campagne. Hypothèses à distinguer : divergence des IDs à partir d'un bloc de vérification, divergence de texte seule, ou condition de protocole (cache, ordre, budget) ; aucune correction moteur n'est autorisée dans cet audit.
- Baseline : même checkout/release, Qwen et draft locaux, MLX 0.32.2, prompt exact `docs/audit-runtime-2026-09-04.md`, contexte 8192, greedy temp=0/top_k=1/pénalité=1 ; 128 tokens atteints dans les deux modes et hashes intermode différents de façon stable sur deux répétitions. Protocole : un bridge neuf, séquence normal/DFlash à budget 128 avec `conversation_id` distinct et cache OFF, conserver les événements `complete`, comparer hashes, longueurs et premier caractère divergent du texte ; si l'écart est réel, localiser le premier budget divergent par quelques budgets bornés dans le même processus, sans prétendre que le texte prouve l'identité des logits. Sorties brutes sous `docs/measurements/`; un seul modèle GPU. Statut **en cours**, gain performance **non mesurable tant que la parité échoue**.
- Contrôle indépendant dans un bridge neuf terminé : normal `cea6d766333b2e34`, DFlash `073037bd2bbe85dc`, chacun 128 tokens, `reuse_prompt_cache=false`, identifiants de conversation distincts et modes effectifs confirmés. Le **texte aussi diverge**, premier caractère différent à l'index 208 (normal « It’s a French-language technical report… » ; DFlash « It’s a report about an audit… »). DFlash a accepté 83/215 propositions. Preuve intégrale `docs/measurements/audit-2026-09-22-qwen-parity-diagnostic.json`, stderr bridge et harness vides. Cela écarte une simple différence du hash ou un cache partagé ; cause numérique à localiser entre formes de prefill 256/128, vérification et gestion d'état. Aucun gain DFlash n'est validé sur ce prompt long.
- Budget borné à 64 tokens, même prompt/contexte et outil bridge existant, un passage mesuré après warmup : **échec de parité confirmé** (hashes `315648164fb501f4` et `8fb3398f4387e57a`, DFlash 44/90 propositions acceptées). À 48 tokens, le contrôle AUDIT-07 restait égal ; premier token différent donc situé entre les sorties 49 et 64 pour ce protocole, sous réserve que limiter le budget modifie les derniers blocs de propositions. Preuves `docs/measurements/audit-2026-09-22-qwen-parity-64.jsonl` et `.stderr.log`. Décision **bloqué numériquement** sur le long contexte ; audit seulement, pas de correction moteur dans ce lot. Ne pas extrapoler la parité 48 à des sorties longues.
- Révision du 23 septembre : le blocage a été levé dans le checkout local par FIX-09 ; les constats rouges ci-dessus restent les références avant correction.

### OPT-2026-09-22-RUST-FIX-09 — parité Qwen/DFlash au contexte long — validé dans le code local

- Nouvelle autorisation utilisateur : corriger la divergence reproductible de la génération DFlash, ce qui remplace la restriction « audit sans modification du moteur » pour ce problème précis. AUDIT-07 est interrompu jusqu'à résolution ; les autres pistes restent documentées mais sans campagne supplémentaire.
- Baseline : checkout local `fbfb357edffee4494a1be152666cf1ef33e47c9d` sale, release MLX 0.32.2 / SDK 26.5, Qwen3.6-35B-A3B EXL3 2.49 bpw + draft DFlash2 local, prompt `docs/audit-runtime-2026-09-04.md` (2825 tokens), contexte 8192, greedy température 0/top_k 1/pénalité 1, cache OFF. 48 tokens identiques ; 64 et 128 divergent en hash et texte. Référence brute AUDIT-07/08 ci-dessus. Test régressif à construire sur la vraie génération avec préfixe long et états exacts avant/après vérification ; il doit échouer sur cette base et réussir après correction.
- Hypothèse initiale : les formes de prefill différentes (normal 256, DFlash 128), l'état retenu après rejet de propositions, ou une différence d'arrondi dans la vérification peuvent changer les décisions après ~48 tokens. Localiser par comparaison de logits/états bit à bit sous tokens forcés, puis corriger à la racine sans modifier silencieusement la politique greedy. Vérifier les budgets 48/64/128, plusieurs formes de contexte et les chemins d'acceptation/rejet. Toute mesure de vitesse après correction sera appariée à la même baseline et devra rapporter le coût réel du bridge, mémoire et acceptation ; aucune valeur de gain présumée.
- Qualité : `cargo fmt --check`, build/test release ciblés, Clippy applicable et tentative Kani sur tout contrat CPU nouvellement isolé. Kani ne prouvera pas les graphes MLX/Metal ; contrôle GPU bit-à-bit de logits et états requis. Aucun install/push/release demandé. Statut **en cours** avant changement de code.
- Première compilation du test de reproduction **échouée avant exécution** : le test appelait `Qwen35Moe::eval_state()`, méthode absente ; dans le bridge ce wrapper est un no-op pour Qwen. Source corrigée en retirant cet appel, sans résultat numérique de cet essai. Log `docs/measurements/audit-2026-09-23-prefill-chunk-regression-before.log`.
- Reproduction numérique avant correction **réussie (test rouge)** : `cargo test --release --locked --features mlx,chat long_prompt_prefill_chunk_widths_match_exactly -- --ignored --nocapture --test-threads=1`, prompt réel 2825 tokens, dernier logit FP16 **différent** entre prefill target en chunks 256 (bridge normal) et 128 (bridge DFlash). Échec à l'assertion des logits avant même la comparaison des états, exécution 19,49 s ; preuve `docs/measurements/audit-2026-09-23-prefill-chunk-regression-before-run.log`. Ceci identifie une cause de dérive préexistante au decode spéculatif : les formes de batch changent l'arrondi numérique. Décision de correction : même chunk Qwen 256 dans les deux modes et capture draft conservée, puis vérifier indépendamment la parité après decode. Le test exploratoire 128=256 doit être remplacé par le contrat réel « capture et absence de capture à forme 256 ».
- Modification locale minimale : `dflash::PREFILL_CHUNK` **128 → 256**, utilisé aussi par la frontière de capture alignée et le cache de préfixe ; fenêtre draft 2048 divisible par 256. Contrôle GPU `long_prompt_dflash_capture_keeps_target_exact` **réussi** : le vrai prompt 2825 tokens, logits FP16 finaux et 80 états octet par octet identiques entre target chunk256 avec et sans capture DFlash ; log `docs/measurements/audit-2026-09-23-prefill-capture-after.log`, 15,42 s. Ce test isole le prefill, pas encore la parité des 128 sorties du bridge.
- Bridge après alignement 256, même prompt/contexte, warmup et deux répétitions : **48 tokens égaux**, mais **64 divergent encore** avec les mêmes deux hashes qu'avant (`315648164fb501f4` / `8fb3398f4387e57a`), 44/90 acceptés DFlash. Le script sort 1 au contrôle de parité ; 128 non tenté dans cette invocation après l'échec. Preuves `docs/measurements/audit-2026-09-23-qwen-long-after.jsonl` et `.stderr.log`. L'alignement du prefill résout une différence numérique initiale mais **pas** le bug utilisateur ; seconde cause dans la vérification/commit des états ou un choix de token à localiser. Aucun gain validé, correction en cours.
- Prochain contrôle ciblé, avant exécution : au préfixe réel 2825, puis après 48 tokens greedy, comparer bloc cible M=6 entre boucle `forward` token-major et `verify_tokens_exact_with_dflash_capture`, logits FP16 et tous états ; tester les préfixes retenus 1..6 de `commit_dflash_verification`. La baseline de ce contrôle est le chemin normal au même préfixe et aux mêmes IDs forcés. Les essais historiques de vérification ne couvraient qu'un préfixe de 3 tokens ; cette **nouvelle forme longue** justifie la répétition. Si ce contrôle passe, instrumenter les propositions/acceptations réelles du premier bloc divergent sans réécrire le moteur.
- Contrôle long préfixe **rouge** : préfixe 2825, tests de commit pour retenus 1..6 passent au début ; après 48 tokens greedy forcés, les retenus 1..5 passent aussi, mais les logits FP16 du bloc M=6 diffèrent du token-major. Le test s'arrête avant le contrôle de l'état complet retenu=6. Log `docs/measurements/audit-2026-09-23-long-verify-before.log`, 12,74 s d'exécution. Cela reproduit une seconde dérive dans la vérification cible à contexte long, sans draft ni bridge. Prochaine précision : relever premier rang de logits différent et état final M=6, puis corriger le chemin de vérification plutôt que masquer la divergence dans le sampler.
- Précision obtenue avec le même test : pour le bloc forcé de six tokens après 48 sorties, les préfixes retenus 1..5 ont des états exacts ; au sixième, premier logit différent sur la **ligne 5** (index zéro) et premier état différent à l'index **41**, soit le second état de la couche 20 si deux états/couche. Log `docs/measurements/audit-2026-09-23-long-verify-state.log`. Le problème porte sur le dernier rang du bloc, pas seulement le texte. Étape suivante avant essai : comparer les activations ligne 5 couche par couche entre exécution token-major et vérification layer-major au même snapshot long, pour identifier le premier opérateur divergent ; pas de changement de kernel tant que cette couche n'est pas connue.
- Comparaison des activations forcées ligne 5 : **premier écart à la couche 20, élément 0**, les couches 0..19 étant bit-à-bit identiques sous le même snapshot après 48 tokens ; log `docs/measurements/audit-2026-09-23-long-verify-layer.log`. C'est une couche Gated DeltaNet, cohérente avec le premier état divergent (index 41). Prochain contrôle isolé : comparer ses projections `qkv/z/a/b` multi-lignes à la ligne mono-token, puis la préparation convolution et la mise à jour récurrente. Ce nouveau contrôle vise la forme M=6 après un état long et ne répète pas les essais historiques sur le petit préfixe.
- Les projections de la **ligne 5** `qkv/z/a/b` de la couche 20 sont identiques en batch et mono-token ; la première différence apparaît dans la sortie `q` de `gated_delta::prepare_qkv`, élément 0. Log `docs/measurements/audit-2026-09-23-long-verify-prepare.log`. Il faut encore distinguer un input `qkv` d'une ligne antérieure ou le noyau de convolution préparatoire : prochain contrôle compare les six entrées `qkv` forcées, puis l'état conv final, avant toute modification de calcul.
- Les **six** entrées `qkv` projetées à la couche 20 sont maintenant confirmées bit-à-bit identiques, chacune comparée à sa projection mono-token ; l'écart reste dans `prepare_qkv` M=6 (sortie `q` ligne 5), log `docs/measurements/audit-2026-09-23-long-verify-qkv-all.log`. Avant modification : vérifier que l'historique convolution avant la sixième ligne correspond exactement à la concaténation initiale plus les cinq premières entrées QKV. Si oui, le noyau préparatoire spécialisé T=6 n'a pas le même arrondi que six appels T=1 sur cet état et doit suivre la même séquence arithmétique que le chemin normal.
- Ce dernier contrôle a révélé que l'**historique convolution avant la sixième ligne diffère déjà** de la concaténation attendue malgré six projections QKV exactes ; le test s'arrête avant de comparer la sortie du noyau (`docs/measurements/audit-2026-09-23-long-verify-conv-input.log`). Il faut diagnostiquer la mise à jour de l'état convolution T=1 contre la fenêtre reconstruite T=6, plutôt que modifier aveuglément le noyau de convolution. Le prochain essai imprimera l'indice/valeur du premier écart et la taille de la différence.
- L'écart touche **24 576/24 576 éléments FP16** de l'historique long de trois pas (conv_length=4), premier élément `0x2323` reconstruit contre `0x30d0` stocké après cinq pas ; log `docs/measurements/audit-2026-09-23-long-verify-conv-values.log`. Ce n'est pas un simple arrondi à 1 ULP. Test suivant : appeler `gated_delta::prepare_qkv` directement sur une seule ligne à la couche 20 et confronter son `state_out` à `concat(state_in, qkv).slice(...)` ; cela distinguera le noyau et une éventuelle double projection du modèle.
- **Révision du 23 septembre : les comparaisons de QKV et d'historique des trois lignes précédentes sont invalides comme preuve de cause.** Le test diagnostique projetait l'activation brute avant `input_norm`, alors que `LinearLayer::forward` normalise avant d'appeler `GatedDelta`. Le contrôle direct du noyau T=1, sur cet input diagnostique, a produit un état cohérent avec sa concaténation, mais le contrôle contre `layer.forward` échoue dès le token 0 (`docs/measurements/audit-2026-09-23-long-verify-direct-state.log`). Cela révèle l'erreur du banc, pas une défaillance prouvée du noyau. Le constat solide reste le premier écart de sortie/état **à la couche 20** et aux logits ligne 5. Prochain essai : comparer les mêmes projections et fenêtres **après** la normalisation `input_norm`, en reprenant le test corrigé ; ne pas changer le moteur sur la base des mesures invalides.
- Banc corrigé après `input_norm` : la comparaison du state_out T=1 avec l'historique concaténé passe ; les six entrées `qkv` et `z`, ainsi que la projection dense `a` à la ligne 5, sont exactes. La **première différence fiable** est `in_proj_b` dense à la couche 20, ligne 5, élément 28, entre `matmul` M=6 et `matmul` M=1 (`docs/measurements/audit-2026-09-23-long-verify-normalized.log`). Le checkpoint contient `in_proj_b.weight` dense, sans module EXL3 ; `value_heads=32`, ce qui satisfait les préconditions du `Projection::forward_dense_rows_exact` déjà utilisé pour le routeur MoE. Hypothèse de correction : utiliser ce chemin existant pour `a` et `b` dans la vérification multi-lignes, puis refaire le test des logits/80 états et le bridge 64/128. Ne pas changer `forward` mono-token ni la politique greedy.
- Essai local `forward_dense_rows_exact` sur `a` et `b` pendant la vérification avec historique : le test long **reste rouge au rang 5, état 41** malgré les rangs 0..4 exacts (`docs/measurements/audit-2026-09-23-long-verify-exact-gates.log`). Ce résultat négatif est conservé ; l'écart `b` observé entre matmul M6 et M1 n'est donc pas la seule cause, ou le helper exact n'est pas exact pour cette projection. Test ciblé en cours : comparer directement la sortie du helper exact au M1 à la couche 20, puis la préparation QKV/convolution et la récurrence. Ne pas conclure que ce prototype corrige le bridge.
- Le helper dit exact reste différent du M1 pour `b` au même élément 28 (`docs/measurements/audit-2026-09-23-long-verify-exact-gates-layer.log`) : son contrat historique porte sur le routeur MoE, pas sur toutes les projections denses Qwen. **Nouvel essai avant modification :** dans la vérification DFlash seulement, projeter `a` et `b` ligne par ligne avec `Projection::forward` M1 existant, puis concaténer. Baseline = vérification M6 encore rouge ; protocole = test forcé logits/80 états au contexte 2825 puis 48 tokens, bridge 64/128, coût decode mesuré séparément si parité rétablie. Le changement préserve le calcul canonique M1 mais peut coûter des soumissions supplémentaires ; gain performance non présumé.
- Correction locale des portes denses `a/b` par appels M1 **validée sur le test GPU forcé** : `long_prefix_verification_keeps_exact_state` passe pour le préfixe réel 2825 avant et après 48 tokens greedy ; retenus 1..6, logits FP16 de M6 et **80 états octet par octet** identiques au chemin token-major. Log `docs/measurements/audit-2026-09-23-long-verify-serial-gates.log`, 13,78 s. Ce contrôle n'exerce pas encore les propositions draft réelles ni les événements bridge ; la vitesse n'est pas mesurée. L'alignement de chunk 256 et les projections M1 forment ensemble le candidat de correction.
- **Échec de parité du bridge malgré ce test ciblé** : même prompt/contexte, warmup, deux répétitions, 48 tokens égaux, mais à 64 les hashes restent `315648164fb501f4` / `8fb3398f4387e57a` ; DFlash 46/80 propositions acceptées, script sort 1 et n'atteint pas 128. Log `docs/measurements/audit-2026-09-23-qwen-long-fixed.jsonl`, stderr associé. Les tokens forcés `[1..6]` n'exercent pas les blocs draft réels ; aucune correction complète n'est déclarée. Prochain test préenregistré : rejouer les vrais blocs DFlash en conservant un snapshot de référence target dans **le même modèle résident**, comparer après chaque commit le prochain logit sous le même token sonde et les IDs générés, puis localiser le premier bloc fautif. Aucun deuxième modèle Qwen simultané, mémoire bornée.
- Le nouveau test de blocs a d'abord échoué **au prefill**, avant toute proposition : `prefill_round` normal et DFlash donnent des logits FP16 différents sur ce prompt (`docs/measurements/audit-2026-09-23-dflash-blocks-before.log`, 19,13 s), bien que les blocs complets utilisent maintenant 256. Cause source : 2825 mod 256 = **9 tokens de queue** ; `NativeChatModel::forward_many` choisit `Qwen::prefill_serial` sous 24, alors que `DFlashChat::prefill` appelait `forward_tokens_with_dflash_capture` batch9. Le test précédent « capture vs sans capture » comparait deux batchs de même forme, pas cette bifurcation bridge. **Nouvelle hypothèse/essai :** capturer chaque token de la queue via le chemin Qwen M1 sans projeter inutilement le vocabulaire avant le dernier, concaténer les features draft, et garder le head final identique à `prefill_serial`. Référence = normal sur 2825, contrôles = logits/80 états exacts au prefill, vraie suite de blocs et bridge 64/128 ; gain performance non présumé.
- Chemin `prefill_serial_with_dflash_capture` ajouté pour la queue `<24`, utilisé seulement par DFlash ; les blocs complets restent en 256. Test GPU `long_prompt_dflash_capture_keeps_target_exact` étendu au **prompt réel 2825 = 11×256+9** : logits FP16 finaux, 80 états octet par octet et forme de capture 9×16384 comparés au prefill normal sérialisé, **réussis** (`docs/measurements/audit-2026-09-23-serial-tail-exact.log`, 33,38 s). Le test vérifie aussi tous les chunks batch256 avec/sans capture. Étape restante : vrais blocs draft, parité des IDs 64/128 et coût decode.
- Test GPU des **vrais blocs** `long_prompt_dflash_blocks_match_target` **réussi** (`docs/measurements/audit-2026-09-23-dflash-blocks-tail-fixed.log`, 33,55 s) : le même modèle résident alterne snapshots de référence et spéculatifs ; prefill FP16 exact, 128 IDs exacts, après chaque commit le prochain vecteur de logits FP16 sous un token sonde est égal à la référence ayant consommé les mêmes tokens. Les propositions/acceptations réelles sont utilisées. Ce test établit la parité sur ce prompt et ce budget, sans être une preuve universelle de tous les prompts ni de tous les 80 états à chaque bloc ; le contrôle forcé de 80 états couvre le cas M=6 avant et après 48 tokens. Bridge réel et mesures encore à faire.
- **Bridge réel long validé** : `scripts/smoke-dflash-bridge.py ... --tokens 48,64,128 --repeats 2 --context-length 8192 --prompt-file docs/audit-runtime-2026-09-04.md` sort 0, mode effectif confirmé, budgets atteints, mêmes hashes et textes normal/DFlash à **48/64/128**, notamment `cea6d766333b2e34` à 128 (référence normale initiale). Raw `docs/measurements/audit-2026-09-23-qwen-long-final.jsonl`, stderr vide. Médianes decode tok/s normal→DFlash : 48 **36,64→34,14** (ratio 0,932), 64 **36,58→32,61** (0,891), 128 **36,54→27,80** (0,761, 83/216 acceptés). Prefill DFlash ~7,7–8,0 s vs normal ~7,2–7,4 s ; les conditions machine ont changé depuis la baseline rouge et les anciens ratios n'étaient pas comparables numériquement, donc aucun pourcentage causal de coût du correctif n'est revendiqué. Sur ce long prompt, DFlash est désormais correct mais plus lent que normal ; documenter cette limite, ne pas vendre un gain. Vérifier encore prompt court, cache de préfixe, frontières et verif source.
- **Bridge court régressif validé** : prompt par défaut 20 tokens, contexte 4096, budgets 48/128/256, deux passages alternés après warmup, mêmes hashes/texte normal et DFlash, tous budgets atteints (`docs/measurements/audit-2026-09-23-qwen-short-final.jsonl`, stderr vide). Decode médian normal→DFlash : 48 **39,58→60,58 tok/s** (×1,531, 38/45 acceptés), 128 **39,90→41,89** (×1,050, 91/171), 256 **39,94→38,04** (×0,952, 178/382). L'environnement est plus lent que les mesures du 22 septembre des deux côtés ; ces ratios ne quantifient pas seuls le coût du correctif. DFlash conserve son intérêt sur la sortie courte et régresse sur la longue dans ces conditions.
- Cache de préfixe bridge régressif (`--check-prefix-cache`, même binaire) **réussi** : normal et DFlash ont chacun deux essais OFF/ON à 575 tokens, 512 tokens cachés ON, même hash `1add3c529217b3a7` entre quatre sorties et modes ; TTFT OFF→ON normal 1,508→0,308/0,311 s, DFlash 1,622/1,658→0,323/0,325 s. Annulations aux événements `context_usage` et `delta`, reprises exactes, changements de système/conversation invalidés par le script. Preuve `docs/measurements/audit-2026-09-23-prefix-final.jsonl`, stderr vide. Ces temps sont des observations de contrôle, pas un nouvel essai de gain du cache.
- Kani 0.68.0 sur la constante de chunk modifiée : `cargo kani --no-default-features --lib --harness draft_capture_keeps_full_window_at_both_boundaries` **34 obligations, zéro échec, trois couvertures satisfaites**, toutes longueurs `usize` symboliques y compris MAX ; vérifie que la fenêtre de 2048 et la frontière du checkpoint restent alignées avec le chunk 256. Log `docs/measurements/audit-2026-09-23-kani-capture.log`. **Vérification bornée/source CPU uniquement** : ne prouve pas les calculs GPU, le bridge ni la parité de génération ; ceux-ci ont leurs contrôles dynamiques ci-dessus.
- Frontières de queue testées après formatage : avec préfixe Qwen 256, queues **1, 9 et 23 tokens** capturées par le nouveau chemin M1, logits FP16 finaux et 80 états octet par octet égaux au `prefill_serial` normal ; forme des captures exacte. Le même test conserve le prompt réel 2825 et l'égalité des chunks batch256. `long_prompt_dflash_capture_keeps_target_exact` **réussi**, 36,71 s, log `docs/measurements/audit-2026-09-23-serial-tail-boundaries.log`. Le seuil ≥24 reste testé par les contrôles batch 24/25 existants ; besoin de relancer les tests généraux et Clippy.
- Vérifications générales intermédiaires : `cargo fmt --all -- --check` **réussi** ; `cargo test --locked --features mlx,chat` **43 tests non ignorés réussis**, 41 GPU/modèles ignorés, log `docs/measurements/audit-2026-09-23-cargo-test.log`. Premier Clippy strict **échoué sur le test ajouté uniquement** (`length as i32` était déjà `i32`), aucun résultat de lint pour le programme complet ; log `docs/measurements/audit-2026-09-23-clippy.log`. Corriger le cast puis relancer avant décision.
- Deuxième passe Clippy **échouée sur la correction du mauvais cas de test** : la nouvelle boucle de queues 1/23 utilise `usize` pour indexer les tokens et nécessite un cast vers `i32` pour la forme du tenseur ; l'ancienne boucle 23/24/127 utilise déjà `i32` et ne le nécessite pas. Log `docs/measurements/audit-2026-09-23-clippy-final.log`. Les deux assertions ont été corrigées selon leurs types ; lint complet à relancer.
- Passe Clippy finale **réussie** : `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`, log `docs/measurements/audit-2026-09-23-clippy-pass.log`. Les deux échecs précédents de typage dans les assertions de test sont conservés ci-dessus ; aucune alerte restante. Reste la revue du diff et le rapport d'audit interrompu.
- Décision finale du 23 septembre : **validé localement pour le checkpoint Qwen, les queues 1/9/23, les préfixes retenus 1..6 et les bridges 48/64/128** ; pas de preuve universelle MLX/Metal. Les trois corrections numériques sont l'alignement des chunks target à 256, le prefill de queue capturé en M1 et les projections `a/b` en M1 pendant la vérification DFlash. Le long prompt 128 est redevenu bit-à-bit identique en logits/états dans les contrôles ciblés et identique en IDs/texte sur le bridge ; DFlash y reste plus lent (12,526 contre 10,728 s de génération complète). Rapport : [audit decode/prefill du 23 septembre](docs/audit-decode-prefill-2026-09-23.md). Intégration : **code local uniquement**, binaire release reconstruit, aucune app installée, aucun push ni publication.
- Recontrôle final du 23 septembre : première invocation `cargo test --locked --features mlx,chat` et `cargo fmt --all -- --check` interrompue avant exécution (`cargo` absent du `PATH` de ce shell). Avec `PATH=/Users/justin/.cargo/bin:$PATH`, le formatage passe, mais le test s'arrête avant compilation C++ car `MLXL3_MLX_ROOT` n'était pas défini. Ces deux échecs sont des erreurs d'environnement, pas des résultats numériques ; relance avec le répertoire MLX installé dans `.venv` enregistrée ci-après.
- Relance finale `PATH=/Users/justin/.cargo/bin:$PATH MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx cargo test --locked --features mlx,chat` **réussie** : 26 tests lib + 3 tests binaire + 14 tests contrats = **43 non ignorés réussis**, 41 tests GPU/modèles explicitement ignorés. `cargo fmt --all -- --check`, `git diff --check` et tous les liens locaux du rapport passent. Les tests GPU nommés et bridges réels ont leurs logs dédiés ci-dessus ; aucun processus modèle ne tourne à la clôture.

### OPT-2026-09-23-RUST-PERF-117 — objectif DFlash +50 % sur le cas long — diagnostic terminé, objectif non atteint

- Demande utilisateur : rendre DFlash 50 % plus rapide après le correctif de parité. Référence principale, tant que la métrique n'est pas précisée autrement : durée **complète** du bridge Qwen sur le prompt réel 2 825 tokens, 128 générés, contexte 8 192, greedy, cache de préfixe OFF. FIX-09 mesurait en deux répétitions alternées normal **10,728 s** contre DFlash **12,526 s**, decode **36,54 contre 27,80 tok/s**, 83/216 propositions acceptées, 44 blocs, mêmes IDs/texte. Objectif 1,5× contre normal : au plus **7,152 s** DFlash au total. Les chiffres de PERF-112/113 sur un autre prompt et l'ancien code ne sont pas une baseline interchangeable ; largeur N≠5 déjà rejetée sur 128 tokens. Aucun gain candidat présumé.
- Contrainte calculée avant essai : prefill DFlash actuel ~7,74–7,81 s et normal ~7,22–7,28 s ; même un decode instantané ne donnerait pas 1,5× sur ce prompt froid. Il faut donc améliorer **prefill et decode** sous un contrôle bit-à-bit, ou constater que la cible n'est pas atteinte. Le simple fallback normal plafonne au niveau de la référence et ne satisfait pas +50 %. Hypothèse de diagnostic : quantifier sur le vrai chemin les temps draft, vérification cible, commit/cache et prefill avant de choisir un changement ; rechercher des travaux redondants, sans refaire les kernels rejetés.
- Protocole prévu avant tout code/benchmark : conserver FIX-09 comme référence historique puis lancer une baseline adjacente sur le bridge actuel, deux passages alternés normal/DFlash après warmup, même fichier, contexte, budget et checkpoint ; sortir JSON brut, parité, acceptation, TTFT, temps complet, mémoire physique/MLX. Un profil ciblé dans un test ignoré, **un seul modèle GPU résident**, décomposera les blocs longs ; les barrières introduites seront signalées comme perturbantes et ne seront pas additionnées au temps bridge. Ensuite seulement, un candidat isolé sera journalisé avec son protocole et comparé A/B après logits FP16 et 80 états exacts aux cas applicables, puis bridge court/long. M5 24 Gio, MLX 0.32.2, SDK 26.5, batterie **61 % en décharge** au départ ; aucun processus modèle. État : **en cours**, code local, app non installée/non publiée.
- Baseline adjacente exécutée avec `scripts/smoke-dflash-bridge.py ./target/release/mlxl3-rs ... --tokens 128 --repeats 2 --context-length 8192 --prompt-file docs/audit-runtime-2026-09-04.md`, warmup puis alternance, sortie `docs/measurements/perf-117-baseline-long.jsonl`, stderr vide. Normal : **9,595/9,616 s** complet, **6,442/6,446 s** prefill, **40,32/40,09 tok/s** decode. DFlash : **11,329/10,966 s** complet, **6,991/6,923 s** prefill, **30,84/31,43 tok/s** decode, 83/216 acceptés en 44 blocs dans chaque passe. Hash intermode `cea6d766333b2e34`, budget 128 atteint, empreinte physique du processus ~17,32 GB. Les temps diffèrent de FIX-09 avec la chauffe/charge du jour mais le classement reste le même. Sur cette paire, 1,5× complet face à normal impose ≤**6,404 s**, sous le prefill DFlash actuel ≥**6,923 s** ; aucun changement limité au decode ne suffit. Profil des blocs encore à faire, aucun gain candidat revendiqué.
- Profil test ciblé `long_prompt_dflash_blocks_match_target` en release, même prompt et vrais blocs, **réussi** avec 128 IDs et comparaisons de logits après commit ; `docs/measurements/perf-117-long-stage-profile.log`. Les 44 blocs, 83/216 acceptés prennent **4,069 s** dans le test, dont appels draft/sélection **0,846 s**, vérification target/argmax **3,218 s**, construction commit/cache **0,004 s** ; la somme n'est pas un temps GPU pur et le travail différé du cache peut être payé au bloc suivant. L'instrumentation n'ajoute pas de barrière et restera cantonnée au test avant retrait. Le coût dominant actuel est la vérification target, mais même sa suppression intégrale laisserait le prefill au-dessus du seuil complet visé. Décision diagnostic : chercher d'abord si les cinq propositions sont trop larges pour les **38 %** acceptés sur ce nouveau long prompt, puis évaluer séparément une piste prefill. Aucun gain modèle déclaré.

### OPT-2026-09-23-RUST-PERF-118 — largeur DFlash au préfixe 2 825 — validé en diagnostic, N=2 fixe non intégré

- Différence précise avec PERF-113 rejeté : son prompt court avait **101/140 = 72 %** de propositions acceptées à 128 tokens, alors que le vrai prompt long actuel n'en accepte que **83/216 = 38 %** et vérifie 44 blocs à ~3,218 s de target. Hypothèse : N=2/3/4 peut réduire le travail de vérification gaspillé, même avec plus de blocs. Cela ne peut pas à lui seul satisfaire l'objectif PERF-117 sur le temps complet.
- Baseline : PERF-117 N=5, même checkpoint/MLX 0.32.2, 2 825/128, contexte 8 192, cache OFF, 83/216, 44 blocs, decode DFlash **4,040–4,118 s** dans le bridge, résultat bit à bit exact contre normal aux contrôles FIX-09. Protocole : balayer N=2,3,4,5 dans un seul test GPU résident, avec vrai prefill et mêmes 128 IDs imposés/greedy, état cible restauré entre variantes ; warmup, ordre contrebalancé, relever acceptation/blocs et temps draft/target/total sans double modèle. Un N prometteur seulement passera dans le vrai bridge alterné, avec logits/états exacts et comparaison complète. Rejeter si le gain se noie dans les variations ou si la parité échoue ; retirer le prototype test-only et ne pas relancer les formes déjà rejetées hors cette nouvelle distribution. Statut **en cours**, aucun code de production changé pour cet essai.
- Balayage test-only **réussi** et tous les 128 IDs égaux au normal : `docs/measurements/perf-118-widths-long.log`, un seul modèle résident, prefill DFlash initial conservé puis snapshots restaurés, warmup de chaque forme et ordre **5/2/3/4/4/3/2/5**. N=5 : **4,059/4,056 s**, 44 blocs, 83/216 ; N=4 : **3,625/3,674 s**, 44 blocs, 83/171 ; N=3 : **3,483/3,468 s**, 49 blocs, 78/145 ; N=2 : **3,141/3,138 s**, 54 blocs, 73/107. Le temps target N=2 baisse **3,229→2,359 s** dans ces appels synchronisés ; draft reste ~0,78–0,83 s. C'est un gain de **decode du test**, pas encore du bridge ni du temps complet. Il confirme que PERF-113 ne se transpose pas à cette nouvelle distribution d'acceptation ; N=2 est candidat. Le prefill inchangé garde l'objectif +50 % complet hors d'atteinte par cette seule piste.

### OPT-2026-09-23-RUST-PERF-119 — politique de largeur DFlash selon acceptation — validé localement, objectif +50 % non atteint

- Source : PERF-118, long prompt 38 % d'acceptation et N=2 ~0,92 s plus rapide que N=5 en decode testé, alors que le prompt court 48 tokens accepte 38/45 propositions et profitait déjà de N=5. Hypothèse : conserver N=5 tant que l'acceptation est haute, puis passer à N=2 après plusieurs blocs faibles réduit la régression longue sans sacrifier le cas court. Cette logique change le **travail spéculatif**, pas les logits du target ni ses commits ; la sortie doit rester strictement identique.
- Baselines : PERF-117 bridge complet N=5 normal **9,595/9,616 s**, DFlash **11,329/10,966 s** à 2 825/128 ; FIX-09 court 20/48 DFlash **60,58 tok/s** contre normal **39,58**, 38/45 acceptés. Protocole avant code production : tester le seuil sur les traces de blocs réels dans un test local, comparer 128 IDs à la référence normale et mesurer decode N=5/N=2/adaptatif en ordre contrebalancé. Puis intégrer la règle minimale seulement si le bridge complet long/128 et court/48,128 reste bit-à-bit exact et montre un gain reproductible ; contrôler les 80 états sur le test ciblé existant. Kani sur la décision CPU si changement retenu, cargo build/test/Clippy. Rejet si le coût ou l'instabilité annule le gain. Aucun gain complet +50 % présumé, état **en cours**.
- Prototype test-only, `docs/measurements/perf-119-adaptive-long.log` : après huit blocs, limiter à N=2 si l'acceptation cumulée est inférieure à 60 %, sinon N=5. Vrais blocs sur 2 825/128, tous les 128 IDs égaux au normal ; ordre 5/adapt/2/2/adapt/5 après warmup. N=5 **3,994/3,989 s**, 44 blocs, 83/216 ; adaptatif **3,301/3,366 s**, 51 blocs, 76/131 ; N=2 **3,144/3,139 s**, 54 blocs, 73/107. La règle réduit le decode du test de ~0,65 s avec ce contexte sans forcer N=2 sur le court à forte acceptation. L'amélioration reste **test-only** et la cible +50 % complète hors d'atteinte avec ce seul changement ; prochain contrôle = vrai bridge apparié après intégration minimale, puis décision garder/rejeter. Le prototype de largeur instrumenté sera retiré avant clôture.
- Référence binaire sauvegardée **avant modification de production** dans `/tmp/mlxl3-perf117-baseline`, SHA-256 `41489e985d717f82d2fc76702ede49632ce894ae74a1ecb858bb0b3583ad0456`. La campagne candidate comparera ce binaire et le nouveau release sur le même prompt/poids avec ordre alterné ; les différences de processus/thermique resteront une limite. Aucun autre processus modèle actif.
- Premier bridge candidat B1 après build release, `docs/measurements/perf-119-candidate-long-b1.jsonl` : parité hash/texte **réussie** à 128 (`cea6d766333b2e34`), normal **9,769/10,045 s** complet et **3,199/3,268 s** decode, DFlash adaptatif **10,813/10,456 s** complet et **3,499/3,373 s** decode, 76/131 acceptés en 51 blocs. Prefill DFlash **7,108/7,080 s** ; empreinte et conditions à consolider après A/B. Le candidat semble moins lent que N=5 historique mais reste **plus lent que normal** dans les deux paires et très loin du seuil complet. Ne pas encore attribuer le gain à la règle sans l'ancien binaire adjacent ; ordre B/A/B en cours.
- Ancien binaire A adjacent, `docs/measurements/perf-119-baseline-long-a.jsonl` : même hash, normal **9,700/10,150 s**, DFlash N=5 **11,706/11,325 s** complet et **4,232/4,148 s** decode, 83/216 acceptés. Prefill DFlash **7,234/7,174 s**. Le decode du candidat B1 (**3,499/3,373 s**) est ~0,65–0,86 s plus court, cohérent avec le test de blocs ; la variation de prefill ~0,1 s entre processus interdit d'attribuer la différence totale entièrement au code. Dernier B2 puis décision.
- Dernier B2, `docs/measurements/perf-119-candidate-long-b2.jsonl` : hash/texte exacts, normal **9,855/10,015 s**, DFlash adaptatif **11,026/10,706 s** complet et **3,598/3,427 s** decode, 76/131 acceptés. Les médianes A contre B1/B2 ont un decode DFlash **4,190 s → 3,436/3,513 s** (environ 16–18 % de moins sur cette métrique), mais une durée complète **11,516 s → 10,635/10,866 s** encore **plus lente que normal** (~9,93 s dans B2). Ce B/A/B confirme un gain decode local, pas +50 % complet. Le coût de prefill domine ; vérifier maintenant le prompt court, Kani et parité d'états avant de garder le candidat.
- Bridge court B, `docs/measurements/perf-119-candidate-short.jsonl`, prompt par défaut 20 tokens/contexte 4096, warmup et deux répétitions par budget : **48 tokens** hash/texte égaux, normal **1,480/1,472 s**, DFlash **1,163/1,108 s**, 38/45 acceptés ; **128 tokens** hash/texte égaux, normal **3,333/3,335 s**, DFlash **3,173/3,106 s**, 88/156 acceptés. Le cas court/48 conserve N=5 et son avantage ; 128 peut passer en N=2 après baisse d'acceptation. Comparaison avec l'ancien binaire A sur les mêmes budgets encore à exécuter, aucune conclusion causale sur ce couple seul.
- Ancien binaire court A, `docs/measurements/perf-119-baseline-short.jsonl` : mêmes hashes/texte. À 48, normal **1,463/1,466 s**, DFlash N=5 **1,149/1,102 s**, 38/45 ; différence avec B sous le bruit, donc pas de régression court/48 mesurable. À 128, normal **3,321/3,353 s**, DFlash N=5 **3,211/3,158 s**, 91/171 ; candidat B **3,173/3,106 s**, amélioration ~0,04–0,05 s seulement et non attribuable avec certitude. La règle adaptative ne prétend pas +50 % sur 128 ; son signal utile est le cas long à faible acceptation.
- Instrumentation temporelle et balayage de largeur test-only retirés après conservation des logs ; la production garde seulement `speculative::adaptive_proposals` et son appel bridge. Kani 0.68.0 : `cargo kani --no-default-features --lib --harness adaptive_work_never_exceeds_the_existing_budget`, **34 obligations, zéro échec, 2/2 couvertures satisfaites** (`docs/measurements/perf-119-kani-adaptive.log`). Propriété source CPU pour tous les compteurs `usize` : la règle ne dépasse jamais le budget/contexte du contrat précédent, limite à deux après le seuil faible, conserve l'ancien maximum sinon. Ne prouve ni temps GPU, ni logits/états ; les contrôles modèle suivent.
- Contrôles après retrait des prototypes : `cargo fmt --all -- --check`, `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`, `cargo test --locked --features mlx,chat` **réussis** (nouveau test CPU de seuil compris ; log `docs/measurements/perf-119-clippy.log` et `perf-119-cargo-test.log`). Test GPU release `long_prompt_dflash_blocks_match_target` **réussi**, 31,25 s, vrai prompt 2 825/128, mêmes IDs et prochain vecteur de logits FP16 après chaque commit contre une référence séquentielle ; `docs/measurements/perf-119-long-blocks-exact.log`. La comparaison forcée de tous les états à M=6 reste à rejouer et aucun gain universel n'est inféré.
- Test GPU `long_prefix_verification_keeps_exact_state` **réussi** après intégration : au vrai préfixe 2 825 avant/après 48 tokens, retenus 1..6 du bloc M=6, logits FP16 et **80 états octet par octet** identiques à l'exécution cible séquentielle ; 13,72 s, `docs/measurements/perf-119-long-states-exact.log`. La règle de largeur ne modifie pas le calcul du vérificateur ; ce contrôle confirme qu'il reste exact dans cette forme. Les hashes/texte du bridge et vrais blocs complètent la preuve dynamique, sans la rendre universelle.
- Décision finale : **garder la règle adaptative dans le code local** car le vrai decode long baisse de ~16–18 % en B/A/B, la sortie reste exacte, et le court/48 ne régresse pas de manière mesurable. Le gain complet est seulement ~0,65–0,88 s selon les paires et le DFlash candidat reste plus lent que le mode normal sur 2 825/128 ; **ne pas annoncer +50 %**. Ce seuil sur une requête froide demanderait simultanément un prefill beaucoup plus court et un decode spéculatif réellement plus rapide que le normal, alors que l'acceptation de ce draft n'est que 38 % en N=5 ; aucune variante exacte mesurée ici ne l'atteint. Le rapport [audit révisé](docs/audit-decode-prefill-2026-09-23.md) expose les chiffres, les limites thermiques et les preuves. Dernier `cargo build --release --locked --features mlx,chat` **réussi** (`docs/measurements/perf-119-final-build.log`). Intégration **code local seulement**, aucune app installée, aucun push/release ; aucun essai ou processus GPU laissé en cours.

### OPT-2026-09-23-RUST-PERF-120 — objectif 70 tok/s en conversations courantes — diagnostic terminé

- Demande utilisateur : 70 tok/s **decode dans le vrai bridge**, priorité aux conversations courantes plutôt qu'au document de 2 825 tokens. Hypothèse : le rendement du draft et le temps par bloc diffèrent sensiblement selon langue et longueur ; une mesure française courte avec sorties 48/128 peut distinguer un coût de calcul d'une baisse d'acceptation. Aucun gain présumé, aucun changement de production pour cette référence. Différence avec PERF-112/113 : vrai bridge et prompt de conversation français ; avec PERF-119 : distribution de prompt nouvelle.
- Référence connue PERF-119 : prompt anglais 20/48, DFlash 66,07 tok/s et 38/45 acceptés ; 20/128, 46,7–46,9 tok/s et 88/156. Cas long 2 825/128, 3,373–3,598 s de decode DFlash adaptatif, 76/131 acceptés. Les anciennes conditions matérielles diffèrent ; ne pas les utiliser comme A/B causal.
- Protocole avant mesure : utiliser `scripts/smoke-dflash-bridge.py` sur le release local actuel, Qwen3.6-35B-A3B EXL3 2,49 bpw + draft DFlash2, contexte 4 096, cache OFF, température 0/top_k 1/pénalité 1 ; prompt français conservé sous `docs/measurements/perf-120-prompt-fr.txt`, sorties 48 et 128, deux warmups et deux répétitions alternées normal/DFlash dans **un seul processus**. Relever tok/s decode, durée complète/TTFT, acceptés/proposés, blocs, mémoire et parité hash/texte ; sortie brute `docs/measurements/perf-120-conversation-fr.jsonl`, stderr adjacent. Si la distribution l'exige, choisir ensuite un seul candidat sur le coût dominant, journalisé avant modification/essai. Contrôles numériques logits FP16 et 80 états, plus vérification source formelle et build, obligatoires pour un changement de calcul. M5 24 Gio, MLX 0.32.2, SDK 26.5, batterie 43 % en décharge, aucun processus modèle au départ. État **en cours**, code local, app non installée/non publiée.
- Première mesure terminée sans erreur (stderr vide), batterie 43→41 % : prompt français **56 tokens**. À 48 tokens générés, normal **44,45/45,01 tok/s**, DFlash **65,83/65,20 tok/s** (decode 0,714/0,721 s), 37/45 acceptés en 9 blocs ; durée complète DFlash **1,018/0,980 s**, TTFT 0,304/0,259 s. À 128, normal **44,88/44,24 tok/s**, DFlash **50,72/50,69 tok/s** (2,504/2,505 s), 94/159 acceptés en 33 blocs ; durée complète DFlash **2,808/2,764 s**, TTFT 0,304/0,259 s. Hash et texte strictement égaux dans chaque paire, budget atteint ; empreinte physique ~15,13–15,37 GB, mémoire MLX active et cache dans le JSONL. Le 70 tok/s réel n'est donc pas atteint sur ce prompt, surtout à 128. Les 48 premiers tokens acceptent 37/45 propositions, les 80 suivants seulement 57/114 par différence des totaux ; cette soustraction ne révèle pas le détail de chaque bloc. La règle cumulative de PERF-119 peut garder N=5 longtemps après ce changement de distribution : hypothèse à discriminer, pas encore un gain candidat. Ancien prompt anglais PERF-119 et celui-ci ne forment pas un A/B causal.
- Décision PERF-120 : **diagnostic terminé**. La piste à tester est la mémoire d'acceptation de la politique, non un changement de logits/kernel. Le coût cible à 48 est ≤0,671 s pour 47 tokens de decode ; à 128 ≤1,814 s pour 127 tokens. Le candidat suivant doit être évalué sur les deux budgets, sans annoncer 70 à partir d'un seul court.

### OPT-2026-09-23-RUST-PERF-121 — acceptation récente pour choisir N=5 ou N=2 — validé localement, 70 tok/s non atteint

- Hypothèse : sur les conversations où le draft accepte bien au début puis mal ensuite (PERF-120 : 37/45 puis 57/114), une fenêtre de huit blocs déclenche N=2 avant la règle cumulative et réduit le temps de decode à 128 sans ralentir 48. Différence avec PERF-119 : distribution nouvelle, fenêtre récente au lieu du cumul ; le travail de target reste identique à largeur fixée. Aucun gain de 70 tok/s présumé.
- Baseline : binaire release local PERF-119 actuel à archiver avant modification ; PERF-120 prompt français 56/48 **65,20–65,83 tok/s**, 37/45, et 56/128 **50,69–50,72 tok/s**, 94/159, textes/hashes exacts contre normal. Protocole : remplacer uniquement les compteurs utilisés pour la décision par ceux des huit derniers blocs, conserver les compteurs totaux pour les statistiques ; test CPU qui échoue sur l'ancienne politique pour la bascule après bonne puis mauvaise acceptation ; Kani sur le contrat de largeur et le budget. B/A/B **un processus modèle à la fois**, mêmes 48/128 du prompt français, deux répétitions alternées par budget et par binaire après warmups, puis long 2 825/128 si signal prometteur. Contrôler hashes/texte, logits FP16/80 états sur le test ciblé existant, build release, Clippy, tests. Rejeter/restaurer si le gain est absent, fragile ou si une parité échoue. M5 24 Gio, MLX 0.32.2, SDK 26.5, batterie 41 % en décharge avant candidat, code local seulement ; app non installée/non publiée.
- Binaire A sauvegardé `/tmp/mlxl3-perf120-baseline`, SHA-256 `a5cc1e9600e9df7cbe9f278714894f1c3784ff891f0a0f02be1f593efb7d1f99`. Régression CPU créée avant correction : 8 blocs initiaux 40/40 puis 8 récents 8/40 donnent cumul 48/80 = 60 %, ancienne règle renvoie **N=5** mais le contrôle attend N=2 ; `cargo test --locked --features mlx,chat dflash_width_tracks_recent_acceptance` échoue comme attendu (`docs/measurements/perf-121-red-test.log`). La candidate garde les cumuls publics et transmet maintenant les huit blocs récents à la même fonction de budget. Validation/mesures à suivre ; aucun gain encore attribué.
- Le même test CPU passe après correction (`perf-121-green-test.log`) et le build release passe (`perf-121-build.log`). Premier bridge B1, `perf-121-candidate-b1.jsonl`/stderr vide, un seul processus : 48 tokens **65,39/65,70 tok/s**, 37/45 en 9 blocs, hash/texte identiques à A ; 128 tokens **52,54/52,36 tok/s**, 91/141 en 36 blocs, mêmes hash/texte. L'amélioration brute vs PERF-120 est ~1,7–1,8 tok/s à 128 et aucune à 48, mais A adjacent et B2 sont nécessaires pour exclure dérive machine. L'objectif 70 n'est pas atteint ; aucun gain définitif attribué.
- Ancien binaire A adjacent `perf-121-baseline-a.jsonl`, stderr vide : 48 **64,84/65,79 tok/s**, 37/45 ; 128 **44,01/50,85 tok/s**, 94/159. Le premier 128 A est un passage lent isolé alors que le normal reste ~44,7 tok/s ; sa cause est non mesurée et il ne faut pas l'utiliser seul pour attribuer un gain. Les hashes restent exacts. Dernier B2 puis décision sur les passes stables et le contrôle long.
- Dernier B2 `perf-121-candidate-b2.jsonl`, stderr vide : 48 **65,77/65,85 tok/s**, 37/45 ; 128 **51,75/52,21 tok/s**, 91/141 en 36 blocs, hashes/textes identiques. Face aux passages A stables ~50,7–50,9, le candidat donne ~52,0 tok/s (environ +2–3 %) à 128 ; le premier A lent est exclu du raisonnement. À 48, variation sous le bruit ; 70 toujours non atteint. Poursuivre le contrôle long, car une fenêtre récente peut osciller sur un prompt difficile alors que le cumul PERF-119 restait N=2.
- Contrôle candidat long `perf-121-candidate-long.jsonl`, stderr vide, même 2 825/128 que PERF-119 : hash exact `cea6d766333b2e34`, DFlash **3,382/3,325 s** de decode (**37,55/38,19 tok/s**), 80/147 acceptés en 47 blocs ; normal adjacent **3,137/3,176 s**, génération totale DFlash **10,568/10,193 s** contre normal **9,576/9,647 s**. La fenêtre récente permet davantage de propositions acceptées/blocs moins nombreux que le cumul PERF-119 (76/131, 51 blocs), mais les conditions machine diffèrent. A long adjacent à mesurer avant attribution ; le seuil 70 et +50 % total restent hors de portée ici.
- Ancien binaire A long adjacent `perf-121-baseline-long-a.jsonl`, stderr vide : même hash, DFlash **3,426/3,433 s** de decode (**37,07/37,00 tok/s**), 76/131 en 51 blocs ; normal **3,217/3,194 s**. B candidat est ~0,04–0,11 s plus court en decode et ne régresse pas le long dans ces passes, malgré la différence de forme. Le temps complet reste plus lent que normal. Les processus sont séquentiels et les dérives prefill/machine empêchent une précision fine au-delà de ce signal ; tests numériques/source encore à faire.
- Test GPU release `long_prompt_dflash_blocks_match_target` **réussi** (29,85 s d'exécution) sur le vrai long prompt : 128 IDs et prochain vecteur de logits FP16 après chaque bloc/commit identiques au target séquentiel ; `perf-121-long-blocks-exact.log`. Le contrôle forcé des 80 états et Kani restent en cours.
- Contrôle forcé `long_prefix_verification_keeps_exact_state` **réussi** (13,77 s) : au préfixe 2 825, retenus 1..6 pour M=6, logits FP16 et **80 tenseurs d'état octet par octet** identiques au chemin target séquentiel ; `perf-121-long-states-exact.log`. Ce test du vérificateur ne prouve pas la qualité du draft ni toute largeur/prompt possible ; les hashes bridge et le test de blocs couvrent les formes mesurées.
- Kani 0.68.0 sur `adaptive_work_never_exceeds_the_existing_budget` : **34 obligations, zéro échec, 2/2 couvertures**, toutes valeurs `usize` des cinq compteurs du harnais, `cargo kani --no-default-features --lib --harness adaptive_work_never_exceeds_the_existing_budget` (`perf-121-kani-budget.log`). Il prouve la borne de travail de la fonction CPU pour ses entrées, mais **pas** la mise à jour de la fenêtre `VecDeque`, le GPU ou la performance. `cargo test --locked --features mlx,chat` : **45 tests non ignorés réussis**, dont le nouveau contrôle de politique (`perf-121-cargo-test.log`) ; Clippy strict tous targets/features demandés, format Rust, diff check et build release réussis (`perf-121-clippy.log`, `perf-121-build.log`). Vérifications modèle ciblées ci-dessus ; autres formes non garanties universellement.
- Décision : **garder la fenêtre récente dans le code local**. Gain reproductible mais faible sur 56/128 (environ +2–3 % de tok/s de decode sur les passes stables), aucun gain mesurable à 56/48, long non régressé dans l'A/B adjacent. Le but **70 tok/s en usage courant n'est pas atteint** : 65–66 à 48 et ~52 à 128 sur le prompt français ; le +50 % sur durée totale demeure non atteint sur le long. Ne pas présenter ces chiffres comme un gain universel ou attribuer le temps total aux seules largeurs. Aucun essai/processus GPU en cours ; batterie 35 % en décharge à la clôture. Rapport [révisé](docs/audit-decode-prefill-2026-09-23.md) ; app installée, push et release **non faits**.

### OPT-2026-09-23-RUST-PERF-122 — validation sur deux prompts réels fournis par l'utilisateur — diagnostic terminé

- Entrées exactes fournies : « salut, comment ça va ? » et « invente une matrice carré d'ordre 3 et calcul son determinant ». Hypothèse : leurs distributions d'acceptation peuvent différer du prompt de contrôle PERF-120 ; elles déterminent si 70 tok/s est atteint en usage réel et si la fenêtre PERF-121 aide. Aucun changement de code pour cet essai. Différence précise avec PERF-120/121 : vrais prompts de l'utilisateur, forme/langue/longueur de réponse inconnues.
- Baseline : binaire A archivé `/tmp/mlxl3-perf120-baseline` (règle cumulative, SHA-256 dans PERF-121) et release B actuel (fenêtre récente) ; PERF-121 ne donne aucun débit sur ces prompts. Protocole : `scripts/smoke-dflash-bridge.py` Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, contexte 4 096, cache OFF, greedy température 0/top_k 1/pénalité 1 ; fichiers de prompts conservés sous `docs/measurements/perf-122-*.txt`, budgets 48/128, warmups et deux répétitions alternées normal/DFlash dans un seul processus par prompt et par binaire, **sans simultanéité GPU**. Mesurer tps decode, acceptés/proposés, TTFT, durée complète, mémoire, hash/texte ; si EOS avant budget, marquer non comparable à ce budget et ne pas présenter de ratio. Commencer par B ; lancer A adjacent uniquement pour une différence de politique interprétable. MLX 0.32.2, SDK 26.5, M5 24 Gio, batterie ~35 % en décharge ; sortie brute JSONL/stderr adjacents. Statut en cours, code local, app non installée/non publiée.
- Prompt utilisateur salut, candidat B, `perf-122-salut-b.jsonl`, stderr vide : 48 tokens atteints avec hash exact normal/DFlash, DFlash **55,35/55,00 tok/s**, 36/52 acceptés en 11 blocs, normal **43,08/43,54** ; 128 tokens atteints avec hash exact, DFlash **50,04/50,14 tok/s**, 88/144 en 39 blocs, normal **45,10/44,86**. Durée complète DFlash **1,235/1,191 s** puis **2,901/2,861 s**. Le 70 n'est pas atteint sur cette conversation. Après le seuil de huit blocs, l'acceptation 36/52 (~69 %) ne force pas la largeur N=2 au court ; à 128 la fenêtre peut varier, mais un ancien binaire n'est utile que si la forme de décision diffère. Continuer avec la matrice avant décision globale.
- Prompt utilisateur matrice, candidat B, `perf-122-matrice-b.jsonl`, stderr vide : 48 tokens hash/texte égaux, DFlash **70,12/70,82 tok/s**, 38/42 acceptés en 9 blocs, normal **44,77/44,62** ; 128 tokens hash/texte égaux, DFlash **67,66/66,23 tok/s**, 102/120 acceptés en 24 blocs, normal **44,93/44,57**. Durée complète DFlash **0,863/0,814 s** à 48 et **2,068/2,067 s** à 128. Cette question dépasse effectivement 70 tok/s **à 48 seulement**, pas à 128 ni sur le salut. Le rendement draft élevé (~85 % à 128) est une nouvelle distribution précise par rapport au rejet historique de N=6/7 à 72 % (PERF-113) ; une largeur 6 conditionnelle pourrait être testée séparément, sans extrapoler le gain. Aucune conclusion universelle.
- Ancien binaire A adjacent sur le salut (`perf-122-salut-a.jsonl`, stderr vide) : 48 DFlash **57,46/57,53 tok/s**, mêmes 36/52 et 11 blocs ; le normal adjacent **45,17/45,37** est aussi plus rapide que lors de B (**43,08/43,54**), donc l'écart court B/A vient de la dérive machine et **ne mesure pas une régression de politique**. À 128 A **49,55/49,39 tok/s**, 88/147 en 39 blocs contre B **50,04/50,14**, 88/144 en 39 blocs, mêmes hashes/textes ; gain faible de B, pas 70. La matrice à 128 reste N=5 (120 propositions/24 blocs), donc l'ancien binaire n'ajouterait pas un contrôle de politique distinct. Décision PERF-122 : diagnostic terminé ; 70 atteint sur un des quatre couples prompt/budget, pas sur l'usage fourni dans son ensemble. Batterie ~33 % en décharge après ces runs, aucun processus modèle actif.

### OPT-2026-09-23-RUST-PERF-123 — six propositions quand l'acceptation récente est haute — rejeté

- Différence précise avec les balayages N=6/7 rejetés PERF-73/79/113 : ces prompts avaient un rendement à 128 de ~72 % (PERF-113) et N=6 réduisait au plus un bloc ; la question de matrice de l'utilisateur atteint **102/120 = 85 %** à 128, 24 blocs et 66,23–67,66 tok/s. Hypothèse : après huit blocs, si les huit derniers acceptent ≥80 %, N=6 peut économiser des vérifications entières et porter cette sortie à 70 tok/s sans changer les tokens ; N=5 autrement, N=2 sous 60 % comme PERF-121. Aucun gain universel présumé, aucune modification du calcul cible à largeur égale.
- Baseline code local archivée `/tmp/mlxl3-perf121-baseline`, SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`. Protocole avant code : test CPU rouge/vert de la décision ≥80 % et des bornes de contexte/sortie ; Kani actualisé sur N≤6 et non dépassement des budgets. Construire release, mesurer d'abord le prompt matrice 48/128 du vrai bridge, deux warmups et deux répétitions alternées normal/DFlash, puis A adjacent et B2 si le signal est positif ; relever acceptation/blocs/decode/total/TTFT/mémoire/hashes. Tester aussi le salut 48/128 et le prompt anglais 128 pour régression ; long 2 825/128 seulement si la politique peut y atteindre N=6. Si gain utile, exiger logits FP16/80 états sur M=7, vrais blocs et suite Rust/Clippy ; sinon restaurer le code et journaliser le rejet. Un seul processus modèle à la fois, M5 24 Gio, MLX 0.32.2, SDK 26.5, batterie ~33 % en décharge. Statut en cours, app non installée/non publiée.
- Test CPU rouge sur l'ancien code : acceptation 32/40 après huit blocs renvoie N=5 au lieu de N=6 (`perf-123-red-test.log`) ; le test passe avec la candidate (`perf-123-green-test.log`), y compris borne terminale et zéro proposition. Kani 0.68.0 `adaptive_work_respects_budgets_and_selected_width` **67 obligations, zéro échec, 3/3 couvertures** (`perf-123-kani-width.log`) : pour tous les `usize` symboliques du harnais, N≤6, N<context/output, branche basse ≤2, haute égale à la borne 6, sinon borne 5. Il ne prouve pas le gain GPU ni la qualité numérique. Build release et bridge à suivre ; aucun gain annoncé.
- Build release réussi (`perf-123-build.log`). Premier bridge matrice B1 `perf-123-matrice-b1.jsonl`, stderr vide, parité hash/texte : à 48 **70,69/70,76 tok/s**, 38/42 en 9 blocs, similaire à PERF-122 ; à 128 **53,60/65,45 tok/s**, 103/125 en 23 blocs contre référence PERF-122 **66,23/67,66**, 102/120 en 24 blocs. Le premier 128 a compilé la nouvelle forme M=7 pendant la mesure (hypothèse de cause, compilation non isolée) et **ne doit pas être pris pour steady-state** ; le second passage chaud ne montre pas de gain, malgré un bloc en moins. A adjacent puis B2 avant rejet/conservation, pas de 70 à 128.
- A adjacent `perf-123-matrice-a.jsonl`, stderr vide, hash exact : à 48 **67,76/68,45 tok/s** alors que normal baisse aussi à 43,10/43,07, illustrant la dérive machine ; à 128 **68,20/74,16 tok/s**, normal **44,65/48,25**, 102/120 en 24 blocs. Le second passage rapide reflète un changement de conditions non isolé, pas un effet du code A. **Correction du protocole avant nouvelle mesure :** lancer chaque variante directement avec `--tokens 128 --repeats 3` pour que les deux warmups précompilent M=7 et pour comparer trois passages alternés au même budget ; ordre B2/A2 séquentiel, sorties `perf-123-matrice-warm128-{b2,a2}.jsonl`. Le B1/A initial reste conservé comme échec de protocole sur la première passe M=7, aucun gain revendiqué.
- B2 avec warmup 128 effectif (`perf-123-matrice-warm128-b2.jsonl`, stderr vide) : hash exact, DFlash **72,08/72,00/71,27 tok/s**, 103/125 en 23 blocs ; normal **48,68/48,20/46,65 tok/s**. Ce passage atteint 70 sur la matrice/128, contrairement à B1 dont le premier 128 compilait M=7, mais A2 sous même protocole reste nécessaire : l'accélération du normal montre que le régime machine a aussi changé. Aucun gain causal de N=6 encore établi.
- A2 adjacent avec même warmup 128 (`perf-123-matrice-warm128-a2.jsonl`, stderr vide) : hash exact, DFlash N=5 **74,95/75,01/74,79 tok/s**, 102/120 en 24 blocs ; normal **48,90/49,16/48,88**. B2 N=6 **71,27–72,08** est **~4 % plus lent** malgré un bloc de moins ; les trois passages A2 et B2 dépassent 70 dans ce régime machine, donc ce seuil ne démontre pas un gain du candidat. Décision : **rejeter N=6 conditionnel**, restaurer source/test/contrat N≤5, reconstruire release et revérifier. Les sorties exactes suffisent pour conclure la performance de ce prototype ; contrôle numérique M=7 non entrepris puisque le candidat est rejeté. Aucun code N=6 ne doit rester.
- Restauration terminée : `native/src/speculative.rs` revenu à la politique N=5/N=2 de PERF-121, test ciblé et Kani (34 obligations, zéro échec, 2/2 couvertures) réussis (`perf-123-restored-test.log`, `perf-123-restored-kani.log`), `cargo fmt --all -- --check`, `git diff --check`, release build réussis (`perf-123-restored-build.log`). Le SHA-256 du release restauré est **exactement** `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, identique au binaire A archivé avant essai (`cmp` = 0). Aucun kernel/code N=6 publié ou conservé ; l'essai reste documenté comme rejeté, sans claim de 70 dû au code.

### OPT-2026-09-23-RUST-PERF-124 — salut à 128 avec échauffement identique à la matrice — diagnostic terminé

- Hypothèse de contrôle : la mesure PERF-122 du salut 48/128 (DFlash ~50 tok/s à 128) échauffait 48 tokens, tandis que PERF-123 sur la matrice a montré 74,79–75,01 tok/s avec échauffement direct à 128 et régime machine différent. Répéter **seulement** le salut/128 sous ce protocole corrigé permet de savoir si le seuil 70 reste non atteint dans le même type de campagne. Différence précise de protocole et de forme vs PERF-122 ; aucun changement de production ni gain présumé.
- Baseline : release restauré PERF-121 SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, M5 24 Gio, MLX 0.32.2, SDK 26.5, batterie **28 % en décharge** au départ. Protocole : `scripts/smoke-dflash-bridge.py ... --tokens 128 --repeats 3 --context-length 4096 --prompt-file docs/measurements/perf-122-salut.txt`, warmup normal et DFlash au même budget, alternance dans un seul processus, cache OFF, greedy, contrôler hash/texte, tokens générés, acceptation/blocs, TTFT/durée et mémoire. `docs/measurements/perf-124-salut-warm128.jsonl`/stderr. Après cette passe, stopper les charges GPU sur batterie basse et consigner le résultat ; app non installée/non publiée. État en cours.
- Résultat `perf-124-salut-warm128.jsonl`, stderr vide : 128 tokens atteints, hash/texte normal et DFlash égaux (`b38c8e6fb96d7548`), normal **48,92/48,70/48,70 tok/s**, DFlash **55,40/55,32/55,24 tok/s**, 88/144 acceptés en 39 blocs à chaque passe ; decode DFlash **2,293/2,296/2,299 s**, durée complète **2,643/2,594/2,653 s**. Le warmup et régime plus rapide améliorent le débit par rapport à PERF-122 (~50), mais ne rapprochent pas le salut de 70. À conditions semblables, la matrice atteint ~75 avec 102/120 acceptés : le rendement du draft explique largement la différence de blocs (39 contre 24), sans constituer une causalité de chaque milliseconde. Batterie 27 % après mesure, aucun processus modèle ; charges GPU arrêtées pour cette campagne. Décision : **diagnostic terminé, 70 non atteint sur tous les prompts réels**. Aucun changement de code.

### OPT-2026-09-23-RUST-PERF-125 — rang des corrections target sur les deux prompts réels — diagnostic terminé

- Hypothèse : sur « salut, comment ça va ? », les refus plus fréquents que sur la matrice proviennent soit de corrections target absentes du top-16 BF16 des logits draft, soit du score de transition qui préfère un autre candidat présent. Ce diagnostic de **forme/prompt nouveau** précise si les anciens essais de calibration par position (PERF-81) et top-32 (PERF-82), rejetés sur un prompt anglais à 48 tokens, ont une chance sur le salut à 128. Aucun réglage du sélecteur n'est présumé gagnant.
- Référence : binaire local PERF-121 restauré, SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, M5 24 Gio, MLX 0.32.2, SDK 26.5 ; PERF-124 salut 128 : 88/144, 39 blocs, 55,24–55,40 tok/s ; PERF-123 matrice 128 : 102/120, 24 blocs, 74,79–75,01 tok/s. Batterie 27 % en décharge : **aucune mesure GPU avant branchement secteur**.
- Protocole : instrumentation temporaire des premiers refus dans `DFlashChat::advance`, sans changement de sélection ; caster la ligne draft rejetée en BF16 comme le shader, compter les logits qui battent l'ID target selon le même ordre valeur puis ID, écrire rang, position, N et acceptés par bloc dans stderr. Lancer le bridge une fois par prompt à 128 tokens avec warmup à 128, cache OFF, même modèle résident et contrôler hash/texte, compteurs et mémoire ; conserver les sorties brutes `docs/measurements/perf-125-rank-{salut,matrice}.{jsonl,stderr}`. Les synchronisations et copies de ce diagnostic invalident ses temps comme mesure de performance. Retirer l'instrumentation, reconstruire et comparer le binaire restauré bit à bit. Classer les refus rang <16, 16–31, ≥32 avant tout essai de changement. App non installée/non publiée.
- Secteur détecté à 26 % (`pmset`, AC attaché, pas encore en charge). Instrumentation temporaire compilée avec `cargo check --release --locked --features mlx,chat --bin mlxl3-rs` et `cargo build --release --locked --features mlx,chat --bin mlxl3-rs` ; format Rust vérifié, build dans `docs/measurements/perf-125-build.log`. Statut : **en cours**, mesure GPU à suivre ; aucun résultat de rang encore.
- Première forme terminée, `perf-125-rank-salut.{jsonl,stderr}` : warmup et passage mesuré reproduisent exactement **39 blocs, 88/144 acceptés** ; texte/hash target égaux au normal (`b38c8e6fb96d7548`). Les 22 premiers refus par passe ont les mêmes rangs BF16 : **21/22 dans le top-16** (dont un au rang 0, beaucoup au rang 1), **0 entre 16 et 31**, **1 au rang 217**. Donc augmenter K à 32 ne peut pas résoudre ces refus ; le score de sélection/transition est une piste sur ce prompt nouveau. Le débit instrumenté (46,40 tok/s) est **invalide comme performance** à cause de la synchronisation/copie de 248 320 logits par refus. Matrice à mesurer ensuite dans le même protocole ; pas de conclusion de gain.
- Deuxième forme terminée, `perf-125-rank-matrice.{jsonl,stderr}` : warmup/passage mesuré reproduisent **24 blocs, 102/120 acceptés** et hash/texte target exact (`70fb796b1d06bb21`). Les sept premiers refus par passe ont les rangs **[3, 0, 16, 2, 16, 1, 2]** : cinq dans le top-16 et deux au rang 16 (0-based), aucun au-delà de 31. Les valeurs se répètent à l'identique entre warmup et mesure. Sur la matrice, top-32 pourrait seulement agir sur deux refus, mais l'essai PERF-82 ne gagnait rien sur son autre prompt et aucun gain de cette forme n'est établi. Le débit instrumenté (65,95 tok/s) est également **invalide comme mesure de performance**. Décision diagnostique : examiner d'abord le score de transition sur le salut ; aucun changement de sélecteur encore. Restauration du code/binaire diagnostique à vérifier avant clôture.
- Restauration **terminée** : instrumentation retirée de `main.rs`, release reconstruit (`perf-125-restored-build.log`), SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0` et `cmp` octet par octet égaux à `/tmp/mlxl3-perf121-baseline`; `git diff --check` réussi. Secteur en charge à 29 %, aucun processus modèle. Décision : **diagnostic terminé**, pas de gain revendiqué, code de production inchangé par PERF-125.

### OPT-2026-09-23-RUST-PERF-126 — poids de transition 0,5 sur conversations françaises — rejeté

- Hypothèse : PERF-125 montre que 21/22 corrections target refusées sur « salut » figurent déjà dans le top-16 BF16 ; l'arbitrage unary/transition peut être mal calibré pour cette distribution. Le poids **0,5** avait donné la même acceptation que 0,25 sur le prompt anglais court de PERF-79, mais n'a pas été mesuré sur ces **deux prompts utilisateur à 128 tokens** : nouvelle forme et protocole justifiant une revalidation. Son effet peut être négatif ; aucun gain présumé. L'essai top-32 PERF-82 ne sera pas répété sur le salut où il ne pourrait agir que sur 1/22 refus.
- Baseline : release PERF-121/125 restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, poids 0,25, greeting 55,24–55,40 tok/s et 39 blocs/88 acceptés, matrice 74,79–75,01 tok/s et 24 blocs/102 acceptés en steady-state après warmup 128. M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge à 29 %, cache OFF, contexte 4096, greedy, app non installée/non publiée.
- Protocole : modifier temporairement seulement la constante shader 0,25→0,5, reconstruire release, lancer `scripts/smoke-dflash-bridge.py` avec warmup 128 et une répétition sur le salut, puis la matrice si le salut réduit les blocs ; garder JSONL/stderr `perf-126-*.{jsonl,stderr}`. Vérifier hash/texte target, acceptés/proposés/blocs, decode, TTFT, durée complète, mémoire. Si le candidat passe ce crible, mesurer A/B/A/B adjacents avec 3 répétitions par binaire et contrôler logits/états sur la forme applicable avant intégration. Rejeter si pas de réduction de blocs ou si la matrice régresse ; restaurer code/binaire et vérifier SHA. Un seul processus modèle à la fois, pas de temps GPU purs revendiqués.
- Statut : **en cours, avant changement**.
- Crible salut B poids 0,5 terminé (`perf-126-salut-b.{jsonl,stderr}`, stderr vide) : hash/texte exacts et 128 tokens, **90/139 acceptés en 36 blocs**, contre référence 0,25 **88/144 en 39 blocs**. DFlash instrument-free **51,14 tok/s**, durée complète **2,868 s**, normal adjacent **43,10 tok/s** ; la machine est plus lente que PERF-124 (normal ~48,7–48,9), donc **aucun gain de débit causal** à déduire de 51,14 contre 55,3. La réduction de trois blocs justifie le crible matrice puis A/B adjacent. Statut en cours ; code local candidat 0,5, pas intégré.
- Crible matrice B poids 0,5 terminé (`perf-126-matrice-b.{jsonl,stderr}`, stderr vide) : hash/texte exacts, 128 tokens, **102/120 acceptés en 24 blocs**, identique au poids 0,25 ; DFlash **66,38 tok/s**, normal adjacent **42,74** (régime plus lent), durée complète DFlash **2,112 s**. Pas de changement de blocs, aucune régression d'acceptation observée sur cette forme ; débit non comparable aux anciens ~75 sans A adjacent. Suite : A/B apparié sur salut et matrice, avec warmup 128 identique et un seul processus à la fois.
- A1 salut poids 0,25 (`perf-126-salut-a1.{jsonl,stderr}`, stderr vide) : trois passes exactes, hash `b38c8e6fb96d7548`, **39 blocs, 88/144** chaque fois ; DFlash **48,87/49,01/48,97 tok/s**, normal **43,00/43,14/43,09**, durée complète DFlash **2,987/2,931/3,025 s**. C'est la référence adjacente dans le régime du crible B (~51,14), mais B1 à trois passages est nécessaire avant attribution.
- B1 salut poids 0,5 (`perf-126-salut-b1.{jsonl,stderr}`, stderr vide) : trois passes exactes même hash, **36 blocs, 90/139** chaque fois ; DFlash **50,40/50,40/50,37 tok/s**, normal **41,34/40,72/42,92**, durée complète DFlash **2,915/2,864/2,986 s**. Par rapport à A1, +~2,9 % de tok/s de decode et trois blocs économisés alors que normal ralentit ; avantage plausible mais **pas encore confirmé A2/B2**. Le seuil 70 reste loin.
- A2 salut poids 0,25 (`perf-126-salut-a2.{jsonl,stderr}`, stderr vide) : hash exact, **39 blocs, 88/144**, DFlash **48,86/48,85/45,64 tok/s**, normal **43,08/42,73/40,18**. Le troisième passage ralentit dans les deux modes ; utiliser médianes et normal adjacent pour contrôler cette dérive. B2 suit avant décision.
- B2 salut poids 0,5 (`perf-126-salut-b2.{jsonl,stderr}`, stderr vide) : même hash, **36 blocs, 90/139**, DFlash **50,20/48,87/47,97 tok/s**, normal **43,00/42,46/41,78**. Les médianes A2/B2 sont **48,85/48,87**, soit aucun gain de débit absolu résolu dans cette paire ; la dérive par passe reste visible. La baisse de trois blocs est certaine, mais B1/B2 ensemble ne démontre pas un gain proche de 70. Vérifier la matrice avant décision de conservation ou rejet.
- A1 matrice poids 0,25 (`perf-126-matrice-a1.{jsonl,stderr}`, stderr vide) : hash exact `70fb796b1d06bb21`, **24 blocs, 102/120**, DFlash **66,17/66,07/65,45 tok/s**, normal **43,08/43,00/42,47**, durée complète DFlash **2,118/2,074/2,138 s**. B1 adjacent suit.
- B1 matrice poids 0,5 (`perf-126-matrice-b1.{jsonl,stderr}`, stderr vide) : même hash, **24 blocs, 102/120**, DFlash **65,35/66,12/64,44 tok/s**, normal **42,83/42,57/42,54** ; médianes DFlash A1/B1 **66,07/65,35**, normal **43,00/42,57**. Aucune amélioration de blocs ni de ratio relatif résolue ; A2/B2 complète le protocole avant décision.
- A2 matrice poids 0,25 (`perf-126-matrice-a2.{jsonl,stderr}`, stderr vide) : même hash, **24 blocs, 102/120**, DFlash **66,49/66,10/65,17 tok/s**, normal **42,81/42,95/42,83** ; régime voisin de A1, mais troisième passage un peu plus lent. B2 suit.
- B2 matrice poids 0,5 (`perf-126-matrice-b2.{jsonl,stderr}`, stderr vide) : même hash, **24 blocs, 102/120**, DFlash **62,16/63,47/72,53 tok/s**, normal **40,62/46,39/49,69**. Les deux modes accélèrent fortement d'un passage à l'autre ; la passe DFlash >70 reflète ce régime et **ne démontre pas un gain du poids**. Aucune réduction de blocs sur la matrice, tandis que salut A2/B2 a des médianes presque égales malgré trois blocs de moins. Décision : **rejeter 0,5 comme optimisation E2E non démontrée** et restaurer 0,25 ; ne pas engager le contrôle GPU d'états d'un candidat rejeté. Logs négatifs conservés. Restauration à vérifier.
- Restauration **terminée** : constante Metal 0,25 rétablie, release reconstruit (`perf-126-restored-build.log`) SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, `cmp` égal au binaire pré-essai ; format Rust et `git diff --check` réussis. Aucun code 0,5 conservé, aucun processus modèle. Statut final **rejeté**, app non installée/non publiée.

### OPT-2026-09-23-RUST-PERF-127 — N=3 dans la branche d'acceptation récente basse — rejeté

- Hypothèse : la politique PERF-121 passe de N=5 à N=2 après huit blocs si l'acceptation récente est <60 %. Sur le salut/128, 88/144 (~61 % global), elle alterne N=5/N=2 et finit à 39 blocs ; une largeur **N=3 seulement dans cette branche** peut amortir davantage les blocs sans coût M=6 partout. Différence précise des essais historiques : PERF-118 a rejeté N=3 **fixe** sur 2 825 tokens à 38 % d'acceptation, PERF-113 a préféré N=5 sur un autre prompt à 72 % ; ce cas français intermédiaire et politique conditionnelle n'ont pas été mesurés. Aucune promesse de 70 tok/s.
- Baseline : release restauré PERF-121 SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, salut PERF-124 55,24–55,40 tok/s et 39 blocs/88 acceptés, matrice PERF-123 74,79–75,01 et 24 blocs/102 acceptés, long PERF-121 3,325–3,382 s decode et 47 blocs/80 acceptés. M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge ; un seul modèle à la fois.
- Protocole : prototype local `adaptive_proposals` N=2→3 uniquement sous <60 %, release, crible bridge salut/128 après warmup 128 avec un passage, puis matrice/128 et long 2 825/128 **si** le salut gagne en blocs ou débit. Cache OFF, greedy, comparer hash/texte, budgets, acceptés/proposés/blocs, tps decode, durée totale/TTFT, mémoire. Fichiers `docs/measurements/perf-127-*.{jsonl,stderr}`. Si signal positif et pas de régression longue, A/B adjacent puis logits/états bit à bit, tests, Kani/build pour un code retenu. Sinon restaurer source et binaire exacts. Aucun gain micro présenté comme gain complet ; app non installée/non publiée.
- Statut : **en cours, avant changement**.
- Crible salut N=3 (`perf-127-salut-b.{jsonl,stderr}`, stderr vide) : hash/texte exacts, 128 tokens, **89/154 acceptés en 37 blocs** contre N=2 référence **88/144 en 39 blocs**. DFlash **54,25 tok/s**, durée complète **2,676 s**, normal adjacent **50,60 tok/s** ; ce régime machine est plus rapide que l'A/B PERF-126 (normal ~40–43), donc le nombre 54,25 n'est **pas** un gain causal. Deux blocs de moins mais dix vérifications supplémentaires : matrice puis long à contrôler avant A/B.
- Crible matrice N=3 (`perf-127-matrice-b.{jsonl,stderr}`, stderr vide) : hash exact, **102/120 en 24 blocs**, donc la branche basse ne s'active pas ; DFlash **75,95 tok/s**, normal **49,49** sous régime rapide. Ce 70+ confirme seulement la performance déjà obtenue avec N=5 sur la matrice, **pas** un gain N=3. Contrôle long suivant, car PERF-118 avait favorisé N=2 sur ce cas à faible rendement.
- Crible long N=3 (`perf-127-long-b.{jsonl,stderr}`, stderr vide) : hash/texte exacts `cea6d766333b2e34`, **80/165 en 47 blocs**, contre N=2 historique PERF-121 **80/147 en 47 blocs**. La largeur accrue n'accepte **aucun** token supplémentaire sur ce long prompt et fait vérifier 18 propositions de plus ; decode mesuré **3,249 s** et durée complète **9,700 s**, normal adjacent **2,737 s** de decode/**8,479 s** complet. Les valeurs absolues ne sont pas comparables à PERF-121 dont la machine était plus lente ; A adjacent requis pour chiffrer le coût N=3. Le gain de deux blocs du salut n'est pas suffisant pour intégrer une régression longue potentielle.
- A adjacent long N=2 (`perf-127-long-a.{jsonl,stderr}`, stderr vide) : même hash/texte, **80/147 en 47 blocs**, decode DFlash **3,014 s** contre B N=3 **3,249 s** (**+7,8 % de durée de decode**), normal adjacent quasi identique **2,739/2,737 s** pour A/B ; complet **9,484 s** A contre **9,700 s** B. L'écart vient du travail supplémentaire sans token gagné ; il suffit à **rejeter N=3 global sous <60 %**, même si le salut a deux blocs de moins. Aucun contrôle d'états du prototype rejeté ; restaurer code, build et hash avant clôture.
- Restauration **terminée** : politique N=2 sous <60 % rétablie, release reconstruit (`perf-127-restored-build.log`), SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0` et `cmp` égaux au binaire PERF-121 ; `cargo fmt --all -- --check` et `git diff --check` réussis. Aucun N=3 candidat conservé, aucun modèle actif. Statut final **rejeté** ; app non installée/non publiée.

### OPT-2026-09-23-RUST-PERF-128 — poids de transition 1,0 sur le salut français/128 — rejeté

- Hypothèse : 21/22 corrections target refusées sur le salut figurent dans le top-16 BF16 (PERF-125), mais 0,5 ne réduit que trois blocs et ne gagne pas clairement du temps (PERF-126). Tester **1,0** sur ce prompt français à 128 vérifie si un poids de transition plus fort corrige un nombre **nettement** plus élevé de blocs. PERF-79 avait écarté 1,0 sur un **prompt anglais à 48 tokens** (76 % contre 86,7 % d'acceptation à 0,25) ; cette nouvelle forme/prompt est la raison précise de l'essai, pas une répétition silencieuse. Aucun effet positif présumé.
- Baseline : release actuel/restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, poids 0,25 ; salut 128 **39 blocs, 88/144**, 55,24–55,40 tok/s dans PERF-124 après warmup 128, matrice 24 blocs/102/120. M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge, un seul processus GPU.
- Protocole : prototype local constante Metal 0,25→1,0, build release, bridge salut/128 avec warmup à 128 et un passage mesuré, cache OFF, contexte 4096, greedy ; sortie `docs/measurements/perf-128-salut-b.{jsonl,stderr}`. Contrôler hash/texte, budget, acceptés/proposés/blocs, decode/TTFT/complet, mémoire. Rejeter immédiatement si les blocs ne baissent pas **d'au moins cinq** ou si l'acceptation régresse, en restaurant source/binaire exacts ; sinon matrice et A/B adjacent, puis validation logits/états avant conservation. Les débits non appariés ne sont pas un gain. Aucun install/push/release.
- Statut : **en cours, avant changement**.
- Crible poids 1,0 terminé (`perf-128-salut-b.{jsonl,stderr}`, stderr vide) : hash/texte exacts `b38c8e6fb96d7548`, 128 tokens, mais **87/141 acceptés en 40 blocs**, contre poids 0,25 **88/144 en 39 blocs**. Le poids plus fort augmente donc les vérifications au lieu d'en retirer cinq ; DFlash **56,57 tok/s** et normal adjacent **51,88** sous machine plus rapide ne constituent aucun gain causal face aux anciennes valeurs. Décision : **rejet immédiat**, sans matrice ni A/B selon le critère préenregistré ; restaurer source/binaire et vérifier SHA. Aucun contrôle d'états nécessaire pour un prototype non retenu.
- Restauration **terminée** : poids 0,25 remis, release reconstruit (`perf-128-restored-build.log`) SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, `cmp` égal au binaire pré-essai, `git diff --check` réussi. Statut final **rejeté** ; aucun candidat 1,0 conservé, aucun processus modèle, app non installée/non publiée.

### OPT-2026-09-23-RUST-PERF-129 — profil des blocs du salut/128 — diagnostic terminé

- Hypothèse diagnostique : la vérification target dominait 3,218/4,069 s dans le test long PERF-117, mais le salut a un contexte court, 39 blocs et un rendement du draft différent. Une répartition sur **ce nouveau prompt réel** peut dire si un chantier de kernel target ou de draft a le potentiel nécessaire aux ~0,48 s de decode à économiser pour passer de 55,3 à 70 tok/s sur 127 tokens decode. Les temps synchronisés au choix de token sont des temps muraux d'étapes, **pas** des temps GPU purs ni un gain.
- Référence : release PERF-121/128 restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, salut/128 PERF-124 **2,293–2,299 s** decode, **39 blocs, 88/144**, 55,24–55,40 tok/s. Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge ; un seul modèle.
- Protocole : instrumentation temporaire dans `DFlashChat::advance`, mesurer par bloc le mur de `dflash_input+draft.forward_hidden+dflash_logits+select_greedy`, puis `verify_tokens_exact_with_dflash_capture+chat_greedy_ids`, et le reste commit/cache ; warmup et un passage bridge salut/128, cache OFF, contexte4096, greedy. Garder `docs/measurements/perf-129-salut.{jsonl,stderr}` et log build ; contrôler hash/texte, budgets, blocs, mémoire, comparer seulement les **proportions** d'étapes, sans somme présentée comme GPU. Retirer sonde et comparer SHA du release restauré. Aucun candidat de calcul ni app installée/publiée.
- Statut : **en cours, avant instrumentation**.
- Diagnostic terminé (`perf-129-salut.{jsonl,stderr}`, stderr de trace seulement) : warmup et passage mesuré reproduisent **39 blocs, 88/144**, hash/texte exacts `b38c8e6fb96d7548`. Passage mesuré, somme des 39 temps muraux : draft/input/head/sélection **0,430105 s**, vérification target+argmax **1,808969 s**, commit/cache CPU immédiat **0,002809 s** ; target représente **80,7 %** de ces étapes, médiane target par bloc **55,122 ms**. Decode bridge instrumenté **2,243952 s** contre ~2,29 s PERF-124 sous autre régime ; ses chiffres absolus ne sont pas une nouvelle baseline. À 127 tokens decode, 70 tok/s impose ≤**1,814 s** : la seule étape target actuelle en occupe presque toute la durée. Il faut donc réduire nettement **nombre de vérifications** ou **coût target par vérification**, pas seulement la soumission draft. Ces étapes synchronisent le choix de token et peuvent payer des graphes différés ; ne pas les appeler temps GPU. Retirer la sonde et restaurer le binaire avant clôture.
- Restauration **terminée** : sonde retirée, release reconstruit (`perf-129-restored-build.log`), SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, `cmp` égal au binaire PERF-121 ; format Rust et `git diff --check` réussis. Aucun processus modèle. Décision : **diagnostic terminé**, aucun gain revendiqué ni code de calcul conservé ; app non installée/non publiée.

### OPT-2026-09-23-RUST-PERF-130 — intervalles du score au premier refus du salut — diagnostic terminé

- Hypothèse diagnostique : PERF-125 trouve 21/22 corrections dans le top-16, mais les poids globaux 0,5 (PERF-126) et 1,0 (PERF-128) n'apportent pas un gain utile. Les intervalles de poids unary+transition qui auraient sélectionné la correction **au premier refus, à prédécesseur fixé** peuvent dire si une calibration simple par position reste plausible ; cette analyse ne prédit pas à elle seule la trajectoire E2E après changement de token. PERF-81 avait balayé les positions 0/4 sur un autre prompt anglais sans effet : la nouvelle forme française/128 justifie le diagnostic, pas une répétition de son ancien balayage.
- Référence : release restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, salut/128 39 blocs, 88/144, trace des rangs `perf-125-rank-salut.stderr`, profil `perf-129-salut.stderr`. Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge, cache OFF/contexte4096/greedy.
- Protocole : sonde temporaire après `select_greedy` lisant candidats top-16, unary BF16 et edges FP32 du seul prédécesseur réellement choisi, sans modifier les tokens. Une passe bridge salut/128 après warmup128, sorties `perf-130-salut.{jsonl,stderr}` ; contrôler hash/texte/blocs. Joindre la trace des rangs PERF-125 par bloc/position et calculer hors GPU, pour les 21 refus couverts, les poids positifs où la cible gagne parmi 16 candidats. Si les intervalles ne se recoupent pas ou ne peuvent corriger assez de refus pour ~9 blocs économisés, arrêter la calibration ; sinon préenregistrer un candidat E2E. Temps instrumentés invalides pour la performance. Retirer sonde et vérifier binaire SHA avant clôture ; aucun install/push/release.
- Statut : **en cours, avant instrumentation**.
- Diagnostic terminé (`perf-130-salut.{jsonl,stderr}` et `perf-130-intervals.json`) : 128 tokens, hash/texte exacts `b38c8e6fb96d7548`, **39 blocs, 88/144** ; les 39 traces mesurées et les 39 warmups concordent avec PERF-125, donc jointure par ordre et première position de refus valide sur cette forme. Pour les **21** corrections couvertes par top-16, comparer le score BF16-unary + poids×edge FP32 au prédécesseur réellement choisi : seulement **8/21** ont un intervalle de poids **non négatif** où elles gagnent localement ; **13/21** ne peuvent être choisies par aucun poids non négatif à ce prédécesseur. Un poids global 0–5 ne corrige au mieux que **5** de ces 21 premiers refus locaux ; même un poids distinct par position ne peut en corriger au mieux que **7** (0 à position 0, 3 à position 1, 2 à position 2, 0 à position 3, 2 à position 4 ; certains exigent des poids >10). Ces bornes portent **seulement sur les lignes/prefixes de la trajectoire actuelle** ; changer un token changerait les étapes suivantes. Elles ne sont pas une borne E2E universelle. Décision : **ne pas poursuivre une calibration scalaire par position** comme piste 70 tok/s sur ce salut ; les anciens essais PERF-81/126/128 et ce diagnostic rendent ce coût injustifié. Les temps instrumentés sont invalides pour la performance. Sonde à retirer et binaire à restaurer.
- Analyse exploratoire **post hoc, non préenregistrée**, sur les mêmes traces (aucun nouveau calcul GPU) : parmi les 88 propositions acceptées de la trajectoire, 86 étaient le candidat unary de rang 0 ; parmi les 22 premiers refus, neuf corrections sont au rang 1. Forcer le rang 1 quand la marge score(rang0)−score(rang1) <1,25 aurait localement corrigé sept de ces neuf refus, mais aurait aussi changé cinq positions déjà acceptées. Les blocs futurs changeraient ; cette comparaison ne prouve aucun gain E2E et ne fonde aucun prototype. L’écart de protocole est conservé explicitement.
- Restauration **terminée** : sonde retirée, release reconstruit (`perf-130-restored-build.log`), SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, `cmp` égal au binaire PERF-121 ; format Rust et `git diff --check` réussis, aucun processus modèle. Décision finale : **diagnostic terminé**, aucun changement de calcul conservé ni gain revendiqué ; app non installée/non publiée.

### OPT-2026-09-23-RUST-PERF-131 — potentiel d'un seul pas de lookahead du sélecteur — rejeté au crible hors ligne

- Hypothèse : le score glouton courant choisit presque toujours le meilleur unary, alors que 21/22 corrections du salut figurent dans le top-16 et 13 ne peuvent gagner avec aucun poids scalaire non négatif au prédécesseur fixé (PERF-130). Un **seul pas** de valeur future parmi les 16 candidats peut discriminer autrement sans ajouter de projection modèle ; c'est distinct du Viterbi global PERF-78, rejeté sur un prompt anglais/48 parce qu'il changeait trop tôt le chemin. Ce n'est qu'un diagnostic hors ligne, pas un gain présumé.
- Référence : release restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge ; salut/128 39 blocs/88 acceptés/144 proposés, matrice/128 24 blocs/102/120. Traces des premières corrections `perf-125-rank-{salut,matrice}.stderr`. Un processus modèle à la fois, cache OFF, greedy, contexte4096.
- Protocole : sonde temporaire dans `select_greedy` copiant après sélection les 16 IDs, unary BF16 et **toutes** les transitions 16×16 de chaque position ; aucun token ni calcul cible changé. Exécuter le bridge salut et matrice à 128 après warmup128, garder `docs/measurements/perf-131-{salut,matrice}.{jsonl,stderr}`, contrôler hashes/compteurs. Hors GPU, joindre les labels des préfixes acceptés et premiers refus de PERF-125 et comparer, au prédécesseur courant figé, `score local + λ × meilleur score de la position suivante` pour λ∈[0,1,5]. N'envisager une candidate E2E que si elle corrige localement **au moins huit** premiers refus du salut avec **au plus deux** positions acceptées dégradées et sans dégrader la matrice ; sinon rejeter sans changer le sélecteur. Les trajectoires après changement ne sont pas prédictibles par ce replay. Les temps de la sonde sont invalides comme benchmark. Retirer code et vérifier SHA du binaire, app non installée/non publiée.
- Statut : **en cours, avant instrumentation**.
- Première forme salut terminée (`perf-131-salut.{jsonl,stderr}`) : 78 graphes pour warmup+mesure, **39 blocs, 88/144** à chaque passe, hash/texte exacts `b38c8e6fb96d7548`, aucune erreur hors trace. Copie CPU de tous les edges pendant ce diagnostic ; débit instrumenté non interprétable. La matrice suit avant l’analyse hors ligne conjointe et aucune candidate n’est encore retenue.
- Deuxième forme matrice terminée (`perf-131-matrice.{jsonl,stderr}`) : 48 graphes warmup+mesure, **24 blocs, 102/120** à chaque passe, hash/texte exacts `70fb796b1d06bb21`, stderr de trace seulement. Les deux trajectoires de référence sont intactes ; comparer maintenant hors GPU les labels des propositions acceptées et premières corrections. Aucun débit du bridge sondé n’est utilisé.
- Replay local hors GPU terminé, λ=0..1,50 par pas 0,01, avec le prédécesseur réel fixé et le score `unary[p,j]+0,25·edge[p,pred,j]+λ·max_k(unary[p+1,k]+0,25·edge[p+1,j,k])` ; reconstruction λ=0 exactement égale aux propositions enregistrées sur tous les blocs. Salut : parmi 21 premiers refus couverts, **au mieux 3 corrigés**, mais aussi **3 positions acceptées changées** (λ≈0,85–0,92) ; meilleur solde local seulement **2 corrigés/1 cassée** (λ≈0,27–0,34). Matrice : au mieux 3/5 refus couverts corrigés avec 1 position acceptée changée (λ≈1,22–1,29). Le critère préenregistré de **≥8 corrections et ≤2 cassures sur le salut** échoue largement ; décision **rejeter le lookahead à un pas avant toute candidate GPU E2E**. Ces comptes locaux ne prédisent pas la trajectoire après une proposition différente et ne sont pas un gain de decode. Traces brutes conservées et grille complète dans `docs/measurements/perf-131-offline.json` ; sonde à retirer.
- Restauration **terminée** : sonde retirée, release reconstruit (`perf-131-restored-build.log`), SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, `cmp` octet par octet égal au binaire PERF-121 ; `cargo fmt --all -- --check` et `git diff --check` réussis, aucun processus modèle. Décision finale : **rejeté au crible hors ligne**, aucun lookahead intégré, aucun débit candidat revendiqué ; app non installée/non publiée.

### OPT-2026-09-23-RUST-PERF-132 — largeur N=4 pour acceptation récente intermédiaire — rejeté

- Hypothèse : PERF-129 mesure par bloc du salut target+argmax médian **55,12 ms** à N=5 contre **34,74 ms** à N=2, mais N=2 peut accroître le nombre de blocs. Sur la trajectoire salut/128 actuelle, après les huit premiers blocs, **14 blocs N=5** partent avec acceptation récente entre **60 et 80 %** ; y proposer N=4 peut économiser une ligne de vérification sans perdre beaucoup de tokens acceptés. Différence avec PERF-113/118 : N=4 y était **fixe** sur des prompts/débits de draft distincts ; ce test est une branche conditionnelle sur le prompt français/128 réel. Le gain 70 n'est pas présumé.
- Baseline : release restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2, M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge ; salut 39 blocs/88/144 et 55,24–55,40 tok/s warm128 (PERF-124), matrice 24 blocs/102/120 et ~75 tok/s warm128 (PERF-123), long 47 blocs/80/147 et 3,014 s decode adjacent (PERF-127). App non installée/non publiée.
- Protocole : prototype local `adaptive_proposals` : après huit blocs, N=2 si taux récent <60 %, N=4 si 60–<80 %, sinon N=5 ; bornes existantes conservées. Release et bridge salut/128 après warmup128, une répétition, cache OFF/greedy/contexte4096 ; si nombre de blocs et temps prometteurs, matrice/128 puis long 2825/128, mêmes contrôles hash/texte, TTFT/complet/decode, acceptation, mémoire. Sorties `perf-132-*.{jsonl,stderr}`. Ne retenir que si une comparaison A/B adjacente montre au moins **5 %** de baisse du temps de decode sur le salut, sans régression mesurable sur matrice/long, puis contrôles logits/80 états et Kani sur la politique avant intégration. Sinon restaurer code/binaire identiques. Un seul modèle à la fois, aucun gain micro extrapolé.
- Statut : **en cours, avant changement**.
- Premier crible B salut/128 terminé (`docs/measurements/perf-132-salut-b.{jsonl,stderr}`), release candidat construit (`perf-132-build.log`) : hash/texte exacts `b38c8e6fb96d7548`, 128 tokens, **39 blocs, 87/136**, decode **2,596315 s / 48,915 tok/s**, durée complète **3,055706 s**, TTFT **0,459359 s**, empreinte processus **15,859 Go**. Normal adjacent **2,903767 s / 43,736 tok/s**. Le régime machine est plus lent que PERF-124, donc 48,9 contre 55,3 n'est pas une régression causale démontrée ; nombre de blocs inchangé et un token accepté de moins. Une référence A immédiatement adjacente avec le binaire archivé est requise avant décision ; N=4 reste prototype non retenu.
- Référence A adjacente terminée (`perf-132-salut-a.{jsonl,stderr}`), même hash/texte/128 : **39 blocs, 88/144**, decode **2,581937 s / 49,188 tok/s**, complet **2,950754 s**, TTFT **0,368792 s** ; normal adjacent **2,872198 s / 44,217 tok/s**. B est **+0,56 %** plus lent en decode et **+3,56 %** en durée complète malgré huit propositions de moins ; différence petite et potentiellement bruitée, mais le critère préenregistré de **≥5 % de baisse decode** échoue et aucun bloc n'est supprimé. Décision : **rejeté au crible**, ne pas lancer matrice/long inutiles, restaurer source et release exactement ; aucune intégration ni app installée/publiée.
- Restauration terminée : branche N=4 retirée, release reconstruit (`perf-132-restored-build.log`) ; SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0` et `cmp` identiques au binaire PERF-121. `cargo fmt --all -- --check` et `git diff --check` passent. Statut final **rejeté**, aucun changement exécutable conservé.

### OPT-2026-09-23-RUST-PERF-133 — normalisations groupées du vérificateur Qwen — rejeté

- Hypothèse : `LinearLayer::forward_verification_impl` normalise actuellement chaque ligne M=1 séparément avant `concatenate`, puis fait encore une normalisation par ligne après l'attention. Regrouper les lignes **avant** l'appel RMSNorm réduit les noeuds/dispatchs de la vérification M=2..6 sans changer mathématiquement les lignes. Le format M peut toutefois choisir un kernel MLX différent et changer des bits ; l'identité réelle est le premier filtre. Les essais historiques de batch M=6 sur les portes GDN et routeur dense ont échoué numériquement sur certaines entrées ; cette piste est distincte car elle ne change que RMSNorm, mais elle exige le même contrôle strict. Baseline release PERF-121 restauré SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0`, salut/128 **39 blocs, 88/144**, matrice/128 **24 blocs, 102/120**. M5 24 Gio, MLX 0.32.2, SDK 26.5, secteur en charge, cache OFF/greedy.
- Protocole : variante I, concaténer les valeurs puis normaliser `input_norm` en un appel, garder `post_norm` inchangé ; lancer le test GPU `long_prefix_verification_keeps_exact_state` sur le préfixe réel 2825 avant/après 48 tokens, retenus 1..6, logits FP16 et 80 états octet par octet. Si exact, variante IP ajoute le regroupement `post_norm` et refait le même test ; sinon restaurer la variante fautive immédiatement. Une variante numériquement exacte seulement passe au bridge chaud salut/128 puis matrice/128 et long/128 si signal prometteur, comparaisons A/B adjacentes, hashes/texte/blocs/TTFT/complet/decode/mémoire. Retenir seulement si gain decode ≥5 % sur le salut sans régression mesurable sur matrice/long, tests, Clippy, build et tentative Kani selon skill ; sinon restaurer source/release bit à bit. Les temps de tests GPU ciblés ne valent pas gain E2E. Logs `docs/measurements/perf-133-*`; aucun install/push/release.
- Statut : **en cours, avant changement**.
- Variante I (`input_norm` groupé seulement) : `cargo test --release --locked --features mlx,chat long_prefix_verification_keeps_exact_state -- --ignored --nocapture --test-threads=1` **réussi** (`docs/measurements/perf-133-input-norm-exact.log`). Sur le prompt 2 825, avant/après 48 tokens, retenus 1..6 : logits FP16 exacts à chacun des deux contrôles applicables et **80 états sans premier octet différent** à chacun des douze commits. Test 14,47 s après compilation ; ce temps n'est pas un benchmark d'inférence. Variante IP suivante ; aucun gain revendiqué.
- Variante IP (`input_norm` et `post_norm` groupés) : le même test GPU **réussi** (`perf-133-input-post-exact.log`, 14,03 s après compilation) avec 12/12 contrôles de commits exacts et les logits applicables exacts. Ce contrôle ciblé autorise un crible E2E, pas une recommandation : le comportement des graphes et le coût de concat/slice restent à mesurer.
- Premier crible B salut/128 (`perf-133-salut-b.{jsonl,stderr}`) : release candidat SHA-256 `886cb85220d0f8c26098849031e2f2be73a1593fa2aa1fb93ba3af9d356842c3`, hash/texte exacts `b38c8e6fb96d7548`, **39 blocs, 88/144**, decode **2,588456 s / 49,064 tok/s**, complet **2,959701 s**, TTFT **0,371220 s** ; normal adjacent **2,865552 s / 44,320 tok/s**. Malgré moins d'appels RMSNorm dans le graphe, le débit est voisin de la référence historique récente sous régime machine voisin ; référence A adjacente nécessaire avant décision. Aucun gain établi.
- Référence A immédiatement adjacente (`perf-133-salut-a.{jsonl,stderr}`) : même hash/texte et **39 blocs, 88/144**, decode **2,602066 s / 48,807 tok/s**, complet **2,982051 s**, TTFT **0,379964 s** ; normal **2,907203 s / 43,685 tok/s**. Le candidat B paraît **0,52 %** plus rapide en decode et **0,75 %** en complet, mais le normal adjacent était lui-même **1,43 %** plus rapide dans B ; aucun gain isolé et le seuil préenregistré de **5 %** échoue. Décision **rejeté au crible E2E**, sans campagne matrice/long ni vérification formelle supplémentaire d'un code à supprimer ; restauration source/release en cours après interruption de la session. Aucun gain revendiqué.
- Restauration terminée : `input_norm`/`post_norm` revenus à la forme par ligne, release reconstruit (`perf-133-restored-build.log`) et identique **octet par octet** au binaire PERF-121, SHA-256 `5156b0635464a6c06df14cae1e93e04142ac23149058ca1e90f73f2e03e6b0d0` ; `cargo fmt --all -- --check` et `git diff --check` passent. Statut final **rejeté**, pas de changement exécutable conservé. La batterie était passée à 100 % puis débranchée au contrôle final ; la paire A/B était déjà terminée. App non installée/non publiée.

### OPT-2026-09-24-RUST-AUDIT-10 — nouvelles pistes dans le code local — inspection statique terminée

- Demande : rechercher de nouvelles optimisations dans le projet, decode prioritaire puis prefill. Référence source : `fbfb357edffee4494a1be152666cf1ef33e47c9d` et changements locaux actuels, dont l'API compatible OpenAI ajoutée depuis le bilan précédent. Aucun nouvel essai après PERF-133 n'est inscrit à l'ouverture de cette inspection ; ne pas attribuer de boost à une fonctionnalité ajoutée sans mesure.
- Hypothèse : des coûts de synchronisation, de calcul finalement jeté, de copie d'états ou de soumission restent hors des variantes de largeur/sélecteur déjà rejetées. Protocole **statique uniquement** : suivre bridge/API → prefill/decode → projections, attention, états, sampling et sorties ; confronter chaque piste aux entrées et rapports antérieurs, identifier les fonctions concernées, les risques d'identité numérique et le contrôle décisif à préenregistrer avant toute future mesure. Les pistes déjà présentes ou rejetées doivent être identifiées comme telles, notamment PERF-98 et PERF-133 sur RMSNorm.
- Livrable prévu : `docs/nouvelles-pistes-performance-2026-09-24.md`, classement par impact potentiel, coût et preuve ; performance de chaque candidat **non mesurée**. Aucune compilation ni charge GPU prévue pour cette inspection, aucun changement exécutable, installation ou publication. Statut **en cours**.
- Résultat : [rapport livré](docs/nouvelles-pistes-performance-2026-09-24.md), **onze pistes** : six decode, deux prefill, trois API. Priorités moteur : dernier historique GDN écrit deux fois (60 Mio logiques par bloc sur ce Qwen, **pas** une mesure de RAM physique ou de gain), projections `a/b` avec dimension batch distincte à cribler numériquement, évaluation groupée des sorties MLX ; puis heads des chunks intermédiaires. Deux pistes sont des diagnostics préalables, sans modification recommandée avant preuve (rétention du stockage des slices, évictions des factories).
- Constats API : normaliseur sans `conversation_id` et ID de requête neuf, donc cache de conversation non réutilisé ; absence de `dflash2` transmis, donc decode normal ; attente SSE dans la boucle de lecture pouvant propager la lenteur client au bridge. Ce sont des constats source, **aucune mesure HTTP nouvelle**. Capacités existantes à rendre accessibles, pas nouveaux kernels.
- Rapprochement historique : le simple déplacement du slice avant le head LFM avait déjà échoué en PERF-16 (118/65 536 logits FP16 différents) ; seul un éventuel calcul de tuile utile conservant l'arithmétique QMM serait une nouvelle expérience. PERF-98 et PERF-133 recouvrent la même famille de normalisations groupées sans gain établi ; ne pas répéter. PERF-06/33 rejettent les clés courtes de factories ; la piste d'évictions doit d'abord montrer un phénomène différent. Gemma reste **non mesuré**, faute de checkpoint utilisable.
- Décision : **inspection terminée**, hypothèses à tester et gains **non mesurés** ; aucune nouvelle optimisation validée/intégrée. Lecture et recherches `rg`, empreintes SHA-256 des neuf sources principales dans le rapport ; seuls ce rapport et cette entrée ont été écrits pendant AUDIT-10. Aucun benchmark, test modèle, build, installation ou publication ; contrôles documentaires et `git diff --check` à la clôture. Les tests/proofs d'implémentation ne sont pas applicables à cette modification documentaire.
- Contrôles de clôture **réussis** : `git diff --check`, existence des 24 liens documentaires, correspondance des neuf empreintes source et onze identifiants de pistes distincts. Aucun essai AUDIT-10 encore en cours.

### OPT-2026-09-24-RUST-PERF-134 — retirer le dernier historique GDN dupliqué — synthèse terminée, pic mémoire validé, vitesse non concluante

- Autorisation : « vas y gp », après AUDIT-10. Hypothèse D1 : en vérification DFlash T=2..8, le commit complet utilise `state_out` et ne lit jamais la dernière tranche d'historique. Réduire l'historique à T−1 et ne plus écrire cette tranche retire 60 Mio logiques par bloc sur Qwen local (30 × 32 × 128 × 128 × 4 octets), sans modifier la récurrence. T=1 conserve son comportement actuel, y compris une tranche, afin d'éviter un nouvel output Metal vide. Aucun gain de temps présumé.
- Référence : HEAD `fbfb357edffee4494a1be152666cf1ef33e47c9d` + état local AUDIT-10/API HTTP. Reconstruire et archiver la baseline actuelle avant édition (l'ancien binaire PERF-121 ne contient pas la nouvelle API). M5 24 Gio, MLX 0.32.2 ; départ batterie 94 %, demande de secteur envoyée. Préserver les modifications existantes. Aucun install/push/release.
- Protocole : test physique déterministe T=1..8 comparant toutes les sorties et tous les historiques utiles avec la récurrence mono-token, plus assertion de capacité T−1 (T=1 inchangé), rejet T=0/9 ; vérifier que le contrôle de capacité échoue avant correctif. Kani sur le contrat Rust de capacité et les bornes des commits partiels, sans prétendre prouver Metal. Tests existants `long_prefix_verification_keeps_exact_state` (2 825, avant/après 48, logits FP16 et 80 états) et `selective_dflash_commit_matches_exact_prefix`, puis bridge salut/matrice/long à 128, warmup128, cache OFF, greedy, contexte4096, un seul modèle à la fois. Comparaison A/B/B/A, deux répétitions par passage si le crible est exact ; sorties `docs/measurements/perf-134-*`. Mesurer durée complète, TTFT, decode, blocs/acceptation et mémoire séparées. Retenir comme boost seulement un gain reproductible dépassant le bruit ; une économie d'allocation exacte sans gain stable sera explicitement classée comme optimisation mémoire. Rejeter toute divergence ou régression stable. Fmt, Clippy, tests Rust applicables, build et inspection CI à la clôture.
- Statut : **en cours, avant modification exécutable**.
- Baseline actuelle construite et archivée `/tmp/mlxl3-perf134-a`, SHA-256 `949b2bc1faf4672d4adb75a2349fd78a4d0e893b6223cdb79615ade8f66c11af` (`perf-134-baseline-build.log`). Test de capacité ajouté puis exécuté sur le kernel inchangé : **rouge attendu à T=2**, historique `[2,1,4,16,128]` contre `[1,1,4,16,128]`, après réussite T=1. Preuve `perf-134-history-red.log`. Correction suivante : capacité bornée commune Rust/Kani, garde d'écriture Metal, arithmétique inchangée.
- Correctif : test physique T=1..8 **réussi**, chaque sortie/état/historique utile comparé octet par octet à la récurrence sérielle déterministe, formes invalides T=0/9 rejetées (`perf-134-history-green.log`). Kani 0.68 **réussi**, fonction de capacité effectivement appelée en production, entrées `time` et `retained` i32 symboliques, zéro hypothèse excluant les bornes, quatre couvertures satisfaites (`perf-134-kani.log`). Ne prouve pas le shader. Secteur confirmé, batterie 93 % en charge, avant campagnes modèle. Contrôles longs et E2E encore en cours.
- Inspection CI : `.github/workflows/rust.yml` référençait trois scripts absents du dépôt (`native/check_parity.py`, `native/test_streaming_parity.py`, `native/test_kernel_sources.py`). Étapes impossibles remplacées par le test existant de validation du mode benchmark bridge ; fmt/Clippy/build/tests Rust et Kani conservés sur push/PR. Les tests Rust codec/streaming restent exécutés ; ceci ne remplace pas une comparaison Python inexistante. Pytest épinglé, aucun déploiement GitHub effectué. Validation locale du workflow à la clôture.
- Parité modèle **réussie** : contrôle long 2 825 avant/après 48, commits retenus 1..6, logits applicables et 80 états exacts (`perf-134-long-exact.log`) ; commits 1..8 exacts (`perf-134-commit-exact.log`). Test négatif du banc bridge réussi (`perf-134-bridge-harness-test.log`). Avant première comparaison de débit : seuil de boost retenu ≥3 % de baisse médiane reproduite dans les deux passages B et au-delà de la dérive des contrôles normaux ; en dessous, aucun gain de vitesse revendiqué. Une économie mémoire peut être retenue séparément si pas de régression stable >3 % du temps complet. Campagne A/B/B/A en attente de fin des compilations/vérifications CPU.
- Validation avant mesures **réussie** : `cargo kani --lib --no-default-features`, **22 harnesses / zéro échec** (`perf-134-kani-all.log`) ; Clippy strict tous targets avec mlx,chat (`perf-134-clippy.log`) ; tests release mlx,chat **49 réussis, 42 ignorés** (`perf-134-tests.log`), hors tests GPU ciblés déjà exécutés séparément ; fmt/diff propres. Workflow YAML parsé localement, aucune exécution GitHub distante sans push. B archivé `/tmp/mlxl3-perf134-b`, SHA-256 `a27da41c6122e83951ddfcad6509bdf0f26b57fe5b8b0b2cddc931905c70446b`. Conditions et empreintes conservées dans `perf-134-conditions.json`. Tous les compilateurs/provers arrêtés avant benchmark.
- Mesure salut/a1 terminée, deux répétitions après warmup128 : DFlash complet 2.680337 s, decode 2.313207 s / 54.904 tok/s, TTFT 0.367104 s ; normal decode 2.724101 s, pic MLX DFlash 13068496860 octets. Hash exact b38c8e6fb96d7548, blocs [39, 39], acceptés/proposés [[88, 144], [88, 144]]. Preuves `docs/measurements/perf-134-salut-a1.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure salut/b1 terminée, deux répétitions après warmup128 : DFlash complet 2.591894 s, decode 2.267105 s / 56.019 tok/s, TTFT 0.324762 s ; normal decode 2.667560 s, pic MLX DFlash 12999286748 octets. Hash exact b38c8e6fb96d7548, blocs [39, 39], acceptés/proposés [[88, 144], [88, 144]]. Preuves `docs/measurements/perf-134-salut-b1.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure salut/b2 terminée, deux répétitions après warmup128 : DFlash complet 2.756319 s, decode 2.424740 s / 52.381 tok/s, TTFT 0.331550 s ; normal decode 2.662820 s, pic MLX DFlash 12999282620 octets. Hash exact b38c8e6fb96d7548, blocs [39, 39], acceptés/proposés [[88, 144], [88, 144]]. Preuves `docs/measurements/perf-134-salut-b2.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure salut/a2 terminée, deux répétitions après warmup128 : DFlash complet 2.811505 s, decode 2.470842 s / 51.403 tok/s, TTFT 0.340627 s ; normal decode 2.746002 s, pic MLX DFlash 13068496860 octets. Hash exact b38c8e6fb96d7548, blocs [39, 39], acceptés/proposés [[88, 144], [88, 144]]. Preuves `docs/measurements/perf-134-salut-a2.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure matrice/a1 terminée, deux répétitions après warmup128 : DFlash complet 1.972601 s, decode 1.809607 s / 70.183 tok/s, TTFT 0.162951 s ; normal decode 2.733611 s, pic MLX DFlash 13069229524 octets. Hash exact 70fb796b1d06bb21, blocs [24, 24], acceptés/proposés [[102, 120], [102, 120]]. Preuves `docs/measurements/perf-134-matrice-a1.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure matrice/b1 terminée, deux répétitions après warmup128 : DFlash complet 1.965635 s, decode 1.805075 s / 70.387 tok/s, TTFT 0.160522 s ; normal decode 2.704050 s, pic MLX DFlash 13000007148 octets. Hash exact 70fb796b1d06bb21, blocs [24, 24], acceptés/proposés [[102, 120], [102, 120]]. Preuves `docs/measurements/perf-134-matrice-b1.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure matrice/b2 terminée, deux répétitions après warmup128 : DFlash complet 1.953385 s, decode 1.789089 s / 70.999 tok/s, TTFT 0.164255 s ; normal decode 2.678185 s, pic MLX DFlash 13000007124 octets. Hash exact 70fb796b1d06bb21, blocs [24, 24], acceptés/proposés [[102, 120], [102, 120]]. Preuves `docs/measurements/perf-134-matrice-b2.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure matrice/a2 terminée, deux répétitions après warmup128 : DFlash complet 1.943882 s, decode 1.784918 s / 71.153 tok/s, TTFT 0.158929 s ; normal decode 2.673175 s, pic MLX DFlash 13069213140 octets. Hash exact 70fb796b1d06bb21, blocs [24, 24], acceptés/proposés [[102, 120], [102, 120]]. Preuves `docs/measurements/perf-134-matrice-a2.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure long/a1 terminée, deux répétitions après warmup128 : DFlash complet 10.525197 s, decode 3.493786 s / 36.366 tok/s, TTFT 7.031374 s ; normal decode 2.975339 s, pic MLX DFlash 13505326812 octets. Hash exact cea6d766333b2e34, blocs [47, 47], acceptés/proposés [[80, 147], [80, 147]]. Preuves `docs/measurements/perf-134-long-a1.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure long/b1 terminée, deux répétitions après warmup128 : DFlash complet 11.838634 s, decode 3.768655 s / 33.699 tok/s, TTFT 8.069947 s ; normal decode 3.649610 s, pic MLX DFlash 13436120796 octets. Hash exact cea6d766333b2e34, blocs [47, 47], acceptés/proposés [[80, 147], [80, 147]]. Preuves `docs/measurements/perf-134-long-b1.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure long/b2 terminée, deux répétitions après warmup128 : DFlash complet 11.391381 s, decode 3.674389 s / 34.567 tok/s, TTFT 7.716957 s ; normal decode 3.269214 s, pic MLX DFlash 13436120796 octets. Hash exact cea6d766333b2e34, blocs [47, 47], acceptés/proposés [[80, 147], [80, 147]]. Preuves `docs/measurements/perf-134-long-b2.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- Mesure long/a2 terminée, deux répétitions après warmup128 : DFlash complet 12.259265 s, decode 4.045119 s / 31.996 tok/s, TTFT 8.213990 s ; normal decode 3.001481 s, pic MLX DFlash 13505322716 octets. Hash exact cea6d766333b2e34, blocs [47, 47], acceptés/proposés [[80, 147], [80, 147]]. Preuves `docs/measurements/perf-134-long-a2.{jsonl,stderr,conditions.json}`. Résultat individuel, conclusion après alternances.
- **Pause demandée par l'utilisateur** (« stop tu reprend apres »). Le dernier passage venait de se terminer : les 12 passages A/B/B/A sont conservés dans `docs/measurements/perf-134-summary.json` et leurs logs bruts ; aucun processus modèle/benchmark/Kani encore actif au contrôle `ps`. Aucun nouvel essai lancé après cette demande.
- État de reprise : correctif D1 et tests conservés **localement comme candidat**, parité et contrats validés, baisse du pic MLX observée d'environ 66 Mio sur les trois cas ; **aucun boost de vitesse validé**. Forte variation des temps, particulièrement sur le long (également dans les contrôles normaux), avec charge externe observée ; ne pas comparer ce régime aux anciens chiffres ni annoncer un gain. Synthèse statistique et décision finale mémoire/performance encore à terminer avant nouvelle campagne. D2/D3/P1 **non commencés**. A/B archivés dans `/tmp/mlxl3-perf134-{a,b}`, aucune app installée ni publication.
- À la reprise : relire cette entrée et les 12 sorties existantes, terminer la décision PERF-134 sans relancer silencieusement ces mesures, puis préenregistrer le crible numérique D2 `[T,1,H]` si poursuivi. La CI réparée a été parsée et son test bridge exécuté localement avec pytest 9.1.1 ; son pin actuel 8.4.2 est à aligner avec la version effectivement vérifiée. Les 22 harnesses Kani, 49 tests Rust, Clippy, fmt et tests GPU ciblés sont déjà consignés ci-dessus. Pas de reprise automatique programmée.

- Révision du **3 octobre 2026**, pendant AUDIT-11, après la nouvelle demande de recherche : synthèse des **12 passages existants uniquement**, sans reprise du benchmark suspendu. Les 24 répétitions DFlash mesurées et 24 contrôles normaux retrouvent toutes les médianes du résumé, avec hashes/blocs/acceptations constants ; preuve [relecture indépendante](docs/measurements/audit-2026-10-03-perf134-review.json). Économie de pic d'allocation MLX **65,996 à 66,016 Mio** dans les quatre comparaisons A/B de chaque cas : **validée sur ces mesures**, distincte d'une économie de RAM processus.
- Décision finale vitesse : **non concluante**, aucun boost revendiqué. Face à la médiane des passages A, decode salut B1 −5,22 % mais B2 +1,37 % ; matrice +0,43/−0,45 % ; long −0,02/−2,52 %, avec contrôles normaux long +22,13/+9,40 %. Le seuil reproduit ≥3 % au-delà de la dérive n'est pas établi. Temps complet long B1 +3,92 %, B2 −0,01 % : absence de régression stable >3 % non démontrée proprement dans ce régime. **Synthèse terminée ; candidat mémoire local conservé tel qu'inspecté, comportement numérique précédemment vérifié, décision de promotion différée jusqu'à conditions stables.** Aucune installation/publication ni nouvelle validation GPU/Kani effectuée ; ancien état de pause ci-dessus conservé comme historique. Le pin CI pytest 8.4.2 reste non revalidé face au test historique en 9.1.1 ; cette recherche documentaire ne le modifie pas.

### OPT-2026-10-03-RUST-AUDIT-11 — nouvelles optimisations générales et Qwen3.6 DFlash2 — recherche terminée

- Demande : « cherches de nouvelles optimisations generales et aussi et surtout a qwen3.6 35b a3b et a son dflash2 ». Recherche et classement d'expériences nouvelles, avec priorité au chemin Qwen/DFlash2 ; préserver les changements locaux et les conclusions négatives.
- Antécédents lus : index complet du journal et entrées pertinentes, rapports decode-roadmap-2026-09-04, general-performance-2026-09-07, audit-runtime-2026-09-04, audits des 22/23 septembre, nouvelles-pistes-performance-2026-09-24, PERF-116 et PERF-134. L'économie d'historique PERF-134 reste un candidat mémoire local, sans boost établi ; ses mesures existantes seront synthétisées sans nouvelle exécution.
- Hypothèse : des mécanismes distincts des variantes déjà rejetées restent dans les transactions récurrentes, le cache draft, les projections exactes, la soumission MLX et les chemins généraux. Rechercher leurs occurrences dans le moteur puis confronter aux sources primaires actuelles (MLX, DFlash/Splash, Qwen et Apple). Une idée publiée ailleurs n'est pas un gain démontré sur ce moteur.
- Baseline : HEAD `fbfb357edffee4494a1be152666cf1ef33e47c9d` plus checkout modifié à l'ouverture ; checkpoint local Qwen3.6-35B-A3B EXL3 2,49 bpw et draft DFlash2. Empreintes des sources et métadonnées à conserver avec le rapport. Environnement présent à relever ; pas d'hypothèse sur le maintien des conditions thermiques/alimentation de septembre.
- Protocole : inspection statique, recherche web explicitement demandée et analyse des preuves existantes uniquement. Pour chaque piste : emplacement réel, nouveauté/antécédent, impact potentiel, risque numérique, première expérience décisive, seuil de décision. Aucun benchmark, chargement modèle, installation ou publication dans cette phase ; toute expérience GPU ou modification exécutable exigera sa propre entrée préalable et les vérifications du skill formal-proof-skill.
- Livrable : `docs/recherche-optimisations-qwen-dflash-2026-10-03.md` et preuves d'inspection sous `docs/measurements/audit-2026-10-03-*`. Gains des nouvelles pistes : **non mesurés**. Intégration : recherche locale, aucune nouvelle optimisation exécutable à ce stade.
- Résultat : [rapport livré](docs/recherche-optimisations-qwen-dflash-2026-10-03.md), **sept pistes** avec emplacement, antécédents, limites numériques et premier protocole : journal compact GDN avec replay récurrent seul ; budget de résidence MLX borné ; consolidation du cache draft par append en gardant les QMM de huit lignes ; nouveaux chemins MLX 0.32.3 en build isolé ; diagnostic Q4/BF16 et adaptation du draft au target EXL3 ; petit arbre de vérification ; propositions copiées puis vérifiées. Priorités : **journal GDN, résidence, cache draft prefill**. Calcul conservateur à T=6 : 300 Mio d'historiques logiques contre 60 Mio d'état initial + 3,54 Mio de journal minimal ; différence **236,46 Mio théorique**, aucune extrapolation au pic MLX, à la RAM ou au débit.
- Sources primaires actuelles : MLX 0.32.3 publié le 29 septembre, DFlash2/Splash, kernels `bstnxbt/dflash-mlx`, Apple Metal et papier CopySpec. [Manifest des révisions/blobs](docs/measurements/audit-2026-10-03-upstream.json). Vérification du dispatch : le nouveau NAX D=256/masque demande ≥1024 queries et ne s'applique pas automatiquement aux chunks 256 ni aux queries 1/8 actuelles. Le budget de résidence est disponible dès MLX 0.32.2 mais absent du bridge Rust inspecté. Les nouveaux kernels GDN upstream n'établissent pas l'identité numérique avec les gates FP32 locales.
- Relecture des traces PERF-125 sans GPU : salut top-2 **10/22** premiers refus, top-4 **15/22**, top-16 **21/22** ; matrice **2/7**, **5/7**, **5/7**. [Rangs et empreintes](docs/measurements/audit-2026-10-03-refusal-ranks.json). Trajectoires warmup/mesure identiques ; ces couvertures ne prédisent ni les continuations de branches ni un gain E2E. Les pistes connues D2/D3/P1, attention multi-queries et diagnostics de slices/factories restent des continuations, pas de nouvelles découvertes ; anciens rejets conservés.
- Vérification documentaire : relectures Python déterministes, validité JSON, neuf liens locaux résolus, calculs de dimensions, **12 empreintes de sources exécutables et quatre métadonnées modèle inchangées**, `git diff --check` réussi. [Inventaire](docs/measurements/audit-2026-10-03-inventory.json), [contrôles](docs/measurements/audit-2026-10-03-checks.json). macOS 27.2 build 26B5091g, Rust 1.98.1, MLX 0.32.2/MLX-LM 0.32.0 ; secteur indiqué mais batterie 50 % en décharge au relevé, thermique non mesuré. CI inspectée, sans modification ni exécution GitHub. Aucun modèle chargé, benchmark relancé, build, test GPU ou preuve Kani exécuté dans cette recherche ; futures variantes et performances **non vérifiées**.
- État final : **recherche validée comme livrable documentaire**, hypothèses de performance **non mesurées**, aucune nouvelle optimisation intégrée. Synthèse PERF-134 terminée ci-dessus, sans promotion du candidat local. Aucun nouvel essai en cours, app non mise à jour et aucune publication effectuée par cette recherche.

### OPT-2026-10-03-RUST-PERF-135 — journal compact GDN avec replay du seul état accepté — en cours

- Autorisation : « vas y test les », après AUDIT-11. Commencer les trois priorités puis appliquer les premiers filtres aux autres pistes du rapport. Nouvelle hypothèse : enregistrer delta FP32, décroissance FP32 exacte et clés préparées FP16 au lieu des historiques complets ; commit partiel par replay récurrent uniquement, commit complet par état final. Distinct de PERF-71 (replay modèle) et PERF-134 (dernière copie seulement). Préserver l'arithmétique et les gates du kernel local.
- Baseline : checkout courant HEAD `fbfb357edffee4494a1be152666cf1ef33e47c9d` modifié, candidat mémoire PERF-134 déjà présent ; checkpoint Qwen3.6-35B-A3B EXL3 2,49 bpw + DFlash2 Q4, MLX 0.32.2, M5 24 Gio/macOS 27.2. Archiver les sources avant modification et le release de référence. Conditions initiales : secteur indiqué, batterie 43 % en décharge ; thermique non mesuré, lecture `ps` refusée par sandbox. Relever les conditions de chaque passe séparément.
- Protocole : capacité/bornes Rust vérifiées par Kani ; test physique déterministe T=1..8 et chaque préfixe, sorties et états comparés en bits à la transition mono-token, rejets T=0/9, puis commits modèle 1..8 et préfixe long 2825 avant/après 48 tokens, 80 états et logits. Si exact : bridge salut/matrice/long, greedy/cache OFF/contexte4096, warmup128 puis deux répétitions mesurées par passage A/B/B/A, un seul modèle à la fois. Mesurer complet/TTFT/decode, compteurs et mémoire ; protocole appliqué séquentiellement, ni prover ni compilation pendant les mesures. Si premier crible régressif ≥3 %, une référence adjacente décide avant élargissement. Seuil vitesse ≥3 % reproduit au-delà de la dérive normale ; mémoire seule retenable si pas de régression stable >3 % du temps complet. Sinon restauration exacte de ce prototype.
- Vérification finale de tout code retenu : fmt, Clippy strict, build, tests ciblés et suite applicable, Kani, CI existante inspectée et réparation si nécessaire. Metal non couvert par Kani ; tests différentiels bornés ≠ preuve universelle. Logs `docs/measurements/perf-135-*`, aucun install/push/publication. Statut **avant prototype**, métriques nouvelles non mesurées.
- Baseline sauvegardée : sources présentes à l'ouverture sous le répertoire indiqué dans `docs/measurements/perf-135-baseline.json` (52 fichiers), release `/tmp/mlxl3-perf135-a`, SHA-256 `a27da41c6122e83951ddfcad6509bdf0f26b57fe5b8b0b2cddc931905c70446b`. Première commande build interrompue immédiatement car `cargo` absent du PATH ; relance par `/Users/justin/.cargo/bin/cargo` **réussie**, sans changement de source. Prototype local en place ; test numérique petite forme et forme Qwen réelle T=1..8/tous préfixes et erreurs 0/9 lancé, résultat en attente.
- Premier contrôle physique **bloqué par la sandbox**, après compilation réussie : MLX « No Metal device available », aucune transition numérique exécutée (`perf-135-tape-exact.log`). Relance du même test avec accès GPU hors sandbox nécessaire ; ce résultat ne rejette pas le mécanisme et n'est pas un échec de parité.
- Contrôle GPU hors sandbox **réussi** (`perf-135-tape-exact-gpu.log`) : petite forme Hk=2/Hv=4/Dv=16 et forme Qwen Hk=16/Hv=32/Dv=128, T=1..8, tous les préfixes ; états FP32 et sorties FP16 égaux en bits à la transition mono-token, entrées T=0/9 et retained invalides rejetées. Replay naïf sans changement de réduction/gates exact sur ces fixtures ; contrôle modèle long et Kani suivants. Temps de test 1,55 s, **pas une mesure de performance**.
- Contrôle modèle long **réussi**, 2 825 tokens avant/après 48 générations, retained=1..6 : logits applicables exacts et **80 états sans premier octet différent**, `perf-135-long-exact.log` (21,91 s de test, pas un benchmark). Contrôle court retained=1..8 et preuve de bornes suivants.
- Commits courts retained=1..8 **réussis** (`perf-135-commit-exact.log`). Kani **réussi**, 66 propriétés sans échec et 5/5 couvertures, `time`/`retained` i32 entièrement symboliques, contrat `gdn_tape_prefix` appelé par le replay de production (`perf-135-kani.log`) ; ne prouve pas Metal. Release candidat construit et archivé `/tmp/mlxl3-perf135-b`. Début des mesures A/B/B/A salut, compilation et prover terminés. Lectures thermique/swap refusées dans la sandbox, à noter dans les conditions ; aucune température inventée.
- Salut A1 terminé, deux répétitions après warmup128, texte/tokens exacts ; résultats bruts `perf-135-salut-a1.{jsonl,stderr}`, médianes dans `perf-135-summary.json`. Résultat de référence individuel, décision après passages B et A final.
- Salut B1 terminé, même hash/blocs/88 sur 144 : decode **2,318523 s** contre A1 **2,362810 s** (−1,87 %), complet **2,652366 s** contre **2,699337 s** ; contrôle normal **2,680568 s** contre **2,713084 s** (−1,20 %). Pic MLX **12 731 434 408** contre **12 999 286 748 octets** (−255,44 Mio). Premier signal mémoire, vitesse sous le seuil et non isolée ; poursuite B2/A2. Conditions B1 relevées après sa passe, batterie 38 % en décharge, à ne pas qualifier de relevé préalable.
- Salut B2 terminé : hash/blocs/acceptation inchangés, decode **2,339101 s**, complet **2,671388 s**, normal **2,667023 s**, pic MLX **12 731 434 408 octets**. Économie d'allocation B1 reproduite exactement ; aucun boost isolé ≥3 % à ce stade. Passage A2 suivant.
- Salut A2 terminé : decode **2,388513 s**, complet **2,720183 s**, normal **2,683913 s**, pic MLX **12 999 290 844 octets** ; hash/blocs/acceptation exacts. Sur les médianes des passages, environ −1,97 % decode B/A, sous le seuil vitesse ; pic économisé ≈255,44 Mio dans chaque B. Élargissement matrice/long pour contrôler les régressions de ce candidat mémoire.
- Campagne complémentaire matrice/long en cours ; une seule passe GPU active.
- long a2 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 2.981531 s, complet 9.260566 s, TTFT 6.279006 s, pic MLX 13436116700 octets ; normal decode 2.787387 s. Preuves `perf-135-long-a2.{jsonl,stderr,conditions.json}`, résumé `perf-135-summary.json` ; résultat individuel.
- long b2 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 2.933665 s, complet 9.185920 s, TTFT 6.252228 s, pic MLX 13210788332 octets ; normal decode 2.774447 s. Preuves `perf-135-long-b2.{jsonl,stderr,conditions.json}`, résumé `perf-135-summary.json` ; résultat individuel.
- long b1 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 2.999197 s, complet 9.427752 s, TTFT 6.428528 s, pic MLX 13210788332 octets ; normal decode 2.795001 s. Preuves `perf-135-long-b1.{jsonl,stderr,conditions.json}`, résumé `perf-135-summary.json` ; résultat individuel.
- long a1 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 3.002531 s, complet 9.269237 s, TTFT 6.266681 s, pic MLX 13436120796 octets ; normal decode 3.161025 s. Preuves `perf-135-long-a1.{jsonl,stderr,conditions.json}`, résumé `perf-135-summary.json` ; résultat individuel.
- Correction du lancement long/A1 : chemin du prompt erroné (`perf-134-long-prompt.txt` absent), arrêt du script **avant tout chargement modèle**, trace conservée `perf-135-long-a1-prompt-error.stderr`. Le vrai protocole historique utilise `docs/audit-runtime-2026-09-04.md` (2 825 tokens) ; relance avec ce fichier. Aucun résultat numérique ni temps d'inférence dans l'essai échoué.
- perf-135-long-a1 échoué, code retour 1 ; voir logs, décision différée avant nouvelle passe.
- matrice a2 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 1.776901 s, complet 1.933951 s, TTFT 0.157013 s, pic MLX 13000023508 octets ; normal decode 2.754258 s. Preuves `perf-135-matrice-a2.{jsonl,stderr,conditions.json}`, médianes dans `perf-135-summary.json` ; résultat individuel.
- matrice b2 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 1.714788 s, complet 1.873288 s, TTFT 0.158462 s, pic MLX 12732326296 octets ; normal decode 2.680063 s. Preuves `perf-135-matrice-b2.{jsonl,stderr,conditions.json}`, médianes dans `perf-135-summary.json` ; résultat individuel.
- matrice b1 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 1.714935 s, complet 1.882418 s, TTFT 0.167447 s, pic MLX 12732342704 octets ; normal decode 2.690629 s. Preuves `perf-135-matrice-b1.{jsonl,stderr,conditions.json}`, médianes dans `perf-135-summary.json` ; résultat individuel.
- matrice a1 terminé, deux répétitions après warmup128, hash/blocs/acceptation constants : decode 1.761352 s, complet 1.922036 s, TTFT 0.160649 s, pic MLX 12999998980 octets ; normal decode 2.662096 s. Preuves `perf-135-matrice-a1.{jsonl,stderr,conditions.json}`, médianes dans `perf-135-summary.json` ; résultat individuel.
- Campagne **terminée** : 12 passages A/B/B/A, 24 répétitions DFlash et 24 contrôles normaux, mêmes hashes/blocs/acceptations sur les trois prompts. [Décision détaillée](docs/measurements/perf-135-decision.json), [médianes et empreintes](docs/measurements/perf-135-summary.json). Candidat **validé comme économie de pic d'allocation MLX locale**, sans boost général démontré ≥3 % et sans régression stable >3 % du temps complet dans les deux passages B de ces cas. Économie environ 255 Mio court/matrice et 215 Mio long ; ne pas l'additionner à PERF-134 ni la présenter comme RAM processus. Prototype conservé pour les vérifications finales et base des expériences suivantes ; app non installée/non publiée.

- Révision du 2026-10-05, reprise après interruption : **aucun code ni benchmark PERF-136 lancé**. Les lignes matrice/long ont été réclassées dans PERF-135 : leur ancien emplacement dans PERF-136 était une erreur de rédaction, sans essai de résidence. PERF-135 reste une économie d'allocation mesurée, pas une accélération générale démontrée.
- Validation complémentaire PERF-135 préenregistrée : le dépôt est désormais au HEAD `0d89e01` (v1.2.0), avec le journal/replay GDN conservé et de nouveaux chemins MTP ; le release courant diffère du binaire mesuré le 3 octobre. Répéter les contrôles de format, Clippy, build, suite Rust et contrats Kani sur ce HEAD, puis les tests GPU ciblés du replay et des commits courts/longs pour vérifier la compatibilité avec les changements intervenus. **Aucun benchmark de performance répété**, les chiffres ci-dessus restent attachés aux anciens binaires archivés. Inspecter la CI et son run si accessible. État : contrôles en cours ; pas de code exécutable modifié pour cette reprise, pas d'installation/publication par cette tâche.

### OPT-2026-10-03-RUST-PERF-136 — budget de résidence MLX borné — avant code, non testé

- Nouvelle hypothèse AUDIT-11 : le bridge Rust ne demande aucun budget de résidence, alors que MLX 0.32.2 le supporte. Tester un budget borné par `recommendedMaxWorkingSetSize` et la mémoire totale, lié à la durée de vie du modèle ; restaurer le précédent sur drop/erreur. Ne changer aucune limite système ni arithmétique. Baseline : candidat mémoire PERF-135, sorties exactes et mesures ci-dessus ; archiver ce release avant ajout. MLX 0.32.2, M5 24 Gio/macOS 27.2, power relevé à chaque passe, températures non mesurées.
- Protocole : wrapper FFI setter minimal, budget calculé avec arithmétique Rust vérifiée Kani, erreurs/limites et restauration testées sur GPU. Comparer budget 0 / budget retenu / budget retenu / 0 sur salut et long si signal, warmup128 puis 2 répétitions, greedy/cache OFF/contexte4096, un seul modèle à la fois ; sorties/compteurs exacts, complet/TTFT/decode/pic/empreinte. Première réponse et reprise après inactivité séparées si utiles. Budget n'alloue pas de mémoire par lui-même. Retenir une accélération seulement ≥3 % reproduite au-delà des contrôles normaux ; aucun benchmark commencé avant tests du wrapper. Logs `perf-136-*`, jamais de sysctl modifié, aucun install/push/publication. Statut **avant code**.
