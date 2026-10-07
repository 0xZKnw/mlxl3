# v1.4.2 — recherche supplémentaire vers 80–85 tok/s

Campagne du 7 octobre 2026 pour la [PR draft #26](https://github.com/0xZKnw/mlxl3/pull/26). **Aucun des quatre nouveaux prototypes ne satisfait le seuil de conservation préenregistré. Ils sont tous retirés, avec leurs tests provisoires.** Le contrôle d'une configuration MTP1 existante dérive fortement et est écarté. Aucun nouveau +10% ni débit stable de 80–85 tok/s n'est validé.

Le code exécutable conservé reste celui de `b389fcce8b6ae1624595dd832a2d174a61d57025`, déjà publié dans la tête `37be560b3afcd3b8d0b19aca2de08276c2647522`. Ses mesures confirmées de phase3 restent **75–77 tok/s sur le prompt code avec tête compacte/MTP2**, et 69–70 avec tête stock. Ce sont les conditions du [rapport pipeline](engine-v1.4.2-pipeline.md), pas une garantie tous prompts ou tous modèles. Cette phase ajoute seulement le journal et les preuves. Aucun merge, release ni remplacement de l'app installée.

## Quatre essais, tous retirés

Chaque essai a son préenregistrement dans [opti.md](../opti.md), avant prototype ou mesure. Le seuil était un gain code d'au moins 3%, confirmé en ordres opposés, avec tous les contrôles decode à moins de 3% de dérive. Les gains de comparaisons distinctes ne s'additionnent pas.

| Essai / ordre | FR decode | Code decode | Médianes code A → B | Décision |
| --- | ---: | ---: | ---: | --- |
| Réparation KV asynchrone / ABBA | +0,94% | +1,27% | 78,08 → 79,07 tok/s | Sous seuil ; pas de BAAB |
| Gate/up EXL3 NT4 pour 48 lignes / ABBA | +1,89% | +3,46% | 75,96 → 78,59 tok/s | Signal initial, confirmation nécessaire |
| Même gate/up / BAAB | +1,88% | +1,80% | 78,42 → 79,83 tok/s | Seuil non confirmé ; retrait |
| Réparation KV + gate/up / ABBA | +3,81% | +2,41% | 76,83 → 78,67 tok/s | Code sous seuil ; pas de BAAB |
| Draft GPU → vérificateur avec pipeline / ABBA | +0,93% | +1,39% | 78,06 → 79,14 tok/s | Sous seuil ; pas de BAAB |

Toutes ces comparaisons ont des sorties et hashes exacts, non vides. Les contrôles decode sont ≤0,87% pour la réparation, ≤2,91% puis ≤0,44% pour gate/up, ≤1,21% pour la combinaison et ≤1,12% pour la liaison GPU. Le passage gate/up à 80,009 tok/s ne constitue pas un débit garanti ; sa confirmation globale reste sous seuil. Les latences complètes code diminuent respectivement de 1,14%, 3,27% puis 1,62%, 2,22% et 1,22%, sur les mêmes fenêtres.

La réparation réutilise `Array::async_eval_all` pour soumettre ensemble les deux graphes KV acceptés, plutôt que deux évaluations synchrones. Elle ne modifie ni acceptation ni calcul numérique. La source conceptuelle est [TensorFold au SHA cb2ebf054](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen3_5_moe/family.py), absorption des KV suivie d'une soumission asynchrone. L'[API officielle MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html) confirme le mécanisme ; la version locale reste 0.32.2.

Le gate/up cible une forme précise : trois tokens de vérification × top8 × deux projections = 48 lignes, H2048 → 512, codebooks K2/K3 sur M5. Le NT4 existant démarrait à 64 lignes. Le prototype ne touche pas le down ni le shader. C'est distinct du down48 historique PERF76 et du NT4 de la grande projection de vocabulaire, déjà rejetés. La combinaison a été mesurée directement, sans addition des gains individuels.

La liaison GPU réévalue explicitement **MTP07**, rejeté sur l'ancienne v1.3 sans pipeline, notamment en D3. La raison nouvelle est le pipeline cible de phase3 : son premier groupe peut démarrer le draft GPU pendant que le CPU construit la suite. Un type interne fermé limite les IDs aux sorties argmax/remap du head validé ; lecture CPU différée après vérification, lookup conservé sur le chemin host. La qualité passe, mais le gain réel reste insuffisant. L'ancien rejet n'est pas effacé.

## Protocole et identité

Apple M5 Air 10 CPU / 10 GPU, 24 Gio, macOS 27.2, MLX 0.32.2 / MLX-LM 0.32.0. L'utilisateur a branché le Mac avant les nouvelles mesures. Toutes les campagnes ci-dessus sont sur secteur, batterie en charge puis 100%. `pmset` ne publie aucune alerte thermique ; températures et fréquences ne sont pas mesurées. Aucune autre inférence, compilation ou preuve n'est lancée simultanément par la campagne.

Qwen3.6-35B-A3B EXL3 2,49 bpw, tête MTP affine4 stock pour les contrôles concernés et compacte pour les timings : 79 591 IDs logiques / 79 616 lignes paddées, projection+map 87,77 MiB supplémentaires. Même binaire et même head dans chaque paire, seuls les opt-in changent. MTP2, greedy, contexte 4096, cache OFF, warmup au budget128 puis deux générations mesurées au budget128 par prompt et passage, pause2s, quatre passages ABBA ou BAAB. Chaque génération mesurée a 128 tokens générés, dont 127 comptés dans le decode ; prompts FR27 / code44 tokens. Le runner existant enregistre texte, hashes d'IDs, nombres de tokens, allocations MLX, empreinte processus, conditions et runtime.

Les identités exactes des prototypes sont dans `*-identity.json`, leurs sources dans l'archive. Le binaire final propre de référence est `build/engine-v1.4.2/async-repair-before/mlxl3-rs`, SHA256 `5d36ec02714c9091ca49d8219be4dbacf3f3ca51344180baefb076d55e1a6049`, runtime 37be/1.4.2/release. `phase-5-finalization.json` relève checkpoints/head par chemins résolus, tailles et mtimes, SHA256 des JSON et versions ; les grands fichiers de poids ne sont pas rehachés intégralement. Aucun poids ou vocabulaire amont n'est empaqueté.

## Configuration MTP1 : contrôle dérivé, écarté

Les débits de Tune à 96 tokens étaient seulement des diagnostics, sur d'autres prompts et parfois des prototypes retirés. Pour vérifier la configuration **existante** MTP1 avec le pipeline final, le runner inchangé a comparé A et B identiques : même binaire propre, même tête compacte, profondeur1 et mêmes options, protocole128 décrit ci-dessus. A/B n'est donc pas un comparatif d'optimisations.

Malgré la parité exacte, la dérive atteint **−14,94% pour FR A et +28,13% pour code A**, au-delà de 3%. Les médianes brutes code70,33 / 74,51 tok/s et l'écart A/B apparent+5,94% ne démontrent ni un gain ni une vitesse reproductible. Ces fenêtres ne prouvent pas non plus une régression D1 contre D2. Cause inconnue. **Arrêt immédiat : aucun BAAB, D3 ou essai long supplémentaire**, conformément à la demande d'écarter les essais qui dérivent.

## Correction, preuves et restauration

Avant chaque chronomètre : contrôles physiques explicitement sélectionnés, sans filtre vide, depuis snapshots et oracles target mono-token. Récursion stock/compact D1..3 sur24 blocs, 80 états cible, hidden et KV exacts ; adversaires acceptation/refus/réparation ; rollback T1..8/tous les préfixes conservés. Le nouveau chemin GPU passe aussi 27 prefixes retenus sur prefills23/24/257, formes et finitude explicites des logits/raw/hidden/états, erreurs vocabulaire/ancre/largeur/contexte et reset. Le helper KV renforcé contrôle deux tableaux non vides Float16 de mêmes formes, rang4 et finitude.

Chaque prototype passe le bridge compact avec Tune : 43 requêtes terminales uniques, 35 completions, cinq annulations, deux erreurs attendues et un Tune ; budgets, préfixe ≥256, récupération et hashes vérifiés. Après retrait de tous les prototypes, le **binaire final propre** repasse également ces 43 requêtes en51,84s, avec les quatre assertions finales vraies. Ces durées de tests ne sont pas des benchmarks.

Kani vérifie l'implémentation CPU de la garde gate/up sur domaines scalaires complets : une assertion et quatre covers réussis, sans hypothèse. La garde device/rangs0..3/vocabulaires i32 complets passe191 obligations et cinq covers, sans hypothèse ni boucle ; après correction de lint, le harnais affecté repasse. Ce sont des vérifications bornées CPU. La tentative réelle Array/MLX échoue par ICE `intrinsics.rs:243` avant obligations : **MLX/Metal/FFI/concurrence non formellement vérifiés**. Aucun stub ajouté. La suite Kani39/39 du code conservé reste celle de phase3 et de la CI37be ; elle n'est pas relancée pour des changements uniquement documentaires.

Mutations détectées : mauvais hidden pour réparer les KV par le test GPU, garde gate/up omettant les deux projections par le test CPU, garde device omettant l'égalité des vocabulaires par le test CPU. Chaque source est restaurée puis le contrôle affecté repasse. Les logs conservent aussi un filtre zéro test rejeté, les premiers échecs de format/Clippy, la commande `cargo` initialement absente du PATH et le parseur externe trop strict corrigé sans refaire l'inférence. Une restauration `copy2` avait rétabli un ancien mtime et réutilisé le binaire mutant ; SHA vérifié et recompilation forcée **avant** toute mesure ou validation de restauration. Aucun de ces échecs n'est effacé.

Format, Clippy strict MLX/chat et builds release des prototypes passent après corrections. Le retrait final restaure byte pour byte les cinq sources concernées à37be/b389, tests compris, et fmtcheck passe. Sources/binaire/prototypes sont distingués dans les manifestes ; aucun test expérimental ne devient une preuve du code finalement conservé. Les suites complètes et huit jobs CI réussis de37be concernent ce code inchangé. La CI du nouveau commit documentaire est suivie séparément dans la PR.

Les [notes officielles MLX0.32.3](https://github.com/ml-explore/mlx/releases/tag/v0.32.3) ont aussi été examinées : les spécialisations GQA12/16, M5 Ultra et prefill à masque array ne correspondent pas au parcours testé (ratio8, M5 Air, masque causal). La nouvelle voie GatedDelta et les changements de précision exigeraient des contrôles numériques distincts. Aucune mise à jour de dépendance ni performance0.32.3 n'est mesurée ou recommandée ici.

Les résultats négatifs, commandes, sources prototypes, mutations et restaurations sont conservés dans [l'archive phase5](measurements/engine-v1.4.2/phase-5-raw.tar.gz), avec [manifeste vérifié](measurements/engine-v1.4.2/phase-5-manifest.json). Les quatre archives précédentes restent inchangées. Toutes les entrées de cette campagne sont fermées dans [opti.md](../opti.md).
