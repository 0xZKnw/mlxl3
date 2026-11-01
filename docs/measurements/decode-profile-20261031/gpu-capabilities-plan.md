# DECODE-PROFILE-08 — capacités GPU et protocole proposé

Inventaire en lecture seule du31 octobre2026, sous `DECODE-PROFILE-08-PLAN/TOOLS-01`. Aucun enregistrement, modèle, benchmark, compilation ou preuve lancé par cet agent. Aucun changement de `xcode-select`, installation ou code moteur. Le root coordonne séparément le téléchargement MetalToolchain.

## Capacités effectivement observées

- `xcode-select -p` = `/Library/Developer/CommandLineTools`. Dans cet état, `xcrun xctrace` échoue. Avec **`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`** : **xctrace27.2(27B52)** ; appareil local `MacBook Pro de Justin`, UDID `163033D7-EA71-571F-B657-F2F01CC65CEC`.
- Templates listés : Metal System Trace, Time Profiler, System Trace, CPU Profiler, CPU Counters, Processor Trace notamment. Instruments listés : GPU, Metal Application, Metal GPU Counters, Metal Resource Events, Thread Activity. Présence du logiciel ≠ compteur matériel qualifié.
- `record --show-recording-options` est explicitement documenté comme **sans enregistrement et sans cible obligatoire**. Sur Metal System Trace, options Time Profiler : `contextSwitchSampling=false`, `highFrequencySampling=false`, `recordKernelStacks=false`, `recordWaitingThreads=false`. Sur CPU Profiler : `highFrequency=false`, `recordKernelCallstacks=false`. Processor Trace expose buffers/ring/decode, mais son support matériel n’a pas été vérifié.
- Métadonnées `--instrument 'Metal GPU Counters'` et `--instrument 'Metal Application'` : **`{}`**. Elles ne donnent ni sélection CLI de counterset ni garantie de données non vides. Le SDK expose `MTLDevice.counterSets` et `supportsCounterSampling`, avec jeux Timestamp/StageUtilization/Statistic déclarés ; **les jeux réellement exposés par ce M5 n’ont pas été interrogés**. Aucun buffer de compteurs n’a été créé.
- Au moment de l’inventaire : `xcodebuild -showComponent MetalToolchain -json` → **build27B47,`status=uninstalled`** ; `xcrun metal --version` refuse explicitement faute du composant. Help supporte `-downloadComponent MetalToolchain`, `-showComponent MetalToolchain -json`, `-importComponent` et `-deleteComponent`. Le root a ensuite préenregistré son installation ; ne pas confondre cette observation initiale avec son état final. Aucun téléchargement supplémentaire démarré ici.

Les labels internes MLX sont conditionnés à **`#ifdef MLX_METAL_DEBUG`** dans le header installé `mlx/backend/metal/utils.h:27–45` : un environnement `MLX_METAL_DEBUG=1` ne recompile pas le wheel. Ces labels concatènent des noms sur un **command-buffer**, pas une association exclusive kernel→couche. Les noms de pipelines custom demeurent utiles indépendamment de ce macro.

## Témoin physique court recommandé après installation

**Réutiliser un test existant sans modèle**, sélection exacte :

```text
/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/build/issue59-t18a-20261031/libtest-r2/prototype-engine
weighted_epilogue::capture_tests::captures_real_legacy_and_fused_epilogue_commands
--exact --ignored --nocapture --test-threads=1
```

Binaire de l’identité T18a r3 : SHA256 `de3616e3d2e0c3d9507e075cf9af4dfe8b3290f0b291ac704b3e1df537f8d910` ; le revérifier avant lancement. Env avant processus : `MTL_CAPTURE_ENABLED=1`, `MLXL3_WEIGHTED_CAPTURE_OUTPUT=<nouveau dossier absolu existant et vide>`. Le test refuse d’écraser `legacy.gputrace`/`fused.gputrace`. Source actuelle : `native/src/weighted_epilogue/capture_tests.rs:8–109`.

Il chauffe stock/fusion, capture deux graphes frais et vérifie **25600 F32/102400octets finis, bitexact**, hors capture. Géométrie Hot10/Cold5/ordre10/largeur2560 ; aucune banque modèle chargée. Il démarre la capture **puis synchronise pour faire tourner le buffer MLX préexistant** : cette rotation était nécessaire au T05-v2 et doit rester. Valeurs et entrées sont déjà évaluées avant capture.

**Nouvelle propriété à qualifier**, distincte du simple recontrôle numérique : après installation, ouvrir/rejouer puis **profiler** les deux captures dans Xcode, relever les noms, timestamps et countersets effectivement disponibles. Référence historique T05-v2 : legacy5dispatchs, fusion1,2CB/1encodeur compute chacun ; vérifier les vrais commandes/cadres, sans déduire ces nombres du compteur de requêtes Rust. Des captures peuvent contenir un CB vide de synchronisation : ne pas exiger un temps non nul de celui-ci.

Acceptation du témoin :1test/0échec/0ignoré, sorties non vides conformes, pipeline custom identifié, au moins un intervalle GPU **physiquement mesuré, fini et strictement positif** sur les dispatchs actifs ; unité et mode de collecte conservés. Pour chaque counterset : noms + unité + données non vides associées au workload, ou **non qualifié**. Capturer la table source et les refus, pas seulement un total UI. Une timeline, un nom et un `0ns` ne valident pas le temps.

Comparaison à [T04 ancien](../issue59-t04-20261031/xcode-dispatch-inspection.json) :3→2dispatchs connus,2CB/1compute, **`gpu_time_measured=false`**,0ns affiché après replay, pas de profil GPU après replay ; outil shader optionnel alors absent. Le nouveau témoin vise précisément cette lacune. L’installation ne garantit pas qu’elle la corrige : si les temps/counters restent nuls ou vides, garder cet échec et arrêter l’escalade vers le modèle pour cette propriété.

Si le sampling Instruments est trop bref pour cette petite épilogue, cela ne prouve pas l’absence de GPU. Second témoin possible **à préenregistrer séparément**, toujours sans modèle : `array::tests::readback_profiling_preserves_layouts_errors_and_overlap --exact --ignored --nocapture --test-threads=1`. Il exerce39cas GPU/3CPU-F64 et une disponibilité retardée, avec marqueur `ULTRA_READBACK ... delayed_overlap=true ... parity=byte_exact` ; utile pour un signal plus long. Ne pas répéter des captures minuscules indéfiniment ni transformer une table vide en succès.

Proposition de gardes pour le premier témoin : secteur/quiet, un seul processus GPU,180s maximum, admission légère2Go, même ancre explicite0Swapout/Swapin≤256Mio et sortie incomplète sur interruption ; captures cumulées≤1Gio. Le root doit figer ce protocole/identité et la disponibilité réelle avant exécution. Aucun de ces lancements n’a été effectué ici.

## Mesurer ensuite les projections avec les bons buffers

Le SDK `MTLCommandBuffer.h:286–294` indique que GPUStartTime vaut0 avant démarrage et GPUEndTime0 avant notification de complétion. Lire les valeurs sur un CB terminé et retenu est donc nécessaire. Dans MLX0.32.2, `gpu::finalize` termine/commit puis remplace le buffer ; **prendre `get_command_buffer()` après `async_eval` récupère le nouveau buffer vide**.

Pour une projection isolée : évaluer inputs/poids et synchroniser en dehors de l’intervalle ; retenir le CB courant **avant** la sortie fraîche/`async_eval` ; mesurer construction et soumission hôte séparément ; lire les timestamps du CB retenu après sa complétion. Garder buffers/pipelines vivants et préserver erreurs/cleanup. Cibles proposées par [gpu-priority](../post-t01-priority-20261031/gpu-priority.md) : out6144→2560K4/MUL1 GDN layer0 et QSA layer3, puis QKV2560→10240K4/MUL1 layer0.

**Un CB retenu ne couvre pas nécessairement la projection.** Le seuil40ops/40MB peut déclencher des commits internes ; le graphe peut également inclure des ancêtres. Vérifier tous CB/encodeurs/dispatchs de la fenêtre. Si plusieurs buffers existent, calculer l’union des intervalles pertinents depuis la trace ; ne pas prendre uniquement le premier ni sommer des enveloppes qui se recouvrent. Une hypothèse mono-buffer non vérifiée invalide le total. Ne pas changer silencieusement les seuils pour obtenir une capture commode.

La durée du buffer inclut ordonnancement/fences/stalls internes : elle n’est pas un temps ALU exclusif. Le profilage isolé, avec seulement cette chaîne en vol, permet une attribution **à la chaîne** plus forte que le temps fenced modèle ; il ne prédit pas le gain en décode naturel. Les anciennes projections synchronisées ne fournissent pas ces temps physiques.

## Reprendre les12ms avec une attribution honnête

Après témoin qualifié : une courte acquisition AR naturelle sur la baseline, sans fences supplémentaires, avec contrôles hors Instruments avant/après. Sélectionner Metal System Trace et Thread Activity ; ajouter Metal Application/Metal GPU Counters seulement selon le témoin. Le mode `--show-recording-options` a accepté cette combinaison sans enregistrement. Options CPU supplémentaires reconnues ci-dessus ; commencer court, exporter la cadence réellement observée, ne pas supposer que highFrequency signifie une fréquence donnée. Processor Trace/CPU Profiler restent des recours conditionnels, pas des prérequis déjà qualifiés.

Garder les deux soumissions précoces. Marquer début/fin/rôle/couche sur le thread hôte, conserver les horloges et refaire le calage `time-info`/timebase ; rattacher les interruptions comme dans l’analyse précédente. Calculer CPU-running∩GPU-idle∩soumission à partir des intervalles, puis utiliser les piles pour localiser hash/lookup, wrappers/traversée, allocations, bindings/barrières et commits. **Les samples, même plus fréquents, ne deviennent pas des ms exclusives par fonction.**

Un CB peut regrouper Cold précédent, HC, GDN/QSA et routeur ; son label unique ou son heure de commit ne suffit donc pas pour facturer tout son temps à une couche. Pour une identité exclusive primitive→dispatch, il faudra soit les événements Metal qualifiés, soit un ledger diagnostique MLX explicitement préenregistré et validé. Ne pas prétendre obtenir cette correspondance depuis un nom de shader partagé entre couches.

En priorité : qualifier le témoin et les timestamps, mesurer une chaîne dense isolée, puis seulement acquérir la courte fenêtre naturelle. **Pas de reconstruction complète MLX pour activer un macro de label à ce stade ; pas de modèle complet sous un capteur encore vide.** TurboEXL3 est volontairement hors de cette tâche et sera consulté après les nouvelles mesures, selon la demande utilisateur.
