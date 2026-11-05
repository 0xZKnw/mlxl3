# P08 AR r2 : plan de lecture de la capture récupérée

31 octobre 2026. Revue des sources et du TOC uniquement ; **aucun export,
test, build, proveur, GPU ou modèle lancé**. Préenregistrements relus :
`DECODE-PROFILE-08-NATURAL-01`, `CONTROLLER-FIX03` et `STOP-WITNESS-01`.
Ne pas modifier les anciennes analyses ni les 551 sources de l'identité r2.

## Statut et provenance

La capture `natural-r2/AR-combined-MAIN-plan-capture-1.trace` est récupérable,
mais le workflow reste **interrompu au délai de jointure de 60 s**. Le résultat
du contrôleur conserve `failed_or_interrupted`, `parity:false`, le PID moteur
20329 et le PID collecteur 20450, tous deux joints ; zéro nouveau Swapin/out.
Cinq résumés natifs sont fermés, sans `control_after`, `final-snapshot.json`
ni `complete.json`. Cela ne constitue pas un test natif complet réussi.

`natural-r2/AR-combined-toc.xml` décrit un run 1 de 16 s, windowed 8 s,
Time Profiler haute fréquence, piles kernel et échantillonnage de commutations.
La présence d'un schéma dans le TOC ne prouve ni lignes présentes ni couverture.
Le `return-exit-status=0` du processus dans le TOC ne remplace pas le résultat
du contrôleur ni le footer natif absent.

Identité source : état local 7978fceac30d991a8c06d72114fc932b2443f555 modifié,
`natural-r2/identity.json`. Binaire réellement visé :
`build/decode-profile-20261031/projection-01/libtest/prototype-engine`, SHA-256
`f7ad8a297d244b5cc9a97c8c0f85e20e818ae6acc78a7586d4dee3caf5340e3e`.
Ne pas réutiliser l'UUID de l'ancien libtest de3616e ou les anciennes adresses
Rust ; vérifier le nouvel UUID/image/load-address avant symbolisation. Le
store de symboles MLX F949D357-CE50-3CDC-BD53-D49566E9F72C est présent dans
cette capture ; cela ne prouve pas à lui seul la résolution de chaque frame.

## Sélecteurs exacts proposés

Préfixe commun : `/trace-toc/run[@number="1"]/data/`.
Chaque sélection doit produire **un seul nœud/schema** ; enregistrer commande,
exit, stderr, SHA du XML, attributs du schéma, nombre de lignes et bornes de
temps. Ces XPath sont dérivés du TOC lu ; leur exécution n'est pas encore
vérifiée. Exécuter les exports séquentiellement après le témoin CPU en cours,
avec timeout et jointure, sans nouvelle capture.

| Fichier futur | XPath après le préfixe commun |
| --- | --- |
| `time-info.xml` | `table[@schema="time-info"]` |
| `thread-info.xml` | `table[@schema="thread-info"]` |
| `process-info.xml` | `table[@schema="process-info"]` |
| `thread-state.xml` | `table[@schema="thread-state"]` |
| `context-switch.xml` | `table[@schema="context-switch"]` |
| `context-switch-sample.xml` | `table[@schema="context-switch-sample" and @callstack="kernel user"]` |
| `syscall.xml` | `table[@schema="syscall"]` |
| `syscall-name-map.xml` | `table[@schema="syscall-name-map"]` |
| `time-sample-100us-user-kernel.xml` | `table[@schema="time-sample" and @sample-rate-micro-seconds="100" and @callstack="user kernel"]` |
| `time-sample-1000us-user.xml` | `table[@schema="time-sample" and @sample-rate-micro-seconds="1000" and @callstack="user"]` |
| `time-profile-high-switch-kernel.xml` | `table[@schema="time-profile" and @high-frequency-sampling="1" and @context-switch-sampling="1" and @needs-kernel-callstack="1"]` |
| `time-profile-standard-user.xml` | `table[@schema="time-profile" and @high-frequency-sampling="0" and @context-switch-sampling="0" and @needs-kernel-callstack="0"]` |
| `metal-gpu-intervals.xml` | `table[@schema="metal-gpu-intervals"]` |
| `metal-gpu-execution-points.xml` | `table[@schema="metal-gpu-execution-points"]` |
| `metal-gpu-submission-to-command-buffer-id.xml` | `table[@schema="metal-gpu-submission-to-command-buffer-id"]` |
| `metal-command-buffer-completed.xml` | `table[@schema="metal-command-buffer-completed"]` |
| `metal-object-label.xml` | `table[@schema="metal-object-label"]` |
| `metal-application-encoders-list.xml` | `table[@schema="metal-application-encoders-list"]` |
| `metal-shader-profiler-intervals.xml` | `table[@schema="metal-shader-profiler-intervals"]` |
| `gpu-shader-profiler-interval.xml` | `table[@schema="gpu-shader-profiler-interval"]` |
| `metal-shader-profiler-shader-list.xml` | `table[@schema="metal-shader-profiler-shader-list"]` |

Ordre minimal : temps/identités/états/syscalls, puis GPU intervals, puis les
deux tables `time-profile`. Les deux tables brutes `time-sample` et les piles
de commutation qualifient la provenance des échantillons avant toute agrégation.
Labels/shaders sont une seconde étape ; ne pas exporter tout kdebug ni tout
le TOC de données dans un seul XML. Un schéma optionnel vide reste « non
disponible » ; une table indispensable vide interdit son attribution.

Variante minimale retenable après inspection du XML : exporter seulement le
profil haute fréquence vers le nom attendu `time-profile.xml` dans un dossier
d'analyse neuf, et le profil basse fréquence dans un autre fichier. Cela permet
de réutiliser le lecteur historique inchangé **si** chaque export possède bien
un unique schéma compatible. Les ID/ref sont locaux à chaque fichier : ne
jamais partager le dictionnaire de références entre les deux exports. Les
contrôles de provenance des samples, de couverture >=16 et de sélection des
40 premiers pas restent nécessaires séparément. Aucun nouveau parser ni
changement de l'analyse historique avant d'avoir constaté les schémas réels.

Exemple de commande à préparer, **non exécutée par cette revue** :

```sh
/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace export \
  --input /Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/decode-profile-20261031/natural-r2/AR-combined-MAIN-plan-capture-1.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile" and @high-frequency-sampling="1" and @context-switch-sampling="1" and @needs-kernel-callstack="1"]' \
  --output /Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/decode-profile-20261031/natural-r2/exports/time-profile-high-switch-kernel.xml
```

## Hypothèses historiques à adapter ou contrôler séparément

Sources relues : `ultra-profile-20261027/attribution/analyze.py` et
`attribution/short/async-stack-analysis-source.txt`. Leur réutilisation peut
éviter du nouveau code, mais ne valide pas à elle seule les hypothèses suivantes.

1. **Tables séparées.** L'ancien `table()` choisit le premier `.//schema`
   puis tous les `.//row`. Il faut refuser plusieurs nœuds/schémas et lire
   séparément les deux profils CPU. Conserver les colonnes et types réels,
   les références XML typées et leurs définitions ; aucune concaténation des
   deux séries 1 ms / 0,1 ms, aucun dénominateur partagé implicite. Les samples
   de commutation ne deviennent pas des samples périodiques parce qu'ils
   portent un poids. Si le schéma ne distingue pas les déclencheurs, rapporter
   la limite et s'appuyer sur les tables brutes séparées pour cette distinction.
2. **Identités.** Filtrer le PID 20329, résoudre les TID via `thread-info`
   et valider l'unique thread du harnais, sans supposer l'ancien TID ni prendre
   un ID de référence XML pour un TID. Conserver les autres threads moteur,
   workers GCD et callbacks MLX séparément. `worker` natif est un index 0..3,
   pas un TID ; ne pas lui attribuer une identité OS fixe entre les lots.
3. **Horloges indépendantes.** Lire le schéma `time-info` réel et ses unités ;
   ne pas utiliser la date TOC ni assimiler directement mach ticks à des ns.
   Construire le décalage initial à partir de `decode_epoch` et de l'origine
   native du fichier `trace_1-decode.json`, puis le contraindre par les seuls
   appels preadv/jobs indépendants. Vérifier le nom via `syscall-name-map`,
   PID/TID, errno, bytes retournés, fd/offset quand présents et cardinalité.
   Réutiliser 1 248 012 B uniquement après vérification des tailles effectives
   des jobs ; éviter un appariement temporel ambigu entre couches proches.
   Aucun calage par corrélation GPU ou par recherche du meilleur résultat.
4. **Calage borné.** Les timings par job natifs sont relatifs au lot, leurs
   champs `job/worker/start/end/bytes/failed` ne sont pas des timestamps OS.
   Transformer les offsets avant comparaison. Pour chaque appariement assuré,
   calculer les contraintes de containment, leur intersection et l'incertitude
   résiduelle ; conserver les échecs/ambigus. Une tolérance 30 µs historique ne
   doit pas devenir une égalité ni être élargie pour forcer un résultat.
5. **Couverture et sélection.** Après calage, identifier toutes les transitions
   complètes dont l'index est >=16 dans la couverture commune réellement
   exportée. Prendre les 40 premières si disponibles, minimum 16 pour attribution,
   sans retirer les lentes ; publier la liste entière, les marges, les phases
   et les effectifs. L'ancien seuil `n > 5` est insuffisant. Les bornes du
   premier/dernier syscall ou GPU ne prouvent pas une collecte continue des
   autres tables ; signaler les trous/pertes et rejeter les bords incertains.
6. **Après calage, refaire la sélection.** L'ancien script choisit les pas
   avant de corriger l'offset. Recalculer après correction les pas et tous les
   sous-intervalles, vérifier leurs ancêtres et les 48 couches, conserver le
   drain terminal hors `output_step`. Pas de nouveau profil pour remplir un
   quota de pas absent de la trace récupérée.
7. **États CPU et GPU.** Mesurer des unions d'intervalles demi-ouverts, Running,
   Runnable, Blocked, Interrupted et résidu inconnus par TID. Séparer GPU du
   PID moteur, de tous propriétaires et propriétaire inconnu ; CPU actif et
   GPU actif se recouvrent. GPU actif signifie intervalle de commande, pas
   ALU utile. Les labels/shaders peuvent rester vides malgré le témoin Xcode
   positif : ne pas transférer ses latences replay au budget du modèle.
8. **Fonctions et piles.** L'ancien script mappe un sample Interrupted au ns
   précédent le début d'interruption si Running. Garder cette attribution
   dérivée distincte des points Running directs ; avec context-switch-sample,
   vérifier la nature de la pile avant d'appliquer cette règle. Rapporter
   comptes, poids, fréquences observées et piles inconnues séparément. Une
   pile kernel d'interruption n'identifie pas automatiquement la fonction
   utilisateur qui a causé l'interruption. Les parts par fonction sont
   statistiques, pas des durées exclusives exactes.
9. **Deux async.** Conserver les deux rôles par parent
   `gpu_resident_hot_prepare`, deux appels ordonnés après validation du graphe
   natif, plutôt qu'un label unique. Recalculer leur CPU Running ∩ GPU idle
   sur cette acquisition ; ne pas recopier les 12,142 ms historiques ni
   sommer des frames inclusives d'une même pile.
10. **SSD critique.** Les unions preadv et leur recouvrement GPU décrivent
    l'attente POSIX observée. Le tail après la dernière commande indépendante
    est une borne conditionnelle à l'ordonnancement fixe ; il ne prouve pas
    une lecture NAND physique, ni une causalité SSD pour tout idle du lot.
    Les nombres historiques 5 244 reads, 1 248 012 B/read et les résultats
    d'anciens états hot-ready ne sont pas des observations de r2.

## Limites finales à conserver

Le run ne possède pas de contrôle après ni de qualification native six phases.
Les résumés fermés peuvent être vérifiés séparément contre le golden, mais
pas transformés en `full_run_validated`. Le coût de l'observateur combiné,
la compression mémoire et les interactions kernel restent des conditions
de capture, pas un gain release. L'analyse r2 sera un diagnostic partiel
des données effectivement couvertes ; aucun résultat MTP/DF ou CPU-only
non acquis ne peut être rempli à partir de ce seul AR.
