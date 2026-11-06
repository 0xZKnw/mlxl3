# AR39 — Flash Next natif : profilage des 128 tokens (6 novembre 2026)

Le CLI qualifié tourne à **9,7302 puis 9,9218 tok/s** dans les témoins frais après les fermetures d’applications par l’utilisateur. Le profil CPU/SSD permet une partition du thread principal sur un intervalle de sortie corrélé aux **127 décodages**. La capture système GPU reste partielle ; le diagnostic interne apporte séparément sept lignes complètes et leurs command buffers.

Aucun gain moteur n’est livré par cette campagne. Les 14 lignes Rust ajoutées sont sous `cfg(test)` : elles restaurent le comptage HC de la voie asynchrone T05 et observent ses producteurs déjà terminés. Aucun calcul, shader, commit, attente, réglage par défaut ou application installée n’est modifié.

## Identités et protocole

- Base de contenu natif : `61d7ec798812eac869c95daf0e62092847b51408`. Avant l’inférence, 257 empreintes de sources/fixtures/helpers sont contrôlées. Le CLI provient de la qualification AR2 : **SHA-256 `725fc7e18dce92ed8b16ced1679df8574872f7e595350cfcad5d8b045edcd4bd`**, révision embarquée **`5675eabf5694-dirty`**. Les sources natives compilées ont été comparées au contenu de cette base ; il ne s’agit pas d’une recompilation attribuée au nouveau SHA.
- Qwen3.8-Flash-Next EXL3 moyen 2,05 bpw/H4, checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`. Cinq métadonnées vérifiées ; pas de nouveau hachage intégral des poids.
- Apple M5, 24 Gio, macOS 27.2 / 26B123, Xcode/xctrace 27.2, SDK 27.2, déploiement 26.2. Secteur/100 % au contrôle initial ; température et fréquence non mesurées.
- MAIN : 87 tokens de prompt, 128 générés / 127 décodages, limite de contexte 4096, cache experts 3 Gio / 53 slots par couche, quatre lecteurs, profondeur 0, lecteurs dynamiques 0, cache de préfixe 0, température 0/top-k 1/pénalité 1. AR direct Metal, sans MLX, MTP ou DFlash.
- Treize réglages acceptés ON ; T05 et AR2 par leurs défauts M5. AR1/AR3/AR4 et autres expérimentations absentes ou OFF. Un modèle GPU à la fois, compilations/proveurs/exportations séparés de l’inférence. Aucun agent secondaire.

[Plans et empreintes](plan-metal.json), [plan CPU/IO](plan-cpu-io.json), [protocole alternatif sans warmup](plan-full-metal.json), [outillage](toolchain.json), [état source final](source-state.json), [contrôleur antérieur préservé](controller-before-full/index.json).

## Débit et surcharge des observateurs

| Phase après reprise | tok/s mesurés | État |
|---|---:|---|
| Témoin avant, warmup + mesure | 9.730156239398045 | Golden complet, joint |
| Time Profiler + threads/FS/syscalls | 8.481176003615078 | Acquisition sauvegardée, fenêtre partielle |
| Metal System Trace, fenêtre glissante | 7.439368615073141 | Acquisition sauvegardée, GPU visible 1,648 s |
| Témoin après, warmup + mesure | 9.921780336138784 | Golden complet, joint |

Les témoins dérivent de +1,97 %. Les débits instrumentés ne sont pas des débits livrés et leurs durées ne sont pas reportées telles quelles sur le témoin. Le témoin antérieur à la pause (7,3977 tok/s) appartient à d’autres conditions ; sa différence après fermeture d’applications n’est pas un boost de code.

[Contrôle avant](resume-control-before/report.json), [CPU/IO](cpu-io-window/report.json), [Metal](resume-metal-window/report.json), [contrôle après](resume-control-after/report.json).

## Budget hôte corrélé aux 127 décodages

133 écritures stdout positives du PID cible correspondent exactement aux frontières du suffixe NDJSON de la requête mesurée. L’intervalle entre la première sortie et la complétion vaut **14,974382042 s**, contre **14,974338458 s** annoncées par le CLI : écart **43,584 µs**. Cela valide une cohérence temporelle hôte ; cela ne prouve pas une latence individuelle pour chaque token.

| Partition du thread principal | secondes | part |
|---|---:|---:|
| Bloqué dans `psynch_cvwait` | 8.031637074 | 53.636 % |
| Bloqué dans `semaphore_timedwait` | 4.012889962 | 26.798 % |
| Running | 2.767221112 | 18.480 % |
| Runnable | 0.115945617 | 0.774 % |
| Autre blocage | 0.032687499 | 0.218 % |
| Interrupted + Preempted | 0.014000778 | 0.093 % |

La partition a **zéro recouvrement et zéro temps non attribué** dans les états exportés de cet intervalle. Les durées d’appels sont intersectées avec les états bloqués ; elles ne sont pas ajoutées une deuxième fois.

Les piles dominantes de `cvwait` passent par les attentes Metal (`wait_command`, `RouteOutput::indices`…) ; les huit piles conservées détaillent 92,93 % de son temps d’appel. Les piles de sémaphore détaillées couvrent 100 % et passent par la réception des complétions SSD/`ReadBatch::ready_min`. Une attente Metal n’est pas une mesure de calcul par kernel.

[Budget et piles](decode-budget.json), [CSV](host-budget.csv), [corrélation stdout](stdout-correlation.json), [fréquences d’échantillons](sample-frequency.json), [analyse des tables CPU](analysis-cpu-io/cpu-summary.json). Les poids Time Profiler avec context switches restent des poids de l’outil : ils peuvent dépasser la fenêtre retenue et ne constituent pas un budget de CPU occupé. Les cellules indisponibles restent `null`.

## Lectures pendant l’intervalle corrélé

**19 554 `pread` entièrement contenus**, aucun appel à proratiser aux frontières, **22 965 319 052 octets retournés** : 22,965 Go décimaux, environ 180,829 Mo par décodage et 1,534 Go/s de trafic logique observé. Ce n’est pas une télémétrie NAND.

| Octets retournés par appel | Appels |
|---:|---:|
| 1 248 012 | 17 375 |
| 625 420 | 1 026 |
| 622 592 | 1 026 |
| 5 120 | 127 |

Les deux tailles intermédiaires s’additionnent à un record expert ; elles sont cohérentes avec le split déjà activé. Les petites lectures ont une pile `FlashNext::embedding_bits`. Par tailles de records, on infère 18 401 remplissages experts pour 60 960 routes, soit environ 69,8 % de résidence : **inférence sous les invariants du lecteur, pas un compteur direct de hits ni une garantie d’exhaustivité des événements**.

Sur la fenêtre CPU plus large de 17,917 s : 59 298 échantillons, 1 488 970 syscalls, 90 496 événements FS et 840 570 états de threads. Les 22 980 `pread` de cette fenêtre ont une médiane de 0,640459 ms et un p95 de 1,007917 ms, avec 26,925 Go retournés. Cette fenêtre inclut une fin de prefill ; ces valeurs ne doivent pas être divisées par 127.

[Lectures observées, CSV compressé](analysis-cpu-io/reads.csv.gz), [tables agrégées](analysis-cpu-io/cpu-summary.json).

## Diagnostic interne : autre population, pas une mesure release

Le premier essai échoue sur un ancien comptage HC qui ne couvrait pas le submit T05. [Échec original](internal-native7/report.json). Après instrumentation de test seulement, **un test physique sélectionné passe**, avec comparaison contrôle/observé du hidden, des logits et des 84 tenseurs de continuation / 28 803 072 mots finis, plus contrôle initial indépendant des logits et du premier tenseur archivé. Les voies réelles exigent **336 producteurs P et 336 soumissions HC asynchrones** sur 7 × 48 couches.

- **13 303 événements**, moyenne de ligne **107,787 ms** ; 2 103 commandes GPU distinctes, zéro doublon observé. Union des intervalles de commandes : **57,883 ms par ligne**. Ce ne sont pas des temps par kernel ou un taux d’utilisation GPU.
- En moyenne **3 393,86 demandes de buffers Metal / ligne**, **357 392 351 octets demandés / ligne** et **7,742 ms dans le wrapper d’allocation**. Ce dernier inclut ses contrôles/observateur ; ni les octets demandés ni ce temps ne sont une empreinte RAM ou une durée pure de `newBuffer`.
- 336 MoE, 3 360 routes, 2 263 hits et 1 097 records lus. Sur les 549 cohortes froides positives, 338 sont des singletons. Les marqueurs de première vague existent dans 248 cas ; ne pas les représenter comme les 301 MoE avec lectures.
- Les horloges CPU `Instant`, GPU/Metal et timelines locales MoE ne sont pas alignées. Les sommes d’intervalles GPU de propriétaires peuvent se recouvrir.

[Résumé](internal-summary.json), [336 lignes de couches](internal-layers.csv), [événements](internal-native7-t05/events.json.gz), [log exact compressé](internal-native7-t05/native7.log.gz), [qualité](internal-native7-t05/report.json), [binaire/source](internal-freeze.json), [sous-groupes de commandes centrales](gpu-central-parent-scopes.json).

## Couverture GPU et limites conservées

Le Metal System Trace sauvegardé ne conserve pour le PID cible que **58,259162500 à 59,907444083 s**, soit 1,648281583 s : 4 302 lignes GPU, 4 276 encoders, union d’activité exportée 0,728944692 s. Aucun timing ShaderTimeline n’est disponible. Le catalogue de 1 868 shaders/PSO ne prouve pas leur exécution. [Analyse](gpu-window-summary.json).

Une capture normale d’une seule requête 128, sans fenêtre glissante, expire à la finalisation après le budget extérieur de 300 s. La génération respecte le golden en contrôle postérieur, mais la capture reste **failed/parity false**. [Échec](full-metal/report.json), [validation numérique distincte](full-metal/posthoc-numeric-only.json). Aucune relance Metal supplémentaire.

Les deux captures sauvegardées contiennent un signal `RunIssues` **`Data stream: Time Mapping`**, type 2/subtype 0 à t=0. Sa signification n’est pas résolue ici ; aucune garantie générale d’absence de pertes ou de calibration CPU/GPU n’en est déduite. [Audit](trace-issues-audit.json). L’export `ThreadActivity` a dépassé 60 s ; les tables sous-jacentes effectivement exploitées sont identifiées. [Empreintes des exports](export-input-hashes.json), [reçus des exports et timeout](export-results.json).

Les ressources exportées couvrent également la destruction du processus ; leur minimum n’est pas une empreinte stationnaire. [Ressources](resource-summary.json). Ni température, fréquence, énergie, occupation matérielle ni temps GPU par kernel n’est qualifié.

## Vérification et reproduction

- Rust : **412 passés / 100 ignorés**, deux cibles vides explicites ; Clippy strict `direct-metal,chat` et `mlx,chat`, rustfmt, build release libtest passent. [Commandes/logs](rust-checks/checks.json).
- Python : **438 passés / 44 sautés Linux uniquement / 60 subtests**, puis 15 tests d’analyse ciblés après deux corrections de lint ; inventaire MLX **283 fichiers** conforme. Les mutations budget128→32 et cache de références XML sont détectées. Les échecs initiaux/receipts restent présents.
- Kani 0.68 : première sélection sans `--tests` inapplicable, puis deux harnesses réels de l’API de trace tentés avec `--tests` ; **ICE du compilateur Apple/aarch64, aucune propriété vérifiée**. Pas de stub ou de domaine réduit pour obtenir un succès. [Tentatives](rust-checks/kani-tests.json). CrossHair absent localement ; [diagnostic](final-crosshair.log). Cela ne prouve pas Metal ou la concurrence.
- Les workflows PR/main exécutent les contrôleurs/analyseurs et les contrats de profilage pertinents. Le résultat GitHub du commit publié est à inspecter séparément.

```sh
python scripts/profile_native_ar.py --plan PLAN --out NOUVEAU_DOSSIER --mode control --timeout 300 --run-after-root-grant
python scripts/profile_native_ar.py --plan PLAN --out NOUVEAU_DOSSIER --mode trace --timeout 300 --run-after-root-grant
python scripts/analyze_native_ar_profile.py --directory EXPORTS_CPU --pid PID_CIBLE --out SORTIE
python scripts/analyze_native_metal_trace.py EXPORTS_GPU --pid PID_CIBLE --output SORTIE.json
```

Les plans contiennent des chemins locaux et refusent les empreintes différentes ; adapter un plan exige un nouveau préenregistrement et une nouvelle identité. Les [snapshots des collecteurs/analyses ponctuelles](repro-snapshots) et le [schéma du parcours](architecture.mmd) décrivent leurs entrées figées. Les bundles et XML système restent privés/non versionnés ; seuls les extraits filtrés et leurs identités sont publiés. Aucun essai en cours, modèle, compilateur ou proveur n’est laissé actif à la clôture.
