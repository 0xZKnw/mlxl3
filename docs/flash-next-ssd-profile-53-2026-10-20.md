# Qwen Flash Next SSD — profil 53 du 20 octobre 2026

La collecte contient **3 989 383 événements**, **602 571 lectures d’experts instrumentées** et **50 679 lignes de statistiques par composant**. Le lot 3 Gio est complet ; le lot 6 Gio est interrompu pour pression mémoire. Ce sont des données de diagnostic, sans nouveau gain global qualifié.

La cible est Qwen3.8 Flash Next EXL3 annoncé 2,05 bpw, checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, Apple M5/24 Gio, MLX 0.32.2, secteur, applications laissées ouvertes. L’instrumentation Rust est `cfg(test)` ; les timestamps disque C++ sont optionnels et désactivés par défaut dans le moteur. L’application installée reste celle d’avant cette collecte.

## Couverture et protocole

| Lot | Contrôles release | Profils naturels | Profils synchronisés | Statut |
|---|---:|---:|---:|---|
|3 Gio, OFF/MTP ×MAIN/HOLD |4 |4 |4 |12/12 cas complets, tous joints, 0 nouveau Swapout |
|6 Gio |1 terminé |1 interrompu |0 |245 104 640 octets de nouveau Swapout système sur le contrôle ; 10 autres cas non lancés |

Chaque cas naturel utilise un processus neuf, 128 sorties (127 pas décodés), puis six générations : warmup, contrôle sans trace, trois traces, contrôle sans trace. Les resets remettent les états sémantiques à zéro et conservent les poids/fréquences LFU. Les compteurs avant/prefill/après sont enregistrés à chaque génération. Un contrôle release neuf, une warmup et une mesure 128, précède chaque cas.

Les profils synchronisés sont un lot distinct : 33 sorties (32 pas), warmup et oracle sans trace puis deux passages avec évaluations intermédiaires. Les deux passages sont conservés séparément. Le premier peut inclure des compilations de graphes propres à ces frontières ; il n’est pas un warmup exclu des résultats.

Paramètres constants : 53 slots par couche à 3 Gio (3 174 942 528 octets de banques sur 48 couches), 4 lecteurs, overlap 1, LFU/auto, LES 0, prédicteur F32, gates résidentes 0, cache allocateur 512 Mio, contexte maximal 4096. MTP1 greedy ; tête 512 experts résidents. Les prompts et commandes exacts sont dans les plans.

La limite processus est **12 000 000 000 octets décimaux** de footprint physique, distincte du RSS et des allocations MLX. Après le contrôle 6 Gio, la garde 2 s a été rendue cumulative depuis le premier contrôle : 256 Mio de Swapins et 256 Mio de Swapouts. Total de la campagne à la fin des modèles : Swapouts 245 104 640 octets, Swapins 185 171 968 octets. Aucune nouvelle écriture swap sur les cas 3 Gio. Ces compteurs sont globaux ; ils ne désignent pas le processus qui a été paginé. Le swap préexistant n’a pas été effacé.

## Parcours naturel : temps observés

Ces débits sont ceux de l’unité de diagnostic. Chaque ligne regroupe 381 pas décodés issus de trois répétitions. Le débit total vaut 381/temps total, la plage garde les trois résultats. Les quatre contrôles release sont des mesures individuelles, sans qualification de gain.

|Mode/prompt |Débit total diagnostic |Plage des traces |Témoin release |Pic processus diagnostic |
|---|---:|---:|---:|---:|
|OFF MAIN |9.475 tok/s |9.232–9.652 |9.413 |7.470 Go |
|OFF HOLD |9.501 tok/s |9.312–9.671 |9.660 |7.476 Go |
|MTP MAIN |9.724 tok/s |9.374–10.047 |9.321 |8.334 Go |
|MTP HOLD |8.811 tok/s |8.417–9.188 |8.727 |8.333 Go |

Dérive max/min des trois traces : OFF MAIN 4.55%, OFF HOLD 3.86%, MTP MAIN 7.18%, MTP HOLD 9.16%. Les deux cas MTP dépassent 5% ; les valeurs restent descriptives. Les contrôles sans trace adjacents dérivent de 1,00/3,57/1,05/2,00% dans cet ordre. Aucun échantillon n’a été supprimé ni relancé pour bruit.

|Mode/prompt |Frontière route CPU incl., ms/token |Appel lecture batch, ms/token |Octets retournés par preadv/token |Jobs/token |
|---|---:|---:|---:|---:|
|OFF MAIN |57.37 |30.88 |179.78 Mo |144.05 |
|OFF HOLD |56.39 |32.23 |193.28 Mo |154.87 |
|MTP MAIN |52.87 |28.83 |195.65 Mo |156.77 |
|MTP HOLD |58.43 |32.00 |217.23 Mo |174.06 |

La frontière route attend aussi les opérations GPU amont. Le batch lecture comprend préparation FFI, dispatch, lecture et attente des workers ; les kernels hot peuvent avancer en parallèle. Chaque job lit un record d’expert et peut effectuer plusieurs appels `preadv` ; le nombre de syscalls n’est pas instrumenté. Les octets sont ceux effectivement retournés par `preadv`, pouvant provenir du cache fichier. Ce ne sont pas les octets NVMe physiques. Ces intervalles sont **à l’intérieur** du décodage et de la vérification MTP ; ils ne s’ajoutent pas à leurs parents.

## MTP : vérification et acceptation

|Prompt |Acceptés/proposés 128 |Tête, ms/token |Vérification, ms/token |Réparation, ms/token |Part vérification dans decode |
|---|---:|---:|---:|---:|---:|
|MAIN |54/72 (75.00%) |3.96 |97.64 |0.54 |94.95% |
|HOLD |48/78 (61.54%) |4.20 |108.03 |0.51 |95.19% |

La vérification exécute le modèle cible. Ses~95% ne représentent pas 95% de surcoût supprimable. Les trois répétitions MAIN calculent 435 lignes cible pour 381 pas émis, avec 54 lignes spéculatives non retenues ; HOLD calcule 471 lignes pour 381 pas, avec 90 non retenues. Les bonus déjà en file sont des sorties à très faible latence et sont conservés dans `output-steps.csv`, avec une classe explicite.

Les compteurs acceptation/blocs de l’unité et du bridge release coïncident pour chaque prompt. Les répétitions suivent la même séquence déterministe ; elles ne sont pas trois tâches indépendantes.

|Pas décodés où débute le bloc |MAIN accepté/proposé |HOLD accepté/proposé |
|---|---:|---:|
|1–32 |16/16 (100.0%) |16/16 (100.0%) |
|33–64 |13/19 (68.4%) |11/21 (52.4%) |
|65–96 |14/18 (77.8%) |11/22 (50.0%) |
|97–127 |11/19 (57.9%) |10/19 (52.6%) |

Les intervalles classent un bloc par son premier pas de sortie ; un bonus peut sortir dans l’intervalle suivant. Les profils synchronisés 33 sorties ont 16/16 acceptations sur les deux prompts, alors que les parcours 128 finissent à 75% et 61,54%. Une mesure courte n’est donc pas une estimation de l’acceptation sur le parcours complet.

## Cache et réutilisation réelle entre les deux lignes

|Mode/prompt |Hits par occurrence |Cohortes entièrement prêtes |Experts communs moyens par paire |
|---|---:|---:|---:|
|OFF MAIN |69.99% |11.75% |— |
|OFF HOLD |67.74% |10.30% |— |
|MTP MAIN |68.71% |4.08% |3.843 |
|MTP HOLD |68.31% |3.24% |3.288 |

Ces fractions sont comptées sur les vraies cohortes (10 occurrences OFF, 20 pour les paires MTP), sans hypothèse d’indépendance des hits. Les moyennes de 3,843/3,288 experts communs concernent les 48 couches cible ; la tête MTP résidente est séparée. Les lectures sont déjà dédupliquées par slot ; cela ne démontre pas une réutilisation du décodage EXL3 sur GPU. Les fichiers `layers.csv` et les routes brutes permettent une analyse par couche.

## Exemple de variation conservé

MAIN MTP :

|Trace |Decode |Misses par occurrence |Octets retournés |
|---|---:|---:|---:|
|trace_1 |12.640371 s |21 780 |24 846 670 908 |
|trace_2 |12.991917 s |21 780 |24 849 166 932 |
|trace_3 |13.547667 s |21 780 |24 847 918 920 |

Le temps augmente de 7,18% entre les extrêmes, avec le même nombre de misses et une étendue du volume lu inférieure à 0,011%. Dans la partition exclusive du **temps hôte** entre trace 1 et 3, les attentes de routes (tête incluse) augmentent de 0,385 s, les appels lecture de 0,183 s ; le reste est notamment dans la construction hôte et d’autres attentes. Cela n’attribue pas ces temps exclusivement au GPU ou au SSD, et ne prouve pas rétroactivement la cause exacte du 9,39→7,62 de 49. Les compteurs globaux de compression et les observations système contemporaines sont conservés.

## Profils synchronisés : repères par composant

Les étapes évaluent les vrais tableaux, en conservant les calculs et leurs états. Les barrières portent le decode OFF MAIN 32 pas de~3,4 s à~16,6 s. Les valeurs ci-dessous incluent dispatch, scheduling, attente et éventuelle compilation : **ce ne sont ni des temps GPU exclusifs, ni une répartition du decode naturel**. Deux passages sur MAIN, 64 pas au total ; MTP a une acceptation 100% dans ce préfixe.

|Étape évaluée |OFF ms/pas émis |MTP ms/pas émis |
|---|---:|---:|
|EXL3 trellis QMV |54,308 |34,486 |
|Entrée EXL3 : échelles/gathers/Hadamard |40,481 |22,572 |
|Sortie EXL3 : Hadamard/échelles |37,929 |20,135 |
|HC lecture attention |16,785 |16,893 |
|HC lecture MLP |16,475 |17,074 |
|Projection sortie attention |15,274 |15,785 |
|Shared gate projection |12,049 |12,068 |
|Shared up projection |12,082 |12,255 |
|Shared down projection |11,811 |11,983 |
|GDN delta |9,559 |9,762 |

Les distributions par couche, phase et forme sont disponibles. Une valeur plus faible dans ce protocole ne qualifie pas un gain sur le parcours naturel. Xcode/xctrace et le compilateur Metal CLI sont absents ; la compilation Metal du runtime MLX fonctionne et les contrôles physiques ont été exécutés.

## Fichiers à utiliser

|Fichier |Lignes |Contenu/unités |
|---|---:|---|
|[phases.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/phases.csv) |48 |temps en secondes, mémoire en octets, VM en pages/événements, compteurs avant/après |
|[components.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/components.csv) |50 679 |scopes par couche/phase/partie ; durées en ns, count et host_self_total_ns |
|[layers.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/layers.csv) |2 316 |hits/misses, cohortes prêtes, partage, lectures et déséquilibre des workers |
|[workers.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/workers.csv) |9 189 |latences ns par worker logique, quantiles et octets retournés |
|[output-steps.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/output-steps.csv) |1 844 |serial/accepted/rejected/queued/fallback, offsets, routes, lectures et spans MTP |
|[system-timeline.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/system-timeline.csv) |951 |GPU/disque système échantillonnés 2 s ; CPU dans les JSON bruts |
|[acceptance-by-position.csv](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/acceptance-by-position.csv) |24 |acceptation par intervalle et répétition |

- [Synthèse numérique](measurements/flash-next-ssd-impl/opt-53-detailed-profile/research-summary.json), [résultat export](measurements/flash-next-ssd-impl/opt-53-detailed-profile/exports/result.json), [schéma](measurements/flash-next-ssd-impl/opt-53-detailed-profile/schema.json).
- [Protocole initial](measurements/flash-next-ssd-impl/opt-53-detailed-profile/protocol.json), [amendement mémoire](measurements/flash-next-ssd-impl/opt-53-detailed-profile/protocol-v2-memory.json), [protocole export](measurements/flash-next-ssd-impl/opt-53-detailed-profile/export-protocol.json).
- [Sources/binaires figés](measurements/flash-next-ssd-impl/opt-53-detailed-profile/campaign-identity-v2.json), [dépendances et checkpoint](measurements/flash-next-ssd-impl/opt-53-detailed-profile/dependencies-final.json), plans et logs sous chaque cas.
- Chaque dossier `profile` valide contient ses résumés, ses événements `*-prefill.json.gz` / `*-decode.json.gz`, et une analyse compressée. Les 48 traces brutes incluent les petits contrôles de qualité. Les `parent` sont des indices dans le même tableau, les temps sont relatifs à sa racine.

Les CSV conservent leurs fins de ligne CRLF standard ; leur lecture a été vérifiée intégralement. Le contrôle Git reconnaît ces CR avec `cr-at-eol` et garde ses autres règles.

Les quantiles suivent le rang empirique `ceil(p*n)-1` et conservent les zéros. Les colonnes min/p 50/p 90/p 95/p 99/max/total des composants et workers sont en ns. `host_self_total_ns` retire les enfants directs : sa somme sur tous les scopes égale exactement la durée de la racine. Cela partitionne le temps hôte observé, avec les attentes de travail GPU différé. La latence par sortie MTP mélange travail cible et bonus déjà en file ; utiliser la classification pour les séparer.

## Vérification et limites

Révision de couverture : les notes préliminaires parlaient de 16 positions pour l’oracle couplé. La relecture du fichier réellement chargé et du test confirme **5 positions**. Le contrôle des activations intermédiaires des 48 couches vérifie formes/finitude et journalise les écarts ; sa tolérance RMSE est infinie. L’égalité indépendante stricte porte sur les logits complets, les routes et les deux états GDN capturés. Les contrôles de paires, états et séquences décrits ci-dessous ont une portée distincte. Voir [oracle-scope.json](measurements/flash-next-ssd-impl/opt-53-detailed-profile/oracle-scope.json).

- `cargo fmt`, Clippy strict et build release `--features mlx,chat` passent. Suite complète MLX : 131 passés, 128 ignorés recensés ; suite CPU chat 109 passés/4 ignorés. Les tests physiques sont exécutés seuls et sélectionnés explicitement.
- Trois oracles physiques : référence indépendante sur 5 positions (248 320 logits et routes des 48 couches par position, états GDN sur les 2 couches capturées), paires/commits 1 et 2, session MTP budgets 1/2/3/17 avec acceptations et rejets. Le bridge du moteur actuel passe 1/2/3/17, annulation/reprise/fallback et configurations invalides. Deux petits profils 17 sorties vérifient aussi les fences.
- Chaque génération mesurée a 248 320 logits finis et des états cible/tête non vides, comparés octet par octet au premier parcours sans trace du même mode. Les 128 IDs/texte/hash concordent avec le golden externe ; les 33 IDs concordent avec son préfixe naturel. L’égalité des états 128 porte sur les répétitions du même mode ; elle n’est pas une preuve formelle serial/MTP sur tous les contextes.
- Python : 229 tests ciblés passent ; mutations retirant le contrôle des octets, la garde cumulative et un contrôle de span de vérification détectées, puis sources restaurées et contrôles repassés. Horloges racine/wall croisées à 20 ms ou 1%, conservation exacte du temps hôte, lignes cible/offsets/compteurs/bytes par bloc, quantiles vérifiés indépendamment avec NumPy sur les données complètes.
- Lecteur C++ : 235 cas sous ASan/UBSan et 235 sous TSan, 4 workers, 1/2/5/512 records, troncatures, invalides, drainage des frères et reprise. Cela reste une vérification par tests.
- Kani 0.68 sur le vrai parcours Rust/MLX et le cycle CPU : ICE `intrinsics.rs:243`, non vérifiés. CBMC 6.11 sur la vraie source C++ : parse error libc++Apple avant obligations, non vérifié. CrossHair absent pour les scripts. Aucun résultat ci-dessus ne constitue une preuve complète MLX/Metal/allocateur/concurrence.
- CI observée sur `27709c98` : Linux, macOS, contrats Flash et régression réussis ; Kani global encore en cours au relevé. Le nouveau commit des exports doit être vérifié séparément. Le job global avait précédemment fini avec une étape annulée ; la CI globale n’est pas déclarée verte.

Commandes et sorties exactes : `checks.json`, `final-checks.json`, `full-mlx-tests.json`, `verification-attempts.json`, `*tests*.log`, plans `run.json` et résultats `run-result.json`. Les scripts de tests numériques utilisent les références indépendantes déjà enregistrées ; aucune autre inférence, compilation ou preuve n’a tourné pendant une campagne GPU.

Les hashes des dépendances ont été pris en fin de campagne ; leurs mtime/ctime précèdent le premier profil. Ce n’est pas une paire de hashes avant/après. Les sept shards du checkpoint ont leurs tailles et dates enregistrées, sans rehash intégral de leur contenu pendant 53.

Couverture restante : lot 6 Gio incomplet, contexte long et DFlash2/DSpark non reprofilés ici, fréquence/GPU exclusif/NVMe physique par modèle non mesurés, pression compression globale sans attribution par processus. Le chargement global a son `load_seconds` ; les lectures d’embeddings et de poids denses ne sont pas ventilées par syscall. Deux prompts fixes et des répétitions ne représentent pas une évaluation générale. Les contrôles release et profils `cfg(test)` ont des allocations et une supervision différentes ; leur écart n’isole pas à lui seul le coût du logger.

Pour régénérer les exports sans lancer de modèle, depuis la racine du dépôt, avec un dossier de sortie neuf :

```sh
/Users/justin/Documents/mix-stq1_0/.venv/bin/python scripts/summarize_flash_profile.py \
  docs/measurements/flash-next-ssd-impl/opt-53-detailed-profile \
  /tmp/mlxl3-flash53-export
```

Le source natif mesuré correspond aux empreintes publiées avec `c15c63f5` ; les binaires portent encore l’identité Git `fa9813` + sources locales au build. Le contrôleur cumulatif correspond à `27709c98`. Les outils d’export ajoutés après capture ne modifient pas ces binaires.
