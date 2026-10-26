# Issue 52 — qualification des modes Flash Next vers 15 tokens/s

Source native : `a585a30b2b2ae0c1f2743b72dc74de17a71ff71d` (main audité).
Branche locale : `codex/issue52-throughput`. Aucun changement de calcul,
de poids ou de défaut Desktop. Les prototypes QSA B1 et Cold A1 restent
distincts : leurs résultats antérieurs ne qualifient pas un nouveau gain.

## Implémentation

`scripts/benchmark_flash_modes.py` réutilise le vrai bridge résident via
`benchmark-flash-bridge.py`. Celui-ci accepte désormais un environnement
explicite par enfant et conserve ses réglages dans le rapport. Le pilote
neutralise les expériences MLXL3 héritées, vérifie le SHA-256 du moteur avant
et après chaque bras, puis compare hashes et textes complets de toutes les
réponses, chauffes comprises. Registres jetables, délais bornés et jonction des
enfants/lecteurs restent ceux du pilote existant.

Les rapports incomplets, processus non joints, modes ou histogrammes de
largeur incorrects, profils incohérents, clipping de contexte, sorties vides,
divergences et valeurs non finies échouent explicitement. Une interruption
conserve `parity: false`. Les mesures invalides ne peuvent empêcher l'écriture
du statut d'échec. Une campagne terminée avec parité n'implique ni un gain
stable ni l'atteinte de la cible.

## Protocole fixé avant les mesures

- Apple M5 standard, 10 cœurs GPU, 24 Gio, MLX 0.32.2, alimentation secteur
  observée avant/après chaque bras. Un modèle GPU à la fois, aucune compilation
  ni aucun proveur pendant les timings.
- Cible Flash Next EXL3 2.05bpw `65c89531`, draft DFlash2 `a22a1744`, Q4/ring,
  greedy, contexte 4096, cache de prompt désactivé. Deux prompts fixes MAIN/HOLD
  du pilote historique, sans données personnelles.
- Neuf configurations dans l'ordre ci-dessous puis dans l'ordre inverse.
  Chaque bras est un nouveau bridge ; une chauffe de 128 tokens par prompt,
  puis une génération mesurée de 128 tokens/127 décodés par prompt. Les deux
  prompts partagent le processus et le cache experts du bras : ce protocole
  diffère des anciens bras à un seul prompt. Total prévu : 18 processus,
  36 chauffes et 36 générations mesurées.
- Livré : OFF, MTP1, Auto, Auto avec PRECAP seulement ; cache experts
  automatique et allocateur 512 Mio, Hot/Wide/63/UNIQUE désactivés.
- Expérimental : Auto, B2, B3, B4, PRECAP ; cache experts 3 Gio/53 slots,
  allocateur 64 Mio et Hot/Wide/63/UNIQUE activés. B2 signifie **une** proposition,
  B3 deux, B4 trois. Le protocole ne réactive aucun candidat rejeté.
- Les modes d'une même configuration sont comparés sur le même binaire.
  Comparer Auto livré à Auto expérimental change plusieurs réglages et ne
  permet pas d'attribuer le gain à un flag isolé. PRECAP contre Auto dans un
  même régime isole le préplafonnement.
- Mesures conservées : decode, TTFT interne et de bout en bout, temps complet,
  lignes draft calculées/vérifiées, acceptation globale, demandes logiques
  d'experts et mémoire processus/MLX disponible. Les octets demandés incluent
  le cache fichier ; ils ne sont pas une mesure du trafic physique NVMe.
  Les temps du profil sont des durées hôte inclusives, pas GPU exclusives.
- Stabilité : étendue max/min des mesures d'un mode/prompt ≤ 3 %. La cible
  n'est atteinte pour un cas que si **toutes** ses mesures atteignent 15 tok/s
  et passent ce contrôle. Aucun bras instable n'est effacé ou relancé pour
  sélectionner un meilleur résultat.

## Résultats partiels — arrêt demandé par l'utilisateur

La campagne a été interrompue proprement avant le push demandé : **11 bras
terminés sur 18**, le douzième interrompu. Tous les processus et lecteurs ont
été joints. Le rapport global reste `status: interrupted`, `parity: false` ;
la parité des bras terminés a passé, mais aucune qualification globale du
débit ni aucun gain stable n'est déclaré. Aucun résultat mesuré n'atteint 15 tok/s.

Première passe, 128 tokens émis / 127 décodés par mesure :

| Configuration | MAIN tok/s | HOLD tok/s |
|---|---:|---:|
| shipped-off | 10.585727 | 10.224325 |
| shipped-mtp1 | 10.832035 | 9.954839 |
| shipped-auto | 10.458101 | 9.904878 |
| shipped-precap | 11.608523 | 11.088425 |
| tuned-auto | 11.173433 | 10.346722 |
| tuned-b2 | 11.792101 | 11.294650 |
| tuned-b3 | 10.689912 | 10.037851 |
| tuned-b4 | 10.211819 | 9.908257 |
| tuned-precap | 10.390244 | 11.362206 |

La seconde passe PRECAP expérimental donne 12,182561/11,333915 tok/s,
contre 10,390244/11,362206 au premier passage. La forte variation MAIN
interdit de présenter 12,18 comme un gain confirmé. Le passage inverse B4
est également conservé. Les bras inverses restants ne sont pas exécutés.

PRECAP livré réduit les lignes draft calculées de MAIN de 384 à 183 dans la
première paire. Le cache automatique a cependant choisi 88 puis 90 slots,
respectivement 4,91 et 5,02 Gio : cette observation ne suffit pas à attribuer
la totalité de l'écart de débit au seul flag. B2 reste le meilleur premier
passage commun MAIN/HOLD, mais son témoin inverse manque.

Le temps avant le premier token reste voisin de 8–9 secondes sur ces prompts
après chauffe. Le budget de 15 tok/s est 66,67 ms/token : les mesures ne
qualifient aucun mécanisme permettant actuellement de l'atteindre.

[État partiel et nombres de réponses](measurements/issue52/partial-results.json),
[rapport interrompu](measurements/issue52/matrix/campaign.json),
[commandes et identité](measurements/issue52/matrix-identity.json),
[validation physique](measurements/issue52/quality.json).

## Vérification exécutée

- `python -m pytest -q tests/test_flash_modes.py tests/test_flash_benchmark.py` :
  **111 passés**, aucun sauté. Couvre la matrice réelle avec enfants jetables,
  les modes ignorés, distributions incohérentes, clipping, divergence,
  interruption, changement de binaire, non-fini et échec durable.
- Suite Python complète exécutée avant les derniers ajouts de tests du pilote :
  **1822 passés, 115 sautés, 10 avertissements**. Sauts : dépendances de
  conversion/torch absentes, modèle Ling absent et Metal explicitement exclu ;
  avertissements Modal préexistants. Les derniers changements du pilote sont
  couverts par les 111 tests ciblés ; aucune suite n'a été relancée après la
  demande d'arrêt.
- `cargo test --locked --no-default-features` : **240 passés, 5 ignorés**
  (154 lib, 35 binaire, 16 contrats, 35 intégration CPU).
- Build livré `cargo build --locked --release --features mlx,chat`,
  Clippy strict `--all-targets --features mlx,chat -- -D warnings`, `cargo fmt
  --all -- --check`, Ruff format/lint et `py_compile` passent.
- Suite lib MLX release : **181 passés, 155 ignorés**. Tests physiques sélectionnés
  seuls : pair, oracle QSA indépendant, captures/tous commits, runtime 42 cas
  (budgets 1/2/3/8/17/64, sept configurations), PRECAP 512 tokens : **5 passés**.
  Tokens/offsets/états exacts contre le séquentiel ; les assertions numériques
  des tests pair/oracle/préfixes passent. Le vrai bridge passe **43 contrôles**,
  parité, erreurs, annulation et reprise comprises.
- Mutations temporaires des contrôles de parité et de persistance des erreurs
  non finies : détectées ; sources restaurées byte à byte, tests relancés.
  Le premier échec de fixture et les résultats des mutations sont conservés.
- `crosshair check scripts.benchmark_flash_modes.number
  scripts.benchmark_flash_modes.counter --per_condition_timeout 20 --report_all` :
  trois conditions confirmées sur le code réel. **Recherche symbolique limitée,
  pas preuve formelle de la campagne, du GPU, de la concurrence ou des I/O.**
  Aucun Rust exécutable modifié ; aucun nouveau résultat Kani revendiqué.
- CI : le workflow PR/push-main inclut les nouveaux contrôles CPU ; YAML vérifié.
  Sur la base exacte a585a30b : régression/conversion SUCCESS, Rust CANCELLED.
  La nouvelle branche est poussée à la demande de l'utilisateur, sans attendre
  sa CI après le push.

Les logs bruts, erreurs et commandes sont conservés sous
`docs/measurements/issue52/`. Préenregistrement et clôture dans `opti.md`.

## Limites

Les prompts courts et la génération 512 ne qualifient pas les seuils QSA
2050/2051/2052/2053, 4095/4096, ni le contexte long réel du modèle. Application
installée, autres familles/Mac, temps GPU exclusifs, trafic NVMe physique,
contrôleur OFF avant draft et gains de nouveaux kernels non vérifiés. Le code
numérique reste celui du main audité ; aucun défaut de production n'est promu.
