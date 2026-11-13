# Qwen3.6 35B : profilage decode, prefill et MTP

Campagne des 12–13 novembre 2026 sur le moteur 2.0.8. [Données des six profils GPU et des trois profils CPU](measurements/qwen36-35b-m5-profile.json). Les noms de fichiers désignent les archives locales de mesure. **Profilage uniquement : aucun kernel ni
source de production modifié pendant cette campagne.** Les changements locaux
antérieurs sont conservés. Aucun moteur installé remplacé, aucun push ou build.

Le résultat principal est un classement stable du **decode AR chaud** :
projections EXL3 groupées, tête vocabulaire K6, experts routés, puis projections
de sortie. Ces quatre postes représentent environ 83 % du cumul shader observé.
La récurrence GDN seule représente environ 3,2 % : elle n'est pas le premier
poste de ce profil. Ce sont des parts d'échantillons GPU, pas des promesses de
gain ni une décomposition exclusive de la latence.

## Périmètre et identités

- Apple M5, 24 Gio, macOS 27.2, MLX 0.32.2, Instruments 27.2.
- Checkpoint local `Qwen3.6-35B-A3B-EXL3-2.49bpw`, experts MCG K3/K2,
  projections denses K4 et tête cible K6. Sidecar `Qwen3.6-35B-A3B-MTP-4bit`,
  affine4/group64, vocabulaire draft compact de 79 616 IDs.
- Greedy, contexte 4 096, cache de prompt désactivé ; cas AR 4 361 tokens avec
  contexte 8 192 préenregistré séparément. Aucun changement de quantification.
- La relance demandée après fermeture des activités utilisateur est étiquetée
  `idle`/`user_apps_closed`. Cette étiquette décrit l'annonce utilisateur ; les
  traces d'activité GPU restent nécessaires pour vérifier les autres processus.
  Alimentation relevée : batterie, 100 %. Ne pas mélanger avec les anciennes
  campagnes sur secteur ou sous d'autres charges.

| Identité | SHA-256 exécutable | Rôle |
|---|---|---|
| Moteur géré installé, 2.0.8 | `e87144bc0317406cab34d1c0babea4143694006bd1e5980257c979b7819fb338` | Profil applicatif principal, avant replay AR local |
| Référence locale replay | `d83526c6a85fbe30142aca86ed87766aab856707b10543453ba65b9c75656b7e` | Profil AR séparé, replay et corrections antérieures |
| Dernier candidat local | `52307481c7b566b91266b450d055767a23323ee6dd5dbc2ef67021d3444aef07` | Relance comparative des trois changements antérieurs, NT4 désactivé |

Le pointeur `current.json`, le manifeste et les bibliothèques du paquet installé
ont été vérifiés ; détails dans host-hooks.md.
Les événements `ready` des captures identifient l'exécutable réellement lancé.
Le runtime `369e0e63fc76-dirty` ne suffit pas à distinguer les deux binaires
locaux : leurs empreintes font foi.

## Relance des performances sans profileur

Comparaison d835 → 5230, identique à la comparaison précédente : chauffe 128,
128 sorties, deux répétitions FR27/code43, ordre ABBA, attente 2 s, AR puis MTP1.
Les 48 réponses ont 128 sorties et les trois signatures identiques entre A/B.
Les résultats ci-dessous sont les moyennes de cette nouvelle série, non
fusionnées avec les anciennes. Ces petits écarts variables n'établissent pas
un gain général des trois changements locaux.

| Mode et prompt | Decode A → B, tok/s | Écart | Prefill A → B, tok/s | Écart | Latence totale |
|---|---:|---:|---:|---:|---:|
| AR FR27 |50,77 →51,49|+1,42 %|205,43 →198,39|−3,43 %|−1,42 %|
| AR code43 |51,98 →52,18|+0,38 %|224,64 →225,57|+0,41 %|−0,35 %|
| MTP1 FR27 |72,03 →74,77|+3,80 %|198,13 →208,15|+5,05 %|−4,36 %|
| MTP1 code43 |72,58 →70,66|−2,64 %|221,57 →212,42|−4,13 %|+2,93 %|

Résultats bruts, répétitions, plages et conditions :
control-idle.json,
AR,
MTP1.
Cette relance compare deux binaires locaux ; elle ne mesure pas un gain par
rapport au moteur installé e871.

## Ce que les instruments permettent d'affirmer

Le template Metal stock désactivait la Shader Timeline. Une copie locale n'a
modifié que son activation ; compteurs matériels et état de performance GPU
restent à leurs défauts. M0 a qualifié des shaders nommés sur le vrai moteur.

La colonne Instruments est **Sample Duration**. Les durées sont échantillonnées
et approximatives ; une ligne n'est pas un dispatch prouvé. Le cumul par famille
classe le travail observé, l'union évite les recouvrements des observations.
Aucun de ces nombres n'est un temps matériel exclusif exhaustif. Les trous ne
sont pas automatiquement des attentes CPU. Preuves et exemples de recouvrement :
m0-gpu-semantics.md.

Les fenêtres primaires sont raccordées à `mach_absolute_time` avec le `time-info`
de chaque trace : envoi→premier delta, puis premier delta→complete. Elles
incluent l'IPC. Le premier head ne délimite pas le prefill long : chaque chunk
peut calculer sa tête. L'ancienne analyse par premier head est archivée et
n'est pas utilisée comme decode long.

## Où va le decode GPU

Cas AR FR27 installé, répétition qualifiée `idle-retry-installed-ar-fr`, deux
requêtes chaudes de 32 sorties/31 pas de decode. Les intervalles après le premier
delta représentent628,36 /637,81ms côté client. Le cumul shader vaut362,48 /
368,31ms, l'union360,62 /366,43ms. La couverture observée est donc partielle ;
le reste n'est pas affecté arbitrairement à des bulles ou au CPU.

| Poste | Cumul shader observé par token émis, ms | Part du cumul observé |
|---|---:|---:|
| Projections EXL3 groupées |2,87–2,92|24,2–25,0 %|
| Tête vocabulaire cible EXL3 K6 |2,78–2,89|23,8–24,3 %|
| Experts routés GU+down K2/K3 |2,65–2,71|22,7–22,8 %|
| Sorties4096→2048 K4, GDN et attention |1,30–1,35|11,1–11,4 %|
| GEMV/dot denses MLX |0,44–0,46|3,8–3,9 %|
| Récurrence/commit GDN |0,378–0,384|3,23–3,24 %|
| Shared down K4 |0,165–0,167|1,41 %|

Les profils AR code43 retrouvent le même ordre : groupées24,7–25,1 %,
tête23,7–24,1 %, sorties11,4 %, experts autour de22 %. Le replay FR retrouve
aussi les mêmes familles, avec une couverture de tête moins complète dans
un des profils. Cela renforce le classement, sans constituer une comparaison
de vitesse installé/replay : les captures sont séquentielles et intrusives.

`mlxl3_rs_grouped` recouvre trois géométries différentes : QKV+Z des30 GDN,
Qgate/K/V des10 attentions et GU shared des40 blocs. Leurs PSO restent distincts
dans les JSON ; leur nom commun ne permet pas encore de répartir leur coût par
géométrie. La sortie4096→2048 est également commune aux GDN et aux attentions.
La carte statique
décrit leurs formes et payloads sans les transformer en compteurs DRAM.

En MTP, la forme du travail change. FR32/depth1 fait15blocs et accepte15/15
propositions ; depth3 fait8blocs et accepte22/24 propositions. La tête cible,
les groupées et les experts restent importants. Le draft affine est visible
séparément des EXL3. Les cumuls observés de MTP1 et MTP3 ont des couvertures
différentes : leur ratio n'est pas le ratio de coût réel. Les statistiques
applicatives et les acceptations restent les dénominateurs utiles.

Six cas ont une attribution shader exploitable, soit12 fenêtres chaudes : AR FR
installé retry, AR code installé, AR FR replay, MTP1 FR, MTP3 FR, MTP1 long.
Les premiers AR FR et les AR 1097/4361 ne contenaient aucun shader du PID cible.
Une répétition préenregistrée a récupéré FR ; les deux AR longs sont restés
vides. Leurs statistiques CPU/application restent utilisables ; leur
attribution GPU est **rejetée**, sans nouvelle boucle de relances.

La configuration « Shader Timeline Enabled » du TOC ne prouve donc pas la
présence des données. Les deux exports raw examinés sur AR 1097 sont aussi
vides, mais d'autres sources internes d'Instruments existent : la cause de
cette perte n'est pas établie. Les échecs et les résumés v1 sont archivés.

## Où va le prefill long

Le prefill1097 du cas MTP1 possède une attribution GPU exploitable. Fenêtre
envoi→premier delta, deux requêtes chaudes :

| Poste | Cumul shader observé, ms | Part du cumul observé |
|---|---:|---:|
| Experts EXL3 TensorOps segmentés |943 /934|environ 60 %|
| Projections denses EXL3 TensorOps |319 /319|environ 20 %|
| Planification des routes experts |67,8 /67,8|environ 4,3 %|

Le premier axe de travail du prefill long se situe donc dans le chemin expert
segmenté ; celui du decode se situe dans les petites projections EXL3 et la
tête. Ce sont des chemins différents. La différence entre ces postes mesurés
n'établit pas un défaut algorithmique : il reste à caractériser les tuiles
effectivement occupées, les réutilisations de poids et les compteurs matériels.
Le plan de routes n'explique pas à lui seul les60 % des matmuls segmentées.

Le MTP1 long montre aussi davantage de SDPA et de copies par token observé que
le court. Les prompts et acceptations changent ; cette observation n'isole pas
causalement le coût du contexte. Les deux cas AR longs restent sans classement
shader valide. Leur prefill applicatif chaud est bien mesuré, autour de 2,15s
pour1097 et 8,28s pour4361 dans leurs contrôles initiaux.

Synthèse complète, PSO distincts, frontières, fragments et activité des autres
PID : m1-gpu-findings.md.

## GPU à 100 % ou bande passante mémoire ?

Les intervalles **Metal GPU Activity Active** donnent une autre information
que les samples de shaders :

| Decode sous instrumentation | Union d'activité GPU du moteur / fenêtre decode |
|---|---:|
| AR FR installé, retry |82,68–82,94 %|
| AR code installé |83,12–83,41 %|
| AR FR replay |87,55–88,28 %|
| MTP1 FR installé |92,31–92,82 %|
| MTP3 FR installé |94,65–94,71 %|

Ce sont des occupations temporelles observées, **pas un taux de saturation des
unités de calcul ou du bus mémoire**. Un intervalle GPU actif peut contenir
attentes mémoire, instructions de décompactage MCG, calcul et synchronisations
internes. Les intervalles non actifs ne prouvent pas à eux seuls une limite CPU.
Les conditions et les observateurs interdisent également de lire cette table
comme un gain isolé du replay ou de MTP3.

Le poids dominant des matmuls EXL3 désigne la lecture/décompression/multiplication
des poids comme premier domaine à étudier. Il ne départage pas bande passante,
latence des lectures, efficacité des chargements ou débit des instructions.
**La saturation DRAM et ALU reste non mesurée.** Aucun débit obtenu en divisant
un payload théorique par une Sample Duration n'est présenté comme compteur DRAM.

## Capture exacte d'un composant, limite des compteurs

Un contrôle codec du moteur installé puis une capture GPU ont exercé le vrai
bundle QKV+Z de la couche GDN0, avec poids du checkpoint et activations F16
synthétiques déterministes. Les8 192 + 4 096 mots de sortie sont finis, non nuls et
bitidentiques au contrôle. Il s'agit d'un composant ; ce n'est pas une requête
de chat ni une mesure de sa latence.

La capture pèse207 966 286 octets réguliers. Le premier essai a expiré pendant
la découverte ; le second a réussi mais le vérificateur a rejeté un alias de
buffer interne. L'alias unique a été vérifié dans le même paquet, sans lien
externe. Une qualification séparée conserve l'échec initial :
m2-gdn-qualification.json.

L'inspection API donne8 command buffers,5 encodeurs compute et13 dispatchs uniques,
dont le grouped à `commands/cb5/ce0/disp2`, relié à `api_calls/api154` :
`dispatchThreads`, grille 49 152×1×1, threadgroup 256×1×1, pipeline `cps4`,5 buffers.
Les autres appels comprennent chargement/copies, Hadamard, cast et scales.

Les replays `overlapping` et `serial`, état GPU `default`, se sont terminés
normalement et ont annoncé une collecte. Pourtant, les sous-arbres de coûts,
shaders et compteurs étaient vides, y compris après `profile load` en serial.
Le succès de l'outil n'a donc pas qualifié de métriques. Aucun temps par
dispatch, compteur DRAM/ALU ou débit matériel n'en est déduit. Le modèle était
fermé et les sessions de replay ont été terminées. La matrice head/MoE prévue
conditionnellement à ces compteurs n'a pas été lancée.
Détails : m2-findings.md.

## CPU, mémoire et perturbation de l'observateur

M1 chauffe 128 sorties avant contrôle, puis32 sous trace, analyse deux requêtes
de 32 sorties, et chauffe32 après détachement avant son contrôle final. Les gros
ralentissements aux transitions sont donc séparés du corps chaud.

Exemples de prefill en secondes, moteur installé :

| Prompt/mode | Contrôle avant | Chauffe sous trace | Profils1 /2 | Chauffe après trace | Contrôle après |
|---|---:|---:|---:|---:|---:|
| AR FR27, première série idle |0,131|1,190|0,156 /0,148|3,503|0,132|
| AR code43 |0,186|0,723|0,187 /0,188|1,052|0,185|
| AR 1097 |2,149|2,633|2,168 /2,172|3,831|2,162|
| AR 4361 |8,287|9,217|8,606 /8,690|25,605|8,282|

La récupération au contrôle suivant montre un effet transitoire de la capture.
Elle ne prouve pas une cause PSO ou VM unique. Dans les intervalles entre
requêtes courtes chaudes, les deltas de décompression sont généralement0–25,
contre environ445k–724k à l'entrée sous trace et740k–788k après détachement.
Ce sont des deltas cumulés entre observations, avec les pauses intermédiaires,
pas des compteurs exclusifs de chaque requête. MTP1/1097 garde10 764 puis27 527
décompressions et25/55ms de stacks indisponibles : corrélation, sans causalité
établie. Le compteur OS est bien `task_vm_info.decompressions`.

Time Profiler M1 montre les chemins `eval_impl`, encodage/driver, allocations et
construction de graphe. Il montre aussi un coût d'instrumentation visible
`__kdebug_trace64`. Ses poids `Running` sur plusieurs threads ne se somment pas
en pourcentage exclusif de latence ; il ne collectait pas les attentes.
Tables détaillées et limites :
m1-cpu-findings.md.

## Passe CPU dédiée aux attentes, 128 sorties

Trois captures distinctes, sans Shader Timeline : AR installé, MTP1 installé,
AR replay. Time Profiler avec attente/context switches, Thread Activity,
System Call Trace et Filesystem Activity. Sept requêtes de 128 sorties par cas,
chauffe sous/après trace et deux profils ; trois signatures identiques aux
contrôles. Tous les modèles sont fermés avant les exports et analyses.

Les **intervalles réels d'état du thread principal**, coupés à la phase
premier delta→complete, donnent les valeurs suivantes. Running, Blocked et
les états restants partitionnent ce thread ; ils ne partitionnent pas
l'exécution du GPU ou des workers.

| Cas | Decode client, ms | Main Running, ms | Main Blocked, ms |
|---|---:|---:|---:|
| AR installé, profil 1 |2593,75|1041,73|1496,67|
| AR installé, profil 2 |2540,76|1016,76|1468,44|
| MTP1 installé, profil 1 |1777,90|746,77|1003,84|
| MTP1 installé, profil 2 |1731,23|725,51|970,98|
| AR replay, profil 1 |2581,79|918,08|1617,46|
| AR replay, profil 2 |2624,41|959,65|1604,40|

AR installé : environ 40 % Running et 58 % Blocked, le reste surtout Runnable,
Preempted et Interrupted. Les syscalls main comprennent environ 946/934ms
d'intervalles `psynch_cvwait` et 612/596ms `iokit_user_client`. Ils recouvrent
les états et ne s'y ajoutent pas. Un nom de syscall ne prouve pas à lui seul
une attente GPU : la chaîne hôte/worker/driver doit rester distinguée.

Le decode AR installé ne montre que127 écritures stdout dans la table FS,
pas de lecture de poids visible. Ce constat ne couvre pas toutes les fautes
VM ni d'éventuels page-ins par mapping mémoire.

Deux pièges de mesure ont été corrigés sans refaire l'inférence :

- Le template CPU ne contient pas `time-info`. Le premier export a échoué
  après une capture complète ; l'échec original reste conservé. La corrélation
  CPU utilise le début UTC du TOC à la milliseconde et les paires d'horloges des
  événements, avec incertitude de bord±2,632ms pour AR et±0,529ms pour MTP/replay.
  Ce n'est pas la conversion Mach précise des profils Metal M1.
- Le TOC possède deux tables `time-profile`. Une sélection par attributs
  attente/context-switch évite de mélanger les deux. Leurs poids variables
  peuvent porter sur des périodes antérieures à leur timestamp : jusqu'à 6,04s
  worker inconnu dans une requête de 2,67s. Ces poids ne sont pas traités comme
  du temps CPU occupé ; les durées ci-dessus viennent de `thread-state`.

Les captures CPU et GPU sont des exécutions différentes. Leur juxtaposition
ne reconstruit pas un chemin critique exact et ne démontre pas un gain replay.

Les piles distinguent deux attentes AR : la condition variable dans `eval_impl`
et `IOSurfaceSharedEvent → mlx::core::Event::wait → array::eval`, sous
`Qwen35Moe::run_tokens`. Elles montrent la synchronisation de l'évaluation du
token. Les quelque 70 intervalles bloqués par token ne constituent pas 70 appels
eval ni 70 synchronisations par couche. En MTP, une attente de coordination
passe par `async_eval_all → verify_mtp` ; un readback distinct passe par
`copy_bytes → to_u32 → Head::draft_chain`.

Cette passe CPU reproduit les grands transitoires de décompression aux
transitions de trace, même sans Shader Timeline. Les attribuer exclusivement
à la recompilation des shaders serait donc injustifié. Les profils chauds
CPU128 gardent 0–244 décompressions supplémentaires entre observations.
Détails, contrôles et preuves :
m3-cpu128-findings.md.

## Reproductibilité et correction

Les protocoles et les échecs sont conservés dans `opti.md`, entrée
`Q36-MAX-PROFILE-20261112`, puis compléments M1-R, M2 et M3. Les réponses des
captures sont comparées à leurs contrôles avec nombre de tokens, hash de tokens,
SHA du texte et de l'historique. Cela contrôle le résultat d'inférence sous
observation ; ce n'est pas une nouvelle validation exhaustive des logits.
Les validations numériques bit à bit antérieures des binaires locaux restent
distinctes de cette campagne de profilage.

Les outils diagnostiques réutilisent les parseurs et processus bornés du dépôt,
valident PID, données non vides, horloges, formes et finitude, et conservent les
échecs. Les auto-tests et recherches finies ne sont pas des preuves complètes.
CrossHair et Nagini sont absents ; aucune preuve de Metal, MLX, de l'OS ou des
outils Apple n'est annoncée. Les sources de production et la CI n'étant pas
modifiées, aucune compilation ni suite GPU/CI supplémentaire n'a été lancée.

Vérifications des diagnostics :13 auto-tests CPU128/horloges/attentes passent ;
analyse GPU v2 :980 cas finis de clipping,9 cas d'erreur, mutation de la borne
multi-chunk détectée puis restaurée. Le contrôleur M2 couvre réponses invalides,
formes/finitude, EOF, exit7, silence/timeout et nettoyage. Ruff `F,E9` passe sur les outils de campagne ; le contrôleur M2
passe aussi le lint complet. Ce sont des tests et recherches finies, pas des preuves de
l'exécution complète. Les logs initiaux en échec et versions archivées restent
dans le répertoire de mesures.

Les623 empreintes de fichiers de référence restent identiques au début de cette
campagne : source-verification.json.
Les modifications exécutables de cette campagne sont limitées aux outils
diagnostiques sous `docs/measurements/`, avec leurs archives ; aucun changement
de précision, modèle, poids, kernel, cache ou moteur livré.

Les XML bruts M0/M1 et CPU128 sont conservés compressés avec vérification du
retour à l'identique : manifest M0/M1
et manifest CPU128.
Les paquets `.trace` et `.gputrace` restent locaux dans
`build/qwen36-max-profile/`.
raw-manifest.json
en conserve chemins, tailles, empreintes et alias. Les commandes enregistrées
doivent être rejouées depuis ce worktree, avec ses artefacts figés.

## Cibles des prochains essais

1. Decode : projections EXL3 groupées et tête K6, puis experts et sorties.
   Conserver les mêmes quantifications et oracles numériques. Le nom grouped
   doit encore être ventilé par géométrie avec des données qualifiées.
2. Prefill long : matmuls experts segmentées, puis denses TensorOps ; relier
   les routes aux tuiles utiles avant de choisir une modification.
3. Hôte : construction/encodage et coordination des évaluations, en gardant
   distincts travail utile et attente de fin GPU. Les observations de replay
   ne suffisent pas à promettre un gain supplémentaire.

Non mesurés : saturation DRAM/ALU, coûts exclusifs de tous les dispatchs,
attribution shader des deux cas AR longs et ventilation exclusive draft/verify.
Aucune optimisation supplémentaire n'est implémentée à partir de ces résultats.
