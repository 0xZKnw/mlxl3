# LES OpenBMB : ce qui manque encore au port Flash Next

Audit du 19 octobre 2026, essai 37. **Le prototype 34 reprend une partie de LES ;
son rejet en débit ne permet pas de conclure que la recette OpenBMB complète
échoue ici.** Plusieurs différences restent réelles. La plus visible dans nos
preuves est le gaspillage de lectures : le trafic fichiers est multiplié par
2,09–2,13 sans MTP, pour seulement 32,8–34,3% de prélectures immédiatement utilisées.
Ce constat provient de mesures existantes, sans nouveau benchmark.

La priorité est de mesurer la qualité du prédicteur sur les *misses réellement
évitables* et son coût GPU, puis de tester une admission GPU persistante à budget
constant. Ajouter des lecteurs ou copier seulement plus vite ne suffira pas à
établir 15 tok/s. Les différences ci-dessous sont des constats de code ; leur
effet causal sur le débit reste à mesurer.

## Sources et portée exactes

- Moteur OpenBMB : [`a64932ec`](https://github.com/OpenBMB/llama.cpp-omni/tree/a64932ec615e1463158129feaf2f0752d0c07a2e),
  également tête de `main` lors de la recherche. Lecture de 23 fichiers source/documentation
  et du schéma LES ; aucune compilation ou exécution du code amont.
- Application : [`MiniCPM-V-Apps fa9d837e`](https://github.com/OpenBMB/MiniCPM-V-Apps/tree/fa9d837e3d38dfde7a9484ec1cc6bc2e5bc9b420),
  tête de `main` observée, 10 fichiers inspectés. Son sous-module référence
  [`tc-mb/llama.cpp-omni 96bbf617`](https://github.com/tc-mb/llama.cpp-omni/tree/96bbf61710d44421ea54e63f5194f9b17c20d877).
  Ce commit est absent du remote OpenBMB consulté ; il a été récupéré depuis son
  dépôt déclaré. Les 22 fichiers de code moteur inspectés sont **identiques en SHA-256**
  entre les deux pins. Les conclusions ne mélangent donc pas deux moteurs différents.
- Comparaison locale : baseline H20 `7a9a2120…`, source retenue HC20 sur
  HEAD `fb782d985972ab9f4eed2926389365c713811095`, puis sources **figées 34** du
  release `bdc77f588e7c8720d4af5158d80cf40cc4d316669da28acfc9de4488aeca9668`.
  Les copies archivées de `qwen4.rs`, `ssd_moe.rs`, `ssd_prefetch.rs` correspondent
  exactement au manifeste 34. Elles évitent de prendre l’instrumentation 36B
  non compilée pour le code mesuré.
- Modèle local : Qwen3.8-Flash-Next EXL3 2,05 bpw, révision
  `65c895314393431c09050b2e04e250836b3a6eb4`, M5/24 Gio, MLX 0.32.2.
  Protocoles 34/36A : secteur, cache 5 Gio/89 slots par couche, allocateur 512 Mio,
  quatre lecteurs, overlap 1, contexte maximal 4096, prompts courts MAIN/HOLD.
  Un contexte maximal 4096 ne constitue pas un test avec 4096 tokens de contexte.

Identités, empreintes et calculs :
[sources.json](measurements/flash-next-ssd-impl/opt-37-les-upstream-audit/sources.json),
[calculs des preuves existantes](measurements/flash-next-ssd-impl/opt-37-les-upstream-audit/existing-evidence-calculations.json),
[disposition des en-têtes du checkpoint](measurements/flash-next-ssd-impl/opt-37-les-upstream-audit/checkpoint-header-layout.json).

## Comparaison des mécanismes

Les liens locaux « C34 » visent les sources archivées du binaire mesuré. « Absent »
signifie absent de ce port, pas nécessairement utile sur ce modèle.

| Mécanisme amont | État local vérifié | Conséquence / décision |
|---|---|---|
| [Prélecture directe dans les slots GPU](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L821-L909) | **Absent pour la spéculation.** C34 `ReaderPool` lit dans 4 buffers CPU ; `import_ready` copie 12 segments seulement après réception des vraies routes. | Une copie et son admission restent sur le chemin critique. Les lectures de demande sont déjà directes depuis l’essai 14. |
| [Prélectures conservées dans un LRU unifié](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L1913-L1923) | **Admission différente.** Une prélecture demandée entre bien dans notre LFU ; une prélecture achevée non demandée ou tardive est recyclée. | Le travail inutilisé immédiatement ne prépare pas un token ultérieur. Sa valeur future n’a pas été mesurée. Garder toutes les mauvaises prédictions pourrait aussi polluer le cache. |
| [Prédictions guidant les évictions](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L2463-L2485) | **Absent.** `ExpertCache::reserve` protège les vrais experts du lot et applique son LFU vieilli ; aucun signal prédictif n’est consommé. | Notre variante « prédiction seule » 36B isole seulement un coût. Elle n’implémente pas l’équivalent de l’option amont `PREFETCH_READ=0`, qui conserve un usage du signal dans le cache. |
| [Capacité réglable couche par couche](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-model.cpp#L2141-L2187) | **Prévu 31, jamais testé.** Les 48 couches reçoivent uniformément 89 slots. | Tracer les courbes de misses par capacité sur un prompt de calibration distinct ; redistribuer les 4272 slots sans augmenter la somme. Les anciens essais négatifs Qwen3.6 ne testent pas ce checkpoint. |
| [Profondeur de prédiction / nombre de lectures distincts](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L128-L132) | **Partiel.** C34 trie toujours 16 candidats par ligne, filtre hits/doublons, puis admet au plus `min(depth,4)` lectures pour tout le modèle. | Les profondeurs 8/10 du test de correction ne testent pas 8/10 lectures en vol. La campagne release 34 ne comparait que 2/4. Augmenter le budget sans améliorer la sélection risque surtout d’augmenter les octets. |
| [Prédiction limitée aux lots d’une ligne](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/models/qwen35moe.cpp#L455-L465) | **Extension locale non isolée.** C34 prédit aussi les 2 lignes de vérification MTP, avec 32 candidats intercalés. | Comparer coût de prédiction, lectures et version complète avant de choisir OFF/une ligne/deux lignes. Le code amont ne démontre pas qu’une première ligne seule serait optimale ici. |
| [Signal du routeur suivant sur `attn_post_norm`](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/models/qwen35moe.cpp#L444-L473) | **Adaptation HC non calibrée.** Notre `next.predict` reçoit le mélange MLP-HC de la couche courante, pas une approximation du mélange appris de la suivante. | Les 4 flux, mélanges HC propres à chaque couche et le PLE peuvent changer la corrélation. Ce n’est pas une divergence numérique : seules les prélectures changent. Tester d’autres signaux nécessite des traces et un coût total, pas seulement un meilleur taux de prédiction. |
| [Cache conservé pour petits lots spéculatifs](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L2313-L2321) | **Déjà présent.** La vérification 2 lignes réutilise les banques de demande ; une cohorte d’au plus 20 experts est protégée. | Aucun passage systématique à un cache froid en MTP observé dans ce chemin. Si capacité < 20, le fallback sériel existant conserve l’exactitude. |
| [Calcul et lectures en parallèle](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-graph.cpp#L1641-L1702) | **Déjà présent dans H20**, puis corrigé 34 pour inclure les prélectures prêtes. Notre split suit la disponibilité et reconstitue l’ordre original ; l’option amont suit un rang fixe et vaut 0 par défaut. | Ne pas annoncer tout le gain de ce mécanisme comme un gain LES supplémentaire. Le découpage amont n’est pas une optimisation manquante à recopier aveuglément. |
| [Préfill distinct et préchauffage hors décodage](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/models/qwen35moe.cpp#L46-L84) | **Absent comme parcours groupé.** `forward_tokens` local appelle `forward` token par token, avec prédiction LES également active ; `reset` conserve les experts. | Un vrai préfill groupé est une piste importante pour TTFT, avec des formes/oracles nouveaux. Une banque de préfill supplémentaire ne doit pas s’ajouter silencieusement au budget. Notre cache est déjà naturellement réchauffé par le préfill. |
| [Regroupement physique des familles d’un expert](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/tools/moe/experts.py#L180-L219) | **Déjà présent.** Les 24 576 records locaux sont chacun contigus, 12 segments/1 248 012 octets ; un `preadv` de demande les disperse dans 7 banques. | Pas de repack complet nécessaire pour obtenir une lecture par expert. L’ordre *entre* records et l’ordre d’envoi des lectures restent deux questions distinctes. |
| [Quantification par couche et groupes de même format](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/models/qwen35moe.cpp#L46-L58) | **Format différent.** Les records experts locaux ont tous la même taille ; le modèle est le checkpoint EXL3 publié. | Requantifier changerait les poids et l’oracle 2,05 bpw. Ce n’est pas un correctif d’ordonnancement restant exact au checkpoint. |
| [DSpark avec tête entraînée et vérification groupée](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/common/speculative.cpp#L835-L914) | **Autre spéculation.** Notre tête native MTP propose 1 token ; les poids Markov/confiance DSpark dédiés à MiniCPM ne sont pas dans cette tête. | Le nombre de tokens proposés ne se transpose pas par un réglage. Le coût cible et les chemins de commit doivent rester mesurés séparément. |
| [Gestion des buffers experts hors mmap intégral](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-model.cpp#L1700-L1750) | **But déjà atteint autrement.** `Checkpoint` lit les en-têtes, garde les inodes inspectés et lit les payloads sur demande. | Le moteur local n’enveloppe pas tous les experts ou les 24,4 Gio de n-grammes dans un seul buffer GPU. Stripper les safetensors ne corrigerait pas ce problème, qui n’est pas présent dans ce parcours. |

Sources locales principales :
[C34 SSD](measurements/flash-next-ssd-impl/opt-34-les-ready/ssd_moe.rs.txt),
[C34 workers](measurements/flash-next-ssd-impl/opt-34-les-ready/ssd_prefetch.rs.txt),
[C34 modèle](measurements/flash-next-ssd-impl/opt-34-les-ready/qwen4.rs.txt),
`native/src/flash_support.rs:138`, `native/ffi/disk_reader.h:53` et
`native/src/checkpoint.rs:38`. Les lignes du code GitHub sont celles du checkout
Git, pas celles des aperçus web qui suppriment certaines lignes vides.

## Ce que nos chiffres permettent déjà de dire

Recalcul à partir des 8 diagnostics légers 34 : chauffe 64 puis 64 tokens générés,
dont 63 pas de décodage, un seul mode/modèle par processus. Même texte/hash 64 H20,
compteurs MTP exacts, zéro swapout système sur les processus terminés. Ces
mesures sont instrumentées ; leurs catégories de temps ne sont pas additives.

| Mode / prompt | Lectures sans LES (Go décimaux) | Lectures LES4 (Go) | Rapport | Prélectures immédiatement utilisées | Imports (ms/token décodé) |
|---|---:|---:|---:|---:|---:|
| Sans MTP / MAIN | 7,785 | 16,617 | **2,134×** | 3456/10533 = 32,81% | 1,2245 |
| Sans MTP / HOLD | 7,746 | 16,152 | **2,085×** | 3509/10244 = 34,25% | 1,2985 |
| MTP / MAIN | 8,455 | 12,862 | **1,521×** | 2887/6418 = 44,98% | 1,0067 |
| MTP / HOLD | 8,423 | 12,863 | **1,527×** | 3034/6592 = 46,03% | 1,0653 |

Les lectures de demande baissent effectivement : MAIN sans MTP passe de 6238
records à 2782, les 3456 autres étant importés. Mais 7077 records spéculatifs sont
lus puis recyclés, soit 8,83 Go supplémentaires. Ce gaspillage explique un risque
de contention même avec une attente de remplissage plus basse. Il ne mesure pas
directement un temps perdu de 8,83 Go/bande passante : le cache fichiers et le
recouvrement interviennent.

Le compteur `late_needed` vaut seulement 9/2 sans MTP et 0/0 avec MTP dans ces
diagnostics. Les seuls retards de lectures utiles ne décrivent donc pas le
problème principal de cette capture. Une lecture mauvaise peut finir à temps.
Les hits/misses LFU restent strictement identiques 0/4 **par construction de
l’admission sur demande** ; cela ne démontre pas que l’I/O utile reste identique.
En MTP, les misses comptent des routes, parfois doublonnées entre lignes : le
trafic physique unique de records doit être dérivé des octets, pas de ce compteur.

Le schéma OpenBMB affiche une couverture de prédiction de plus de 80% sur son
cas top 8. Notre ratio 32–46% mesure l’utilisation immédiate de lectures *après*
filtrage des hits/admission/achèvement, sur top 10 et un autre modèle. Ce ne sont
pas les mêmes dénominateurs. Il manque encore rappel des vrais misses, précision
par rang/couche, coût du gate/readback et utilité à plusieurs tokens de distance.

Les imports représentent environ 1,0–1,3 ms/token dans cette instrumentation.
Supprimer cette copie seule ne démontre pas le passage de 10,51 à 15 tok/s, qui
demanderait environ 30% de réduction du temps par token. L’intérêt du stockage GPU
persistant doit aussi se chercher dans les relectures évitées et l’admission.

Le contrôle causal release 36A a ajouté le témoin qui manquait dans le même
binaire : sans MTP, C0 = 10,1794 et C4 = 10,3236 tok/s ; avec MTP, C0 = 10,1803 et
C4 = 9,7651 tok/s. C’est une seule cellule par mode, donc un **diagnostic**, pas une
confirmation de gain. Le crible 34 à 10,51 reste un crible ; la confirmation OFF
a échoué à +0,0847%. Aucun de ces nombres n’établit une limite matérielle.

## Les lectures, le calendrier GPU et la mémoire demandent des distinctions

**Calendrier.** C34 soumet les vrais experts déjà prêts avant les lectures
manquantes, attend celles-ci, construit le reste du graphe, soumet `output`,
puis lance `next.prefetch`. Le travail chaud a donc déjà pu s’exécuter avant
l’admission des prédictions. Il reste le travail froid/partagé et l’attention
de la couche suivante pour les masquer. Il manque une chronologie
route-prête → armement → fin I/O → première consommation GPU. Avancer l’armement
pourrait agrandir la fenêtre, mais aussi faire concurrencer les lectures de
demande par de mauvaises prélectures. Ce n’est pas un gain statiquement acquis.

**Ordre et appels disque.** Nos quatre lecteurs de demande utilisent déjà GCD,
pas quatre nouveaux threads créés à chaque couche. Les workers spéculatifs sont
persistants. L’amont a également quatre lecteurs par défaut : le contraste
« 4 buffers globaux contre fenêtres par couche » concerne surtout le stockage et
sa durée de vie, pas 48 fois plus de lectures parallèles chez eux.
Les commentaires amont sur des misses triés par ID ne correspondent pas
systématiquement au `process()` courant, qui parcourt les positions de route.
De même, `accumulate_runs()` **compte** les séquences adjacentes ; le chemin lu
n’implémente pas leur fusion en une seule lecture multiexpert.

Inspection des en-têtes locaux seulement : tous les 24 576 records sont contigus,
mais seulement 413 des 511 paires d’IDs consécutifs de chaque couche sont adjacentes
dans le fichier. L’ordre physique observé commence notamment par 10, 100, 101…109, 11.
Un éventuel tri des jobs doit utiliser `(fichier,offset)`, sans toucher aux routes
ni à leur réduction. Son gain n’a pas été mesuré ; le repack complet serait
prématuré. Le `argsort_top_k` amont appelle lui-même un argsort puis une vue :
le nom ne prouve pas un kernel de sélection partielle plus rapide que notre tri.

**Budget d’un port plus fidèle.** Nos 4272 slots consomment 5 331 507 264 octets,
soit 5084,52 Mio ; les 4 buffers CPU ajoutent 4,76 Mio. Ajouter 4 slots GPU **à chacune**
des 48 couches coûterait 228,52 Mio ; 8 coûteraient 457,04 Mio et 16 coûteraient 914,07 Mio.
Une banque partagée de préfill 256 records de ce format ajouterait 304,69 Mio.
On peut envisager de réserver 4 slots dans les 89 existants plutôt que les ajouter,
mais le cache de demande passe alors de 89 à 85 : ce coût doit entrer dans le témoin.
Un LRU unifié peut garder 89 slots au total ; son admission spéculative doit gérer
pending, erreurs, cohortes protégées et durée de vie des lectures GPU.

La copie CPU de 32 avait justement évité de donner des pointeurs GPU aux workers.
La supprimer exige de vérifier ces contrats sur le vrai backend ; aucun résultat
Kani CPU antérieur ne prouve la sûreté de cette concurrence. Une comparaison
future de politiques de cache devra autoriser des hits/misses différents tout en
gardant logits/états identiques : imposer leur égalité rejetterait une optimisation
de cache correcte. 36B conserve cette égalité uniquement pour isoler les coûts.

**Cache fichiers et mémoire au repos.** `F_NOCACHE` permanent a déjà été rejeté
dans 12-F : le relancer inchangé serait inutile. Le mode amont par phase est une
autre option, surtout avec un vrai préfill groupé, pas une garantie RAM.
Le [`park` Metal amont](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/ggml/src/ggml-metal/ggml-metal-device.m#L1674-L1711)
libère les enveloppes Metal mais conserve les pages hôte : une baisse de mémoire
allouée GPU ne prouve pas une baisse égale de RAM. Notre réserve automatique 8 Gio
est évaluée au chargement ; l’override 5 Gio du benchmark et des applications qui
grossissent ensuite ne constituent pas une garantie générale contre le swap.

## Ce que l’application OpenBMB change dans l’interprétation

Le [README de l’app](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/fa9d837e3d38dfde7a9484ec1cc6bc2e5bc9b420/README.md#L179-L195)
décrit MiniCPM35B-A3B, 40 couches, 256 experts/top 8, format Q3_K `mix_q3gud_im-les`,
experts externalisés et modèle commun distinct. Nos 48 couches, 512 experts/top 10,
HC et EXL3 ne reproduisent pas cette charge.

Les [constantes effectivement utilisées](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/fa9d837e3d38dfde7a9484ec1cc6bc2e5bc9b420/MiniCPM-V-demo/Sources/Settings/ModelConst/MiniCPMModelConst.swift#L1067-L1160)
mettent la profondeur par défaut à **8**, malgré des commentaires disant 4. Le
préchargement de départ passe par des tables calibrées ; le choix courant d’une
table 96 peut être tronqué à la capacité du cache par le bridge, indépendamment
du nombre 32 affiché comme défaut Swift. Copier la valeur d’un commentaire ou
les IDs d’une table MiniCPM serait une mauvaise calibration pour Flash.

La même source précise qu’un ancien avantage annoncé de K4 était confondu avec
la capacité du cache. Les profils non uniformes restent des overrides, pas une
redistribution automatiquement activée partout. Les scripts privés cités pour
la calibration (`plan_layer_windows.py`, `analyze_hits.py`, sources des tables)
n’ont pas été trouvés dans les deux arbres publics consultés. Leur méthode et
les réglages exacts des chiffres publiés ne sont donc pas intégralement reproduits.

DSpark utilise une tête entraînée séparée, par défaut désactivée dans l’app
consultée ; le bridge fixe `p_min=0`, donc la présence d’une tête de confiance
dans le moteur ne signifie pas que le filtrage par confiance soit activé dans
cette configuration. La couverture du cache pour les lots de vérification,
la réserve mémoire du draft et le rollback sont des éléments distincts.

Les [résultats publiés](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/docs/les/lookahead-expert-scheduling.md#L19-L44)
séparent LES/chargement, quantification par couche et DSpark : sur iPad M4,
9,81 puis 14,92 puis 17,39 tok/s. Le 20,51concerne un iPhone 18 Pro Max et la recette
complète. Aucun multiplicateur ne s’applique directement à notre 10,51 ni ne
promet 15 tok/s. Le schéma temporel et les chiffres d’un autre modèle ne remplacent
pas les traces locales.

## Erreurs antérieures déjà corrigées, à ne pas compter comme omissions restantes

| Essai | Erreur / limite historique | État réel |
|---|---|---|
| 30 → 32 | Attendre la cohorte spéculative entière | Remplacé par un polling non bloquant et fallback exact.30 rejeté, jamais publié. |
| 32 → 33 | Résultats terminés occupant les workers ; budget consommé par des hits | Deux reproductions échouent sur 32, passent 33. Filtrage/dédoublonnage avant budget. |
| 32 → 33 | Projections prédictives par ligne et ancien top-k quadratique | Projection 1/2 lignes groupée et tri MLX pour les prédictions. Route réelle conservée. |
| 33 → 34 | Experts préchargés prêts attendant encore les vrais misses | Reproduction 4 prêts + 6 froids échoue avant, passe 34 ; imports avant séparation chaud/froid. |
| 33 → 34 | Profil gardant trop d’instantanés et la têteMTP inutilement | Ancien dépassement 283,39 Mio de swapouts conservé  ; 8 diagnostics légers 34 passent à 0 swapout. |
| 34 → 36A | Pas de témoin releaseC34 désactivé contreC34 actif | Contrôle ajouté  ; 8 bras/24 réponses exactes. Diagnostic, pas gain confirmé. |

Le LFU des experts fréquents, les banques fixes, les fichiers ouverts réutilisés,
les lectures groupées directes, la concurrence 4 et le recouvrement des vrais hits
existaient déjà avant LES. Les essais 17/27 de recouvrement et les essais de
kernels 22–25 ne doivent pas être rejoués sans changement précis. Le kernel NT2
sur slots SSD 29/35 reste **jamais prototypé** ; c’est une autre piste, pas un
mécanisme LES amont manquant.

## Ordre des prochains contrôles proposé par l’audit

1. Terminer et qualifier les traces 36B, sans changer la production, puis isoler
   OFF/prédiction/I/Orejouée/complet. Ajouter rappel des misses, précision par
   rang/couche, retards et calendrier d’armement. Conserver les oracles de taille,
   finitude, logits/états, texte et compteurs MTP ; rejeter les traces divergentes.
2. Sur une calibration indépendante, examiner l’entréeHC du prédicteur et les
   deux lignes MTP, puis la redistribution 31. Toute sélection est figée avant
   évaluation MAIN/HOLD. Tester le coût de la prédiction lorsqu’elle ne lit rien ;
   le distinguer d’une vraie protection prédictive du cache.
3. Prototyper la lecture spéculative directe et l’admission persistante dans les
   **89 slots existants**, avec états pending et protections GPU explicites.
   Mesurer aussi les octets gaspillés et les nouvelles misses, pas seulement le
   temps de remplissage. Ne pas appeler « exact » une sortie où les experts réels
   auraient été remplacés par les experts prédits.
4. Tester un ordre physique de jobs seulement si les traces justifient son coût,
   puis reprendre 35/NT2 si les kernels restent dominants. Traiter le préfill
   groupé à part : gain TTFT et gain de décodage ont des contrats différents.

Chaque futur essai exige son propre préenregistrement, correction avant vitesse,
témoin de même mémoire, source/binaire figés et modes sans MTP puis MTP. L’audit 37
n’autorise aucun gain présumé ni aucune addition avec les gains historiques.

## Vérification de cet audit et état de livraison

Commandes : `git clone --filter=blob:none`, `git ls-remote`, `git show`,
`git diff`, `rg`, `nl` et lecture des sources ; SHA-256 des sources comparées ;
analyse Python standard des JSON 34/36A et des seuls en-têtes safetensors.
Les assertions de cohérence passent : 24 576 records × 12 segments contigus,
48 gates F16[512,2560], comptabilité lectures/imports et 22 sources moteur identiques
entre pins. Cela vérifie des données finies et des identités, pas l’inférence.

Aucun nouveau test GPU, build, test moteur, solveur ou benchmark n’a été lancé
pour 37 ; aucune nouvelle preuve formelle n’est revendiquée. La campagne de qualité34 (30 tests physiques Rust + bridge) et ses limites Kani restent des preuves historiques.
Le plan 36B est suspendu : ses ajouts de tests sont **non compilés/non exécutés**.
Les sources courantes de travail ne doivent pas être présentées comme un nouveau
candidat validé. 35 reste différé. Aucun push de gain, release ou remplacement de
l’application n’a eu lieu durant cet audit.

Recherche 37 terminée ; implémentations et gains des mécanismes manquants restent
non vérifiés. Objectif 15 tok/s non atteint.
