# AR r2 : attribution partielle récupérée

31 octobre 2026. **Diagnostic du harnais instrumenté, workflow interrompu,
parité complète du run non validée.** Aucun nouveau modèle, calcul GPU ou
build lancé pour cette analyse. Tous les exports et processus d'analyse
sont terminés et joints.

Le contrôleur r2 s'est arrêté sur la jointure xctrace de 60 s. Cinq résumés
natifs sont fermés ; `control_after`, `final-snapshot.json`, `complete.json`
et le footer de succès natif manquent. Le champ `parity:false` original est
conservé. L'audit `OBSERVER-AUDIT01` a en outre identifié la rétention
d'Arrays intermédiaires sous cfg(test), susceptible d'empêcher la donation
même dans les contrôles. **Ces chiffres ne sont pas un budget du moteur
release ou de l'application.**

## Fenêtre et calage

- PID 20329 ; thread du harnais TID **208681**.
- Binaire libtest SHA-256
  `f7ad8a297d244b5cc9a97c8c0f85e20e818ae6acc78a7586d4dee3caf5340e3e`,
  UUID **79A97D1C-1BE6-306E-B479-2900548D10D6**, base `0x104104000`.
  MLX UUID F949D357-CE50-3CDC-BD53-D49566E9F72C, base `0x106274000`.
- 60 transitions complètes disponibles à l'alignement initial ; sélection
  des **40 premières**, pas **68 à 107**, dont l'index est >=16. La sélection
  reste identique après correction de l'horloge.
- Origine `time-info` : **1793461292.131400108 s** ; elle diffère de la date
  de début TOC. Fenêtre sélectionnée : **9,443394166–13,628516458 s** dans
  l'horloge de la trace. Le dénominateur est 40 transitions/sorties AR.
- Correction native **−25,684 µs**, intervalle résiduel de containment
  **±1,333 µs**, contraint indépendamment par les lectures : **5 709 preadv**,
  **7 124 900 508 octets**, cardinalités et retours exacts. Les 331 lots
  mono-job donnent un écart durée wrapper/syscall de 625 à 2 666 ns.
- Vérification typée supplémentaire : 5 709 identités temps/TID/fd/offset/
  octets uniques et mêmes totaux dans la fenêtre. Six TID effectuent des
  lectures ; un index worker 0..3 n'est pas une identité OS fixe.

Preuves : `partial-attribution.json`, `typed-io-audit.json`,
`analysis-execution.json`, `final-data-checks.json`.

## Temps observés par transition AR

Les axes CPU/GPU se recouvrent. Les colonnes ne doivent pas être additionnées.
GPU actif signifie intervalle de commande observé, pas utilisation ALU.

| Intervalle | Nombre | Wall ms/sortie | CPU Running ms/sortie | CPU Running ∩ GPU idle ms/sortie |
| --- | ---: | ---: | ---: | ---: |
| Transition complète | 40 | 104,6280 | 38,9664 | 22,5327 |
| Deux soumissions async | 3 840 | 22,6124 | 22,3946 | 11,9846 |
| Readback source des routes | 1 920 | 35,9654 | 0,3125 | 0,00245 |
| Lot de lectures | 1 667 | 32,6872 | 7,8811 | 4,7834 |

Les deux async ont des rôles très différents :

| Rôle | Nombre | Wall ms/sortie | CPU Running ∩ GPU idle ms/sortie |
| --- | ---: | ---: | ---: |
| `slots.indices.async_eval` | 1 920 | 18,8015 | **11,9104** |
| Soumission du résultat Hot | 1 920 | 3,8110 | **0,07423** |

Le déplacement aux deux extrémités du fit temporel donne respectivement
11,8589–11,9618 ms et 0,07345–0,07494 ms. C'est une sensibilité aux extrémités,
pas une preuve de bornes globales par recherche continue.

Le readback des routes recouvre **35,8674 ms de GPU moteur** sur 35,9654 ms
wall ; le temps de copie mesuré est **0,00440 ms/sortie**. Son attente ne peut
pas être traitée comme 35,9 ms de GPU inactif supprimables.

L'union des preadv mesure 32,0823 ms/sortie ; le recouvrement GPU moteur des
lots vaut 5,5055 ms/sortie. Le tail après la dernière commande indépendante
est 27,1687 ms, dont 26,9449 ms recouvrent preadv. Ce sont des observations
POSIX et une borne conditionnelle à l'ordonnancement fixe ; elles ne prouvent
pas une lecture NAND physique ni la causalité SSD de tout idle du lot.

## Appels identifiés dans les soumissions exposées

Les ensembles CPU sont conservés séparément. Haute fréquence : 4 835 points
éligibles, dont **4 830 Timer Fired** retrouvés par temps/TID dans le raw et
5 points sans correspondance raw et sans pile exploitable. Les 4 830 points
périodiques sont communs à la série user-only. Rôles HF : 4 808 indices,
27 résultat Hot ; les 5 points inconnus appartiennent au premier rôle.

| Première frame MLX rencontrée dans la pile HF | Points |
| --- | ---: |
| `CommandEncoder::maybeInsertBarrier()` | 736 |
| `CommandEncoder::get_command_encoder()` | 634 |
| `CommandEncoder::dispatch_threads(...)` | 590 |
| `gpu::eval(array&)` | 557 |
| `eval_impl(vector<array>, bool)` | 481 |
| `CommandEncoder::commit(...)` | 200 |
| `CommandEncoder::set_input_array(...)` | 191 |
| `fast::CustomKernel::eval_gpu(...)` | 115 |
| `CommandEncoder::end_encoding()` | 106 |
| `BufferCache<MTL::Buffer>::reuse_from_cache(...)` | 94 |

Ces nombres sont des **points statistiques**, pas des nombres d'appels,
dispatchs, barriers réellement émis ou millisecondes exclusives par fonction.
Une même pile peut contenir plusieurs de ces fonctions ; le tableau choisit
une seule frame MLX par point. Il ne permet pas de convertir les parts en
une ventilation exacte des 11,91 ms.

Exemples complets dans `async-high-stacks.json` :

- À 9 445 811 958 ns : kernel puis `__kdebug_trace64`,
  `-[AGXG17GFamilyComputeContext memoryBarrierWithScope:]`,
  `CommandEncoder::maybeInsertBarrier`, `dispatch_threads`.
- À 9 448 312 041 ns : kernel puis `__kdebug_trace64`, allocation de storage
  IOGPU, début de compute pass AGX, `get_command_encoder`, `Gather::eval_gpu`.
- À 9 443 511 916 ns : hash de chaîne/recherche de library, `get_library`,
  `Compiled::eval_gpu`, `gpu::eval`, `eval_impl`, `async_eval`,
  `mlxl3_array_async_eval`, `SsdMoe::prepare_resident_hot`.

La première frame utilisateur est `__kdebug_trace64` dans **1 126 points**
périodiques ; avec les piles kernel HF, seulement 200 restent leaf directement
à ce symbole, les autres montrent le kernel au-dessus. Cela confirme une
composante trace/driver dans l'observation ; cela **ne mesure pas** combien
de ms seraient récupérées sans Instruments. Allocation tiny 333 points,
free 235 et hashsets/metadata MLX apparaissent aussi.

Les samples périodiques sont pris pendant des interruptions : attribution
au contexte Running juste avant l'interruption, comme dans l'analyse
historique, explicitement étiquetée `interrupt_predecessor`. Les points
Running directs et inconnus restent distingués. Les cinq autres threads
moteur sont rapportés séparément : callbacks de rétention/fin de commande,
recyclage du BufferCache et lectures y apparaissent. Leur présence ne prouve
pas une prolongation du chemin critique du thread d'inférence.

## Limite décisive des deux providers CPU

`sampling-provider-comparison.json` vérifie **15 645 timestamps/TID main
identiques**, aucune différence entre les deux tables raw sélectionnées.
Toutes portent `sample-type=0`, `Timer Fired`. Les tables du TOC sont bien
distinctes, mais n'offrent pas deux populations indépendantes de cadence.

Pour ces points, les poids du profil standard sont 1 000 000 ns ; ceux du
profil HF sont 100 000 ns, avec des points dérivés supplémentaires de poids
variables. La documentation XML indique explicitement qu'un poids est une
contribution statistique **sans durée CPU associée**. L'égalité des populations
et le warning de binding interdisent de conclure 1 ms contre 100 µs depuis ces
seuls poids. La moyenne Running/points vaut 99,626 µs ; elle ne garantit pas
une période fixe. Le trigger des points HF non retrouvés dans les raw n'est
pas établi : l'export `context-switch-sample` a échoué, exit 10, erreur interne.

## Qualité des outils et limites

- Dix tables exportées avec succès, commandes/XPath/SHA/exit/jointures dans
  `export-receipt.json` et `optional-export-receipt.json`. Un export optionnel
  a échoué explicitement ; il n'est pas présenté comme vide ou validé.
- L'analyse reprend une copie de l'outil historique dans ce nouveau dossier ;
  l'original, les anciennes mesures et les sources physiques 551 sont préservés.
  `analysis-source-provenance.json` conserve l'origine et les adaptations.
- Ruff check/format, py_compile, **10 tests** déterministes passent ; le
  selfcheck historique passe 3 375 requêtes d'union + 3 375 d'intersection sur
  domaine fini, 3 cas invalides et 5 corruptions XML. Tests additionnels : alias
  typés, export multi-schema, première sélection 40/index>=16, refus<16.
  Les mutations de seuil et de garde multi-schema sont distinguées par les
  oracles. Cela reste des tests/explorations finies, pas une preuve générale.
- Mutation réelle séparée : contrat bytes 1248012→1248013 dans une copie
  isolée, exports inchangés ; **rejet exit1** sur la comparaison indépendante
  de bytes à la première calibration. Source et échec conservés sous
  `bytecount-mutant/` et `bytecount-mutant.log`.
- Première analyse de piles interrompue sur une absence légitime de backtrace,
  preuve conservée `async-analysis-r1-*`. Correction : préserver les points
  inconnus explicitement, sans leur inventer une fonction. Exécution finale
  des sources formatées réussie et jointe dans `final-data-checks.json`.
- Premier driver d'export arrêté après time-info : Python système sans
  `hashlib.file_digest`, reçu conservé ; reprise Python 3.12 du venv réutilisant
  ce XML sans refaire sa capture/export. CrossHair déjà constaté absent dans
  ce même environnement ; reprise explicite de cette limite, aucune
  installation, aucune preuve formelle revendiquée.

Ces résultats permettent de cibler les soumissions du graphe dépendant des
indices et les coûts d'encodeur/driver/metadata, sous les conditions du
harnais r2. Ils ne quantifient pas le gain d'une optimisation ni le budget
release après correction de la rétention des stages.
