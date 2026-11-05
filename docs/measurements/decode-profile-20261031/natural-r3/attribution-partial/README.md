# AR r3 : diagnostic partiel, stages désactivés après warmup

31 octobre 2026. **Workflow interrompu ; full_run_parity=false.** La trace
sauvegardée permet une attribution partielle. Aucun modèle, build ou calcul
GPU lancé par cette analyse ; tous ses exports et processus sont joints.

R3 a été interrompu par le faux refus RSS d'un collecteur en état Darwin E,
sans swap ni divergence rapportée. Le résultat d'origine conserve ses erreurs
de cleanup et son statut failed. Quatre phases sont fermées, mais il manque
`control_after`, `complete.json`, `final-snapshot.json` et le footer natif de
succès ; aucun de ces absents n'est remplacé par une validation inventée.

## Sources et qualification de la fenêtre

- Libtest **32cc1c366497b0385e4ea6594211a9bef5e8393e42fb42891edc0ef444de3483**,
  UUID **C7BF0A50-5036-358D-AB53-8E8233FF20D6**, base `0x102ba8000`.
  MLX UUID F949D357-CE50-3CDC-BD53-D49566E9F72C, base `0x104e98000`.
- PID **27154**, thread du harnais **291767**, identités résolues dans le XML.
- `stage-policy-evidence.json` : warmup rétention true/count 789 ;
  control_before, trace_1 et trace_2 false/count 0. Ces quatre résumés déclarent
  les 248320 logits/états finals byte-exact et 128 tokens. Le contrôle croisé
  existe avant l'attache ; la qualification complète six phases manque.
- TOC : **un seul provider standard 1000 µs**, sans piles kernel ni sampling
  de commutations. Aucune donnée haute fréquence ne vient de cette capture.
- 52 transitions complètes couvertes à l'alignement initial ; sélection des
  **40 premières**, pas **76–115**, index>=16. Sélection inchangée après fit.
- Origine `time-info` : **1793463830.061990261 s**. Calage indépendant par
  **5985 preadv /7469351820 octets**, correction **−26,989 µs**, intervalle
  résiduel **±1,375 µs**. Les 315 lots mono-job donnent des durées wrapper
  supérieures aux syscalls de 583 à 1583 ns. Le fit ne dépend pas de la GPU.

Preuves : `export-receipt.json`, `toc.xml`, `partial-attribution.json`,
`stage-policy-evidence.json`, `async-standard-report.json`.

## Temps observés par sortie AR

Le dénominateur est 40 transitions complètes. CPU et GPU se recouvrent : ne pas
additionner les axes. GPU actif est une union d'intervalles de commande,
pas une mesure ALU utile ou une ventilation exclusive par kernel.

| Intervalle | Nombre | Wall ms/sortie | CPU Running ms/sortie | CPU Running ∩ GPU idle ms/sortie |
| --- | ---: | ---: | ---: | ---: |
| Transition complète | 40 | 104,5704 | 38,4354 | 22,0897 |
| Deux soumissions async | 3840 | 22,3854 | 22,2612 | **11,8539** |
| Readback source des routes | 1920 | 36,7559 | 0,3286 | 0,00359 |
| Lots de lectures | 1696 | 32,5758 | 7,9900 | 4,8431 |

| Rôle async | Nombre | Wall ms/sortie | CPU Running ∩ GPU idle ms/sortie |
| --- | ---: | ---: | ---: |
| `slots.indices.async_eval` | 1920 | 18,6486 | **11,7676** |
| Soumission du résultat Hot | 1920 | 3,7368 | **0,08622** |

La sensibilité aux deux extrémités du fit donne 11,7136–11,8215 ms pour les
indices et 0,08519–0,08727 ms pour Hot ; elle n'est pas une preuve de bornes
globales sur tous les offsets continus. Le poste des indices reste observé
dans ce harnais stage-free ; les différences avec r2 ne mesurent pas un gain
causal, car provider, fenêtre et politique instrumentée ont changé.

Le readback recouvre **36,6258 ms de GPU moteur** sur 36,7559 ms wall ; la
copie mesurée vaut **0,004312 ms/sortie**. L'attente routes ne représente donc
pas 36,7 ms de GPU inactif récupérables.

I/O : union preadv 31,9595 ms/sortie, recouvrement GPU des lots 5,3963 ms,
tail après la dernière commande indépendante 27,1664 ms, dont 26,9297 ms dans
preadv. Sur 5985 appels, 457 finissent avant ce repère, 1395 commencent après,
4133 le traversent. Le tail est conditionnel à l'ordonnancement fixe ; ces
données POSIX ne mesurent pas la NAND physique et n'établissent pas que tout
idle du lot est causé exclusivement par le SSD.

## Piles CPU standard

1543 points du thread d'inférence dans la sélection, tous retrouvés dans le
raw comme `Timer Fired`, tous de poids 1000000 ns. **497 points** tombent dans
les soumissions exposées : **494 indices**, **3 Hot**. Ils sont attribués au
contexte Running immédiatement avant leur interruption, suivant la méthode
historique ; ce mapping dérivé est explicitement conservé dans les stacks.
Aucune absence de backtrace parmi ces 497 points ; quatre points d'autres
threads sélectionnés sans backtrace sont conservés comme inconnus.

| Première frame MLX rencontrée dans la pile | Points |
| --- | ---: |
| `CommandEncoder::maybeInsertBarrier()` | 73 |
| `CommandEncoder::dispatch_threads(...)` | 68 |
| `CommandEncoder::get_command_encoder()` | 62 |
| `gpu::eval(array&)` | 56 |
| `eval_impl(vector<array>, bool)` | 51 |
| `CommandEncoder::set_input_array(...)` | 26 |
| `BufferCache<MTL::Buffer>::reuse_from_cache(...)` | 17 |
| `CommandEncoder::commit(...)` | 17 |
| `CommandEncoder::end_encoding()` | 12 |
| `array::~array()` | 10 |

La leaf utilisateur est `__kdebug_trace64` dans 123 points, malloc tiny dans 41,
`eval_impl` dans 25, free dans 21 et une instruction AGX dans 17. La présence du
traçage/driver est donc visible, même avec le provider standard. **Ces nombres
sont des points statistiques, pas des nombres d'appels, barriers physiques
ou ms exclusives par fonction.** Ne pas multiplier les poids ou fractions
par 11,7676 ms pour inventer une ventilation exacte.

Cinq autres threads moteur sont séparés dans le rapport. Lectures, callbacks
de fin de commande/rétention et recyclage des buffers y apparaissent ; aucune
somme de leurs poids n'est ajoutée au CPU du thread critique. Leurs index
worker n'identifient pas des TID fixes entre lots.

## Contrôles et limites

- Huit exports réussis (TOC inclus), XPath exacts, SHA, délais 180 s, exit 0
  et jointures dans `export-receipt.json` ; pas de tentative HF.
- Outils r2 copiés dans ce nouveau dossier, origine/SHA/changements dans
  `analysis-source-provenance.json`. Adaptations : PID/fichiers, un seul
  provider standard, statut/limites stage-free. Anciennes mesures et sources
  physiques inchangées.
- Ruff, format, py_compile : réussis. Dix tests de parser/sélection/refus
  et mutations passent. Selfcheck : 3375 requêtes d'union et 3375 d'intersection
  sur domaine fini, 3 cas invalides, 5 corruptions XML. `verification.json`.
- Le contrôle négatif réel byte-count r2 est repris explicitement : mêmes
  gardes numériques/calage, seules métadonnées/provider adaptés ; mutation
  1248012→1248013 rejetée à la première calibration. Il n'est pas présenté
  comme une nouvelle exécution physique r3. CrossHair absent dans le même
  environnement, limite reprise sans installation ni preuve formelle.
- Analyse finale exit 0 en 6,25 s, joined=true ; `analysis-execution.json`.
  La durée de l'analyse n'est pas un benchmark du moteur.
- Trace_1 complète 13,2517 s contre 11,4656 s pour control_before. Sans contrôle
  après ni fin du run, on ne peut pas isoler précisément le coût de l'observateur.
  Les stages sont désactivés, mais spans/sampling/kernel tracing restent.

Ce diagnostic identifie encore les soumissions des indices comme poste CPU
exposé. Il ne valide ni un gain d'optimisation ni une équivalence complète
avec l'application release.
