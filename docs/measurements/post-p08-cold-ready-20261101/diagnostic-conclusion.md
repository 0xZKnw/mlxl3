# Cold en deux vagues : diagnostic et audit final

1 novembre 2026. Essai `COLD-READY-DIAG-01`, après la campagne P1.

## Décision

Le candidat P1 est conservé comme **prototype prometteur, désactivé par défaut**.
L'audit n'a pas trouvé de mauvais flag, de fallback permanent ou de changement
de l'ordre numérique qui expliquerait artificiellement les résultats. Les deux
vagues s'exécutent réellement et disposent souvent d'une fenêtre I/O restante.
La correction qualifiée et le signal de débit justifient sa conservation ; ils
ne suffisent pas à promettre un gain fixe sur chaque prompt/mode.

Un placement améliorable est confirmé : W2 est construit après la jointure des
lecteurs. Son anticipation est une petite piste locale ; elle n'a pas été
implémentée ni mesurée ici. Aucun changement de défaut, d'application installée,
de release ou de publication. Aucun processus de test ne reste en cours.

## Conditions et identité

- M5 24 Go, secteur ; Qwen3.8 Flash-Next EXL3 2.05 bpw, checkpoint `65c89531`.
- Cache 3 Gio, 53 slots par couche, 48 couches, quatre lecteurs. GPU Hot actif,
  Cold Ready actif, lecteur dynamique/shared EXL3/épilogue pondéré/F02/GDN OFF,
  lookahead 0, buffers MLX 40 opérations/40 Mo, cache allocateur 64 Mio.
- AR MAIN : 87 tokens d'entrée, 128 sorties, 127 transitions decode. Une chauffe,
  un contrôle avant, trois traces naturelles, un contrôle après. Aucun fenced
  ni retard artificiel ; rétention des stages désactivée après chauffe.
- Binaire diagnostic : `9adf58319b4c2d2900d41848b9281deaad41f57bfb2d2e062cbaeb5a78f4c1d6`.
  CLI P1 conservé : `42dd74561ec12fea5248a0c1b5368b9da5c04e6b8a9633105a0775274121ce69`.
  Source de base `7978fceac30d991a8c06d72114fc932b2443f555` plus modifications locales.
  `identity-diagnostic-amended.json` et les reçus `diagnostic-source-recheck.json`
  (137 sources avant édition du CI d'analyse) / `diagnostic-binary-recheck.json`
  (trois binaires) fixent les identités. P1 original archivé dans `p1-runtime-source/`.
- Même ancre GPU1 `9fdf6028…`, sans réancrage. Autorisation utilisateur : lectures
  cumulées limitées à 1 Gio ; écritures 32 Mio et autres gardes inchangées.
  Diagnostic : aucune écriture swap nouvelle, cumul maximal 524288 B écrits et
  344129536 B lus. Ces compteurs sont globaux, pas des I/O attribuables au modèle.
- Test physique : 1 passé, 0 échec, 394 autres tests filtrés ; 115,62 s natifs,
  116,33 s supervisées, processus joint. Le refus d'admission antérieur à 256 Mio
  est conservé dans son rapport séparé ; il n'avait chargé aucun modèle.

## Correction

Les six phases ont les mêmes 128 IDs, hash et texte que le golden AR MAIN
indépendant. Le harnais compare aussi les **logits finaux et tous les états
finaux** bit à bit entre phases : 248320 logits finis, forme `[1,248320]`, plus
de 100 buffers d'état non vides. Cela n'est pas une comparaison à un oracle
numérique indépendant à chacun des 128 pas. La qualification P1 indépendante
sur cinq pas, le composant avec lectures retenues, les erreurs/reprise, la paire
MTP et DFlash restent les six contrôles physiques précédents, non répétés ici.

`diagnostic-analysis.json` valide les 1 258 570 événements et leurs relations de
parenté, durées, ordre, nombre et monotonie des snapshots ; aucun lot incomplet
n'est écarté silencieusement. Les données et leurs SHA-256 sont liées au rapport.

## Ce qui arrive réellement aux lectures

Les dénominateurs sont les appels éligibles de `fill_ready`, pas toutes les
couches et pas une fréquence garantie du CLI non instrumenté.

| Observation, trois traces réunies | Préfill | Decode |
|---|---:|---:|
| Appels éligibles | 7012 | 9782 |
| Déjà tous prêts au premier masque, une seule vague | 101 | 266 |
| Appels effectivement scindés | 6911 | 9516 |
| Tous prêts avant soumission W1 | 303 / 4,38 % | 705 / 7,41 % |
| Partiels avant, complets après soumission | 1330 / 19,24 % | 1843 / 19,37 % |
| Encore partiels après soumission | 5278 / 76,37 % | 6968 / 73,22 % |

Les pourcentages des trois dernières lignes portent sur les appels scindés.
« Encore partiels après soumission » établit une possibilité de recouvrement,
pas le début physique des dispatchs : W1 peut attendre derrière Hot. Le groupe
complet après soumission est ambigu, puisque le GPU peut travailler pendant
`async_eval`. Le groupe complet avant soumission a perdu sa fenêtre au point
observé, avec la perturbation propre à l'instrumentation.

## Coûts hôte, millisecondes par token

Sommes sur 261 entrées préfill et 381 transitions decode ; ces durées murales
hôte ne distinguent pas CPU actif, préemption, GPU utile et GPU idle.

| Intervalle | Préfill | Decode |
|---|---:|---:|
| Préparer/lancer le lot de lecteurs | 0,370 | 0,357 |
| Attendre au moins deux jobs prêts | 12,001 | 8,542 |
| Construire W1 | 0,697 | 0,717 |
| Soumettre W1 (`async_eval`) | 1,527 | 1,495 |
| Deux snapshots du diagnostic | 0,023 | 0,023 |
| Joindre les lecteurs après soumission W1 | 7,309 | 5,724 |
| `fill_ready` complet, inclusif | 22,030 | 16,961 |

Le total parent inclut les lignes précédentes : ne pas les additionner au total.
La jointure est une attente exposée au thread hôte, **pas du SSD critique pur** :
le GPU peut avancer pendant cette attente. Les lectures peuvent être servies
par le cache de fichiers ; ce profil ne mesure pas le NAND.

Le decode moyen des trois traces vaut 87,490 ms/transition, contre 85,369 ms
pour les deux contrôles non tracés (+2,49 %). Les contrôles dérivent eux-mêmes
de +1,98 % en temps. On ne peut donc pas soustraire simplement 2,49 % aux coûts
ou aux fréquences observées. Pas de capture GPU corrélée dans ce diagnostic.

## Classement des ajustements locaux

1. **Préparer le graphe W2 pendant la jointure restante.** Le span existant
   `expert_calculate_build` de W2 représente 0,533 ms/token decode et 0,551 ms
   préfill. La somme, lot par lot, de `min(construction W2, jointure restante)`
   donne **0,395 ms decode / 0,422 ms préfill**. C'est une estimation optimiste
   sous durées inchangées, environ 0,47 % de débit au temps decode du contrôle,
   pas un gain mesuré ni une borne stricte sur tout le chemin : les petits
   indices créés avant ce span sont exclus. Risques : contention CPU/lecteurs,
   attente de l'allocateur, erreur de durée de vie et surtout évaluation
   implicite en profilage fenced. Il faut garder W2 strictement lazy jusqu'à
   `finish` réussi, et conserver le placement actuel en fenced. Les caches,
   transactions, occurrences et l'ordre de somme ne doivent pas changer.
2. **Éviter une soumission W1 après expiration de la fenêtre.** Les soumissions
   de ces lots représentent **0,105 ms/token decode / 0,065 ms préfill**, soit
   environ 0,12 % au mieux pour ce seul poste decode. Jeter W1 déjà construit
   puis reconstruire une cohorte complète peut annuler ce petit bénéfice.
   Une vérification supplémentaire n'est pas gratuite ; aucun seuil ajouté.
3. **Alléger la préparation des lecteurs.** Tout le span begin ne vaut que
   **0,357 ms/token decode**. C'est un plafond artificiel en supposant pouvoir
   tout supprimer ; une part est déjà payée par le lecteur stock. Les FD et
   propriétaires MLX assurent les durées de vie : les retirer pour gagner des
   appels serait dangereux. Coût différentiel vs stock non isolé ici.

Ni la jointure de 5,724 ms, ni les 1,495 ms de soumission W1 ne sont des gains
automatiquement récupérables. Supprimer la soumission W1 supprimerait aussi le
mécanisme recherché. Ces ajustements ne ciblent pas directement les ~12 ms de
CPU actif pendant GPU idle du profil P08 : il reste à corréler ce chemin précis
avec les timestamps GPU avant une attribution de ce type.

## Débit P1 déjà mesuré, distinct de ce diagnostic

| Mode | Référence | Deux vagues | Variation brute |
|---|---:|---:|---:|
| AR | 11,210 tok/s | 11,664 tok/s | +4,05 % |
| MTP | 11,438 tok/s | 11,865 tok/s | +3,73 % |
| DFlash | 12,198 tok/s | 12,313 tok/s | +0,94 % |

Les 18 processus précédents sont complets et tous les résultats restent
conservés. AR contre le premier A seulement : +1,53 %, car le dernier A baisse.
DFlash HOLD : −0,66 % ; compression mémoire et dérive empêchent de présenter
les moyennes comme des gains causaux établis ou une absence universelle de
régression. Acceptations et sorties identiques ; lectures logiques DFlash
identiques. Détails : `all-modes-independent-analysis.md` et son JSON.

## Vérifications de l'analyse et limites

`analyze-diagnostic.py --self-check` : cinq cas positifs, 18 traces malformées
refusées, deux contrôles d'intégration (rapport incomplet marqué en échec,
résultat antérieur préservé). Une mutation autorisant la readiness décroissante
est détectée. Le parseur final relit aussi les six traces réelles et le golden.
Ruff format/lint, py_compile et parse YAML CI passent. Six avertissements C408
du premier lint ont été corrigés ; aucun défaut numérique natif changé.

CrossHair tenté, module absent : aucune preuve source du parseur. Le diagnostic
natif avait déjà passé fmt, Clippy strict, build MLX/chat et ce test physique.
Les limites Kani (interruption mémoire) et CBMC (libc++ non pris en charge) de
P1 restent consignées dans `cpu-verification.md` ; aucun proveur lourd relancé.
Ces contrôles ne prouvent pas tous les entrelacements Metal/lecteurs.

Le workflow ajoute les contrôles du parseur aux PR/push main. Le connecteur
GitHub ne retourne aucun run PR pour le SHA de base ; la version locale modifiée
n'est pas publiée et n'a donc pas de run distant correspondant. Pas de CI
distante annoncée verte. DFlash/MTP ne sont pas reprofilés par ce diagnostic AR.
