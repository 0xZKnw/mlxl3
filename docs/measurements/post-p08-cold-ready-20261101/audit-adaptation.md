# COLD-READY P1 — audit d'adaptation pendant GPU1

1 novembre 2026. Lecture seule du runtime gelé et des rapports déjà produits ;
aucun changement moteur, build, proveur ou lancement GPU par cet audit.
Préenregistrement : entrée COLD-READY-01-GPU1 « audit demandé explicitement »
dans `opti.md`. Les chronos128 sont encore en cours au moment de cette revue.

## Conclusion

Le port atteint les chemins attendus et les fallbacks examinés sont corrects.
Il existe cependant un coût concret non isolé : la décision de scinder repose
sur une disponibilité observée **avant** la construction et la soumission de W1.
Si la queue SSD se termine pendant ce travail CPU, le moteur conserve deux
vagues et une soumission supplémentaire sans nécessairement masquer du temps.
Les tests avec un job retenu prouvent le mécanisme et sa correction, mais ne
mesurent pas la fréquence de ce cas sur des lectures naturelles.

Autre limite importante : les 9,048–9,053 ms du crible sont une somme par sortie
sur 1071 lots en40sorties, soit26,775lots/sortie. Le quotient est environ
0,338ms de fenêtre moyenne par lot éligible historique, **pas9ms par vague**.
Ce calcul n'est ni un coût mesuré du candidat ni une promesse de gain.

## Chemins effectifs et périmètre

Sources : `native/src/main.rs`, `qwen4.rs`, `flash_draft.rs`,
`flash_draft_runtime.rs`, `ssd_moe.rs`, `array.rs`, `linear.rs`,
`native/ffi/mlx_bridge.cpp`, `disk_read_batch.h`, `disk_reader.h`.

- AR : bridge → NativeChatModel::Flash.forward → FlashNext::forward_inner →
  Layer::forward_lookahead → SsdMoe B1. Le préfill Flash actuel parcourt aussi
  forward_inner token par token ; ce candidat peut donc affecter son TTFT.
- MTP depth1 : session → target.verify_pair → Layer::forward_pair →
  SsdMoe::forward_lookahead avec deux lignes. La tête MTP elle-même n'est pas
  la source du bénéfice de streaming attendu ; la cible48couches est concernée.
- DFlash GPU1 fixe WIDTH=1, ADAPTIVE=0 : anchor+uneproposition donnent deux
  lignes ; verify_draft → forward_draft_lookahead → forward_draft_rows B2.
  Le B2 est effectivement observé dans verified_widths des rapports17.
  Des blocs DFlash≥3lignes ne prennent pas ce prototype, sauf sous-cohortes
  éventuelles déjà imposées par un cache trop petit ; à53slots la campagne
  B2 n'a pas cette ambiguïté.
- Le loader lit COLD_READY strictement et désactive le candidat avec T18a
  DYNAMIC_READERS=1. GPU1 rapporte COLD_READY=1, DYNAMIC_READERS=0,
  LOOKAHEAD_DEPTH=0, GPU_HOT=1, GPU_HOT_WIDE=1, SHARED_EXL3=0,
  WEIGHTED_EPILOGUE=0. lookahead_depth0 ne crée pas de ReaderPool : le test
  `prefetcher.is_none()` n'annule donc pas silencieusement la campagne.
- Au site d'appel, il faut rows≤2, depth0, aucunpréfetch, Hotprésent, plusde2
  jobs manquants et les flags compatibles. Un cache vide sansHot, les petits
  lots et les grandes lignes conservent le chemin stock.
- Le crible historique contient1071/1696lots avec jobs>2, soit63,15% ; c'est
  un plafond d'éligibilité de cette ancienne trace de lectures, pas la fréquence
  réellement scindée enGPU1. Les175vagues naturelles de l'oracle montrent que
  le prototype s'exécute, sans donner le taux de succès de chevauchementCLI.

L'inspecteur bridge vérifie l'environnement et les modes/goldens, mais laCLI
release ne publie pas les compteurs `cold_ready_waves` (cfgtest). Le chemin
appelant est confirmé par code et la qualification ; sa fréquence par prompt
et le nombre de fallbacks « tout était prêt » restent non mesurés.

## Coûts et pièges identifiés

1. **Disponibilité périmée avant soumission.** wait_ready(2) produit un seul
   masque. partition, indices d'activations, indices de slots, graphes gate/up,
   SwiGLU/down et async_eval suivent sans second contrôle. Cela reste correct :
   un slot prêt ne redevient pas en attente. Mais W1 peut démarrer après le
   dernier retour SSD. Le fallback initial `ready.all()` évite une scission
   seulement si tout était déjà prêt au moment du premier masque.
2. **Coût lecteur payé même sans vague.** Pour chaque lot éligible, begin
   duplique chaque File source via try_clone, copie les propriétairesMLX,
   alloue état/erreurs/contexte de lot et crée un groupeGCD. Les workers font
   une publication sous mutex et notification par job. Le stock utilise
   dispatch_apply_f et n'a ni ces duplicationsFD ni cette readiness. Les
   validations, metadata de sources/destinations et vecteurs d'IO existent
   aussi dans le stock : ne pas attribuer leur coût complet au candidat.
   Le fallback tout-prêt conserve ce surcoût de lecteur, bien qu'il calcule
   une seule cohorte. Aucun chronométrage de ce coût n'est encore disponible.
3. **Coût fixe de deux graphes.** La partition produit deux vecteurs et deux
   sélections d'activations ; deux ExpertSlots et chaînes de calcul remplacent
   une chaîneCold. Le concat prend trois entrées au lieu de deux. Le take de
   sortie et la somme restent uniques : il n'y a pas de scatter ou de somme
   supplémentaire cachée, ni de deuxième banque/copie des poidsEXL3.
4. **Cohortes et factories.** Les kernels gardent K2/MUL1/NT1/SG4/splits1.
   Les petites vagues changent les grilles, temporaires et, avec
   STABLE_EXPERT_KERNELS=0, LOCAL_OUTPUT_DIMS dans les clés de factories.
   Pas de bascule TensorOps/GEMM cachée ; une sous-occupationGPU ou un coût de
   clé/encodage reste plausible. Une chauffe ne garantit pas que tous les
   couples de tailles W1/W2 futurs aient déjà été vus.
5. **Recouvrement à confirmer physiquement.** W1 peut attendre derrière Hot
   dans le fluxGPU. Le signal readyCPU ne dit pas quand son premier dispatch
   commence. LecturesSSD et GPU utilisent la mémoire partagée ; le calcul
   lancé plus tôt peut aussi allonger la queueIO. Aucun effet de contention
   n'est chiffré ici.
6. **Profilage ancien à ne pas lire comme du SSD pur.** `profile_seconds[1]`
   encadre maintenant fill_ready, donc inclut construction/soumissionW1 en
   plus de l'attente. fill_ready ne crée pas actuellement le même span
   `disk_read_batch` détaillé que fill_many. Réutiliser l'ancien analyseur de
   lots sans ajouter des marqueurs adaptés ferait une attribution incorrecte.

Les deux problèmes de la première revue ont été traités dans le source gelé :
T18a force le fallback stock et l'échec de clonageHot est rattrapé avec drain/
synchronisation avant retour. Pas de nouveau défaut numérique ou d'ordre de
somme trouvé dans cette revue ; les qualifications physiques passées ne sont
pas une preuve de tous les entrelacements ou d'une panne du contexteMetal.

## Ce que les rapports17 permettent déjà d'écarter

AR/MTP/DF gardent les mêmes goldens. MTP etDF ont chacun8propositions et8acceptées
sur MAIN/HOLD dans les deux bras. DFlash garde les mêmes volumes demandés :
MAIN2 809 275 012B, HOLD2 708 186 040B, et les mêmes8blocsB2. Une variation de
volumeIO ou de nombre de tokens vérifiés n'explique donc pas ces pairescourtes.

Sur DF HOLD, le temps vérification monte de1,10464225 à1,191207417s (+7,84%),
mais le draft monte aussi de0,055400541 à0,06669421s (+20,39%) et le commit/cache
de0,022097208 à0,024558917s (+11,14%). Ces régions ne sont pas réécrites parCold.
Cela peut refléter une dérive globale ou des effets secondaires du runtime ;
ce n'est pas une preuve que toute la baisse provient de W1. Ne pas jeter ces
échantillons. Sur MAIN, les directions diffèrent. Attendre ABBA128 avant toute
conclusion de débit, sans déduire de l'acceptation100% de17tokens qu'elle se
maintiendra dans128tokens.

## Contrôle suivant le moins coûteux, après les chronos gelés

Avant de modifier la politique ou rejeter l'idée, une courte capture naturelle
avec compteurs/spans séparés suffirait à orienter la suite : lots candidat,
raison de fallback, jobs et occurrencesW1/W2, duréepréparationlot, attente2prêts,
constructionW1, async_evalW1 et jointure finale. Ajouter des instantanés prêts
avant/aprèsconstruction seulement en profilage, et relier début/fin des
lectures au premier dispatchGPU deW1. Séparer préfill/AR/B2 et MAIN/HOLD.

Selon le coût réellement dominant :

- Beaucoup de lots deviennent complets avant async_eval : comparer une variante
  qui évite cette soumission séparée devenue inutile, avec protocole et parité.
  Ne pas simplement jeter/reconstruire le graphe sans compter le travail perdu.
- Coût de préparationlecteur dominant : un bras « lecteurasynchrone mais jointure
  complète avant calcul » isole ce surcoût. Une seule duplicationFD par source
  distincte peut ensuite être examinée en conservant les propriétaires ; ne pas
  retirer les leases pour gagner quelques appels.
- Nombreuses petitesvagues peu utiles : mesurer leur coût selonB1/B2 et taille
  avant de choisir un seuil. Aucun seuil nouveau ou gain théorique n'est fixé
  rétrospectivement sur les chronos en cours.

La préparationW2 avantjointure est auditée séparément par l'autre agent.
Cet audit n'en autorise ni l'évaluationGPU anticipée ni un changement de code
pendant la campagne. Aucun temps CPU/GPU utile du candidat n'est inventé ici.

## Proposition de diagnostic minimal — patch non appliqué

`cold-diagnostic.patch` contient seulement une proposition de54lignes modifiées
pour SsdMoe::fill_ready. `git apply --check` passe sur le source gelé au moment
de l'écriture. Aucun fichier natif n'a été modifié par cette proposition.
Le parent doit préenregistrer, appliquer, formater et compiler après GPU1.

Un span `cold_ready_batch` compte les appels effectivement éligibles. Il porte
status, jobs, cold_rows, ready_initial, w1_rows, w2_rows, split, io_ok. Les
sous-spans `cold_ready_begin`, `cold_ready_wait`, `cold_ready_first_build`,
`cold_ready_first_submit`, `cold_ready_finish` donnent leurs duréesmurales
end_ns−start_ns. Deux sous-spans de snapshot après construction et après
async_eval enregistrent le nombre de jobs prêts et leur propre durée.
Ils utilisent l'API déjà existante ready_for_test, seulement si TRACE est actif.
Une erreur du snapshot est enregistrée `diagnostic_error`, sans modifier la
branche ni remplacer le résultat forward. Les erreurs des opérations réelles
continuent à être propagées ; une trace partielle reste `incomplete` ou
`failed_finish`. L'analyse doit refuser toute erreur/snapshot manquant.
Pas de hold artificiel, pas de nouvel eval, pas de décision conditionnée par
les nouveaux snapshots, pas de modificationW2, pas de nouveau flagCLI.

Le harnais existant n'accepte **pas** natural32 ni un nombre de tokens configuré
par environnement. Ses choix sont natural/context128, verifier64, fenced33,
quality17. Seuls natural/context appliquent StageRetention(false) aux phases
mesurées. quality inclut une phasefenced qui évalue les étapes intermédiaires ;
verifier conserve les stages. Le plus petit choix déjà disponible adapté à la
fenêtre naturelle est donc `MLXL3_PROFILE_KIND=natural`,128sorties/sixphases :
warmup, control_before, trace_1/2/3, control_after. Ne pas inventer un envtokens
ignoré. Créer natural32 exigerait aussi de changer le harnais et ses validateurs.

Sélecteur exact :
`qwen4::profile::flash_detailed_profile_preserves_full_outputs_and_states`
avec `--exact --ignored --nocapture --test-threads=1` sur le nouveau libtest.
Un seul diagnostic initial proposé : AR/MAIN, sans xctrace au premier passage.
Env supplémentaire : PROFILE_KIND=natural, FLASH_PROFILE_MTP=0,
FLASH_PROFILE_PROMPT=MAIN, PROFILE_OUTPUT=nouveaudossierabsolu vide,
SSD_COLD_READY=1. Conserver tous les autres contrôles GPU1 et leurs préfixes
MLXL3_, checkpoint,3Gio,4lecteurs, ancre/gardes courantes et identités nouvelles.
Le budget de durée reste borné par le superviseur ; les deux contrôles non
tracés encadrent trois traces. Un DF/HOLD B2 éventuel est une extension distincte
à décider sur ces données, pas un lancement implicite de cette proposition.

Analyse prévue : compter les lots tout-prêts au premiermasque, les splits dont
ready_after_graph==jobs, ceux encore partiels aprèsasync, et le groupe ambigu
partielavantasync/terminéaprèsasync. Exiger2≤ready_initial<jobs pour les splits,
monotonie des comptes, w1+w2=cold_rows et exactlytwo snapshots par split. Aucun
snapshot pour le fallback initial. Rapporter les comptes par préfill/décode,
trace et tailleW1/W2, puis les distributions de construction/soumission/jointure
et de surcoût des snapshots. Les événementsparents layer/output_step existent
déjà. Pas besoin d'un nouveau framework ou d'un événementgate pour ce premier
objectif : le dénominateur est l'ensemble des appels fill_ready observés.

Interprétation obligatoire : tout-prêt **avant** async_eval indique une fenêtre
expirée au point observé, sous réserve de l'audit de constructionlazy. Tout-prêt
**après** async_eval n'établit pas l'absence de chevauchement : des dispatchs
peuvent avoir commencé pendant cet appel. Encorepartiel aprèsasync prouve que
la queueIO survit à la soumissionCPU, mais ne date pas le début réelGPU.
Le profilage et les snapshots retardent eux-mêmes le thread ; leur plafond de
perturbation est inconnu avant mesure. La durée des snapshots et l'écart entre
contrôles/traces quantifient une partie de cette perturbation, sans permettre
de soustraire simplement un pourcentage aux comptes. Ce sera un diagnostic
instrumenté, ni un débitrelease ni une fréquence exacte du cheminrelease.
