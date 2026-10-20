# Flash Next SSD : synthèse des deux audits et des mesures 44

**Suivi45, après autorisation d'implémentation.** Les deux préalables ci-dessous
sont maintenant fermés sur les fixtures réelles. Une erreur hot après réservation
laissait des experts non remplis dans le cache : reproduite avant correctif,
elle invalide désormais les réservations tout en conservant les leases pending.
La reprise relit les poids et restitue exactement la sortie indépendante.

L'oracle reçoit maintenant le même hidden natif :744960 logits par draft,
RMSErelative DF2=0,0000994933 / DSpark=0,0000908974, propositions exactes et
écart maximal des trois confidencesDS=3,63512e-8. Une mutation de confidence
seule est rejetée, puis les données originales sont restaurées et recontrôlées.
Les15 processus de qualité passent, bridge réel et annulation inclus, avec
zéro nouveau Swapout. Pic d'empreinte processus du bridge :8 370 559 480 octets.
Ces contrôles ne mesurent aucun gain de débit.

ValidationCPU :126 tests Rust passés/121 ignorés,570 Python passés, puis15cas
ciblés du superviseur étendu (sous-ensemble, pas585 tests distincts). Fmt,
Clippy strict `mlx,chat`, Ruff et build release passent. Kani vérifie le vrai
garde sur24 slots arbitraires, unwind27 :1247 obligations,12 inatteignables
dans std/modèles seulement,2covers atteints. La tentative réserve/publication
expire à180s ; MLX avec `--tests` rencontre une ICE ; CrossHair absent.
Ni GPU ni concurrence ne sont prouvés. Sources/binaires/fixtures :125empreintes
revérifiées après la campagne. [Preuves45 et limites](measurements/flash-next-ssd-impl/opt-45-audit-prerequisites/summary.json).
Le profilage45-P est maintenant clos ; les conclusions historiques44 sont conservées.
L'application installée n'est pas remplacée.

**Profil45-P.** Neuf processus joints, zéro nouveau Swapout : DSpark B1/B2/B3/B4
sur deux prompts CAL non-thinking, DFlash2 B4 thinking, puis OFF/MTP sur
MAIN/HOLD thinking. Les20 générations64tokens des drafts ont aussi leurs
états exacts. Les8 générations naturelles retrouvent tous les IDs de43,
le hash/texte H20 et248320logits finits ; leurs états complets ne sont pas
recopiés dans ce diagnostic.

La cible prend environ77% du decode DS/B1 et90% DS/B4, contre1–3% pour
l'appendKV. Les occurrences routées réutilisent un expert du même bloc dans
environ21%(B2),30%(B3),36%(B4) des cas ; l'I/O est déjà dédupliquée.
MTP naturel : vérification5,609/5,677s, tête0,227/0,227s, decode5,960/6,040s
MAIN/HOLD. La tête représente3,8%, plafond théorique d'environ4% si son coût
disparaissait entièrement. Ce n'est pas un gain de débit mesuré.

Les horodatages réutilisent les frontières existantes sans nouveau `eval`.
« Backbone » désigne sa construction/soumission ; la sélection paie son
exécution différée. Route/fill/import sont inclusifs et ne s'additionnent pas
à des duréesGPU exclusives. [Phases, routes et limites](measurements/flash-next-ssd-impl/opt-45-audit-prerequisites/profile-phase/summary.json).
KV groupé et tête MTP résidente sont différés pour ce cycle. Le prochain essai
est29/35 : NT2 sur les slots experts Flash, jamais exécuté sur ces formes,
distinct du NT2 dense10 et du MB2 de15. Aucun gain15tok/s annoncé.

Revue demandée le 20 octobre 2026, sur `1f38776f64e917e699e3554a97ce3caae78a222a`,
Qwen3.8-Flash-Next EXL3 2,05 bpw, M5/24 Gio. Le moteur est inchangé pendant
ce croisement. Les propositions ci-dessous restent des hypothèses à tester.

La priorité commune est de réduire le coût du vérificateur et sa pression
mémoire. Le second audit ajoute surtout **MTP entièrement résident**, **lookup
déjà disponible**, **réutilisation de la déquantification par expert**, et
**prélecture/décodage PLE**. Le premier apporte un meilleur filtre historique :
le B2 exact, l'anticipation de l'expert partagé et plusieurs regroupements
d'évaluations ont déjà été essayés sans gain confirmé.

## Sources et portée

Documents fournis, tous deux fondés sur le commit ci-dessus :

- `MLXL3_PR34_FLASH_NEXT_SSD_DFLASH2_DSPARK_RECHERCHE_2026-10-20.md`,
  SHA256 `412c4898b7805442e581b95b57eb9c07b93b9b2fc069704064a439c765bd5337` ;
- `MLXL3_FLASH_NEXT_SSD_DECODE_AUDIT_2026-10-20.md`,
  SHA256 `3ca99fa4c1a70ac6476e2875156060d7ac4e9f17292b7bff4b326ffae756b742`.

Leurs recommandations ne déclenchent aucune implémentation. Les sources
locales et les essais 15/15B, 27, 31, 43A/B et PERF-135/136 ont été recoupés.
Les tableaux ci-dessous distinguent mécanisme, coût inconnu et résultat réel.
Les nouvelles données 44 sont postérieures au point publié étudié par les
audits ; leurs mentions « GPU44 non exécuté » ne décrivent plus tout l'état local.

Les preuves de ce croisement sont dans
[`review-r3/comparison.json`](measurements/flash-next-ssd-impl/opt-44-draft-scheduler/review-r3/comparison.json).
Le petit manifeste indépendant DSpark y est archivé ; les poids, hiddens et
logits binaires restent hors Git. Aucun modèle supplémentaire n'a été exécuté
pour cette revue, aucune preuve GPU nouvelle n'est revendiquée.

## Ce que les mesures 44 changent

Même unité figée `74fc95a4`, moteur `25b1f44a`, secteur, MLX0.32.2,
cache experts3 Gio/53 slots, allocateur512 Mio, quatre lecteurs,
overlap1/LES0, contexte maximal4096. Les prompts de vitesse restent courts ;
4096 est une limite, pas une longueur de prompt mesurée.

| Phase | Résultat et limite |
| --- | --- |
| Qualité affectée | 10 processus joints : 8 invocations natives et 2 têtes Pony indépendantes. Norme corrigée, modes de template, backbones, propositions et états contrôlés. Lacune sur les confidences natives détaillée plus bas. |
| Mutation de l'ancien résidu | Échec attendu détecté, source restauré et test positif repassé. |
| Bridge release | Budgets1/2/3/17 et annulation passent. Swapouts système+5,31 Mio, sous garde qualité256 Mio ; ce n'est pas une validation de vitesse. |
| Calibration DSpark non-thinking | Quatre préfixes0/3/1/2, deux prompts CAL distincts de MAIN/HOLD. Profil STS ajusté ; zéro nouveau Swapout dans les quatre bras. ECE du troisième préfixe dégradée après ajustement, conservée. |
| Vitesse non-thinking | 16/16 bras, 48 générations128 IDs exactes dont32 mesurées, zéro nouveau Swapout. Aucun gain global qualifié. |
| Vitesse thinking | 2/12 bras complets ; le troisième, MTP, est arrêté sur garde de swap et joint. Neuf bras non lancés, comparaison incomplète. |

Débits descriptifs de la campagne non-thinking, deux bras par mode et prompt ;
chaque bras contient une chauffe et deux générations mesurées :

| Mode | MAIN, tok/s | HOLD, tok/s |
| --- | --- | --- |
| OFF | 9,332 ; 8,732 | 8,834 ; 9,492 |
| MTP1 | 8,211 ; 8,592 | 9,815 ; 9,476 |
| DSpark fixe | 5,492 ; 5,165 | 7,667 ; 7,782 |
| DSpark adaptatif | 3,946 ; 7,783 | 8,831 ; 9,212 |

Le premier MAIN adaptatif a une forte dérive chauffe/mesures ; aucune cause
n'est établie et aucun passage n'est exclu. Sur HOLD, adaptatif/fixe est
positif dans les deux paires, mais ce résultat ne se reproduit pas sur toutes
les paires MAIN. Le surcroît de pic des drafts, environ1,80–1,85 Go, dépasse
également la limite préenregistrée1,5 Gio face à OFF. Le pic absolu maximal
est8 977 865 664 octets ; il ne représente pas toute la mémoire système.

Thinking : OFF7,927 puis DFlash2 7,772tok/s seulement. Le bras MTP suivant
dépasse256 Mio de nouveaux Swapouts système : maximum observé469 221 376
octets, soit447,48 Mio. L'attribution au seul moteur n'est pas démontrée.
Aucune relance ni changement de budget pour obtenir un meilleur chiffre.

Résultats complets :
[`speed-off/result.json`](measurements/flash-next-ssd-impl/opt-44-draft-scheduler/speed-off/result.json),
[`speed-off/summary.json`](measurements/flash-next-ssd-impl/opt-44-draft-scheduler/speed-off/summary.json),
[`speed-on/result.json`](measurements/flash-next-ssd-impl/opt-44-draft-scheduler/speed-on/result.json).
Ni15tok/s ni gain livré ne sont validés.

## Deux préalables de correction

**Oracle de confidence incomplet, confirmé.**
Dans [`flash_draft.rs`](../native/src/flash_draft.rs), le contrôle des744960
logits utilise le hidden CPU indépendant ; les confidences comparées à1e-4
sont aussi celles de ce hidden CPU. Les propositions issues du vrai hidden
natif sont comparées exactement, mais leurs confidences ne reçoivent qu'un
contrôle de compte/finitude/plage dans
[`check_flash_drafts.py`](../scripts/check_flash_drafts.py).
Les anciens résumés de « chaîne couplée » doivent être lus avec cette restriction.

Les trois valeurs natives existantes sont0,340676159/0,275671005/0,228091478,
contre0,341871483/0,275281668/0,226553651 côté référence CPU. Écart absolu
maximal0,001537827 ; trois propositions identiques `[16,15,11007]`.
Les hiddens diffèrent déjà dans les tolérances du backbone : cette observation
ne démontre ni une erreur du sélecteur, ni la cause du mauvais débit.
Elle ne fixe aucune nouvelle tolérance d'acceptation.

La fermeture utile est un oracle indépendant alimenté par **le même hidden
natif exporté**, avec tous les logits et confidences, puis une comparaison
séparée de l'effet des écarts du backbone. Ajouter des cas proches des seuils
de décision et une mutation qui altère seulement `actual_confidence`.

**Réservations après erreur hot, risque confirmé par lecture, non reproduit.**
Dans [`ssd_moe.rs`](../native/src/ssd_moe.rs), `reserve` marque les nouveaux
slots avant le remplissage. Les erreurs de prédiction et de lecture invalident
le cache ; les opérations fallibles de calcul/soumission hot antérieures
peuvent sortir avant ces gardes. `FlashNext::reset` remet les états de session,
sans invalider ces poids. Cette fenêtre était déjà signalée en43.
Une injection d'erreur suivie d'une reprise sur les mêmes experts doit fermer
ce point avant de complexifier overlap/lectures par vagues. Aucune corruption
de sortie n'a été reproduite dans ce croisement.

## Plan commun, sans doublons

| Ordre | Chantier retenu | Premier contrôle qui tranche | Antécédent ou limite |
| --- | --- | --- | --- |
| 0 | Oracle natif et erreurs de réservation | Même hidden vers oracle indépendant ; échec injecté avant fill, puis reprise exacte. | Correction et couverture, sans promesse de TPS. |
| 1 | Attribution du temps | Séparer cible, décision/readback, commit, projection/append KV et attente I/O exposée ; relever compression et Swapin/out. | Le poste actuel « verify+cache » ne mesure pas le SSD seul. |
| 2a | MTP résident | Carte GPU expert→slot stable, sans readback/réservation SSD après `warm_all`, avec repli sur invalidation et contrats de durée de vie. | 512 experts déjà chargés,609,38 Mio de payload. Ce mécanisme ne rend pas la tête gratuite en RAM. |
| 2b | Lookup Flash | Réutiliser `prompt_lookup`/`DFlashCopy` avec vérification Flash à captures optionnelles, puis tester copie de code et absence de répétition. | Helpers présents ; le wrapper CLI n'active le lookup que pour Qwen, pas Flash. Aucun drafter BF16 supplémentaire nécessaire. |
| 3 | KV draft | Grouper les matérialisations indépendantes ; capacité par chunks dans un essai séparé. | Dix appels `eval` ne prouvent pas dix attentes coûteuses. Aliases, RoPE, fenêtre2048 et rollback à conserver. |
| 4 | Mémoire GDN Flash | Adapter le journal de bloc local PERF-135 ; vérifier tous préfixes, convolution et forme Hv48/ratio3. | Déjà conservé sur Qwen3.6 ; économie de pic MLX215–255 Mio ancienne, aucune extrapolation Flash. Un état initial par bloc, pas par token. |
| 5 | Amortissement EXL3 ciblé | Profiler une forme dominante ; réutiliser l'ancien MB2 comme comparaison. M3/M4 ou groupes d'occurrences du même expert seulement si le mécanisme est nouveau. | Le B2 exact15/15B a déjà été essayé et retiré. Désactiver `reference_accumulation` change le contrat numérique. |
| 6 | Drafter Q8 sélectif et vrai parking | Q8 à mêmes slots, puis même mémoire totale ; décision avant backbone avec coût de maintenance/reprise. Recalibrer après changement. | Le préfixe0 actuel paie encore le draft complet. Ne pas garder simultanément MTP/DF2/DSpark en RAM pour un Auto. |
| 7 | Cache par couche et admission après commit | Rejouer les mêmes routes : valeur marginale des slots, vieillissement et suffixes rejetés, puis validation hors réglage. | Continuité de31, pas une nouvelle idée ; déduplication déjà présente et LFU par cohorte pas démontré mauvais. |
| 8 | LES/admission et pipeline disque | Mesurer misses utiles avant échéance, doublons pending et attente du dernier job ; ne retenir qu'un mécanisme à la fois. | LES41 négatif en confirmation ; partagé anticipé27 rejeté. Plus de workers ou Metal I/O ne garantissent rien. |

Les rangs2a/2b et3 sont des candidats localisés à départager par le profil,
pas trois gains à additionner. Une refonte commune `verify_block` devient
utile lorsqu'un composant réellement gagnant doit être partagé entre MTP,
DF2/DSpark et lookup ; elle n'est pas un prérequis pour écrire tout un nouveau moteur.

La réutilisation GPU par expert est distincte de la déduplication I/O :
le dispatch actuel traite les occurrences routées. Un kernel peut partager
le décodage d'une tuile pour plusieurs activations du même expert, mais
seulement si les traces montrent une multiplicité suffisante. Restaurer
l'ordre des dix contributions avant la réduction ; aucune accumulation
atomique réordonnée dans le target exact.

## Apports supplémentaires et objectifs distincts

- **PLE** : seize petites lectures possibles et décodage refait sur hit.
  Prélecture des lignes de la chaîne proposée ou cache décodé sont des pistes
  supplémentaires du second audit. Les clés, le biais, l'arrondi et l'historique
  après rejet doivent rester exacts ; leur poids temporel n'est pas mesuré.
- **Préfixe réduit en cours de vérification** : retirer seulement un suffixe
  après observation de vraies routes, en tronquant tous états/captures associés.
  Idée supplémentaire exploratoire ; elle ne récupère aucun travail déjà
  exécuté et vient après un bloc rentable et son instrumentation.
- **Contexte long** : QSA passe sparse à2052 positions, avec IDs de blocs vers
  CPU et spécialisations par `count`. Tester2051/2052/2053, séparément de la
  fenêtre draft2048. Cache KV sparse ne signifie pas stockage historique borné.
  Ce chemin n'explique pas les débits sur les prompts courts actuels.
- **TTFT et tours suivants** : éviter les têtes de vocabulaire intermédiaires
  du prefill, construire les KV draft par chunks et réutiliser un préfixe typé
  cible/MTP/draft/PLE/template. Objectif distinct du débit decode.
- **Métrologie** : compléter les octets logiques par PLE/embeddings et séparer
  I/O terminée/échouée/spéculative. L'attente exposée ne s'additionne pas aux
  temps GPU qui la recouvrent. Les inter-tokens servis depuis `pending` ne
  représentent pas le coût de production d'un nouveau bloc.
- **Profil DSpark** : associer contenu/binaire/options effectives, dont politique
  du cache et dtype du prédicteur ; équilibrer les prompts dans chaque bras.
  Deux prompts répétés en greedy ne constituent pas32 tâches indépendantes.
  La campagne44 est équilibrée et lie ses SHA hors runtime ; cela ne rend pas
  le profil générique portable entre machines ou budgets.

## Sources primaires revérifiées et limites

La documentation [MLX KV](https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html)
recommande les capacités par chunks et mises à jour de tranches pour éviter
les concaténations croissantes. La documentation
[évaluation paresseuse](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
rappelle le compromis entre trop de frontières et graphes trop grands.
Ces pages sont en0.32.4 ; le runtime local est0.32.2. Aucun gain local n'est
déduit de leurs chiffres ou de leur exemple Python.

Le [README Strata](https://github.com/yibie/strata-mlx) confirme copie depuis
la conversation, précautions de précision M5 et budget système revu après
swap sur un vrai Mac32 Go. Il distingue ses mesures avec experts partiellement
résidents de celles du M4 Max128 Go. Ces formats et contrats diffèrent de l'EXL3
local. La [présentation Apple M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
distingue les gains matriciels de prefill et le decode contraint par la bande
passante sur les architectures qu'elle mesure. Aucun multiplicateur transféré.

Les autres papiers des audits n'ont pas tous été relus dans ce croisement ;
leurs propositions sont conservées comme pistes attribuées, pas comme résultats
reproduits. Aucun nouveau kernel, entraînement, contexte long ou correctif de
production n'a été testé. Les prototypes restent `cfg(test)`, absents du flux
public DF2/DSpark ; l'application installée est inchangée.

## Vérification complémentaire46 : NT2 slots non retenu

L'essai46 a exécuté la piste29/35 jusque dans le vrai bridge. NT2 traite deux
tuiles de sortie ; il ne partage pas le décodage d'un expert entre deux tokens.
Les32cas physiques couvrent les formes2560→640×2 et640→2560, F16/F32,
strides,1/2/10/20routes et huit contrôles de mauvais slot. Ils sont exacts
contre le chemin résident indépendant. Les13processus de qualité passent,
dont logits/états cible, paire et MTP, drafts et bridge avec annulations.

Secteur, cache3Gio/53slots, un warmup et deux mesures128tokens par bras :

| Mode/prompt | NT1 tok/s | NT2 tok/s | Écart descriptif |
| --- | ---: | ---: | ---: |
| OFF MAIN |10,3923|10,4467|+0,523%|
| OFF HOLD |10,2426|10,1773|−0,637%|
| MTP MAIN |8,7421|9,0182|+3,158%|
| MTP HOLD |8,3348|7,8700|invalide : dérive interne16,06%|

**Aucun gain confirmé.** OFF manque le seuil1% sur les deux prompts ; le dernier
bras MTP dépasse la dérive5% autorisée. Pas de confirmation inversée ni de
répétition pour bruit. Huit processus terminés/joints, sorties128 identiques,
zéro nouveau swapout ; pics maximums7 146 608 544octets OFF et8 198 837 528MTP.
Le dispatch NT2, sa garde et son harnais ont été retirés après archivage ;
l'oracle des formes réelles et les contrôles du banc restent conservés.
La première tentative arrêtée après échec Ruff reste archivée séparément.

La preuve Kani de la garde retirée passe18obligations et6covers ; elle ne
prouve pas le kernel. La tentative du vrai test MLX s'arrête sur une ICE,
et CrossHair n'est pas installé. Les métadonnées et sources exactes sont
conservées dans [closure.json](measurements/flash-next-ssd-impl/opt-46-slot-nt2/closure.json),
[la fenêtre complète](measurements/flash-next-ssd-impl/opt-46-slot-nt2/speed-final/speed-result-0.json)
et [le patch retiré](measurements/flash-next-ssd-impl/opt-46-slot-nt2/retired-nt2-source/candidate.patch).

L'avis complémentaire reçu sur45-P est cohérent : les~94% incluent la cible
nécessaire. Sur MAIN, atteindre15tok/s à même acceptation exige1,7599s de moins,
soit31,38% de la vérification actuelle. Les10 810misses par occurrence deviennent
9 893jobs experts après déduplication ; les12 346 582 716octets comptés sont
logiques, sans mesure du trafic NVMe physique. Le cache uniforme et le partage
du décodage entre occurrences restent des pistes distinctes. La coalescence
avec une prélecture pending manque, mais ne s'applique pas au témoin LES0.
Le regroupement de tête43B reste rejeté ; ne pas refaire ce même mécanisme.

## Allocation par couche47 : qualité validée, gain général rejeté

Une capture froide/chaude sur un prompt séparé de MAIN/HOLD couvre le prefill
et127pas decode par génération, avec128tokens exacts face au bridge,248320logits
finits et136buffers d'état identiques. Le replay du vrai ExpertCache Rust
reproduit exactement les compteurs des48couches avant optimisation.

Le solveur répartit les mêmes2544slots entre28 et84par couche. Sur la seconde
phase decode de calibration, les chargements logiques passent de17419à16545
(−5,0175%). Cela ne mesure pas le trafic physique du SSD.

Le nouveau réglage expérimental `MLXL3_SSD_LAYER_SLOTS` accepte48capacités,
20..512par couche, avec une somme exactement égale au budget uniforme courant.
Il est absent par défaut ; `MLXL3_SSD_CACHE_GIB=3` reste le budget employé ici.
Le profil proposé est conservé dans
[selected-slots.txt](measurements/flash-next-ssd-impl/opt-47-layer-cache/selected-slots.txt).
Aucun profil n'est sélectionné automatiquement.

Le crible release128tokens, même binaire avec/sans profil, donne :

| Mode/prompt | Uniforme tok/s | Par couche tok/s | Écart descriptif |
| --- | ---: | ---: | ---: |
| OFF MAIN |9,11731|9,04743|−0,7664%|
| OFF HOLD |8,99984|8,95157|−0,5364%|
| MTP MAIN |8,98468|9,00133|+0,1853%|
| MTP HOLD |8,40335|9,70089|+15,4408%|

**Aucun gain général confirmé** : une seule paire dépasse1%, et le protocole
exigeait les quatre avant confirmation inversée. Le résultat HOLD est une
observation à conserver, sans le promouvoir en boost ; aucune répétition pour
bruit. Huit processus joints,24réponses exactes, zéro nouveau swapout ; pics
processus maximums7 137 335 176octets OFF et8 131 024 152MTP. Le cache contient
exactement3 174 942 528octets d'experts dans les deux configurations.

Les17contrôles physiques/de parcours passent :13avec profil, oracle cible
uniforme par défaut et trois refus CLI avant chargement des experts.
Les tests source donnent130Rust passés/124ignorés,107CPU passés/3ignorés et69Python
ciblés. Fmt, Clippy strict, Ruff et build `mlx,chat` passent. La mutation du
contrôle de budget est détectée puis la source restaurée repasse.

Kani vérifie le validateur numérique réel sur48usize etbudgetusize arbitraires,
sansassume/unwind49 :143obligations et4covers atteints, une obligation
inatteignable dans std::fmt seulement. Le parser expire à120s, le DP à180s,
et le vrai parcours MLX rencontre une ICE : ces parties restent non vérifiées.
Le premier refus CLI avait été attendu avec exit1 alors que le CLI retourne2 ;
l'échec du banc est conservé, puis les trois cas passent avec le bon contrat.

[Résultats et décision47](measurements/flash-next-ssd-impl/opt-47-layer-cache/runtime/closure.json) ·
[qualité](measurements/flash-next-ssd-impl/opt-47-layer-cache/runtime/quality-summary.json) ·
[replay et courbes](measurements/flash-next-ssd-impl/opt-47-layer-cache/allocation.json).
L'application installée reste inchangée ;15tok/s n'est pas atteint.

Le producteur de calibration refuse désormais tout profil par couche avant création
du rapport. L’oracle négatif détecte l’ancienne version et la capture uniforme
repassée produit le JSON intégral byte-identique. Le moteur release reconstruit
est bit-identique au binaire mesuré. Cette dernière capture qualité observe
1,0625Mio de nouveaux Swapouts système, sous la garde qualité256Mio ; elle
n’est pas une mesure de débit et ne change pas les zéro Swapout des huit bras47R.


## Experts partagés48 : correction validée, débit non qualifié, kernel retiré

Un prototype décode une tuile EXL3 une seule fois pour deux occurrences du même
expert et écrit les deux accumulations aux positions originales. Les24cas
partagés passent bit à bit, avec8mauvaises cartes et8mauvais slots détectés.
Le MoE réel froid/chaud/petits caches et erreurs/reprise passe aussi, sans
nouveau swapout. Kani sur20slots symboliques expire120s et le vrai MLX rencontre
une ICE ; aucun de ces chemins n’est présenté comme une preuve.

Le crible isolé0/23/47, B1/B2,48échantillons de16forwards, est non concluant :
dérive jusqu’à106,88%, y compris dans le contrôle B1 au code identique. Les
médianes B2 sont négatives sur les trois couches ; elles ne qualifient pas une
régression fiable. Aucun benchmark complet OFF/MTP lancé et aucune répétition
pour bruit. Processus joint en2,49s, zéro nouveau swapout, sorties exactes et
aucune lecture d’expert chronométrée.

Le prototype est retiré du moteur. Ses sources exactes et mesures restent
archivées ; l’oracle NT1 conservé couvre désormais40cas et8mauvais slots.
130Rust passent/124ignorés,87Python ciblés, format/Clippy/build mlx,chat passent.
L’oracle40cas repasse sur l’unité finale, joint/Swapouts0. Le code de production
est inchangé ; le binaire release porte une nouvelle identité Git et n’est pas
bit-identique au moteur47. Aucun gain global ni15tok/s atteint.

Le premier contrôle négatif48 échangeait seulement l’ordre des deux occurrences
du même expert, une permutation équivalente ; cet échec du banc est conservé.
Le contrôle corrigé échange des lignes de groupes différents et est détecté.

[Décision48](measurements/flash-next-ssd-impl/opt-48-paired-experts/closure.json) ·
[mesures brutes](measurements/flash-next-ssd-impl/opt-48-paired-experts/micro-raw.json) ·
[identité finale](measurements/flash-next-ssd-impl/opt-48-paired-experts/retained-runtime-info.json).


## Gates FP32 résidentes49 : gain composant, débit complet non concluant

L’option `MLXL3_SSD_RESIDENT_GATES=1` matérialise une fois les poids FP32
transposés des48routeurs cible (240Mio), conserve le chemin FP16 et laisse
la tête MTP inchangée. Elle est désactivée par défaut.

384cas/196608logits, probabilités,3840routes/poids passent bit à bit, y compris
FP16 et strides. La mutation de poids est détectée puis restaurée. Le micro
48paires donne19,506→14,892ms (gain composant30,99%,4,615ms économisées), avec
dérive0,40/1,63%, zéro Swapout et un processus joint. Un premier post-traitement
Python3.9 échoue sur zip(strict=True) ; le même raw est traité avec Python3.12,
sans relance matérielle ni changement de critères. Les17contrôles complets du
modèle/états/MTP/drafts/bridge/refus/default passent, tous joints/Swapouts0.

Le débit complet reste non concluant : le premier témoin OFFMAIN varie de
9,3911 à7,6184tok/s, soit23,27% de dérive, avant le premier candidat. Les127tokens
prennent13,5234 puis16,6702s. Le crible s’arrête ; aucun C/MTP vitesse ni nouvelle
confirmation. Mêmes IDs/texte, pic témoin7 140 693 944octets et zéro Swapout ;
les stats ne donnent pas les compteurs SSD par génération. Le relevé CPU fait
après ne permet pas d’accuser une application ou de conclure à une limitation
thermique. Le diagnostic49D est préenregistré pour isoler cache/lectures/attentes.

130Rust passent/126ignorés,105Python ciblés ; format/Clippy/buildmlxchat passent.
Kani du vrai parcours MLX rencontre une ICE et CrossHair est absent : nonvérifiés.
Aucun gain global ni15tok/s livré ; option conservée pour analyse. L’utilisateur
autorise ensuite jusqu’à12Go processus pour les futurs caches plus grands,
sans changer rétroactivement les critères49.

[Décision49](measurements/flash-next-ssd-impl/opt-49-resident-gates/closure.json) ·
[micro](measurements/flash-next-ssd-impl/opt-49-resident-gates/micro-result-v2.json) ·
[qualité](measurements/flash-next-ssd-impl/opt-49-resident-gates/runtime/quality-summary.json) ·
[témoin invalide](measurements/flash-next-ssd-impl/opt-49-resident-gates/runtime/speed-result-0.json).

## Diagnostic49D : la variation se concentre à la frontière des routes

Les9,3911 et7,6184tok/s viennent de deux répétitions du même témoin release,
sans MTP ni gates résidentes. Les127tokens prennent13,5234 puis16,6702s,
soit18,88% de débit en moins et23,27% de temps en plus. La sortie reste exacte,
la RAM autour de7,14Go et les nouveaux Swapouts à zéro. Les compteurs par
génération manquent dans cette capture ; sa cause précise reste inconnue.

Une capture native distincte de trois générations identiques observe le cache,
les attentes existantes et l’activité système, sans ajouter d’évaluationGPU dans
le chronomètre. Elle conserve les limites : diagnostic instrumenté, pas un
nouveau benchmark release. Les routes,248320logits finis et les états complets
restent bit identiques ; le vrai cache CPU reproduit tous les compteurs.

| Decode | Misses experts | Durée observée | Frontière routes, inclusive | Fill, inclusif |
| --- | ---: | ---: | ---: | ---: |
| Phase0 |18315|15,351s|9,233s|3,699s|
| Phase1 |18401|15,663s|9,469s|3,794s|
| Phase2 |18212|16,583s|10,540s|3,725s|

Le cache LFU conserve bien son historique entre générations. Il fait varier les
misses à routes identiques, mais la dernière phase ralentit avec moins de misses.
La frontière qui attend les routesGPU sur CPU augmente de1,071s ; le fill baisse
de0,069s. Ces horloges incluent du travail en attente et se recouvrent : elles
ne mesurent pas séparément le routeur, les kernels experts et le SSD.

Les52échantillons système sont valides. Le GPU AGX affiche52–64% Device/Renderer
avant modèle,50–59% après. Pendant les trois decode, Renderer médian34/36,5/41%.
Une concurrence graphique est plausible ; aucun compteur ne désigne une app
responsable, ne mesure la fréquenceGPU ou ne prouve rétrospectivement la cause49.
Un processus85,15s joint, zéro nouveau Swapout, pic7,431Go,66identités inchangées.

130tests Rust passent/127ignorés et113Python ciblés passent, dont8cas d’erreur et
nettoyage du sampler ; fmt/Clippy strict/buildmlxchat passent. Le vrai KaniMLX
rencontre une ICE ; CrossHair est absent. Aucun gain global supplémentaire établi.

[Analyse49D](measurements/flash-next-ssd-impl/opt-49-variance-diagnostic/analysis.json) ·
[trace](measurements/flash-next-ssd-impl/opt-49-variance-diagnostic/trace.json) ·
[observations système](measurements/flash-next-ssd-impl/opt-49-variance-diagnostic/system-samples.json).

## Cache plus grand50 et lecture du nouvel audit51R

Le plafond autorisé est12 000 000 000octets de footprint processus. Les paliers
4/5/6Gio passent l’oracle indépendant ;6Gio fournit107slots par couche,
6 409 789 632octets d’experts. PaireMTP/commits/états/session et vrai bridge
budgets1/2/3/17+annulation/reprise passent. Six processus joints/Swapouts0,
pic du bridge11 651 556 160octets. Le moteur49 reste inchangé, gatesFP32à0.

Le nouveau protocole repart d’un moteur neuf pour chaque bras, une chauffe128
puis une seule mesure128. Cela fixe l’historiqueLFU avant chaque mesure ;
aucun reset du cache de l’application n’est ajouté. Pour chaque prompt/mode,
deux pairesH-C/C-H, H=3Gio53slots/C=6Gio107, seize bras au total.

| Mode/prompt | Première paire, tok/s H→C | Paire inverse, tok/s H→C | Gains appariés descriptifs |
| --- | ---: | ---: | ---: |
| OFF MAIN |10,248→10,233|9,938→10,243|−0,15/+3,08%|
| OFF HOLD |9,660→9,960|9,703→9,895|+3,11/+1,98%|
| MTP MAIN |10,054→10,626|9,745→10,298|+5,69/+5,67%|
| MTP HOLD |8,986→9,508|9,518→10,003|+5,81/+5,09%|

Les32réponses128tokens sont exactes et les16processus joints/Swapouts0.
Pics mesurésC :10,380Go OFF/11,428Go MTP. Aucun gain global confirmé selon
les règles prospectives : OFF MAIN étendue des gains3,222points>3 ; MTP HOLD
dérive témoins5,917%>5. Les quatre pairesMTP favorables sont conservées comme
signal, sans relance pour bruit ni assouplissement après résultats.

Les compteurs système déjà capturés montrent beaucoup de compression et de
décompression, davantage sur6Gio. Leur fenêtre inclut chargement, prefill,
chauffe et mesure : ils ne permettent pas d’attribuer le phénomène aux seules
pages du modèle ni au seul decode. Zéro Swapout ne prouve pas une absence de
pression mémoire. Les jobsSSD par génération release et le tempsGPU exclusif
restent nonmesurés ; la cause initiale49 n’est toujours pas précisément attribuée.

Le nouvel audit utilisateur,1717lignes/SHA8d4cf718, est lu intégralement et
recoupé avec le code : réductionEXL3 exacte par rangement du scratch, découpage
B4parunion, traitement séparé singles/paires48 et garde de calibration des
prompts sont des pistes distinctes. Le tableau de cohortes et les densités48
sont recalculés depuis les vraies archives. Aucun prototype issu du rapport
n’est lancé pendant les mesures. La résidenceMLX fait l’objet de la revue52D,
avec l’antécédent denseQwen27-05 négatif explicitement conservé.

45nouveaux tests du banc et160Python pertinents passent, y compris vrais
processus factices frais, mauvaise sortie, mémoire, silence, interruption et
jonction. Mutation du plafond mémoire détectée puis restauration byte exacte.
Ruff/format etYAML passent ; sourceRust/Metal/FFI inchangée, binaires49 réutilisés
par empreintes104. CrossHair absent, aucune nouvelle preuve formelle revendiquée.

[OFF50](measurements/flash-next-ssd-impl/opt-50-larger-cache/baseline-closure.json) ·
[MTP50](measurements/flash-next-ssd-impl/opt-50-larger-cache/mtp-closure.json) ·
[compression](measurements/flash-next-ssd-impl/opt-50-larger-cache/compression-observations.json) ·
[revue51R](measurements/flash-next-ssd-impl/opt-51-audit-review/review.json).
