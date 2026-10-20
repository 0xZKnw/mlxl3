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
Le profilage45-P reste à faire ; les conclusions historiques44 sont conservées.
L'application installée n'est pas remplacée.

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
