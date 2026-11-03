# Flash Next : ce que Strata et Splash apportent encore, revue59R

21 octobre2026. Recherche de code et analyse de métadonnées, sans nouvelle
inférence ni mesure de débit. Cible locale : Qwen3.8 Flash Next EXL3
2,05bpw, M5/24Gio, MLX0.32.2, cache experts3Gio, applications ouvertes,
plafond processus12Go décimaux. Les résultats d'autres machines ne sont pas
des gains MLXL3.

**Pour DFlash2, les écarts les plus concrets sont le draft encore en BF16,
son attention construite avec concaténations et masque CPU, la vérification
du modèle cible encore largement par ligne, et le budget de vérification
adaptatif absent du chemin DFlash2.** Reprendre seulement un kernel MoE ne
reprend pas le pipeline qui permet à Splash de l'amortir.

## Versions et portée

| Projet | Révision consultée | Évolution depuis la revue56 |
| --- | --- | --- |
| Niko1221/Strata | `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4` | Quatre commits ; installation/Control, pas de changement dans les chemins Metal comparés. Ancienne révision `cbfd5b92` conservée. |
| yibie/strata-mlx | `992d5f16dcaffe4831735a158e0db054fc4f9160` | Deux commits : expériences e55/e56 et documentation ; moteur inchangé depuis `295c9659`. Projet distinct de Strata. |
| incoai/splash | `1482f6bbc2abf12375332725302019594150b6cb` | Intégration Flash Next et nombreuses modifications du20octobre, dont budgets adaptatifs, mémoire/SSD, projections et HC. |
| MLXL3 | PR34 `79e605fa`, plus prototypes locaux57/58 | Routage hybride56 publié ;57 compact seul négatif ;58 partage EXL3 compilé, GPU et débit non validés. |

Les inventaires GitHub sont complets et non tronqués. Les sources téléchargées
ne sont pas toutes auditées : les index distinguent explicitement la simple
collecte des fichiers et les régions effectivement relues. Aucun code amont
n'a été exécuté. [Strata et strata-mlx](measurements/flash-next-ssd-impl/opt-59-strata-research/sources.json),
[Splash](measurements/flash-next-ssd-impl/opt-59-strata-research/splash-sources.json).

## DFlash2 : même checkpoint, exécution différente

Le Hub donne toujours `incoai/Qwen3.8-Flash-Next-DFlash2` à la révision
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`,603065600 paramètresBF16.
Le `config.json` récupéré à cette révision est identique octet par octet au
fichier local :5couches, largeur2560, intermédiaire7680, captures
`[5,15,25,35,45]`, RoPE10000. Le fixture de test Splash à3couches n'est donc
pas la preuve d'un nouveau drafter plus petit.

Commandes de consultation : `hf models info incoai/Qwen3.8-Flash-Next-DFlash2
--expand sha,lastModified,config,safetensors --format json`, puis
`hf download ... config.json --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6`.
Seul ce petit fichier a été récupéré, aucun poids.
[MétadonnéesHub](measurements/flash-next-ssd-impl/opt-59-strata-research/dflash-hub-metadata.json),
[comparaison locale et calcul du payload](measurements/flash-next-ssd-impl/opt-59-strata-research/dflash-checkpoint-comparison.json).

| Partie | Splash actuel | Chemin Flash Next local | Adaptation utile |
| --- | --- | --- | --- |
| Poids du draft | `DraftCheckpoint.cpp::Builder::projection` convertit les projections BF16 en affine4bits/groupes64, garde normes, convolutions de base et codebooks stockés. | `flash_draft.rs::Draft::load/linear` charge toutes les matrices BF16 et utilise `matmul`. | Réutiliser l'infrastructure Q4 du dépôt après extension des formes Flash ; comparer quantification et kernel à un oracle indépendant. |
| Q/K/V et attention du draft | Projection QKV groupée, préparation norme/RoPE, attention BF16 split/reduce, KV circulaires2048. | Projections séparées, concaténations KV, masque BF16 matérialisé sur CPU dans `Draft::attention`; copies/évaluations dans `append`. | Ring2048, masque calculé dans l'attention et écritures des seules lignes retenues ; adapter au nombre variable de propositions. |
| Convolutions et résiduels | `draft_conv`, normalisations préparant les entrées du kernel suivant, GateUp avec épilogue. | `convolution` construit broadcasts/slices/multiplications et somme ; `residual_norm` est déjà fusionné. | Porter les fusions encore absentes en conservant les points d'arrondi BF16. Ne pas présenter la fusion résiduelle déjà présente comme nouvelle. |
| Sélection | Top16, transitions des candidats, propositions et confidences restent dans les buffers GPU. | Top16 et transitions déjà GPU, puis `to_u32`; DFlash2 renvoie actuellement une confidence vide. | Compléter les probabilités de sélection et le budget GPU ; éviter les transferts seulement après avoir traité la vraie vérification cible. |
| Vérification cible | `Qwen4ExpTarget::addVerify` traite les lignes dans une même organisation de scratch/dispatchs ; les lignes désactivées ne routent aucun expert. | `verify_draft` appelle `Layer::prepare` pour chaque ligne ; seul le MoE est regroupé. `Linear::forward` sérialise plusieurs lignes EXL3 de référence. | Partager les poids des projections réellement indépendantes entre lignes ; garder les dépendances causales et les commits exacts. |
| États GDN | Tape compacte des entrées/clefs/valeurs/gates et application différée du préfixe retenu au pas suivant. | `verify_draft` conserve un `LayerState` après chaque préfixe ; le GDN Flash produit des états complets successifs. | Adapter le principe de tape/replay au chemin Flash, avec parité de chaque état et rejet. Les mécanismes GDN d'autres familles ne valident pas ce port. |

Sources directes : [préparation des poids](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/DraftCheckpoint.cpp#L19),
[draft et contexte](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/DFlashDraft.cpp#L246),
[attention spécialisée](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/DraftAttention.cpp#L140),
[tapeGDN](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/decode/gdn.metal#L23).

### Réduction de poids calculable, débit encore inconnu

L'en-tête local safetensors contient47matrices à quantifier selon la politique
du Builder Splash :475791360paramètres. Le formatAF4G64 stocke36octets pour
64poids, échelle et biais compris. Les autres tenseurs gardent254548480octets.
Le payload théorique devient donc :

`475791360 / 64 × 36 + 254548480 = 522181120 octets`,
contre `1206131200 octets` BF16 ; économie de `683950080 octets`, environ684Mo.

Ce calcul est fondé sur les formes et offsets de l'en-tête, pas sur une
conversion effectuée. Il exclut alignement des images, allocateur, scratch,
KV et modèle cible ; ce n'est pas un picRAM mesuré. Une partie pourrait
éventuellement servir au cache experts, après nouvelle qualification mémoire.
Les4bits changent les propositions du draft : mesurer coût **et acceptation**.
Le modèle cible EXL3 doit toujours décider les tokens et retrouver ses états
exacts. Aucun changement de quantification cible n'est proposé.

### Budget adaptatif : la modification Splash la plus directement pertinente

La [PR487](https://github.com/incoai/splash/pull/487) remplace huit lignes
vérifiées systématiquement par un choix fondé sur les probabilités de survie
et le coût des cycles. Le modèle d'acceptation se réajuste toutes les64boucles
sur jusqu'à4096labels ; les coûts suivent les durées récentes. Le kernel
`draft_verify_rows` choisit les lignes, et le groupement MoE ignore les autres
avant toute lecture d'expert. Les formes de scratch restent fixes.

L'auteur annonce notamment+55% pour Flash Next dans un budget simulant16Go,
contre **son ancien mode huit lignes**, pas contre notre MTP1. Notre adaptatif
DSpark existe déjà en prototype, limité à trois propositions et profil calibré
hors ligne ; il ne constitue pas ce budget DFlash2 dynamique. Splash signale
aussi que son modeauto peut changer les sortiesgreedy entre exécutions ; sa
validation ne remplace pas notre oracle séquentiel exact.

Premier essai discriminant à prévoir : même draft/mêmes kernels, largeur fixe
puis budget adaptatif, compteurs de lignes effectivement vérifiées, experts
uniques lus, acceptation, temps cible/draft et séquence cible exacte. Séparer
cette comparaison du changement BF16→Q4 pour attribuer les résultats.

## SSD : ce qu'il reste à reprendre

### Coalescence et priorité des lectures

Strata possède `where_[layer,expert]` pour l'expert prêt **ou en cours** ; une
demande réutilise son slot et promeut ses morceaux encore en file. Splash
arrête la prise de nouveaux morceaux spéculatifs, garde ceux déjà lus, attend
ceux réellement indispensables et complète seulement le reste. Ses workers
prennent les jobs dans un index partagé ; le travail urgent précède le fond.

Ici `ReaderPool::note_late` compte seulement le retard. La demande peut
réserver un autre slot et relire l'expert en vol. Les lecteurs de demande
répartissent encore les records par indices `worker,worker+4,...` et attendent
le batch entier. Ce ne sont pas encore les mêmes mécanismes.

Plus important pour l'intégration : `SsdMoe::forward_rows` n'active le GPUHot
que si `depth==0 && prefetcher.is_none()`. LES et hybride ne s'additionnent donc
pas aujourd'hui. Retirer simplement cette garde serait incorrect : un expert
publié après le snapshot de slots n'a pas été calculé par le graphehot initial.
Il faut distinguer les experts présents au snapshot, ceux arrivés ensuite et
les vrais misses, sans relire les arrivées tardives.

[Strata promotion/demande](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/metal/src/engine/stream.cpp#L360),
[Splash réutilisation des morceaux](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/ExpertStore.cpp#L468),
[workers et priorité](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/SsdReaders.cpp#L104).

### Alignement réel, au-delà de F_NOCACHE

Les experts EXL3 locaux sont déjà des records contigus de1248012octets,
mais `destinations` les disperse dans12segments des banques. Cela ne garantit
pas l'alignement16Kio des sources **et destinations**. Splash utilise des
places de staging et lectures élargies aux pages, puis un kernel de chargement
dans son layout final ; ses prélectures sont découpées en morceaux512Kio.

L'absence locale de `F_NOCACHE` vient du rejet12F→12G, conservé. Une piste
distincte serait un staging borné et aligné, avec comparaison cachefichier
actuel / staging aligné / staging aligné sans cachefichier. Elle coûte une
copie et des octets de bord : elle peut perdre. Mesurer prélecture/demande,
temps exposé, octets retournés, CPU, RAM et swap ; ne pas appeler ces compteurs
des lectures NVMe physiques. Ne pas réactiver le flag seul.
[Disposition et alignement Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/ExpertStore.cpp#L28),
[constat de strata-mlx](https://github.com/yibie/strata-mlx/blob/992d5f16dcaffe4831735a158e0db054fc4f9160/docs/wiki/concepts/ssd-reads.md).

## Synchronisation et exécution : pourquoi un dispatch indirect ne suffit pas

Strata garde un graphe par forme de fenêtre, ping-pong et état de commit,
puis ne corrige que positions/grilles. Il encode les kernels dans un
ordonnancement commun. Le prototype57 local insère au contraire une primitive
indirecte par GU/down, avec clear du scratch, fermeture d'encodeur et
signaux/attentes d'événements. Ce coût peut dépasser le peu de travail évité
sur des routes absentes déjà masquées. Les micros57 montrent une régression,
mais n'isolent pas causalement chaque coût.

**Autre mécanisme Splash absent ici : la reprise du GPU après les passages
hôte.** Son backend distingue prise en chargeCPU, travailCPU et repriseGPU.
Les lectures decode sont marquées `HostStepGpu::Awake` : un petit kernel borné
maintient le GPU actif pendant le passage hôte. Un autotest vérifie qu'il voit
la libération ; sinon repli. Le GPU suivant attend toujours un événement de
fin du travailCPU. Ce mécanisme ne rend pas les lectures gratuites.

Son kernel utilise `#pragma METAL internals : enable`, une portée système
numérique et `coherent(system)` : ce n'est pas une simpleAPIpublique portable.
Le backend démarre aussi un nouveau commandbuffer après les écrituresCPU pour
leur visibilité. Il ne faut donc pas recopier aveuglément un spin ou le
handshake Strata dans un même buffer. Premier travail utile : mesurer sur le
M5 local la reprise après des durées d'I/O réalistes, puis un prototype borné
avec repli et contrôle consommation/chaleur. Aucun chiffre de délai amont
n'est transféré au M5 de cette campagne.

[Graphe Strata](https://github.com/Niko1221/Strata/blob/9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4/metal/src/engine/window.cpp#L1007),
[frontières hôte Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/MetalBackend.mm#L1142),
[limite du spin](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/host_step.metal).

## Cache global et résistance aux longs prompts

Strata partage une arène entre couches de même classe de taille. Splash
utilise des slots globaux par layout, un recyclage ARC entre récents/fréquents
avec historique des évictions, protège tous les experts du chargement courant
et traite les longs scans de préfill sans les promouvoir comme du decode.

Le cache local est par couche,53slots dans la campagne3Gio, LFU ouLRU.
L'allocation statique55 réglée surMAIN a déjà échoué surHOLD. Un cacheglobal
dynamique ARC est une hypothèse différente, pas la validation de55. Le premier
filtre serait un replay des mêmes routes, mêmes2544slots totaux et coût des
métadonnées compté, état initial explicite et contrôle hors réglage. Aucun
agrandissement à6Gio ni `mlock` de plusieursGo n'est justifié par cette revue.
[Recyclage Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/SlotRecycler.cpp#L162).

## Kernels : nouveautés, doublons et fausses pistes

- **Partage EXL3 des vrais doublons58** : déjà prototypé localement, NT1 pour
  singletons et MB2 pour les paires. Les routes53 avaient3,843/3,288experts
  communs sur20occurrences par paireMAIN/HOLD. Cela permettrait d'éviter au
  plus environ19,2/16,4% de décodages/lectures de poids de ces seuls experts,
  avant surcoût ; ni19% du modèle entier ni19% d'I/OSSD, déjà dédupliqués.
  OraclesGPU et gain réel58 restent à établir.
- **Expérience e56 de strata-mlx** : leurs essais Q2_0 indiquent que l'accès
  aux octets et les petitskernels comptent davantage que la table de décodage.
  Le layout en plans gagne7–9% sur gate à4/8tokens, sans gain mono-token
  correspondant ; il change l'ordre des sommes. Ce n'est pas un kernelEXL3
  à recopier. La bonne reprise est le diagnostic lecture seule / décodage /
  calcul, sur nos vraies formes et48couches, avant un nouveau layout. Gate/up
  experts sont déjà regroupés ici ; les variantes54 scratch/K2 restent rejetées.
- **Projections float et HC de Splash** : lectures vectorielles et grilles
  choisies selon forme, format **et nombre de cœurs** ; pas uniquement
  «GPU M5». Leur PR491 modifie l'ordre des sommes, donc ne respecte pas
  automatiquement notre contrat de routage. Leur PR493 retire une fusion de
  préparation HC parce qu'une étape séparée répartit mieux le travail :
  fusionner davantage peut ralentir. Les anciens rejets HC22 ne sont pas
  effacés ; seule une adaptation avec mécanisme différent mérite un nouvel essai.
- **QSA court** : le code local ignore déjà la sélection sparse quand
  `count<=512`; ce n'est pas un gain à réimplémenter. QSA GPU sans readback
  au-delà de ce seuil et préfill par couches concernent surtout le contexte
  long/TTFT, pas nos mesures decode courtes.
- **Mémoire dynamique** : strata-mlx e53 réduit fortement compression/swap
  mais peut trop vider les experts et perdre du débit ; défautOFF chez eux.
  Une pression basse et un compteur de hits haut ne prouvent pas que les
  experts ne sont pas compressés. Garder les métriques processus et VM.

[e56](https://github.com/yibie/strata-mlx/blob/992d5f16dcaffe4831735a158e0db054fc4f9160/docs/wiki/experiments/e56-expert-kernel-variants.md),
[projections491](https://github.com/incoai/splash/pull/491),
[fusion retirée493](https://github.com/incoai/splash/pull/493),
[pression mémoire e53](https://github.com/yibie/strata-mlx/blob/992d5f16dcaffe4831735a158e0db054fc4f9160/docs/wiki/experiments/e53-follow-pressure.md).

## Ordre proposé pour les essais suivants

1. Terminer la correctionGPU58 déjà préparée ; garder le compact57 seul
   désactivé. Aucun débit ne peut être déduit de sa compilation.
2. DFlash2 : isoler la conversionQ4 du draft, puis l'attention/caches
   spécialisés ; contrôle de l'acceptation et du temps cible à chaque étape.
3. Budget DFlash2 de lignes utile/coût, avec même draft, pour réduire les
   experts lus sur propositions rejetées. Ne pas confondre avec MTP1.
4. SSD : coalescence/priorité + intégration hybride/prefetch correcte ; puis
   alignement/staging borné. Mesurer aussi le coût de repriseGPU.
5. Replay cacheglobalARC à budget fixe avant toute banqueGPU nouvelle.
   Regrouper ensuite les projections cible indépendantes si leur coût domine.

Ce sont des essais à préenregistrer séparément, pas une nouvelle campagne
déjà exécutée. Les sources Splash sont Apache2.0 et StrataMIT ; tout port de
code doit garder les notices appropriées. Les mécanismes Q4/attention/selector
déjà repris dans le dépôt pour35B/27B ne prouvent pas leur intégration dans
le prototype Flash Next, actuellement sous `cfg(test)`.

## État vérifié localement et limites

Pendant la recherche, les contrôlesCPU58 déjà lancés ont fini :108tests
sansfeatures passent/3ignorés ; release`mlx,chat`136passés/139ignorés.
Clippystrict, fmt et builds passent. L'oracle de partition détecte une
mutation de la seconde ligne de paire puis repasse sur source restaurée.
Kani20positions symboliques/unwind21 reste nonvérifié après arrêt à4Gio ;
CBMC6.11 échoue à parser libc++Apple. Le CABI104cas sousASan/UBSan est compilé,
pas exécuté surGPU. Ces résultats ne valident pas Metal ni la vitesse58.
[État58](measurements/flash-next-ssd-impl/opt-58-shared-exl3/verification-status.json).

La CI du commit publié79e605fa : régressions, Linux, macOS/MLX et contrats
Flash réussis ; jobKani global marqué en échec après annulation de son étape
par arrêt du runner. Pas de CI globale verte, pas de run du code local57/58
non commité. [Jobs du SHA exact](measurements/flash-next-ssd-impl/opt-58-shared-exl3/ci-79-jobs.json).

59R n'a lancé ni modèle, ni benchmark GPU, ni conversion de poids. Aucun
nouveau gain publié, aucun push de prototype, aucune application remplacée.
Objectif15tok/s toujours non atteint. Commandes et mesures historiques :
[profil53](flash-next-ssd-profile-53-2026-10-20.md),
[résultats56](flash-next-ssd-strata-56-2026-10-20.md),
[résultat négatif57](measurements/flash-next-ssd-impl/opt-57-compact-plan/micro-summary.json).
