# Strata et Flash Next : comparaison des mécanismes, essai 56

Le premier prototype hybride attendait encore la fin du GPU avant de démarrer
les lectures SSD. Le correctif de lecture MLX supprime cette attente dans le
test de recouvrement. Les composants, le modèle complet et la CLI passent ;
la série sans MTP est terminée et le candidat gagne dans les quatre couples.
La série MTP reste à mesurer. Les contrôles de correction ne mesurent pas le débit.

Sources figées : **Niko1221/Strata `cbfd5b926b794b356195d3cb600191b0c3567540`**,
**yibie/strata-mlx `295c96594dd79cbd080052da8740b1a8e44d7d76`**. Ce sont deux
implémentations distinctes. L’inventaire GitHub n’est pas tronqué ; 26 fichiers
de contexte sont identifiés par URL et SHA-256 dans
[l’index des sources](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/strata-source-index.json).
La revue porte sur les chemins de decode, routage, streaming, groupement et
contrôle mémoire indiqués ci-dessous. Elle ne constitue pas un audit de tous
les formats ou de tous les backends de ces dépôts.

## Ce qui est présent et ce qui reste à reprendre

| Mécanisme dans Strata | Comparaison avec MLXL3 |
| --- | --- |
| Plan GPU déterministe par expert, entrées `(token, poids)`, grille indirecte ; `moe.metal:60–109` | Le prototype56 calcule les slots résidents sur GPU. Sa grille reste fixe et masquée ; il n’a pas encore le plan compact ni les dispatchs indirects de Strata. |
| Décoder les poids une fois puis les appliquer à toutes les entrées de l’expert ; `moe.metal:169–204,249–271` | Les lectures SSD sont déjà dédupliquées ici. Le partage du calcul pour les vrais doublons reste le prochain essai. Les singletons doivent conserver NT1 ; ne pas réactiver le MB2 systématique rejeté dans48. |
| Spécialisation pour 1/2/3/4 entrées et traitements séparés des activations ; `moe.metal:230–243` | Reprendre la spécialisation pour deux occurrences utiles. Le format EXL3, les Hadamard et les conversions FP16 imposent un autre corps numérique. |
| Restitution des résultats dans l’ordre du top-k ; `moe.metal:301–310` | MLXL3 restitue déjà cet ordre. Le prototype56 rassemble les lignes par indices, sans addition de zéros qui pourrait modifier leur représentation. |
| Un graphe de fenêtre préconstruit dans un command buffer ; échanges hôte par `MTLSharedEvent` ; `mtl_rt.mm:315–352` | Ce mécanisme **n’est pas implémenté** par 56. Notre graphe MLX est encore construit par couche. Supprimer la copie tardive ne revient pas à porter ce moteur Metal préconstruit. |
| Cache partagé entre couches, classes de tailles, slots protégés pendant une cohorte ; `stream.cpp:122–148,398–417,646–675` | Nos banques restent séparées par couche avec53 slots chacune. Le replay55 d’une allocation fixe a échoué sur HOLD ; un pool global dynamique serait un essai distinct, pas la validation du profil55. La protection des experts courants existe déjà dans `ExpertCache::reserve`. |
| Réutiliser une prélecture en vol et promouvoir ses morceaux dans la file urgente ; `stream.cpp:360–366,654–663` | **Pas encore équivalent ici** : `ReaderPool::note_late` compte les prélectures tardives ; la demande n’attend pas cette prélecture et peut relire dans un autre slot. La propriété de publication sûre existe ; la fusion/priorisation des demandes reste à tester. |
| Travailleurs SSD persistants, file urgente avant spéculative, abandon des travaux non commencés ; `stream.cpp:304–357` | Les lecteurs de demande utilisent déjà la file globale libdispatch, sans création de pool par couche. La sélection actuelle des jobs est statique par lecteur ; elle n’a pas l’ordonnanceur urgent de Strata. |
| Experts contigus et découpe optionnelle d’un expert entre lecteurs ; `stream.cpp:269–296` | Les records EXL3 sont déjà contigus et lus directement dans les banques avec `preadv`. La découpe parallèle d’un miss isolé est différente de nos quatre lecteurs répartis entre experts. Ne pas présenter ces lecteurs existants comme ce mécanisme. |
| `F_NOCACHE` et lectures alignées dans strata-mlx ; `paged.py:265–269,358–389` | `F_NOCACHE` a été essayé puis retiré volontairement dans12F→12G après un coût I/O important avec le lecteur d’alors. Son absence actuelle n’est pas une fonctionnalité oubliée à réactiver sans nouveau protocole. Les lectures alignées avec gestion des fragments seraient une autre variante. |
| Budget selon mémoire disponible avant chargement, réduction sous pression ; `paged.py:212–258` | Budget auto et plafond allocateur existent ici ; la campagne utilise explicitement3Gio. Le contrôleur de pression dynamique de Strata n’a pas été porté. Le RSS ou les octets MLX ne remplacent pas le footprint du processus ni les compteurs de swap. |
| Projection du routeur lisant les poids FP16 avec accumulation FP32 ; `fused.py:328–391` | Piste distincte de la banque F32 résidente49. Le contrat de sommation et les égalités de routage doivent être vérifiés avant adoption. Les règles de départage du top-k ne se recopient pas sans l’oracle local. |
| Sélection du token et embedding du brouillon sur GPU ; `mtp_chain.metal:27–101` | À examiner avec la spéculation adaptative après le backbone. Le port ne doit pas introduire une synchronisation CPU à chaque proposition. Les paramètres mesurés sur un autre Mac ne deviennent pas des coûts valides pour notre M5. |

Liens directs : [kernels MoE Metal](https://github.com/Niko1221/Strata/blob/cbfd5b926b794b356195d3cb600191b0c3567540/metal/kernels/moe.metal),
[streaming et priorités](https://github.com/Niko1221/Strata/blob/cbfd5b926b794b356195d3cb600191b0c3567540/metal/src/engine/stream.cpp),
[graphe et événements](https://github.com/Niko1221/Strata/blob/cbfd5b926b794b356195d3cb600191b0c3567540/metal/src/mtl_rt.mm),
[cache MLX](https://github.com/yibie/strata-mlx/blob/295c96594dd79cbd080052da8740b1a8e44d7d76/strata_mlx/paged.py),
[routeur MLX](https://github.com/yibie/strata-mlx/blob/295c96594dd79cbd080052da8740b1a8e44d7d76/strata_mlx/fused.py).

Les kernels CUDA partagent aussi le décodage entre les entrées d’un groupe
(`s2_expert_grouped.cu:853–911`). Leur calcul entier/Q8 et leurs transferts
PCIe ne décrivent pas le chemin Metal/EXL3 du M5. Les flags de mesure qui
remplacent des experts manquants par le slot0 donnent des réponses fausses ;
aucun de ces raccourcis n’est repris. La fusion routeur monodispatch signalée
incorrecte sur M1 dans `docs/MAC.md` n’est pas utilisée comme modèle de
synchronisation inter-threadgroups.

## Correctif56B et preuves disponibles

Dans MLX0.32.2, `contiguous()` construit une nouvelle primitive, même si
l’entrée est déjà contiguë. Notre FFI la construisait au moment du readback,
après avoir soumis le calcul résident. Cette primitive attendait donc aussi
le travail placé devant elle dans la file.

Le correctif attend directement l’événement de l’entrée lorsqu’elle est déjà
soumise et contiguë, puis copie ses octets. Les entrées non soumises ou stridées
conservent le chemin antérieur ; aucune attente nécessaire aux données n’est
supprimée. [Source MLX](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/ops.cpp#L6319),
[attente propre à l’array](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.cpp#L135).

Contrôles physiques terminés : lookup :104 cas/514 IDs,40 formes ordinaires et120
masquées,39 cas de lecture sur13 dtypes Metal et3 buffers Float64CPU, erreurs et
reprise avant/après réservation,42 combinaisons de MoE, fallback prélecture,
10 cas de CABI instrumentés ASan/UBSan. Le travail retardé reste effectivement
en vol à la lecture des routes et au début de la demande SSD après correction.
Ce retard artificiel ne mesure pas un gain de génération.

Modèle complet : cinq positions avec248320 logits bit identiques et48 routes
exactes à chaque position ; seulement deux états GDN capturés indépendamment.
Les autres activations ont la vérification historique de taille/finitude et
une tolérance RMSE infinie, donc aucune nouvelle preuve numérique stricte
n’en est déduite. Paires et sessions MTP : logits et tous les états cible/tête
bit identiques au parcours séquentiel. CLI : budgets1/2/3/17, annulations,
reprises, rejets de configuration et fallback passent.

Pic physique natif de la CLI : **8 416 336 424 octets**,18 rapports, aucun nouveau
Swapout pendant ces contrôles. KaniCPU20 slots expire à120s ; KaniMLX rencontre
une erreur interne et CBMC ne parse pas les en-têtes libc++ Apple. Ces
propriétés restent non vérifiées. Les tests Metal et les sanitizers ne sont
pas une preuve formelle ; libmlx lui-même n’est pas instrumenté.

[Qualité et limites](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/quality-summary.json),
[échec initial conservé](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/mechanism/result.json),
[identités et archives source](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/identity-v2.json).

## Mesure et suite

OFF puis MTP,128 sorties, MAIN/HOLD, ordre HCCH, un warmup et une mesure dans
chaque processus neuf. H=v1option0, C=v2option1 ; cache experts3Gio,
allocateur512Mio,4 lecteurs, LFU, prélecture désactivée. M5/24Gio,
macOS27.2, secteur, applications ouvertes. Minimum1% sur chaque paire,
dérive témoin au plus5%, étendue des gains au plus3points, aucune relance pour
bruit. Limite physique12 Go décimaux, garde swap256Mio cumulative toutes les2s,
aucun Swapout nouveau pour qualifier un gain. Les deux binaires, les sources,
les plans, les dépendances MLX et le contrôleur sont figés avant mesure.

Le partage EXL3 pour les vrais doublons vient ensuite. La fusion Hadamard/QMV
nécessite d’abord une analyse de l’arrondi et du travail dupliqué ; les kernels
GGUF de Strata ne résolvent pas ces contraintes à notre place. MTP/DFlash2/
DSpark adaptatif vient après ces travaux. P0/Xcode reste exclu. Aucun modèle
Strata complet n’a été exécuté ici et aucun chiffre de leur matériel n’est
présenté comme une performance de MLXL3.

## Résultats sans MTP

| Prompt | Témoins H (tok/s) | Candidats C (tok/s) | Gains appariés |
| --- | --- | --- | --- |
| MAIN | 8,582 ; 9,413 | 9,947 ; 9,764 | +15,91 % ; +3,74 % |
| HOLD | 9,005 ; 9,313 | 9,315 ; 9,797 | +3,45 % ; +5,20 % |

HOLD satisfait les critères préenregistrés par prompt (dérive3,42 %,
étendue1,75 point). MAIN dépasse les seuils de stabilité (dérive9,68 %,
étendue12,18 points) : le qualificateur global reste `inconclusive_drift`.
L’utilisateur demande de conserver et publier le gain observé malgré cette
dérive. Cette décision d’adoption expérimentale est distincte de la
qualification d’un pourcentage stable. Les quatre couples sont favorables ;
aucun couple n’est écarté et aucune mesure n’est relancée.

Seize réponses128 sorties (warmups inclus) conservent IDs et texte attendus.
Les huit processus sont joints ; pic natif7 144 626 128 octets, zéro nouveau
Swapout. Le maximum de Swapin cumulé depuis l’origine56 est7 061 504 octets,
sous la garde256 Mio. [Résultat brut](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/baseline-result.json)
et [décision de conservation](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/baseline-summary.json).

Le chemin hybride s’active par `MLXL3_SSD_GPU_HOT=1` ; défaut0 conservé le
temps des mesures MTP. Les cas prélecture active, profondeur de lookahead
positive ou plus de deux lignes suivent le chemin existant. Le correctif
FFI conserve l’attente de l’événement des données. Aucune application
installée n’a été remplacée.

## Vérification locale du code conservé

`cargo fmt --all -- --check`, Clippy strict CPU et MLX et le build release
`mlx,chat` passent. La suite CPU compte112 tests réussis/4 ignorés ; la suite
release MLX/chat134 réussis/134 ignorés. Ces tests normalement ignorés ne
sont pas déclarés exécutés par la suite : les contrôles physiques listés plus
haut ont été sélectionnés séparément. Les45 tests Python finaux passent,
avec162 configurations préprocesseur dans l’un d’eux et44 cas de
qualification/orchestration dans l’autre. Ruff et les déclencheurs CI
PR/push sont contrôlés. Les journaux conservent également les suites Python
antérieures, qui se recouvrent et ne s’additionnent pas.

Les mutations « publier un slot pending » et « accepter un gain nul » sont
détectées par les régressions, puis restaurées. Les commandes/diagnostics de
preuve restent dans les fichiers `kani-*-result.json` et
`formal-readback-result.json` ; leurs timeouts/erreurs ne constituent pas une
preuve. [Bilan local](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/final-local-checks.json).

## Interruption de la première tentative MTP et relance56M

La campagne56 d’origine est close : avant le premier témoin MTP, le compteur
Swapout avait déjà augmenté de3 965 779 968 octets depuis la fin du dernier
brasOFF. Le témoin a été arrêté et joint après2,009s, avant tout débit.
L’utilisateur indique avoir lancé puis fermé une autre instance MLXL3 et
demande une relance. Cette information explique le changement de contexte ;
les compteurs globaux ne permettent pas une attribution physique par processus.

Les huit brasOFF restent sans nouveau Swapout dans leur fenêtre. Ils ne
prouvent pas l’absence de swap dans l’intervalle qui suit. L’échec initial
et son origine cumulative sont conservés dans
[memory-interruption.json](measurements/flash-next-ssd-impl/opt-56-hybrid-hot/memory-interruption.json).

Le superviseur56G refuse désormais le lancement si le budget cumulatif est
déjà dépassé. Douze cas d’intégration exercent les vrais scripts et un enfant
jetable ; l’ancien code démarre cet enfant malgré le budget épuisé, le nouveau
le refuse avant `Popen`. Frontières à256Mio, compteurs manquants/booléens/en
recul, grands entiers et erreurs sont couverts. La suite affectée passe206
cas ; CrossHair est absent, donc cette propriété n’est pas formellement prouvée.

La relance56M réutilise exactement les binaires et oracles56, après20s de VM
stable et vérification d’absence d’autre moteur. Son compteur est distinct,
avec le delta de l’ancienne campagne conservé. Un contrôle30s après le dernier
modèle recherche aussi les écritures différées. Le protocole est consigné
avant lancement ; aucun chiffre MTP n’est encore publié à cette étape.

## Résultats MTP1 après fermeture de l’autre instance

| Prompt | Témoins H (tok/s) | Candidats C (tok/s) | Gains appariés |
| --- | --- | --- | --- |
| MAIN | 10,073 ; 9,884 | 10,551 ; 10,221 | +4,74 % ; +3,41 % |
| HOLD | 9,535 ; 9,507 | 9,707 ; 9,748 | +1,80 % ; +2,54 % |

Les quatre couples satisfont cette fois les critères d’origine : gain au
moins1 % partout, dérives des témoins1,912/0,302 % et étendues1,336/0,743
point. Statut **validated_gain**, sans dérogation nécessaire pour le MTP.
Les réponses128 sorties sont exactes,127 tokens de decode chronométrés.
MAIN conserve72 propositions/54 acceptées/72 blocs, HOLD78/48/78, identiques
sur tous les warmups et bras de chaque prompt.

Huit processus joints ; pic natif total8 177 751 296 octets, candidat au
plus8 128 828 744 octets. La campagne entière et les30s après le dernier
modèle conservent **0 octet de nouveau Swapout** ; Swapin cumulé80 855 040
octets, sous256Mio. Ces observations ont une durée bornée, pas une garantie
sur les usages futurs. Les132 identités sont inchangées ; aucun binaire
natif n’a été reconstruit entre la qualité56 et cette campagne56M.

[Qualification et contrôle postflight](measurements/flash-next-ssd-impl/opt-56m-mtp-after-app-close/publication-status.json),
[mesures brutes](measurements/flash-next-ssd-impl/opt-56m-mtp-after-app-close/mtp-result.json),
[VM après30s](measurements/flash-next-ssd-impl/opt-56m-mtp-after-app-close/stability-after.json).
Les gains OFF et MTP sont propres à leurs comparaisons ; ils ne s’additionnent
pas. L’objectif15tok/s n’est pas encore atteint. L’option reste explicite
`MLXL3_SSD_GPU_HOT=1`, sans remplacement de l’app installée.

La suite demandée est : plan compact et dispatch indirect Strata, partage
EXL3 des vrais doublons, puis fusion des prélectures/demandes urgentes.
La [conception57C](measurements/flash-next-ssd-impl/opt-57-compact-plan/design.json)
distingue le dispatch natif de Strata, la grille fixe de l’API MLX installée
et une primitive à examiner avec des frontières d’événements explicites.
Cette primitive n’est encore ni implémentée ni mesurée.
