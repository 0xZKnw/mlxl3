# LFM2.5 — calibration après la V2

Campagnes QUANT-2026-10-15-LFM25-25 à33. **Révision du15octobre : le candidat K8 est refusé pour publication ; les réglages officiels sont maintenant mesurés sur H100.** La V2 publiée reste immuable ; aucune V3 installée ou publiée. Les pilotes30–33 sont clos : aucun des nouveaux profils ne passe tous les garde-fous. Aucune conversion24B supplémentaire n’a été engagée ; les résultats et limites sont conservés ci-dessous.

## Réglages officiels mesurés sur H100 (30–33)

Même source et quantifier officiel épinglés, HQ K4/K5/K6, head16, scalesauto ; quatre profils, deux graines,29projections physiques (neuf experts et deux projections communes). Les poids décodés remplacent temporairement ces projections dans le modèle sourceF16, puis sont restaurés byteexact ; les logits source sont aussi vérifiés bitexact après chaque variante. Un contrôle négatif réel produit KL0,09049084. Validation disjointe :12×1024tokens,128positions scorées par fenêtre, vocabulaire complet128000 et calculF64. **Ces KL partiels ne sont pas les KL du modèle24B entièrement quantifié ni des mesures contre le teacherBF16 final.**

| Réglage | Pilote30,32lignes de calibration | Pilote32,250lignes complètes | Pire domaine32 / contrôle | Pire projection32 / contrôle |
|---|---:|---:|---:|---:|
| MCG standard | 0,0385834 | 0,0477042 | 1,0000 | 1,0000 |
| MCG scaled | 0,0452004 | 0,0377232 | 1,2213 | 1,4781 |
| MUL1 standard | 0,0502436 | 0,0475916 | 1,5400 | 1,0055 |
| MUL1 scaled | 0,0431620 | 0,0379929 | 1,1250 | 1,4644 |

Les pilotes30 et32 ont chacun232quantifications et96fenêtres KL complètes ; leurs durées sont174,96s et279,85s hors préparation Modal. Avec la calibration complète, MCGscaled réduit le KL partiel moyen de20,92%, mais échoue aux seuils préenregistrés de régression10% par domaine et projection. MUL1 n'améliore pas uniformément le résultat. **Aucun profil global ne déclenche une nouvelle conversion24B.** Ces expériences ne sont pas des benchmarks de vitesse. [Pilote30](measurements/quant-lfm25-settings30-20261015/pilot.json), [scores32](measurements/quant-lfm25-settings32-20261015/pilot-scores.json).

L'essai31 a capturé toutes les250séquences mais sélectionné un nouvel expert froid insuffisamment couvert en validation. Il s'est arrêté après72quantifications : rapportfailed, sélectionnulle, aucun résultat complet déduit. L'essai32 conserve exactement les experts déjà couverts du30, avec2048000routes par couche. [Échec31 conservé](measurements/quant-lfm25-settings31-20261015/pilot-first-failed.json).

L'essai33 teste MCGscaled sauf les descentes `w2` des experts, et MCGscaled sur le tronc seul. Il choisit uniquement entre les options de factorisation officielles ; le quantifier CUDA, les poids sources, le budget de bits, les captures et les seuils restent inchangés. Un petit wrapper partitionne les appels officiels scalaire/groupé par option, refuse des options contradictoires sur une Hessienne partagée, et restaure les fonctions même après erreur. Les contrôles globaux doivent reproduire32 à1e-8+1e-6×|référence| avant toute conclusion. Pilote, prévol CUDA/natif puis évaluation24B sont des étapes distinctes ; aucune fidélité finale n'est présumée.

Le pilote33 est terminé en220,29s :232quantifications et96mesures KL. Les164contrôles RMS/KL reproduisent32 exactement (écart maximal0). Le profil `mcg-no-down` atteint KLpartiel0,04587661 (−3,83%), pireprojection1,00809×, mais le domaine multilingue régresse de32,60%. `mcg-trunk-only` atteint0,04401518 (−7,73%), projections toutes≤1,00000×, mais le multilingue régresse de37,13%. Les seuils préenregistrés restent inchangés : sélectionnulle, aucune conversion24B supplémentaire. [Scores33](measurements/quant-lfm25-settings33-20261015/pilot-scores.json), [reproduction des contrôles](measurements/quant-lfm25-settings33-20261015/control-reproduction.json).

Les quatre applications30–33 sont arrêtées, zéro tâche. Solde Modal affiché après actualisation :16,68USD contre17,91USD avant ces essais (écart affiché1,23USD, pas une facture isolée). Aucun nouveau modèle publié/installé. Les prévols CUDA/natif du sélecteur mixte et les cinq tests finaux/holdouts/smokes d’un nouveau24B restent non exécutés : aucun candidat n’a passé le pilote. Le wrapper est un outil expérimental vérifié surCPU, sans validation numérique de son parcours CUDA complet. La PR reste en brouillon.

Vérification CPU33 :774tests passés/1Ling ignoré, puis97ciblés après correction de l'optionCLI candidate pour conserver `--profile` à l'authentification Modal. La mutation réactivant scaled sur une descente protégée est détectée. Ruff/compile/diff passent. CrossHair confirme précédemment2contrats inchangés d'arguments ; les4contrats du sélecteur restent non confirmés à20s/condition, sans contre-exemple. Les tenseursCUDA, les I/O et le KL ne sont pas formellement prouvés. [Preuves et limites33](measurements/quant-lfm25-settings33-20261015/cpu-verification.json).

## Résultat complet du candidat K8

Le convertisseur CUDA officiel1.5.4 a terminé en45min51,54s, avec source déjà en cache. ExpertsK4 et106projections communesK8 donnent **4,27856555bpw réels**, contre4,20971609 enV2 (+206303853octets). Les captures routées et les nouvelles traces n'ont pas été utilisées. Ce temps ne démontre pas un gain de vitesse face à la V2 dont la préparation incluait un téléchargement à froid.

| Protocole figé | KL V2 | KL K8 | Variation KL | PPL K8 | Gates KL/PPL |
|---|---:|---:|---:|---:|---|
| BOS,4088positions | 0,083458 | 0,080079 | −4,05% | 18,160152 | passe / échoue |
| Sans BOS,8192positions | 0,187928 | 0,153575 | −18,28% | 26,824301 | passe / passe |
| Français,1024positions | 0,139829 | 0,129810 | −7,17% | 64,622880 | passe / échoue |
| Python,1024positions | 0,103885 | 0,110854 | +6,71% | 11,607101 | échoue / échoue |
| Court,2048positions | 0,200784 | 0,214752 | +6,96% | 49,826615 | échoue / passe |

Les28fenêtres ont terminé en747,01s, avec16376positions scorées, mêmes IDs, enseignantsBF16, vocabulaire128000 et moteur355b5262 que la V2. Les plafonds figés avant conversion restent inchangés. Les fenêtres BOS6, principale8 et courte5 régressent de plus de20% enKL. **La recette n'améliore pas tous les usages et ne remplace pas la V2.** [Rapport numérique](measurements/quant-lfm25-v3-20261015/quality.json), [log](measurements/quant-lfm25-v3-20261015/quality.log).

Le compactage est sans perte sur les octets conservés et sur16384000logitsF16 testés ; cela ne prouve pas la qualité du quantifié. Les contrôles dialogue/CJK et les smokes supplémentaires du nouveau candidat restent non exécutés, celui-ci ayant déjà échoué aux gates. Les applications cloud sont arrêtées, zéro tâche. L'essai28 de promotion d'expertsK5 est annulé avant prototype/GPU à la demande de l'utilisateur.

## Sélection du pilote, avant les tests complets

Cette sélection a servi à la conversion expérimentale ci-dessus. Son échec final est conservé ; elle ne recommande plus ce candidat pour diffusion.

Le pilote est terminé : 216 quantifications de matrices d’experts et huit contrôles du tronc, sur validation indépendante. Les captures conditionnelles sont rejetées sur ce panel ; les conserver comme options expérimentales ne signifie pas les activer sur le nouveau modèle.

| Corpus | Capture | Ratio géométrique d’erreur | Pire projection | Décision |
|---|---|---:|---:|---|
| Original | Tous tokens | 1,0000 | 1,0000 | Référence retenue |
| Original | Routée | 1,1045 | 1,4570 | Rejetée |
| Original | Affinité p | 1,1056 | 1,4469 | Rejetée |
| Original | Affinité p² | 1,1126 | 1,4282 | Rejetée |
| Traces BF16 | Tous tokens | 1,0012 | 1,0295 | Pas de gain établi |
| Traces BF16 | Routée | 1,1027 | 1,4046 | Rejetée |
| Traces BF16 | Affinité p | 1,1040 | 1,4041 | Rejetée |
| Traces BF16 | Affinité p² | 1,1107 | 1,3971 | Rejetée |

Les deux sondes communes donnent des ratios K8/HQ de **0,27695 et 0,14172**, soit des baisses d’erreur de72,30% et85,83%. Le candidat conserve donc la calibration originale et les experts K4, avec les projections communes K8. Le surcoût estimé est206307328octets : **4,27857bpw compacts**, contre4,20972 en V2. Ces proxys ne sont pas des mesures KL du modèle complet. [Données du pilote](measurements/quant-lfm25-v3-20261015/pilot.json), [sélection figée](measurements/quant-lfm25-v3-20261015/selected-recipe.json).

Le premier prévol sur la fixture à embedding/routeurs construits échoue à4,014% RMS. La matrice de contrôle conserve cet échec : nouvelle/HQ2,180%, historique/HQ2,347%, historique/K8 **0,1923%** ; chaque cas a4096valeurs finies et32/32argmax identiques. La tolérance2% n’a pas changé et aucune fixture n’a été retouchée. Le K8 retenu passe sur la source miniature historique inchangée ; cela ne prouve pas une parité universelle entre moteurs. Les résultats complets sont conservés dans [la matrice des prévols](measurements/quant-lfm25-v3-20261015/probe-control-matrix.json). La publication reste conditionnée aux tests24B complets.

La génération source a produit64conversations BF16 en574,62s, dont39longues ; son remplacement des lignes aléatoires n’est finalement pas utilisé. Transformers5.16.1 utilise grouped_mm en prévol/prefill et bascule officiellement en batched_mm au décodage CUDA, toujours BF16 ; les38biais de routage FP32 restent exacts. Les deux holdouts de régression dialogue/CJK ne partagent aucun13gramme avec la calibration ni les64conversations générées. Ils restent réservés à l’évaluation finale.

## Ce qui justifie un nouvel essai

La V2 atteint KL0,08345755 avec BOS et0,18792807 sans BOS sur les protocoles historiques. Le pic de la cinquième fenêtre se reproduit dans ExLlamaV3 CUDA : ce n'est pas uniquement un problème du moteur MLXL3. GSS, scalesauto, refit, propagation des états et HQ sont déjà actifs. Le bilan et les limites interbackend sont dans [le rapport V2](lfm25-v2-results-2026-10-15.md).

Le code épinglé à `16a49792a3c93d8432d72e6c4bce800841566577` partage la Hessienne gate/up entre tous les experts d'un bloc. Il active tous les experts pendant la capture et utilise XᵀX sans affinités. La sensibilité de sortie est déjà pondérée par les routes ; elle reste conservée telle quelle. Les vrais IDs V2 contiennent39/250lignes de tokens uniformes aléatoires (79 872/512 000), audit24 bitexact avec un RNG indépendant.

## Recherche et choix

- [MoEQuant](https://arxiv.org/abs/2505.03804) distingue couverture entre experts et importance différente des tokens affectés à un même expert. Son objectif par affinité conduit à pondérer les lignes par √p avant XᵀX. Nous comparons séparément routes seules, p et p² ; p² correspond à l'erreur quadratique d'une sortie linéaire multipliée par p et n'est pas présenté comme l'équation du papier. Le [code des auteurs](https://github.com/chenzx921020/MoEQuant) et leurs résultats sur d'autres architectures ne garantissent aucun gain sur LFM/EXL3.
- Le [pipeline de traces officiel ExLlamaV3](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/sc_trace.py) génère des conversations avec les modèles source ou à haute précision, avec raisonnement et plusieurs langues. Les premiers essais du générateur ExLlama F16 ont été interrompus pour lenteur. Les mêmes prompts épinglés sont maintenant lus comme données et générés avec Transformers5.16.1, poids sourceBF16 et grouped_mm natif, avec garde de parité contre eager. L’ordre est mélangé avant les lots64 ; chaque lot est sauvegardé et39traces complètes d’au moins2048tokens sont exigées. Seuls39segments de2048tokens source remplacent les39lignes uniformes, les211lignes réelles sont conservées. Le total reste512000tokens si39traces suffisamment longues sont disponibles ; sinon l'étape échoue explicitement.
- [GPTQv2](https://arxiv.org/abs/2504.02692) vise une cible asymétrique : reproduire les sorties source malgré les entrées déjà perturbées. C'est différent de la propagation existante. Cette méthode demanderait des captures source/quantifié appariées et une adaptation validée à LDLQ/treillis et aux changements de routes ; aucun drapeau actuel ne l'implémente. Nous ne prétendons pas l'avoir ajoutée.
- [EAQuant](https://arxiv.org/abs/2506.13329) et [VSRAQ, juin2026](https://arxiv.org/abs/2606.05688) motivent le contrôle des décisions du routeur et des marges top-k. Nos routeurs sont déjà conservés hors K4 ; nous ne modifions ni leurs biais ni leur distribution pour forcer un équilibre uniforme. Protéger davantage quelques projections communes est une expérience distincte, avec coût de bits affiché. Leurs mécanismes d'apprentissage des paramètres de quantification ne sont pas transférés automatiquement à EXL3.
- [L'optimisation de bitrate officielle](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/doc/optimize.md) reste décrite comme expérimentale pour MoE. Le [moteur public Sushi](https://github.com/beamivalice/sushi) n'expose pas le convertisseur ni sa recette complète ; ses scores et son mélange de précisions ne sont pas une garantie sur LFM.

## Implémentation et limites

`scripts/exl3_moe_calibration.py` applique un patch strictement contrôlé par SHA au chemin Python de capture. Les poids sont ceux de l'ordre réellement trié par le calcul MoE, associés aux mêmes lignes pour up/gate/down. Chaque expert a sa propre Hessienne d'entrée ; gate et up de ce même expert la partagent. Les paramètres du forward courant ne sont pas multipliés par les affinités : seul le calcul de H l'est. Les paramètres d'inférence ne sont pas modifiés.

Une couverture insuffisante ou non finie arrête le convertisseur, au lieu de réutiliser un autre expert ou de produire un repli sans calibration. Le minimum du24B est256lignes et256lignes effectives pondérées. Le minimum de la fixture miniature est1, explicitement distinct. Le journal doit contenir4864Hessiennes d'experts uniques pour les38blocs MoE du24B.

Le pilote utilise les états source aux couches2/20/39 et trois experts par couche choisis uniquement par leurs comptes de calibration : froid admissible, médian, chaud. Les modes all/routed/affinity/squared et les deux corpus donnent216quantifications de matrices, sur27projections physiques ; les données de validation sont12×1024tokens répartis entre six domaines, sans13grammes communs avec les deux calibrations. Ce sont des proxys sur états source fixes, pas une mesure KL du modèle complet, ni une preuve sur toutes les couches.

La sélection minimise la moyenne géométrique des erreurs relatives de projection, sous une limite de régression de10% pour toute projection et chaque domaine agrégé. Les deux projections communes du premier bloc sont comparées à leur précision HQ V2 (K5/K6) puis K8. Une promotion du tronc exige une baisse du proxy d'au moins20% pour chacune et un budget final estimé≤4,35bpw. Une hausse de taille est présentée séparément du gain de calibration.

## Exécution et critères finaux

Compte Modal `compte2`, workspace `xzknw` :25,02USD visibles après actualisation avant lancement. Nouveau Volume privé `mlxl3-lfm25-v3-20261015`, aucun token HF transmis, aucun upload automatique. Préparation H10080GB/CPU8/RAM96GiB≤1800s, retry0 ; conversion≤10800s, processus enfant≤10200s, retry0. La conversion exige un prévol CUDA puis natif avec le même helper/mode/corpus/recette et un fichier d'approbation explicite. Les fichiers de calibration sont contrôlés contre leurs SHA du pilote.

Les tests finaux conservent le même moteur355b5262 et les enseignants BF16 de V2 :8192labels sans BOS,4088labels BOS,2048labels courts,1024labels français et1024labels Python. Rapporter les agrégats, chaque fenêtre, PPL, KL, argmax, comptes et finitude. Ajouter des contrôles chat/multilingues disjoints des prompts source, puis les smokes du bridge. Ne pas choisir une nouvelle recette d'après ces résultats finaux sans enregistrer une nouvelle expérience et un nouveau holdout.

Avant la conversion complète, figer ces gates : KL principal avec et sans BOS réduit d'au moins1% contre V2 ; KL FR/code/court sans régression>2% ; PPL≤1,02×max(PPLV2,PPLteacherBF16), seuil révisé avant mesure pour ne pas rejeter une reproduction exacte du teacher ; smokes réussis, structure/poids/logits finis et sorties complètes. Examiner toute fenêtre dont le KL régresse>20%, même si l'agrégat passe. Une variante qui échoue reste un résultat conservé, sans remplacer le modèle V2 publié. Le KL mesure la fidélité, la PPL sur labels ne lui est pas équivalente ; une PPL sous BF16 n'est pas une preuve de meilleure fidélité.

## Vérification

La campagne doit conserver les commandes, versions, SHA, durées, coûts et résultats négatifs dans `opti.md` et ses mesures. Les tests CPU comparent H à une somme extérieure F64 indépendante, inversent les poids pour contrôler la sensibilité, exercent les garde-fous et le câblage des hooks. CrossHair a tenté trois contrats d'isolation de cartes à20s/condition : **non confirmés**, sans contre-exemple retourné ; les tenseurs, CUDA, les fichiers et la qualité du modèle ne sont pas formellement prouvés. CrossHair confirme les deux postconditions de bornes du vrai `scored_count` ; cela reste une recherche symbolique, pas une preuve du parcours complet. Le test du masque de score détecte une mutation qui score aussi les positions exclues. Les tests GPU de conversion et du bridge restent nécessaires. Aucun changement Rust/Metal du moteur n'est livré par cette recette.

## Recherche29 : ce que Sushi permet réellement de conclure

Sources Sushi figées au commit `47930102459b91c73b80f5bf7ce612f3cc736dd7`, consultées le15octobre2026. Son [README](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/README.md) décrit un convertisseur hybride EXL3/affine. La [séparation publique/privée](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/CLAUDE.md) est explicite : recettes, données et convertisseurs « sashimi » sont privés. Nous pouvons étudier ses contrats et mesures, sans prétendre disposer de sa méthode complète.

| Lecture publiée par les auteurs | KL | Portée de la comparaison |
|---|---:|---|
| Flash-Next Sushi4, publication29septembre | 0,0592 | Référence BF16,16×512,7186positions jusqu'au premier EOS, cache élève8bits |
| MiMo Sushi2,3 | 0,0860 | Référence MOPD stockée, en partieFP8/MXFP4 ; pas un modèle source entièrementBF16 |
| GLM Sushi2,4 livré | 0,074213 | Référence NAX BF16,4×512 ; code0,0429 et prose0,1055 |
| GLM Sushi2,5 expérimental | 0,057151 | Même référence et même écran4×512 que la ligne précédente |
| GLM, comparaison de cache en contexte long | 0,00715 | Cache8bits contre cacheBF16 **du même modèle déjà quantifié**,496positions avant EOS |

Sources directes : [Flash-Next](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/docs/qwen4-kld.md), [MiMo](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/docs/mimo2-kld.md), [GLM](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/docs/glm5-kld.md). Ce sont leurs résultats publiés, pas des reproductions locales. Le0,00715 ne mesure pas la fidélité des poids auBF16 et ne justifie pas une cible identique pour LFM.

Leur [protocole](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/docs/quality-kld.md) score des continuations de l'enseignant et distingue premierEOS/toutespositions. Notre test historique score des séquences imposées, avec des réinitialisations et masques différents. Une comparaison de nombres entre ces populations ne classe pas les quantifications. La lecture de [`scoreRow`](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/src/kld.zig#L247) confirme une sommeF64 sur le vocabulaire complet ; `--top-k10` est une métadonnée du fichier, pas une troncature de leur KL. Leurs petits chiffres ne s'expliquent donc pas par une renormalisation sur dix tokens.

Ce qui est transférable : précisions distinctes pour experts et tronc, conservation des états/biais sensibles, contrôle numérique face à une référence explicite et mesure séparée du cache. Le [contrat des experts](https://github.com/beamivalice/sushi/blob/47930102459b91c73b80f5bf7ce612f3cc736dd7/docs/engine-exl3-experts.md) expose MCG/MUL1, des fenêtres et des taux fractionnaires. Copier un réglage de fenêtre dans un décodeur16bits donnerait d'autres poids : ce n'est pas une option de qualité directement applicable à notre moteur. Sushi emploie aussi MCG ; notre choix MCG n'est donc pas démontré fautif.

## Réglages officiels : base et ablations utiles

La V2 utilise **déjà le quantifier officiel CUDA1.5.4**, commit `16a49792a3c93d8432d72e6c4bce800841566577`. La [documentation officielle](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/doc/convert.md) recommande actuellement MUL1. Elle ne fournit pas de réduction deKL garantie contre MCG. Changer de H100 ou de codebook ne suffit pas à certifier la fidélité.

| Réglage | État V2 | Décision issue de la recherche |
|---|---|---|
| `-b4 --hq` | Actif | Garder la promotion officielle des couches sensibles ; K8 partout est rejeté par le test complet. |
| `-hb16 -eb16` | Actif | Garder les matrices liées à haute précision ; le compactage supprime seulement leur copie redondante vérifiée. |
| `--out_scales auto` | Actif | Garder comme référence ; forcer toutes les scales n'est pas une amélioration établie. |
| `-cr250 -cc2048` | 512000tokens | Référence reproductible ; une hausse exige un vrai corpus plus grand et des lignes complètes. |
| `-cb mul1` | MCG | Première ablation officielle, même budget ; effet partiel mesuré30/32, pas de profil global admissible. |
| `--scaled_hessian` | Désactivé | Deuxième ablation, compatible avec les experts ; option expérimentale, pas un gain acquis. |
| YAQA, `--hessians` | Désactivé | Objectif intéressant, mais couverture des experts fusionnés manquante dans l'outil stock. |

Recette de départ **à comparer**, sans kernel ou capture personnalisée :

```sh
python -m exllamav3.conversion.convert_model \
  -i "$LFM_SOURCE" -w "$LFM_STOCK_WORK" -o "$LFM_STOCK_OUT" \
  -b 4 --hq -hb 16 -eb 16 -cb mul1 \
  --out_scales auto -cr 250 -cc 2048
```

Les dossiers doivent être nouveaux. Aucun lancement de cette recette n'a eu lieu pendant la recherche29. Le parseur dense/MoE natif reconnaît MUL1, mais cela ne remplace pas un prévol physique de ses nouveaux poids. Les taux fractionnaires et les fenêtres Sushi ne sont pas inclus dans cette recette.

Le [vrai `scaled_ldl`](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/quant/exl3_lib/quantize.py) refactorise la Hessienne après avoir intégré les échelles d'entrée propres au tenseur. Cela distingue son objectif de la factorisation partagée utilisée enV2. Il ajoute une décomposition par tenseur, sans changer le format d'inférence. **Hypothèse** : cette métrique peut mieux distribuer l'erreur sur les experts ; le pilote32 mesure un gain KL partiel moyen, accompagné de régressions qui bloquent sa promotion ; le KL24B de cette option reste non mesuré.

## Deux contrôles concrets avant d'augmenter la calibration

Le vrai `split_raw` officiel a été exercé surCPU avec le tokenizer source exact, aux tailles250/1000/2000. À1000, `technical.utf8` demande74lignes2048 mais ne contient que35lignes pleines : le découpage rend1ligne courte et **38vides**. À250, les18lignes demandées sont toutes pleines. À2000, le fichier code manque aussi de données :192pleines,1courte,103vides. Aucune quantification n'a consommé ces lignes. [Rapport et empreintes](measurements/quant-lfm25-v3-20261015/calibration-capacity29.json).

Il faut donc augmenter les textes/traces disponibles avant de monter `-cr`, puis contrôler longueur, IDs, BOS/template, couverture et indépendance des données. Répéter les mêmes textes ou concaténer des lignes vides ne crée pas de diversité. Le remplacement de39lignes par nos traces au pilote25 était presque neutre ; cela ne valide ni un corpus entièrement généré ni une hausse de taille.

Le collecteur [YAQA officiel](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/util/yaqa_hessians.py) ne sélectionne que `nn.Linear`. Son vrai `find_targets`, exécuté sur la structure40blocs en `meta`, couvre **70projections communes sur106**, et **zéro expert fusionné**. Ajouter `w1/w2/w3/in_proj` au motif couvre106projections communes, toujours aucun expert. Les38modules fusionnés représentent7296projections et **95,7435% des valeurs du checkpoint**. [Audit de couverture](measurements/quant-lfm25-v3-20261015/yaqa-stock-coverage.json). Aucun poids chargé, forward, gradient ou gainKL mesuré dans ce contrôle.

`--out_scales yaqa` avec uniquement une diagonale ne réalise pas l'arrondi bilatéral complet. Des facteurs `hout` déclenchent cet arrondi ; garder une capture d'entrée sur les états quantifiés est une ablation distincte de remplacer `hin` par des facteurs source. L'outil stock et le master consulté ont le même collecteur. Il faudrait une adaptation des experts avant de qualifier cette voie de YAQA couvrant LFM2.5.

## Papiers : intérêt et limites d'application

| Source primaire | Apport pertinent | Application ici |
|---|---|---|
| [QTIP, NeurIPS2024](https://arxiv.org/abs/2406.11235) | Treillis et transformation d'incohérence ; base scientifique d'EXL3. | Déjà dans le quantifier ; pas un nouvel interrupteur de qualité. |
| [YAQA, version2](https://arxiv.org/html/2505.22988v2) | Approximation de Hessienne du KL final ; environ30% de réduction empirique contreLDLQ dans les expériences des auteurs. | Piste directement alignée sur notre objectif ; leurs résultats4bits surLlama ne garantissent pas ceux d'unLFM MoE. Le collecteur officiel limite aujourd'hui son application. |
| [GuidedQuant, ICML2025](https://arxiv.org/abs/2505.07004) | Importance des sorties issue des gradients de la loss ; intégration démontrée avecQTIP. | Alternative à étudier pour des facteurs sensibles au résultat final ; pas un flagEXL3 ni la même approximation queYAQA. |
| [GPTAQ, anciennementGPTQv2](https://arxiv.org/abs/2504.02692) | Calibration asymétrique qui tient compte d'entrées déjà déformées. | Différent de la simple propagation actuellement active ; pas implémenté par `--hq`. |
| [FastKron, version2 septembre2026](https://arxiv.org/abs/2608.06291) | Réduit le coût algorithmique d'un arrondi à deux Hessiennes. | Piste de coût pourYAQA, sans garantie supplémentaire deKL. La PRofficielle395 de collecte récursive demeure ouverte, pas dans la roue1.5.4. |
| [A KL Lens on Quantization, avril2026](https://arxiv.org/abs/2604.13440) | Sensibilité de composants hybrides mesurée enKL plutôt qu'enMSE/SQNR. | Motive une validation de sortie pour les blocs conv/attention deLFM ; leurs résultats surSSM ne sont pas une validation deLFM. |
| [XOR-Trellis, septembre2026](https://arxiv.org/abs/2610.00432), [CurveTQ, octobre2026](https://arxiv.org/abs/2610.09212) | Recherche de treillis tenant compte de la courbure, sans rotationHadamard. | Nouveaux codecs/méthodes, pas des réglages de l'EXL3 stock. Pas de changement du format livré pour cette campagne. |

Les titres des papiers peuvent évoluer :2504.02692 est désormais GPTAQ ;2608.06291 est désormais FastKron, appeléBaKron dans la [PR395](https://github.com/turboderp-org/exllamav3/pull/395). Les anciennes références restent identifiables par leurs IDs. La PR distingue ses temps de phases et son code GPU mesuré de la préparation soumise ; nous n'en déduisons aucun temps de conversion pour notreH100.

## Décision29 et prochain protocole

La meilleure base **mesurée et conservée** reste la V2. La prochaine comparaison utile est une matrice2×2 **MCG/MUL1 × Hessienne partagée/scalée**, à bitratesHQ identiques, corpus original et paramètres fixes. Un pilote doit comparer les projections puis la sortie du modèle sur validation distincte : les seuls proxys de reconstruction du25 ont mal prédit les régressions du candidatK8. Les nouvelles options restent à mesurer avant une conversion24B ; cette recherche ne désigne pas un gagnant non testé.

Après cette ablation seulement, tester un corpus réellement agrandi avec des fenêtres complètes et la répartition des domaines annoncée. Ne pas mélanger immédiatement codebook, Hessienne, corpus et nouvelles précisions : leurs effets resteraient impossibles à attribuer. Aucun expert supprimé, seuil deKL relâché ou teacher approximatif pour afficher un meilleur score. Les jeux historiques restent des régressions ; tout réglage sélectionné d'après leur sortie exige un nouvel échantillon final disjoint. Les holdouts dialogue/CJK du25 sont encore disponibles et n'ont pas été scorés sur les candidats.

La [pipeline officielle d'allocation](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/doc/optimize.md) mesure déjà une sensibilité finale et crée des traces en domaine, mais reste annoncée non testée sur modèles sparse. Nous ne la présentons pas comme une recette MoE garantie. Il n'existe pas de réglage publié permettant de promettre KL0,01–0,02 pour ceLFM à4bpw. Les écartsBF16/F16/backend mesurés au23 restent séparés de l'erreur de quantification ; ils ne se soustraient pas auKL.

Sources etSHA : [manifeste documentaire](measurements/quant-lfm25-v3-20261015/settings29-sources.json), [comparaison officielle après1.5.4](measurements/quant-lfm25-v3-20261015/upstream-compare29.json). Aucun nouveau code exécutable produit modifié au29, aucun nouveau modèle converti. Les contrôles du code précédent restent :664tests de la suite complète passés/1Ling ignoré,14nouveaux tests ciblés de compactage passés,2postconditions de bornes de score confirmées parCrossHair ;3contrats de cartes et2de métadonnées nonconfirmés à20s/condition. Ces recherches symboliques ne prouvent pas CUDA/MLX, les I/O ou la qualité. La CI du commitbc5304 a ses checks conversion/régression/Linux/macOS réussis ; Kani restait en cours à la dernière inspection.
