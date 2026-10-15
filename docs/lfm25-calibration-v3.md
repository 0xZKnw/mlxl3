# LFM2.5 — calibration après la V2

Campagne QUANT-2026-10-15-LFM25-25. État initial : recherche et code local ; aucun gain V3 annoncé. L'utilisateur autorise une nouvelle conversion H100 puis les tests et privilégie la fidélité au BF16, sans imposer KL0,01–0,02. La V2 publiée reste immuable.

## Résultats du pilote et candidat retenu

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
