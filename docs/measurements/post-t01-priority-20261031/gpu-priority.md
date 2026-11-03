# POST-T01-PROFILE-01 — priorité GPU, 31 octobre 2026

Lecture seule des sources et des acquisitions existantes ; aucun nouveau modèle,
benchmark, build, test, preuve ou relevé GPU exécuté. Préenregistrement lu dans
`opti.md`, entrée POST-T01-PROFILE-01. Ce classement choisit **où mesurer avant
portage**, sans annoncer un gain ni qualifier une durée GPU exclusive absente.

## Décision provisoire

1. **T09 : mesurer la chaîne dense output 6144→2560 K4/MUL1**, commune aux
   36 couches GDN et 12 QSA, soit 48 appels par pas AR. Premier contrôle voisin :
   GDN QKV 2560→10240, 36 appels. C'est une nouvelle frontière GEMV→finish,
   distincte de T17/T04, du regroupement NT2 et du batching dense déjà essayé.
2. **T12 : gate QSA→préparation de cette projection out**, seulement après avoir
   localisé le coût réel de préparation. Son périmètre couvre 12 couches AR,
   donc il est plus étroit. L'indexeur long contexte ne mérite pas la priorité
   pour les prompts courts étudiés.
3. **T07/T10 : vérification B2 naturelle, conditionnelle** à des répétitions de
   routes ou des spills réellement observés. Aucune raison actuelle de commencer
   par B6–8 ni de réactiver grouped65/66.

La lecture GPU seule ne permet pas de départager T09 d'un port I/O disposant déjà
d'un budget critique mesuré. Elle ne montre aucun kernel individuel dont on
pourrait aujourd'hui promettre l'économie de plusieurs ms/token.

## Durées : ce qui existe et ce qui manque

Le rapport [attribution/README.md](../ultra-profile-20261027/attribution/README.md),
lignes 22–37, 77 et 88–100, donne **63,013 ms/token de commandes GPU moteur
actives** dans la fenêtre courte de 20 pas ; corroboration séparée de 36 pas :
57,233. Ce sont des unions de commandes actives, incluant stalls et plusieurs
opérations, pas des temps ALU utiles ni une somme de coûts de kernels.

Contrôle direct des XML existants :

| Export sous `ultra-profile-20261027/attribution/` | Taille | Lignes `<row>` | Portée |
| --- | ---: | ---: | --- |
| `short/AR-MAIN/gpu-shader-profiler-sample.xml` | 532 B | 0 | Aucun compteur shader attribuable |
| `short/AR-MAIN/metal-application-encoders-list.xml` | 1825 B | 0 | Aucun label d'encodeur attribuable |
| `short/AR-MAIN/metal-gpu-intervals.xml` | 6172439 B | 5542 | Intervalles GPU sans identité de kernel |
| `AR-MAIN/metal-gpu-intervals.xml` | 10504758 B | 9499 | Autre fenêtre, conservée séparément |

Recherche complémentaire dans les mesures du checkout
`/Users/justin/Documents/mix-stq1_0` : aucun export supplémentaire trouvé donnant
une attribution exclusive aux kernels concernés. Les captures T05/T17/T04
qualifient des noms et nombres de dispatchs ; les temps de replay nuls ne
deviennent pas des temps GPU. Les labels absents ne valent pas zéro travail.

Les **39,947 ms de GPU actif dans 40,056 ms d'attente routes** montrent que les
producteurs occupent cette attente, sans identifier le kernel dominant. Les
12,142 ms de CPU actif/GPU idle dans les soumissions sont un autre axe et ne
doivent pas être attribuées à ces projections. Sources : [issue60 archivée](issue60-current.md),
lignes 137–163, 165–194 et 291 ; [issue59 archivée](issue59-current.md), lignes
24–36 et 92.

## Localisation indépendante par trace synchronisée

Source :
`ultra-profile-20261027/attribution/qual-AR/profile/fenced-decode.json`,
SHA-256 `1c2eff7dc3f8ef105a0fce4a3e58689148a5097dc59756e8e4e13ae408d7c072`.
100502 événements, **16** `output_step`, 576 appels GDN et 192 QSA. Fenêtre
decode 8,190885334 s, soit environ 511,930 ms par transition : l'instrumentation
est fortement intrusive. Elle ne décrit pas la latence de production.

Calcul utilisé : somme de `end_ns-start_ns` des événements nommés, divisée par
16 et 1e6. Pour out, classification par ancêtre `layer_i` (QSA si `i%4==3`).
Pour les noms génériques `gated/normed`, filtrage par ancêtre GDN/QSA ; leur
simple total global inclurait d'autres opérations. Toutes les valeurs ci-dessous
sont des **enveloppes hôte synchronisées inclusives, pas du GPU exclusif**.

| Chaîne localisée | Appels/pas | ms/pas synchronisé | Usage |
| --- | ---: | ---: | --- |
| GDN complet `gdn_single_build` | 36 | 95,157444 | Contexte, contient les enfants |
| QSA complet `qsa_build` | 12 | 25,634904 | Contexte, contient les enfants |
| GDN QKV | 36 | 13,383334 | Projection dense voisine à comparer |
| GDN Z | 36 | 11,701261 | Autre projection dense de contrôle |
| GDN delta | 36 | 9,619739 | Récurrence : ne pas confondre avec EXL3 |
| GDN norm | 36 | 8,600596 | T17 ne constitue pas une nouvelle piste |
| GDN gated | 36 | 8,612312 | Même réserve, pas une économie promise |
| Attention output, GDN | 36 | 11,807974 | T09 dense out |
| Attention output, QSA | 12 | 3,896492 | Même géométrie dense out |
| **Attention output, total** | **48** | **15,704466** | **Première chaîne à mesurer physiquement** |
| QSA gated | 12 | 2,942448 | Localisation T12, chevauche le périmètre proposé |
| QSA q_proj | 12 | 4,945352 | Comparateur dense, pas première cible |
| lm_head eval | 1 | 3,523026 | Gros poids, fréquence faible |

Le span `attention_out_projection` est après l'attention et son stage gated
déjà synchronisé (`qwen4.rs:1081–1086`) ; il localise la projection complète
préparation→QMV→finish et son coût hôte, pas seulement le shader QMV. Les durées
de parents/enfants ne s'additionnent pas. Ces 16 transitions diffèrent des
datasets fenced MAIN de 33 sorties/32 transitions publiés dans #60 ; aucune
moyenne n'est faite entre ces acquisitions.

## 1 — T09 : output dense réel, géométrie commune

Sources actuelles : `native/src/qwen4.rs:205–209,267–291,528,668–673` ;
`native/src/linear.rs:276–282,368–440,474–546,554–666` ;
`native/shaders/_qmv_tile_kernel.metal:18–25,40–57,116–142`.

Métadonnées du checkpoint réel, lues sans charger de poids :
`/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531/quantization_config.json`,
entrées `tensor_storage` :

| Projection | Entrée→sortie | K / codebook | Trellis I16 | Octets trellis |
| --- | --- | --- | --- | ---: |
| layer0 `linear_attn.out_proj` | 6144→2560 | K4 / MUL1 | [384,160,64] | 7864320 |
| layer3 `self_attn.o_proj` | 6144→2560 | K4 / MUL1 | [384,160,64] | 7864320 |
| layer0 `linear_attn.in_proj_qkv` | 2560→10240 | K4 / MUL1 | [160,640,64] | 13107200 |
| layer0 `linear_attn.in_proj_z` | 2560→6144 | K4 / MUL1 | [160,384,64] | 7864320 |

La voie Flash force SG4, NT1, accumulation de référence et aucun lane-order
alternatif. Out B1 : `mlxl3_rs_tile_6144_2560_4_2_1_4_2`, deux splits. QKV B1 :
`mlxl3_rs_tile_2560_10240_4_2_1_4_1`, un split. Les entrées gated observées sont
F16 ; préparation avec `suh` F16 et H128 natif F16 ; partiels F32 ; somme des
splits de référence ; cast F16 **avant** H128 sortie ; produit `svh` F16 ; cast
externe F32 dans GDN et `x.dtype()` dans QSA. Dans cette trace, les deux sorties
observées sont F32 `[1,2560]`.

T09 propose une vraie frontière nouvelle GEMV→finish, avec dernier threadgroup
publiant, plutôt qu'un nouveau nombre de tiles ou le simple batching. Elle peut
réduire la dépendance et le nombre d'opérations, mais **son plafond exclusif
récupérable est non mesuré**. Même les 15,704 ms synchronisées ne sont pas un
plafond fiable de latence naturelle. Publication mémoire, initialisation des
compteurs, absence de spin, écrivain final unique, exactitude des FMA/splits et
H128 restent les contraintes décisives. Le reset indépendant a un coût.

**Filtre minimal proposé, non exécuté :** une projection GDN out layer0 réelle,
une QSA out layer3 de même géométrie et un QKV layer0, B1 d'abord, puis GDN B2
de vérification réellement servie. Mesurer séparément préparation, QMV,
somme/finish/H128 et enveloppe complète, sur un command buffer identifié après
chauffe, avec temps GPU physiques non nuls/qualifiés et coût hôte distinct.
Ne pas traiter un reshape comme une copie sans preuve. Comparer préparation
séparée/intégrée uniquement après ces mesures ; commencer par conserver H128.

## 2 — T12 : gate QSA vers préparation out, périmètre réduit

`qwen4.rs:620–625,668–673` produit gate F16 depuis q_proj
`[1,1,24,512]`, applique sigmoid puis produit F16 à l'attention `[1,6144]`,
ensuite démarre out et son produit par `suh` F16/H128. La nouvelle frontière
possible est la préparation de **cette** projection ; garder chaque arrondi
F16, sigmoid stock, strides et ordre H128. Il ne s'agit pas de reprendre la
fusion sigmoid GDN T17 ni le SwiGLU T04 retiré. Douze appels/pas, proxy gated
2,942448 ms et out 3,896492 ms ; aucune durée exclusive/économie établie. Le
travail out est déjà inclus dans la cible T09 : **aucune addition de gains**.

Indexer : l'index_qk_proj existe à court contexte, mais la sélection par scores
et readback de blocs ne s'active que lorsque `(offset+1)/4 >512`
(`qwen4.rs:633–664`), donc à partir de 2052 tokens présents. MAIN/HOLD courts
ne servent pas ce parcours. Les sept lancements supprimés par l'amont ne sont
pas sept lancements locaux manquants. Étudier ce parcours avec contexte long
serait un autre essai, pas une promesse de boost du decode court.

## 3 — T07/T10 : B2 naturel seulement après preuve de coût

Les résumés naturels `ultra-profile-20261027/natural/DF-{MAIN,HOLD}/profile/trace_1-summary.json`
montrent respectivement 69 blocs/138 lignes vérifiées et 70/140 ; seules les
largeurs B2 sont non nulles. T10 amont visant 6–8 lignes n'est donc pas
directement la forme dominante de ces acquisitions.

Le kernel conserve `x_values[MB][4]`, accumulateurs et scratch de réduction
(`_qmv_tile_kernel.metal:20–25,40–57,116–142`). En Flash, le batch dense réel
reste MB2/NT1 (`contracts.rs:626–630`, `linear.rs:591–594`), y compris pour une
cohorte plus large répartie en groupes de deux. **Registres/spills non mesurés** :
un tableau source ne prouve pas un spill compilé. Les 36 GDN peuvent être
batchées ; QSA reste séquentielle (`qwen4.rs:1124–1133,1157–1160`).

T07 demande multiplicité réelle des experts **avec résidence/masques**, coût du
plan GPU et protection des slots en vol. Les essais65/66 ont déjà tenté le
regroupement CPU de routes connues, refusant Hot masqué, sans gain global
qualifié ; les gains artificiels B4/B8 ne sélectionnent pas B2 naturel. Un plan
GPU Hot sans readback serait une différence réelle, mais pas une justification
pour réactiver l'ancien grouped. Premier filtre : histogrammes naturels de
réutilisation Hot/Cold et durée du kernel B2, puis compteurs de spill pour T10.

## Antécédents exclus et limites finales

- T17 et T04 ont déjà qualifié la correction sans établir de boost global ;
  T04 a été retiré après audit. Ne pas compter la disparition de dispatchs comme
  un gain livré.
- NT2 dense10 a eu un microfiltre favorable puis une campagne modèle dérivante
  non concluante ; T09 doit changer la frontière, pas répéter NT2.
- Grouped65/66, scratch/décodage54Q et seuils ops/bytes déjà rejetés ne sont pas
  relancés. Une fusion H128 avec une autre réduction ne conserve pas notre
  contrat numérique.
- T01-P a mesuré le builder hôte seulement. Ses 0,265 ms/pas indicatives ne
  bornent ni les encodeurs ni la chaîne GPU et ne peuvent pas être soustraites
  des 12,142 ms exposées.

Références historiques et provenance primaire : [issue59](issue59-current.md),
lignes 189–227 et 241–249, avec ses liens H65/H66/T-DENSE ; [issue60](issue60-current.md),
lignes 137–194 et 251–291 ; `opti.md` clôtures T17/T04/T01 et historique dense10.
État : **lecture terminée, classement provisoire ; aucun portage activé,
aucun temps exclusif par famille ni gain modèle nouveau mesuré**.
