# COMPOSE-01 — recalcul indépendant AR128

1 novembre 2026. **Les agrégats AR sont reproduits exactement.** Lecture des
huit rapports clos, ordre **A–G–F–C–C–F–G–A**, même CLI
`2b0ab6951ef850baa2bc728bcd19060edfc8ac69bbf598cd230d5bb5452ff838`.
Aucun nouveau test, build, GPU ou benchmark. Données MTP/DF en cours exclues.
[Calculs, lignes et SHA des rapports](AR-independent-analysis.json).

16 réponses mesurées et 16 chauffes complètes ; aucune exclusion. Deux
processus par bras, deux mesures par prompt/bras. Chaque réponse compte
128 sorties, **127 transitions décode**, aucun prompt réutilisé. MAIN évalue
87 tokens, HOLD82 : 169 tokens de préfill par processus. Chaque bras fournit
donc **508 transitions/338 tokens de prompt** mesurés.

Décode = somme des transitions/somme des `decode_seconds` ; préfill = somme
des prompts évalués/somme des `prefill_seconds`. Les chauffes restent dans les
preuves mais hors agrégats de vitesse. TTFT = moyenne arithmétique par réponse,
pas l’inverse du débit préfill. Identités/flags A00/G10/F01/C11 et clôture des
processus/lecteurs contrôlés ; validation golden/gardes reprise des inspections
déjà closes, sans nouvelle exécution du validateur physique.

## Agrégats des deux prompts

| Bras | Décode tok/s | Δ/A | Préfill tok/s | Δ/A | TTFT moyen s | Δ/A |
|---|---:|---:|---:|---:|---:|---:|
|A|10,48155|—|10,27312|—|8,22583|—|
|G|10,41334|−0,65075 %|10,43240|+1,55049 %|8,10016|−1,52778 %|
|F|10,45675|−0,23655 %|10,21204|−0,59461 %|8,27499|+0,59766 %|
|C|10,57961|+0,93561 %|10,46604|+1,87795 %|8,07419|−1,84343 %|

Ce sont les ratios de temps totaux, pas une moyenne des TPS individuels.
Les valeurs concordent avec [AR-summary.json](AR-summary.json), ainsi que
les agrégats décode par prompt.

## MAIN et HOLD séparés

| Bras/prompt | Décode tok/s | Δ/A | Préfill tok/s | Δ/A |
|---|---:|---:|---:|---:|
|A/MAIN|10,69870|—|10,09097|—|
|A/HOLD|10,27303|—|10,47371|—|
|G/MAIN|10,74029|+0,38870 %|10,45851|+3,64222 %|
|G/HOLD|10,10571|−1,62878 %|10,40485|−0,65740 %|
|F/MAIN|10,62692|−0,67101 %|10,04229|−0,48244 %|
|F/HOLD|10,29196|+0,18421 %|10,39852|−0,71783 %|
|C/MAIN|10,74962|+0,47592 %|10,43350|+3,39439 %|
|C/HOLD|10,41490|+1,38099 %|10,50080|+0,25865 %|

Le second G/HOLD à **9,777361648 tok/s** est conservé avec le premier à
10,456870555. Son exclusion changerait la conclusion descriptive ; elle
n’est pas autorisée par ce recalcul.

## Dispersion et dérive

Intervalles ci-dessous = min/max des **deux mesures**, pas intervalles de
confiance. Dérive = dernier TPS/premier TPS−1 du même bras et prompt.

| Bras | Décode MAIN min–max | Dérive MAIN | Décode HOLD min–max | Dérive HOLD |
|---|---:|---:|---:|---:|
|A|10,55654–10,84475|−2,65755 %|10,14503–10,40431|+2,55573 %|
|G|10,73394–10,74665|−0,11832 %|9,77736–10,45687|−6,49821 %|
|F|10,52785–10,72786|+1,89987 %|10,14980–10,43816|+2,84104 %|
|C|10,74765–10,75159|+0,03663 %|10,36409–10,46622|−0,97584 %|

Le contrôle A agrégé par processus semble stable (10,48322→10,47987), mais
les dérives opposées de MAIN/HOLD se compensent. Son préfill MAIN varie de
10,56046 à 9,66145 tok/s, soit **−8,51295 %** ; F/MAIN passe de9,64562 à
10,47298, soit **+8,57760 %**. Le petit effectif et la position différente des
bras dans le temps limitent l’attribution des écarts préfill comme décode.

## Décision permise à ce stade

C présente ici un petit signal AR positif sur les deux prompts, préfill et
TTFT compris. G et F seuls sont mixtes. Cela ne démontre pas une interaction
causale bénéfique : **les pourcentages individuels ne s’additionnent pas**,
et C n’a pas un gain global garanti parce que son agrégat dépasse A.

Conserver les candidats corrects pour composition est cohérent avec POLICY-02 ;
ni la dispersion ni un petit passage négatif ne déclenchent un rejet
automatique. Ce résultat reste **AR seulement**, sans décision d’activation
générale couvrant MTP/DF. Aucune répétition supplémentaire n’est proposée ici
pour améliorer les chiffres ; terminer les modes préenregistrés avant la
décision de portée du paquet.
