# COMPOSE-01 — audit du chemin CLI GDN/F02

1 novembre 2026. **Aucun flag perdu, fallback global ou défaut concret
d’adaptation trouvé.** Lecture des sources figées et des quatre processus
AR128 finalisés du premier passage A/G/F/C uniquement. Le retour est en cours :
aucune conclusion finale de gain, aucun mélange avec les anciens pilotes.
Aucun code surveillé, build, test ou GPU modifié/lancé par cet audit.

## Le CLI emprunte les mêmes chemins qualifiés

`NativeChatModel::from_checkpoint` choisit Flash pour `qwen4_exp`
([main.rs:1406](../../../native/src/main.rs#L1406)). Son `forward` appelle
`FlashNext::forward` → `forward_inner` → `Layer::forward_lookahead` →
`Layer::prepare` → `Gdn::forward` ou `Qsa::forward`. Les stages et compteurs de
tests sont sous cfg(test), sans rétention ajoutée au CLI release.

**GDN.** `Gdn::load` capture strictement PREPARED_CONV à load. Some est
matérialisé après le vrai `half_weight` F16, sans supprimer cet arrondi ;
None n’appelle pas la préparation. `convolve` emprunte désormais le coefficient
Some directement (R2), sans clone de wrapper, ou reconstruit exactement la
transpose/cast stock en None
([qwen4.rs:326](../../../native/src/qwen4.rs#L326)). Ce helper est aussi celui
de chaque ligne de vérification. Aucun test-only hook n’active l’optimisation.
L’epsilon, les stores F16, la somme, SiLU, norme et états restent stock.

**F02.** `Projection::load_logical` choisit EXL3 quand le trellis est présent
([lfm2.rs:180](../../../native/src/lfm2.rs#L180)).
`Exl3Linear::from_checkpoint` capture DENSE_EPILOGUE et, pour Flash, active
l’accumulation de référence/SG4 avant le forward. Aucun autre chemin ne remet
le flag à false. Les deux callsites mono/batch essaient l’épilogue juste après
les partiels, avant somme et Hadamard stock
([linear.rs:574](../../../native/src/linear.rs#L574)).

Le domaine reste K4/Mul1, 6144→2560, SG4, split2, B1/B2, F16/F32, sans biais ;
`apply` vérifie partiels F32 `[2,2560]` ou `[2,2,2560]` et SV F16 `[2560]`.
Les sorties attention ciblées sont les 36 GDN et 12 QSA, déjà qualifiées sur
les 48 poids. GDN transmet un gate F16 `[1,6144]`, QSA un `[1,6144]` dans son
dtype d’activation ; les deux sont admissibles. Les dimensions donnent
384 tiles d’entrée/160 de sortie ; `reference_splits` produit bien 2.
QKV, z et les autres dimensions restent stock. Un appel groupé B>2 n’est pas
admissible ; un chemin séquentiel qui le découpe en B1 peut légitimement
utiliser F02 pour chaque ligne. Ces fallbacks prévus ne sont pas un arrêt global.
Le préfill AR par chunks8 demeure token-séquentiel, donc cette taille ne
transforme pas ses projections en un B8 hors domaine.

Ces constats source et les oracles physiques distinguent les voies réelles
des flags seulement enregistrés. Le CLI n’émet cependant pas un compteur de
dispatch F02 par réponse : cet audit ne prétend pas en compter dans ces runs.

## Coûts et interaction

G ajoute 36 coefficients F32 persistants, payload **5898240 B = 5,625 Mio**,
avec évaluations au chargement ; les F16 originaux restent conservés. Aucun
readback ni synchronisation de forward supplémentaire. Les propriétaires
Array/MLX restent valides ; reset ne détruit pas le coefficient immuable.

F02 alloue une sortie custom et garde les partiels stock ; il ne supprime pas
la QMV ni toute sa construction. Le dtype/reshape final est préservé. Les
coûts de primitive custom, bindings, gestion des ressources, éventuelle
contiguïté et différences de donation restent possibles et non attribués.
Aucune nouvelle copie de poids n’est introduite entre G et F02 : G travaille
dans la convolution, F02 dans la projection de sortie après récurrence/norme.
Leurs pourcentages ne sont pas additifs ; cache GPU, donation et chevauchements
avec le MoE peuvent masquer ou déplacer un petit coût évité.

## Quatre rapports AR128 du premier passage

Répertoires exacts : `runs/compose-pilot128-AR-{1-A,2-G,3-F,4-C}/report.json`.
Même CLI SHA256
`2b0ab6951ef850baa2bc728bcd19060edfc8ac69bbf598cd230d5bb5452ff838` ;
statut validated, processus/lecteur joints, mode baseline/AR. Flags rapportés
PREPARED_CONV/DENSE_EPILOGUE = **A00, G10, F01, C11** ; T17/T05 OFF. Les
compteurs MTP/DF sont null, profils DF absents : aucun mode spéculatif masqué.

Exemple MAIN mesuré, octets ; une seule réponse par bras à ce stade :

| Bras | Active MLX | Empreinte processus | Compression logique courante |
|---|---:|---:|---:|
|A|6579517056|7121737608|1276493824|
|G|6585416320|7123720072|1334657024|
|F|6579518080|7141955488|1396916224|
|C|6585419904|7119132528|1431437312|

G−A active = 5899264 B ; C−F = 5901824 B, cohérents avec le payload préparé
plus de petits temporaires. F−A vaut seulement1024 B. L’empreinte varie sans
suivre cette seule allocation : aucune causalité de débit déduite des
compressions ou décompressions cumulées. Pics de vie natifs des quatre
processus ≤7141971872 B ; swapin logique du processus nul aux deux réponses.
Les inspections supervisées indiquent zéro nouveau Swapout sur ces runs.

Les rapports AR ne contiennent ni demande d’experts, ni hits/misses, ni temps
des lectures : **I/O identique, augmentée ou mieux masquée non mesurée ici**.
Le load observé est 1,008/1,048/1,044/1,069 s pour A/G/F/C, un processus par
bras ; ce n’est pas une attribution du surcoût de chargement au seul flag.

Les neuf sélections qualité G/C et les preuves F02/GDN réutilisées restent le
socle numérique. Le premier passage ne décide pas à lui seul de l’activation ;
attendre la matrice prévue et qualifier sa portée, en conservant aussi les
petits candidats compatibles si le résultat global reste indécis.
