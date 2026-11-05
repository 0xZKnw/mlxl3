# Deux vagues Cold : conserver les contributions et leur ordre

1 novembre 2026 — POSTP08-COLD-READY-01. Revue de code uniquement. Aucun
runtime modifié, build, test, preuve ou GPU exécuté. COMPOSE reste arrêté à la
demande de l'utilisateur. Les chiffres P08 décrivent une disponibilité possible,
pas le gain de cette proposition.

## Conclusion

Le premier prototype peut utiliser **deux cohortes compactes d'occurrences**,
avec les fonctions de calcul existantes, puis une remise en ordre unique avant
la multiplication des poids et la somme Top10 stock. Il n'est pas nécessaire
de créer un kernel EXL3. La forme de chaque lancement change, mais le chemin
slot-major garde ici **K2/MUL1, NT1, SG4, splits1** indépendamment du nombre
d'occurrences. Ce n'est pas une preuve de parité Metal : elle doit être mesurée
sur le vrai chemin pour les deux vagues et leurs erreurs.

Une variante qui calcule deux matrices complètes masquées préserve le nombre
logique de lignes, mais double plusieurs travaux MLX et change le chemin Cold
non masqué vers le chemin resident masqué. Elle n'est pas le premier choix.

## Chemin réellement exécuté

- [ssd_moe.rs:968](../../../native/src/ssd_moe.rs#L968) : `jobs` contient une
  seule lecture par slot Cold manquant, mais `cold` contient **les occurrences**
  de routes en ordre token puis rang Top10. Les hits arrivés via un préfetch
  depuis le snapshot initial peuvent rester dans `cold` sans être dans `jobs`.
- [ssd_moe.rs:986](../../../native/src/ssd_moe.rs#L986) : `fill_many` attend
  actuellement toutes les lectures avant `fill.commit()` et le calcul Cold.
  Une première cohorte prête est donc inutilisée pendant la fin du lot.
- [ssd_moe.rs:1016](../../../native/src/ssd_moe.rs#L1016) : `routed_x.take`
  sélectionne les activations de chaque occurrence, puis `calculate` réutilise
  les banques slot-major. En présence de Hot, un concat + take rétablit les
  contributions canoniques. Avec GPU Hot, son output garde toutes les lignes
  originales, même les lignes masquées ; son offset est `ids.len()`, pas le
  nombre de hits.
- [ssd_moe.rs:1071](../../../native/src/ssd_moe.rs#L1071) : une conversion vers
  `x.dtype()`, une multiplication avec les poids canoniques, puis **une seule**
  somme de `[tokens,10,2560]` sur l'axe Top10. La contribution shared est ajoutée
  après cette somme. Garder cet ordre exactement.

`calculate_slots` ([ssd_moe.rs:348](../../../native/src/ssd_moe.rs#L348)) :
entrée `[n,2560]` F16, deux projections gate/up `[2n,2560]→[2n,640]`, SwiGLU
puis down `[n,640]→[n,2560]`. Chaque route utilise les scales de son slot ;
les activations et sorties de ce parcours restent F16. Le cast F16/F32 final
et les poids du routeur sont appliqués seulement à l'assemblage stock.

## Ce qui change et ce qui ne change pas avec n

[linear.rs:1492](../../../native/src/linear.rs#L1492) sélectionne les splits
slot-major par les dimensions internes, pas par `local_tiles` :
`reference_splits(160,40)=1` en gate/up et `reference_splits(40,160)=1` en down.
Le prédicat de [flash_support.rs:286](../../../native/src/flash_support.rs#L286)
ne double les splits que si `input_tiles/(2*splits)>=128`, faux dans les deux cas.
Slot-major force NT1 et SG4, même sur M5. La branche NT4 liée à `rows>=64` n'est
pas sélectionnée. Il n'y a pas de bascule vers TensorOps/GEMM dans cette fonction.

Le kernel de référence conserve l'ordre FMA/réduction par ligne
([_qmv_mapped_tile_kernel.metal](../../../native/shaders/_qmv_mapped_tile_kernel.metal)).
Les transformations Hadamard portent sur des groupes de128 valeurs de chaque
ligne. Compacter les routes ne demande pas de changer leurs calculs internes.

Cependant `LOCAL_OUTPUT_DIMS`, les grilles, les tailles de buffers et les clés
de factories changent avec `n` lorsque `STABLE_EXPERT_KERNELS=0`. MLX peut aussi
changer allocation, donation, gestion des vues et préparation des opérateurs.
Ne pas activer simultanément STABLE pour masquer ce coût : cela constituerait
un deuxième changement. Préparer/chauffer les tailles effectivement servies
avant le chrono, conserver aussi le TTFT et le coût de chargement à part.

`SHARED_EXL3=1` appelle `ExpertSlots::grouped`, avec un seuil à16 occurrences et
un calcul de rentabilité dépendant des répétitions. Scinder un lot peut activer
ou désactiver ce regroupement. Pour le premier port, conserver **SHARED_EXL3=0**
comme dans la baseline et retomber sur stock si l'option est activée. Même
principe pour `WEIGHTED_EPILOGUE=1` : son interface n'accepte que Hot et Cold,
son adaptation à trois sources serait un autre changement.

## Assemblage compact proposé

Réserver tous les slots et construire la partition Hot/Cold comme aujourd'hui,
avec toutes les occurrences conservées. La readiness concerne un **slot/record
complet** ; un succès partiel `preadv` n'est pas une publication. Le contrôleur
hôte reçoit un ensemble de slots entièrement prêts ; aucun appel MLX depuis
les workers.

1. Former W1 à partir des occurrences Cold dont le slot est prêt, en conservant
   leur ordre dans `cold`. Les Cold déjà prêts sans job sont éligibles. Fixer
   avant test le seuil en **experts distincts** et consigner aussi le nombre
   d'occurrences : en B2, deux experts prêts peuvent représenter quatre lignes.
2. Sur le thread moteur, sélectionner les activations de W1, appeler le
   `calculate` existant, retenir sa sortie/propriétaires puis `async_eval` pour
   lancer effectivement le travail pendant les lectures restantes. Un graphe
   simplement construit ne produit aucun recouvrement.
3. Attendre les lecteurs restants, vérifier tous les résultats, puis calculer
   W2 = complément des occurrences Cold. Au plus deux vagues, pas un lancement
   par expert. Si aucune cohorte utile n'est prête avant la fin, suivre stock
   sans lancement GPU supplémentaire. Éviter les appels `calculate` vides.
4. Concaténer Hot/W1/W2 et construire l'indice inverse par **occurrence originale**.
   Un seul `take` restitue `[tokens*10,2560]` ; conserver ensuite le cast,
   `mul(weights)`, `reshape([tokens,10,2560]).sum(1)` et shared existants.
   Si W1 est un préfixe de `cold`, W1+W2 est déjà dans l'ordre Cold ; ce cas
   simplifie l'indice mais peut attendre un record lent placé tôt.

Pour toute occurrence i, l'indice restauré doit sélectionner exactement une
source : Hot original si le snapshot la couvrait, sinon sa position dans W1
ou W2. Les deux vagues sont disjointes et leur union est `cold`. En GPU Hot,
les lignes de misses de son output masqué ne sont jamais sélectionnées.
Ne pas dédupliquer les résultats par ID d'expert : deux tokens B2 partagent
les mêmes poids mais généralement pas la même activation. Une seule lecture
est correcte ; une seule évaluation réutilisée entre ces tokens serait fausse.

**À ne pas faire :** sommer W1, sommer W2, puis additionner leurs résultats.
Même avec des poids identiques, cela réassocie les additions flottantes et
change le contrat exact. Ne pas ajouter des matrices masquées Hot+W1+W2 pour
fusionner non plus : sélectionner les contributions évite les effets de signe
zéro/non-finis et conserve le chemin de réduction canonique.

## Variante à forme complète masquée

`ExpertSlots::residents` sait construire un vecteur de slots masqué, et
`gather_scales` met les scales absents à0. Le kernel sort avant les lectures du
trellis pour un slot `u32::MAX`/hors capacité
([linear.rs:1318](../../../native/src/linear.rs#L1318),
[_qmv_mapped_tile_kernel.metal:107](../../../native/shaders/_qmv_mapped_tile_kernel.metal#L107)).
On pourrait produire W1 et W2 sur toutes les `10B` lignes, avec deux maps
immuables prêtes/non prêtes, puis sélectionner la source correcte par ligne.

Cela ne conserve **pas** le même chemin kernel Cold qu'aujourd'hui : flag
masked, gather custom et map kernel supplémentaires. Même les lignes absentes
passent dans plusieurs opérations complètes (scales, Hadamard, activation,
sorties), malgré le court-circuit du QMV. Les matrices activations/sorties
sont répétées et l'assemblage doit toujours être une sélection, pas une somme.
Ne jamais partager une map modifiée par CPU avec un kernel en vol. Une map
512IDs qualifie tous les doublons d'un expert ensemble, ce qui convient à une
readiness par record, pas à une sélection arbitraire d'occurrences.

Cette variante ne se justifie que si le coût de compactage/factories a été
mesuré comme dominant et qu'un micro complet démontre le bénéfice. Aucun
bénéfice de ce type n'est mesuré actuellement.

## Coûts, mémoire et propriété des banques

Deux vagues ajoutent au moins une construction de chaîne gate/up–SwiGLU–down
et une soumission asynchrone ; deux `take` d'activations, vecteurs de slots,
scales et temporaires de tailles séparées remplacent une chaîne. Les petites
cohortes peuvent sous-occuper le GPU ; l'encodage sur le thread moteur peut
manger l'avance SSD. La première vague passe derrière le Hot déjà soumis dans
le flux GPU : sa disponibilité CPU ne garantit pas un démarrage GPU immédiat.
Plus de dispatchs peuvent donc coûter davantage que le trou que l'on vise.

Conserver exactement les mêmes banques et 53slots×48couches ; **pas de seconde
banque d'experts ni de staging complet supplémentaire** pour faciliter le port.
Les sorties Cold seules valent au plus100Kio de payload F16 en B2 (20×2560×2),
mais ce chiffre ne borne pas le pic de tous les temporaires MLX, le cache de
factories ou les allocations retenues par des graphes concurrents. Mesurer
la RAM/pic natif et conserver le budget experts3Gio séparément.

Le chemin synchrone actuel retient `last` pour les consommateurs des banques
et invalide les réservations via `ExpertFill` sur erreur. Une W1 soumise crée
un nouveau consommateur même si la lecture de W2 échoue ensuite. Un `last`
remplacé uniquement par W1 peut perdre Hot ; un `last` restant Hot peut perdre
W1. Le guard de calcul doit retenir **toutes** les sorties soumises et les
attendre sur les chemins d'erreur avant toute réutilisation/écriture des slots.
Sur succès, le résultat final dépend des trois contributions et devient le
nouveau `last`. Ne pas appeler `fill.commit` après W1 seulement : des records
encore incomplets survivraient comme hits. Préparer les propriétaires/ranges
avant de lancer les lecteurs ; construire les calculs seulement après la
publication de readiness des slots qu'ils lisent. Le protocole FFI/lecteurs
reste à qualifier avec l'audit de propriété distinct.

## Contrôles minimaux avant mesure

Réutiliser les oracles slot-major existants :
`linear::tests::flash_slot_shapes_match_resident_reference_for_every_route`
(cas routes1..80, F16/F32, strides, doublons, masked/grouped) et
`ssd_moe::tests::two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads`.
Ils ne testent pas encore un calcul pendant des écritures Cold concurrentes.

Ajouter au vrai chemin : B1/B2, allCold/allHot/mixed, aucune W1/W1 seule/deux
vagues, préfixe et slots prêts non contigus, doublons intertokens avec activations
différentes, cache plein/slots53 et petit cache, erreur après soumission W1,
reset/reprise, non-finis/signes zéro selon le contrat existant. Comparer les
contributions de chaque occurrence puis le résultat, les états et logits ;
un mutant inversant deux occurrences doit échouer avant vitesse. Inspecter
les signatures du vrai kernel pour W1/W2 et les fallbacks, pas seulement les
flags. Preuve CPU utile : partition totale/disjointe, index inverse borné,
slot prêt requis ; elle ne prouve ni Metal ni la concurrence des lecteurs.

Le micro doit inclure préparation d'indices, deux chaînes, soumission et
assemblage, avec retards de lecture déterministes pour détecter l'overlap réel.
Une capture temporelle doit montrer W1 commencée avant le dernier retour,
puis un pilote release doit mesurer le gain après tous ces coûts. Aucun gain
de15,53ms ou de15tok/s n'est déduit de cette revue.
