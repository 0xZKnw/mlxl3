# Après T01B : cibler le travail encore exposé

1 novembre 2026. Recherche close, sans nouveau prototype, benchmark ou
compilation. Dynamic + Cold U reste activé localement ; le lanceur prepared
reste désactivé. Référence source `d5a776e53425bb4d0ba61c0774c5399ae7987c06`
avec modifications locales conservées, MLX 0.32.2, M5 24 Go, experts 3 Gio.
[Données extraites et empreintes des sources](evidence.json).

## Premier choix : placer l'expert partagé dans l'attente SSD

Le chemin réel de `native/src/ssd_moe.rs` est actuellement : Hot anticipé,
readback des routes, lancement des lectures dans `fill_ready`, attente de
deux experts, soumission W1, préparation paresseuse W2, jointure de toutes
les lectures, puis `routed.add(&self.shared(x)?)`.
L'expert partagé utilise x et ses propres poids résidents : il ne dépend
pas de la fin des lectures des experts routés.

**Différence testable avec les essais précédents : lancer les lectures avant
de construire/soumettre le shared.** OPT27 construisait le shared avant la
réservation, puis le soumettait avant `fill_many`. OPT73 le soumettait avant
le readback CPU des routes. Ces placements pouvaient retarder le début SSD.
Leurs résultats mixtes/défavorables restent conservés : OPT27 AR MAIN −2,18%,
HOLD −1,77% ; OPT73 AR −4,53%/+4,28%, B2 +6,03%/−4,25%, écrans uniques.
Ils ne valident pas un nouveau placement après lancement des lectures.

Points précis : `fill_ready` commence les lectures ligne 448, attend ligne 453,
soumet W1 ligne 492, joint ligne 550 ; le shared est construit ligne 1344,
ses projections lignes 1372–1387. Les numéros correspondent aux sources
épinglées dans evidence.json.

Premier filtre proposé, à préenregistrer séparément avant code : préparer
le graphe shared pendant les lectures, puis examiner une soumission commune
W1 + shared, en conservant l'ordre W1 prioritaire et la sortie précoce des
routes. Comparer la version « préparation seule » à la version soumise permet
de distinguer travail CPU masqué et calcul GPU masqué. Pas de nouvelle
soumission par petit opérateur, ni de shared calculé deux fois. Les parcours
sans vague Cold gardent leur chemin actuel. Une soumission commune MLX ne
garantit pas à elle seule l'ordre effectif des dispatchs : vérifier la trace.

Dans P09, la construction shared coûte 0,761 / 1,126 / 0,959 ms de CPU actif
pendant GPU inactif par token AR/MTP/DF. Ce sont les enveloppes historiques
de construction, pas les économies garanties. Le GPU shared est évalué plus
tard : sa durée exclusive n'est pas isolée. Les attentes Cold exposées sont
8,357 + 4,744 ms en AR, 4,296 + 10,341 en MTP, 3,730 + 9,171 en DF
(première cohorte / jointure finale, scopes disjoints). Pour le calcul shared,
le plafond idéal est au plus `min(calcul shared déplaçable, trou GPU compatible)` ;
le premier terme reste **non mesuré**. Ne pas lui attribuer toute l'attente.
En particulier, soumettre le shared avec W1 ne peut masquer l'attente initiale
qui précède cette soumission : la jointure finale seule donne une enveloppe
GPU pertinente pour ce placement, encore amputée du temps déjà occupé par W1.

Risques : retarder W1 avec la construction, saturer la bande passante mémoire
en même temps que les lectures, ordre de parcours MLX inattendu, travail
encore vivant sur erreur. Garder les consommateurs et joindre avant de
libérer les banques ; vérifier erreurs/reprise, états/logits et acceptation
AR/MTP/DFlash, puis prefill/decode réels. C'est le premier essai recommandé
pour son mécanisme précis et son périmètre réduit, pas pour un gain chiffré.

## Deuxième choix : réduire une chaîne EXL3 complète, pas son cache de lancement

T01B conservait les shaders, les dispatchs, les allocations de sortie et la
gestion des ressources par MLX. Il retirait surtout des recherches de
pipeline/métadonnées et ajoutait un wrapper. Son pilote AR −1,71% ne justifie
pas de continuer à optimiser ce sous-poste sans autre mesure.

Le dépouillement des échantillons P09 dans `host_async_submit` trouve en AR
34 piles avec `Gather::eval_gpu`, 20 avec `Hadamard::eval_gpu`, 19 avec
`CustomKernel::eval_gpu`, et 224 sans frame primitive identifiable. Les
soumissions avec CPU actif/GPU idle totalisent 12,433 / 9,323 / 7,940 ms en
AR/MTP/DF. **Les nombres de piles ne donnent ni des durées par fonction ni
une attribution complète** ; ils ne soutiennent pas l'hypothèse « tout est
dans le cache CustomKernel ».

Le candidat de portée plus large déjà identifié est le mécanisme DEC-ONE
exact sur une projection dense 6144→2560 K4 : conserver la préparation et
les partiels canoniques, regrouper la réduction et la fin de projection.
Il vise plusieurs nœuds/dispatchs et intermédiaires d'une même chaîne.
La [lecture antérieure](../decode-profile-20261031/turbo-read-01/README.md)
reste la source ; ce n'est pas une nouvelle découverte ni un port exécuté.
Il est distinct du petit épilogue F02 arrêté et de la fusion gather×scale
POSTPROFILE05, corrigée puis retirée après deux comparatifs négatifs.

Avant un port large : qualifier une seule projection complète, en incluant
zéro des compteurs, scratchs et sortie finale dans le coût. Conserver nombre
et ordre des sommes, conversions F16 et arbre Hadamard. Les compteurs
intergroupes et l'ownership GPU en font une piste plus risquée que le
placement shared. Le coût de cette projection dans le modèle courant et
son plafond propre restent **non mesurés** ; les 12,433 ms ne lui appartiennent
pas. Aucun kernel M5 Max ou réglage 40 cœurs à recopier sur le M5 standard.

## Troisième vague Cold : abaissée dans le classement après dépouillement

Les trois traces natives Dynamic+ColdU MTP HOLD existantes contiennent
381 pas de sortie et 9 192 lots séparés en vagues. Filtre déterministe : au
snapshot après W1, au moins deux jobs supplémentaires prêts par rapport au
snapshot initial, et au moins un job encore manquant. On obtient 1 454 lots,
soit 15,82%. Leurs attentes finales hôte totalisent respectivement
1,545 / 1,685 / 1,665 ms par sortie, moyenne 1,631 ms.

Cette enveloppe est un plafond conditionnel pour supprimer l'attente finale
de ces seuls cas à latences figées, avant tout coût de construction,
soumission et calcul supplémentaires. Elle n'est ni du GPU idle corrélé,
ni un gain prédit, ni le plafond de toute politique possible de vagues.
Les soumissions W1 existantes durent déjà 65–68 µs par lot en moyenne.
Une vague de plus risque de consommer une grande partie du bénéfice. Il
faudrait aussi éviter de construire W2 complète puis reconstruire ses morceaux.
La piste reste disponible, mais n'est plus le prochain choix.

## Exclusions et limites de la recherche

- Ne pas enlever la soumission précoce des routes : POSTPROFILE03 a déjà
  montré le retard derrière Hot. L'attente routes P09 AR contient
  33,767 ms de GPU actif sur 33,863 ms inclusives : pas 34 ms récupérables.
- Lectures vers les buffers Metal et workers GCD existent déjà. Le lecteur
  asynchrone ne crée pas quatre nouveaux pthreads à chaque couche.
- Ne pas relancer ARC/pool, shared batching, stabilité des kernels ou la
  fusion gather×scale sous un nouveau nom sans différence concrète.
- Les traces P09 précèdent Dynamic ; le diagnostic Dynamic est MTP HOLD
  avec instrumentation native et sans corrélation GPU. Les gains d'essais
  différents ne s'additionnent pas. Cache fichier et NAND restent distincts.

Contrôles de cette recherche : lecture du code et des patches historiques,
extraction JSON par Python standard, vérification des 27 sorties de chaque
P09, des séquences 1..127 de chacune des trois traces Dynamic, des lots
terminés/masques monotones et des totaux d'échantillons ; empreintes de
15 entrées conservées. L'inspection initiale supposait à tort un objet pour
une trace stockée en liste et s'est arrêtée avant écriture ; inspection puis
extraction corrigée, aucune donnée modifiée ni mesure répétée. Deux chemins
documentaires exploratoires absents ont été remplacés par les fichiers
effectivement présents. Aucun code exécutable ajouté pour cette recherche,
donc aucun nouveau test moteur, Kani, GPU ou run CI. Aucun push ni app modifiée.

Décision : tester d'abord le placement shared **après** démarrage SSD ; garder
la projection complète en deuxième position ; ne pas commencer par W3.
Les deux premières pistes restent proposées, non implémentées/non mesurées.
