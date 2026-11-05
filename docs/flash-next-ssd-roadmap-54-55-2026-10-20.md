# Roadmap Flash Next : essais 54 et 55

Qwen3.8 Flash Next EXL3 2,05 bpw, M5 24 Gio, MLX 0.32.2.
P0/Xcode exclu à la demande de l’utilisateur. Aucun gain de débit livré par ces
deux essais ; le moteur de production conserve les calculs du profil 53.

## 54 : deux kernels exacts, sans gain global qualifié

Sources de départ `33f27ddf`, résultats publiés dans la PR34 au commit
`b2c3fe4a`. NT1/SG4 et ordre exact des additions conservés.

| Variante | Modification | Gain médian composant, selon couche/forme | Décision |
| --- | --- | --- | --- |
| Layout | Quatre partiels contigus, additions séquentielles inchangées | −0,071 à +0,357 % | Aucun cas n’atteint 1 % ; dérive d’un témoin à 13,29 % |
| Décodage | Une fenêtre K2 pour huit codewords | +0,289 à +1,509 % | Trois formes sur six restent sous 1 % |

Le contrôle A/A atteint lui-même 6,00 % de dérive sur une cellule, au-delà des
5 % prévus. Pas de répétition choisie après les résultats, ni de campagne
modèle déduite de ces chronos. Ces pourcentages concernent un MoE isolé chaud,
pas les tok/s. Les trois couches sont 0/23/47, formes B1/B2, quatre couples
alternés, chauffe d’au moins 2 s puis 256 forwards par échantillon. Aucun I/O
chronométré ni nouveau Swapout système dans ces trois processus.

Correction : 120 formes slots GPU et 24 mauvais slots détectés, puis sorties
MoE complètes bit identiques aux références indépendantes. Le témoin conserve
les mêmes tokens prétraités dans 162 configurations. Parser CPU Kani validé
sur chaînes de 0 à 6 octets, unwind8 : 287 obligations, 7 inatteignables,
4 couvertures. Cela ne prouve pas Metal ; le parcours Kani MLX rencontre une
erreur interne. Les prototypes sont retirés du moteur et conservés avec leurs
empreintes dans [les données54](measurements/flash-next-ssd-impl/opt-54-qmv/closure.json).

## 55 : l’allocation calculée sur MAIN ne généralise pas sur HOLD

Extraction des cohortes réelles du profil 53, préfill et decode séparés,
48 couches, OFF puis MTP. La tête MTP résidente est exclue du replay.
MAIN sert au choix, HOLD reste hors du réglage. Budget exact : 2 544 slots,
comme le cache uniforme 53 slots par couche, minimum20 pour les cohortes B2.
Le solveur de l’essai47 est réutilisé.

Les clés/fréquences LFU avant `trace_1` n’ont pas été capturées. Le replay
utilise donc le vrai `ExpertCache` Rust depuis un état froid, réexécute deux
générations identiques et compte les jobs uniques du second decode. Il ne
prétend pas reproduire les hits initialement observés dans53.

| Cas | Référence uniforme, jobs | Profil calculé, jobs | Réduction prédite |
| --- | ---: | ---: | ---: |
| OFF MAIN, réglage | 18 401 | 17 930 | +2,560 % |
| MTP MAIN, réglage | 19 904 | 19 419 | +2,437 % |
| OFF HOLD, contrôle | 19 708 | 19 799 | −0,462 % |
| MTP HOLD, contrôle | 22 186 | 22 306 | −0,541 % |

Le critère préenregistré demandait au moins 1 % sur les quatre cas : rejet.
Aucun benchmark GPU55 ni changement automatique du cache. Ce sont des
prédictions de lectures retournées, pas des mesures NVMe ou de débit.
Replay complet : 15,81 s CPU, footprint12 304 888 octets.

La première extraction utilisait le mauvais chemin gzip. L’échec reste
conservé avec SHA sous `input/extraction.json.gz` ; la correction est testée
sur gzip réel et corrompu, avec erreur bornée et statut explicite.

Vérification retenue : 245 tests Python ciblés, 133 tests Rust MLX/chat
passés et 129 explicitement ignorés ; sélection CPU5 passés/2 ignorés,
Clippy strict, format et déclencheurs CI PR/push vérifiés. Mutations détectées :
chemin gzip fautif, double comptage des misses, réglage utilisant HOLD.
Kani du compteur exact20 entrées expire deux fois à120 s ; le vrai replay
résident rencontre une erreur interne Kani. Ces propriétés restent non
vérifiées. CrossHair absent. Les tests ne constituent pas une preuve complète.

[Allocation, courbes et limites55](measurements/flash-next-ssd-impl/opt-55-cache53/closure.json),
[source et entrées mesurées](measurements/flash-next-ssd-impl/opt-55-cache53/replay-identity.json).

## Suite selon la dernière priorité utilisateur

1. Pipeline hybride GPU/CPU : préparer le calcul résident avant le retour des
   routes, conserver la protection des banques et mesurer le recouvrement SSD.
2. Groupement uniquement des experts effectivement répétés, singletons sur NT1.
3. Analyse de l’arrondi et du travail dupliqué avant une fusion Hadamard/QMV.

La spéculation adaptative reste dans la roadmap après ces travaux sur le
backbone. Aucun temps GPU exclusif acquis, aucune app installée remplacée,
aucune promesse de15tok/s à partir de ces essais.
