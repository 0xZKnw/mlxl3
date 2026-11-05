# HC à deux lignes sur batterie — essai22

**Rejeté : aucun gain moteur22 validé.** Le gain20 a été
poussé dans la [PR34](https://github.com/0xZKnw/mlxl3/pull/34), commit
23292112fc16e0c8da691078394b83eaa32f3883. L’application installée reste inchangée.

Le kernel partage le chargement des poids FP16 entre deux vecteurs FP32,
avec le même ordre de FMA et de réduction que deux GEMV MLX0.32.2.
Seuls les HC sont groupés. L’attention et les historiques PLE restent causaux ;
le snapshot du premier token conserve son propre historique. Le chemin OFF
sert de contrôle. Le prototype15 groupait également EXL3/GDN/shared : cette
piste différente a déjà été rejetée et n’est pas réintroduite.

## Sources et protocole

Apple M5/24Gio, macOS27.2, modèle EXL3 2,05bpw révision65c89531.
À la demande de l’utilisateur, alimentation batterie avec témoins20 refaits.
Les résultats secteur20C ne servent pas de baseline numérique.

H : binaire20 SHA7a9a21205812dee34680df4b85600116b8ecb65c58d7897826206af54a8406a1.
C : binaire22 SHAa634e30e13764dabe995f3d88e82e04621e1a975fb12a1d0f9adf938e87bc2ac.
[Identité des sources et binaires](measurements/flash-next-ssd-impl/opt-22-hc-pair/identity.json).

MAIN/HOLD × OFF/MTP, deux tours H-C puis C-H ; par processus, une chauffe128
puis trois générations128 (127tokens de décodage), médiane des trois.
Contexte4096, cache experts5Gio/89slots, allocateur512Mio, quatre lecteurs,
overlap1. Applications de l’utilisateur conservées ; aucun autre modèle,
compilateur ou proveur pendant les mesures.

Critères préenregistrés : chaque paire OFF ≥−2%, MTP ≥+1%, étendue des gains
≤3points par cellule ; surplus du pic processus ≤256Mio, swapouts système
≤256Mio par bras, batterie avant/après, ratio débit max/min intra-bras ≤1,25.
Arrêt après le premier tour si un seuil de vitesse échoue ; arrêt immédiat
sur divergence, mémoire excessive ou dérive forte. Délai360s/processus,
arrêt et jonction de tous les enfants. Les64préfixes historiques et les128tokens
complets sont comparés avant de valider chaque réponse.
[Protocole et commandes](measurements/flash-next-ssd-impl/opt-22-hc-pair/speed/matrix.json).

## Correction

- Trois tests de primitives : 298matrices HC ; 1 056 392valeurs T1 et
  4 225 568valeurs T2 réelles, finies et bit à bit identiques à MLX T1.
  Oracle CPU indépendant sur21 128valeurs synthétiques ; strides et replis couverts.
- Mutation du vrai shader aliasant la deuxième ligne sur la première détectée ;
  restauration byteexacte et nouveau test réussi. Le mutant n’a pas été chronométré.
- Test HC pair : lecture/injection/écriture contre le parcours sériel,
  strides, formes invalides et refus de paire FP16.
- 12tests physiques Qwen, référence held indépendante11positions et3testsSSD
  réussis. Ensemble des oracles :16positions, logits complets et états pertinents.
- Bridge réel : budgets1/2/3/17, refus de configurations invalides,
  deux annulations/reprises et repli pénalité non unitaire validés.

`cargo fmt --check`, Clippy strict `--all-targets --features mlx,chat`,
build release `--features mlx,chat` réussis. Suite CPU :87passés/2ignorés ;
suite release :107passés/100ignorés (les tests physiques pertinents ont été
sélectionnés séparément). Voir [checks.json](measurements/flash-next-ssd-impl/opt-22-hc-pair/checks.json),
[qualité](measurements/flash-next-ssd-impl/opt-22-hc-pair/quality-inspection.json),
[mutation](measurements/flash-next-ssd-impl/opt-22-hc-pair/mutation.json).

Tentative Kani sur le vrai chemin MLX : trois i8 symboliques indépendants,
aucune hypothèse, deux lignes10240 et320sorties ; erreur interne du compilateur
avant toute obligation. CBMC ne traite pas le MSL. CrossHair absent pour les
nouvelles gardes Python, dont13cas frontières sont testés. Ces chemins ne sont
pas formellement prouvés ; les contrats CPU CI ne prouvent pas le GPU.

## Microbenchmark distinct du débit moteur

Cinq paires alternées,32évaluations par bras après8chauffes, exactitude vérifiée
avant mesure. Médianes T1×2 → T2 :

| Forme | Deux T1 | T2 | Réduction de temps |
|---|---:|---:|---:|
|10240→320|0,313ms|0,271ms|13,2%|
|320→10240|0,420ms|0,323ms|23,2%|
|10240→4|0,296ms|0,262ms|11,7%|

Ces chiffres incluent le lancement/évaluation et ne constituent pas un gain
end-to-end. [Sortie brute](measurements/flash-next-ssd-impl/opt-22-hc-pair/micro.log).

## Mémoire et limites

La suite de correction Qwen a ajouté1 168 244 736octets de swap système.
Ce chiffre ne peut pas être attribué à une génération isolée : le harnais
exerce plusieurs chargements et parcours. Les tests held, SSD, bridge réel
et micro suivants ont ajouté0octet. Le pic processus et le swap du parcours de génération ont été évalués
séparément dans chacun des bras du benchmark ci-dessous.
Le swap déjà occupé par macOS ne signifie pas que le modèle écrit encore
dans le swap ; les deux métriques restent distinctes.

## Résultat et décision

| Cas | H20, tok/s | C22, tok/s | Écart |
|---|---:|---:|---:|
|MAIN OFF|9,794|9,925|+1,34%|
|HOLD OFF|9,365|9,677|+3,34%|
|MAIN MTP|10,224|10,187|−0,37%|
|HOLD MTP|8,906|9,292|+4,34%|

Huit bras terminés,32réponses complètes exactes ; huit bras suivants non lancés
conformément au crible préenregistré. Le seuil MTP+1% échoue sur MAIN ; le
gain du micro ne se traduit pas en gain moteur retenu. Prototype retiré,
source20 conservée. Plus grand surplus de pic candidat43 647 000octets ;
25 362 432octets de swapouts système supplémentaires sur MAIN MTP C, zéro
sur les sept autres bras. Aucun dépassement des bornes mémoire.

[Inspection finale](measurements/flash-next-ssd-impl/opt-22-hc-pair/assessment.json),
[résultats](measurements/flash-next-ssd-impl/opt-22-hc-pair/speed/result.json),
[exécutions](measurements/flash-next-ssd-impl/opt-22-hc-pair/speed/execution.json).
L’objectif15tok/s n’est pas atteint.
