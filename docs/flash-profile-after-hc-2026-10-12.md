# Profil Flash après GEMV HC mixte — essai21

Diagnostic du prototype20, moteur7a9a2120/tests19a12f37, source36bae3ad
avec le patch HC mixte enregistré. La campagne courte20 n'a pas satisfait
son critère de stabilité ; ce profil ne valide pas son gain. La confirmation20B
est distincte. Apple M5/24Gio, MLX0.32.2, EXL3 2,05bpw65c89531.

| Coût instrumenté par pas | MAIN | HOLD |
|---|---:|---:|
| Parcours natif |101,21ms|101,63ms|
| Remplissage experts |27,67ms|27,15ms|
| Frontière de lecture des routes |54,75ms|55,70ms|
| Préparation HC/attention, passe intrusive |56,81ms|57,12ms|
| MoE, passe intrusive |73,46ms|73,85ms|
| Injection HC, passe intrusive |10,93ms|10,98ms|
| Vérification cible MTP / token émis |83,69ms|86,05ms|
| Part de la vérification cible dans MTP |93,86%|93,79%|

Le diagnostic16 avait une préparation intrusive à98,60/92,13ms et une
frontière de routes à96,76/90,01ms. Les frontières incluent le travail GPU
amont ; les durées intrusives incluent leurs barrières et ne s'additionnent
pas au parcours natif. Ce sont deux diagnostics séparés, pas une comparaison
appariée de débit release ni un profil GPU exclusif.

La vérification reste dominante. Le kernel HC mixte20 motive un essai
nouveau22 : partager chaque chargement de poids FP16 entre deux lignes de
vérification. Le prototype15 utilisait encore deux GEMV denses séparés ;
22 ne reprend pas ses projections EXL3/GDN/shared groupées. Aucun gain22
n'est encore mesuré. Les remplissages SSD restent autour de27ms par pas.

Un processus125,53s, MAIN/HOLD32pas, chauffe T1/MTP, cache5Gio/89slots,
allocator512Mio, lecteurs4, overlap1, contexte4096 et tête résidente.
Applications conservées, **batterie** (le protocole mentionnait secteur par erreur), pas de compilateur/proveur/autre modèle.
64logits complets et tous états cible exactement identiques entre passes,
MTP tokens/états exacts ;16propositions/16acceptations par prompt sur ces
préfixes32, qui ne remplacent pas les taux des générations64 ou128.
Swapouts système supplémentaires0octet ; RAMprocessus non mesurée.

[Commande, conditions et identité avant/après](measurements/flash-next-ssd-impl/opt-21-profile/execution.json),
[données intégrales](measurements/flash-next-ssd-impl/opt-21-profile/summary.json),
[sortie brute](measurements/flash-next-ssd-impl/opt-21-profile/run.log).
Le test GPU ne constitue pas une preuve formelle.
