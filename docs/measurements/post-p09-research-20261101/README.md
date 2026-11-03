# Pistes decode après P09 — 1 novembre 2026

Révision ultérieure : Dynamic + Cold U a été qualifié puis activé localement.
Le prototype étroit de lanceur prepared reste OFF après un pilote AR négatif.
Le [nouveau classement fondé sur les traces et le code](../post-p09-targeted-research-20261101/README.md)
remplace les priorités de ce rapport historique sans en effacer les hypothèses.

**Recherche terminée ; aucun nouveau candidat exécuté, aucun gain annoncé.**
Premier essai recommandé : composer les lecteurs dynamiques avec Cold U,
aujourd'hui mutuellement exclusifs. Ensuite viser le coût d'évaluation/encodage
MLX avec un lanceur préparé, pas un nouveau cache de factories.

Les chiffres viennent du [rapport P09](../../decode-profile-p09-20261101.md),
publié dans [l'issue65](https://github.com/0xZKnw/mlxl3/issues/65).
[Extraction, calculs et empreintes locales](evidence.json),
[sources primaires consultées](upstream.json). Source locale HEAD
d5a776e53425bb4d0ba61c0774c5399ae7987c06 avec modifications P09/MTPHEAD
préexistantes conservées ; MLX installé0.32.2. Recherche sans GPU/build/proveur.

## Distance réelle à15tok/s

MAIN, médianes release P09 ; chaque mode reste séparé.

| Mode | Débit actuel | Temps/token | Temps à retirer pour15 | Hausse de débit nécessaire |
|---|---:|---:|---:|---:|
| AR |11,487tok/s|87,055ms|20,389ms|30,58%|
| MTP |12,137tok/s|82,391ms|15,725ms|23,59%|
| DFlash B2 |12,463tok/s|80,235ms|13,568ms|20,35%|

Ce ne sont donc pas seulement2–3%. Les petits gains restent utiles en
composition ; leurs pourcentages ne s'additionnent pas. Le gain prefill TAIL2
n'est pas compté comme gain decode. HOLD et contexte long restent à qualifier
pour chaque futur ensemble.

## 1. Lecteurs dynamiques + Cold U : meilleur premier essai

Constat vérifié dans `native/src/ssd_moe.rs` au chargement :
`cold_ready = cold_ready && !MLXL3_SSD_DYNAMIC_READERS`.
`native/ffi/disk_read_batch.h` distribue encore les jobs par bandes fixes,
`i = worker; i += workers`. Le lecteur qui termine tôt ne reprend pas le
travail d'un autre. `DiskWorkQueue` dynamique existe déjà, mais seulement dans
le lecteur synchrone. **Mettre le flag actuel à1 ne teste pas la composition :
cela retire Cold U.**

Nouvelle différence avec T18a : conserver la première vague GPU et la
construction anticipée W2, tout en distribuant dynamiquement les lectures
restantes. Les anciens résultats T18a (+2,15% AR,+2,94% MTP,−1,66% DF,
campagnes différentes et dérive) sont mixtes et ne prédisent pas ce résultat.
Ils sont dans le [bilan T18a](../issue59-t18a-20261031/README.md).

P09, fenêtre commune87–113, temps CPU **Blocked ∩ GPU idle** :

| Phase | AR ms/sortie | MTP ms/sortie | DFlash ms/sortie |
|---|---:|---:|---:|
| Attente première cohorte `cold_ready_wait` |8,3570|4,2957|3,7303|
| Jointure finale `cold_ready_finish` |4,7443|10,3407|9,1712|

Ces enveloppes montrent où chercher. Elles ne mesurent pas le seul déséquilibre
des lecteurs : file système, latence commune, contention et vrai travail de
lecture subsistent. Le plafond propre du rééquilibrage reste **non mesuré**.
Les enveloppes POSIX exposées globales P09 (AR16,87/MTP15,31/DF13,43ms)
se recouvrent avec ces phases et ne doivent pas être additionnées.

Port minimal proposé : réutiliser la file existante dans `DiskReadBatch`,
garder4workers,3Gio,les mêmes jobs et destinations, la publication après
record complet et les consommateurs Hot/W1/W2 protégés. Aucun MLX dans les
workers. Préserver aussi le mode statique. Risques : contention atomique,
perturbation des premières lectures, collisions mémoire/I/O et invalidation
des slots sur erreur. Aucune nouvelle politique de cache nécessaire.

Premier filtre, à préenregistrer avant implémentation : analyser le déséquilibre
des fins de workers dans les traces actuelles ; ne pas interpréter un replay
à latences figées comme une prédiction de SSD. Puis tests de jobs retardés,
couverture unique/exhaustive, erreurs/annulation et jointure avant libération,
oracle physique Cold avec chevauchement réel, états/logits/bridge. Comparer
Static+U à Dynamic+U, même binaire ; ne pas réintroduire Dynamic sansU comme
nouveau candidat. Pilote AR/MTP/DF court alterné, qualité avant vitesse.

## 2. Préparer le lancement pendant l'évaluation, puis les chaînes stables

Dans P09, `host_async_submit` contient12,4327ms AR,9,3226ms MTP et7,9399ms DF
de CPU Running pendant GPU idle. C'est le budget de **toutes** les soumissions
dans la trace, pas le gain du seul changement proposé.

Le [filtre T01A](../issue59-t01-20261031/README.md) n'avait mesuré que la
construction d'un Hot GU sans évaluation :5,529µs/appel, projection historique
0,265ms pour48appels. Il ne réfute pas T01B. Dans
[CustomKernel::eval_gpu de MLX0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp),
chaque évaluation alloue les sorties, construit les listes d'entrées,
rehache le source, cherche la bibliothèque/pipeline et encode les bindings.
Bibliothèque et pipeline sont déjà cachées : **pas de recompilation JIT par
appel présumée**. Les samples P09 pointent surtout encodeur/dispatch/barrières ;
ils ne prouvent pas que le hash domine.

Premier contrôle utile : qualifier le coût de cette évaluation sur une vraie
chaîne Cold, puis une primitive/lanceur gardant la pipeline et les métadonnées
stables, avec entrées et scratchs explicites. Garder exactement les shaders,
arrondis et frontières SSD. Une chaîne préparée peut amortir le graphe et
certains allocations/bindings ; elle n'élimine pas magiquement les dispatchs.
Commencer étroit, avant un backend Metal entier ou des ICB.

Risques : ABI interne MLX, entrées mutables capturées/stales, donation,
réutilisation prématurée de scratchs, barrières et propagation des erreurs.
Le premier `slots.indices().async_eval()` doit rester une sortie précoce :
POSTPROFILE03 a déjà démontré qu'enlever cet appel retarde les routes derrière
Hot. Préparer un graphe n'autorise pas à l'évaluer avant publication des poids.
Coût propre et gain modèle **non mesurés**, pas12ms promis.

## 3. Une troisième vague Cold, uniquement si elle remplit un trou réel

Aujourd'hui `fill_ready` prend un snapshot après deux experts prêts, soumetW1,
prépareW2 et attend **toutes** les autres lectures. En B2, la jointure finale
reste longue dans P09. Un groupe intermédiaire déjà prêt pourrait occuper le
GPU avant le dernier lecteur ; les occurrences doivent ensuite retrouver
l'ordre Top10 canonique, sans changer la somme.

Ce n'est pas le premier port Cold déjà livré ni la varianteD abandonnée qui
supprimait certaines soumissions. Limiter à trois vagues ; ne pas créer un
graphe par expert. Avant code, rechercher dans les traces des groupes assez
larges/précoces pour payer construction+soumission+calcul. La première vague
actuelle coûte déjà, en AR,0,653ms de construction et1,819ms de soumission
CPU/GPUidle cumulées par sortie ; la seconde valeur est incluse dans les
12,433ms de soumission totale. Les extrapoler àW3 serait invalide.

Priorité après le rééquilibrage : celui-ci peut supprimer une partie des mêmes
trous. Plafond propre non mesuré, enveloppe de jointure ci-dessus seulement.
Risque principal : plus de travail CPU et petites cohortes GPU inefficaces.

## 4. DEC-ONE exact : potentiel plus large, port plus risqué

[Turbo, arrivée du dernier groupe](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/src/kernels/dense/dec_gemv_arrive.metal)
écrit les partiels, publie leur disponibilité puis laisse le dernier groupe
exécuter la fin de projection, sans spin des autres. Le mécanisme reste
distinct de F02 (petit épilogue dense) arrêté par l'utilisateur. Ne pas
réactiver F02/GDN silencieusement.

Commencer avec préparation séparée, projection6144→2560K4, splits et ordre
MLXL3 inchangés. Compter remise à zéro des compteurs, finish complet et scratchs.
Ne pas importer sa transformation affine MUL1 différée, ses arrondis ni son
réglage160groupes/40cœurs sur M5standard. Risques de publication intergroupes,
registres et contention ; plafond propre non mesuré. La
[lecture détaillée antérieure](../decode-profile-20261031/turbo-read-01/README.md)
reste applicable. Priorité derrière les essais plus petits ci-dessus.

## Recherche amont : pistes filtrées

- **MLX configuration dynamique** : [PR4647](https://github.com/ml-explore/mlx/pull/4647),
  fusionnée27octobre, permet de changer des options à l'exécution ; le code
  Metal relit désormais les seuils lors de `needs_commit`. Notre MLX0.32.2
  les fige dans `Device`. Intérêt potentiel : réglage decode distinct du
  prefill, sans reprendre une grosse hausse globale et son pic mémoire.
  Mais les essais20/80 anciens étaient mixtes et la nouvelle API ne prouve
  aucun gain. Vérifier les coupures réelles avant changement de dépendance.
- **Piège identifié puis corrigé** : notre banque GU fait43 417 600octets
  =41,40625Mio, mais le runtime0.32.2 additionne `array::data_size()` en
  **éléments**. Sa vue UInt32 compte10 854 400éléments (~10,35unitésMi), pas
 41,4. Elle ne force donc pas seule le seuil40. Ce contresens initial ne
  sert pas de justification. Source :
  [device.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L345)
  et le contrat du header installé `mlx/array.h:346`. Le commentaire amont
  [#4521](https://github.com/ml-explore/mlx/issues/4521#issuecomment-5949792521)
  signale aussi ce point. Les résultats M3Ultra/M5Max ne sont pas transposés.
- **Sushi vérification différée** : [contrat courant épinglé543c25f](https://github.com/beamivalice/sushi/blob/543c25fdd0e1907bf28ad22ed60830b8a536e994/docs/engine-expert-streaming.md).
  Il garde un successeur GDN spéculatif et reconstruit en cas de miss ; pas
  une accélération générale B2. Notre Hot part déjà avant le readback et
  garde les contributions utiles même avec misses. L'ancien diagnostic
  local ne trouvait que13,3–13,6% de transitions admissibles entièrementHot,
  soit beaucoup de reconstructions potentielles. Ces taux sont historiques,
  pas remesurés P09 : priorité basse sans sélecteur meilleur démontré.
- **Fenêtre K2, petits épilogues, ARC/pool** : ne pas les renommer en idées
  nouvelles. T02 a déjà confronté la fenêtreK2 aux essais Flash54Q ; F02/GDN
  et pools ont leurs résultats et audits. Lire notamment
  [T02](../issue59-20261029/T02-read.md). Le denseK4 reste distinct mais
  son coût n'est pas attribué ; priorité derrière le CPU/I/O observé.

## Clôture

Les trois fichiers P09 sont complets,parité qualifiée,27sorties87–113 ; leurs
empreintes sont conservées. Extraction initiale stoppée par une attente de
nom de statut incorrecte, corrigée après inspection sans changer les données.
Calculs arithmétiques, chemins et diff documentaire contrôlés. Aucun code
exécutable modifié : pas de nouveau test GPU, Kani ou CI ; preuves antérieures
non réattribuées aux pistes proposées. Aucun push, publication ou app modifiée.

Décision : **recherche close, candidats proposés non implémentés**. Garder
ColdU comme référence et tester d'abord la composition avec les lecteurs
dynamiques ; les plafonds ci-dessus restent des enveloppes diagnostiques
instrumentées, non des millisecondes release garanties récupérables.
