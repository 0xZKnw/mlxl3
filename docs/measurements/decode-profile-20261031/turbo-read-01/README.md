# TurboEXL3 relu directement après P08

31 octobre 2026. Lecture par le coordinateur, sans exécuter ni compiler
TurboEXL3, sans nouveau port moteur ou gain mesuré.

Sources primaires épinglées : [davidtai/turboexl3,
087c75724863e71710d4615453d3d131d8f7accb](https://github.com/davidtai/turboexl3/tree/087c75724863e71710d4615453d3d131d8f7accb).
[Identités des fichiers lus](source-identity.json). L'ancien catalogue #59
utilisait345bee9e ; T09/T10/T12/T01B y étaient déjà décrits. Cette lecture les
confronte au nouveau profil, elle ne les présente pas comme des découvertes.

## Premier choix : T09, mécanisme DEC-ONE adapté

[Dense.decCfgs/decLinear](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/src/dense.zig#L333)
sert1–4lignes. Il choisit une préparation séparée ou intégrée au GEMV. Le
dernier groupe arrivé pour un bloc de128sorties exécute le finish, après
publication des partiels ; les autres groupes terminent sans attendre.
Les compteurs sont entiers, réinitialisés pour chaque appel ; les additions
flottantes finales gardent un ordre fixe, sans atomiques flottantes.

Le [kernel d'arrivée](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/src/kernels/dense/dec_gemv_arrive.metal)
montre les fences device, le compteur et le retour des groupes non derniers.
Le [finish](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/src/kernels/dense/dec_finish_v.metal)
additionne les plans dans l'ordre des slices. Ce n'est pas une boucle de spin.
La variante intégrée a toujours un lancement de zéro des compteurs : «un GEMV»
ne signifie pas un seul dispatch au total.

Cela correspond au poste observé dans P08 : graphes, allocations, bindings,
encodeur et barrières de nombreuses petites opérations. Notre voie dense
Flash conserve cast/produit/H128, QMV partiel, éventuelle somme, puis
cast/H128/scale/cast. `MLXL3_FUSE_OUTPUT` existe dans le shader générique,
mais les entrées Flash concernées le fixent à0 ; cette branche n'est pas un
DEC-ONE Flash déjà qualifié.

**Portage direct exclu ; mécanisme à adapter.**

- `ksFor` vise160threadgroups, explicitement quatre par cœur d'un GPU40cœurs.
  Ce réglage ne se copie pas au M5 standard. Changer le nombre de splits change
  aussi l'ordre des sommes ; les splits MLXL3 doivent rester canoniques pour
  le contrat exact.
- Préparer dans chaque groupe peut répéter H128 et augmenter les registres/
  mémoire partagée. Commencer avec préparation séparée permet de tester le
  mécanisme de finish sans ajouter cette autre variable.
- Turbo dit byte-identique à **sa propre chaîne**, elle-même à tolérance de
  son oracle. Notre produit d'entrée F16 matérialisé et nos arbres H128,
  arrondis, codebooks et réductions doivent être reproduits. Ses fragments
  `dec_prep1` et `dec_finish_store` ne prouvent aucune parité avec MLXL3.
- Les compteurs et plans temporaires doivent appartenir à chaque appel,
  rester vivants jusqu'à consommation et fonctionner pour appels répétés,
  GPUHot, graphes différés et erreurs. Les compteurs ont au moins8éléments
  chez Turbo pour éviter le binding MLX des petites constantes : détail à
  conserver ou remplacer explicitement, pas à oublier.

Une première cible raisonnable est la projection dense6144→2560 K4
déjà capturée/oraclée : préparation stock, réduction/finish regroupés, mêmes
partiels et somme, puis seulement envisager la préparation intégrée.
Ce n'est encore qu'une proposition ; aucun essai T09 n'a été lancé.
L'essai devra inclure initialisation, mémoire et projection complète, puis
parité logits/états/bridge avant tout A/B modèle. Le plafond propre à T09
reste inconnu ;11,85ms est le budget de toutes les soumissions exposées.

Comparaison de source : `ksFor` et `decCfgs` sont identiques à345bee9e.
Les six blobs Metal DEC-ONE inspectés sont inchangés ; dans `decLinear`, le
changement lu concerne l'état du message debug unique (`Global`), pas les
lancements. Ce n'est pas un nouveau kernel à rejouer pour le seul changement
de commit. La justification d'un futur essai est son adaptation MLXL3 exacte
et le profil nouvellement attribué.

## Pistes suivantes et exclusions

| Piste | Correspondance réelle | Priorité/limite |
| --- | --- | --- |
|T12 `gatedDecode`|[decb.zig](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/src/decb.zig#L395) lit directement gate/strides et réunit sigmoid, produit et préparation out.|Après T09 ou filtre ciblé moins risqué ; respecter les valeurs F16, pas importer la table de sigmoid comme une approximation.|
|T12 merge QSA|`mergeGatedDecode` attend des plans split-K de son cœur QSA.|Notre chemin SDPA ne produit pas ces plans ; pas de port direct. Examiner séparément layouts query/gate et indexeur/pooling.|
|T01B lanceur préparé|[kit/kernel_launch.zig](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/kit/kernel_launch.zig) réutilise kernels/configs et délègue à HostGpu.launch. Sans host lié, fallback mlx-c standard.|Notre cache factory128 existe déjà. La partie eval/pipeline est distincte ; aucune preuve qu'elle récupère les12ms. T01A construction seule a déjà un petit budget.|
|T10 GEMV lean|Les chunks6–8lignes réduisent les entrées vivantes tout en gardant les FMA ;9–16lignes utilisent deux chunks.|Pertinent si ces formes sont chaudes ; pas une accélération AR1ligne présumée.|
|Lecteur de codewords R64|[helpers.metal](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/src/kernels/lane/helpers.metal) réutilise une fenêtre64/96bits par lane.|Hypothèse GPU K2/K4, pas remède direct au coût CPU ; auditer nos chemins K3 existants et préserver les valeurs décodées.|
|Préfill NAX/GEMM|[kit/nax.zig](https://github.com/davidtai/turboexl3/blob/087c75724863e71710d4615453d3d131d8f7accb/kit/nax.zig) teste GPU M5 et OS ; dense17–32lignes a une voie registres, puis matmul/déquantification au-delà.|Vraie piste future de préfill par blocs, mais Flash local reste séquentiel et ses experts ont de petits lots. Ce sont les accélérateurs du GPU, pas l'ANE séparé.|
|Tronc FP16 résident/unrotated|Turbo peut garder les poids déquantifiés et joindre des projections.|Budget RAM potentiellement prohibitif sur24Go et arrondis différents pour unrotated. Pas de cache FP16 de tous les experts à3Gio.|
|Fusion router/routed/shared|Partage des launches chez Turbo résident.|Préserver la disponibilité précoce des routes et le départ SSD local. Sa voie router fold servie2–4lignes ne supprime pas gratuitement le readback AR.|

Les résultats T04/SwiGLU, T17, pools/ARC et T18 restent attachés à leurs essais
historiques. Une fusion dense complète est distincte de la petite frontière
SwiGLU×suh retirée ; «moins de dispatchs» n'est pas en soi un gain de tok/s.
Les performances M5 Max/résidentes de Turbo ne sont pas reprises comme gains
disponibles sur notre M5/3Gio/SSD.

La lecture couvre README, KERNELS, dense/decb/decode/lane, kit/sdk et les
fragments Metal pertinents ; pas l'intégralité des backends ni les benchmarks
upstream exécutés. Licence et NOTICE ont été consultés. Aucun code Turbo
n'est intégré ou activé dans MLXL3 par cette revue.
