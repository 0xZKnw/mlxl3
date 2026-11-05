# Après F02 : contre-audit depuis nos profils et notre code

1 novembre 2026 — POSTP08-LOCAL-REVIEW-02. Lecture locale uniquement : aucun
amont consulté, aucun code moteur/contrôleur modifié, aucun build, GPU, test ou
benchmark lancé. F02 reste OFF ; AR n’est pas concluant, MTP128 incomplet et
DF128 non lancé. L’admission RAM reste une condition distincte de cette revue.

## Le budget que nous avons réellement

[P08r3, fenêtre corrigée](../decode-profile-20261031/natural-r3/attribution-partial/README.md) :
40 transitions AR MAIN, harnais sans rétention de stages après chauffe,
104,5704 ms/sortie. Le run six phases n’est pas complet. Les spans, traces et
échantillons restent des observateurs ; aucune ventilation exacte du CLI ne
peut être déduite de cette fenêtre.

- Deux soumissions : 22,3854 ms wall, **11,8539 ms CPU Running∩GPU idle**.
  La première `slots.indices().async_eval()` porte 11,7676 ms exposées ; elle
  évalue aussi les producteurs précédents, pas seulement la cache-map.
- Readback routes : 36,7559 ms, dont 36,6258 ms recouvrent une commande GPU.
  Copie : **0,004312 ms**. Retirer le transfert ne rend pas 36,8 ms au moteur.
- Lots I/O : 32,5758 ms wall, 7,9900 ms CPU Running, **4,8431 ms CPU
  Running∩GPU idle**. Tail conditionnel après dernière commande indépendante :
  27,1664 ms, dont 26,9297 dans `preadv`. C’est la file POSIX observée,
  pas une mesure NAND ni une durée entièrement supprimable.
- Les 497 échantillons des soumissions exposées désignent encodeur, graphe,
  allocations et driver ; leurs proportions ne fournissent pas des ms par
  fonction. Les budgets ci-dessus se recouvrent et ne s’additionnent pas.

## 1. Vérifier les constantes réellement reconstruites dans GDN

**Appel concret.** `Gdn::load`, [qwen4.rs:211](../../../native/src/qwen4.rs#L211),
stocke `half_weight(...).transpose([0,2,1])`. `half_weight`,
[lfm2.rs:119](../../../native/src/lfm2.rs#L119), force F16 puis évalue le poids.
Donc `Gdn::convolve`, [qwen4.rs:304](../../../native/src/qwen4.rs#L304),
reconstruit bien une transpose puis **F16→F32** du même coefficient avant
chaque produit. Le chemin mono et la boucle de vérification multi-ligne
appellent ce même helper. Le scalaire epsilon F16 est aussi recréé dans la
closure de normalisation Q/K. `Array::astype` passe réellement à `mx::astype`
([array.rs:1101](../../../native/src/array.rs#L1101),
[mlx_bridge.cpp:512](../../../native/ffi/mlx_bridge.cpp#L512)).

**Ce qui est établi, et ce qui manque.** Ce cast ne peut pas être une identité
de dtype ni une simple réinterprétation des mêmes octets. La transpose peut
être une vue. Cette lecture ne prouve toutefois ni un dispatch autonome, ni
une allocation supplémentaire par token : fusion/optimisation/réutilisation
MLX doivent être observées. Temps propre et plafond récupérable : **inconnus**.
Ne pas lui attribuer les 11,85 ms de l’ensemble du graphe.

**Premier test falsifiable.** Avant cache persistant, capturer un GDN complet
sur entrée/état déjà évalués, identifier les conversions du coefficient et
leurs buffers réels ; distinguer les casts d’activations. Si elles ne donnent
aucun travail évitable, fermer cette piste. Sinon, qualifier uniquement une
préparation du même coefficient/epsilon à `load`, puis comparer le GDN complet
mono et tous préfixes multi-lignes, états/histoires/logits inclus, avant débit.

**Pièges.** Partir du poids **déjà arrondi F16 par `half_weight`**, pas du
checkpoint F32 original ; conserver forme/broadcast, epsilon F16 et toute la
somme/SiLU/norme stock. Aucun nouveau kernel requis. Pour 36 GDN, le payload
conv passe de 2,8125 Mio F16 à 5,625 Mio F32 : surplus net 2,8125 Mio seulement
si l’ancien stockage est libéré ; garder les deux ajoute 5,625 Mio. Compter
aussi les durées de vie réelles et l’effet donation. Ce n’est pas une nouvelle
fusion conv/récurrence PERF109/Rapid, ni l’épilogue sigmoid T17.

## 2. Cribler Cold par groupes prêts avant tout port T18b

**Frontière concrète.** Dans `SsdMoe::forward_rows_prepared`, le premier Hot est
soumis avant les routes ; ensuite `fill_many(&jobs)` attend **tout** le lot
([ssd_moe.rs:986](../../../native/src/ssd_moe.rs#L986)) avant construction du
calcul Cold ([ssd_moe.rs:1017](../../../native/src/ssd_moe.rs#L1017)). Le C++
reçoit tous les records et attend `read_disk_records`
([mlx_bridge.cpp:327](../../../native/ffi/mlx_bridge.cpp#L327)).

Différence falsifiable par rapport à T18a : exécuter le calcul d’un sous-groupe
Cold déjà complètement lu pendant les autres lectures ; T18a change seulement
la répartition des jobs entre lecteurs et garde la barrière globale de fin.
Le tail P08 de 27,1664 ms est un **plafond conditionnel très large**, pas une
estimation du gain T18b. Le calcul Cold devrait désormais consommer une partie
de cet intervalle ; nombre de groupes et coût GPU/CPU ajouté sont inconnus.

**Premier filtre sans GPU nouveau.** Pour chaque lot P08r3, utiliser les fins
des jobs et leurs slots réels pour calculer l’intervalle entre un groupe
admissible entièrement prêt et la fin du lot. Garder les petits lots et les
cas sans avance. Si cette avance est nulle ou trop courte relativement à un
calcul Cold complet, ne pas écrire l’ordonnanceur. Ce filtre mesure un slack
possible ; il ne simule pas gratuitement le coût de nouveaux kernels/commits.

**Pièges avant toute implémentation.** Ne publier que les slots effectivement
remplis ; garder les réservations, owners et utilisateurs GPU vivants jusqu’à
leur fin, y compris quand un autre job échoue. Le `last` unique actuel ne doit
pas perdre un groupe encore en vol. Conserver l’ordre canonique des 10 routes,
les poids et la réduction finale stock ; une concaténation dans l’ordre de
complétion réordonnerait les additions. Vérifier aussi le choix de kernel
quand le nombre de lignes du sous-groupe change. Trop de groupes peuvent
augmenter précisément le coût d’encodage déjà dominant. Le recouvrement
Hot/Cold actuel ne prouve pas automatiquement ce nouvel ordonnancement.

## 3. Amortir la préparation des records, sans supprimer les gardes I/O

**Chaîne locale.** `fill_many` recrée sources/destinations/vues
([ssd_moe.rs:301](../../../native/src/ssd_moe.rs#L301)).
`Array::read_records` refait les bornes de tous les segments, `File::metadata`
par source puis six vecteurs FFI
([array.rs:805](../../../native/src/array.rs#L805)). Le C++ reconstruit
pointeurs/capacités, valide/évalue les banques, puis un vecteur de destinations
par record ([mlx_bridge.cpp:300](../../../native/ffi/mlx_bridge.cpp#L300)).

Des capacités réutilisées et des descriptions immuables par couche pourraient
réduire allocations/marshalling, sans changer les lectures. Mais **aucun coût
propre n’est mesuré**. Les 4,8431 ms de CPU exposé dans tous les lots bornent
ce qui est observé sur cet axe ; elles incluent beaucoup d’autres tâches,
notamment du noyau dans les lectures. Ce n’est pas un budget acquis au C++.

**Premier test falsifiable.** Distinguer, sur le même lot et hors compilation,
préparation Rust, validation/marshalling C++, attente des workers et `preadv`.
Compter les allocations réellement présentes. Seulement si la préparation
exposée est matérielle : réutiliser d’abord les capacités/descripteurs en
préservant les mêmes validations, puis exercer les erreurs source/destination
et la reprise avant un micro A/B du chemin réel.

**Pièges.** Ne pas supprimer `last.eval()` ou les protections de slots, ni
réutiliser un pointeur de banque dont le propriétaire a disparu. Un cache de
longueur de fichier change la réaction à une troncature : le contrôle de
lecture courte et l’invalidation avant publication restent obligatoires.
Des banques déjà évaluées ne justifient pas de retirer aveuglément `a.eval()`.
Ce travail est distinct de l’ARC/pool partagé et des lecteurs dynamiques T18a.

## Pistes déjà réfutées ou insuffisamment attribuées

- **Une soumission au lieu de deux** : POSTPROFILE03 a échoué sur le véritable
  test d’overlap ; les routes attendaient le Hot retardé. Préserver les deux
  points de disponibilité. Le shared anticipé est aussi déjà 73T, avec
  contrôles physiques et écrans OFF/B2 ; sa position tardive dans le code
  actuel n’en fait pas une idée neuve.
- **Routage spéculatif/GDN différé** : GPU Hot existe ; notre ancienne borne
  du préfixe admissible était seulement 0,38–0,47 ms/token de GPU idle. Elle
  ne constitue pas un accès aux 11,85 ms CPU. Coût de rollback/rebuild et
  surfaces PLE/QSA restent à payer.
- **Cache de factory T01A** : le diagnostic source-seule n’exécute pas le
  graphe. Projection mesurée de sa closure : 0,0335 ms ; T01B pendant eval
  reste non mesuré. Ne pas transformer les 497 frames en budget de lookup.
- **Fusions petites** : POSTPROFILE05-B puis C corrigé, T04 retiré et T17
  exact après correction `precise::exp` n’ont pas qualifié de gain général.
  F02 réduit bien 8→5 dispatchs, mais son AR n’est pas concluant. Une baisse
  de dispatchs ou du temps hôte inclusif ne garantit ni idle réduit ni débit.
- **ARC/pool** : POOL04 ARC a donné −3,485 % MAIN/−2,145 % HOLD, plus de
  demande logique et coût de réservation/import ; runtime retiré après audit,
  replay conservé. Ni tout pool global ni ARC O(1) n’ont été réfutés, mais
  refaire la même politique n’est pas une nouvelle expérience.
- **T18a** : option déjà présente, Static reste défaut ; pilotes AR +2,15 %,
  MTP +2,94 %, DF −1,66 % dans des fenêtres distinctes. Aucun gain additionné,
  aucune reprise automatique du flag comme solution générale.

Je qualifierais d’abord les constantes GDN (petit risque numérique, budget
inconnu) et le slack Cold sur traces (potentiel d’ordonnancement, risque plus
fort). L’amortissement I/O vient après preuve de son coût propre. Aucune de
ces trois pistes ne reçoit ici un pourcentage de gain attendu.
