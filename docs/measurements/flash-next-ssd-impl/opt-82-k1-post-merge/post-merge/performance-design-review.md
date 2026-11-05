# Revue de conception K1 après fusion — 25 octobre 2026

La suppression Cold fonctionne et le candidat n’est pas un fallback silencieux. En revanche, « moins d’arithmétique Cold » ne signifiait pas ici « primitive globalement moins coûteuse ». Deux avantages de la référence ont été insuffisamment pris en compte : SwiGLU était déjà préparé par `mx::compile`, et Hadamard natif pouvait réutiliser son buffer par donation. Le custom garde par ailleurs toutes les grilles et les écritures des sorties, ainsi que les casts et les multiplications de scales.

La lecture de M est terminée. Aucun checkout, build, test, proveur ou benchmark n’a été modifié ou lancé par cette revue. Le parent peut retirer les cinq sources actives sans attendre une autre recherche.

## Identité et observation réelle

Base fusionnée `9c8474ca1292e29bae3c7d4d6383bd1820549349`, patch K1 `9782c74c49613acc520a38975376eefac5a52217941cfc254909702521dd358b`, manifeste de 100 sources `2b4eb648ff2e0c4362bb88b1705c2022a6acb6d1896f1f2766923ed8d1ca98c4`. Les noms de fichiers et lignes ci-dessous désignent cette source gelée, conservée dans les archives K1, et non un checkout susceptible d’être retiré ensuite.

DFlash2 B2 fixe, MAIN/HOLD, warmup128 puis mesure128 (127 décodés), cache3 Gio/53 slots, allocateur64 Mio, AC ; HC puis CH. Les 16 réponses sont identiques au gold. Les compteurs d’acceptation et les demandes logiques d’experts restent appariés.

| Ordre | Prompt | Référence H tok/s | Candidat C tok/s | C/H − 1 |
|---|---|---:|---:|---:|
| HC | MAIN | 11,399755 | 11,254234 | −1,277 % |
| HC | HOLD | 10,903516 | 10,896760 | −0,062 % |
| CH | MAIN | 11,576340 | 11,439044 | −1,186 % |
| CH | HOLD | 11,128820 | 10,946912 | −1,635 % |

Aucun gain de débit observé sur ces quatre paires. Ces répétitions courtes ne prouvent pas un ralentissement universel ni la cause exclusive. Voir [résultats complets](real-tps-review.md) et [JSON](real-tps-review.json). Les sorties de swap système sont nulles ; 65536 octets d’entrées système sont observés au post30 HC. La compression privée est non nulle. Ces observations ne sont pas une mesure des sorties de swap privées.

## Vérification du vrai chemin éligible

`ssd_moe.rs:520–542` : `prepare_resident_hot` appelle `routed_input`, qui effectue le `take` puis impose **Float16**. `ExpertSlots::residents` (`linear.rs:1275…`) fournit des indices UInt32 et **masked=true**. `calculate_slots` (`ssd_moe.rs:339–380`) conserve Half dans l’expansion GU, puis le chemin EXL3 restitue le dtype de son entrée : GU arrive donc bien en Half à l’activation, dont la sortie Half alimente down.

| Étape H128 | Forme B2 réelle | Repeat | Groupes |
|---|---|---:|---:|
| Entrée GU | [40,20,128] | 2 | 800 |
| Sortie GU | [40,5,128] | 2 | 200 |
| Entrée down | [20,5,128] | 1 | 100 |
| Sortie down | [20,20,128] | 1 | 400 |

Les quatre formes, les dtypes et la capacité53 satisfont les gardes du helper. L’activation reçoit `[20,2,640]` Half et lance 100 groupes de128. Pour B1–B8, les routes vont de10 à80 et restent dans le domaine du helper. Le chemin Hot reste conditionné par les options Hot/overlap, la profondeur0, la largeur autorisée et une map résidente non vide. La voie Cold ordinaire est unmasked et garde volontairement la référence.

Il s’agit d’une **garantie statique d’éligibilité des appels Hot**, pas d’un compteur réel des dispatchs durant le décodage M. Aucun `FAST_CALLS` du vrai décodage M n’a été collecté. Les compteurs des composants et les anciennes traces K1D ne doivent pas être présentés comme cette mesure absente.

## Différences de conception confirmées

1. **SwiGLU natif est déjà compilé.** `native/ffi/mlx_bridge.cpp:533–537` met `multiply(multiply(gate,sigmoid(gate)),up)` dans un `thread_local mx::compile`. `Array::swiglu` appelle ce chemin. K1 remplace donc un graphe compilé par un custom scalaire ; il n’a pas démontré une réduction de trois dispatchs vers un. Le booléen `true` de cette API signifie `shapeless`. Aucun capture du graphe généré n’autorise à affirmer son nombre exact de dispatchs ou sa vectorisation effective.
2. **H128 natif peut donner son buffer ; le custom alloue une sortie.** Le `Hadamard::eval_gpu` amont, lignes167–200, utilise `out.copy_shared_buffer(in)` si l’entrée row-contiguous est donatable. Le `CustomKernel::eval_gpu` de MLX0.32.2 utilise `allocator::malloc(out.nbytes())` pour chaque sortie sans branche de donation. Cette différence peut ajouter un coût de buffer et d’écriture sur les Hot. La donation réellement obtenue à chaque site et le coût correspondant ne sont pas mesurés ; une allocation MLX peut être servie par le pool.
3. **Cold est bien sauté avant les charges et barrières, mais la grille reste complète.** `hot_transform.rs:56–76` branche uniformément par groupe avant les lectures `values`, les radix et toutes les barrières. Les huit threads écrivent néanmoins les128 zéros positifs. Le candidat B2 lance encore1500 groupes H128 et écrit192000 Half, soit384000 octets. `hot_activation.rs:18–28` évite bien les lectures GU/SwiGLU Cold mais lance encore les100 groupes et écrit12800 Half. Les groupes d’activation sont uniformes :640 est divisible par128.
4. **Casts/scales et étapes voisines restent pleins.** `linear.rs:1437–1446` garde le cast Half et le produit SU avant le H128 masqué ; `1636–1647` garde le cast de sortie, puis SV et la restitution du dtype. K1 ne réduit ni les QMV EXL3, ni les demandes SSD, ni la réduction finale. Les31–32 % de positions Cold ne peuvent donc pas être convertis en pourcentage de temps MoE économisé.
5. **Le même custom s’applique aux géométries où il perd.** H128 conserve les8 threads/groupe natifs N128 ; ce choix n’est pas à lui seul une erreur de grille. En revanche, aucune sélection par rentabilité ne protège les tableaux de largeur640 ou les masques presque tout Hot. Les kernels élémentaires génériques MLX définissent `WorkPerThread<Half>=4`, alors que l’activation K1 fait un élément/thread ; cela motive une comparaison, sans démontrer la vectorisation réellement choisie par le SwiGLU compilé.

L’en-tête Hadamard installé MLX0.32.2 est byte-identique au fichier amont gelé : SHA `9c4b749b4d2985203416c2da42dbdf7d59070c6c49c81f6112907b6ce407edc5`, commit amont `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. Le `custom_kernel.cpp` primaire conservé a le SHA `92548fba86690996a30725ef96bb8e0e299b69a4559e71c60340e7e11c1cac5e` ; [source MLX0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp).

## Ce que les composants permettent de conclure

Les704 échantillons H128 couvrent32 cellules :17 médianes favorables et15 défavorables. Trois des quatre géométries allHot sont défavorables, jusqu’à−5,70 %. Les192 échantillons d’activation couvrent8 cellules :6 favorables,2 défavorables. Ce mélange confirme qu’il ne suffit pas de supprimer les calculs Cold pour garantir un meilleur chemin ; il ne permet pas d’attribuer le résultat tok/s à la donation ou à la vectorisation.

Ces micros mesurent du wall incluant construction/dispatch/fin GPU, pas du temps GPU exclusif. H128 répète32 fois la même transformation au lieu des quatre transformations intercalées avec QMV/scales. L’activation inclut31 récurrences communes. Les fixtures utilisent des masques réels mais des activations synthétiques et ne constituent pas une pondération de la trace entière. Les A/A sont variables ; aucun gain global ne doit être formé en additionnant ces cellules. Voir [revue des micros](micro-review.md).

La qualité physique et les six mutants MSL détectés étayent la correction du calcul et des masques dans leurs domaines testés. Ils ne prouvent pas un gain de performance. Les preuves CPU ne prouvent ni MLX/MSL ni l’allocateur/concurrence. Les43 requêtes bridge K1 et les tok/s OFF/MTP ne figurent pas dans cette campagne K1PM et restent non exécutés ici.

**Décision : conserver les preuves négatives et retirer K1 actif.** La faiblesse identifiée est un choix de remplacement trop peu attentif aux optimisations de la primitive de référence, pas une suppression Cold qui ne s’exécuterait pas. Aucun nouveau prototype ou benchmark n’est demandé par cette revue.
