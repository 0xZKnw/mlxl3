# Issue 59 — T03/T04/T05/T06, frontières exactes et premier candidat

29 octobre 2026. Lecture de sources seulement : aucun code exécutable modifié, aucun modèle, benchmark, build ou proveur lancé. Aucun gain ni nombre de dispatchs MLXL3 n’est établi par cette revue.

## Identités et sources

- TurboEXL3 : commit `345bee9e35920ee77292055d873fca891146529c`, lu via le connecteur GitHub. `fin7` est construit par concaténation de fichiers ; `fin7.metal` n’est pas un fichier autonome à cette révision.
- MLXL3 : worktree `7978fceac30d991a8c06d72114fc932b2443f555` modifié, après retrait du runtime POOL04. Sources réellement examinées : `ssd_moe.rs` SHA-256 `f5145c58c83da2e6e08920b83eddc069f5fd4523759162e1234d042441522672`, `linear.rs` `42106754300edc10d4b73b24a63720681626805acd0d97e27737642b973be18e`, `mlx_bridge.cpp` `efc08fcbe039e24a5cc0dac7e9483ae612a8a09fef2e285559262c13aab4ccc8`. Le snapshot publié de l’issue, `f3f2c1d5…`, ne remplace pas cet état local.
- [POSTPROFILE05](../post-sushi-20261028/README.md), ses résultats B/C et son archive ont été relus. Le port corrigé C avait supprimé les vues supplémentaires, gardé les Hadamard natifs, passé les oracles, puis échoué à qualifier un gain release.
- Snapshot local de l’issue 58 : `build/post-sushi-20261028/issue58.md`, SHA-256 `4c4ff3f867af7f57313efd332cf2d52572353c0b585731866529290058c1ca55`.

Blobs Git amont utiles, tous sous le commit Turbo épinglé :

| Fichier | Blob Git | Ce qui a été vérifié |
|---|---|---|
| [decode.zig](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/decode.zig#L1488-L1603) | `25889e8e0fe792781839ef11bf293b33391be2ce` | Assemblage fin7, shared, et protocole gu3sf |
| [fin7_4.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/fin7_4.metal) | `1c4a31c391004807f1ce8e5cd9b40b22efdb5254` | H128, scales gate/up, activation et préparation down |
| [fin7_planes.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/fin7_planes.metal) | `45cbd83ace0d828280201f9a25fed4511c7cc241` | Somme séquentielle des plans split-K |
| [dfin7.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/dfin7.metal) | `e9c3937a4359089c7c8ff3e685330ee063e76f33` | Finish down puis chaîne FMA pondérée des rangs |
| [dfin7s.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/dfin7s.metal) | `9f5e13307bc16bfd8d68c73fcd44f394e0faa4f4` | Branche shared et addition finale |
| [shared.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/shared.metal) | `a0074181be22e1f30ee9be4b327978b67193d8fb` | Décodeur dense séparé, préparation et finish |
| [shared_fin_head_4.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/shared_fin_head_4.metal) | `67947dd5ed37f91afdae97c493b281f3322cfa38` | Table sigmoid et intermédiaires OUT_T |

## Ce qui ne peut pas être copié tel quel

Le routé local termine chaque projection par réduction des plans de référence, cast F16, Hadamard MLX, produit avec `svh`, puis cast de sortie. À l’entrée, le produit entre activation et `suh` est F16 avant le Hadamard. Les splits du chemin SSD sont indépendants du nombre de routes et doivent le rester.

`fin7_4` travaille avec ses H128 en registres F32, associe le facteur H128 aux scales de sortie avant un cast, puis multiplie l’activation arrondie par `down_suh` en F32 avant le prochain H128. Cela ne reproduit pas automatiquement les frontières F16 locales. Le Hadamard MLX installé utilise aussi un buffer de type de sortie entre ses passes radix ; garder seulement le facteur `1/sqrt(128)` ne suffit pas à prouver l’identité.

`dfin7` effectue la pondération au moyen d’une chaîne de FMA dans l’ordre des rangs. Le local caste d’abord les contributions vers le dtype de x, multiplie les scores, puis appelle `sum(1,false)` MLX sur `[tokens,10,2560]`. Ni une chaîne FMA, ni dix additions dans un ordre choisi ne sont un remplacement établi de cet arbre.

Le SwiGLU local est **déjà une fonction `mx::compile`** (`mlx_bridge.cpp:569–572`), qui contient sigmoid et ses deux produits. Il ne reste pas trois lancements indépendants sigmoid/mul/mul à supprimer sur cette seule base. La table sigmoid Turbo et ses intermédiaires OUT_T ne sont pas présumés identiques à cette chaîne compilée, notamment pour les arrondis F16.

## Tri des quatre propositions

| Piste | Déjà essayé ou présent | Frontière réellement distincte | Limite avant prototype |
|---|---|---|---|
| T03 | #58/07 correspond à POSTPROFILE05 B puis C : gathers/scales fusionnés, H128 natif gardé ; C non gagnant après correction. K1 avait déjà remplacé la préparation/H128. | #58/11 peut encore supprimer `routed_input`/expansion en adressant la ligne source depuis le rang, ou produire directement un layout consommé sans autre gather. | Démontrer les copies réellement matérialisées ; un broadcast ou reshape seul ne prouve aucun lancement. Ne pas relancer POSTPROFILE05-C inchangé. |
| T04 | Gate/up projetés ensemble ; SwiGLU déjà compilé ; préparations et fins EXL3 restent séparées. | Un petit épilogue SwiGLU + produit F16 d’entrée down, en gardant le H128 natif ; ensuite seulement fusion des scales de sortie GU. | Préserver l’arrondi de la sortie SwiGLU puis celui du produit `suh`. Copier fin7 complet réintroduirait les changements H128/arrondis rejetés. |
| T05 | Concaténation Hot/Cold, restauration par `take`, cast, pondération et somme sont encore exprimés séparément. Un copieur 80H a été préparé ailleurs, sans exécution retrouvée. | Consommateur unique sélection + cast + pondération, puis **somme MLX inchangée**. Les producteurs et leur overlap restent identiques. | Capturer le coût et les lancements des copies/consommateurs actuels ; aucune réduction Metal nouvelle dans la première variante. |
| T06 | 68 groupe les trois projections shared sur plusieurs lignes ; 73 anticipait leur calcul sans retirer leur chaîne de lancements. Aucun gain global qualifié par ces essais. | Travaux shared dans les grilles routées, avec poids et décodeur propres, sans recalcul Hot et Cold. Le finish au dernier groupe est une variante plus profonde. | K4 shared contre K2 routé sur le checkpoint local ; sigmoid/casts différents ; publication des partiels et compteurs privés nécessaires au fin fold. |

Le snapshot 58 conservé appelle **/42 le replay GDN accepté**. La fusion du consommateur Hot/Cold est **/43**, et l’écriture canonique par les producteurs est /28. La référence T05 « #58/28 et /42 » de l’issue 59 semble donc contenir un décalage de numéro ; cela ne doit pas faire relancer une expérience GDN pour ce travail.

Les essais [65](../../flash-next-ssd-grouped-exl3-65-2026-10-21.md) et [66](../../flash-next-ssd-fixed-width-66-2026-10-21.md) partageaient le décodeur entre occurrences d’experts connus CPU, avec plan et restauration, sans qualifier un gain général. La voie Hot masquée garde un chemin distinct. T05 ne change ni le plan de regroupement ni les calculs des experts. [68](../../flash-next-ssd-shared-expert-68-2026-10-22.md) et 73 ne démontrent pas qu’une fusion de lancements T06 serait inutile ; ils interdisent seulement de présenter leur batching ou déplacement inchangé comme nouveau.

## Premier candidat proposé : T05 réduit à son consommateur pondéré

Commencer **uniquement sur le chemin mixte Hot + Cold**. Un consommateur lit la contribution choisie directement dans l’une des deux sorties, applique le cast et le produit déjà existants, et produit les contributions pondérées dans l’ordre canonique. Le `reshape([rows,10,H])`, `sum(1,false)` et l’addition shared restent ceux du moteur actuel. Les chemins all-Hot et all-Cold restent des contrôles inchangés dans cette première variante.

Cette frontière retire des matérialisations après les projections, sans toucher aux FMA, split-K, Hadamard ou sigmoid. Elle peut raccourcir le graphe soumis par le CPU, tout en laissant Hot soumis avant le readback des routes et les lectures Cold. C’est un candidat à observer puis tester, pas une affirmation que MLX réalise aujourd’hui un lancement par nœud.

### Contrat de dtype exact

Soient `Tx = x.dtype`, `Tw = weights.dtype`, `Tp = promotion_MLX(Tx,Tw)`. Les contributions source sont F16. Le consommateur doit réaliser **dans cet ordre** :

1. Sélectionner une contribution F16 unique depuis Hot ou Cold.
2. La convertir vers Tx, comme `values.astype(x.dtype())`.
3. Effectuer la promotion vers Tp, puis multiplier par le poids promu vers Tp.
4. Stocker un produit de type Tp ; appeler ensuite la somme MLX existante.

| Tx | Tw | Tp / sortie pondérée |
|---|---|---|
| F16 | F16 | F16 |
| F16 | F32 | F32 |
| F32 | F16 | F32 |
| F32 | F32 | F32 |

Le dtype ne se déduit pas seulement du type de la banque. Préserver le cast **avant** la pondération, le score par rang top10 et le produit séparé de la somme. Aucune accumulation FMA ni addition de zéro masqué dans ce kernel.

### Sources Hot et Cold

- **Hot GPU anticipé** : sa sortie physique contient `rows*10` lignes, y compris les lignes masquées. La limite entre les deux tableaux est donc `hot.shape[0]`, **pas le nombre de hits**. Les rangs Cold choisissent seulement leur ligne Cold compacte ; les zéros masqués Hot ne participent jamais au produit ou à la somme.
- **Hot compact historique** : sa hauteur physique est le nombre de contributions Hot. La même table d’origine existante indique la ligne Hot ou Cold ; ne pas supposer que tous les hits sont un préfixe.
- **Hot déjà calculé avant découpage de cohorte** : conserver la vue capturée et sa table d’origine. Le consommateur ne doit ni relancer le calcul ni lire les banques expertes après une éviction autorisée.
- **Doublons d’experts** : une occurrence reste une contribution distincte avec son entrée et son poids, même quand l’expert est le même. Une source peut aussi être répétée si la table de `take` l’autorise ; une table d’origine n’est pas nécessairement une permutation de toutes les lignes physiques.
- **All-Hot / all-Cold** : laisser leurs branches actuelles intactes au premier essai. S’ils sont étendus ensuite, utiliser une identité de rang et le même contrat Tx/Tw/Tp ; n’introduire aucun faux buffer Cold/Hot ou lecture d’un placeholder.

Les deux sorties sont des dépendances GPU explicites du nouveau consommateur. Ne pas ajouter de `to_u32`, `eval` ou barrière CPU. Garder la rétention `last`, les protections de slots, les erreurs/annulation et la soumission Hot initiale. Les entrées réellement stridées exigent des strides corrects ou la copie déjà nécessaire ; aucune contiguïté supplémentaire ne doit être supposée silencieusement.

### Antécédent 80H à réutiliser, pas à renommer

Le patch préparé [80H](</Users/justin/Documents/mix-stq1_0/build/flash-opt-80/moe/80H/source.patch>) SHA-256 `d40e6b1be7cbc907d8c1a74dbda7ac8ae175127bf3645e81bae1614305b8f2f4` est un copieur de bits F16 avec sélection Hot/Cold. Il laisse cast, pondération et somme inchangés. Son `commands.json` porte `prepared_not_executed` pour les étapes retrouvées ; aucun gain ou test exécuté n’en est inféré. Son source et ses oracles peuvent servir de contrôle de sélection, sans écraser ce travail privé ni réutiliser des identités obsolètes.

La nouveauté de la proposition T05 étroite est d’absorber **aussi cast et pondération**, en gardant la somme stock. Une simple réapplication du copieur 80H ne répondrait pas à cette distinction.

### Vérification requise avant toute vitesse

Préenregistrer d’abord le candidat et la baseline finale. Montrer, sur une capture physique hors chrono release, que les copies/consommateurs ciblés existent et qu’au moins un lancement utile est réellement supprimé ; sinon fermer le filtre sans nouveau portage.

Comparer les contributions pondérées **avant la somme** puis la sortie entière : Tx/Tw ci-dessus, B1/2/3/4/8, 0..10 hits par token, Hot masqué/compact, toutes partitions, strides, doublons et cohortes coupées. Couvrir ±0, petites valeurs, grandes valeurs finies, poids nuls et poison des régions non sélectionnées. Les tests doivent détecter une mauvaise frontière Hot/Cold, une mauvaise source ou un cast déplacé. Rejeter formes, origines et longueurs de poids invalides avant toute lecture GPU.

Ensuite seulement : MoE réel en attente/échec/récupération, logits/états indépendants, paires/MTP et préfixes DFlash, bridge/annulation, puis mesures release alternées. Les sorties greedy seules ne prouvent pas le contrat. Si le candidat déçoit, auditer d’abord la voie exécutée, les copies de contiguïté, les conversions de dtype, les allocations de métadonnées et l’overlap réellement conservé.

## T06 et finish au dernier groupe : après les fusions simples

Les headers safetensors locaux de la couche 0 montrent shared gate/up `I16[160,40,64]`, shared down `[40,160,64]`, contre expert routé gate `[160,40,32]` : **K4 shared, K2 routé**, bien que l’intermédiaire soit 640 dans les deux cas. Une grille commune nécessite donc deux chemins de décodage/paramétrage, comme le fait Turbo pour ses travaux denses, avec chaque `suh/svh` propre.

Le shared doit être calculé une seule fois par token, dans la branche désignée, même si Hot et Cold déclenchent chacun un `calculate_slots`. Son résultat doit être conservé jusqu’à l’addition finale. Une fusion qui attend tous les misses pour lancer un travail auparavant masqué peut perdre plus d’overlap qu’elle ne retire de lancements.

`gu3sf` ajoute publication cohérente de partiels, compteur d’arrivées et un seul écrivain final, sans boucle d’attente entre groupes. Il inclut un lancement d’initialisation des compteurs et du scratch privé par appel. Cette variante peut retirer un niveau dépendant sans diminuer le nombre total de lancements ; ni ce protocole ni son gain ne sont validés pour MLXL3. Elle n’est pas le premier candidat minimal.

Limites : aucun décompte de dispatchs locaux ni gain, aucune parité numérique nouvelle, pas de preuve Metal/MLX/concurrence. Les conclusions portent sur les frontières de code lues et les preuves historiques identifiées ; elles ne qualifient aucune option à activer.
