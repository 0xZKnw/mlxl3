# T04 — mécanisme amont et frontière nouvelle minimale

31 octobre 2026. Revue en lecture seule préenregistrée `ISSUE59-T04-SCOPE-01`, lue en tête de `opti.md` dans le worktree `slotstream-flash/mix-stq1_0`. Aucun prototype, build, test, proveur ou GPU lancé pour cette revue. T17 et T05 restent OFF ; aucun gain nouveau mesuré. La phase d’implémentation suivante est préenregistrée séparément sous `ISSUE59-T04-COLD-01`.

**Conclusion : la frontière distincte est SwiGLU compilé → produit F16 avec down.suh, avant le H128 natif.** Elle ne consiste ni à refusionner les trois opérations internes du SwiGLU ni à reprendre les gathers/scales de POSTPROFILE05-C. Le plus petit candidat plausible serait d’étendre la fonction MLX compilée existante à la quatrième multiplication, en consommant les scales déjà rassemblées ; sa fusion et son identité numérique restent à vérifier physiquement.

## Source amont épinglée, portée réelle

TurboEXL3 : commit `345bee9e35920ee77292055d873fca891146529c`. Les documents locaux `issue59-20261029/upstream/KERNELS.md`, leur manifeste, `fusions-read.md` et la révision `update-v2/issue.md` ont été réutilisés. KERNELS : blob `875e6cb836d4c4d8bfdd8c569f66301536fbbdef`, SHA-256 `9f868e123a69eb0813403b3b99195e1098e6887aae1d29e83814b0b81ba8abb7` recalculé.

L’extrait numérique primaire absent des fichiers locaux conservés a été relu via le connecteur GitHub, sans action mutante : [fin7_4.metal](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/decode/fin7_4.metal), blob retourné `1c4a31c391004807f1ce8e5cd9b40b22efdb5254`, identique au pin de la revue précédente. `fin7` est assemblé par `decode.zig` ; ce nom ne désigne pas un fichier autonome.

Le mécanisme pertinent de `fin7` est de réunir, entre les GEMV GU et down : somme des plans, finishes gate/up, SwiGLU et préparation de l’entrée down. H128 se fait dans ses registres. Son décompte 11 lancements/8 niveaux → 4/4 au singleton ou 5/5 à plusieurs lignes décrit **Turbo**, pas un comptage de MLXL3. Le chemin shared `fin7s` emploie la table sigmoid de son hôte ; ce contrat ne doit pas être attribué au routé `fin7_4`.

Dans le fragment routé relu, `gv` et `uv` sont obtenus avec un cast half après le facteur H128 et les scales de sortie. L’activation est calculée en F32 comme `(gv / (1 + metal::exp(-gv))) * uv`. Le code arrondit `hv` en half, puis calcule `float(half(hv)) * float(down_suh)` **en F32**, avant son H128 en registres. Il n’insère pas notre arrondi F16 du produit avec `suh` avant ce H128. Le préfill Turbo utilise une exponentielle précise, tandis que cette forme decode emploie `metal::exp` ; les documents amont l’indiquent explicitement.

Ainsi, ni sa formule de sigmoid, ni ses finishes/H128, ni son produit scale F32 ne peuvent être copiés comme un remplacement bit exact de notre chaîne. Conserver le mécanisme de frontière et les points d’arrondi locaux est l’adaptation pertinente ; le port complet de `fin7` élargirait inutilement l’essai.

## Chaîne locale effectivement lue

Sources locales, SHA-256 recalculés :

| Fichier | SHA-256 |
|---|---|
| `native/src/ssd_moe.rs` | `4aa0b893e3a6c53ee6c5348a1def4d6dce040ffb00ce9c8fdb61990d56f31e0b` |
| `native/src/linear.rs` | `42106754300edc10d4b73b24a63720681626805acd0d97e27737642b973be18e` |
| `native/src/array.rs` | `e19a7c0929231b60fd944ee5907724c34208995dcbdf82eeb0a88d17bb788378` |
| `native/ffi/mlx_bridge.cpp` | `efc08fcbe039e24a5cc0dac7e9483ae612a8a09fef2e285559262c13aab4ccc8` |

`SsdMoe::calculate_slots` (`ssd_moe.rs:344–405`) produit GU `[n,2,640]`, sélectionne gate/up `[n,640]`, appelle `Array::swiglu`, puis rassemble down.suh `[n,640]` pour `linear::expert_slots`. Ici n compte les occurrences physiques d’experts, pas les tokens ; un bloc B2 n’est pas nécessairement n=2. Hot masqué et Cold compact doivent rester distingués.

Le SwiGLU (`mlx_bridge.cpp:568–572`) est déjà une fermeture `mx::compile(..., true)` contenant `multiply(multiply(gate, sigmoid(gate)), up)`. Son intérieur n’offre donc pas trois primitives indépendantes à supprimer sur le seul texte Rust. Aucun cast F32 n’y est ajouté. Le produit suivant avec `suh` est construit **après l’appel à cette fermeture**, dans `expert_mapped_layout` (`linear.rs:1423–1437`), puis le H128 MLX sur `[n,5,128]` est conservé.

La frontière de graphe est donc réelle dans les sources : sortie SwiGLU dans le dtype GU, cast vers F16, puis produit F16 avec down.suh. Le dtype GU peut être F16 ou F32, car la projection termine par `astype(x.dtype())` ; ne pas déduire F16 universellement de ses poids EXL3. Cela ne constitue pas encore une preuve du nombre de dispatchs ou de buffers matérialisés. La nouvelle fermeture peut refuser une fusion ou introduire une copie selon les strides ; seule une capture physique du chemin exact peut qualifier le travail supprimé.

Contrat minimal : garder le SwiGLU stock dans le dtype GU F16 ou F32, puis son cast F16 et son produit F16 avec les scales F16 avant le H128 MLX ; restaurer le dtype GU pour la sortie down. Ne pas réassocier gate×up×scale ou déplacer les frontières de conversion. Les vues gate/up, leurs strides et les zéros des slots Hot absents ne deviennent pas automatiquement gratuits ou interchangeables.

**Piège de raccord concret :** `input_pretransformed=true` signifie entrée déjà passée par le H128. Passer directement une simple sortie SwiGLU×suh sous ce flag supprimerait la rotation. À l’inverse, repasser le produit dans le chemin actuel avec `suh` l’appliquerait deux fois. L’adaptation doit exprimer une unique multiplication et le même H128 ; elle doit conserver les gardes slot-major, masque/groupes, splits et somme existants.

## Pourquoi POSTPROFILE05-C ne clôt pas ce minimum T04

[POSTPROFILE05](../post-sushi-20261028/README.md) a remplacé gather+produit par un produit banké indexé, aux entrées/sorties des projections GU et down, Cold/Hot. Il gardait le producteur SwiGLU séparé et le H128 natif. Après découverte de huit appels reshape ajoutés par calcul, C les a retirés puis a repassé les qualités et mesures. Le résultat corrigé reste négatif : décode agrégé MAIN −3,81%, HOLD −0,54%, sans boost préfill ; port retiré après audit et restauration vérifiée. Ces résultats ne sont pas effacés.

T04 minimal conserve les gathers de scales ; son hypothèse différente est de **supprimer la frontière producteur SwiGLU/consommateur down-input**. Il absorbe une multiplication dans le producteur compilé existant, sans shader banké supplémentaire, sans H128 en registres, sans changement de projection ou de réduction. Reprendre seulement le produit banké C, ou remplacer le SwiGLU compilé par un autre SwiGLU de même frontière, répéterait une piste déjà traitée sans propriété nouvelle.

## Différence falsifiable à préenregistrer ensuite

Sur une première cohorte définie explicitement, comparer le vrai chemin stock et le candidat mono-sortie jusqu’au produit prêt pour H128, puis la projection complète. Si le stock matérialise un SwiGLU compilé et un Multiply F16 distincts et que le candidat les remplace par un seul producteur sans copie supplémentaire, la différence de lancements est nouvelle. **Aucun nombre actuel n’est revendiqué.** Si le compilateur réalise déjà la même frontière ou si la nouvelle closure laisse le produit séparé, fermer ce filtre sans annoncer de travail supprimé.

La comparaison doit exercer les buffers GU réels stridés et les scales non identiques/non unitaires, vérifier le F16 complet après produit et après H128, ainsi que la sortie de projection avec masques/groupes/splits inchangés. Un oracle à plusieurs sorties ne suffit pas à qualifier le codegen du vrai candidat mono-sortie, comme l’a montré la limite du diagnostic T17. Les chemins non sélectionnés restent des contrôles ; ne pas introduire une attente qui retarde Hot derrière les misses Cold.

Plafond temporel récupérable : **non mesuré**. Il porte seulement sur cette frontière, pas sur les 12 ms historiques d’encodage ni sur les chiffres Turbo. Oracles numériques, qualité modèle/états et capture doivent précéder tout chrono ; une variante doit être préenregistrée avant modification, conformément à `opti.md`.
