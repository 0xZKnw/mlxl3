# T17 AUDIT-PERF-01 — frontières numériques et coût possible des `volatile`

31 octobre 2026. Audit documentaire après lecture du préenregistrement
`T17-AUDIT-PERF-01` dans [opti.md](../../../opti.md). Aucun prototype, retrait
de qualificatif, build, test, preuve ou mesure GPU exécuté pour cet audit.
Seul ce rapport est ajouté. Le shader courant reste `FIX-01`, SHA-256
`1fa5b6ae214104186021ff2aa43a5e7b083e838fb0a410cf886326aaf8f9ca44`.

**Conclusion : les deux `volatile` constituent une hypothèse de surcharge
locale plausible, pas une cause établie du pilote ni une correction déjà
validée.** Une variante conservant `precise::exp`, les intermédiaires F32 et
le mode Safe pourrait être plus légère. Elle exige un nouveau
préenregistrement et la qualification du vrai kernel mono-sortie avant toute
mesure. Le diagnostic historique à trois sorties ne suffit pas.

## Ce qui diffère réellement du chemin stock

La vraie API [apply](../../../native/src/gdn_epilogue.rs) transmet chaque
norme F32 directement, avec `z` F16, et retourne une seule sortie F16. Le
[shader](../../../native/shaders/flash_gdn_epilogue.metal) calcule :

1. `gate` F32 issu du F16 ;
2. `y = 1 / (1 + precise::exp(abs(gate)))` en F32 ;
3. `probability` F32, sélection symétrique `gate < 0 ? y : 1-y` ;
4. `product = normalized * probability` F32 ;
5. conversion du produit en F16.

Le sigmoid stock installé contient la même symétrie. Les primitives stock
écrivent leur sortie F32 en buffer device après le sigmoid, puis après le
produit, avant le cast final F16. Ces écritures sont des frontières numériques
réelles entre kernels. Le stock n'utilise pas nos deux variables locales
`volatile`. La fusion remplace leurs buffers intermédiaires par des scalaires
locaux ; elle n'introduit ici aucune nouvelle allocation Metal de taille
`rows × 6144` pour ces scalaires.

Les assertions de l'oracle [stock](../../../native/src/gdn_epilogue/tests.rs)
suivent justement `astype(F32) → sigmoid → multiply(F32) → astype(F16)` par
ligne, avec concaténation F16 pour plusieurs lignes. Une valeur finale F16
identique ne permet pas à elle seule de conclure que tous les intermédiaires
F32 ont les mêmes bits ; inversement, un petit écart F32 peut déplacer une
sortie F16 à une frontière d'arrondi.

## Safe, fonctions précises et contraction : trois points distincts

La [spécification Apple MSL](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
édition du 4 juin 2026, §1.6.3, pages 15–17, distingue les transformations
arithmétiques et le choix des fonctions FP32. Safe interdit les
transformations arithmétiques dangereuses, mais règle la contraction sur
**on**, pas **off** : contraction permise dans une instruction source, pas
entre instructions comme en mode fast. `#pragma METAL fp contract(off)`
permet de désactiver spécifiquement la contraction.

Le choix FP32 fast/precise est séparé. Le header du SDK Xcode installé indique
Fast par défaut pour `mathFloatingPointFunctions`. MLX 0.32.2 règle seulement
`MathModeSafe` sur ce macOS ; il ne règle pas cette propriété. En revanche,
la recette des kernels stock utilise `-fno-fast-math`, équivalent selon la
spécification à Safe **et** aux fonctions précises. L'appel explicite
`metal::precise::exp` du correctif reste donc nécessaire ; enlever les
`volatile` ne justifierait jamais de revenir à `metal::exp`.

Le code actuel sépare la sélection du sigmoid, le produit et le cast final
en instructions typées F32. Il ne présente pas de motif utilisateur
`normalized * probability + c` à contracter en FMA. Cette observation rend
plausible une variante sans `volatile`, sans en démontrer l'identité au
stock. La spécification §8.1/8.5 autorise le FTZ dans certains calculs ;
§8.6 distingue la conversion F16/F32, dont le rétrécissement F32→F16 utilise
l'arrondi au pair par défaut et préserve les subnormaux F16 générés.

Ces règles ne promettent pas des bits universellement identiques entre le
metallib stock et un JIT d'une autre chaîne. L'historique des options ayant
produit le **binaire installé** reste moins précisément connu que la recette
source du tag ; le vrai oracle physique reste indispensable.

## Pourquoi DIAG-02 ne qualifie pas la suppression des `volatile`

Le [log final du diagnostic](t17-final-diagnostic.log) établit seulement
qu'avec `precise_exp` sans `volatile`, les trois sorties coïncident avec le
stock sur **un couple norme/gate répété 6 144 fois**. L'exhaustif
63 488 gates × 16 normes × B1/B2/B4/B8 a passé sur le shader **avec** les deux
`volatile`, pas sur cette variante sans qualificatifs.

Il existe une seconde limite concrète : le diagnostic dérive le shader avec
**trois sorties device observables**, `p_out` F32, `product_out` F32 et
`output` F16. Cela maintient des intermédiaires F32 observables même quand
les qualificatifs sont supprimés. La vraie API n'expose que `output` F16.
Son codegen peut donc différer de celui du capteur. Le contrôle du capteur
contre `apply` ne porte que sur la variante `precise_exp` **avec** les
qualificatifs ; il ne valide pas un `apply` mono-sortie sans `volatile`.

L'ancien échec `03f1` contre `03f2` était localisé au chemin `exp` : le
correctif de fonctions précises a rétabli le cas, et supprimer `volatile`
seul ne l'a pas corrigé. Le conserver comme preuve d'un coût des `volatile`
ou comme preuve de leur inutilité serait une nouvelle conclusion sans
données. Voir aussi l'[audit numérique historique](diag-source-audit.md).

## Ce que les `volatile` peuvent coûter, et ce qui n'est pas mesuré

Les deux scalaires sont dans l'espace **thread**, privé à chaque thread
MSL (§4.3, page 120). Ce sont des qualificatifs du langage, pas le storage
mode Metal `private`, pas `MTLPurgeableState.volatile`, et pas une primitive
de synchronisation CPU/GPU. Les enlever ne supprime directement ni appel
FFI, ni requête de graphe, ni attente de routes, ni soumission CPU.

Une contrainte de lecture/écriture volatile peut empêcher certaines
optimisations et conduire à de la matérialisation locale ou à des accès
scratch. La [référence LLVM sur les accès volatiles](https://llvm.org/docs/LangRef.html#volatile-memory-accesses)
confirme qu'ils ne sont pas librement éliminables/réordonnables et que leur
réalisation dépend de la cible. **Ce n'est pas une preuve du lowering AIR
ou de l'ISA du compilateur Apple utilisé ici.**

Nous n'avons ni désassemblage qualifié du kernel, ni nombre de spills,
ni taille scratch, ni compteur de trafic privé, ni durée GPU isolée
comparant avec/sans `volatile`. On ne peut donc affirmer « deux variables =
deux spills DRAM », convertir leur présence en millisecondes, ni expliquer
la baisse de prefill DFlash HOLD par cette seule lecture source. La capture
Xcode a qualifié les dispatchs **4→1 / 10→1**, pas ces coûts internes ;
voir [inspection](xcode-dispatch-inspection.json).

Même si la variante était plus rapide au niveau kernel, son effet modèle
peut être masqué par les lectures, les autres kernels et le CPU. Aucun
plafond récupérable en ms ou tok/s n'est calculable à partir de ces sources
seules. Le budget de cache 3 Gio ne serait pas réduit ou prêté par une
suppression de deux qualificatifs locaux.

## Plus petit essai éventuel et conditions de décision

**Hypothèse à préenregistrer, pas modification réalisée :** supprimer
uniquement les deux qualificatifs `volatile`, en gardant les déclarations
`float`, les instructions séparées, la symétrie, `precise::exp`, Safe,
les pointeurs/formes, le stockage des entrées et la sortie F16 identiques.
Ne pas remplacer cette expérience par le seul capteur DIAG-02. Un pragma
local `contract(off)` serait une autre variante explicite si l'on voulait
interdire toute contraction ; il n'est pas déjà nécessairement activé par
Safe et ne doit pas être ajouté silencieusement à cette variante minimale.

La qualification devrait appeler le **vrai `apply` mono-sortie** : premier
smoke et contre-exemple d'arrondi, puis la matrice existante complète
(352 cas / 5 013 504 sorties, 42 refus, 64 layouts/alias, cas infinis et
non finis). Elle doit conserver les bits stock, finitude/cardinalités,
immutabilité et compteurs. Passer ensuite les qualités GDN réelles B1/B2/B4/B8,
les tapes/continuations, les logits/états indépendants et MTP/DFlash/bridge
affectés, sur identité nouvelle et mêmes gardes. La géométrie Rust inchangée
ne transforme pas le shader en code formellement prouvé.

En cas d'écart, auditer le contre-exemple avant rejet ; en cas de parité,
mesurer seulement alors le débit et le prefill avec chauffe, identité
figée, alternance et isolation prévues. Un résultat positif sur le cas
uniforme ou une baisse de dispatchs ne justifie pas une promotion. Cet audit
ne préenregistre et ne lance lui-même aucun de ces essais.

## Sources effectivement consultées et identité

| Source primaire | Référence |
| --- | --- |
| Spécification MSL Apple officielle | PDF lu en mémoire, 14 564 497 octets, 383 pages, édition 2026-06-04 ; SHA-256 `41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5`. Pages 15–17, 120, 368, 375–376. Aucun fichier PDF ajouté au dépôt. |
| Header SDK Xcode 27.2 `MTLLibrary.h`, lignes 256–338 | `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers/MTLLibrary.h` ; SHA-256 `476081df0bf347bb66397578427692ef9ee94551eb070b7f6358823731937323`. MathMode et fonctions FP32 séparés, défaut Fast des fonctions, arrondi de conversion au pair. |
| MLX `device.cpp` archivé, lignes 37–79 et 650–668 | `/Users/justin/Documents/mix-stq1_0/build/flash-strata57/device.cpp` ; SHA-256 `008b67e599f5508ab670a91f2ea80372162a8767b5226fbbeb9f3d23c85cdb7c`, blob Git `2f25f894e494a4328886148bb6e3332d9f9b298a`. Safe réglé seul, langage runtime 4.1 sur macOS 27. |
| MLX installé `backend/common/metal_kernel.h` | Sous `/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx/include/mlx/` ; SHA-256 `b64ca4cebcba530d8ec1e0a0f23b57313507a37d05e3615ded3796b563cc83fc`. `CompileOptions{}` choisit Safe. |
| MLX installé `kernels/unary_ops.h`, lignes 308–313 ; `unary.h`, lignes 1–18 ; `binary_ops.h`, lignes 208–212 | Même racine installée ; `unary_ops.h` SHA-256 `0a5492b65ae39ecb6d8b04e64ea5007e0a4ff60d8d9428559bfbfd03387ece2a`, blob `1337f6b52edb510f4fb199dee5e443e48e02b675`. Formule Sigmoid et écritures typées, produit simple. |
| [MLX CMake stock v0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/CMakeLists.txt), lignes 12–38 | Relu via connecteur GitHub ; blob `ecabbda550780ae2e18ee8e002914f03d12bca4d`, compilation `-fno-fast-math`. |
| [MLX compiled.cpp v0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/compiled.cpp), lignes 205–220 | Relu via connecteur GitHub ; blob `e95d7b8d5f1ea879bb12a067bfe04cde7739fd9b`. Le JIT de graphes émet aussi des temporaires typés, sans `volatile` ; contexte distinct de notre custom kernel, sans garantie de parité du nouveau mono-sortie. |
| FFI locale `native/ffi/mlx_bridge.cpp`, lignes 681–703 | SHA-256 `efc08fcbe039e24a5cc0dac7e9483ae612a8a09fef2e285559262c13aab4ccc8` ; appel custom kernel avec `mx::CompileOptions{}`. Aucun réglage numérique alternatif ajouté par cet audit. |

Les consultations, hashes et extractions PDF ont été des opérations de
lecture CPU ; aucune chaîne de compilation ni app Xcode n'a été lancée.
