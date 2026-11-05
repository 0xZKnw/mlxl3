# 79R-C — revue statique des kernels et de la frontière MLX/Metal

23 octobre 2026. **Aucun gain mesuré et aucun code modifié.** Checkout documentaire `88cbfd22`, runtime `89174fcc`. La recherche 79R est préenregistrée dans `opti.md`. Les cinq constats ci-dessous décrivent des redondances ou opportunités à vérifier ; ils ne valident ni une parité numérique ni une accélération. Aucun modèle GPU, benchmark, build, test ou preuveur n’a été exécuté pour cette revue. Les nombres historiques conservent leur statut initial.

## Priorité : supprimer du travail cible

Le témoin H72W-R récent, relu par root, passe 10,494 s/11,238 s MAIN et 10,984 s/11,851 s HOLD dans la vérification, contre 0,516 s et 0,529 s dans le draft. Le draft représente 4,59 %/4,46 % du décodage : le rendre gratuit aurait un plafond idéal d’environ 4,8 %/4,7 % de débit, avant tout effet d’acceptation. Les pistes ring/selector restent utiles mais ne peuvent pas fournir seules le saut vers 15 tok/s.

Le profil 64 localisait le MoE hôte inclusif à 72,3–75,2 ms/sortie et la préparation GDN groupée à 0,74–0,81 ms. Ces frontières peuvent attendre du GPU amont et se recouvrent ; ce ne sont pas des temps GPU exclusifs. Le passage instrumenté avec barrières ralentissait fortement l’exécution : ses durées ne doivent pas être additionnées comme une décomposition du parcours naturel.

Les cibles immédiates les plus vérifiables sont **la projection d’index QSA partiellement inutile** et **les métadonnées Rust reconstruites avant le cache C++**. Les petites projections denses GDN encore sérialisées ont un correctif limité et déjà disponible, mais leur coût exclusif doit être confirmé avant une longue campagne.

## 79C-01 — Calculer seulement la tête clé de l’index QSA quand les quatre requêtes ne sont pas utilisées

**Travail cible redondant démontré statiquement ; optimisation non mesurée.**

Sources : `native/src/qwen4.rs:591,606,616,649,659,664,1764,1771`, `native/src/linear.rs:347,357,389,409,441`, `native/src/flash_support.rs:284`.

**Chemin réel.** FlashNext/verify_draft → Layer::prepare_rows → Qsa::forward → append_cache ; MtpHead::extend_cache → append_cache.

**Cas concret.** append_cache produit cinq têtes de 128. Seule la dernière est nécessaire au cache. Les quatre premières ne sont lues par Qsa::forward que lorsque (ancien_offset+1)/4>512, soit à partir du 2052e token. extend_cache abandonne toujours le résultat qk et ne nécessite jamais ces quatre requêtes.

**Formes.** Entrée F16[1,2560], sortie 640. Checkpoint réel index layer 3 : trellis I16[160,40,32], suh F16[2560], svh F16[640], K2/Mul1 ; les 128 sorties utiles occupent les tuiles 32..40 et un bloc Hadamard entier.

**Plus petit changement.** Ajouter un chemin EXL3 de plage de sorties 512..640, avec stride physique 40 inchangé et décalage de tuile 32. Conserver toute la transformation d’entrée 2560 et les splits calculés pour la projection 640 d’origine. Employer la projection complète quand Qsa::forward utilise effectivement ses requêtes ; employer la plage clé seule dans extend_cache. Repli dense/full projection+slice.

**Travail supprimé.** 32 des 40 tuiles de sortie de cette seule petite projection ; ni 80 % du QSA entier, ni 80 % du modèle. Applicable aux 12 couches QSA et au cache de tête MTP.

**Différence avec les essais précédents.** La sélection sparse évitée aux contextes courts était connue ; elle ne supprime pas la projection d’index 640 actuellement exécutée. Différent du regroupement QSA général de 63/64R.

**Risques.** Ne pas remplacer le stride 40 par 8. Ne pas recalculer les splits d’après la largeur réduite ; le cas réel a split 1. Préserver SG4/NT1, partition de sommes FP32, cast F16, Hadamard 128 et svh[512..640]. Conserver toutes les clés historiques pour franchir ultérieurement le seuil 2052. Distinguer besoin de requêtes du forward et absence de besoin dans extend_cache.

**Oracle requis.** Comparaison bit à bit full 640.slice(512..640) contre range sur les poids réels, entrées finies F16/F32/strides et B1/B2/B4/B8 ; logits et états index_tail/blocks/KV à 2051/2052/2053, tous préfixes et rollback/cancel/reprise, cache-only MTP.

**Essai discriminant à préenregistrer.** Préenregistrer d’abord la plage exacte et ses oracles. Compter les tuiles réellement lancées et profiler la seule projection, puis OFF/MTP1/DFlash B2 MAIN/HOLD avec même cache 3 Gio et contrôle mémoire. Aucun débit prévu.

## 79C-02 — Réutiliser le chemin dense exact multi-lignes pour les deux portes GDN de 48 sorties

**Appels de projection ligne par ligne confirmés ; adaptation numérique à vérifier.**

Sources : `native/src/qwen4.rs:166,349,352`, `native/src/lfm2.rs:249,258,273,304`, `native/src/qwen35.rs:311,3451`.

**Chemin réel.** DFlash verify_draft / verify_pair multi-token → Gdn::forward_rows → verification_projection(a/b).

**Cas concret.** verification_projection ne groupe que les projections EXL3. a et b du checkpoint sont denses F16[48,2560], donc chaque fenêtre B exécute B projections puis concatenate pour chacune des deux portes, sur 36 couches GDN.

**Formes.** Entrée F16[B,2560], poids F16[48,2560], sortie F16[B,48], B2..8. Ces formes satisfont les critères dimensionnels du helper forward_dense_rows_exact déjà utilisé par Qwen3.5, mais jamais par Qwen4.

**Plus petit changement.** Branchement étroit a/b denses sans biais vers le helper existant pour 2..8 lignes, après validation dtypes et oracle réel. Garder B1 et formes non admissibles sur le chemin actuel. Pas de GEMM arbitraire, pas de réécriture du scan.

**Travail supprimé.** Réduit les appels de construction des portes de 2B à 2 par couche : 72(B−1) appels de projection potentiellement évités par fenêtre sur 36 couches, plus les concatenations. Le helper garde une grille Y par ligne : ce constat ne prouve pas un partage des chargements de poids.

**Différence avec les essais précédents.** 63 groupe les projections EXL3 ; le helper dense exact existant reste inutilisé ici. Différent de HC MB2(22), de la fusion GDN mono-ligne 25 et du shared EXL3 (68).

**Risques.** Le nom exact du helper ne prouve pas l’égalité avec le GEMV MLX pour 48 sorties. Préserver l’ordre FP32 et le cast F16 ; vérifier les strides/dtypes plutôt que se fier seulement aux dimensions. Ne pas confondre le nombre d’appels Rust avec le nombre final de soumissions Metal.

**Oracle requis.** Poids a/b réels 48×2560, B2..8 et strides, oracle B×T1, finitude et nombre de valeurs, états GDN complets et chaque préfixe ; mutation d’indice de ligne détectée. Bridge DFlash et MTP séparément.

**Essai discriminant à préenregistrer.** Compter appels/nœuds/dispatchs puis coût exclusif portes avant une campagne globale. Si ce coût est faible, arrêter après mesure composant ; le profil 64 ne justifie pas une campagne entière sur GDN.

## 79C-03 — Éviter de reconstruire les métadonnées immuables avant chaque cache hit de kernel

**Allocations et copies hôte confirmées ; coût exposé non mesuré.**

Sources : `native/src/linear.rs:357,517,1401,1536,1540,1546`, `native/src/array.rs:1366,1395,1403,1412`, `native/ffi/mlx_bridge.cpp:646,660,661,669`.

**Chemin réel.** Projections EXL3 denses et experts → reference_reduction_header/codebook_header → array::metal_kernel → FFI C++ → cache.find.

**Cas concret.** Chaque appel référence construit 256 String par to_string, joint les positions dans une nouvelle chaîne, reconstruit les headers, et convertit la source MSL complète ainsi que les noms en CString avant le cache.find C++. Le cache de factories ne peut pas éviter ce travail déjà effectué côté Rust.

**Formes.** Toutes les fenêtresB1/B2/B4/B8 ; shaders tile/mapped d’environ 10/14 Ko copiés en CString à l’appel. Aucune donnée tenseur n’est nécessaire pour mémoriser ces seules chaînes.

**Plus petit changement.** Commencer par un OnceLock du préfixe de réduction de référence immuable ; conserver le suffixe propre à chaque géométrie. Ensuite seulement, si le profil l’exige, des descripteurs CString immuables par signature complète. Ne mémoriser ni pointeur d’array, ni slot expert, ni budget.

**Travail supprimé.** Construction de 256 petites String par appel de référence, jointure puis allocations/copies de métadonnées ; aucun calcul GPU supprimé.

**Différence avec les essais précédents.** 19 mesure lookup/factory C++ (~0,4 % historique) et 67 stabilise les signatures. Ce constat vise le travail Rust situé avant ces intervalles, pas une nouvelle politique de cache factory.

**Risques.** Conserver la clé complète nom+noms I/O+header+source ; ne pas créer de collision de variantes. Conserver erreurs de NUL, validations de tailles et durées de vie FFI. Le compilateur/allocateur peut déjà rendre ce coût négligeable : pas de gain présumé.

**Oracle requis.** Identité octet par octet des descripteurs, tests des variantes et collisions, mêmes graphes/sorties/logits ; aucun changement numérique attendu.

**Essai discriminant à préenregistrer.** Profiler les spans CPU formatage et CString avant FFI, sans eval ajouté. Comparer coût au token et temps effectivement exposé. Ne pas construire un graphe persistant général avant preuve que ce coût compte.

## 79C-04 — Conserver une vue FP32 dérivée des blocs d’index QSA aux contextes longs

**Conversion répétée confirmée ; précision d’un reliquat long contexte, pas un boost des prompts courts.**

Sources : `native/src/qwen4.rs:616,622,667,677,704`.

**Chemin réel.** Qsa::forward, uniquement lorsque (offset+1)/4>512.

**Cas concret.** Chaque token convertit l’ensemble des blocs d’index F16 en FP32 pour le produit des scores. Un nouveau bloc 128 est ajouté tous les 4 tokens ; les anciens blocs immuables sont reconvertis dans une nouvelle expression.

**Formes.** Blocs F16[1,1,count,128]. À count 513..1024, une vue FP32 matérialisée représente environ 0,25..0,5 Mio par QSA, soit environ 3..6 Mio pour 12 couches.

**Plus petit changement.** Cache dérivé FP32 mis à jour seulement après ajout d’un nouveau bloc, en convertissant APRÈS l’arrondi F16 canonique. Garder l’état F16 de référence ; reconstruire/invalider la vue sur restore/reset/cancel.

**Travail supprimé.** Conversions répétées des blocs existants, pas les copies de concaténation ni le calcul des scores ou les readbacks de sélection.

**Différence avec les essais précédents.** L’optimisation QSA long contexte est un reliquat déjà connu ; la conversion complète bloc→FP32 est une frontière concrète à isoler plutôt qu’un nouveau port QSA générique.

**Risques.** Un cache dérivé incorrect au rollback contamine les scores. Ajouter quelques Mio persistants et davantage de snapshots peut annuler le bénéfice. Confirmer une matérialisation GPU réelle plutôt qu’attribuer seulement un nœud astype.

**Oracle requis.** Blocs F16 inchangés, vue FP32 égale à astype intégral, logits et états à 2051/2052/2053 et 4096, restaurations de tous préfixes et annulation/reprise.

**Essai discriminant à préenregistrer.** Mesurer uniquement sur contextes >2051 avec même KV et trace d’allocations/casts. Ne pas lancer MAIN/HOLD courts pour attendre un gain sur un chemin inactif.

## 79C-05 — Éviter les groupes vides et les lignes de requêtes rembourrées du ring draft B2

**Géométrie surdimensionnée confirmée ; priorité secondaire car draft ≈ 4,5 % du décodage.**

Sources : `native/src/flash_draft.rs:600,634,896,925,946,959`, `native/shaders/flash_draft_ring.h:18,22,33,37,42,197`.

**Chemin réel.** Draft DFlash2 Q4, cinq couches, attention circulaire → ring_attention. Pas le vérificateur cible.

**Cas concret.** B2 est rembourré à 8 positions ; 4 têtes Q par tête KV donnent M32 alors que 8 lignes Q suffisent. La grille garde 8 splits par tête KV. À contexte ancien ≤ 128, seuls split 0 et split 7 sont utiles, les 6 autres ne font qu’écrire des maxima −inf puis retournent.

**Formes.** Q BF16[1,32,B,128], K/V cache[1,8,2048,128]/[1,8,128,2048]. Partials FP32[8,8,4160], soit 1 064 960 octets logiques, puis sortie BF16[1,32,8,128] découpée à B.

**Plus petit changement.** Premier essai limité : liste CPU des splits actifs déduite de position connue, conserver leurs indices logiques et l’ordre de réduction à 8 splits. Réduction doit ignorer les slots jamais écrits AVANT lecture. Garder d’abord le layout partials. Un M compact 4B est un second mécanisme distinct, à qualifier séparément.

**Travail supprimé.** Groupes vides et éventuellement calculs de lignes Q rembourrées ; pas de réduction prouvée de durée. Les groupes vides ont déjà un early-return.

**Différence avec les essais précédents.** Différent d’un nouvel A/B ring ON/OFF 60/61 inchangé : géométrie active et indexation proposées changent réellement le travail. Ne pas simplement réduire le nombre logique de splits, ce qui changerait l’ordre des sommes.

**Risques.** Lire des partials non initialisés si la réduction ne saute pas les slots absents. Modifier le nombre de splits ou la géométrie MPP peut changer l’arrondiFP32/softmax et les propositions. Les 1 Mio sont une taille logique temporaire, pas un pic physique ajouté cinq fois. Gain idéal de tout le draft gratuit seulement ≈ 4,7–4,8 % sur les témoins B2 récents.

**Oracle requis.** Ancien ring contre nouveau bit à bit B1/B2/B4/B8, positions 0/1/127/128/255/1024/2047/2048/2049/4096 ; cache wrap, partials empoisonnés et mutant omettant split 7, oracle attention BF16 indépendant, commit et tous préfixes.

**Essai discriminant à préenregistrer.** Mesurer coût du ring et dispatchs actifs avant TPS global. Priorité après les chantiers cible : le témoin H72W-R attribue seulement 4,46–4,59 % au draft.

## Annexe exploratoire — une LUT Mul1 exacte, sans changement de quantification

`linear.rs:1546–1555` : Mul1 calcule un hash 32 bits, additionne ses quatre octets, puis applique une expression half à `0x6400+S`, avec `S∈[0,1020]`. Une table de 1021 half, environ 2042 octets, pourrait remplacer seulement le réinterprétage et le half-muladd final. Le hash et la somme SWAR resteraient nécessaires. Ce n’est ni une table int8 approchée ni la LUT 8 entrées K3 rejetée dans PERF89, qui confondait les bits de transition avec l’état 16 bits.

La génération doit préserver exactement l’expression Metal et ses arrondis. Une table CPU produite avec une contraction FMA différente n’est pas un oracle. Un futur composant devrait comparer les 65 536 états 16 bits, puis mesurer si la lookup ne coûte pas davantage que l’arithmétique supprimée. Les essais 54 n’ont pas établi de gain global des variantes de décodage. **Piste basse priorité, aucun shader créé, aucun gain attendu chiffré.**

## Éléments écartés ou déjà connus

- Le cache PLE conserve encore les mots compressés et redécode les 160 valeurs même sur hit (`qwen4.rs:800–852`). Un cache de valeurs F16 ajouterait environ 3,72 Mio au maximum 16 384 lignes, hors métadonnées : 82→320 octets/ligne. Mais le cache déquantifié est déjà proposé dans l’audit 45/70R ; ce n’est pas une découverte 79 et PLE n’occupe qu’une seule couche. Aucun taux de hit ni coût CPU exclusif nouveau n’a été mesuré.
- Les constantes de convolution GDN appartiennent déjà au prototype 78 non appliqué. Les fusions pré/post GDN multi-token et HC/PLE/QSA groupés sont des reliquats 64R/72R. Les répétitions de portes/exp uniformes dans le shader GDN pourraient être éliminées par le compilateur : sans inspection d’instructions, ce n’est pas du travail matériel redondant démontré.
- Les FHT d’entrée de shared gate/up, GDN qkv/z et QSA q/k/v ne sont pas mutualisables naïvement : leurs vecteurs SU sont différents dans le checkpoint réel. Les hashes se trouvent dans `input-scale-sha.json`. Une fusion conservant les transformations séparées serait un mécanisme différent.
- HC écriture 18/24, HC MB2(22), GDN mono-ligne 25, overlap shared 27, variantes 54, grouped EXL3 (65/66), signatures 67 et shared multirow 68 n’autorisent aucune répétition inchangée. Leurs résultats négatifs/non qualifiés restent valables. Le helper dense GDN proposé ici change une frontière précise non exploitée par 63.
- Le cache des factories C++ est déjà efficace d’après 19 ; aucune raison de réinventer un cache général. La piste 79C-03 se situe avant le cache lookup et doit d’abord démontrer son coût. Les prototypes 72/73 retirés ne sont pas réactivés ; 71P/78 ne sont pas appliqués.

## Ce que les nouvelles sources amont apportent réellement

Les identités primaires exactes sont conservées par root dans `upstream.json`. Splash main `3135ab45b3da525a8a2dbb8a1aa72ce095dda8a9` est inchangé depuis 72. Strata main `7ddc17a38fe7f31637e895ecfba58c2cd43d65ec` ajoute depuis 5c8dd322 seulement du packaging et des tests. Leurs architectures de fenêtres/dispatchs et leurs plans actifs restent des références de conception, pas un boost livré.

[PR569 Splash](https://github.com/incoai/splash/pull/569), tête `2c12b35eb036fd3462dcbe1216d12985d238ed28`, optimise le zero-point GGUF sur Apple9/M4 et précise qu’Apple10 n’est pas modifié. Cela ne fournit donc pas un gain direct M5/EXL3.

[PR3299 Strata](https://github.com/Niko1221/Strata/pull/3299), tête `dd0c130fe9a4370f148f30b25cd546434c5182c5`, porte des kernels Gyrotrellis TQ2_T/TQK6/TQK7. La discussion décrit leur hash/lookup comme ALU-bound et envisage une LUT int8. Ces formats et leurs critères numériques diffèrent de notre EXL3 ; leurs mesures ne prouvent pas que notre décodeur est ALU-bound. Seul le principe d’une table exacte sur un domaine réellement réduit est repris dans l’annexe.

## Portée et clôture

Les formes checkpoint ont été lues dans les en-têtes safetensors sur CPU ; aucune charge de tenseurs GPU ni nouvelle exécution numérique. Les fichiers source concernés et leurs SHA-256 sont dans `kernels.json`. Les appelants ont été retracés depuis le vérificateur, la couche et le bridge FFI jusqu’aux shaders ; les formes de la seule source MSL n’ont pas été prises comme preuve du parcours réel.

Toute implémentation future exige son propre préenregistrement, les contrôles de source appropriés, l’oracle bit à bit des logits/états et le bridge réel avant mesure de vitesse. Les limites mémoire restent inchangées ; un compteur ledger nul ne démontre pas l’absence de swap-out du modèle. Cette revue statique est **close**, les cinq optimisations sont **non mesurées et non implémentées**, et 15 tok/s n’est pas promis.
