# Diagnostic 37 — attribution CPU des clés du cache QKV

Date : 2026-10-15. Analyse documentaire de l’essai `QUANT-2026-10-15-LFM25-37`, après ses trois étapes originales complètes et avant clôture de la reprise des six bras. Aucun modèle, poids, kernel ou réglage n’a été modifié pour cette analyse ; aucun GPU, test ou benchmark supplémentaire n’a été lancé.

## Portée et provenance

Le rapport analysé est `build/lfm25-tuning37-fourth-cloud.json`, **21 868 octets**, SHA-256 `2200bcee7cb2e4525d112d70a4d84087fa8e183a47b930d1fc2afb3f3613246b`. Champs lus : `stages.{cold-a,warm-a,cold-b}.run.cache_records_after`, `cache_after`, `runtime`, ainsi que les manifests `capture.windows`. Le parent global de ce quatrième lancement reste en échec après le segfault de `routes-a` ; les trois captures complètes et leurs anciens échecs sont conservés. Cette inspection ne transforme pas le parent en validation complète de causalité.

Les sources officielles locales sous `build/research/exllamav3-1.5.4/` sont au commit **`16a49792a3c93d8432d72e6c4bce800841566577`**, avec absence de modifications Git sur les fichiers cités. Environnement enregistré : ExLlamaV3 `1.5.4+cu128.torch2.10.0`, Torch `2.10.0+cu128`, Python 3.12.10, CUDA 12.8, driver 580.95.05, H100 80GB HBM3, 132 SM, compute capability 9.0. Le code classe Hopper comme `cc=4`, distinct de la valeur matérielle 9.0.

La configuration V2 réellement chargée a SHA-256 `d6cb556687ddd8382a90cbb957b4fbbdf95ed607bbc4800dc85fd5e5510edade` ; son descripteur EXL3 a SHA-256 `71908578d6be39abe2e2839ae2adfa087f9ea74aaaa51a6812ff5950d48ae61f`. Le petit fichier source local `build/lfm25-source-metadata/config.json` (1 442 octets, SHA-256 `b0c46662afe3881e49cf542e683bf5ea8a5e0b780a429b6141a48defc33531bf`) sert uniquement à vérifier les dimensions : hidden 2048, 32 têtes Q, 8 têtes KV, head_dim dérivé64, 40 couches dont 10 d’attention aux indices 2,6,10,14,18,22,26,30,34,38. Son SHA ne remplace pas celui de la configuration V2 chargée.

## Les cinq géométries enregistrées

`m/k/n` désignent lignes d’activation / dimension de réduction / colonnes de sortie. `K` est ici le nombre de bits EXL3. Toutes les entrées correspondent à MCG (`cb=1`), `half_k=false`, device CUDA0, classe Hopper4 et maximum132SM. Les tuples de géométrie sont `(tag, block_dim, num_sms, concurrency)` ; `num_sms` devient `gridDim.x`, donc un nombre de blocs par groupe, pas une mesure d’occupation matérielle effective.

| Clé disque uint64 hex | Projection / chemin | Paramètres de hash | Cold A et warm A | Cold B |
|---|---|---|---|---|
| `d104cf09a0c41648` | Conv.in | GEMM m2, k2048, n6144, K6, sortieF16 | (2,512,132,1) | identique |
| `9e89a622d6375b61` | Conv.out / attention.out | GEMM m2, k2048, n2048, K6, sortieF32 | (2,512,80,1) | identique |
| `80b19f9604b919d7` | Dense gate + up fusionnés | MGEMM m1, k2048, n11776, K5, sortieF16, batch1→2, non sliced | (4,256,66,2) | identique |
| `e56c31831b47bdf0` | Dense.down | GEMM m2, k11776, n2048, K5, sortieF32 | (2,512,132,1) | identique |
| **`5b575c27644bb1d9`** | **Q/K/V fusionnés et découpés** | **MGEMM m1, k2048, n512, K6, sortieF16, batch1→6, sliced** | **(2,512,20,6)** | **(2,512,16,6)** |

Le hash GEMM utilise `MAX(size_m,2)` : m2 ne signifie donc pas deux tokens réellement décodés. Le hash MGEMM garde ici m1. Le découpage QKV produit quatre tranches Q de512 colonnes et une tranche de512 pour chacun de K et V ; six groupes de sortie pour une activation d’entrée. La classe Python calcule la largeur par PGCD et construit les pointeurs/tranches. L’appel MGEMM reçoit `had_src_list`, ce qui distingue cette clé de six matrices indépendantes et de la fusion K/V seule.

Caches après exécution :

| Cache | Octets | SHA-256 |
|---|---:|---|
| Cold A / warm A, `/data/diagnostic37-fixed-tune-trace/cache-a.bin` | 176 | `1cd9c092bd749dc10fdc017a936477ed9ef0d6072d4af401711d5a7ce5bb0f65` |
| Cold B, `/data/diagnostic37-fixed-tune-trace/cache-b.bin` | 176 | `abcbacd301784a5c0a7ca39bc506cdb6b223b8124a9c70f6ec90a661d4f0416a` |

Le warm A a `bytes_unchanged=true` et aucune nouvelle entrée. Les manifests de logits cold A et warm A sont identiques pour les deux fenêtres. Cold B reproduit les SHA des logits V2 historiques23 : fenêtre1 `d4c667ec6a41a5afb336235c6fd4d1912dc36ac599fd13daee41daa13a9301a2`, fenêtre5 `71d258db615c0c8e136b820724a15b39bfe8f59fea31a699f753ec8f20e43fb6`. Ce sont des constats des manifests existants, pas une nouvelle comparaison numérique F64.

## Reproduction CPU du hash

Les opérations ci-dessous restituent le calcul entier observé dans les sources. Elles ont été exécutées via `python3` sur stdin, sans créer de fichier exécutable. Domaine : les cinq familles de formes V2 ci-dessus et les attributs matériels enregistrés. Des candidats GEMM simples puis MGEMM, avec drapeaux et tailles de batch bornés, ont été calculés ; les cinq clés du rapport ont chacune trouvé une correspondance dans ces familles. Cela n’est ni une recherche de collisions exhaustive ni une preuve symbolique générale du hash ou des kernels.

Le mélange utilise une multiplication modulo2^64, avec seed `1469598103934665603` et facteur `1099511628211`. Pour la clé QKV changée, les valeurs mélangées dans l’ordre sont :

```text
half_k=0, m_bucket=1, size_k=2048, size_n=512, K=6,
c_fp32=0, device=0, cc=4, max_num_sms=132, cb=1,
min(bszm_in,24)=1, min(bszm_out,24)=6
```

Reproduction condensée du cas QKV :

```python
factor = 1099511628211
mask = (1 << 64) - 1
h = 1469598103934665603
for value in (0, 1, 2048, 512, 6, 0, 0, 4, 132, 1, 1, 6):
    h = ((h ^ value) * factor) & mask
print(f"{h:016x}")                       # 594eede5032dbc52
h ^= 0x9e3779b97f4a7c15                   # distinction sliced
print(f"{h:016x}")                       # c779945c7c67c047
h = ((h ^ 4) * factor) & mask             # salt COOP_AUTOTUNE_VERSION=4
print(f"{h:016x}")                       # 5b575c27644bb1d9
```

## Le changement modifie réellement les arrondis intermédiaires

Tag2 correspond aux tuiles K32/N128 et block_dim512. Pour k2048/n512, chaque tranche QKV contient64 tuiles K et4 tuiles N, soit256 tuiles. Le code partitionne linéairement ces tuiles avec `gridDim.x`, par divisions entières des bornes de chaque bloc.

| gridDim.x | Intervalles K en unités de32, pour chacun des quatre groupes N128 | Sommes partielles par groupe N |
|---:|---|---:|
| 20 | [0,12), [12,25), [25,38), [38,51), [51,64) | 5 |
| 16 | [0,16), [16,32), [32,48), [48,64) | 4 |

Ces intervalles sont une seconde reproduction CPU de l’arithmétique entière du code pour cette seule forme et ces deux géométries ; aucune activation ni sortie GPU n’a été simulée.

Le kernel utilise des accumulateurs FP32 dans les blocs sur Hopper. Mais les blocs d’une même colonne échangent leur somme via C, de type F16 pour ce MGEMM : `write_sum_gl` convertit par `__floats2half2_rn`, puis `read_sum_gl` relit par `__half22float2` et ajoute aux accumulateurs. Le protocole de verrouillage impose l’ordre des sommes partielles, du bas vers le haut de K. Il ne rend pas identiques deux partitionnements différents. La géométrie20 a quatre écritures intermédiaires arrondies plus la sortie finale ; la géométrie16 en a trois plus la sortie finale. Les tailles des sous-sommes changent également.

C’est un mécanisme précis de variation des logits à poids fixes dans le bundle QKV utilisé par les dix couches d’attention. Il ne démontre pas que16SM soit plus fidèle au BF16, que20SM soit incorrect, ni que toute la perte KL du modèle quantifié provienne de ce seul mécanisme. La reprise des contrôles de routage et l’analyse numérique complète restent nécessaires ; aucune amélioration de qualité ou vitesse n’est revendiquée ici.

## Sources officielles figées et empreintes

Base des liens : dépôt `turboderp-org/exllamav3`, commit complet ci-dessus. Empreintes SHA-256 des fichiers locaux effectivement lus :

| Fichier et lignes utiles | SHA-256 |
|---|---|
| [exl3_gemm.cu — hash54–113, GEMM252, MGEMM594–598](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/exl3_gemm.cu#L54-L113) | `1754f22a4dcbd9a732121e5bb6e079bff4d7ec54005a8d8e19b610fd5ff30598` |
| [coop_autotune.cu — version28, salt68–72, lancement635–667](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/coop_autotune.cu#L635-L667) | `b685a7e18b470b86b68dfdddedc9d419b40e21462cb31c7738f44c105e86888c` |
| [exl3_kernel_map.cuh — géométries73–75](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/exl3_kernel_map.cuh#L73-L75) | `68afe01ded6198adf74dd124f0cd151fe536f918c0067ed77844e34e7688b20e` |
| [exl3_gemm_inner.cuh — partition110–119 ; lecture633–674 ; écriture724–762 ; réduction842–874](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh#L842-L874) | `3aa69924e7e32e0b442798cd0b82e08b82af4557bad37f7425bd0d47f7701364` |
| [exl3_gemm_kernel.cuh — traitement sliced147–167, appel inner234–236](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh#L147-L167) | `38d2b2911241253351ad1894cbb4cce76fe270a67bf0eeefcca75429d28b5dc7` |
| [exl3_devctx.cuh — classe Hopper25](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/exl3_devctx.cuh#L22-L26) | `7f3d88fcbac664e4a00ec0374332056d3bd24355951a78e99ec34a09a739ed93` |
| [exl3_devctx.cu — attribution CC32–45](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/exllamav3_ext/quant/exl3_devctx.cu#L32-L45) | `7115effb88a65468297015ced4f2ad50156502e859f8d207d16fb7fb413c882c` |
| [multilinear.py — largeur/tranches83–111](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/multilinear.py#L83-L111) | `8d8dd402f9a03c6cda253dd5b63378f76cf7a4af903273228d1858bc2fe583cb` |
| [attn.py — sélectionbundle498–531, appel740–783](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/attn.py#L740-L783) | `2bea03819bae9aacdc4c4a4a2315593860ef5f79aadf99e94eeafe1b81d22072` |
| [architecture/lfm2_moe.py — dimensions54–60, types118–169](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/architecture/lfm2_moe.py#L118-L169) | `7a2672ec64de9e00d13d535c636a874e8f9fc632f42b7ecc16cb6c130fac9a29` |

## Contrôle documentaire court du chemin Metal natif

Source MLXL3 locale au HEAD `6ace5dfb499c2810416bbdb4eb562dbe879baf33`, fichiers cités sans modification locale. Il s’agit du code actuel du checkout, pas d’une validation du binaire installé ni d’une comparaison de logits Metal/CUDA.

Le dispatch LFM2 choisit `Exl3Group` pour trois projections compatibles (`native/src/lfm2.rs:284–309`) et l’attention le consomme pour Q/K/V (`:335–350,379`). Les deux chemins de décodage examinés, groupé et séparé, conservent leurs sommes partielles dans des tableaux **Float32**, réduisent ces tableaux, puis convertissent le résultat global en F16 avant le Hadamard de sortie :

- `Exl3Group::forward`, `native/src/linear.rs:825–850` : allocation Float32, `partials.sum(0,false)`, puis castF16. Shader `_qmv_mapped_tile_kernel.metal:18–19,41,213–221` : partials/accumulateurs/somme écrite sont float.
- `Exl3Linear::forward`, `native/src/linear.rs:369–398` : même séquence pour une projection séparée. Shader `_qmv_tile_kernel.metal:30,130–149` : accumulateurs et sommes partielles float ; `MLXL3_FUSE_OUTPUT=0` dans ce chemin.

Ces deux chemins ne présentent donc pas l’échange interbloc par C F16 identifié dans le MGEMM CUDA ci-dessus. Cela ne prouve pas leur égalité au BF16 : entrées/transformées et sortie finale restent F16, et les ordres de calcul diffèrent. Les autres chemins (préremplissage tensoriel, experts, autres modèles) ne sont pas validés par cette brève inspection.

| Fichier natif lu | SHA-256 |
|---|---|
| `native/src/linear.rs` | `48cffac804ccaa9a8e0a66616b85c0c8afbeecead2665e168b2c4f11cf478748` |
| `native/src/lfm2.rs` | `9a500aa4d6a4197e063c8f7917f41843bf4347469ca8d6d3db43ad2e57ed58c8` |
| `native/shaders/_qmv_mapped_tile_kernel.metal` | `5a4e911376902e4e3e6f8209fce62f393a18f3a22f86afc203689b3ae69041a6` |
| `native/shaders/_qmv_tile_kernel.metal` | `594c573571ef6be4e8276299f05c7ebc264eb39f95a6edb9a3fe30c0b998d2cd` |

La vérification de cette note se limite aux lectures de sources/manifests et aux calculs entiers CPU bornés décrits. Aucun vérificateur symbolique n’a été invoqué et aucune preuve générale de correction numérique n’est revendiquée.
