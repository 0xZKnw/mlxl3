# Inventaire Qwen27B small-M — M5, 6–7 octobre 2026

Source : `inventory-timings.json`. 16 formes/bundles originaux × M=2/3/4/6/8 ; les partielles FP32 sont finies et identiques à la référence M1/NT1 ligne par ligne avant chronométrage. Trellis I16, codebook MUL1/CB2, scales et transforms FP16 originaux. Les counts désignent le nombre de projections/bundles de cette forme dans le checkpoint.

Chaque temps inclut soumission hôte, MLX et synchronisation GPU : ce n’est pas un temps GPU pur. Trois warmups puis vingt répétitions par série, ordre alterné. Les deux séries EXL3 exécutent **le même kernel de production** ; leurs différences ne sont pas des gains. La chaîne applique huit projections avec scales/transforms réels et feedback synthétique borné, puis divise le temps par huit. Elle ne mesure pas le coût attribué à une couche du modèle.

L’ancienne erreur d’inventaire SG4 pour les projections séparées K2 est conservée dans `inventory-initial-sg-diagnostic.json`. Le tableau ci-dessous et les timings utilisent SG8 pour ces projections ; seuls les groupes K2 utilisent SG4.

| Projection représentative | Occurrences | Input → outputs | Chemin | K | CB | M | MB | NT | SG | Splits | Isolé séries 1 / 2 (ms) | Chaîne séries 1 / 2 (ms/projection) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| layers.0.linear_attn.in_proj_qkv + layers.0.linear_attn.in_proj_z | 47 | 5120 → 10240 + 6144 | grouped | 2 | 2 | 2 | 2 | 2 | 4 | 1 | 0.847854 / 0.828458 | 0.578831 / 0.566742 |
| layers.0.linear_attn.in_proj_qkv + layers.0.linear_attn.in_proj_z | 47 | 5120 → 10240 + 6144 | grouped | 2 | 2 | 3 | 1 | 2 | 4 | 1 | 1.288104 / 1.346312 | 1.343281 / 1.269617 |
| layers.0.linear_attn.in_proj_qkv + layers.0.linear_attn.in_proj_z | 47 | 5120 → 10240 + 6144 | grouped | 2 | 2 | 4 | 2 | 2 | 4 | 1 | 1.178145 / 1.212021 | 1.172266 / 1.166151 |
| layers.0.linear_attn.in_proj_qkv + layers.0.linear_attn.in_proj_z | 47 | 5120 → 10240 + 6144 | grouped | 2 | 2 | 6 | 2 | 2 | 4 | 1 | 1.686292 / 1.641625 | 1.731466 / 1.742922 |
| layers.0.linear_attn.in_proj_qkv + layers.0.linear_attn.in_proj_z | 47 | 5120 → 10240 + 6144 | grouped | 2 | 2 | 8 | 2 | 2 | 4 | 1 | 2.141895 / 2.101458 | 2.195904 / 2.195221 |
| layers.0.mlp.gate_proj + layers.0.mlp.up_proj | 5 | 5120 → 17408 + 17408 | grouped | 1 | 2 | 2 | 2 | 2 | 8 | 1 | 1.259937 / 1.289041 | 1.261510 / 1.306187 |
| layers.0.mlp.gate_proj + layers.0.mlp.up_proj | 5 | 5120 → 17408 + 17408 | grouped | 1 | 2 | 3 | 1 | 2 | 8 | 1 | 2.515271 / 2.581208 | 2.611255 / 2.650620 |
| layers.0.mlp.gate_proj + layers.0.mlp.up_proj | 5 | 5120 → 17408 + 17408 | grouped | 1 | 2 | 4 | 2 | 2 | 8 | 1 | 2.295271 / 2.356562 | 2.380729 / 2.379536 |
| layers.0.mlp.gate_proj + layers.0.mlp.up_proj | 5 | 5120 → 17408 + 17408 | grouped | 1 | 2 | 6 | 2 | 2 | 8 | 1 | 3.644521 / 3.356938 | 3.632432 / 3.641029 |
| layers.0.mlp.gate_proj + layers.0.mlp.up_proj | 5 | 5120 → 17408 + 17408 | grouped | 1 | 2 | 8 | 2 | 2 | 8 | 1 | 4.698542 / 4.404792 | 4.663099 / 4.733148 |
| layers.10.mlp.gate_proj + layers.10.mlp.up_proj | 58 | 5120 → 17408 + 17408 | grouped | 2 | 2 | 2 | 2 | 2 | 4 | 1 | 1.226438 / 1.259729 | 1.247732 / 1.213078 |
| layers.10.mlp.gate_proj + layers.10.mlp.up_proj | 58 | 5120 → 17408 + 17408 | grouped | 2 | 2 | 3 | 1 | 2 | 4 | 1 | 2.514876 / 2.552042 | 2.566099 / 2.616906 |
| layers.10.mlp.gate_proj + layers.10.mlp.up_proj | 58 | 5120 → 17408 + 17408 | grouped | 2 | 2 | 4 | 2 | 2 | 4 | 1 | 2.294729 / 2.351917 | 2.340370 / 2.327094 |
| layers.10.mlp.gate_proj + layers.10.mlp.up_proj | 58 | 5120 → 17408 + 17408 | grouped | 2 | 2 | 6 | 2 | 2 | 4 | 1 | 3.493229 / 3.334521 | 3.522800 / 3.478935 |
| layers.10.mlp.gate_proj + layers.10.mlp.up_proj | 58 | 5120 → 17408 + 17408 | grouped | 2 | 2 | 8 | 2 | 2 | 4 | 1 | 4.644146 / 4.350917 | 4.602768 / 4.599328 |
| layers.63.mlp.gate_proj + layers.63.mlp.up_proj | 1 | 5120 → 17408 + 17408 | grouped | 3 | 2 | 2 | 2 | 2 | 8 | 1 | 1.305541 / 1.297083 | 1.337591 / 1.372807 |
| layers.63.mlp.gate_proj + layers.63.mlp.up_proj | 1 | 5120 → 17408 + 17408 | grouped | 3 | 2 | 3 | 1 | 2 | 8 | 1 | 2.594062 / 2.718833 | 2.791802 / 2.742128 |
| layers.63.mlp.gate_proj + layers.63.mlp.up_proj | 1 | 5120 → 17408 + 17408 | grouped | 3 | 2 | 4 | 2 | 2 | 8 | 1 | 2.417604 / 2.469167 | 2.528396 / 2.532693 |
| layers.63.mlp.gate_proj + layers.63.mlp.up_proj | 1 | 5120 → 17408 + 17408 | grouped | 3 | 2 | 6 | 2 | 2 | 8 | 1 | 3.575125 / 3.717875 | 3.945008 / 3.921044 |
| layers.63.mlp.gate_proj + layers.63.mlp.up_proj | 1 | 5120 → 17408 + 17408 | grouped | 3 | 2 | 8 | 2 | 2 | 8 | 1 | 4.840146 / 4.996583 | 5.173300 / 5.167792 |
| lm_head | 1 | 5120 → 248320 | separate | 3 | 2 | 2 | 2 | 2 | 8 | 1 | 9.425000 / 8.709104 | 9.586943 / 9.655435 |
| lm_head | 1 | 5120 → 248320 | separate | 3 | 2 | 3 | 3 | 1 | 8 | 1 | 16.442354 / 16.550937 | 16.061570 / 16.047307 |
| lm_head | 1 | 5120 → 248320 | separate | 3 | 2 | 4 | 2 | 2 | 8 | 1 | 18.666833 / 18.669459 | 19.276109 / 19.029349 |
| lm_head | 1 | 5120 → 248320 | separate | 3 | 2 | 6 | 2 | 2 | 8 | 1 | 28.481563 / 28.697458 | 29.213859 / 29.160810 |
| lm_head | 1 | 5120 → 248320 | separate | 3 | 2 | 8 | 2 | 2 | 8 | 1 | 38.768312 / 39.124167 | 39.253234 / 39.010315 |
| layers.0.linear_attn.out_proj | 49 | 6144 → 5120 | separate | 2 | 2 | 2 | 2 | 2 | 8 | 1 | 0.495708 / 0.512813 | 0.250417 / 0.248771 |
| layers.0.linear_attn.out_proj | 49 | 6144 → 5120 | separate | 2 | 2 | 3 | 1 | 4 | 8 | 1 | 0.598251 / 0.597333 | 0.552435 / 0.449057 |
| layers.0.linear_attn.out_proj | 49 | 6144 → 5120 | separate | 2 | 2 | 4 | 2 | 2 | 8 | 1 | 0.580229 / 0.582917 | 0.439104 / 0.594188 |
| layers.0.linear_attn.out_proj | 49 | 6144 → 5120 | separate | 2 | 2 | 6 | 2 | 2 | 8 | 1 | 0.750916 / 0.770646 | 0.654615 / 0.660565 |
| layers.0.linear_attn.out_proj | 49 | 6144 → 5120 | separate | 2 | 2 | 8 | 2 | 2 | 8 | 1 | 0.943666 / 0.963271 | 1.017396 / 1.013797 |
| layers.0.mlp.down_proj | 1 | 17408 → 5120 | separate | 1 | 2 | 2 | 2 | 2 | 8 | 4 | 0.770083 / 0.794438 | 0.634974 / 0.630992 |
| layers.0.mlp.down_proj | 1 | 17408 → 5120 | separate | 1 | 2 | 3 | 1 | 4 | 8 | 4 | 1.342521 / 1.312730 | 1.428615 / 1.418000 |
| layers.0.mlp.down_proj | 1 | 17408 → 5120 | separate | 1 | 2 | 4 | 2 | 2 | 8 | 4 | 1.253958 / 1.268417 | 1.439495 / 1.413477 |
| layers.0.mlp.down_proj | 1 | 17408 → 5120 | separate | 1 | 2 | 6 | 2 | 2 | 8 | 4 | 1.977958 / 1.821208 | 2.021094 / 2.020925 |
| layers.0.mlp.down_proj | 1 | 17408 → 5120 | separate | 1 | 2 | 8 | 2 | 2 | 8 | 4 | 2.372500 / 2.416062 | 2.605435 / 2.586630 |
| layers.1.mlp.down_proj | 60 | 17408 → 5120 | separate | 2 | 2 | 2 | 2 | 2 | 8 | 4 | 0.765083 / 0.728459 | 0.661760 / 0.647385 |
| layers.1.mlp.down_proj | 60 | 17408 → 5120 | separate | 2 | 2 | 3 | 1 | 4 | 8 | 4 | 1.339708 / 1.318583 | 1.415716 / 1.393536 |
| layers.1.mlp.down_proj | 60 | 17408 → 5120 | separate | 2 | 2 | 4 | 2 | 2 | 8 | 4 | 1.311396 / 1.373438 | 1.417630 / 1.446771 |
| layers.1.mlp.down_proj | 60 | 17408 → 5120 | separate | 2 | 2 | 6 | 2 | 2 | 8 | 4 | 1.861458 / 1.855438 | 2.050687 / 2.086685 |
| layers.1.mlp.down_proj | 60 | 17408 → 5120 | separate | 2 | 2 | 8 | 2 | 2 | 8 | 4 | 2.500895 / 2.463001 | 2.654750 / 2.635922 |
| layers.11.self_attn.k_proj | 18 | 5120 → 1024 | separate | 3 | 2 | 2 | 2 | 2 | 8 | 8 | 0.356563 / 0.281396 | 0.089693 / 0.090029 |
| layers.11.self_attn.k_proj | 18 | 5120 → 1024 | separate | 3 | 2 | 3 | 1 | 4 | 8 | 8 | 0.300625 / 0.302771 | 0.127286 / 0.127232 |
| layers.11.self_attn.k_proj | 18 | 5120 → 1024 | separate | 3 | 2 | 4 | 2 | 2 | 8 | 8 | 0.288270 / 0.287875 | 0.134161 / 0.134323 |
| layers.11.self_attn.k_proj | 18 | 5120 → 1024 | separate | 3 | 2 | 6 | 2 | 2 | 8 | 8 | 0.329666 / 0.334271 | 0.169396 / 0.168172 |
| layers.11.self_attn.k_proj | 18 | 5120 → 1024 | separate | 3 | 2 | 8 | 2 | 2 | 8 | 8 | 0.380521 / 0.382563 | 0.204495 / 0.207388 |
| layers.11.self_attn.q_proj | 2 | 5120 → 12288 | separate | 1 | 2 | 2 | 2 | 2 | 8 | 1 | 0.805334 / 0.844479 | 0.601755 / 0.430964 |
| layers.11.self_attn.q_proj | 2 | 5120 → 12288 | separate | 1 | 2 | 3 | 1 | 4 | 8 | 1 | 0.988250 / 0.963125 | 0.934122 / 1.016635 |
| layers.11.self_attn.q_proj | 2 | 5120 → 12288 | separate | 1 | 2 | 4 | 2 | 2 | 8 | 1 | 0.955125 / 0.970083 | 0.894914 / 0.942719 |
| layers.11.self_attn.q_proj | 2 | 5120 → 12288 | separate | 1 | 2 | 6 | 2 | 2 | 8 | 1 | 1.347958 / 1.317375 | 1.401763 / 1.375906 |
| layers.11.self_attn.q_proj | 2 | 5120 → 12288 | separate | 1 | 2 | 8 | 2 | 2 | 8 | 1 | 1.746208 / 1.761249 | 1.863357 / 1.882255 |
| layers.15.self_attn.q_proj | 14 | 5120 → 12288 | separate | 2 | 2 | 2 | 2 | 2 | 8 | 1 | 0.774667 / 0.640146 | 0.447805 / 0.543883 |
| layers.15.self_attn.q_proj | 14 | 5120 → 12288 | separate | 2 | 2 | 3 | 1 | 4 | 8 | 1 | 1.055813 / 1.012020 | 1.006154 / 0.982010 |
| layers.15.self_attn.q_proj | 14 | 5120 → 12288 | separate | 2 | 2 | 4 | 2 | 2 | 8 | 1 | 0.999791 / 0.975146 | 1.033995 / 1.040885 |
| layers.15.self_attn.q_proj | 14 | 5120 → 12288 | separate | 2 | 2 | 6 | 2 | 2 | 8 | 1 | 1.384645 / 1.369312 | 1.404727 / 1.434648 |
| layers.15.self_attn.q_proj | 14 | 5120 → 12288 | separate | 2 | 2 | 8 | 2 | 2 | 8 | 1 | 1.760146 / 1.849021 | 1.945029 / 1.945674 |
| layers.2.linear_attn.out_proj | 15 | 6144 → 5120 | separate | 3 | 2 | 2 | 2 | 2 | 8 | 1 | 0.491042 / 0.467250 | 0.270763 / 0.269112 |
| layers.2.linear_attn.out_proj | 15 | 6144 → 5120 | separate | 3 | 2 | 3 | 1 | 4 | 8 | 1 | 0.611541 / 0.629396 | 0.483940 / 0.502372 |
| layers.2.linear_attn.out_proj | 15 | 6144 → 5120 | separate | 3 | 2 | 4 | 2 | 2 | 8 | 1 | 0.630917 / 0.628876 | 0.479398 / 0.502862 |
| layers.2.linear_attn.out_proj | 15 | 6144 → 5120 | separate | 3 | 2 | 6 | 2 | 2 | 8 | 1 | 0.820855 / 0.822812 | 0.716281 / 0.710078 |
| layers.2.linear_attn.out_proj | 15 | 6144 → 5120 | separate | 3 | 2 | 8 | 2 | 2 | 8 | 1 | 1.042750 / 1.024271 | 1.027013 / 1.016542 |
| layers.24.mlp.down_proj | 3 | 17408 → 5120 | separate | 3 | 2 | 2 | 2 | 2 | 8 | 4 | 0.845625 / 0.845000 | 0.690656 / 0.786409 |
| layers.24.mlp.down_proj | 3 | 17408 → 5120 | separate | 3 | 2 | 3 | 1 | 4 | 8 | 4 | 1.411334 / 1.445292 | 1.388464 / 1.576823 |
| layers.24.mlp.down_proj | 3 | 17408 → 5120 | separate | 3 | 2 | 4 | 2 | 2 | 8 | 4 | 1.392875 / 1.391875 | 1.552018 / 1.544758 |
| layers.24.mlp.down_proj | 3 | 17408 → 5120 | separate | 3 | 2 | 6 | 2 | 2 | 8 | 4 | 1.966479 / 2.066917 | 2.171747 / 2.169219 |
| layers.24.mlp.down_proj | 3 | 17408 → 5120 | separate | 3 | 2 | 8 | 2 | 2 | 8 | 4 | 2.683750 / 2.707438 | 2.872995 / 2.915151 |
| layers.27.self_attn.k_proj | 14 | 5120 → 1024 | separate | 4 | 2 | 2 | 2 | 2 | 8 | 8 | 0.541687 / 0.265750 | 0.085659 / 0.086638 |
| layers.27.self_attn.k_proj | 14 | 5120 → 1024 | separate | 4 | 2 | 3 | 1 | 4 | 8 | 8 | 0.294520 / 0.304979 | 0.123938 / 0.122521 |
| layers.27.self_attn.k_proj | 14 | 5120 → 1024 | separate | 4 | 2 | 4 | 2 | 2 | 8 | 8 | 0.298708 / 0.388292 | 0.121516 / 0.121852 |
| layers.27.self_attn.k_proj | 14 | 5120 → 1024 | separate | 4 | 2 | 6 | 2 | 2 | 8 | 8 | 0.327958 / 0.327021 | 0.155839 / 0.153471 |
| layers.27.self_attn.k_proj | 14 | 5120 → 1024 | separate | 4 | 2 | 8 | 2 | 2 | 8 | 8 | 0.366729 / 0.367125 | 0.191242 / 0.193253 |
| layers.61.linear_attn.in_proj_qkv | 1 | 5120 → 10240 | separate | 1 | 2 | 2 | 2 | 2 | 8 | 1 | 0.543250 / 0.558542 | 0.370010 / 0.374406 |
| layers.61.linear_attn.in_proj_qkv | 1 | 5120 → 10240 | separate | 1 | 2 | 3 | 1 | 4 | 8 | 1 | 0.852770 / 0.868605 | 0.831167 / 0.754833 |
| layers.61.linear_attn.in_proj_qkv | 1 | 5120 → 10240 | separate | 1 | 2 | 4 | 2 | 2 | 8 | 1 | 0.814125 / 0.839604 | 0.749586 / 0.717503 |
| layers.61.linear_attn.in_proj_qkv | 1 | 5120 → 10240 | separate | 1 | 2 | 6 | 2 | 2 | 8 | 1 | 1.136292 / 1.149563 | 1.217607 / 1.176958 |
| layers.61.linear_attn.in_proj_qkv | 1 | 5120 → 10240 | separate | 1 | 2 | 8 | 2 | 2 | 8 | 1 | 1.487854 / 1.514187 | 1.605807 / 1.594102 |
| layers.61.linear_attn.in_proj_z | 1 | 5120 → 6144 | separate | 2 | 2 | 2 | 2 | 2 | 8 | 1 | 0.431959 / 0.414416 | 0.252833 / 0.255310 |
| layers.61.linear_attn.in_proj_z | 1 | 5120 → 6144 | separate | 2 | 2 | 3 | 1 | 4 | 8 | 1 | 0.601792 / 0.660021 | 0.523414 / 0.463971 |
| layers.61.linear_attn.in_proj_z | 1 | 5120 → 6144 | separate | 2 | 2 | 4 | 2 | 2 | 8 | 1 | 0.647938 / 0.598271 | 0.574281 / 0.448742 |
| layers.61.linear_attn.in_proj_z | 1 | 5120 → 6144 | separate | 2 | 2 | 6 | 2 | 2 | 8 | 1 | 0.766417 / 0.821375 | 0.657846 / 0.668367 |
| layers.61.linear_attn.in_proj_z | 1 | 5120 → 6144 | separate | 2 | 2 | 8 | 2 | 2 | 8 | 1 | 0.982250 / 0.950688 | 0.901729 / 0.928263 |

## Projections GDN étroites FP16

Source : `gdn-columns-screen.json`, couche0, 40 paires AB/BA après trois warmups. A = GEMV M1 sérialisés ; B = GEMV à vecteurs colonnes batchés, avec concaténation des poids a/b pour la paire. M3 est aussi contrôlé sur les 96 poids et les 48 paires du checkpoint. Toutes les 162 sorties comparées et les chaînes contrôlées sont finies et identiques en bits.

K/codebook et MB/NT/splits EXL3 ne s’appliquent pas à ces poids FP16. Le dispatch GEMV canonique est BM1/BN8/SM1/SN32/TM4/TN4, soit huit SIMD groups, d’après la source MLX0.32.2 pour K≥16×N ; ceci est une lecture du dispatch, pas un relevé de compteur GPU. Les résultats ne constituent pas une parité du modèle complet pour B.

| Projection | Input → output | M | Isolé A / B (ms) | Chaîne A / B (ms/projection) | Gain chaîne apparié (%) |
|---|---|---:|---:|---:|---:|
| in_proj_a | 5120 → 48 | 2 | 0.496667 / 0.461291 | 0.057643 / 0.052026 | 3.58 |
| in_proj_a | 5120 → 48 | 3 | 0.272459 / 0.268875 | 0.063310 / 0.062930 | 2.16 |
| in_proj_a | 5120 → 48 | 4 | 0.267312 / 0.266084 | 0.050635 / 0.046271 | 9.17 |
| in_proj_a | 5120 → 48 | 5 | 0.223062 / 0.223479 | 0.049255 / 0.045482 | 7.71 |
| in_proj_a | 5120 → 48 | 6 | 0.214000 / 0.221417 | 0.051497 / 0.046206 | 11.80 |
| in_proj_a | 5120 → 48 | 7 | 0.212375 / 0.222687 | 0.058078 / 0.047448 | 23.29 |
| in_proj_a | 5120 → 48 | 8 | 0.219583 / 0.222687 | 0.061234 / 0.048945 | 23.71 |
| in_proj_b | 5120 → 48 | 2 | 0.218229 / 0.221583 | 0.044112 / 0.041198 | 6.65 |
| in_proj_b | 5120 → 48 | 3 | 0.221166 / 0.222187 | 0.044203 / 0.042469 | 4.10 |
| in_proj_b | 5120 → 48 | 4 | 0.222375 / 0.222229 | 0.047557 / 0.043568 | 9.16 |
| in_proj_b | 5120 → 48 | 5 | 0.218583 / 0.226333 | 0.048430 / 0.045375 | 7.62 |
| in_proj_b | 5120 → 48 | 6 | 0.218375 / 0.225500 | 0.050492 / 0.047263 | 7.25 |
| in_proj_b | 5120 → 48 | 7 | 0.216375 / 0.234479 | 0.058648 / 0.050156 | 16.79 |
| in_proj_b | 5120 → 48 | 8 | 0.226375 / 0.233125 | 0.060505 / 0.051786 | 17.68 |
| a+b | 5120 → 96 | 2 | 0.314167 / 0.303396 | 0.085107 / 0.076573 | 11.71 |
| a+b | 5120 → 96 | 3 | 0.237438 / 0.230021 | 0.067401 / 0.049810 | 27.31 |
| a+b | 5120 → 96 | 4 | 0.237125 / 0.234000 | 0.067768 / 0.052677 | 28.56 |
| a+b | 5120 → 96 | 5 | 0.242188 / 0.233313 | 0.079336 / 0.057518 | 39.46 |
| a+b | 5120 → 96 | 6 | 0.258688 / 0.254667 | 0.091388 / 0.064646 | 42.66 |
| a+b | 5120 → 96 | 7 | 0.265709 / 0.253979 | 0.107500 / 0.068617 | 55.22 |
| a+b | 5120 → 96 | 8 | 0.284812 / 0.250500 | 0.131857 / 0.077675 | 70.56 |

Aucun poids préparé ni chemin GDN ajouté au moteur. Les deux tableaux proviennent de fenêtres différentes et leurs temps ne doivent pas être additionnés pour prédire un débit modèle.
