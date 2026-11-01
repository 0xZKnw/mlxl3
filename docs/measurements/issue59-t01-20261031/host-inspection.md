# T01-P-01 — audit indépendant des données hôte

31 octobre 2026. **Cohérence des données validée, aucune parité numérique ou performance modèle qualifiée.** Lecture/calculs sur les artefacts seulement ; aucun code moteur, benchmark, compilation, inférence ou preuve lancé.

Les [données](host-profile-r1.json) contiennent 24 échantillons : 8 lots ×3 modes ×256 constructions, **6 144 appels**, plus 64 appels de chauffe dont un miss. Ordre off/coarse/fine puis inverse alterné vérifié. Les 8 échantillons fine totalisent **2 048 appels C++, zéro miss, zéro éviction**, et aucune factory n'est chronométrée sur ces hits. Chaque sortie est signalée pending ; zéro readback hôte déclaré. Les partitions Rust sont exactes, les phases C++ restent dans leur total, lui-même dans FFI Rust. Les bornes u64 et compteurs par mode sont contrôlés indépendamment du validateur Rust.

La vraie signature est `mlxl3_rs_expert_2560_640_2_2_1_4_1_slots1_resident53` : header **1 842 octets**, source **14 733 octets**, corps byte-identique au shader figé. Les 500 fichiers listés et le libtest **20a0006f54e00bbae2e937b9c86f73fcc24dfe8262d36b7c5d1179984cae8e0c** (11 036 592 octets) correspondent à l'[identité](instrumented-identity-r1.json), sans écart. Cela ne fige pas les documents/opti non listés. Le [superviseur](host-r1-plan-result.json) termine exit0, processus joint, zéro nouveau Swapout/Swapin.

Temps externes par construction, en **µs** :

| Mode | Moyenne | Médiane | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| off | 5.529215 | 5.509439 | 5.276043 | 5.801105 |
| coarse | 5.369507 | 5.399494 | 5.160969 | 5.547527 |
| fine | 5.883117 | 5.875082 | 5.671711 | 6.212566 |

Ce sont des statistiques de **8 moyennes de lot**, pas des latences individuelles. L'horloge englobe le vrai `expert_slots`, les destructions de graphes et les contrôles de métadonnées. Le JSON [d'inspection](host-inspection.json) conserve tous les résultats en ns, les phases et les ratios par lot.

Ratios observés des moyennes : coarse/off **0.971116×**, fine/off **1.064006×**, fine/coarse **1.095653×**. Ils décrivent les modes et leur effet observateur dans cette série ; aucune différence n'est soustraite pour fabriquer un gain. Les postes C++ sont inclus dans FFI Rust et ne doivent pas lui être ajoutés. `invoke` comprend plusieurs travaux, dont construction signature/source et primitives ; sa durée n'isole pas la génération seule.

**Limites :** aucun graphe évalué, aucune sortie numérique inspectée, aucun temps de hash/pipeline/encodeur GPU, SSD ou tok/s. L'audit n'extrapole pas ces µs à 48 appels par token et ne réfute pas un futur T01B. Un signal faible peut fermer ce filtre après audit ; il ne démontre pas l'inefficacité universelle d'un launcher préparé.

État après l'audit : l'identité500 et le binaire20a décrivent exclusivement
le prototype instrumenté archivé, depuis retiré. Voir le [reçu](retirement-result.json)
et la [vérification finale](verification.md) pour le moteur conservé.
