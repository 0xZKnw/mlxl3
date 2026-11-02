# Réduire le coût d'encodage — Sushi, Strata et TensorFold

28 octobre 2026, recherche POSTPROFILE-04-UPSTREAM. La piste commune est de réduire le nombre de primitives et de réutiliser une description du calcul, en conservant une frontière de disponibilité des routes avant la fin du Hot. Aucun gain local nouveau n'est validé par cette revue. Le moteur retenu est revenu à son binaire `a60eb4ad…`, identique à celui d'avant POSTPROFILE-03.

Les sources sont épinglées dans [sources.json](sources.json) : Sushi `3fc47d8a`, Strata natif `5e8a4cca`, TensorFold `a1e6d342`, strata-mlx `94817fd0`. Vingt fichiers téléchargés et vérifiés par blob Git et SHA-256, lecture ciblée des chemins pertinents. Deux fichiers du pool TensorFold sont identiques à la revue précédente. Le HEAD de la recette Mia a été relevé ; ses nouveaux changements n'ont pas été audités ici. Aucun moteur tiers exécuté ni benchmark supplémentaire.

## Ce qui répond réellement au profil

Le [diagnostic local](../../ultra-profile-20261027/attribution/README.md) attribue 11,897 ms/token de CPU actif pendant GPU inactif au premier `slots.indices().async_eval()`, contre 0,245 ms au second `output.async_eval()`, dans une capture instrumentée de vingt pas. Le premier soumet aussi les producteurs amont : Cold précédent, réduction, HC, attention/GDN et routeur. Optimiser seulement les kernels du Hot ne cible donc pas directement toute cette enveloppe. Ces temps ne constituent pas un plafond de gain mesuré en release.

POSTPROFILE-03 confirme une contrainte : supprimer la première soumission fait attendre les routes jusqu'au travail Hot artificiellement retardé. Le test existant échoue avec `ready_at_routes=true` et `ready_at_demand=true`. Le prototype a été retiré avant toute mesure de vitesse. Un futur graphe doit garder l'événement précoce des routes et l'ownership des banques.

| Projet | Mécanisme vérifié | Portée pour MLXL3 |
| --- | --- | --- |
| Sushi, qui utilise aussi MLX | Fusions EXL3, préparation/finition indexées, closures compilées ; sa documentation rapporte l'absence de gain des réglages de buffers | Piste courte : réduire les primitives nécessaires au Cold et au Hot, sans changer nos arrondis |
| Strata natif Metal | Cache de plans par forme, buffers persistants, mise à jour des seuls paramètres variables, échanges CPU/GPU aux coupures d'un command buffer | Piste architecturale pour éviter de reconstruire le graphe MLX par couche |
| TensorFold natif Metal | Encoder conservé dans une session et retrait non bloquant des commandes ; API de commandes indirectes | Exemple d'architecture ; les utilisations ICB retrouvées sont des tests/microbenchmarks |
| strata-mlx | Documentation et contexte d'un moteur distinct | Aucun mécanisme nouveau qualifié ici pour supprimer l'encodage de notre EXL3 |

## Sushi : moins d'opérations, pas une suppression des attentes

[engine-kernels.md:53–62](https://github.com/beamivalice/sushi/blob/3fc47d8a460970b1e86b24de1cab3b0f2067bc72/docs/engine-kernels.md#L53) décrit un decode limité par les frontières de dispatch ; les réglages `MLX_MAX_OPS_PER_BUFFER` et `MLX_METAL_FAST_SYNCH` n'y ont pas aidé. C'est cohérent avec nos essais 01/02 négatifs, sans transférer leurs chiffres à notre machine.

[prepareIndexed/finishIndexed](https://github.com/beamivalice/sushi/blob/3fc47d8a460970b1e86b24de1cab3b0f2067bc72/src/exl3/expert_exl3_kernels.zig#L1905) travaillent directement depuis les slots et les banques d'échelles. [moeSwigluFusedWithShared:3675–3756](https://github.com/beamivalice/sushi/blob/3fc47d8a460970b1e86b24de1cab3b0f2067bc72/src/exl3/expert_exl3_kernels.zig#L3675) fusionne les étapes autour des projections. Attention : `preparedMidOn:10497–10508` exclut explicitement Qwen2560/640/top10 ; son chemin MiMo préparé n'est pas un gain Flash-Next oublié.

Leurs corps `PREPARE_SOURCE:928–960` et `FINISH_SOURCE:963–995` ne reproduisent pas nos stores FP16 : produit et rotation peuvent rester FP32 avant le store final. Un port verbatim viole donc potentiellement notre contrat. La variante locale raisonnable serait de fusionner seulement gather+produit, en gardant les Hadamard et réductions MLX actuels, et de l'exercer aussi sur le Cold.

Les [closures GDN/HC:275–331](https://github.com/beamivalice/sushi/blob/3fc47d8a460970b1e86b24de1cab3b0f2067bc72/src/qwen4_forward.zig#L275) et [HcPrepared:809–890](https://github.com/beamivalice/sushi/blob/3fc47d8a460970b1e86b24de1cab3b0f2067bc72/src/qwen4_hc.zig#L809) montrent comment conserver un graphe avec les tensors mutables en entrées explicites. Mais notre FFI compile déjà activation, gate, mix et SwiGLU ; les gates sont déjà fusionnées avec la récurrence GDN. Leur HC préparé vise aussi un autre format BF16/affine8 et des largeurs2–8. Leur documentation précise que compiler une même formule ne garantit pas les mêmes bits : pas de `compile` global aveugle.

`specRoute/specMatches` et GDN différé restent les mécanismes de la [revue précédente](../../sushi-20261027/README.md). `expert_stream.zig` et `qwen4_forward.zig` ont les mêmes blobs qu'à cette revue. Ils masquent de l'attente sur les hits, sans supprimer le travail d'encodage sur les misses. Leurs possibilités locales déjà mesurées restent distinctes des12ms.

## Strata : le plan est réutilisé, les commandes sont encore encodées

[Window::forward:1117–1144](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/metal/src/engine/window.cpp#L1117) retrouve `wcache_`, ajuste positions et grille QSA, puis exécute le plan. Les scratchs sont des buffers persistants. [run_graph_seg:315–352](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/metal/src/mtl_rt.mm#L315) encode les segments, puis échange des signaux avec le CPU pour les routes/remplissages. Cela donne un exemple concret de frontières de phase, avec traitement d'erreur, plutôt qu'une attente supprimée.

Limite essentielle : `run_graph_seg` parcourt et encode encore les dispatchs à chaque exécution. Le plan CPU est caché ; ce n'est pas une relecture ICB du modèle entier. `Window::run_ds:1223–1225` appelle ce chemin, tandis que les appels `run_icb` retrouvés concernent tests et microbenchmarks. Un port nécessiterait un adaptateur Metal/EXL3, scratchs, événements et leases compatibles MLX ; leurs kernels GGUF ne sont pas notre arithmétique.

## TensorFold : distinguer API disponible et chemin servi

[command_session.zig:28–47](https://github.com/ashhart/TensorFold/blob/a1e6d342462d08bb0c58aae8a990909470e46235/zig/src/metal/command_session.zig#L28) garde un encoder serial pendant la commande, puis suit les commandes en vol et leurs erreurs. [icb.zig](https://github.com/ashhart/TensorFold/blob/a1e6d342462d08bb0c58aae8a990909470e46235/zig/src/metal/icb.zig) expose un enregistrement réutilisable. Les recherches d'appels retrouvent `zig/tests/ordering.zig` et `zig/tests/dispatch_bench.zig`, pas une intégration ICB Flash-Next servie démontrée. Ne pas présenter cette API comme un boost de leur modèle complet.

La [recette:257–261](https://github.com/ashhart/TensorFold/blob/a1e6d342462d08bb0c58aae8a990909470e46235/docs/recipes/qwen3.8-flash-next.md#L257) dit encore que le moteur natif ne sert pas de pack EXL3. Le pool global CLOCK/leases reste pertinent pour les lectures critiques, avec les mêmes2544slots/3Gio dans un premier replay ; il ne supprime pas directement les primitives MLX. Les blobs `fsr/cache.zig` et `flashnext/fsr_storage.zig` sont inchangés : [analyse existante](../../tensorfold-20261027/README.md), pas un nouvel essai ni un gain établi.

## Priorités après cette revue

1. **Fusion ciblée Cold + Hot**, garder les rotations/réductions natives et l'événement précoce des routes. Le code actuel construit quatre gathers d'échelles par `calculate_slots` ; enlever ces primitives est une hypothèse concrète, leur coût CPU exclusif reste non mesuré. Ne pas inventer un pourcentage depuis leur nombre.
2. **Réutilisation d'un graphe de bloc à formes stables**, avec poids, slots et tensors mutables explicites. Mesurer construction/encodage séparément ; nos caches de factories existent déjà et ne sont pas un graphe préparé.
3. **Plan natif Metal et scratchs persistants**, inspiré de Strata, si les étapes précédentes laissent un coût important. Risques plus grands : synchronisation entre runtime MLX et Metal, aliasing, annulation, états récurrents et RAM.
4. **ICB et pool global**, pistes séparées à filtrer avant port. ICB : valider d'abord la compatibilité des kernels/buffers et une phase réelle ; pool : replay indépendant de la politique et protection des lecteurs GPU.

L'ancien [K1](../../flash-next-ssd-impl/opt-82-k1-post-merge/post-merge/real-tps-review.md) demeure rejeté : ses quatre comparaisons DFlashB2 étaient défavorables. Le snapshot source confirme qu'il remplaçait le Hadamard masqué et l'activation Hot, tout en conservant les quatre gathers ; le Cold utilisait encore le Hadamard stock. Ce n'est pas une raison de relancer K1 inchangé. Une fusion des gathers/produits, sans remplacer le Hadamard et couvrant le Cold, serait un essai distinct à préenregistrer.

Aucune de ces pistes n'a une vitesse M5 validée ici. Aucun nouveau code exécutable, modèle chargé, port, push ni remplacement de l'app pendant la recherche. Les sources moteur et le binaire retenus sont identiques à la baseline antérieure au prototype03.
