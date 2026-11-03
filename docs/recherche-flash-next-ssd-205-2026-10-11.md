# Qwen3.8 Flash Next à 2,05 bpw avec streaming SSD dans MLXL3

La voie concrète est de porter l'architecture Flash Next dans le moteur natif, de conserver ses petites matrices en RAM et de reprendre le cache et le relais SSD de `localqwen3.8flash`, avec les kernels EXL3 déjà présents ici. Les poids d'experts du checkpoint sont déjà contigus : une seconde banque de fichiers compactés n'est pas nécessaire. Le gain de débit reste à mesurer après validation numérique.

Recherche du 11 octobre 2026, préenregistrée sous `RESEARCH-2026-10-11-FLASHNEXT-SSD-205` dans [opti.md](../opti.md). Checkout étudié : MLXL3 `1ac910d0a40efcad049c6e5a8cf081d770723465`, avec les modifications MTP locales préexistantes conservées ; projet SSD `44166b76b8c3727a607af311157fa45c69b8d974`, fichiers suivis propres. Le code distant plus récent de MLXL3 n'a pas été audité pour cette recherche. Sources et empreintes : [inventaire](measurements/flash-next-ssd-205/inventory.json).

## Checkpoint exact et place sur le SSD

Le checkpoint est [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), branche **`2.05bpw_h4_ng4`**, révision immuable **`65c895314393431c09050b2e04e250836b3a6eb4`**. La branche `main` contient la fiche, sans les shards de poids. La fiche demande ExLlamaV3 ≥1.4.5 ; son moteur CUDA ne fournit pas directement l'exécution Metal. [Fichiers de la révision](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/65c895314393431c09050b2e04e250836b3a6eb4).

| Fichiers publiés | Taille, en Gio (2³⁰ octets) | Placement proposé |
| --- | ---: | --- |
| Cinq shards principaux | 33,9084 | Petites matrices résidentes, experts et embedding token lus à la demande |
| `ngram_embedding.safetensors` | 24,4380 | SSD et cache de lignes borné |
| `mtp_hyper_connection_mixer_patch.safetensors` | 0,0122 | RAM si MTP activé |
| Total poids, en-têtes compris | **58,3587** | SSD ; ce total n'est pas une consommation RAM |

Avec tokenizer, index et autres fichiers, l'arbre publié représente **58,5012 Gio**. Le réglage 2,05 bpw ne s'applique pas uniformément à tous les tenseurs : experts K2/MUL1, tête K4, n-grammes K4 et matrices de mélange notamment F16. Les valeurs et la qualité de cette quantification n'ont pas été examinées.

## Les surfaces qui manquent ici

La [configuration épinglée](measurements/flash-next-ssd-205/hf-205-config.json) déclare `qwen4_exp` : 48 couches, dimension 2560, 512 experts par couche dont 10 sélectionnés, expert partagé, GDN et attention QSA, quatre flux HyperConnections, PLE et tête MTP dédiée. La [documentation officielle](https://github.com/QwenLM/Qwen3.8-Flash-Next) et les [sources ExLlamaV3 épinglées](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/architecture/qwen4_exp.py) donnent les références d'architecture.

| Entrée actuelle | Travail nécessaire |
| --- | --- |
| [Dispatch natif](../native/src/main.rs), ligne 1360 | Ajouter un modèle `qwen4_exp`, ses caches, snapshots et états de session |
| [Lecteur checkpoint](../native/src/checkpoint.rs), ligne 180 | Charger les deux sidecars nommés ci-dessus, avec validation spécifique de leurs formats |
| [Exl3SwitchGlu](../native/src/moe.rs), ligne 620 | Remplacer le chargement de tous les experts par des banques de slots réutilisables |
| [expert_mapped](../native/src/linear.rs), ligne 889 et [shader mappé](../native/shaders/_qmv_mapped_tile_kernel.metal) | Garder le calcul EXL3 et adapter les strides pour des experts stockés par slot |
| [Bridge Desktop](../apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift), hub et catalogue | Télécharger la révision et ses sidecars, reconnaître le modèle, exposer chargement/streaming/annulation et libérer les ressources au changement de modèle |

Le routeur générique accepte déjà 512 experts. Le fast path de préremplissage segmenté dans `moe.rs` et son `route_plan` sont limités à 256 : prévoir un parcours exact pour les cohortes plus larges. Conserver les dix experts du vrai routeur ; une prédiction de préchargement n'autorise pas à éliminer une route.

## Experts : lecture directe et kernels existants

L'inspection des **sept en-têtes safetensors**, sans lire leurs payloads, trouve **25 088 records d'experts contigus**, dont 24 576 pour le modèle et 512 pour le MTP. Chaque record fait **1 248 012 octets**, avec trois treillis K2, leurs échelles `suh`/`svh` et marqueurs MUL1. Dimensions 640 et 2560 : elles respectent l'alignement 128 des kernels EXL3 actuels. Les [résumés d'en-têtes](measurements/flash-next-ssd-205/hf-205-headers.json) et [quatre autres shards](measurements/flash-next-ssd-205/hf-205-remaining-headers.json) conservent les tailles, formes, comptes et empreintes des en-têtes.

Un expert compact oQ3 du projet SSD faisait 2 150 400 octets. La réduction de record est donc **41,96 %**, à expert identique en dimensions. Il s'agit d'octets sérialisés, sans comparaison numérique entre quantifications ni mesure de vitesse. À titre arithmétique, 10 Gio répartis sur 48 couches offrent 179 records par couche, avant alignement et métadonnées ; le budget réel doit aussi couvrir matrices fixes, contexte, temporaires, MTP et lectures en vol.

Le moteur actuel concatène les experts dans une banque `[tiles_K, experts × tiles_N, mots]`. Un expert y occupe des segments disjoints. Passer à une disposition `[slot, tiles_K, tiles_N, mots]`, avec offsets/alignements validés, permet de remplir directement ses treillis et échelles par `preadv` dans des buffers MLX persistants. Les marqueurs validés peuvent être lus séparément ou absorbés dans des destinations auxiliaires. Conserver le décodage MUL1, Hadamard, arrondis et réductions du shader actuel ; adapter l'adressage et le masquage des slots inutilisés. Éviter allocations, concaténations ou copies de banques entières à chaque miss.

Réutiliser la gestion des lectures courtes/EINTR du [bridge C++](../native/ffi/mlx_bridge.cpp), les réservations de cohorte, leases, vieillissement LFU et fences du [storage SSD](</Users/justin/Documents/localqwen3.8flash/qwen_ssd/storage.py>) et de [StreamingGLU](</Users/justin/Documents/localqwen3.8flash/qwen_ssd/engine.py>). Un slot ne peut être remplacé avant la fin de ses consommateurs GPU. Les lectures, erreurs, annulations et arrêts doivent être bornés et rejoindre tous les workers.

Le [relais SSD](</Users/justin/Documents/localqwen3.8flash/benchmarks/relay.py>) fournit le protocole à porter dans la FFI native : routes GPU disponibles → réservation/publication des slots → calcul des hits pendant les lectures des misses → signal de remplissage → calcul des misses. Ses événements Metal et son service persistant évitent les synchronisations CPU entre chaque opération. Son parcours optimisé actuel vise **T2, MTP1**, avec jusqu'à 20 routes ; les autres formes conservent un fallback exact. Les kernels affines oQ3 ne décodent pas les treillis EXL3.

## N-grammes : le deuxième flux SSD

Le sidecar utilise **`exl3_ngram_trellis`**, un format de lignes différent des matrices EXL3. Il contient 320 001 536 lignes de dimension 160, réparties en 128 shards ; chaque ligne occupe **82 octets**. Les 16 têtes demandent au plus 1 312 octets logiques par token, mais les petits accès aléatoires peuvent provoquer beaucoup plus de trafic physique.

Porter le hash avec overflow 64 bits, reste signé normalisé et segmentation EOS ; dédupliquer/trier les IDs et grouper les lectures voisines. Précharger les lignes connues pendant les premières couches, avant le PLE de la couche 1. Ajouter un petit décodeur Metal : échelle F16 en premier mot, treillis circulaire MUL1, arrondis et biais par tête selon le [codec de référence](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/modules/quant/exl3_lib/ngram_codec.py) et le [décodeur CUDA](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/exllamav3_ext/ngram.cu). Les transformations Hadamard des matrices ne s'appliquent pas à ces lignes.

L'embedding token est encore un autre cas : BF16, 1,1841 Gio sérialisé, lisible par lignes. Le reste du modèle texte et la tête représentent environ **3,0508 Gio de payload** hors experts, embedding token, MTP et vision. Ce chiffre aide à préparer le budget ; il ne prédit ni allocations MLX ni RSS.

## Ce que l'ancien projet établit déjà

La [passe 73](</Users/justin/Documents/localqwen3.8flash/results/round73/REPORT.md>) a validé le relais sur oQ3, M5 de base/24 Gio, SSD interne, MLX 0.32.2, MTP1 : 11,8853→12,2701 tok/s MAIN et 10,6153→11,0041 HOLDOUT, IDs/RNG/cache exacts. Les prompts étaient courts avec capacité de contexte configurée à 65 536. Ce résultat justifie de reprendre le mécanisme, sans prédire son résultat sur EXL3. La [passe 14](</Users/justin/Documents/localqwen3.8flash/results/round14/REPORT.md>) concerne des dictionnaires affine oQ3 : sa compaction n'est pas applicable aux échelles EXL3.

Les essais SSD16, 26, 36, 49, 50, 113 et 114 ont également été consultés : pipeline gate/up/down séparé plus lent, global pool sous son seuil, prédiction Markov faible, préchargement répétitif presque entièrement couvert et derniers essais MTP/relais non retenus. Ne pas les relancer sans nouvelle raison explicite. Ici, LOAD09 a rejeté le mmap ; LOAD10 accélère le chargement initial, sans établir un gain de streaming durant l'inférence. Le journal de nettoyage SSD115 conserve aussi un OOM de génération en présence d'un autre moteur résident.

## Ordre d'intégration et qualification

1. **Référence correcte, MTP désactivé.** Architecture, PLE/n-grammes, lecteur de slots et parcours CLI/bridge, lectures synchrones. Comparer sorties intermédiaires non vides et finies, logits, GDN/QSA et HyperConnections avec une référence indépendante du **même checkpoint EXL3**. Le patch de norme spécifique au checkpoint Vontra oQ3 ne doit pas être appliqué automatiquement. Préserver séparément les dtypes et contrats de Flash Next.
2. **Cache et recouvrement I/O.** Banques persistantes, LFU borné, protection de toute la cohorte, puis relais. Comparer avec le parcours synchrone inchangé, y compris les cohortes de plus de 256 experts, hits/misses mixtes et préremplissage. La nouvelle base K2 justifie cet essai ; elle ne justifie pas des routes approximatives.
3. **MTP1 et relais T2.** Charger le patch mixer, dont les dimensions utilisent les quatre flux (10 240). La [tête de référence](https://github.com/turboderp-org/exllamav3/blob/3a513ac9f91aae777f317ca2341b8e445b62c64e/exllamav3/architecture/qwen4_exp_mtp.py) doit guider la capture de l'état avant collapse. Vérifier acceptation/rejet, rollback de tous les caches et RNG. Les 512 experts de draft représentent 0,5951 Gio sérialisé ; choisir leur résidence selon le budget et l'acceptation mesurée.

Tests à préenregistrer avec l'implémentation : limites d'offset/shard/ligne et mauvais formats ; hash/EOS/treillis circulaire ; lectures partielles, processus silencieux et erreurs de worker ; cohorte entièrement froide, eviction avec GPU en vol, budgets insuffisants ; changement de modèle, reset et annulation ; préremplissage aux frontières 255/256/257 et routage sur 512 experts ; frontières QSA et états MTP. Confirmer qu'un oracle détecte une mutation pertinente avant la qualification finale.

Charger `formal-proof-skill` et ses références lors des changements exécutables, tenter Kani sur les vrais contrats Rust de tailles/offsets/budgets/slots/transitions et le vérificateur approprié pour la FFI. Exécuter format, lint strict, build `mlx,chat`, suites ciblées/complètes et contrôles physiques Metal isolés. Les workflows PR/push actuels ont été inspectés : leur voie CPU/Kani ne qualifie pas la FFI MLX ni les nouveaux shaders. Étendre les contrôles touchés et vérifier les runs du commit effectivement livré.

Mesurer ensuite, une variante GPU à la fois : même checkpoint, prompts MAIN/HOLDOUT figés, warmup, répétitions et ordre alterné, contexte 4096 puis contextes réellement visés. Relever TTFT, prefill, decode, taux de miss/acceptation MTP, octets logiques et physiques, mémoire MLX et RSS séparément, conditions d'alimentation, versions et empreintes binaires. À cache entièrement froid et sans MTP, les experts du modèle exigeraient **599 045 760 octets par token** : le hit rate et le recouvrement restent décisifs. Aucun objectif de tok/s n'est établi par cette recherche.

## État final et limites

**Recherche terminée ; implémentation et performances non vérifiées.** Sept en-têtes ont été lus par requêtes HTTP Range bornées ; tailles, formes et contiguïté documentaire contrôlées. Les résumés conservent leurs hashes ; les en-têtes complets n'ont pas été archivés. Les payloads, index complet et tokenizer n'ont pas été téléchargés ; les SHA LFS publiés ne sont pas des hashes de poids vérifiés localement. Les sources upstream sont conservées comme texte avec leur licence MIT.

**0 test d'inférence, 0 benchmark, 0 propriété formelle exécutés** pour cette tâche documentaire. Les contrôles de JSON, comptes et empreintes sont détaillés dans [validation.json](measurements/flash-next-ssd-205/validation.json). Aucun code moteur, app installée, poids utilisateur ou historique négatif modifié ; aucun essai/processus de cette recherche restant en cours. Vision, contextes longs, qualité 2,05 bpw et débit sur la machine restent à qualifier. L'architecture complète et l'ordonnancement GPU/SSD ne sont pas prouvés par les métadonnées.
