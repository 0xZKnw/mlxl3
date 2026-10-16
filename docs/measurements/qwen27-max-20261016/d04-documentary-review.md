# D04 — vectorisation Mul1 : revue documentaire du 16 octobre 2026

Décision : **déprioriser D04 comme prochaine optimisation**. Le décodeur natif est encore scalaire au niveau du texte MSL, mais son bytesum utilise déjà SWAR. La vectorisation de plusieurs codewords est une famille déjà essayée sur Qwen3.8-27B/M5 : l'ancien prototype `uint4/half4` est retrouvé dans Git et le rapport associé le classe parmi les variantes rejetées/non activées. Passer à `uint2/half2` ne suffit pas, à lui seul, à établir une nouvelle hypothèse de rendement. Cette revue n'exécute aucun prototype ou test et ne mesure aucun gain.

## Périmètre et provenance

Lecture dans le worktree `/Users/justin/.codex/worktrees/qwen27-smallm/mix-stq1_0`, tête `7ce611c1c0db9a704bbe35abccc6293fc6ee7995` avec changements locaux du parent. Recherche couverte par le préenregistrement Q27-2026-10-16-14. Q27-08, Q27-06 et le rapport historique de septembre ont été relus. Q27-19 est une autre modification : ordre des lectures d'activations, sans modification du décodeur Mul1. Ses mesures provisoires ne sont pas un résultat confirmé de cette note.

| Source | Révision/date et passage pertinent |
|---|---|
| Native actuel | `native/src/linear.rs:1236–1248`, `codebook_header` ; `native/shaders/_qmv_tile_kernel.metal:92–123` et `_qmv_mapped_tile_kernel.metal:170–204` |
| Essai historique | Commit `8be9b7bc5c426555894bd299673bb8b80799ec9d`, 6 septembre 2026 ; `benchmarks/benchmark_qwen38_decode.py:37–58`, blob Git `9f85d6bc3ecf71d5984367577919afc96e3f315b` |
| Conclusion historique | `docs/qwen38-decode-local.md`, étude datée du 5 septembre, même commit ; blob `3a9977e44a98ee84cce011cdab35cb96eb957456`, lignes 40–44 |
| Sushi | [code primaire épinglé](https://github.com/beamivalice/sushi/blob/70dc0e807e4113179389377253fe8270d5a3333b/src/exl3/expert_exl3_kernels.zig#L985-L1009), 16 octobre 2026 ; `codebookHelpers` et `exl3_pairh` |
| ExLlamaV3 | [code primaire épinglé](https://github.com/turboderp-org/exllamav3/blob/058b8630fd081f3ff46fce6246955b157c3a2e3b/exllamav3/exllamav3_ext/quant/codebook.cuh#L36-L55), 14 octobre 2026 ; `decode_mul1_product_2` |

Les copies Sushi/ExLlama sont celles de `projects-source-manifest.json` : SHA-256 respectifs `e8c1b1a0dd19dd40444297f2406ca2ea5e3c1f01165b6d472dba74125653c185` et `de6606f17b5d29d6d998dd2987cc3c8474028c738edfca54c2d2d541d9a9f989`. Aucune nouvelle source distante ni code externe exécuté.

## Ce qui existe déjà

Le code natif masque chaque codeword à 16 bits, multiplie en `uint` par `0x83DCD12D`, additionne les octets avec deux lanes indépendantes de 16 bits dans un mot 32 bits, puis effectue la conversion half avec les constantes binaires `0x1EEE` et `0xC931`. Le résultat est converti en float pour les FMA d'accumulation FP32. SWAR réduit donc déjà l'arithmétique scalaire de somme des octets ; il ne traite pas deux codewords ensemble.

Le shader dense prépare quatre poids avant la boucle MB. Le shader mapped conserve les appels dans cette boucle, mais Q27-06 a déjà testé le hoisting explicite à M2 : résultat proche de zéro pour NT2, aucun des vingt filtres passé. L'hypothèse que le compilateur réalise déjà le partage reste une inférence ; aucune ISA n'a été inspectée ici. Déplacer ces appels n'est pas une nouvelle piste D04.

Le driver historique retrouvé contient précisément une fonction `mlxl3_decode_four(uint4)` : masquage 16 bits, multiplication vectorielle, même somme SWAR vectorielle, bitcast `ushort4` vers `half4`, expression half multiplicative/additive avec les mêmes constantes, retour `float4`. Ses quatre composantes remplacent les quatre décodages scalaires du QMV sans réassociation des FMA. Le rapport concerne l'ancien checkpoint `b09c100c0a2885bbf573eac3b7e2c385327d4083`, 27B/M5, greedy AR, batterie ; il rejette les vecteurs packed et le byte-dot FP16. Les valeurs de débit publiées concernent le SWAR retenu, **pas** la variante packed : aucun pourcentage propre à cette dernière ne peut être repris. Les anciens drivers sont absents du checkout mais retrouvés dans Git ; les mesures brutes propres au packed n'ont pas été retrouvées pendant cette revue.

## Transférabilité et risque numérique

Sushi utilise le même calcul avec deux lanes `uint2`, bitcast `ushort2` vers `half2`, puis un `fma` half explicite. Cela fournit une réalisation Metal possible, mais aucune comparaison scalaire/vectorielle isolée transférable à notre kernel. Le nombre de composantes dans le source ne prouve ni l'instruction émise ni son débit. Le changement quatre lanes vers deux peut modifier la pression de registres, mais cette différence n'est pas quantifiée et n'établit pas un bénéfice.

ExLlamaV3 utilise deux sommes CUDA `__dp4a` et un `__hfma2`. Ces instructions ne sont pas des gains disponibles automatiquement par remplacement syntaxique en MSL. La multiplication entière, les bitcasts et les voies indépendantes sont portables ; le débit des instructions CUDA ne l'est pas.

Sur les entiers, deux ou quatre composantes restent indépendantes : multiplication modulo 2^32, somme de deux octets bornée à 510 par sous-lane, somme finale entre `0x6400` et `0x67FC`. Il n'y a donc pas de report entre les sous-lanes 16 bits ni de dépassement ushort final. Ce raisonnement documentaire ne prouve pas le shader. La frontière délicate reste l'arrondi half : remplacer l'expression locale par `fma` explicite ou par une expression élargie float exige de vérifier le résultat compilé, sans supposer son identité.

## Seule modification concrète éventuelle et premier oracle

Une reprise conditionnelle consisterait à remplacer quatre appels scalaires par deux décodages `uint2/half2`, en gardant masques, constantes, frontières half et **les quatre FMA FP32 scalaires dans le même ordre**. Ce serait une variante de largeur de l'essai packed antérieur, pas un nouvel algorithme. Avant nouveau prototype, il faudrait un motif précis : différence d'ISA compilée, baisse démontrée de registres/spills, ou coût dominant du décodeur révélé par un profil. Aucun de ces éléments n'est acquis ici ; je ne recommande pas de lancer cet essai maintenant.

Si ce motif apparaît, l'oracle de départ doit comparer à l'implémentation scalaire compilée indépendante : les 65 536 codes dans chaque lane, paires adverses et permutations (zéro, maximum, motifs alternés et hauts bits masqués), compte attendu, valeurs finies et bits half/float exacts. Un oracle CPU indépendant de multiplication enveloppante/somme explicite des quatre octets/arrondi half doit aussi rester actif. Une mutation de constante ou de lane doit échouer. Ensuite seulement : partielles FP32 QMV stock et chaînes FP16 sur les formes réelles M1/M2 et K1..4, avant le filtre de vitesse apparié. Une parité des textes seule serait insuffisante.

## Conséquence pour l'objectif 15 tok/s

À titre arithmétique, un débit hypothétique de 13,335 tok/s demanderait encore environ 12,49 % de débit, donc 11,1 % du temps total supprimé, pour atteindre 15. Même un véritable facteur deux sur le seul décodage Mul1 demanderait que cette portion représente au moins 22,2 % du temps initial. Cette fraction n'est pas mesurée. D04 ne constitue donc pas une promesse crédible pour le reste de l'objectif.

Ligne de synthèse à remplacer :

| ID / priorité | Mécanisme, source et point local | Nouveauté, contrat et premier oracle | État réel |
|---|---|---|---|
| **D04 R / conditionnel** | Décodage de codewords indépendants `uint2/half2` ; E/S et essai packed `8be9b7bc`, `linear.rs::codebook_header` | SWAR déjà présent ; ancien `uint4/half4` retrouvé et rejeté. La seule largeur deux ne justifie pas une répétition ; exiger un motif ISA/registres/profil, préserver arrondi half et ordre FP32. Oracle exhaustif par lane puis QMV stock | Famille déjà essayée, aucun nouvel essai ni gain. Le cas exact half2 courant n'est pas mesuré ; pas de transfert du CUDA `dp4a` |

Commandes documentaires utilisées : `rg` sur journal/code/notes, `sed` sur sources, `git log -- docs/qwen38-decode-local.md`, `git show 8be9b7bc:<chemin>`, `git ls-tree -r --name-only 8be9b7bc`, `git rev-parse 8be9b7bc:<chemin>` et `git grep` historique. Deux recherches initiales ont rencontré des noms/globs absents ; elles ont été remplacées par inventaire des chemins et lecture via Git. Aucun test, build, proveur, benchmark ou GPU lancé, aucun fichier du worktree modifié. La différence half2/half4 et le débit courant restent non vérifiés expérimentalement.
