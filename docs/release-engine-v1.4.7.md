# MLXL3 moteur 1.4.7 — accélération Qwen3.8-27B

## Nouveautés et réglages

- Le moteur regroupe par défaut les MLP denses aux largeurs de vérification **M2 et M4**, sur les formes M5/EXL3 qualifiées du27B. M3 conserve le parcours en série. `MLXL3_EXPERIMENTAL_DENSE_MLP_BATCH=0` restaure le parcours précédent ; MB3 reste expérimental et désactivé par défaut.
- **Option `MLXL3_EXL3_LANE_ORDER=1`** : disposition des activations adaptée aux lectures directes half4 des projections EXL3 Mul1 qualifiées, sur M5 et M1/M2 par appel. Cette option reste **OFF par défaut** ; absence,0 ou valeur invalide gardent le chemin stock. Elle doit être définie dans l'environnement au démarrage du moteur.
- Les clés Tune distinguent le batch MLP et la disposition des activations : les anciens profils ne sont pas réutilisés comme s'ils avaient été mesurés avec ces réglages.
- Les outils de benchmark comparent deux têtes MTP et refusent les campagnes incomplètes ou divergentes. L'export expérimental Mul1 est contrôlé indépendamment du runtime ; aucun nouveau draft ou checkpoint n'est imposé.

## Performances mesurées

Modèle **Qwen3.8-27B EXL3**, tête MTP4bit d'origine, **MacBook Pro M5 standard,10cœurs GPU,24Go**, MLX0.32.2. Les chiffres concernent le décodage MTP1 avec les mêmes poids, sampler greedy, contexte4096 et cache prompt désactivé.

| Comparaison | Français | Code | Conditions |
|---|---:|---:|---|
| MLP M2/M4 automatique contre MLP en série |9,18→11,08tok/s (+20,7%)|9,01→10,94tok/s (+21,4%)|Campagne Q27-03 sur secteur, disposition stock |
| Option lane1 contre lane0, MLP automatique commun |**10,97→13,65tok/s (+24,4%)**|**10,77→13,38tok/s (+24,3%)**|Campagne Q27-19 sur batterie, ABBA puis BAAB |

Ces gains ne s'additionnent pas : les témoins et conditions diffèrent entre campagnes. Les **15tok/s ne sont pas atteints**. L'option lane1 est nécessaire pour les chiffres13,65/13,38 ; le défaut MLP seul correspond à l'autre comparaison.

Q27-19 conserve huit processus terminés/code0, quatre valeurs par bras et prompt, warmup64tokens,130tokens français jusqu'à EOS et256tokens code. Tokens, texte, historique et compteurs MTP sont identiques, y compris au contrôle historique ; les logits/hidden/états ont aussi des oracles numériques séparés. La variation max/min combinée côté candidat est2,83% en français et2,49% en code, sous le seuil3%. Chaque candidat dépasse chaque témoin d'au moins3%.

Les mesures sont celles de la source prérelease identifiée dans les rapports. Les sept sources natives concernées restent byte-identiques après intégration sur main et changement de numéro de version ; la validation du paquet livré est publiée séparément. Aucun nouveau débit du paquet n'est inventé. [Rapport complet et34mécanismes examinés](qwen27-max-optimisations-20261016.md), [première campagne MLP](qwen27-optimization-20261016.md), [essais négatifs MTP](qwen27-vers-15-20261016.md).

## Correctifs et décisions conservées

- Les sélections de géométrie et d'options gardent leurs replis et profils Tune distincts.
- Les comparaisons de projections, MLP, états et continuations vérifient nombres de valeurs, formes, finitude et égalité bit à bit ; une permutation volontairement corrompue est détectée.
- Les prototypes de tête draft réduite, de fusion GDN et les filtres sans bénéfice qualifié sont retirés. Une mesure microkernel n'est pas présentée comme un gain de génération.
- Aucun poids, quantification cible, KV dtype, sampler ou format de conversation n'est modifié.

## Installation et compatibilité

Artefacts **MLXL3-Engine-v1.4.7-arm64.tar.gz**, manifeste **MLXL3-Engine-v1.4.7-arm64.json** et validation **MLXL3-Engine-v1.4.7-validation.json**. Apple Silicon arm64, macOS26.2+, Desktop1.4.0+, MLX0.32.2. Distribution signée ad-hoc, sans notarisation Apple ; poids et modèles séparés.

Desktop1.4.6 est recommandé pour l'updater indépendant ; les anciens clients peuvent devoir mettre à jour le DMG pour bénéficier du correctif d'installation.

Cette release concerne **le moteur uniquement**. Desktop reste1.4.6/build28 ; son DMG reste latest. La publication moteur utilise **latest=false**. Aucun remplacement de l'app ou installation locale n'est effectué par cette publication. Le manifeste de validation joint publie le SHA source exact, les noms, tailles et SHA-256 des artefacts, ainsi que les résultats réellement inspectés.

## Vérification et limites

- Intégration1.4.7 : **179 tests Rust passés/85 ignorés et442 tests Python réussis**, dont les contrôles de packaging. Format et Clippy strict mlx,chat passent. Les trois harnais Kani ciblés sont rejoués avec succès. Les contrôles d'intégration et de packaging de la version livrée sont consignés dans le manifeste joint, sans confondre une CI en cours avec un succès.
- GPU physique :20projections réelles, mutation détectée sur246227valeurs finies ;384comparaisons MLP ;40configurations de préfixes24/257,100commits,200tokens de continuation,248320logits,5120valeurs cachées et128tableaux d'état. Les32cycles Session comprennent29blocs spéculatifs et3cycles AR.
- Bridge :sixbudgets, deuxcas de préfixe, deuxannulations suivies de reprises exactes, troisrequêtes invalides et repli de pénalité. Le paquet signé est vérifié sur cette frontière, avec délais et fermeture des processus.
- Kani CPU :79obligations/11couvertures pour clé, permutation u32 complète et garde de formes i32/usize/bools, sans assume ni stub et aucune obligation inaccessible. Les contrats MLP antérieurs et leurs bornes restent documentés. Ce model checking ne prouve ni MLX, ni Metal, ni la concurrence ou le modèle complet.
- La vraie API Array reste non vérifiée après erreur interne Kani intrinsics.rs:243. CrossHair indisponible dans l'environnement des nouveaux scripts ; ses tentatives et les échecs de contrôleurs corrigés sont conservés. Les tests ou contrôles ignorés ne sont pas transformés en preuve.
- Les179/85 et résultats physiques ci-dessus décrivent la qualification locale de l'implémentation ; consulter le manifeste pour les nouvelles commandes d'intégration et l'état de chaque job CI du commit publié. Pas de gain RAM revendiqué ; long contexte, autres Mac et qualité globale sur corpus ne sont pas requalifiés. Températures et fréquences non mesurées.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), intégration du moteur et validation sur le Mac physique.
- [Sushi / beamivalice](https://github.com/beamivalice/sushi/tree/70dc0e807e4113179389377253fe8270d5a3333b), idée de disposition des activations et de lectures directes par lane ; cette release en adapte le mécanisme au codec et aux oracles MLXL3.
- [MLX](https://github.com/ml-explore/mlx), runtime Apple Silicon et opérateurs de référence ; [ExLlamaV3](https://github.com/turboderp-org/exllamav3), format EXL3 ; [PonyExl3](https://github.com/beamivalice/PonyExl3), travaux de conversion et d'inférence EXL3.
- [MTPLX](https://github.com/youssofal/MTPLX), travaux amont MTP et architecture de référence ; [Qwen](https://github.com/QwenLM), modèles et architecture ; [Kani](https://github.com/model-checking/kani), vérification des contrats CPU.


## Artefacts et validation du paquet publié

Source du moteur : 957b19469e2a4ed0cb788046cad8e75c100c583d. Le binaire signé annonce ce commit propre, version1.4.7/profilrelease.

| Fichier | Octets | SHA-256 |
|---|---:|---|
| MLXL3-Engine-v1.4.7-arm64.tar.gz | 66960079 | bd952551a310df8cbec8e2ba12513e0544998e794270f11c4b1819f74baaca30 |
| MLXL3-Engine-v1.4.7-verification.tar.gz | 36984 | 7b0fd5c99cbd93784ff8f1dedee402266e5be0218f50f10b2e8349717226f02f |
| MLXL3-Engine-v1.4.7-arm64.json | 13587 | ee9786aac69d681adebb80111e9f6aae1aad11b9c676a232cecb333500594074 |
| MLXL3-Engine-v1.4.7-validation.json | 79462 | 5b3345f5741d1e5d96f85785e89b253f4421cf6f049dbb9233a2f80833934d9b |

Paquet reproductible, signatures/arm64/macOS26.2/hashes vérifiés. Le vrai installateur sélectionne1.4.7 depuis Desktop1.4.6 enstorejetable, exécute le moteur relocalisé puis vérifie le fallback. Le bridge du paquet signé passe sixbudgets, deuxpréfixes, deuxannulations et troisrefus ; les quatre sorties FR/code sous défaut livré et lane1 ont les hashes/compteurs historiques exacts. Aucun gain de vitesse supplémentaire attribué à ce contrôle fonctionnel.

CI du commit exact : [Desktop/protocole](https://github.com/0xZKnw/mlxl3/actions/runs/38940141542) réussi ; [RustLinux/macOS](https://github.com/0xZKnw/mlxl3/actions/runs/38940141824) réussis, étapeKaniusageAPI réussie ; [conversion/packing](https://github.com/0xZKnw/mlxl3/actions/runs/38940141850) réussis. **La suite Kani complète reste en cours au relevé de publication ; aucune CI globale verte ni preuve complète n'est proclamée.** Les trois harnais ciblés de cette optimisation passent localement.
