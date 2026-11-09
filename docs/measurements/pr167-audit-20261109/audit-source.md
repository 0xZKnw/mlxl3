# MLXL3 — audit de la PR #167 et plan correctif pour Codex

## MTP D2–D4, Q02 et Q04

**Date : 9 novembre 2026.**  
**Dépôt :** [0xZKnw/mlxl3](https://github.com/0xZKnw/mlxl3).  
**PR examinée :** [#167 — expert substitution and MTP D1–D4 pilots](https://github.com/0xZKnw/mlxl3/pull/167).  
**Tête figée :** `cf7ba1724bd3a8e925d4206ff04518bbc5294c0f`.  
**Base :** `0fd9fbd1d9f81ff046b4c9a9d495b29d4cb119db`.  
La tête a été revérifiée à la fin de la collecte : elle était inchangée.

> **Mission pour Codex :** améliorer le coût réel des propositions et de leur vérification, corriger les surcoûts du prototype Q04, puis reprendre les variantes Q02 réellement décrites dans l’issue #165. Commencer par les corrections exactes et les formes effectivement exécutées. Conserver les expériences négatives ; ne pas présenter le pilote actuel comme une évaluation définitive du potentiel de D2–D4 ou de Q02.

Cette revue confronte les sources Rust/Metal, les différences de la PR, les mesures JSON, les passages pertinents du journal `opti.md`, les issues #160/#165/#166 et les sources primaires Sushi/TurboEXL3. Trois analyses parallèles ont été recoupées, puis un complément a vérifié les formes B2/B3 et leurs kernels. Les tableaux dérivés ci-dessous sont recalculés depuis les rapports existants. **Aucun nouveau benchmark Metal, test physique du modèle ou changement de moteur n’a été exécuté pendant cette revue.**

---

## 1. Verdict

**Il existe plusieurs défauts d’efficacité et limites d’implémentation concrets. Il n’y a pas, dans les parcours examinés, de preuve que la récursion utilise un mauvais hidden ou que la règle finale de substitution Q04 soit fondamentalement incorrecte.**

| Sujet | Constat | Conclusion opérationnelle |
|---|---|---|
| D2 et vérification B3 | Les projections éligibles traitent trois lignes en deux groupes MB2. Le deuxième groupe n’a qu’une ligne utile. | Les projections multiligne étudiées utilisent une géométrie MB2. Tester une géométrie B3 exacte avant de juger D2. |
| D2–D4 : draft | Chaque proposition termine une sélection de token hôte ; les hidden récursifs Half sont copiés et convertis sur CPU puis remis dans un nouveau buffer. | Supprimer d’abord les copies/conversions hôtes évitables ; mesurer les attentes et les lectures d’embeddings séparément. |
| D2–D4 : travail utile | Plus de propositions, plus de réparations KV et davantage de lignes cible calculées puis abandonnées. | Piloter la profondeur sur le temps par token émis, avec histogramme des préfixes acceptés. |
| Q04 en bloc | La préparation désactive le picker GPU ; sélection CPU et upload des IDs précèdent le lancement Hot. | La voie bloc ne bénéficie pas pleinement du choix GPU anticipé du scalaire. |
| Q04 scalaire | Le pilote chronomètre le choix GPU et un choix CPU complet de validation. | Séparer politique, transitions d’état et oracle ; éviter le double calcul dans la voie de performance. |
| Q02 | Seule une petite fusion exacte norm→prepare du draft a été essayée puis retirée. | L’affine différée/centrée MUL1 et les variantes H128/épilogues restent à implémenter et à évaluer. |
| Comparaison des résultats | AR, D1 scalaire et matrice D1–D4 ne proviennent pas tous du même état du code. | Refaire une comparaison courte sur un binaire unique après les changements ciblés. |

Les limites B3 et une partie de l’ordonnancement sont **héritées du backend antérieur** ; le nouveau pilote D2–D4 les exerce davantage. Les copies récursives et l’adaptateur Q04 bloc appartiennent au prototype de cette PR. Il faut distinguer une régression introduite d’une capacité qui n’a pas encore été optimisée.

Les pilotes sont explicitement des chemins de test. Le module `mtp_pilot.rs` est sous `cfg(test)`, Q04 est un prototype de test, et l’API Flash de production demeure limitée à D1. Les débits publiés ne décrivent donc pas une nouvelle version déjà intégrée à l’application. [M14], [R1]

---

## 2. Ce que montrent vraiment les mesures

### 2.1 Même matrice de profondeurs

M5 24 Gio, Qwen3.8-Flash-Next EXL3 ~2,05 bpw, head4, cache cible de 3 Gio/53 slots par couche, quatre lecteurs, LFU, routage résident activé, HC workspace désactivé. Un échauffement de 64 sorties puis 64 sorties mesurées, dont **63 étapes de decode chronométrées**, une seule réponse mesurée par cellule. [R1], [R2]

| Mode expérimental | Q04 désactivé | Q04 10 % | Q04 20 % |
|---|---:|---:|---:|
| D1, vérification effective B2 | 9,861 tok/s | 10,201 tok/s | 10,275 tok/s |
| D2, vérification B3 | 8,929 tok/s | 9,038 tok/s | 9,557 tok/s |
| D3, blocs ≤3 | 8,749 tok/s | 8,324 tok/s | 8,697 tok/s |
| D4, blocs ≤3 | 9,173 tok/s | 8,953 tok/s | 9,191 tok/s |

**Attention au contexte :** 4096 est la capacité configurée. Les rapports de cette matrice contiennent un prompt de **47 tokens** ; l’offset final vaut 47 + 63 = **110**. Ce n’est pas une mesure à 4096 tokens de contexte actif. Le diagnostic #166 utilisait, lui, un prompt de 87 tokens et un autre protocole. [R3], [R4], [R5], [R6], [R10]

### 2.2 Comptabilité du travail, Q04 désactivé

Les quatre réponses mesurées émettent les mêmes 63 tokens après le premier token. On peut reconstruire le travail à partir des histogrammes et du code du pilote :

| Mesure | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|
| Propositions draft | 35 | 52 | 71 | 81 |
| Propositions acceptées | 27 | 36 | 39 | 42 |
| Acceptées / proposées | 77,14 % | 69,23 % | 54,93 % | 51,85 % |
| Rounds spéculatifs | 35 | 26 | 24 | 21 |
| Sorties par round spéculatif, bonus inclus | 1,77 | 2,38 | 2,63 | 3,00 |
| Tail scalaire D0 | 1 | 1 | 0 | 0 |
| Lignes cible réellement exécutées, tail inclus | 71 | 79 | 79 | 76 |
| Lignes cible excédentaires par rapport aux 63 retenues | 8 | 16 | 16 | 13 |
| Conversions récursives hôtes Half→F32 | 0 | 26 | 47 | 60 |
| Appels de rattrapage `head.extend`, tail inclus | 28 | 37 | 39 | 42 |

**Nature de ces chiffres :** propositions, acceptations et histogrammes viennent des JSON ; conversions, rattrapages et lignes cible sont des déductions du flot de contrôle. Ce ne sont pas des durées mesurées de chaque composant. [M1], [R3], [R4], [R5], [R6]

Calcul reproductible :

- rounds = somme de `draft_lengths` ;
- tail D0 = 63 − rounds − acceptées, dans ces réponses sans EOS ;
- lignes cible = somme de `(largeur × nombre de blocs)` + tail D0 ;
- conversions récursives = somme de `((profondeur − 1) × nombre de rounds à cette profondeur)` ;
- rattrapages = acceptées + tail D0, pour ces trajectoires.

Contrôle indépendant : `(47 + lignes_cible) × 48` reproduit exactement `device_hot_submissions` : **5664 / 6048 / 6048 / 5904**. Ce compteur inclut le prefill.

**Interprétation :** D4 obtient réellement plus de sorties par round. Son acceptation moyenne plus basse ne suffit donc pas à invalider l’algorithme. Mais il paie 81 propositions et 42 rattrapages, contre 35 et 28 en D1, sans réduire assez le travail de la cible. Les proportions d’acceptation agrégées ne donnent pas la probabilité conditionnelle d’accepter le deuxième, troisième ou quatrième token : cette télémétrie manque.

### 2.3 À Q04 20 %, D4 demande plus de lectures

| Mesure sur le decode | D1 / 20 % | D4 / 20 % |
|---|---:|---:|
| Lignes-couches exécutées | 3408 | 3792 |
| Lignes-couches retenues | 3024 | 3024 |
| Équivalents lignes cible, sur 48 couches | 71 | 79 |
| Records froids demandés | 7234 | 8089 |
| Octets logiquement demandés | 9 028 118 808 | 10 095 169 068 |
| Débit | 10,275 tok/s | 9,191 tok/s |

D4 demande ici **11,82 % d’octets en plus** pour la même quantité de sorties utiles. Cela contredit l’idée que davantage de profondeur amortit automatiquement le SSD dans ce prototype. Ce sont des octets de records demandés par le moteur ; ils ne mesurent pas les octets effectivement lus sur la NAND ni les effets du cache système. [R7], [R8]

---

## 3. MTP : corrections classées par priorité

### MTP-1 — P1 : le passage de B2 à B3 franchit une limite de géométrie MB2

**Statut : fait de code confirmé ; importance temporelle à mesurer. Backend préexistant.**

`verify_rows_key` et `verify_specialized_key` acceptent bien B2, B3 et B4. Il serait donc faux d’écrire « B3 perd forcément les spécialisations et tombe dans un GEMV générique ». En revanche :

- les kernels de vérification utilisent `MLXL3_QMV_MB = 2` ;
- `Projection::encode_expert_stage` lance `ceil(batch / 2)` groupes sur cet axe ;
- le chemin experts × consommateurs applique aussi `ceil(rows / 2)` ; [M12]
- pour une dernière ligne absente, les entrées sont mises à zéro et les écritures masquées, mais les boucles d’accumulation et de réduction du template restent présentes. [M4], [M5], [M6]

Sur les blocs de la matrice sans Q04, en excluant le tail scalaire :

| | D1 / B2 | D2 / B3 |
|---|---:|---:|
| Blocs vérifiés | 35 | 26 |
| Lignes des blocs | 70 | 78 |
| Groupes MB2 par projection éligible | 35 | 52 |
| Emplacements de lignes dans ces groupes | 70 | 104 |
| Emplacements masqués | 0 | 26 |

Soit **48,6 % de groupes MB2 supplémentaires** pour **11,4 % de lignes de blocs supplémentaires**. Les deux groupes B3 exécutent leur propre parcours de chargement/décodage des poids. Cela ne prouve ni +48,6 % de temps GPU, ni +48,6 % de trafic DRAM, et encore moins un doublement des lectures SSD : caches matériels et recouvrement interviennent.

**Correction demandée :**

1. Instrumenter les PSO/formes réellement utilisés : B1/B2/B3/B4, dimensions, quantification, splits, NT, MB, rôle GDN/expert/head, groupes et fallbacks.
2. Tester un PSO **MB3 exact** sur les projections dominantes, en gardant le même ordre de réduction par ligne et les mêmes frontières Half.
3. Comparer MB3 à MB2+tail spécialisé B1, ainsi qu’à la voie MB2 actuelle. Un tail B1 peut éviter des accumulations inutiles mais ne supprime pas le second parcours des poids.
4. Étendre seulement ensuite aux experts ayant trois consommateurs réellement communs. Les experts consommés par une ou deux lignes doivent garder leurs chemins adaptés.
5. Mesurer ressources compilées, registres et occupation : une variante qui partage davantage les poids peut néanmoins être plus lente.

**Contrainte importante :** MB4+NT2 avec l’accumulation de référence dépasse le budget de scratch indiqué par le code. MB3+NT2 emploie déjà 24 Kio pour le seul tableau `reference_partials`, auxquels s’ajoutent les autres tableaux. Lire `staticThreadgroupMemoryLength` et les limites du pipeline réel. Ne pas réactiver discrètement la réduction Q01 pour faire entrer un MB4 ; Q01 a été retiré sur demande de l’utilisateur. [M4], [M6], [R9]

La variante temporelle GDN choisie pour trois lignes est nommée B4, **mais elle s’arrête sur la longueur réelle**. Ne pas accuser une quatrième mise à jour GDN imaginaire. Le gaspillage décrit ici concerne la géométrie QMV MB2. [M7]

### MTP-2 — P1 : garder le hidden récursif sur GPU, sans inventer un gain de synchronisation

**Statut : copies et conversions confirmées dans le nouveau pilote. Coût en millisecondes non isolé.**

Dans `propose_hidden`, si une nouvelle proposition doit suivre :

1. `output.export_bytes` construit un `Vec<u8>` ;
2. Rust convertit les 10 240 Half en `Vec<f32>` ;
3. `f32_storage` crée un nouveau buffer pour la proposition suivante.

Cela se produit **26 / 47 / 60 fois** sur les réponses D2 / D3 / D4 sans Q04. [M1]

**Nuance matérielle essentielle :** la sortie HC est en `StorageModeShared`. Dans cette branche, `export_bytes` lit la mémoire partagée et ne lance pas automatiquement un blit avec attente. Il faut parler de copies CPU, allocations et conversion sur CPU, pas de 60 téléchargements PCIe ou de 60 attentes GPU supplémentaires démontrées. [M8], [M13]

**Correction demandée :**

- Ajouter une conversion Half→F32 sur GPU ou un chemin de consommation du Half qui reproduit exactement la conversion attendue.
- Réutiliser des buffers bornés par profondeur et expliciter leur durée de vie.
- Garder un contrôle de finitude adapté, sans imposer une copie intégrale des 10 240 éléments dans chaque round de performance.
- Vérifier les valeurs converties, les offsets et la continuation après rejet ; préserver exactement le hidden pre-mixer à quatre streams.

Le hidden choisi par le pilote est cohérent avec Sushi : la récursion repart du stream **4 × 2560** après le layer MTP, avant le mixer destiné à la tête de vocabulaire. Aucun défaut de type « il réinjecte le mauvais hidden de 2560 éléments » n’a été démontré. [S1], [S2]

### MTP-3 — P1/P2 : construire une chaîne de draft moins synchrone, avec un budget mémoire explicite

**Statut : différence d’architecture confirmée ; refonte à mesurer.**

Le pilote appelle successivement `mixed`, le layer, le mixer puis `device_greedy_head`, qui rend un ID au CPU, pour chaque proposition. Sushi construit une chaîne dont les tokens sont des arrays GPU différés et enchaîne les hidden sans synchronisation hôte à chaque étape. Ce mécanisme amont est pertinent ; ses gains ne sont pas transférables tels quels. [M1], [S1]

**Obstacle concret à traiter avant de copier ce schéma :** MLXL3 récupère actuellement les embeddings par token avec lecture d’une ligne sur miss et un petit cache. Garder tous les IDs exclusivement sur GPU suppose un chemin de gather compatible, des embeddings résidents ou une stratégie de prélecture bornée. Charger silencieusement toute la table d’embeddings ferait changer le budget mémoire et le cache experts. [M9]

**Plan :**

1. Corriger le hidden CPU de MTP-2.
2. Mesurer séparément sélection de l’ID, attente de fin de commande, lookup embedding et calcul du draft.
3. Si le profil le justifie, introduire une API qui conserve les IDs GPU et encode plusieurs étapes en respectant la dépendance autorégressive.
4. Publier RAM réellement vivante et octets de cache experts perdus/gagnés.
5. Garder l’EOS, l’annulation et la validation du préfixe corrects même si la chaîne exécute un suffixe finalement inutile.

Une fusion de commandes réduit potentiellement du coût de soumission ; elle ne rend pas indépendantes les propositions autorégressives.

### MTP-4 — P1 diagnostic : réparer l’état du draft coûte, mais ce n’est pas un second modèle complet

Le pilote restaure le snapshot après la première proposition, puis appelle `head.extend` sur les hidden cible vérifiés pour les autres positions retenues. Cette réparation maintient le contrat actuel du draft : les activations produites pendant la spéculation ne sont pas celles du target. La supprimer changerait ce contrat et pourrait dégrader l’acceptation. Cela ne démontre pas, à lui seul, une corruption des sorties finales : celles-ci restent vérifiées par la cible. Une autre politique de cache draft doit donc être évaluée comme un changement du drafter, avec ses propres garanties et mesures. [M1]

Le M5 dispose déjà du chemin **KV-only catchup** dans `FlashMtpHead::extend`. Il ne faut donc pas écrire que chaque réparation réexécute systématiquement tout le MoE du draft et son LM head. Ces appels ajoutent du travail en projections, commandes et gestion d’état, dont le coût temporel reste à isoler. [M10]

Sushi fournit une piste précise : conserver l’historique accepté à ajouter et fusionner son append avec la première proposition du round suivant. Adapter cette idée au KV-only natif, sans revenir à un append complet plus cher. [S1]

**Correction demandée :**

- compter les lignes KV à rattraper et leur temps ;
- prévoir un rattrapage groupé ou différé avec offsets et durée de vie des hidden explicites ;
- qualifier les branches aucun/tous/partiellement acceptés, EOS et arrêt de génération ;
- imputer le travail différé une seule fois au débit utile.

Les copies de snapshots et les transitions QSA/GDN méritent un profil, mais aucun chiffre ne permet ici de les déclarer dominantes.

### MTP-5 — P1 : choisir profondeur et largeur sur le coût utile

Avec la limite de trois lignes demandée pour ce pilote :

- D1 vérifie ancre + une proposition : B2 ;
- D2 vérifie ancre + deux propositions : B3 ;
- D3 peut nécessiter 3 + 1 lignes ;
- D4 peut nécessiter 3 + 2 lignes.

Le deuxième chunk n’est exécuté que si le préfixe le justifie. **Le découpage 3+2 est intentionnel, pas un faux D4 ni un bug de borne.** Il réduit toutefois la réutilisation entre chunks. D3 peut payer un second passage peu amorti sur une ligne ; cela constitue une explication plausible de son résultat inférieur à D4, sans que cette unique mesure établisse la causalité. [M1], [R1]

La quantité à optimiser est :

**Débit utile = tokens réellement émis / durée complète du round**, en incluant draft, vérification, réparation, publication et orchestration.

Dans l’implémentation, utiliser une formule ou métrique numérique, pas additionner des spans hôte inclusifs qui se chevauchent.

**Livrables :**

- histogramme de longueur acceptée, rejet à chaque profondeur et EOS ;
- acceptation conditionnelle par rang ;
- lignes cible exécutées/retenues et lectures des suffixes rejetés ;
- temps par composant et temps mural par token émis ;
- profondeur effective sélectionnée et raison de sélection.

Commencer par un choix simple D1/D2 fondé sur des moyennes lissées de coût. Étendre à D3/D4 si le backend et l’acceptation le justifient. Une comparaison B4 pour D3 doit être une variante explicitement nommée ; ne pas changer discrètement la convention Bmax3. Même après les corrections, un contrôleur peut légitimement choisir D1 sur certains prompts.

### MTP-6 — P2 : la tête cible reste évaluée ligne par ligne

`BlockTxn::greedy` exécute mixer et tête pour chaque ligne effectivement consultée. Cette sélection paresseuse évite de projeter les lignes rejetées, ce qui est utile. Mais elle ne mutualise pas la lecture des poids de la tête entre lignes. [M2]

Comparer, seulement si le profil montre un coût significatif, une projection groupée des lignes probables avec la sélection paresseuse actuelle. Mesurer le compromis entre poids partagés et tête calculée inutilement après rejet. La tête du draft, appelée 35 à 81 fois dans cette matrice, mérite aussi un compteur séparé. Une tête draft approximative/shortlist serait une expérience distincte ; ne pas la mélanger au premier correctif exact.

### MTP-7 — P2 robustesse : vérifier l’annulation juste avant publication

**Défaut de contrôle confirmé par lecture ; aucune exécution concurrente de reproduction effectuée dans cet audit. Ce point n’explique pas les tok/s.**

Le pilote vérifie l’annulation au début et dans plusieurs boucles, mais pas après sa dernière réparation/projection avant de publier un résultat. Si l’annulation arrive pendant le dernier `head.extend`, l’opération peut réussir puis publier un token et un préfixe avancé. La session de production, elle, effectue un contrôle final avant son chemin de restauration sur erreur. [M1], [M15]

Ajouter ce contrôle **avant le traitement succès/erreur**, de façon que toutes les branches, y compris D0, empruntent la restauration target + draft sur annulation. Vérifier les compteurs et l’état de session après une reprise. Un test avec injection à cette frontière doit distinguer la version précédente du correctif.

### MTP-8 — Préserver les optimisations déjà présentes

Le vérificateur n’est pas une simple répétition générale du chemin AR : les cohortes SSD, la réutilisation des records, le traitement au fil des complétions et les kernels experts × consommateurs existent. Ne pas recycler les anciens constats de #161 comme s’ils étaient toujours présents. En revanche, `prepare_block_inputs` groupe les projections GDN, tandis que les producteurs QSA/PLE restent principalement préparés ligne par ligne. C’est une piste de regroupement supplémentaire à profiler après les corrections prioritaires. [M11], [M16]

---

## 4. Q04 : garder la politique, retirer les surcoûts du prototype

### Q04-1 — P1 : le bloc repasse par le CPU avant le Hot

**Statut : ordre d’exécution confirmé dans la PR.**

Pendant la préparation des lignes, `Preparing` empêche `encode_pick` d’encoder le choix GPU. Ensuite `forward_block_routed_cancellable` exécute, pour chaque ligne, **tout `block_pick`** : lecture des IDs/probabilités/scores, choix Rust, mise à jour de la famine, allocation/upload des IDs puis réservation du cache logique et conservation du préfixe. `submit_resident_hot_block` démarre seulement après cet ensemble. [Q1], [Q2], [Q7]

Le bloc perd donc l’anticipation du choix GPU dont bénéficie le scalaire. **Cela ne peut pas expliquer les ralentissements lorsque Q04 est désactivé** : c’est un problème supplémentaire propre à la composition Q04 + bloc.

**Correction progressive :**

- Première étape : scratch préalloué, lectures de métadonnées regroupées, sélection des quelques meilleurs candidats sans tris/allocations complets inutiles, table compacte d’IDs.
- Deuxième étape : choix et transitions logiques sur GPU, puis lancement Hot dès que possible.
- Conserver un oracle CPU indépendant hors mesure pour qualifier les décisions.

**Ne pas lancer naïvement tous les picks en parallèle sur un même masque.** La résidence logique de la ligne suivante dépend de la transition de la ligne précédente. La version performante doit conserver cet ordre sémantique, même si elle optimise l’ordonnancement physique.

### Q04-2 — P1 : le mode performance ne doit pas recalculer l’oracle complet

En scalaire, `moe.rs:463–515` lit les 512 probabilités, reconstruit des vecteurs/masques, exécute `quality_select`, lit le choix GPU et compare les deux. La demande SSD est réservée ensuite. Une partie du travail peut être masquée par les experts résidents déjà soumis, mais le choix CPU reste sur le chemin qui précède la demande des manquants. Le picker alloue aussi des buffers ready/starved/output par couche. [Q3], [Q4]

Sushi possède également un contrôle hôte de son choix spéculé ; la seule présence d’un oracle n’est donc pas une preuve de mauvaise traduction. La correction consiste à mesurer puis supprimer le coût superflu du pilote, en conservant son contrat. [S3]

**Attention : `quality_select` met aussi à jour les compteurs de famine.** Retirer cet appel ou seulement désactiver l’assertion ne suffit pas : on supprimerait soit une transition sémantique nécessaire, soit presque aucun coût.

Transitions minimales à préserver pour les experts originaux : **hit → compteur inchangé ; miss substitué → incrément si le compteur est inférieur à trois ; miss non substitué → remise à zéro**. Cette dernière règle s’applique aussi aux experts protégés, à la limite de substitutions atteinte et à l’absence de candidat. [Q6]

**Correction demandée :**

- séparer calcul du choix, transition de politique et validation ;
- faire produire les choix/deltas nécessaires par la voie active ;
- appliquer les transitions sans refaire un tri complet ;
- conserver l’oracle complet dans les tests et un mode de validation explicite ;
- préallouer les buffers par session/profondeur.

Tester l’égalité des IDs, poids, compteurs et états logiques avant/après cette séparation. Ne pas comparer une voie sans oracle qui aurait aussi changé la politique.

### Q04-3 — P2 : réduire les copies de métadonnées sans perdre le rollback

`DecisionSnapshot` contient notamment 48 × 512 compteurs ; `DecisionFrame` retient un état de cache par ligne/couche. Les `Rc::make_mut(...).reserve(...)` peuvent cloner `ExpertCache` quand un snapshot possède encore l’état précédent. Cela conserve correctement des préfixes indépendants, mais peut générer beaucoup d’allocations et de copies CPU. **Son poids temporel n’est pas mesuré dans les rapports actuels.** [Q5], [Q7], [Q8]

Profiler nombre et volume des clones. Si nécessaire, employer une arène réutilisable ou des deltas bornés par profondeur. Garder l’état exact LFU/famine/positions, y compris la décroissance périodique des compteurs. Un « cache logique simplifié » qui change ses décisions n’est pas une optimisation exacte.

### Q04-4 — P1 instrumentation : distinguer chaleur logique et présence physique

La séparation actuelle est nécessaire : une lecture issue d’un suffixe rejeté peut rester en cache physique sans modifier le target logique. Sinon, le résultat dépendrait du chunking et de l’ordre d’arrivée des lectures. [Q1], [Q5], [Q7]

Il faut mesurer :

- remplaçant logiquement présent mais physiquement absent ;
- original logiquement absent mais physiquement présent ;
- lectures/évictions provoquées par les suffixes rejetés ;
- octets demandés par token émis ;
- part des lectures réellement critiques et part recouverte par le calcul ;
- mêmes compteurs avec Q04 désactivé, où les rapports actuels indiquent `policy: null`.

Ne pas « corriger » un remplaçant physiquement absent en remplaçant le masque logique par l’état physique instantané. Optimiser plutôt le placement, la rétention ou la prélecture compatibles avec la décision logique.

### Q04-5 — Ce qui est déjà corrigé ou volontaire

Dans MLXL3 au head audité, après comparaison à la révision Sushi citée :

- la sélection traite les misses logiques ;
- les poids de mélange d’origine sont conservés ;
- les candidats sont distincts et hors sélection d’origine ;
- les trois meilleurs experts sont protégés ;
- au plus trois remplacements, avec poids original ≤0,10 ;
- les hits préservent le compteur de famine ; la récupération après trois reports est conservée ;
- le draft est exclu de la politique Q04 cible. [Q6], [S4], [M1]

La protection des trois meilleurs experts, la limite de trois remplacements et le plafond de poids 0,10 sont les adaptations conservatrices de MLXL3 ; elles ne sont pas des contraintes copiées telles quelles de Sushi.

Les anciens défauts du replay — choisir un expert déjà en cache, mauvaise transition du compteur, hypothèse SLRU au lieu de LFU — et le recalcul du gate appartiennent à des expériences archivées. **Ne pas les présenter comme des bugs encore présents au head audité.** [R9]

Sushi emploie un seuil en log-score ; MLXL3 réutilise ses probabilités canoniques. Le rapport mathématique est apparenté, mais les frontières d’arrondi peuvent différer. Revenir aux log-scores serait une variante de politique à qualifier, pas une réparation automatiquement exacte.

---

## 5. Q02 : le résultat négatif ne concerne pas les variantes principales

### 5.1 Prototype réellement testé

Archive : `eeecec02290a35a3667b43745f469c63e33a214c`.  
Retrait : `ac71f0a3bbc7c10ffd6f04156ce3ffca8c942a25`.

L’essai fusionne la normalisation et la préparation EXL3 du hidden MTP en préservant les arrondis exacts. Il évite un intermédiaire normalisé de **40 Kio**, soit **80 Kio d’écriture/lecture explicites**, et utilise un facteur de normalisation scalaire. Mais :

- les deux dispatchs restent présents ;
- aucune lecture SSD n’est supprimée ;
- aucune frontière de synchronisation CPU/GPU n’est supprimée par ce seul changement ;
- l’allocation du facteur remplace celle de l’intermédiaire, sans supprimer automatiquement une allocation. [T1], [R9]

La moyenne contrôlée passe de **10,399104 à 10,366439 tok/s : −0,314 %**, avec **−0,786 % de dérive entre contrôles**. Cela signifie **gain non démontré**, pas régression précisément établie. Le premier pilote plus court à environ +4,94 % n’a pas été confirmé par cette rotation. [T1]

Le chemin projeté rejoint bien `encode_expert_stage`. Le scratch réutilisable excluait déjà `mtp_hidden` ; aucune perte de spécialisation utile n’a été démontrée dans cette petite fusion. Ne pas inventer un mauvais dispatch pour expliquer son résultat. [T2], [T7]

### 5.2 Ce qui n’a pas été implémenté

Au head, le kernel QMV décode encore les poids Half avant les FMA. Le décodeur MUL1 conserve l’affine et son arrondi Half par poids. **L’affine différée de TurboEXL3 n’est pas ce que le pilote Q02 a testé.** Les variantes H128 et épilogues proposés dans l’issue ne sont pas non plus épuisés par cet essai. [M6], [T3], [T4]

La fusion exacte archivée peut rester une petite brique à réexaminer en composition après audit de son coût. La priorité est de nommer séparément les expériences, pour ne plus confondre :

- Q02-NORM-PREPARE-EXACT ;
- Q02-MUL1-DEFERRED ;
- Q02-MUL1-CENTERED ;
- Q02-H128/EPILOGUES, découpé par consommateur.

### 5.3 Variante MUL1 affine différée : port précis

Dans le mécanisme amont, `t` est la somme des quatre octets du codeword hashé, entre 0 et 1020. Avec :

- A = 887 / 131072 ;
- C = −3,453125 ;

le poids affine non arrondi vaut A × t + C. On peut accumuler :

```text
Sxt = somme(x_hat[i] * t[i])
Sx  = somme(x_hat[i])
acc = A * Sxt + C * Sx
```

Turbo applique cette transformation après la réduction des partiels, plutôt que l’affine Half à chaque poids. **Cela retire un arrondi individuel des poids ; ce n’est pas bit-exact avec le calcul canonique actuel.** Il faut un profil numérique explicite et une évaluation de la qualité ajoutée sur le même checkpoint. [T4], [T5]

**Points de correction obligatoires :**

1. Borner la variante au codebook MUL1 concerné ; ne pas modifier les autres décodeurs.
2. Calculer `Sx` depuis l’entrée réellement consommée, après les préparations/scalings/arrondis pertinents. La somme du hidden original n’est pas interchangeable.
3. Garder des sommes séparées gate/up si leurs facteurs `Suh` diffèrent.
4. En split-K, appliquer `C × Sx_global` une seule fois après assemblage, **ou** utiliser la somme locale de chaque partition. Ne jamais répéter la correction globale dans chaque split.
5. Préserver les formes spécialisées, la disposition des partiels et les sorties B1/Bn.
6. Mesurer le coût de calcul de `Sx`, sa réutilisation et le travail supprimé ; une affine moins chère peut perdre son avantage si la préparation est répétée pour chaque consommateur.
7. Évaluer indépendamment erreur de projection, cohérence du nouveau target AR/Bn, puis qualité modèle.

### 5.4 Variante centrée : expérience séparée

Avec u = t − 512 :

```text
acc = A * somme(x_hat[i] * u[i]) + 0.01171875 * somme(x_hat[i])
```

La constante vient de C + 512A. Cette écriture peut diminuer une annulation numérique entre deux termes importants ; ce n’est pas la preuve d’une meilleure qualité finale et elle ne recrée pas l’ancien arrondi Half par poids. Comparer différée non centrée, centrée et canonique comme trois calculs distincts. [T4], [I1]

### 5.5 H128 et épilogues : reprendre le mécanisme complet

Turbo fournit une fusion de réduction gate/up, correction affine, Hadamard, scaling, SwiGLU puis préparation de down. Sa préparation conserve les arrondis et les sommes requises pour le consommateur suivant. Ce mécanisme est beaucoup plus large que l’intermédiaire norm→prepare du draft. [T6]

Commencer par une fusion dont le consommateur est clairement identifié et les buffers évités mesurables. Garder normalisations sensibles, routeur et états récurrents dans leur contrat prévu. Ne pas activer un fast-math global et attribuer ensuite tout écart à « Q02 ».

**Q01 reste retiré.** Les variantes Q02 doivent pouvoir être testées sans réintroduire cette réduction sous un nouveau nom.

---

## 6. Protocole correctif court et exploitable

### 6.1 Référence et instrumentation

Lire le `AGENTS.md` courant et les entrées `I165-Q02`, `I165-Q04`, `I164` pertinentes. Avant chaque nouvel essai, ajouter la raison nouvelle et le protocole à `opti.md`. Réutiliser les contrôles antérieurs non affectés ; ne pas répéter une campagne complète pour chaque changement de scratch.

**Fixer un binaire unique pour AR / D1 scalaire / D1 bloc / D2–D4.** Les anciens 11,44 tok/s AR et 10,79 tok/s D1 scalaire ont été mesurés avant certaines additions du cache logique et des préfixes. Leur comparaison avec le D1 bloc 10,28 n’isole pas le seul coût du bloc. [R9]

Séparer les options de sémantique des options d’observation :

- profil exact ou Q04, tau, nombre de substitutions, politique logique ;
- mode oracle complet ou mesure de performance ;
- profondeur demandée/effective et largeur maximale ;
- kernels/PSO réellement choisis ;
- options effectives de cache, Hot, HC, catchup et lecteurs.

L’option `MLXL3_NATIVE_MTP_BLOCK_VERIFY=0` du superviseur ne signifie pas que le pilote profond est scalaire : celui-ci appelle directement son vérificateur de bloc. Rapporter la voie exécutée, pas déduire le backend d’un flag isolé. [R2], [M1]

### 6.2 Compteurs utiles

Pour chaque configuration, conserver au minimum :

| Famille | Données |
|---|---|
| Résultat | tokens émis, temps decode, temps total, TTFT, contexte actif et capacité |
| Spéculation | rounds, profondeur effective, propositions, acceptations par rang, histogramme du préfixe retenu |
| Cible | lignes exécutées/retenues/rejetées, widths, PSO B1/B2/B3/B4, groupes MB2/MB3 |
| Draft | forwards, têtes vocabulaire, conversions, copies, allocations, lignes KV à réparer |
| Routage | temps choix CPU/GPU, temps avant demande SSD, Hot lancé et utile |
| I/O | records uniques demandés, octets par token utile, cache logique/physique, lectures rejetées |
| État | snapshots/clones, mémoire réellement vivante, compression/swap observés |
| Provenance | SHA source, hash binaire, checkpoint/pack, flags effectifs, ordre des bras et conditions |

Profilage et vitesse peuvent être deux passages courts distincts si l’observation est intrusive. Le diagnostic #166 contient des spans hôte **inclusifs**, avec chevauchements ; leur somme ne fournit ni la durée GPU exclusive ni un plafond d’accélération. [R10]

### 6.3 Tests qui ciblent les risques réels

**Pour une optimisation exacte de transport/ordonnancement :**

- CPU Half→F32 contre nouveau chemin GPU, frontières/valeurs finies et formes complètes ;
- B1/B2/B3, sorties et états du même target ;
- rejet à chaque rang, acceptation complète, EOS, budget de sortie, contexte épuisé ;
- annulation après soumission et avant publication ;
- réparation KV et continuation sur plusieurs tokens ;
- mêmes valeurs de préfixe, pas seulement un texte final similaire.

**Pour Q04 :**

- égalités de scores, seuils tau, poids autour de 0,1, rangs non triés et candidats distincts ;
- trois reports puis chargement à l’occasion suivante, avec hit intermédiaire ;
- décroissance LFU à sa frontière ;
- différents chunkings donnant le même préfixe accepté ;
- cache physique perturbé par des lectures rejetées, cache logique identique ;
- état logique, compteurs, IDs, poids, logits et états QSA/GDN après commit.

**Pour Q02 numérique :**

- oracle indépendant du composant, chaque codebook et forme affectée ;
- tests split-K pour détecter la correction affine appliquée plusieurs fois ;
- sommes d’entrée gate/up distinctes ;
- cohérence du nouveau calcul entre AR et vérification ;
- petit filtre qualité attribuable, puis qualification de la composition retenue.

Appliquer les exigences de preuve et de tests du dépôt aux fonctions exécutables effectivement modifiées. Les preuves CPU historiques ne couvrent pas automatiquement Metal, la nouvelle transaction complète ou la concurrence. Garder leur portée explicite.

### 6.4 Mesure après correction

Ordre conseillé :

1. Baseline même binaire, puis modification MTP-2 isolée.
2. Microbenchmark de formes MB3, puis petit pilote D1/D2 avec la variante retenue.
3. Q04 sans oracle complet dans le chronométrage, à choix/counters identiques.
4. Q04 bloc avec moins de travail CPU avant Hot.
5. Composition des petits candidats compatibles.
6. Reprise Q02 affine/H128 sous profils numériques séparés.

Commencer avec le même prompt court pour attribuer l’effet. Alterner les bras et répéter suffisamment les contrôles pour distinguer la variation observée ; éviter de dérouler D1 puis D2 puis D3 puis D4 une seule fois et de traiter l’ordre comme équilibré. Étendre à quelques prompts représentatifs et à un contexte réellement plus long uniquement pour qualifier les candidats qui restent intéressants.

Conserver les résultats négatifs, avec distinction entre **aucun gain démontré**, **ralentissement reproductible**, **gain local**, **non concluant** et **intégration effective**. Ne pas additionner les pourcentages de microbenchmarks.

---

## 7. Qualité et limites à conserver dans la conclusion

Les 24 séquences chaud/mesuré identiques aux références AR du même profil montrent une cohérence locale de ces trajets. Elles ne prouvent ni la qualité globale de Q04, ni toutes les transitions de rejet/annulation.

Le petit contrôle de 384 tokens sur six domaines rapporte :

- Q04 10 % : PPL agrégée +0,4608 %, pire domaine +3,1913 % ;
- Q04 20 % : PPL agrégée +0,0852 %, pire domaine +5,5298 %.

Ces variations de perplexité ne sont pas des pourcentages de mauvaises réponses. La meilleure moyenne à 20 % ne garantit pas une meilleure qualité générale qu’à 10 %. Les contrôles finaux de profondeur n’ont pas été suivis d’une nouvelle campagne complète de qualité/preuve ; le rapport l’indique. [R1]

L’analogie avec le quasi ×2 de Sushi sur un autre modèle et un M5 Max avec un autre cache ne fournit pas une cible chiffrée pour ce M5. Les sources amont servent ici à identifier **comment** partager le travail, pas à promettre leur gain.

**État de la CI au cours de l’audit :** au SHA figé, Flash profile compatibility et EXL3 conversion contracts ont réussi ; Regression checks a échoué dans le job `desktop-and-protocol`, étape `Run scripts/check-e2e.sh` ; Native Rust checks et Apple compatibility contracts étaient encore en cours à la dernière consultation. Le log de ce job donne **2740 tests passés, un échec, 146 ignorés et 123 sous-tests passés**. L’échec concerne `test_silence_is_bounded_and_reaped`, avec `PermissionError` pendant le nettoyage du faux processus CLI silencieux. Ce relevé ne prouve pas un lien entre l’échec CI et les ralentissements. Reconsulter le head corrigé avant de conclure. [CI1]

---

## 8. Livrable attendu de Codex

Produire un correctif avec un compte rendu lisible comprenant :

1. le défaut d’efficacité traité, son emplacement et la voie réellement exécutée ;
2. le changement exact et les invariants conservés ;
3. les tests ciblés exécutés, les échecs rencontrés et les limites matérielles éventuelles ;
4. les compteurs avant/après et le débit utile sur le même protocole ;
5. le résultat en composition des candidats retenus ;
6. ce qui reste ouvert : vrai B3, chaîne draft, Q04 bloc, affines Q02 ;
7. l’état d’intégration réel du code et les références des preuves.

**Premier objectif concret :** rendre D2 économiquement compétitif avec D1 sur une géométrie adaptée, puis permettre au moteur de choisir sa profondeur. Le pilote actuel établit que son implémentation coûte trop cher sur ce prompt ; il n’établit pas que le MTP profond est condamné, ni qu’un correctif garantit 20 ou 40 tok/s.

---

## Sources figées

Toutes les sources MLXL3 ci-dessous concernent la tête `cf7ba172…`, sauf les deux références explicites du prototype Q02 archivé. Les numéros de lignes sont des repères de revue ; ils peuvent bouger après correction.

[M1]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/mtp_pilot.rs#L63-L215
[M2]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/flash_verify.rs#L29-L50
[M4]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/shaders/flash_projection.metal#L41-L52
[M5]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/projection.rs#L766-L865
[M6]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/shaders/_qmv_tile_kernel.metal#L100-L287
[M7]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/shaders/flash_gdn.metal#L176-L203
[M8]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/view.rs#L103-L126
[M9]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/flash.rs#L473-L529
[M10]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/mtp.rs#L274-L287

[Q1]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/route_swap_trace.rs#L245-L283
[Q2]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/moe_block.rs#L103-L135
[Q3]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/moe.rs#L463-L552
[Q4]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/route_swap_trace.rs#L285-L333
[Q5]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/route_swap_trace.rs#L43-L91
[Q6]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/route_swap.rs

[R1]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/README.md
[R2]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-campaign.py
[R3]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-speed/B3-D1-A/report.json
[R4]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-speed/B3-D2-A/report.json
[R5]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-speed/B3-D3-A/report.json
[R6]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-speed/B3-D4-A/report.json
[R7]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-speed/B3-D1-C/report.json
[R8]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q04-20261109/q04-mtp-depth-speed/B3-D4-C/report.json
[R9]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/opti.md#L27895-L28141
[R10]: https://github.com/0xZKnw/mlxl3/issues/166

[T1]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/docs/measurements/issue165-q02-20261108/README.md
[T2]: https://github.com/0xZKnw/mlxl3/blob/eeecec02290a35a3667b43745f469c63e33a214c/native/src/gpu/projection.rs#L545-L880
[T3]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/shaders/exl3.metal#L8-L19
[T4]: https://github.com/davidtai/turboexl3/blob/c7ce66c12f6992eabda5db1d11934a6fe5efec33/src/core/kernels/lane/helpers.metal#L125-L178
[T5]: https://github.com/davidtai/turboexl3/blob/c7ce66c12f6992eabda5db1d11934a6fe5efec33/src/core/kernels/dense/dfinish_v.metal#L9-L19
[T6]: https://github.com/davidtai/turboexl3/blob/c7ce66c12f6992eabda5db1d11934a6fe5efec33/src/core/kernels/decode/fin3.metal#L11-L39
[I1]: https://github.com/0xZKnw/mlxl3/issues/165

[S1]: https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/mtp_round.zig#L335-L455
[S2]: https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/qwen4/qwen4_forward.zig#L1204-L1309
[S3]: https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/trunk.zig#L3492-L3660
[S4]: https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/expert_stream.zig#L523-L572
[CI1]: https://github.com/0xZKnw/mlxl3/actions/runs/42624522473

[M11]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/flash.rs#L88-L120
[M12]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/projection.rs#L2048-L2076
[M13]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/hc.rs#L850-L868
[M14]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/mtp.rs#L7-L14
[M15]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/mtp.rs#L546-L560
[M16]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/moe_block.rs#L205-L401
[Q7]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/gpu/route_swap_trace.rs#L462-L595
[Q8]: https://github.com/0xZKnw/mlxl3/blob/cf7ba1724bd3a8e925d4206ff04518bbc5294c0f/native/src/flash_support.rs#L1360-L1373
[T7]: https://github.com/0xZKnw/mlxl3/blob/eeecec02290a35a3667b43745f469c63e33a214c/native/src/resident_scratch_policy.rs#L11-L21

