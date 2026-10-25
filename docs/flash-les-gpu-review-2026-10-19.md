# Revue LES après le prototype GPU 38

Le port GPU 38 reste exact dans les tests numériques, mais sa politique d’admission ne reproduit pas le LRU LES. Deux contre-exemples CPU utilisent la véritable bibliothèque du moteur, sans recopier son algorithme.

| Constat | Preuve | Portée |
|---|---|---|
| Une prélecture neuve non demandée est choisie comme victime dès la prédiction suivante. Sa fréquence LFU reste zéro ; les entrées anciennes ont déjà une fréquence positive. | Cache plein24/89, cohortes10 ; prélecture511 puis510 : même slot réutilisé. Avec une vraie demande511 intercalée,511 reste. | Écart de politique reproduit ; son coût réel n’est pas encore isolé. |
| Une entrée valide peut être évincée alors qu’un slot libre existe. Le critère LFU/âge ne donne pas une priorité absolue aux slots libres. | Seed39, cache89, transition1471 : victime56, libre78. L’observateur indépendant suit Ready/Pending/Libre ; ≤4pending, cohortes10. | Contre-exemple avec publications réussies/échouées et doublons. Aucun cas trouvé en10000transitions de cache24 ; cela n’est pas une preuve d’absence. |
| Publication d’une prélecture : l’âge ne progresse pas dans38. | `publish_prefetch` réutilise `self.clock`, tandis que LES incrémente l’horloge à publication. | Écart de récence ; à corriger avec le choix LRU pour LES. |
| Quatre lecteurs ont été confondus avec un plafond de quatre enregistrements pour le lot. | `budget.min(4)` et un Pending par worker ; LES permet un lot8, consommé en deux vagues par quatre lecteurs. | Limite déjà signalée37, toujours présente38. Le GPU persistant permet de tester8 dans les banques existantes, sans huit buffers CPU. Son bénéfice reste inconnu. |

LES choisit d’abord un slot libre, puis un ancien slot pour la prélecture ; les demandes réelles protègent les prédictions avant le repli LRU. [Admission/publication amont](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L821-L909), [évictions de demande](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L2463-L2558), [lot et lecteurs](https://github.com/OpenBMB/llama.cpp-omni/blob/a64932ec615e1463158129feaf2f0752d0c07a2e/src/llama-moe-slot.cpp#L353-L450). Le pin `main` vérifié reste `a64932ec615e1463158129feaf2f0752d0c07a2e`.

La lecture des banques disponibles avec `array::eval` n’impose pas une barrière GPU globale : cette piste est écartée par le code MLX0.32.2. [Source MLX](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.cpp#L132-L161). Le non-wait des prélectures, l’exclusion Pending et la possession des banques jusqu’au join correspondent au contrat attendu. L’entrée HC et la prédiction de deux lignes MTP restent les hypothèses37 ; ce ne sont pas de nouveaux bugs démontrés par cet audit.

Les tests de logits et états protègent l’exactitude du modèle. Ils ne prouvent ni une bonne durée de rétention des prélectures, ni leur couverture des vraies lectures manquantes. C’est cette partie des tests qui était insuffisante38.

Preuves : [admission](measurements/flash-next-ssd-impl/opt-39-les-gpu-review/admission-reproduction.json), [slots libres](measurements/flash-next-ssd-impl/opt-39-les-gpu-review/free-slot-reproduction.json), [sources et empreintes](measurements/flash-next-ssd-impl/opt-39-les-gpu-review/sources.json). Les deux exécutions terminent avec l’assertion attendue sur la version38 ; ce sont des tests déterministes, pas une preuve formelle ou un résultat de débit.

Le crible38OFF termine avec24réponses128exactes, zéro swapout supplémentaire et +2,84%MAIN/+2,18%HOLD pour le budget4 face au meilleur témoin de chaque prompt. Sa confirmation et le cribleMTP ne sont pas lancés : corriger la politique avant de qualifier le candidat final. Ce gain reste non confirmé et n’est pas poussé.



## Correction 39A et contrôle de la première couche

L’activation du nouveau LRU était incomplète : `protect_predictions` ne touche jamais la couche 0. Un test physique remplit 89 slots avec de vrais records, rend un expert non routé fréquent mais ancien, puis appelle le véritable `SsdMoe::forward` avec un pool attaché et sans prédiction entrante. Avant correction, la sortie égale déjà l’oracle indépendant, mais l’expert ancien reste sous LFU. Le test échoue sur cette assertion précise ; il passe après activation dans le chemin du pool. [Reproduction avant correction](measurements/flash-next-ssd-impl/opt-39a-les-lru/layer0-before/quality.json), [sources et binaire avant correction](measurements/flash-next-ssd-impl/opt-39a-les-lru/layer0-before/identity.json).

Le candidat 39A conserve le LFU sans pool LES, choisit d’abord les slots libres, utilise la récence dans toutes les couches LES et avance l’horloge à publication. Il ne fabrique aucune demande pour favoriser une prélecture. Les quatre reproductions d’admission et les 20 000 transitions déterministes de la revue 39 passent désormais. Les banques, les quatre workers, le plafond de quatre lectures et le prédicteur HC/MTP restent ceux de 38. Le port reste partiel.

Les contrôles passent : 96 tests CPU / 2 ignorés, 116 tests release MLX / 108 ignorés, 100 tests Python ; format, Clippy strict CPU et MLX/chat, build livré. Les 24 tests physiques explicitement sélectionnés passent ensuite, ainsi que le bridge de production : budgets 1/2/3/17, refus, annulation et reprise. Chaque processus termine avec zéro swapout supplémentaire. Ils vérifient notamment les logits de 248 320 valeurs, les états sur 16 positions et le prompt de contrôle, la tête MTP et les commits 1/2. Les tests ignorés de la suite générale ne sont pas tous exécutés. [Qualité physique](measurements/flash-next-ssd-impl/opt-39a-les-lru/quality.json).

Cinq vérifications Kani 0.68 passent sur des entrées symboliques : publication sur 24 slots et horloge u64 entière, admission sur 24 slots avec 0..4 pending, puis victimes, priorité libre et LRU sur 10 slots. Les 4 715 obligations n’ont aucun échec, les 19 couvertures sont satisfaites ; 64 obligations de bibliothèque ou de branches non exécutables sont inatteignables. Les fonctions vérifiées et les harnais restent identiques après le raccordement de la couche 0. Ces vérifications CPU bornées ne prouvent pas le GPU, MLX ou la concurrence. La réservation générale de 2 routes / 10 slots dépasse 120 s ; le harnais MLX déclenche un ICE, CBMC ne parse pas le SDK macOS et CrossHair est absent. Ces chemins restent non vérifiés. [Contrôles](measurements/flash-next-ssd-impl/opt-39a-les-lru/checks.json), [contrôles après raccordement](measurements/flash-next-ssd-impl/opt-39a-les-lru/affected-checks.json), [propriétés inchangées](measurements/flash-next-ssd-impl/opt-39a-les-lru/reused-kani-properties.json).

Le binaire 39A est figé : `52b7b6bb77981ef8f7a76749b0077e422b692ccfdbbcad6d7da7e0312c771aea`, base `fb782d98` modifiée. [Identité complète](measurements/flash-next-ssd-impl/opt-39a-les-lru/identity.json). La parité a été validée avant le crible de débit. Celui-ci a été arrêté à la demande de l’utilisateur ; aucun gain 39A n’est confirmé.


## État publié à la demande de l’utilisateur

Les tests et benchmarks sont arrêtés. Le crible OFF a six bras complets (18 réponses de 128 tokens exactes), un bras interrompu et un témoin non lancé. Tous les enfants sont joints. Les bras complets n’ajoutent aucun swapout.

| Prompt | H initial | LES2 | LES4 | H final |
|---|---:|---:|---:|---:|
| MAIN | 10,26849 | 10,15725 | 10,05825 | 9,88135 |
| HOLD | 9,83247 | 9,62429 | interrompu | non lancé |

Débits en tok/s, médianes de deux générations de 128 tokens après une chauffe ; 127 tokens décodés chronométrés. M5 24 Gio sur secteur, applications ouvertes, cache 5 Gio / 89 slots par couche, allocator 512 Mio, quatre lecteurs, chevauchement activé, MLX 0.32.2, checkpoint EXL3 2,05 bpw `65c89531`. Prompts courts, limite de contexte 4096. Les pics de footprint physique des bras complets vont de 9,29265 à 9,29957 Go décimaux. Le swap déjà alloué n’est pas une mesure des écritures ajoutées.

Les candidats MAIN restent entre les deux témoins ; le critère de gain face au témoin le plus rapide n’est pas atteint. Ce tableau incomplet ne démontre pas une régression générale ou un gain. Le crible MTP, les confirmations et les diagnostics naturels supplémentaires ne sont pas lancés. La qualité MTP a été vérifiée auparavant ; son débit 39A reste non mesuré. Les scripts de diagnostic naturel sont une préparation non exécutée et non vérifiée. Le lot de huit experts n’est pas implémenté. [Résultat interrompu](measurements/flash-next-ssd-impl/opt-39a-les-lru/screen/result.json), [bras complets et interrompu](measurements/flash-next-ssd-impl/opt-39a-les-lru/screen/execution.json), [état explicite](measurements/flash-next-ssd-impl/opt-39a-les-lru/stopped-by-user.json).

Le code est publié pour inspection dans la PR à la demande de l’utilisateur, avec LES désactivé par défaut (`MLXL3_SSD_LOOKAHEAD_DEPTH=0`). Les budgets 2/4 utilisent le cache GPU et le LRU expérimental ; les valeurs 8/10 restent plafonnées à quatre jobs globaux dans ce candidat. Aucun gain LES ni objectif 15 tok/s n’est annoncé. Aucune application installée n’est remplacée.
