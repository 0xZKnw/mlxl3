# MLXL3 — priorité SSD, puis revue complète

**Revue du 2 novembre 2026 — M5 24 Go, FlashNext EXL3, objectif 15 tok/s.**

## Oui : concentrer maintenant les essais sur le chemin SSD est justifié

Pas pour « augmenter les Go/s » sans contexte, mais pour **réduire le temps pendant lequel le GPU attend les experts nécessaires**. P09 trouve, en DFlash, **13,427 ms/token tracé** à l'intersection lectures POSIX / attente hôte / CPU bloqué / GPU inactif. En AR : 16,870 ms ; MTP : 15,312 ms. C'est un signal d'attente exposée assez important pour prioriser ce chantier.

**Ce n'est pas 13,4 ms de NAND prouvée.** `preadv` peut dépendre du cache de fichiers, de la VM/compression, du scheduling et du stockage. Les octets lus par le programme ne sont pas les octets physiques NVMe. Le profilage publié n'isole pas tout cela. Et il s'agit d'une trace ralentie : **on ne soustrait pas ses 13,427 ms au temps release**.

L'objectif release reste **12,463→15 tok/s**, soit **13,57 ms/token à économiser** sur MAIN, et 17,23 ms sur HOLD. Le SSD paraît donc un axe sérieux, mais cette proximité de nombres issus de mesures différentes ne prouve pas qu'il suffit à atteindre 15.

**Ma priorité mise à jour : SSD d'abord ; callbacks et kernels en annexe**, comme candidats complémentaires si le temps reste exposé. Source : [dernière issue #65](https://github.com/0xZKnw/mlxl3/issues/65). La revue porte sur main **d5a776e53425bb4d0ba61c0774c5399ae7987c06** ; les changements locaux visibles le 1er novembre peuvent être plus avancés.

## A. Ce que fait réellement le chemin actuel

1. Le GPU calcule les routes et le Hot ; le CPU récupère les IDs.
2. Le cache réserve les slots nécessaires. Les demandes Cold sont **dédupliquées par slot** : un expert utilisé par les deux tokens de B2 n'est pas automatiquement lu deux fois. [Construction des jobs](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L1202-L1209).
3. ColdReady lance les workers puis attend **deux jobs uniques terminés**, pas deux occurrences calculées ni une quantité donnée de travail GPU. Il construit W1 avec toutes les occurrences associées aux slots prêts, lance W1, construit W2 paresseusement, puis joint les lecteurs. [Chemin complet](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L450-L554), [partition des occurrences](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/cold_ready.rs#L1-L17).
4. Le scheduler async publié attribue les jobs par stride statique entre quatre workers. Un worker qui finit tôt ne prend pas le prochain job d'un worker lent. [Boucle](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/ffi/disk_read_batch.h#L94-L121).
5. Dans main, DynamicReaders désactive encore ColdReady. Le correctif local « Dynamic + Cold » mentionné dans les captures doit être identifié et testé comme **baseline actuelle**, pas proposé comme si personne n'y travaillait. [Garde](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L175-L183).

## B. Les expériences SSD dans l'ordre

### 1 — Attribuer la queue d'attente résiduelle, après Dynamic + Cold

**Priorité maximale, pas un nouveau grand profilage général.** Sur le vrai binaire local, capturer pour chaque batch : nombre de jobs uniques et occurrences, worker, début/fin de chaque lecture, ready2, début/fin construction W1, soumission W1, fin I/O, début GPU Cold, fin du calcul. Conserver un identifiant batch/job/couche/token commun et des horloges qualifiées.

Distinguer :
- **déséquilibre workers** : certains workers libres alors qu'un autre a encore plusieurs jobs ;
- **un seul retardataire final** : tous les autres jobs finis, dernière lecture bloque W2 ;
- **retard de publication/soumission** : données prêtes mais CPU/encoder pas encore prêt ;
- **W1 trop petit** : il finit avant le dernier I/O ;
- **saturation/VM** : tous les workers avancent lentement ensemble, compression/swap/latences changent.

Un pool dynamique traite surtout le premier cas, pas nécessairement le dernier. Plus de workers peut empirer contention/VM ; le code limite actuellement à quatre et cela fait partie de la configuration qualifiée. Mesurer avant d'élargir.

**Critère de réussite :** baisse de l'union I/O-exposée et du temps release, pas seulement un meilleur débit de `preadv`. Si les lectures sont déjà masquées, passer au coût de soumission au lieu d'optimiser le SSD hors chemin critique.

### 2 — Nouvelle piste ciblée : priorité aux experts qui rendent W1 utile en B2

**Hypothèse source étayée, efficacité inconnue.** L'ordre des jobs vient des occurrences Cold ; la déduplication garde la première occurrence. Le seuil de publication est deux jobs. Or un slot partagé entre les deux lignes B2 débloque **deux occurrences de calcul** pour le même record lu ; un autre n'en débloque qu'une.

Essai minimal : garder scheduler/cache/octets identiques et ne changer que l'ordre de lancement, en privilégiant les jobs à multiplicité B2 élevée. Conserver strictement l'ordre de restauration et de réduction des sorties. Le critère n'est pas le poids du router : tous les experts choisis restent obligatoires. Pas de suppression d'experts ou approximation.

Puis, séparément, comparer le seuil fixe deux jobs à un seuil borné de **travail GPU prêt**. L'idée est de donner à W1 assez de calcul pour masquer la queue I/O, sans attendre trop de données avant de le lancer. Les formes GU/Down et le nombre d'occurrences donnent un premier proxy ; une adaptation sophistiquée n'est pas justifiée au premier essai.

**À falsifier :** même nombre d'octets/jobs, W1 plus utile, queue GPU-inactive raccourcie, coût total W1+W2 non accru. Si les doublons sont rares sur les routes actuelles, ou si la première lecture lente reste dominante, arrêter cette piste. Ne pas transposer une densité mesurée sur un autre prompt/batch. Aucun gain chiffré disponible ; nouveauté limitée à cette politique précise, pas à l'idée générale de prioriser l'I/O.

### 3 — Coalescence uniquement des extents demandés réellement adjacents, surtout en queue

**Piste conditionnelle, priorité derrière 1–2.** Chaque record contient 12 segments pour 1 248 012 octets ; ils sont déjà lus via `preadv` vers les banques, sans imposer un tampon de staging. [Layout](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_support.rs#L36-L55), [lecteur](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/ffi/disk_reader.h#L61-L85).

Avant tout prototype, compter les paires **demandées simultanément**, dans le même fichier et d'offsets exactement adjacents. Les IDs voisins ne suffisent pas : le layout historique est partiellement lexicographique. Pas de relecture de trous ni de préfetch supplémentaire pour créer artificiellement la contiguïté.

Si les extents sont assez fréquents, concaténer leurs vecteurs destination permettrait de réduire le nombre de syscalls sans recopier les poids. Mais l'API actuelle impose un nombre commun de segments par record ; il faut une représentation explicite des groupes, pas détourner silencieusement ce contrat. La limite locale de 64 segments borne à cinq records complets de 12 segments par groupe.

**Risque majeur :** publier seulement à la fin d'un gros groupe retarde W1. Garder les premiers jobs critiques indépendants et tester d'abord une coalescence de queue. Conserver provenance de chaque expert, erreurs/lectures partielles/EOF, leases et publication de tous les slots concernés. La littérature historique mentionne déjà la coalescence ; cette adaptation n'est pas présentée comme une invention ni un gain démontré.

### 4 — Cache : même 3 Gio, mêmes leases, preuve de churn avant nouvelle politique

Le cache actuel est déjà LFU avec **division des fréquences par deux tous les 16 appels de réservation**, puis ancienneté pour départager. Ce n'est pas un LFU figé sans vieillissement. [reserve/victim](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_support.rs#L961-L1050).

Faire d'abord un replay exact des routes **B2 actuelles**, avec préfill/chauffe et état initial établis : même trace doit reproduire chaque slot/hit avant toute variante. Mesurer évictions puis réutilisations rapides, lectures logiques uniques, multiplicité B2 et exposition réelle de ces misses. Une économie de misses n'est pas automatiquement une économie de temps si ce sont des lectures déjà masquées.

Les essais global LFU/CLOCK/ARC et pool partagé ont déjà produit des résultats négatifs ou indécis. **Ne pas remettre ARC ou davantage de RAM comme solution par défaut.** Les politiques offline sans leases en vol ne sont pas qualifiées pour le runtime. Une politique « coût I/O exposé » ne mérite un prototype que si le replay et les traces identifient un biais récurrent exploitable sans ajuster après coup aux deux prompts.

## C. Trois fausses bonnes idées éliminées par l'historique

- **Lire Gate/Up puis Down pour lancer le GPU plus tôt** : déjà essayé dans ISSUE46-A1, numériquement exact, puis MAIN −3,236 % et HOLD −3,516 %, retiré. Ce n'est pas une nouvelle optimisation à relancer inchangée. [Journal](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L17712-L17769).
- **Supprimer les fstat serait le gros gain SSD** : déjà mesuré dans l'ancien P08, doubles fstat 0,126 ms/sortie wall, seulement 0,005 ms hors GPU actif. Les validations/métadonnées ont d'autres coûts, mais ces appels seuls sont faible priorité. [Mesure](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L20837-L20858).
- **Si toutes les lectures ont fini, ne plus soumettre W1** : déjà étudié P2/D. Même sans overlap I/O, W1 peut recouvrir la construction CPU de W2/merge/shared. Il faut comparer le chemin complet, pas supprimer cette soumission sur ce seul argument. [Ablation](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L21813-L21830).

## D. Première action concrète recommandée

**Pas un port de plus tout de suite :** établir si le dernier Dynamic+Cold local réduit bien les retardataires et où subsiste l'attente exposée. Ensuite premier candidat petit et distinct : **ordre des jobs par multiplicité B2**, sans toucher au cache, au nombre de readers, au format ou au calcul. Si le diagnostic ne montre ni doublons utiles ni queue masquable, ne pas l'implémenter ; choisir le mécanisme réellement présent dans les traces.

Shared construit après le départ I/O reste un complément pertinent déjà en cours ; callbacks MLX peuvent ensuite réduire le temps de publication/soumission qui empêche de profiter de données déjà prêtes. **SSD et CPU de soumission ne sont donc pas deux univers indépendants.**

Aucun benchmark ou changement exécuté pour ce rapport. Pas de chiffre de gain inventé. La revue technique générale, les défauts confirmés et les liens détaillés suivent en annexe ; son ordre d'essais général est remplacé ici par la priorité SSD demandée.

---

# Annexe — revue générale et autres mécanismes

# MLXL3 — revue critique vers 15 tok/s

**2 novembre 2026. Revue de code/documentation, sans modification ni benchmark sur le Mac.**

## Verdict utile

**15 tok/s n'est pas exclu, mais ce n'est pas encore démontré.** La meilleure mesure release MAIN publiée dans la dernière issue est **12,463 tok/s en DFlash B2** ; il faut enlever **13,57 ms par token**, soit **16,91 % du temps** (ou gagner 20,36 % de débit). Il ne suffit donc pas d'un petit kernel plus rapide. Sur HOLD, 11,919→15 demande **17,23 ms/token, 20,54 % du temps**.

Les meilleurs paris sont désormais : **coûts de soumission/lifetime MLX**, **fusion ciblée des projections par rôle et par mode**, et **ordonnancement I/O/Shared/Cold réellement recouvrant**. Le router top10 reste une qualification à terminer, pas un nouveau miracle. Les kernels quantifiés existants ne sont pas « mauvais » : plusieurs simplifications exactes déjà essayées accélèrent des micros mais pas le modèle.

Trois corrections concrètes de diagnostic avant de coder :
1. **Le benchmark tuned n'est pas la configuration par défaut du produit.** Les options demandées et celles effectivement actives doivent être affichées séparément.
2. **DynamicReaders=1 désactive ColdReady dans main**, malgré ColdReady=1 demandé. Les captures locales peuvent contenir une correction non publiée : ne pas confondre les versions.
3. **Le timeout I/O de 30 s n'est pas une borne de retour** : le nettoyage peut attendre indéfiniment. C'est un défaut de robustesse, pas une explication des TPS actuels.

## Périmètre et niveau de preuve

- Dernière issue créée vérifiée : [#65, profilage P09](https://github.com/0xZKnw/mlxl3/issues/65), corps intégral et absence de commentaires ; recoupement avec #59, #60, #52 et le journal `opti.md`.
- Source publiée revue : **d5a776e53425bb4d0ba61c0774c5399ae7987c06**, tête de main consultée, avec inspection des PR récentes. Les prototypes du soir du 1er novembre ne sont pas dans cette tête.
- Revue des chemins Rust/FFI/Metal SSD, projections, router, DFlash/MTP, options et MLX upstream.
- Les données brutes P09 sont annoncées **locales/non publiées**. Les chiffres ci-dessous sont ceux du rapport publié, pas une reconstruction indépendante des traces.
- MLX upstream **v0.32.2**, commit **1f8e74e3f12f31365464a6867c6579f0e9b29d85**, consulté pour le mécanisme des callbacks. **Identité avec la bibliothèque installée non établie** : `build.rs` consomme `MLXL3_MLX_ROOT`, vérifie des fichiers et lit les macros de version, mais ne fixe pas leur hash ni les éventuels patches. [Contrat de build](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/build.rs#L77-L112).
- Aucun test GPU/CPU, benchmark, build, correctif, push ou commentaire effectué pour cette revue. Analyse statique ≠ validation numérique/performance.

## 1. Le budget réel : ce qu'on peut et ne peut pas additionner

### Mesures release, comparables entre elles dans P09

| Mode | MAIN tok/s | HOLD tok/s | LONG tok/s |
|---|---:|---:|---:|
| AR | 11,487 | 11,423 | 10,935 |
| MTP | 12,137 | 11,172 | 11,678 |
| DFlash B2 | 12,463 | 11,919 | 12,193 |

Configuration : M5 standard 24 Go, FlashNext EXL3 2,05 bpw, cache experts 3 Gio, 53 slots × 48 couches, quatre readers statiques, Hot/Wide/Unique/overlap/ColdU actifs ; dynamique désactivé ; DFlash B2 = **un token proposé**, adaptive OFF.

**MTP ne gagne pas partout.** Sur LONG, le débit decode de DFlash est meilleur, mais le temps total publié est AR 47,225 s contre DFlash 49,226 s et MTP 50,110 s. Il faut choisir la métrique utile, pas seulement la colonne favorable.

DFlash MAIN, autre mesure release instrumentée par timers internes : draft 3,864 ms/token, verify 74,927, commit 1,582, résiduel 0,012, total 80,385. Ce total ne coïncide pas exactement avec 1000/12,463 : ce sont des agrégations rapportées distinctes, à ne pas fusionner au dixième de milliseconde.

Même supprimer **tout draft + tout commit** laisserait environ 74,94 ms/token, soit **13,34 tok/s**. C'est un plafond irréalisable de cette seule intervention, pas une prévision. Pour 15, il faut surtout réduire le **chemin verify** ou améliorer le nombre de tokens utiles par vérification sans augmenter trop son coût. « Verify » est inclusif, pas 75 ms de GPU arithmétique.

### Fenêtre tracée P09 : séparée des débits release

| ms/token tracé | AR | MTP | DFlash |
|---|---:|---:|---:|
| Durée | 97,861 | 102,622 | 90,544 |
| GPU actif | 59,175 | 66,386 | 58,665 |
| CPU actif, GPU inactif | 19,545 | 17,106 | 14,834 |
| CPU bloqué, GPU inactif | 17,910 | 17,185 | 16,060 |
| Soumission MLX inclusive | 24,897 | 26,915 | 23,336 |
| Soumission MLX ∩ CPU actif ∩ GPU inactif | 12,433 | 9,323 | 7,940 |
| Attente route inclusive | 33,863 | 31,328 | 26,570 |
| GPU actif pendant attente route | 33,767 | 30,732 | 26,464 |

**Les trois premières catégories d'activité partitionnent approximativement le temps ; les autres lignes sont des intersections/inclusions et ne s'ajoutent pas.** Le route wait n'est presque jamais une bulle de 34 ms récupérable : il attend du GPU actif. Le 12,4 ms de soumission exposée AR n'est ni le budget DFlash ni le coût exclusif des callbacks. La trace a un surcoût observateur d'environ 12–22 % et de la dérive ; aucune conversion proportionnelle fixe vers release n'est justifiée.

L'annexe C2 finale donne un gain prefill de 4,91/4,81/2,81 % MAIN/HOLD/LONG, mais decode +1,05/+0,41/−0,74 %. **Pas de gain decode établi, ni de preuve que l'allocateur explique la différence.**

## 2. Constats de code établis et défauts à corriger

### F1 — Options demandées ≠ options actives : important pour la reproductibilité

**Confiance élevée ; importance P1 diagnostic/livraison, pas bug arithmétique.**

Dans main, GPU_HOT, GPU_HOT_WIDE et UNIQUE_COHORT ont pour défaut 0. ColdReady a pour défaut 1, puis est forcé à false si DynamicReaders=1. [Chargement des options](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L150-L183).

Le benchmark distingue explicitement shipped/tuned et injecte Hot/Wide/Unique/multi-token + cache allocateur 64 Mio ; tuned-b2 impose adaptive=0 et width=1. [Profils](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/scripts/benchmark_flash_modes.py#L28-L50). Le bridge Swift hérite de l'environnement, ce qui ne garantit pas ces réglages lorsqu'on lance l'app normalement. [Bridge](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift#L150-L165).

**Correctif proposé :** reçu de configuration effective au chargement, avec requested/effective/reason, mode, largeur réellement vérifiée, cache, slots, nombre/politique readers, hash du binaire/libMLX. Tests déterministes pour Dynamic+Cold et configuration app/CLI. Ne pas activer aveuglément toutes les expériences dans l'app ; qualifier un profil exact. Ce problème de configuration est en partie déjà connu (#52), pas présenté comme une découverte nouvelle.

### F2 — Timeout I/O non borné de bout en bout

**Confiance élevée sur le chemin statique ; P2 robustesse ; occurrence réelle non observée.**

`wait_ready` lève après 30 s. L'erreur remonte par FFI/Rust ; `ReadBatch::drop` libère ensuite le batch C++. Son destructeur appelle `join`, qui fait `dispatch_group_wait(...FOREVER)`. Si un worker est bloqué dans un `preadv` qui ne revient pas, **l'appel peut rester bloqué après le timeout affiché**.

Sources : [timeout et join](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/ffi/disk_read_batch.h#L38-L80), [Drop Rust](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/array.rs#L624-L664), [pont FFI](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/ffi/mlx_bridge.cpp#L400-L430), [lecture](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/ffi/disk_reader.h#L59-L85).

**Ne surtout pas corriger en détruisant les banques pendant que le worker écrit.** Soit documenter le timeout comme détection seulement et garder un drain possédé hors du chemin UI, soit concevoir une isolation/annulation avec propriété de mémoire et ressources sûre. Test : worker injecté bloqué, deadline externe du processus de test, vérifier retour/annulation, durée et absence d'accès après libération. Aucun gain TPS attendu.

### F3 — Petite allocation répétée évitable : indices de réplication des routes

**Confiance élevée sur la répétition, faible sur gain mesurable ; P3.**

`routed_input` recrée un Vec puis un Array UInt32 `[0×10,1×10,…]` à chaque appel. Ce contenu dépend seulement du nombre de lignes, contrairement au slot-map qui varie avec les évictions. [Code](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L783-L805).

Candidat simple : cache **borné** d'indices immuables par nombre de lignes dans le contexte runtime, sans global cross-stream hasardeux. Compter d'abord les allocations/FFI réellement économisées. Ne pas confondre suppression du Vec avec suppression du `take` GPU. Le slot-map de 512 experts **ne doit pas** être mis en cache comme constant : il dépend du cache résident. Validation lignes 1/2/4/8, même contenu, absence de croissance non bornée, mêmes sorties. Plutôt un nettoyage composable qu'un chantier vers 15 à lui seul.

## 3. Priorité haute : batching des callbacks/lifetimes MLX

**Mécanisme source confirmé upstream ; bénéfice et équivalence locale non mesurés.** C'est une piste réellement différente du prepared launcher FFI déjà essayé.

Dans MLX 0.32.2, chaque primitive GPU construit un `unordered_set<shared_ptr<Data>>`, y place les entrées/siblings, retire une sortie éventuellement donnée, puis le plus souvent ajoute un completion handler capturant ce set. [eval.cpp](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/eval.cpp#L29-L68).

Avec beaucoup de petites primitives, le nombre de sets/nœuds/callbacks peut coûter du CPU sans changer les calculs. Mais les buffers Metal sont créés avec références non retenues ; ces captures participent à leur sécurité de durée de vie. `end_encoding` maintient aussi temporaires, fences et suivi des sorties : ce n'est pas une liste de callbacks supprimables arbitrairement. [Gestion encoder](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/device.cpp#L323-L350), [fin d'encodage/commit](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/device.cpp#L436-L560).

### Deux expériences isolées, pas un grand patch mélangé

**A. Regrouper les handlers seulement.** Accumuler au niveau du command buffer les conteneurs de rétention déjà construits, puis une libération groupée à completion. Garder les mêmes objets possédés, événements, notifications et points de commit. Ce premier essai réduit les handlers, pas tous les sets.

**B. Après A, remplacer les allocations de conteneurs.** Représentation compacte/stockage réservé, puis éventuellement déduplication. Cette étape modifie davantage les références et le moment des destructions : l'évaluer séparément.

Le test de donation MLX dépend de `use_count()==1`. [array.h](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/array.h#L289-L296). **Je n'ai pas démontré un changement de donation atteignable dans ce modèle** ; c'est une propriété à mesurer/protéger, pas un bug attribué. Il faut également préserver le retrait des sorties données, les erreurs GPU et les événements nécessaires à d'autres streams.

**Mesures falsifiables :** handlers/command-buffer, allocations de sets/nœuds, temps CPU exclusif allocation/destruction/encoder, CPU actif pendant GPU inactif, buffers retenus/peak mémoire, donations, nombre et tailles des command buffers. Puis débit release ABBA MAIN/HOLD/LONG pour AR/MTP/DF. Refuser une variante qui réduit un compteur mais augmente les copies, la pression mémoire ou le chemin critique. Le budget de 7,940 ms/token de soumission exposée DFlash tracée englobe bien plus que cette piste ; aucun gain de 8 ou 12 ms ne lui est attribué.

## 4. Projections : faire la bonne fusion au bon endroit

La fusion complète aperçue dans les captures locales a un pilote AR prometteur (~5,8 %), mais aussi une régression MTP (~−7,8 %) et un premier DFlash positif sans contrôle retour complet disponible. **Ces captures ne constituent pas une validation de la tête GitHub.**

Le dispatch existant d'épilogue est déjà spécialisé : input6144/output2560, k4 Mul1, splits2, rows1..2 et autres gardes. [Sélection](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/linear.rs#L382-L410). Une garde de forme ne distingue pas la couche cible et une projection de même forme dans la tête MTP.

**Expérience utile : matrice rôle × mode.** Baseline ; fusion sur les 48 couches cibles seulement ; tête MTP seulement ; les deux. Mesurer AR/MTP/DF, rows1/2, acceptation et état du cache. C'est le moyen de départager « fusion B2 intrinsèquement mauvaise » et « mauvais rôle ajouté au chemin MTP ». Ne pas activer globalement une forme parce que l'AR gagne. La régression et sa cause restent historiques/non établies par ce checkout.

### Variante kernel, conditionnelle : retuiler l'épilogue exact

L'épilogue S2 actuellement publié utilise **8 threads par threadgroup**. [Lancement](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/dense_epilogue.rs#L80-L92). Il traite des blocs H128 avec stockage half et barrières. [Shader](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/shaders/flash_dense_epilogue.metal#L1-L95).

Essayer un empaquetage de plusieurs blocs dans un groupe plus large **en préservant exactement l'arbre d'opérations et les arrondis**, seulement si le profil courant montre cet épilogue sur le chemin critique. Mesurer registres/spills/barrières/durée réelle ; 8 threads ne prouve pas à lui seul une occupation matérielle de 25 %. Ne pas ressusciter F02 inchangé : ses résultats antérieurs étaient mixtes. Cette géométrie est une hypothèse à tester, pas une optimisation établie ni forcément applicable au dernier prototype local.

## 5. I/O : overlap utile plutôt que « davantage de readers »

Le chemin ColdReady lance le batch puis attend deux jobs ; le shared expert est construit beaucoup plus tard, après le retour de la phase de lecture/calcul routé. [Début batch/attente](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L449-L458), [fin et shared](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L1345-L1423).

Piste déjà visible dans les travaux locaux : construire Shared **après** le départ SSD, pendant le travail OS, plutôt qu'avant ce départ ou après tout le drain. Mais construire un graphe n'est pas encore le soumettre : il faut distinguer build CPU, encode, exécution GPU et disponibilité du résultat. Faire avancer Shared ne doit pas retarder le premier Cold disponible ni augmenter les ressources GPU en concurrence au mauvais endroit.

**Test :** chronologie startSSD → ready2 → submitCold1 → Shared → readfinish, union des périodes GPU/CPU/I/O sur la même fenêtre ; puis release. Le gain maximal n'est pas la somme de toutes les durées déplacées. Les captures évoquent 0,76–1,13 ms de build CPU exposé, pas une promesse de gain GPU équivalent.

Les workers async publiés utilisent un stride statique. [Boucle](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/ffi/disk_read_batch.h#L94-L121). Une file dynamique **compatible avec ColdReady** exige publication des jobs, erreurs et ownership corrects ; changer seulement un flag main coupe ColdReady. Qualifier ensemble, pas additionner le gain individuel des deux.

Une troisième vague Cold avait une éligibilité faible (~16 %) et un plafond tracé annoncé ≤1,63 ms/token avant surcharge ; pas le premier chantier. Ce plafond ne s'ajoute pas à l'overlap Shared sur les mêmes intervalles.

## 6. Router top10 : reprendre le bon dossier, pas réinventer

Le fast path publié cible **256 experts/top8/F16**, pas Flash512/top10. Le fallback de Flash fait un classement toutes-paires : chaque thread compare sa clé aux 512 clés, soit 262144 comparaisons par ligne avant sélection. [Dispatch et boucle](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/router.rs#L37-L109).

Un candidat **SIMD01** existe déjà dans l'historique : top10 local par sous-groupe puis réduction des candidats. Son ancienne mesure 5 Gio était instable/non concluante ; la reprise sur 3 Gio a été préparée mais pas qualifiée physiquement jusqu'au bout. [Journal](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L19168-L19195).

**À faire :** reprendre ce candidat, mesurer d'abord le coût GPU exclusif du router ; respecter ordre des égalités, NaN, ±0, ordre de sommation/normalisation FP16/F32 et rows1..8. Ne pas utiliser routewait33ms comme budget du topk. Ce chantier peut réduire une vraie inefficacité algorithmique, mais les dépendances précédentes et le petit volume peuvent limiter le gain global.

## 7. Ce que je ne recommande pas de refaire à l'identique

- **Masked Hot H128/activation** : l'idée « ne pas calculer les branches Cold masquées » paraît neuve en lisant main, mais **82K1PM l'a déjà essayée**, avec nombreux tests exacts puis DFlash sans gain : −1,28/−0,06 % et −1,19/−1,63 % dans l'ordre inverse. Rejet documenté, pas une piste à revendre. [Historique](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L17570-L17615).
- **Simple gather_scales×input fusion POSTPROFILE05** : correct numériquement mais résultats réels mixtes ; ne pas réétiqueter « nouvelle fusion » sans travail matériel supplémentaire supprimé.
- **mx.compile autour de Hadamard/scales** : RUST-PERF04 corrigé numériquement, finalement −4,49 % decode et retiré. Le premier problème shapeless avait été corrigé ; ce n'est pas la cause finale du rejet. [Historique](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L6016-L6064).
- **T17 sigmoid/GDN epilogue** : attention à l'équivalence de l'exp et aux arrondis ; exactitude obtenue, bénéfice final non confirmé, garder OFF sans nouvelle justification.
- **Cache experts pooled ARC04** : les mêmes banques ne signifient pas les mêmes lectures ; historique défavorable. Pas « davantage de cache » sans garde mémoire sur 24 Go.
- **Supprimer la soumission Hot initiale ou les drains** : peut casser l'overlap ou la sécurité des banques/états. [Drain modèle](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/qwen4.rs#L1619-L1637).
- **Éliminer le dernier append du draft** : vrai travail parfois inutile si génération terminée, déjà identifié ; contrat de reprise à préserver. Effet de fin de requête, trop faible seul pour 15. [Append](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_draft_runtime.rs#L150-L170).

## 8. Plan d'exécution recommandé et critères d'arrêt

| Ordre | Expérience | Résultat qui justifie de continuer | Résultat qui l'arrête |
|---|---|---|---|
| 0 | Figer le vrai binaire local + diff + libMLX + config effective | Baseline répétable, identification des fusions déjà présentes | Identités/config divergentes : aucune conclusion comparative |
| 1 | Callback batching A, puis B séparé | Baisse du coût CPU exposé, même calcul/ownership, gain release sans régression mémoire | Handler count baisse mais pas temps utile, copies/retention augmentent |
| 2 | Fusion rôle cible/tête × AR/MTP/DF | Attribution de la régression, bénéfice stable du rôle ciblé | Gain AR seulement et perte MTP/DF non maîtrisée |
| 3 | Dynamic compatible ColdReady + position Shared | Bulle critique réellement raccourcie, pas seulement déplacée | W1 retardé, plus de contention, gain traces sans gain release |
| 4 | Finir SIMD01 sur la baseline actuelle | Topk exclusif baisse et bénéfice modèle ou composition démontrée | Temps router négligeable, coût de coordination annule le bénéfice |
| 5 | Indices immuables + géométrie épilogue séparément | Moins d'allocations ou kernel plus court, exact et borné | Seulement un micro favorable sans apport au modèle |

Pour chaque candidat : préenregistrement bref dans le journal, une seule différence au départ, baseline/candidat dans les deux ordres, MAIN/HOLD et LONG, mêmes checkpoints/prompt/tokens/largeurs/cache/thermique ; numérateur de débit = tokens émis utiles, pas tokens vérifiés. Compter acceptations et coût de chaque itération pour MTP/DF. Qualité : logits/états imposés, routes/masks/duplicates/NaN/±0 selon le contrat concerné, refus/rollback/reprise pour spéculation, erreurs/synchronisation pour MLX. Le test de sortie finale seule n'est pas suffisant.

Ne pas faire une campagne infinie pour « prouver » un gain de bruit : si les intervalles et contrôles montrent une variation comparable à l'effet, conserver le statut indécis et changer de mécanisme. Une optimisation simple, exacte et peu risquée peut être gardée pour composition si cela est documenté, mais ne pas annoncer son pourcentage comme acquis.

## Conclusion

**Il reste des problèmes concrets à traiter, mais pas une réserve prouvée de 20 % dans un unique kernel.** Le meilleur levier nouveau à vérifier est le coût de rétention/callbacks de MLX ; le meilleur levier déjà amorcé est la fusion de projections **sélective**, avec l'ablation de la tête MTP. Le profilage montre aussi de l'attente I/O exposée, mais les gains d'overlap partagent des intervalles et ne s'additionnent pas librement.

Une composition qui enlève réellement 13,57 ms/token à la baseline release MAIN atteindrait 15 ; le code lu ne permet pas d'affirmer qu'elle existe déjà. Priorité à établir les gains locaux du 1er novembre et leur identité, puis mesurer les trois mécanismes distincts ci-dessus — pas à relancer les optimisations exactes déjà rejetées.

