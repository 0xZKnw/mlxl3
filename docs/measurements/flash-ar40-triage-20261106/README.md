# AR40 — tri des issues 147 à 152 pour Flash Next AR

Revue du 6 novembre 2026, base `71530f09dfbc2f698b240d6bbfd6ebea09bd39fc`. Aucun code moteur modifié, aucun benchmark GPU lancé. Cette revue choisit les mécanismes à essayer ; elle ne qualifie pas leur correction ni leur débit.

## Références et état réel

- [#147](https://github.com/0xZKnw/mlxl3/issues/147) fournit le profil AR39 : témoins release historiques 9,730 / 9,922 tok/s, diagnostic interne distinct, limites GPU/horloges inchangées.
- [#148](https://github.com/0xZKnw/mlxl3/pull/148) est fusionnée : outillage et instrumentation de test, aucun gain runtime.
- [#149](https://github.com/0xZKnw/mlxl3/issues/149) décrit 14 mécanismes AR ; [#150](https://github.com/0xZKnw/mlxl3/issues/150) en décrit 32, avec recouvrements et replays historiques.
- [#151](https://github.com/0xZKnw/mlxl3/pull/151) est fusionnée : distribution du moteur natif et intégration LFM3 ; aucun nouveau boost Flash démontré.
- [#152](https://github.com/0xZKnw/mlxl3/issues/152) contient 26 fiches Regolith, dont beaucoup concernent préfill, MTP ou reprise. Leur rang global ne correspond pas au rang pour le decode AR court.
- Les deux commentaires/annexes de #150 et les deux de #152 ont été consultés pour les sources, estimations et contrats. Aucun script venant d'un commentaire n'a été exécuté tel quel.

Les sources des pistes principales (`gdn.rs`, `qsa.rs`, `indirect_hot.rs`, `device_hot_cache.rs`, `device_hot_policy.rs`, `ssd_reads.rs`, shaders Flash/GDN/QSA) n'ont pas changé entre le pin des plans `ed3bfdb8` et la base actuelle. Les nouveaux shaders LFM3 ne sont pas des optimisations Flash. Les projections et sources partagées ont évolué depuis le binaire AR39 : une future campagne doit construire/figer ses propres artefacts et refaire ses témoins.

## Priorités retenues

| Priorité | Fiche et recouvrement | Motif et condition |
|---|---|---|
| 1 | #150 P02 : attendre HC/GDN à la frontière CPU des routes | `Metal::command` attend immédiatement ; `forward_pending`/`forward_after_hc` l'utilisent avant la soumission P. Jusqu'à 36 rendez-vous peuvent être reportés. Commencer avec les mêmes CB, conserver owners et rollback, puis considérer une fusion distincte. Ce n'est ni AR3 ni AR4. |
| 2, plus gros levier I/O documenté | #150 P01 / #149 N09 : capacité T05 qualifiée, d'abord 64 slots | Admission, snapshots, bindings et shader sont tous fixés à 53 ; augmenter simplement le cache coupe T05. Le replay historique annonce ~12 % de misses en moins à 64, ~25–27 % à 80 : à vérifier sur le workload retenu, sans convertir ces pourcentages en tok/s. 64 ajoute environ 0,614 Gio de records seuls. |
| 3 | #150 P03+P07 / #149 N02+N05 : workspace et tables T05 | Sept buffers sont encore demandés par couche (336/token). Commencer par les cinq numériques ; map/arguments demandent des versions immuables. Recycler seulement après H et son consommateur F. Différent de C04 ou du scratch HC précédemment abandonné. |
| 4, travail mort le plus net | #150 P08 : K128 seul dans l'indexeur QSA court | `qk.encode_row` produit 640 sorties avant de tester `count>512`. Les quatre Q sont sans consommateur sur AR39, mais K et son historique restent nécessaires. Retirer 80 % de cette projection seulement, avec adressage global/splits/arrondis conservés et transition sparse à 2052 positions. |
| 5 | #150 P05 : lancer la première lecture pendant la préparation des destinations suivantes | `start_shared_reusing` collecte toutes les destinations avant `start_payloads`. Avancer l'enqueue peut réduire le retard avant les premiers octets ; une erreur tardive d'allocation doit drainer les jobs déjà soumis. |
| 6, après qualification des lifetimes | #150 P04 / #149 N01 : records retirés réutilisables sous T05 | Le réemploi courant exige `device_hot.is_none()`. Ce n'est pas un flag oublié. Ownership et réserve mémoire sont plus difficiles que le workspace ; ne pas enlever ce garde sans nouvel état de retrait. |

Ordre économique proposé : **P02 isolé**, puis **P03 et P08** qualifiés séparément et composés si compatibles ; **P01/64 slots** comme expérience de capacité distincte avec compteur T05 effectif. P05 peut rejoindre le lot I/O. Chaque mesure suivante part de la composition réellement conservée. Ce choix ne suppose aucun seuil minimal de gain et ne révoque pas les candidats simples, exacts et bornés à effet indécis.

## Kernels à garder pour une seconde passe

- #149 N06 déduplique les préparations Q/K partagées entre trois value heads GDN ; le code `hk=h/3` confirme le calcul répété. Mais un prépass ajoute 36 dispatchs et du trafic : ce n'est pas une répétition du B01 scalaires, ni un gain garanti.
- #150 P12 et #152 G17 ciblent géométrie/loads d'état ou épilogue GDN→H128. Ils nécessitent de vérifier barrières, spills et parité du segment complet. Le temps composite HC/GDN n'est pas le temps du seul shader whole-GDN.
- #152 G10/G11 (scratch QSA alterné et prepare en registres) sont des micro-optimisations compatibles à étudier. Leurs sensibilités publiées sont petites ; elles ne justifient pas à elles seules un objectif 11–12 tok/s.
- #152 G16 enlève jusqu'à 34 matérialisations/dispatchs de mix, mais peut recalculer le mix quatre fois et ajouter ~3,82 Mio de trafic logique/token. Priorité inférieure aux attentes et allocations plus directement localisées.

Les grands lots #152 G01/G04/G05 concernent préfill/verify ; G03 et G09 le MTP. G07 vise le contexte sparse absent d'AR39. Ces pistes restent hors de la campagne AR actuelle. Ne pas rouvrir AR3/AR4, P04 historique de #120, L3/L5, SLRU ou les anciens kernels OFF sans nouveau motif.

## Contrôle indépendant des lectures existantes

Le CSV AR39 a le SHA attendu `cd5d9e2bcffcc302cdddf89eeac0d5b44033f2979d83f6371edb9c66721567fe`. Recalcul indépendant par fusion des intervalles, puis balayage des deltas :

- Fenêtre : `[39704398458,54678780500]` ns ; 19 554 appels entièrement contenus, sans erreur ni appel frontalier.
- 127 embeddings exclus ; 19 427 appels experts, 22 964 668 812 octets.
- Union experts : **4 429 262 048 ns**, **29,57893 %** de la fenêtre ; concurrence maximum 4.
- Débit logique rapporté à cette union : **5,18476 Go/s**. Ce n'est ni la NAND, ni une mesure du GPU idle, ni un débit moteur.

Cela confirme la priorité d'avancer les demandes et de réduire les misses plutôt que de multiplier aveuglément les lecteurs. [Reçu](read-window-review.json). Le replay complet de capacité n'a pas été relancé ; ses pourcentages sont attribués à #150 et à sa préhistoire reconstruite.

## Vérifications non exécutées

Aucun prototype, build, test numérique nouveau, Kani ou test GPU : cette livraison est une revue et une réanalyse CPU de données existantes. Aucun nouveau tok/s mesuré. Aucun changement de défaut, publication, merge, release ou remplacement d'app. Journal mis à jour ; aucun essai en cours.
