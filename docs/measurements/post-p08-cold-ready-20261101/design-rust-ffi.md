# COLD-READY-01 — contrat Rust/FFI et port minimal proposé

1 novembre 2026. **Conception seulement**, après lecture du préenregistrement,
de `ssd_moe.rs`, `flash_support.rs`, `array.rs`, `ssd_prefetch.rs` et du lecteur
C++. Aucun runtime édité, build, test, GPU ou nouveau benchmark lancé. Le crible
P08 précède le choix de taille ; les 15,53 ms de premier retour disponible ne
sont ni une cohorte prête ni un gain promis.

## 1. Contrat actuel à conserver

`forward_rows_prepared` soumet déjà le Hot, attend les routes, réserve les slots,
puis construit les jobs uniques par slot. `fill_many`
([ssd_moe.rs:295](../../../native/src/ssd_moe.rs#L295)) prépare les 12 segments
de chaque expert, passe dans `Array::read_records`
([array.rs:805](../../../native/src/array.rs#L805)), puis dans
`mlxl3_arrays_preadv` ([mlx_bridge.cpp:289](../../../native/ffi/mlx_bridge.cpp#L289)).
Les validations Rust/C++ bornent sources et destinations ; le C++ refuse aussi
les chevauchements entre **tous** les records du lot. `read_disk_record`
([disk_reader.h:60](../../../native/ffi/disk_reader.h#L60)) gère EINTR et lectures
partielles jusqu’aux 12 segments, et échoue sur erreur/troncature. Il modifie ses
iovecs : un même record préparé ne doit pas être exécuté deux fois.

`read_disk_records` utilise les lecteurs GCD, attend tous les workers même si un
job échoue, puis propage la première erreur dans l’ordre des jobs. Ce retour
synchrone protège actuellement la durée de vie des pointeurs et fichiers.

**Le cache demande n’a pas d’état Ready distinct.** `ExpertCache::reserve`
([flash_support.rs:907](../../../native/src/flash_support.rs#L907)) inscrit déjà
`keys[slot]=id`. `pending` concerne les prélectures, pas ces demandes.
`ExpertFill` invalide les keys si sa transaction échoue ; `commit` marque
seulement la réussite globale. Il n’attend ni CPU ni GPU. Le commentaire de la
resident map suppose donc que les fills précédents sont entièrement terminés.

Avant les writes, `last.eval()` protège les utilisateurs de la visite précédente.
Le Hot courant peut ensuite lire des slots distincts des writes Cold : ce
recouvrement existe déjà. Après succès, la sortie finale retenue dans `last`
contient les dépendances des sorties Hot/Cold. Sur erreur, le chemin actuel
attend explicitement le Hot ; il n’a pas encore de sorties Cold anticipées.

## 2. Premier port : une seule vague anticipée, puis le reste

Proposition volontairement bornée : **préfixe canonique de la liste Cold**,
une seule cohorte anticipée, puis une cohorte contenant le reste. Taille fixe
issue du crible, sans autotuning. Distinguer deux unités : un job SSD remplit un
slot/expert unique ; la liste Cold contient des **occurrences** de routes.
Un slot prêt peut servir plusieurs occurrences, avec des activations distinctes
entre lignes : jamais réutiliser une sortie sur le seul ID d’expert.

Un préfixe est prêt si chaque occurrence est soit un hit déjà complet, soit
associée à un job entièrement réussi. Il n’est pas prêt au premier segment ou
au premier syscall court. Le préfixe et le suffixe sont concaténés dans l’ordre
Cold initial ; on réutilise ensuite le remapping Hot/Cold et la somme Top10
stock. Un ensemble arbitraire «les premiers finis» ajouterait scatter/remapping
et risque d’ordre : à exclure du premier port.

Séquence proposée :

1. Conserver les deux async existants et toutes les réservations initiales.
   Valider/préparer **tout** le lot, puis lancer les mêmes lecteurs statiques.
2. Attendre une progression CPU, puis examiner le masque de jobs complets.
   Si tout le lot est déjà fini, garder le calcul Cold stock, sans le scinder.
3. Si le préfixe est complet et des lectures restent en vol, calculer ses
   occurrences par le vrai `calculate`/`calculate_slots`, puis `async_eval`.
   Les kernels et leur arithmétique restent ceux du moteur.
4. Joindre le lot entier, vérifier tous les retours, effectuer le commit global
   des poids, calculer le suffixe et fusionner dans l’ordre stock.
5. Installer la sortie finale dans `last` avant de libérer la garde des sorties
   anticipées. Aucun nouveau reserve, prefetch ou resident-map de cette couche
   ne doit observer la transaction intermédiaire.

Limiter d’abord l’activation aux B1/B2 target, lookahead0/prélecture absente,
T18a/GDN/F02/T05 OFF, mêmes 3 Gio et 4 lecteurs. Cas sans partition utile,
all-Hot, zéro/une lecture, grand batch ou configuration non prise en charge :
fallback existant. Cela évite de modifier simultanément les prélectures et
leurs réservations `pending`, ainsi que les fenêtres uniques terminées à part.

## 3. Minimum API asynchrone

Les primitives `Array::prepare_read` / `PreparedRead::run(token)` existent
([array.rs:525](../../../native/src/array.rs#L525)). Elles démontrent le modèle
de durée de vie : File cloné et owners MLX conservés côté inférence, worker OS
uniquement. Cependant, `ReaderPool` publie des prélectures et ne garde que quatre
pending : le réutiliser tel quel pour un lot demande serait incorrect. Créer
quatre nouveaux threads Rust **par couche et token** serait aussi une régression
probable face au pool GCD actuel.

Le plus petit ajout conservant le backend actuel serait un **handle C++ de lot**
et quatre entrées FFI privées, noms proposés :

| API | Contrat |
|---|---|
| `disk_batch_begin(...,workers,out)` | Même validation complète que `read_records`, y compris non-chevauchement inter-records ; tous owners, états, erreurs et contextes de workers alloués avant lancement. |
| `disk_batch_poll(handle,wait_ns,states,bytes,count)` | Attente bornée sur progression, sans busy-spin ; copie un état Pending/Complete/Failed par job et ses octets. Ne touche pas MLX. |
| `disk_batch_finish(handle,...)` | Joint **tous** les workers, y compris après erreur ; restitue l’erreur de plus petit index comme la baseline et les compteurs réels. Idempotent après jointure. |
| `disk_batch_free(handle)` | Filet de sécurité : jointure avant destruction ; libération des owners MLX uniquement sur thread d’inférence. |

Le handle conserve les `mx::array` owners créés sur le thread d’inférence et
les records OS. Les File clonés restent dans le wrapper Rust jusqu’à jointure.
Les callbacks GCD accèdent exclusivement aux pointeurs/iovecs, états et erreurs
du job ; aucun Array, cache Rust, callback Rust ou API MLX dans ces workers.
Conserver l’attribution statique T18a OFF : la nouveauté est la consommation
avant join, pas une nouvelle politique de lecture.

Publication d’un résultat : écrire octets/erreur, puis état terminal **release** ;
poll le lit **acquire** avant de considérer les 12 segments visibles. Une simple
atomique relaxed de progression ne publie pas les buffers. Aucun compteur seul
ne permet de déduire quels slots sont prêts. **Ne lire ni bytes, ni erreur, ni
timings non atomiques d’un job Pending** pendant que son worker les modifie :
retourner une valeur vide jusqu’à son état terminal acquis. Toutes les métadonnées et vectors
restent stables pendant les callbacks ; pas de réallocation après lancement.

Wrapper Rust non-Send/non-Sync, comme PreparedRead, sans `unsafe impl Send`
d’Array. Un emprunt `&mut banks` conservé dans le handle empêcherait les calculs
concurrents dans Rust ; le mécanisme doit posséder les owners et faire respecter
explicitement les leases de **plages disjointes**, pas escamoter ce conflit avec
une durée de vie prétendument sûre. La frontière privée appelle l’unsafe existant
seulement sous ces préconditions validées.

## 4. Erreur, unwind et annulation : risque principal

**`last` unique n’est pas une garde suffisante.** Remplacer le Hot par une seule
sortie Cold indépendante ne prouve pas que tous les utilisateurs des banques
sont terminés. La nouvelle transaction locale doit conserver le Hot et toutes
les sorties GPU anticipées, même si un calcul/soumission ultérieur échoue.

Une garde de transaction doit, sur toute sortie d’erreur ou unwind : joindre
les lecteurs, attendre **toutes** ces sorties GPU sans s’arrêter à la première
erreur, puis seulement autoriser l’invalidation/libération/réutilisation des
slots et owners. `ExpertFill` doit tomber après ce drainage. Préférer une garde
composite avec ordre explicite plutôt que dépendre par accident de l’ordre de
Drop de plusieurs variables. Le chemin succès ne rajoute pas d’attente GPU :
la sortie finale conservée possède les dépendances, comme aujourd’hui.

Annuler signifie arrêter la requête après drainage, pas tuer un thread dans
`preadv` ni libérer un buffer encore écrit. Une attente poll peut expirer sans
libérer les leases ; finish/free attendent les writers. Aucun destructeur ne
peut promettre un délai OS absolu si un syscall reste bloqué ; le superviseur
reste le dernier délai du processus. Le premier prototype peut drainer tous
les jobs même après annulation, comme sur erreur, sans nouvelle API cancel.

Le compteur `read_bytes` et le cache ne doivent annoncer le succès que lorsque
le lot complet réussit. Conserver séparément les octets physiques retournés sur
échec, sans les transformer en hit ou poids publié. Une complétion tardive ne
doit jamais publier un ancien slot après reset/une autre réservation.

## 5. Contrôles à préenregistrer avant le port

- CPU/Kani sur la vraie sélection de préfixe : hits, slots répétés, erreurs,
  sous-ensemble complet, tout fini, nul/une lecture, bornes et mapping canonique.
- C++ lecteur réel avec fichiers jetables : retards déterministes d’un job,
  lectures partielles/EINTR/troncature, erreurs multiples, première erreur
  canonique, free/finish avant libération, états acquire/release et absence
  d’alias entre destinations. Sanitizers/race tests appropriés ; tentative de
  vérification source avec limites GCD/atomiques explicites, sans faux backend MLX.
- GPU composant : empoisonner les banques non prêtes, vérifier sorties finies
  et byte-exactes ; forcer le dernier job lent et prouver une vraie sortie GPU
  de cohorte avant sa fin. Aucun nombre de requêtes de graphe n’est un dispatch.
  Tester erreur après première soumission, reprise et reset sans writers restants.
- Puis états/logits/commits, bridge AR/MTP/DF et pilote borné. Le nombre de vagues
  peut changer le choix de kernel par nombre d’occurrences et la pression de
  buffers ; la somme finale identique ne suffit pas à prouver toutes les valeurs.

Pas de banque d’experts supplémentaire : même payload 3 Gio. Contextes, Files,
états de jobs et sorties temporaires sont néanmoins une mémoire réelle à borner.
Deux vagues limitent ce coût et le surcroît d’encodage ; leur gain net reste
non mesuré jusqu’au crible puis au chemin physique qualifié.
