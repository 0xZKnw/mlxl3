# T18A-01 — audit du raccord Rust/FFI, 31 octobre 2026

Revue de sources uniquement, après préenregistrement T18A-01. Aucun code modifié,
aucune compilation, exécution de test, preuve ou acquisition GPU par cet agent.
Les builds et qualifications restent sous orchestration du coordinateur.

## Conclusion et point à corriger

Aucun défaut de distribution, de copie ou de jointure identifié dans le port
PR61. **Un défaut concret de diagnostic pré-FFI reste ouvert au moment de cette
lecture** : après un lot réussi, `Array::read_records` peut refuser le lot suivant
en Rust (metadata fichier, taille source, banque/destination) avant de rejoindre
`mlxl3_arrays_preadv`. Dans ce cas, son reset C++ ne s'exécute pas et
`SsdMoe::fill_many` attache les timings/politique du lot précédent aux nouveaux
jobs avec `ok:false` (`array.rs:811–841`, `ssd_moe.rs:323–327`). Le profiler est
activé une fois par phase (`qwen4/profile.rs:163–178`), pas une fois par lot.

Le validateur refuse immédiatement `ok:false` (`profile_flash.py:148–149`) :
**ce défaut ne permet pas une fausse validation numérique**, mais attribue des
lectures anciennes à un échec nouveau. Correction minimale suggérée : remettre
le profil à zéro uniquement lorsqu'un span `disk_read_batch` est actif, juste
avant le call Rust, puis vérifier succès Dynamic → refus pré-FFI → timings vides.
Constat transmis au coordinateur ; aucun correctif exécuté ici.

## Raccord vérifié

| Propriété | Constat de source |
| --- | --- |
| Option | `disk_schedule` accepte seulement absent/`0`/`1` ; le `thread_local const requested` est initialisé à la première entrée valide qui atteint le parseur. Après initialisation, un changement d'environnement sur le même thread ne change pas le bras. Un parseur qui lève n'achève pas l'initialisation C++. |
| Petits lots | `effective_disk_schedule` applique exactement requested Dynamic && workers>1 && count>workers. Il est appelé avant le clamp `min(workers,count)` ; les deux évaluations donnent la même politique. Le reader reçoit le résultat explicitement sans deuxième garde. |
| Limites | Rust et bridge refusent count hors1..512 et workers hors1..4 ; le reader valide toutes les destinations avant démarrage. La queue est construite après clamp avec workers<=count. |
| Option invalide | Le parseur lève avant `read_disk_records`, donc avant les écritures `preadv`. L'évaluation MLX préalable des banques reste présente ; ce n'est pas une garantie d'absence de toute opération MLX/GPU. |
| Profil effectif | Le booléen est reset sur enable et à chaque entrée FFI ; fixé à la politique effective immédiatement avant le reader, conservé en cas d'erreur I/O. Aucun cache de la seule valeur demandée. Réserve pré-FFI décrite ci-dessus. |
| Attribution | CAS relaxed PR61 inchangé ; counter borné à count, aucun wrap après épuisement. Le compteur distribue des IDs et ne publie pas les données des records. Static conserve ses bandes et n'exécute pas de CAS. |
| Erreurs et publication | Chaque travailleur capture l'erreur et continue les autres jobs. `dispatch_apply_f` est synchrone ; scan des erreurs dans l'ordre original après la jointure. Les écritures partielles d'un record en erreur restent possibles comme auparavant, mais les slots ne deviennent pas valides. |
| Parents et reprise | `ExpertFill` reste vivant sur le parcours de demande puis `commit` seulement après `fill_many` réussi ; son Drop invalide sinon (`flash_support.rs:860–885`, `ssd_moe.rs:850,982–988`). Le Hot en vol est toujours attendu dans la branche d'erreur. `warm_all` invalide sur échec (`ssd_moe.rs:471–474`). |
| Payload/compteurs | Aucun changement aux jobs, offsets, destinations ni déduplication. `read_bytes` n'est augmenté qu'après succès (`ssd_moe.rs:318–330`). Hits/misses, cache3Gio et EXL3 restent inchangés. Les lecteurs persistants spéculatifs gardent leur chemin propre. |

## Portée des tests nouveaux et des preuves

Le sélecteur physique
`array::tests::disk_schedule_profile_matches_real_batch_assignment`
(`array.rs:1485–1572`) passe par **la vraie** `Array::read_records` avec fichiers
jetables et banques MLX. Ses nombres annoncés sont corrects :

- workers1/2/4 × counts1/4/5/9/17/512 = **18 cas** ; somme548 records par groupe
  de workers, soit **1644 records**, **52608 octets utiles** (32 par record).
- Flag1 : workers1 donne6 Static ; workers2 donne1 Static+5 Dynamic ; workers4
  donne2 Static+4 Dynamic : **9 Dynamic et9 Static**. Flag0 : **18 Static**.
- Chaque sortie `[64]` UInt8 est comparée intégralement à un oracle fichier
  indépendant, avec deux fragments de16 octets et32 canaris. Les timings exigent
  exactement count jobs, index dans l'ordre, worker valide, bytes32, aucun échec,
  intervalles non superposés pour chaque worker. Aucune répartition équilibrée
  artificielle n'est imposée en Dynamic.
- Reset vérifié avant chaque lot réussi. Ce test **ne vérifie pas** le rejet
  d'une option invalide par la FFI : son préambule Rust refuse ce cas avant le
  call. Ajouter un sélecteur distinct lancé en processus option invalide permet
  d'exercer le vrai rejet avant écriture, sans mutation globale de l'environnement.

Les tests C++ reader couvrent les deux politiques, vrais GCD/preadv, sentinelles,
intervalles réels par worker, deux erreurs différentes (jobs0/3), sélection de
l'erreur du plus petit job après drainage et reprise. Leurs **308 cas** incluent
les contrôles staged préexistants. Les logs produits par le coordinateur et lus
ici annoncent `308 disk reader cases passed` et
`4084 concurrent disk queue cases passed; 4096 effective policy cases passed` ;
cette lecture ne remplace pas l'inspection de sa campagne complète.

Le harness C++ appelle la queue réelle et sa garde effective ; domaine<=8jobs,
<=4workers avec ordres symboliques de claims productifs. Il ne prouve ni CAS
concurrent, ni GCD/preadv, ni visibilité Metal. Les assumes Static excluent les
workers déjà épuisés du choix productif ; deux claims d'épuisement sont vérifiés
ensuite pour chacun. Le test concurrent porte les autres interleavings observés,
sans devenir une preuve exhaustive de concurrence. Le vrai parcours Kani MLX
reste soumis aux limites externes connues ; aucun résultat formel nouveau n'est
revendiqué par cette revue.

## Identité lue et état

Sources lues avant correction du constat pré-FFI :

| Fichier | SHA-256 |
| --- | --- |
| `native/src/array.rs` | `acce79abd817f70a9a2cad28265a367f85777dadadf702f267c3718120e3f7cb` |
| `native/src/ssd_moe.rs` | `80cb4c295a49f5e0e99f2f905d2c005e0162d76da51483bfc550b61e6b0f64cf` |
| `native/ffi/disk_work_queue.h` | `319079e04db06d663e93aa72d73983c3a1ff9c930f52587982451e2093dfdc8c` |
| `native/ffi/disk_reader.h` | `232587bc59483be17f9a3e5d53e9c841b1799c87d99e82409368caa48b241ee7` |
| `native/ffi/mlx_bridge.cpp` | `092c42ddbc41350c5f8ae16b0b301902d8a13cc350413102842eddf42579b7fa` |

Diff au snapshot pré-T18 relu : raccord Array ajouté seulement sous cfg(test),
SSD seulement politique du payload test ; CPP seulement option/queue/profil T18.
Les modifications readback antérieures ne sont pas attribuées à T18 et restent
préservées. **Audit source terminé avec un diagnostic à corriger, qualification
globale en cours ; aucun gain de débit ou validation GPU annoncé.**
