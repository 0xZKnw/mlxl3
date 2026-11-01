# T17 — bilan des vérifications exécutées

État documentaire au 31 octobre 2026. Ce bilan porte sur la correction et les
contrôles du prototype local `ISSUE59-T17-GATE-01`, après `FIX-01` et la
requalification `POST-XCODE-01`. Les chronos sont traités séparément : **aucun
gain de débit, de prefill ou de TTFT n'est établi par ce document**. Le flag
`MLXL3_FLASH_GDN_EPILOGUE` reste expérimental, désactivé par défaut. Ce bilan
n'atteste ni publication, ni remplacement de l'application installée.

Le protocole et ses amendements, y compris les échecs, sont conservés dans
[opti.md](../../../opti.md), sections `ISSUE59-T17-GATE-01`, `DIAG-01`,
`DIAG-02`, `FIX-01`, `CAPTURE-02`, `MUTATION-01` et `POST-XCODE-01`.

## Identité et périmètre de production

Le changement testé consomme directement les buffers normalisés **F32** de
chaque ligne et les gates **F16**. Il remplace uniquement
cast F16→F32 / sigmoid F32 / produit F32 / cast F32→F16, sans concaténation F32
ajoutée. La normalisation, la récurrence GDN, la convolution, les états/tapes et
la projection de sortie restent sur leur chemin existant. Domaine du kernel :
`rows=1..8`, normes `[1,1,48,128]` par ligne, gates `[1,rows,48,128]`, sortie
`[rows,6144]` F16. `FIX-01` emploie `metal::precise::exp` et conserve les deux
frontières intermédiaires F32 `volatile`.

Les [identités finales](final-identity.json) figent 183 fichiers et les binaires
suivants. La [seconde identité](post-xcode-identity.json) étend le manifeste à
185 fichiers pour les éléments du protocole après Xcode, sans reconstruction
des binaires de test ou du CLI.

| Artefact | SHA-256 |
| --- | --- |
| CLI normal, `build/issue59-t17-20261029/engine/prototype-engine` | `202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa` |
| Libtest normal, `build/issue59-t17-20261029/libtest-final/prototype-engine` | `a751cf592d9e28622d1c062a9f93213f4c55d8a6dcce0fa8b1c035be16c6b862` |
| Shader `FIX-01` restauré | `1fa5b6ae214104186021ff2aa43a5e7b083e838fb0a410cf886326aaf8f9ca44` |
| Python du lanceur bridge | `30b2a6bf9706029b609327ca5db722cd458239df96efc87e1f8e37beed49db2e` |

Modèle : `Qwen3.8-Flash-Next-exl3-205-65c89531`, EXL3 2,05 bpw ; draft
`Qwen3.8-Flash-Next-DFlash2-a22a1744` pour DFlash2 ; MLX **0.32.2**, Apple M5
24 Gio, macOS 27.2. Les manifestes identifient les sources locales modifiées,
les exécutables, MLX et son metallib, ainsi que les métadonnées des checkpoints.
**Ils ne hachent pas l'intégralité des fichiers de poids.**

Le budget cible reste **3 Gio**, soit 53 slots par couche sur 48 couches,
4 lecteurs, lookahead 0, GPU Hot WideUnique et recouvrement existants. Le
protocole préparé utilise le même CLI final avec flag 0/1 ; T05/T04/T18 sont
désactivés. Les options complètes et variables effectives sont dans les plans
et leurs résultats, plutôt que déduites d'une valeur par défaut.

## Contrôles locaux CPU, MLX et CI

L'[inspection CPU finale](final-cpu-inspection.json) et les deux plans
[CPU complet](full-cpu-plan-result.json) / [MLX final r2](good-final-cpu-r2-plan-result.json)
établissent les résultats suivants. Les commandes ont été exécutées depuis
la racine `/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0`, où se
trouve `Cargo.toml`, séquentiellement, hors campagne GPU.

| Commande | Résultat observé |
| --- | --- |
| `cargo clippy --locked --all-targets --no-default-features -- -D warnings` | Réussite, exit 0 ; [log](t17-full-clippy-cpu.log). |
| `cargo test --locked --release --no-default-features -- --test-threads=1` | **256 passés, 0 échec, 8 ignorés** ; [log](t17-full-cpu-suite.log). |
| `cargo fmt --all -- --check` | Réussite finale r2 ; [log](t17-final-good-fmt-r2.log). |
| `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings` | Réussite finale r2 ; [log](t17-final-clippy-mlx-r2.log). |
| `cargo build --locked --release --features mlx,chat --bin mlxl3-rs` | CLI normal reconstruit, exit 0 ; [log](t17-final-build-good-cli-r2.log). |
| `cargo test --locked --release --features mlx,chat -- --test-threads=1` | **293 passés, 0 échec, 185 ignorés**, toutes cibles agrégées ; [log](t17-final-mlx-suite-r2.log). |
| `cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json` | Identification explicite du libtest final ; [log](t17-final-good-libtest-inventory-r2.log). |

Les nombres CPU sont 170 + 35 + 16 + 35 tests passés ; deux cibles annoncent
explicitement zéro test. Les six résultats MLX sont respectivement
`198/0/175`, `44/0/8`, `16/0/0`, `35/0/0`, `0/0/2`, `0/0/0`
(passés/échecs/ignorés). **Les tests ignorés ne sont pas comptés comme exécutés.**
Les sélections physiques décrites ci-dessous ont été exécutées séparément avec
`--exact --ignored --nocapture --test-threads=1`.

La première passe finale avait été arrêtée par Clippy Rust 1.98 sur cinq usages
de `chunks_exact` de taille constante. La correction des tests vers
`as_chunks::<2/WIDTH>().0.iter()` a conservé cardinalités et oracles ; la passe
r2 ci-dessus a réussi et les six micros ont été rejoués sur le libtest final.
L'[échec initial](good-final-cpu-plan-result.json) et son
[diagnostic](t17-final-clippy-mlx.log) restent conservés.

La CI PR/main existante a été inspectée et adaptée aux contrôles concernés,
avec MLX 0.32.2 épinglé, Clippy/build MLX et sélecteurs CPU. Le parseur YAML Ruby
a accepté le workflow ; `actionlint` était absent. **Aucun run GitHub du code
local non publié n'a été exécuté : la CI distante n'est pas déclarée verte.**

## Vérification source et limites des proveurs

| Tentative réelle | Résultat et portée |
| --- | --- |
| `cargo kani --lib --no-default-features --harness gdn_epilogue_geometry_checks_symbolic_shapes_and_addresses --solver kissat` | **148 obligations sans échec, 9/9 couvertures satisfaites, unwind 17**, assertions d'unwinding conservées. [Plan/résultat](geometry-kani-r2-plan-result.json), [log complet](t17-geometry-kani-r2.log). |
| Même harnais, première borne 7 | Assertion d'unwinding du `memcmp` interne insuffisamment déroulé ; pas une réfutation de la géométrie de production. Résultat non vérifié à cette borne, conservé dans le [premier plan](geometry-kani-plan-result.json) et le [log](t17-geometry-kani.log). Le passage à 17 n'a pas réduit le domaine ni changé la production. |
| `cargo kani --lib --tests --features mlx,chat --harness actual_gdn_epilogue_smoke_entry` | **Non vérifié** : ICE `kani-compiler/src/intrinsics.rs:243:17`, exit 1 avant obligations ; [résultat](actual-mlx-kani-plan-result.json), [log](t17-actual-mlx-kani.log). |
| `cargo kani --lib --tests --features mlx,chat --harness actual_gdn_capture_size_filesystem_entry` | **Non vérifié** : même ICE avant obligations, exit 1 ; [résultat](mutation-01-cpu-plan-result.json), [log](t17-cap02-kani-fs.log). |
| CBMC 6.11 sur le vrai fichier `native/shaders/flash_gdn_epilogue.metal` | **Non vérifié** : source MSL non prise en charge, « Failed to figure out type of file », exit 1 avant obligations ; [commande/résultat](cbmc-metal-result.json), [log](cbmc-metal.log). Cette tentative portait sur le shader antérieur à `FIX-01` et n'établit aucune propriété de l'un ou l'autre shader. |

Le harnais Kani de géométrie appelle la vraie fonction `geometry`. Il explore
un nombre de lignes `usize` symbolique, des dimensions `i32` symboliques et
des rangs 0 à 5, puis un index `u32` symbolique dans toute sortie acceptée. Il
vérifie les formes exactes admises, les lignes 1..8, la cardinalité maximale
49 152, la décomposition ligne/colonne et les bornes d'adressage F32/F16. Les
couvertures incluent notamment B1/B2/B4/B8, le dernier élément de B8, zéro,
`usize::MAX`, un rang 5 et une dimension négative. Il ne démontre pas un
contrat pour des slices de rang supérieur à 5.

Il s'agit d'une vérification bornée de cette géométrie CPU, **pas d'une preuve
de l'exponentielle Metal, de MLX/FFI, du filesystem, de l'allocateur, de la
concurrence ou de la génération complète**. Le statut de superviseur
`completed_expected_outcome` des essais ICE signifie seulement que l'échec
attendu a été correctement collecté, pas qu'une preuve a réussi.

## Défaut numérique reproduit, diagnostic et contrôle négatif

Le premier smoke a réussi, puis le premier exhaustif a réellement échoué sur
le libtest `081d1c847136947fd15c0c4df94e318752608d8fcf51d86f56733cd6dbebee75`,
exit 101. **Les 352 cas n'avaient pas tous été terminés** : ce premier essai ne
valide pas la matrice. Voir [résultat initial](micro-gpu-plan-result.json) et
[log de divergence](t17-micro-exhaustive.log).

`DIAG-01` a reproduit le défaut sans modifier la production : norme F32 bits
`3f801001`, gate F16 `c8dc`, sortie `03f1` au lieu de `03f2`. La reproduction
attendait exit 101 ; elle n'est pas une parité du candidat. Voir
[inspection](diag-01-inspection.json) et [log](t17-diag-01-reproduce-exhaustive.log).

`DIAG-02` a isolé l'effet sur une fixture uniforme de 6 144 valeurs et cinq
variantes instrumentées. L'original donne probabilité F32 `387c4074`, produit
`387c5ffe`, sortie F16 `03f1`. MLX stock donne `387c4076`, `387c6000`, `03f2` :
le produit stock est exactement au milieu de deux F16 et son arrondi au pair
est `03f2`. `precise::exp` rétablit ces trois bits stock sur les 6 144 valeurs.
Supprimer `volatile` seul ne corrige rien ; fournir la probabilité stock en
entrée rétablit le produit/cast. Le capteur à trois sorties a été comparé à
l'API réelle. Voir [diagnostic initial](t17-diag-02-boundaries.log) et
[diagnostic final](t17-final-diagnostic.log). Le test final conserve
délibérément les variantes historiques divergentes ; sa réussite ne signifie
pas que ces cinq variantes ont toutes la parité.

Le contrôle `MUTATION-01` a ensuite omis temporairement le facteur gate dans le
**vrai shader**. Le smoke a échoué sur sa première valeur : norme 1, gate 0,
probabilité 0,5 ; mutant `3c00` (1), stock `3800` (0,5). Le résultat est bien
**0 passé, 1 échec, 0 ignoré**, exit 101 attendu, et non un échec de compilation
ou un filtre vide. Libtest mutant :
`4ac5bdc7b908a625483fd98d1bcc1b341170037adb120235e265c93b466ad00d`.
Le shader `FIX-01` a été restauré dans un `finally`, puis son hash `1fa5…`
contrôlé avant les qualifications normales. Preuves :
[log du mutant](t17-mutation01-omit-gate-smoke.log),
[résultat supervisé](mutation-01-gpu-plan-result.json),
[cycle de vie/restauration](mutation-01-lifecycle.json).
**Le contrôle négatif est concluant pour la sensibilité de l'oracle ; il ne
constitue pas une parité du mutant.**

## Qualité du composant sur le binaire final

Les sept sélections du [plan final léger](final-small-gpu-plan-result.json)
ont chacune réellement terminé avec **1 passé, 0 échec, 0 ignoré**. Les oracles
vérifient formes, dtypes, cardinalités non nulles et finitude des fixtures
finies avant comparaison complète ; aucune réussite ne repose sur deux
sorties vides. Les compteurs de requêtes du graphe MLX ne sont pas présentés
comme des nombres de dispatchs physiques.

| Sélection | Couverture observée et preuve |
| --- | --- |
| Smoke B1/B2 | **4 cas, 36 864 sorties F16** ; normes F32 non représentables en F16, gates non triviales, ancres CPU indépendantes `z=±0 → sigmoid=0,5`, parité stock bit à bit et sources immuables. [Log](t17-final-smoke.log). |
| Exhaustif des gates finies | **352 cas**, **63 488 patterns F16 finis × 16 ancres F32 × 4 formes B1/B2/B4/B8 = 4 063 232 paires** ; **5 013 504 sorties avec padding**, toutes bit à bit stock. Zéros signés, subnormaux, frontières normales, petites valeurs F32 et arrondis inclus. [Log](t17-final-exhaustive.log). |
| Refus de métadonnées | **42 cas** de formes/rangs/dtypes/comptes invalides ; refus avant requête de kernel, **0 requête, 0 readback**, aucune mutation des sources. [Log](t17-final-invalid.log). |
| Strides et alias | **64 cas = 8 nombres de lignes × 4 masques de strides × 2 dispositions d'alias** ; entrées non contiguës acceptées via la matérialisation MLX, padding empoisonné, sources immuables, parité complète. [Log](t17-final-strides-aliases.log). |
| Infinis et produits non finis | **4 cas spéciaux à sortie finie + 4 cas de produits non finis**, B1/B2/B4/B8. Bits exacts pour les valeurs non NaN ; **classification NaN seulement**, sans contrat de payload NaN. [Log](t17-final-inf-nonfinite.log). |
| Diagnostic de frontière conservé | 6 144 éléments par variante ; l'API corrigée et le capteur `precise_exp` doivent correspondre au stock, les deux variantes historiques erronées doivent reproduire leur divergence. [Log](t17-final-diagnostic.log). |
| Capture B1/B2 | Quatre packages stock/fusion, **18 432 valeurs F16 = 36 864 octets** complets, finis, bit à bit ; comparaison et immutabilité vérifiées hors capture. [Log](t17-final-capture-02.log). |

L'exploration des gates F16 est exhaustive dans ce domaine fini, **pas pour
toutes les normes F32**, dont seules les 16 ancres sont croisées. Pour les
produits minuscules et le FTZ, l'oracle physique stock est la référence ; ce
test ne prétend pas imposer à MLX un résultat CPU IEEE strict différent.

## Capture, taille des packages et commandes physiques

Le premier essai de capture a échoué dans le nouveau contrôleur de taille
après création des quatre packages : il rejetait des liens internes légitimes
d'Apple. Ce n'était pas un échec numérique démontré du kernel. Les preuves
[du premier essai](fix-01-capture-plan-result.json) et
[des liens](fix-01-capture-links.json) sont conservées. Le helper corrigé
accepte seulement les liens résolus vers des fichiers réguliers dans la
racine canonique, compte leur taille `lstat` sans recompter la cible, et
rejette échappement, dangling, répertoire, boucle et types non pris en charge.

Le test CPU
`gdn_epilogue::capture_tests::capture_size_validates_internal_links_rejections_and_sparse_limits`
a passé **18 cas répartis en 4 groupes**, y compris fichiers sparse aux
frontières de 1 Gio, packages vides/manquants et refus de liens. Voir
[log CPU](t17-cap02-filesystem-cpu.log) et
[plan](mutation-01-cpu-plan-result.json). Il compte les tailles **logiques**
des fichiers et des liens, pas les blocs disque, suppose le package stable
après `finish` et vérifie le plafond **après écriture**, sans limiter activement
la capture en cours.

La matrice finale occupe **730 655 969 octets < 1 Gio**. En comptant aussi
l'essai initial échoué et conservé, le cumul est **1 461 346 528 octets** :
**le budget initial de 1 Gio cumulé a été dépassé**. La réussite du contrôle
par matrice ne corrige pas cette déviation ; aucune capture supplémentaire
n'a été ajoutée pour la répétition après Xcode.

Les captures suivent le protocole corrigé `start → synchronize → construction
du graphe/eval → finish`, avec inputs et warmup préalables. L'
[inspection Xcode réelle](xcode-dispatch-inspection.json), Xcode 27.2 build
27B52, confirme les commandes suivantes :

| Graphe | Dispatchs compute observés | Command buffers / encodeurs compute |
| --- | --- | --- |
| Stock B1 | **4** : cast F16→F32, sigmoid, produit F32, cast F32→F16 | 2 / 1 |
| Fusion B1 | **1**, kernel dédié, grille 6 144, threadgroup 128 | 2 / 1 |
| Stock B2 | **10** : deux fois cast/sigmoid/produit/cast, puis deux copies de concaténation | 2 / 1 |
| Fusion B2 | **1**, kernel dédié, grille 12 288, threadgroup 128 | 2 / 1 |

Zéro encodeur render/draw observé. Aucun blit observé, mais les deux buffers
n'ont été développés dans le détail que pour stock B1 et fusion B2 ; **aucun
compteur blit universel n'est revendiqué pour les autres buffers non
développés**. Le toolchain optionnel de shader debug n'était pas installé ;
aucun téléchargement n'a été lancé. Le replay a fonctionné, puis Xcode a été
quitté (`isRunning=false`). Ces captures ne fournissent ni temps GPU, ni
compteurs ALU/spills, ni mesure de GPU idle. La baisse **4→1 / 10→1 dispatchs
du composant** n'établit pas une accélération du modèle.

## Intégration au modèle, états et chemins spéculatifs

Les six sélections du [plan modèle final](prepared/final-model-quality-plan-result.json)
ont chacune terminé avec **1 passé, 0 échec, 0 ignoré**. L'
[inspection récapitulative](final-model-inspection.json) correspond aux logs
suivants, sur le checkpoint réel, avec le flag T17 activé :

| Test exact | Contrat exercé |
| --- | --- |
| `qwen4::tests::flash_gdn_epilogue_matches_stock_serial_outputs_states_and_prefix_tapes` | B1/B2/B4/B8 ; chemins stock/fusion chargés dans la vraie GDN, **une requête par cohorte**, sorties et états bit à bit, **N−1 tapes de préfixe et état final N**, continuation exacte et refus atomiques. [Log](prepared/t17-final-model-gdn-integration.log). |
| `qwen4::tests::flash_matches_independent_reference_states_and_logits` | **5 positions × 248 320 logits complets**, finis et bit à bit avec la fixture indépendante ; routes et états récurrents couplés exacts, argmax, reset et refus invalides. [Log](prepared/t17-final-model-oracle.log). |
| `qwen4::tests::flash_components_match_reference_on_identical_inputs` | Composants comparés à la référence sur entrées identiques, avec leurs seuils propres et les écarts détaillés dans le log ; ne pas transformer toute cette batterie en une preuve universelle bit à bit. [Log](prepared/t17-final-model-components.log). |
| `qwen4::tests::flash_pair_verification_preserves_serial_logits_and_all_states` | Paires aux préfixes 0..4 ; logits/états complets bit à bit, commits de 1 ou 2 tokens, continuation, refus de transactions/commits/contextes invalides. [Log](prepared/t17-final-model-pairs.log). |
| `qwen4::tests::flash_mtp_session_preserves_greedy_sequence_target_and_head_states` | **Budgets 1/2/3/17** ; séquence greedy, états cible/tête et alignement exacts, absence de sortie pendante, refus atomiques, acceptation **et** rejet effectivement couverts (budget 17 : 12 propositions, 5 acceptées, 12 blocs). [Log](prepared/t17-final-model-mtp.log). |
| `qwen4::draft::flash_draft_captures_and_all_prefix_commits_match_serial` | **36 cas de préfixes**, largeurs 1..8 ; captures, commit/rollback, logits et états cible exacts, continuation. Footer : **64 000 valeurs de capture**, `byte_equal=true`, au-delà du seul seuil RMSE 0,004. [Log](prepared/t17-final-model-dflash.log). |

Les deux premières préparations des qualités modèle avaient été refusées
avant PID : membre de texte requis absent du plan, puis marge mémoire
7 985 725 440 < 8 500 000 000 octets. Elles ne sont pas comptées comme des
tests modèle exécutés. Le plan final a été admis après libération de mémoire.

Les durées de superviseur 1,805 / 3,575 / 3,403 / 9,769 / 10,019 / 27,174 s
incluent préparation/chargement/vérifications : **ce ne sont pas des mesures
de vitesse du modèle**. Le pic natif de vie enregistré pour le dernier test
DFlash est 7 409 178 864 octets ; ce nombre n'est ni le budget de cache de
3 Gio ni une mesure d'allocation MLX équivalente.

## Répétition après Xcode et conditions d'exécution

L'initialisation de composants système lors de l'ouverture de Xcode 27.2 a
motivé un préenregistrement de requalification, sans rebuild ni nouveau
kernel. Le [plan après Xcode](post-xcode-quality-plan-result.json) a répété les
**six micros et les six qualités modèle**, dans des processus neufs, avec les
mêmes CLI/libtest/sources/MLX/metallib. Les **12/12 répétitions** ont chacune
annoncé exactement **1 passé, 0 échec, 0 ignoré**, avec processus joints.
Les logs portent les noms `t17-final-*-post-xcode.log` et
`t17-final-model-*-post-xcode.log` ; par exemple l'
[exhaustif](t17-final-exhaustive-post-xcode.log), l'
[intégration GDN](t17-final-model-gdn-integration-post-xcode.log) et
[DFlash](t17-final-model-dflash-post-xcode.log).

Le bilan est donc **13 sélections de qualité uniques finales**
(6 micros + 1 capture + 6 modèle) et **12 répétitions explicites après Xcode**.
Ce ne sont ni 25 contrats différents, ni des répétitions de débit. La capture
n'a pas été répétée. Les résultats des versions intermédiaires et des
contrôles négatifs restent séparés de cette qualification finale.

Les plans finaux attestent la collecte des processus et **zéro nouvelle
écriture swap cumulative depuis l'ancre `T17-w1` pendant les qualifications,
le bridge et le pilote initial, jusqu'avant `T17-DF-RECHECK-01`**. Cette
borne temporelle est essentielle : ce n'est pas un constat de zéro swap
sur toute la suite de la campagne. Les contrôles secteur, activité
concurrente, swapins et RSS sont assurés par le superviseur. La garde
d'admission de cette première phase est de 8,5 Go pour le modèle complet
et de 2 Go pour les seuls micros/captures, avec swapins ≤ 256 Mio et
RSS ≤ 10 Gio. Le seuil de footprint natif < 12 Go est vérifié dans les
footers effectivement collectés : notamment le pic de vie DFlash de
7 409 178 864 octets et chaque réponse du bridge. Cela ne constitue pas
un échantillonnage continu universel du footprint natif, ni une mesure du
pic mémoire de l'interface Xcode. Une seule inférence GPU est admise à la
fois ; compilations, proveurs et replays sont séparés. Ces observations
n'affirment pas l'absence de swap historique avant l'ancre.

Le recheck DFlash ultérieur `T17-DF-RECHECK-01` sous w1 a été **interrompu
après 1 048 576 octets (1 Mio) de nouvelles écritures swap globales, avec
zéro réponse produite** ; le processus a été arrêté et joint. Ce résultat
reste conservé et n'est ni une comparaison C/A terminée ni une divergence
numérique. Le zéro swap des qualifications et du pilote antérieurs n'est
pas étendu à cet essai interrompu. Voir le [bilan du pilote et des deux
rechecks](pilot-results.md).

`T17-DF-RECHECK-02` a ensuite ouvert **une nouvelle ancre w2 explicite**, après
60,27 secondes d'observation stable, sur secteur et sans moteur concurrent,
avec une garde d'admission renforcée à **12 000 000 000 octets** disponibles
estimés avant chaque processus. Le vrai [fichier d'ancre w2](gpu-anchor-w2.json)
enregistre `vm_bytes.Swapins = 53807169536` et
`vm_bytes.Swapouts = 140865945600` ; ce sont les compteurs absolus de départ,
pas les deltas de la nouvelle phase. L'[inspection w2](df-recheck-w2-inspection.json)
établit **2 processus terminés et joints, 0 nouvel octet Swapout et 0 nouvel
octet Swapin depuis w2**. Les réponses, modes, goldens et footers mémoire
ont été contrôlés. L'ancienne ancre w1 n'est pas réécrite et ses résultats
ne sont pas fusionnés avec cette répétition sous d'autres conditions.

La correction est qualifiée dans les formes, valeurs, fixtures, binaires et
conditions décrits. D'autres modèles, versions MLX/macOS, toutes les valeurs
F32, tous les contextes et toute concurrence restent hors de cette couverture.
Les inspections bridge, les chronos AR/MTP/DFlash et les rechecks sont
rapportés séparément dans [pilot-results.md](pilot-results.md) ; aucun
résultat de vitesse n'est déduit de ces seuls succès de qualité ou des
échecs attendus des contrôles négatifs.
