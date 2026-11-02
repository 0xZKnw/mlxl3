# POSTP08-F02-01 — épilogue dense S2 exact

Mise à jour du 1 novembre 2026. **Prototype local, flag par défaut 0.**
Correction composant/modèle et réduction 8→5 dispatchs qualifiées dans les
limites ci-dessous. Bridge17 AR/MTP/DF terminé ; pilote AR128 **non concluant**
(+3,06 % observé, dérive de la référence). MTP128 incomplet après interruption
par compilation externe ; DF128 non lancé. La reprise du 1 novembre est refusée
avant modèle pour marge mémoire insuffisante, malgré l'autre chat désormais idle.
**Aucun gain intégré, aucune app mise à jour, aucun push.** Les anciennes preuves
et les échecs sont conservés ; le plafond total swap autorisé de 32 Mio reste
attaché à l'ancre originale, sans réinitialisation.

## Changement et identité

Reprise de la [conception F02 du 25 octobre](historical-F02/review.md), jamais
implémentée/mesurée à l’époque. Le candidat conserve préparation et QMV stock,
puis remplace somme S2/cast/H128/SV par un épilogue indépendant de 8 threads
par bloc de 128 valeurs. Périmètre : Flash `qwen4_exp`, référence, K4/MUL1,
6144→2560, SG4, splits2, B1/B2, F16/F32, SV F16, sans biais. Deux sites
mono/batch ; hors domaine, stock.
`MLXL3_FLASH_DENSE_EPILOGUE=0|1` est strict et capturé au chargement.

L’ordre reste `(p1+0)+(p0+0)` F32→F16, radix16→F16, radix8→F16, scale F32→F16,
puis SV F16→F16. Le dtype et la forme de l’appelant sont conservés. Pas de
compteur intergroupe, readback, eval ou synchronize ajouté. Le vieux
`FUSE_OUTPUT` NT8/split1 n’est pas activé. Le scale stock vaut `0x3db504f3`.

HEAD `7978fceac30d991a8c06d72114fc932b2443f555` modifié localement ; état
préexistant préservé. [Sauvegardes](before/identity.json), [diff candidat](candidate.patch),
[identité v1 et commandes](build-identity.json). Sources primaires MLX 0.32.2
épinglées dans le dossier historique ; modèle
`Qwen3.8-Flash-Next-exl3-205-65c89531`, Apple M5, macOS 27.2, MLX 0.32.2.

| Artefact immuable | SHA-256 |
|---|---|
| CLI final | `6f35338c74d940c28bee9d37bd2284c5c505926fa918d5133325b38db5f6eee5` |
| Libtest v1 | `e5e464948a54007cf209991d4ddb112eda3d520b37dab1c07d4c18fb8b147a5e` |
| Libtest v2, extension cfg(test) uniquement | `43c55cd1a2d2895121a064a16013f7d226941b39ff33d2156e1e9497ec62e240` |

Binaires sous `build/post-p08-dense-epilogue-20261031/{engine,libtest,libtest-v2}/prototype-engine`.
L’extension v2 ne remplace ni CLI 6f ni lib e5 ; [identité v2](quality48-v2/build-identity.json).

## Vérifications réalisées

- Format, Clippy release strict MLX/chat tous targets et builds CLI/libtest
  passent. CPU lib : **174 passés, 0 échec, 8 ignorés** ; 3 tests ciblés couvrent
  parser, métadonnées/refus et bits du scale. [Logs](cpu-lib-suite.log).
- Kani 0.68.0/CBMC 6.11/kissat : **201 obligations SUCCESS, 9/9 covers,
  zéro échec/inatteignable**, unwind17. Rows et dimensions i32, dtypes booléens
  symboliques ; rangs partiels 0..5/SV 0..3, sans assume ajouté. Le vrai prédicat
  Rust et son lancement sont vérifiés ; les calculs d’adresses sont un modèle
  arithmétique de l’indexation, **pas une preuve source du MSL/H128/MLX ou de
  la concurrence**. Les constructs non pris en charge détectés ailleurs dans
  la crate ne sont pas atteints. [Commande/résultat](kani-result.json),
  [obligations individuelles](kani.log).
- Mutant jetable du vrai module : B2 accepte 2559 au lieu de 2560 ; l’assertion
  `layout.is_some()==expected` échoue réellement, 1/202, après compilation.
  Contre-exemple imprimé, production inchangée. [Preuve négative](proof-mutant/result.json).
- Lib MLX v1 : **202 passés, 0 échec, 183 ignorés**. Physique isolé : 12 cas
  partiels finis byte-exact + 10 non finis intentionnels (classes/signes/valeurs
  finies, sans contrat payload NaN), 15 refus avant kernel + 2 reprises,
  21 projections complètes dont 3 fixtures indépendantes et 18 cas B1/B2/B3
  F16/F32. Requêtes attendues : 1 pour ON/B1–2, 0 pour OFF/QKV/B3 ; ce sont
  des demandes de graphe,
  pas des dispatchs. [Plans et reçus v1](light-v1/identity.json).
- Mutant shader : lecture de ligne 0 en B2, écriture de ligne correcte.
  **Exit101 attendu**, 1 test échoué, `F02_FIRST_MISMATCH rows=2`, demi-mot2561.
  Aucun OOB recherché. Shader normal restauré `ac5dfbdc…`, 106 sources normales
  et archives 6f/e5 vérifiées. Le mutant `ad89a6e8…` correspond explicitement
  au snapshot mutant, pas au fichier live. [Identité](negative-offset/identity.json),
  [reçu physique](negative-offset/negative-plan-result.json).
- Extension v2 compilée, format/Clippy passent. Nouveau test physique :
  **192 paires ON/OFF exactes** (48 out×B1/B2×F16/F32), stock explicitement
  OFF, 1 test passé/0 échec/0 ignoré, 1,39 s de processus. B2 utilise l’oracle
  de projection stock, pas une nouvelle fixture externe indépendante B2.
  [Reçu](quality48-v2/physical-plan-result.json), [contrôles](quality48-v2/checks.json).
  Le test ignoré supplémentaire donne **202/184 attendus** pour la suite
  ordinaire v2, qui n’a pas été réexécutée : ne pas confondre avec le résultat v1.
  L’ancien test batch sous env ON ne constitue pas seul un oracle indépendant F02.

La [qualité modèle physique v2](model-quality-v2/summary.json) passe également,
six sélections isolées, chacune 1 passée/0 échec/0 ignoré, processus joints et
zéro nouvelle écriture swap :

| Sélection | Durée processus |
|---|---:|
| AR, oracle/logits/états | 5,57 s |
| Composants des 48 couches | 3,05 s |
| Paires/transactions multi-token | 9,42 s |
| Session MTP | 9,70 s |
| Tête MTP, fixture held-02, 10 positions | 2,06 s |
| DFlash, 36 préfixes de largeurs 1 à 8 | 26,80 s |

Ces durées qualifient l’exécution des tests et ne mesurent pas le gain de
vitesse. Les oracles incluent logits/états byte-exact ; les composants et
fixtures conservent leurs tolérances existantes, sans nouvel assouplissement.
La tête MTP a une [identité dédiée](model-quality-v2/head-identity.json) ; le
CLI et les sources production restent inchangés.

Commandes de vérification exécutées, Cargo appelé par son chemin absolu
`/Users/justin/.cargo/bin/cargo`, `CARGO_BUILD_JOBS=1`,
`MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx` :

```text
cargo fmt --check
cargo test --locked --release --no-default-features --lib -- --test-threads=1
cargo kani --lib --no-default-features --harness dense_epilogue_geometry_bounds_real_launch_and_addresses --jobs 1
cargo clippy --locked --release --features mlx,chat --all-targets -- -D warnings
cargo build --locked --release --features mlx,chat --bin mlxl3-rs --message-format=json
cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json
```

Les tests GPU ignorés sont sélectionnés individuellement sous superviseur
avec `--exact --ignored --nocapture --test-threads=1` ; commandes/options
complètes et sorties attendues dans les plans liés, sans sélection vide.

## Micro et captures

[Micro](light-v1/micro-summary.json) : 96 échantillons×32 `forward+eval`
séquentiels, 3 chauffes/bras/cellule, 12 rounds AB/BA, sorties exactes/non vides,
aucun échantillon exclu. Wall construction/soumission/complétion, pas temps GPU
exclusif ni graphe naturel de 48 couches.

| Projection | B | OFF µs | ON µs | Débit local agrégé |
|---|---:|---:|---:|---:|
| GDN0 out |1|312,60|283,46|+10,28 %|
| GDN0 out |2|386,81|382,95|+1,01 %|
| QSA15 out |1|282,65|283,25|−0,21 %|
| QSA15 out |2|325,69|327,14|−0,44 %|

Le premier OFF GDN B1 est 507,12 µs : conservé, sans attribuer le +10,28 % à un
gain stable. Résultat composant **mixte**, pas gain modèle établi.

Six captures B1 OFF/ON, sorties exactes aux fixtures, borne commune de 1,5 Gio
respectée. Inspection physique Xcode 27.2, sans profilage supplémentaire :

| Projection | Dispatchs OFF→ON |
|---|---:|
| GDN0 out |8→5|
| QSA15 out |8→5|
| GDN0 qkv, contrôle |6→6|

Deux command buffers/un compute encoder pour chaque capture. [Décompte final](captures-v1/dispatch-inspection.json),
[métadonnées OFF](captures-v1/OFF/metadata.json), [ON](captures-v1/ON/metadata.json).
Xcode quitté et workload arrêté. **Aucun temps GPU natif ou gain tok/s déduit
du replay/nombre de dispatchs.**

[Audit d’adaptation](adaptation-audit-v1.md) : aucun défaut concret imposant
une correction. Donation/allocation du custom, temporaires volatile et coût de
graphes/bindings restent des limites non attribuées. Pas de nouvelle copie
contiguë démontrée en production, ni nouveau SSD/sync/readback.

## Contrôleur, CI et échecs conservés

[Inspecteur initial](inspection-verification.json) : 52 cas synthétiques (12 positifs,
40 négatifs), mutation supprimant la validation du flag détectée, source
restaurée ; 82 tests bridge/POOL passent, dont faux enfants silencieux/terminés
et cleanup. Ruff strict/format/compile passent. CrossHair absent : non vérifié
formellement. CI PR/main câble ces contrôles et le harnais Kani ; YAML passe,
actionlint absent, **aucun run distant du code local non publié revendiqué**.

Les échecs restent archivés : format initial, message d’assertion inspecteur r1,
CLI Python refusant initialement `--timeout` (corrigé sans changer son défaut 300 s),
test du timeout qui supposait à tort avoir reçu `ready` avant 0,2 s, et erreur
d’index du précontrôle de plan avant PID. Voir [historique inspecteur](inspection.md)
et [précontrôle](prepared/plan-structure-precheck-failure.json).
Le dialogue Xcode empêchant initialement QKV ON est conservé dans l’inspection ;
le complément après redémarrage confirme 6 dispatchs, sans remplacer de mesure.
Les deux mutants sont des échecs attendus, pas des positifs numériques.
Les refus préflight de métadonnées/goldens restent archivés ; aucun PID modèle
n’a démarré sur ces échecs. Le [préflight final](prepared/final-preflight.json)
valide les 18 plans ; cette validation de plans n’est pas une mesure bridge.

L’amendement S32 ajoute le [runner dédié](bounded-gpu-runner.py), sans changer
le superviseur partagé : **48 tests CPU passés**, limites inclusives 0/32 Mio,
refus de 32 Mio+1, types/reset/Swapin, ancre altérée, baseline conservée entre
commandes, dépassement enregistré avant rejet et snapshots finaux. Le vrai
superviseur crée et joint seulement des faux enfants Python ; VM/power/quiet
sont remplacés par des fixtures déterministes. La mutation réelle acceptant
cap+1 est détectée, source normale inchangée. Ruff/format/py_compile passent ;
[reçus et limites](swap32-amendment/runner-verification.json). Les quatre échecs
initiaux du routage MONITOR dans le fixture temporaire, avant enfant, restent
[conservés](swap32-amendment/runner-checks-r1.txt).

L’[inspecteur S32](swap32-amendment/inspector-checks.json) passe **89 cas**
(17 positifs, 72 négatifs ; les 52 précédents conservés), dont accord
manifest/identity/anchor, plafond absent/invalide et dépassement cumulatif.
La mutation supprimant ce dernier contrôle est détectée puis restaurée.
CrossHair reste absent : diagnostic conservé, aucune preuve Python annoncée.
Le [workflow local](swap32-amendment/ci-validation.json) conserve PR/main et
Kani, et câble ces nouveaux contrôles ; pas de run CI distant revendiqué.

## Incident, conditions et reprise S32

[Ancre F02-w1](anchor-w1.json) prospective explicite : les 262144 octets de
Swapout historiques P08 restent visibles. Qualité et AR17 A/C ont terminé avec
zéro nouvelle écriture depuis cette ancre ; ces résultats complets restent
valides dans leur portée, sans gain modèle déduit du bridge17.

La chauffe suivante **MTP17-A, F02 OFF**, a été interrompue après **2 chauffes
et 0 mesure** : compteur absolu 262144→786432 B, soit **524288 B nouveaux
(512 Kio)**, Swapin0. L’ancienne garde zéro a arrêté et joint le groupe ;
rapport incomplet, inspection échouée, aucun succès ou gain réutilisé.
[Reçu interrompu](prepared/f02-bridge17-MTP-1-A-plan-result.json).
Cet incident sous flag0 n’établit aucune causalité du candidat F02.

L’utilisateur a ensuite explicitement autorisé un **plafond TOTAL de 32 Mio**
(33554432 B) depuis **la même ancre originale 262144 B**, SHA
`052bdca4c7197f2980a992cb7290de26f5b3001841652b8d2aef92c6ea08fab8`.
Les 512 Kio sont déjà consommés : **31,5 Mio restent à l’admission S32**, pas
32 Mio par processus. Aucun réancrage ni changement rétroactif des échecs.
[Autorisation/protocole](swap32-amendment/protocol.json). Le monitor conserve
son attente de 2 s entre contrôles, hors coût des commandes système : arrêt
sur dépassement observé, maxima réels
conservés ; ce mécanisme ne constitue pas un quota swap imposé par l’OS.

Toutes les autres gardes restent inchangées : secteur/quiet, un modèle GPU,
aucun build/proveur concurrent ; RSS ≤10 Gio, pic natif <12e9 octets,
Swapin ≤256 Mio. Budget fixe : 3 Gio/53 slots×48, 4 lecteurs, allocator 64 Mio,
MLX 40 ops/40 MB, Hot/Wide/Unique/overlap1, lookahead0, T04/T05/T17/T18 OFF.
Admission légère : 2 Go ; modèle : 3 instantanés sur 20 s ≥10,5 Go, sans baisse automatique.

[Admission S32](prepared-s32/admission.json) : **14,15 / 14,20 / 14,18 Go**
estimés disponibles, 512 Kio cumulés depuis l’ancre. L’[identité S32](final-identity-s32.json)
fige **636 sources**, mêmes CLI6f/lib43c55 ; [18 préflights passent](prepared-s32/final-preflight.json).
Le [manifeste S32](prepared-s32/manifest.json) réutilise les
[2 AR17 complets](prepared-s32/reused-AR17-compatibility.json) sans GPU, et
prépare **16 plans à noms/sorties neufs** (4 bridge17 MTP/DF, 12 pilotes128).
Ces plans préparés ne valent pas, à eux seuls, résultats complets.

## Résultats modèle inspectés et interruption

[Bridge17](prepared-s32/bridge17-summary.json) : six processus AR/MTP/DF A/C,
24 réponses chauffe comprise, goldens/modes/compteurs/mémoire/jointures valides.
Débits descriptifs seulement : AR10,631→10,494, MTP12,439→12,376,
DF12,219→12,646 tok/s. Deux AR anciens réutilisés explicitement, autres frais
sous S32 ; pas de gain établi avec 16 transitions mesurées par prompt.

[AR128 ACCA](prepared-s32/AR128-root-summary.json), quatre processus complets,
MAIN/HOLD, deux chauffes puis deux mesures par processus :

| Mesures uniquement | A | C | Écart observé |
|---|---:|---:|---:|
| Décode agrégé, tok/s |10,36265|10,68016|+3,06 %|
| Préfill agrégé, tok/s |10,21360|10,64952|+4,27 %|

La référence A1→A4 dérive de −4,13 % en décode et −6,12 % en préfill ;
C2→C3 varie aussi de −1,92 % en décode. **Non concluant, aucun gain causal établi.**
Aucun échantillon exclu ni répétition pour sélectionner un résultat favorable.
[Contre-analyse](prepared-s32/AR128-independent-analysis.json),
[conditions](prepared-s32/AR128-conditions.json), [audit mémoire](memory-audit.md).
Les compteurs de compression corrèlent avec l'ordre ; leur coût causal reste
inconnu. Le maximum RSS des enfants n'est pas un RSS instantané du moteur.

MTP128 : A1 et C2 complets/goldens exacts, mêmes propositions et acceptations
MAIN72/54 et HOLD78/48. Nouveaux Swapouts respectifs1 638 400 et589 824octets,
consignés sous le plafond autorisé. C3 est interrompu après4,11s par la garde
quiet (`cargo install wasm-bindgen-cli` d'un autre chat), processus joint,
rapport incomplet et inspection échouée conservés. A4 et DF128 non lancés.
Ce couple incomplet ne qualifie pas un gain MTP.

[Admission du 1 novembre](resume-20261101/admission.json) :6,714/6,547/6,160Go
estimés disponibles, sous10,5Go. Secteur, absence de compilateur,636sources et
4binaires conformes ; **aucun modèle lancé**. Depuis l'ancre originale :
15 400 960octets de nouvelles écritures swap (14,6875Mio sur32),
5 570 560octets lus. L'augmentation depuis la pause est une observation système,
pas une écriture attribuée au candidat. Aucun réancrage, aucune garde abaissée.

**État : interrompu pour ressources**, qualité conservée, vitesse globale non
validée, candidat OFF. Une future reprise doit préenregistrer ses nouveaux
noms/sorties et comparer les deux bras dans ses conditions contemporaines ;
la pause du31octobre au1novembre interdit de rebaptiser l'ensemble ACCA continu.
Ancien C3 échoué et MTPA1/C2 restent visibles. Aucun retrait décidé sans gain
mesuré ni audit : les deux audits d'adaptation n'ont trouvé aucun défaut concret.

La [revue locale suivante](../post-p08-local-review-20261101/README.md) avance
sans GPU ni changement moteur. Pas de nouveau port de kernel amont.
