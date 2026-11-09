# PR161 : nos coûts MTP face à Regolith

> Audit historique du runtime initial et du premier correctif HC/routeur. Les sections suivantes conservent leurs constats à cette révision. Plusieurs limites ont depuis été corrigées et qualifiées localement : voir [les correctifs finaux, mesures et nouvelle lecture TurboEXL3/Regolith](pr161-fixes-2026-11-08.md).

Le problème principal est structurel : notre B2 programme encore deux calculs cible largement scalaires. Il partage des lectures SSD, mais très peu du calcul des poids entre tokens. Il paie la deuxième ligne même lorsque la proposition est rejetée. Une régression supplémentaire, la perte du producteur HC/routeur fusionné dans les cohortes SSD, est corrigée localement et qualifiée sur le modèle réel.

## Sources et périmètre

- PR161 : base `28cf4c210490f9a028e9bdda5b1e067e1cdbca13`, tête `ebacd5ecdec128c249607eac264c28a4ba836ec2`, toujours identiques lors du contrôle final. Les problèmes de structure ci-dessous existent dans cette tête ; la campagne initiale ne constitue pas une comparaison base/tête.
- Code testé et correctif : `/Users/justin/.codex/worktrees/pr161-speed-20261108/mix-stq1_0`. Le checkout utilisateur contient des modifications antérieures et n'a pas été réinitialisé. Le correctif exécutable est uniquement dans le worktree isolé.
- Le Regolith pertinent est [ptr.pet/regolith](https://tangled.org/ptr.pet/regolith), hébergé sur Tangled : révision `92a523f036092df5a072d24db1cb4823863ca862`. Son installateur épingle Strata `296b59d2f73c8f04a805df29c2fa97655adace78`. Le projet GitHub homonyme de sandbox Linux n'est pas celui-ci.
- Inspection du patch réel `scripts/strata_metal_complete.diff`, de l'installation, des audits MTP, QSA, catch-up, GDN et des essais rejetés ; aucun installateur ni téléchargement de poids Regolith exécuté. [Identités des sources](measurements/pr161-speed-20261108/regolith-sources.json).
- Même modèle local EXL3 moyenne 2,05 bpw, Apple M5/24 Gio. Aucun changement de poids, routage, contexte, précision ou critère d'acceptation pour le correctif.

## 1. Fusion HC/routeur perdue dans le chemin SSD : corrigé

**Défaut confirmé par un test rouge sur l'ancien runtime.** `LayerResidual::prepare_after_attention` effectuait l'injection HC et la lecture MLP, puis `forward_block_cancellable` soumettait un routeur séparé par ligne. Le chemin AR possédait déjà `read_after_write_with_gate`, qui exécute ces producteurs ensemble. La séparation ajoutait `largeur × 48` soumissions de routeur par bloc, soit **96 en B2** lorsque la fusion est éligible.

Le correctif réutilise ce helper existant, conserve le `RouteOutput` dans la ligne préparée et le transmet au scheduler. Il garde le routeur séparé en repli lorsque la fusion est désactivée. Les formes et le nombre de routes sont validés avant mutation du cache. Le helper attend son command buffer : les routes préparées ne sont pas des producteurs GPU laissés en vol lors d'une annulation. Aucun shader ni nouvel interrupteur.

Fichiers : [gdn.rs:659](measurements/pr161-speed-20261108/hcroute-delivery.patch), [flash_verify.rs:119](measurements/pr161-speed-20261108/hcroute-delivery.patch), [moe_block.rs:57](measurements/pr161-speed-20261108/hcroute-delivery.patch). Les compteurs physiques vérifient le producteur réellement exécuté, pas seulement la présence de l'option.

**Diagnostic court après parité**, bridge de production, MAIN, 32 sorties/31 decode, contexte4096, température0, pénalité1, prefix reuse OFF, cache explicite3 Gio/53 slots par couche, quatre readers. Chaque processus fait un warmup MTP32 puis une réponse mesurée, sans vider les caches entre les deux. B2/cohortes/shared QMV/KV-only identiques. Ordre A1 ancien / B corrigé / A2 ancien :

| Binaire | Decode tok/s | Decode s | TTFT s | Total génération s |
|---|---:|---:|---:|---:|
| A1 | 7,016 | 4,418 | 10,898 | 15,316 |
| B | 7,444 | 4,165 | 11,011 | 15,175 |
| A2 | 7,212 | 4,298 | 10,537 | 14,835 |

Même texte et hash `bfa49090cb33d4fe`, 15/15 propositions acceptées pour chacun. B est entre +3,21 % et +6,09 % en débit decode par rapport aux deux contrôles ; les contrôles dérivent de +2,79 %. **Un seul point corrigé, aucun gain stable établi**, aucun gain TTFT ou total démontré. Ce prompt court n'exerce pas le coût des rejets. Batterie80→79 %, température non mesurée. Les trois processus montrent environ7,40 Go de footprint et une compression logique différente (environ2,41/2,85/2,50 Go) ; aucun swapin logique du processus enregistré. Les gardes RSS/swap système de la longue campagne n'ont pas été reprises pour ce diagnostic court. Tous les enfants/readers sont joints.

Ce protocole chaud à53 slots est distinct de l'ancien128 tokens à90 slots : ne pas comparer leurs débits. [Plan](measurements/pr161-speed-20261108/hcroute-speed-plan.json), [résultats](measurements/pr161-speed-20261108/hcroute-speed-summary.json).

## 2. B2 ne partage pas encore le calcul cible entre tokens

**Confirmé dans le code, coût séparé non mesuré.** Dans `stage_block_inner`, la boucle est couche puis ligne. Sans cohortes, chaque ligne appelle `layer.forward(...).complete(gpu)`. Avec cohortes, les préparations attention/PLE et les écritures finales restent par ligne. Dans `encode_block_ready`, les experts prêts sont regroupés sur l'axe des experts **d'une même ligne**, puis le scheduler recommence pour la suivante.

Ainsi un expert commun à deux tokens peut être lu une seule fois par le reader SSD, tout en étant consommé par deux calculs GPU séparés. L'option QMV multiligne sert actuellement l'expert partagé ; `projection.rs` exclut explicitement les records routés, les projections génériques et `mtp_hidden` de cette spécialisation. Le nom « B2 » ne suffit pas à obtenir un amortissement des poids sur le GPU.

Sources : [flash_verify.rs:97](measurements/pr161-speed-20261108/hcroute-delivery.patch), [moe_block.rs:423](measurements/pr161-speed-20261108/hcroute-delivery.patch), [projection.rs:758](measurements/pr161-speed-20261108/hcroute-delivery.patch).

Regolith regroupe les projections sur plusieurs lignes et construit un plan d'experts consommé par plusieurs tokens. Son ordonnanceur sépare travail résident et lectures manquantes autour de la coupure SSD. Mécanismes vérifiés dans le [patch source](https://tangled.org/ptr.pet/regolith/blob/main/scripts/strata_metal_complete.diff), notamment `Window::run`, `run_ds`, `mtl_rt::run_graph_seg` et les plans MoE. Ses gains ne sont pas transposables à nos kernels EXL3 sans une implémentation et une parité propres.

**Priorité suivante :** un véritable calcul multiligne des projections réutilisables et des consommateurs d'un même expert routé, en gardant l'ordre d'accumulation par rang et les états temporels canoniques. L'ancien essai `FLASHSSD-2026-10-12-15` reste non concluant à cause de la dérive ; il ne valide pas notre runtime actuel et ne doit pas être relancé inchangé.

## 3. B2 paie les rejets ; D1 ne réduit pas les passes cible acceptées

**Code confirmé et temps de cycles disponibles.** `advance_with_eos_cancellable` propose un token puis calcule `[anchor, proposal]` avant de lire les logits qui décident l'acceptation. Un rejet conserve seulement la première ligne. D1 évite ce successeur rejeté ; sur acceptation, il reste deux forwards cible ordinaires plus le draft et son rattrapage.

Premier tour MAIN128 : 54 acceptations/72 propositions. Les18 rejets coûtent **2,454 s en D1 contre5,218 s en B2** ; les54 acceptations,15,366 s contre15,699 s. Le surcoût des cycles rejetés représente2,764 s des3,091 s de différence decode, soit environ89 %. C'est une décomposition d'un tour enregistré, pas une attribution universelle ni un benchmark isolé de kernel. [Cycles](measurements/pr161-speed-20261108/cycle-analysis.json), [mtp.rs:425](measurements/pr161-speed-20261108/hcroute-delivery.patch).

À75 % d'acceptation et sans autre économie, le coût d'une vérification B2 doit descendre sous environ1,75 passe AR, moins le coût du draft/rattrapage, pour être avantageux. Notre structure se rapproche encore de deux passes. `measured_depth` existe dans `speculation_plan.rs`, mais n'a pas de consommateur dans la session de production ; ni ce modèle de coût ni le suffix lookup ne pilotent cette décision. Le `SpecGate` de Regolith est intégré à son ordonnanceur ; ce n'est pas actuellement notre cas. **Piste :** mesurer le coût réel des blocs et choisir AR/D1/B2 selon celui-ci, sans confondre acceptation élevée et accélération.

## 4. Draft et rattrapage trop découpés ; option GPU existante neutralisée par le harnais

**Chemins confirmés, contribution temporelle non isolée.** `mixed` relit l'embedding via CPU, convertit BF16→F16 puis l'envoie au GPU. Dans le chemin sans device greedy, les248320 logits F16 reviennent au CPU ; `greedy_id` les renvoie au GPU pour normalisation, puis construit les probabilités F32 côté CPU. Le harnais fixe `head.device_greedy=false`.

Le GPU greedy normalisé existe déjà pour le proposeur et D1. Il ne faut donc pas réécrire cette fonction ni prétendre qu'elle manque. En B2, la sélection des logits du bloc continue d'utiliser les vecteurs hôtes. Une simple substitution par argmax brut ne préserve pas nécessairement le contrat de normalisation Half et les égalités.

Après une acceptation, `head.extend` effectue immédiatement le rattrapage ; la proposition suivante recommence sa chaîne. Regolith combine le rattrapage et la première étape draft suivante en `(a+1)` lignes, puis ne calcule la query QSA que pour la dernière (`defer_catchup`, `QSA_LAST_Q`) ; voir son [audit r40](https://tangled.org/ptr.pet/regolith/blob/main/audits/r40_rocm14_metal_optimizations.md). Les timings de cet audit sont les siens, pas un gain MLXL3 mesuré.

Sources : [mtp.rs:81](measurements/pr161-speed-20261108/hcroute-delivery.patch), [mtp.rs:190](measurements/pr161-speed-20261108/hcroute-delivery.patch), [mtp.rs:453](measurements/pr161-speed-20261108/hcroute-delivery.patch). Nuance : notre `fc_hidden` reçoit déjà les quatre flux HC via `encode_rows`. Le kernel spécialisé conserve une réduction distincte ; annoncer « quatre appels scalaires manquants » ou activer aveuglément le QMV partagé serait incorrect. L'[audit fc_hidden Regolith](https://tangled.org/ptr.pet/regolith/blob/main/audits/opus_r28_mtp.md) sert à cibler la réutilisation effective des poids, pas seulement la forme de l'API.

## 5. QSA : copies du préfixe après rollback

**Condition et copies confirmées, coût à long contexte non mesuré.** Après une vérification qui publie deux lignes puis ne conserve que la première, le high-water `published` du slab peut rester au-delà de l'offset restauré. `prepare_append` appelle `qsa_append` et, si `offset < published`, alloue un slab puis recopie keys, values et blocs indexés du préfixe. Le coût augmente avec le préfixe à copier. [qsa.rs:721](measurements/pr161-speed-20261108/hcroute-delivery.patch).

Cette copie protège les snapshots et n'est pas un bug de logits démontré. La retirer exige une propriété transactionnelle sur la queue spéculée et les snapshots externes encore vivants. Abaisser simplement `published` lors d'un restore serait dangereux. Regolith a une sauvegarde ciblée de la queue brute QSA ; son ancien [audit suffix/rollback](https://tangled.org/ptr.pet/regolith/blob/main/audits/sol_spec_mtp_decode.md) explique pourquoi restaurer seulement un compteur ne suffit pas. Nos slabs ne sont pas son implémentation : son contre-exemple ne prouve pas un bug numérique chez nous.

Précision vérifiée après lecture de la revue jointe : pour nos12 couches QSA, keys+values représentent `12 × 2 × 1024 × L = 24576 × L` octets logiquement copiés, hors index, si les12 couches suivent ce fork au préfixe conservé de longueurL. Cela donne96 Mio pourL=4096 et1,5 Gio pourL=65536 ; ce sont des exemples de volume, pas des durées ni le contexte effectivement rempli de nos essais. L'append exige une capacité supérieure àL. Si la deuxième ligne a fait croître le slab vers un nouveau buffer, l'ancien slab du préfixe retenu n'est pas avancé par cette écriture : le fork supplémentaire au cycle suivant n'est donc pas systématique. Le coût éventuel appartient à l'append suivant, pas forcément au cycle de rejet chronométré.

## 6. Le harnais refroidit les experts du draft

**Défaut de qualification du protocole, pas du modèle.** `clear()` vide les caches experts cible et draft, puis le conditionnement commun exécute seulement AR. Les tenseurs fixes du draft sont résidents ; ses experts routés ne sont pas réchauffés. Le champ `draft_resident_for_all_arms=true` décrit donc trop largement l'état réel.

Premier tour : tous les modes commencent à8929,116 Mo alloués Metal. AR y reste ; chaque mode MTP finit à9176,383 Mo, soit+247,267 Mo pendant le decode. Les allocations seules ne mesurent pas toute la latence de ce phénomène. Les résultats128 tokens sont conservés et qualifiés **experts draft froids**, sans prétention de débit MTP résident. Source : [mtp_campaign.rs:25](measurements/pr161-speed-20261108/hcroute-delivery.patch). Aucun nouveau grand benchmark lancé pour corriger cette interprétation. Le diagnostic A/B/A utilise le vrai warmup MTP décrit plus haut.

## 7. Revue jointe : contre-exemple de rétention LFU reproduit

**Nouveau constat exécuté sur les vrais composants CPU, sans modification du runtime.** Cache froid11 slots, LFU, aucune prédiction ni lecture en vol ; routes `{0..9}` puis `{0..4,10..14}`. Le plan de cohortes déduplique les experts en ordre de première utilisation, réserve0..10 dans une première vague, puis11..14 dans la seconde. La fréquence est incrémentée une fois par réservation unique, pas par consommateur. Les protections portent seulement sur la vague réservée. La deuxième vague peut donc évincer0..3, pourtant communs aux deux lignes.

| Observation exécutée | Deux réservations scalaires | Réservations par cohortes |
|---|---:|---:|
| Lectures nécessaires dans le premier bloc | 15 | 15 |
| Experts de la deuxième route encore présents | 10 | 6 |
| Lectures nécessaires si cette route se répète | 0 | 4 |

Résidents finaux scalaires : `{0..4,9..14}` ; cohortes : `{4..14}`. Même résultat en décalant les IDs de0,1,255 ou497, jusqu'à l'expert511. Contrôles15,20,53 et90 slots : les15 experts tiennent, aucune relecture sur la route suivante dans les deux chemins. Le défaut décrit à11 slots est **réel**, mais ce scénario précis **n'explique pas nos mesures récentes à90 puis53 slots**, où l'union B2 de20 experts au maximum tient en une vague. La sous-comptabilisation des usages communs reste visible dans le code même sans dépassement ; son effet sur une suite de blocs chaude n'a pas été mesuré.

Chemin reproduit : `resident_expert_cohorts` → `DemandPlan::reserve` → `ExpertCache::reserve` → publication des vrais leases → demande suivante. Aucun simulateur réécrivant le LFU, aucun GPU ni fichier expert. Sources : [speculation_plan.rs:202](measurements/pr161-speed-20261108/hcroute-delivery.patch), [moe_block.rs:163](measurements/pr161-speed-20261108/hcroute-delivery.patch), [flash_support.rs:1438](measurements/pr161-speed-20261108/hcroute-delivery.patch), [victime:1551](measurements/pr161-speed-20261108/hcroute-delivery.patch).

Le test isolé passe **1 test,74 autres filtrés**, avec16 exécutions de scénario. Il attend le mauvais comportement existant pour le reproduire ; ce n'est pas une preuve qu'une politique corrigée fonctionne. [Source du probe](measurements/pr161-speed-20261108/attachment-cache-repro.rs), [sortie](measurements/pr161-speed-20261108/attachment-cache-test.log), [commandes et portée](measurements/pr161-speed-20261108/attachment-cache-execution.json). Deux préparations ont échoué avant exécution : oubli du lien libc, puis incompatibilité entre bitcode RustLLVM22 et linker AppleLLVM21. Le lien via Rust thin LTO et les dépendances déjà disponibles a permis le test ; diagnostics conservés.

Kani0.68/CBMC6.11 a été tenté sur ces mêmes modules,11 slots, deux routes, décalage d'IDs `u8` symbolique0..255, unwind18, aucune hypothèse ajoutée, couvertures0/255. **Timeout CBMC à45s**,0 harness vérifié et aucune couverture déclarée satisfaite ; ce n'est ni une preuve ni un contre-exemple formel. Processus joint, limite globale60s respectée. [Diagnostic Kani](measurements/pr161-speed-20261108/attachment-cache-kani.log).

Correction à concevoir : séparer déduplication des lectures et usages logiques du cache, définir la rétention après chaque préfixe retenu et conserver la protection des générations. Changer seulement le poids LFU d'un expert ne résout pas forcément sa protection entre vagues. Pas de nouvelle politique implémentée pendant cette vérification de la revue.

## 8. Projection du vocabulaire : une cible concrète supplémentaire

La revue jointe précise un site que l'analyse générale du calcul scalaire ne détaillait pas assez : la fin de `stage_block_inner` fait un `head.submit_fp16_view(...).read()` **par ligne**, même en B2/cohortes. La tête valide une forme2560×248320, soit635699200 poids logiques. Ce nombre n'est ni le volume physique EXL3 ni une durée GPU mesurée. [Appel:flash_verify.rs:170](measurements/pr161-speed-20261108/hcroute-delivery.patch), [forme:flash.rs:304](measurements/pr161-speed-20261108/hcroute-delivery.patch).

Deux choix ont des effets différents : projeter plusieurs lignes ensemble pour amortir les poids, ou différer la seconde tête jusqu'à l'acceptation pour ne pas la payer au rejet. Le choix exige le coût de cette projection sur notre chemin réel et le taux d'acceptation ; aucun n'est implémenté ni chronométré dans cet addendum. Un second head paresseux ne supprime pas à lui seul le calcul des48 couches de la deuxième ligne déjà effectué.

La revue jointe est donc largement confirmée, avec deux mises à jour : son défaut HC/routeur est déjà corrigé dans HCROUTE01, et l'identité des mesures n'est plus inconnue grâce aux manifestes/binaries conservés. Les traces de cycles permettent la décomposition des rejets donnée plus haut ; elles ne donnent toujours pas une attribution exhaustive kernel/SSD/copies. Le texte fourni s'arrête au début du point7 ; seule la partie effectivement reçue a été examinée.

## Fausses pistes écartées et limites de comparaison

- **Pas de gros memcpy GDN de rollback à supprimer ici.** Nos `snapshot` et `restore` clonent des handles. Le `fold()` de Strata éliminé par `dual2/spec1` n'existe pas dans ce chemin natif. Des frames GDN sont encore allouées, mais un pool doit respecter les snapshots externes ; son coût n'a pas été isolé.
- **Pas de boucle SSD qui soumet à vide.** `ReadBatch::ready(false)` attend au moins une complétion, puis draine celles déjà disponibles. Une fragmentation en petites cohortes peut coûter des soumissions ; elle n'a pas été attribuée par mesure séparée.
- Les~42 tok/s annoncés par Regolith utilisent Swift fine-tuné, GGUF IQ2, une autre configuration dense/draft, un vocabulaire draft réduit, sink64/window256 et un cache KV différent. Le [README](https://tangled.org/ptr.pet/regolith) précise également le bonus MoCE0,85 et l'arène experts wired6,5 Go. Ce routage favorise les experts résidents et change la sélection cible : ce n'est pas une accélération exacte de notre checkpoint/routage EXL3. Leur [journal d'essais rejetés](https://tangled.org/ptr.pet/regolith/blob/main/01-wrong-turns-and-traps.md) documente lui-même des changements de réponses sous MoCE.

## Vérification et livraison locale

Le test ajouté échoue sur le runtime initial avec `SSD cohort lost the canonical joint HC/router producer`, puis passe sur le correctif. Le test SSD de réutilisation possédait aussi une préparation incorrecte à20/40 slots : l'invalidation conservait les âges, donc dix experts ne remplissaient pas tous les backings. La préparation est corrigée par des lectures réelles ; les assertions de réutilisation sont conservées. Les échecs initiaux restent archivés.

| Contrôle final | Résultat et portée |
|---|---|
| Build release lib+CLI, défaut direct-metal + `chat` | Passe ; binaires figés et empreintes enregistrées |
| `cargo fmt --all --check`, `git diff --check` | Passent |
| `cargo clippy --locked --all-targets --features chat -- -D warnings` | Passe |
| Suite bibliothèque release | **354 passés,127 ignorés**,0 échec ; les ignorés ne sont pas déclarés exécutés |
| `pytest tests/test_native_coverage.py` et inventaire | **6 passés**,2 sous-tests ; marqueur `.take`1→2 revu comme transfert d'un handle de routes |
| Cinq invocations physiques, quatre fonctions distinctes | Oracle B2–B4/QMV OFF puis ON, composant SSD, annulation après soumission réelle, session B2/EOS : toutes passent, processus joints |
| Propriétés physiques | Logits F16, états retenus, hidden et quatre continuations égaux aux oracles scalaires ; tous les préfixes B2–B4, repli joint désactivé, erreurs/rollback, capacités10/20/40, F16/F32, routes répétées/variées, read-once/réutilisation ; compteurs du chemin fusionné |
| Bridge de production A/B/A | Trois réponses mesurées32 tokens et trois warmups ; mêmes tokens, mode et slots vérifiés ; arrêt et readers joints |
| Tentative Kani sur la fixture physique | **Non vérifiée** : harness ignoré non exposé, échec `Failed to match ... harness` conservé ; aucune propriété Metal/FFI/concurrence prouvée |

La qualification initiale avait aussi exécuté un harness Kani CPU réel de réutilisation : une assertion et cinq couvertures satisfaites, booléens et `usize` symboliques, sans hypothèses supplémentaires. Il ne prouve pas le nouveau raccordement GPU.

Les workflows existants couvrent PR et pushes main, Clippy/tests/build Linux/macOS et Kani CPU ; les tests sur modèle physique restent sélectionnés manuellement. Au SHA publié exact, Flash profile compatibility est vert ; Native Rust checks, Regression checks et EXL3 conversion sont en échec, Apple compatibility est annulé. [Snapshot des statuts](measurements/pr161-speed-20261108/ci-head.json). Ces statuts précèdent le correctif local et ne le valident pas ; leurs causes n'ont pas été investiguées dans cet audit de performance. Aucun run GitHub du patch local, aucun build MLX de compatibilité ni vérification des autres modèles.

Artefacts : [patch complet avec inventaire](measurements/pr161-speed-20261108/hcroute-delivery.patch), [patch Rust correspondant aux binaires](measurements/pr161-speed-20261108/hcroute-final.patch), [validation compacte](measurements/pr161-speed-20261108/hcroute-validation.json), [identité test](measurements/pr161-speed-20261108/hcroute-final-binary.json), [identité CLI](measurements/pr161-speed-20261108/hcroute-cli.json), [campagne interrompue et résultats](measurements/pr161-speed-20261108/README.md). Logs complets et binaires restent dans le worktree, `build/pr161-speed-20261108/` ; les preuves essentielles sont copiées avec ce rapport. Journal des expériences : `opti.md`.

**État :** correctif HC/routeur local validé sur ce parcours ; signal de vitesse court non concluant. Les autres coûts sont diagnostiqués et priorisés, pas implémentés. Aucun push, merge, publication ni remplacement de l'application installée.
