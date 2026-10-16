# Campagne physique LFM : protocole avant exécution

État au 16 octobre 2026 : **plan, pas résultat**. Aucun modèle, benchmark ou test
GPU n’a été lancé pour rédiger ce document. Les chemins de binaires candidats,
empreintes et prompts doivent être gelés par le coordinateur
avant la première commande. Un flag sans consommateur réel ne valide aucun chemin.

## Références et hypothèses

- `opti.md`, **LFM-01 / K02-K03a** et **LFM-02 / K04** sont les préenregistrements.
- **PERF-16** : le routeur multi-row était exact isolément, mais le prefill MoE
  complet32/64 divergeait dans les caches. Le microbloc actuel conserve opérateurs,
  normes, résiduels, gates et head M1 ; il faut prouver que ce nouveau chemin est
  réellement parcouru. Une ancienne mesure batch ou un texte identique ne suffit pas.
- **PERF-54/56**, **PERF-57/59**, **PERF-60** distinguent gates M1, experts groupés
  et TensorOps segmenté ; ne pas réactiver le chemin segmenté pour obtenir un gain.
- **AUDIT-03 du22septembre** : les heads LFM intermédiaires sont paresseux. **PERF-16**
  a rejeté le slice placé avant le head (118/65536logits F16 modifiés). Le head final
  canonique reste inchangé ici ; K05/K06 ne sont pas incorporés à cette comparaison.
- LFM2.5-24B-A2B V2compact : vocabulaire128000, hidden2048,64experts/top4,
  40couches dont38MoE,10attention et30convolutions. Les poids ne changent pas.
  Le ready physique de la campagne UI et ses hashes sont conservés dans
  `../lfm-mtp-ui-20261016/physical-ready.json` ; les revérifier avant cette campagne.
- Le rapport `../../lfm25-measurements-2026-10-14.md` contient un ancien débit
  absolu, autre binaire/checkpoint/protocole. Ce n’est pas la baseline A/B présente.
  Les KL0,428/0,323/0,157 y correspondent à des fenêtres/BOS différents, pas à une
  amélioration successive. Aucun objectif KL n’est déduit d’un changement moteur exact.

## Outillage réutilisé et limites

`benchmarks/compare_native.py` convient au débit réel bridge : un `JsonProcess`
par passe, fermé et rejoint avant la suivante ; warmup identique au prompt mesuré ;
contrôle des hashes tokens/texte/historique, nombres positifs/finis et cache réutilisé0 ;
rapport atomique intermédiaire et `failed` après erreur/divergence/interruption.
Il ne compare pas les logits ou les états. `parity:true` dans ce rapport signifie
seulement égalité de ces sorties et achèvement de toutes les passes.

La petite adaptation de cette campagne ajoute les ordres **ABBAAB** et **BAABBA**,
et un `--campaign-timeout` optionnel : échéances de chargement/requête, sondes5s et
pause entre passes sont plafonnées par le même budget absolu. Sans option, les délais
antérieurs restent inchangés. Le driver exige désormais un code de sortie0 après
fermeture et join ; une fin forcée ou nonnulle refuse la campagne, même si les
générations étaient valides. Après expiration, le cleanup `JsonProcess` conserve sa
borne existante0,25s puis terminate1s puis kill/wait5s, soit au plus6,25s d’attentes
additionnelles pour l’enfant courant ; la dernière écriture de rapport suit.
Ce budget n’est pas une garantie temps réel du système de fichiers ou de l’ordonnanceur.
Aucun sous-processus natif non joint ne doit rester avant le chargement suivant.

`benchmarks/benchmark_bridge.py` n’est **pas** utilisé pour qualifier ce lot : ses
`readline()` ne portent pas de deadline, le warmup n’a pas la forme des mesures et
il ne vérifie pas parité/cache/admission comme `compare_native`. Aucun deuxième
harnais Python n’est ajouté.

`benchmarks/eval_lfm25_quality.py` calcule fullvocab/NLL/KL correctement pour son
protocole déclaré, mais appelle **`forward` sériel sans `--batch`**. Il contourne
`prefill_moe_exact`. Un score inchangé obtenu ainsi ne valide donc pas K03. Il peut
contrôler la voie M1 de K04 seulement si le cache candidat est effectivement actif.
Il ne reproduit pas le protocole BOS/seconde moitié de l’ancienne KL0,157.

`forward --states` sérialise tous les états dans une ligne JSON : vers266tokens,
plus de5,4Mo de données K/V brutes, puis leur représentation décimale et les128000logits,
peuvent dépasser `JsonProcess.MAX_JSON_LINE=16MiB`. Ne pas sauter des états ou lever
cette garde pour obtenir du vert : utiliser le comparateur Rust direct ci-dessous.

## Gates CPU avant toute résidence GPU

Le coordinateur fige et archive : commit et diff local (pas seulement le numéro de
version), SHA256 des sources touchées, binaires et dylibs/MLX0.32.2, fichiers du
checkpoint/config/index/tokenizer, prompts UTF-8 et IDs après template. Il prépare
les exécutables release et tests physiques par `cargo test --no-run` **avant** le
créneau GPU, relève les chemins de test produits et vérifie leurs noms avec `--list`.
La commande physique utilise ensuite directement ce binaire ; aucun `cargo test`
qui pourrait compiler pendant une mesure. Aucune preuve ni compilation concurrente.

Registre à compléter avant lancement :

| Identité | Valeur prévue | État |
|---|---|---|
| Python | `/Users/justin/Documents/mix-stq1_0/.venv/bin/python` | existant, version/hash à relever |
| Modèle | `/Users/justin/Documents/mix-stq1_0/models/LFM2.5-24B-A2B-EXL3-4bpw-v2` | existant, poids inchangés |
| Baseline source connue | `0bebf7446a08db1e13879394288a8a4bacab1632` | UI/mémoire/packaging, sans K03/K04 |
| Binaire baseline existant | `dist/MLXL3 Desktop.app/Contents/Resources/runtime/mlxl3` | SHA256395fc4b6f6cab735c7e3130348c1bb42a3beaad1e86acf22e475ac68df91d8c1 selon livraison ; vérifier/copie immuable |
| Baseline figée | `build/audit-api-lfm/baseline-0bebf744/mlxl3` | créée par root, dépendances voisines et manifeste `build/audit-api-lfm/baseline.json` ; vérifier SHA avant GPU |
| Candidat K03 proposé | `build/audit-api-lfm/candidate-lfm-01/mlxl3` | chemin candidat final à confirmer après gates CPU |
| Candidat K04 proposé | `build/audit-api-lfm/candidate-lfm-02/mlxl3` | chemin candidat final à confirmer après gates CPU |
| Test Rust physique | chemin exact produit par `--no-run` | à inscrire, ne pas deviner un hash de fichier cible |
| Raccordement K03 bridge | `MLXL3_LFM_MOE_PREFILL=0/1`, défaut0 gelé par root | raccordement main/témoin effectif encore bloquant pour mesure bridge |
| Raccordement K04 | `MLXL3_LFM_KV_CAPACITY=0/1` proposé par agent | bloquant tant que flag non consommé |

Les noms de chemins proposés servent à conserver les artefacts entre rebuilds, pas
à prétendre qu’ils existent. Si root choisit d’autres chemins, mettre cette table
à jour avant le premier essai. Conserver le moteur compilé et ses dépendances côte
à côte pour que la relocalisation n’en change pas le runtime.

## Correction physique : K02 puis K03 puis K04, un seul processus à la fois

Exécuter ces tests uniquement après feu vert root, avec un superviseur borné et
preuves `exit`, timeout, durée, stderr/stdout et count nonnul. Les noms qualifiés
ci-dessous doivent correspondre au `--list` du binaire exact ; zéro test sélectionné
est un échec. La notation `$LFM_TEST_BIN` désigne le chemin enregistré, pas un build.

```sh
"$LFM_TEST_BIN" --ignored --exact router::tests::biased_router_matches_independent_sort_through_64_experts --nocapture --test-threads=1
"$LFM_TEST_BIN" --ignored --exact router::tests::biased_router_rejects_invalid_inputs_before_dispatch --nocapture --test-threads=1
```

K02 : oracle CPU indépendant de tri des bits F16,13lignes, E1/2/31/32/33/63/64,
K1/4/full, lignes distinctes/ties0/31/32/63, scores/IDs dans l’ordre contractuel.
Mutation d’un offset de ligne doit être détectée et source restaurée avant la suite.
Délai120s maximum pour la fixture routeur ; aucune mesure de vitesse de ce test.

K03 : `MLXL3_LFM_TEST_MODEL` pointe sur V2 ; `MLXL3_LFM_TEST_LENGTH` sélectionne un
cas, `MLXL3_LFM_TEST_PREFIX` vaut0ou17. Commencer par8/0, puis matrice1..8,
23/24/25/63/64/65/128/266 × préfixes0/17, **60s par cas**. La sélection absente
parcourt l’union mais ne fournit pas seule une borne par cas ; le coordinateur
supervise chaque sélection lorsqu’une borne stricte est requise.

```sh
MLXL3_LFM_TEST_MODEL="$LFM_MODEL" MLXL3_LFM_TEST_LENGTH=8 MLXL3_LFM_TEST_PREFIX=0 "$LFM_TEST_BIN" --ignored --exact lfm2::prefill_tests::lfm_moe_microblocks_match_serial_logits_routes_states --nocapture --test-threads=1
MLXL3_LFM_TEST_MODEL="$LFM_MODEL" "$LFM_TEST_BIN" --ignored --exact lfm2::prefill_tests::lfm_moe_route_capture_is_passive --nocapture --test-threads=1
```

Référence A = `forward([token])` canonique, reset, B = `prefill_moe_exact` sur les
mêmes IDs. Comparer les **128000logits F16 au point terminal du prefill**, puis les
17continuations imposées ; ce test ne relève pas les logits de chaque position
intermédiaire et ne calcule pas une PPL de tout le prefill. Vérifier formes, dtypes,
nombre et finitude ; comparer **50 états nommés** (20 K/V + 30 conv) et toutes les
routes/affinités des 38 MoE aux points prévus et pendant les continuations forcées. Trace passive ON/OFF128 doit conserver sorties/états exacts. Vérifier les
témoins `tokens=T`, `blocks=ceil(T/8)`, `max_block=min(8,T)`, `router_rows=38*T`,
`grouped_moe_calls=38*nombre_de_blocs_de_longueur_supérieure_à1`. Un test M1 qui ne
parcourt aucun expert groupé ne remplace pas les cas2..8.

K04 : d’abord cache synthétique avec oracle concat CPU, puis modèle. Les noms de
module proposés doivent être confirmés par `--list` ; délai120s par fixture.

```sh
"$LFM_TEST_BIN" --ignored --exact lfm_cache::gpu::tests::capacity_cache_matches_cpu_and_reuses_storage --nocapture --test-threads=1
"$LFM_TEST_BIN" --ignored --exact lfm_cache::gpu::tests::capacity_cache_preserves_aliases_and_delayed_branches --nocapture --test-threads=1
"$LFM_TEST_BIN" --ignored --exact lfm_cache::gpu::tests::capacity_cache_full_strided_input_and_validation --nocapture --test-threads=1
"$LFM_TEST_BIN" --ignored --exact array::tests::slice_update_preserves_exact_bits_aliases_and_storage --nocapture --test-threads=1
"$LFM_TEST_BIN" --ignored --exact array::tests::slice_update_rejects_invalid_metadata_without_mutation --nocapture --test-threads=1
```

Sans alias retenu : identité de stockage stable hors croissance256, avec handle
réellement remplacé et anciennes vues abandonnées avant `eval`. Avec alias : ancien
contenu inchangé, COW et branche indépendante, appends différés/reset. Dtypes,
formes/axes/budget invalides refusés. Un simple appel `slice_update` réussi ne prouve
pas la donation ; si chaque append recopie la capacité, ne pas passer au benchmark.
Le module ne dispose pas encore d’un snapshot de session LFM et n’en crée pas ici.

Après primitive valide : même modèle, même chunking/opérations qu’après décision
LFM-01, K04OFF puis K04ON, fullvocab+50états exacts,23/24/255/256/257, continuation,
reset et erreurs.4095/4096/4097 seulement si exécutable de test et budget réellement
prévus : sinon marquer long contexte non vérifié. Préserver causalité et spécialisation
SDPA malgré strides/capacité. Le test modèle K04 attend son raccordement et son nom.

## Mesure bridge après toutes les gates

Prompts à figer : **FR266, code266, FR environ600, code environ600 tokens après le
vrai template**, pas une estimation caractères. Fichiers proposés sous
`build/audit-api-lfm/prompts/{fr-266,code-266,fr-600,code-600}.txt`. La tokenisation CPU
peut les préparer sans modèle GPU ; relever IDs/SHA et le vrai `stats.prompt_tokens`.
Utiliser les mêmes fichiers pour chaque variante. Pas de MTP/DFlash/MCP, greedy
(temp0, top_k0, répétition1), contexte4096, usage266/600+≤48, cache de préfixeOFF. Le driver transmet explicitement `--context-length4096` ; une annotation intermédiaire évoquant131072 était erronée et son contrôle a interrompu la première campagne. Cette erreur et ses résultats sont conservés dans le journal. Fin naturelle avant48
est conservée et identique dans les deux variantes ; ne pas ajouter silencieusement
une contrainte de génération.

L’outil exécute aussi un prompt court fixe auxiliaire (`short`) ; le fichier demandé
est le résultat **`document`**. Ne pas présenter le court fixe comme le cas266/600.
Chaque passe recharge une seule variante puis effectue1warmup de48tokens et1mesure
de48tokens par forme. Six passes **ABBAAB** donnent3paires AB,BA,AB et3échantillons
mesurés par variante ; `--repeats 1` est donc intentionnel. Les fichiers FR/code sont
quatre appels séquentiels soumis à une **même échéance externe de 900 s** pour la
série. Le coordinateur relève le temps restant avant chaque appel et le transmet
à `--campaign-timeout` ; il ne réinitialise pas 900 s pour chaque fichier. La
confirmation BAABBA éventuelle est une série séparée, prévue avant ses résultats.

Commande K03 après raccordement root du flag gelé (défaut0). Le candidat doit
attester le microbloc actif ; la baseline figée ancienne ignore ce flag et conserve
sa voie M1 canonique. `compare_native` applique les overrides par variante :

```sh
cd /Users/justin/.codex/worktrees/lfm-mtp-ui-state/mix-stq1_0
LFM_PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
LFM_MODEL=/Users/justin/Documents/mix-stq1_0/models/LFM2.5-24B-A2B-EXL3-4bpw-v2
LFM_BASE=build/audit-api-lfm/baseline-0bebf744/mlxl3
LFM_K03=build/audit-api-lfm/candidate-lfm-01/mlxl3
LFM_K04=build/audit-api-lfm/candidate-lfm-02/mlxl3
LFM_PROMPT=build/audit-api-lfm/prompts/fr-266.txt
LFM_BUDGET=900
"$LFM_PY" benchmarks/compare_native.py "$LFM_MODEL" --baseline "$LFM_BASE" --candidate "$LFM_K03" --baseline-env MLXL3_LFM_MOE_PREFILL=0 --candidate-env MLXL3_LFM_MOE_PREFILL=1 --output docs/measurements/audit-api-lfm-20261016/lfm-01-fr-266-abbaab --prompt-file "$LFM_PROMPT" --tokens 48 --warmup-tokens 48 --mtp-depth 0 --dflash-mode 0 --repeats 1 --order ABBAAB --settle-seconds 2 --load-timeout 120 --request-timeout 120 --campaign-timeout "$LFM_BUDGET"
```

Vérifier **le témoin de branchement du candidat**, pas la seule présence du flag
dans le JSON du driver. Pour code266/FR600/code600,
changer fichier, nouveau répertoire de sortie et **budget restant de la même série** ;
aucun lancement parallèle. `LFM_BUDGET=900` dans cet exemple représente seulement
la borne du premier appel, à réduire du temps déjà écoulé avant les suivants.
Tous les overrides MLXL3 hérités sont relevés et nettoyés ou figés avant le lancement,
car l’outil archive les overrides explicites mais pas tout l’environnement parent.
`MLXL3_HOME` doit être isolé dans un dossier jetable partagé par les deux variantes ;
aucun registre, conversation, calibration Tune ou paramètre utilisateur réel modifié.

K04 doit comparer le **même exécutable** avec uniquement son flag qui change, à K03
fixé dans l’état finalement conservé. Si K03 est retenu, passer aussi explicitement
`--baseline-env MLXL3_LFM_MOE_PREFILL=1 --candidate-env MLXL3_LFM_MOE_PREFILL=1` ;
si rejeté, figer0 sur les deux variantes et l’indiquer dans le manifeste. Commande conditionnelle au flag implémenté :

```sh
"$LFM_PY" benchmarks/compare_native.py "$LFM_MODEL" --baseline "$LFM_K04" --candidate "$LFM_K04" --baseline-env MLXL3_LFM_KV_CAPACITY=0 --candidate-env MLXL3_LFM_KV_CAPACITY=1 --output docs/measurements/audit-api-lfm-20261016/lfm-02-fr-266-abbaab --prompt-file "$LFM_PROMPT" --tokens 48 --warmup-tokens 48 --mtp-depth 0 --dflash-mode 0 --repeats 1 --order ABBAAB --settle-seconds 2 --load-timeout 120 --request-timeout 120 --campaign-timeout "$LFM_BUDGET"
```

Ready/statistiques ou autre témoin gelé doit attester K03 actif sur B et K04OFF/ON
respectivement ; aucun compteur de diagnostic n’est laissé actif dans la mesure
sans vérifier son coût identique. Les prompts et binaires sont hashés par l’outil,
mais les poids/dépendances/sources nécessitent le manifeste complémentaire figé.

## Lecture des mesures et décision

Relever par passe : `prefill_seconds`/tok/s, TTFT, decode_seconds/tok/s, durée totale,
ready/load hors débit chaud, prompt/evaluated/generated/cached tokens, hashes et
mode effectif. Garder les observations individuelles et les trois rapports de
paires avant la médiane. L’outil conserve les conditions power/thermal/swap avant
chaque passe ; relever après campagne et préciser température/fréquences physiques
non mesurées si l’outil ne les fournit pas.

Pour chaque métrique positive, calculer les3rapports B/A des paires(0,1),(2,3),(4,5)
en respectant leurs labels inversés, puis médiane ; publier aussi les3valeurs A/B.
Dérive par variante = écart premier/dernier échantillon / premier, plus plage
(max-min)/médiane à titre de contrôle. Critères du préenregistrement : dérive≤3%,
pas de régression decode>3%, parité de sortie et numérique vérifiée. Un gain≥3%
exige confirmation **BAABBA** dans un nouveau dossier avec mêmes sources/prompts et
critères ; une dérive supérieure rend le résultat non concluant, même si la médiane
semble rapide. Ne pas additionner K03 et K04 ; qualifier ensuite la combinaison
finalement conservée contre une référence identifiée.

Mémoire : valeurs de `stats.memory` déjà archivées (MLXactif/cache/pic, footprint,
pic lifetime), modèle sur disque distinct. Relever longueur/capacité/backing des
K/V lors du contrôle de correction, sans instrumentation propriétaire supplémentaire
qui empêche la donation dans le timing. Surcoût paliers V2 maximum prévu5222400octets ;
une métrique mémoire non exposée reste **non mesurée**, pas0. Les pics lifetime
incluent warmup/ancien prompt du même processus et doivent être décrits comme tels.

Toute divergence, NaN, forme/état manquant, compteurs de chemin nuls, timeout ou
processus anormal arrête immédiatement la variante ; garder le rapport `failed`,
les sorties partielles et stderr, puis fermer/rejoindre avant la suivante.
Pas de moyenne gagnante calculée sur les seuls cas réussis. Un timeout est un essai
interrompu, ni un contre-exemple numérique ni un succès.

## Complément PPL/KL M1 si demandé après parité moteur

Le BF16 source et le corpus existants sont utilisés séquentiellement par l’évaluateur ;
il ne charge pas deux modèles GPU ensemble. La teacher est couche par couche puis
libérée avant chaque enfant natif. Cette commande est **complémentaire et différente**
de l’ancienne mesure BOS, et ne qualifie pas K03 :

```sh
"$LFM_PY" benchmarks/eval_lfm25_quality.py /Users/justin/Documents/mix-stq1_0/models/source/LFM2.5-24B-A2B "$LFM_MODEL" --native-binary "$LFM_K04" --data /Users/justin/Documents/mix-stq1_0/benchmarks/data/wikitext-2-raw/wiki.test.raw --work-dir build/audit-api-lfm/quality-m1-new --output docs/measurements/audit-api-lfm-20261016/quality-m1-new.json --tokens 2048 --window 256 --timeout 120
```

Lancer séparément référence et candidat, dossiers neufs, mêmes teacher/tokenIDs/BOS/
fenêtres/labels. Cet outil recalcule la teacher à chaque exécution ; éviter cette
campagne coûteuse si les128000logits de la voie affectée sont déjà bitexacts et que
son seul objectif serait de les comparer encore par une métrique plus faible.
Aucune nouvelle quantification, Modal, modification de poids ou promesse de baisse
KL n’appartient à ce lot.


## Adaptation du driver : résultats CPU conservés

Le driver et les régressions ont été lus avec `formal-proof-skill` et sa référence
Python ; aucune dépendance ajoutée. La CI `rust.yml` couvre déjà ces fichiers sur
PR et push ; son run du futur commit doit encore être inspecté par root.

Avant correctif, le vrai CLI refuse `ABBAAB` et `--campaign-timeout` (exit2) ; les
deux nouveaux tests d’ordres sont rouges. Après correctif, la suite pertinente
(driver, JsonProcess, ancien bridge et évaluateur qualité) passe **108 tests** sans
skip. Les 11 nouveaux cas sont aussi exécutés seuls après renforcement des assertions.
Une première reprise a échoué sur 2 attentes d’étape : sous charge CPU, le budget
de 0,4 s expirait déjà avant ready ; le driver arrêtait correctement. Cet échec
est conservé. Budgets de test passés à 2 s et silences à 3 s (sonde interpasses
3 s / silence10 s), sans changer le driver ; reprise finale **11 passés**,14,20 s.
Une mutation retirant seulement `min(campaign_deadline)` est détectée : un ready
arrive après le budget global alors qu’il doit être refusé. Source restaurée exacte.

Ruff strict, format et `py_compile` passent. CrossHair a exploré le vrai helper
`bounded_deadline` : **une postcondition analysée**, retour≤deadline globale lorsque
présente, marquée « Confirmed over all paths » dans son modèle. Borne configurée20s
par condition, superviseur30s ; ni l’I/O, ni les processus, ni l’horloge système
complète ne sont prouvés. Les tests CLI restent des vérifications échantillonnées.

Commandes, sorties rouges/mutation/vertes et empreintes sources sont conservées sous
`build/audit-api-lfm/benchmark-tool/`. Aucun modèle GPU ou benchmark physique exécuté
par cette adaptation. Root actualise `opti.md` après réception des résultats.


### Complément préenregistré avant essai : code de sortie du bridge après EOF

La revue indépendante trouve une limite héritée : `JsonProcess.close()` récole le
bridge, mais le driver ne contrôle pas son code de sortie après les réponses
valides. Un enfant qui finit par exit1 ou crash après le dernier `complete` pourrait
être marqué `complete/parity:true`. Avant la campagne, ajouter au seul driver le
contrôle `returncode==0` **après fermeture et join**, enregistrer ce code, puis
refuser une terminaison anormale (y compris forcée) sans masquer une erreur de
requête déjà levée. La politique de qualification de cette campagne exige un
arrêt normal ; un arrêt forcé n’est pas une preuve complète de santé du runtime.

Régression prévue : vrai faux bridge qui émet toutes les réponses valides puis
sort1 à EOF, et variante qui reste bloquée à EOF puis est terminée par le cleanup
existant. Vérifier rapport `failed`, absence de faux `parity:true`, réponses utiles
conservées et aucun enfant survivant ; ancien driver rouge, fix vert. Les tests
CPU attendent la fin du créneau de compilation root pour stabiliser les délais.
Aucune modification de `native_json_process.py`, aucun nouveau GPU ni proveur.


### Résultat CPU final du complément EOF et du workflow

Les deux régressions du vrai CLI ont été exécutées sur une copie isolée de l’ancien
driver : **2 échecs attendus**, car il produisait exit0 et `parity:true` après un
exit1 ou une terminaison forcée à EOF. La source active n’a jamais été permutée.
Sur le driver corrigé : **110 tests pertinents passés, 0 ignoré**,51,26s ; puis les
**13 cas affectés passés /44 désélectionnés**,14,82s. Ces suites couvrent le driver,
JsonProcess, l’ancien bridge et l’évaluateur qualité ; elles ne constituent pas
la suite complète du dépôt. Les réponses utiles sont conservées dans le rapport
`failed`, le code de sortie1 ou SIGTERM est enregistré, `parity` reste null et aucun
résumé global n’est certifié.137PID enfants de fixtures contrôlés : aucun vivant.

Ruff strict, format, `py_compile` et `git diff --check` passent. Le workflow Rust
a été chargé via YAML BaseLoader (clé `on` conservée) : triggers PR/push, permissions
lecture, harnais Kani API réel et7 tests CPU MLX existants avec sélection positive.
Ses13 blocs `run` passent `bash -n`. Cela ne remplace ni l’exécution Kani/MLX locale
du coordinateur ni les jobs GitHub du futur commit exact.

Le helper `bounded_deadline` est byteidentique à la source du contrôle CrossHair
historique, SHA256 `bf37e534335754f149edcfcbf5fad9f451052cc079a853485a4e939e0edf33f6`.
Ce résultat est réutilisé sans relancer le proveur : la nouvelle logique EOF dépend
de l’I/O et des processus, hors sous-ensemble pur praticable ici. Aucune preuve
complète de l’I/O, du cycle de vie des enfants ou du système n’est revendiquée.
Le délai EOF normal reste0,25s : une destruction saine plus lente peut être rejetée,
sans être faussement qualifiée.

Voir le [rapport final de l’outillage](benchmark-tool.md), le
[manifeste et empreintes](benchmark-tool-verification.json) et l’
[archive brute](benchmark-tool-evidence.tar.gz), dont chaque entrée décompressée
a été comparée byte à byte. Les échecs de timing initiaux et la mutation détectée
restent conservés. Aucun modèle ni test GPU lancé par cette adaptation.
