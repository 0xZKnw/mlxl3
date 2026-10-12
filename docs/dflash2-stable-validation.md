# DFlash2 — support stable, préparation et Tune

État : implémentation locale sur `codex/dflash2-stable`, initialement basée sur
`9f2a572e22a1560399ac8d77a3088573ed8e74cb`, puis synchronisée avec le main
`db52b7304a057ad8f0d8fcf464cd66b22b44fe08` (correctif RAM PR29).
App et moteur **1.4.3, build 25**, construits et contrôlés localement.
La comparaison finale MTP/DFlash est terminée ; son contrôle de dérive limite
la conclusion de vitesse. L'app installée et les releases n'ont pas été remplacées.
Le checkout personnel et ses changements MTP antérieurs sont préservés.

## Parcours livré dans le code

Le switch Generation → DFlash2 sélectionne le draft selon la configuration
réelle du modèle chargé, le télécharge avec reprise et SHA-256 épinglés,
puis attend son chargement par le bridge résident. Les paramètres greedy
s'appliquent automatiquement. OFF libère le draft et le cache ; une réponse
erronée ou silencieuse libère aussi le composer après un délai borné à30s.
Le changement de cible invalide la préparation précédente. Le réglage ON/OFF
persiste ; les profils de Tune sont séparés du MTP et invalidés quand les
fichiers du draft, le moteur, le contexte ou le modèle changent.

Tune réutilise le moteur et les contrôles MTP : baseline, Auto adaptatif2/5,
fixe2, fixe7 ; warmup32tokens puis un prompt de code et un prompt français
96tokens en ordres inversés. MTP conserve ses deux prompts de code antérieurs.
Les IDs doivent être identiques à la cible. Le score exclut acceptation nulle,
mesure trop courte ou divergente ; un gain≤3% conserve la baseline. Annulation
et résultats invalides conservent le réglage précédent. Les propositions sont
toujours vérifiées et les caches engagés dans l'ordre exact de la cible.

## Checkpoints vérifiés

| Cible | Draft publié | Captures cible | Géométrie draft | Paquet épinglé |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | [IncoAI DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) | 1,6,11,16,22,27,32,37 | hidden2048,6couches,FFN6144,mask248077 | [Splash](https://huggingface.co/incoai/Qwen3.6-35B-A3B-Splash/tree/0f4714b2db37b5f3c42a10de07281e74f88e4adc/draft),479444992octets |
| Qwen3.8-27B | [Configuration IncoAI](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json) | 5,19,33,47,61 | hidden5120,5couches,FFN17408,mask248070 | [Splash](https://huggingface.co/incoai/Qwen3.8-27B-Splash/tree/9d27070b71f7142c6b6025f03ac011d70a73cb48/draft),1266040832octets |

La recherche du7octobre n'a identifié aucune nouvelle tête DFlash2 35B
officielle démontrée supérieure : l'historique du dépôt BF16 comporte une
seule publication « DFlash 2 release ». La variante
[mlx-community4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit)
est une autre représentation de ce draft. Aucun gain de qualité/acceptation
du BF16 ou de cette conversion n'a été mesuré ici. Les métadonnées actuelles
et les commits sont archivés sous `measurements/dflash2-stable/` ; une ancienne
révision citée dans l'historique retourne404, donc son égalité au checkpoint
actuel n'est pas affirmée.

Le lecteur paramètre les dimensions, les captures et la convolution deux taps
avec la famille. Le shader conserve le calcul et l'ordre BF16. Le downloader
ne charge que les six/sept fichiers draft, sans poids cible Splash.

## Optimisations et mesures

Les projections Q4, kernels KV-only, suffixe utile de contexte, commit exact
et adaptation à l'acceptation étaient déjà présents ; aucun gain historique
n'est revendiqué comme nouveau. Les nouveaux essais sont préenregistrés et
suivis dans [opti.md](../opti.md), avec résultats négatifs conservés.

| Essai35B / M5 Air24Go, batterie | Résultat | Décision locale |
| --- | --- | --- |
| Consolidation des appends KV, ABBA court/document2825 | TTFT−1,45%/−0,11%, decode document−1,66%, parité vraie, dérives≤2,82% | Rejeté sous le seuil3%, ancien append sériel restauré |
| Pipeline cible existant, opt-in1 contre0, fixe7 | Qualité bit exacte ; contrôles vitesse dérivent3,17–5,26%, dernière passe interrompue | Non concluant, pipeline DFlash reste OFF par défaut |
| Auto2/5/7 après huit blocs à≥75% d'acceptation, ABBA français/code | Français62,02→61,57tok/s (−0,73%), code70,57→68,73tok/s (−2,61%), parité vraie, dérives≤0,454% | Rejeté, Auto2/5 conservé ; fixe7 reste disponible |

Ces essais sont distincts ; leurs résultats ne s'additionnent pas. Les sources,
binaires et diagnostics rejetés restent conservés, sans servir de preuve du code
final. L'essai cache27B a été interrompu à la demande de l'utilisateur : rapport
failed/paritynull, A terminé et B partiel, aucun gain dense établi. Aucun nouveau
modèle27B n'a été chargé après cette instruction.

Sur le moteur final signé, la calibration DFlash français/code choisit Auto :
baseline **50,81**, Auto **65,14**, fixe2 **62,49**, fixe7 **54,63 tok/s**.
Chaque mode produit les mêmes deux hashes non vides et190tokens de decode.
Le Tune MTP, sur ses deux prompts de code inchangés, choisit profondeur1 :
baseline **50,50**, MTP1 **73,68**, MTP2 **69,28**, MTP3 **67,32 tok/s**.
Ces calibrations n'ont pas les mêmes prompts ; leurs débits ne sont donc pas
une comparaison directe. Les anciennes calibrations et celle du prototype
Auto2/5/7 retiré restent archivées avec leurs sources.

### Comparaison directe du moteur signé

Même moteur, cible EXL3 2,49 bpw, contexte 4096, greedy, cache OFF, lookup OFF,
MTP stock profondeur 1 contre DFlash Auto 2/5. Warmup de 128 tokens par prompt
et passage, deux répétitions de 128 tokens, ordre MTP/DFlash/DFlash/MTP.
Le pipeline garde ses défauts livrés : MTP M5 ON et DFlash OFF.

| Prompt | MTP decode | DFlash decode | TTFT MTP / DFlash | Durée complète MTP / DFlash |
| --- | --- | --- | --- | --- |
| Français court, 27 tokens | 70,52 tok/s | 61,28 tok/s | 127,30 / 137,55 ms | 1,9284 / 2,2102 s |
| Document technique, 2825 tokens | 56,64 tok/s | 42,55 tok/s | 5,7635 / 6,2892 s | 8,0096 / 9,2745 s |

Les 24 générations, warmups compris, produisent chacune 128 tokens avec hashes
de tokens, texte et historique non vides identiques. Les compteurs de chaque
algorithme restent identiques entre répétitions : court MTP 57/69 acceptés en
69 blocs, DFlash 91/136 en 35 blocs ; document MTP 52/74 en 74 blocs, DFlash
80/147 en 47 blocs. Une acceptation totale supérieure ne suffit donc pas à
établir un avantage de vitesse ; le travail par bloc diffère.

**Vitesse non concluante selon le protocole préenregistré** : dérives decode
court MTP −1,68 % / DFlash +0,31 %, document MTP −5,32 % / DFlash −2,74 %.
Le contrôle MTP document dépasse la limite de 3 %. MTP est observé devant dans
ces passes, mais l'écart n'est pas qualifié comme résultat confirmé ou
classement universel. Aucune confirmation BAAB n'a été lancée puisque son
critère préalable n'est pas satisfait. Aucun boost DFlash nouveau établi.

Batterie en décharge, 29/27/26/25 % aux quatre relevés ; température et
fréquences non mesurées, aucune autre inférence, compilation ou preuve.
Pic d'allocation MLX et empreinte physique restent distincts : médianes
document MTP 13,185 / 14,858 Go, DFlash 13,212 / 15,604 Go. Ces valeurs ne
certifient pas un gain mémoire, et ne s'ajoutent pas aux anciens essais.
[Décision et conditions](measurements/dflash2-stable/comparison-final-143-decision.json),
[résultats bruts](measurements/dflash2-stable/compare-final-143-abba/results.json),
[protocole, commande et identités](measurements/dflash2-stable/comparison-final-143-plan.json).

Le nettoyage RAM déjà fusionné dans main est conservé et appliqué aussi après
le nouveau Tune DFlash, succès/erreur/annulation. Les checkpoints de préfill
restent réutilisables ; son gain mémoire historique appartient à
[PR29](https://github.com/0xZKnw/mlxl3/pull/29), pas à un nouveau kernel DFlash.
Le coût de réallocation n'est pas inféré de ce gain de RAM.

Les chiffres DFlash2 devant MTP de la
[fiche officielle](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2#evaluation)
portent sur SGLang/GB300, sept propositions, sampling/thinking et4096tokens.
Ils ne prédisent pas le débit greedy/EXL3 de ce Mac.

## Vérification et limites

- Rust CPU `cargo test --locked --features chat` :60passés/3ignorés.
- Python `python -m pytest -q` :360passés/4sauts historiques (PonyExl3 absent,
  modèle Ling absent). Les tests de transport emploient des processus et
  fichiers jetables ; silence, JSON invalide, EOF, divergence, sorties vides,
  nombres non finis et rapport interrompu ne certifient jamais une parité.
  Les cinq régressions finales du benchmark échouent avant correction : texte
  vide, historique vide/absent/nonstring et historique divergent malgré les
  mêmes IDs/texte. Le driver corrigé compare aussi le SHA de l'historique.
- Kani0.68.0 `cargo kani --lib --no-default-features` : suite41/41 sur la
  première politique finale Auto2/5 (5607SUCCESS/104covers), puis41/41 sur le
  prototype Auto2/5/7 (5615SUCCESS/106covers).72obligations internes à Rust
  std et aux intrinsics Kani inatteignables, aucune assertion applicative
  inatteignable, aucun échec/indéterminé. Les deux nouveaux harnais couvrent
  les deux géométries et les budgets `usize` complets sans hypothèse, largeur
  maximale7 et réserve d'une place cible.144obligations/6covers initiales.
  Mutations maximum8 et seuil75% remplacé par2/3 détectées ; restauration
  byte exacte et relance ciblée. La politique2/5 est restaurée après son
  crible : les preuves du prototype ne sont pas attribuées au défaut final.
  Avec RAM,42harnais sont disponibles : les41corps CPU antérieurs sont
  inchangés et le seuil512Mio u64 a passé sa preuve ciblée/quatrecovers.
  La suite42 complète n'a pas été répétée ; les trois harnais concernés
  passent chacun1/1 sur la version finale (modes, géométrie, seuilRAM).
  Ces trois relances vérifient respectivement 114/30/4 obligations et
  4/2/4 couvertures. Les budgets et compteurs `usize`, ainsi que le cache
  inutilisé `u64`, sont symboliques sur leur domaine entier ; les géométries
  sont les deux familles publiées. Cela ne modélise pas l'allocateur MLX.
  Aucun assume sur ces domaines scalaires.
- GPU Apple M5/24Go, MLX0.32.2 : captures/vérification/commit MoE et dense
  passent chacun24vérifications/108commits,80/128états finis bit exacts ;
  logits contre exécution token-major, états contre appels mono-token.
  Les contrôles supplémentaires finitude/nombre de logits/captures et le
  pipelineMoE1 contre oracleOFF passent également. Convolution2048 et5120 :
  14sorties chacune bit exactes contre référence scalaire BF16, erreurs grid/
  dtype couvertes, boucles chronométrées désactivées.
- Bridge MoE réel sans Tune :41requêtes réussies, parité non vide de tokens
  et historique, budgets1/2/3/8/17/64, trois politiques, erreurs mode/MTP/
  pénalité/draft absent ou étranger, annulation et reprise, cache256 et miss.
  Bridge dense identique41/41 avant la suspension des nouveaux essais27B.
  Bridge35B avec Tune avant RAM45/45, puis avec RAM/candidat45/45. La
  version finale signée1.4.3 passe **45/45 DFlash et43/43 MTP**, Tune inclus,
  après retrait du prototype. Rapports `bridge-moe-final-143.json` et
  `mtp-moe-final-143.json`. Le nettoyage RAM et le nouveau Tune27B restent
  non vérifiés physiquement, conformément à la suspension demandée.
- Desktop Swift6/SDK26.5 : suite finale complète réussie, avec erreurs,
  réponse malformée et silence30s sur OFF. Le SDK27 local échoue sur
  des macros SwiftUI préexistantes ; diagnostic conservé.
- Rust release `mlx,chat` :**73passés/62ignorés** sur la version finale ;
  précédemment72/62 puis74/62 avec le prototypeAuto2/5/7 retiré.
  Contrôles GPU sélectionnés
  explicitement et seuls. fmt et Clippy strict `mlx,chat` passent.
  CI PR/push inspectée et étendue aux nouveaux scripts et
  tests ; les3jobsNative et1Desktop de la base9f2a572e passent, sans valider
  les changements locaux. Sur main/db52, Desktop et les trois jobs Native
  passent également (`main-native-ci-final.json`). Aucun run GitHub de ces changements locaux n'est
  disponible. Les checks DFlash et RAM sont conservés ensemble dans la CI.

Kani vérifie les contrats CPU bornés, pas MLX/Metal/FFI, la concurrence ni la
génération entière. CrossHair a été tenté mais n'est pas installé. CBMC6.11.0
ne reconnaît pas le fragment Metal `.metal`. Shader et Swift restent sans
preuve source formelle ; compilation, tests numériques physiques et E2E
constituent des contrôles distincts, pas une preuve complète.

Les logs, configurations publiées, empreintes et rapports bruts sont dans
[`measurements/dflash2-stable/`](measurements/dflash2-stable/). Les binaires
de comparaison et leurs sources avant optimisation restent des artefacts
locaux ; ils ne font pas partie d'une release. Les durées des tests de
correction exécutés pendant des compilations sont diagnostiques seulement.

## Commandes et artefacts

Les commandes principales, lancées dans le worktree avec Rust1.98.1,
Python3.12.14, MLX0.32.2 et Swift6/SDK26.5, sont :

```sh
export PATH=/Users/justin/.cargo/bin:$PATH
export MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --release --features mlx,chat
cargo build --locked --release --features mlx,chat
cargo kani --lib --no-default-features
python -m pytest -q
scripts/check-desktop.sh
scripts/build-macos-app.sh
python scripts/check_dflash2.py ENGINE TARGET DRAFT --foreign-draft FOREIGN --output REPORT
python scripts/check-mtp-depths.py ENGINE TARGET HEAD --tune
```

Le bundle contrôlé est `dist/MLXL3 Desktop.app` ; l'archive moteur est
`dist/MLXL3-Engine-v1.4.3-arm64.tar.gz`. La signature adhoc, l'architecture
arm64, les cinq membres du manifest et les SHA archive/bundle concordent.
`runtime-info` confirme1.4.3, `db52b7304a05-dirty`, release, MLX et chat activés.
Le SHA-256 du moteur signé est
`74ba7a825174d314c822715f25ef5f12a15647f86868f1c89b8ac49f8d61be16`.
Identité et empreintes sources : `final-package-143.json`.

L'updater a installé, relocalisé et exécuté cette vraie archive dans des
répertoires jetables, puis vérifié rejet et fallback ; timeline/MCP release
et20tests packaging passent. Le numéro1.4.3 permet de préférer ce bundle à
un ancien moteur managed1.4.2. Ce contrôle n'a pas modifié l'app personnelle.

Les options des campagnes128tokens/warm128/deuxrépétitions/ABBA, flags,
conditions pmset/swap/therm, empreintes des binaires et prompts sont dans chaque
`results.json`. Le rapport de checkpoints précise les grands poids non hashés ;
les paquets draft téléchargés ont tous été vérifiés contre leurs SHA épinglés.
Les logs de correction exécutés pendant des compilations ne mesurent pas la
vitesse. Les campagnes sont closes, y compris les interruptions conservées.
Le support des deux familles est intégré localement ; la nouvelle qualification
Tune/nettoyage RAM et les optimisations 27B restent différées à la demande.

Preuves brutes archivées sans perte pour publication : chaque chemin original
dans les anciens rapports JSON est répertorié dans `raw-evidence-index.json`
et récupérable dans `raw-evidence.tar.gz` de son dossier de mesures. Les sources
historiques et les rapports originaux restent également dans ces archives.
