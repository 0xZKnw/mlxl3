# Desktop1.4.8 — vérification du paquet final

Source compilée `e032a8e814375dea3bedc529930e1918ff401306`,32sources Swift identiques aux harnais et au candidat E2E complet. Moteur public1.4.8 byteexact, aucune recompilation du moteur.

`PYTHONPATH=src scripts/check-e2e.sh` :840testsPython réussis/115sautés et suiteDesktop réussie. Préflight courant du DMGfinal et ancienupdaterv1.3.0 réussis. Les commandes détaillées sont conservées localement dans `compile-commands.json` et `release-validation.json`. Le relevé public contient seulement les identités publiques, résultats agrégés et liens CI.

Swiftstrict/SIL : contrôle de compilation, pas preuveformelle SwiftUI.786cas finis/63images/scroll natif/9formules×2styles, paritéworker63 et régressions Unicode/clipboard/annulation/historique passent. Contrôleur41cas sans fenêtre, mutation de garde/fail-closed, Ruffstrict réussis ; CrossHair absent.

## CPU et ordonnancement UI

11cas×8passes,24appends mesurés après6warmup. CPU et footprint du processus UI, p95 du retard d’arrivée des callbacks, pas FPS/latence pixel.

|Cas|CPU avant(s)|CPU après(s)|ÉcartCPU|p95 avant(ms)|p95 après(ms)|
|---|---:|---:|---:|---:|---:|
|prose 64Kio 50ms|0.5011|0.6250|+24.7%|7.59|7.20|
|prose 64Kio slow|0.8796|0.7871|-10.5%|10.41|8.26|
|prose 64Kio burst|0.4497|0.5081|+13.0%|12.97|8.19|
|code 64Kio 50ms|0.4105|0.3794|-7.6%|5.15|7.11|
|thinking 64Kio 50ms|0.8380|0.8272|-1.3%|7.93|8.12|
|prose 256Kio 50ms|0.8023|0.7224|-10.0%|8.48|7.46|
|code 256Kio 50ms|0.4977|0.5259|+5.7%|6.39|7.16|
|thinking 256Kio 50ms|0.9006|0.9284|+3.1%|8.40|9.30|
|prose 1024Kio 50ms|0.5195|0.7541|+45.2%|7.19|7.76|
|code 1024Kio 50ms|1.1280|1.1859|+5.1%|12.65|12.50|
|thinking 1024Kio 50ms|0.8239|0.8246|+0.1%|8.59|6.75|

La médiane du CPU total par passage, pour les 11 cas, passe de 7,536881 à 8,0340415 s (+6,596 %). Aucun gain CPU global revendiqué.

## Génération

8passesABBA+BAAB,48tokenswarmup puis2×128tokens, contexte4096,temp0,topK1,MTP1,MCPoff.35B2,49bpw+tête4bits, moteur fixé et un processus à la fois. Les16sorties et compteursMTP sont exacts/nonvides.

- Prompt0 :65.758456→65.162602tok/s (-0.906%). PairesB/A% :[14.720270932004297, 1.673461871124604, -1.0157220607118544, -3.6160479783471255]. Dérives interblocs :{'a': 13.337002606287207, 'b': 2.8644880136427187}.
- Prompt1 :67.606316→68.880986tok/s (+1.885%). PairesB/A% :[4.734917245437753, 2.36555349508496, -1.46877852053523, -1.4615161131799836]. Dérives interblocs :{'a': 1.8055872985869525, 'b': -3.1045172293327927}.

La cible≤1% reste non concluante : dispersion/dérive et faible effectif. Aucun gain de débit établi. Anciennes mesuresAPP02 séparées, rejet de visibilitéinitial conservé.

## CI et limites

[Desktop exactsource SUCCESS](https://github.com/0xZKnw/mlxl3/actions/runs/38953567273),840passés/115sautés et natifcomplet ; conversion SUCCESS. StatutsRust/Kani séparés dans lessnapshotsJSON. BufferMetalexplicitementsauté, pas nouveaucontrôle27B, FPS/énergie/gestes physiques nonmesurés. Tests visibles terminés, aucune nouvelle fenêtre ouverte après demande d’arrêt. Appinstallée nonremplacée.

Publication vérifiée : [Desktop1.4.8](https://github.com/0xZKnw/mlxl3/releases/tag/v1.4.8), tag source exact, canal stable, deux actifs re-téléchargés et SHA-256 identiques. Moteur public inchangé, app installée non remplacée. Les jobs Rust Linux et macOS ont ensuite réussi ; Kani reste en cours au dernier relevé.

Actualisation après publication : le [job Kani du moteur hérité](https://github.com/0xZKnw/mlxl3/actions/runs/38953567272)
a terminé en **échec par interruption (code 143)** pendant
`dflash_copy_len_10`, après 65 harnais réussis. Les obligations déjà
rapportées n’affichent aucun échec ; le harnais interrompu et la suite complète
restent **non vérifiés**, sans cause d’interruption établie. Les jobs Rust Linux
et macOS sont réussis. Les sources du moteur et de Kani sont identiques à la
baseline ; aucun changement ni nouvelle exécution du moteur n’est ajouté ici.
