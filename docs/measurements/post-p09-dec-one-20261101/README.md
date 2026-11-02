# DEC-ONE exact sur Flash — 1 novembre 2026

**Clos : abandon demandé par l'utilisateur, expérimentation désactivée.**
La qualité a passé les contrôles ciblés ; le gain global reste non concluant.
R2 était prometteur en AR et DFlash mais défavorable en MTP ; R3 et le
comparatif direct n'ont pas établi un bénéfice stable de l'exclusion de tête.
Les sources, binaires et résultats, y compris négatifs, restent conservés.
`r3-summary.json`, `direct-summary.json` et `r3-exclusion-source-audit.json`
complètent les résultats historiques ci-dessous. Aucun benchmark en cours.
Dynamic+ColdU reste activé ; ColdShared, prepared et F02 restent désactivés.
Aucun push, publication ou remplacement de l'application installée.

## Résultats AR

Même binaire release R2, flag `MLXL3_FLASH_DEC_ONE=0/1`, M5 24 Go sur secteur,
MLX 0.32.2, Flash EXL3 2.05 bpw checkpoint `65c89531`, experts 3 Gio,
quatre lecteurs dynamiques. MAIN/HOLD, 64 sorties (63 transitions), ordre
A/C/C/A, une chauffe et une mesure par prompt et processus.

| Prompt | A tok/s | C tok/s | Débit | Temps prefill A → C |
|---|---:|---:|---:|---:|
| MAIN | 10,60627 | 11,13028 | +4,941 % | 8,31531 → 8,22866 s |
| HOLD | 10,58873 | 11,29546 | +6,674 % | 7,67485 → 7,38997 s |
| Moyenne | 10,59750 | 11,21287 | +5,807 % | 7,99508 → 7,80932 s |

Le prefill moyen baisse de 2,323 %. `r2-pilot-summary.json` conserve chaque
mesure ; les rapports contiennent aussi toutes les chauffes. Les 16 réponses
concordent avec le préfixe de la référence externe128 ; textes et hashes A/C
égaux. L'écran antérieur AR32 A/C était presque neutre (MAIN −0,022 %,
HOLD −0,055 %) et reste conservé. Longueur, ordre et variations interpassages
interdisent d'en faire une garantie de +5,8 % sur tout contexte.

## Port et audit demandé

Seules les projections Flash 6144→2560, K4/Mul1, S2/SG4/NT1, B1/B2,
sans biais et à échelles F16 sont admissibles. Les autres chemins gardent
leur fallback. Les 48 projections de sortie attention du checkpoint
correspondent à ce périmètre (`checkpoint-output-audit.json`). Elles sont
chargées via `Projection::load_logical` puis `Exl3Linear::from_checkpoint`.

**Correction de portée R3 :** R2 touchait aussi la projection
`mtp.layers.0.self_attn.o_proj`, soit 49 projections admissibles. R3 garde
B1/B2 dans les 48 couches cibles et remet seulement cette projection MTP
sur le chemin stock. L'oracle indépendant de la tête passe sur R2 et R3 :
ce périmètre supplémentaire n'est pas un bug numérique démontré.

MTP et DFlash2 width1 partagent `prepare_rows` et le calcul MoE à deux
lignes (GDN B2, QSA B1 sériel). Leurs propositions et acceptations diffèrent,
mais le seul row_limit2/8 n'explique pas une différence de kernel sur ces
deux lignes. Aucun défaut de fallback, copie CPU ou synchronisation ajoutée
spécifique à MTP n'a été trouvé dans ce port.

Le corps QMV canonique est réutilisé sans modification. Après publication
des partiels, le dernier des 16 groupes d'un bloc128 exécute la réduction,
Hadamard et les échelles, en conservant l'ordre des sommes et les arrondis.
Le mécanisme d'arrivée vient du TurboEXL3 épinglé à `087c757`, sans importer
son calcul affine DEFER ni son réglage de splits. Aucun spin ni atomique
flottante. Sources primaires et licence sont enregistrées dans ce dossier.

Chaque appel possède un output UInt32 privé : partiels, 20 compteurs,
puis sortie F16. MLX initialise cet output à zéro ; la sortie est une vue
sans copie sur sa fin. Taille logique B1/B2 : 25 680 / 51 280 octets.
La capture B1 montre une allocation arrondie de 32 Kio. Aucun input MLX
n'est muté ; les références gardent le stockage vivant pendant l'évaluation.

**Défaut trouvé et corrigé avant R2 :** R1 reconstruisait le texte du shader
à chaque projection. R2 le construit une fois via `OnceLock`, sans capturer
d'array ou de poids. Le micro R1 dérive fortement : GDN B1/B2 +0,72/+1,18 %
de temps, QSA B1/B2 +17,86/−2,13 %. Ses 48 lots restent disponibles ; ils
ne valident pas la vitesse R2 et n'ont pas servi à rejeter le candidat.

**Coût restant identifié, non modifié pendant la comparaison :** le fill
initialise tout le buffer, bien que seuls les compteurs nécessitent zéro.
Il remplace un dispatch supprimé par la fusion. Les fences, compteurs et
ressources du finish peuvent aussi coûter ; leur coût exclusif et l'effet
sur l'occupation GPU ne sont pas mesurés. Ce sont des limites concrètes,
pas une attribution certaine de chaque variation de tok/s.

Six captures R2 ont été inspectées dans Xcode 27.2, sans option de profilage
du replay : GDN **8→5 dispatchs**, QSA **8→5**, QKV témoin **6→6**, toujours
deux command buffers et un compute encoder. Le kernel DEC-ONE B1 est bien
exécuté, grid 20480×1×2, groupe128 ; fill6420U32 puis sortie offset20560B
dans le même buffer. `captures-r2/dispatch-inspection.json` documente les
observations et limites. Les champs `enabled` et `kernel_requests` du
harnais réutilisé concernent F02, maintenu à zéro dans les deux bras.
Xcode a été fermé avant les mesures MTP/DFlash ; aucun tok/s ne vient du replay.

## Contraste AR / MTP / DFlash et diagnostic de portée

R2, même protocole64 ACCA sur MAIN/HOLD :

| Mode | A → C tok/s moyens | Débit | Temps prefill |
|---|---:|---:|---:|
| AR | 10,59750 → 11,21287 | +5,807 % | −2,323 % |
| MTP | 11,45065 → 11,13588 | −2,749 % | +2,787 % |
| DFlash2 width1 | 11,39348 → 11,65828 | +2,324 % | +3,244 % |

MTP MAIN est dispersé (C2 10,05961, C3 11,51496 tok/s), HOLD quasi neutre.
La confirmation unique MAIN C/A donne 11,85133 → 10,92191 tok/s (−7,843 %),
prefill −5,117 %. Aucun échantillon n'est supprimé. Activité utilisateur
rapportée, horaire/charge non quantifiés ; pas une explication démontrée.

Trois diagnostics naturels MAIN128 comparent R2/OFF, R2/ON et R3/ON avec
le harnais existant : six phases, rétention des intermédiaires seulement
pendant warmup, zéro intermédiaire retenu ensuite, logits/états complets
exacts internes et golden externe. Tous passent. Tokens, acceptations,
hits/misses et octets SSD demandés sont identiques **phase par phase**
entre les trois bras. L'ancien harnais léger a été annulé avant exécution
car il conservait les intermédiaires cfg(test).

Moyennes des trois traces, **ms par transition, intervalles hôte inclusifs
qui se chevauchent : ne pas additionner les lignes** :

| Poste | R2 OFF | R2 ON | R3 ON, tête exclue |
|---|---:|---:|---:|
| Total décode | 89,046 | 98,875 | 90,601 |
| Vérification cible | 83,519 | 92,918 | 85,218 |
| Proposition tête MTP | 3,877 | 4,149 | 3,757 |
| Lecture des routes, attente incluse | 30,055 | 36,033 | 30,931 |
| Soumission asynchrone MLX | 18,475 | 20,088 | 19,087 |
| Attente première partie Cold | 11,676 | 12,234 | 11,481 |
| Fin Cold | 10,377 | 10,084 | 10,031 |

L'écart R2 se trouve principalement dans la vérification ; la tête ne
porte que 0,272 ms sur 9,829 ms d'écart total observé. La demande SSD
n'augmente pas. Ces chiffres ne donnent pas une partition GPU utile/idle,
ni des lectures SSD physiques. Les coûts CPU/GPU exclusifs du nouveau
kernel restent non mesurés.

R3 revient près du témoin mais ce n'est pas une preuve causale de la tête :
le prefill varie lui aussi, Equibop et WindowServer étaient actifs pendant
R2/ON (snapshot conservé), et la compression mémoire varie. `pmset -g therm`
ne rapportait aucun warning enregistré ; cela ne mesure pas les fréquences.
Les trois traces sont des diagnostics, pas des débits release.
Voir `diagnostic-summary.json` et les trois dossiers `diagnostic-*`.

R3 release64, une paire par mode (MTP puis AR en A/C ; DFlash en C/A),
une chauffe et une mesure MAIN/HOLD par processus :

| Mode | A → C tok/s moyens | Débit | Temps prefill A → C |
|---|---:|---:|---:|
| AR | 10,87043 → 11,03118 | +1,479 % | 7,91287 → 7,84691 s (−0,834 %) |
| MTP | 11,71410 → 11,68356 | −0,261 % | 7,75668 → 7,78548 s (+0,371 %) |
| DFlash2 width1 | 12,31561 → 12,29915 | −0,134 % | 8,01296 → 7,87577 s (−1,712 %) |

Les 24 réponses warm/mesure sont exactes au préfixe externe128 et les
hashes/acceptations A/C identiques ; processus et lecteurs joints. AR/C
ajoute384 Kio de swap global, autres passages R3 release zéro. Les grands
gains R2 ne sont pas reproduits à cette amplitude ; la forte baisse MTP
non plus. Le code reste un candidat prometteur local OFF, réutilisable en
composition, sans annonce de gain stable global ni installation d'app.

Audit supplémentaire demandé : `r3-exclusion-source-audit.json` confirme
que le seul changement de `linear.rs` est le filtre au chargement,
que `dec_one::apply` et le shader sont inchangés et que les 48 vrais
préfixes du checkpoint passent ce filtre. Le bridge ne charge la tête MTP
que lorsque MTP est actif ; AR/DFlash2 déclarent MTP inactif. Cela ne
prouve pas l'identité du code machine entre deux builds. Quatre passages
directs R2/ON contre R3/ON sont préenregistrés pour vérifier un effet de bord
sur AR/DFlash2 ; leurs résultats restent séparés des comparaisons contre OFF.

## Vérification

- R2 : 75 contrôles de projections exactes (trois références externes,
  B1/B2/F16/F32/strides, quatre sorties en attente évaluées à rebours,
  fallback B3/QKV, inputs inchangés).
- R2 : oracle indépendant cible logits/états ; MTP avec états cible/head,
  séquence et acceptation ; DFlash 36 préfixes, largeurs1..8, captures64 000
  valeurs, logits et états exacts face au parcours sériel.
- Initialisation FFI : valeurs invalides rejetées avant GPU ; 32 outputs
  privés avec valeurs initiales alternées0/7 et évaluations à rebours.
- Kani0.68 : fonction réelle `dec_one::layout`, domaine i32 symbolique
  sans hypothèse, 83 obligations et quatre couvertures satisfaites.
  Non-recouvrement, alignement, comptage et borne51 280 octets vérifiés.
  Cela ne prouve ni Metal, ni les fences, ni MLX ou la génération complète.
- R3 : 136 obligations Kani et sept couvertures satisfaites ; même domaine
  i32 pour le layout, sélection symbolique parmi cinq préfixes pour la portée
  cible/MTP (pas une preuve sur toutes les chaînes UTF-8).
- R3 : cinq sélections physiques passent, dont le chargement réel de la
  projection MTP exclue avec zéro requête DEC-ONE ; les 75 contrôles cible,
  états cible/MTP, 36 préfixes DFlash et oracle tête indépendant passent.
  Suite complète mlx/chat : 314 passés, 207 ignorés ; fmt, Clippy strict
  tous targets et build release passent. Les ignorés physiques non sélectionnés
  restent non exécutés.
- CBMC sur le vrai C++ : erreur de parsing des en-têtes libc++ avant
  obligations ; FFI C++ non vérifiée formellement. Diagnostic conservé.
- R1 : fmt, Clippy strict, build réel mlx/chat et suite lib218 passés,
  196 ignorés. Les tests physiques ci-dessus sont sélectionnés explicitement.
  Tests de l'orchestrateur Kani44 ignorés sur macOS faute `prlimit` Linux.
  CI PR/main adaptée, YAML vérifié ; aucun run distant du diff non publié.

Les commandes exactes, options et identités sont dans les plans JSON,
`identity-r2.json` et `identity-modes-r2.json`. Builds avec
`CARGO_BUILD_JOBS=1`, `MLXL3_MLX_ROOT` pointant vers MLX0.32.2 installé et
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

```
cargo build --locked --release --features mlx,chat --bin mlxl3-rs
cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json
cargo kani --lib --no-default-features --harness dec_one::dec_one_storage_contract
python docs/measurements/decode-profile-p09-20261101/run-mtp-head.py PLAN IDENTITY
```

CLI R2 `91c8f6892abf235fd7b0cf80761c31fb52224a2f85519437aa28215c369c8008`,
libtest R2 `02859cd3dbff0df033dcef359746c70a0bfe32f7d93e46bbdd227705290c72ed`.
CLI R3 `32425d756a1f9b7446a85eee5bcb9883a8b06ceeb20ff618b722e20761653c21`,
libtest R3 `e89a17293acb5b4263c696e1333d40a818b772c840787fcc65041fd22400675d`.
Base `d5a776e53425bb4d0ba61c0774c5399ae7987c06` avec modifications locales
préexistantes conservées. `own-change-r2.diff` isole ce port et ses contrôles.
`own-change-r3.diff` conserve le port final et `r2-to-r3-scope.diff` isole
l'exclusion de la tête. Aucune ablation B1 seul n'a été compilée ou mesurée.

Même ancre MTPHEAD, plafond cumulatif32 Mio écritures/256 Mio lectures swap,
admission modèle10,5 Go, petits composants2 Go. AR64 A1 ajoute320 Kio,
autres processus de qualification/captures/AR sans nouvel événement durant
leur exécution. Aucun réancrage. Les erreurs d'outillage et résultats R1
restent dans `opti.md` et les artefacts ; aucune relance jusqu'au vert.

