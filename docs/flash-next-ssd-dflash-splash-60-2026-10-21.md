# Flash Next SSD : DFlash2 Q4, cache circulaire et budget Splash — expérience 60

État : implémentation publiée dans la PR34, correction numérique et bridge passés, réduction mémoire Q4 validée. Les quatre campagnes de débit sont closes. L'adaptatif améliore les quatre paires contre DFlash fixe, avec une dispersion trop forte sur HOLD ; il reste derrière MTP1 sur les quatre paires fraîches. **15 tok/s non atteints.** Aucune application installée remplacée.

## Sources et portée

- Base MLXL3 : `79e605fab417a5f844bc6dd49f6d35eee436a757`, worktree isolé `flash-splash60`. Les prototypes 57/58 ne sont pas incorporés dans le moteur de cette expérience.
- [Splash](https://github.com/incoai/splash/tree/1482f6bbc2abf12375332725302019594150b6cb), commit `1482f6bbc2abf12375332725302019594150b6cb`, licence Apache-2.0. Le kernel d'attention circulaire est adapté de leur implémentation Metal MPP ; attribution dans le shader et `THIRD_PARTY_NOTICES.md`.
- Cible Qwen3.8 Flash Next EXL3 2.05bpw : révision `65c895314393431c09050b2e04e250836b3a6eb4`.
- Draft IncoAI DFlash2 : révision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, cinq couches, captures 5/15/25/35/45. Les poids locaux d'origine restent inchangés.
- Revue des mécanismes et limites amont : [rapport 59](flash-next-ssd-strata-splash-59-2026-10-21.md).

## Implémentation

Les 47 matrices de projection du draft passent de BF16 à la quantification affine 4 bits par groupes de 64 de MLX, avec paramètres BF16. Le produit utilise QMM MLX ; ce travail ne porte pas le kernel Q4 MPP spécialisé de Splash. Normes, convolutions et codebooks restent BF16. Le payload calculé sur les formes passe de 1 206 131 200 à 522 181 120 octets, soit 683 950 080 octets économisés ; cette différence de poids n'est pas une mesure de l'empreinte physique du processus.

Le cache KV circulaire conserve 2 048 positions par couche, avec au plus deux écritures physiques non recouvrantes par append. L'attention MPP parcourt directement ces slots, applique la fenêtre de chaque ligne et réduit huit partitions dans un ordre fixe. Le chemin n'assemble pas de cache chronologique ou de masque quadratique CPU. Les mises à jour passent par `slice_update` de MLX, sans écriture brute dans un tableau encore utilisé.

Le budget adaptatif transpose le modèle probabilités/coût de Splash sur une seule séquence. Les confiances du sélecteur Markov DFlash2 choisissent de zéro à sept propositions, toujours vérifiées par le modèle cible. Seuls les tokens acceptés et le premier refus fournissent des labels. La rétention est bornée à 4 096 labels, le réajustement probabiliste intervient tous les 64 cycles et une exploration déterministe tous les 32 cycles. L'état repart à neuf à chaque requête ; sur une génération courte, l'ajustement probabiliste peut ne pas se déclencher. Les coûts observés s'adaptent dès les premiers cycles.

Le bridge Flash dispose maintenant du vrai parcours DFlash2 : chargement, préfill avec captures, propositions, vérification, commit, génération, annulation, erreurs et réinitialisation. Il annonce la capacité DFlash2 et l'absence du Tune historique : le budget est en ligne. Le mode 1 permet le budget adaptatif, le mode 2 fixe deux propositions et le mode 3 en fixe sept. Les contrôles expérimentaux sont `MLXL3_FLASH_DRAFT_Q4`, `MLXL3_FLASH_DRAFT_RING` et `MLXL3_FLASH_DRAFT_ADAPTIVE`, valeurs `0` ou `1`, tous à `1` par défaut dans ce prototype DFlash2 activé explicitement. La configuration mémoire mesurée ci-dessous utilise **Q4=1, RING=0, ADAPTIVE=0**, mode 1 ; son débit ne valide pas celui de la configuration adaptative. Aucun support DSpark ou changement de calcul LFM2 n'est ajouté ici.

Le draft BF16 brut est fourni explicitement au bridge avec `dflash_draft_path`, et Q4 est calculé en mémoire. L'installateur `dflash-draft` et le téléchargement automatique Desktop conservent leur périmètre legacy de drafts packed ; ils ne sont pas étendus ni validés pour Flash dans cette expérience.

## Correction et vérification

Les identités des sources, archives et binaires sont dans [les mesures 60](measurements/flash-next-ssd-impl/opt-60-splash-dflash/). La version native v3 a passé les comparaisons suivantes sur le M5 physique, un seul couple cible/draft chargé :

- Q4 : 64 groupes et 4 096 codes/paramètres comparés bit à bit à un oracle scalaire indépendant. Produit avec accumulation indépendante, lignes 1/4/8/17 : RMSE relative 0 ; lignes 256 : 0,00357168, sous la tolérance préenregistrée de 0,004. Une mutation des nibbles est détectée. Formes invalides et valeurs non finies rejetées. La v6 ajoute et teste physiquement le rejet de paramètres non finis issus d'entrées finies extrêmes.
- Cache circulaire : six appends jusqu'à 6 148 positions, puis 28 attentions aux offsets 0/1/2047/2048/2049/4095/8193 et largeurs 1/2/4/8, comparées à une attention chronologique SDPA indépendante. K/V physiques exacts ; RMSE relative maximale 0,0000726013, sous 0,004.
- Sélection et confiances : huit cohortes BF16/Q4, largeurs 1/2/3/7, comparées à un calcul Markov scalaire ; erreurs de confiance au plus 1e-4.
- Génération native : 20 cas B/Q/R/A × budgets 1/2/3/17/64, tokens identiques à la cible séquentielle et 136 états sémantiques par cas comparés en forme, finitude et octets. Annulation, reprise et entrées invalides incluses. Pic d'empreinte physique 8 945 229 024 octets, aucun nouveau Swapout jusque 30 secondes après la fin.
- Bridge de production v6 : 43 contrôles passés en 307,95 secondes, budgets 1/2/3/8/17/64, trois modes, conflits et erreurs, annulation au préfill et après émission, reprise, cache de conversation ignoré explicitement, mémoire idle et désactivation. Pic physique 8 937 904 704 octets, aucun nouveau Swapout jusque 30 secondes après la fin. Les contrôles v6 gardent la même origine VM cumulative que v3 ; aucun compteur remis à zéro entre les processus.

Le contrôle du diff avant push a ensuite demandé de retirer une ligne vide en fin du shader embarqué. La source v7 et ses binaires sont archivés séparément ; aucune opération numérique ne change. Les six appends et 28 attentions ont été revérifiés sur v7, avec les mêmes erreurs bornées, avant les mesures Q/R et R/A. La mesure B/Q v6 n'exécute jamais ce shader (`RING=0`) : ses chemins Q4 et cible ne sont pas affectés par cette correction de fin de fichier.

Contrôles CPU : suite release `mlx,chat`, 139 tests passés et 138 ignorés ; ces tests ignorés ne sont pas déclarés exécutés par la suite générale. Suite Python pertinente : 562 tests passés, puis 69 tests benchmark/qualificateur affectés par les derniers changements. Format, Ruff et Clippy strict passent. L'échec initial d'une fixture de démarrage trop courte a été reproduit sur HEAD puis corrigé dans le test ; les journaux négatifs sont conservés.

Kani sur le vrai helper de ring : domaines i32 symboliques complets, unwind 3, 41 obligations et deux couvertures passent. Kani sur le choix de préfixe : sept probabilités et huit coûts f64 symboliques, unwind 9, 270 obligations dont une inatteignable de formatage std, deux couvertures passent ; ce contrôle porte sur les bornes et rejets, pas sur l'optimalité générale. L'optimalité est également testée contre une énumération indépendante de 16 384 distributions et 128 issues chacune. Les contrats existants de cohorte, acceptation et commit passent leurs harnais ciblés.

La tentative Kani sur le vrai chemin MLX atteint une erreur interne du compilateur ; CBMC sur le vrai CABI échoue sur le parsing du SDK libc++ Apple ; CrossHair est absent. MLX, Metal, concurrence et génération complète ne sont pas formellement prouvés. Les premières compilations échouées, l'erreur MPP `const bfloat` et les essais interrompus restent dans les artefacts et le journal.

## Protocole de débit

M5 24 Gio, secteur, applications de l'utilisateur ouvertes. Cache experts 3 Gio / 53 slots, cache allocateur 512 Mio, quatre lecteurs SSD, overlap et GPU Hot activés, LES et lookahead désactivés. Contexte 4 096, greedy, mêmes prompts MAIN/HOLD avec le template thinking, 128 tokens émis / 127 décodés. Un warmup et une mesure par nouveau processus, ordre alterné HCCH pour chaque prompt. Aucun build, preuve ou autre inférence simultanée.

| Variante | Draft | Cache KV | Vérification mode 1 |
|---|---|---|---|
| B | BF16 | concaténation | fixe, 3 propositions |
| Q | Q4 | concaténation | fixe, 3 propositions |
| R | Q4 | circulaire | fixe, 3 propositions |
| A | Q4 | circulaire | adaptative, au plus 7 propositions |

Ordre prévu : B/Q, puis Q/R, puis R/A si le mécanisme justifie de poursuivre ; contrôles OFF et MTP1 frais pour le meilleur DFlash2. Un résultat de composant ou un temps provenant des tests de correction ne constitue pas un gain de débit.

Chaque campagne conserve une origine VM cumulative : préflight 20 secondes, garde toutes les deux secondes, limites de 256 Mio Swapin/Swapout et 12 milliards d'octets d'empreinte physique. Un gain exige zéro nouveau Swapout, contrôle différé de 30 secondes inclus. Seuils préenregistrés : au moins 1 % par paire, dérive des témoins au plus 5 %, étendue des gains au plus trois points ; les résultats instables restent non concluants.

## Mesures B/Q

| Prompt | BF16 (tok/s) | Q4 (tok/s) | Effets appariés |
|---|---|---|---|
| MAIN | 8,376 ; 8,757 | 8,578 ; 8,943 | +2,42 % ; +2,12 % |
| HOLD | 8,474 ; 8,601 | 8,299 ; 8,725 | −2,06 % ; +1,45 % |

Les huit processus sont joints, avec 16 réponses exactes (warmup et mesure). La dérive des témoins vaut 4,554 % sur MAIN et 1,498 % sur HOLD. L'étendue des effets sur HOLD atteint 3,508 points, au-delà des trois préenregistrés ; une paire y est négative. Décision : **`inconclusive_drift`, aucun gain global de tok/s établi**. Les passages MAIN favorables ne deviennent pas une promesse de gain général.

Les quatre comparaisons de pic physique donnent des économies de **490 013 056 / 486 769 000 / 487 620 968 / 486 785 480 octets**, soit environ 486–490 Mo. Le pic BF16 est de 8,873–8,875 Go et le pic Q4 de 8,385–8,389 Go. La réduction mémoire est validée pour cette configuration et ce protocole. Zéro nouveau Swapout sur les huit bras et les 30 secondes postflight, même origine cumulative que la correction native. [Données et décision](measurements/flash-next-ssd-impl/opt-60-splash-dflash/speed-v6-BQ/analysis.json).

Le premier point de publication `dc33e3e4` comprend cette réduction mémoire et le vrai parcours DFlash2 numériquement qualifié. Les anciens résultats négatifs 57 et l'état incomplet 58 sont conservés comme preuves historiques, sans incorporation de leurs prototypes dans le moteur.

## Mesures Q/R

| Prompt | Q4 concat (tok/s) | Q4 circulaire (tok/s) | Effets appariés |
|---|---|---|---|
| MAIN | 8,317 ; 9,018 | 8,468 ; 9,026 | +1,81 % ; +0,09 % |
| HOLD | 7,951 ; 8,459 | 8,479 ; 7,987 | +6,65 % ; −5,58 % |

Décision : **`inconclusive_drift`, aucun gain global établi**. Les témoins dérivent de 8,421 % et 6,391 % ; l'étendue sur HOLD vaut 12,232 points et les deux effets y changent de signe. Le cache circulaire ajoute environ 37–41 Mo de pic sur ces contextes courts, tout en bornant sa capacité KV à 2 048 positions. Les huit processus sont joints, 16 réponses exactes, aucun nouveau Swapout jusque 30 secondes après. [Données et décision Q/R](measurements/flash-next-ssd-impl/opt-60-splash-dflash/speed-v7-QR/analysis.json).

## Mesures R/A

La demande utilisateur confirme de tester l'adaptatif même sans gain Q/R grâce à l'intérêt mémoire de Q4.

| Prompt | Q4 circulaire fixe (tok/s) | Q4 circulaire adaptatif (tok/s) | Effets appariés |
|---|---|---|---|
| MAIN | 8,697 ; 8,763 | 9,789 ; 9,746 | +12,57 % ; +11,21 % |
| HOLD | 8,221 ; 8,632 | 8,849 ; 8,854 | +7,64 % ; +2,58 % |

Les quatre paires sont favorables. MAIN satisfait les seuils préenregistrés : dérive 0,766 %, étendue 1,351 point. HOLD a une dérive de 4,998 % et une étendue de 5,062 points, supérieure aux trois permis. Décision globale stricte : **`inconclusive_drift`**, avec un signal positif à confirmer, sans annonce de boost général.

Les compteurs expliquent un mécanisme concret : sur MAIN, le fixe vérifie 182 lignes dans ses blocs, l'adaptatif 136 ou 138 ; sur HOLD, 186 contre 139 ou 138. Il écarte donc environ un quart des lignes de vérification, avec davantage de petits blocs. Cela ne mesure pas le temps GPU exclusif. MAIN compte 50–51 cycles, avant le premier refit probabiliste à 64 ; HOLD atteint 64–65 cycles, donc ce refit arrive en toute fin. Le gain ne peut pas être attribué à une longue calibration probabiliste : l'adaptation du coût commence bien plus tôt.

Le pic adaptatif atteint **8,859–8,910 Go**, contre environ 8,42 Go pour le fixe, notamment avec les états temporaires des vérifications larges. L'économie de pic de 486–490 Mo mesurée en Q4 fixe ne doit pas être transférée au pic du mode adaptatif. Huit processus joints, 16 réponses exactes, zéro nouveau Swapout jusque 30 secondes après. [Données R/A](measurements/flash-next-ssd-impl/opt-60-splash-dflash/speed-v7-RA/analysis.json).

## Comparaison finale OFF / MTP1 / DFlash2 adaptatif

La règle enregistrée avant R/A retient A comme candidat exploratoire : quatre effets favorables et au moins 1 % géométrique sur chaque prompt. Elle ne transforme pas l'échec du critère de dispersion en gain validé. Deux témoins OFF frais précèdent huit bras MTP1/A alternés, mêmes binaire v7, modèle, paramètres et alimentation.

| Prompt | OFF, témoin descriptif (tok/s) | MTP1 (tok/s) | DFlash2 A (tok/s) | A contre MTP1 |
|---|---|---|---|---|
| MAIN | 10,480 | 10,546 ; 10,154 | 9,994 ; 10,013 | −5,24 % ; −1,39 % |
| HOLD | 10,458 | 9,586 ; 9,695 | 8,985 ; 9,283 | −6,27 % ; −4,25 % |

**A ne dépasse MTP1 sur aucune des quatre paires.** Le classement global du qualificateur reste `inconclusive_drift` : sur MAIN, l'étendue des effets atteint 3,851 points, au-delà de trois, malgré une dérive témoin de 3,864 %. HOLD a une dérive de 1,137 % et une étendue de 2,014 points. Ces limites ne permettent pas d'annoncer un gain ; les quatre observations vont dans le sens opposé. OFF est un témoin descriptif, sans matrice appariée complète contre chaque mode.

Pics physiques finaux : environ **7,13–7,14 Go OFF**, **7,92–8,20 Go MTP1**, **8,86–8,90 Go DFlash2 A**. Les dix processus sont joints. Les **68 générations** des quatre campagnes (34 processus, warmup et mesure, chacune de 128 tokens) ont le même texte et la même empreinte de tokens que les témoins OFF frais. Cela complète les comparaisons numériques natives de 136 états ; une empreinte de tokens n'est pas une preuve formelle.

Zéro nouveau Swapout sur toute la campagne cumulative v3, y compris le postflight final de 30 secondes. Swapin cumulé final : **240 009 216 octets**, sous la garde de 268 435 456 octets. Un seul couple cible/draft chargé à la fois, aucune compilation ni preuve locale pendant les mesures. [Données finales](measurements/flash-next-ssd-impl/opt-60-splash-dflash/speed-v7-final/analysis.json).

## Vérification CI et reproduction

Au commit de code `dc33e3e48c472dfd875923679304ab3114e55ad0`, les [régressions Desktop/protocole](https://github.com/0xZKnw/mlxl3/actions/runs/39584144970), Linux natif, macOS MLX et Kani Flash passent. Le [workflow natif global](https://github.com/0xZKnw/mlxl3/actions/runs/39584145219) reste rouge : le job Kani global reçoit un shutdown runner / exit 143 sur le harnais préexistant `cache_allocation_is_optimal_and_exact_on_two_layers_two_levels`, dont le code n'a pas changé. Ce n'est pas un contre-exemple ; le même problème était enregistré au commit79. Les nouveaux harnais de ring et de préfixe ont leurs obligations et couvertures passées dans le job Kani Flash distinct. Journaux bruts et nettoyés conservés avec les mesures.

Les commandes GPU exactes, variables d'environnement, modèles et ordres sont dans les `run.json`, avec leurs empreintes de matrice et les identités des binaires. Les plans QR/RA v6 ont été préparés mais jamais lancés, remplacés par v7 après la seule correction de fin de fichier du shader. Contrôles locaux utilisés :

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --features mlx,chat -- -D warnings
cargo clippy --all-targets --no-default-features --features chat -- -D warnings
cargo test --release --features mlx,chat
cargo test --locked --no-default-features
python -m pytest -q
cargo kani --lib --no-default-features --harness circular_writes_are_bounded_disjoint_and_cannot_wrap
cargo kani --lib --no-default-features --harness adaptive_prefix_is_bounded_and_rejects_invalid_probabilities
```

Le build utilise `MLXL3_MLX_ROOT` pointant vers MLX0.32.2 et le runtime Apple physique M5/macOS27.2. Après fermeture de la campagne GPU, la suite CPU sans features passe **110 tests / 3 ignorés** (94 lib + 16 contrats), et la suite Python complète passe **915 tests / 4 skips**, en 105,74 secondes. Les skips concernent `ponyexl3.convert` / `ponyexl3.convert.measure` absents et le modèle Ling local absent ; ces parcours de conversion ne sont pas validés par ce run. Format et Ruff repassent, toutes les empreintes de source/binaire v7 sont recontrôlées. Les tests physiques ignorés par la suite générale restent distingués des tests sélectionnés explicitement. La GUI et son installateur automatique Flash DFlash2 ne sont pas validés ici. [Synthèse vérifiable](measurements/flash-next-ssd-impl/opt-60-splash-dflash/validation-summary.json) et [commandes CPU finales](measurements/flash-next-ssd-impl/opt-60-splash-dflash/final-cpu-commands.json).
