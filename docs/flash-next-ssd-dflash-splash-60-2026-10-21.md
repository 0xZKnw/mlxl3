# Flash Next SSD : DFlash2 Q4, cache circulaire et budget Splash — expérience 60

État : correction numérique et bridge passés, réduction mémoire Q4 validée ; débit global B/Q non concluant. Cache circulaire et budget adaptatif qualifiés numériquement, leur débit reste à mesurer. Aucune application installée remplacée.

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

## Correction et vérification

Les identités des sources, archives et binaires sont dans [les mesures 60](measurements/flash-next-ssd-impl/opt-60-splash-dflash/). La version native v3 a passé les comparaisons suivantes sur le M5 physique, un seul couple cible/draft chargé :

- Q4 : 64 groupes et 4 096 codes/paramètres comparés bit à bit à un oracle scalaire indépendant. Produit avec accumulation indépendante, lignes 1/4/8/17 : RMSE relative 0 ; lignes 256 : 0,00357168, sous la tolérance préenregistrée de 0,004. Une mutation des nibbles est détectée. Formes invalides et valeurs non finies rejetées. La v6 ajoute et teste physiquement le rejet de paramètres non finis issus d'entrées finies extrêmes.
- Cache circulaire : six appends jusqu'à 6 148 positions, puis 28 attentions aux offsets 0/1/2047/2048/2049/4095/8193 et largeurs 1/2/4/8, comparées à une attention chronologique SDPA indépendante. K/V physiques exacts ; RMSE relative maximale 0,0000726013, sous 0,004.
- Sélection et confiances : huit cohortes BF16/Q4, largeurs 1/2/3/7, comparées à un calcul Markov scalaire ; erreurs de confiance au plus 1e-4.
- Génération native : 20 cas B/Q/R/A × budgets 1/2/3/17/64, tokens identiques à la cible séquentielle et 136 états sémantiques par cas comparés en forme, finitude et octets. Annulation, reprise et entrées invalides incluses. Pic d'empreinte physique 8 945 229 024 octets, aucun nouveau Swapout jusque 30 secondes après la fin.
- Bridge de production v6 : 43 contrôles passés en 307,95 secondes, budgets 1/2/3/8/17/64, trois modes, conflits et erreurs, annulation au préfill et après émission, reprise, cache de conversation ignoré explicitement, mémoire idle et désactivation. Pic physique 8 937 904 704 octets, aucun nouveau Swapout jusque 30 secondes après la fin. Les contrôles v6 gardent la même origine VM cumulative que v3 ; aucun compteur remis à zéro entre les processus.

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

Le point de publication comprend cette réduction mémoire et le vrai parcours DFlash2 numériquement qualifié. Les anciens résultats négatifs 57 et l'état incomplet 58 sont conservés comme preuves historiques, sans incorporation de leurs prototypes dans le moteur. Les comparaisons Q/R, R/A, OFF et MTP1 restent à faire ; **15 tok/s non atteints**.
