# Hot précoce et réutilisation du résultat — expérience64H

21 octobre 2026. **Correction physique validée en v4, gain de débit rejeté.** Le premier essai de [réunification64U](flash-next-ssd-unique-cohorts-64-2026-10-21.md) était exact mais non concluant sur le débit, avec quatre paires négatives.64H corrige un défaut d'ordonnancement : sa voie v2 attendait les routes CPU avant de préparer Hot. L'essai64L suivant traite séparément le suivi LFU des contributions Hot réutilisées au repli ; il ne requalifie pas les mesures v4.

## Constat et mécanisme

Les deux traces du diagnostic v2 observent chacune240 grandes fenêtres avec Hot préparé après la fin du span de remontée CPU. [Audit des cohortes et de leur ordre](measurements/flash-next-ssd-impl/opt-64-unique-cohort/hot-order-v2-retry/cohort-order-audit.json). Les sorties, logits et états sont exacts, mais la campagne **échoue à la garde mémoire** :103 809 024octets de nouveau Swapout au second processus, pic physique8 468 012 016octets. Aucun gain n'est qualifié par ces traces. L'échec initial du lanceur, avant chargement du modèle faute de dossier de sortie, est conservé aussi.

Le correctif conserve le même routage et les mêmes kernels :

1. Attendre les anciens lecteurs des banques, capturer expert→slot et soumettre Hot pour la fenêtre complète, avant la lecture CPU des IDs.
2. Récupérer les IDs une fois et chercher le plus grand préfixe dont l'union tient dans les slots non pending.
3. Si toute la fenêtre tient, protéger tous ses experts sélectionnés et remplir Cold pendant Hot. Réutiliser directement son résultat et ses entrées routées.
4. Si elle doit être découpée, terminer le Hot complet avant toute éviction. Garder ses valeurs et ne réserver que les experts initialement Cold dans chaque sous-bloc. Attendre les consommateurs Cold précédents avant réemploi des banques.
5. Assembler les contributions de chaque occurrence dans l'ordre original, avec les mêmes poids. Un expert Cold devenu hit au sous-bloc suivant doit encore être calculé pour les nouvelles activations ; il ne fait pas partie du résultat Hot initial.

Le repli peut donc évincer un ancien expert Hot sans le relire ni recalculer sa contribution. Il garde toutefois une barrière avant les évictions dans ce cas. Aucune déquantification EXL3 partagée entre occurrences, aucun kernel GDN et aucun agrandissement du cache ne sont inclus.

## Identité et conditions

Base publiée `bde5e44f702cb61158bd1b5bdf67c364568a686b`, code local64H ; [manifeste v4](measurements/flash-next-ssd-impl/opt-64-unique-cohort/identity-v4.json),146 sources/deux artefacts, archive `ROOT/build/flash-opt-64/hot-v4/source.tar.gz`.

- CLI : `5f94eaa05cb0fdcaf310e94043cea134a8ff1debfbb79bd1e8161474119ebcf9`.
- Tests MLX v4 : `9c11915ff71d93f09c9489fd656558130d0e3d63fc76ed80d047a9d46fd666a4`.

Qwen3.8 Flash Next EXL3 2,05bpw65c89531 ; DFlash2a22a1744 Q4/ring/adaptatif7 ; M5 10 cœurs GPU/24Gio, MLX0.32.2, secteur et applications ouvertes. Cache3Gio/53slots, quatre lecteurs,63/Hot/Wide1, lookahead0/LES0, allocateur512Mio. `MLXL3_SSD_UNIQUE_COHORT` reste strict0/1, défaut0 ; application installée inchangée.

## Vérification

Commandes de base exécutées sur la correction :

```text
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --no-default-features
cargo test --locked --features mlx,chat
cargo build --locked --release --features mlx,chat
cargo test --locked --release --lib --features mlx,chat --no-run --message-format=json
cargo kani --lib --tests --features mlx,chat --harness actual_unique_window_demand_entry
```

Format/Clippy/build passent ;115 tests CPU passés/3 ignorés et144 MLX/chat passés/144 ignorés, tests physiques séparés. Le changement v4 ne modifie que le cinquième test physique ; la production et la CLI sont identiques à v3, avec nouvelle compilation native et Clippy. La tentative Kani sur le chemin GPU réel échoue encore avant obligations dans `intrinsics.rs:243` : non prouvé. Le helper CPU inchangé conserve sa vérification20 IDs0..511/capacité0..512/unwind21 ;80 IDs dépassent180s. Ces contrôles ne prouvent ni MLX, ni Metal, ni la concurrence.

La mutation qui remet la préparation Hot après le readback échoue sur le nouvel assert d'ordre, avec exactement un test sélectionné. Source restaurée octet par octet ; [résultat de mutation](measurements/flash-next-ssd-impl/opt-64-unique-cohort/mutation-hot-v3/result.json).

La première campagne physiquev3 passe les24cas uniques, le Hot mono-ligne asynchrone, son erreur/reprise et les12cas Wide. Le cinquième test attendait toujours deux cohortes sous pending, même avec UNIQUE1 :4 passés/1 échoué, logs conservés. v4 corrige uniquement ce test pour exercer UNIQUE0/1, sixcas de prélectures protégées. Les preuves des quatre tests inchangés sont réutilisées ; manifeste et CLI identiques hors ce fichier de test.

Les24cas vérifient notamment B6+B2 et B4+B4 sous pending, une seule remontée CPU, une seule soumission Hot, contribution Hot réutilisée après éviction, hit Cold calculé pour les nouvelles activations, erreur I/O et reprise. Sur le cas F32 contrôlé :55records Cold demandés après les cinq experts initialement chauds, aucune relecture des anciens Hot.

La reprisev4 couvre ensuite36préfixes DFlash/logits/états, les sessions MTP budgets1/2/3/17 avec acceptation et rejet, dix commits pair avec logits/états exacts et les43contrôles du vrai bridge, erreurs et annulation/reprise incluses. [Plan et résultats de qualité](measurements/flash-next-ssd-impl/opt-64-unique-cohort/quality-hot-v4/).

## Débit et mémoire

La qualité v4 et son postcontrôle30s passent : six cas pending,36préfixes DFlash, sessions MTP et dix commits pair,43contrôles bridge. Les quatre tests physiques v3 inchangés et la mutation d'ordre détectée sont réutilisés avec identité de la production vérifiée.

Matrice DFlash adaptatif MAIN/HOLD HCCH :128 sorties/127 pas décodés par réponse, warmup puis mesure dans chaque processus neuf ; seul UNIQUE0/1 varie dans le même CLI.

| Prompt | Témoin H | Candidat C | Candidat C | Témoin H | Gains appariés |
| --- | ---: | ---: | ---: | ---: | --- |
| MAIN |10,6970|10,7994|10,5107|10,6212|+0,957% / −1,040%|
| HOLD |9,8882|9,8471|9,7577|9,6115|−0,416% / +1,522%|

Débits en tok/s. Dérive des témoins HCCH0,715%/2,879%, étendue des gains1,997/1,937points : stabilité admise, gain≥1% dans chaque paire absent, **gain rejeté**. Huit processus joints,16réponses exactes, empreinte physique maximale8 255 920 680octets, Swapin524288octets dans la fenêtre et0nouveauSwapout depuis l'origine qualitév3. [Résultat et qualification v4](measurements/flash-next-ssd-impl/opt-64-unique-cohort/speed-result-hot-v4.json).

Les témoins OFF/MTP avant/après sont clos aussi. OFF MAIN10,5114→10,2894, HOLD10,0452→8,7963 ; MTP MAIN10,9259→10,6418, HOLD10,0134→9,5350tok/s. La dérive OFF/HOLD dépasse5% : comparaison absolue des trois modes **non concluante** sur cette longue fenêtre, distincte du rejet HCCH court. Aucun débit15tok/s ni dépassement reproductible de MTP établi. [Clôture et contrôles](measurements/flash-next-ssd-impl/opt-64-unique-cohort/closure-hot-v4.json).

La qualitév3 commence après30s sans modèle puis20s de précontrôle, avec une nouvelle origine explicite après le diagnostic mémoire échoué. v4 **conserve cette même origine**, sans remise à zéro après le test historique échoué. Zéro nouveau Swapout est exigé dans la nouvelle campagne ; le précédent échec de103,8Mo reste visible et n'est pas requalifié. Incoming256Mio par fenêtre courte, empreinte physique≤12 milliards d'octets, postcontrôle30s et processus joints. Aucun build/proveur/autre modèle pendant les contrôles GPU.

Les durées de vérification sont hôte inclusives, les octets des records demandés ne sont pas les octets NVMe physiques, et le contrôleur adaptatif peut changer les largeurs. P0/Xcode, application installée et CI du correctif restent non validés à ce stade.
