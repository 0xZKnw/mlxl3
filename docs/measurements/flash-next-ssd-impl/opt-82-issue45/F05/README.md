# 82F05 — softmax et top-10 FP32 groupés

**Clôturé : qualité passée sur le domaine testé, aucun gain observé sur H-C, prototype effectivement retiré. Aucun boost F05 livré.** Les seuls hunks F05 des trois fichiers suivis ont été inversés, sans écraser de travail privé. Le contre-audit `withdrawal-static-pre-K1.json` a constaté ces trois fichiers exactement identiques à HEAD b5898866 **avant** la nouvelle application K1. Le test F05 est préservé sans référence de module et ne compile plus dans le moteur actif. Les futurs changements K1 sont distincts ; l’application installée est inchangée.

## Mécanisme essayé

Sur FP32 / B2…B8, les GEMV de gates restaient strictement scalaires par ligne ; seuls les logits étaient concaténés avant un softmax précis et un top-10 multi-lignes normalisé. FP16 et B1 gardaient l’ancien chemin. Flag `MLXL3_SSD_BATCH_ROUTES=0|1`, strict, lu au chargement, défaut 0. Les deux sites de routage — fenêtre unique et chemin standard/cohortes — étaient reliés. Aucun shader/FFI, poids, SSD, lease, fence, politique de cache ou calcul expert modifié. F01 et F06 absents, aucun gain combiné.

## Écran réel H-C : aucun signal positif

M5 10 cœurs / 24 Gio, Flash Next 2,05 bpw `65c89531`, draft Q4 `a22a1744`, MLX 0.32.2. Secteur, cache 3 Gio / 53 slots, allocateur 64 Mio, GPU Hot/Wide/UNIQUE/multi-token actifs, prélecture/LES/shared/stable désactivés. DFlash2 fixe B2 : 128 tokens de warmup puis 128 mesurés dont 127 decode par processus. Même CLI `bb7d25de` et contrôleur 82B `8935`, seul BATCH_ROUTES diffère H0/C1.

| Prompt | H tok/s | C tok/s | Écart C/H | Warmup→mesure H / C |
|---|---:|---:|---:|---:|
| MAIN | 10,180175 | 9,807010 | −3,6656 % | +4,9314 % / −0,6502 % |
| HOLD | 9,652619 | 8,869685 | −8,1111 % | +0,7037 % / −5,9518 % |

Retrait conservateur ; pas de confirmation C-H, de débit OFF/MTP1 ni de replay de vitesse après cet écran négatif. Ces paires uniques ne démontrent pas une régression causale universelle. La dérive HOLD-C de −5,95 % est conservée, sans explication thermique, compression ou ordonnanceur inventée.

Les huit réponses warmup/mesure ont texte et hash gold60 exacts, 128/127, modes/compteurs/largeurs/octet SSD logique identiques entre bras par phase. MTP inactif a des compteurs JSON null ; les histogrammes ont huit cases, index 1 = ancre + une proposition. Pas de trafic SSD physique mesuré. Vérification/draft/commit sont des intervalles inclusifs côté hôte, pas des temps GPU exclusifs à ajouter aux totaux block/decode. Voir `cross-review-screen.{json,md}` et `summary-final.json`.

## Qualité et vérification inspectées

- CPU : 117 lib réussis / **cinq** ignorés, seize intégrations ; l’ancien « six ignorés » de métadonnées est conservé et corrigé à partir du log. Suite native : 130 réussis / 150 ignorés. Format, Clippy CPU+MLX et builds inspectés ; quinze sélections physiques non vides et un refus CLI non Unicode exécutés explicitement.
- 28 cas de bords en FP32, B2/B3/B4/B8 : ±0, égalités, extrêmes, NaN/Inf intentionnels ; comportement historique non fini préservé sans qualifier arbitrairement ces valeurs de finies.
- 1 004 cas de gates réelles sur les 48 couches et deux activations oracle par couche, dont 404 fast paths ; dtypes F16/F32, vues contiguës/stridées, largeurs et chemin resident_gate. Logits face aux GEMV mono-ligne, probabilités/routes face à la voie scalaire, tri CPU indépendant des 512 probabilités. Compteurs réels imposant B appels de gate et une branche groupée admissible.
- 24 cas MoE complets : cache 20/53, quatre slots pending, lignes 1/2/3/8, legacy/U et cinq vrais experts Hot. Sorties exactes et claims protégés. La chronologie Hot avant readback est explicitement vérifiée pour **quatre** fenêtres U débordantes, pas généralisée aux vingt autres. Les **32 cas historiques U** supplémentaires couvrent repli 6+2/4+4, éviction, pending et erreur de lecture/reprise ; il ne s’agit pas de 32 cas de dtype UInt32.
- Neuf processus loaders à environnement fixé : défaut/0/1 et six littéraux invalides ; plus byte 0xff refusé par une CLI distincte. Aucun changement d’environnement après MLX. Les invalides natifs ont exit 0 parce que le test vérifie le refus attendu.
- 36 préfixes DFlash, captures externes (64 000 valeurs exactes observées), logits/états/commits exacts serial/groupé sous flag 1 commun ; quatre budgets MTP, quinze propositions et cinq acceptations, séquence et états cible/tête exacts. Aucun différentiel intégral cible 0/1 revendiqué au-delà des composants/routes. Bridge : 43 requêtes, parité, erreurs attendues, annulation/reprise. Son transport ne certifie pas les événements tardifs de fermeture ; aucun incident tardif observé. Le contrôleur TPS 82B vérifie sa file et son EOF.

## Preuve, mutations et échecs

Kani sur le vrai helper de forme : 97 obligations répertoriées réussies et cinq couvertures satisfaites ; quatre dimensions i32 et précision booléenne symboliques, rang 0…4, unwind 5. Une hypothèse borne le rang pour indexer le tableau symbolique fixe de quatre éléments. Contrat d’admissibilité seulement : ni GEMV, softmax, top-10, Metal, allocator, slots ou concurrence ne sont prouvés.

Première tentative MLX : nom du test ignoré non sélectionné, échec conservé. Seconde : wrapper réel correctement sélectionné, ICE Kani `intrinsics.rs:243`, environ 4,62 s / 748,5 Mo RSS groupe, sous limites 180 s / 4 Gio ; propriété numérique MLX **non vérifiée**, aucun stub.

Trois mutants détectés puis sources restaurées et contrôles affectés repassés : admission B1, retrait de normalisation, remplacement des GEMV scalaires. Ce dernier échoue sur le compteur d’appels ; ne pas prétendre qu’une divergence numérique GEMM a été mesurée. Mutants debug et normal release sont des profils différents, aucun débit comparé entre eux.

Le premier préflight vitesse appelait `/bin/uname`, absent : FileNotFoundError et **aucun lancement GPU**, grâce à errexit. Chemin corrigé vers `/usr/bin/uname`, identités renouvelées, pas de relance de qualité non affectée. Les variantes préparatoires modifiant l’environnement après load ont été retirées **sans exécution** ; patchs/manifeste historiques conservés. Les échecs d’application/préparation et versions précédentes restent visibles.

## Mémoire et portée

Écran H-C : six phases, enfants/lecteurs et sampler joints, secteur AC ; **zéro swap-out système, 131 072 octets de swap-in sur MAIN-C**, origine inconnue. Pic physique 7 846 877 496 octets ; compression logique courante maximale 3 947 593 728 octets ; ledger swap-in propre zéro aux snapshots.

Bridge, séparément : **136 314 880 octets de swap-out système (130 Mio)** et 851 968 de swap-in, pic processus 7,937 Go, compression logique max 3,995 Go. Garde de 256 Mio passée, critère zéro global non atteint. Ces compteurs système ne donnent pas le propriétaire des pages. Ledger zéro ne prouve pas l’absence de swap-out privé ; les caps compression 77B restent FAILED. Aucun diagnostic de fuite ou de cause exclusive de latence établi. Les octets logiques compressés ne sont pas la taille physique du compresseur.

## Archive et retrait

`evidence/source-snapshot.tar.gz` contient les 86 sources du gel, toutes concordantes. `integrated-source.patch` conserve les trois fichiers suivis et le nouveau test ; `tracked-integrated.patch` est le patch inversé. `withdrawal-static-pre-K1.json` date le retrait avant K1 ; ne pas comparer le nouveau W/K1 aux sources F05 mesurées comme s’il s’agissait d’une modification silencieuse de ces résultats.

Les copies complètes base/candidate/application-check et leurs doublons superseded ne sont pas livrées : base Git, patchs et snapshot conservent la reproduction utile. Patchs, manifestes, versions abandonnées, logs, erreurs, commandes/options et preuves sont conservés. Les anciennes mentions « préparé » des documents datés ne remplacent pas ce README et `summary.json` : **runtime retiré**.

Publication documentaire uniquement : aucun runtime/binaire/modèle, test détaché natif ou boost livré ; application installée inchangée. Aucune CI de base ou d’archive n’est présentée comme une CI verte d’un moteur F05 publié. Aucun gain cumulé, 15 tok/s non atteint.
