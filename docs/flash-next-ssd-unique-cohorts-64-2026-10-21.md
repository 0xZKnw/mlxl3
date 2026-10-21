# Réunification des cohortes SSD — expérience 64U

21 octobre 2026. **Version v2 : correction validée, débit non concluant et aucun gain publié. Révision du recouvrement Hot en cours dans64H.** Cet essai fait suite au [profil64P](flash-next-ssd-verifier-profile-64-2026-10-21.md) : 688/768 fenêtres qui avaient été découpées tenaient théoriquement dans53 experts distincts. Cela ne concernait pas toutes les vérifications et ne prédisait aucun pourcentage de gain.

## Changement isolé v2, archivé

`MLXL3_SSD_UNIQUE_COHORT=1` calcule les routes de la fenêtre complète une seule fois, puis choisit le plus grand préfixe dont les experts distincts tiennent dans les slots effectivement disponibles. `available_slots()` exclut les prélectures en cours ; `reserve()` protège les experts sélectionnés et le chemin existant attend les consommateurs précédents avant réutilisation des banques. Le repli conserve les IDs et les poids routés, sans relancer le routeur. Une fenêtre B8 peut rester B8, ou devenir B6+B2 lorsque c'est le plus grand préfixe admissible.

La voie est limitée à GPU Hot + Wide + overlap, sans préfetcher ni lookahead. Le Hot de chaque cohorte reste soumis avant ses lectures SSD. Cette version lit toutefois les routes complètes avant de préparer ce Hot : son effet sur le recouvrement hôte/GPU doit être mesuré. Les calculs EXL3 par occurrence restent inchangés ; aucun partage de déquantification du prototype58 n'est inclus.

Option stricte0/1, défaut0. Aucun slot, lecteur ou tampon SSD supplémentaire. OFF et MTP1 ordinaires utilisent une ou deux lignes et conservent leur chemin avec53 slots.

## Sources et protocole

Base publiée `bde5e44f702cb61158bd1b5bdf67c364568a686b`, modifications locales64U. [Identité v2](measurements/flash-next-ssd-impl/opt-64-unique-cohort/identity-v2.json) :146 fichiers source et deux artefacts figés ; archive sous `ROOT/build/flash-opt-64/unique-v2/`. Un ancien résumé comptait147 sources par erreur ; la liste effective est inchangée.

- CLI : `cf6861fd4eef308d1218b73239553024648f5de5008d793cc986c544cc516c6a`.
- Binaire de tests MLX : `d61c2ef5dd8a20c77585334b76ecd8ac03693b2b409153635db32c7b638fae91`.

Qwen3.8 Flash Next EXL3 2,05bpw `65c89531`, draft DFlash2 `a22a1744` Q4/ring/adaptatif7 ; M5 10 cœurs GPU/24Gio, MLX0.32.2, secteur, applications ouvertes. Cache3Gio/53slots, quatre lecteurs, Hot1/Wide1, vérificateur63 activé, lookahead0/LES0, allocateur512Mio.

Débit :128 sorties/127 pas décodés, un warmup puis une mesure dans chaque processus neuf. DFlash MAIN puis HOLD, ordre HCCH : témoin UNIQUE0, candidat1, candidat1, témoin0. UNIQUE est la seule option modifiée. Qualification préexistante : au moins1% dans chacune des quatre paires, dérive des témoins≤5%, étendue des gains≤3points par prompt. OFF/MTP avant et après contrôlent les chemins inchangés et la comparaison entre modes ; ils ne servent pas à attribuer un faux gain64U à MTP.

Un ancien plan63 a initialement fourni quatre témoins avec MULTI_TOKEN0. Les deux OFF restent admissibles pour leur chemin B1 inchangé ; les deux MTP0 sont conservés comme protocole distinct et exclus de la comparaison MTP63=1. La reprise préenregistrée ne refait que les deux témoins MTP avec63 activé. Aucun bras DFlash n'avait démarré lors de cette correction.

Un seul modèle GPU à la fois ; aucun build ou proveur pendant les mesures. Précontrôle20s et postcontrôle30s, empreinte physique≤12 milliards d'octets, zéro nouveau Swapout depuis l'origine qualitév1. Chaque courte campagne a une origine Swapin explicite et la même borne256Mio. L'historique d'un essai interrompu n'est pas effacé.

## Correction et vérification

- Tests CPU :115 passés/3 ignorés ; suite `mlx,chat` :144 passés/144 ignorés, tests physiques sélectionnés séparément. Format, Clippy strict et build de la CLI livrée passent.
- Helper :4096 cas déterministes contre un oracle `BTreeSet`, frontières de formes et IDs, capacité exacte et slots pending. Mutation `unique > available` en `>=` détectée ; source restaurée et test repassé.
- MoE sur GPU physique :20 cas, fenêtres6..8, capacités20/24/53, slots pending protégés, repli maximal, Hot asynchrone, erreur I/O et reprise. Sorties octet par octet contre captures indépendantes et calcul séquentiel ; une seule lecture des routes, contre deux avec l'ancien découpage.
- Cible complète :36 préfixes/commits,248320 logits finis, états et captures exacts contre l'oracle séquentiel. Vrai bridge/CLI :43 contrôles, dont erreurs et annulation/reprise. [Clôture qualité v2](measurements/flash-next-ssd-impl/opt-64-unique-cohort/quality-v2/campaign-result.json).
- Kani sur le helper réel :20 IDs symboliques0..511, capacité0..512, unwind21 ;376 obligations SUCCESS, trois UNREACHABLE dans libcore identifiées, trois couvertures0/1/2 lignes satisfaites. L'égalité avec l'oracle indépendant est SUCCESS. La tentative80 IDs dépasse180s ; la tentative du parcours GPU réel échoue dans Kani sur `intrinsics.rs:243`, avant obligations. **80 IDs, GPU et concurrence ne sont pas prouvés.** [Commandes/résultats](measurements/flash-next-ssd-impl/opt-64-unique-cohort/proof-results.json), [obligations individuelles](measurements/flash-next-ssd-impl/opt-64-unique-cohort/kani20-results.json).

La première campagne qualité a échoué sur une assertion de cohorte du test, pas sur une divergence numérique : son oracle F16 supposait à tort le routage F32 synthétique. L'échec et ses logs restent conservés. Seul l'oracle du test a été corrigé ; la CLI v1/v2 a la même empreinte. Les trois processus qualitév2 sont joints et le postcontrôle30s passe sans nouveau Swapout.

## Résultats de débit

Huit processus joints/16 réponses exactes, postcontrôle30s passé. [Résultats et qualification](measurements/flash-next-ssd-impl/opt-64-unique-cohort/speed-result.json), [clôture v2 avec témoins admissibles](measurements/flash-next-ssd-impl/opt-64-unique-cohort/closure-v2.json).

| Prompt | Témoin0, tok/s | Candidat1, tok/s | Gains appariés | Dérive des témoins |
|---|---:|---:|---:|---:|
| MAIN |10,542334 /9,992408 |10,448746 /9,883825 |−0,888% /−1,087% |5,503% |
| HOLD |9,672298 /8,861013 |9,447801 /8,444980 |−2,321% /−4,695% |9,156% |

**Non concluant selon le seuil préenregistré de5% de dérive ; les quatre paires observées sont négatives.** Aucun boost64U, ni victoire DFlash face à MTP. Les lignes vérifiées varient avec le feedback adaptatif : MAIN136 chez les témoins contre140/138 chez les candidats ; HOLD136/139 contre141/141. Le temps global ne peut donc pas être attribué à un kernel seul. Une baisse du nombre de cohortes ne suffit pas à qualifier un gain de génération.

Pic physique de cette matrice :8 259 345 008octets ;79 527 936octets de Swapin dans sa fenêtre, zéro nouveau Swapout depuis l'origine qualitév1.146 sources/deux artefacts sont recontrôlés après les mesures.

| Témoins inchangés | Avant, tok/s | Après, tok/s | Dérive |
|---|---:|---:|---:|
| OFF MAIN |10,438127 |9,916740 |5,258% |
| OFF HOLD |9,843351 |9,511525 |3,489% |
| MTP63 MAIN |10,507146 |10,400213 |1,028% |
| MTP63 HOLD |9,920115 |9,665622 |2,633% |

Ces contrôles sont deux mesures dans le temps, pas des paires64U OFF/ON. Les deux premiers MTP63=0 restent archivés séparément et sont exclus du tableau. Les [plans, commandes et résultats bruts](measurements/flash-next-ssd-impl/opt-64-unique-cohort/) restent conservés.

## Révision64H en cours

Le changement v2 a déplacé le readback CPU avant la préparation Hot. C'est une perte de recouvrement à examiner avant de rejeter le mécanisme de réunification.64H profile la version v2 figée, puis rétablira Hot avant readback sur un snapshot stable, en réutilisant les contributions Hot même dans le repli. Aucun partage EXL3 des experts répétés n'est inclus. Les résultats v2 ci-dessus resteront distincts des nouvelles mesures.

Le [rapport64H](flash-next-ssd-early-hot-64-2026-10-21.md) suit cette correction séparément. Le diagnostic v2 confirme l'ordre tardif sur240 fenêtres par trace, mais échoue à la garde mémoire avec103 809 024octets de nouveau Swapout ; ses temps ne constituent pas un gain qualifié. Cet échec reste distinct de la matrice v2 ci-dessus, qui avait passé ses contrôles mémoire.

## Limites

Les temps de vérification sont des durées hôte inclusives ; les lectures comptent des octets de records retournés, pas les octets NVMe physiques. La politique adaptative peut modifier les largeurs vérifiées et le nombre de propositions entre répétitions. Ne pas additionner les gains63, Wide et64U mesurés dans des fenêtres distinctes. P0/Xcode, application installée et15tok/s restent hors validation de cet essai.
