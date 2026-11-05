# Réunification : hits, lectures et largeurs réelles —64D

21 octobre2026. **Diagnostic clos, compteurs contre-vérifiés ; aucun gain de débit qualifié.** Source moteur et artefacts v6 de `aa5bc75f` inchangés. Le correctif LFU64L est exact, mais la réunification économise peu de lectures sur ces prompts. La prochaine piste est de supprimer du décodage EXL3 répété, en adaptant réellement le groupement aux grandes fenêtres.

## Protocole et preuves

Qwen3.8 Flash Next EXL3 2,05bpw `65c89531`, draft DFlash2 `a22a1744` Q4/ring/adaptatif7, M5/24Gio/MLX0.32.2, secteur100%, applications ouvertes. Cache3Gio/53slots,4lecteurs, allocateur512Mio,63/Hot/Wide1,LES/lookahead0. UNIQUE0/1 seul paramètre variable. Quatre processus dans l'ordre MAIN0/MAIN1/HOLD1/HOLD0,64sorties/63tokens décodés. Chacun exécute oracle OFF, chauffe, témoin avant, deux traces naturelles, témoin après :24générations et8traces. Aucun `eval()` de profilage ajouté. Pas de compilation/proveur/autre modèle simultané.

Les quatre enfants sont joints en92,194/91,712/88,506/90,962s. Les sorties,248320logits complets et tous les états cible sont exacts ; le dataset vérifie aussi l'oracle externe. Pic physique8 461 097 944octets (8,461Go décimaux), incoming maximal1 769 472octets et0nouveau Swapout depuis l'origine conservée121 528 483 840octets. Pré20/post30passent. Le RSS est distinct de cette empreinte physique.

[Commande et matrice](measurements/flash-next-ssd-impl/opt-64-unique-cohort/cache-width-64D-command.txt), [résultat et origine](measurements/flash-next-ssd-impl/opt-64-unique-cohort/cache-width-64D/campaign-result.json), [dataset/oracles](measurements/flash-next-ssd-impl/opt-64-unique-cohort/cache-width-64D/dataset-result.json), [identité native](measurements/flash-next-ssd-impl/opt-64-unique-cohort/identity-v6.json), [analyse complète](measurements/flash-next-ssd-impl/opt-64-unique-cohort/cache-width-64D/cache-width-analysis.json).

Le posttraitement attribue les événements à leur fenêtre cible, vérifie48couches et10routes par ligne, puis contre-vérifie hits/misses/octets aux compteurs backend après−prefill et à `expert_demand_bytes`. Il distingue le dernier pas ordinaire des vérifications. Les9tests déterministes couvrent regroupement, pas final et données incomplètes/corrompues ; Ruff et pytest passent. CrossHair tenté : module absent, analyse Python **non prouvée**. CI ajoutée sur PR/push ; les résultats locaux ne signifient pas que sa nouvelle exécution est verte.

## Résultat discriminant

Somme de deux traces par cellule ; Mo=10⁶octets, moyenne par token émis :

| Prompt | UNIQUE | Hits logiques | Misses logiques | Taux hit | Mo demandés/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAIN | 0 | 42023 | 21337 | 66,324% | 181,685 |
| MAIN | 1 | 41776 | 22544 | 64,950% | 177,475 |
| HOLD | 0 | 42600 | 20760 | 67,235% | 178,703 |
| HOLD | 1 | 42415 | 20945 | 66,943% | 174,335 |

Les records demandés diminuent de2,317%MAIN et2,444%HOLD. Ce sont des octets d'appels de lecture de records EXL3, **pas des octets physiques NVMe mesurés**. Les profils naturels ne décomposent pas le temps GPU exclusif.

Un taux hit inférieur ne démontre pas une politique moins bonne : un expert absent utilisé deux fois dans une seule cohorte produit deux occurrences misses mais une seule lecture. Deux cohortes successives produisent un miss puis un hit avec le même volume lu. Un test de régression reproduit ce cas. L'adaptatif et les réservations par cohorte changent aussi ; ce diagnostic n'isole pas l'effet causal du correctif LFU seul.

| Prompt/UNIQUE/trace | Histogramme de largeur cible (largeur:fenêtres) |
| --- | --- |
| MAIN/0/1 et2 | 1:3,2:7,3:2,5:1,6:1,8:4 |
| MAIN/1/1 et2 | 1:2,2:8,3:2,5:1,6:1,8:4 |
| HOLD/1/1 | 1:5,2:6,3:2,4:2,5:2,8:3 |
| HOLD/1/2 | 1:3,2:7,3:2,4:2,5:2,8:3 |
| HOLD/0/1 | 1:5,2:6,3:2,4:2,5:2,8:3 |
| HOLD/0/2 | 1:7,2:5,3:2,4:2,5:2,8:3 |

MAIN vérifie66lignes en UNIQUE0 contre67en UNIQUE1 ; HOLD66lignes cible dans tous les bras, pas ordinaire final compris. Ces traces64sorties ne sont **pas** la reconstruction des largeurs du bras vitesse128sorties à8,433tok/s. Le [rapport64L](flash-next-ssd-hot-demand-64-2026-10-21.md) garde tous les résultats de vitesse et la panne temporaire/reprise des témoins.

## Suite : partage du décodage, comparaison amont

La revue fixe [Strata `011c9e60`](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/kernels/moe.metal#L167) et [Splash `1482f6bb`](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/shared/moe.metal#L136). Strata décode les poids hors de la boucle des occurrences (`gu_body`/`dn_body`), avec1–4accumulateurs et des groupes de4pour les grands groupes. Splash construit les groupes d'experts actifs avant les tuiles et utilise un repli adapté aux petits groupes ; son scratch/dispatch persistant diffère de MLX. Leurs formats de quantification ne valident pas nos arrondis EXL3.

L'ancien prototype58 ne traitait que B2, avec exactement deux occurrences ; trois occurrences revenaient aux singletons. Son plan GPU utilisait un seul SIMD32 et des shuffles : remplacer simplement20par80routes est incorrect. Il n'a jamais été qualifié sur GPU ni en débit. Le compact57 seul avait régressé ; il ne doit pas être réactivé implicitement.

Dans les cohortes réellement exécutées, la fraction théorique d'occurrences dont on pourrait éviter un deuxième décodage passe de28,49→36,70%MAIN et27,09→31,36%HOLD avec UNIQUE. En partageant seulement par paires, les plafonds deviennent22,00→26,38% et20,66→22,79%. Ce sont des **bornes supérieures de décodages de poids**, avant séparation Hot/Cold, pas des gains de calcul total, SSD ou tok/s. Le prochain essai doit préserver Hot anticipé, partager seulement les vrais experts répétés et mesurer son surcoût de plan/accumulateurs. Aucune promesse de15tok/s.
