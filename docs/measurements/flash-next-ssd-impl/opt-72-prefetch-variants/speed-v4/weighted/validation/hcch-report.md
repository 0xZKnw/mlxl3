# 72 weighted — bilan HCCH

**Décision actuelle — retrait utilisateur72W-WITHDRAW.** Le vrai A/B contreB2sansprélecture estdéfavorable danssonHC : MAIN−1,557172 %, HOLD−18,111644 %. L’utilisateurannule72W-KEEP ; aucunevarianteweightedàpublier, CHannulé/nonlancé. Lesanciensgainsvslegacyet décisionsci-dessoussont historiques ; lesdonnéesnumériqueset caps77Bfailed restentinchangées.

Gain global non qualifié. Les critères réels sont listés ci-dessous ; statut enum hérité inconclusive_drift, qui inclut aussi une étendue excessive des gains sans que la dérive H échoue forcément. Aucun gain promu. Les plafonds de compression77B restent échoués indépendamment du débit.

| Prompt | Gain HC | Gain CH | Dérive H | Étendue des gains | Moyenne C |
|---|---:|---:|---:|---:|---:|
| MAIN | +19.951846 % | +16.512950 % | 0.033323 % | 3.438896 points | 9.451969495 tok/s |
| HOLD | +6.780078 % | +12.245518 % | 9.802431 % | 5.465439 points | 10.052576043 tok/s |

Critères non satisfaits :
- {"prompt": "MAIN", "gate": "maximum_observed_gain_spread_points", "actual": 3.4388960533176203, "maximum": 3}
- {"prompt": "HOLD", "gate": "maximum_h_control_drift_percent", "actual": 9.802431067489014, "maximum": 5}
- {"prompt": "HOLD", "gate": "maximum_observed_gain_spread_points", "actual": 5.465439412061901, "maximum": 3}

Les étendues décrivent deux paires et ne sont pas des intervalles de confiance. Les gains ne sont ni additionnés ni combinés avec ceux des autres candidats. Les quatre paires emploient leurs témoinslegacy propres. Un écran positif seul ne prouve pas un gain reproductible ; une baisse isolée ne prouve pas une régression causale.

| Prompt | Sans prélecture B2 frais | MTP frais | Moyenne candidat vs B2 frais | Moyenne candidat vs MTP frais |
|---|---:|---:|---:|---:|
| MAIN | 10.222852654 | 9.833896963 | -7.540783 % | -3.883786 % |
| HOLD | 9.486925854 | 8.907714222 | +5.962418 % | +12.852476 % |

Références fraîches non appariées, comparaisons descriptives seulement. Elles ne valident pas un dépassement du meilleur moteur sans prélecture ou de MTP.

16 réponseswarm+mesure ont passé golden60textecompletnonvide/hash/128sorties127decode ; huit enfants et lecteurs joints, samplers et pré20/post30 secteur complets. Même acceptation et mêmes largeurs vérifiées H/C dans les deux ordres. Prélectures réellement soumises/complètes/utilisées. Aucun bridge43candidatv3 exécuté ; packed43 n’est pas transféré. Aucun nouveau contrôle logits/états pendant ce benchmark.

Picphysiquemaximal7853824288B, deltasledger_swapins[0, 0, 0, 0, 0, 0, 0, 0]B. SwapoutsHC/CH{'HC': [0, 0, 0, 0, 0, 0], 'CH': [0, 0, 0, 0, 0, 0]} ; SwapinsHC/CH{'HC': [0, 0, 32768, 0, 0, 0], 'CH': [0, 0, 131072, 0, 0, 0]}. Huit analyses77Bfailed/source_errorsvide, capscompressioninchangés. Ces compteurs ne prouvent pas l’absence d’écritures swap du processus.

Provenance/protocole et compteurs complets dans screen/summary.json, validation/summary.json et validation/hcch-analysis.json. CLI77v1gelé69b919ae, mêmes poids/drapeaux horsPREFETCH_VARIANTlegacy→weighted ; codearithmétique/driver/observer inchangés. Aucun code/plan/seuil modifié ni GPU/build/proveur/test lourd lancé par cet analyste.

## Révision de décision72W-KEEP — 23octobre2026

L’utilisateur demande explicitement de conserver ce gain malgré sa variabilité. Les verdictsstricts ci-dessus décrivent les critères72S originaux et restent inchangés ; ils ne rejettent plus l’intégration. Weighted est désormais **retenu à la demande utilisateur, validation bridge43 en attente**, avec les quatre gainsvslegacy positifs et leur plage conservée (MAIN+16,512950 à+19,951846 %, HOLD+6,780078 à+12,245518 %). Aucuneaddition, aucuneprésomptiondeboostvsnofetch/MTPnonappariés, aucun15tok/sannoncé. Les capscompression77B échoués restent échoués ; l’app installée n’est pas modifiée. La publication du chemin opt-in reste conditionnée au bridge43weighted, après les témoins d’encadrement récents.

Actualisation après bridgeweightedv4 :43/43 validés, paritétrue etpost30joint. La mention «bridgeenattente» ci-dessus décrit la décision initiale72W-KEEP ; la qualificationfonctionnelle propreaucandidat estdésormaisclose. Publicationopt-inenattenteduparent, nouvelleA/B72W-Ren cours séparément. Lesverdictsstrictshistoriqueset77Bfailedrestentinchangés.
