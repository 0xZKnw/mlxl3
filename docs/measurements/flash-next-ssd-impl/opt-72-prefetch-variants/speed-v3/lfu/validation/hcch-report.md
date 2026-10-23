# 72 lfu — bilan HCCH

Gain global non qualifié. Les critères réels sont listés ci-dessous ; statut enum hérité inconclusive_drift, qui inclut aussi une étendue excessive des gains sans que la dérive H échoue forcément. Aucun gain promu. Les plafonds de compression77B restent échoués indépendamment du débit.

| Prompt | Gain HC | Gain CH | Dérive H | Étendue des gains | Moyenne C |
|---|---:|---:|---:|---:|---:|
| MAIN | +3.307757 % | +1.380000 % | 2.123494 % | 1.927756 points | 8.923999412 tok/s |
| HOLD | +4.079236 % | +0.839914 % | 1.209188 % | 3.239323 points | 8.629764206 tok/s |

Critères non satisfaits :
- {"prompt": "HOLD", "gate": "minimum_pair_gain_percent", "phase": "CH", "actual": 0.8399135390610546, "minimum": 1}
- {"prompt": "HOLD", "gate": "maximum_observed_gain_spread_points", "actual": 3.2393227844469186, "maximum": 3}

Les étendues décrivent deux paires et ne sont pas des intervalles de confiance. Les gains ne sont ni additionnés ni combinés avec ceux des autres candidats. Les quatre paires emploient leurs témoinslegacy propres. Un écran positif seul ne prouve pas un gain reproductible ; une baisse isolée ne prouve pas une régression causale.

| Prompt | Sans prélecture B2 frais | MTP frais | Moyenne candidat vs B2 frais | Moyenne candidat vs MTP frais |
|---|---:|---:|---:|---:|
| MAIN | 10.222852654 | 9.833896963 | -12.705389 % | -9.252665 % |
| HOLD | 9.486925854 | 8.907714222 | -9.035189 % | -3.120329 % |

Références fraîches non appariées, comparaisons descriptives seulement. Elles ne valident pas un dépassement du meilleur moteur sans prélecture ou de MTP.

16 réponseswarm+mesure ont passé golden60textecompletnonvide/hash/128sorties127decode ; huit enfants et lecteurs joints, samplers et pré20/post30 secteur complets. Même acceptation et mêmes largeurs vérifiées H/C dans les deux ordres. Prélectures réellement soumises/complètes/utilisées. Aucun bridge43candidatv3 exécuté ; packed43 n’est pas transféré. Aucun nouveau contrôle logits/états pendant ce benchmark.

Picphysiquemaximal7853807904B, deltasledger_swapins[0, 0, 0, 0, 0, 0, 0, 0]B. SwapoutsHC/CH{'HC': [0, 0, 0, 0, 0, 0], 'CH': [0, 0, 0, 0, 0, 0]} ; SwapinsHC/CH{'HC': [0, 0, 0, 0, 0, 0], 'CH': [0, 0, 0, 573440, 0, 0]}. Huit analyses77Bfailed/source_errorsvide, capscompressioninchangés. Ces compteurs ne prouvent pas l’absence d’écritures swap du processus.

Provenance/protocole et compteurs complets dans screen/summary.json, validation/summary.json et validation/hcch-analysis.json. CLI77v1gelé69b919ae, mêmes poids/drapeaux horsPREFETCH_VARIANTlegacy→lfu ; codearithmétique/driver/observer inchangés. Aucun code/plan/seuil modifié ni GPU/build/proveur/test lourd lancé par cet analyste.
