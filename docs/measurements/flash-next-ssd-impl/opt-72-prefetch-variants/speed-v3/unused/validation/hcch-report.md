# 72 unused — bilan HCCH

Le gain HC ne se reproduit pas. MAIN passe de+2,4957 % enHC à−1,7806 % enCH, HOLD de+1,6643 % à+0,2745 %. La dispersion MAIN dépasse les3points autorisés. Les deux dérives H restent sous5 % (MAIN0,9535 %, HOLD4,1405 %) : ce critère passe. L’échec vient de MAINCH négatif et de l’étendue MAIN4,2763points>3, ainsi que HOLDCH0,2745 %<1. L’enum historique inconclusive_drift inclut aussi une dispersion excessive des gains par paire ; il ne signifie pas ici une dérive H hors limite. Aucun gain promu. Les plafonds de compression77B restent échoués indépendamment de cette conclusion.

| Prompt | Gain HC | Gain CH | Dérive H | Étendue des gains | Moyenne C |
|---|---:|---:|---:|---:|---:|
| MAIN | +2.495661 % | -1.780636 % | 0.953535 % | 4.276298 points | 8.845054052 tok/s |
| HOLD | +1.664345 % | +0.274507 % | 4.140457 % | 1.389838 points | 8.647801629 tok/s |

Les étendues ci-dessus décrivent deux paires, sans intervalle de confiance. Les gains ne sont ni additionnés ni combinés avec ceux de packed. Les quatre paires emploient leurs témoinslegacy propres.

| Prompt | Sans prélecture B2 frais | MTP frais | Moyenne unused vs B2 frais | Moyenne unused vs MTP frais |
|---|---:|---:|---:|---:|
| MAIN | 10.222852654 | 9.833896963 | -13.477633 % | -10.055453 % |
| HOLD | 9.486925854 | 8.907714222 | -8.845059 % | -2.917837 % |

Ces références fraîches sont non appariées avec unused, donc comparaisons descriptives seulement. Elles montrent que ce candidat ne démontre pas de dépassement du moteur sans prélecture ni de MTP dans cette campagne.

16 réponseswarm+mesure ont passé golden60textecompletnonvide/hash/128sorties127decode ; huit enfants et lecteurs joints, samplers et pré20/post30 secteur complets. Même acceptation et mêmes largeurs vérifiées H/C dans les deux ordres. Prélectures réellement soumises/complètes/utilisées. Aucun bridge43unusedv3 exécuté ; packed43 n’est pas transféré. Aucun nouvel oraclelogits/états pendant ce benchmark.

Picphysiquemaximal7852595464B, surplus C de chaque paire sous256Mio ; ledger_swapinslogiquesdelta0B pour les huit processus. Swapins etSwapoutsglobaux0B dans HC etCH. Les huit analyses77B restentfailed avecdonnéesvalides etsource_errorsvide, capscompressioninchangés. Cescompteursneprouventpasl’absenced’écrituresswapduprocessus.

Provenance/protocole et compteurs complets dans screen/summary.json, validation/summary.json et validation/hcch-analysis.json. CLI77v1gelé69b919ae, mêmes poids/drapeaux horsPREFETCH_VARIANTlegacy→unused ; codearithmétique/driver/observer inchangés. Aucun code/plan/seuil modifié ni GPU/build/proveur/test lourd lancé par cet analyste.
