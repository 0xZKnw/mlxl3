# T18a — pilote DFlash128 w2

**Quatre processus DFlash w2 validés et inspectés, résultat de vitesse mixte.** Ordre ACCA, deux processus indépendants par bras ;8 mesures+8 chauffes, aucune exclusion. [Données/gardes/hashes](pilot128-DF-inspection.json).

| Prompt DF | A tok/s | C tok/s | Decode C/A | Préfill A→C tok/s | Préfill C/A |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAIN | 11.832049 | 11.889242 | +0.483% | 10.1950→10.4105 | +2.113% |
| HOLD | 11.417654 | 11.002653 | -3.635% | 10.3779→10.4236 | +0.440% |
| ALL | 11.621159 | 11.428779 | -1.655% | 10.2829→10.4168 | +1.302% |

Decode = somme des127 transitions mesurées / somme des `stats.decode_seconds` ; préfill = somme des87/82 tokens d’entrée / somme des `stats.prefill_seconds`. ALL combine les deux prompts dans DFlash w2 seulement. Chauffes exclues des agrégats.

| Dérive decode | A1→A2 | C1→C2 |
| --- | ---: | ---: |
| MAIN | +0.831% | +1.312% |
| HOLD | +0.424% | -6.425% |
| ALL | +0.624% | -2.782% |

**C2 HOLD à10,649172 tok/s reste inclus**, après C1 à11,380405 (−6,425 %). Les témoins HOLD évoluent11,393541→11,441870 (+0,424 %). Plages MAIN A11,783314–11,881188/C11,812240–11,967255 ; HOLD A11,393541–11,441870/C10,649172–11,380405. La baisse agrégée ne prouve pas seule une causalité de la file dynamique. La préférence utilisateur accepte les variations modestes sans seuil automatique ; ne pas retirer avant audit d’adaptation ni relancer une longue matrice uniquement pour stabiliser les décimales. Aucun gain général qualifié dans ce bilan.

Les16 réponses complètes ont texte/hash golden128, comptes128/127 et prompts87/82 exacts, DFlash actif mode1/proposals1/draft attendu, MTP inactif et préfixe non réutilisé. Compteurs identiques par prompt entre bras/chauffes/mesures : MAIN[69, 58, 69], HOLD[70, 56, 70] (proposés/acceptés/blocs). Quatre superviseurs et inspections externes complets ; lecteurs/processus joints. Identité541/541 empreintes sans écart, trois binaires conformes, CLI `1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436`. Admission commune10,5Go, ancrew2 Swapouts/Swapins0 ; nouveaux totaux 0/0B. Footprint/pic vie maximaux aux footers 7737694400/7818975448B <12Go ; pas de capteur continu universel.

Ce rapport n’agrège aucun mode ou fenêtre avant redémarrage. Les seuls textes/hash CLI ne prouvent pas128 états numériques ; qualité native distincte conservée. Aucun moteur, test, build, preuve ou GPU lancé par cet audit ; seuls ces documents sont écrits, sans modification d’opti ou du code.
