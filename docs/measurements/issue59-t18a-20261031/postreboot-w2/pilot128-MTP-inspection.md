# T18a — pilote MTP128 w2

**Quatre processus MTP w2 validés et inspectés ; DFlash reste hors de ce bilan.** Ordre ACCA, deux processus indépendants par bras,8 mesures+8 chauffes. Aucun résultat MTP w1 ajouté, aucun échantillon exclu. [Données, gardes et hashes](pilot128-MTP-inspection.json).

| Prompt MTP | A tok/s | C tok/s | Decode C/A | Préfill A→C tok/s | Préfill C/A |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAIN | 10.802165 | 11.380673 | +5.355% | 9.8428→10.0309 | +1.911% |
| HOLD | 10.383112 | 10.458170 | +0.723% | 9.6482→10.1378 | +5.075% |
| ALL | 10.588494 | 10.899938 | +2.941% | 9.7474→10.0825 | +3.438% |

Decode = somme des127 transitions mesurées / somme des `stats.decode_seconds` ; préfill = somme des87/82 tokens d’entrée / somme des `stats.prefill_seconds`. ALL combine MAIN/HOLD dans MTP w2 seulement. Chauffes exclues des agrégats.

| Dérive decode | A1→A2 | C1→C2 |
| --- | ---: | ---: |
| MAIN | -8.048% | -2.252% |
| HOLD | -2.696% | +1.151% |
| ALL | -5.356% | -0.493% |

**Dernier témoin conservé : MAIN11,274879→10,367494 tok/s (−8,048 %), HOLD10,526956→10,243146 (−2,696 %).** Les plages mesurées MAIN sont A10,367494–11,274879/C11,252549–11,511749 ; HOLD A10,243146–10,526956/C10,398645–10,518382. Le pilote donne un signal pratique positif, mais son amplitude dépend de ce mouvement des contrôles. Selon la préférence utilisateur, cette dérive ne déclenche pas à elle seule un rejet ou une nouvelle longue campagne pour stabiliser les décimales ; elle reste visible. Pas de décision d’intégration globale dans ce rapport.

Les16 réponses complètes ont texte/hash golden128 et comptes128/127 vérifiés, prompts87/82, mode MTP actif/profondeur1, DFlash inactif, cache de préfixes non réutilisé, flags corrects et compteurs identiques par prompt entre bras/chauffes/mesures : MAIN[72, 54, 72], HOLD[78, 48, 78] (proposés/acceptés/blocs). Quatre inspections externes et superviseurs complets, lecteurs/processus joints. Identité actuelle541/541 SHA sans écart et trois binaires conformes, CLI `1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436`. Admission10,5Go commune ; nouvelle ancrew2 Swapouts/Swapins0, nouveaux totaux 0/0B. Footprint/pic vie maximaux aux footers 7770659344/7863753160B <12Go, pas capteur continu universel.

Les oracles numériques natifs restent une preuve distincte des seuls textes/hash CLI. Aucune somme avec AR ou DFlash, aucun plafond théorique transformé en gain réel ; DFlash encore à examiner. Aucun moteur, test, build, proveur ou GPU lancé par cet audit ; seuls ces documents sont écrits, sans changement d’opti, manifeste ou code.
