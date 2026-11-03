# T18a — pilote AR128, inspection des quatre processus

**AR terminé et données validées, autres modes en attente dans ce rapport.** Ordre ACCA, deux processus indépendants par bras ; chacun chauffe MAIN/HOLD puis mesure une réponse128 par prompt :8 mesures+8 chauffes. Aucun échantillon exclu. [Données et hashes](pilot128-AR-inspection.json).

| Prompt AR | A tok/s | C tok/s | Decode C/A | Préfill A→C tok/s | Préfill C/A |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAIN | 10.440963 | 10.771817 | +3.169% | 10.2839→10.4917 | +2.021% |
| HOLD | 10.429647 | 10.549104 | +1.145% | 10.3392→10.5257 | +1.804% |
| ALL | 10.435302 | 10.659297 | +2.147% | 10.3106→10.5082 | +1.916% |

Débit decode = somme des127 transitions par réponse mesurée / somme des `stats.decode_seconds`. Préfill = somme des87/82 tokens d’entrée / somme des `stats.prefill_seconds`. ALL agrège les deux prompts **dans AR seulement**, sans moyenne arbitraire des TPS. Chauffes exclues des agrégats.

| Dérive decode | A1→A2 | C1→C2 |
| --- | ---: | ---: |
| MAIN | -0.253% | -0.838% |
| HOLD | -3.869% | -1.092% |
| ALL | -2.079% | -0.967% |

Identité r3 :531/531 SHA sources/dépendances relus, zéro écart ; CLI `1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436` et autres binaires conformes. Les16 réponses complètes (chauffes comprises) ont texte/hash identiques aux goldens externes128, comptes128/127 et87/82, modes AR/counters inactifs, préfixe non réutilisé, bon flag0/1 et T04/T17/T05=0. Quatre inspections externes et superviseurs complets, lecteurs/processus joints, zéro nouveau Swapout sous T18A-w1 ; Swapin cumulé0B. Footprint/pic vie aux footers maximaux 7158421480/7158421480B, sous12Go. Pas de capteur natif continu universel revendiqué.

Ces petites différences sont descriptives, avec seulement deux processus par bras ; leur dérive et les modes MTP/DFlash restent nécessaires au jugement. **Aucun gain global établi ou intégration décidée ici.** Le texte/hash CLI seul ne prouve pas128 états numériques ; la qualité native distincte garde sa portée. Aucun moteur, build, test, preuve ou GPU lancé par cet audit ; aucun code/manifeste/opti modifié.
