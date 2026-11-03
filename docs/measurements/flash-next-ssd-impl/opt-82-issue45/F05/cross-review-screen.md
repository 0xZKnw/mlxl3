# 82F05 — contre-audit de l’écran H-C

Les huit réponses warmup/mesure sont exactes au texte **et** au hash gold60, avec 128/127, DFlash2 fixe B2 actif, MTP inactif et compteurs MTP JSON null. Les acceptations, largeurs, histogrammes et demandes logiques SSD sont identiques H/C par phase.

| Prompt | H tok/s | C tok/s | C/H |
|---|---:|---:|---:|
| MAIN | 10.180175 | 9.807010 | −3,6656 % |
| HOLD | 9.652619 | 8.869685 | −8,1111 % |

Aucun signal positif. Retrait conservateur, sans confirmation C-H, OFF/MTP ou replay de vitesse. Ces paires uniques ne démontrent pas une régression causale universelle, particulièrement avec la dérive warmup→mesure de HOLD-C.

Variations warmup→mesure :

- MAIN-0-H : +4,9314 % ; décompressions propres +30867, compression logique +4636672 octets.
- MAIN-1-C : −0,6502 % ; décompressions propres +31628, compression logique +70205440 octets.
- HOLD-0-H : +0,7037 % ; décompressions propres +30677, compression logique +150700032 octets.
- HOLD-1-C : −5,9518 % ; décompressions propres +30918, compression logique +166494208 octets.

Six phases et leurs enfants/lecteurs clos et joints, secteur AC ; sampler joint, 131 observations sans erreur. Swap-out système zéro ; swap-in système **131 072 octets sur MAIN-C**, origine inconnue. Douze snapshots : pic processus **7.847 Go**, compression logique courante maximale **3.948 Go**, ledger swap-in propre zéro. Le ledger ne prouve pas l’absence de swap-out privé et les caps 77B restent FAILED. Le bridge précédent a, séparément, 130 Mio de swap-out global et n’est pas qualifié zéro global.

Intervalles de profil inclusifs côté hôte, sans somme avec decode/block et sans attribution GPU exclusive. Aucun trafic physique SSD mesuré. Les 86 sources de l’archive concordent avec le gel ; les binaires/pilotes mesurés restent figés. Le retrait a été constaté avant K1 dans `withdrawal-static-pre-K1.json` : les changements K1 futurs sont distincts. Aucun test, GPU, build, preuveur ou code modifié par cet audit.
