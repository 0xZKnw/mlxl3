# 73 OFF — écran réel EarlyShared H-C

État : diagnostic hétérogène, sans confirmation CH ni promotion. MAIN ralentit de 4,533 % et HOLD accélère de 4,280 %. Un seul point mesuré par bras/prompt ; les seuils de répétition exigent au moins +1 % sur les deux prompts, condition non satisfaite. Aucun effet reproductible établi.

| Prompt | H EARLY_SHARED=0 | C EARLY_SHARED=1 | Écart C/H |
|---|---:|---:|---:|
| MAIN | 8.461713086 tok/s | 8.078141880 tok/s | -4.533021 % |
| HOLD | 8.119024926 tok/s | 8.466547406 tok/s | +4.280347 % |

Protocole : OFF confirmé dans les 8 réponses, aucun MTP ni DFlash actif, pipeline=0/lookahead=0, GPU Hot activé. Seule différence H-C : MLXL3_SSD_EARLY_SHARED 0→1. Cache experts 3 Gio, 53 slots, allocateur 64 Mio, contexte 4096, greedy, 128 sorties dont 127 décodées, warmup128 puis mesure128. CLI77v1 SHA 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. Plan et quatre fichiers d’identité rehashés inchangés.

Contrôles : les huit réponses warmup+mesure ont le texte complet non vide, le token_hash et les comptes128/127 exactement égaux au golden60 OFF indépendant. Les tableaux bruts d’IDs token ne sont pas persistés dans ces rapports ; ne pas confondre cette égalité avec une comparaison directe d’IDs ou des logits/états. Les contrôles numériques précédents restent distincts. Les quatre benchmarks sont validated, reaped et reader_joined. Les six étapes pré20/4enfants/post30 sont terminées comme prévu, code0 et reaped ; le sampler est joint. Toutes les frontières sont sur secteur. Pré20 20,042 s ; post30 30,067 s.

Mémoire : pic physique maximal7 153 358 752 B (<12 Go), surplus C MAIN−1 572 840 B et HOLD+2 359 344 B. Aucune hausse des ledger_swapins logiques du moteur depuis ready ni warm→mesure. Swapouts système0 B sur toutes les phases ; Swapins système par bras MAIN H/C32768/393216 B, HOLD H/C65536/0 B, dans la borne stock256 Mio. Ils ne sont pas attribués au modèle.

Les quatre analyses de mémoire aux seuils77B inchangés restent failed : compression courante/pic >512 Mio et lifetime depuis ready >1 Gio. Identités et compteurs valides, source_errors vide. Aucun échec77B requalifié et aucun seuil modifié.

| Bras | Δ lifetime compression warm→mesure | Δ jauge compression warm→mesure | Δ décompressions | Δ ledger_swapins |
|---|---:|---:|---:|---:|
| MAIN H | 228048896 B | +222937088 B | 1382 | 0 B |
| MAIN C | 13156352 B | +2473984 B | 1791 | 0 B |
| HOLD H | 169017344 B | +162365440 B | 1528 | 0 B |
| HOLD C | 4423680 B | -278528 B | 1398 | 0 B |

Ces deltas de phase sont uniquement descriptifs et n’annulent pas les échecs de vie du protocole77B. Les faibles deltas C ne prouvent ni moins de RAM résidente ni moins d’écritures swap attribuées au processus. Aucun nouveau code exécutif, GPU, build, proveur ou test lourd lancé par cette analyse.

Preuves : summary.json, quatre report.json et memory-analysis.json, run-result.json, system-samples.json, run.json et identity.json dans ce dossier. Le bridge43 propre à EarlySharedv3 n’a pas été relancé avant cet écran selon l’amendement72S/73S : pas de promotion possible sans ce contrôle et sans qualification mémoire.
