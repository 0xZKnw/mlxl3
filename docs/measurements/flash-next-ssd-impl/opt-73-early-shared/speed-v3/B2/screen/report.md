# 73 B2 — EarlyShared H-C

Écran diagnostique hétérogène : MAIN+6,026925 %, HOLD−4,253012 %. La condition CH≥1 % sur les deux prompts échoue. Un point par bras/prompt ne permet pas de confirmer un gain ni d’isoler le bruit. Aucun gain promu.

| Prompt | H EarlyShared0 | C EarlyShared1 | Écart | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 9.542746168 tok/s | 10.117880309 tok/s | +6.026925 % | 12.413202/11.657539 s |
| HOLD | 9.068805258 tok/s | 8.683107892 tok/s | -4.253012 % | 12.949441/13.497944 s |

Le déplacement du temps de vérification suit celui du débit sur les deux prompts. Les routes/demandes observables n’ont pas changé : MAIN58acceptés/69proposés/69blocs,138lignes vérifiées ; HOLD56/70/70,140lignes. Même demande expertsH/C :23930630100B MAIN et25474420944B HOLD. Aucun prefetch soumis dans les profils, conformément au pipeline0/lookahead0. Les métriques de vérification sont inclusives : ne pas les additionner au decode_seconds ou à une sous-phase GPU.

Protocole : DFlash2fixeB2 confirmé, WIDTH1/ADAPTIVE0, MTPinactif, GPUHot etwideactivés, cache3Gio53slots, allocateur64Mio, contexte4096, 128sorties127decode, warmup128+mesure128. Seul EARLY_SHARED0→1 diffère entre H/C. Les quatre fichiers d’identité CLI/observer/driver/plan sont rehashés inchangés, CLI77v1 SHA 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. Les165sources de référence sont relues : seule la CI a changé, comme prévu pour le commit89174fcc. Les sources arithmétiques restent identiques ; un changement du HEAD Git ne transforme pas ce binaire gelé en build du nouveau commit.

Qualité : huit réponses warmup+mesure exactement égales au golden60 OFF pour texte complet non vide, token_hash et comptes128/127. Pas de tableaux bruts d’IDs persistés ni de nouvelle vérification logits/états dans cet écran. Quatre enfants validés/reaped/reader_joined, six étapes pré20/post30 complètes, samplerjoint, secteur à toutes les frontières. Les anciens contrôles numériques72/73 restent distincts, le bridge43 spécifique73v3 non relancé ne peut être annoncé validé par transfert.

Mémoire : picphysique7850121528B, surplusC MAIN−1310768B/HOLD−16973920B ; ledger_swapins logiques du moteur sans hausse depuis ready ou warm→mesure. Swapouts système0B. Swapins système262144Bpré20,65536BMAINH et0autresphases, sousstock256Mio ; aucune attribution au modèle. Les quatre analyses77B échouent aux capscompression inchangés, identités et compteurs valides/source_errorsvide ; aucune requalification.

| Bras | Δlifetime compression warm→mesure | Δjauge compression | Δdécompressions | Δledger_swapins |
|---|---:|---:|---:|---:|
| MAIN H | 19382272 B | +5668864 B | 31184 | 0 B |
| MAIN C | 8503296 B | +3735552 B | 30808 | 0 B |
| HOLD H | 173080576 B | +165740544 B | 30588 | 0 B |
| HOLD C | 15613952 B | +4669440 B | 30768 | 0 B |

Deltas warm→mesure descriptifs, pas de seuil futur appliqué rétrospectivement. L’absence de ledger_swapins ne prouve pas l’absence d’écritures swap du processus. Aucun GPU/build/proveur/test lourd lancé par l’analyste. Aucun changement de code, plans ou seuils.

Preuves locales : summary.json, quatre report.json et memory-analysis.json, run-result.json, system-samples.json, run.json, identity.json.
