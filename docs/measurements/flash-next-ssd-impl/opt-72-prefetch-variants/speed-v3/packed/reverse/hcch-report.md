# Packed72 HCCH : résultat diagnostique final

Packed accélère le chemin de prélecture legacy sur les quatre paires. La campagne reste non concluante pour un gain : le témoin HOLD dérive, et les plafonds de compression77B restent échoués. Comparer à legacy ne démontre pas un gain face au moteur sans prélecture.

|Prompt|Ordre|LegacyH(tok/s)|PackedC(tok/s)|Écart|
|---|---|---:|---:|---:|
|MAIN|HC|9.122354|10.305966|+12.9748%|
|MAIN|CH|8.714262|9.809459|+12.5679%|
|HOLD|HC|8.695171|9.626069|+10.7059%|
|HOLD|CH|7.834093|9.523039|+21.5589%|

## Stabilité et référence sans prélecture

- MAIN : dériveH 4.6830% (limite5%), étendue des gains 0.4070points (limite3). Packed moyen 10.057713tok/s, contre 10.222853 pour le témoin frais sans prélecture : -1.6154% descriptif. Les pointsPacked isolés donnent +0.8130% et -4.0438% face à ce même témoin antérieur ; comparaison non appariée.
- HOLD : dériveH 10.9914% (limite5%), étendue des gains 10.8530points (limite3). Packed moyen 9.574554tok/s, contre 9.486926 pour le témoin frais sans prélecture : +0.9237% descriptif. Les pointsPacked isolés donnent +1.4667% et +0.3807% face à ce même témoin antérieur ; comparaison non appariée.

## Qualité et mémoire

Les16réponses (chauffe et mesure, huitprocessus) correspondent aux golden60MAIN/HOLD : texte complet, hash,128sorties et127tokens décodés. Les tableaux d’IDs bruts ne figurent pas dans ces rapports. Les acceptations, propositions et nombres de blocs mesurés sont identiques entrebras de chaque prompt. B2/WIDTH1/ADAPTIVE0 et binaire77v1SHA69b919ae inchangés.

Pré20/post30 sur secteur réussis dans les deuxphases ; tous enfants, lecteurs et samplers joints. ScreenMAINH observe458752B de nouveauxSwapouts système, admis par le stock256Mio ; reverse observe0B. La hausse n’est ni effacée ni attribuée au modèle.

Pic physique maximal 7853988128B ; pic de compression logique maximal 3982131200B ; plus grand delta de compressed_lifetime depuisready 3030024192B. Les huit analyses échouent sur les plafonds de compression77B inchangés ; données/identités/cumuls valides, source_errorsvide. Delta logicalledger_swapins depuisready nul dans tous les processus ; cela ne mesure pas les écritures swap du processus.

Les quatre nouvelles analysesCH restent enregistrées failed dans leurs memory-analysis.json. Aucun seuil relevé, aucune réutilisation des seuils78, aucune requalification77B. Surplus physiquesCvsH : +13516872/+3473408B MAIN, −16498688/+3014680B HOLD, tous sous256Mio.

Les durées inclusives du draft/vérification/commit et les compteursAPI/prélecture parbras sont conservés dans hcch-analysis.json. Les octetsAPI ne sont pas des lectures physiquesNVMe ; les temps de vérification ne sont pas duGPUexclusif.

Le qualificateur60 historique exige zéroSwapout global ; ses entrées n’ont pas été falsifiées pour obtenir un verdictvert. Les mêmes seuils de ratios/dérive/étendue ont été calculés avec les donnéesréelles. Statut final : inconclusive_drift et mémoire nonqualifiée, donc aucune promotion.

Aucun GPU, build, prouveur ou test lourd lancé pour cette analyse. Une première commande éphémère a échoué au parsingPython avant touteinstruction ; son diagnostic est conservé dans analysis-attempt-v1-error.txt et la reprise a analysé les mêmes données.
