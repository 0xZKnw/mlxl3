# Témoins après72S/73S

Les quatre contrôles d’encadrement après sont terminés, sans nouvelle optimisation. Huitréponseswarm+mesure exactes golden60 (texte complet nonvide/hash/128sorties127decode), modesB2fixe/MTP1 confirmés, prélecturepipeline0/lookahead0/EarlyShared0. Les quatre enfants/lecteurs, sampler et pré20/post30 sont joints, sursecteur.

| Mode | Prompt | Avant tok/s | Après tok/s | Variation descriptive |
|---|---|---:|---:|---:|
| MTP | HOLD | 8.907714222 | 9.903072863 | +11.174120% |
| B2 | HOLD | 9.486925854 | 10.269771465 | +8.251837% |
| MTP | MAIN | 9.833896963 | 10.799178310 | +9.815858% |
| B2 | MAIN | 10.222852654 | 11.021135664 | +7.808809% |

Chaque référence n’a qu’un point par mode/prompt. Leur variation indique que les anciennes références ne sont pas appariées aux candidats ; elle ne valide pas un gain. La nouvelle72W-R H=no-prefetch/C=weighted est séparée, après le bridgeweighted43, et les anciens résultatsvslegacy ne la remplacent pas. Weighted reste retenu selon72W-KEEP ; les variations des références ne rejettent pas sonintégration demandée.

Picphysiquemaximal7888149008B. Ledger_swapinspropre sanshausse depuisready ; Swapoutsglobauxparphase{'pre20': 0, 'MTP-HOLD-after': 0, 'B2-HOLD-after': 0, 'MTP-MAIN-after': 0, 'B2-MAIN-after': 0, 'post30': 0}, Swapins{'pre20': 0, 'MTP-HOLD-after': 0, 'B2-HOLD-after': 0, 'MTP-MAIN-after': 0, 'B2-MAIN-after': 0, 'post30': 0}, sous256Mio/phase. Les quatreanalyses77B sontfailed avecdonnéesvalides/source_errorsvide, capscompressioninchangés ; aucunerequalification. Lesdeltaswarm→mesure sontdanssummary.json. Cescompteursneprouventpasl’absenced’écrituresswapduprocessus.

Provenance : CLI77v1SHA69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c, mêmesmodèles/options/cache3Gio53slots/alloc64Mio/ctx4096/greedy/128tokens, warmup128+mesure128. Ordre inversédescontrôlesavant :MTPHOLD/B2HOLD/MTPMAIN/B2MAIN. Commandes/env exacts dansrun.json, hashplan vérifié. Aucuneinférence/build/proveur/testlourd lancé par cet analyste.
