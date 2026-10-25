# 72W-R — screen

Nouvelle comparaison appariée : H=B2 sans prélecture, C=B2weighted. Les anciens gains contre legacy ne remplacent pas ces mesures. La consigne initiale72W-KEEP est révoquée : l’utilisateur retire weighted après le vraiA/B contre sansprélecture. CHannulé/nonlancé, pertesconservées ; aucunseuilvariation n’est utilisé pour justifier ce nouveau retrait.

| Prompt | H sans prélecture | C weighted | Écart C/H | Vérification H/C |
|---|---:|---:|---:|---:|
| MAIN | 11.301206241 | 11.125226993 | -1.557172% | 10.494344/10.657954s |
| HOLD | 10.716598138 | 8.775646048 | -18.111644% | 10.984327/13.409279s |

H/C ne diffèrent que par pipeline0→1, lookahead0→2 et variantlegacy→weighted. DFlashfixeB2/WIDTH1/ADAPTIVE0, draftQ4/ring, GPUHot/Wide, 3Gio53slots/allocateur64Mio/ctx4096/greedy, warmup128+mesure128sorties127decode. Mêmes acceptations et largeurs vérifiées sur les deux bras. Les oracles sont les huit réponsesgolden60 textecompletnonvide/hash/comptes, pas une nouvelle comparaison directe des logits/états ou arraysbrutsIDs.

| Prompt | Demande H | Demande C | Prélecture C | Demande+prélecture C |
|---|---:|---:|---:|---:|
| MAIN | 23930630100B | 27061892208B | 7708970124B | 34770862332B |
| HOLD | 25474420944B | 29235929112B | 7868715660B | 37104644772B |

Ces volumes sont des compteurs moteur d’enregistrements, pas des octets physiquesNVMe ni une mesure de saturationSSD. Les temps de vérification sont inclusifs et ne s’additionnent pas à leurstotaldecode. Les profilsH sans prélecture sont nuls, C soumet/termine/utilise réellement des prélectures.

Huitréponses exactes, quatreenfants/lecteurs etsamplerjoints ; pré20/post30secteur. Picphysique7851661600B ; ledgerpropre sanshausse=True. Swapoutsglobauxparétape[0, 0, 0, 0, 0, 0], Swapins[0, 0, 0, 65536, 0, 0]. Les4analyses77Bfailed/source_errorsvide sontconservées, aucune requalification des capscompression. Deltasready→fin etwarm→mesure danssummary.json. Ces compteursneprouventpasl’absenced’écrituresswapprivées.

Plans/identité/commandesenv/sourcehashes conservés ; CLI77v1SHA69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c. Les changementsdeHEAD/publication ne sontpasunrebuildduCLI. AucunGPU/build/proveur/testlourd lancé par cetanalyste. Aucuneaddition des gains et aucune conclusion universelle15tok/s.

État final des sources : les6fichiersRustprototypes etlaCI ont été restaurés auHEAD89174fcc aprèsarchive938b2e9c ; le CLIgelé69b919 réellementmesuré estinchangé. Les données passées restentvalablespourceprototype archivé, paspourlecheckoutrestauré. Publication documentaire uniquement, aucune autre mesure.
