# T17 — résultats du pilote AR/MTP/DFlash128

31 octobre 2026. **Phase T17 close : qualification numérique validée dans le domaine exercé, performance NON CONCLUANTE ; aucun boost validé.** Le pilote initial conserve un préfill DFlash HOLD plus lent dans les deux processus ON ; cette forte pénalité n’est pas reproduite dans la répétition ciblée sous des conditions différentes. Aucun gain global ni cause du premier écart n’est établi. Le prototype reste local, expérimental et désactivé par défaut (`MLXL3_FLASH_GDN_EPILOGUE=0`), sans promotion. Application installée inchangée ; aucun push, commit, PR ou release pour ce candidat, aucun run distant de cet arbre déclaré vert.

Les premiers tableaux décrivent exclusivement les 12 processus du pilote initial. Une section distincte clôt ensuite le recheck DFlash C→A : premier essai interrompu sous l’ancre w1, puis répétition w2 après observation et admission mémoire renforcée. Aucun chiffre du recheck n’est ajouté aux agrégats du pilote ; les échecs et anciennes conclusions restent conservés.

## Périmètre et identité

Le kernel T17 conserve la norme native F32, la récurrence GDN, les états/tapes et la projection de sortie. Il fusionne l’épilogue gate sigmoid/produit/cast, avec `metal::precise::exp` et deux intermédiaires F32 `volatile` après le correctif `FIX-01`. Aucune LUT ni réserve mémoire supplémentaire n’a été introduite. La capture du composant montre 4→1 dispatchs en B1 et 10→1 en B2 ; elle ne mesure ni un gain modèle ni le GPU idle.

| Élément | Valeur figée |
|---|---|
| Source locale | `7978fceac30d-dirty` ; modifications locales préservées |
| CLI commun A/C | `202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa` |
| Libtest final des qualités | `a751cf592d9e28622d1c062a9f93213f4c55d8a6dcce0fa8b1c035be16c6b862` |
| Shader FIX-01 | `1fa5b6ae214104186021ff2aa43a5e7b083e838fb0a410cf886326aaf8f9ca44` |
| Matériel / système / MLX | Apple M5, 24 Gio ; macOS 27.2 ; MLX 0.32.2 |
| Cible | `Qwen3.8-Flash-Next-exl3-205-65c89531`, EXL3 2,05 bpw |
| Draft DFlash | `Qwen3.8-Flash-Next-DFlash2-a22a1744` |
| Bras A / C | Même CLI ; `MLXL3_FLASH_GDN_EPILOGUE=0` / `1` |
| Autres candidats | T05 forcé OFF ; T04/T18 absents du runtime local, OFF |
| Cache experts | 3 Gio configurés ; 53 slots × 48 couches, payload 3 174 942 528 octets |
| I/O / Hot | 4 lecteurs ; GPU Hot, Hot Wide, Unique et overlap actifs ; lookahead 0 |
| Allocateur / soumissions | Cache allocateur configuré à 64 Mio ; `MLX_MAX_OPS_PER_BUFFER=40`, `MLX_MAX_MB_PER_BUFFER=40` |
| Génération | Contexte 4096, greedy température 0/top-k 1/pénalité 1 ; MCP OFF, réutilisation de préfixe OFF |
| MTP / DFlash | MTP profondeur 1 ; DF mode 1, largeur fixe 1, adaptatif OFF, Q4/RING actifs |

[Identité finale](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/final-identity.json) (183 sources/options) et [extension après Xcode](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/post-xcode-identity.json) (185) identifient les sources, exécutables, MLX/metallib et métadonnées des checkpoints. Ils ne hachent pas l’intégralité des poids. Les rapports de pilote ont tous le même SHA CLI et les mêmes métadonnées runtime hors temps de chargement/mémoire ; leurs environnements reportés diffèrent uniquement par T17. Les variables `MLX_MAX_*` sont vérifiées dans les plans/identités, car le filtre environnement du rapport n’enregistre que `MLXL3_*`.

## Protocole et contrôles

Pour chaque mode AR, MTP et DF : ordre **A–C–C–A**, soit 4 processus, deux par bras. Chaque processus produit successivement chauffe MAIN, chauffe HOLD, mesure MAIN, mesure HOLD (`--repeats 1`, pas de `--compare` ni de prompt personnalisé). Total : **12 processus, 24 chauffes et 24 mesures**, donc 48 réponses complètes. Par mode : 8 chauffes et 8 mesures ; par prompt et bras : seulement **2 mesures**.

MAIN comporte 87 tokens de prompt ; HOLD 82. Chaque réponse contient 128 tokens générés et 127 transitions de décode. Contextes finaux respectifs 215 et 210, aucun token de prompt réutilisé, un round sans outil. Les trois modes sont analysés séparément : leurs gains ne sont ni additionnés ni utilisés comme réplications du même traitement.

Chaque processus dispose d’un rapport durable et d’une inspection externe v2, contrôlés avant le suivant : identité du CLI, flags, réponses non vides, texte/hash golden128 complets, comptes, modes et compteurs actifs, durées finies, empreinte, processus et reader joints. Le statut natif `validated` seul n’effectue pas les comparaisons externes.

| Golden indépendant | Hash 128 tokens | Provenance |
|---|---|---|
| MAIN | `f6b5f974c6db2e7a` | Ancienne référence AR `natural-OFF-c3-MAIN/profile/warmup-summary.json` |
| HOLD | `9d02c1a97e015196` | Ancienne référence AR `natural-OFF-c3-HOLD/profile/warmup-summary.json` |

Les fichiers `docs/measurements/sushi-20261027/gold-AR-{MAIN,HOLD}.json` contiennent les 128 IDs, texte et hash. Le hash FNV64 sur les IDs uint32 little-endian a été recomputé ; les inspections vérifient également l’égalité complète du texte à 128, dans AR/MTP/DF, chauffes comprises.

| Mode/prompt | Propositions / acceptées / blocs, identiques A/C |
|---|---:|
| AR MAIN/HOLD | Compteurs MTP et DF inactifs : `null` |
| MTP MAIN | 72 / 54 / 72 |
| MTP HOLD | 78 / 48 / 78 |
| DF MAIN | 69 / 58 / 69 |
| DF HOLD | 70 / 56 / 70 |

Les champs actifs sont des entiers et les blocs sont effectivement non nuls ; MTP depth 1 et DF mode 1/proposals 1 sont contrôlés. DF MAIN/HOLD vérifie respectivement 138/140 lignes. Le premier inspecteur v1 avait rejeté les compteurs inactifs `null` avant de lancer un autre processus ; ce refus et le rapport natif restent conservés. L’inspection v2 préserve ces valeurs brutes et refuse toujours `null` sur des compteurs actifs.

Campagne sur secteur, avec contrôles quiet/mémoire/swap et un seul modèle GPU à la fois, hors compilation/proveur/Xcode/replay. Garde d’admission 8,5 Go, RSS supervisé ≤10 Gio, footprint natif <12 000 000 000 octets, Swapin cumulé ≤256 Mio, zéro nouveau Swapout depuis l’ancre `T17-w1`. Les 12 processus sont joints, leurs 12 lecteurs également ; les 48 réponses passent l’inspection externe. Les contrôles ne prétendent pas mesurer tous les pics natifs entre deux échantillons.

## Décode : agrégats du pilote

Calcul par bras et prompt : **Σ transitions / Σ secondes de décode**, pas moyenne des TPS individuels. Les lignes ALL agrègent MAIN et HOLD au sein du même mode. Les chauffes sont exclues.

| Mode/prompt | A (tok/s) | C (tok/s) | Variation C/A |
|---|---:|---:|---:|
| AR/MAIN | 10.413438 | 10.461347 | +0.460% |
| AR/HOLD | 10.232231 | 10.207064 | -0.246% |
| AR/ALL | 10.322039 | 10.332642 | +0.103% |
| MTP/MAIN | 11.050266 | 11.138336 | +0.797% |
| MTP/HOLD | 10.178883 | 10.237115 | +0.572% |
| MTP/ALL | 10.596691 | 10.668727 | +0.680% |
| DF/MAIN | 11.735541 | 11.713367 | -0.189% |
| DF/HOLD | 11.188145 | 11.254292 | +0.591% |
| DF/ALL | 11.455307 | 11.479242 | +0.209% |

AR reste proche de l’égalité. MTP présente un petit signal positif sur les deux prompts ; DFlash est mixte. Ces écarts restent des observations de ce pilote, sans qualification de gain global ni promesse de 15 tok/s.

## Préfill et TTFT

Préfill = Σ tokens évalués / Σ secondes de préfill. TTFT = moyenne arithmétique des TTFT par réponse. La ligne ALL mélange les deux prompts uniquement à l’intérieur du mode ; elle ne constitue pas un prompt supplémentaire.

| Mode/prompt | Préfill A → C (tok/s) | Variation | TTFT A → C (s) | Variation |
|---|---:|---:|---:|---:|
| AR/MAIN | 10.150843 → 10.099876 | -0.502% | 8.571209 → 8.614472 | +0.505% |
| AR/HOLD | 10.069126 → 10.235128 | +1.649% | 8.144113 → 8.012062 | -1.621% |
| AR/ALL | 10.111028 → 10.165052 | +0.534% | 8.357661 → 8.313267 | -0.531% |
| MTP/MAIN | 9.863628 → 9.869019 | +0.055% | 8.820897 → 8.816045 | -0.055% |
| MTP/HOLD | 9.856849 → 9.896377 | +0.401% | 8.319610 → 8.286407 | -0.399% |
| MTP/ALL | 9.860338 → 9.882274 | +0.222% | 8.570254 → 8.551226 | -0.222% |
| DF/MAIN | 10.224518 → 10.182861 | -0.407% | 8.509451 → 8.544353 | +0.410% |
| DF/HOLD | 10.178491 → 8.923625 | -12.329% | 8.056665 → 9.189600 | +14.062% |
| DF/ALL | 10.202133 → 9.530330 | -6.585% | 8.283058 → 8.866976 | +7.050% |

DF/HOLD : préfill **−12,329%**, TTFT **+14,062%** ; préfill moyen 8,056204 → 9,189091 s, soit +1,132887 s. Les deux mesures ON valent 8,911257 et 8,936028 tok/s ; OFF 10,080991 et 10,277896. Ce coût est visible et n’est pas masqué par la faible variation positive du décode.

## Plages observées et dérive

Les intervalles suivants sont des **minimums–maximums des deux mesures**, pas des intervalles de confiance. Avec ce petit effectif et l’ordre fixe des prompts, aucune significativité statistique n’est revendiquée.

| Mode/prompt | Décode A / C (tok/s) | Préfill A / C (tok/s) | TTFT A / C (s) |
|---|---|---|---|
| AR/MAIN | 10.186–10.652 / 10.457–10.465 | 9.938–10.373 / 9.980–10.223 | 8.388–8.755 / 8.511–8.718 |
| AR/HOLD | 10.093–10.376 / 10.168–10.246 | 9.763–10.395 / 10.185–10.286 | 7.889–8.399 / 7.973–8.052 |
| MTP/MAIN | 10.953–11.149 / 11.117–11.160 | 9.769–9.960 / 9.844–9.894 | 8.735–8.907 / 8.794–8.838 |
| MTP/HOLD | 10.081–10.279 / 10.208–10.267 | 9.797–9.917 / 9.872–9.921 | 8.269–8.370 / 8.266–8.307 |
| DF/MAIN | 11.604–11.870 / 11.600–11.829 | 10.148–10.302 / 10.106–10.260 | 8.445–8.573 / 8.480–8.609 |
| DF/HOLD | 11.095–11.283 / 11.251–11.257 | 10.081–10.278 / 8.911–8.936 | 7.979–8.135 / 9.177–9.202 |

Dérive du décode entre le premier et le dernier OFF :

| Mode | MAIN | HOLD |
|---|---:|---:|
| AR | -4.375% | -2.730% |
| MTP | +1.788% | +1.963% |
| DF | +2.295% | +1.695% |

La dérive AR atteint −4,375%, supérieure aux petits écarts moyens du pilote. L’acceptation de la variance par l’utilisateur est conservée : la variance seule ne motive pas un retrait automatique. Elle limite l’attribution du petit signal observé, sans prouver à elle seule que ce signal est nul.

## Mémoire, I/O et audit de l’adaptation

| Observation sur les 48 réponses | Valeur |
|---|---:|
| process_footprint_bytes | 7124965304–7788780120 octets |
| process_lifetime_peak_bytes | 7124965304–7881677376 octets |
| mlx_active_bytes | 6579356288–7294084064 octets |
| mlx_cache_bytes | 48062402–67609858 octets |
| mlx_peak_bytes | 6592220576–7439735688 octets |
| process_compressed_logical_bytes | 1126154240–2627796992 octets |

Le pic natif de vie maximal est **7881677376 octets**, distinct du budget experts de 3 Gio et de la mémoire active MLX. Les 48 compteurs process-ledger Swapin sont nuls et les inspections rapportent zéro nouveau Swapout. Cela ne prouve pas l’absence de compression, de lecture du modèle ou d’attente CPU/GPU.

L’audit avant décision ne trouve aucun défaut concret expliquant la pénalité DF/HOLD. Il confirme :

- Le ralentissement se produit sur la quatrième réponse des deux processus ON ; les chauffes HOLD restent normales.
- Demandes logiques du décode DF, compteurs et lignes identiques A/C : HOLD mesuré 25 808 888 160 octets. Aucun compteur I/O préfill ou hits/misses n’est disponible.
- Temps moyen de vérification DF/HOLD 10,525191 → 10,451495 s ; draft/commit proches, chargement draft de l’ordre de 35–37 µs aux mesures. Ces horloges décode n’expliquent pas le préfill.
- Mémoire active MLX DF/HOLD strictement identique en moyenne, 7 144 222 848 octets ; footprint ON supérieur de seulement 7 258 112 octets.
- Décompressions corrélées mais non attribuables au préfill : ON 50 577/39 355 pages depuis la réponse précédente, OFF 28 709/27 445. Un MAIN ON normal compte aussi 47 746 pages.
- Le cache C++ de factories est borné à 128 clés, mais son occupation/éviction n’est pas mesurée ; T17 ajoute deux clés dans cette campagne B1/B2. Les positions ring sont des données, pas une nouvelle clé par token. Cache factory et cache des pipelines Metal sont distincts.
- Aucun readback, eval/fence ou concaténation F32 ajouté au raccord ; la sortie F16 est déjà allouée dans le stock. Un coût des deux `volatile` est plausible, sans spill, scratch ou temps GPU isolé mesuré. Le diagnostic à trois sorties ne qualifie pas leur suppression dans le vrai kernel mono-sortie.

**Cause inconnue.** Le pilote ne décompose pas I/O préfill critique/masquée, GPU utile/idle, CPU actif/synchronisation, cache factory, allocations ou codegen. Aucune correction de production, activation par défaut ou conclusion d’inefficacité de l’idée n’est établie par cet audit.

## Recheck DFlash C→A : phase distincte, clôturée

Le recheck a été préenregistré pour examiner la reproductibilité du coût DF/HOLD à la quatrième réponse, sans changer le kernel ni rechercher une confirmation de gain. Les deux phases ci-dessous sont distinctes du pilote et l’une de l’autre.

**Premier essai w1 interrompu.** La tentative de chargement C a été arrêtée après 2,096 s de supervision, sur +1 048 576 octets de nouveau Swapout cumulatif par rapport à `T17-w1` ; Swapin cumulé +12 500 992 octets. Disponibilité estimée avant PID : 11 951 964 160 octets. Aucune réponse, chauffe ou mesure terminée ; bras A non lancé. Le rapport natif demeure `incomplete`, sans événement `ready` enregistré : ne pas transformer cette durée en temps de chargement réussi ou en benchmark. Le processus arrêté est joint.

Preuves conservées : [rapport natif partiel à zéro réponse](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck/t17-df-recheck128-1-C/report.json) ; [résultat supervisé de l’interruption](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck/t17-df-recheck128-1-C-plan-result.json).

**Nouvelle condition explicite et ancre w2.** L’ancienne ancre w1 n’a pas été réécrite. Après préenregistrement `T17-DF-RECHECK-02`, quatre instantanés ont couvert 60,27 s sans modèle, compilation, preuve ou Xcode, sur secteur et en conditions quiet. Swapout est resté stable à 140 865 945 600 octets ; la disponibilité minimale était de 12 543 442 944 octets. L’admission est passée explicitement à **12 000 000 000 octets avant chaque processus**, contre 8,5 Go dans le pilote. La nouvelle ancre identifiée w2 fixe ce Swapout et un Swapin initial de 53 807 169 536 octets. Ce changement de conditions interdit une fusion silencieuse des campagnes.

Voir [observation et admission renforcée](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-admission.json) et [ancre w2 explicite](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/gpu-anchor-w2.json). L’[identité w2](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-identity.json) fige 190 empreintes de sources/binaires/dépendances ; leur contrôle final les retrouve inchangées, avec le même CLI `202f96…`, libtest `a751cf…` et shader FIX-01.

**Recheck w2 terminé.** Deux processus neufs C→A, mêmes paramètres cible/draft/cache 3 Gio/53 slots/40 ops/40 MB, mêmes MAIN/HOLD et quatre réponses par processus : **4 chauffes + 4 mesures**, 128 tokens/127 transitions. Chaque réponse passe les contrôles externes texte/hash128, modes, compteurs et empreinte ; les deux processus et lecteurs sont joints. Depuis w2 : **0 octet de nouveau Swapout et 0 octet de Swapin**. Une seule mesure par prompt et bras : aucune plage ni précision statistique ne peut être estimée dans ce recheck.

Les débits utilisent les mêmes définitions Σ tokens/Σ durées ; TTFT est la moyenne des réponses. Les lignes ci-dessous ne remplacent pas celles du pilote :

| DF w2/prompt | Décode A → C (tok/s) | Variation C/A | Préfill A → C (tok/s) | Variation C/A | TTFT A → C (s) |
|---|---:|---:|---:|---:|---:|
| MAIN | 11.680283 → 11.718396 | +0.3263% | 10.184694 → 10.066507 | -1.1604% | 8.542722 → 8.643067 |
| HOLD | 11.154298 → 10.950704 | -1.8252% | 10.162639 → 10.143281 | -0.1905% | 8.069229 → 8.084711 |
| ALL | 11.411233 → 11.321551 | -0.7859% | 10.173981 → 10.103613 | -0.6917% | 8.305975 → 8.363889 |

HOLD préfill : **C 10,143281 contre A 10,162639 tok/s, −0,1905%**. La pénalité initiale de −12,33% n’est donc **pas reproduite sous cette admission renforcée, l’ordre inverse et une nouvelle ancre**. Ce résultat ne démontre ni que la RAM causait le premier écart ni que le flag était innocent ; ordre, état résiduel et conditions diffèrent. La cause initiale reste inconnue.

Décode agrégé w2 : **C 11,321551 contre A 11,411233 tok/s, −0,7859%** ; aucun boost confirmé. MAIN et HOLD restent séparés dans le tableau, sans addition de leurs variations ni mélange avec AR/MTP ou le pilote w1. Résultats et gardes : [inspection w2](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-inspection.json) ; [manifeste w2](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-manifest.json).

**Décision de clôture.** Numérique qualifiée dans le domaine testé, performance non concluante ; prototype local conservé OFF, sans promotion. L’audit n’a pas établi d’autre défaut concret dans le raccord, les dtypes/layouts/fallbacks, ownership/copies/donation/synchronisations ou cache de factories. L’écart initial réel de 2 ULP sur la probabilité F32, localisé au chemin exponentielle, reste corrigé par `metal::precise::exp` ; ce correctif et ses preuves ne sont pas retirés. Une éventuelle variante réelle mono-sortie sans `volatile` reste non testée et exigerait son propre préenregistrement. Tous les processus GPU/compilation/preuve de cette phase sont arrêtés.

## Qualification et pièces conservées

Les qualités numériques ont précédé la vitesse. Les preuves et échecs restent dans le [bilan de vérification](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/verification.md) : 13 sélections physiques finales uniques, 12 répétitions explicites après Xcode, oracles exhaustifs dans leur domaine fini, intégration GDN/états/tapes, modèle et chemins MTP/DFlash ; Kani géométrie CPU bornée, MLX/Metal non formellement prouvés. La baisse de dispatch du composant ne fournit pas un plafond de débit modèle.

Le code local reste OFF par défaut après clôture non concluante des performances. CI locale inspectée et checks passés selon ce bilan ; code non publié, aucun run GitHub de cet arbre déclaré vert. Application installée inchangée. Les résultats du pilote et du recheck restent séparés ; aucune moyenne additionnant AR/MTP/DF ni aucun mélange de leurs ancres et conditions n’est présenté ici.

- [Inspection agrégée du pilote](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/pilot128-inspection.json) et [manifest des 18 plans bridge/pilote](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/prepared/manifest.json).
- [Audit données](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/audit-perf-data.md) et [48 statistiques/provenance/hashes](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/audit-perf-data.json).
- [Audit cache et raccord](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/audit-perf-source.md) ; [audit numérique et volatile](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/audit-perf-numeric.md).
- [Qualités modèle finales](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/final-model-inspection.json) ; [inspection réelle des dispatchs](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/xcode-dispatch-inspection.json).
- [Premier recheck w1 interrompu](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-manifest.json) et [recheck w2 clos, distinct du pilote](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-inspection.json) ; [admission w2](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-admission.json) ; [identité 190 empreintes](/Users/justin/.codex/worktrees/slotstream-flash/mix-stq1_0/docs/measurements/issue59-t17-20261029/df-recheck-w2-identity.json).

Les 12 rapports bruts et inspections v2 restent dans `prepared/runs/t17-pilot128-{AR,MTP,DF}-{1-A,2-C,3-C,4-A}/`. Leurs empreintes sont listées dans l’audit données. Aucun build, test, preuve ou lancement GPU effectué pour rédiger ce document.
