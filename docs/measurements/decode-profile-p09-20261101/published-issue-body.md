## Résultat principal

Le profilage P09 prévu est terminé : CLI natif **AR, MTP et DFlash2**, prompts courts et contexte de 512 tokens, puis captures corrélées CPU/GPU/POSIX. **Les ~34 ms d’attente des routes ne sont pas 34 ms de GPU inutilisé : le GPU travaille pendant presque toute cette attente.** Les postes à examiner sont surtout l’encodage/soumission MLX exposé et les attentes Cold non masquées.

Cette issue consolide le rapport local `docs/decode-profile-p09-20261101.md`. Les étapes interrompues et anciennes conclusions restent conservées dans `opti.md`. Les prototypes décrits à la fin ne constituent pas une optimisation publiée dans le moteur ou l’application.

## 1. Référence, matériel et protocole

- Référence intégrée : [`d5a776e53425bb4d0ba61c0774c5399ae7987c06`](https://github.com/0xZKnw/mlxl3/commit/d5a776e53425bb4d0ba61c0774c5399ae7987c06), après intégration des PR #61–#64.
- Apple M5 standard, 24 Go ; secteur ; un seul modèle GPU à la fois, sans compilation/proveur/autre inférence concurrente.
- Qwen3.8 Flash-Next EXL3 2.05 bpw, checkpoint `65c89531` ; draft DFlash2 `a22a1744`.
- Experts : **3 Gio**, 53 slots × 48 couches, quatre lecteurs statiques ; GPU Hot/Wide, Unique Cohort, overlap et Cold U actifs.
- MAIN : 87 tokens d’entrée ; HOLD : 82. 128 sorties, dont 127 comptées en decode.
- Ordre CLI : AR–MTP–DF–DF–MTP–AR ; une chauffe par prompt/processus, puis deux mesures par prompt/processus. Soit quatre mesures par mode/prompt.
- DFlash **B2 = une proposition**, largeur fixe, adaptation désactivée. Ces résultats ne décrivent pas toutes les largeurs DFlash.
- LONG : 512 tokens réels, 64 sorties/63 decode ; une chauffe + deux mesures, un processus par mode AR puis MTP puis DF. Corpus différent : l’écart avec MAIN/HOLD n’isole pas un effet pur de longueur.
- Cache fichier macOS non purgé. Les lectures POSIX/logiques ne sont pas assimilées à des octets NAND.

Le CLI de référence est inchangé. Le libtest ajoute sous `cfg(test)` l’export des lectures Cold asynchrones : l’ancien analyseur ne comptait que le fallback synchrone. Régression du diagnostic reproduite sur le même lot : zéro lecture comptée auparavant, dix après correction.

SHA-256 :
- CLI référence : `5fe19abf4bacdc6890c7f42211164a026afedb1e4909130c5a24f0e069fe08a3`.
- Libtest de profilage : `7145dbf164333ca0630a027c9b1acbbf3817643dc3632e39e9ef9d6a6a1523dc`.
- Identités de sources : `identity.json`, `identity-s32.json`, `identity-long.json`, avec empreintes des sources et dépendances du dépôt. Les références initiales restent conservées.

<details>
<summary>Options déterminantes</summary>

```text
MLXL3_ALLOCATOR_CACHE_MIB=64
MLXL3_SSD_CACHE_GIB=3
MLXL3_SSD_READERS=4
MLXL3_SSD_GPU_HOT=1
MLXL3_SSD_GPU_HOT_WIDE=1
MLXL3_SSD_UNIQUE_COHORT=1
MLXL3_FLASH_MULTI_TOKEN=1
MLXL3_SSD_OVERLAP=1
MLXL3_SSD_COLD_READY=1
MLXL3_SSD_LOOKAHEAD_DEPTH=0
MLXL3_SSD_CACHE_POLICY=auto
MLXL3_SSD_SHARED_EXL3=0
MLXL3_SSD_DYNAMIC_READERS=0
MLXL3_FLASH_PREFILL_HEADS=1
MLXL3_FLASH_DRAFT_WIDTH=1
MLXL3_FLASH_DRAFT_ADAPTIVE=0
MLXL3_FLASH_DRAFT_Q4=1
MLXL3_FLASH_DRAFT_RING=1
MLXL3_FLASH_SWIGLU_DOWN=0
MLXL3_FLASH_GDN_EPILOGUE=0
MLXL3_SSD_WEIGHTED_EPILOGUE=0
MLXL3_FLASH_DENSE_EPILOGUE=0
MLXL3_FLASH_PREPARED_CONV=0
MLX_MAX_OPS_PER_BUFFER=40
MLX_MAX_MB_PER_BUFFER=40
```

Le flag PREFILL_HEADS n’évitait pas encore les projections intermédiaires du prefill MTP dans cette référence.
</details>

## 2. Débits et prefill CLI non instrumentés

| Mode | Prompt | Decode médian, tok/s | Étendue | MAD | Prefill médian, s | Génération médiane, s |
|---|---|---:|---:|---:|---:|---:|
| AR | MAIN | 11,487 | 11,471–11,544 | 0,012 | 7,824 | 18,885 |
| AR | HOLD | 11,423 | 11,383–11,608 | 0,034 | 7,289 | 18,413 |
| MTP | MAIN | 12,137 | 11,911–12,233 | 0,087 | 8,159 | 18,624 |
| MTP | HOLD | 11,172 | 11,066–11,348 | 0,082 | 7,641 | 19,002 |
| DFlash B2 | MAIN | 12,463 | 12,318–12,518 | 0,030 | 8,033 | 18,219 |
| DFlash B2 | HOLD | 11,919 | 11,737–12,107 | 0,143 | 7,553 | 18,243 |

Quatre mesures par case. Dérive des médianes premier/dernier processus : 0,03 à 1,99 % selon la case ; pause de reprise conservée. **Ce sont des références, pas des gains avant/après.** Les médianes de durées séparées ne s’additionnent pas.

| LONG512, 64 sorties | Prefill médian, s | Decode médian, tok/s | Temps complet médian, s |
|---|---:|---:|---:|
| AR | 41,463 | 10,935 | 47,225 |
| MTP | 44,714 | 11,678 | 50,110 |
| DFlash B2 | 44,058 | 12,193 | 49,226 |

Les sorties longues sont identiques entre modes ; ce contrôle long compare texte/tokens, pas tous les états numériques du contexte512.

### Démarrage et mémoire

| Processus de référence | Chargement annoncé, s | Spawn → ready, s | Pic footprint annoncé, Go décimaux |
|---|---:|---:|---:|
| AR1 | 0,589 | 1,630 | 7,137 |
| MTP2 | 1,021 | 1,763 | 7,886 |
| DF3 | 1,004 | 1,825 | 7,822 |
| DF4 | 1,008 | 1,824 | 7,833 |

L’écart spawn/chargement inclut démarrage/checkpoint/transport, sans ventilation mesurée. Footprint, RSS et mémoire MLX allouée sont distincts. Le pic `RUSAGE_CHILDREN` d’un superviseur multiprocessus est cumulatif : il ne donne pas un pic individuel indépendant pour chaque sous-processus.

## 3. Attribution CPU/GPU/POSIX sur une fenêtre commune

Fenêtre **87–113 : 27 tokens**, intersection des couvertures des trois captures, choisie par couverture et non par vitesse. Horloges recalées sur les lots POSIX synchrones et leur nombre/octets ; incertitude de l’ordre de 1–2 µs dans les analyses initiales.

| Temps instrumenté, ms/token émis | AR | MTP | DFlash B2 |
|---|---:|---:|---:|
| Fenêtre complète | 97,861 | 102,622 | 90,544 |
| GPU modèle actif | 59,175 | 66,386 | 58,665 |
| CPU inférence actif pendant GPU inactif | 19,545 | 17,106 | 14,834 |
| CPU inférence bloqué pendant GPU inactif | 17,910 | 17,185 | 16,060 |
| Attente des routes, inclusive | 33,863 | 31,328 | 26,570 |
| Dont GPU modèle actif | 33,767 | 30,732 | 26,464 |
| Soumissions MLX, inclusives | 24,897 | 26,915 | 23,336 |
| Dont CPU actif pendant GPU inactif | 12,433 | 9,323 | 7,940 |
| Lectures POSIX ∩ attente hôte ∩ CPU bloqué ∩ GPU inactif | 16,870 | 15,312 | 13,427 |
| Lectures POSIX ∩ GPU modèle actif | 5,844 | 9,644 | 8,257 |

**Ces lignes se chevauchent : ne pas les additionner.** GPU inactif signifie aucune activité GPU observée, y compris celle d’autres processus. GPU actif ne mesure pas l’utilisation ALU ni le temps exclusivement utile. Les partitions CPU ont un résidu nul dans ces fenêtres ; les budgets inclusifs et exclusifs sont conservés séparément.

Les 16,87 ms AR sont une intersection d’attentes exposées, pas une preuve causale de temps NAND entièrement récupérable. Une partie du GPU Cold W1 dépend déjà des lectures : le calcul n’est pas intégralement indépendant des I/O.

Contrôle détaillé de l’analyse AR initiale (fenêtre distincte78–117, conservée) : 5 885 lectures, 7 344 550 620 octets, 1 054 lots Cold et 637 lots synchrones, chaque lecture rapprochée exactement une fois. Calage −25 282 ns, demi-largeur1 438 ns. Les 20,35 ms POSIX exposées annoncées sur cette ancienne fenêtre ne remplacent pas les 16,87 ms de la fenêtre commune.

## 4. Quels appels MLX sont visibles pendant le GPU idle ?

Sur AR commun : 977 échantillons du thread d’inférence, dont 527 éligibles CPU actif/GPU idle. Les états juste avant une interruption d’échantillonnage sont explicitement distingués d’un état directement Running.

Symboles MLX les plus proches :
- `CommandEncoder::get_command_encoder` :45 échantillons ;
- `gpu::eval` :43 ;
- `dispatch_threads` :42 ;
- `maybeInsertBarrier` :40 ;
- `eval_impl` :25 ;
- `commit` :21.

111 points n’ont pas de frame MLX. Les piles complètes, allocateur, driver et inconnus sont conservés. **Ces comptes ne permettent pas d’attribuer des millisecondes exactes à chaque fonction.** Des appels de traçage du driver sont visibles : l’observateur lui-même coûte du temps.

## 5. Coût des observateurs

| Decode complet, s | Contrôle avant | Capture Xcode | Trace2 | Trace3 | Contrôle après |
|---|---:|---:|---:|---:|---:|
| AR | 11,455 | 12,789 | 13,505 | 10,988 | 10,532 |
| MTP | 10,489 | 12,178 | 15,698 | 10,394 | 9,963 |
| DFlash | 10,054 | 11,625 | 12,124 | 10,089 | 9,735 |

La capture ajoute environ12–22 % selon le témoin retenu. Les contrôles dérivent : aucun facteur unique de correction n’est justifié. Trace2 peut chevaucher la sauvegarde Xcode ; ne pas attribuer toute sa durée aux seuls spans. Trace3 est environ3,6–4,3 % au-dessus du contrôle suivant, comparaison séquentielle également soumise à dérive.

**Les chronos CLI non tracés décident des gains.** Les ms instrumentées ne sont pas converties directement en gains release. Au relevé AR, `pmset -g therm` ne rapportait aucun avertissement ; température physique et fréquence GPU restent non mesurées.

## 6. Vérification spéculative, acceptation et queues

Durées murales inclusives des compteurs DFlash release, normalisées par token de decode émis :

| Phase, ms/token | MAIN | HOLD |
|---|---:|---:|
| Proposition draft | 3,864 | 3,932 |
| Vérification cible | 74,927 | 77,671 |
| Commit/cache | 1,582 | 1,604 |
| Résidu hors de ces compteurs | 0,012 | 0,695 |
| Decode complet moyen | 80,385 | 83,903 |

La vérification représente environ93 % de ce budget ; supprimer ce calcul supprimerait aussi la vérification requise. Ce budget hôte inclusif n’est pas du temps CPU actif. La chronologie corrélée précédente ne doit pas être additionnée à ces timers.

- MTP :54 propositions acceptées/72 sur MAIN (75 %),48/78 sur HOLD (61,54 %),une queue AR dans les deux cas.
- DFlash :58/69 (84,06 %) et56/70 (80 %),138/140 lignes vérifiées,zéro/une queue AR.
- La queue AR HOLD est hors des trois timers : son résidu n’est pas intégralement un overhead.
- Demande logique d’experts DFlash :23,889/25,809 Go par requête ; pas un volume NAND physique.

## 7. Classement des pistes et plafonds

| Priorité | Piste | Plafond observé/théorique | Principaux risques |
|---|---|---|---|
| 1 | Réduire graphe/encodage exposé en gardant la disponibilité précoce des routes |12,43 ms/token AR dans cette trace ; pas intégralement supprimables | Déplacer une attente, supprimer l’overlap Hot, ajouter fences/copies, overhead de trace |
| 2 | Mieux masquer les attentes Cold |16,87 ms/token AR exposées, borne sous masquage parfait sans surcoût | Dépendance W1 aux lectures, durée de vie des slots, RAM3Gio, évictions/fallbacks |
| 3 | Éviter les logits intermédiaires MTP |87 heads,452,87 ms inclusives sur10,739 s de prefill tracé ; plafond de débit optimiste ~4,4 % | États/KV/annulation/dernière projection et comportement mémoire ultérieur |
| 4 | Draft/commit DFlash | Suppression fictive totale :~4,9–5,0 % de débit pour draft,~2,0 % pour commit | Bornes irréalistes ; préserver acceptation, états et travail cible |

Les plafonds ne s’additionnent pas. Réduire l’attente des routes ne récupère pas ses33,86 ms : le GPU est actif pendant33,77 ms. L’essai historique POSTPROFILE03 a déjà montré que retirer le premier `slots.indices().async_eval()` fait attendre les routes jusqu’à la fin du Hot ; il n’a pas été relancé.

Lectures directes Metal, chemin mono-lecteur inline et recouvrement partiel existent déjà. Ni le cache de préfixes ni une réduction du TTFT ne doivent être présentés comme un gain automatique de decode natif.

## 8. Qualité, gardes et limites

- Capture : six phases par mode, comparaison exacte des logits et états cible/head/draft, puis golden externe ; processus et traceurs joints.
- CLI court : mêmes128 tokens et texte entre modes. Hashes MAIN `f6b5f974c6db2e7a`,HOLD `9d02c1a97e015196`.
- Oracle indépendant : cinq tokens×48 couches,248320 logits et états/reset ;177 vagues Cold naturelles.
- Analyseur :13 cas synthétiques couvrant aussi données tronquées/corrompues,mappings,parité absente,octets/nombres faux,fenêtre non couverte et trous CPU.3 375 unions+3 375 intersections finies,cas d’intervalles/XML invalides. Ce sont des tests finis, pas une preuve générale.
- Suite Python pertinente :1302 tests réussis avant l’entrée de reprise ;18 tests de garde réussis après son ajout. Suite Rust CPU :276 réussis,9 ignorés. Format/Clippy strict `mlx,chat`,build release et Ruff passent. Le build CPU seul ne valide pas MLX.
- CI PR/main inspectée et étendue localement ; aucun run distant de ce diff non publié.
- Kani sur le chemin réel MTP : tentative avec token u32 et booléens symboliques,deux tokens ; arrêt à la limite mémoire512Mio avant qualification des obligations. CrossHair absent. **Aucune preuve formelle MLX/Metal/concurrence/génération complète.**

Gardes : admission estimée libre+file-backed≥10,5Go,secteur,un seul modèle ; RSS≤10Gio,footprint≤12Go. Estimation récupérable et RAM physique ne sont pas synonymes.

Le premier protocole zéro écriture a interrompu MTP5 après192Kio de swap global. Reprise explicitement autorisée à32Mio depuis la même ancre,incident compris,lectures≤256Mio. MTP5 et AR6 ont ensuite été achevés ; l’essai interrompu reste exclu de l’agrégat complet mais conservé.

Fin du profilage long :0,75Mio écrits,68 108 288 octets lus depuis l’ancre P09. Après une pause, les compteurs globaux ont dépassé ces plafonds avant les nouveaux tests ; origine non attribuée à un processus. Une **nouvelle campagne MTPHEAD**,explicitement autorisée,conserve l’ancienne ancre et l’incident. Aucun réancrage silencieux.

## 9. Optimisation issue du profilage : statut séparé

**C1 : seul dernier head du prefill MTP.** Qualité physique et bridge réel validés ; même binaire,A-C-C-A,quatre mesures parvariante/prompt.

| Charge | Prefill A→C1, s | Réduction prefill | Decode A→C1, tok/s | Variation decode | Réduction temps complet |
|---|---:|---:|---:|---:|---:|
| MAIN87/128sorties |8,168→7,897|3,32 %|12,113→11,954|−1,31 %|0,59 %|
| HOLD82/128sorties |7,671→7,512|2,07 %|11,149→11,136|−0,12 %|0,74 %|
| LONG512/64sorties |44,432→42,954|3,33 %|11,906→11,718|−1,58 %|2,93 %|

C1 n’est pas validé comme réglage livré sans régression de decode. L’audit confirme le bon chemin,les états exacts,l’absence de copies/coût mémoire important ajoutés et les synchronisations conservées.

**TAIL2/C2 : deux derniers heads.** Hypothèse : restaurer la présence des logits précédents pendant le dernier token rapproche le comportement d’allocation terminale de la référence. Ce n’est pas une cause démontrée. Les tests physiques et le bridge passent ; pilote court prometteur mais confirmation en cours à la publication. Un prefill HOLD de8,479s est conservé, pas supprimé comme outlier. **Aucun gain C2 confirmé à ce stade.**

Contrôles MTP : longueurs1/2/7/8/9/17,deux réglages,logits finis/états/KV exacts,trois pas de continuation et compteurs,cinq cas d’erreur/reprise,comptage explicite des heads. Contrôles AR/DFlash également passés sur leurs chemins inchangés.

## 10. Artefacts et reproductibilité

Racine locale : `docs/measurements/decode-profile-p09-20261101/`.

- `baseline-complete-summary.json`,`long-{AR,MTP,DF}-inspection.json`.
- `AR-common/attribution.json`,`MTP-common/attribution.json`,`DF-common/attribution.json` ; fenêtres initiales conservées séparément.
- Captures `.trace`,exports XML,profils natifs,rapprochements I/O,piles CPU,sources exactes de l’analyseur et empreintes.
- Plans de chaque processus,contrôles de parité,conditions mémoire/alimentation,reçus de jointure et erreurs conservées.
- `mtp-head/{short,long}-inspection.json`,`mtp-head-tail2/` pour les expériences distinctes.
- Journal `opti.md` et rapport `docs/decode-profile-p09-20261101.md`.

**Cette publication met le rapport dans l’issue. Les captures brutes, sources locales de diagnostic et prototypes P09 ne sont pas poussés avec cette action** : les chemins ci-dessus sont des références d’artefacts locaux, pas des liens GitHub téléchargeables. Aucun remplacement de l’application installée.

