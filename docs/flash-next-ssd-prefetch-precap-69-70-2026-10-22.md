# Flash Next SSD : prélecture du vérificateur et draft préplafonné (69–70)

État final du 22 octobre : correction du pic mémoire70M et gain PRECAP70
contre l'ancien adaptatif validés et poussés. Les témoins OFF/MTP1/B2 sont
terminés ; B2 fixe reste le plus rapide observé. Aucun gain de débit69
qualifié, options69/70 expérimentales désactivées par défaut. L’app installée
reste inchangée ; les 15 tok/s ne sont pas atteints.

## Implémentation et périmètre

- **69**, `MLXL3_SSD_PREFETCH_PIPELINE=1` avec lookahead2 : le véritable
  `verify_draft` transmet le MoE suivant, prédit ses experts sur les deux
  premières positions au maximum et conserve GPU Hot anticipé, y compris sur
  les fenêtres réunifiées ou découpées B1–B8. Le pool existant reste borné à
  quatre lectures. Une demande du même expert/couche/cache peut récupérer une
  lecture en cours ; l’échéance commune est nominalement 1 ms. Un timeout garde
  le lease d’écriture et utilise le repli urgent sûr. Une complétion invalide
  n’est jamais publiée. L’ordonnanceur OS peut dépasser l’échéance nominale.
- **70**, `MLXL3_FLASH_DRAFT_PRECAP=1` : le contrôleur choisit une à trois
  propositions **avant** les cinq couches du draft, soit B2/B3/B4. Il mesure les
  tokens réellement émis par temps de cycle, essaie chaque forme puis explore
  périodiquement. Le choix de préfixe après confiance reste disponible. Le
  contrôleur est réinitialisé avec la requête. Les modes fixes restent séparés.
- Le draft DFlash2 est non causal : raccourcir sa forme peut changer ses
  propositions. La référence est la génération cible séquentielle, jamais un
  préfixe de propositions B8 présumé équivalent.
- Compteurs ajoutés sans `eval()` de profilage : histogrammes des largeurs
  réellement calculées/vérifiées, positions calculées, prélectures soumises,
  complétées, utilisées, réutilisées à la demande, timeouts, attente, octets
  complétés et demandes tardives. Les octets d’API incluent les hits du cache
  fichier et ne mesurent pas les octets physiques NVMe. Les compteurs asynchrones
  sont relevés séparément : leur différence n’est pas une demande SSD exacte.

Les 3 Gio d’experts et leurs 53 slots/couche sont conservés. Aucune modification
LFM2, kernel EXL3 ou politique adaptative de cache supplémentaire. Le partage
EXL3 directement sur GPU Hot (71P) reste préenregistré, sans prototype ; il
n'est inclus dans aucun résultat de ce rapport. Les étapes69/70 sont closes.

## Sources et protocole

M5 10 cœurs GPU, 24 Gio, macOS27.2, MLX0.32.2, Rust1.98.1. Base publiée
`ba97af8d93dce22829efd587dd37907de35d89f1`. Sources locales modifiées gelées
dans [identity-v3.json](measurements/flash-next-ssd-impl/opt-70-pre-draft/identity-v3.json)
et son archive158fichiers. Pour la publication,
[identity-v4.json](measurements/flash-next-ssd-impl/opt-70-pre-draft/identity-v4.json)
ne change que le timeout CI Flash15→30min ; les158sources numériques/outils et
les deux binaires sont identiques hormis ce fichier de workflow. Binaires70v2 :

| Artefact | SHA-256 |
| --- | --- |
| CLI | `05e03a2c0a8c7c42c72440fb214275e75730f6e59d3832c28d984d359706d13c` |
| Tests natifs | `a8b30526870efc2450ea08cd4f194fca7563a593a40b6eb9f715864422b03689` |

Checkpoints cible `65c895314393431c09050b2e04e250836b3a6eb4`, draft
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`, quantification EXL3 2.05bpw,
draft Q4/cache circulaire, contexte4096, GPU Hot Wide/63/UNIQUE actifs,
shared65/68 et stable67 désactivés. Revues amont :
[69](flash-next-ssd-hot-prefetch-review-69-2026-10-22.md),
[70](flash-next-ssd-splash-strata-followup-70-2026-10-22.md), Splash1482f6bb,
Strata011c9e60. Le préplafonnement est une adaptation locale ; Splash calcule
encore huit positions avant son choix de préfixe dans le chemin examiné.

Pour les vitesses, voir les plans Q64C : MAIN/HOLD,128 sorties/127 décodées,
un warmup et une mesure par enfant, témoins frais avant/après. 69 compare
H sans prédiction à C avec pipeline/lookahead2 sous LRU identique. 70 compare
ADAPT7 à PRECAP sous autoLFU identique, pipeline69 désactivé. Validation HCCH
conditionnelle après écran positif≥1% sur les deux prompts ; dériveH≤5%,
étendue≤3points, surplusC≤256Mio. Pas de victoire contre MTP à partir de points
sélectionnés : il faudra une comparaison alternée dédiée si une variante passe.

## Vérifications et limites déjà connues

- CPU/chat :130 passés,5 ignorés. Variante livrée `mlx,chat` :153 passés,
  153 ignorés ; fmt, Clippy strict et build release passent. Les tests physiques
  sont lancés seuls et explicitement, jamais assimilés aux tests ignorés.
- Python après correction du checker :989 passés,4 ignorés ;18 tests65P
  préexistants exclus de la variante publiable.51 contrôles ciblés de transport
  et checker. Le défaut SIGTERM a été reproduit avant correction. Les deux
  interruptions réelles vérifient le rapport failed/parity=null, le progrès
  conservé et l’absence du processus natif après nettoyage. Les snapshots sont
  atomiques avant/après chaque requête. SIGKILL reste non interceptable.
- Kani0.68 / CBMC6.11 : géométrie69 vérifiée sur tous les `usize` symboliques,
  choix70 et transition70 sur états/entiers symboliques complets, unwind5,
  couvertures5/5,4/4 et4/4 atteintes. La propriété supplémentaire de classement
  des rapports128bits a expiré à120,137s : non prouvée. Mutations de géométrie,
  câblage des prédictions et classement détectées. Les tentatives Kani du vrai
  chemin MLX échouent avec ICE243 avant obligations. CrossHair absent. Ces
  vérifications CPU ne prouvent ni Metal, ni l’allocateur, ni la concurrence.
- Qualité69 native à64Mio :32 cas MoE,36 préfixes avec logits248320/états/captures
  bit à bit, sessionMTP et dix paires passent ; pic8,160Go. Test de rétention :
  64 cohortes/256 lectures réelles, quatre pending au plus, owners1→5→1,
  aucun pending restant, aucune croissance de mémoire MLX active, cache libéré
  après drop. Ce test borné ne prouve pas toute la mémoire de l’inférence.

## Mémoire : conserver les deux échecs

| Fenêtre | Résultat |
| --- | --- |
|69 initiale, cache temporaire512Mio|106037248B de nouveaux swap-outs pendant la qualitéMTP ; arrêt, aucun TPS|
|69Q64, cache temporaire64Mio|Qualité native passée ; bridge arrêté après282,308s sur50987008B supplémentaires ; aucun TPS|
|69Q64C, après fermeture d’applications confirmée|Validation en cours ; nouveau point de départ explicite157024256B, sans effacer les deux échecs|

Les deux premiers incidents ont atteint environ8,2–8,5Go de compresseur
système. La présence simultanée d’autres processus n’établit pas une cause
exclusive. Ne pas prétendre que64Mio a résolu la pression. L’ancien checker ne
persistait pas son avancement avant SIGTERM : le nombre de requêtes achevées
dans Q64 est perdu et reste inconnu. Son rapport initial n’a pas été reconstruit.

Q64C conserve les experts3Gio, le cache temporaire64Mio, pré20/post30, zéro
nouveau swap-out par rapport à157024256B, Swapins cumulés depuis reboot≤256Mio
et empreinte≤12Go. Après les tests CPU,60s stables et compresseur≤7Go sont requis.
Une nouvelle garde échouée ferme la campagne. Les probes2s ne peuvent empêcher
un burst entre deux observations.

## Résultats Q64C

Le bridge69 passe **43/43 contrôles** en322,333s, pic **7933418072B**,
zéro nouveau swap-out pendant l’exécution et le postflight30s. Les contrôles
incluent erreurs, cache/conversation et annulation/reprise. Les profils confirment
des prélectures consommées pendant les vrais blocs DFlash2. Preuve :
`q64c-bounded/bridge69/campaign-result.json`.

Deux incidents de protocole restent conservés : une tentative arrêtée
explicitement après dix requêtes car son délai600s était jugé trop court, puis
le refus immédiat d’un délai parent1260s au-delà du maximum900s du superviseur.
La version finale respecte checker840s/parent900s. L’extrapolation initiale
des21–23s par préfill froid était trop pessimiste : les requêtes suivantes sont
tombées autour de8s et le bridge entier a terminé en322s. Aucun de ces arrêts
ne constitue une divergence ou une mesure de débit.

La qualité70 est close également : **512 tokens et136 composants d’état**
identiques à la cible séquentielle, pic8086100424B, histogramme des propositions
calculées `[0,14,43,191,0,0,0,0]` (indices = nombre de propositions, plus
l’ancre). Le bridge70 passe43/43 en302,631s, pic7938562552B. Six valeurs
invalides du flag69 sont rejetées avant ready ; PRECAP1/ADAPTIVE0 est rejeté
avec récupération OFF exacte. Tous les enfants sont joints, zéro nouveau
swap-out et postflight commun30s passés. Le premier plan d’observation à vide
omettait `env={}` : il a été rejeté avant lancement, corrigé dans `boundary-settlement-v2`.
Le témoin MTP déclenché par erreur à la suite de ce rejet a été interrompu
pendant son préflight, sans inférence ni ligne mesurée ; dossier conservé.

### Écrans69, aucune promotion

| Mode | Prompt | H sans prédiction | C pipeline/lookahead2 | Écart écran |
| --- | --- | ---: | ---: | ---: |
|B2 fixe|MAIN|11,134690|10,489017|−5,799%|
|B2 fixe|HOLD|10,401663|9,921178|−4,619%|
|Adaptatif antérieur|MAIN|10,380755|9,975263|−3,906%|
|Adaptatif antérieur|HOLD|9,198679|9,519339|+3,486%|

Unités : tok/s. Un A/B par prompt, aucun C-H conditionnel éligible : **aucun
gain qualifié, aucune régression stable établie**. Seize réponses128tokens sont
exactes, aucun nouveau swap-out, surplus mémoire C≤256Mio, pic maximal7,946Go.
Le témoin MTP frais avant69 est11,048225/10,175733tok/s MAIN/HOLD, et après69
10,991800/10,238255, soit−0,511%/+0,614% de dérive. Les24réponses (mesures et
warmups) sont exactes ; pic vitesse global8202261832B. Aucune victoire intermode
n’est déduite de ces écrans.

Sur B2, acceptation et nombre de blocs sont identiques H/C, mais les octets
complétés de l’API augmentent **12,573% MAIN /10,905% HOLD**. La vérification
passe10,674715→11,377182s et11,373112→11,925580s. L’attente des demandes en
prélecture ne représente que0ms/3,826ms cumulés, sans timeout. Le coût n’est
donc pas expliqué par une longue attente de promotion des lectures en vol.
Cela ne mesure pas la bande passante physique du SSD. En adaptatif, les nombres
de blocs changent49→50 et61→55 ; les écarts ne peuvent pas être attribués
exclusivement au pipeline.

### Revue69R2 : admission des prélectures, sans prototype

La politique locale `auto|lru` ne propose pas de littéral `lfu` ; la prélecture
appelle `protect_predictions`, qui active de toute façon LRU. Les candidats
sont intercalés par rang sur les deux lignes, jusqu’à32 IDs protégés ; une
complétion devient récente dans LRU. `victim` ne donne pas de priorité distincte
aux records préchargés jamais consommés. Strata classe ses topK par meilleur
score sur la fenêtre et peut placer les prélectures à l’arrière de LRU ;
Splash trie par somme des poids et précharge des fragments avant staging GPU.
Attention : Strata choisit l’insertion arrière par défaut seulement au-dessus
de1,5Mo par expert ; nos records1248012B sont du côté de son défaut MRU avant.
Son choix dépend de mesures M4/formats différents. Une priorité locale aux
prélectures inutilisées serait donc une hypothèse à tester, pas un oubli de
portage démontré par son code.
Ce sont des différences d’architecture à étudier, pas un résultat de vitesse.
La prochaine hypothèse est de distinguer les prélectures inutilisées des
experts confirmés par demande, en conservant les exclusions pending/routes.
Aucune modification de cache ou de kernel n’a encore été introduite pendant
les mesures gelées. La position de `prediction.async_eval` avant les routes
actuelles est également identifiée ; son coût exclusif reste non mesuré.

### Arrêt avant les débits70

Le premier témoin B2-70 a été arrêté et joint après2,015s au chargement, sur
**131072B** de nouveaux swap-outs système. Avant le chargement : compresseur
5425283072B, mémoire libre6055690240B. Dernière sonde : compresseur8390541312B,
libre128188416B, compteur Swapouts157024256→157155328B. Aucun ready, warmup ou
point de débit70 terminé ; son pic physique processus est **non mesuré**.
Après nettoyage, le compresseur est retombé autour de5,16Go et aucun moteur ne
reste actif. Cette coïncidence temporelle ne démontre pas une cause exclusive.

PRECAP et ses autres témoins sont explicitement non lancés. La qualité70 reste
valide ; aucun gain70, zéro-swap global sur toute la session ou atteinte15tok/s
ne peut être annoncé. Le protocole envisagé pour la suite réutilisera l’option
existante du benchmark qui mesure les deux prompts dans un même processus,
avec nouveaux témoins et nouvelle fenêtre documentée. Ce protocole n’est pas
encore exécuté et ne résout pas à lui seul la cause de pression mémoire.

Aucune option promue. Les preuves brutes,
commandes, identités et interruptions restent dans
[69](measurements/flash-next-ssd-impl/opt-69-prefetch-pipeline/) et
[70](measurements/flash-next-ssd-impl/opt-70-pre-draft/).

## 70M : suppression du pic inutile au chargement

Le loader remplissait immédiatement de zéros les sept banques de chaque
couche : 48 × 53 × 1248012 = **3174942528 octets** de slots encore libres.
Les banques réservent maintenant la même capacité avec l'allocateur MLX
déjà utilisé par le lecteur direct. Une lecture ne rend un slot accessible
qu'après remplissage ; les kernels Hot ignorent les slots absents et pending.
Les 12 segments SSD couvrent entièrement chaque slot. Aucune capacité, règle
de routage, quantification ou opération arithmétique n'est modifiée.

La réservation synchrone inscrit les clés avant l'I/O : le snapshot Hot est
pris avant cette réservation, et les nouveaux slots Cold ne sont dispatchés
qu'après le retour réussi du lecteur. `ExpertFill` invalide les clés sur erreur.
Pour les lectures asynchrones, les slots pending sont exclus du snapshot et
la complétion valide publie le record. La sûreté dépend aussi de cet ordre ;
il ne faut pas introduire un snapshot entre réservation Cold et fin de lecture.

Le constructeur Rust est privé et `unsafe` : son contrat interdit toute
lecture d'une région non remplie. Les tests empoisonnent les banques libres
avec des NaN dans les scales pour exposer une lecture prématurée. Ce contrat
ne remplace pas une preuve de l'exécution Metal ou de la concurrence.

Comparaison de chargement uniquement, quatre bridges sans génération, ordre
C-H-H-C, même checkpoint, secteur, 3 Gio / 53 slots par couche, cache temporaire
MLX 64 Mio, contexte4096, pipeline/préplafonnement désactivés. Chaque enfant
atteint ready puis ferme stdin, avec observation30s après sortie.

| Passage | Version | Pic physique processus |
| --- | --- | ---: |
| 1 | C, banques réservées | 3738718936 B |
| 2 | H, banques mises à zéro | 6928372296 B |
| 3 | H | 6924997240 B |
| 4 | C | 3734704832 B |

Réduction moyenne **3189972884 B**, soit **3,19 Go / 46,05 %** du pic de
chargement. Les quatre processus sont joints et conservent zéro nouvelle
sortie de swap, ancre157155328 B. Les anciens échecs restent conservés ;
l'ancien binaire ne swappe pas dans cette nouvelle fenêtre non plus.
Le résultat démontre la suppression de pages inutilement touchées au
chargement, sans attribuer tous les anciens swap-outs à cette seule cause.
La capacité se remplit avec les experts réellement utilisés : cette baisse
ne prédit pas une réduction identique pendant une longue génération.

Qualité initiale :154tests Rust passés/154ignorés, fmt/Clippystrict/build
`mlx,chat` passés ; quatre contrôles physiques sélectionnés passent, dont
16cas de prélecture avec poison,32cas de cohortes/pending/repli et des
lectures échouées suivies de reprises. La mutation qui remet zeros dans le
constructeur est détectée, puis restaurée. Kani vérifie le vrai calcul des
destinations pour chaque slot symbolique0..511, unwind14 :228obligations,
2couvertures atteintes ;1assertioninatteignable de formatage dans la branche
d'erreur exclue. KaniMLX reste bloqué par l'ICE `intrinsics.rs:243`, CBMC par
le parseur libc++Apple. Aucune preuve GPU/FFI complète.

Les contrôles modèle complets passent :36préfixes/logits/états/captures64000,
sessionMTP et dixpaires, puis512tokens PRECAP et136composants d'état exactement
égaux à la cible séquentielle. Histogramme calculé `[0,6,6,213,0,0,0,0]` :
B2/B3/B4 réellement atteints. Les deux vrais bridges passent **43+43contrôles**,
annulations/reprises/cache/erreurs compris, en313,608 et297,460s.

Pic maximal de génération contrôlée **8134777504 B**, pré20/post30 de chaque
phase et zéro nouveau swap-out à l'ancre157155328 B. Tous les enfants sont
joints. Le pic de génération reste voisin de celui des anciennes campagnes :
le correctif évite le remplissage inutile au démarrage, en conservant la
capacité complète lorsque les experts finissent par l'occuper.

Les débits PRECAP70 restent non mesurés à cette clôture. Leur reprise70S2
est préenregistrée : B2/MTP avant, adaptatif H-C-C-H conditionnel sur MAIN/HOLD,
B2/MTP après, mêmes3Gio/64Mio,128sorties, nouveaux témoins, aucune reprise
silencieuse du point interrompu. Aucun15tok/s ni victoire sur MTP présumé.
[Commandes, identités, résultats bruts et limites70M](measurements/flash-next-ssd-impl/opt-70-load-memory/).

## 70S2 : préplafonnement validé contre l'ancien adaptatif

Après publication de70M au commit82cbef71, nouveaux témoins et comparaison
H-C-C-H, mêmes source/binaire/options hors préplafonnement :

| Prompt | Ancien adaptatif H | PRECAP C | Gain par paire |
| --- | --- | --- | --- |
| MAIN | 10,969865 / 10,812826 | 11,375660 / 11,357701 | +3,699 % / +5,039 % |
| HOLD | 10,032978 / 9,860747 | 10,826402 / 10,886474 | +7,908 % / +10,402 % |

Unités tok/s ; moyennes C **11,366681 / 10,856438**. Les dérives H sont
1,452 % / 1,747 %, les étendues de gains1,340 /2,494points : le critère
HCCH préenregistré passe. Les16réponses128tokens (warmups compris) sont
exactes, les huit enfants sont joints et le postflight30s conserve zéro
nouveau swap-out. Pic C≤7891507464B, H≤7950457240B. Le gain concerne
l'adaptatif qui calculait jusqu'à sept propositions avant de tronquer ;
il ne démontre pas un gain face à B2 fixe.

Témoins avant : B2fixe11,470853/10,912513 etMTP1 11,219774/10,204433tok/s.
Les témoins après sont désormais terminés, voir ci-dessous. PRECAP demeure
une option expérimentale désactivée ; aucun15tok/s annoncé.
[Données et critères](measurements/flash-next-ssd-impl/opt-70-load-memory/speed70/precap-gain.json).

## 70S2 clos : classement actuel et coût restant

Même binaire70M/v2, cache expert3Gio/53slots, allocator64Mio, contexte4096,
secteur, un warmup128 puis une mesure128 par processus,127tokens décodés.
Les deux témoins B2/MTP encadrent le HCCH PRECAP ; OFF est un témoin final
supplémentaire. Les moyennes suivantes conservent tous les points prévus.

| Mode | MAIN, tok/s | HOLD, tok/s | Mesures par prompt | Pic physique maximal |
| --- | ---: | ---: | ---: | ---: |
| OFF | 10,728288 | 10,337428 | 1 | 7141824416 B |
| MTP1 | 11,135342 | 10,200603 | 2 | 7890901520 B |
| DFlash2 B2 fixe | **11,544644** | **11,001097** | 2 | 7847909712 B |
| DFlash2 PRECAP | 11,366681 | 10,856438 | 2 | 7891507464 B |

MTP après :11,050911/10,196773tok/s, dérive −1,505%/−0,075% par rapport à
avant. B2 après :11,618435/11,089681tok/s, dérive +1,287%/+1,624%. Les
contrôles restent sous5%. B2 et PRECAP dépassent les deux points MTP de
chaque prompt ; c'est un classement descriptif sur ces deux prompts et ces
effectifs, pas une qualification universelle contre MTP. PRECAP est qualifié
contre l'ancien adaptatif, pas contre B2. Passer de la moyenne B2 MAIN à15
demanderait encore environ30% de débit supplémentaire.

**36 réponses de128tokens exactes**,18warmups et18mesures, hachages identiques
aux golden indépendants60. Tous les enfants sont joints, toutes les phases
pré20/post30 passent et **zéro nouveau swap-out système** est observé depuis
l'ancre explicite157155328B. Cela n'efface pas les échecs antérieurs. Aucun
autre modèle, build ou proveur n'a tourné pendant les mesures GPU.

Les profils existants, sans nouvel `eval()` de mesure, expliquent le gain :

| Moyenne HCCH, ancien adaptatif → PRECAP | MAIN | HOLD |
| --- | ---: | ---: |
| Temps draft | 1,192475 → 0,673266 s | 1,399065 → 0,773872 s |
| Temps vérification cible | 10,296577 → 10,317046 s | 11,076570 → 10,628498 s |
| Positions calculées par le draft | 381 → 199,5 | 442 → 223 |
| Lignes cible vérifiées | 138 → 145 | 140 → 136,5 |
| Octets d'API SSD par token décodé | 180,805 → 186,671 Mo | 196,542 → 198,861 Mo |

La vérification représente encore92,3%/90,9% du décodage PRECAP. Sa durée
inclut routage, attentes SSD, backbone et MoE ; ce n'est ni un temps GDN
exclusif, ni un temps GPU exclusif. Les octets d'API peuvent provenir du cache
fichier : aucun volume physique NVMe n'est déduit. La réduction du draft
n'a pas supprimé le travail de la cible.

[Matrice finale et limites](measurements/flash-next-ssd-impl/opt-70-load-memory/speed70/summary.json),
[profils détaillés](measurements/flash-next-ssd-impl/opt-70-load-memory/speed70/precap-components.json).

Les159sources exécutables et les deux binaires sont inchangés depuis le
correctif publié82cbef71, puis le gain publié2945cd6b :

| Artefact70M/v2 réellement mesuré | SHA-256 |
| --- | --- |
| CLI | `8bb0733f5352a6d7e46515ff25df965007f417c181bd1fc8c9c148eb001f9804` |
| Tests natifs | `01137fb55487c92aa6a1c366346bc17394c8e55980b05a587dd99ffb73449012` |

[Identité complète](measurements/flash-next-ssd-impl/opt-70-load-memory/identity-v2.json),
[dépendance MLX](measurements/flash-next-ssd-impl/opt-70-load-memory/runtime-dependency.json),
[vérification finale des empreintes](measurements/flash-next-ssd-impl/opt-70-load-memory/identity-final-check.json).
Les anciens binaires70v2 du début de ce rapport identifient les essais
antérieurs au correctif mémoire ; ils ne sont pas ceux de70S2.

Commandes locales finales, toutes réussies avant les mesures GPU :

```sh
cargo fmt --check
cargo clippy --release --all-targets --features mlx,chat -- -D warnings
cargo test --release --features mlx,chat -- --test-threads=1
cargo build --release --features mlx,chat
cargo kani --lib --no-default-features --harness slot_reads_initialize_exact_disjoint_bank_regions
```

La suite normale livre154tests passés/154ignorés. Les quatre tests composant,
quatre tests modèle et deux bridges43 ont été exécutés séparément ; leurs
succès n'activent pas tous les154tests ignorés. Les989tests Python/4ignorés
précèdent70M, qui ne modifie pas ces sources Python, empreintes recontrôlées.
Les18tests65P préexistants restent exclus de cette publication. Les preuves
CPU bornées, l'ICE KaniMLX, l'échec de parsing CBMCApple et le timeout du
harnais optimalité128bits restent des résultats distincts, pas une preuve
complète du moteur. [Commandes et codes de retour](measurements/flash-next-ssd-impl/opt-70-load-memory/checks-final.json),
[Kani des intervalles](measurements/flash-next-ssd-impl/opt-70-load-memory/kani-slots.json).

CI2945cd6b inspectée : jobs natifs Ubuntu/macOS et régression Desktop/protocole
réussis. Le job Kani global échoue après un arrêt du runner, exit143, pendant
`cache_allocation_is_optimal_and_exact_on_two_layers_two_levels`, avant
conclusion de ce harnais ; aucun contre-exemple dans les obligations terminées.
Flash Kani reste en cours à cette observation. Aucune CI entièrement verte
revendiquée. [Log exact du job arrêté](measurements/flash-next-ssd-impl/opt-70-load-memory/ci-2945cd6b-kani-job.log).

## 71R : revue Splash/Strata après les mesures, sans nouveau prototype

Les HEAD consultés sont toujours Splash1482f6bbc2abf12375332725302019594150b6cb
et Strata011c9e600a93bbc9fd451c563501b5831c32bbca. Six fichiers ciblés ont
été recontrôlés byte à byte contre ces commits, pas seulement les README.
[Sources, SHA-256 et périmètre](measurements/flash-next-ssd-impl/opt-71-research/sources.json).
Cette revue ne prétend pas couvrir tout le code amont.

| Piste | Différence réelle et prochaine vérification utile |
| --- | --- |
| Ordre Hot/prédiction | `SsdMoe::forward_draft_lookahead` soumet `prediction.async_eval` avant `forward_draft_rows`, donc avant les routes/Hot courants. Le chemin local `forward_rows` possède déjà une soumission après Hot, mais passer simplement par `forward_lookahead(depth>0)` désactiverait le fast path hybride actuel et changerait les limites B8. Une expérience distincte devrait conserver Hot anticipé, borner la prédiction à deux lignes et déplacer seulement sa soumission. Mesurer ensuite attente critique et octets, avec un témoin sans prédiction. |
| Départ des workers après reprise GPU | Splash sépare `planAhead` de `startAhead` et déclenche le second dans le callback suivant la libération GPU. Notre pool est borné, quatre workers avec un job chacun, mais cette frontière native de reprise n'existe pas dans le wrapper MLX. Il faut mesurer le coût de reprise avant de porter un ordonnanceur entier. |
| Partage EXL3 directement sur Hot B2 | Splash groupe surGPU les routes et dispatch les seules tuiles actives.71P appliquerait le partage seulement aux experts Hot présents une fois dans chaque ligne, en écrivant directement aux positions existantes. Le prototype CPU65/66 et ses `concatenate`/`take` ne sont pas réactivés. Le coût du plan et la pression registres peuvent annuler le gain ; oracle bit à bit puis B2 réel indispensables. |
| Fenêtres persistantes | Strata réutilise `Window::Cached` selon largeur, tête, état de fold et modes ; seules certaines positions/QSA sont patchées. Notre cache de factories Metal ne remplace pas ces buffers et listes de dispatch persistants. Chantier distinct et plus lourd : d'abord séparer construction hôte, attente SSD et exécution GPU, sans supprimer les protections de slots ni la causalité. |

Sources primaires : [Splash ExpertStore, `planAhead`/`startAhead`](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/ExpertStore.cpp#L594),
[Splash MoE, plan et grilles indirectes](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.cpp#L195),
[Strata Window, cache de fenêtres](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/window.cpp#L1083).
Les économies décrites dans leurs commentaires ne sont pas des gains MLXL3.

Garde-fous tirés de l'historique et du code actuel :

- Le vrai dispatch indirect a déjà été implémenté dans57P : aucun cas mixte
  stable positif, puis échec du postflight mémoire. Le refaire isolément ne
  constitue pas une nouvelle piste ; le mécanisme nouveau71P serait le travail
  de déquantification réellement évité, sans regroupement CPU.
- Shared expert groupé68 et grouped EXL3 CPU65/66 restent négatifs. Les
  résultats B4/B8 ne sont pas remplacés par le seul meilleur microbenchmark.
- Strata choisit MRU ou arrière LRU selon la taille d'expert, seuil1,5Mo.
  Nos records1,248012Mo relèveraient de son côté MRU/top10. L'éviction
  préférentielle des prélectures inutilisées reste une hypothèse locale69R2,
  motivée par les octets supplémentaires, pas une correction amont évidente.
  [Initialisation Strata](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/stream.cpp#L151).
- Le draft Splash utilise AF4G64 ; notre draft affine4/groupe64 lit des plans
  MLX et des scales/bias FP16. Le format général est proche, mais les plans,
  arrondis et épilogues doivent être vérifiés avant toute substitution. La
  cible EXL3 trellis et son codebook arithmétique restent distincts des kernels
  GGUF/IQ et de leurs tables constantes. Copier ces kernels n'est pas un port
  exact automatique. [Draft Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/DFlashDraft.cpp#L69).
- Les experts sont déjà des records contigus avec12segments/preadv ; ni un
  second repacking, ni davantage de cache sans mécanisme nouveau ne sont
  retenus. Le maintien GPU par API Metal non publique de Splash n'est pas porté.

Décision71R : revue close, aucun nouveau débit ni gain annoncé. Priorité
suivante proposée : isoler l'ordre de prédiction69, puis71P Hot B2 avec
sortie directe ; fenêtres natives seulement si les délais hôte le justifient.
71P est différé, sans code ni mesure en cours. Pour un port futur, conserver
les attributions Apache2.0 Splash et MIT Strata. Aucun changement exécutable
ni nouvelle inférence n'a été effectué pendant71R.
