# Flash Next SSD : prélecture du vérificateur et draft préplafonné (69–70)

État : correction du pic mémoire70M validée, options69/70 expérimentales
désactivées par défaut. Aucun gain de débit69 qualifié ; reprise du débit70
préparée après correction du chargement. L’app installée reste inchangée.

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
EXL3 directement sur GPU Hot (71) reste préenregistré, sans prototype, jusqu’à
la clôture de ces deux étapes, conformément à la dernière instruction.

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
