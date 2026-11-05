# 66F — DFlash2 réel en B2/B4/B8

Demande utilisateur : tester ces trois largeurs en génération réelle, même si
leur coût est important. Cette demande autorise la campagne malgré le crible
synthétique 65T non concluant ; son verdict et ses données restent conservés.
La campagne est terminée : aucun gain du partage EXL3 n'est qualifié et les
témoins trop variables empêchent une conclusion entre modes.

## Largeur réellement vérifiée

B désigne l'ancre plus les propositions du draft. Les anciens modes fixes
étaient mode1=B4, mode2=B3, mode3=B8. Mode2 ne permettait donc pas le test B2.

L'option expérimentale `MLXL3_FLASH_DRAFT_WIDTH` accepte les littéraux 1..7
avec `MLXL3_FLASH_DRAFT_ADAPTIVE=0`. Sans cette option, les comportements des
modes restent identiques. Une valeur invalide ou une combinaison avec le
drapeau adaptatif actif est rejetée avant tout chargement du draft.

| Test | Propositions | Fenêtre cible maximale | Réglage |
|---|---:|---:|---|
| B2 | 1 | 2 | WIDTH=1, ADAPTIVE=0 |
| B4 | 3 | 4 | WIDTH=3, ADAPTIVE=0 |
| B8 | 7 | 8 | WIDTH=7, ADAPTIVE=0 |

Le benchmark conserve maintenant l'événement `generation_mode` du vrai bridge.
Le validateur exige sa largeur exacte et des compteurs de propositions et de
lignes vérifiées cohérents. Les fenêtres finales peuvent se raccourcir : après
plusieurs refus, plusieurs petites fenêtres sont légitimes en fin de budget.
Cette précision du validateur a été identifiée et testée avant tout débit 66F.

## Qualité et provenance

Base publiée `5ab2c7f8`, état local 66F. Binaire CLI figé :
`156cc2639c46599e1106504bd9bbd304adf4f7fe775bdbdad8cf455fa48aac21`.
Natif de tests :
`aa1f6223750a838a4346236e7ebd261d023c054da746475727ba02164df52d65`.
[`identity-v1.json`](measurements/flash-next-ssd-impl/opt-66-fixed-width-real/identity-v1.json)
identifie 154 sources, les artefacts et l'archive locale. Les kernels,
les banques SSD et la politique du cache sont inchangés depuis 65T.

- Rust chat : 124 passés/5 ignorés ; MLX chat : 147 passés/146 ignorés.
  fmt, Clippy strict et build MLX/chat passent. Les physiques sont sélectionnés
  explicitement et exécutés seuls.
- Python complet : 992 passés/4 sautés, 102,61 s. Les quatre absences connues
  concernent ponyexl3 et le modèle Ling. Les 18 tests de diagnostic 65P en
  attente sont inclus dans cette collecte, sans mesure 65P.
- Kani du parseur réel : 0..3 octets symboliques, UTF8 valide, booléen adaptatif,
  unwind5 ; 263 obligations, 0 échec, 8 obligations libcore UTF8/align/fmt
  inatteignables, 4 covers satisfaites. Ce bornage ne prouve pas MLX/Metal.
- Kani du vrai lecteur d'options MLX : ICE `intrinsics.rs:243` avant obligations,
  non vérifié. La première tentative sur un test ignoré n'avait sélectionné
  aucun harnais ; cet échec est conservé. CrossHair est absent.
- Douze processus sans checkpoint exercent le vrai lecteur d'options : valeurs
  1/3/7, défaut fixe3, valeurs invalides, Unicode et conflit avec l'adaptatif.
  Tous passent avec exactement un test sélectionné par processus.
- Le test physique runtime passe 30 cas en 404,660 s : cinq budgets
  1/2/3/17/64, configurations du draft existantes, nouvelles largeurs fixes
  1 et 7, comparaison des tokens et de tous les états à l'oracle séquentiel,
  annulation/reset. Pic physique 8 730 303 616 octets, aucun nouveau Swapout.
  Cette durée de validation n'est pas un débit.
- Le bridge réel forcé B2 passe 43 contrôles en 284,964 s ; ses 27 événements
  DFlash annoncent une seule proposition. Pic 8 151 816 648 octets. Le post30 s
  passe, sans nouveau Swapout sur l'ancre conservée.

Le protocole v2 corrige seulement le validateur Python : déficit maximal des
propositions finales `width*(width-1)/2`, prompts positifs laissant les 128
sorties dans contexte4096. Les nouveaux cas de queues multiples et de contexte
invalide sont couverts : 176 contrôles ciblés passent, puis 17 contrôles du
validateur final. Les deux binaires sont identiques à la qualité v1 ;
[`identity-v2.json`](measurements/flash-next-ssd-impl/opt-66-fixed-width-real/identity-v2.json)
identifie cette réutilisation et les deux seuls fichiers Python modifiés.

Commandes et logs :
[`opt-66-fixed-width-real`](measurements/flash-next-ssd-impl/opt-66-fixed-width-real).
Les preuves physiques inchangées 65T (experts, préfixes cible, MTP) restent
disponibles dans le rapport 65 ; aucune nouvelle preuve Metal n'est revendiquée.

## Campagne réelle prévue avant les mesures

M5 24 Gio, Qwen3.8-Flash-Next EXL3 2.05bpw `65c89531`, DFlash2 `a22a1744`,
draft Q4 et cache circulaire activés, cache experts 3 Gio/53 slots, quatre
lecteurs, allocateur 512 Mio, contexte4096, 63/Hot/Wide activés, UNIQUE=1,
secteur et applications ouvertes. Un seul modèle GPU à la fois.

Chaque processus effectue un warmup et une génération mesurée de 128 sorties,
soit 127 tokens décodés. Les tok/s sont `127 / decode_seconds` ; le coût du
draft, de la vérification, du commit et de l'orchestration est inclus. Les
temps de profil sont des durées hôte inclusives, pas des temps GPU exclusifs.
Texte et hash doivent égaler l'oracle OFF128 indépendant déjà conservé.

1. Quatre témoins frais OFF/MTP1, MAIN/HOLD, sans override de largeur.
2. Pour chaque B2/B4/B8 : MAIN puis HOLD, SHARED=0 puis1. Les trois largeurs
   sont testées même si elles régressent. Ce premier A/B ne qualifie pas un gain.
3. Si les deux prompts dépassent +1 % lors de l'écran, seconde paire C-H par
   prompt, prévue à l'avance. Le HCCH doit avoir chaque gain ≥1 %, dérive des
   témoins ≤5 %, étendue des gains ≤3 points. Tous les échantillons sont gardés.
4. Témoins OFF/MTP1 après, puis clôture de la comparaison des modes.

Témoins avant terminés, mêmes 127 tokens décodés et sorties exactes :

| Mode | MAIN tok/s | HOLD tok/s |
|---|---:|---:|
| OFF | 10,567761 | 9,832489 |
| MTP1 | 10,487449 | 10,101739 |

Quatre processus joints, pré20/post30 passés, aucun nouveau Swapout.

Les trois fenêtres sont terminées :

| Fenêtre | Partage EXL3 | MAIN tok/s | HOLD tok/s |
|---|---:|---:|---:|
| B2 | 0 | 11,427601 | 10,818333 |
| B2 | 1 | 10,942969 | 10,593349 |
| B4 | 0 | 9,277117 | 9,226583 |
| B4 | 1 | 8,883878 | 8,556508 |
| B8 | 0 | 6,770375 | 6,020686 |
| B8 | 1 | 5,841979 | 4,789241 |

Les trois écrans sont sous le seuil : pas de seconde paire C-H déclenchée,
aucun gain du partage qualifié. Un seul A/B ne prouve pas une régression stable.
H/C font exactement le même nombre de propositions, d'acceptations, de blocs,
de lignes vérifiées et de lectures d'experts pour chaque prompt et largeur.
La différence de débit H/C ne vient donc pas de ces volumes de travail.

| Sans partage | MAIN B2 | MAIN B4 | HOLD B2 | HOLD B4 |
|---|---:|---:|---:|---:|
| Acceptation des propositions | 84,06 % | 59,56 % | 80,00 % | 57,55 % |
| Lignes vérifiées/token décodé | 1,087 | 1,433 | 1,102 | 1,465 |
| Vérification ms/token émis dans les blocs | 82,04 | 101,56 | 86,79 | 102,15 |
| Vérification ms/ligne cible | 75,50 | 70,87 | 78,11 | 69,75 |
| Octets d'experts/token décodé, Mo décimaux | 188,43 | 227,16 | 200,59 | 232,38 |

Ces temps de vérification incluent routage, SSD et calcul ; ils ne localisent
pas à eux seuls un kernel lent. B4 réduit le coût par ligne tout en augmentant
le travail total par token utile. Les octets sont les demandes d'experts
comptabilisées par le moteur, sans assimilation à une mesure d'I/O physique.

B2/B4/B8 passent parité, pré20/post30, sources inchangées et zéro nouveau Swapout.
Pic maximal B2 : 8 203 622 832 octets ; B4 : 8 230 378 048 octets.
B8 atteint au maximum 8 307 825 384 octets. Sur MAIN sans partage, il vérifie
271 lignes pour 127 tokens décodés, avec 92 propositions acceptées sur236
(38,98 %) et 302,20 Mo de demandes d'experts par token décodé.

Témoins après terminés, toujours exacts :

| Mode | MAIN tok/s | HOLD tok/s | Dérive MAIN | Dérive HOLD |
|---|---:|---:|---:|---:|
| OFF | 9,747471 | 10,067484 | 8,42 % | 2,39 % |
| MTP1 | 10,880380 | 9,293503 | 3,75 % | 8,70 % |

Les dérives OFF MAIN et MTP HOLD dépassent 5 %. B2 semble favorable sur son
écran, mais cette campagne ne valide donc aucun avantage entre DFlash2 et MTP.
Les huit témoins et douze processus des trois fenêtres restent tous conservés :
40 générations warmup/mesurées exactes au total, sans retrait d'outlier.
Le HCCH dédié MTP/B2 (66M) a été préparé puis laissé non lancé à la demande de
l'utilisateur, qui donne priorité au push puis aux nouvelles pistes Splash.
Le partage reste expérimental, désactivé par défaut.

[`summary.json`](measurements/flash-next-ssd-impl/opt-66-fixed-width-real/summary.json)
contient les données et profils complets. Les archives source v1/v2 sont
maintenant copiées dans les mesures, avec tailles et SHA256 dans
[`published-sources.json`](measurements/flash-next-ssd-impl/opt-66-fixed-width-real/published-sources.json).

Le tout premier lancement B2 avait été refusé par argparse avant le modèle
(option CLI `--timeout` inexistante). Échec exit2, enfant joint, données conservées
dans B2. Commandes corrigées et vérifiées avec le vrai parseur sans inférence ;
les mesures réelles B2 sont dans B2-retry-cli. L'origine de Swapins du premier
lancement est conservée pour la reprise.

Pré20/post30 par phase, empreinte ≤12e9 octets, incoming ≤256 Mio par fenêtre
déclarée et aucun nouveau Swapout depuis l'ancre 121 528 483 840. Les erreurs,
interruptions, dispersions et non-lancements doivent rester explicites. Une
fenêtre coûteuse n'est pas tronquée pour augmenter son débit.

Le diagnostic 65P du coût CPU et des groupes homogènes 2/3/4 est en attente à la
demande de l'utilisateur. Ses modifications de tests sont conservées dans
des archives locales (`pending-sources.json`), sans remplacer le runtime testé
ici ; ce diagnostic non exécuté n’est pas une optimisation livrée. La revue
65R Strata/Splash a confirmé les mêmes HEADs ; elle reprendra après ces tests
réels. Aucune installation ni release de l'application n'est faite.

## CI et limites

Le parent publié5ab2c7f8 a ses jobs natifs Linux/Mac, régression Python et
contrats Flash Kani réussis. Son job Kani global a échoué avec l'étape de preuve
annulée ; la cause précise n'est pas établie et la récupération du log d'échec
n'a renvoyé aucun contenu. L'état exact est conservé dans ci-5ab-final.json.
Le nouveau commit doit avoir sa propre CI inspectée ; aucune CI complète verte
n'est déduite de ces succès partiels. L'ICE Kani de MLX et l'absence de preuve
Metal restent explicites. Application installée et réglages par défaut inchangés.
