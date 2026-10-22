# 68 — shared expert multi-token et garde swap

Base publiée : f8c9202ed37015c7adf3d787e5471f654f064cea, PR34.
État : prototype expérimental, campagne terminée, qualité physique passée. Aucun nouveau
gain de génération qualifié, aucune application installée ou publiée.

## Choix après le rapport utilisateur

Le [MoE de Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.cpp)
traite aussi le shared expert dans ses groupes, dispose d'un passage gate/up
commun et calcule son facteur scalaire dans la sélection. Les
[fenêtres de Strata](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/window.cpp)
conservent les dispatchs et leurs coupures SSD par taille. Ces mécanismes
ne sont pas entièrement portés par cette expérience.

La piste retenue ici est plus étroite : SsdMoe charge déjà les projections
EXL3 avec les kernels exacts multi-token63, mais sa boucle shared ne leur
transmet qu'une ligne. L'option stricte `MLXL3_SSD_SHARED_BATCH=0|1`, défaut0,
envoie une matrice aux trois projections gate/up/down quand elles sont EXL3
et que la fenêtre contient2..8lignes. Les conversions FP16 et les opérations
élémentaires restent identiques. Le multiplicateur dense reste ligne par
ligne pour garder l'ordre GEMV, ainsi que l'ordre de la somme finale.
T1 et les autres formats gardent leur chemin séquentiel. Sans63, le repli
interne EXL3 reste séquentiel et exact.

Le shared B2 existait déjà dans le prototype15, avec plusieurs autres
changements simultanés. Son rejet15B est conservé :68 isole ce seul levier
sur le moteur63/64, avec GPUHot et cache3Gio, et étend le contrôle àB3..8.
Ce n'est ni un nouveau format de poids ni une nouvelle déquantification.
LFM2 et les shaders communs ne sont pas modifiés.

## Mémoire et conditions

L'échec67,397737984octets de nouvelles sorties de swap système, reste
publié. Le Mac a redémarré le22octobre14:47:51Europe/Paris ; les compteurs
étaient revenus à0. Une première création d'origine batterie est refusée
car le chargeur vient d'être rebranché. Le lanceur suivant refuse l'origine
absente ; aucun enfantGPU ne démarre. Les erreurs et statuts non lancés sont
conservés dans origin-precondition-failure.json et origin-attempt-result.json.

L'origine acceptée68 est sur secteur98%, macOS27.2/26B5112a, M5/24Gio,
Swapins0/Swapouts0. Les applications utilisateur restent ouvertes. Toutes
les phases exigent la même alimentation, pré20/post30sans nouveau swap-out,
entrées≤256Mio depuis l'origine explicite, empreinte physique≤12e9octets,
un modèleGPU et aucun compilateur/proveur concurrent.

La garde cumulative de scripts/profile_flash.py refusait auparavant les
sorties uniquement au contrôle final, avec une tolérance live256Mio héritée.
Elle les refuse maintenant dès un échantillon positif et avant le lancement
suivant. Le superviseur garde le nettoyage des enfants et la restauration de
la photographie parbras. Le polling2s ne peut empêcher un burst entre sondes.

## Vérification

- Quatre contre-exemples de garde échouent sur l'ancien code.85tests ciblés
  passent après correction, dont un vrai enfant endormi interrompu et joint
  après un octet simulé de sortie, et le sampler rejoint.
- SuitePython1005passés/4ignorés ; elle inclut18tests65P locaux non publiés.
  Ponyexl3 et le modèle Ling manquent pour les cas ignorés.
  La variante destinée à la publication, excluant ce seul fichier65P, a été
  vérifiée séparément après fermeture des modèles :987passés/4ignorés,
  104,70s, commande `python -m pytest -q --ignore=tests/test_flash_grouped_diagnostics.py`.
- Rustchat125passés/5ignorés ; MLXchat148passés/148ignorés, contrôles physiques
  exécutés séparément. Fmt/Ruff/Clippy strict et build livré passent.
- CrossHair est tenté mais absent. Kani sur le vrai chemin MLX échoue
  en5,674s sur l'ICE intrinsics.rs:243 avant obligations, enfantjoint,
  pic676855808octets. Aucune preuve formelle Python/MLX/Metal revendiquée.
- ComposantGPU :96cas par exécution, trois couches0/23/47, B1..8,
  FP16/FP32 et strides,288comparaisons d'intermédiaires,15formes/dtypes
  invalides, mutationgate/up détectée/restaurée. Deux exécutions distinctes
  avec63activé puis désactivé passent. Sorties bit à bit, aucun readback
  ajouté, pic625656792octets, post30passé et0nouveauSwapout.
-32commandes de benchmark passent le vrai parserCLI, inférence interceptée ;
  l'ancienne optiontimeout invalide est refusée. Ce contrôle n'est pas un
  benchmark ni un oracle du modèle.
- Six valeurs invalides de la nouvelle option sont aussi rejetées par la
  vraie CLI au chargement, avant `ready`, dont une valeur non Unicode :
  `invalid-control-v2/checks.json`, pic617038760octets, enfants joints et0swap.
  Le premier lanceur attendait à tortexit1 au lieu du contratexit2 ; son
  échec et ses cinq cas non lancés restent dans `invalid-control`.
- Modèle :32casMoE,36commits de préfixes avec248320logits,64000captures
  indépendantes et tous états exacts ; sessionsMTP et10paires exactes.
  Quatre enfants joints, pic8 160 647 984octets, post30sans nouveau swap-out.
- Bridge réel :43contrôles passent en304,914s, refus/configuration,
  annulation/reprise, enfant et readerjoints. Pic8 301 517 520octets,
  pré20/post30passés,0nouveauSwapin/Swapoutdepuisorigine0.

155sources et deux binaires gelés dans identity-v1.json. CLI8607d08e…,
natif10ddc2da… ; archive source-v1.tar.gz avec SHA complet. Les contrôles
MoE/préfixes/MTP et bridge43 ont fermé leurs phases avant la première mesure
de vitesse. La parité est validée par tests physiques, pas prouvée formellement.

## Protocole de vitesse préenregistré

Après qualité : micro shared B2/B4 séparé, trois couches,8chauffes,
5paires alternées×96appels.12traces fenced hors chronos conservent les
intervalles de complétion hôte+attenteGPU. Les parents recouvrent leurs
enfants : ne pas les additionner, ni les appeler tempsGPU exclusif.

OFF contrôle, MTP1, DFlash2 B2fixe puis adaptatif7, MAIN/HOLD,
128sorties/127tokens décodés,1chauffe+1mesure parprocessus. H=SHARED_BATCH0,
C=1 ; SHARED_EXL3 et STABLE_EXPERT_KERNELS restent0,63/HotWide/UNIQUE1,
cache3Gio53slots,4lecteurs, alloc512Mio, contexte4096, Q4/ring.
Quatre bras d'écran parmode ; si chacun des deux prompts gagne≥1%, C-H
supplémentaire. QualificationHCCH : chaque gain≥1%, dériveH≤5%, étendue
≤3points, picC≤H+256Mio. Une seule paire favorable ne qualifie pas un gain.

Les anciennes valeurs66secteur et cette campagne ne sont pas mélangées.
L'objectif15tok/s reste non atteint. Les pistes Hot+prélecture, planGPU
indirect et fenêtres persistantes ne sont pas présentées comme livrées.

## Résultats du composant isolé

Les vingt mesures sont terminées, avec sorties exactes, pic physique de
625509288 octets et zéro nouveau swap-out, contrôle différé de trente
secondes compris. Chaque échantillon contient 96 appels sur les trois couches.

| Lignes | Variations de cadence par paire | Médiane des variations appariées |
| --- | --- | --- |
| B2 | −6,57 / +16,69 / +44,88 / +15,29 / +9,99 % | +15,29 % |
| B4 | +33,51 / +29,41 / +29,81 / +34,50 / +29,53 % | +29,81 % |

Ce sont des durées murales de composant, incluant construction hôte,
soumission et attente. Elles ne qualifient ni un gain de génération ni un
temps GPU exclusif. B2 présente une variabilité importante. Les douze traces
avec barrières sont enregistrées séparément : leurs intervalles parent/enfant
se recouvrent et ne s'ajoutent pas aux chronos natifs. Le nombre de marqueurs
de complétion de la projection gate passe de 2 à 1 pour B2, de 4 à 1 pour B4 ;
le nombre réel de dispatchs Metal reste non mesuré.

Voir les [paires brutes et leur résumé](measurements/flash-next-ssd-impl/opt-68-shared-expert/micro/summary.json),
les [intervalles bruts](measurements/flash-next-ssd-impl/opt-68-shared-expert/micro/fenced-events.csv)
et leur [décomposition](measurements/flash-next-ssd-impl/opt-68-shared-expert/micro/fenced-breakdown.json).

## Génération réelle — campagne terminée

| Mode | Prompt | Shared séquentiel, tok/s | Shared groupé, tok/s | Écart d'écran |
| --- | --- | ---: | ---: | ---: |
| OFF | MAIN | 9,693746 | 10,068810 | +3,8691 % |
| OFF | HOLD | 9,930709 | 8,819980 | −11,1848 % |
| MTP1 | MAIN | 11,096802 | 11,124272 | +0,2475 % |
| MTP1 | HOLD | 10,031994 | 10,123211 | +0,9093 % |
| DFlash2 B2 | MAIN | 11,205302 | 11,191089 | −0,1268 % |
| DFlash2 B2 | HOLD | 10,017284 | 10,012462 | −0,0481 % |
| DFlash2 adaptatif | MAIN | 10,174639 | 9,988597 | −1,8285 % |
| DFlash2 adaptatif | HOLD | 9,350320 | 9,237501 | −1,2066 % |

Ces écrans sont clos, préflight et postflight compris, avec huit réponses
exactes par mode. Aucun ne déclenche la confirmation C-H prévue : OFF donne
des résultats contradictoires, et les deux écarts MTP restent sous 1 %.
Le regroupement est inactif en T1 ; l'écart OFF ne permet donc pas d'attribuer
une cause exclusive au calcul groupé. Une seule paire ne démontre pas non
plus une régression stable. Les pics sont respectivement de 7144216600 et
8203146544 octets, sans nouveau swap-out. Les compteurs MTP restent identiques
entre variantes : MAIN 54 acceptations sur 72 propositions, HOLD 48 sur 78,
aucun bloc lookup. La voie réelle MTP regroupe bien deux lignes avant le MoE.

DFlash2 B2 fixe est aussi clos : huit réponses exactes, préflight/postflight
passés, pic de 8202902032 octets et zéro nouveau swap-out. Les deux écarts
restent sous le seuil de confirmation. H et C exécutent le même travail :
MAIN 58 acceptations sur 69 propositions, 69 blocs et 138 lignes vérifiées ;
HOLD 56 sur 70, 70 blocs et 140 lignes. Les octets d'experts demandés restent
respectivement de 23930630100 et 25474420944, identiques entre variantes.
Il ne s'agit pas d'une mesure des octets physiques du contrôleur SSD.

La vérification par position émise, avec le dénominateur `accepted+blocks`,
passe de 83,520 à 83,795 ms sur MAIN, de 93,529 à 93,507 ms sur HOLD. Ce
compteur vaut 126 sur HOLD contre 127 tokens décodés, le premier token ne
venant pas d'un bloc. La vérification par ligne vaut respectivement
76,862→77,116 ms et 84,176→84,156 ms. Ces durées restent inclusives du calcul
cible, du routage et du streaming ; elles ne s'additionnent pas aux sous-phases.

Sur ces passages frais, le témoin B2 dépasse MTP1 d'environ 0,98 % sur MAIN
mais reste inférieur de 0,15 % sur HOLD. L'ordre n'est pas une comparaison
alternée dédiée entre modes : aucun avantage global DFlash2>MTP1 n'est
qualifié.

L'adaptatif est aussi clos, avec huit réponses exactes, contrôles pré/post
passés et pic de8254347816octets. MAIN conserve78acceptations,90propositions,
49blocs et139lignes dans les deux variantes. HOLD passe de71/86/55 à72/87/54,
avec141lignes dans les deux cas. Les décisions temporelles peuvent changer
les fenêtres même lorsque les sorties restent exactes. MAIN demande
23166846756→23034557484octets experts, HOLD24852910968→24931535724.
Ces différences interdisent d'attribuer tout l'écart à un seul kernel.

Les seize bras d'écran produisent au total32réponses exactes de128tokens,
chauffes comprises. Aucun mode n'est éligible aux bras C-H conditionnels,
qui ne sont donc pas lancés. Aucun nouveau gain en tok/s n'est validé ;
`MLXL3_SSD_SHARED_BATCH` reste désactivé par défaut. Les résultats négatifs
et la variabilité sont conservés. Ni les15tok/s ni une victoire globale
DFlash2>MTP1 ne sont atteints dans cette campagne.

La suite des recherches est dans les revues
[69 — prélecture/Hot](flash-next-ssd-hot-prefetch-review-69-2026-10-22.md)
et [70 — autres écarts Splash/Strata](flash-next-ssd-splash-strata-followup-70-2026-10-22.md).
Elles ne constituent pas des optimisations livrées.

## Commandes et limites de vérification

Les matrices complètes, environnements et deadlines sont conservés dans
chaque `run.json`, avec les lanceurs `quality-command.txt` et
`speed-command.txt`. Les suites utilisent les commandes suivantes, avec
Rust1.98.1, MLX0.32.2 et Kani0.68.0/CBMC6.11.0 :

```text
cargo fmt --check
cargo clippy --all-targets --features mlx,chat -- -D warnings
cargo test --features chat
cargo test --release --features mlx,chat
cargo build --release --features mlx,chat
python -m ruff format --check scripts/profile_flash.py tests/test_flash_profile.py
python -m ruff check scripts/profile_flash.py tests/test_flash_profile.py
python -m pytest -q --ignore=tests/test_flash_grouped_diagnostics.py
cargo kani --lib --features mlx,chat --tests --harness actual_shared_batches_preserve_serial_outputs
python -m crosshair check scripts/profile_flash.py --analysis_kind=asserts
```

Les deux dernières tentatives restent non vérifiées : ICE Kani avant
obligations et CrossHair absent. La tentative MLX appelle l'implémentation
réelle, sans modèle formel de Metal ou de la concurrence. Les cas physiques
finis, même avec mutations détectées, ne sont pas une preuve complète.
Les tests ignorés dans les suites génériques ne sont pas déclarés exécutés
du seul fait que certains contrôles GPU ciblés passent séparément.

Les workflows PR/push existants couvrent déjà le script modifié, ses tests,
Rust CPU et la compilation/lint MLX. Le parentf8c9202e a une régression verte
et deux jobs native verts, mais pas une CI globale verte : Kani a été
interrompu sur un signal d'arrêt, et FlashKani annulé à sa limite de temps.
Les résultats du nouveau commit sont à inspecter après publication.

Preuves : [opt-68-shared-expert](measurements/flash-next-ssd-impl/opt-68-shared-expert).
Le [résumé structuré](measurements/flash-next-ssd-impl/opt-68-shared-expert/summary.json)
conserve les16points, compteurs et SHA des rapports ; les dix séries système
sont compressées sans perte, avec SHA et aller-retour vérifié dans
[l'index](measurements/flash-next-ssd-impl/opt-68-shared-expert/compressed-system-samples.json).
