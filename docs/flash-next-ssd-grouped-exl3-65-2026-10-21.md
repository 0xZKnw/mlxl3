# 65 — partage EXL3 des experts répétés, Flash Next SSD

État au 21 octobre 2026 : prototype expérimental, `MLXL3_SSD_SHARED_EXL3=0`
par défaut. Aucun gain de génération n'est encore qualifié. L'application
installée n'a pas été remplacée. Les résultats négatifs 64U/64H/64L restent
dans les rapports précédents.

## Mécanisme et périmètre

Une contribution CPU connaît déjà son slot expert. Le plan regroupe ses
occurrences réelles par paquets de quatre au maximum. Un kernel par largeur
présente (1, 2, 3 ou 4) charge et décode une tuile EXL3 K2 MUL1 une fois,
puis l'applique à autant d'activations avec des accumulateurs indépendants.
Les partiels compacts sont concaténés et restaurés dans l'ordre original
avant la réduction habituelle. Plan, métadonnées et restauration font partie
des chronométrages.

Les Hadamard, échelles FP16, ordre FMA, quatre SIMD groups et réduction de
référence sont conservés. Aucune deuxième ligne fictive, déquantification
résidente, lecture CPU de routes supplémentaire, primitive C++ ou allocation
de banque expert supplémentaire. Les slots ont la même durée de protection.

GPU Hot anticipé conserve son chemin GPU et son snapshot. Ce prototype ne
regroupe que les contributions dont les slots sont déjà connus du CPU,
notamment Cold. Il ne faut donc pas annoncer le regroupement de tout le MoE,
ni attribuer la totalité de la vérification à ce kernel.

Points d'entrée : `SsdMoe::calculate` → `ExpertSlots::grouped` →
`expert_mapped_layout` → `_qmv_mapped_tile_kernel.metal`. Le chemin masqué Hot
et les modèles autres que Flash SSD conservent leur calcul précédent.

## Comparaison avec Strata et Splash

- [Strata, moe.metal, 011c9e60](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/kernels/moe.metal)
  partage le décodage hors boucle des occurrences, avec spécialisations de
  largeur et découpage en paquets. Ce principe motive les groupes réels 1..4.
- [Splash, MoE.cpp, 1482f6bb](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.cpp)
  et ses kernels MoE choisissent le calcul selon la taille des groupes et
  réutilisent leur infrastructure de dispatch et de buffers. Nos formats EXL3,
  réductions et orchestration MLX diffèrent : leurs pourcentages ne sont pas
  transposés.
- Le prototype 58 ne couvrait que B2 et son plan GPU à 32 lanes n'était pas
  extensible tel quel aux 80 occurrences de B8. Le compact 57 seul avait
  régressé. Aucun des deux n'est repris comme une optimisation déjà qualifiée.

Les cinq fichiers effectivement lus, révisions et SHA-256 figurent dans
[`upstream-read-set.json`](measurements/flash-next-ssd-impl/opt-65-grouped-exl3/upstream-read-set.json).
La revue n'est pas un audit exhaustif des deux dépôts.

## 65G v1 : qualité exacte, crible global non concluant

Artefacts figés sur base `5443cd05`, sources locales modifiées :
CLI `dc74319faf7e6aaa2ff0a1dea18ebc5e670b1c7ed38b93005036030a4e60ca5d`,
natif `c5ad4dbf54c87eb408c9518ec9154cf87a81b60136a74cf57a6067232eb9093a`.
[`identity-v1.json`](measurements/flash-next-ssd-impl/opt-65-grouped-exl3/identity-v1.json)
identifie 147 sources, les binaires et l'archive locale.

Qualité : 88 formes GU/down F16/F32 avec strides et 1..80 routes, 264 cas
masqués, 8 contrôles de mauvaise origine ; 32 cas MoE/repli/pending/erreur I/O,
36 préfixes DFlash avec 248 320 logits et tous les états, 43 contrôles du
bridge (annulation/reprise comprises), Session MTP budgets 1/2/3/17 et
10 commits de paires. Comparaisons bit à bit, finitude et tailles contrôlées.
Le bridge initial s'est arrêté après 36/43 au budget global 330 s ; sa reprise
600 s passe 43/43 en 327,222 s. Le premier échec reste conservé.
Pic du bridge complet : 8 382 454 432 octets, zéro nouveau Swapout et post30 s
passé. Cette durée de validation n'est pas une mesure de tok/s.

Crible : couches 0/23/47 × B1/B2/B4/B8 × contributions entières/mixtes,
24 cellules, 4 paires alternées, warmup ≥1 s par cellule, 128 forwards par
échantillon, plan/restauration inclus. B1/B2 proviennent de deux positions
capturées réelles ; B4/B8 répètent ces deux positions, configuration favorable
aux répétitions, et ne représentent pas les fenêtres adaptatives naturelles.
Aucune lecture SSD dans le temps mesuré.

| Formes | Gain composant médian observé | Limite |
|---|---:|---|
| B1 | −0,41 à +0,62 % | témoins, plusieurs dispersions >3 points |
| B2 plein, couches 0/47 | −2,15 / −1,61 % | régressions stables |
| B2 plein, couche 23 | +3,84 % | davantage de répétitions |
| B2 mixte | −0,73 à +0,70 % | deux cellules sans répétition, une très instable |
| B4 | +21,33 à +34,21 % | plusieurs dispersions excessives |
| B8 | +50,22 à +66,69 % | composant favorable, pas des tok/s |

Les 192 échantillons sont conservés sans filtre dans
[`micro-v1`](measurements/flash-next-ssd-impl/opt-65-grouped-exl3/micro-v1).
Verdict préenregistré : `inconclusive_drift`, `full_model_eligible=false`,
`throughput_gain_confirmed=false`. Aucun benchmark complet v1 n'a été lancé.
Processus joint en 47,154 s, pic composant 619 545 536 octets, zéro nouveau
Swapout/post30 s passé. Le qualificateur v1 est archivé séparément ; sa
compatibilité avec le qualificateur actuel a été vérifiée hors ligne.

## 65T v2 : sélection avant calcul groupé

Calibration uniquement sur les comptes du crible v1 : les B2 négatifs
économisent 1 décodage sur 20 (5 %), celui positif en économise 4 (20 %).
Le plus petit B4 mixte positif possède 16 routes. Le seuil fixé avant
modification est donc **au moins 16 routes CPU et 20 % de décodages économisés**
après découpage en paquets de quatre. Sinon NT1, sans upload du plan groupé.
MAIN/HOLD de génération restent hors de cette calibration.

Le compteur n'alloue pas de table de métadonnées. Le retour immédiat sous
16 routes évite aussi de le parcourir. Aucun seuil n'a été balayé en fonction
des nouveaux résultats. Le test physique attend 56 formes groupées et
32 fallbacks, plus 264 cas masqués et 8 mutations de mauvaise origine sur
20 routes réellement éligibles.

Le nouveau crible garde les 24 cellules et 4 paires mais passe à 1 024 forwards
par échantillon : les fenêtres de 64–120 ms et les outliers A/A de v1 étaient
trop courts. Les bornes restent dérive ≤5 %, étendue des gains ≤3 points,
chaque gain ≥1 % si le groupement est activé, ≥−1 % sinon. Le booléen réellement
activé est enregistré et contre-vérifié par le qualificateur.

Artefacts : CLI `56041ac6f97a98056d5194eb8917231a715b6c2fb2c0d845320d0e4f9830eda3`,
natif `42483c79f3ae76018a47ae3295cccc82c9e6099125cfa6acb765bd6798a1955f`.
[`identity-v2.json`](measurements/flash-next-ssd-impl/opt-65-grouped-exl3/identity-v2.json)
couvre 151 sources, dont les scripts de protocole modifiés, et les deux
binaires. Qualité physique v2 terminée : 88 formes (56 groupées, 32 fallbacks),
264 masquées, 8 mauvaises origines détectées, 32 cas MoE, 36 préfixes/logits/états,
43 contrôles bridge et les deux tests MTP passent. Tous les enfants sont joints,
les trois post30 s passent ; pic du bridge 8 350 014 112 octets et aucun nouveau
Swapout depuis l'ancre conservée.

Crible v2 terminé : 196 608 forwards exacts, processus joint en 199,207 s,
pic composant 624 804 800 octets, post30 s passé et aucun nouveau Swapout.
Toutes les cellules qui activent le groupement passent le seuil de gain
apparié : B2 plein couche23 +3,082 % médian, B4 +21,373 à +33,233 %,
B8 +49,221 à +66,536 %. Les deux B2 pleins défavorables de v1 reviennent à
NT1 (+0,152/−0,024 % médians).

Le verdict global reste **non concluant**, `full_model_eligible=false` :
B2 plein0 dérive candidate 6,439 %, B2 mixte0 témoin 16,102 %, B4 mixte23
dérives 11,667/7,112 %, B1 plein47 candidate 45,506 %. B1 plein0 est stable
mais un gain apparié est sous −1 % (médiane −0,443 %). Aucune de ces observations
n'est retirée. Les données et chaque paire sont dans
[`micro-v2`](measurements/flash-next-ssd-impl/opt-65-grouped-exl3/micro-v2).
La condition préenregistrée n'étant pas remplie, les plans macro128 préparés
ne sont pas lancés. Aucun gain de tok/s ni aucune promotion n'est annoncé.

Les archives source des deux versions sont publiées avec
[`published-sources.json`](measurements/flash-next-ssd-impl/opt-65-grouped-exl3/published-sources.json),
pour que les résultats v1 restent vérifiables quand le code évolue.

## Vérification locale et limites

Commandes avec `CARGO_TARGET_DIR` du projet principal et MLX 0.32.2 :

```text
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --features chat
cargo test --locked --release --features mlx,chat
cargo build --locked --release --features mlx,chat
python -m pytest -q
python < docs/measurements/flash-next-ssd-impl/opt-65-grouped-exl3/proof-v2-command.txt
```

v2 : Rust chat 123 passés/5 ignorés, MLX chat 146 passés/145 ignorés ; les tests
physiques sont sélectionnés séparément. Ruff passe. Python ciblé 120 passés ;
suite complète seule 963 passés/4 sautés (ponyexl3 et modèle Ling absents).
La première suite complète pendant compilation a eu 21 erreurs système
`EAGAIN35`, toutes conservées ; la cause système précise n'est pas établie.

Kani : gate de production sur tout `usize`, 68 obligations réussies et
3 covers satisfaites. Compteur de chunks sur 0..8 IDs u32 symboliques,
unwind9 : 453 obligations, 0 échec, 1 assertion libcore/fmt inatteignable,
3 covers satisfaites. Les deux tentatives initiales sur les Vec du plan
8/80 ont atteint 180 s : elles restent **non vérifiées**, domaine non réduit.
L'entrée MLX a rencontré ICE `intrinsics.rs:243` avant obligations.
CrossHair est absent. Ces contrôles ne prouvent pas Metal, MLX, les durées
de vie concurrentes ni la génération complète.

La CI ajoute les tests Python du qualificateur et du timeout, ainsi que les
deux harnesses Kani CPU réussis. Les runs du futur commit doivent encore être
inspectés. Sur `5443cd05`, Linux/macOS/Regression passaient, Kani global
échouait avec une étape annulée et Flash Kani était annulé : pas de CI globale
verte annoncée.

## Suite autorisée

Si le crible v2 passe, comparer sur le même binaire OFF/MTP1/DFlash2 adaptatif
7, MAIN et HOLD, 128 sorties/127 tokens décodés, HCCH et témoins avant/après.
Les témoins OFF/MTP1 utilisent SHARED=0 ; le HCCH DFlash2 compare SHARED=0/1.
UNIQUE=1 est fixe. Cache expert 3 Gio/53 slots, 4 lecteurs,
63/Hot/Wide activés, Q4/ring, allocateur 512 Mio, contexte 4096, secteur et
applications ouvertes. Pré20/post30, empreinte ≤12e9 octets, incoming ≤256 Mio
par fenêtre déclarée, zéro nouveau Swapout depuis l'ancre 121 528 483 840.

Après cette expérience, isoler le coût CPU du plan et les groupes de 2/3/4
occurrences demandés dans la capture utilisateur. La revue 65D a identifié
l'encodeur MLX existant comme piste pour écrire les origines disjointes dans
un seul output, sans clear ni nouvel encodeur brut ; c'est une hypothèse non
mesurée, pas un gain. Toute nouvelle variante exige un préenregistrement.
