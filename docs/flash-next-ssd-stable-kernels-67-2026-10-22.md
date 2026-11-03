# 67S — kernels EXL3 stables inspirés de Splash

Campagne commencée le21octobre et poursuivie le22octobre2026, après publication
47075104 de la campagne réelle66F. Objectif : tester un mécanisme distinct du
partage des poids65, dont aucun gain de génération n'a été qualifié.

## Recherche et adaptation

HEADs GitHub revérifiés : Splash1482f6bbc2abf12375332725302019594150b6cb et
Strata011c9e600a93bbc9fd451c563501b5831c32bbca, inchangés.

Dans [MoE.cpp de Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.cpp),
les dimensions viennent de paramètres de lancement et les grilles des descripteurs
de travail. Les mêmes buffers de travail servent aux passages du plan. Notre
chemin EXL3 de slots inscrit au contraire le nombre de routes dans
LOCAL_OUTPUT_DIMS et MLXL3_GROUP_OUTPUT_DIMS, créant des headers distincts.

Le prototype transmet ces deux strides via les formes1D des entrées tile_map et
tile_sub déjà disponibles. Il conserve les mêmes grilles, plans, banquesSSD,
protections, FMA, réductions, conversions FP16 et Hadamard. Aucun changement de
format GGUF/Table16, de code C++, d'ABI, de source Metal partagée ou de LFM2.
Ce n'est pas un graphe Metal persistant.

Point d'intégration vérifié dans le code MLX0.32.2 : la détection des paramètres
`*_shape` examine le corps source, pas le header. Le prototype injecte donc un
préambule dans ce seul corps EXL3. Les formes sont fournies au lancement ; le
[cache de bibliothèques MLX](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp)
utilise le hash de la source. Le choix constant/device de MLX sous8éléments peut
encore produire plusieurs signatures. La réduction des clés de notre factory
ne constitue donc pas le nombre exact de PSO compilés ni leur coût.

Option expérimentale stricte `MLXL3_SSD_STABLE_EXPERT_KERNELS=0|1`, défaut0.
SHARED reste0 pour toute mesure de génération67. Les sources consultées et leurs
SHA sont dans research-source.json ; les rapports57/58/63/64/65 et opti.md ont
été relus avant ce nouveau test.

## Vérification avant débit

- fmt, Clippy strict, build livré MLX/chat passent. Rustchat125passés/5ignorés,
  MLXchat148passés/146ignorés ; les tests physiques sont exécutés seuls ensuite.
- Kani : vrai calcul de stride avec trois u32 symboliques, aucun assume,
  oracle u128 indépendant, refus du zéro et des dépassements d'i32. Quatre
  couvertures satisfaites. Mutation omettant le facteur width : deux assertions
  détectent le défaut. Source restaurée byteexacte, preuve et test ciblé repassés.
- La tentative Kani du vrai chemin MLX échoue sur l'ICE intrinsics.rs:243,
  avant obligations. Ce chemin, Metal et la concurrence ne sont pas prouvés.
- GPU : 176formes GU/down, F16/F32 et vues stridées, groupes1..4/routes1..80,
  528masques froid/mixte/chaud ; valeurs finies, formes complètes et égalité
  bit à bit aux références résidentes indépendantes. Mauvais slots/origines
  détectés, entrées invalides rejetées, aucun nouveau readback.
- Les vrais triplets nom/header/corps sont comparés par égalité de chaînes :
  **64 clés distinctes à12**, pendant ces mêmes contrôles. C'est une propriété
  structurelle observée, pas un gain de tok/s ou une mesure de misses.
- 32cas MoE et36préfixes DFlash avec logits/états et captures indépendantes
  passent. Les dix paires MTP et la session MTP passent aussi, ainsi que les
  43 contrôles du vrai bridge, dont annulation/reprise. Les deux phases qualité
  ont fermé leur postflight30 sans nouveau swap-out. Pics respectifs :
  8 151 817 032 et 8 299 420 200 octets.

CLI figé f6d4ce4c94957f374d983778eec874529003b359a39624e2550ea5b9b46534b2 ;
natif a3da0c6009adfc984d5e240798c638793284b9310e0e4f700e6c051711e5b310.
154sources, deux binaires et archive identifiés dans identity-v1.json, état local
modifié depuis47075104. Aucun résultat d'un prototype retiré ne valide ce code.

Données : [opt-67-stable-expert-kernels](measurements/flash-next-ssd-impl/opt-67-stable-expert-kernels).

## Protocole de génération préenregistré

M5 24Gio, cible65c89531 EXL3 2.05bpw, drafta22a1744 Q4/ring, cache3Gio/53slots,
4lecteurs, allocateur512Mio, 63/HotWide/UNIQUE1, LES/lookahead0, contexte4096,
secteur et applications ouvertes. Un seul modèleGPU, aucune compilation ni
preuve pendant les mesures. Pré20/post30, pic physique≤12e9octets, incoming≤256Mio
sur une fenêtre explicite et aucun nouveau Swapout depuis121528483840.

OFF, MTP1 puis DFlash adaptatif7 : MAIN/HOLD,1warmup et1mesure de128sorties,
127tokens décodés par processus. H=flag0/C=flag1, autre configuration identique.
Un premier A/B par prompt pour chaque mode ; si les deux gains≥1%, seconde
paire C-H par prompt, prévue avant les résultats. HCCH : chaque gain≥1%, dérive
H≤5%, étendue≤3points. Tous points, erreurs et interruptions conservés. Les
largeurs choisies par l'adaptatif et les volumes de travail peuvent varier ;
ils seront publiés avec les profils inclusifs.

## Résultats et clôture

| OFF, 127 tokens décodés | H, option0 | C, option1 | Écran |
|---|---:|---:|---:|
| MAIN |10,548254 tok/s|10,402035 tok/s|−1,3862 %|
| HOLD |9,933862 tok/s|10,204738 tok/s|+2,7268 %|

Les quatre processus sont joints, les huit réponses complètes sont exactes,
le postflight30 passe sans nouveau swap-out. Pic : 7 143 692 192 octets.
L'écran MAIN ne satisfait pas le seuil ; la confirmation C-H n'est pas lancée.
Un A/B isolé ne prouve pas non plus une régression stable.

MTP : le témoin MAIN termine à11,071174 tok/s, pic8 199 263 512 octets.
Le candidat suivant est arrêté et joint après37,130s sur la garde swap.
La campagne est **échouée**, sans débit candidat qualifié : le compteur système
passe de121 528 483 840 à121 926 221 824 octets, soit **397 737 984 octets
de nouvelles sorties de swap**. L'activité simultanée de syspolicyd et trustd
est observée, mais ne démontre pas une cause exclusive. Aucun moteur de cette
campagne ne reste actif. Le postflight est échoué, pas validé pendant30s.
Les deux bras HOLD, DFlash2 adaptatif et la combinaison67C sont **non lancés**.
Il n'y a ni reprise silencieuse ni nouvelle baseline dans cette campagne.

Le rapport utilisateur a conduit à recouper seulement maintenant l'essai19,
qui avait déjà observé58 créations initiales, aucune éviction et environ0,40 %
du temps en lookup/construction de factories. Ce recoupement aurait dû précéder
67. Les nouvelles formes groupées justifiaient de compter les variantes, mais
leur réduction64→12 ne démontre pas un coût critique. Aucun gain de tok/s,
de RAM ou de swap n'est établi ; option expérimentale conservée à0.

Les quatre séries de relevés système sont compressées sans perte avec SHA
dans compressed-system-samples.json ; source-v1.tar.gz correspond exactement
aux154sources testées. Les échecs, rapports partiels et plans non lancés restent
dans les preuves. CI47075104 inspectée : régressionPython et nativeLinux/macOS
réussies ; Kani global échoué (étape annulée), KaniFlash annulé ; ne pas
qualifier ce workflowNative vert. Diagnostic ci-4707-final.json conservé.

## Autres écarts Splash conservés comme pistes

| Mécanisme | Ce qui manque ici | État |
|---|---|---|
| Contributions groupées écrites directement |65 concatène puis restaure les partiels ; l'encodeur MLX public étudié65D pourrait écrire les origines disjointes dans une sortie commune | API identifiée, prototype non lancé ; respecter allocations/dépendances et couverture totale |
| Scratchs et graphe persistants | Nous reconstruisons les graphes MLX et temporaires à chaque fenêtre | Non porté ; le simple cache de kernels67 ne remplace pas cette architecture |
| Commit GDN fusionné au scan suivant | Notre tape rejoue encore le préfixe conservé séparément | Non porté ; garder états observables, limites de contexte, annulation et reprise |
| GDN/HC/PLE/QSA groupés |63 groupe des projections, mais pas toute la préparation ni les opérations causales | Paliers possibles ; le mono-token25 négatif ne doit pas être relancé sans nouveau mécanisme |
| Sélecteur/draft | Splash regroupe davantage QKV et sélection | Non porté ; leurs calculs FP32 ne sont pas interchangeables avec notre sélection BF16 |

Les sources GDN et les écarts numériques sont détaillés dans la
[revue64](flash-next-ssd-splash-64-research-2026-10-21.md). Aucun gain de ces pistes
n'est déduit de la mesure67. Les15tok/s restent un objectif.
