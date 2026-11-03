# Réduire les dispatchs HC — recherches23 et essai24

Après le rejet du regroupement22, la source retenue reste20, commit23292112.
L’essai24 intègre SiLU(x/4) et2sigmoid(x/4) directement après les réductions
GEMV down et injection. Poids FP16, calcul FP32 et ordre des sommes conservés ;
aucune nouvelle copie persistante ni modification du cache experts.
La tête MTP FP16 conserve les primitives historiques.

## Source de la piste

[Sushi, HC](https://github.com/beamivalice/sushi/blob/2c602b9e41b3f0a9577f7cdb30a22dca5b2e6951/src/qwen4_hc.zig)
regroupe la lecture HC en trois kernels : normalisation/injection, down/activation,
up/mélange. Le format affine8bits et l’ordre des sommes sont différents du
checkpoint local ; ce code n’est pas copié. Seule l’idée de fusion d’épilogue
est reprise dans notre GEMV exact. Les réductions et chargements locaux restent
ceux adaptés de MLX0.32.2. La formule sigmoid doit aussi suivre MLX :
`y=1/(1+exp(abs(x)))`, puis `x<0 ? y : 1-y`.

[Les résultats amont](https://github.com/beamivalice/sushi/blob/2c602b9e41b3f0a9577f7cdb30a22dca5b2e6951/docs/qwen4-perf.md)
ne sont pas des mesures sur ce M5 ; modèles, mémoire et matériel diffèrent.
[Commits](measurements/flash-next-ssd-impl/opt-23-upstream/heads.json) et
[empreintes des fichiers consultés](measurements/flash-next-ssd-impl/opt-23-upstream/sources.json)
sont conservés. Les pistes SSD13 sont déjà documentées : quatre lecteurs,
slots stables et LFU vieillissant restent actifs. Aucune répétition de cacheLRU,
préchargement aveugle, étatsBF16 ou suppression d’experts.

## Protocole24

Avant vitesse : comparer tous les GEMV et leurs épilogues à la chaîne MLX,
avec entrées réelles/synthétiques, strides, zéros, valeurs extrêmes finies,
biais, formes et FP16. Tester une mutation de formule, puis les logits/états
et le vrai bridge. Tentatives Kani/CBMC sur les sources effectives.

Si exact : deux tours H-C puis C-H, MAIN/HOLD×OFF/MTP sur batterie vérifiée,
une chauffe128 puis trois générations128 (127tokens décodés) par processus.
Mêmes M5/24Gio, modèle65c89531 EXL3 2,05bpw, MLX0.32.2, contexte4096,
cache5Gio/89slots, allocator512Mio, readers4 etoverlap1. Baseline20 immuable
7a9a2120, nouvelles mesures. Tous textes/hashes128 et préfixes64 exacts.

Chaque paire doit gagner≥1%, étendue≤3points par cellule, surplus du pic
processus≤256Mio, swapouts système≤256Mio/bras ; ratio débit max/min≤1,25.
Arrêt après le premier tour si un seuil échoue, immédiatement sur erreur,
divergence ou dérive forte. Un seul modèleGPU, aucune compilation/proveur
pendant la vitesse, applications conservées. Délai360s et enfants joints.

**Rejeté : aucun gain24 validé. Prototype retiré,20 conservé.**

## Vérification locale avant débit

Deux tests de primitives passent :298matrices,1 056 392sorties GEMV finies
bitexactes, puis2 112 784sorties d’épilogues identiques à la chaîneMLX
indépendante. Le même contrôle couvre les entrées synthétiques/dyadiques,
strides, zéros, valeurs extrêmes, biais, autres formes et FP16. Une mutation
du facteur1/4 en1/2 est détectée ; shader restauré et test repassé, binaire
de test revenu exactement au SHAinitial.

`cargo fmt --check`, Clippy strict `--all-targets --features mlx,chat`,
build release `--features mlx,chat`, suite CPU87passés/2ignorés et
suite release107passés/98ignorés réussissent. Les11testsQwen physiques
sont exécutés chacun dans un processus, puis l’oracle held et3testsSSD :
tous passent sans swapout système supplémentaire. Cette isolation évite
de conserver les caches de plusieurs tests dans un même processus ; elle
ne modifie ni le moteur ni le protocole de vitesse. Le bridge passe aussi : budgets1/2/3/17, refus et annulations/reprises ;95 354 880octets de swapouts supplémentaires pendant ce contrôle, sous la borne256Mio.

Kani sur le vrai chemin, deux i8symboliques indépendants sansassume :
erreur interne avant obligations. CBMC ne traite pas le MSL ; CrossHair
absent pour les gardesPython (14cas frontières passent). Pas de preuveGPU.
[Commandes et résultats](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/checks.json),
[mutation](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/mutation.json),
[preuve et limites](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/proof-inspection.json),
[identités](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/identity.json).

Candidat24 moteur86d58932, testscee96230 ; source23292112 plus
[patch archivé](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/source.patch).

## Résultat24 sur batterie

| Cas | H20, tok/s | C24, tok/s | Écart |
|---|---:|---:|---:|
|MAIN OFF|9,431|9,395|−0,38%|
|HOLD OFF|9,223|9,017|−2,23%|
|MAIN MTP|9,799|10,221|+4,31%|
|HOLD MTP|9,398|9,391|−0,08%|

Les critères du premier tour échouent ; huit bras terminés/32réponses
exactes, huit bras suivants non lancés. Aucun swapout supplémentaire
sur les huit bras ; plus grand surplus du pic candidat8 372 152octets.
Le résultat MTPMAIN seul ne valide pas cette fusion. Code retiré, preuves
conservées dans [assessment.json](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/assessment.json)
et [speed/result.json](measurements/flash-next-ssd-impl/opt-24-hc-epilogues/speed/result.json).
