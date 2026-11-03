# Préparation GDN exacte pour Flash — essai25

**Rejeté : gain insuffisant selon le protocole préenregistré.** Prototype retiré ;
le moteur conserve le gain20 publié dans la PR34. Hypothèse testée : remplacer la chaîne de
préparation GDN de Flash par un dispatch produisant convolution, Q/K/V et
historique. Les projections et la récurrence FP32 restent inchangées.

Le kernel local `gated_delta::prepare_qkv` dePERF109 couvre déjà cette fusion
pour les autres Qwen. Il utilise toutefois RMSFP32 ; Flash utilise une somme
de carrés FP16 puis une normalisationL2 avec epsilonFP16. Une réutilisation
directe changerait les calculs. Le prototype adapte la structure aux formes
Flash fixes : une ligne,10240canaux, quatre coefficients convolutionnels,
16têtesQ/K et48têtesV de128dimensions.

Le kernel produit aussi la convolution entière pour que les tests puissent
comparer chaque composant, puis toutes les valeurs des états et logits.
Aucune augmentation du cache experts ou copie persistante de poids.
Les principes de fusion de [Sushi](https://github.com/beamivalice/sushi/blob/2c602b9e41b3f0a9577f7cdb30a22dca5b2e6951/docs/qwen4-perf.md#gdn-decode-recur)
ont été examinés ; ses chiffres surM5Max ne sont pas des résultats locaux.
Le prototype n’intègre pas la récurrence dans ce kernel.

Protocole avant vitesse : huit cas déterministes et409600sorties attendues,
poids réels, strides, historique initial/nonvide, zéros, activations faibles
et grandes, métadonnées invalides. Chaque sortie doit être nonvide, de forme
etdtype exacts, finie, bit à bit identique à la chaîneMLX originale.
Mutation de l’historique, restauration puis recontrôle. Ensuite oracles
modèle indépendants, transactionsMTP, vrai bridge et contrôles de build/preuve.
Le débit est évalué seulement après ces contrôles, sur la baseline retenue.

## Premiers contrôles et corrections

Clippy strict et compilation releaseMLX/chat passent. Le premier oracle
refuse Q : l’accumulation des carrés étaitFP32. La première correctionF16
ne suffit pas, car le compilateur peut contracter le carré avec l’addition
et réassocier les produits normalisés. Les deux échecs restent archivés.

La version corrigée impose les arrondisF16 intermédiaires du carré et du
produit normalisé, avant l’opération suivante, puis utilise la réduction
SIMDF16 de MLX. La convolution conserve son accumulationFP32. Les huitcas
passent alors :409600valeurs finies exactes, y compris la norme débordante
versinf dont les sorties normalisées restentzéro. Aucune tolérance relâchée.

[Premier échec](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/primitives.json),
[échec avec accumulationhalf seule](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/repair-first.json),
[oracle corrigé](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/rounding-primitives.log),
[sourcesMLX consultées](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/mlx-reduction-sources.json).
Le kernel est comparé à la chaîne MLX originale, puis confronté aux
fixtures indépendantes et au vrai bridge.

## Qualification avant vitesse

La mutation du décalage historique t+1→t est détectée sur la cinquième
sortie ; shader restauré byteexact et oracle repassé. La première erreur
de norme avait déjà montré que le contrôle numérique n’est pas vacuant.

Format, Clippy strict, build releaseMLX/chat,87testsCPU/2ignorés et
107testsrelease/99ignorés passent. Kani sur le vrai helper (troisi8
indépendants diviséspar128, sansassume, cinqsortiesMLX) échoue dans son
compilateur avant obligations ; CBMC ne prend pas en charge le MSL.
Aucune preuve du GPU n’est revendiquée.

Les12testsQwen sélectionnés séparément, l’oracle tenu à part et3testsSSD
passent sans swapout supplémentaire. Ils couvrent16positions indépendantes
avec logits complets et états, commits1/2, sessions et têteMTP.
Le bridge valide budgets1/2/3/17, refus et annulations/reprises en186,09s, sans swapout supplémentaire. [Qualité](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/quality.json),
[contrôles](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/checks.json),
[mutation](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/mutation.json),
[limites de preuve](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/proof-inspection.json).

Baseline H :20/7a9a2120. Candidat25 : moteurbae34c75, tests9e11868f,
source23292112 plus [patch](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/source.patch)
et [shader](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/source.metal).
[Identités complètes](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/identity.json).

Protocole de débit préenregistré : batterie avant/après chaque bras, mêmes
M5/24Gio, MLX0.32.2, modèle65c89531 EXL3 2,05bpw, contexte4096, cache5Gio
(89slots), allocateur512Mio, readers4/overlap1. Deux tours H-C/C-H sur
MAIN/HOLD×OFF/MTP, unechauffe128 puis troisgénérations128 par processus,
127tokens décodés, médiane3. Parité128complète et préfixes64. Chaque paire
≥1%, étendue≤3points ; surpluspic≤256Mio, swapouts≤256Mio/bras,
ratiointra≤1,25. Arrêt au premiertour si unseuil échoue, immédiatement
sur erreur ; processus bornés360s et enfants joints. Applications conservées,
unseul modèle et aucun compilateur/proveur pendant la vitesse.

Revue des bornes du shader :80groupes×128canaux=10240,32lanes×4valeurs.
Indices des poids≤40959, historique entrant/sortant≤30719, Q/K≤2047,
V≤6143. Le wrapper n’accepte que les formes/dtypes correspondants et la
FFI matérialise les strides noncontigus. C’est une inspection du code,
pas une preuve du parallélismeMetal ; les refus et stridessont aussi testés.


## Résultat sur batterie et décision

Les deux prompts ont de nouveaux témoins20 sur batterie, sans utiliser les
chiffres de la campagne secteur20C. Batterie62→51%, applications ouvertes.

| Cas | H20, tok/s | C25, tok/s | Écart |
|---|---:|---:|---:|
| MAIN sans MTP | 9,998 | 10,123 | +1,247% |
| HOLD sans MTP | 9,748 | 9,784 | +0,367% |
| MAIN avec MTP | 10,323 | 10,421 | +0,943% |
| HOLD avec MTP | 8,889 | 9,025 | +1,522% |

Deux cellules échouent au seuil de1%. Le premier tour s’arrête donc après
8bras et32réponses complètes exactes ; les8bras inversés ne sont pas lancés.
La stabilité sur deux tours n’est pas établie. Ces écarts ne constituent pas
un gain intégré, même lorsque le signe est positif.

Tous les bras ont ajouté **0octet de swapout système**. Aucun pic candidat
ne dépasse son témoin : pic maximal8,659Gio sans MTP,9,414Gio avec MTP,
contre8,659/9,561Gio pour20. Ce sont les pics du processus, distincts des
allocations MLX. Les compteurs MTP restent identiques entre variantes et
répétitions : MAIN72propositions/54acceptations, HOLD78/48.

L’inspection finale contrôle aussi les32textes et hashes complets, les
médianes, la finitude des débits, les identités source/binaire et la fin de
tous les processus. Le patch et le shader restent archivés ; les sources
exécutables sont restaurées exactement au commit23292112. Aucun nouveau
kernel25 n’est livré et l’objectif15tok/s reste non atteint.

[Inspection finale](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/assessment.json),
[résultat de campagne](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/speed/result.json),
[commandes et conditions](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/speed/execution.json).

## État de livraison et limites

Le commit23292112 publie le gain HC20 dans la
[PR34](https://github.com/0xZKnw/mlxl3/pull/34). Les essais22,24 et25 restent
rejetés ; leurs variantes et leurs échecs numériques initiaux sont archivés.
Le profil21 reste pertinent pour le moteur20 conservé : environ27ms par pas
pour remplir les experts, et environ94% du temps MTP dans la vérification
cible. Ces mesures instrumentées ne s’additionnent pas et ne constituent
pas un nouveau benchmark. Le cache experts reste5Gio dans ces campagnes.

Sur23292112, les jobs natifs Linux/macOS, les contrats Flash Kani ciblés et
les régressions passent. La suite Kani complète atteint sa limite d’une
heure, comme sur la base ; la CI globale n’est donc pas verte. Les tentatives
Kani/MLX et CBMC/MSL locales ne prouvent pas le GPU. Les contextes testés
sont courts malgré la limite configurée4096 ; pas de validation de débit à
4096tokens effectivement remplis ni d’installation de l’application.


La restauration est reconstruite avec `cargo build --locked --release
--features mlx,chat` : succès, diff natif nul. Le SHA du binaire devient
4d4f9780, car deux chaînes indiquent désormais23292112 au lieu de36bae3ad.
Les sections d’instructions et de données inspectées sont identiques,
à ces chaînes près. Les débits restent attachés au binaire20 archivé7a9a2120.
[Build de restauration](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/retained-build.json),
[inspection du binaire](measurements/flash-next-ssd-impl/opt-25-gdn-prepare/retained-binary-inspection.json).
