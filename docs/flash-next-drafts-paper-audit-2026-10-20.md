# Qwen3.8 Flash Next : audit du prototype DFlash2 / DSpark

Le prototype à trois propositions est fonctionnel en greedy, mais il ne mesure
pas une implémentation optimisée des deux méthodes. Plusieurs écarts concrets
restent dans la fidélité numérique du draft, son protocole d’évaluation et le
coût de vérification. Aucun résultat actuel ne permet de conclure que DSpark ou
DFlash2 ne peuvent pas accélérer ce moteur ; aucun gain ni objectif de 15 tok/s
n’est validé non plus.

Cette revue répond à la demande du 20 octobre de relire le dernier document.
Elle n’ajoute aucune modification exécutable ni inférence. Les instructions de
l’audit fourni sont traitées comme des propositions à examiner.

## Périmètre et mesures conservées

- Audit fourni : chapitre 9 de `MLXL3_PR34_AUDIT_GLOBAL_V2_2026-10-19.md`,
  SHA-256 `9d450f3a307848552f749c608902c1c1ac30b50ffb103287f846b8ce5dab9012`.
- Branche `codex/flash-next-ssd-205`, HEAD `615b254df577ca502bf99e994548b7231502de4f`,
  avec modifications locales. Prototype limité à `cfg(test, feature="chat")` ;
  aucune activation dans le bridge ou l’application.
- Source draft : `f8c9bbaff1d6f554c8355576b2e2328563eb5280692683613de66e5c46972ce5` ;
  unité positive figée phase4 :
  `c918e6e1b353a5cee82e4b03de180317daea25d4a91b8527bd31ffc4cd6e4837`.
- Cible EXL3 2,05 bpw `Qwen3.8-Flash-Next-exl3-205-65c89531` ; drafts BF16
  Inco `a22a1744e85c17364dcc54a6a420af7135d4c7f6` et DSpark
  `aafaf1bf59fdc4639183600d7747474990623171`.
- M5 24 Gio, MLX 0.32.2, secteur, applications conservées ; cache experts
  3 Gio / 53 slots, allocateur 512 Mio, quatre lecteurs, overlap actif, LES
  désactivé, contexte maximal 4096. Une chauffe 128 puis deux générations 128
  par bras, dont 127 tokens de décodage chronométrés.

La campagne a été interrompue pour cette revue : neuf bras terminés, un
interrompu et rejoint, six non lancés. Les huit bras MAIN forment un bloc
équilibré ; HOLD ne possède qu’un témoin. Les 27 générations terminées ont
leurs 128 IDs exacts. Aucun nouveau Swapout système dans les bras terminés ;
les Swapins ne sont pas tous nuls. Ce n’est pas une campagne complète sur deux
prompts. [Matrice, conditions, commandes et résultats](measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed/result.json).

| Variante sur MAIN | Débits des deux bras, tok/s |
| --- | --- |
| Sans spéculation | 9,093 ; 8,867 |
| MTP natif | 9,264 ; 9,388 |
| DFlash2, trois propositions | 7,985 ; 8,110 |
| DSpark, trois propositions | 5,788 ; 5,910 |

Ce sont des observations du prototype natif de test, pas des gains livrés.
Les pics d’empreinte processus, incluant chargement/prefill/chauffe, sont
7 155 341 240 octets pour le premier OFF, 8 859 966 304 pour DFlash2 et
8 997 346 240 pour DSpark. Ils proviennent de
`process_lifetime_peak_bytes`, et sont distincts des compteurs d’allocation MLX.

## Écarts identifiés

**1. La normalisation du résidu ne reproduit pas le contrat du guide Inco.**
`Draft::sublayer_output` additionne deux tenseurs BF16 puis `forward` normalise
ce résultat déjà arrondi. Le kernel fusionné SGLang calcule sa somme de carrés
sur la somme FP32 avant l’arrondi du résidu. Les deux calculs ne sont donc pas
interchangeables. La version native Python SGLang possède aussi une addition
FP32 avant sa variance ; elle ne suffit cependant pas à représenter tous les
arrondis du kernel CUDA. [Kernel exact du guide, révision 3c560089](https://raw.githubusercontent.com/sgl-project/sglang/3c560089e1818a5217a73f2168273485a5350341/python/sglang/kernels/jit/csrc/elementwise/fused_add_rmsnorm.cuh).

Un contrôle fini CPU préenregistré, sans modèle, confirme la différence :
320 sorties BF16 différentes sur 2560 valeurs finies, pour le premier motif
du seed 20261020, poids unitaires et epsilon 1e-6. Les variances sont
2,4300642013549805 et 2,4255690574645996.
[Entrées et sorties exactes en bits](measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed/residual-norm-counterexample.json).
Cela confirme l’écart de formules ; son effet sur les propositions et
l’acceptation du modèle reste non mesuré.

**2. L’oracle ne ferme pas encore la chaîne réelle du draft.**
`scripts/check_flash_drafts.py` calcule les vraies captures, caches KV et cinq
couches avec NumPy, puis teste le sélecteur avec un autre jeu de hidden/logits
synthétiques. Les activations du backbone sont acceptées avec une RMSE relative
jusqu’à 2 %. Le test de sélecteur détecte bien l’ancien facteur 0,25 ou un
prédécesseur Markov figé, mais il ne compare pas les propositions issues du
vrai backbone et de la tête EXL3 complète à une référence indépendante.
La parité finale du target ne remplit pas ce manque : le vérificateur peut
corriger un mauvais brouillon. Les grandes positions près de 2048 ne sont pas
couvertes par cette campagne courte.

**3. DSpark est évalué dans un autre mode que son checkpoint publié.**
`prompt()` appelle `ChatTokenizer::render_values` sans `enable_thinking`.
Le template cible sélectionne alors thinking et `xhigh`. Le checkpoint DSpark
utilisé a été entraîné sur des réponses non-thinking et évalué dans ce mode.
L’incidence sur l’acceptation n’est pas mesurée. Il faut deux comparaisons
distinctes, avec témoins appariés dans chaque mode ; désactiver thinking ne
constituerait pas un gain à comportement identique.
[Fiche du checkpoint DSpark épinglé](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark/blob/aafaf1bf59fdc4639183600d7747474990623171/README.md).
Le guide DFlash2 Flash Next utilise au contraire `xhigh` ; cet écart de mode
ne s’applique donc pas aux deux drafts indistinctement.

**4. La confiance DSpark est calculée puis jetée.**
La ligne `let (proposals, _) = draft.select(...)` conserve une largeur fixe.
Il manque la calibration STS, les probabilités de survie cumulées et une table
de coût propre au moteur pour choisir le préfixe à vérifier. Le papier combine
ces éléments ; ses expériences de qualité à largeur fixe désactivent toutefois
explicitement le scheduler. Le prototype statique est donc un contrôle initial
légitime, prévu aussi par l’audit V2, mais il ne teste pas le système adaptatif
complet. [Papier DSpark, §§3.2 et 4.2](https://arxiv.org/html/2607.05147v1).
Le SGLang inspecté retombe également sur verify-all lorsque sa table SPS n’est
pas initialisée ; importer son planner sans profil local ne résoudrait rien.
[Planner SGLang épinglé](https://raw.githubusercontent.com/sgl-project/sglang/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/speculative/dspark_components/dspark_planner.py).

**5. Les sélecteurs conservent des synchronisations CPU évitables.**
DFlash2 extrait les candidats puis attend un argmax par position. La référence
prépare en parallèle les scores entre tous les candidats voisins, puis suit
le chemin sur GPU. [Description DFlash2](https://inco.ai/blog/dflash2/).
DSpark effectue un `to_u32()` puis un `to_f32()` par proposition, soit six
lectures synchrones pour trois propositions ; la référence possède un chemin
greedy fusionné et une boucle maintenue sur le GPU. Les dépendances Markov
restent séquentielles, mais elles n’imposent pas ces retours CPU.
Leur coût séparé n’est pas mesuré.

**6. Le vérificateur cible est encore majoritairement scalaire.**
`verify_draft` boucle sur `layer.prepare` pour chaque token, avec projections,
attention GDN/QSA et HC séparées, puis groupe le MoE. Les têtes restent des
graphes mono-ligne indépendants. Grouper le MoE n’équivaut donc pas à exécuter
tout le target en bloc. La récurrence GDN exige de conserver l’ordre ; ses
projections indépendantes peuvent être étudiées séparément. Toute nouvelle
forme matricielle doit conserver la parité numérique choisie.

Les snapshots retiennent les états de chaque préfixe. À B4, les seuls anciens
états GDN représentent logiquement 324 Mio, contre 108 Mio à B2. Ces clones
sont des références de tableaux ; aucune copie physique de 324 Mio par bloc
n’a été démontrée. Passer directement à sept propositions augmenterait encore
la rétention et la pression sur les 53 slots experts.

Les poids propres sont aussi entièrement BF16 dans ce premier port. L’exception
des projections dynamiques de largeur 640 justifie leur maintien BF16, mais
n’oblige pas à conserver toutes les autres matrices dans ce dtype. La
quantification du draft reste une piste mémoire avec qualité à requalifier,
pas un gain de débit déjà observé.

## Ce que le profil permet réellement de prioriser

Sur les deux mesures du premier bras DFlash2, le temps moyen est 15,908 s :
1,063 s de draft et 14,845 s de vérification **avec commit et entretien des KV
draft**, soit 93,3 %. DSpark prend 21,943 s, dont 1,749 s de draft et
20,194 s de vérification/entretien, soit 92,0 %. Ce découpage ne mesure pas les
kernels cibles isolément. Les compteurs I/O actuels incluent le prefill ; ils
ne permettent pas d’attribuer le ralentissement au SSD seul.

À acceptation et vérification inchangées, rendre le draft gratuit ne donnerait
qu’une borne idéale de 8,56 tok/s pour DFlash2 et 6,29 pour DSpark sur ces bras.
Ce sont des calculs à partir des temps conservés, pas de nouveaux benchmarks.
Atteindre 15 tok/s avec la même acceptation demanderait respectivement environ
47 % et 61 % de temps total en moins. Optimiser seulement les quelques lectures
CPU du selector ne peut donc pas suffire.
[Calculs dérivés et empreinte des données d’entrée](measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed/audit-derived-costs.json).

L’ordre de travail justifié est : fermer l’oracle couplé et qualifier les
arrondis du résidu ; mesurer coût et survie pour B2/B3/B4 dans des modes
appariés ; optimiser la vérification et instrumenter ses lectures SSD ; enfin
calibrer la sélection de préfixe et supprimer les synchronisations du selector.
La quantification et le journal de récurrence visent aussi le budget mémoire.
La causalité de ces pistes reste à tester, aucune addition de gains n’est faite.

Les captures de couches, moyennes HC FP32 puis BF16, masques distincts,
RoPE 10000/10000000, largeur de backbone adaptée, décalage DSpark=0 et coefficient
DFlash2=1 sont déjà présents. La révision du guide `3c560089` diffère de
`e00930c5`, mais les expressions de score inspectées confirment ce coefficient ;
aucune nouvelle erreur de capture ou de convolution n’a été établie.
[Sources primaires et SHA](measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed/guide-research.json).

La qualité précédant la vitesse reste conservée : 121 tests Rust réussis et
116 ignorés dans la suite générale ; 448 tests Python réussis ; sept processus
physiques phase4 réussis, cinq anciens contrôles indépendants réutilisés avec
leurs conditions 4 Gio explicites, trois mutations détectées/restaurées puis
positifs réexécutés. Les deux contrats CPU Kani ont 137 obligations réussies et
six couvertures atteintes ; les tentatives sur les chemins MLX réels ont
rencontré l’ICE Kani `intrinsics.rs:243`. CrossHair absent. Ces contrôles ne sont
pas une preuve complète du GPU ni du draft de bout en bout.
[Qualité et limites](measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts4-quality/qualified-bundle.json).

Pour reproduire les résultats positifs, utiliser l’unité phase4 figée dans
`build/flash-opt-43/drafts/phase4`, identifiée ci-dessus. Le binaire de test
mutable dans `target/release/deps` a servi aux mutations et doit être reconstruit
avant réutilisation. Les sources ont été restaurées et leur SHA revérifié.
Les échecs antérieurs de chargement, budget1 et garde swap sont conservés dans
le journal. Aucune nouvelle CI du code local n’a été exécutée sur GitHub.
