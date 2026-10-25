# 70R — pistes Splash et Strata après le shared expert

22 octobre 2026. Revue statique terminée, sans nouveau prototype ni benchmark70.
Les quatre modes de68 sont mesurés séparément dans le
[rapport68](flash-next-ssd-shared-expert-68-2026-10-22.md). Le regroupement du
shared expert n'y qualifie pas un gain de génération. La réduction du coût
d'un composant isolé ne suffit donc toujours pas.

Les HEADs distants restent Splash `1482f6bbc2abf12375332725302019594150b6cb`
et Strata `011c9e600a93bbc9fd451c563501b5831c32bbca`. Dix fichiers sources
relus, passages ciblés : draft, sélection, budget, HC, GDN, vérificateur,
graphe, ordonnanceur et fenêtres/streaming. Leurs copies locales ont été
comparées octet par octet aux blobs distants épinglés :
[manifest70](measurements/flash-next-ssd-impl/opt-68-shared-expert/research-70-source.json).
Cela ne constitue pas une revue exhaustive des deux projets. Les constats
déjà présents dans59/61/63/64R/65 sont conservés comme tels.

## Ce que les nouvelles mesures permettent de choisir

Dans68, B2 séquentiel donne 11,205/10,017 tok/s MAIN/HOLD, contre
10,175/9,350 pour l'adaptatif. Comparaisons entre modes non alternées : elles
orientent la recherche, sans qualifier une accélération stable. B2 dépasse
MTP1 sur MAIN d'environ0,98 %, reste inférieur d'environ0,15 % sur HOLD.

Sur MAIN, le draft B2 prend0,535 s pour69blocs ; l'adaptatif prend1,209 s
pour49blocs, dont139lignes vérifiées contre138 enB2. La vérification prend
10,607 contre11,124 s. L'adaptatif demande légèrement moins d'octets experts,
23,167 contre23,931 milliards, mais ne gagne pas de temps dans ces passages.
Ce sont des durées inclusives, pas des temps GPU exclusifs ni des débits NVMe.
Les captures runtime sont dans les rapports bruts68 ; les histogrammes exacts
des largeurs ne sont pas exposés dans ces statistiques du bridge.

## Pistes et différences concrètes

| Priorité d'étude | Mécanisme amont | Écart MLXL3 et premier contrôle utile |
| --- | --- | --- |
| 1 | Prélecture urgente compatible avec le calcul en cours | Dans `verify_draft`, aucun MoE suivant n'est transmis ; activer lookahead crée pourtant le prefetcher et désactive le Hot anticipé. Brancher réellement les prédictions de vérification et conserver le Hot, puis mesurer lectures utiles, attentes et octets supplémentaires. Voir69R pour les protections des slots. |
| 2 | Préparation de l'entrée réutilisée entre opérateurs, formes de draft maîtrisées | Évaluer un plafond de propositions avant l'exécution du draft sur ce M5, avec coûts et acceptation par largeur. Notre B2 fixe le fait déjà ; la combinaison adaptative n'existe pas encore. Ce plafond n'est pas une fonctionnalité Splash oubliée. |
| 3 | Plan d'experts GPU et écriture directe des occurrences |65 groupe seulement les chemins dont les slots sont connus du CPU. Le Hot anticipé passe directement par `ExpertSlots::residents` puis `calculate_slots`, sans ce partage. Un prototype B2 GPU peut chercher les experts communs aux deux lignes et écrire directement chaque contribution, sans planCPU/concat/take supplémentaires. |
| 4 | HC groupé et entrée préparée pour la projection suivante | `HyperConnection` reçoit toutes les lignes et peut produire les tables consommées par la projection suivante. Nos HC et captures restent par position. Un nouveau test doit réduire aussi les préparations/intermédiaires, pas répéter le HC deux lignes rejeté22. |
| 5 | Commit GDN absorbé par le passage suivant | Splash lit la tape pending dans le scan suivant ; Strata peut différer le fold jusqu'à sa prochaine fenêtre. Notre replay est déjà paresseux dans MLX, mais reste un kernel séparé. Fusionner réellement les mises à jour enregistrées, conserver un seul préfixe différé borné et les observateurs d'état. |
| 6 | Enchaînement natif et mesure des étapes hôte | Scratchs persistants, événements et séquences préparées réduisent les reconstructions et permettent de distinguer prise en charge CPU, I/O et reprise GPU. Nos banques experts sont persistantes, le graphe complet ne l'est pas. Une mesure des attentes précède une réécriture de ce moteur. |

### 1 — Vérificateur et prélecture : corriger l'appelant aussi

La [revue69](flash-next-ssd-hot-prefetch-review-69-2026-10-22.md) identifie
deux défauts d'intégration distincts : demandes pouvant relire un expert
encore en vol, et vérificateur DFlash ne produisant pas les prédictions de
la couche suivante. Les compléter n'autorise pas à enlever simplement les
gardes Hot : les snapshots excluent les destinations pending, et toute
complétion tardive doit rester dans le Cold si elle n'était pas capturée.
Les compteurs doivent aussi conserver la fréquence réelle des demandes LFU.

Ce mécanisme peut réduire une attente ; il peut aussi saturer davantage le
SSD avec de mauvaises prédictions. Les mesures68 utilisent lookahead0 et
ne mesurent aucun bénéfice de cette proposition.

### 2 — Splash ne réduit pas le draft avant sa sélection adaptative

Dans [DFlashDraft::addDecode](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/DFlashDraft.cpp#L244),
les lignes du draft sont fixées par les huit positions par lane. Le
[DraftSelector](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/DraftSelector.cpp#L56)
traite les sept propositions, puis un opérateur distinct choisit le nombre
de lignes cible. Le plafond avant le draft serait donc une adaptation
MLXL3 motivée par nos mesures66/68, plutôt qu'une copie manquante.

Le draft DFlash n'est pas causal entre ses positions masquées. Réduire ses
lignes peut changer les propositions ; ce n'est pas équivalent à tronquer
les sept propositions d'un draftB8 déjà exécuté. La correction finale reste
contrôlable par le modèle cible, mais l'acceptation et les confidences doivent
être remesurées. L'oracle du draft tronqué ne doit pas exiger artificiellement
l'égalité au préfixe du draftB8.

Notre budget conserve les nœuds initiaux0,60/0,69/0,85/1,00 de
[VerifyBudget](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/VerifyBudget.cpp#L12),
issus d'autres moteurs, formats et budgets mémoire. Les coûts apprennent
dès le premier bloc ; le modèle d'acceptation se réajuste tous les64blocs.
Les49/54–55blocs de nos mesures adaptatives n'atteignent pas ce réajustement.
Le coût réel de chaque largeur sur M5/EXL3/3Gio mérite donc une calibration
indépendante, validée sur d'autres prompts et des générations longues. Il
ne faut pas ajuster le prior après chaque mauvais chiffre MAIN/HOLD.

### 3 — Partage EXL3 du Hot : un prototype distinct de65/66

Le regroupement65 ne modifie pas la voie GPUHot anticipée ; ce choix est
déjà documenté dans le [rapport65](flash-next-ssd-grouped-exl3-65-2026-10-21.md).
Le nouveau motif proposé cible cette voie, avec B2 : chaque ligne a dix
experts distincts, donc un expert commun a au plus deux occurrences. Le GPU
peut théoriquement identifier ce couple depuis les slots stables, décoder
une tuile une fois et garder deux accumulations indépendantes.

Il faudrait préserver l'ordre SG4/splits/FMA, les conversions et les
positions de sortie d'origine. Les routes absentes restent masquées, les
uniques gardent leur chemin actuel, sans addition atomique qui changerait
la réduction. La pression de registres et les groupes masqués peuvent
annuler le gain : aucun prototype ni chrono n'existe pour cette proposition.
Elle ne justifie pas de promouvoir le partage65 rejeté par les écrans66.

### 4 à6 — Réutiliser le travail, sans changer la trajectoire numérique

[HyperConnection](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/HyperConnection.cpp#L58)
peut émettre une entrée déjà préparée pour l'opérateur suivant. Cela inspire
la suppression de conversions/préparations répétées, mais nos Hadamard EXL3
dépendent des échelles propres à chaque projection. Les formats Table16/GGUF
ne permettent pas de partager aveuglément ces transformations. HC dense
doit garder son ordre GEMV exact. PLE n'est présent que sur la couche1 dans
ce modèle ; son optimisation ne se multiplie pas par48.

Dans [GDN::addDecode](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/GDN.cpp#L207),
la tape pending et la tape du nouveau passage sont disjointes, puis le scan
fournit les états. [Strata::Window::commit](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/window.cpp#L1569)
diffère éventuellement son fold et le matérialise aussi avant checkpoint.
Notre tape contient déjà les deltas calculés : reprendre son algèbre, sans
recalculer les deltas comme le format amont. Les tests doivent couvrir commit
partiel/total, continuation ordinaire, annulation, reprise et fin de contexte.
Une petite durée hôte de commit n'isole pas le coût de son replay paresseux.

Le [CommandGraph](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/CommandGraph.hpp#L69)
exprime explicitement les étapes CPU entre les dispatchs. Le
[HostStepRunner](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/MetalBackend.mm#L615)
documente un compromis entre polling court et sommeil, distinct du kernel
GPU de maintien actif. Ce dernier ne doit pas être copié sans étude de ses
mécanismes Metal. Les gains mesurés sur leur M5Max ne sont pas des gains
prévisibles sur notre M5. Ni le simple cache des factories67 ni davantage de
threads de lecture ne reproduisent cet ordonnanceur.

## Essais à ne pas répéter tels quels

- HC deux lignes22 et fusion GDN mono-ligne25 : rejetés, prototypes retirés.
- Dispatch compact57 et groupedEXL3 CPU65/66 : ne pas réannoncer un gain réel
  à partir d'un microbenchmark favorable.
- Cache des factories19/67 et second repackingSSD : coût ou mécanisme déjà
  traité, sans nouvelle preuve de chemin critique.
- Constantes de tables IQ Strata : notre codebook EXL3 est un décodage
  arithmétique, pas la même table à déplacer vers la mémoire constante.

Les quatre premiers chantiers du tableau ont des contrats précis et des
oracles réutilisables. Aucun n'est implémenté ou validé par cette revue.
