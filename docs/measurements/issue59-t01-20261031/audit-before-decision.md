# T01-P-01 — audit du filtre avant décision

31 octobre2026. Lecture de `host-r1-plan-result.json`, `host-profile-r1.json`,
du collecteur et de son test réel ; aucun nouveau chrono, build, test,
proveur, GPU ou changement de code. **Aucun défaut bloquant d'adaptation ou
de collecte identifié.** Les limites ci-dessous bornent la décision.

## Réalité du chemin et des comptes

- Processus terminé exit0, joint en0,398s ;0nouveau Swapout/Swapin aux
  observations. Rapport `validated_host_only`, pas une mesure modèle.
- Fixture synthétique utilisant **`linear::expert_slots` réel**, slots
  résidents masqués53, GU20×2560→640 K2/MUL1, sortie `[20,640]`F16 ;
  partial `[1,20,640]`F32, grid102400, threadgroup128. Neuf slots présents
  et une route masquée ; les valeurs ne sont pas calculées.
- Nom exact HotGU, masque/slot-major, stride12800 et géométrie vérifiés.
  Header1842octets ; source14733octets **identique** au shader actuel,
  SHA256`02f91b3a97be9c6323fbc0a4c9d4d1ef13f68504eef1065b60d2c50a7def98c8`.
- Warmup64 :1miss initial+63hits. Puis8lots, ordre off/coarse/fine et inverse,
  256builders par mode/lot, **6144builders mesurés** au total.
- Chaque mode représente2048builders. Les compteurs Rust voient2048coarse
  et2048fine ; **C++ voit seulement2048fine**, tous sans miss/éviction,
  max2entrées. Les zéros off/coarse sont des compteurs désactivés, pas une
  observation directe de leurs misses. Signature inchangée/cache non modifié.
- Chaque sortie contrôlée a la forme/type attendus et reste pending ;
  host_readbacks_delta=0 sur tous les lots. Trace de signature non fenced,
  terminée avant les lots. Le code n'appelle pas eval/async_eval/synchronize.
  « GPU execution not_requested » est correct ; aucun temps GPU ni comptage
  physique de dispatch n'est mesuré.

## Temps réellement inclus

Moyennes des8lots, ns cumulées divisées par2048appels du mode concerné :

| Poste | µs/appel | Portée |
|---|---:|---|
| boucle externe off | 5,5292 | builder complet+contrôles shape/type/pending+drop |
| boucle externe coarse | 5,3695 | même boucle, deux timestamps internes supplémentaires |
| boucle externe fine | 5,8831 | mêmes travaux et tous les compteurs/timers fins |
| Array Rust fine « total » | 2,8536 | validation→adoption des sorties |
| Rust fine FFI | 2,0277 | contient tout le C++ profilé |
| C++ lookup | 0,6504 | recherche clé complète, pas seulement hash/nom |
| C++ closure/invoke | 0,6975 | `arrays(...)`, génération signature/source, primitive et graphe neufs |
| C++ total | 1,9927 | enveloppe inclusive et observateur C++ |

Les phases C++ sont **dans** RustFFI, lui-même dans Rusttotal puis boucle
externe. Ne pas les additionner. La closure n'isole pas la génération de
source seule : des nouvelles entrées/primitive/sorties resteraient requises
avec T01A. Le cache de cette fixture n'a que2entrées, sans le working set
du modèle complet ; aucune absence de churn modèle n'en est déduite.
Off utilise le même libtest instrumenté avec collecteurs désactivés ; ce
n'est pas une mesure du CLI release antérieur à l'instrumentation.

Deux détails de scope à garder visibles : Rusttotal est horodaté à la fin
de l'adoption, **avant** l'enregistrement du collecteur et la destruction
des CString/Vec locaux ; C++total, via son destructeur RAII, inclut ses
destructions locales, scans strlen et une partie des timers. Les coûts de
formatage linear et les nœuds autour du QMV restent dans la boucle externe.
Il n'y a donc pas de partition exclusive complète du coût host par fonction.

Coarse extérieur est inférieur à off (−0,160µs), fine lui est supérieur
(+0,354µs). Ce résultat non monotone montre l'effet observateur et la
variabilité à cette échelle ; **ne pas soustraire arbitrairement** ces
différences pour « corriger » les phases. Huit lots d'un seul processus
ne constituent pas une distribution de performances sous charge modèle.

## Compte historique et décision bornée

[historical-call-count.json](historical-call-count.json) rapproche la même
signature, header/source byte-identiques, d'une trace ARMAIN du27octobre :
6096requêtes sur127pas,48parpas et127parchacune des48couches. Cela établit
un compte sur **cette trace historique B1**, pas sur toutes les formes/modes.

La multiplication descriptive des coûts host actuels par ces48requêtes donne
0,2654ms/pas pour la boucle entière et0,03348ms/pas pour l'invocation de
closure incluse. Ce sont des **projections conditionnelles**, pas un plafond
de gain modèle prouvé : fixture non évaluée, autre date/charge, travail
obligatoire dans les enveloppes, pas de concurrence GPU/CPU reproduite.
Aucun ajout/retrait aux12,142ms historiques n'est permis.

**La génération source seule apparaît petite dans ce filtre.** Ne pas
porter T01A au seul motif de supprimer cette génération est une décision
raisonnable, sans prétendre que le launcher préparé est universellement
inutile. T01B (hash/source dans eval_gpu, résolution pipeline et encodeur)
reste **non mesuré et non réfuté**. Aucun gain tok/s, numérique ou mémoire
du modèle ne résulte de ce diagnostic ; aucune nouvelle évaluation du modèle
n'est nécessaire pour confirmer ce que ce filtre a effectivement observé.
