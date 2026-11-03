# LES — revue complémentaire 42, 19 octobre 2026

Revue du runtime publié dans `e64d1d439a798887bf955ada27c99d789684df32`
(tête documentaire `8cca95b2e70d312c14d8bf977408088a2aecc0a3`). Le moteur
mesuré est toujours `a3fd917d625c20e4dd4e4144a84fcca5fe37ff0262c67caf4598f53e4597c59d`.
Cette revue ne modifie aucun code exécutable et ne produit aucun nouveau profil.
Les observations de code ci-dessous établissent des mécanismes, pas leur durée.

Le journal 42 reconnaît que la première lecture de `SsdMoe`/`ReaderPool` a
précédé son préenregistrement. Les lectures suivantes et ce rapport ont lieu
après cette entrée. Aucun prototype ni benchmark 42 n'a été lancé.

## Ce qui reste réellement dans le code

| Priorité | Observation et scénario | Conséquence et contrôle nécessaire |
| --- | --- | --- |
| P1 | Le routage réel emploie toujours la projection dense générique. Pour deux lignes MTP, `forward_lookahead` appelle `route` séparément sur chaque ligne. | L'option prédictive FP16 n'accélère pas ce chemin. Deux graphes de GEMV et de promotion des poids sont construits par couche. Un éventuel partage du cast FP32 entre les deux GEMV pourrait conserver leur forme et leurs arrondis ; son coût matériel et son gain restent **non mesurés**. |
| P1 | `verify_pair` appelle deux fois `logits`, et `logits` fait un `eval` bloquant avant de retourner. | Deux évaluations séparées de la tête vocabulaire sont certaines. Construire les deux graphes inchangés puis les évaluer ensemble est une piste distincte du regroupement MB2 rejeté dans l'essai 15. Il faut vérifier tous les logits, états et commits 1/2 avant de chronométrer. |
| P1/P2 | `protect_predictions(ids)` protège les 16 candidats d'une ligne ou jusqu'à 32 des deux lignes, avant le filtrage du cache et l'admission d'au plus quatre nouveaux jobs. La même fonction `victim` sert à la demande et à la prélecture. | La protection influence les évictions même pour des candidats non admis. Avec 71 slots, le masque peut désigner jusqu'à 32 experts ; cela ne mesure ni leur présence réelle ni une pollution effective de 45 %. Une ablation à capacité identique et un replay des routes doivent isoler cette politique. |
| P1/P2 | `output.async_eval()` peut être appelé dès qu'un candidat manque, puis `prefetch` peut admettre zéro lecture : pool plein, expert déjà pending, réserve de slots ou échec de préparation. | Le test de soumission GPU et la décision effective d'I/O restent séparés. Le cas est déjà décrit par LES-08 dans l'audit fourni ; 41 ne l'a pas corrigé. Compter les soumissions sans admission avant de modifier leur ordre. |
| P2 | `note_late` compte un expert encore en vol sans le rendre disponible. La demande réserve alors un autre slot et peut relire le même expert. La publication spéculative ultérieure est rejetée comme doublon. | Le repli reste exact et évite d'attendre toute une cohorte, mais peut doubler une lecture. Il faut mesurer les échéances individuelles, sans transformer le repli en attente spéculative bloquante. |
| P2 | Les quatre workers spéculatifs n'ont chacun qu'un job pending. Ils sont indépendants des quatre lecteurs de demande. | Une profondeur 8/10 n'admet pas huit jobs globaux. Jusqu'à quatre lectures spéculatives et quatre lectures de demande peuvent se concurrencer ; aucune priorité explicite de la demande n'est implémentée. Augmenter seulement la file peut aggraver cette concurrence. |

Sources : [routage et boucle MTP](../native/src/ssd_moe.rs#L272),
[admission et protection](../native/src/ssd_moe.rs#L550),
[slots et évictions](../native/src/flash_support.rs#L251),
[pool et lectures tardives](../native/src/ssd_prefetch.rs#L104),
[projection dense](../native/src/lfm2.rs#L201),
[tête et vérification MTP](../native/src/qwen4.rs#L1191).

Le volume logique d'une sortie de conversion de gate `[512,2560]` en FP32 est
5 Mio. Les 48 routes réelles représentent donc 240 Mio de sorties de conversion
par token, avant lectures ultérieures ; les deux graphes de routage représentent
480 Mio par paire MTP si chacun matérialise sa conversion. Ce ne sont ni une
mesure de trafic physique, ni un coût temporel, ni 480 Mio de RAM permanente.
Le parcours `eval_impl` MLX inspecté visite les identités des nœuds ; aucune
preuve matérielle de fusion ou de suppression de ces conversions n'est produite
ici. La quantification du coût reste à faire sur le runtime exact.

Le signal de prédiction HC et la combinaison des deux lignes MTP restent à
calibrer sur les **misses réellement nécessaires** et leurs échéances. L'exactitude
du texte lorsque la prélecture se trompe valide le repli, pas l'utilité du signal.

## Hypothèses écartées ou déjà couvertes

- Les shards ne sont pas rouverts pour chaque expert : `TensorInfo::source`
  réutilise le descripteur inspecté. Les `metadata` répétés sont des contrôles de
  troncature ; `prepare_read` duplique un descripteur pour sa durée de vie. Il ne
  faut pas supprimer ces contrôles sans conserver leurs garanties.
- La factory Metal est déjà cachée avec une recherche hétérogène. La fabrication
  des chaînes Rust subsiste, mais les variantes de clé courte PERF-06/PERF-33 ont
  déjà été rejetées. Cette revue ne les relance pas.
- La tête vocabulaire convertit déjà son entrée en FP16 : elle ne promeut pas ses
  poids en FP32 à cause d'une entrée HC FP32. Le problème observé ici est la
  séparation des évaluations, pas cette promotion hypothétique.
- `Projection::forward_dense_rows_exact` existe, mais renvoie du FP16. Il ne
  remplace donc pas directement la gate HC FP32 ; son nom ne prouve pas son
  exactitude pour cette forme. L'historique FIX-09 documente déjà une divergence
  de ce helper sur une autre projection, à ne pas ignorer.
- NT2 désigne deux tuiles de sortie, pas deux tokens. Le prototype de slots
  29/35 n'a toujours pas été exécuté ; il ne constitue aucun gain acquis.
- Le préfill calcule des sorties qui peuvent être inutilisées. Cela concerne le
  temps avant le premier token ; aucun gain de débit de décodage n'en est déduit.

Sources : [descripteurs](../native/src/checkpoint.rs#L96),
[préparation des lectures](../native/src/array.rs#L698),
[cache de factory](../native/ffi/mlx_bridge.cpp#L594), et historiques dans
`opti.md` : PERF-06/33, FLASHSSD-15/26/29/35 et audit 40.

## Mesures 41 à distinguer de cette revue

Le crible OFF 41 à 4 Gio trouve FP16 éligible sur les deux prompts, mais sa
confirmation alternée est **rejetée** dès la première paire : 10,110099 tok/s
pour C0-LFU contre 9,833681 tok/s pour C4-F16-LRU, soit −2,73408 %. Les huit
réponses de cette paire sont exactes et les deux bras n'ajoutent aucun swapout.
Les six bras non lancés restent explicitement absents. Le crible n'est donc pas
un gain reproductible démontré. Le crible MTP est en cours au moment de cette
revue ; ses résultats devront être lus dans le journal 41, sans les attribuer à 42.

Complément à la clôture 41 : le crible MTP est terminé, sans candidat éligible.
Le moteur LFU sans LES atteint 10,222442 tok/s sur MAIN et 9,796803 sur HOLD.
L'avance MAIN sur le témoin R0 le plus rapide est de 0,925995 %, sous le seuil
préenregistré de 1 %. LES FP16 atteint 10,127525 / 9,604047 tok/s, sous LFU sur
les deux prompts. Les compteurs de référence sont 54/72 propositions acceptées
sur MAIN et 48/78 sur HOLD ; ces taux ne mesurent pas la durée de vérification.
Les 26 bras terminés de la campagne 4 Gio n'ajoutent aucun swapout.
Voir le [bilan 41](measurements/flash-next-ssd-impl/opt-41-les-critical-path/summary.json).

Ces mesures emploient l'override `MLXL3_SSD_CACHE_GIB=4`. Le budget automatique
de production reste celui de `automatic_cache_slots` et peut dépasser 4 Gio
selon la mémoire estimée disponible. Cette campagne ne valide pas son comportement
matériel ni celui de longues conversations. Elle ne remplace pas l'app installée.

Preuves : [confirmation OFF](measurements/flash-next-ssd-impl/opt-41-les-critical-path/baseline/confirm-les/result.json)
et [identité de la revue](measurements/flash-next-ssd-impl/opt-42-followup-review/identity.json).

Décision 42 : revue terminée, pistes identifiées et hiérarchisées. Aucun nouveau
bug de correction reproduit par cette lecture, aucun correctif ou gain 42
revendiqué. Un profil léger du routage réel, de la tête MTP et des admissions
utiles doit précéder le prochain prototype. Aucun réglage LES supplémentaire
n'est activé par défaut ; l'objectif de 15 tok/s n'est pas atteint.
