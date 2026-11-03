# T18a w2 — audit final de l'adaptation DFlash, 31 octobre 2026

Lecture seule des sources et des quatre reçus DFlash après clôture des pilotes.
Aucun port, build, test, preuve, benchmark ou GPU lancé par cet agent. Sources
et binaires restent ceux de l'identité w2 ; défaut encore Static. Ce document
audite le candidat avant décision, sans remplacer l'agrégation du coordinateur.

## Mécanisme et chemins effectivement admissibles

- Les quatre rapports `runs/t18a-w2-pilot128-DF-{1-A,2-C,3-C,4-A}/report.json`
  sont complets ; même CLI `1fea100a…`, même modèle/draft, cache3Gio, quatre
  lecteurs, overlap/Hot/Wide/Unique1, lookahead0, width1/adaptive0/multi-token1,
  T04/T17/T05=0. Option lecteurs **0/1/1/0**, DFlash mode1. Aucun changement de
  mode ou de budget trouvé entre A et C.
- `flash_draft.rs:989–1028` concatène les deux entrées de vérification puis
  appelle `forward_draft_lookahead`. À profondeur0, le parcours rejoint
  `forward_draft_rows` (`ssd_moe.rs:495–508`), puis le même `fill_many` que AR.
  Ce n'est pas un chemin résident alternatif ignorant le nouveau reader.
- Les jobs sont les misses Cold dédupliqués par slot (`ssd_moe.rs:965–972`).
  Le bridge choisit Dynamic seulement si requested1 ET workers>1 ET jobs>workers
  (`mlx_bridge.cpp:324–327`, `disk_work_queue.h:18–20`). Avec quatre lecteurs,
  tout lot>4 est donc éligible ; les lots1..4 restent Static. Les reçus release
  **n'enregistrent pas l'histogramme par lot ni les retries CAS** : aucun taux
  d'utilisation Dynamic de ce pilote n'est inventé.
- Le même compteur `expert_demand_bytes` est observé dans les quatre processus,
  pour chaque phase/prompt : chauffeMAIN23664803544B, chauffeHOLD25647894612B,
  mesureMAIN23889445704B, mesureHOLD25808888160B. Soit respectivement
  **18962/20551/19142/20680 records** de1248012B. Il mesure les octets logiques
  du target, pas des octets NAND (`flash_draft_runtime.rs:115–126,215–220`,
  `qwen4.rs:1698–1708`). Aucun surcroît de demande/cache observable entre A/C.
- Le vrai stockage ON déjà qualifié imprime **9 lots Dynamic/9 fallback**,
  18cas/1644records/52608octets avec canaris ; le MoE indépendant ON/OFF couvre
  froid/chaud/B2/erreurs/reprise. L'oracle DFlash w2 confirme36préfixes,
  largeurs1..8, états/logits target byte-exact et captures64000valeurs. Voir
  [qualité w2](quality-inspection.md) et [qualité initiale](../quality-inspection.md).
  Ces contrôles empêchent de qualifier une voie silencieusement absente dans
  l'intégration, sans fournir un profil par lot des mesures de vitesse.

## Fidélité PR61, allocations et synchronisations

Le CAS borné PR61 ([patch queue](../upstream/native_ffi_disk_work_queue.h.patch.txt), tête
`a36ba5077bf386466660d6b478058fbe6880245c`) est conservé : load/CAS relaxed,
index unique, compteur arrêté à count, sentinelle stable. Le worker prend son
job suivant après son propre retour de lecture. La différence préenregistrée
est la garde des petits lots au bridge ; aucune double politique dans le reader.

Le reader suit exactement le changement PR61 de boucle d'affectation. Même
GCD `dispatch_apply_f`, même maximum4, même `preadv`, mêmes segments/destinations,
validations, capture des erreurs et drainage complet avant retour. Exceptions
réémises dans l'ordre des jobs après jointure ; publication/commit du cache
inchangée. Les banques et leurs propriétaires restent vivants pendant le call.

Aucun clone/copie de record, nouveau buffer Metal, cast, allocation de banque,
fence ou synchronisation GPU ajouté au bras Dynamic. Le vecteur d'erreurs et les
autres préparations existent dans les deux bras ; la queue est sur pile dans
les deux. Les timings sont désactivés en release. Les **CAS/retries et la
modification de l'affectation aux workers sont des coûts réels intentionnels**,
sans mesure exclusive de leur durée. Leur effet sur le service OS/cache fichier
peut varier ; ce n'est pas un défaut de port démontré.

Le défaut de diagnostic stale trouvé avant mesure est corrigé uniquement sous
cfg(test), avec régression sur le vrai `fill_many`. Il ne change pas ce chemin
release. Aucun autre défaut concret d'adaptation découvert dans cette relecture.

## Mémoire : chaque chauffe et mesure DFlash

Données : `stats.memory` et `stats.dflash_profile` des quatre rapports.
Compression = `process_compressed_logical_bytes` en Mio ; décompressions =
compteur cumulatif du processus. Delta entre deux footers successifs, **pas un
compteur exclusif du decode** : l'intervalle inclut aussi préparation/prefill et
travail entre générations. Premier delta depuis le footer `ready`.

| Processus | Phase | Decode s | Vérification s | Compressé Mio | Décomp. cumulées | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|1A|chauffe MAIN|12.565733|10.336927|951.359|53829|+53827|
|1A|chauffe HOLD|11.381581|10.572889|956.062|90951|+37122|
|1A|mesure MAIN|10.777952|10.060270|1004.922|125119|+34168|
|1A|mesure HOLD|11.146666|10.357249|1004.406|140073|+14954|
|2C|chauffe MAIN|10.846943|10.131438|1283.266|66152|+66150|
|2C|chauffe HOLD|11.233033|10.430160|1296.734|86161|+20009|
|2C|mesure MAIN|10.751560|10.045097|1309.297|100144|+13983|
|2C|mesure HOLD|11.159532|10.356536|1311.828|110838|+10694|
|3C|chauffe MAIN|11.055752|10.322748|1559.703|89687|+89685|
|3C|chauffe HOLD|11.059234|10.269436|1554.391|90057|+370|
|3C|mesure MAIN|10.612292|9.928223|1554.328|90060|+3|
|3C|mesure HOLD|11.925809|11.120567|1554.328|90060|+0|
|4A|chauffe MAIN|10.893877|10.188648|1146.938|52203|+52201|
|4A|chauffe HOLD|11.316380|10.517417|1177.641|71856|+19653|
|4A|mesure MAIN|10.689166|9.981504|1184.609|87808|+15952|
|4A|mesure HOLD|11.099585|10.303137|1186.469|101716|+13908|

3C/HOLD est plus lent que2C/HOLD de **0,766277s** de decode, dont **0,764030s**
dans l'enveloppe de vérification. Draft et commit ne portent pas l'essentiel de
l'écart. Cette enveloppe mêle CPU/GPU/I/O/cache ; elle n'isole ni atomiques ni
SSD physique.

Point important : de3C/MAIN à3C/HOLD, `process_decompressions` reste **90060**,
compressed current **1629831168B** et lifetime **1670594560B** restent identiques.
Le HOLD lent **ne correspond pas à une hausse observée des décompressions natives**
sur cet intervalle. Le niveau comprimé supérieur de3C n'établit donc pas cette
causalité. Les états mémoire diffèrent néanmoins entre processus.

## Conditions et limites d'attribution

Les quatre reçus `t18a-w2-pilot128-DF-*-plan-result.json` ont exit0, processus
joints, **0Swapout/Swapin** et maximums observés0, même admission10,5Go/ancrew2.
Les vérifications quiet sont non vides (43/41/42/41) ; leurs derniers snapshots
ne montrent ni Brave/Ollama/Deezer, ni compilateur ni autre moteur identifié.
Aux bornes : secteur100%, aucun avertissement thermique/performance enregistré.
Cela ne mesure pas les fréquences GPU/CPU ni les conditions à chaque token.

RSS enfants rapporté identique **6724009984B** pour les quatre ; métrique distincte
du footprint natif. Pic vie natif maximal3C **7818975448B**, contre
7804508280/7814830320/7805458648B pour1A/2C/4A. Pics MLX proches de7,289Go.
Pas de nouvelle allocation massive visible imputable à la queue.

Le compresseur système occupé augmente pendant le processus3C de
3800858624→4074586112B (+273727488B), contre une baisse pour les autres.
Ce sont des bornes du processus complet, incluant activité système/autres
processus ; aucun placement temporel sur le seul HOLD ni cause exclusive ne
peut en être tiré. Pas de temps par syscall, de file physique SSD ou de compteur
de contention dans ces reçus.

**Conclusion bornée : adaptation fidèle et correction qualifiée dans les domaines
exercés ; aucune panne concrète trouvée expliquant le HOLD3C. Le ralentissement
observé reste à conserver dans le bilan, sans l'attribuer arbitrairement à la
compression ni déclarer l'idée inutile.** Conserver un opt-in est compatible
avec cet état ; aucune activation générale, règle automatique par mode ou
suppression du candidat n'est décidée par cette revue.

Le bilan agrégé du coordinateur conserve **DFlash−1,6554%, dont HOLD−3,6347%**.
Ce résultat ne justifie pas une activation générale. Le code reste disponible
en opt-in, défaut Static ; les observations AR/MTP favorables gardent leur
portée propre. Audit clos, aucune nouvelle mesure demandée pour cette clôture.
