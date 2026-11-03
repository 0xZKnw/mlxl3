# GDN-CONST-01 — audit d’adaptation et du micro

1 novembre 2026. **Aucun défaut numérique ou de branchement concret trouvé.**
Le prototype est qualifié dans le composant ; le micro montre des signaux
modestes et un B1 très dispersé. Aucun gain modèle, dispatch supprimé ou temps
GPU exclusif n’est établi. Revue et calculs CPU sur les données existantes
uniquement ; aucun nouveau test, build, GPU ou changement de source surveillée.

## Adaptation réelle

- `Gdn::load` ([qwen4.rs:212](../../../native/src/qwen4.rs#L212)) part toujours
  de `half_weight`, donc du coefficient **arrondi F16**, puis de sa vue
  `[10240,4,1]`. `prepare_conv` applique exactement la transpose `[2,1,0]`,
  le cast F32 et `eval` une fois. Aucun raccourci depuis le checkpoint F32.
  Le flag strict est lu à load, pas à chaque forward ; défaut OFF.
- `convolve` ([qwen4.rs:326](../../../native/src/qwen4.rs#L326)) prend le
  coefficient préparé en Some, ou reconstruit la chaîne stock en None.
  Produit, réduction, store F16, SiLU, Q/K, epsilon F16 et récurrence restent
  stock. Mono et toutes lignes de `forward_rows` passent par ce même helper.
  Aucun readback, synchronization globale, stream ou accès SSD ajouté au
  forward. L’évaluation supplémentaire appartient au chargement ON.
- Some fait un `Array::try_clone`, qui appelle
  [mlx_bridge.cpp:184](../../../native/ffi/mlx_bridge.cpp#L184) :
  `new mx::array(arr(p))`. C’est un wrapper hôte et une référence supplémentaire,
  **pas une copie des données GPU**. Son coût est inconnu. Un emprunt en Some
  pourrait l’éviter, avec temporaire seulement en None ; ce serait un changement
  prospectif, pas un défaut expliquant déjà les résultats. Je ne l’ai pas fait.
- Les propriétaires Array restent vivants dans Gdn et dans les dépendances
  MLX ; aucun pointeur brut ni cache global. `reset` efface histoire/état,
  conserve le coefficient immuable. L’ancien F16 est volontairement gardé :
  **5,625 Mio F32 supplémentaires pour 36 couches**, pas le net 2,8125 Mio.
- **Strides non mesurés** : la forme logique préparée est `[1,4,10240]`, mais
  le cast MLX peut conserver un ordre col-major. L’API Rust actuelle ne donne
  pas les strides. Les deux branches exécutent la même préparation à des
  moments différents ; les 1474560 valeurs des 36 poids sont exactes face au
  transpose/conversion CPU. Cela valide les valeurs, pas un layout physique
  identique ni un dispatch autonome. Ne pas ajouter `contiguous` sans preuve.

## Le micro compare bien les deux branches

Le helper `pair` charge deux GDN sous flag ON, vérifie Some, puis met
explicitement le stock à None ; les poids F16 stock sont conservés. T17 est
forcé false dans les deux, F02 et T17 sont aussi contrôlés OFF par environnement.
Les instances de poids sont distinctes mais leurs valeurs proviennent du même
checkpoint. Ce protocole compare l’exécution stationnaire, **pas le chargement
OFF contre ON**, ni le surcoût de RAM/TTFT du modèle complet.

Les entrées, coefficients et un état/histoire non vide après deux pas sont
évalués avant mesure. Chaque bras repart des mêmes snapshots retenus ; leur
clone/restauration est hors chrono. Trois chauffes par bras et largeur,
12 rounds AB/BA, 16 forward+eval par échantillon. `StageRetention(false)`
prévient la rétention des stages ; leur vecteur vide est vérifié. Les outputs
intermédiaires sont évalués et libérés, seuls ceux du seizième forward sont
gardés pour comparaison. Bytes/finitude/état final sont contrôlés hors chrono.
La qualité séparée couvre les trajectoires et préfixes, sans prétendre que le
micro compare les bytes des seize états intermédiaires de chaque échantillon.

Le chrono inclut construction, soumission, attentes et destruction des tapes
multi-lignes ; il n’est pas un graphe naturel de 36 GDN entrecoupés de MoE/I/O.
Les snapshots retenus affectent potentiellement la donation dans les deux
bras. La restauration et les readbacks hors chrono sont symétriques, mais
peuvent influencer caches/thermique du passage suivant. Le JSON `complete` est
créé seulement après toutes les validations ; aucune sortie partielle n’est
promue comme complète.

## Recalcul indépendant des 96 échantillons

[micro.json](micro.json), SHA256
`0fa2460af76b8cfed8ab6a8a9fa72d197b6667354c0f85316cbfcc29e2515e1e`.
Les 96 clés `(B,round,bras)` sont uniques, tous les temps finis et positifs,
chaque paire contient OFF/ON dans l’ordre prévu ; 1536 forwards et 24 chauffes.
Aucun échantillon exclu. Moyennes : somme des durées divisée par 192 forwards
par bras/largeur. Débit agrégé : `ΣtempsOFF/ΣtempsON−1` à travail égal.

| B | OFF µs/forward | ON µs/forward | Débit agrégé | Médiane des 12 ratios appariés | Paires ON plus rapides |
|---|---:|---:|---:|---:|---:|
|1|762,376|734,113|+3,8499 %|+0,9362 %|7/12|
|2|801,189|799,403|+0,2234 %|+0,2967 %|8/12|
|4|1310,059|1307,509|+0,1950 %|+0,5492 %|9/12|
|8|2323,042|2310,500|+0,5428 %|+0,6573 %|11/12|

B1 varie de 612,289 à 1409,651 µs OFF et 608,581 à 1496,945 µs ON ; ses ratios
appariés vont de **−21,19 % à +71,43 %**. La médiane des six premières paires
vaut −2,564 %, celle des six dernières +1,452 %. Les deux bras connaissent
des passages lents, qui restent inclus : +3,85 % n’est pas un gain stable
qualifié. B2/B4/B8 ont de petits signaux positifs ; aucune addition entre B,
aucune multiplication par 36 couches ni extrapolation en tok/s.

## Prochain contrôle utile

Une capture OFF/ON bornée est justifiée pour **vérifier le mécanisme**, pas
pour retester les chronos jusqu’à disparition de la variance. Le plus petit
cas est le vrai `Gdn::convolve` B1 : qkv et histoire identiques déjà évalués,
coefficient préparé évalué avant capture, outputs conv/Q/K/V exacts après.
Une sous-chaîne ainsi capturée doit être nommée comme telle ; elle ne compte
pas les dispatchs du GDN complet. Inspecter casts, buffers/layout et éventuels
travaux de matérialisation avant d’affirmer que le cast récurrent disparaît.
Ces captures **ne sont pas encore faites**. Même si elles confirment un
dispatch en moins, les oracles complets et le vrai bridge précèdent toute
mesure modèle ; le gain naturel peut être partiellement masqué par l’I/O/GPU.
