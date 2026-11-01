# Après F02 : choisir les essais à partir du moteur local

1 novembre 2026 — POSTP08-LOCAL-REVIEW-02. **Analyse effectuée, aucun nouveau
candidat moteur implémenté ni benchmark lancé.** Le comparatif F02 reste
interrompu pour ressources, flag OFF ; sa [situation exacte](../post-p08-dense-epilogue-20261031/README.md#résultats-modèle-inspectés-et-interruption)
est conservée. Cette revue indépendante avance pendant ce blocage et ne le
transforme pas en essai terminé. Aucun projet amont nouvellement consulté.

## Ce que le profil permet de viser

Fenêtre [P08r3](../decode-profile-20261031/natural-r3/attribution-partial/README.md) :
40 transitions AR, pas76–115,104,5704ms/sortie. Trace instrumentée partielle,
sans contrôle final : **ni un débit release, ni une mesure complète des modes
MTP/DFlash**. « GPU actif » désigne des intervalles de commandes, pas de l'ALU
utile sans stalls. Les axes CPU/GPU/I/O se recouvrent.

-36,7559ms d'attente des routes recouvrent36,6258ms de GPU moteur ; la copie
ne prend que0,004312ms. Supprimer le readback ne supprime pas ses producteurs.
-11,8539ms de CPU Running pendant GPU idle se trouvent dans les deux soumissions,
dont11,7676ms dans `slots.indices().async_eval()`. Cet appel soumet aussi Cold
précédent/combinaison/HC/attention/routeur : son nom ne localise pas le coût
au petit tableau des indices. Les piles situent graphe, encodeur et ressources,
sans donner de millisecondes exclusives à chaque fonction.
-27,1664ms de queue I/O conditionnelle suivent la dernière commande indépendante.
Ce sont des lectures POSIX, pas des lectures NAND identifiées. Elles ne sont
pas toutes supprimables et ne s'ajoutent pas aux11,85ms comme un gain acquis.

## Deux filtres supplémentaires sur les données existantes

[Résultat détaillé et empreintes](trace-filter.json). Lecture des XML originaux,
PID27154/TID291767, références XML résolues, fenêtre historique inchangée.
Le parseur/union d'intervalles existants sont réutilisés sans réexécuter leur
analyse productrice. Les5985 `preadv`,7 469 351 820octets et1696lots historiques
sont retrouvés ; chaque lecture appartient exactement à un lot et les retours
sont sans erreur. Aucune nouvelle capture ou I/O du modèle.

### Doubles contrôles de fichier : faible priorité

`fill_many` appelle `TensorInfo::source`, puis `Array::read_records` ; chacun
vérifie la longueur du fichier. Le code fait bien deux contrôles, mais leur
poids mesuré est très faible :

| Appel dans fill_many | Nombre | Somme wall / sortie | Portion sans aucune activité GPU observée / sortie |
|---|---:|---:|---:|
| TensorInfo::source → fstat |5985|0,093070ms|0,002816ms|
| Array::read_records → fstat |5985|0,033084ms|0,002340ms|
| Total |11970|0,126154ms|0,005155ms|

Le temps système `cputime` fourni par Instruments dépasse parfois `duration` ;
il reste brut dans l'artefact et **n'est pas une partition CPU valide**.
Ces durées ne comptent pas toute la préparation Rust/C++ et leur recouvrement
ne prouve pas la causalité. Même l'effacement fictif de tous les fstat ne retire
que0,126ms de leurs enveloppes dans cette fenêtre ; on ne supprime pas les
protections de troncature pour ce poste. Réutiliser des vecteurs reste une
hypothèse distincte, sans coût propre attribué.

### Experts Cold prêts avant la barrière du lot : crible positif, gain inconnu

1381/1696lots contiennent plusieurs jobs. Pour chaque lot, le premier retour
`preadv` réussi précède le dernier ; la somme de ces écarts vaut15,8480ms/sortie.
Après intersection avec la partie située après la dernière commande GPU
indépendante du lot, elle vaut **15,5326ms/sortie**. Calcul :
`max(0, dernier_retour - max(premier_retour, frontière_GPU_du_lot))`, puis somme
sur1696lots/division par40sorties. Calage historique ±1375ns conservé.

C'est une **avance de disponibilité avec l'ordonnancement figé**, pas un temps
GPU récupéré. Elle ne soustrait ni le coût des contributions Cold, ni les
nouveaux graphes/commits, ni la publication CPU/worker, ni la contention I/O/GPU.
Elle ne prouve pas qu'une première cohorte assez large soit disponible assez
tôt. Elle justifie un crible par groupes, pas encore un ordonnanceur complet.

## Classement des prochains essais

| Ordre | Essai local précis | Pourquoi | Plafond propre / risque |
|---|---|---|---|
|1, petit filtre|Préparer une seule fois les coefficients immuables de convolution GDN dans la forme/F32 déjà demandée, puis vérifier le travail réellement évité.|`Gdn::convolve` reconstruit ce cast à chaque token dans36couches. Aucun nouveau kernel nécessaire.|Coût propre non mesuré. Risque faible à moyen : conserver l'arrondi F16 initial, forme/strides, donation et mémoire.|
|2, potentiel plus large|Calculer une cohorte Cold complètement chargée pendant la fin des lectures du lot.|La barrière actuelle attend tous les jobs avant de construire Cold ; le nouveau crible montre une avance disponible.|15,53ms est une enveloppe de disponibilité, pas un gain théorique atteignable. Risque élevé : propriétaires, erreurs, réservations et soumissions supplémentaires.|
|3, diagnostic seulement|Amortir les descriptions et capacités de vecteurs des lectures.|Travail effectivement répété Rust/FFI/C++, mais non attribué en temps propre.|Inconnu ; le seul sous-poste fstat mesuré est trop petit pour être prioritaire. Conserver toutes les validations.|

Le classement choisit d'abord le filtre le moins coûteux ; il ne prétend pas
que les constantes GDN possèdent le plus gros potentiel. L'analyse indépendante
et les références source détaillées sont dans [independent-review.md](independent-review.md).

### Essai1 : conditions pour passer au prototype

Vérifier sur le GDN complet que le cast F16→F32 du coefficient crée réellement
du travail répétitif : distinguer simple vue, copie/cast, buffer réutilisé et
fusion MLX. Si aucun travail évitable n'existe, clore la piste. Sinon déplacer
exactement la préparation existante au chargement, en partant de
`half_weight` et non du checkpoint F32 d'origine. Le payload conv devient
5,625Mio au lieu de2,8125Mio pour36couches ; vérifier la libération de l'ancien
stockage. Ne pas mélanger initialement l'epsilon et d'autres optimisations.

Qualité : poids préparés exacts, GDN mono et multi-ligne2..8, états/histoires,
reset, répétitions/parents libérés, puis oracles modèle AR/MTP/DFlash et vrai
bridge. Un mutant utilisant le poids avant son arrondiF16 doit être détecté
par une fixture discriminante. Kani sur les invariants Rust touchés, limites
FFI/Metal explicites. Mesurer GDN entier (coûts de préparation inclus/exclus
nommés), puis pilote complet préfill/décode/RAM ; aucune activation sur la seule
baisse du nombre de dispatchs. Ce protocole est une proposition : **aucun
nouvel essai exécuté ou préenregistré comme démarré**.

### Essai2 : conditions avant architecture asynchrone

Mesurer l'avance d'une cohorte de taille fixée et le coût complet de son
calcul/soumission avant de réécrire les lecteurs. Limiter les vagues plutôt
qu'émettre un dispatch par expert. Conserver l'ordre canonique token/rangTop10
pour l'assemblage et la somme stock ; le changement de taille ne doit pas
modifier le kernel ou son ordre numérique sans qualification.

Publier après les12segments complets de chaque expert, conserver banques et
réservations jusqu'à la fin des consommateurs, y compris si un autre job échoue.
Le champ `last` unique et `ExpertFill` doivent couvrir toutes les vagues en vol.
Aucun appel MLX dans les workers. Tests de retard ciblé, échec/troncature,
annulation, duplication d'ID, mixedHot/Cold, cache miss/reprise, puis états
et logits exacts et vrai débit. T18a modifie l'attribution des lectures mais
conserve la jointure globale : il ne teste pas ce mécanisme T18b.

## Pièges écartés

- Fusionner les deux soumissions : POSTPROFILE03 a retardé les routes derrière
  Hot et échoué au test d'overlap. Conserver l'événement précoce des routes.
- Avancer shared :73T déjà essayé ; le voir tard dans le code ne le rend pas neuf.
- Refaire un cache de factories :T01A coûte peu ; T01B pendant eval non mesuré.
- Préfill naïvement groupé/réordonné : P01/P01B/P01L déjà qualifiés puis retirés.
  P01L a augmenté le TTFT de10–13% avec mêmes I/O et métadonnées LFU ; les
  synchronisations supplémentaires restent une explication plausible non isolée.
  HEADONLY, déjà retenu, supprime les têtes intermédiaires inutiles.
- Assimiler le routage spéculatif aux36,8ms : le préfixe différable observé ne
  recouvrait que0,38–0,47ms de GPU idle, rollback et reconstruction à payer.
- Importer ARC/pool ou un réglage de lecteurs sans bilan : historiques négatifs
  ou mixtes conservés. Une lecture supplémentaire peut théoriquement aider,
  mais seul le temps complet après coûts/misses/pression mémoire le décide.
- Étendre F02 parce que8→5dispatchs : sa correction passe, son gain AR reste
  non concluant et MTP/DF incomplets. Aucun pourcentage additionné à cette revue.

## Vérification et état

Lecture des chemins production, historiques et profils, recalcul des comptages
et fenêtres, contrôle de636sources/4binaires F02 : inchangés. Aucun code exécutable
ni CI modifié par cette revue ; pas de nouveau build/test GPU/Kani nécessaire
pour ces seuls documents. Les vérifications antérieures F02 restent attachées
à leurs sources et domaines, sans devenir une preuve de ces pistes futures.
Le [recalcul indépendant](trace-filter-independent-audit.md) reproduit exactement
les résultats ; son premier filtre incorrect, corrigé, reste conservé.
[Contrôles finaux](final-checks.json) : liens locaux et identités conformes,
`git diff --check` réussi. Aucun benchmark natif nouveau ni pourcentage de boost validé.
