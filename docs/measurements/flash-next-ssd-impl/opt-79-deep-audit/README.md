# Audit Flash Next vers 15 tok/s — 79R

MLXL3 exécute Qwen3.8 Flash Next EXL3 2,05 bpw avec des experts SSD, un cache borné et des kernels Metal via MLX. Cette revue suit une conversation sur le M5 10 cœurs / 24 Gio, avec les applications de l’utilisateur ouvertes, cache experts 3 Gio et allocateur 64 Mio. Elle concerne le moteur Flash Next et ses utilitaires partagés, pas une optimisation du modèle LFM2.

**15 constats concrets sont documentés, plus deux diagnostics annexes. Aucun nouveau gain n’est mesuré.** Trois agents ont suivi les appelants réels et les historiques ; le parent a vérifié les constats prioritaires, les métadonnées du checkpoint et les sources amont. Le bilan précédent, retrait de weighted compris, est publié dans la PR34 au commit88cbfd22. Les sources exécutables et la CI de ce checkout restent celles de89174fcc ; les prototypes72/73 ne sont pas réactivés.

## Ce que les données imposent comme priorité

Les chiffres suivants viennent des deux témoins sans prélecture du vrai A/B72W-R, pas d’une nouvelle campagne. Les phases sont des durées hôte inclusives, pas des temps GPU exclusifs ; des travaux paresseux peuvent être payés à une frontière ultérieure.

| Témoin | Débit | Draft / décodage | Vérification / décodage | Latence à économiser pour15 tok/s |
|---|---:|---:|---:|---:|
| MAIN | 11.301 tok/s | 4.59% | 93.38% | 21.82 ms/token |
| HOLD | 10.717 tok/s | 4.46% | 92.69% | 26.65 ms/token |

À acceptation et autres coûts inchangés, rendre tout le draft gratuit ne donnerait ici qu’environ +4,7 à +4,8% de débit. Il faudrait encore retirer environ26 à31% du temps de vérification enregistré pour atteindre15 tok/s si les autres postes restaient fixes. Ce calcul idéal oriente le travail ; ce n’est ni une prédiction de gain ni une preuve de décomposition GPU. Les améliorations du sélecteur seules ne suffiront donc pas. [Calculs et données sources](phase-and-compression.json).

## À corriger : qualité du diagnostic sur erreur

1. **Les octets d’un batch synchrone partiellement échoué ne sont pas comptés.** Le compteur ne monte qu’après succès de tout le batch, alors que certains lecteurs peuvent avoir déjà terminé. Remonter les octets réellement lus avec le statut d’échec, ou préciser le nom et le contrat du compteur. Sans cela, le diagnostic d’une reprise sous-estime l’I/O ; ce point n’accélère pas les réponses réussies. [79B-07](ssd-memory.md#7-les-octets-synchrones-dun-batch-échoué-disparaissent-des-compteurs-moteur).

## À optimiser : travail démontré, gains à mesurer

| N° | Constat et scénario | Plus petit mécanisme à tester | Conséquence si laissé tel quel | Preuves / historique |
|---:|---|---|---|---|
|2|En QSA court, seules128 des640 sorties de l’index sont utilisées ; le cacheMTP n’utilise jamais les512 autres.|Projection EXL3 limitée à512..640, mêmes strides et sommes ; conserver toutes les clés pour le passage à2052.|32/40 tuiles de cette petite projection restent calculées inutilement.|[79C-01](kernels.md), piste nouvelle ; OFF/MTP/DFlash.|
|3|Chaque appel EXL3 reconstruit256 chaînes de positions et les CString MSL avant d’atteindre le cacheC++.|Mémoriser d’abord le préfixe de réduction immuable, octet par octet ; mesurer le tempsRust exposé.|Le cache de factories ne supprime pas ce formatage déjà payé côtéRust.|[79C-03](kernels.md), distinct du profil19 et des signatures67.|
|4|Les portes GDN a/b denses48 sont calculées ligne par ligne sur36 couches.|Brancher étroitement le helper dense multi-lignes existant après oracle48×2560.|72(B−1) appels de projection et des concaténations persistent par fenêtre ; pas autant de soumissions GPU prouvées.|[79C-02](kernels.md), non couvert par le regroupement EXL3 de63.|
|5|B2 mixte reconstruit Hot/Cold par concat puis gather.|Copieur direct gardant le snapshot initial et l’ordre des10 contributions.|Des lignes masquées sont copiées puis jetées ; coût non mesuré.|[79B-02](ssd-memory.md), backlog74 A06, pas une découverte nouvelle.|
|6|L’entrée routéeFP32 est dupliquée10fois avant conversionFP16.|Convertir lesB lignes avant le gather ; aucun changement de réduction.|PourB2, grand intermédiaire nominal204800B au lieu de10240B avant gatherFP16.|[79B-01](ssd-memory.md), petite piste nouvelle ; réduction du pic global non démontrée.|
|7|Un batch deNexperts froids demande2N métadonnées sur les handles ouverts.|Dédupliquer les contrôles dans un batch en gardant inode, bornes et validation avant écriture.|Préparation CPU répétée ; la suppression naïve casserait la détection de troncature.|[79B-03](ssd-memory.md), backlog74 précisé ; maximumB2 entièrement froid1920 appels/48couches.|
|8|NeufVecRust et des listesC++ sont reconstruits par batchCold.|Réutiliser uniquement les capacités et métadonnées stables, avec invalidation si banques changent.|Allocations hôte répétées ; aucun coût exclusif nouveau connu.|[79B-04](ssd-memory.md), backlog74.|
|9|Les lecteurs synchrones ont une affectation statique des records.|Index de prochain travail à nombre de lecteurs inchangé, si les traces prouvent une traîne.|Un lecteur lent peut retenir le dernier job pendant que les autres sont libres.|[79B-05](ssd-memory.md), backlog56 ; pas simplement davantage de threads.|
|10|En mode fixe, les confidences ne pilotent ni choix ni apprentissage, mais sont relues surCPU.|Supprimer d’abord ce retourCPU en gardant la validation GPU de la probabilité sélectionnée.|Une sortie de diagnostic et ses constructions restent payées par bloc.|[79R-A01](verifier.md), nouvelle redondance ; ne pas supprimer le contrat d’erreur.|
|11|En singleton B2,16 prédécesseurs identiques produisent16 lignes de scores ; seul rang0 est accessible.|Spécialisation singleton conservant transitionMarkov, arrondis et tie-break.|16× plus de termes dans ce seul petit sous-calcul ; aucun facteur16 sur le draft complet.|[79R-A02](verifier.md), nouvelle redondance ; GEMM→GEMV peut diverger.|

Les premiers essais rentables à préparer sont2/3/4/5 : ils touchent le chemin cible, majoritaire. Le point6 est un petit correctif de graphe simple à qualifier. Les points10/11 sont précis mais leur gain global est plafonné par le faible poids du draft. Ce classement est une proposition de tests, pas un classement de gains établis.

## Pistes secondaires ou conditionnelles

| N° | Travail présent et scénario | Mécanisme minimal / limite | Preuves |
|---:|---|---|---|
|12|Le même masque BF16 est lu/alloué par proposition.|Une ligne de5120B conservée et diffusée ; préciser identité du checkpoint et invalidation. Ne pas réutiliser l’embedding publicF16.|[79R-A03](verifier.md), coût probablement faible.|
|13|RingB2 est paddé à8positions et garde8splits, dont6vides aux contextes≤128.|Grille des splits actifs en conservant indices et ordre logique8 ; ne jamais lire les partials non écrits. La compaction de lignesQ est un autre essai.|[79C-05](kernels.md), cible le draft seulement.|
|14|En QSA>2051, les anciens blocs d’index sont recastésF32 à chaque token.|Vue dérivée mise à jour après nouvel arrondiF16, invalidée lors des restores ; ajoute3–6Mio à ces contextes.|[79C-04](kernels.md), inactif sur les prompts courts.|
|15|Prélecture4workers et demande4lecteurs peuvent coexister.|Budget commun seulement après preuve de contention ; ne pas annuler les écritures encore en vol.|[79B-06](ssd-memory.md), inactif sans prélecture, ne justifie pas réactiver72.|

## Diagnostic annexe : compression du moteur, sans attribution de latence

Les témoins sans prélecture se terminent avec environ3,036Go MAIN et1,752Go HOLD de mémoire logique compressée propre au moteur. Les deltas warmup terminé→mesure terminée sont30726 et30635 décompressions, avec ledgerSwapins logique inchangé. **Cet intervalle comprend le préfill, la génération et le travail de requête ; il ne permet pas de compter les décompressions du seul décodage.** Les candidats weighted ont des deltas proches30697/30923, même quand le débit baisse ; les comptes ne démontrent donc ni une cause unique ni un coût constant. Les caps77B restent échoués.

Un futur diagnostic peut prendre un relevé propre à la frontière préfill/décodage, sans evalGPU ajouté, puis séparer chargement, préfill et décodage. Aucune fuite, aucune écriture swap privée et aucune durée GPU ne sont déduites de ces compteurs. Ne pas réduire le cache ni forcer le câblage mémoire sans test discriminant. [Données et limites](phase-and-compression.json).

## Splash et Strata : ce qui se transpose vraiment

- Splash main3135ab45 reste inchangé depuis72 ; les quatre commits récents de Strata vers7ddc17a3 concernent packaging/tests. Les SHA, dates et URL primaires sont dans[upstream.json](upstream.json).
- Les fenêtres réutilisables de[Strata](https://github.com/Niko1221/Strata/blob/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec/metal/src/engine/window.cpp#L265) et les étapes hôte/événements de[Splash](https://github.com/incoai/splash/blob/3135ab45b3da525a8a2dbb8a1aa72ce095dda8a9/runtime/metal/CommandGraph.hpp#L63) restent des écarts d’architecture connus. Une première suppression des métadonnées immuables n’exige pas de réécrire ce moteur. Un port de graphe natif devra démontrer le coût hôte exposé, préserver les pointsSSD et les protections des slots.
- [Splash569](https://github.com/incoai/splash/pull/569) est ouvert et vise Apple9/M4, GGUF ; Apple10 reste inchangé. Aucun boost directM5/EXL3 à récupérer de ce patch.
- [Strata3299](https://github.com/Niko1221/Strata/pull/3299), ouvert, porte d’autres formats trellisGyro. L’idée de table exacteMul1 locale porte sur1021half, environ2042B, pas un requant8bit. C’est une annexe exploratoire à tester exhaustivement sur65536états et à microbenchmarker ; un lookup peut être plus lent que le calcul actuel.
- [Strata3882](https://github.com/Niko1221/Strata/pull/3882) retire sa propre comparaison DFlash/MTP invalide, ne teste pas Metal et signale une dépendance de stream manquante. Il ne fournit aucun résultat transportable à notre M5.

## Contrôles, limites et décision

Les domainesA/B/C ont leurs scénarios, sources et SHA, contrats, correctifs minimaux et tests discriminants dans[verifier](verifier.md),[SSD/mémoire](ssd-memory.md) et[kernels](kernels.md). [La revue croisée](cross-review.md) a cherché des contre-exemples : ne pas choisir Hot d’après un hit devenu vrai après le snapshot ; ne pas supprimer la validation de confiance ou la transitionMarkov ; conserver les confiances en adaptatif singleton.

Les suites et preuves historiques ne valident pas les futurs changements. **Aucun test, benchmark, build, modèle GPU ou proveur nouveau n’a été lancé.** Les contrôles de cette revue sont les lectures source, la vérification des appelants et des formes, les hashes, l’analyse des rapports déjà publiés et les métadonnées GitHub primaires. Les quelques tenseursSuh lus surCPU sont petits et leurs hashes montrent que sharedgate/up, GDNqkv/z et QSAq/k/v ne peuvent pas partager naïvement le même Hadamard d’entrée.

L’historique négatif est conservé : F_NOCACHE12, HC22, préparationGDN25, variantes54, grouped65/66, shared68, prélecture72 et earlyshared73 ne sont pas de nouveaux essais à relancer. Le cache PLE déquantifié, le commitGDN différé, HC/PLE groupés, les10evalKV et les têtes intermédiaires de préfill sont des reliquats déjà identifiés.71PHotB2 et78constantes restent préparés/non appliqués. Aucun gain nouveau n’est livré, aucune application installée remplacée.

Verdict : commencer par retirer du travail cible mesurable, puis compléter avec les économies du draft. Les15tok/s restent un objectif non atteint ; cette revue ne démontre pas que les pistes suffiront. Le périmètre lu est le chemin Flash Next/SSD/DFlash et ses utilitaires, pas un audit de sécurité de toute l’interface, des téléchargements ou des autres modèles.
