# MLXL3 — review 2 : DFlash, kernels et objectif 15–20 tok/s

**2 novembre 2026 — deuxième revue indépendante. La première revue SSD reste inchangée.**

## Verdict

**Oui, un vérificateur réellement optimisé pour plusieurs tokens peut accélérer DFlash sans améliorer sensiblement l’AR.** Le bon objectif est « plus de tokens acceptés par passage coûteux », pas « accélérer seulement le petit modèle draft ». Mais **15 tok/s est une cible à tester ; 20 tok/s n’est pas démontré par les mesures actuelles**.

La piste la plus intéressante n’est pas un kernel magique : c’est un **chemin B3/B4 cohérent** qui conserve le recouvrement SSD, partage davantage les tuiles EXL3 et évite de refaire les états récurrents. Aujourd’hui, élargir le bloc change plusieurs chemins défavorablement. Deux constats particulièrement actionnables :

1. **ColdReady est volontairement limité à deux lignes et désactivé avec les experts EXL3 groupés.** Passer à B3/B4 ne conserve donc pas les mêmes avantages que le B2 actuel.
2. **Le batch EXL3 de référence Flash partage ses tuiles par paires seulement**, même pour B3/B4/B8. Un bloc plus large ne signifie pas une seule déquantification pour toutes ses lignes.

Les kernels GDN, les experts groupés et QSA ont déjà un historique fourni. Je distingue ci-dessous les nouvelles interactions des anciens prototypes à ne pas recycler.

## 1. Périmètre et provenance

Revue statique de main **`d5a776e53425bb4d0ba61c0774c5399ae7987c06`**, dernière [issue #65](https://github.com/0xZKnw/mlxl3/issues/65), historique `opti.md` et rapports associés, notamment #59 et la revue64R. Les essais locaux récents non publiés ne sont pas assimilés à ce main. Aucun benchmark, accès au Mac, changement de code ou envoi à Codex dans cette revue.

Sources amont consultées :

- [DFlash officiel, revision07ebd93](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a), README et backend MLX.
- [Splash, revision031a25a](https://github.com/incoai/splash/tree/031a25a6125af2d7acf7096580afe56eb33d65bc), vérificateur Qwen4, GDN, kernels MoE et budget de vérification.
- [Checkpoint exact incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), et [présentation DFlash2](https://inco.ai/blog/dflash2/).
- [DFlash original](https://arxiv.org/abs/2602.06036), [DDTree](https://arxiv.org/abs/2604.12989), [DCOCO](https://arxiv.org/abs/2610.12570).

Le moteur local valide bien DFlash2 Flash-Next : H2560, cinq couches, captures5/15/25/35/45, masque248077, RoPE10000, sélecteur rang256/top16, convolutions dynamiques à deux taps. Ce n’est ni le DSpark voisin, ni le modèle dense Qwen27B des exemples MLX. [Métadonnées locales](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_draft_metadata.rs#L7-L120).

La fiche exacte Flash-Next utilise GB300, cible FP8, draft non quantifié et sampling ; B4 signifie trois propositions. Elle rapporte 2,83 à3,26 tokens utiles par vérification selon les tâches, et un plafond B4 lié à son implémentation QSA. **Cela prouve que la famille peut accepter plusieurs propositions, pas que le Q4/EXL3 local atteindra ces valeurs.** Le backend MLX officiel recommande des blocs≤5 avec quantification à cause de ses matmuls ; ce seuil n’est pas une loi pour les kernels EXL3 personnalisés.

## 2. Ce que 15 et20 imposent mathématiquement

### Noms et formule sans hypothèse d’indépendance

Ici **B = lignes vérifiées = une ancre + K propositions**, donc B2=une proposition, B3=deux, B4=trois. Dans MLXL3, `width=1` signifie B2, pas un bloc de deux propositions. Le réseau DFlash2 inclut lui aussi l’ancre dans ses lignes ; le head de propositions exclut cette première ligne. [Boucle de session](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_draft_runtime.rs#L96-L225).

Pour un bloc non tronqué en fin de réponse :

**L(B) = 1 + Σ(j=1..K) P(les j premières propositions sont toutes acceptées).**

**Débit = L(B) / [temps draft + vérification + commit + administration].**

On doit mesurer l’histogramme des longueurs de préfixes acceptés. Le taux80–84% de la première proposition ne dit pas que chaque proposition suivante a la même probabilité ; ne pas utiliser arbitrairement `0,84^j`. Pour le résultat final, utiliser total des tokens / total du temps, pas la moyenne de ratios par bloc.

### Référence publiée et budget

P09 release MAIN : AR11,487 ; DFlash12,463 tok/s. HOLD : DFlash11,919. Cache3GiB,53slots×48couches, quatre lecteurs, B2 fixe, draftQ4/ring, ColdReady activé. Le profil release MAIN décompose environ80,385ms/token : draft3,864 ; vérification74,927 ; commit1,582 ; résiduel0,012. Les agrégations ne sont pas strictement identiques au débit médian publié.

- Pour15 : budget66,667ms/token ; environ13,57ms à gagner par rapport au débit MAIN,17,23ms surHOLD.
- Pour20 : budget50ms/token ; environ30,24ms à gagner surMAIN.
- Même supprimer fictivement **draft et commit** laisse environ74,94ms/token, soit **13,34tok/s** dans cette décomposition. Il faut attaquer la vérification ou son amortissement.

MAIN compte58propositions acceptées pour69cycles : L≈1,841. À débit12,463, cela correspond à un ordre de grandeur148ms/cycle — rapprochement de mesures, pas nouvelle mesure chronométrée. Avec le même L,15 exige≤122,7ms/cycle et20≤92,0ms. Même une acceptation B2 parfaite plafonnerait autour13,54tok/s si ce coût de cycle restait inchangé.

| L réellement mesuré | Temps maximal du cycle pour15 | Pour20 |
|---:|---:|---:|
|1,80|120ms|90ms|
|2,30|153,3ms|115ms|
|2,60|173,3ms|130ms|
|2,80|186,7ms|140ms|
|3,00|200ms|150ms|

Exemple : B4 avec L=2,8 et180ms donne15,56tok/s ; le même bloc à220ms ne donne que12,73. **Ce sont des seuils de décision, pas des prédictions d’acceptation ou de vitesse.**

### Pourquoi on peut dépasser le plafond AR sans dépasser la bande passante physique

Si l’AR déplace W octets par token et dispose d’une bande passante effective β, son plafond simplifié vaut β/W. Un bloc qui déplace W_B octets et rend L tokens vaut au mieux β×L/W_B. Si W_B/L<W, le débit de tokens peut dépasser le plafond AR ; aucune mémoire ne transfère plus vite que β.

Sur un MoE, **W_B n’est pas constant** : chaque ligne peut choisir d’autres experts. Avec dix routes par ligne, l’union est au plus10B, et les activations/calculs restent propres à chaque occurrence. Les lectures Cold sont déjà dédupliquées par slot. Réutiliser un expert évite potentiellement une lecture, mais pas automatiquement tous ses produits. À53slots par couche, les grands blocs peuvent déclencher des cohortes ; le chemin `unique_cohorts` sait déjà exploiter l’union réelle. Ne pas supposer B8 toujours découpé5+3, ni inventer un taux de réutilisation. [Admission/cohortes](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L940-L1010).

Mesurer par couche et par bloc : union d’experts, multiplicité, uniques Cold, octets logiques, cohortes, attente exposée et tokens finalement gardés. Les octets `pread` ne sont pas une mesure des octets physiques NAND.

## 3. Priorité A — rendre B3/B4 compatible avec le pipeline ColdReady

**Fait, pas bug de correction :** `split_cold` impose `rows<=2`, `depth==0`, pas de prefetcher, pas de `shared_exl3`, pas d’épilogue pondéré, plus de deux jobs et un résultatHot. Le commentaire indique une qualification volontaire B1/B2 avant élargissement. Sinon, le code repasse par `fill_many`. [Garde exacte](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/ssd_moe.rs#L1224-L1265).

**Conséquence :** un écran naïf B2→B4 mélange l’effet d’acceptation avec la perte du découpage Cold W1/W2. Activer les experts groupés modifie également ce chemin. Ce sont des interactions importantes apparues après les anciens essais de largeur.

**Proposition :** construire un chemin B3/B4 borné qui partitionne les occurrences selon les slots réellement prêts, conserve toutes les routesHot/Cold et les leases, puis lance le calcul des occurrences prêtes pendant les lectures restantes. Variante suivante seulement : groupement EXL3 à l’intérieur de chaque vague, sans attendre artificiellement un expert appartenant à l’autre vague.

Ce n’est pas « retirer `rows<=2` ». Il faut auditer les maps, bornes20routes historiques, réutilisation des slots, consommation différée des arrays, erreurs partielles et barrière de fin. Le premier prototype doit garder exactement le même calcul EXL3 pour isoler le scheduling ; le deuxième seulement ajoute le kernel groupé.

**Test falsifiable :** à B3/B4 fixe, les traces doivent montrer une réduction de lecture exposée ou de latencecycle, pas seulement davantage de tâches lancées. Comparaison même nombre de tokens/acceptation, puis débit naturel. Si le GPU était déjà occupé pendant les nouvelles lectures masquées, le gain peut être nul. Dépendance avec la première revue SSD, sans la remplacer : ici l’objectif spécifique est préserver l’amortissement multi-token.

## 4. Priorité B — vrai kernel EXL3 Flash reference-MB3/MB4

**Écart précis :** `flash_reference_batch_layout` retourne toujours `(2,ceil(B/2))` pourB2..B8. `Exl3Linear::forward_qmv_batch` conserve NT1 et les réductions de référence. B4 utilise donc deux groupes de deux lignes. [Contrat](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/contracts.rs#L624-L688), [dispatch](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/linear.rs#L598-L710).

Le flag MB3 existant concerne certaines projections `input5120/output16384 ou34816` : **il ne couvre pas les projections Flash H2560**. Le MB4 particulier du FC MTP n’est pas un kernel Flash universel non plus.

**Kernel à expérimenter :** une déquantification/lecture de tuile EXL3 partagée entre trois ou quatre accumulateurs indépendants, tout en conservant l’ordre FMA et les réductions de chaque ligne. Cibles initiales : GDN qkv2560→10240, z2560→6144, out6144→2560 ; puis tête et projections réellement présentes dans les traces. Ne pas activer tous les formats/splits d’un coup.

Le shader générique possède déjà la dimensionMB, donc la première version peut être une spécialisation de géométrie plutôt qu’une réécriture. Mais sa mémoire threadgroup de référence croît avecMB : `MB×NT×SG×256` floats, soit8KiB àMB2/NT1/SG4 et16KiB àMB4, hors autres tableaux. Les accumulateurs et registres croissent aussi ; moins de relectures peut coûter davantage en occupancy. [Shader réel](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/shaders/_qmv_tile_kernel.metal#L14-L44).

**Pourquoi DFlash plus qu’AR :** le cheminB1 reste inchangé ; seules les vérifications élargies réutilisent les poids. Ce n’est pas les experts groupés65T : ici plusieurs tokens utilisent la même projection dense garantie, sans tri par expert. Cela ne promet pas une moitié du temps : seule une partie des lectures/déquantifications est amortie, les opérations parligne restent.

**Historique à respecter :** MB4 et blocs EXL3 ont déjà des échecs sur Qwen27B (`Q27-07/12`) ; ce n’est pas une invention universelle. Aucun gain Flash reference-MB3/4 n’est établi ici. Garder MB2 en contrôle et abandonner une géométrie qui régresse, même si le microbenchmark de déquantification est meilleur.

## 5. Experts EXL3 groupés : utile seulement sous conditions, déjà essayé

Le groupement par slot existe : seules les occurrences d’un même expert peuvent partager sa déquantification. Le chemin impose notamment K2/MUL1, slot-major et nonmasqué. [Implémentation](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/linear.rs#L1518-L1620).

65T a déjà essayé des groupes de multiplicité1..4 avec ordre conservé. 66F a ensuite testé B2/B4/B8 : plusieurs écrans régressent et certaines fenêtres dérivent ; pas de gain global qualifié. Par exemple les contrôles B4 étaient autour9,2tok/s, B8 autour6–7, bien loin d’une croissance automatique. Ces essais précèdent ColdReady actuel : **ne pas attribuer rétrospectivement leurs échecs à sa garde récente.** [Historique65/66](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L15900-L16120).

Nouvelle condition d’intérêt : groupe minimal construit dans la vagueReady courante, sélectionné par multiplicité observée et coût total ; uniques gardés dans le kernel simple. Le gain net doit dépasser tri/maps/gather/scatter, dispatch supplémentaire et perte éventuelle de recouvrement. L’inspiration [MoE Splash](https://github.com/incoai/splash/blob/031a25a6125af2d7acf7096580afe56eb33d65bc/runtime/metal/kernels/shared/moe_gguf.metal) adapte aussi les tuiles au nombre de lignes vivantes, mais ses poidsGGUF ne sont pas EXL3 : pas de port direct ni de pourcentage transférable.

## 6. GDN : fusion de vérification et commit différé, pas simple sigmoid

Le moteur groupe déjà les projections et utilise un scan multi-token avec **tape compacte**. Celle-ci conserve état initial, clés, deltaFP32 et decay ; un rejet partiel reconstruit le préfixe retenu dans un kernel de replay. Ce n’est donc pas un moteur qui recopie naïvement tout l’état à chaque token. [Tape/replay](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/gated_delta.rs#L262-L330), [préparation multirow](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/qwen4.rs#L353-L487).

Ce qui reste : fenêtres de convolution et certaines préparations/postnormes parligne ; replay distinct. Splash laisse un préfixe pending que le scan suivant incorpore en chargeant l’état, et fusionne préparation/scan/gate via mémoire threadgroup. [Kernel GDN amont](https://github.com/incoai/splash/blob/031a25a6125af2d7acf7096580afe56eb33d65bc/runtime/metal/kernels/decode/gdn.metal#L7-L95).

**Adaptation locale raisonnable :** fusionner le replay des delta/decay déjà calculés avec le scan suivant, sans recalculer les dot-products pending à la manière de l’autre format. Ajouter ensuite préparation causale B2..B4 et post-traitement si leur coût est exposé. Un état `[1,48,128,128]` F32 vaut3MiB ; les36couches GDN représentent108MiB de payload. Éviter une lecture+écriture séparée peut être intéressant, mais ce n’est ni un gain mesuré ni forcément du traficDRAM économisé.

**Déjà connu :** cette direction figure dans la [revue64R](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/docs/flash-next-ssd-splash-64-research-2026-10-21.md). La préparation mono-token25 a été rejetée ; T17 sigmoid/normgate a réduit les dispatchs mais n’a pas confirmé de boostDFlash. Je ne propose pas de les relancer inchangés.

Contrats difficiles : états exacts de tous les préfixes, convolution, fin de réponse, lecture d’état, passageB1/AR, abandon de vérification, erreur/reprise et durée de vie bornée des tapes. Splash calcule enBF16 avec ses propres normes ; Flash local impose ses arrondisFP16/L2. Une formule mathématiquement équivalente ne suffit pas.

## 7. QSA et coût de politique : ne pas perdre l’amortissement ailleurs

### QSA

`Layer::prepare_rows` conserve le chemin séquentiel pourQSA, même en multi-token ; les12couches ne bénéficient donc pas du même groupement queGDN. [Branche exacte](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/qwen4.rs#L1173-L1245).

Mais q/outMB2 a déjà été essayé dans ISSUE46-B1, validé numériquement puis retiré faute de gain stable. La piste distincte serait un blocQSA complet qui partage les opérations indépendantes/indexation, tout en conservant pooling causal, rangs ex æquo, masques et snapshots. Gros chantier, priorité après les deux verrous précédents ; pas un simple changement `for`→GEMM. [HistoriqueQSA](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/opti.md#L17751-L17828).

### Politique B2/B3/B4

Le budget adaptatif et un **precap avant draft** existent déjà. Sansprecap, certaines politiques calculent sept propositions puis en vérifient moins ; réinventer « choisir la largeur avant de drafter » serait recycler le code. [Budget actuel](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_draft_budget.rs#L1-L180).

En revanche, le coût du vérificateur est modélisé par une droite après apprentissage. Les ruptures ColdReadyB2/B3, groupesMB2/MB4 et cohortes peuvent rendre cette approximation inadéquate. Tester des coûts empiriques séparésB2/B3/B4, avec coûts de draft inclus, plutôt qu’un modèle toujours plus sophistiqué. `PreDraftBudget` a déjà un compteur par largeur : l’exploiter/comparer avant de le remplacer.

Ne pas prétendre connaître les routes des futurs tokens avant leur calcul. Les statistiques récentes de cache peuvent aider une estimation, pas fournir gratuitement l’union exacte future. Mesurer le regret de la politique contre les largeurs fixes sur plusieurs prompts, avec exploration bornée et sans surajustementMAIN/HOLD.

## 8. Draft : compléments peu coûteux, pas la solution entière

Le top16, le sélecteur DFlash2, le ring2048 et les convolutions dynamiques existent déjà. [Sélection](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_draft.rs#L560-L649).

Pistes complémentaires :

- B2 a une seule proposition ; les16prédécesseurs du sélecteur sont tous l’ancre. Spécialiser ce cas peut éviter des calculs identiques, mais ce manque est déjà noté dans79R. Préserver top16, ex æquo et arrondisBF16 ; le sélecteur Splash FP32 n’est pas identique.
- Le ring attention padde q/k/v jusqu’à huit lignes et produit un buffer huitlignes avant slicing. Une géométrie B2/B3/B4 compacte peut réduire padding/temporaire ; **ne pas annoncer8/B fois moins de calcul**, car `FLASH_RING_ROWS` masque déjà du travail. [Ring local](https://github.com/0xZKnw/mlxl3/blob/d5a776e53425bb4d0ba61c0774c5399ae7987c06/native/src/flash_draft.rs#L738-L811).
- Fusionner convolution dynamique + résiduel/norme ou regrouperQKV/GateUp peut réduire dispatchs. Les shapesH2560 et sorties640 des kernels dynamiques ne correspondent pas automatiquement aux wrappers EXL3 legacy alignés256.
- Un sélecteur toutGPU ne supprime pas les IDsCPU exigés par les embeddings/PLE sur disque. Ne pas promettre de boucle entièreGPU sans changer ce contrat.

Avec draft≈3,9ms/token dansP09, ces chantiers sont secondaires pour15, insuffisants seuls pour20.

## 9. Recherche plus exploratoire et qualité

DCOCO étudie la correction contextuelle entre candidats adjacents et DDTree des arbres de propositions. Ce sont des axes de recherche réels, pas des kernels plug-and-play pour ce checkpoint. DFlash2 possède déjà un correcteur entre candidats ; ajouter « un sélecteur » ne constitue pas une nouveauté. Un autre correcteur exige poids/entraînement compatibles et mesure d’acceptation. DDTree nécessite des états récurrents parbranche, des masquesQSA adaptés et davantage d’unions d’experts : la pressionSSD/3GiB peut annuler le gain d’acceptation.

Changer uniquement le draft peut conserver la qualité cible **si la vérification est correcte** ; modifier quantification, experts, attention ou états de la cible ne le peut pas implicitement. Le chemin local analysé utilise `chat_greedy_ids` et `greedy_accept` : son exactitudegreedy ne prouve pas un sampling spéculatif distributionnel. Pour le sampling, il faut le test d’acceptation et la correction de distribution appropriés, avec probabilités cohérentes ; ne pas remplacer cela par un simple argmax.

## 10. Plan de travail conseillé et critères d’arrêt

1. **Figer une baseline actuelle** incluant les éventuels correctifs locaux déjà qualifiés. B2/B3/B4 fixes, mêmes prompts/checkpoints, greedy,3GiB, mêmes paramètres, sans autre charge.
2. Mesurer L et histogrammes de préfixe, tempscycle, draft/verify/commit, union/multiplicitéCold, cohortes, formes/projections. Séparer trace intrusive et débitrelease ; ne pas additionner les temps inclusifs.
3. **PrototypeA : ColdReadyB3/B4 seul**, sans groupedEXL3. Oracle complet des48couches/états, erreurs et slots réutilisés ; puis comparaison chronologique symétriqueA/B/B/A avec contrôles de dérive.
4. **PrototypeB : Flash reference-MB3/4 seul**, formes réelles, B2 inchangé. Microbench seulement comme filtre ; validation finale sur débit, acceptation et états.
5. CombinerA+B seulement si chacun utile ; mesurer la combinaison, ne pas additionner les pourcentages. Puis GDNpending et groupementReady-aware selon le coût restant.
6. Politique adaptative et draft optimisé après stabilisation du coût des largeurs. Garder les options rejetées hors défaut.

Tests numériques : chaque préfixe retenu1..B, tous les logits pertinents, étatsGDN/QSA/PLE/hidden/offset, continuation après rejet, reset, fin de contexte, erreur et reprise. Conserver les réductionsEXL3 exactes ou annoncer explicitement un changement de contrat ; pas simplement vérifier que le texte d’un prompt paraît identique.

Tests mémoire :3GiBexperts inchangés, un modèle, aucun nouveauSwapout, contrôle de l’empreinte et des buffers temporaires/tapes ; pas de cache silencieusement agrandi. Le pleinB8 n’est pas prioritaire tant queB3/B4 ne paie pas.

**Décision15 :** le totalrelease doit réellement passer66,7ms/token sur plusieurs prompts représentatifs, pas seulement un kernel. **Décision20 :**50ms/token ; à L=2,8 cela exige un cycle≤140ms. Si les mesures montrent que le coût irréductible de vérification dépasse ce budget avec l’union d’experts observée, arrêter de promettre20 et identifier explicitement le changement matériel/format/modèle qui manquerait.

### Conclusion courte

Je miserais sur **ColdReady multi-lignes + EXL3 FlashMB3/MB4**, puis **GDNpending fusionné**, plutôt que sur le draft seul ou un B8 forcé. C’est une voie crédible pour que DFlash progresse beaucoup plus que l’AR. **15 reste un objectif raisonnable de campagne ;20 est un objectif conditionnel nécessitant simultanément une meilleure acceptation multi-token et une vérification nettement mieux amortie. Aucun des deux n’est acquis.**

