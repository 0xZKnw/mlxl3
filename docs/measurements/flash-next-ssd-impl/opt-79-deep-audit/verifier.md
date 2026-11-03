# 79R-A — audit statique du vérificateur et du draft DFlash2

La revue isole **trois redondances nouvelles**, dont deux dans la voie fixe B2 courante. Aucune n’est un gain mesuré. Le mécanisme le plus simple à tester d’abord est la suppression du retour CPU des confiances en mode fixe, en préservant son contrôle d’erreur. Le second réduit le calcul Markov singleton ; le troisième évite de relire une petite ligne BF16 constante.

Préenregistrement : `opti.md:14960–14965`. HEAD documentaire `88cbfd2232ab23a0a7ca1d8dc312792347ca9d31`, runtime de référence `89174fcc`. Worktree : `/Users/justin/.codex/worktrees/flash-splash60/mix-stq1_0`. Cible EXL3 2.05bpw65c89531 et draft Q4a22a1744 repris du journal ; aucun poids chargé.

## Entrée réellement utilisée

- main.rs:3235-3252 validation greedy/configure_dflash2
- flash_draft_runtime.rs:14-55 Options::read : WIDTH1/ADAPTIVE0 => fixe B2
- main.rs:1827-1831 prefill_round => prefill_dflash2
- flash_draft_runtime.rs:285-310 reset et préfill capturé
- main.rs:2508-2537 => advance_dflash2 avec reste effectif
- flash_draft_runtime.rs:312-323 => Session::advance
- flash_draft_runtime.rs:131-159 draft_input/forward/select/verify_draft
- flash_draft.rs:711-852 head BF16/sélection ; flash_draft_top16.metal:1-12 validity
- flash_draft_runtime.rs:163-188 acceptance/commit/append ; confidence observée seulement en adaptatif

B2 signifie ici une proposition (`WIDTH=1`) plus l’ancre vérifiée : `ADAPTIVE=0`, Q4/ring selon la configuration existante. Les sélecteurs test-only ne remplacent pas le parcours de production.

## 79R-A01 — La voie fixe B2 calcule une confiance qui ne pilote aucune décision

**Statut :** redondance démontrée ; gain non mesuré. Priorité 1.

**Nouveauté :** 61/64 ont décrit le sélecteur et ses readbacks ; la dépendance conditionnelle ADAPTIVE=0 n’y était pas isolée. Les essais63/68/70/72/73 ne testaient pas ce mécanisme.

**Scénario :** Flash DFlash2, MLXL3_FLASH_DRAFT_ADAPTIVE=0, WIDTH=1 pour B2 ; également toute largeur fixe1..7. PRECAP ne peut pas être actif dans cette configuration.

**Mécanisme établi par lecture :** scores.argmax choisit les rangs et IDs. scores.softmax_precise et les gathers de probabilités ne nourrissent que confidence. En fixe, runtime conserve seulement le contrôle de longueur ; ni choose ni observe ne lisent les valeurs. Le sélecteur impose pourtant un second to_f32 après le to_u32 des tokens/validité.

**Plus petit changement à étudier :** D’abord demander une sortie sans valeurs de confiance au seul appel fixe : conserver exactement la probabilité sélectionnée et sa validation sur GPU, emballer son indicateur UInt32 avec tokens/top16.valid dans le readback existant ; supprimer seulement le to_f32/copie hôte. Les gathers encore nécessaires à cette validation restent présents. Une seconde expérience, distincte, pourrait supprimer softmax et ses gathers si elle préserve exactement le contrat de validité.

**Contrat et risques :** Ne pas supprimer simplement le contrôle invalid confidence : top16.valid ne valide que les candidats avant les projections Markov. Ne pas remplacer le contrat par tous les scores finis : des -Inf ou des lignes de prédécesseurs non consultées peuvent être compatibles avec une probabilité sélectionnée valide. Pas de changement de précision BF16/F32, ordre du classement ou tie-break.

**Preuves source :**

- [native/src/flash_draft_runtime.rs:24–35](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft_runtime.rs#L24)
- [native/src/flash_draft_runtime.rs:131–146](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft_runtime.rs#L131)
- [native/src/flash_draft_runtime.rs:180–197](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft_runtime.rs#L180)
- [native/src/flash_draft.rs:775–794](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft.rs#L775)
- [native/src/flash_draft.rs:822–850](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft.rs#L822)

**Contrôles discriminants proposés, non exécutés :**

- Comparer ancien/nouveau sélecteur fixe pour counts1/2/3/7, Q4=0/1, logits et hiddens non nuls, égalités, strides et ancres limites : mêmes propositions exactes ; voie adaptative conserve confidences et choix exacts.
- Réutiliser oracle scalaire splash_confidence_matches_independent_markov_scores et oracle couplé tête/backbone, puis36 commits/logits248320/captures64000/136 états, bridge43 et annulation/reprise sur la voie fixe réellement appelée.
- Mutation : logits top16 tous finis, mais NaN/+Inf dans un codebook/projection consulté => ancienne et nouvelle voie doivent refuser/reset ; tester séparément -Inf admissible et corruption d’une ligne non consultée.
- Instrumentation test-only du vrai sélecteur : mutation réactivant le to_f32 en fixe doit échouer au contrôle du nombre de readbacks, sans se contenter de sorties identiques.

**Portée performance :** Aucun coût exclusif ni TPS mesuré ; une seule frontière hôte réellement bloquante n’est pas déduite de deux appels. À tester isolément avant toute combinaison.

## 79R-A02 — Le premier pas Markov traite16 prédécesseurs identiques alors qu’un seul est accessible

**Statut :** calcul et gather redondants démontrés ; gain et parité du futur kernel non mesurés. Priorité 1.

**Nouveauté :** Le sélecteur générique existe depuis43/60 et les comparaisons61/64 sont connues. Cette spécialisation count1/ligne0 ne répète ni grouped EXL3 rejeté ni fusion router-GEMV.

**Scénario :** Tout bloc DFlash commence par l’ancre ; en B2 count=1, ce cas est l’ensemble du sélecteur Markov.

**Mécanisme établi par lecture :** predecessors commence par anchor broadcast[1,16]. Le gather donne16 vecteurs BF16 identiques de256 éléments. pred*projection puis matmul produit16 lignes identiques de16 scores, mais rank commence à0 et ne lit que maps[0,0] et probabilities[0,0,:]. Pour count1, le produit16x256x16 comporte65536 termes contre4096 pour1x256x16 ; la multiplication élémentaire a4096 sorties contre256. Ce rapport16 porte uniquement sur ce sous-calcul, pas sur la tête vocabulaire ni le draft complet.

**Plus petit changement à étudier :** Spécialiser seulement rows==1 : gather d’un seul predecessor anchor, produit BF16 avec projected[1,256],16 scores successeurs, argmax/tie-break identiques ; conserver les voies rows>1 et adaptatives comme témoins. Une optimisation ultérieure de la première ligne des grandes fenêtres est distincte.

**Contrat et risques :** La transition Markov reste indispensable : ce n’est pas un argmax brut des logits. Réduire la forme matmul peut changer le kernel et l’ordre de réduction FP32 avant l’arrondi BF16 ; l’égalité des lignes mathématiques ne suffit pas à prouver une implémentation bit exacte. Conserver les casts BF16 puis F32 et la validation des probabilités de A01.

**Preuves source :**

- [native/src/flash_draft.rs:748–794](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft.rs#L748)
- [native/src/flash_draft.rs:963–1005](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft.rs#L963)
- [native/src/flash_draft_splash_tests.rs:163–243](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft_splash_tests.rs#L163)

**Contrôles discriminants proposés, non exécutés :**

- Oracle scalaire existant avec Q4=0/1, rows1, codebooks non nuls et logits non uniformes ; étendre les cas de tie-break/ancres0 et248319.
- Comparer les16 scores réellement utilisés, proposition et confiance à la première ligne du chemin générique, puis aux mêmes hiddens couplés de production ; vérifier formes, compte et finitude.
- Mutation d’ancre ou suppression de la composante Markov dans une fixture construite pour inverser le gagnant doit être détectée.
- Oracle cible séquentiel, snapshots et bridge43 ; micro séparé de DFlash B2 MAIN/HOLD, sans mélanger A01.

**Portée performance :** Non mesuré ; les256 scores sont petits devant la projection sur248320 tokens et les48 couches cibles.

## 79R-A03 — La même ligne BF16 du masque est relue et réallouée à chaque proposition

**Statut :** I/O logique et allocation répétées démontrées ; impact probablement limité, non mesuré. Priorité 3.

**Nouveauté :** 72R a vérifié l’identité/dtype du masque et exclu le bug GGUF Splash4dd195e. Ce constat concerne la répétition de notre pread BF16, pas la quantification du masque.

**Scénario :** Chaque Session::advance non terminal avec count>=1 ; même ligne kind.mask()248077 pour chaque ligne après l’ancre.

**Mécanisme établi par lecture :** draft_input boucle count+1 fois et appelle Array::from_file pour chaque ligne. La FFI alloue puis exécute pread_exact ; ce n’est pas une vue mmap ou un cache de lignes. Le masque identique coûte count lectures logiques de5120 octets et count allocations ; les blocs suivants recommencent. Le cache OS peut supprimer l’accès physique au SSD mais pas ces appels et allocations.

**Plus petit changement à étudier :** Conserver une unique ligne BF16 du masque pendant la vie du modèle/session compatible et la diffuser vers les count positions. Continuer à lire l’ancre variable, conserver le dtype BF16 et l’identité exacte du fichier/offset ; invalider avec le remplacement de checkpoint/session selon le contrat existant.

**Contrat et risques :** Il n’y a ici aucun astype FP16 du vocabulaire entier : Array::from_file lit uniquement [1,2560] BF16, puis concat et norme BF16. Ne pas réintroduire le bug d’arrondi du tiny-mask GGUF. La sémantique d’un checkpoint modifié en place doit être explicitée avant le cache. Ne pas appeler `FlashNext::embedding` à la place : cette API renvoie une copie Float16, tandis que `draft_input` consomme la ligne BF16 brute ; le cache doit conserver cette dernière (`qwen4.rs:1445–1458`).

**Preuves source :**

- [native/src/flash_draft.rs:1112–1124](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/flash_draft.rs#L1112)
- [native/src/array.rs:596–608](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/src/array.rs#L596)
- [native/ffi/mlx_bridge.cpp:261–267](https://github.com/0xZKnw/mlxl3/blob/88cbfd2232ab23a0a7ca1d8dc312792347ca9d31/native/ffi/mlx_bridge.cpp#L261)

**Contrôles discriminants proposés, non exécutés :**

- Comparer byte à byte draft_input ancien/nouveau pour counts1/2/3/7, ancres différentes, checkpoints de test séparés et reset/reconfiguration.
- Mutation de la ligne/offset du masque doit être détectée par oracle BF16 indépendant ; vérifier explicitement la ligne ancre non réutilisée par erreur.
- Comptage sur l’entrée réelle FFI : première utilisation une lecture masque, réutilisations zéro, nouveau modèle une lecture ; ne pas appeler ces compteurs des lectures NVMe physiques.
- Oracle draft/cibles/états et acceptations fixes/adaptatives inchangés, mémoire ajoutée utile5120 octets hors overhead non mesuré.

**Portée performance :** Aucun syscall/timing physique mesuré. Faible priorité : B2 ne répète qu’une lecture masque par bloc et son payload vaut5Kio.

## Coûts connus à ne pas revendre comme découvertes

- **61-KV :** append_ring/append impose deux eval par couche sur5 couches ; handles et donation ne prouvent ni copies intégrales ni dix attentes matérielles. Déjà61:110, non réessayé.
- **61-prefill :** prefill_dflash2 fait tête/logits à chaque token alors que seul le dernier est retourné ; déjà61:116. Piste TTFT connue, pas nouveau gain decode. Évaluer streams/drain demeure nécessaire si la tête était élidée.
- **63-commit :** Le replay GDN partiel est paresseux et peut être payé dans la vérification suivante ; déjà63:17. Ne pas attribuer tout ce temps à la seule phase enregistrée commit_cache_seconds.
- **63-terminal :** count==0 effectue forward ordinary et ajoute ses octets, mais pas draft_seconds/verification_seconds/verified_rows/blocks. Budget bridge2 rend ce chemin explicite. Compteur terminal/octet et métriques séparées connus63v5:30/85 ; limite d’interprétation, pas nouveau bug.
- **63-backbone :** HC/PLE/QSA encore largement ligne par ligne et commit fusionné suivant absents : déjà63/64. Pas une découverte79R-A.
- **70-precap :** Le plafond avant draft existe et a été mesuré contre ancien adaptatif ; ne pas le présenter comme nouveau ni comme gain vs B2.
- **76-switch :** Options différentes rechargent les47 matrices Q4 ; correctif76 préparé connu, n’affecte pas la session B2 stable.
- **72-mask :** Tiny-mask Splash GGUF déjà exclu pour le BF16 brut local ; le cache de ligne A03 est distinct.
- **72-73-withdrawn :** Prélectures variantes72 et EarlyShared73 retirées après vraie comparaison sans prélecture ; aucune réactivation.

## Conclusions négatives de la lecture

- Le clone des handles trace/snapshots ne démontre pas une copie de payload.
- La causalité GDN/QSA et l’ancre DFlash interdisent de supprimer arbitrairement leurs lignes.
- chat_greedy_ids normalise avant argmax pour préserver le contrat d’arrondi ; raw argmax n’est pas un remplacement exact établi.
- Le target bonus est requis par greedy_accept ; les logits vérifiés ne sont pas tous inutiles parce qu’un préfixe est ensuite rejeté.

## Vérification et limites

Lectures uniquement (`rg`, `nl`/`sed`, `git rev-parse`, `git status`, `shasum -a256`) et écriture de ces deux rapports. **Zéro test, benchmark, modèle GPU, build ou proveur exécuté.** Les tests existants mentionnés servent de bases futures ; leurs anciens résultats ne valident pas un futur correctif. Les preuves Kani/CBMC et le comportement MLX/Metal restent hors portée de cette revue statique. Aucune source exécutable, entrée de journal ou configuration n’a été modifiée par cet agent. Les travaux65P/74 existants sont préservés.

Le détail machine, les empreintes des huit sources et les contrôles proposés sont dans [verifier.json](verifier.json). Trois redondances ne suffisent pas à démontrer les15tok/s : il faut mesurer leur coût exclusif et tester chaque mécanisme séparément avant de les combiner.
