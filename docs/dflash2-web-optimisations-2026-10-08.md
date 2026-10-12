# DFlash2 : recherche amont et optimisation du 35B — 8 octobre 2026

État final local : worktree `codex/dflash2-stable`, base `db52b7304a057ad8f0d8fcf464cd66b22b44fe08` modifiée, aucun commit/push/publication. Le bundle généré 1.4.3/build 25 contient la copie du contexte **en option, défaut OFF**, validée physiquement sur le 35B. Un premier comparatif est positif, mais sa confirmation échoue sur la stabilité des témoins : aucun gain général établi et aucune supériorité sur MTP annoncée. Les deux prototypes de transfert ont été retirés. Le 27B reste arrêté à la demande ; l’application personnelle installée reste inchangée. Aucun essai lancé n’est encore en cours.

## Sources primaires examinées

- [Splash](https://github.com/incoai/splash/tree/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7),6octobre2026 : propositions MetalBuffer consommées directement par `QwenTarget::addVerifyInput`, sélection dans `runtime/ops/DraftSelector.cpp`. Ses projections affine/GGUF et ses états BF16 ont un autre contrat numérique que notre cible EXL3/FP16.
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx/tree/60803233af4589e18588b9bacbb03880801c828a),20août2026 : soumission `async_eval`, replay GDN, copies du contexte dans `engine/spec_epoch.py::_copy_draft_for_block`, désactivation conservatrice après zéro acceptation. Son README indique60s de repos entre mesures. Le replay compact est déjà intégré ici ; il ne constitue pas une nouvelle optimisation.
- [DFlashMLX amont](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py),18août2026 : captures après les couches sélectionnées et rollback récurrent. Aucun décalage d'index de capture identifié dans la comparaison statique des sources.

Révisions, blobs et SHA256 des14fichiers lus : [manifest](measurements/dflash2-web-02/source-manifest.json). Les chiffres publiés sur GB300/sampling/4096tokens ne prédisent pas le classement sur M5/greedy/EXL32,49bpw. La comparaison locale MTP précédente reste nonconcluante en vitesse, avec MTP observé devant ; voir [validation précédente](dflash2-stable-validation.md).

## Tête35B

La recherche publique duHub trouve la [tête officielle IncoAI](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) au commit `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` :526251520paramètresBF16, unique commit «DFlash2release» du25septembre. Les deux autres correspondances sont des conversions4bitMLX etAMXQ8 de cette famille, sans nouvelle tête entraînée identifiée. Aucune supériorité n'est inférée de la précision ou du nom. Le draft packedQ4 existant est conservé ; une comparaison BF16/Q4 nécessiterait son propre chargeur, oracle et protocole.

[Catalogue et historique bruts](measurements/dflash2-web-02/draft-head-search.json). La recherche couvre les dépôts correspondants et les sources examinées, pas tous les checkpoints privés ou nonindexés.

## DEVICE-DRAFT : premier candidat

Le chemin précédent matérialise les propositions sur CPU avant de construire la vérification. Le prototype conserve leurs IDs surGPU, soumet le graphe du draft avec l'API MLX déjà présente, puis construit la cible et lit les IDs après les logits. Les poids, calculs du sélecteur, largeursAuto2/5 et arithmétique cible sont conservés. Un essai voisin MTP avait été retiré après un gain inférieur3% ; ce nouvel essai concerne les six couches DFlash et une soumission explicite avant la cible.

La revue a trouvé un contre-exemple : des logits tousNaN peuvent produire le sentinel0xffffffff. Le shader conserve les IDs originaux pour les refuser, et émet séparément des IDs bornés pour le gatherGPU. Une entrée invalide provoque un reset avant commit et incrément d'offset. Ce prototype utilisait `MLXL3_DFLASH_DEVICE_DRAFT=1` et a été retiré après les deux campagnes non concluantes.

### Vérification avant vitesse

- `cargo fmt --all --check`, `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`, buildrelease et `cargo test --locked --release --features mlx,chat` :74passés/63ignorés. GPU exécuté séparément.
- `cargo kani --lib --no-default-features --harness dflash_device_draft_reserves_anchor_and_matches_vocabulary` :108obligations/4couvertures réussies ; i32/u32 complets, sans hypothèse. Même harnais avecfeaturesmlx-chat réussi : il vérifie toujours la garde scalaire, aucune FFI ni exécutionGPU.
- Mutation du vocabulaire détectée. Première restauration de test enéchec àcause du binairemutant gardé parCargo après restauration du mtime ; source byteexacte, logconservé, rebuild forcé et1/1test repassé.
- Test physique `qwen35::tests::dflash_device_proposals_match_host_and_serial_states` :1/1,21vérifications device/host/oracle indépendant,105préfixes commités,80états finis, logits et captures bitexactes ; préfill23/24/256,1..7propositions, erreursanchor/vocab/contexte et3sélecteursNaN rejetés/reset.
- `scripts/check_dflash2.py` sur le vrai bridge35B :45/45, Tune, budgets1/2/3/8/17/64, cache, annulations, erreurs et historique exact nonvide. Aucun nouveau parcours GPU27B.
- CBMC6.11 sur le vrai shaderMSL refusé avant obligations (langage nonpris encharge), ESBMC/Frama-C absents. PropriétésMetal/MLX/concurrence nonvérifiées formellement. Compilation réelle et tests physiques ne constituent pas une preuve complète.

[Identité et sources](measurements/dflash2-web-02/device-identity.json), [contrôles locaux](measurements/dflash2-web-02/device-local-checks-final.json), [qualitéGPU](measurements/dflash2-web-02/device-gpu-quality.json), [mutation](measurements/dflash2-web-02/device-mutation.json).

### Mesure initiale : nonconcluante

Même moteur SHA`ed5cf0c868e7a636ffbdf5c94880484f13163ed1074956c855bba9220e3a4561`, A0/B1, M5Air24Gio/MLX0.32.2, targetEXL32,49bpw/draftQ4, Auto1/pipeline0/lookup0/contexte4096/cacheOFF/greedy,128warm/128mesurés/2répétitions/ABBA/repos2s. Un seul modèle, aucun build/proveur concurrent. Secteur,51%encharge aupréflight ; conditions parpasse dans lesbruts, température/fréquences nonmesurées.

| Prompt | A CPU | B GPU | Écart observé |
|---|---:|---:|---:|
| Français court |57,1281tok/s|61,1921tok/s|+7,114%|
| Code matrice |63,9600tok/s|70,1243tok/s|+9,638%|

Parité des tokens/texte/historiques nonvides et compteurs identiques. Dérives FR A−13,497%/B−8,716%, code A−12,731%/B−5,768% : lecritère≤3% échoue. **Aucun gain établi et aucune BAAB de cette campagne.** [Décision et bruts](measurements/dflash2-web-02/device-abba-decision.json).

Une seule répétition est préenregistrée pour corriger ce protocole :60s fixes avant chaque passe, autres paramètres et critères identiques. La cause physique de la dérive reste inconnue. [Plan](measurements/dflash2-web-02/device-cooldown-plan.json). Statut final : ABBA terminé en 311,47 s. Français 54,8100 → 65,0213 tok/s (+18,630 %), code 65,9322 → 74,4345 (+12,896 %), agrégation du harnais inchangée. Dérive du contrôle français A −9,377 %, donc résultat **non concluant** malgré les autres contrôles ≤3 %. Aucune BAAB et aucune autre variation de repos. [Décision](measurements/dflash2-web-02/device-cooldown-decision.json). Prototype retiré ; sources antérieures restaurées exactement, preuves conservées ([retrait](measurements/dflash2-web-02/device-retirement.json)).

## Copie du contexte : état initial de la recherche

CopySpec peut éviter le réseau draft lorsqu'un passage du contexte correspond à la suite courante. Notre matcherMTP borné à1024IDs existe déjà et peut être réutilisé avec vérification cible et désactivation après refus total. Cette piste était non prototypée à ce stade ; son adaptation distincte CONTEXT-COPY est décrite plus bas, après préenregistrement.


## Comparaison de moteurs MLX demandée ensuite

Les sources ont été téléchargées et lues, sans exécuter les moteurs étrangers ni charger un autre modèle. [Inventaire des révisions](measurements/dflash2-web-02/engine-index.json), [32 fichiers et empreintes](measurements/dflash2-web-02/engine-source-manifest.json).

| Moteur et révision examinée | Mécanisme concret | Adaptation à MLXL3 |
|---|---|---|
| [mlx-dspark](https://github.com/ARahim3/mlx-dspark/tree/d6042f38f0aa7dd3e03a2a406111d725056d45a3) | `dflash_generate` garde le sélecteur, les IDs et l'argmax cible dans le graphe jusqu'à l'acceptation. `small_m_qmm` partage la lecture des poids entre 5–16 lignes et calibre les formes. | Graphe et transfert groupé prototypés ci-dessous. Le kernel affine BF16 ne lit pas notre EXL3 et change l'ordre d'accumulation : pas de copie directe dans le chemin exact FP16. |
| [oMLX](https://github.com/jundot/omlx/tree/cc1fdc9a24053224521a8dc6e1350d64e8ec16f4), [fork dflash-mlx épinglé](https://github.com/jundot/dflash-mlx/blob/71f7c2cae42a968ddc981972a8ad35368dc587a9/dflash_mlx/engine/spec_epoch.py) | Le chemin historique concatène IDs vérifiés et posterior avant un seul transfert hôte ; captures asynchrones hors profiling. Le nouveau drafter oMLX conserve aussi des caches par requête et peut préparer le bloc suivant avant lecture de l'acceptation. | Un transfert hôte par bloc adapté. L'anticipation du bloc suivant nécessiterait une transaction supplémentaire sur les caches draft et la validation de l'acceptation dynamique : non implémentée. |
| [TensorFold MLX 0.6.6](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248) | Head réduit pour les propositions seulement, copie du contexte, arbres et kernels à calcul par ligne invariant. Le sélecteur/tree impose une politique propre à ce moteur. | Pistes head réduit/copie identifiées. La vérification séquentielle exacte et le replay compact existent déjà ici. Les kernels affine ne remplacent pas ceux EXL3. La branche principale v1.0.2 a migré vers Zig : le code MLX analysé est explicitement le tag 0.6.6, pas son main actuel. |
| [Yunshu](https://github.com/YuhuanStudio/Yunshu/tree/8888514d42091c359b8cdf19142e869e0d784759) | `CopyDraft` évite le réseau pendant une reprise du contexte et rattrape une seule fois les captures/caches lors du retour au draft ; contrôleur de bénéfice et backoff. | Matcher MTP local adapté dans CONTEXT-COPY ci-dessous, avec validation cible et historique des seuls tokens commités. Qualité physique validée, vitesse globale non confirmée. |
| [IronMLX](https://github.com/apepkuss/ironmlx/tree/cac4352b722a8c630c58e154827566f92b4c584c) | Moteur Rust/MLX : IDs draft asynchrones, entrée cible construite directement sur GPU, profils de vérification exacts selon quantification et largeur, tests physiques de rollback. | Confirme le mécanisme GPU choisi et la nécessité des profils numériques. Notre type opaque et les tests de cache restent propres à MLXL3 ; les qualifications affine ne sont pas transférables à EXL3. |
| [vllm-mlx](https://github.com/waybarrios/vllm-mlx/tree/80e7fdec7e8641c8f81b01ece2d7cc256348b76f) | Aucun chemin DFlash2 identifié dans les arbres examinés ; demandes DFlash toujours ouvertes dans les sources consultées. | Pas une référence d'intégration DFlash2 à reproduire sur la seule base du nom vLLM. |

Les chiffres publics concernent surtout le 27B affine sur M4 Pro ou M5 Max. La page [méthodes Yunshu](https://github.com/YuhuanStudio/Yunshu/blob/8888514d42091c359b8cdf19142e869e0d784759/docs/BENCHMARKS.md) précise que son ancien comparatif utilisait en réalité MTP et un SHA non capturé, alors que TensorFold utilisait DFlash2 ; les bruts JSONL sont privés. Ces chiffres ne permettent pas de désigner un gagnant sur notre 35B EXL3/M5 Air. La source exploitable est le mécanisme observé dans le code, suivi de nos propres oracles et mesures.

## SINGLE-TRANSFER : adaptation complète du graphe

Le candidat précédent rapatriait encore les propositions séparément, puis évaluait les logits cible avant de lancer leur sélection. Cette version construit la normalisation greedy existante et son argmax avant la lecture. Elle concatène les propositions originales et le posterior en un paquet UInt32 de 2N+1 valeurs, puis effectue un seul transfert hôte. Les captures sont soumises conjointement et matérialisées avant commit. Elle conserve la soumission anticipée du draft, les mêmes embeddings par ligne, le même calcul cible et les mêmes largeurs Auto2/5.

Cette adaptation reprend les mécanismes de `mlx-dspark::dflash_generate`, du fork oMLX et d'IronMLX, avec notre normalisation FP16 et notre protection contre le sentinel NaN. Tous les IDs originaux, la longueur et la géométrie sont validés avant l'incrément d'offset ; une erreur remet le cache à zéro. Le parsing CPU est testé indépendamment, sans traiter les copies bornées comme des propositions valides.

Prototype historique retiré après mesure : `MLXL3_DFLASH_SINGLE_TRANSFER=1`, ancien défaut OFF, option maintenant absente du code conservé. L'ancienne option DEVICE_DRAFT est supprimée. Le profil Tune passe à `dflash-v3:transfer=0/1`, partagé entre l'événement ready et le résultat Tune. Cela empêche de réutiliser un profil mesuré avec une autre politique. Aucun changement GPU27B, aucune installation personnelle ou publication.

### Contrôles réalisés sur le prototype retiré

- Format, Clippy strict, buildrelease `mlx,chat` passent. Suite Rust livrée : **76 passés / 63 ignorés** ; CPU sans MLX :43/1. Les tests GPU ignorés sont sélectionnés séparément.
- Mutation de la borne de vocabulaire détectée ; restauration et test ciblé réussis.
- Kani paquet : premier unwind16 insuffisant pour `memcmp`, échec conservé. Avec unwind65 : **210 obligations réussies, 3 internes inatteignables, 4/4 couvertures satisfaites**, paquet0..15 valeurs symboliquesu32, largeurusize complète, sans hypothèse ni stub. Même contrôle avecfeaturesmlx-chat réussi, sans constituer une preuve GPU/FFI. La garde scalaire conserve son domainei32/u32 complet. Suite Kani CPU complète :44/44 réussis,5929 obligations réussies,116 couvertures satisfaites et75 obligations internes inatteignables ; aucune assertion applicative inatteignable.
- CBMC6.11 refuse encore la source Metal réelle avant obligations. Shader, MLX et concurrence non vérifiés formellement.
- Contrôle physique avant vitesse réussi :21cas/105commits/80états, logits/captures finis bitexacts et greedy normalisé identique, erreurs/NaN/reset. Vrai bridge45/45, Tune/cancel/cache/budgets et clétransfer1 contrôlés. [Qualité](measurements/dflash2-web-02/single-gpu-quality.json).

[Identité](measurements/dflash2-web-02/single-identity.json), [contrôles locaux](measurements/dflash2-web-02/single-local-checks.json), [mutation](measurements/dflash2-web-02/single-mutation.json), [premières tentatives formelles](measurements/dflash2-web-02/single-formal-checks.json), [tentatives corrigées](measurements/dflash2-web-02/single-formal-checks-corrected.json).

### Protocole de vitesse préenregistré

Même futur binaire A0/B1, mêmes checkpoints, Auto1/pipeline0/lookup0/contexte4096/cacheOFF/greedy. Français court et code matrice historiqueSHA4f326…,256tokens de warmup puis4répétitions de128tokens,60s fixes avant chaque passe, ordreABBA. Seuils : code≥3%, français≥−3%, dérive de chaque contrôle≤3%, puis confirmationBAAB aux mêmes seuils. Pas de variation du protocole jusqu'à un résultat vert.

Résultat ABBA359,26s : français59,3862→66,6973tok/s (+12,311%), code66,6609→75,7354 (+13,613%), sorties et compteurs exacts. Mais dérives du contrôle A−6,677%/−9,893% (B−0,204%/−0,208%) : **non concluant**, aucune BAAB et aucun gain établi. [Décision](measurements/dflash2-web-02/single-abba-decision.json). Prototype effectivement retiré et sources baseline restaurées exactement ; [retrait et empreintes](measurements/dflash2-web-02/single-retirement.json). Les chiffres de ce prototype ne valident pas le code conservé ensuite.

## CONTEXT-COPY : éviter le réseau draft pendant une copie vérifiable

Ce nouveau candidat adapte les copies de contexte observées dans Yunshu, TensorFold et BST. Il réutilise le matcher MTP local : suffixe exact de8IDs, fenêtre maximale1024IDs, continuation de1..3propositions. La cible vérifie toujours toute la suite, puis ses captures avancent le cache DFlash même si le réseau draft a été évité. Les références primaires et les essais MTP précédents ont été relus ; leur signal de vitesse ne valide pas ce candidat à six couches DFlash.

L'état de copie appartient à la requête. Il est initialisé à partir du prompt complet après prefill, y compris sur restauration d'un checkpoint. Après commit, il mémorise seulement l'ancre et les propositions acceptées ; la correction en attente et les propositions refusées sont exclues. Un refus total de copie la désactive pour le reste de la requête, avec fallback neural. Les copies sont limitées au modeAuto du35B, largeur≤min(3,largeurAuto) ; les modesfixes2/7 et le27B restent sur leur chemin précédent.

La même méthode configure le chat et les échantillons Tune. La clé ready/Tune porte `dflash-v4:copy=0/1`, empêchant le recyclage d'un ancien profil, et le compteur `dflash_lookup_blocks` mesure les blocs effectivement vérifiés depuis une copie. Option locale `MLXL3_DFLASH_CONTEXT_COPY=1`, **défaut OFF conservé après confirmation non concluante**. L’historique utilise un tableau fixe de 1024 u32 (4 Kio inline dans chaque DFlashChat, initialisé avec le prompt seulement lorsque la copie est active), sans allocation sur le tas ; aucun nouveau shader/poids/FFI/dépendance. Le temps complet et le prefill sont conservés dans les résultats bruts ; aucun gain mémoire processus ni coût isolé d’initialisation n’est établi.

### Vérification de CONTEXT-COPY

- `cargo fmt --all --check`, Clippystrictmlx-chat et buildrelease réussis. Après correction de fixture et ajout du test de flux : suite livrée **77passés/65ignorés**, CPUchat **63/2ignorés**. Huit graines fixes ×512commits, soit4096transitions comparées à une file indépendante ; préfixes0..7/1023/1024/1025 et plusieurs débordements de fenêtre. Ce test ne remplace pas une preuve.
- Python complet360passés/4sauts historiques (convertisseur ponyexl3 absent et modèleLing absent), aucune nouvelle erreur.
- Premier test CPU ne compilait pas (E0689 compteur de fixture ambigu) : corrigé enusize, échec conservé. Mutation retirant le one-strike détectée (1test échoué), source restaurée byteexacte et1test réussi. Tests couvrent fenêtres0..2048 et préfixes0..1025, petits budgets, IDsu32 extrêmes et désactivation persistante.
- Kani sur la vraie transition CPU : domaine prévu de21longueurs0..20, IDs/ancreu32, préfixe accepté0..7, flags et index de sortie symboliques, unwind33 sans hypothèse ni stub. Les tentatives Vec puis tableau fixe agrégé ont été interrompues pour coût de calcul, sans contre-exemple fonctionnel. La partition longueur0 passe avec KiSSAT (126,96s), puis le harnais final exact repasse en128,39s :161obligations réussies,12assertions inatteignables attendues (8std,1branche accepted≥1024 hors domaine,3ancienne région prompt vide),8/9couvertures satisfaites avec celle de l’ancien prompt inatteignable àlen0. La série des21longueurs n’a pas terminé. La dernière représentation CBMC (sensibilité des champs1024/CaDiCaL) dépasse le timeout180s. Le contrat disabled passe séparément :291obligations réussies,14internes inatteignables,2/2couvertures, historique0..20 et largeurusize complète. La commande de suite CPU complète est tentée avec un plafond global180s, sans exclusion ni changement des propriétés. Les longueurs non terminées restent **non vérifiées**. La troncature réelle1024 est testée par un oracle Vec indépendant, hors du domaine de ce harnais ; aucune preuve GPU/Metal/FFI/concurrence. Les assertions inatteignables de région «ancien prompt» pourlen0 sont attendues, cette région doit être vérifiée dans les partitionslen>0.
- Bundle local reconstruit avec SDK26.5/MACOSX26.2. Moteur final à tableau fixe : SHA256 `ad0bef499cd1850795d5ad9c8a54055d91299e9564fa55ebeb898e7ffde03c64`, 9 766 336 octets. Desktop complet réussi en295,12s : parcours un clic, Tune, restore/cancel/OFF, téléchargement/transport/erreurs et tests UI. Ces tests Swift/JSON précèdent uniquement le changement de stockage interne Rust ; leurs sources sont inchangées. Ancien moteur74ba sauvegardé ; aucune app personnelle remplacée.
- Qualité physique réussie avant vitesse :4tests explicitement sélectionnés,1passé chacun. **9blocs forcés/N1..3/accept0..N**, oracle sériel indépendant, hidden/selector finis et bitexacts ; **24vérifications cible/108commits/80états** finis bitexacts ; **35completions nonvides** avec budgets1/2/3/17/256, cache froid/chaud et modesAuto/fixes, **96copies réellement vérifiées** ; **4samplesTune** exacts avec copieAuto>0/fixes0. Vrai bridge signé :**45/45**, Tune/cancel/erreurs/cache/petits budgets. Clé ready/Tune identique, suffixe dflash-v4:copy=1. Aucun nouveau GPU27B.

[Contrôles Rust après flux](measurements/dflash2-web-02/copy-stream-checks.json), [qualité physique](measurements/dflash2-web-02/copy-gpu-quality.json), [premier échec de fixture](measurements/dflash2-web-02/copy-gpu-quality-fixture-failure.json), [contrôles Rust du build](measurements/dflash2-web-02/copy-fixed-checks.json), [frontières CPU/Python](measurements/dflash2-web-02/copy-boundary-checks.json), [mutation](measurements/dflash2-web-02/copy-mutation.json), [tentatives initiales](measurements/dflash2-web-02/copy-formal-checks.json), [partition interrompue](measurements/dflash2-web-02/copy-partition-formal.json), [dernière représentation](measurements/dflash2-web-02/copy-field-formal.json), [identité finale](measurements/dflash2-web-02/copy-identity.json), [packaging](measurements/dflash2-web-02/copy-package-checks.json).

### Protocole de vitesse préenregistré

Même moteur signé final Aenv0/Benv1,35BEXL32,49/draftQ4/M5Air24Gio/MLX0.32.2, Auto1/pipeline0/MTPLookup0/ctx4096/cacheOFF/greedy. Français court contrôle et copiecode de la fixture MTP historique (deux fonctions répétées8fois),256warm/256mesurés/4répétitions/reposfixe60s/ABBA. [Prompt exact](measurements/dflash2-web-02/copy-prompt.txt), SHA25694cab9306c719db06d1049b4a7969f783c9ddb5c47ad977dd751def5fe7406db. Gaincopie≥3%, FR≥−3%, toutesdérives≤3%, puis BAAB aux mêmes seuils. Contrôlecode nonrépétitif requis avant éventuel défautON. En cas de dérive : pas de nouvelle variation de protocole, maintien seulement opt-in si qualité validée et aucune régression ; aucun gain revendiqué. En cas d'erreur de correction ou régression, retrait. ABBA lancé après qualité validée ; gain non annoncé avant analyse des quatre passages et des dérives. [Plan exact](measurements/dflash2-web-02/copy-speed-plan.json).

### Résultats de vitesse et décision finale

La première campagne a été interrompue lorsqu’une suite Desktop externe a démarré. Seul notre benchmark a été arrêté ; les processus externes ont été laissés intacts. Ce résultat partiel ne valide aucune vitesse. Une unique reprise du **même protocole** a été préenregistrée après la fin de cette suite, avec surveillance des compilations, prouveurs et autres inférences. Les deux campagnes complètes suivantes n’ont détecté aucune interférence externe. [Interruption](measurements/dflash2-web-02/copy-abba-interference.json), [reprise](measurements/dflash2-web-02/copy-speed-isolated-plan.json).

| Campagne | Français A → B | Copie de code A → B | Dérives entre passages | Décision |
|---|---:|---:|---|---|
| ABBA isolée | 56,84 → 56,95 tok/s (+0,19 %) | 80,15 → 88,41 tok/s (+10,31 %) | Toutes ≤0,992 % | Qualifie pour confirmation |
| Confirmation BAAB | 55,89 → 55,15 tok/s (−1,33 %) | 80,37 → 88,01 tok/s (+9,50 %) | Français A +4,421 %, B −6,683 % ; copie A +0,385 %, B −0,928 % | **Non concluante**, seuil 3 % dépassé |

A désigne `MLXL3_DFLASH_CONTEXT_COPY=0`, B la valeur `1`, dans le **même moteur signé ad0bef…**. L’agrégation est celle du harnais : médiane des deux médianes de passage, soit leur moyenne ; ce n’est pas la médiane des huit répétitions regroupées. Les bruts conservent aussi le prefill, TTFT, temps complet, compteurs, mémoire et conditions par passage. La batterie était autorisée et en décharge ; températures et fréquences ne sont pas mesurées, la cause de la dérive reste inconnue.

Chaque campagne complète contient 32 générations mesurées de 256 tokens, plus huit warmups, avec mêmes hashes de tokens, texte et historique **non vides**. Le contrôle français ne copie aucun bloc. La fixture de code copie 48 blocs par génération B, contre zéro en A. Le signal sur la copie est positif dans les deux ordres, mais **la confirmation globale ne satisfait pas les critères fixés avant l’essai**. Aucun chiffre n’est présenté comme accélération générale livrée ou comme comparaison avec MTP. [ABBA et bruts](measurements/dflash2-web-02/copy-abba-isolated-decision.json), [BAAB et bruts](measurements/dflash2-web-02/copy-baab-decision.json).

La copie du contexte est donc conservée **uniquement en option, défaut OFF**. Le contrôle code non répétitif, l’activation par défaut sur M5 et la réorganisation conditionnelle des harnais/du budget CI n’ont pas été lancés, leur condition n’étant pas satisfaite. Aucune nouvelle variation de repos ou de warmup n’a été essayée après cette dérive. [Plans conditionnels clos](measurements/dflash2-web-02/copy-code-control-plan.json), [promotion non lancée](measurements/dflash2-web-02/copy-default-conditional-plan.json).

### Vérification et limites de livraison

Les commandes finales applicables sont consignées dans les JSON liés ci-dessus : format, Clippy strict, `cargo test --locked --no-default-features --features chat` (63 passés / 2 ignorés), `cargo test --locked --release --features mlx,chat` (77 / 65), build et packaging SDK 26.5, suite Python (360 / 4 sauts), suite Desktop complète. Quatre tests GPU explicitement sélectionnés et les 45 contrôles du vrai bridge signé passent ; les autres tests GPU ignorés ne sont pas présentés comme exécutés. Les sources production du draft, de la cible et du sélecteur sont inchangées depuis le retrait des transferts ; les seuls nouveaux calculs retenus sont le matcher CPU et son câblage chat/Tune.

La suite Kani CPU complète a effectivement été lancée, puis interrompue au plafond global de 180 s : **36 harnais terminés avec succès, suite incomplète**. Les contrats de copie longueur initiale 0 et désactivation ont des résultats ciblés réussis ; les longueurs 1..20 restent non vérifiées faute de ressources. Les 4096 transitions déterministes couvrent des débordements de la fenêtre 1024, hors de ce domaine formel. Le timeout CBMC 180 s sur longueur 4 n’est ni un succès ni un contre-exemple. [Bilan formel partiel](measurements/dflash2-web-02/copy-final-formal.json), [contrôle exact du harnais final longueur 0](measurements/dflash2-web-02/copy-final-zero.json), [dernière tentative longueur 4](measurements/dflash2-web-02/copy-field-formal.json). Aucune preuve complète de génération, Metal, MLX, FFI, allocateur ou concurrence n’est revendiquée.

Les workflows PR/push existants ont été inspectés : Kani 0.68.0, Rust/Clippy, compilation MLX 0.32.2 et contrôles Python/Swift. Les runs réussis de la base db52 ne valident pas ce diff local non commité ; **aucun run GitHub du candidat n’existe**. La durée totale des nouveaux harnais dans la limite CI actuelle de 30 minutes reste non vérifiée. Aucun modèle 27B n’a été chargé dans cette campagne ; seul son en-tête draft sert au contrôle négatif de famille. La comparaison BF16/Q4 des poids de tête et le tuning final 27B restent différés.

Le bundle généré peut être testé localement ; l’application personnelle et le DMG publié n’ont pas été remplacés. Le bouton de téléchargement/chargement et Tune proviennent de l’intégration stable précédente, dont [le rapport](dflash2-stable-validation.md) reste distinct. Les anciennes mesures des prototypes retirés restent archivées et ne valident pas ce bundle.

## Remerciements

[IncoAI/Splash](https://github.com/incoai/splash), [mlx-dspark](https://github.com/ARahim3/mlx-dspark), [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) et son [fork oMLX](https://github.com/jundot/dflash-mlx), [TensorFold](https://github.com/ashhart/TensorFold), [Yunshu](https://github.com/YuhuanStudio/Yunshu), [IronMLX](https://github.com/apepkuss/ironmlx) et [z-lab](https://github.com/z-lab/dflash) : architecture et mécanismes d'inférence cités précisément ci-dessus. Aucun chiffre amont n'est annoncé comme gain livré dans MLXL3.

Preuves brutes archivées sans perte pour publication : chaque chemin original
dans les anciens rapports JSON est répertorié dans `raw-evidence-index.json`
et récupérable dans `raw-evidence.tar.gz` de son dossier de mesures. Les sources
historiques et les rapports originaux restent également dans ces archives.
