# Flash Next : état publié pour analyse, 20 octobre 2026

Ce point de contrôle publie les essais43 et le prototype44 demandé après
[l’audit des papiers](flash-next-drafts-paper-audit-2026-10-20.md).
**Aucun gain de débit44 n’est encore validé.** Les prototypes DFlash2/DSpark
restent dans les tests natifs ; ils ne sont pas activés dans le bridge ou l’app.

## Suivi après le point de contrôle `1f38776f`

La [synthèse des deux audits et des mesures44](flash-next-ssd-audits-synthesis-2026-10-20.md)
décrit l'état le plus récent. Les sections suivantes conservent le point
initial publié avant GPU, avec ses contrôles CPU et ses limites à cette date.

Depuis, dix processus de qualité affectée, la mutation de l'ancien résidu,
le bridge release et les quatre bras de calibration DSpark sont terminés.
La campagne non-thinking16/16 garde48 générations128IDs exactes et zéro nouveau
Swapout, mais aucun gain global ne satisfait les critères. Les drafts dépassent
aussi le surcoût de pic autorisé face à OFF. La campagne thinking est arrêtée
après2/12bras, le MTP suivant ayant dépassé la garde de swap système ; tous
les processus sont joints. Aucun résultat15tok/s ni gain livré n'est établi.

La revue confirme une **couverture numérique incomplète des confidences du
hidden natif** : le contrôle complet de tête/confidence indépendant utilise
le hidden CPU de référence. Les vrais IDs proposés et les états contrôlés
restent exacts, mais cela ne remplace pas l'oracle de probabilité manquant.
Les résultats bruts et cette restriction sont conservés dans la synthèse ;
aucune nouvelle optimisation n'est implantée à partir des audits.

## Code à examiner

- `native/src/flash_draft.rs` : backbone BF16, capture et vérification de blocs,
  résidu/norme suivant le kernel du guide SGLang, sélection sur GPU,
  préfixes forcés0..3 et ordonnanceur greedy DSpark utilisant confidences
  calibrées et coûts locaux B1..B4. Le modèle de coût réel reste à mesurer.
- `scripts/check_flash_drafts.py` et `scripts/check-flash-reference.py` : oracle
  NumPy et chaîne réelle vers la tête EXL3 Pony pour contrôler logits complets,
  propositions et confidences. La nouvelle chaîne couplée attend son essai GPU.
- `scripts/calibrate_flash_draft.py` : calibration STS et profil local contrôlé,
  avec données distinctes des prompts de validation. Aucun profil réel encore
  ajusté ; le support reste limité au greedy.
- `native/src/flash_support.rs` : contrats CPU de cohortes, commit et scheduler.
  `ssd_moe.rs` conserve la limite publique à deux lignes et ajoute le chemin de
  test large ; le fallback interne utilise des cohortes bornées par le cache.
- `native/src/tokenizer.rs` : rendu thinking explicite, valeur par défaut
  préservée. Les sources `lfm2.rs` et `array.rs` ne sont pas modifiées.

## Vérification inspectée

Les commandes exactes et leurs sorties sont conservées dans
[`opt-44-draft-scheduler`](measurements/flash-next-ssd-impl/opt-44-draft-scheduler/).

| Contrôle | Résultat |
| --- | --- |
| `cargo fmt --all --check` | passe |
| Clippy strict, tous targets, `mlx,chat` | passe |
| `cargo test --locked --release --features mlx,chat` |125 passés,120 ignorés |
| `cargo build --locked --release --features mlx,chat` | passe |
| Suite Python pertinente |475 passés,0 ignoré,77,98s |
| Ruff format et lint strict des Python modifiés | passe |
| Kani, helper réel de préfixe |216 obligations réussies,2 couvertures |
| Kani, norm/sélecteur MLX et template | ICE `intrinsics.rs:243`, non vérifiés |
| CrossHair | absent, non exécuté |
| CBMC6.11, shader Metal | type de fichier non pris en charge, non vérifié |

Le harnais CPU explore des longueurs0..3, des probabilités/coûts f64
symboliques avec unwind5 : bornes du préfixe et validité des entrées acceptées.
Cette vérification ne prouve ni l’optimalité générale, ni MLX, ni Metal,
ni la génération complète. Les premières erreurs et résultats négatifs
restent présents dans les logs. Les contrôles GPU ignorés, mutations44,
calibration, nouvelle parité bridge et mesures de débit44 ne sont pas exécutés
au moment de cette publication. La CI du nouveau commit doit être inspectée
après push ; les résultats locaux ne valent pas succès de la CI.

## Débit et mémoire : limites des résultats existants

La campagne43 a été interrompue à la demande humaine après9/16 bras complets :
27 générations128 IDs exactes, un bras interrompu et joint, six non lancés.
Sur le prompt MAIN, OFF :8,87–9,09 tok/s ; MTP :9,26–9,39 ;
DFlash2 statique :7,99–8,11 ; DSpark statique :5,79–5,91.
Ce sont des résultats de prototypes43, **pas des mesures du scheduler44**.
Le protocole complet à deux prompts n’est pas validé et aucun gain de ces
drafts n’est annoncé. Leurs temps de vérification/caches dominaient le décodage.

Conditions43 : M5/24Gio, secteur, cible2,05bpw révision65c89531,
drafts BF16 épinglés, cache3Gio/53slots, allocateur512Mio, quatre lecteurs,
overlap activé et LES désactivé. Les bras complets n’ont pas ajouté de
Swapouts. Le précédent essai qualité4Gio a été arrêté par la garde mémoire
et reste documenté. Aucun débit15tok/s n’a été atteint.

## Identité et suite autorisée

[`phase1-identity.json`](measurements/flash-next-ssd-impl/opt-44-draft-scheduler/phase1-identity.json)
contient les SHA des74 sources et l’identité des modèles/budgets. Binaires
44 figés, compilés mais pas encore exécutés sur GPU :

- unité : `74fc95a49823d7f793ea685933a7fe51c749ff85d50707b323eff985b4f12bc2` ;
- moteur : `25b1f44a19bcec5dfd154c09074de151282273e664b2ebcf2dd9436d3c8ffe46`.

Les poids et binaires construits restent hors Git ; les hashes des modèles
sont ceux vérifiés en43, explicitement réutilisés. L’app installée reste
inchangée. Après publication demandée, reprendre les contrôles physiques
affectés, puis la calibration et les témoins de vitesse appariés sous gardes
mémoire/alimentation. Consigner chaque nouveau protocole dans `opti.md`
avant de le lancer.
