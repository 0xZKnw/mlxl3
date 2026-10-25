# Audit masque DFlash2 et limites prélecture72 —23octobre2026

Audit en lecture seule de `/Users/justin/.codex/worktrees/flash-splash60/mix-stq1_0`, sans GPU, compilation, prouveur ni benchmark. Les résultats ci-dessous proviennent des lectures CPU déjà terminées ; leur persistance n'a pas relancé ces contrôles.

## Conclusion masque

Le correctif Splash[4dd195e](https://github.com/incoai/splash/commit/4dd195e1581d9a5d597e9e0eeff35573c81a8340) ne correspond pas au défaut de notre checkpoint : il traite un masque Qwen3.5-122B/Ornith très faible détruit par les échelles FP16 d'une quantification GGUFQ8/K. Le masque local de Flash Next est un row BF16 non quantifié et n'est pas écrasé.

Le vrai parcours est `flash_draft_runtime.rs:131 Session::advance` → `flash_draft.rs:1112 draft_input` → lecture directe BF16 des embeddings cible, anchor en ligne0 et masque248077 ailleurs → concaténation BF16 → `flash_draft.rs:673 Draft::forward`, qui exigeBF16 et applique sa première norme. Il ne passe pas par `qwen4.rs:1445 embedding()`, qui effectue effectivement un castFP16 pour le modèle cible ordinaire. Le CABI (`mlx_bridge.cpp:55,255,525`) conserve le dtype demandé et délègue la norme à MLX ; aucun castFP16 local n'est inséré. La quantification Q4 des47matrices du draft ne s'applique pas à cet input.

Le loader vérifie le masque contre la configuration et l'embedding cible contre BF16/[248320,2560]. Config draft : `DFlash2DraftModel`, BF16, hidden2560, bloc8, masque248077. Checkpoints épinglés : cible `65c895314393431c09050b2e04e250836b3a6eb4` ; draft `a22a1744e85c17364dcc54a6a420af7135d4c7f6`. Ces révisions ne sont pas des SHA256 de tous les poids ; le checkpoint complet n'a pas été rehashé.

## Preuve CPU du row local

Fichier : `/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531/model-00001-of-00005.safetensors`. Header8189220B ; offset absolu1278343468 ; lecture5120B ; tensor `model.language_model.embed_tokens.weight`, row248077.

| Propriété | Résultat |
|---|---:|
| Valeurs finies |2560/2560|
| BF16 non nulles |2550/2560|
| Minimum |−0,0169677734375|
| Maximum |0,0166015625|
| Moyenne absolue |0,004154026923060883|
| Non-nulles après FP16 hypothétique |2550/2560|
| Valeurs arrondies par FP16 |5/2560|
| Valeurs devenues zéro |0|

SHA256 du row : `860615491bdf6786178b8c14e96397f51e3895c556d0336e2b5ad8cc8e061a19`. `evidence.json` conserve la commande Python stdlib réellement exécutée, sa sortie précise et les lectures de code/config. Le calcul FP16 est un contrôle CPU hypothétique, pas le dtype du parcours livré.

Le patch primaire fourni par le sousagent recherche est conservé dans `/Users/justin/Documents/mix-stq1_0/build/flash-opt-72/upstream-review-agent/splash-mask-4dd195e.json`, SHA256 communiqué `04c034f86536c26f289ac4155ecdd6aa02751fc13085be119f7110a72020d246`. Son gain2,3→4,9tokens/cycle concerne Ornith et ne prédit aucun gain local. Les tentatives directes web eturllib du présent audit ont respectivement échoué cachemiss etDNS ; aucun code amont copié.

## Revue72/73 : constats distincts

- **Garde weighted v1 incorrecte** : `flash_draft.rs:1943` attendait toujours47×16×min(width,2) IDs. Weighted émet16IDs agrégés/couche. Le log `prefix-weighted.log` échoue sur cette garde avant comparaison des logits larges ; pas de divergence numérique démontrée. Correction variante-aware stricte nécessaire, à revalider par le parent.
- **Late n'overlappe pas toute l'attente** : `ssd_moe.rs:974` prépare la prédiction après readback routes, `wait_demand` et réservation. Son nouvel ordre suitHot mais ne masque pas les attentes antérieures. Limite de performance, pas bug numérique.
- **Shared anticipé sansHot** : `ssd_moe.rs:819` peut soumettre shared avant scheduling/readback des routes lorsqu'aucunHot n'existe ; ralentissement potentiel à départager par les écransOFF. Aucun effet chronométré ici.
- **Bounded protège un préfixe** : `ssd_moe.rs:1178` protège les premiersmin(ids,budget,4), pas nécessairement les experts finalement admis. Exemple résidentA/B puis froidC/D, budget2 : protègeA/B, peut chargerC/D. Conforme au préenregistrement actuel ; ne pas le décrire comme protection des admissions effectives.
- **Fréquence LFU par cohorte** : `flash_support.rs:979 reserve` déduplique, puis incrémente chaque expert distinct une seule fois. Un expert répété deux fois dansB2 gagne1, alors que deux réservationsB1 gagnent2. Le clock avance par réservation, la décroissance survient tous les16ticks ; `publish_prefetch` avance aussi ce clock. Comportement existant, pas nouveau bug confirmé ; passage à fréquence par occurrence ou clock séparé exige hypothèse/protocole propre et mesures réelles.
- **Slots/I/O** : aucun nouveau défaut concret identifié dans le diff72/73 : pending exclus, cohorte demandée préservée, échecs invalidés, lecturesHot retenues. Absence de constat n'est pas preuve de sûreté universelle.

Pas de correctif masque, sidecarBF16 ou nouveau testGPU proposé. Acceptation, tok/s et concurrenceGPU/SSD restent non mesurés/non prouvés par cet audit. Aucun fichier du worktree modifié.
