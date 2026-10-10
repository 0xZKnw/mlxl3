# Tune MTP : retour utilisateur du 10 octobre 2026

État : campagne GPU terminée, correctifs locaux vérifiés ; packaging et test du moteur embarqué en préparation. Base b0c62850b59a26df408731ac069cf85118c5fb24 ; binaire diagnostique embarqué source63bf989/SHA366cccff760e1c8ee3656c9750d7e38199d39adb0a6b0e22a1b2671fcb2c64ea. App installée inchangée.

## Interface et calibration

Le lancement Tune forçait `showInspector=true`. Régression reproduite sur la vraie méthode avant correction (exit133 : Header tuning must keep the conversation visible). Le code local supprime la navigation forcée et utilise un contrôle de barre avec spinner circulaire, pourcentage, phase et action d’arrêt. Résultat/erreur proposent un bouton Détails ; un inspecteur déjà ouvert reste ouvert. La baseline est expliquée : aucun gain MTP supérieur à 3 % confirmé sur ces deux prompts, pas une affirmation universelle.

La clé de calibration omettait la politique du cache allocateur. Le nouveau tag allocator-cache-v1 utilise défaut0 et chaque limite MiB+1 (0..4096), validée par le même contrat `mib_budget` ; clés séparées par politique, valeurs invalides refusées. Aucun calcul de génération modifié. Callers : ready/runtime_key → MTPConfigurationKey → restore/save ; Tune utilise la même runtime_key. Tests sur le vrai constructeur, exploration exhaustive4098politiques et Kani symbolique prévu ; ne prouve ni sysctl/FS/env, ni MLX/Metal.

## Ce que signifie la baseline

Dernier résultat utilisateur (seul champ studio.mtpSelections consulté) sur Qwen3.8-27B : D0=7,813, D1=7,404, D2=5,927, D3=5,642 tok/s. Quatre modes éligibles, 190decode tokens/mode, mêmes deux hashes non vides. D0 est donc le choix conforme au seuil de 3 % ; les propositions sont acceptées mais leur vérification et l’entretien du draft ont un coût.

L’ancien profil à D1=70,670 tok/s concerne Qwen3.6-35B-A3B (MoE), pas Qwen3.8-27B (dense64couches/hidden5120). Les mesures historiques dense étaient déjà beaucoup plus lentes ; pas de comparaison directe des deux architectures. Le protocole Tune reste quatre warmups32tokens puis deux prompts96tokens × quatre modes en ordre inverse, soit896tokens maximum et douze préfills. La durée est cohérente avec plusieurs minutes à ces débits. Le MTP adaptatif est exclu du Tune qui compare des sessions fixes.

## Campagne en cours et limites

Matrice Cpacked1/cache0, Bpacked1/cacheMLXdéfaut, Dpacked1/cache256Mio, Apacked0/cacheMLXdéfaut, puis Bconfirmation. Même binaire/modèle/head EXL3 nominal2bit/head4bit, M5Air24Gio/macOS27.2/MLX0.32.2/context4096 ; adaptatif1 constant, aucune compile/preuve concurrente. Deadline ready120s/Tune360s, transport JsonProcess borné, shutdown/reap, échecs conservés, pas de registre/conversations/préférences de production modifiés.

L’utilisateur a précisé après confirmation initiale que le Mac n’était finalement pas branché. Le lancement avec gardeAC a échoué avant création du processus modèle ; preuve initial-no-ac conservée. Révision enregistrée avant reprise : mesures surbatterie **indicatives**, aucun gain secteur ou universel établi. Les débits dérivent fortement ; aucune attribution causale de petites différences. Source exacte du pilote mesuré archivée séparément de sa correction ultérieure : étatpassed d’un cas n’est maintenant posé qu’après confirmation exit0/reap ; tous les cas complets devront confirmer exit0.

[État brut](measurements/tune-feedback-20261010/diagnostic-status.json), [identité](measurements/tune-feedback-20261010/driver-measured-identity.json), [journal](../opti.md). Analyse MLX de triage sans nouveau kernel : [backend CustomKernel MLX0.32.2](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/custom_kernel.cpp), entrée packée via factory déjà cachée dans le bridge ; aucune conclusion de vitesse tirée de la source seule.

## Résultats du diagnostic terminé

| Cas | Temps total (s) | D0 | D1 | D2 | D3 | Choix |
|---|---:|---:|---:|---:|---:|---|
| C-zero | 153.62 | 7.524 | 7.145 | 5.709 | 5.292 | Baseline |
| B-default | 172.86 | 6.675 | 6.395 | 5.108 | 4.597 | Baseline |
| D-256 | 189.51 | 6.03 | 5.707 | 4.652 | 4.271 | Baseline |
| A-dense | 195.55 | 5.889 | 5.559 | 4.52 | 4.12 | Baseline |
| B-default-repeat | 199.61 | 5.665 | 5.578 | 4.462 | 4.094 | Baseline |

Débits en tok/s. **5/5 processus rejoints, 20 modes / 3 800 tokens decode**, hashes identiques et positifs, acceptations non nulles pour D1..3. Dérive du contrôle B répété : [-15.14, -12.78, -12.65, -10.94] % pour D0..3. **Vitesse non concluante** : aucun gain/perte causal du cache/packing établi. Le réglage cache0 ne suffit pas à expliquer la lenteur. Baseline conforme dans tous les cas et aucune différence de sortie détectée sur ces prompts. [Synthèse](measurements/tune-feedback-20261010/diagnostic-summary.json).

## Vérification du correctif

CPU67passés/3ignorés ; livrémlx/chat84passés/71ignorés, format/Clippystrict passent. Desktopcomplet passé puis libellé final revérifié par compilation Swift6warnings-as-errors et hardening/rendus9états ; 240oraclescontext,114assertionsidle,64grillesTune, transport69tests passés. Pilote1selftest/9corruptions rejetées. Les ignored ne sont pas comptés comme GPU physiques exécutés.

Kani0.68 : **1/1harnais réel,68obligations/4coversSATISFIED,0échec/inatteignable**, domaineOption/usize64bit symbolique complet, sansassume ni stub/boucle. Collisiondefault/zero détectée par mutant isolé ; original passé. La suite entière n’est pas rejouée localement : contrats anciens inchangés, CIcomplète raccordée. PreuveCPU ne prouve pas MLX/Metal/FS/env/concurrence/génération. CrossHair tenté absent, pas de preuve sourceSwift disponible. [Synthèse](measurements/tune-feedback-20261010/checks-summary.json), [sources vérifiées](measurements/tune-feedback-20261010/checked-source-identity.json), [logs compressés sans perte](measurements/tune-feedback-20261010/log-archives.json), snapshotsPNG fixtures dans le même dossier.

Échecs conservés : avantfix navigation forcée, gardeAC avant créationchild ; harnaisRust édition2015 au lieu2024 (diagnostic outildisponible, fichierbrutinitial absent), globSwiftMath.build évalué avant présenceSwiftMath.o (diagnostic outildisponible, fichierbrutinitial absent). Réparations concernent uniquement les harnais, puis contrôles affectés réussis. Le libellé final dit MessageMTP afin de couvrir aussi une erreur de tête horsTune.
