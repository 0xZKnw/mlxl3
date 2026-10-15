# Essais20/21 : H100, contrôles appariés et tête liée

État de ce premier enregistrement : conversion24B en cours, aucune qualité v2
mesurée et aucune publicationHFv2. Le journal racine conserve le protocole
préenregistré et chaque échec. Le [rapport](../../lfm25-calibration-v2.md)
distingue le chemin Metal conditionnel et le convertisseur CUDA effectivement
utilisé surH100.

## Code et prévol

- `code-verification.json` identifie les sources finales et leursSHA256.
  `full-tests-final-typed.log` :570passés/1ignoré (modèle Ling local absent).
  `runner-tests.log` :12tests ; `runner-offline-tests.log` :12sansprofil/token,
  serveurpointé surlocalhostinjoignable, aucunappelcloud.
- `calibration-final-tests.log` :52tests de calibration, dont deux vrais
  bridges1.4.4. `cache-boundary-before.log.gz` conserve les deux nouvelles
  régressions0octet/NPZ avantfix. Les versions antérieures des suites et le
  lint initial en échec restent conservés et identifiés.
- `calibration-crosshair-final.log` :3conditions d'alignement sur entiers Python
  symboliques, sansborne ajoutée. `crosshair.log` :2contrats des options/budget
  de lancement (les deuxvaleurs du booléenprobe). Recherche de contre-exemples,
  pas preuve de NumPy/MLX/Metal/CUDA/I/O ni de la qualitéglobale.
- `probe-biased-native.json` :395projections,4096logits finis, CUDAdecode/natifdecode
  RMSrelative0,0019083≤0,02 et32/32argmax ; biais F32 réels source. Les essais à
  biaisnuls restent **échoués** (`probe-native*.json`), leurs causes ne sont pas
  toutes démontrées. Les erreurs du tokenizer/API/cache/CLI sont également
  conservées. Les inputs sont identifiés par les manifestes `probe-*-manifest`.
- `cloud-start-report.json` :H10080GB, dépendances réellement chargées et cinq
  hashes de codePython de la roue, égaux au commitCUDA16a4979. SourceHF1769516.
  AppModalap-50iSv73B3hlsIcCd4mM7KB, timeout4h/retries0, candidatprivé.

## Qualité

- `calibration-leakage.json` :zéro13gramme exact commun entre les sixsources de
  calibration et WikiTexttest, tokenizer source,294644tokens test.
- `v1-control-8192.json` :8192tokens/8×1024/sansBOS, huitteachersF32SHA identifiés,
  moteur1.4.4 empaqueté SHA355b5262. PPL/KL reproduites exactement ; NLLteacher
  parfenêtre vérifiée à1e−12. Ancien binaire99830c identifié séparément.
- `supplemental-teachers.json`, `corpora/manifest.json` :1024tokens/2×512 en
  français technique etPython, enseignantsBF16 indépendants du candidat.
  L'essai initial tropcourt948tokens est conservé sanspadding/répétition.
- `v1-other-quality.json` :contrôles courts, BOS etdeuxcorpuscomplémentaires.
  Le premier contrôle dehashBOS utilisait8193au lieu des8192inputs historiques ;
  il échoue avantGPU, puis la portée corrigée reproduit exactement lehash.
  Les métriques historiques court/BOS sont aussi reproduites exactement.

Les gates sont ceux d'opti20 :PPL etKL principales en baisse>0,1% relatif,
examen parfenêtre etsmokesFR/code/calcul/JSON ; seuil complémentaire2% par
métrique/corpus technique. Aucun agrégat de protocoles différents.

## Compaction sans requantification

`head-compact-probe.json` :copie de tête32768octets supprimée sur modèlejetable,
1596autrestenseurs dtype/shape/SHAbyteexacts,4096logitsF16 natifs bitexacts et
mutation négative détectée. Le premier garde supposait à tort unchamp
quant_format surlespoidsplain ; la première sérialisationMLX metadata={} créait
`__metadata__:null`, refusé parlemoteur. Échecs préservés, nouvelartefact avec
metadata `format=pt` vérifié. Cela ne valide pas encore la compaction24B ; le
contrat262144000valeurs etle contrôle128tokens restent àexécuter après20.

Certains logs sont compressés sansperte en`.log.gz` pour conserverleursespaces
originaux. `compressed-logs.json` et lesmanifestes donnent lesSHA vérifiés.
Les fichiers temporaires de poids/teachers sont locaux ou surleVolumeprivé,
pas inclus dans cette preuveGit.
