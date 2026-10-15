# Essais20–23 : H100, qualité et diagnostic de calibration

Révision finale : conversion53min29s, intégrité/compaction validées,5protocolesqualité et6smokes réussis. ComparatifGGUF mixte, contrôles CUDA/source terminés. [Rapport complet](../../lfm25-v2-results-2026-10-15.md). La publication est suivie dans `hf-v2-publication.json` ; les états initiaux ci-dessous sont conservés comme historique.

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


## Résultats finaux et commandes

- `cloud-complete-report.json` : fonction3209,232s,4shardsbruts/SHA ; `cloud-full-conversion.log.gz` et son manifeste conservent le log complet byteexact.
- `raw-integrity.json` :7402projections,556356224valeursfloat finies. `head-compact-full.json` :262144000valeurs de tête liées vérifiées,782autrestenseurs byteexacts et16384000logits natifs bitexacts ; modèlefinal12614197321octets/4,209716bpw.
- `v2-quality.json` :28fenêtres/5protocoles, gates touspassés. Le boolNumPy du rapport final a provoqué une erreur de sérialisation **après** toutes les fenêtres ; trace dans `.log`, instantané `v2-quality-before-finalization-failed.json`, reprise d'agrégats sansGPU contrôlée à1e−12.
- `functional-smokes.json/.events.jsonl` :6réponses finales, vrai bridge1.4.4, échéances180s, exit0/rejoint ; FR,391,JSON,PythonAST et2tours mémoire.
- `gguf-full-kl.json` :16fenêtres/logitsF32 complets via bibliothèque officielle llama.cpp b11620, mêmesID/teachers. AncienPPL18,3467 reproduit ; comparaison àv2 mixte. Pilote exact `.cpp.txt`, en-têtes/source/binaireSHA et8casnégatifs dans `gguf-driver-*.json`. Bruts8,39GB conservés localement dans `build/gguf-kl22`.
- `diagnostic23-comparison.json` :deuxfenêtres surlesmêmespoids, Native/CUDA ; le picKL existe danslesdeux. `diagnostic23-source-comparison.json` :sourceCUDA-F16 nonquantifiée, qui permet d'établir une forteperte liéeauxpoids/recette surlafenêtre5. LeslogitsF16 complets sont locaux dansbuild, SHAetdimensions danslesrapports. Scripts exacts archivés `.py.txt` ; échechydratation initial conservé, puis sourcePython montée etdeuxcaches créés avantallocationmodèle.
- `local-v2-install.json`, `local-v2-registered-smoke.json` :clonesAPFS indépendants du modèle final,14fichiersSHA+manifesteidentiques, anciennesentrées du registre conservées, vrai bridge paraliasrépond391/exit0. Moteur/app installés nonremplacés, UI nonexercée.

Commandes de vérification des sources exécutables inchangées depuisc639 :

```bash
PYTHONPATH=benchmarks:scripts:src:references/PonyExl3 python -m pytest -q
ruff format --check scripts/quantize_lfm25_moe.py scripts/quantize_lfm25_modal.py tests/test_lfm25_moe_conversion.py tests/test_lfm25_modal.py tests/lfm25_calibration_contracts.py tests/lfm25_modal_contracts.py
ruff check scripts/quantize_lfm25_moe.py scripts/quantize_lfm25_modal.py tests/test_lfm25_moe_conversion.py tests/test_lfm25_modal.py tests/lfm25_calibration_contracts.py tests/lfm25_modal_contracts.py
python -m compileall -q scripts/quantize_lfm25_moe.py scripts/quantize_lfm25_modal.py
python -m pip check
crosshair check tests/lfm25_calibration_contracts.py --report_all
crosshair check tests/lfm25_modal_contracts.py --report_all
```

570passés/1Lingignoré ; contratsCrossHair limités à3conditionsarithmétiques et2conditionsbooléennes. Pasdepreuve formelle duCUDA/Metal/I/O. Le pilote C++ a été compilé avecASan/UBSan/Wall/Wextra/Werror ; la tentativeCBMC échoue carl'outil estabsent. CI du codec639 :conversion/packing/nativeLinux/macOS/Desktop réussis, Kaniannulé. Lescommitsdepreuves suivants sont distingués. AucuntestGPUvide ouignoré n'estcompté commeparité.

## Audit24 du corpus exact

`calibration-composition.json` : les512000IDs réellement employés contiennent79872tokensuniformes (15,6%),39lignesMT19937seed0/modulo125017, oracle NumPy bitexact. Mutation et tri lexicographique erroné sont détectés aprèscontrôlepositif. Le fichier safetensors source exact reste dansbuild/SHA10a4a6ee… ; échecs initiaux du conteneur liste etdu vocabulairepad conservés. Aucun nouveaugainqualité n’estmesuré etaucune calibration n’estmodifiée.
