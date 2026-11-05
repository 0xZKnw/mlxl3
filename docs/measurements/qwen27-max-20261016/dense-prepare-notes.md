# Q27-17 — préparation dense SwiGLU → suh → H128

État : préparationCPU terminée, **aucunGPU lancé par l'agent**, aucun résultat numérique ou gain encore établi. Le parent lance seul après Q27-16. Runtime, poids, app et shaders du dépôt inchangés par cet agent. Préenregistrement lu dans opti.md Q27-17 avant création des deuxfichiers.

## Contrat et sources réellement lues

Chemin : qwen35.rs::Mlp::forward(Dense) → Array::swiglu → mlxl3_array_binary(operation6) → mx::compile((gate*sigmoid(gate))*up,shapeless=true) ; puis Exl3Linear::forward/_batch → castFP16 × suhFP16 → hadamard_transform128(scale=1/sqrt128).

Le MoE::glu_down_input existant fournit une structure de préparation mais ses intermédiairesF32 diffèrent : **aucune utilisation commeoracle**, aucunpointflottantempruntédecehelper.

Sources primaireMLX0.32.2 fixes à1f8e74e3f12f31365464a6867c6579f0e9b29d85, copiéesenlectureseule dans dense-prepare-source/manifest.json :

- mlx/backend/metal/compiled.cpp, lignes205–221 : chaqueprimitive est affectée àuntemporaire dudtype de sonArray ; iciFP16 pourSigmoid etlesdeuxMultiply.
- kernels/unary_ops.h::Sigmoid : `auto y=1/(1+metal::exp(metal::abs(x)))`, puis retour `(x<0)?y:1-y` deT. Kernel reprend cetteexpression etlesaffectationshalf, pasexpprécisinventé nihalf→F32surleSwiGLU.
- kernels/binary_ops.h::Multiply : multiplicationdeT, retourT.
- hadamard.cpp : pourN128, m1,n1=1,n2=128,max_radix16,readwidth4, normalisationF32. AucunedécompositionH>8192.
- kernels/hadamard.h::hadamard_n : radix16F32 (4stages) puisstoreT ; radix8F32 (3stages) puisstoreT ; sortieT(buf×scaleF32). Kernel fait lesmêmesbutterfliesavecbarrières etroundhalfaprèsstages3et6, puisnormalisationhalf.

Troisheaders installés (unary_ops,binary_ops,hadamard) sont comparésoctetpourcentetsauxsourcespinnées ; égalitévraie etempreintes dans dense-prepare-cpu-status.json. `mlx/backend/metal/jit/compiled.cpp` essayé404, conservé ; le vrai fichier estbackend/metal/compiled.cpp. Aucunecompilationouexécutiondecodesourceexterne.

## Prototype et premier oracle physique

Deuxfichiers : dense_prepare.py ettest_dense_prepare.py. Scriptn'importeMLXquedansrun ; importsCPU/tests n'ouvrentpasdecontexteGPU. Kernel128threads/bloc, troisstadescompiletime (SwiGLU, échelle, WHTcomplet) pourlocaliserunéventuelécart avanttoutevitesse ; aucuneécriturediagnostiqueremplaceles temporairessupprimés dansl'armcomplète.

Oracleindépendant : véritablemx.compile de lamêmeexpression queFFI puismx.multiplyFP16 etmx.hadamard_transform128. Pasunsecondshaderaveclesmêmesformules. Toutescomparaisons utilisent assert_exact existant : forme, dtype, nombre nonnul etfinitude puisbytes.

Plan GPU codé, conditionnéparleparent :

1. Les63488patternsgateFP16finis, rembourrésà4×H17408 etup2^-10/scalesalternés±1 pourgarderH128fini. Troisstadesexercés pourchaquepartie :12comparaisonsde17408valeurs. IlnecomparepaspourNaN/Inf.
2. Exigertousles64projectionsdown ducheckpoint, dtypeI16/input17408/output5120/K1..4/Mul1, scalesFP16réellesfiniesetSHA. Chaqueéchelle×M1/M2×deuxsaltspseudoaléatoiresdéterministes :256comparaisonscompletes. Totalattendu268 ; aucunfiltrevideaccepté.
3. PourunreprésentantdechaqueKprésent, lesdeuxxhat passentdans lemêmeQMVstockcanonique ; partiellesFP32doiventêtreprésentesfiniesbitexactes. Aucunshaderdownmodifié.
4. Mutationdukernelretirantleseularrondiaprèsradix16 : doitdonnerdesvaleursfiniesdifférentesetêtre refuséeparlevraioracleMLX. Sansdétection, arrêt. Pasdemesureavantles268contrôlescomplets.
5. EnsuiteunreprésentantparKprésent×M1/M2, huitpréparationsdépendantes avecfeedbacktanhFP16commun ; troiswarmups puis40pairesAB/BA avecparentpaired/speed_gate. Allocationcache256Mio, coûtpréparation+feedback+soumissionMLX+sync inclus. Ce n'estpasunforwardMLPcomplet niundébitdetokens. Seuils : médiane≥3%,≥80%pairespositives, chaqueordre≥3%, dérive≤3%.

Leparentdoitencadrerleprocessusparledélai600s etcleanup/joinqu'ila préparé ; leprototype necréepasd'enfant. mainenregistre running/parityfalse avantimportMLX, catchBaseException/KeyboardInterrupten failed/parityfalse puisfinallysave. MauvaisCLI/refusécrasementn'écraseaucunancienrapport.

## ContrôlesCPU et erreurs conservées

- `.venv/bin/ruff format dense_prepare.py test_dense_prepare.py` : passe.
- `.venv/bin/ruff check --select E,F,I,B,UP,SIM,PLW dense_prepare.py test_dense_prepare.py` : premièrefoisE402(importsaprèsajoutsys.path) etE501(stringtrop longue), corrigéenimportanthelpersdansrun etsplittantstring. Repasséstrictsansdiagnostic.
- `.venv/bin/python -m pytest -q test_dense_prepare.py` : **7passés/0skip**, dernier0,01s. Formes/types/bornes, source mutationrefusmarqueurabsent/dupliqué,64recordsstricts/codebook/dimensions, CLIoverwrite/40pairs, divergences/timeout/KeyboardInterrupt sauvegardésfalse.
- `.venv/bin/python -m py_compile dense_prepare.py test_dense_prepare.py` : passe.
- `.venv/bin/python -m crosshair check dense_prepare.py --analysis_kind=asserts` : réellementtenté, échoue **No module namedcrosshair**. AucunepreuvePython,MSL,MLXouGPUrevendiquée ; pasd'installationoutildelourd.

Identitésfigées : dense_prepare.py SHA05475c20b3762e53f4c2933375d4ef647694e25eaf948d531d13e28724006d4b ; test_dense_prepare.py SHAd836d1b1458e826b8c2263410d4e48999d23e043d704507e2406f4c1a7fff0de. Lesbrutscommandesontété inspectés ; résumé avecéchecinitialetlimites dans dense-prepare-cpu-status.json.

Commandeproposéeauparent, **nonexécutéeparcetagent** :

```sh
/Users/justin/Documents/mix-stq1_0/.venv/bin/python /tmp/mlxl3-q27-max-20261016/dense_prepare.py --checkpoint '/Users/justin/Library/Application Support/io.mlxl3.desktop/Models/Qwen3.8-27B-exl3-6a9ca9d0' --output /tmp/mlxl3-q27-max-20261016/dense-prepare-gpu.json --iterations 40
```

Aucungainlocalannoncé. Uneparité oumicrovitesseéventuelle nevaliderapasencore leraccordementnatif,lesétatsdumodèle,lebridge,nil'appinstallée.
