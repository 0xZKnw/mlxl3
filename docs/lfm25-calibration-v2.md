# LFM2.5-24B-A2B : préparation qualité v2

État au 15 octobre2026 : conversion v2 et évaluations terminées. Le gate8192tokens donne **PPL26,86163/KL0,187928**, contre29,47950/0,323047 en v1. Les quatre protocoles complémentaires et les six smokes du moteur1.4.4 passent. La comparaison GGUF est mixte et le diagnostic source/CUDA montre une fenêtre sensible àla quantification : [résultats complets et prochaines pistes](lfm25-v2-results-2026-10-15.md). La [v2 HF](https://huggingface.co/0xzknw/LFM2.5-24B-A2B-EXL3-4bpw/tree/v2) est publiée et son dossier v2 enregistré pour test local ; v1 reste conservée. Les paragraphes de préparation ci-dessous décrivent le protocole initial, avec ses limites.

## Ce que l'audit établit

L'[audit v1](lfm25-quality-audit-2026-10-14.md) compare12 projections réelles,
48tenseurs et17359884valeurs : aucune divergence avec Pony sur ce panel.
Cela exclut plusieurs erreurs de kernel envisagées, sans prouver les7402projections.
Le principal écart de recette est la calibration : seulement2048tokens source,
et les mêmes entrées offertes à chaque expert. La calibration d'origine utilisait
les états BF16 avant toute quantification, sans propagation des erreurs accumulées.

La contre-épreuve18 améliore les sorties linéaires sur les **tokens test routés**
de trois matrices down : −2,02%, −12,92% et −4,45% d'erreur RMS relative.
Les contrôles tous-tokens se dégradent sur deux matrices ; l'amélioration train
est beaucoup plus grande. Ce résultat ne remplace pas une PPL/KL globale.

## Chemin Metal conditionnel

[`quantize_lfm25_moe.py`](../scripts/quantize_lfm25_moe.py) ajoute
`--routed-calibration`, sans modifier la recette legacy par défaut.
L'instrumentation observe les véritables appels SwitchLinear, les IDs d'experts
et les entrées w1/w3/w2, y compris les activations SwiGLU déjà calculées.
Elle prend en charge les routes triées et les entrées diffusées, sans recalcul
dense des experts pour reconstruire les captures.

Toutes les séquences demandées sont parcourues avant plafonnement de chaque
projection. Un expert doit recevoir au moins `min(calibration_rows,256)` lignes
réelles. Une couverture insuffisante fait échouer la conversion ; aucune entrée
n'est empruntée à un autre expert. Cette limite évite aussi le changement de
backend Hadamard sous192lignes ; elle ne garantit pas une Hessienne de rang plein.

`coverage.json` conserve les lignes observées/retenues, formes, empreintes et
statuts complete/failed/interrupted. Les caches vérifient les données, la source,
le corpus et la recette. Les dossiers legacy peuvent contenir des liens entre
experts : le nouveau chemin les rompt avant écriture et ne partage gate/up
qu'au sein d'un même expert. Les hooks sont restaurés sur erreur/annulation.

Une recette locale possible est2048lignes par projection,128séquences de1024
tokens train (131072tokens présentés au routeur). Le corpus fourni contient
203982tokens source. Le maximum des captures dédupliquées est36,67Go hors
en-têtes ; réserver aussi les12–13Go du modèle et les fichiers temporaires.
Cette recette n'a pas été chronométrée sur le24B.

## Ce que Sushi permet de conclure

Sources primaires épinglées : [Sushi2df81de](https://github.com/beamivalice/sushi/tree/2df81de2938fdbcf519604945b1e01d1dbd6766c),
[format public](https://github.com/beamivalice/sushi/blob/2df81de2938fdbcf519604945b1e01d1dbd6766c/docs/pack-format.md),
[pack Qwen4f25c0c](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw/tree/f25c0c4acdc44289a9cc6dab7f0eb788d1125ded).
Le convertisseur est privé : le format et les métadonnées ne révèlent pas son
algorithme complet d'optimisation.

Le pack «4bpw» associe des experts EXL3K4/MCG et un tronc largement affine8bits,
dont attention, GDN et experts partagés. La table ngram BF16 sur SSD n'est pas
comptée dans sa taille GPU annoncée. Ce n'est donc pas une comparaison uniforme
à4bits de tous les paramètres. Les métadonnées d'un shard expert lu par requête
HTTP bornée indiquent LDLQ rotated, calibration imatrix-diagonal, échelles de
sortie svh et `g_scale:gss`. Notre v1 utilisait déjà la recherche GSS.

Le pack utilise une fenêtre W15 alors que MLXL3 attend W16 : les octets ne sont
pas interchangeables. Les chiffres KL de Qwen4 ne sont pas comparables à ceux
de LFM avec d'autres données, contextes et moteurs.

Protéger les752877568paramètres de projections non expertes de LFM en K6 au lieu
de K4 ajouterait environ188,22Mo de payload (+0,06281bpw), hors alignement et
métadonnées. C'est une estimation de taille, **aucun gain qualité mesuré**.
Le CLI Pony uniforme ne propose pas encore cette allocation fixe ; `--head-bits`
ne protège pas à lui seul attention et convolution. Le CUDA officiel propose
déjà `--hq` pour ces couches, ce qui évite de recréer un allocateur local.

## Conversion Modal/H100 et vérification

Le [convertisseur ExLlamaV3v1.5.4](https://github.com/turboderp-org/exllamav3/tree/16a49792a3c93d8432d72e6c4bce800841566577)
supporte Lfm2MoeForCausalLM,64experts/top4 et le routeur sigmoid+biais.
Ses [options](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/doc/convert.md)
incluent un corpus mixte de512000tokens, des états recalculés couche par couche,
des échelles de sortie guidées par la sensibilité et HQ. Il exécute des kernels
CUDA officiels ; nos kernels Metal restent utilisés uniquement sur Apple.
Le module de lancement est [`quantize_lfm25_modal.py`](../scripts/quantize_lfm25_modal.py).
Les poids du candidat restent sur un Volume Modal privé jusqu'au gate qualité.

Le lancement utilise le clientModal1.5.5 (`pip install -e '.[dev]'`), unH10080Go,
8CPU/96GioRAM et un timeout4h sans retry. La roue CUDA12.8/Torch2.10/Python3.12
est vérifiée parSHA256, puis cinq fichiers Python de conversion sont comparés
au commit épinglé avant téléchargement du modèle. Le prévol utilise64experts,
top4, attention, convolution, MLPdense et les biais F32 des vraies couches2/39.
Ses395projections et4096logits sont nonvides/finis ; CUDAdecode contre natifdecode
donne RMSrelative0,0019083 sous tolérance0,02 et32/32argmax identiques.

Les essais synthétiques précédents à biais tous nuls restent **hors tolérance**
(RMS0,0329 puis0,0397), malgré32argmax égaux. Les erreurs d'API du premier harnais
et de téléchargement sont archivées. Ce prévol avec biais réels ne prouve pas
la cause de tous les écarts précédents ni la qualité finale du24B.

La recette complète est `-b4 -hb16 -cbmcg --out_scales auto --hq -cr250 -cc2048`.
Les experts restentK4, les MLPdenses sontK5 et attention/convolutionK6. Le code
amont effectue aussi un refit des échelles par canal aprèsLDLQ. Le head lié
reste non quantifié et peut être dupliqué dans l'export ; la taille et le bitrate
physique seront mesurés et indiqués séparément de «K4experts».
Les1403213tokens des six textes de calibration n'ont aucun13gramme exact commun
avec les294644tokens du WikiText test sous le tokenizer source. Ce contrôle
n'exclut pas tout recouvrement sémantique possible.

La calibration conditionnelle Metal et la recette CUDA sont deux variantes
distinctes. Le prévol doit vérifier export, codebook, formes et logits dans le
véritable moteur MLXL3 avant le lancement complet. La quantification n'est pas
considérée comme réussie à partir du seul code de sortie du convertisseur.

Les idées supplémentaires YAQA/QTIP/finetuning restent des pistes : l'extracteur
YAQA public vise principalement nn.Linear et n'est pas une solution validée pour
ces experts fusionnés. Aucune de ces techniques n'a été silencieusement appliquée.

## Résultats de vérification locale de la préparation Metal

[Manifeste et logs](measurements/quant-lfm25-calibration-v2-20261014/README.md) :
50tests ciblés, puis556tests complets réussis et1test Ling ignoré (modèle absent).
Les deux passages de CLI synthétique exécutent le bridge du moteur livré1.4.4.
Le panel réel inclut64experts/top4, toutes les395projections d'une petite
architecture, les routes triées/non triées, les queues et les entrées w2 propres
à chaque expert. Oracle down indépendant : RMSrelative≤1,026e−5, tolérance1e−4.
Les tests couvrent cache corrompu, expert froid, disque, annulation, changements de
recette et reprise byte exacte. Une mutation tous-tokens est détectée par l'oracle.

CrossHair confirme trois contrats d'alignement sur entiers Python symboliques.
Il s'agit d'une recherche de contre-exemples, pas d'une preuve de MLX/Metal,
du routage flottant ou de la qualité finale. Les échecs initiaux et corrections
restent dans les journaux. CI modifiée pour PR/pushmain ; aucun run de la nouvelle
tête locale non publiée ne peut encore être annoncé vert.

Révision finale du15octobre : deux cas de cache supplémentaires (fichier0octet
et archiveNPZ renommée.npy) invalident aussi le marqueurcomplete et ferment
l'archive. Les52tests ciblés et570tests complets passent,1Ling reste ignoré.
Les erreurs initiales et lelint de type corrigé restent dans les preuves de20.

Le gate v2 sera le protocole historique8192tokens/8×1024/sansBOS, même teacher et
même moteur : **PPL<29,47949896 et KL<0,32304706**, baisse relative>0,1% des deux,
avec examen par fenêtre et smokes FR/code/calcul391/JSON. Les autres protocoles
seront conservés séparément. Pas de publication HF si le gate échoue.

La v1 a été réévaluée sur ce même moteur1.4.4 empaqueté : les huit fenêtres,
PPL etKL reproduisent exactement le résultat historique. Deux corpus supplémentaires
du dépôt, français etPython, utilisent chacun1024tokens/2×512 et un enseignantBF16
source indépendant ; tolérance de non-régression complémentaire2% relatif par
métrique/corpus. Les protocoles court etBOS restent séparés. Ces petits corpus
techniques ne sont pas des benchmarks généraux de français ou de programmation.
