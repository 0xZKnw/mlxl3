# GEMV HC mixte Flash — essai20

État : qualité numérique et gain de débit validés ;20C est terminé. Le kernel
est retenu dans le code de la PR34, sans installation de l’application. Les
essais20 et20B restent non concluants ou interrompus ; leurs données sont conservées.

## Changement

Les projections HC utilisaient une entrée FP32 et des poids FP16, ce qui
matérialisait une conversion des poids en FP32 avant chaque multiplication.
Le nouveau kernel lit directement les poids FP16 et reproduit les partitions,
les FMA et l'ordre de réduction du GEMV MLX0.32.2, avec une sortie FP32.
Il ne conserve aucune copie FP32 des poids et n'augmente pas le budget cache.

Seules les formes10240→320,320→10240 et10240→4, avec une seule ligne FP32,
empruntent ce chemin. Les poids denses sont chargés en FP16 par le constructeur
existant. Les autres formes, projections et entrées FP16 de la tête MTP suivent
leur chemin d'origine. La cible utilise ces HC pendant le décodage ordinaire
et pendant la vérification MTP. Aucun changement de quantification ou de
sélection des experts ; les prototypes HC-write, NT2 et T2 antérieurs restent
retirés.

Sources primaires : [dispatch MLX0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp#L1053),
[GEMV MLX](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/gemv.h).
Attribution Apple/MIT dans THIRD_PARTY_NOTICES.md.

## Correction avant vitesse

- 298matrices HC réelles :1 056 392valeurs FP32 finies, formes et octets exacts
  face au matmul MLX original.
- 21 128valeurs synthétiques FP32, oracle CPU indépendant par sommes dyadiques
  exactes, zéros, entrées stridées, annulations, fallbacks FP16/batch/biais et
  largeur invalide. Muter l'ordre BN8 du shader réel fait échouer l'oracle ;
  restauration et nouveau succès, binaire de tests revenu au SHA initial.
- 11tests Qwen physiques, oracle tenu à part de11positions et3tests SSD.
  Les références indépendantes couvrent ensemble16positions, logits complets
  et états cible. Les commits1/2, la tête MTP et les sessions passent.
- Bridge réel : budgets1/2/3/17, cinq configurations refusées, deux parcours
  annulation/reprise, fallback avec pénalité et réponses OFF/MTP exactes.
  Sa durée183,63s est un contrôle fonctionnel, pas une mesure de performance.
- Fmt, Clippy strict, build livré mlx,chat ;87tests CPU réussis/2ignorés,
  107tests release réussis/98ignorés. Les tests physiques sont sélectionnés
  explicitement et exécutés seuls.

Kani0.68 est tenté sur le vrai appel Rust→MLX : deux i8 symboliques indépendants,
sans hypothèse, diffusés en poids320×10240 et entrée1×10240. Son compilateur
échoue avant les obligations (intrinsics.rs:243). CBMC6.11 ne reconnaît pas
l'entrée MSL. Ces chemins restent **non vérifiés formellement** ; aucune preuve
GPU n'est revendiquée. [Inspection des preuves et domaines](measurements/flash-next-ssd-impl/opt-20-hc-gemv/proof-inspection.json).

Les fonctions de benchmark existantes passent31régressions. Le callback de
parité128 est contrôlé sur8cas réels, dont hash/texte/préfixe divergents, vide,
compte tronqué et NaN. La garde secteur ajoute5cas ; CrossHair est absent du
runtimePython3.12 utilisé, donc aucune preuve symbolique de cette garde.

## Identités et conditions

Apple M5/24Gio, macOS27.2, MLX0.32.2. Modèle
Qwen3.8 Flash Next EXL3 2,05bpw, révision65c895314393431c09050b2e04e250836b3a6eb4.
Source HEAD36bae3adc7be6f19ecc47f2fbe1daf856e30cf2b avec patch natif enregistré.
Moteurs immuables : témoin bd89a7d6fa49bec7587df603d62f9e767cac70525fffa0d2cb8c34fd8d57686e,
candidat7a9a21205812dee34680df4b85600116b8ecb65c58d7897826206af54a8406a1.
Tests candidats19a12f37219a617f8aaa1b07d411501435b3dfcc66eec5ea1d7a083c719fa69e.
[Manifeste complet](measurements/flash-next-ssd-impl/opt-20-hc-gemv/identity.json).

Contexte4096, cache experts5Gio/89slots par couche, allocator512Mio,
quatre lecteurs SSD, overlap hot/cold actif. Applications utilisateur laissées
ouvertes. Un modèle à la fois, sans compilateur/proveur concurrent.
Relevés VM, alimentation et thermique par bras ; les compteurs de swap sont
ceux du système entier, le pic mémoire est le lifetime max physical footprint du processus
(proc_pid_rusage), pas la seule allocationMLX ni le RSS.

**Correction des conditions :20,21 et20B se sont déroulés sur batterie**,
contrairement au secteur indiqué par erreur dans le protocole initial.
Leurs relevés bruts ne sont pas modifiés. L'utilisateur a rebranché le Mac
avant20C ; cette série vérifie AC Power avant et après chaque bras.
Ces contrôles ponctuels ne certifient pas une alimentation continue.

## Microbenchmark distinct

Cinq paires alternées par forme,32évaluations après8chauffes par bras.
Médianes témoin/candidat : down468,07/273,52µs, up439,38/341,82µs,
inject260,51/246,26µs. Une paire injection est défavorable et reste conservée.
Ces chronos muraux ne sont ni un débit de génération ni un tempsGPU exclusif.
RAM processus et swap par micro non mesurés.

## Campagnes de débit

20 : chauffe64 puis64tokens générés/63décodés, MAIN/HOLD×OFF/MTP,
H-C/C-H/H-C prévus.18bras terminés, un interrompu et cinq non démarrés.
Les neuf paires complètes vont de+33,42 à+43,20%, mais les étendues dépassent
le plafond préenregistré de3points. **Non concluant pour un gain stable**,
même si toutes les réponses complètes sont exactes et les swapouts nuls.
Les deux erreurs initiales de lancement Python3.9/chaîneJS, antérieures
à toute mesure, sont aussi conservées.
[Résultat brut20](measurements/flash-next-ssd-impl/opt-20-hc-gemv/speed/result.json),
[qualification](measurements/flash-next-ssd-impl/opt-20-hc-gemv/speed/assessment.json).

20B :128tokens/127décodés, une chauffe et trois mesures par processus,
deux tours H-C/C-H prévus. Cinq bras terminés, un interrompu et dix non
démarrés. La dérive HOLD candidat4,02/7,78/7,80tok/s et la condition batterie
incorrecte interdisent une conclusion stable. Tous résultats conservés.
[Résultat brut20B](measurements/flash-next-ssd-impl/opt-20-hc-gemv/confirmation-128/result.json),
[arrêt et conditions](measurements/flash-next-ssd-impl/opt-20-hc-gemv/confirmation-128/early-stop.json).

20C reprend exactement les binaires et les128tokens/trois mesures de20B
sur secteur vérifié.16bras prévus, médiane des trois mesures par bras.
Chaque gain apparié doit dépasser1%, avec une étendue≤3points par cellule,
un pic candidat≤témoin+256Mio et≤256Mio de swapouts système par bras.
La série s'arrête sur divergence, alimentation incorrecte ou ratio max/min
supérieur à1,25 entre les trois mesures d'un bras. Pas de sélection du
meilleur résultat, pas de cumul avec les campagnes64tokens.
[Plan20C](measurements/flash-next-ssd-impl/opt-20-hc-gemv/confirmation-ac/matrix.json),
[résultat20C](measurements/flash-next-ssd-impl/opt-20-hc-gemv/confirmation-ac/result.json).

## Résultat20C retenu

Les16bras et64réponses sont terminés et exacts. Les huit gains appariés
respectent le seuil1% et les étendues par cellule restent entre0,035 et1,761point,
sous la limite3points. Médianes ci-dessous : médiane des deux passages,
chaque passage étant la médiane de trois générations après chauffe.

| Mode / prompt | Témoin tok/s | Candidat tok/s | Gains appariés |
|---|---:|---:|---:|
| OFF / MAIN |7.11|9.85|+38.36% / +38.49%|
| OFF / HOLD |7.06|9.63|+37.03% / +35.92%|
| MTP1 / MAIN |7.02|10.11|+44.11% / +44.08%|
| MTP1 / HOLD |6.42|9.26|+45.02% / +43.26%|

Pic candidat OFF 8.659Gio,
MTP 9.644Gio. Le plus grand surplus
apparié est 166.2Mio, sous256Mio.
Les16bras ont chacun0octet supplémentaire de swapout système. Les compteurs
MTP sont identiques entre variantes : MAIN72propositions/54acceptations,
HOLD78propositions/48acceptations sur chaque génération128.
Ce gain compare le kernel HC au moteur précédent ; il ne signifie pas que
MTP dépasse OFF sur tous les prompts.15tok/s restent non atteints.
[Qualification indépendante des rapports](measurements/flash-next-ssd-impl/opt-20-hc-gemv/confirmation-ac/assessment.json).

## Coûts restants et limites

Le [profil21](flash-profile-after-hc-2026-10-12.md) du même prototype place
la vérification cible autour de94% du coût MTP sur ses préfixes32tokens.
Ce diagnostic sur batterie n'est pas une comparaison appariée de débit.
L'essai22 de HC à deux lignes est préparé hors checkout et n'a pas été exécuté.
15tok/s ne sont pas atteints ou validés par ces essais.

La CI du HEAD de départ36bae3ad réussit Linux, macOS/MLX, Kani ciblé et
régressions Desktop ; le job Kani complet est annulé après sa limite1h, confirmée par
l’annotation GitHub. Les tests AppleGPU
locaux et la limite du vérificateur restent distincts de ces contrôles.
L'application installée n'a pas été remplacée.
