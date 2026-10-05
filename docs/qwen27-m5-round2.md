# Qwen3.8 27B EXL3 sur M5 — deuxième campagne, 5 octobre 2026

Le changement retenu élargit les tuiles de sortie du décodage mono-token,
de deux à quatre tuiles EXL3 par groupe, sur trois formes mesurées du modèle.
Il conserve les poids, le codebook, les produits et l'ordre des réductions.
Le MTP reste hors de cette campagne.

## Portée du changement

La sélection s'applique au GPU M5, en MUL1, avec une entrée de 5120 valeurs :

| Projection | Bits K | Sortie |
| --- | ---: | ---: |
| Bundle GDN qkv/z | 2 | 16384 |
| Bundle MLP gate/up | 2 | 34816 |
| Tête vocabulaire | 3 | 248320 |

Les autres formes et le corps multi-token du transformeur gardent leur
dispatch existant. La tête finale ne traite que le dernier token après un
préfill et peut donc aussi bénéficier du nouveau réglage mono-token.
`MLXL3_DENSE_DECODE_NT4=0` rétablit le réglage précédent pour comparaison.
Les shaders et leurs réductions ne sont pas modifiés. Le choix passe par
`contracts::dense_decode_nt4`, appelé par `Exl3Linear::forward` et
`Exl3Group::forward`, eux-mêmes utilisés par les projections Qwen et le bridge.

Le fork a reçu les mises à jour de sa branche, puis celles du parent jusqu'à
`c73c2a8414f68acf2b70b1a7e8ccba4a334e8639`, moteur 1.3.1 et PR 23 fusionnée.
La campagne conserve les conflits de journal résolus et les backups d'autostash.
Ce travail concerne les sources du fork ; aucune installation ni publication
du moteur n'est réalisée par cette campagne.

## Mesures et correction

Machine : MacBook Air M5, CPU/GPU 10 cœurs, 24 Gio, macOS 27.2.
MLX 0.32.2, Rust 1.99.0, build release `mlx,chat`.
Checkpoint local : `Qwen3.8-27B-exl3-6a9ca9d0`, dense, 64 couches,
hidden 5120, intermediate 17408, vocabulaire 248320, EXL3 mixte K1/K2/K3/K4.
Aucune requantification ni modification du sampler.

Le filtre synthétique compare les partielles FP32 **avant** l'arrondi F16 :
48 matrices couvrent dense/groupé, K1..K4, trois codebooks, SG4/split1 et
SG8/split4, chacune en NT1/2/4/8. Toutes sont finies et identiques en bits.
Les quatre grandes formes sont aussi comparées exactement. NT1 et NT8 sont
plus lents sur ces grandes formes.

La confirmation de 40 passages alternés donne les médianes suivantes,
soumission CPU + MLX + synchronisation GPU comprises :

| Microbenchmark | NT2 | NT4 |
| --- | ---: | ---: |
| Gate/up K2 | 0,9713 ms | 0,9074 ms |
| Qkv/z K2 | 0,5919 ms | 0,5672 ms |
| Tête K3 | 5,7753 ms | 5,4591 ms |

Ces durées ne sont pas des temps GPU purs et ne s'additionnent pas pour
annoncer un gain de génération. Le down MLP était déjà en NT4 en production.

Le premier test apparié utilise un seul modèle chargé, un snapshot commun
après 69 IDs synthétiques, 16 tokens imposés par passage, quatre warmups et
huit paires dans les ordres AB/BA alternés. Les huit paires sont plus rapides
avec NT4 : médiane A 2,203987 s, B 2,088347 s, gain apparié médian **5,565 %**.
Chaque paire compare les 248320 logits F16 finaux, tous finis, et les 128 états
internes en bits. Les copies CPU de ces oracles sont hors chronomètre.
Ce résultat mesure le forward natif sur un préfixe imposé.

La répétition appariée sur le code final, après retrait de la sonde C++, donne
**4,774 %** de gain apparié médian, encore huit paires gagnantes sur huit.
Médianes A 2,137583 s et B 2,039958 s pour 16 tokens, soit 7,485 → 7,843 tok/s
sur ce préfixe imposé. Les logits et 128 états restent exacts. Le test final
contre les références sauvegardées passe aussi aux 12 étapes : préfills de
23/128/256 tokens puis trois décodages pour chacun, 248320 logits finis et
128 états par étape. Les deux campagnes appariées soutiennent un gain du
forward natif d'environ **5 % sur ce Mac et ce protocole**.

La première campagne bridge ABBA donne un signal positif de 6,32 % et 10,27 %
sur les deux prompts, mais une forte dérive de la référence interdit d'en
faire un gain causal établi. La confirmation BAAB donne −6,33 % sur le prompt
court et +5,85 % sur le document ; le préfill court varie aussi de −5,87 %.
Son corps multi-token est inchangé, mais sa tête finale est en M=1 et bénéficie
du changement : le temps préfill n'est donc pas un contrôle totalement intact.
La dernière passe B chute
de 8,040 à 6,749 tok/s sur le court et de 7,921 à 6,166 sur le document.
**Le gain de débit bridge reste non concluant.** Tous les tokens et le texte
sont identiques, cache désactivé, greedy, 24 tokens, deux mesures et un warmup
par prompt/passe, contexte4096, repos25 s, un seul modèle à la fois.
Les campagnes sont sur batterie, températures/fréquences non mesurées ;
l'absence d'avertissement `pmset` n'établit pas une fréquence stable.

## Autres pistes examinées

L'audit des métadonnées du checkpoint trouve aussi six grands bundles MLP
encore en NT2 : cinq K1 et un K3, tous 5120 → 34816, avec huit SIMD groups.
Deux fenêtres de 40 passages confirment un gain micro de 8,73–9,17 % pour K1
et 3,94–4,44 % pour K3, avec partielles FP32 exactes. Le prototype passe les
12 étapes de l'oracle checkpoint, les tests et Kani. Dans le modèle, toutefois,
deux fenêtres de huit paires donnent seulement +0,218 % puis +0,547 % médian,
avec quatre paires gagnantes sur huit à chaque fois. Toutes les paires AB
perdent et toutes les BA gagnent : l'effet d'ordre domine le petit signal.
**L'extension K1/K3 est retirée de la production.** Les sources Rust retenues
sont restaurées en bits à celles de la campagne08 ; seuls le microfiltre
élargi, le patch expérimental et les preuves restent archivés. Les 77
couvertures Kani du prototype sont distinctes des 74 du code retenu.
Ces microgains ne sont pas des gains supplémentaires de génération.

Une fusion RMSNorm128 + gate SiLU utilise une table F32 de 256 Kio calculée
par le même graphe MLX pour tous les patterns F16. Elle conserve les deux
arrondis F16 de la norme et le produit final F32. Les 40 fixtures passent,
y compris les gates F16 finies exhaustives. Les timings varient avec la forme :
48 lignes 0,244812 → 0,231042 ms ; 144 lignes 0,231500 → 0,233354 ms ;
384 lignes 0,257730 → 0,246334 ms. **Pas de gain modèle établi**, aucune fusion
intégrée au moteur. Les deux échecs initiaux de compilation sont conservés.

Le diagnostic des clés du cache Metal trouve 16527 appels, 59 misses,
aucune éviction et 59 entrées sur 128 possibles. La construction/recherche
C++ des clés prend 21,655 ms cumulées sur 19,494 s de requêtes instrumentées,
soit environ 0,111 %. Cela n'inclut ni les CString Rust ni la compilation JIT.
Le cache ne thrash pas ici ; le changement de cache n'est pas retenu.
L'instrumentation a été archivée puis retirée avant les dernières mesures.

L'AR asynchrone de MTPLX reste une piste non mesurée. Il exige une adaptation
de la frontière d'évaluation/streaming du moteur natif ; aucun gain n'est
attribué à sa seule présence dans MTPLX. Les anciennes pistes lazy global,
transpositions, changements de réductions et fusions de tête rejetées ne
sont pas réintroduites.

## Vérification et limites

Le [skill fourni par l'utilisateur](https://github.com/0xZKnw/formal-proof-skill)
est lu à la révision `5f00bb443c9d0d14f705a3098381850166a83e5c`, avec les
références Rust, C++ et Python. Le nouveau harnais Kani appelle la fonction
Rust de production avec des entrées i32/i32/usize/bool/bool symboliques,
sans hypothèse : formes admissibles, codebook/GPU, sortie positive alignée128,
couvertures K2, K3 et fallback. Il ne prouve pas le calcul Metal, MLX,
les allocations ni la génération complète.

Kani, CBMC et CrossHair étaient absents lors des premières tentatives, dont
les diagnostics restent conservés. Kani officiel0.68.0 a ensuite été installé
dans l'outillage de travail isolé, sans dépendance projet. Sur ARM Apple, avec
nightly2026-08-21, le nouveau contrat passe7obligations et3couvertures, sans
hypothèse ni obligation inatteignable. La suite CPU passe32/32harnais :
4473obligationsSUCCESS,70UNREACHABLE préexistantes,74couverturesSATISFIED,
aucun échec ni résultat indéterminé. Les bornes des anciens harnais restent
celles de leurs sources. C'est du **model checking Rust CPU** ; aucune preuve
source Python/C++ locale n'est revendiquée.
Les tests GPU sont des tests échantillonnés, hormis l'exploration finie des
patterns de gates F16 ; ils ne constituent pas une preuve générale du modèle.

Contrôles locaux : 61 tests Rust passent, 54 tests GPU/modèles sont ignorés
dans la suite générale et doivent être sélectionnés explicitement ; 117 tests
Python pertinents passent, dont 16 nouveaux cas pour les microfiltres.
Fmt, Ruff, py_compile, Clippy strict et build release `mlx,chat` passent.
Trois mutations temporaires sont détectées : retrait du garde M5 et
suppression du refus de résultats vides dans chacun des entrypoints Python.
Les sources sont restaurées puis retestées. Le test d'override vérifie
imbrication, restauration après panique et isolement entre threads.

La collecte Python globale échoue sur des dépendances/parcours hors changement
(`mlx_lm`, accès GPU sandbox), avec un skip `ponyexl3`. Un ancien test QMM à
délai court a échoué une fois sur la raison d'erreur attendue, puis passé seul
et dans la suite pertinente complète. Ces limites restent visibles dans les logs.

Sources primaires examinées :
[MTPLX](https://github.com/youssofal/MTPLX/tree/9882703f3105363ddc37eca9f97aa09a1d387112),
[normalisation MLX 0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/rms_norm.metal).
Les idées BF16/affine de MTPLX ne sont pas utilisées comme substitutions
automatiques des kernels EXL3/F16.

Le [journal](../opti.md) contient les protocoles, échecs et décisions.
Les [mesures brutes](measurements/qwen27-m5-round2/) contiennent les campagnes,
provenances, traces de tests, mutations et tentatives de vérificateurs.
Le binaire mesuré sans sonde est identifié dans `provenance.json`, SHA-256
`e28a993cb044310923d862d2f6cd6c6d0f3333dd781e402e9217b0e6a088573f` ;
il n'est pas commité. Les workflows du fork étaient initialement désactivés,
malgré l'indicateur général d'API `enabled=true`. Leur activation a été vérifiée
dans l'UI du propriétaire et par les deux workflows devenus `active` dans
l'API. Un nouveau push documentaire déclenche les contrôles du même code.
Le commit`2592d95` passe trois jobs GitHub : Linux/macOSnative chacun50testsRust
et117Python, et DesktopE2E119Python/4skips plus les contrôles Swift/protocole.
Le job Kani distant a attendu15minutes sans obtenir de runner GitHub puis a
été annulé ; l'annotation confirme l'échec d'attribution après plusieurs
tentatives. **La CI entière n'est pas annoncée verte**. Les32preuves Kani
locales ARM sont distinctes de cette indisponibilité de runner.
Les [résultats natifs](https://github.com/HENK0O/mlxl3/actions/runs/37372734634)
et [Desktop](https://github.com/HENK0O/mlxl3/actions/runs/37372734558), leurs JSON
et logs compressés sont archivés. La clôture documentaire suivante ne change
aucun fichier exécutable ; ses runs doivent aussi être inspectés séparément.

Révision après inspection de `20debe916f2f77cdbfe068a0e85e54a3ec6c43d5` : les
trois jobs [Native/Kani](https://github.com/HENK0O/mlxl3/actions/runs/37374854568)
sont réussis. Les logs confirment 32 harnais, 4473 checks SUCCESS, 70
UNREACHABLE préexistants et 74 couvertures satisfaites, sans échec. Les jobs
Linux/macOS passent chacun 50 tests Rust et 117 Python. Le premier essai
[Desktop](https://github.com/HENK0O/mlxl3/actions/runs/37374854512) est annulé
après la limite configurée de 25 minutes ; ses logs sont absents et aucune
assertion défaillante n'est disponible. Une relance unique du même job/SHA
est demandée. Les JSON, annotations et diagnostics sont conservés dans
`ci-20debe9/` ; la réussite native ne suffit pas à annoncer toute la CI verte.
Les commits documentaires suivants devront être distingués de ce SHA.

Clôture : la deuxième tentative Desktop sur ce même SHA est **réussie**,
119 tests Python passent avec quatre skips, puis les contrôles Swift,
lifecycle, imports, streaming, bridge et CLI passent. Les **quatre jobs de
20debe9 sont verts**, Kani compris. La première annulation reste archivée.
Les métadonnées et logs de la tentative2 sont dans `ci-20debe9/`.

Après retrait du prototype K1/K3, les empreintes des sources Rust, FFI et
build sont identiques à celles de08. Fmt, Clippy, build et 61 tests Rust
passent à nouveau (54 ignorés). La reconstruction porte une autre empreinte
binaire, `53335a7b9740e2a6d9150f71c4eb63f71ee2bc698af378bb0c03047f575ebba1` :
le build inclut la révision Git dans les diagnostics. Il n'est pas présenté
comme le binaire chronométré. `final-source-provenance.json` distingue ces
artefacts ; le binaire mesuré original reste conservé.
