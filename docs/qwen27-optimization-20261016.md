# Qwen3.8-27B : optimisation du verifier MLP, 16 octobre 2026

Le binaire final atteint **11,08 tok/s en français** et **10,94 tok/s en
code**, en MTP1 avec batch MLP, contre **9,18 et 9,01 tok/s** avec le MLP
série : **+20,71 % et +21,37 %**, sorties identiques. La fenêtre ABBA finale
passe les seuils de stabilité avec le cache allocateur par défaut dans les
deux bras. Ce sont des mesures du moteur local, pas de l’app installée.

Le prototype précédent avait validé +20,83 à +25,17 % sur deux fenêtres
ABBA/BAAB avec cache allocateur 256 Mio. Les protocoles et les binaires sont
conservés séparément ; le +25 % historique n’est pas le gain du binaire final.

Le code local active le batch MLP dense uniquement pour **2 et 4 lignes**
sous les gardes existantes M5/hidden5120/groupedEXL3/downEXL3. Le batch de
3 lignes seul régresse : il revient au chemin série. MB3 groupé reste opt-in.
`MLXL3_EXPERIMENTAL_DENSE_MLP_BATCH=0` désactive le batch ; absence ou `1`
l’active, toute autre valeur le désactive. La clé Tune passe à `smallm-v2`
pour empêcher la réutilisation d’une calibration de l’ancienne politique.
Aucun poids, shader, dtype ou sampler n’est modifié.

## Périmètre et sources

Worktree `/Users/justin/.codex/worktrees/qwen27-smallm/mix-stq1_0`, branche
`codex/qwen27-smallm`, base `7ce611c1c0db9a704bbe35abccc6293fc6ee7995`,
avec modifications locales. Le checkout personnel modifié reste préservé.
La demande autorise l’optimisation du 27B ; les instructions du rapport joint
sont des propositions techniques à vérifier. Aucun push, release ou
remplacement de l’app installée n’a été effectué.

MacBook Pro Mac17,2, M5 à 10 cœurs GPU, 24 Go, macOS 27.2/26B5091g,
MLX 0.32.2, Rust 1.98.1, Kani 0.68.0/CBMC 6.11.0, Python 3.12.14.
Secteur, batterie 100 % pendant les campagnes. Températures et fréquences
non mesurées ; les relevés pmset ne signalent pas d’avertissement thermique.
Les anciennes campagnes Air/batterie restent distinctes.

Modèle `Qwen3.8-27B-exl3-6a9ca9d0`, EXL3 mixte K1/K2/K3/K4, 64 couches,
hidden 5120, MLP 17408, vocabulaire 248320 ; tête `Qwen3.8-27B-MTP-4bit`.
Les 16 fichiers cible/tête sont identiques en SHA-256 à la campagne historique.
L’inventaire trouve 16 géométries/290 projections regroupées, dont 64 bundles
MLP gate/up : 58 K2, 5 K1, 1 K3. [Identité et empreintes](measurements/qwen27-20261016/identity.json),
[inventaire](measurements/qwen27-20261016/inventory.json),
[comparaison historique](measurements/qwen27-20261016/historical-checkpoint-comparison.json).

Le prototype mesuré utilise le moteur SHA-256
`d92c5d6c63971a41131df87a1c76d440b90e4471661a1f30f29ebd23f1852e97`.
Le moteur final utilise
`4db446744abcd61ec0ff497fe83b4990202ebde1bf48ec28526d9f4f26793b55`.
Le libellé embarqué `7ce611c1c0db-dirty` ne suffit donc pas à les distinguer.
[Sources/patch du prototype](measurements/qwen27-20261016/measured-sources.json),
[identité finale](measurements/qwen27-20261016/final-source-identity.json),
[patch final testé](measurements/qwen27-20261016/final-runtime.patch).

## T0 : coût du verifier, avant choix de la politique

Les options existaient déjà ; elles sont isolées dans le même binaire :
A=batch0/MB30, B=batch1/MB30, C=batch0/MB31, D=batch1/MB31.
Préfixe imposé de 69 tokens, quatre blocs dépendants/sample, M1..4,
deux warmups par cellule puis huit tours mesurés : 160 samples dont
128 mesures. Les largeurs sont entrelacées par tour ; les ordres Williams
[A,B,D,C], [B,C,A,D], [C,D,B,A], [D,A,C,B], répétés deux fois,
équilibrent positions/prédécesseurs et donnent quatre paires AB/quatre BA.

Timer : forward AR M1 ou verify+commit complet M2..4 et synchronisation GPU
finale. Prefill, tête draft, rendu texte et copie des oracles exclus. Ce n’est
pas un débit utilisateur. Les logits (M×248320), hidden (5120) et 128 tableaux
d’état sont présents, finis et identiques en bits entre configurations dans
toutes les cellules. Cache allocateur commun 256 Mio, pipeline off.

Promotion à cette frontière : gain apparié médian ≥3 %, au moins 7/8 paires
positives, gain médian ≥3 % dans chaque ordre, dérive première/dernière moitié
≤3 % pour référence et candidat. Un résultat positif instable reste non concluant.

| Largeur / option | A, ms/bloc | Candidat, ms/bloc | Gain apparié | Décision |
| --- | ---: | ---: | ---: | --- |
| M2 / B batch seul | 199,679 | 162,221 | +21,355 %, 8/8 | Validé à cette frontière |
| M3 / B batch seul | 321,375 | 334,483 | −3,954 %, 2/8 | Rejeté, stable |
| M3 / C MB3 seul | 321,375 | 307,652 | +4,711 %, 8/8 | Validé à cette frontière seulement |
| M4 / D batch+MB3 | 394,488 | 309,793 | +24,460 %, 8/8 | Validé à cette frontière |
| M2 / D | 199,679 | 159,027 | +23,137 % | Non concluant : dérive −3,662 % |
| M3 / D | 321,375 | 294,016 | +10,790 % | Non concluant : dérive +5,041 % |
| M4 / B | 394,488 | 315,832 | +23,532 % | Non concluant : dérive +3,159 % |

M4/C échoue aussi au seuil de dérive (+5,653 %). Aucun gain AR établi.
M1/A vaut 117,438 ms/bloc ; rhoM2/A=1,700 et rhoM4/A=3,359,
contre rhoM2/B=1,415 et rhoM4/D=2,640 (chaque rho utilise l’AR de sa configuration).
Ne pas additionner ces gains à ceux du bridge.
[Mesures brutes](measurements/qwen27-20261016/factorial-profile-cache256.json),
[analyse complète des 16 cellules](measurements/qwen27-20261016/factorial-analysis.json).

## Bridge réel MTP1 : prototype

Même moteur/checkpoint/tête pour A et B, batch0 contre batch1, MB3off.
Contexte 4096, greedy/temp0/top_k0/pénalité1, cache de prompt off, MCP off,
pipeline/adaptive/lookup0, cache allocateur 256 Mio. Un modèle GPU à la fois,
aucune compilation ni preuve pendant les mesures. Un warmup de 64 tokens par prompt
et passe, sortie max 256, une mesure/prompt/passe, ordres ABBA puis BAAB.
Tous les IDs/texte/histoire/nombres de tokens et compteurs MTP sont égaux.
Les huit processus finissent code 0 et sont joints.

| Fenêtre / prompt | A médiane, tok/s | B médiane, tok/s | Gain |
| --- | ---: | ---: | ---: |
| ABBA / français | 9,146 | 11,060 | +20,928 % |
| ABBA / code | 8,723 | 10,919 | +25,175 % |
| BAAB / français | 9,086 | 10,979 | +20,831 % |
| BAAB / code | 8,812 | 10,914 | +23,845 % |

Chaque B dépasse chaque A d’au moins 3 % et la dérive max/min de chaque bras
reste ≤3 % dans chaque fenêtre. Le français produit 130 tokens, le code 256 :
ce n’est pas une preuve de 256 tokens soutenus en français. En ABBA, la durée
complète médiane passe de 15,131 à 12,688 s en français et de 30,557 à 24,645 s
en code. Le prefill n’est pas attribué au batchMLP ; les variations de 0,22/2,31 %
restent des variations de contrôle.
[ABBA](measurements/qwen27-20261016/bridge-d1-abba/results.json),
[BAAB](measurements/qwen27-20261016/bridge-d1-baab/results.json),
[contrôle des seuils et compteurs](measurements/qwen27-20261016/bridge-d1-summary.json).

## Binaire final : MTP1 et MTP3 validés

Q27-03 conserve le même protocole bridge (prompts/budgets/options),
A=flag0 contre B=absence/défaut, MB3 absent, cache allocateur absent/défaut
commun. Les clés réelles contrôlent pipeline off, allocator-cache-v1=0,
smallm-v2=0/1. Quatre processus code 0, joints ; warmup64 pour chaque prompt/passe,
130 tokens FR/256 code mesurés, activitéMTP et compteurs identiques.

| MTP1 / prompt | A médiane, tok/s | B médiane, tok/s | Gain | Dérive A / B |
| --- | ---: | ---: | ---: | ---: |
| Français | 9,176 | 11,077 | +20,710 % | 0,244 / 1,783 % |
| Code | 9,011 | 10,936 | +21,371 % | 1,431 / 1,030 % |

Chaque B dépasse chaque A d’au moins 3 %. FR : 71 proposés/60 acceptés ; code :
142 proposés/113 acceptés, mêmes blocs et hashes IDs/texte/histoire. Durée complète
FR 15,083→12,674 s, code 29,594→24,611 s. TTFT FR 1,024→1,026 s et
code 1,292→1,293 s : pas de gain prefill/TTFT attribué au MLP.

Mémoire relevée après génération, médianes A→B (Gio=2³⁰ octets), sans revendication
de réduction RAM : pic MLX FR 9,216→9,216/code 9,354→9,354 ; footprint processus
FR 9,739→9,772/code 10,354→10,375. Pic MLX identique, différence de footprint ≤0,34 %,
non attribuée au batch. Les caches MLX finaux FR~0,65 Gio et code~1,25 Gio
restent distincts de la RAM processus et du pic de durée de vie.

[MTP1 final, seuils et commandes](measurements/qwen27-20261016/final-bridge-d1-default-status.json),
[brut et conditions par passe](measurements/qwen27-20261016/final-bridge-d1-default-abba/results.json).

MTP3 passe aussi le filtre ABBA final (4 processus code 0/joints) :

| MTP3 / prompt | A médiane, tok/s | B médiane, tok/s | Gain | Dérive A / B |
| --- | ---: | ---: | ---: | ---: |
| Français | 7,176 | 8,762 | +22,097 % | 0,018 / 0,209 % |
| Code | 6,652 | 8,067 | +21,283 % | 0,084 / 1,491 % |

Même protocole final que MTP1, MB3 absent : M4 est confirmé sur le chemin
réel conservé. Chaque B dépasse chaque A de ≥3 %. Mêmes 130/256 tokens et compteurs
(FR 138 proposés/85 acceptés, code 294/157), sorties identiques entre bras et à MTP1.
Durées complètes FR 19,002→15,752 s, code 39,628→32,895 s. MTP1 reste plus rapide
sur ces prompts ; ces deux fenêtres successives ne sont pas un nouveau test
apparié D1 contre D3 ni une garantie universelle de meilleur mode.

Pic MLX inchangé (9,216/9,354 Gio pour FR/code), allocations actives identiques
à 0,3 Mio près. En MTP3/code, cache MLX final 1,282→1,392 Gio (+113 Mio) et
footprint processus 10,346→10,498 Gio (+155 Mio,~1,47 %) ; le batch conserve des
buffers de formes différentes dans le cache allocateur non borné. La hausse
du cache explique l’essentiel de cette différence, sans preuve d’attribution
pour chaque page processus. Pic de durée de vie 11,162→11,165 Gio (~+3 Mio).
Aucun gain RAM revendiqué ; aucun changement de politique mémoire pour compenser.

[MTP3 final, seuils/commandes](measurements/qwen27-20261016/final-bridge-d3-default-status.json),
[brut et conditions](measurements/qwen27-20261016/final-bridge-d3-default-abba/results.json).

Tune production est terminé (117,38 s, code 0, processus joint), corpus interne
différent des prompts FR/code : quatre modes, 190 tokens de décode/mode,
deux hashes identiques, quatre lignes éligibles, compteurs d’acceptation valides
et règle des 3 % vérifiée.
Un auto-test du validateur historique rejette neuf corruptions avant lancement.

| Mode Tune | Décode, tok/s | Acceptés / proposés |
| --- | ---: | ---: |
| AR | 9,175 | 0 / 0 |
| MTP1 | 10,710 | 81 / 108 |
| MTP2 | 6,975 | 105 / 169 |
| MTP3 | 7,905 | 114 / 224 |

Sélection effective **MTP1**, clé finale smallm-v2=1. Cette passe qualifie le
parcours Tune et son choix sur ces échantillons ; elle n’établit pas un gain
stable AR/MTP sur le corpus bridge ni un meilleur mode pour tous les usages.
Aucune préférence ou calibration Desktop modifiée.
[Résultat et commande Tune](measurements/qwen27-20261016/final-tune-status.json),
[événements complets](measurements/qwen27-20261016/final-tune.jsonl),
[auto-test validateur](measurements/qwen27-20261016/final-tune-validator-selftest.log).

## Correction, régressions et contrôles négatifs

- MLP physique : 384 comparaisons (64 couches×M2/3/4×MB3 off/on) exactes,
  sorties F16 finies ; erreurs d’entrée vide/rang/largeur/9 lignes refusées.
- Modèle prototype : préfixes 24/257×M2/3/4/8×5 configurations, tous les commits
  partiels : 40 cellules, 170 commits et 170 continuations de deux tokens contre
  replay AR indépendant ; logits/hidden/128 états présents, finis, exacts.
- Sessions MTP : 32 blocs D1/D2/D3 et transitions adaptatives D0..3, comparaison
  greedy et caches target/rawKV/headKV/hidden exacts ; budgets, profondeurs et
  artefacts incomplets refusés.
- Code final : MLP 384 comparaisons rejouées, fallbackM3 seul (préfixes 24/257,
  cinq configurations, 30 commits/30 continuations) et 32 blocs Session sous flag absent
  réussis. Les M2/M4 conservent le calcul du prototype ; M3 modifié est requalifié.
- Cinq vrais démarrages bridge : clés smallm-v2 par défaut=1, flag0=0,
  flag1=1, valeur invalide=0, défaut+MB31=3 ; absence=flag1 et invalid=flag0.
  Les cinq processus finissent code 0, sans modifier le registre utilisateur.
- E2E bridge budgets1/2/3/4/17/64 : tous exacts au premier essai. Le contrôle
  préfixe échoue ensuite car la fixture ne contient que 240 tokens, sous la
  frontière 256. Relance avec 429 tokens réussie (231,88 s, code 0, processus joint) : deux préfixes
  réutilisés ≥256 tokens, séparation des conversations, trois requêtes invalides refusées,
  annulations context_usage/delta et reprises exactes, fallback pénalité 1,1 exact.
  Échec initial conservé ; la clé ready confirme le pipeline automatique (dense off),
  car MLXL3_PIPELINE=0 du pilote est inconnue ; les mesures suivantes utilisent
  MLXL3_QWEN_PIPELINE=0 explicitement.
- Mutation temporaire de la garde réelle réactivant M3 : Kani détecte le
  contre-exemple. Source restaurée et contrôles ciblés relancés avec succès.
- Mutation de l’analyseur supprimant le filtre de dérive : test échoue,
  source restaurée puis 15 tests réussis. Les matrices vides/incomplètes,
  duplications, mauvais ordre, non-finitude et fausses parités sont refusées.
- Rapport physique préexistant : refus avant chargement, octets inchangés
  (échec attendu). Tous les pilotes ont des délais et nettoient leurs processus.

[Contrôles GPU initiaux](measurements/qwen27-20261016/gpu-status-cache256.json),
[GPU final](measurements/qwen27-20261016/gpu-final-status.json),
[clés effectives](measurements/qwen27-20261016/final-ready-keys.json),
[E2E final](measurements/qwen27-20261016/final-bridge-boundaries-long-status.json).
Ces tests échantillonnés ne prouvent pas tous les prompts/contextes/modèles.

## Commandes et vérification source

Environnement Rust : PATH inclut `/Users/justin/.cargo/bin` ;
`MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx`.

```sh
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --release --features mlx,chat
cargo build --locked --release --features mlx,chat
cargo kani --lib --no-default-features --harness dense_mlp_batch_stays_within_supported_domain --output-format terse
cargo kani --lib --no-default-features --harness smallm_kernel_key_is_injective --output-format terse
cargo kani --lib --no-default-features --harness grouped_mb3_preserves_rows_and_all_fallbacks --output-format terse
cargo kani --lib --no-default-features --jobs 2 --output-format terse
python -m pytest -q tests/test_smallm_profile.py tests/test_smallm_benchmark.py tests/test_compare_native.py tests/test_native_json_process.py tests/test_qmm_tiles.py tests/test_decode_microbenchmarks.py
ruff format --check benchmarks/analyze_smallm_profile.py tests/test_smallm_profile.py
ruff check benchmarks/analyze_smallm_profile.py tests/test_smallm_profile.py
```

Format/Clippy strict/build mlx,chat réussis. Suite Rust finale : 175 passés,
83 ignorés, zéro échec (lib86 passés/73 ignorés, bin42/8, contrats14/0,
APItransport33/0, runtimeGPU0/2). Les tests GPU ignorés sont distingués des
contrôles physiques explicitement sélectionnés ci-dessus. Python : 225 passés.
[Commandes/status Rust](measurements/qwen27-20261016/rust-final-status.json),
[sortie Rust](measurements/qwen27-20261016/rust-final-tests.log),
[sortie Python](measurements/qwen27-20261016/python-final-tests.log).

Le model checking borné Kani vérifie les vrais contrats scalaires CPU : garde MLP sur tout le
domaine usize64/i32/bool (6 obligations, 6 covers réussies), injectivité des quatre
masques smallm (3 obligations/4 covers), garde MB3 existante. Aucun assume/stub ajouté,
pas de boucle dans ces gardes. Cela ne prouve ni Metal/MLX ni la génération,
l’environnement, le cache ou la concurrence. Kani général interrompu après
433 s par choix de portée/coût, avant la limite 900 s : 59 harnais réussis,
zéro échec, deux anciens lookup/DFlash inachevés parmi 61 démarrés et suite
restante non exécutée. Aucun timeout ou parcours inachevé déclaré réussi.
Les obligations inatteignables de harnais anciens sont conservées dans le log.
[Contrats finaux](measurements/qwen27-20261016/kani-final-status.json),
[contre-exemple mutation](measurements/qwen27-20261016/kani-m3-mutation.log),
[suite interrompue](measurements/qwen27-20261016/kani-summary.json).

CrossHair tenté sur le vrai analyseur Python, indisponible dans le venv.
Aucune preuve formelle Python proclamée. Aucun nouveau framework de preuve.
[Diagnostic](measurements/qwen27-20261016/crosshair-source-attempt.log).

## Échecs, mémoire et CI

Le premier long test numérique sous cache allocateur MLX par défaut est arrêté
après 136,82 s avant la première cellule terminée : pression mémoire/swap global
3,58→15,73 Gio, footprint processus 11,8 Gio et 8,6 Gio de régions swappées.
Aucune divergence numérique ni débit obtenu ; ce n’est pas une preuve
que 256 Mio accélère le modèle. Reprise préenregistrée avec cache 256 Mio dans
tous les bras ; test complet réussi 273,45 s, durée fonctionnelle, pas un débit.
[Échec initial](measurements/qwen27-20261016/gpu-status.json),
[mémoire](measurements/qwen27-20261016/prefix-memory-pressure.txt).

Les allocations MLX actives/cache/pic et footprint processus sont enregistrés
séparément dans les rapports bruts ; aucun gain RAM établi. Le swap global
inclut le reste du système. Un moteur Desktop 35B a été vu transitoirement
entre campagnes closes puis a disparu avant le lancement suivant ; il n’a
pas été interrompu par ce travail. Pas de deuxième modèle pendant les mesures.

CI : workflows inspectés (PR/push, Kani CPU, Rust/Linux/macOS, Python,
build MLX/chat épinglé 0.32.2). Étape analyseur/filtres ajoutée au workflow natif.
Le run de base a les jobs Linux/macOS et conversion réussis ; Kani distant
était en cours au premier relevé puis a été annulé (run 38908579908) ;
ce job ne valide donc pas la suite complète. Regression checks échoue dans l’ancien préflight
updater `lipo -verify_arch`, hors du chemin Qwen ; échec conservé. Aucune CI du
code local modifié, aucun push. [Relevé base](measurements/qwen27-20261016/ci-base.jsonl),
[statut final du run natif de base](measurements/qwen27-20261016/ci-base-latest.json),
[échec antérieur](measurements/qwen27-20261016/ci-base-failure.log).

Restent horsqualification : autres modèles/matériels, longs contextes,
concurrence, DFlash 27B complet et défaut allocateur sous les longs harnais
de rollback. MB3 n’est
pas promu ; pas de nouveau TensorOps/KV approximatif/quantification/drafter.
Le premier lot T0/T1 apporte le gain de génération recherché. Aucun nouveau kernel
T2 n’est nécessaire pour cette activation du chemin existant ; T3/T4/T6..9
restent des expériences futures, pas des optimisations livrées. Le choix Tune
existant et sa nouvelle clé sont exercés sans forcer un mode gagnant. [Journal complet](../opti.md).

Campagne close : aucun essai Q27 ou processus GPU de cette campagne en cours.
Code local prêt, application installée inchangée ; aucune publication.
