# Flash Next : lectures SSD bornées, essai 14

État : quatre lecteurs retenus après la campagne 14B, qualité et critères comparatifs validés localement. La campagne 14 reste non concluante selon son propre critère de stabilité. Défaut retenu dans le code de cette branche ; application installée inchangée.

## Changement et sources

Les experts déjà présents restent dans le cache LFU. Pour les experts absents, le moteur réserve toute la cohorte puis remplit directement les douze segments de chaque record EXL3 dans les sept banques existantes. La nouvelle FFI groupe les records et répartit les lectures entre au plus quatre invocations de `dispatch_apply_f`. Cette fonction attend la fin de toutes les invocations, même lorsqu'un record échoue. Le moteur invalide alors le cache et termine le calcul des hits avant de permettre une reprise.

Les intervalles sont vérifiés avant la première écriture, y compris les alias entre tableaux MLX. Aucun tampon de poids intermédiaire, nouveau format, changement d'experts ou approximation numérique n'est ajouté. Le lecteur séquentiel reste sélectionnable avec `MLXL3_SSD_READERS=1` ; les valeurs permises sont 1 à 4. Le candidat final utilise 4 par défaut.

Le principe des lectures concurrentes vient de la [comparaison des moteurs SSD](flash-ssd-upstream-2026-10-12.md). Aucun kernel tiers n'est copié. L'ordonnancement utilise l'API système [Apple `dispatch_apply_f`](https://developer.apple.com/documentation/dispatch/dispatch_apply_f) et le contrat de [libdispatch](https://github.com/apple-oss-distributions/libdispatch/blob/main/dispatch/queue.h).

## Correction vérifiée avant le débit

- Lecteur C++ réel : 72 cas déterministes passent sous ASan/UBSan et TSan. Couverture de 1 à 4 lecteurs, lots de 1/2/5/512 records, sentinelles, dispersion, chevauchements, débordements, fichier tronqué, descripteur invalide, achèvement des autres lectures et reprise.
- Rust : format, Clippy strict et build livré `mlx,chat` passent ; 87 tests CPU passent avec 2 ignorés, et 107 tests release avec 95 ignorés. Les tests physiques sont lancés séparément.
- 29 contrôles physiques passent sur le premier binaire : les modes 2 et 4 lecteurs conservent les logits et états des 16 positions de référence ; les tests MTP et 90 paires de cohortes couvrent les caches de 10/20/143 slots, l'overlap, les hits/misses et les erreurs. Le chemin sans variable d'environnement du candidat final est ensuite revérifié sur les logits/états, la tête MTP et la session MTP.
- Le bridge réel passe les budgets 1/2/3/17 tokens en ordre OFF/ON/ON/OFF, la configuration MTP, les annulations et les reprises. Les paramètres de lecteurs invalides sont rejetés. Il s'agit de contrôles fonctionnels, pas de chiffres de vitesse.
- Pilote de mesure : budget fixe configurable, 461 tests Python passent, 4 sont sautés faute d'adaptateurs Pony ou de modèle Ling. Les 27 tests ciblés du pilote et Ruff passent. Deux mutants isolés sont détectés : autoriser les alias et ignorer le budget demandé.

Kani vérifie la garde réelle des destinations sur tout le domaine `usize` du Mac 64 bits, sans hypothèse ni boucle : 12 obligations, 3 couvertures. La suite ciblée totalise 10 harnais, 1 102 obligations et 34 couvertures, avec 3 obligations inatteignables dans les anciens contrats. Cela ne prouve ni MLX, ni Metal, ni la concurrence. Le chemin FFI et la suite complète échouent dans le compilateur Kani ; CBMC ne parse pas la libc++ locale ; CrossHair ne trouve pas de fonction vérifiable dans le pilote I/O. Les commandes erronées initiales et les tentatives corrigées sont conservées.

Les [contrôles physiques](measurements/flash-next-ssd-impl/opt-14-readers/quality.json), [contrôles du bridge](measurements/flash-next-ssd-impl/opt-14-readers/bridge-quality.json), [builds](measurements/flash-next-ssd-impl/opt-14-readers/build.json), [tentatives formelles](measurements/flash-next-ssd-impl/opt-14-readers/formal-02.json) et [mutants](measurements/flash-next-ssd-impl/opt-14-readers/mutants-formal.json) donnent les commandes et les logs.

## Protocole et crible

Apple M5, 24 Gio, macOS 27.2, MLX 0.32.2 ; checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, EXL3 `2.05bpw_h4_ng4`. Applications utilisateur ouvertes, alimentation secteur. Budget experts figé à 5 Gio (89 slots par couche), cache allocateur 512 Mio, overlap actif, contexte 4096, greedy. Deux prompts MAIN/HOLD, 64 tokens produits dont 63 de décodage. Un seul modèle GPU, aucune compilation ou preuve pendant la mesure. Les compteurs système sont relevés avant/après ; les pics du processus viennent du moteur.

La [matrice préenregistrée](measurements/flash-next-ssd-impl/opt-14-readers/matrix.json) précise ordre, seuils de gain, dérive et pression mémoire. Le [crible](measurements/flash-next-ssd-impl/opt-14-readers/screen.json) comporte quatre chauffes puis un passage par cellule :

| Lecteurs | OFF MAIN/HOLD (tok/s) | MTP MAIN/HOLD (tok/s) | Pic processus (Gio) | Swapouts système (Mio) |
|---|---:|---:|---:|---:|
| 1 | 6,42 / 6,45 | 6,43 / 6,22 | 9,79 | 0 |
| 2 | 7,28 / 7,27 | 7,29 / 7,16 | 9,94 | 0 |
| 4 | 7,38 / 7,40 | 7,52 / 7,32 | 9,67 | 0 |

Ce crible en ordre fixe sélectionne quatre lecteurs ; il ne confirme pas un gain causal. La chauffe du mode 2 montre une dérive. Le mode 1 utilise aussi la nouvelle FFI groupée et n'est donc pas le moteur historique.

## Comparaison 14 : non concluante

Témoin immuable H `5e423506` contre candidat `bd89a7d6`, dans l'ordre H-C / C-H / H-C, trois passages par cellule avec quatre chauffes à chaque processus. Les identités sont conservées dans [le manifeste final](measurements/flash-next-ssd-impl/opt-14-readers/default4-identity.json). Les six bras de la [comparaison](measurements/flash-next-ssd-impl/opt-14-readers/confirmation.json) sont complets, les sorties exactes et les swapouts système nuls.

Les médianes passent de 6,50/6,44 à 7,35/7,39 tok/s sans MTP et de 6,37/6,29 à 7,42/7,29 avec MTP. Tous les couples sont favorables, mais le spread du candidat atteint 7,6–9,1 %, au-delà du seuil préenregistré de 5 %. Ce [résultat non concluant](measurements/flash-next-ssd-impl/opt-14-readers/result.json) ne valide donc pas la rétention du défaut. C2 reste dans les résultats ; sa cause de variation n'est pas établie.

Une seule nouvelle campagne 14B est prévue avec une chauffe puis une mesure par processus, un prompt et un mode à la fois, en rapprochant les paires H/C. Elle utilisera de nouvelles données et un critère comparatif explicite : chaque gain apparié ≥1 % et minimum candidat supérieur de ≥1 % au maximum témoin dans chaque cellule. L'essai 14 garde sa conclusion. Objectif 15 tok/s encore non atteint. L'application installée n'est pas remplacée.

## Confirmation 14B : critères comparatifs satisfaits

La nouvelle campagne est terminée : 24 processus, chacun avec une chauffe et une mesure de 64 tokens ; 12 paires H/C en ordre alterné. Le modèle, les deux binaires, le budget de 5 Gio et les autres paramètres restent identiques. Chaque processus utilise un seul prompt et un seul mode. Les [commandes et conditions](measurements/flash-next-ssd-impl/opt-14-readers/adjacent/execution.json), la [matrice préalable](measurements/flash-next-ssd-impl/opt-14-readers/adjacent/matrix.json) et les [résultats complets](measurements/flash-next-ssd-impl/opt-14-readers/adjacent/result.json) sont conservés séparément de 14.

| Mode / prompt | Témoin médian (tok/s) | Candidat médian (tok/s) | Gains des trois paires |
|---|---:|---:|---|
| OFF / MAIN | 6,68 | 7,29 | +7,38 %, +5,57 %, +9,09 % |
| OFF / HOLD | 6,64 | 7,05 | +6,06 %, +6,04 %, +5,53 % |
| MTP / MAIN | 6,52 | 7,13 | +9,29 %, +9,37 %, +10,42 % |
| MTP / HOLD | 6,35 | 6,89 | +8,29 %, +8,54 %, +7,41 % |

Les douze gains appariés dépassent 1 %. Dans chaque cellule, le passage candidat le plus lent reste au moins 1 % au-dessus du témoin le plus rapide. Les 48 réponses gardent le texte et le hash de référence. Toutes les sources natives et les binaires gardent leurs empreintes. Ce résultat établit un avantage répété sur les passages observés, pas un débit absolu constant ni un résultat à 15 tok/s.

Les pics du candidat atteignent 8,684 Gio sans MTP et 9,451 Gio avec MTP. L'écart maximal à son témoin apparié est de +20,5 Mio, sous le seuil de 256 Mio ; les 24 bras enregistrent zéro nouvelle écriture de swap système. Le swap antérieur peut rester alloué. Ces valeurs ne se mélangent pas aux pics de la campagne avec changements de mode dans un même processus.

Le pilote expose `--prompt MAIN|HOLD`, `--repeats` et `--cache-gib`. Ses 31 tests ciblés passent, ainsi que 465 tests Python complets (4 sautés). Le mutant ignorant le filtre de prompt échoue ; Ruff et le format passent. CrossHair n'a pas de propriété I/O vérifiable ici. Le code natif et ses preuves/limites restent ceux de 14 ; 29 exécutions physiques couvrent 15 tests distincts, puis 3 contrôles supplémentaires exercent le défaut sans variable d'environnement. La CI du commit antérieur `bdc3889b` passe Linux, macOS et les contrats Kani ciblés ; le Kani complet est annulé après environ une heure, pas validé.
